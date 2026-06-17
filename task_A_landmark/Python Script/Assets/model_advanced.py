import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialAttention(nn.Module):
    """Applies Spatial Attention to focus on the actual cranium boundary."""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)

class FeaturePyramidFusion(nn.Module):
    """Fuses multi-scale features from ResNet so the model sees both macro and micro details."""
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidFusion, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        # x is a list of features: [c1, c2, c3, c4] from ResNet
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="bilinear", align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))
            
        return results[0] # Return the highest resolution fused feature map

class SoftArgmax2D(nn.Module):
    """Regression Voting: Converts 2D heatmaps into continuous (x, y) coordinates cleanly."""
    def __init__(self):
        super(SoftArgmax2D, self).__init__()

    def forward(self, heatmaps):
        # heatmaps: [Batch, 4 Points, Height, Width]
        B, C, H, W = heatmaps.shape
        
        # 1. Apply Spatial Softmax so all pixels in a heatmap sum to 1.0 (probabilities)
        heatmaps = heatmaps.view(B, C, -1)
        heatmaps = F.softmax(heatmaps, dim=-1)
        heatmaps = heatmaps.view(B, C, H, W)
        
        # 2. Create X and Y coordinate grids (normalized 0 to 1)
        y_grid = torch.linspace(0.0, 1.0, steps=H, device=heatmaps.device).view(1, 1, H, 1)
        x_grid = torch.linspace(0.0, 1.0, steps=W, device=heatmaps.device).view(1, 1, 1, W)
        
        # 3. Every pixel "votes" on the coordinate weighted by its probability
        y_preds = torch.sum(heatmaps * y_grid, dim=(2, 3)) # [B, 4]
        x_preds = torch.sum(heatmaps * x_grid, dim=(2, 3)) # [B, 4]
        
        # 4. Stack and flatten into [x1, y1, x2, y2, x3, y3, x4, y4] to match our dataset labels
        coords = torch.stack([x_preds, y_preds], dim=-1).view(B, C * 2)
        return coords

class AdvancedLandmarkRegressor(nn.Module):
    def __init__(self, num_points=4): # 4 anatomical points
        super(AdvancedLandmarkRegressor, self).__init__()
        
        # 1. Backbone
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).conv1.weight.data.sum(dim=1, keepdim=True)
        
        # 2. FPN (Fuses layer 1, 2, 3, 4)
        self.fpn = FeaturePyramidFusion([64, 128, 256, 512], out_channels=128)
        
        # 3. Spatial Attention
        self.attention = SpatialAttention(128)
        
        # 4. Predict 4 Heatmaps
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_points, kernel_size=1) # 4 output channels
        )
        
        # 5. Regression Voting
        self.soft_argmax = SoftArgmax2D()

    def forward(self, x):
        # Extract features
        x0 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x0 = self.backbone.maxpool(x0)
        c1 = self.backbone.layer1(x0) 
        c2 = self.backbone.layer2(c1)  
        c3 = self.backbone.layer3(c2)  
        c4 = self.backbone.layer4(c3)  
        
        # Pass through FPN, Attention, and Head
        fused = self.fpn([c1, c2, c3, c4])
        attended = self.attention(fused)
        heatmaps = self.heatmap_head(attended)
        
        # Vote for final coordinates
        coords = self.soft_argmax(heatmaps)
        return coords