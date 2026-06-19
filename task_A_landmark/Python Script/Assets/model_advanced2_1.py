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
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="bilinear", align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))
        return results[0]  # highest-resolution fused map (stride 4 -> 64x64 @ 256 input)

class SoftArgmax2D(nn.Module):
    """
    Spatial-softmax regression voting. Softmax (not sigmoid) is the right parameterization:
    it suppresses background pixels exponentially, so the soft-argmax centroid is not pulled
    toward the image center by diffuse background mass.
    Returns (coords [B, C*2] in [0,1], prob [B, C, H, W] summing to 1 over H,W).
    """
    def __init__(self, temperature=1.0):
        super(SoftArgmax2D, self).__init__()
        self.temperature = temperature  # >1 sharpens; 1.0 == original baseline behavior

    def forward(self, logits):
        B, C, H, W = logits.shape
        prob = F.softmax(logits.view(B, C, -1) * self.temperature, dim=-1).view(B, C, H, W)
        y_grid = torch.linspace(0.0, 1.0, steps=H, device=logits.device).view(1, 1, H, 1)
        x_grid = torch.linspace(0.0, 1.0, steps=W, device=logits.device).view(1, 1, 1, W)
        y_preds = torch.sum(prob * y_grid, dim=(2, 3))  # [B, C]
        x_preds = torch.sum(prob * x_grid, dim=(2, 3))  # [B, C]
        coords = torch.stack([x_preds, y_preds], dim=-1).view(B, C * 2)
        return coords, prob

class AdvancedLandmarkRegressor(nn.Module):
    def __init__(self, num_points=4, temperature=1.0):
        super(AdvancedLandmarkRegressor, self).__init__()

        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        pretrained_conv1 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).conv1.weight.data
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = pretrained_conv1.mean(dim=1, keepdim=True)

        self.fpn = FeaturePyramidFusion([64, 128, 256, 512], out_channels=128)
        self.attention = SpatialAttention(128)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_points, kernel_size=1)  # logits
        )
        self.soft_argmax = SoftArgmax2D(temperature=temperature)

    def forward(self, x, return_heatmaps=False):
        x0 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x0 = self.backbone.maxpool(x0)
        c1 = self.backbone.layer1(x0)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        fused = self.fpn([c1, c2, c3, c4])
        attended = self.attention(fused)
        logits = self.heatmap_head(attended)

        coords, prob = self.soft_argmax(logits)
        if return_heatmaps:
            return coords, prob
        return coords