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

        return results[0]  # Return the highest resolution fused feature map (stride 4 -> 64x64 @ 256 input)


def soft_argmax_2d(prob):
    """
    Regression voting over an already-normalized probability map.
    prob: [B, C, H, W] summing to 1 over (H, W).
    Returns coords [B, C*2] in [0, 1], interleaved as [x1, y1, x2, y2, ...].
    Grid uses linspace(0, 1) so pixel i maps to coordinate i/(W-1), keeping it
    consistent with the [0,1]-normalized labels and the existing Tester.
    """
    B, C, H, W = prob.shape
    y_grid = torch.linspace(0.0, 1.0, steps=H, device=prob.device).view(1, 1, H, 1)
    x_grid = torch.linspace(0.0, 1.0, steps=W, device=prob.device).view(1, 1, 1, W)
    y_preds = torch.sum(prob * y_grid, dim=(2, 3))  # [B, C]
    x_preds = torch.sum(prob * x_grid, dim=(2, 3))  # [B, C]
    coords = torch.stack([x_preds, y_preds], dim=-1).view(B, C * 2)
    return coords


class AdvancedLandmarkRegressor(nn.Module):
    def __init__(self, num_points=4):  # 4 anatomical points
        super(AdvancedLandmarkRegressor, self).__init__()

        # 1. Backbone
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        pretrained_conv1 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).conv1.weight.data
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Average (not sum) the RGB filters so grayscale activation stats match the pretrained scale
        self.backbone.conv1.weight.data = pretrained_conv1.mean(dim=1, keepdim=True)

        # 2. FPN (Fuses layer 1, 2, 3, 4)
        self.fpn = FeaturePyramidFusion([64, 128, 256, 512], out_channels=128)

        # 3. Spatial Attention
        self.attention = SpatialAttention(128)

        # 4. Predict 4 heatmap logits
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_points, kernel_size=1)  # 4 output channels (logits)
        )

        self.eps = 1e-8

    def forward(self, x, return_heatmaps=False):
        # Extract features
        x0 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        x0 = self.backbone.maxpool(x0)
        c1 = self.backbone.layer1(x0)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)

        fused = self.fpn([c1, c2, c3, c4])
        attended = self.attention(fused)
        logits = self.heatmap_head(attended)

        # Bounded, non-negative heatmaps in [0,1] -> comparable to peak-1 Gaussian targets (MSE).
        heatmaps = torch.sigmoid(logits)

        # Normalize to a probability distribution for sub-pixel soft-argmax + JS regularization.
        denom = heatmaps.sum(dim=(2, 3), keepdim=True) + self.eps
        prob = heatmaps / denom

        coords = soft_argmax_2d(prob)

        if return_heatmaps:
            # Return the unnormalized [0,1] heatmaps; the loss normalizes them itself for the JS term.
            return coords, heatmaps
        return coords