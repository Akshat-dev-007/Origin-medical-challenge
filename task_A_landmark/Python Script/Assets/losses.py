import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_targets(coords_gt, H, W, sigma):
    """
    Render peak-1 Gaussian heatmaps centered on the ground-truth landmarks.
    coords_gt: [B, 2P] normalized in [0,1], interleaved [x1,y1,x2,y2,...].
    Returns: [B, P, H, W] with peak value 1.0 at each landmark.
    """
    B = coords_gt.shape[0]
    P = coords_gt.shape[1] // 2
    c = coords_gt.view(B, P, 2)  # (x, y)
    dev = coords_gt.device

    ys = torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
    # Map normalized coords to heatmap-pixel centers (i <-> i/(dim-1), matching soft-argmax grid)
    cx = (c[..., 0] * (W - 1)).view(B, P, 1, 1)
    cy = (c[..., 1] * (H - 1)).view(B, P, 1, 1)

    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    return g  # peak 1


def js_divergence(p, q, eps=1e-12):
    """Mean Jensen-Shannon divergence between two distributions normalized over (H, W)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=(2, 3))
    kl_qm = (q * (q / m).log()).sum(dim=(2, 3))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm  # [B, C]
    return jsd.mean()


def wing_loss(residual, w=10.0, eps=2.0):
    """
    Wing loss (Feng et al., 2018). residual is the per-coordinate error in PIXELS.
    Log regime for small errors (sub-pixel precision), ~L1 for large errors (robust to outliers).
    """
    x = residual.abs()
    c = w - w * math.log(1.0 + w / eps)
    loss = torch.where(x < w, w * torch.log(1.0 + x / eps), x - c)
    return loss.mean()


class LandmarkLoss(nn.Module):
    """
    Three complementary terms:
      - l_heatmap : MSE between predicted [0,1] heatmaps and explicit peak-1 Gaussian targets.
                    This is the dominant spatial supervision (shapes the whole map).
      - l_reg     : Jensen-Shannon divergence between the normalized predicted distribution
                    and the normalized Gaussian -> keeps each heatmap unimodal and tight.
      - l_coord   : Wing loss on the soft-argmax coordinates (sub-pixel accuracy, outlier-robust).

    Note: l_heatmap and l_reg both pull toward the same Gaussian and are partially redundant
    (MSE penalizes per-pixel deviation incl. far-field mass; JS is a proper divergence on shape).
    Keep both at first, then ablate w_reg -> 0 to confirm it earns its place. The trainer prints
    each component so you can rebalance the weights for your data.
    """
    def __init__(self, sigma=2.0, img_size=256,
                 w_heatmap=10.0, w_coord=1.0, w_reg=0.5,
                 wing_w=10.0, wing_eps=2.0, eps=1e-8):
        super().__init__()
        self.sigma = sigma
        self.img_size = img_size
        self.w_heatmap = w_heatmap
        self.w_coord = w_coord
        self.w_reg = w_reg
        self.wing_w = wing_w
        self.wing_eps = wing_eps
        self.eps = eps

    def forward(self, coords_pred, heatmaps, coords_gt):
        # coords_pred, coords_gt: [B, 2P] normalized [0,1]; heatmaps: [B, P, H, W] in [0,1]
        H, W = heatmaps.shape[2:]

        g1 = gaussian_targets(coords_gt, H, W, self.sigma)          # peak-1 target
        l_heatmap = F.mse_loss(heatmaps, g1)

        # Normalized distributions for the JS regularizer
        p = heatmaps / (heatmaps.sum(dim=(2, 3), keepdim=True) + self.eps)
        gn = g1 / (g1.sum(dim=(2, 3), keepdim=True) + self.eps)
        l_reg = js_divergence(p, gn)

        # Coordinate loss in pixel units on the 256 canvas (matches the eval metric)
        residual_px = (coords_pred - coords_gt) * self.img_size
        l_coord = wing_loss(residual_px, self.wing_w, self.wing_eps)

        total = self.w_heatmap * l_heatmap + self.w_coord * l_coord + self.w_reg * l_reg
        components = {
            "heatmap": l_heatmap.item(),
            "coord": l_coord.item(),
            "reg": l_reg.item(),
        }
        return total, components