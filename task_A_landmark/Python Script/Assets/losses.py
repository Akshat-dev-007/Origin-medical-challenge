import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_targets(coords_gt, H, W, sigma):
    """Peak-1 Gaussian heatmaps centered on GT landmarks. coords_gt: [B, 2P] in [0,1]."""
    B = coords_gt.shape[0]
    P = coords_gt.shape[1] // 2
    c = coords_gt.view(B, P, 2)
    dev = coords_gt.device
    ys = torch.arange(H, device=dev, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, 1, W)
    cx = (c[..., 0] * (W - 1)).view(B, P, 1, 1)
    cy = (c[..., 1] * (H - 1)).view(B, P, 1, 1)
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    return g


def js_divergence(p, q, eps=1e-12):
    """Mean Jensen-Shannon divergence; p, q are normalized over (H, W)."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=(2, 3))
    kl_qm = (q * (q / m).log()).sum(dim=(2, 3))
    return (0.5 * kl_pm + 0.5 * kl_qm).mean()


class LandmarkLoss(nn.Module):
    """
    Canonical DSNT-style objective:
      l_coord = mean per-point radial (L2) distance in normalized [0,1] coords
                -> directly minimizes your eval metric (radial error / 256).
      l_reg   = Jensen-Shannon divergence between the softmax heatmap and a normalized
                Gaussian centered on GT -> this IS the explicit Gaussian target AND the
                distribution regularizer, done on the softmax map (no sigmoid bias).

    w_reg = 0.0 reproduces the pure softmax soft-argmax baseline (your 27.99 px run).
    Sweep w_reg in {0, 0.1, 0.5, 1.0} after confirming recovery at 0.
    """
    def __init__(self, sigma=2.0, w_coord=1.0, w_reg=0.02, eps=1e-8):
        super().__init__()
        self.sigma = sigma
        self.w_coord = w_coord
        self.w_reg = w_reg
        self.eps = eps

    def forward(self, coords_pred, prob, coords_gt):
        # coords: [B, 2P] normalized [0,1]; prob: [B, P, H, W] summing to 1 over (H, W)
        B = coords_pred.shape[0]
        P = coords_pred.shape[1] // 2
        pred_pts = coords_pred.view(B, P, 2)
        gt_pts = coords_gt.view(B, P, 2)
        l_coord = torch.norm(pred_pts - gt_pts, dim=2).mean()

        if self.w_reg > 0:
            H, W = prob.shape[2:]
            g = gaussian_targets(coords_gt, H, W, self.sigma)
            gn = g / (g.sum(dim=(2, 3), keepdim=True) + self.eps)
            l_reg = js_divergence(prob, gn)
        else:
            l_reg = torch.zeros((), device=coords_pred.device)

        total = self.w_coord * l_coord + self.w_reg * l_reg
        return total, {"coord": l_coord.item(), "reg": float(l_reg.item())}