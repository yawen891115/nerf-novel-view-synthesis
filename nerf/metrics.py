"""Image quality metrics used in the NeRF paper: PSNR, SSIM and LPIPS.

All functions take float images in [0, 1] with shape (H, W, 3).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def mse_to_psnr(mse_value: torch.Tensor | float) -> float:
    """PSNR = -10 log10(MSE) for images in [0, 1]. Higher is better."""
    value = float(mse_value)
    return math.inf if value == 0.0 else -10.0 * math.log10(value)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return mse_to_psnr(mse(pred.clamp(0, 1), target))


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).float()


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11, sigma: float = 1.5,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity with an 11x11 Gaussian window (sigma 1.5), averaged over
    pixels and channels; the standard setting used by NeRF evaluation code."""
    if pred.device.type == "mps":
        pred, target = pred.cpu(), target.cpu()
    x = pred.clamp(0, 1).permute(2, 0, 1)[:, None].double()       # (3, 1, H, W)
    y = target.permute(2, 0, 1)[:, None].double()
    w = _gaussian_window(window, sigma, x.device).double()[None, None]

    def filt(img):
        return F.conv2d(img, w)                                     # valid region only

    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x ** 2
    syy = filt(y * y) - mu_y ** 2
    sxy = filt(x * y) - mu_x * mu_y
    c1, c2 = k1 ** 2, k2 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float((num / den).mean())


class LPIPS:
    """Learned perceptual distance (lower is better). Needs `pip install lpips`; the first
    call downloads the backbone weights. Papers differ in the backbone (VGG or AlexNet), so
    always report which one you used."""

    def __init__(self, net: str = "vgg", device="cpu"):
        import lpips  # imported lazily so the rest of the code works without it
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        def prep(img):  # (H, W, 3) in [0, 1] -> (1, 3, H, W) in [-1, 1]
            return img.clamp(0, 1).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        return float(self.model(prep(pred), prep(target)))
