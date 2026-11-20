"""Small helpers shared by the scripts."""

from __future__ import annotations

import random

import numpy as np
import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to8b(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """Float image in [0, 1] -> uint8 image for saving."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def depth_to_gray(depth: torch.Tensor, near: float, far: float) -> np.ndarray:
    """Map depth in [near, far] to a uint8 image (near = bright) for quick inspection."""
    d = ((far - depth) / (far - near)).clamp(0, 1)
    return to8b(d)
