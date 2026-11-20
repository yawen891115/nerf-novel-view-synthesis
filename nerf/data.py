"""NeRF-synthetic ("Blender") data loading, camera rays and training-ray sampling.

Dataset layout (one folder per scene, e.g. data/nerf_synthetic/lego):

    transforms_train.json   {"camera_angle_x": ..., "frames": [{"file_path": "./train/r_0",
    transforms_val.json                                    "transform_matrix": 4x4 c2w}, ...]}
    transforms_test.json
    train/r_0.png ...       800x800 RGBA images (alpha = object mask)

Camera convention (OpenGL / Blender): in camera space the camera looks down -z,
x points right and y points up. `transform_matrix` maps camera space to world space.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


# --------------------------------------------------------------------------- loading
def load_blender_split(basedir: str | Path, split: str, half_res: bool = False,
                       white_bkgd: bool = True, skip: int = 1):
    """Read one split. Returns (images (N,H,W,3) float32 in [0,1], poses (N,4,4), H, W, focal,
    names) where `names` are the frame file names, used when saving renders."""
    basedir = Path(basedir)
    with open(basedir / f"transforms_{split}.json") as f:
        meta = json.load(f)
    frames = meta["frames"][::skip]

    imgs, poses, names = [], [], []
    for frame in frames:
        path = basedir / (frame["file_path"] + ".png")
        imgs.append(imageio.imread(path))
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
        names.append(Path(frame["file_path"]).name)
    imgs = np.stack(imgs).astype(np.float32) / 255.0        # (N, H, W, 4) RGBA
    poses = np.stack(poses)

    H, W = imgs.shape[1:3]
    focal = 0.5 * W / math.tan(0.5 * float(meta["camera_angle_x"]))   # pinhole focal in pixels
    if half_res:
        # 2x2 box filter = area interpolation for an exact factor of 2.
        imgs = imgs.reshape(-1, H // 2, 2, W // 2, 2, 4).mean(axis=(2, 4))
        H, W, focal = H // 2, W // 2, focal / 2.0

    rgb, alpha = imgs[..., :3], imgs[..., 3:]
    if white_bkgd:
        rgb = rgb * alpha + (1.0 - alpha)     # transparent pixels become white
    return rgb.astype(np.float32), poses, H, W, focal, names


# --------------------------------------------------------------------------- rays
def pixel_rays(rows: torch.Tensor, cols: torch.Tensor, H: int, W: int, focal: float,
               c2w: torch.Tensor):
    """Rays through pixels (rows[k], cols[k]). Returns origins and directions, each (K, 3).

    The pixel (row j, column i) maps to the camera-space direction
    ((i - W/2) / f, -(j - H/2) / f, -1), which is rotated into world space by c2w.
    Like the official code we use integer pixel coordinates (no +0.5 offset).
    """
    i, j = cols.to(c2w.dtype), rows.to(c2w.dtype)
    dirs = torch.stack([(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -torch.ones_like(i)], -1)
    rays_d = dirs @ c2w[:3, :3].T                 # rotate: d_world = R d_cam
    rays_o = c2w[:3, 3].expand_as(rays_d)         # every ray starts at the camera centre
    return rays_o, rays_d


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """One ray per pixel of an H x W image. Returns origins and directions, each (H, W, 3)."""
    rows, cols = torch.meshgrid(torch.arange(H, device=c2w.device),
                                torch.arange(W, device=c2w.device), indexing="ij")
    rays_o, rays_d = pixel_rays(rows.reshape(-1), cols.reshape(-1), H, W, focal, c2w)
    return rays_o.reshape(H, W, 3), rays_d.reshape(H, W, 3)


class RaySampler:
    """Training batches: pick one random training image, then `n_rand` distinct random
    pixels of it (the official "no_batching" mode used for the Blender scenes).
    During the first `precrop_iters` steps only the central crop is sampled."""

    def __init__(self, images: np.ndarray, poses: np.ndarray, H: int, W: int, focal: float,
                 n_rand: int, precrop_iters: int = 0, precrop_frac: float = 0.5,
                 device: str | torch.device = "cpu"):
        self.images = torch.from_numpy(images).to(device)    # (N, H, W, 3)
        self.poses = torch.from_numpy(poses).to(device)      # (N, 4, 4)
        self.H, self.W, self.focal = H, W, focal
        self.n_rand = n_rand
        self.precrop_iters, self.precrop_frac = precrop_iters, precrop_frac
        self.device = device

    def sample(self, step: int):
        """Returns rays_o (n_rand, 3), rays_d (n_rand, 3) and target colours (n_rand, 3)."""
        img_i = int(torch.randint(len(self.images), (1,)))
        H, W = self.H, self.W
        if step <= self.precrop_iters:
            dH, dW = int(H // 2 * self.precrop_frac), int(W // 2 * self.precrop_frac)
            row0, col0, h, w = H // 2 - dH, W // 2 - dW, 2 * dH, 2 * dW
        else:
            row0, col0, h, w = 0, 0, H, W

        flat = torch.randperm(h * w, device=self.device)[: self.n_rand]   # without replacement
        rows, cols = row0 + flat // w, col0 + flat % w
        rays_o, rays_d = pixel_rays(rows, cols, H, W, self.focal, self.poses[img_i])
        target = self.images[img_i, rows, cols]
        return rays_o, rays_d, target


# --------------------------------------------------------------------------- video path
def pose_spherical(theta_deg: float, phi_deg: float, radius: float) -> np.ndarray:
    """Camera-to-world matrix on a sphere around the origin, looking at the origin
    (the 360-degree path used by the official code for Blender scenes)."""
    def trans_t(t):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], np.float32)

    def rot_phi(p):
        c, s = math.cos(p), math.sin(p)
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float32)

    def rot_theta(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, 0, -s, 0], [0, 1, 0, 0], [s, 0, c, 0], [0, 0, 0, 1]], np.float32)

    c2w = trans_t(radius)
    c2w = rot_phi(math.radians(phi_deg)) @ c2w
    c2w = rot_theta(math.radians(theta_deg)) @ c2w
    flip = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], np.float32)
    return flip @ c2w
