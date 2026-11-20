"""Differentiable volume rendering (Sec. 4) and the coarse-to-fine renderer (Sec. 5.2).

For one ray with samples t_1 < ... < t_N, densities sigma_i and colours c_i:

    delta_i = t_{i+1} - t_i                         distance between samples
    alpha_i = 1 - exp(-sigma_i * delta_i)           chance the ray stops in segment i
    T_i     = prod_{j<i} (1 - alpha_j)              chance the ray reaches segment i
    w_i     = T_i * alpha_i                         contribution of sample i
    C(r)    = sum_i w_i * c_i                       rendered colour (Eq. 3)

Every step is made of differentiable tensor operations, so the photometric loss on C(r)
back-propagates into the MLP.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import NeRFConfig
from .data import get_rays
from .encoding import PositionalEncoding
from .model import NeRF
from .sampling import sample_pdf, stratified_samples


def raw2outputs(raw: torch.Tensor, z_vals: torch.Tensor, rays_d: torch.Tensor,
                raw_noise_std: float = 0.0, white_bkgd: bool = False) -> dict:
    """Turn raw network outputs along rays into pixel values.

    raw:    (R, S, 4) raw [r, g, b, sigma] at each sample
    z_vals: (R, S)    sample depths t_i
    rays_d: (R, 3)    ray directions (not normalised)
    """
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)  # last: "infinite"
    deltas = deltas * rays_d.norm(dim=-1, keepdim=True)    # t-units -> world distance

    rgb = torch.sigmoid(raw[..., :3])
    sigma = raw[..., 3]
    if raw_noise_std > 0.0:
        sigma = sigma + torch.randn_like(sigma) * raw_noise_std
    alpha = 1.0 - torch.exp(-F.relu(sigma) * deltas)

    # T_i: exclusive cumulative product of (1 - alpha). The 1e-10 keeps gradients finite.
    ones = torch.ones_like(alpha[:, :1])
    trans = torch.cumprod(torch.cat([ones, 1.0 - alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
    weights = alpha * trans                                  # (R, S)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)         # (R, 3)
    depth_map = (weights * z_vals).sum(dim=-1)               # expected depth
    acc_map = weights.sum(dim=-1)                            # opacity, in [0, 1]
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])      # empty space shows white
    return {"rgb": rgb_map, "depth": depth_map, "acc": acc_map, "weights": weights}


class NeRFRenderer(nn.Module):
    """Holds the coarse and fine MLPs and renders batches of rays with them."""

    def __init__(self, cfg: NeRFConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_pos = PositionalEncoding(cfg.multires)
        self.embed_dir = PositionalEncoding(cfg.multires_views)
        dims = dict(depth=cfg.netdepth, width=cfg.netwidth, skips=cfg.skips,
                    pos_dim=self.embed_pos.out_dim(3), dir_dim=self.embed_dir.out_dim(3))
        self.coarse = NeRF(**dims)
        self.fine = NeRF(**dims) if cfg.n_importance > 0 else None

    # ------------------------------------------------------------------ helpers
    def _query(self, net: NeRF, pts: torch.Tensor, viewdirs: torch.Tensor) -> torch.Tensor:
        """Run the MLP on (R, S, 3) points in chunks of `point_chunk` to bound memory."""
        flat_pts = pts.reshape(-1, 3)
        flat_dirs = viewdirs[:, None, :].expand_as(pts).reshape(-1, 3)
        chunk = self.cfg.point_chunk
        outs = [net(self.embed_pos(flat_pts[i:i + chunk]), self.embed_dir(flat_dirs[i:i + chunk]))
                for i in range(0, flat_pts.shape[0], chunk)]
        return torch.cat(outs, dim=0).reshape(*pts.shape[:-1], 4)

    # ------------------------------------------------------------------ main API
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, perturb: bool = False) -> dict:
        """Render R rays. Returns fine results as 'rgb'/'depth'/'acc' and coarse as '*_coarse'."""
        cfg = self.cfg
        n_rays = rays_o.shape[0]
        viewdirs = F.normalize(rays_d, dim=-1)       # the MLP gets unit view directions
        noise = cfg.raw_noise_std if perturb else 0.0

        # 1) Coarse pass: stratified samples.
        z_vals = stratified_samples(cfg.near, cfg.far, cfg.n_samples, n_rays, perturb,
                                    rays_o.device)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]    # r(t) = o + t d
        coarse = raw2outputs(self._query(self.coarse, pts, viewdirs), z_vals, rays_d,
                             noise, cfg.white_bkgd)
        out = {"rgb_coarse": coarse["rgb"], "depth_coarse": coarse["depth"],
               "acc_coarse": coarse["acc"]}
        if self.fine is None:
            out.update(rgb=coarse["rgb"], depth=coarse["depth"], acc=coarse["acc"])
            return out

        # 2) Fine pass: more samples where the coarse weights are large.
        z_mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        z_fine = sample_pdf(z_mids, coarse["weights"][:, 1:-1], cfg.n_importance, det=not perturb)
        z_fine = z_fine.detach()                     # no gradient through the sampling itself
        z_all, _ = torch.sort(torch.cat([z_vals, z_fine], dim=-1), dim=-1)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_all[..., None]
        fine = raw2outputs(self._query(self.fine, pts, viewdirs), z_all, rays_d,
                           noise, cfg.white_bkgd)
        out.update(rgb=fine["rgb"], depth=fine["depth"], acc=fine["acc"])
        return out

    @torch.no_grad()
    def render_image(self, H: int, W: int, focal: float, c2w: torch.Tensor) -> dict:
        """Render a full H x W image, `ray_chunk` rays at a time (deterministic sampling)."""
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
        chunk = self.cfg.ray_chunk
        pieces = [self.forward(rays_o[i:i + chunk], rays_d[i:i + chunk], perturb=False)
                  for i in range(0, rays_o.shape[0], chunk)]
        keys = ("rgb", "depth", "acc")
        return {k: torch.cat([p[k] for p in pieces], 0).reshape(H, W, -1).squeeze(-1)
                for k in keys}
