"""The NeRF MLP F_theta: (encoded position, encoded direction) -> (RGB, density).

Architecture from Fig. 7 of the paper:

    gamma(x) -> 8 x [Linear(256) + ReLU], with gamma(x) concatenated again after layer 5
             -> density sigma (1 number; depends on position only)
             -> 256-d feature (no activation)
    [feature, gamma(d)] -> Linear(128) + ReLU -> RGB (3 numbers)

The network returns raw values; `rendering.raw2outputs` applies ReLU to sigma and a
sigmoid to RGB, exactly where the official code applies them.
"""

import torch
from torch import nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, depth: int = 8, width: int = 256, pos_dim: int = 63,
                 dir_dim: int = 27, skips: tuple = (4,)):
        super().__init__()
        self.skips = set(skips)

        layers = []
        in_dim = pos_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            # The layer after a skip receives [gamma(x), h], so it is wider.
            in_dim = width + pos_dim if i in self.skips else width
        self.pts_linears = nn.ModuleList(layers)

        self.sigma_linear = nn.Linear(width, 1)
        self.feature_linear = nn.Linear(width, width)
        self.view_linear = nn.Linear(width + dir_dim, width // 2)
        self.rgb_linear = nn.Linear(width // 2, 3)

    def forward(self, x_enc: torch.Tensor, d_enc: torch.Tensor) -> torch.Tensor:
        """x_enc: (N, pos_dim), d_enc: (N, dir_dim) -> raw (N, 4) = [r, g, b, sigma]."""
        h = x_enc
        for i, layer in enumerate(self.pts_linears):
            h = F.relu(layer(h))
            if i in self.skips:
                h = torch.cat([x_enc, h], dim=-1)

        sigma = self.sigma_linear(h)          # density depends on position only
        feature = self.feature_linear(h)
        h = F.relu(self.view_linear(torch.cat([feature, d_enc], dim=-1)))
        rgb = self.rgb_linear(h)              # colour may also depend on view direction
        return torch.cat([rgb, sigma], dim=-1)
