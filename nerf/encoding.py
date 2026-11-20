"""Positional encoding gamma(p), Eq. 4 of the NeRF paper.

An MLP fed raw (x, y, z) coordinates can only fit smooth functions. Mapping every
coordinate p to sines and cosines of increasing frequency lets the network represent
fine detail:

    gamma(p) = (p, sin(2^0 p), cos(2^0 p), ..., sin(2^(L-1) p), cos(2^(L-1) p))

Two details follow the authors' code rather than the formula in the paper:
the raw input p is kept as the first entry, and the frequencies are 2^k without the
factor pi. With L = 10 a 3-D position becomes 3 + 3*2*10 = 63 numbers; with L = 4 a
viewing direction becomes 3 + 3*2*4 = 27 numbers.
"""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # 1, 2, 4, ..., 2^(L-1). A buffer moves with .to(device) but is not trained.
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs), persistent=False)

    def out_dim(self, in_dim: int = 3) -> int:
        return in_dim * (int(self.include_input) + 2 * self.num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> (..., D * (1 + 2L))."""
        scaled = x[..., None, :] * self.freqs[:, None]                   # (..., L, D)
        enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # (..., L, 2D)
        enc = enc.flatten(start_dim=-2)                                  # (..., 2LD)
        if self.include_input:
            enc = torch.cat([x, enc], dim=-1)
        return enc
