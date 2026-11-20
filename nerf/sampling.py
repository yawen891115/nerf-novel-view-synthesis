"""Where along each ray we evaluate the network (Sec. 4 and Sec. 5.2).

* `stratified_samples`: split [near, far] into N_c even bins and draw one random point
  in each bin (Eq. 2). Random jitter means the MLP sees continuous positions during
  training instead of a fixed grid.
* `sample_pdf`: hierarchical sampling. The coarse network's weights w_i say where the
  ray hits matter; we normalise them into a piecewise-constant PDF along the ray and
  draw N_f more samples from it by inverse-transform sampling (Eq. 5).
"""

import torch


def stratified_samples(near: float, far: float, n_samples: int, n_rays: int,
                       perturb: bool, device=None) -> torch.Tensor:
    """Return sorted depths t of shape (n_rays, n_samples)."""
    steps = torch.linspace(0.0, 1.0, n_samples, device=device)
    z = near * (1.0 - steps) + far * steps            # evenly spaced, (S,)
    z = z.expand(n_rays, n_samples)
    if perturb:
        # Jitter every sample uniformly inside its own bin (as in the official code).
        mids = 0.5 * (z[:, 1:] + z[:, :-1])
        upper = torch.cat([mids, z[:, -1:]], dim=-1)
        lower = torch.cat([z[:, :1], mids], dim=-1)
        z = lower + (upper - lower) * torch.rand_like(z)
    return z


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, n_samples: int,
               det: bool = False) -> torch.Tensor:
    """Inverse-transform sampling from a piecewise-constant PDF.

    bins:    (R, B+1) bin edges along each ray.
    weights: (R, B)   unnormalised probability of each bin.
    Returns  (R, n_samples) new depths (not sorted).
    """
    weights = weights + 1e-5                                  # avoid division by zero
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)   # (R, B+1), starts at 0

    if det:   # evenly spaced quantiles (used at test time)
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device).expand(cdf.shape[0], -1)
    else:     # random quantiles (used for training)
        u = torch.rand(cdf.shape[0], n_samples, device=bins.device)
    u = u.contiguous()

    # For each u find the CDF interval [cdf[below], cdf[above]] that contains it.
    idx = torch.searchsorted(cdf, u, right=True)
    below = (idx - 1).clamp(min=0)
    above = idx.clamp(max=cdf.shape[-1] - 1)
    cdf_lo, cdf_hi = cdf.gather(-1, below), cdf.gather(-1, above)
    bin_lo, bin_hi = bins.gather(-1, below), bins.gather(-1, above)

    # Linear interpolation inside the bin.
    denom = cdf_hi - cdf_lo
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    frac = (u - cdf_lo) / denom
    return bin_lo + frac * (bin_hi - bin_lo)
