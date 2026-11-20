import math

import torch

from nerf.rendering import raw2outputs
from nerf.sampling import sample_pdf, stratified_samples


def test_stratified_samples_one_per_bin():
    z = stratified_samples(2.0, 6.0, n_samples=8, n_rays=100, perturb=True)
    assert z.shape == (100, 8)
    assert (z >= 2.0).all() and (z <= 6.0).all()
    assert (z[:, 1:] >= z[:, :-1]).all()                   # still sorted after jitter
    grid = stratified_samples(2.0, 6.0, 8, 1, perturb=False)[0]
    assert torch.allclose(grid, torch.linspace(2.0, 6.0, 8))


def test_sample_pdf_puts_samples_where_the_weight_is():
    bins = torch.linspace(0.0, 1.0, 11).expand(4, 11).contiguous()   # 10 bins of width 0.1
    weights = torch.zeros(4, 10)
    weights[:, 7] = 1.0                                                # all mass in [0.7, 0.8]
    s = sample_pdf(bins, weights, n_samples=256)
    inside = ((s >= 0.7) & (s <= 0.8)).float().mean()
    assert inside > 0.98


def test_sample_pdf_matches_the_target_distribution():
    torch.manual_seed(0)
    bins = torch.linspace(0.0, 1.0, 5)[None]              # 4 bins
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    s = sample_pdf(bins, weights, n_samples=200_000)
    hist = torch.histc(s, bins=4, min=0.0, max=1.0) / s.numel()
    assert torch.allclose(hist, weights[0] / weights.sum(), atol=0.01)


def test_deterministic_sampling_is_sorted():
    bins = torch.linspace(0.0, 1.0, 11).expand(2, 11).contiguous()
    s = sample_pdf(bins, torch.rand(2, 10), n_samples=32, det=True)
    assert (s[:, 1:] >= s[:, :-1]).all()


def test_volume_rendering_of_a_uniform_medium():
    """Constant density sigma over a segment of length L: opacity must be 1 - exp(-sigma L)."""
    sigma, L, n = 0.7, 2.0, 2000
    z = torch.linspace(2.0, 2.0 + L, n + 1)[None]          # n intervals + a far-away last sample
    raw = torch.zeros(1, n + 1, 4)
    raw[:, :n, 3] = sigma
    raw[:, n:, 3] = 0.0                                     # nothing after the segment
    raw[..., :3] = 10.0                                     # sigmoid(10) ~ 1: white medium
    out = raw2outputs(raw, z, torch.tensor([[0.0, 0.0, -1.0]]))
    assert math.isclose(out["acc"].item(), 1 - math.exp(-sigma * L), rel_tol=1e-3)
    assert torch.allclose(out["weights"].sum(-1), out["acc"])


def test_opaque_first_sample_hides_everything_behind_it():
    z = torch.tensor([[2.0, 3.0, 4.0]])
    raw = torch.tensor([[[10.0, -10.0, -10.0, 1e4], [-10.0, 10.0, -10.0, 1e4],
                         [-10.0, -10.0, 10.0, 1e4]]])
    out = raw2outputs(raw, z, torch.tensor([[0.0, 0.0, -1.0]]))
    assert torch.allclose(out["rgb"], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-3)  # red wins
    assert math.isclose(out["depth"].item(), 2.0, rel_tol=1e-4)


def test_white_background_for_empty_space():
    z = torch.linspace(2.0, 6.0, 16)[None]
    raw = torch.zeros(1, 16, 4)
    raw[..., 3] = -5.0                                      # relu -> zero density everywhere
    out = raw2outputs(raw, z, torch.tensor([[0.0, 0.0, -1.0]]), white_bkgd=True)
    assert torch.allclose(out["rgb"], torch.ones(1, 3)) and out["acc"].item() == 0.0
