import numpy as np
import torch
from skimage.metrics import structural_similarity

from nerf.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from nerf.config import load_config
from nerf.data import RaySampler
from nerf.metrics import mse_to_psnr, psnr, ssim
from nerf.rendering import NeRFRenderer


def _setup(seed):
    torch.manual_seed(seed)
    cfg = load_config(None, ["netdepth=2", "netwidth=16", "skips=[0]", "n_samples=8",
                             "n_importance=8", "n_rand=32", "precrop_iters=0"])
    model = NeRFRenderer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    sampler = RaySampler(rng.random((2, 8, 8, 3), dtype=np.float32),
                         np.stack([np.eye(4, dtype=np.float32)] * 2), 8, 8, 10.0, n_rand=32)
    return cfg, model, opt, sampler


def _train(model, opt, sampler, steps, start=0):
    for step in range(start + 1, start + steps + 1):
        o, d, rgb = sampler.sample(step)
        out = model(o, d, perturb=True)
        loss = ((out["rgb"] - rgb) ** 2).mean() + ((out["rgb_coarse"] - rgb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_resume_gives_exactly_the_same_run(tmp_path):
    _, a, opt_a, sampler_a = _setup(0)
    _train(a, opt_a, sampler_a, 6)                 # 6 steps in one go

    _, b, opt_b, sampler_b = _setup(0)
    _train(b, opt_b, sampler_b, 3)                 # 3 steps, save, "crash"
    save_checkpoint(tmp_path, 3, b, opt_b)
    _, c, opt_c, sampler_c = _setup(123)           # fresh process with different init
    step = load_checkpoint(latest_checkpoint(tmp_path), c, opt_c)
    _train(c, opt_c, sampler_c, 3, start=step)     # resume for the remaining 3 steps

    for p, q in zip(a.parameters(), c.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_keep_only_newest_checkpoints(tmp_path):
    _, m, opt, _ = _setup(0)
    for step in (10, 20, 30, 40):
        save_checkpoint(tmp_path, step, m, opt, keep=2)
    assert [p.name for p in sorted(tmp_path.glob("step_*.pt"))] == ["step_0000030.pt",
                                                                     "step_0000040.pt"]


def test_psnr_and_ssim():
    assert abs(mse_to_psnr(0.01) - 20.0) < 1e-9
    assert mse_to_psnr(0) == float("inf")
    rng = np.random.default_rng(1)
    gt = rng.random((40, 50, 3)).astype(np.float32)
    pred = np.clip(gt + 0.05 * rng.standard_normal(gt.shape), 0, 1).astype(np.float32)
    ref = structural_similarity(gt, pred, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False)
    ours = ssim(torch.from_numpy(pred), torch.from_numpy(gt))
    assert abs(ours - ref) < 1e-3
    assert psnr(torch.from_numpy(gt), torch.from_numpy(gt) * 0 + 0.5) < 15


def test_checkpoints_are_read_onto_the_cpu(tmp_path, monkeypatch):
    """Regression test for GPU resumes: if a checkpoint were loaded with map_location='cuda',
    the saved RNG states would become CUDA tensors and torch.set_rng_state would fail.
    Loading must always go through the CPU."""
    _, m, opt, _ = _setup(0)
    save_checkpoint(tmp_path, 5, m, opt)
    real_load = torch.load
    seen = []

    def checked_load(*args, **kwargs):
        seen.append(kwargs.get("map_location"))
        assert kwargs.get("weights_only") is True
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    assert load_checkpoint(latest_checkpoint(tmp_path), m, opt) == 5
    assert seen == ["cpu"]
