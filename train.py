"""Train NeRF on one NeRF-synthetic scene.

    python train.py --config configs/lego_paper.yaml
    python train.py --config configs/lego_paper.yaml --set n_iters=200000 expname=lego_200k

Re-running the same command resumes from the newest checkpoint in logs/<expname>/checkpoints.
Watch progress with:  tensorboard --logdir logs
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nerf.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from nerf.config import load_config, save_config
from nerf.data import RaySampler, load_blender_split
from nerf.metrics import mse, mse_to_psnr, psnr
from nerf.rendering import NeRFRenderer
from nerf.utils import depth_to_gray, pick_device, seed_everything


def check_resume_config(cfg, path) -> None:
    """Reject changes to the data/model/optimization recipe for an existing run."""
    if not path.is_file():
        raise SystemExit(f"Cannot resume without {path}; use a new expname.")
    previous = load_config(str(path))
    adjustable = {"n_iters", "i_print", "i_img", "i_weights", "keep_ckpts"}
    changed = [f.name for f in fields(cfg) if f.name not in adjustable
               and getattr(cfg, f.name) != getattr(previous, f.name)]
    if changed:
        raise SystemExit(f"Resume configuration differs ({', '.join(changed)}); use a new expname.")


def learning_rate(cfg, step: int) -> float:
    """Exponential decay from lrate to lrate_final over lrate_decay_steps (Sec. 5.3)."""
    ratio = cfg.lrate_final / cfg.lrate
    return cfg.lrate * ratio ** (step / cfg.lrate_decay_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML file in configs/")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set n_iters=1000 half_res=true")
    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true",
                        help="allow pickle loading only for an old checkpoint you trust")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    device = pick_device(args.device)
    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(cfg.seed)
    expdir, ckpt_dir = cfg.expdir, cfg.expdir / "checkpoints"
    config_path = expdir / "config.yaml"
    last = latest_checkpoint(ckpt_dir)
    if last is not None:
        check_resume_config(cfg, config_path)
    print(f"device={device}  experiment dir={expdir}")

    # ---------------------------------------------------------------- data
    images, poses, H, W, focal, _ = load_blender_split(cfg.datadir, "train", cfg.half_res,
                                                       cfg.white_bkgd)
    val_imgs, val_poses, val_H, val_W, val_focal, _ = load_blender_split(
        cfg.datadir, "val", cfg.half_res, cfg.white_bkgd, skip=8)
    print(f"train images: {len(images)} at {H}x{W}, focal={focal:.2f}px")
    sampler = RaySampler(images, poses, H, W, focal, cfg.n_rand, cfg.precrop_iters,
                         cfg.precrop_frac, device)

    # ---------------------------------------------------------------- model
    renderer = NeRFRenderer(cfg).to(device)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=cfg.lrate, betas=(0.9, 0.999),
                                 eps=cfg.adam_eps)
    n_params = sum(p.numel() for p in renderer.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M (coarse + fine)")

    start = 0
    if last is not None:
        start = load_checkpoint(last, renderer, optimizer, trust_legacy=args.trust_legacy_checkpoint)
        if cfg.n_iters < start:
            raise SystemExit(f"n_iters ({cfg.n_iters}) is below the saved step ({start}).")
        print(f"resumed from {last.name} (step {start})")
    save_config(cfg, config_path)

    writer = SummaryWriter(str(expdir / "tensorboard"))
    pbar = tqdm(range(start + 1, cfg.n_iters + 1), initial=start, total=cfg.n_iters,
                dynamic_ncols=True)
    t0 = time.time()
    try:
        for step in pbar:
            renderer.train()
            rays_o, rays_d, target = sampler.sample(step)
            out = renderer(rays_o, rays_d, perturb=cfg.perturb)

            # Loss of Sec. 5.3: squared error of the coarse AND the fine rendering.
            loss_fine = mse(out["rgb"], target)
            loss_coarse = mse(out["rgb_coarse"], target)
            loss = loss_fine + loss_coarse if renderer.fine is not None else loss_fine

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lr = learning_rate(cfg, step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step % cfg.i_print == 0:
                train_psnr = mse_to_psnr(loss_fine.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{train_psnr:.2f}")
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/psnr", train_psnr, step)
                writer.add_scalar("train/psnr_coarse", mse_to_psnr(loss_coarse.item()), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/iters_per_sec", cfg.i_print / (time.time() - t0), step)
                t0 = time.time()

            if step % cfg.i_img == 0:
                renderer.eval()
                k = (step // cfg.i_img) % len(val_imgs)
                pose = torch.from_numpy(val_poses[k]).to(device)
                res = renderer.render_image(val_H, val_W, val_focal, pose)
                gt = torch.from_numpy(val_imgs[k]).to(device)
                writer.add_scalar("val/psnr", psnr(res["rgb"], gt), step)
                writer.add_image("val/rgb", res["rgb"].clamp(0, 1), step, dataformats="HWC")
                writer.add_image("val/depth", depth_to_gray(res["depth"], cfg.near, cfg.far),
                                 step, dataformats="HW")

            if step % cfg.i_weights == 0 or step == cfg.n_iters:
                save_checkpoint(ckpt_dir, step, renderer, optimizer, cfg.keep_ckpts)
    except KeyboardInterrupt:
        saved = latest_checkpoint(ckpt_dir)
        print(f"\nInterrupted. Restart from {saved}." if saved else
              "\nInterrupted before the first checkpoint; rerun to start over.")
        return
    finally:
        writer.close()
    print(f"done. checkpoints in {ckpt_dir}. Next: python evaluate.py --expdir {expdir}")


if __name__ == "__main__":
    main()
