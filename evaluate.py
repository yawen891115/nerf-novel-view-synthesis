"""Render views of a trained model and report PSNR / SSIM / LPIPS.

    python evaluate.py --expdir logs/lego_paper                 # all 200 test images
    python evaluate.py --expdir logs/lego_paper --testskip 8    # quick check on 25 images
    python evaluate.py --expdir logs/lego_paper --lpips none    # skip LPIPS

Writes renders and metrics.json to a new directory for each evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from nerf.checkpoint import latest_checkpoint, load_checkpoint
from nerf.config import load_config
from nerf.data import load_blender_split
from nerf.metrics import LPIPS, psnr, ssim
from nerf.rendering import NeRFRenderer
from nerf.utils import depth_to_gray, pick_device, to8b


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expdir", required=True, help="logs/<expname> of a training run")
    parser.add_argument("--ckpt", default=None, help="checkpoint file (default: newest)")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true",
                        help="allow pickle loading only for an old checkpoint you trust")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--testskip", type=int, default=None, help="use every k-th image")
    parser.add_argument("--lpips", default="vgg", choices=["vgg", "alex", "none"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set ray_chunk=4096")
    args = parser.parse_args()

    expdir = Path(args.expdir)
    cfg = load_config(str(expdir / "config.yaml"), args.set)
    device = pick_device(args.device)
    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    renderer = NeRFRenderer(cfg).to(device).eval()
    ckpt = Path(args.ckpt) if args.ckpt else latest_checkpoint(expdir / "checkpoints")
    if ckpt is None:
        raise SystemExit(f"no checkpoint found in {expdir / 'checkpoints'}")
    step = load_checkpoint(ckpt, renderer, restore_rng=False,
                           trust_legacy=args.trust_legacy_checkpoint)

    skip = cfg.testskip if args.testskip is None else args.testskip
    if skip < 1:
        parser.error("--testskip must be at least 1")
    manifest = Path(cfg.datadir) / f"transforms_{args.split}.json"
    checkpoint_hash, manifest_hash = file_sha256(ckpt), file_sha256(manifest)
    images, poses, H, W, focal, names = load_blender_split(cfg.datadir, args.split,
                                                           cfg.half_res, cfg.white_bkgd, skip)
    outdir = Path(tempfile.mkdtemp(prefix=f"eval_{args.split}_{step:07d}_", dir=expdir))
    lpips_fn = None if args.lpips == "none" else LPIPS(args.lpips, device)
    print(f"checkpoint step {step}: rendering {len(images)} {args.split} images at {H}x{W}")

    rows = []
    t0 = time.time()
    for k in tqdm(range(len(images))):
        res = renderer.render_image(H, W, focal, torch.from_numpy(poses[k]).to(device))
        pred, gt = res["rgb"].clamp(0, 1), torch.from_numpy(images[k]).to(device)
        row = {"image": names[k], "psnr": psnr(pred, gt), "ssim": ssim(pred, gt)}
        if lpips_fn is not None:
            row["lpips"] = lpips_fn(pred, gt)
        rows.append(row)
        imageio.imwrite(outdir / f"{names[k]}.png", to8b(pred))
        imageio.imwrite(outdir / f"{names[k]}_depth.png", depth_to_gray(res["depth"], cfg.near,
                                                                        cfg.far))
    seconds = time.time() - t0

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "image"}
    report = {"checkpoint": str(ckpt), "checkpoint_sha256": checkpoint_hash,
              "split_manifest_sha256": manifest_hash, "step": step, "split": args.split,
              "testskip": skip, "lpips_backbone": None if lpips_fn is None else args.lpips,
              "device": str(device), "torch_version": str(torch.__version__),
              "config": asdict(cfg), "num_images": len(rows),
              "resolution": [H, W], "seconds_per_image": seconds / len(rows),
              "mean": {k: v if math.isfinite(v) else str(v) for k, v in summary.items()},
              "per_image": [{k: str(v) if isinstance(v, float) and not math.isfinite(v) else v
                             for k, v in row.items()} for row in rows]}
    with open(outdir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("mean:", "  ".join(f"{k.upper()} {v:.4f}" for k, v in summary.items()))
    print(f"saved renders and metrics.json to {outdir}")


if __name__ == "__main__":
    main()
