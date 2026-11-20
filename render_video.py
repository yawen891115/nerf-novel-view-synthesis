"""Render a 360-degree fly-around video of a trained scene (the path used by the official code).

    python render_video.py --expdir logs/lego_paper --frames 120

Saves <expdir>/video_<step>.mp4 (needs imageio-ffmpeg; falls back to a GIF without it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from nerf.checkpoint import latest_checkpoint, load_checkpoint
from nerf.config import load_config
from nerf.data import load_blender_split, pose_spherical
from nerf.rendering import NeRFRenderer
from nerf.utils import pick_device, to8b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elevation", type=float, default=-30.0, help="phi in degrees")
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-legacy-checkpoint", action="store_true",
                        help="allow pickle loading only for an old checkpoint you trust")
    args = parser.parse_args()

    expdir = Path(args.expdir)
    cfg = load_config(str(expdir / "config.yaml"))
    device = pick_device(args.device)
    renderer = NeRFRenderer(cfg).to(device).eval()
    ckpt = latest_checkpoint(expdir / "checkpoints")
    if ckpt is None:
        raise SystemExit(f"no checkpoint found in {expdir / 'checkpoints'}")
    step = load_checkpoint(ckpt, renderer, restore_rng=False,
                           trust_legacy=args.trust_legacy_checkpoint)
    _, _, H, W, focal, _ = load_blender_split(cfg.datadir, "test", cfg.half_res,
                                              cfg.white_bkgd, skip=1000)  # only intrinsics

    frames = []
    for theta in tqdm(np.linspace(-180, 180, args.frames + 1)[:-1]):
        c2w = torch.from_numpy(pose_spherical(theta, args.elevation, args.radius)).to(device)
        frames.append(to8b(renderer.render_image(H, W, focal, c2w)["rgb"]))

    out = expdir / f"video_{step:07d}.mp4"
    try:
        imageio.mimwrite(out, frames, fps=args.fps, quality=8)
    except Exception as err:  # no ffmpeg backend installed
        out = out.with_suffix(".gif")
        print(f"mp4 failed ({err}); writing {out.name} instead")
        imageio.mimwrite(out, frames, duration=1000 / args.fps, loop=0)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
