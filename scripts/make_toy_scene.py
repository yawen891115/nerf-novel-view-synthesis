"""Generate a small synthetic scene in exactly the NeRF-synthetic ("Blender") format.

It lets you test the whole pipeline (loading, rays, training, evaluation, video) in a
few minutes on a CPU before paying for a GPU. Three spheres and a box are ray-traced
analytically with simple diffuse shading and saved as RGBA PNGs plus
transforms_{train,val,test}.json, using the same camera convention and field of view
as the real lego scene.

    python scripts/make_toy_scene.py --out data/toy_scene --res 64
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

CAMERA_ANGLE_X = 0.6911112070083618          # horizontal field of view of the real lego scene
RADIUS = 4.0                                 # camera distance from the origin
LIGHT = np.array([0.5, 0.3, 1.0]) / np.linalg.norm([0.5, 0.3, 1.0])

SPHERES = [  # (centre, radius, colour)
    (np.array([0.0, 0.0, 0.0]), 0.6, np.array([0.90, 0.20, 0.20])),
    (np.array([0.9, 0.3, 0.2]), 0.35, np.array([0.20, 0.80, 0.30])),
    (np.array([-0.6, 0.7, -0.3]), 0.4, np.array([0.20, 0.40, 0.90])),
]
BOX = (np.array([-0.9, -0.9, -0.9]), np.array([-0.3, -0.3, -0.5]), np.array([0.95, 0.80, 0.20]))


def look_at(cam_pos: np.ndarray) -> np.ndarray:
    """Camera-to-world matrix for a camera at cam_pos looking at the origin (world up = +z).
    Camera axes: x = right, y = up, z = backwards (the camera looks down -z)."""
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -forward, cam_pos
    return c2w


def trace(origins: np.ndarray, dirs: np.ndarray):
    """Closest hit for each ray. Returns (hit mask (N,), shaded colour (N, 3))."""
    n = len(dirs)
    best_t = np.full(n, np.inf)
    color = np.zeros((n, 3))
    normal = np.zeros((n, 3))

    for centre, radius, albedo in SPHERES:
        oc = origins - centre
        b = np.sum(dirs * oc, axis=1)
        disc = b * b - (np.sum(oc * oc, axis=1) - radius ** 2)
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        hit = (disc > 0) & (t > 0) & (t < best_t)
        best_t[hit] = t[hit]
        p = origins[hit] + t[hit, None] * dirs[hit]
        normal[hit] = (p - centre) / radius
        color[hit] = albedo

    lo, hi, albedo = BOX
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - origins) / dirs, (hi - origins) / dirs
    t_near, t_far = np.minimum(t1, t2), np.maximum(t1, t2)
    t_enter, t_exit = t_near.max(axis=1), t_far.min(axis=1)
    hit = (t_exit >= np.maximum(t_enter, 0)) & (t_enter > 0) & (t_enter < best_t)
    best_t[hit] = t_enter[hit]
    axis = t_near.argmax(axis=1)
    box_normal = np.zeros((n, 3))
    box_normal[np.arange(n), axis] = -np.sign(dirs[np.arange(n), axis])
    normal[hit] = box_normal[hit]
    color[hit] = albedo

    mask = np.isfinite(best_t)
    shade = 0.25 + 0.75 * np.clip(normal @ LIGHT, 0.0, None)
    return mask, color * shade[:, None]


def render(c2w: np.ndarray, res: int, supersample: int = 3) -> np.ndarray:
    """Render an RGBA uint8 image. Pixel (row j, col i) is centred on (i, j), which matches
    the integer pixel coordinates used for rays in nerf/data.py."""
    focal = 0.5 * res / math.tan(0.5 * CAMERA_ANGLE_X)
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    jj, ii, oy, ox = np.meshgrid(np.arange(res), np.arange(res), offsets, offsets, indexing="ij")
    x, y = ii + ox, jj + oy
    dirs_cam = np.stack([(x - 0.5 * res) / focal, -(y - 0.5 * res) / focal, -np.ones_like(x)], -1)
    dirs = dirs_cam.reshape(-1, 3) @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(c2w[:3, 3], dirs.shape)

    mask, color = trace(origins, dirs)
    mask = mask.reshape(res, res, -1)
    color = color.reshape(res, res, -1, 3)
    coverage = mask.mean(axis=2)
    hits = np.maximum(mask.sum(axis=2), 1)[..., None]
    straight_rgb = (color * mask[..., None]).sum(axis=2) / hits      # average colour of the hits
    rgba = np.concatenate([straight_rgb, coverage[..., None]], axis=-1)
    return (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)


def camera_positions(split: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if split == "test":                  # circle at 30 degrees elevation, like lego's test set
        az = np.linspace(0, 2 * np.pi, n, endpoint=False)
        el = np.full(n, math.radians(30.0))
    else:                                      # random views on the upper hemisphere
        az = rng.uniform(0, 2 * np.pi, n)
        el = np.arcsin(rng.uniform(0.1, 0.95, n))
    return RADIUS * np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/toy_scene")
    parser.add_argument("--res", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_val", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        (out / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for k, pos in enumerate(camera_positions(split, n, rng)):
            c2w = look_at(pos)
            imageio.imwrite(out / split / f"r_{k}.png", render(c2w, args.res))
            frames.append({"file_path": f"./{split}/r_{k}", "transform_matrix": c2w.tolist()})
        with open(out / f"transforms_{split}.json", "w") as f:
            json.dump({"camera_angle_x": CAMERA_ANGLE_X, "frames": frames}, f, indent=2)
        print(f"{split}: {n} images of {args.res}x{args.res} -> {out / split}")


if __name__ == "__main__":
    main()
