"""Experiment settings.

Every run is described by one `NeRFConfig`. Values come from a YAML file in `configs/`
and can be overridden on the command line, for example:

    python train.py --config configs/lego_paper.yaml --set n_iters=200000 expname=lego_200k
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class NeRFConfig:
    # ----- experiment -----
    expname: str = "lego"
    logdir: str = "logs"          # checkpoints, TensorBoard files and renders go to logdir/expname
    seed: int = 0

    # ----- data (NeRF-synthetic / "Blender" format) -----
    datadir: str = "data/nerf_synthetic/lego"
    half_res: bool = False        # True: 400x400 instead of 800x800 (faster, not the paper setting)
    white_bkgd: bool = True       # composite RGBA images onto white, as in the paper
    near: float = 2.0             # t_n in Eq. 1
    far: float = 6.0              # t_f in Eq. 1
    testskip: int = 1             # evaluate on every k-th test image (1 = all 200)

    # ----- model (Fig. 7 of the paper) -----
    netdepth: int = 8             # number of 256-wide layers before the density output
    netwidth: int = 256
    skips: tuple = (4,)           # re-inject the encoded input after this layer
    multires: int = 10            # L for positions, Eq. 4
    multires_views: int = 4       # L for viewing directions

    # ----- sampling (Sec. 5.2) -----
    n_samples: int = 64           # N_c: stratified samples for the coarse network
    n_importance: int = 128       # N_f: extra samples for the fine network (0 = no fine network)
    perturb: bool = True          # stratified jitter during training
    raw_noise_std: float = 0.0    # noise added to raw density (used for real scenes, not Blender)

    # ----- optimization (Sec. 5.3) -----
    n_iters: int = 500_000
    n_rand: int = 1024            # rays per step (official Blender config; paper text: 4096)
    lrate: float = 5e-4           # start learning rate
    lrate_final: float = 5e-5     # decays exponentially to this value ...
    lrate_decay_steps: int = 500_000  # ... at this step
    adam_eps: float = 1e-7        # paper: TensorFlow's Adam default
    precrop_iters: int = 500      # first steps sample only the image centre (avoids all-white)
    precrop_frac: float = 0.5

    # ----- memory: rays and points processed at once -----
    ray_chunk: int = 8192         # rays per chunk when rendering full images
    point_chunk: int = 16384      # points per chunk when calling the MLP

    # ----- logging and checkpoints -----
    i_print: int = 100            # console + TensorBoard scalars
    i_img: int = 5000             # render one validation image into TensorBoard
    i_weights: int = 10000        # save a resumable checkpoint
    keep_ckpts: int = 3           # keep the newest checkpoints; zero keeps all
    tf32: bool = True             # allow TF32 matmuls on Ampere/Ada GPUs (big speed-up, tiny error)

    @property
    def expdir(self) -> Path:
        return Path(self.logdir) / self.expname


def _parse_value(text: str):
    """Turn 'true', '3', '1e-4', '[4]' ... into Python values the same way YAML does."""
    return yaml.safe_load(text)


def load_config(path: str | None, overrides: list[str] | None = None) -> NeRFConfig:
    """Read a YAML file, apply `key=value` overrides and return a checked NeRFConfig."""
    values: dict = {}
    if path is not None:
        with open(path) as f:
            values.update(yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, text = item.split("=", 1)
        values[key.strip()] = _parse_value(text)

    known = {f.name for f in dataclasses.fields(NeRFConfig)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    if "skips" in values:
        values["skips"] = tuple(values["skips"])
    return NeRFConfig(**values)


def save_config(cfg: NeRFConfig, path: Path) -> None:
    data = dataclasses.asdict(cfg)
    data["skips"] = list(data["skips"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)
