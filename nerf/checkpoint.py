"""Resumable checkpoints: model + optimizer + step + random-number-generator states.

Training can resume at the latest completed checkpoint.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def _rng_state() -> dict:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {"torch": torch.get_rng_state(),
             "numpy": (name, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, cached),
             "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    if isinstance(keys, torch.Tensor):
        keys = keys.numpy().astype(np.uint32)
    np.random.set_state((name, keys, pos, has_gauss, cached))
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        for device, rng in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng, device)
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(ckpt_dir: Path, step: int, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, keep: int = 3) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": _rng_state()}, tmp)
    tmp.replace(path)          # atomic: a crash never leaves a half-written checkpoint
    if keep > 0:
        for old in sorted(ckpt_dir.glob("step_*.pt"))[:-keep]:
            old.unlink()
    return path


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    return ckpts[-1] if ckpts else None


def load_checkpoint(path: Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    restore_rng: bool = True, *, trust_legacy: bool = False) -> int:
    """Load weights (and optimizer/RNG state if given). Returns the saved step.

    Everything is read onto the CPU first: the random-number states must stay CPU byte
    tensors, and load_state_dict copies the weights to whatever device the model is on.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=not trust_legacy)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if restore_rng:
        _set_rng_state(ckpt["rng"])
    return int(ckpt["step"])
