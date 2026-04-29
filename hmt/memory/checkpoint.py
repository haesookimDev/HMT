"""Stage 7.2: NVMe checkpoint-only policy.

Persists training state to a single ``.pt`` file for crash-recovery resume.
Captures:

  - model ``state_dict`` (works for ``CompressedLinear`` too — same params)
  - optimizer ``state_dict`` (PyTorch standard; low-rank state shapes too)
  - projector snapshots (for ``LowRankAdamW`` — P, Q tensors keyed by
    parameter qualified name)
  - training step counter + run identity
  - Python / numpy / torch RNG states

NOT captured: data-loader iterator position. Resume re-seeds and re-streams
from the dataset's natural ordering. For exact same-batch resume, freeze the
data with non-streaming + a fixed seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hmt.optim.lowrank_adamw import LowRankAdamW
from hmt.optim.projector import LayerProjector


@dataclass
class CheckpointMeta:
    step: int
    config_name: str = ""


def _gather_projectors(model: torch.nn.Module, optim: LowRankAdamW) -> dict[str, dict]:
    """Snapshot ``optim``'s projectors keyed by parameter qualified name.

    ``id(param)`` is unstable across processes; qualified names are durable.
    """
    out: dict[str, dict] = {}
    for name, p in model.named_parameters():
        proj = optim.projectors.get(id(p))
        if proj is None:
            continue
        out[name] = {
            "mode": proj.mode,
            "rank": int(proj.rank),
            "out_dim": int(proj.out_dim),
            "in_dim": int(proj.in_dim),
            "P": proj.P.detach().cpu() if proj.P is not None else None,
            "Q": proj.Q.detach().cpu() if proj.Q is not None else None,
            "method": getattr(proj, "method", "full"),
        }
    return out


def _restore_projectors(
    model: torch.nn.Module,
    optim: LowRankAdamW,
    snap: dict[str, dict],
) -> int:
    name_to_param = dict(model.named_parameters())
    n = 0
    for name, info in snap.items():
        p = name_to_param.get(name)
        if p is None:
            continue
        proj = LayerProjector(
            mode=info["mode"],
            rank=int(info["rank"]),
            out_dim=int(info["out_dim"]),
            in_dim=int(info["in_dim"]),
            P=info["P"].to(device=p.device, dtype=p.dtype) if info["P"] is not None else None,
            Q=info["Q"].to(device=p.device, dtype=p.dtype) if info["Q"] is not None else None,
            method=info.get("method", "full"),
        )
        optim.attach_projector(p, proj)
        n += 1
    return n


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config_name: str = "",
) -> None:
    """Write a complete training-state checkpoint to ``path`` (.pt)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict = {
        "step": int(step),
        "config_name": config_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "projectors": (
            _gather_projectors(model, optimizer)
            if isinstance(optimizer, LowRankAdamW)
            else None
        ),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: Optional[str | torch.device] = None,
) -> CheckpointMeta:
    """Restore training state from ``path``. ``model``/``optimizer`` should
    already be constructed with the same architecture/hyper-params used at save."""
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Attach projectors BEFORE loading optimizer state. ``attach_projector``
    # clears state for safety on live retrofits, but here state is empty so
    # the clear is a no-op; the subsequent ``load_state_dict`` then populates
    # state with the saved low-rank shapes.
    if isinstance(optimizer, LowRankAdamW) and ckpt.get("projectors"):
        _restore_projectors(model, optimizer, ckpt["projectors"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    rng = ckpt.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return CheckpointMeta(
        step=int(ckpt["step"]),
        config_name=str(ckpt.get("config_name", "")),
    )
