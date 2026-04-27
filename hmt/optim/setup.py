"""Helpers to wire up LowRankAdamW against an HF model.

Selects target ``nn.Linear`` modules by regex on qualified module name, then
attaches / refreshes SVD-based projectors using their current ``.weight.grad``.
"""
from __future__ import annotations

import re
from typing import Iterable

import torch

from hmt.optim.lowrank_adamw import LowRankAdamW
from hmt.optim.projector import LayerProjector, ProjectionMode, make_projector_from_grad


def select_target_params(
    model: torch.nn.Module,
    *,
    pattern: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return ``(qualified_name, weight)`` for ``nn.Linear`` modules whose
    qualified name matches ``pattern`` (``re.search`` semantics)."""
    pat = re.compile(pattern)
    out: list[tuple[str, torch.nn.Parameter]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not pat.search(name):
            continue
        if not mod.weight.requires_grad:
            continue
        out.append((name, mod.weight))
    return out


def attach_projectors_from_grads(
    optim: LowRankAdamW,
    target_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    mode: ProjectionMode,
    rank: int,
) -> dict[str, LayerProjector]:
    """Build an SVD-based projector from each target's current gradient and
    attach it to ``optim``. Returns name->projector for logging."""
    out: dict[str, LayerProjector] = {}
    for name, p in target_params:
        if p.grad is None or p.ndim != 2:
            continue
        proj = make_projector_from_grad(p.grad, mode=mode, rank=rank)
        optim.attach_projector(p, proj)
        out[name] = proj
    return out


def refresh_projectors_from_grads(
    optim: LowRankAdamW,
    target_params: Iterable[tuple[str, torch.nn.Parameter]],
) -> int:
    """Recompute SVD bases for already-attached projectors using current
    gradients. Returns the number refreshed."""
    n = 0
    for _name, p in target_params:
        proj = optim.projectors.get(id(p))
        if proj is None or p.grad is None:
            continue
        proj.refresh_(p.grad)
        n += 1
    return n
