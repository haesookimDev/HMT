"""Helpers to wire up LowRankAdamW against an HF model.

Selects target ``nn.Linear`` modules by regex on qualified module name, then
attaches / refreshes SVD-based projectors using their current ``.weight.grad``.
Supports either a fixed per-layer rank or an ``EnergyRankScheduler``.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import torch

from hmt.optim.lowrank_adamw import LowRankAdamW
from hmt.optim.projector import (
    LayerProjector,
    ProjectionMode,
    SVDMethod,
    make_projector_from_grad,
    make_projector_with_scheduler,
)
from hmt.optim.rank_scheduler import EnergyRankScheduler


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
    rank: Optional[int] = None,
    scheduler: Optional[EnergyRankScheduler] = None,
    method: SVDMethod = "full",
) -> dict[str, LayerProjector]:
    """Build an SVD-based projector from each target's current gradient.

    Provide either ``rank`` (fixed) or ``scheduler`` (energy-based per-layer).
    """
    if (rank is None) == (scheduler is None):
        raise ValueError("specify exactly one of `rank` or `scheduler`")
    out: dict[str, LayerProjector] = {}
    for name, p in target_params:
        if p.grad is None or p.ndim != 2:
            continue
        if scheduler is not None:
            proj = make_projector_with_scheduler(
                p.grad, mode=mode, scheduler=scheduler, method=method,
            )
        else:
            proj = make_projector_from_grad(
                p.grad, mode=mode, rank=int(rank), method=method,
            )
        optim.attach_projector(p, proj)
        out[name] = proj
    return out


def refresh_projectors_from_grads(
    optim: LowRankAdamW,
    target_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    align_state: bool = True,
) -> int:
    """Recompute SVD bases for already-attached projectors using current
    gradients. Returns the number refreshed.

    With ``align_state=True`` (default), ``m`` and ``v`` are rotated from the
    pre-refresh (P, Q) basis into the new one so that EMA history is preserved
    instead of becoming meaningless coordinates in a stale basis. This is the
    BACKLOG F.1 fix; disable by setting ``align_state=False`` to reproduce the
    old (pre-fix) behavior.
    """
    n = 0
    for _name, p in target_params:
        proj = optim.projectors.get(id(p))
        if proj is None or p.grad is None:
            continue
        if align_state:
            P_old = proj.P.clone() if proj.P is not None else None
            Q_old = proj.Q.clone() if proj.Q is not None else None
            proj.refresh_(p.grad)
            optim.realign_state(p, P_old, Q_old, proj.P, proj.Q, proj.mode)
        else:
            proj.refresh_(p.grad)
        n += 1
    return n
