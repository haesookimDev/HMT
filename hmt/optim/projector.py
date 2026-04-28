"""Low-rank gradient projector.

Provides:
  - ``LayerProjector``: holds projection bases ``P`` and/or ``Q`` plus shape
    metadata, with ``project(grad)`` and ``reconstruct(low_update)`` ops.
  - ``update_projection_basis(grad, mode, rank)``: SVD-based basis computation.
  - ``make_projector_from_grad(grad, mode, rank)``: factory.

Modes
-----
* ``two_sided``: ``Ĝ = P^T G Q``     (state shape ``r × r``) — README HMT spec
* ``left``    : ``Ĝ = P^T G``       (state shape ``r × in_dim``) — vanilla GaLore
* ``right``   : ``Ĝ = G Q``         (state shape ``out_dim × r``)

For SVD-derived bases, ``two_sided`` reconstructs the rank-r SVD truncation of
``G`` exactly. ``left`` / ``right`` keep more information per-layer at the
cost of larger optimizer state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from hmt.optim.spectrum import randomized_svd

ProjectionMode = Literal["two_sided", "left", "right"]
SVDMethod = Literal["full", "randomized"]


@dataclass
class LayerProjector:
    mode: ProjectionMode
    rank: int
    out_dim: int
    in_dim: int
    P: Optional[torch.Tensor] = None  # [out_dim, rank]
    Q: Optional[torch.Tensor] = None  # [in_dim, rank]
    method: SVDMethod = "full"  # SVD method to use for refresh_

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.mode == "two_sided" and (self.P is None or self.Q is None):
            raise ValueError("two_sided mode requires both P and Q")
        if self.mode == "left" and self.P is None:
            raise ValueError("left mode requires P")
        if self.mode == "right" and self.Q is None:
            raise ValueError("right mode requires Q")
        if self.P is not None and tuple(self.P.shape) != (self.out_dim, self.rank):
            raise ValueError(f"P shape {tuple(self.P.shape)} != ({self.out_dim}, {self.rank})")
        if self.Q is not None and tuple(self.Q.shape) != (self.in_dim, self.rank):
            raise ValueError(f"Q shape {tuple(self.Q.shape)} != ({self.in_dim}, {self.rank})")

    @property
    def low_rank_shape(self) -> tuple[int, int]:
        if self.mode == "two_sided":
            return (self.rank, self.rank)
        if self.mode == "left":
            return (self.rank, self.in_dim)
        return (self.out_dim, self.rank)

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if self.mode == "two_sided":
            return self.P.T @ grad @ self.Q
        if self.mode == "left":
            return self.P.T @ grad
        return grad @ self.Q

    def reconstruct(self, low_update: torch.Tensor) -> torch.Tensor:
        if self.mode == "two_sided":
            return self.P @ low_update @ self.Q.T
        if self.mode == "left":
            return self.P @ low_update
        return low_update @ self.Q.T

    @torch.no_grad()
    def refresh_(self, grad: torch.Tensor) -> None:
        """Recompute P/Q from ``grad``'s SVD. Mutates self in-place."""
        P, Q = update_projection_basis(grad, mode=self.mode, rank=self.rank, method=self.method)
        if P is not None:
            self.P = P
        if Q is not None:
            self.Q = Q


def _compute_top_singular(
    grad: torch.Tensor,
    *,
    rank: int,
    method: SVDMethod,
    oversample: int = 8,
    n_iter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-``rank`` ``(U, S, Vh)`` of ``grad`` via full or randomized SVD."""
    if method == "full":
        U, S, Vh = torch.linalg.svd(grad.detach().float(), full_matrices=False)
        return U[:, :rank].contiguous(), S[:rank].contiguous(), Vh[:rank, :].contiguous()
    if method == "randomized":
        return randomized_svd(grad, rank=rank, oversample=oversample, n_iter=n_iter)
    raise ValueError(f"unknown svd method: {method}")


@torch.no_grad()
def update_projection_basis(
    grad: torch.Tensor,
    *,
    mode: ProjectionMode,
    rank: int,
    method: SVDMethod = "full",
    oversample: int = 8,
    n_iter: int = 2,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute SVD-based projection bases for one 2D gradient.

    Returns ``(P, Q)`` in ``grad``'s dtype/device. Either tensor is ``None``
    when ``mode`` does not need it.
    """
    if grad.ndim != 2:
        raise ValueError(f"grad must be 2D, got shape {tuple(grad.shape)}")
    out_dim, in_dim = grad.shape
    eff_rank = min(rank, out_dim, in_dim)

    U, _S, Vh = _compute_top_singular(
        grad, rank=eff_rank, method=method, oversample=oversample, n_iter=n_iter,
    )
    P = None
    Q = None
    if mode in ("two_sided", "left"):
        P = U.to(dtype=grad.dtype, device=grad.device).contiguous()
    if mode in ("two_sided", "right"):
        Q = Vh.T.to(dtype=grad.dtype, device=grad.device).contiguous()
    return P, Q


def make_projector_from_grad(
    grad: torch.Tensor,
    *,
    mode: ProjectionMode,
    rank: int,
    method: SVDMethod = "full",
    oversample: int = 8,
    n_iter: int = 2,
) -> LayerProjector:
    if grad.ndim != 2:
        raise ValueError(f"grad must be 2D, got shape {tuple(grad.shape)}")
    out_dim, in_dim = grad.shape
    eff_rank = min(rank, out_dim, in_dim)
    P, Q = update_projection_basis(
        grad, mode=mode, rank=eff_rank, method=method, oversample=oversample, n_iter=n_iter,
    )
    return LayerProjector(
        mode=mode, rank=eff_rank, out_dim=out_dim, in_dim=in_dim, P=P, Q=Q, method=method,
    )


def make_projector_with_scheduler(
    grad: torch.Tensor,
    *,
    mode: ProjectionMode,
    scheduler,
    method: SVDMethod = "full",
    oversample: int = 8,
    n_iter: int = 2,
) -> LayerProjector:
    """Use ``scheduler.select_rank`` to choose a rank from this gradient's
    spectrum, then build a projector at that rank.

    Computes singular values up to ``max(scheduler.candidates)``.
    """
    if grad.ndim != 2:
        raise ValueError(f"grad must be 2D, got shape {tuple(grad.shape)}")
    out_dim, in_dim = grad.shape
    max_candidate = int(max(scheduler.candidates))
    probe_rank = min(max_candidate, out_dim, in_dim)
    U, S, Vh = _compute_top_singular(
        grad, rank=probe_rank, method=method, oversample=oversample, n_iter=n_iter,
    )
    eff_rank = min(scheduler.select_rank(S), out_dim, in_dim)
    P = None
    Q = None
    if mode in ("two_sided", "left"):
        P = U[:, :eff_rank].to(dtype=grad.dtype, device=grad.device).contiguous()
    if mode in ("two_sided", "right"):
        Q = Vh[:eff_rank, :].T.to(dtype=grad.dtype, device=grad.device).contiguous()
    return LayerProjector(
        mode=mode, rank=eff_rank, out_dim=out_dim, in_dim=in_dim, P=P, Q=Q, method=method,
    )
