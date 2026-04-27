"""Tests for hmt.optim.projector — Stage 1.1 + 1.2."""
from __future__ import annotations

import pytest
import torch

from hmt.optim import LayerProjector, make_projector_from_grad, update_projection_basis


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / a.norm()).item()


@pytest.mark.parametrize("mode", ["two_sided", "left", "right"])
def test_shapes(mode):
    torch.manual_seed(0)
    out_dim, in_dim, rank = 64, 96, 16
    G = torch.randn(out_dim, in_dim)
    proj = make_projector_from_grad(G, mode=mode, rank=rank)
    assert proj.rank == rank
    assert proj.out_dim == out_dim and proj.in_dim == in_dim
    low = proj.project(G)
    assert tuple(low.shape) == proj.low_rank_shape
    rec = proj.reconstruct(low)
    assert tuple(rec.shape) == (out_dim, in_dim)


def test_two_sided_full_rank_recovers_grad():
    """At rank = min(out, in), two-sided reconstruction equals G (SVD truncation)."""
    torch.manual_seed(0)
    G = torch.randn(16, 24)
    proj = make_projector_from_grad(G, mode="two_sided", rank=16)
    rec = proj.reconstruct(proj.project(G))
    assert _rel_err(G, rec) < 1e-5


def test_left_full_rank_identity():
    """At rank = out_dim with mode=left, P is orthonormal so P P^T = I; reconstruction is exact."""
    torch.manual_seed(0)
    G = torch.randn(8, 32)  # out_dim=8 ≤ in_dim, full rank for left = 8
    proj = make_projector_from_grad(G, mode="left", rank=8)
    rec = proj.reconstruct(proj.project(G))
    assert _rel_err(G, rec) < 1e-5


def test_round_trip_below_one_percent_at_low_rank_for_low_rank_grad():
    """Synthetic low-rank gradient: round trip at the true rank should be ~exact."""
    torch.manual_seed(0)
    out_dim, in_dim, true_rank = 48, 64, 8
    A = torch.randn(out_dim, true_rank)
    B = torch.randn(true_rank, in_dim)
    G = A @ B  # exact rank-8

    proj = make_projector_from_grad(G, mode="two_sided", rank=true_rank)
    rec = proj.reconstruct(proj.project(G))
    assert _rel_err(G, rec) < 1e-4


def test_bf16_round_trip_within_one_percent():
    """1.1 verify: BF16 round-trip error < 1% on full-rank synthetic data."""
    torch.manual_seed(0)
    G_fp32 = torch.randn(32, 48)
    G_bf16 = G_fp32.to(torch.bfloat16)
    proj = make_projector_from_grad(G_bf16, mode="two_sided", rank=32)
    rec = proj.reconstruct(proj.project(G_bf16))
    err = _rel_err(G_bf16.float(), rec.float())
    assert err < 0.01, f"bf16 round-trip error {err:.4f} exceeds 1%"


@pytest.mark.parametrize("mode", ["two_sided", "left", "right"])
def test_higher_rank_monotonically_reduces_error(mode):
    """1.2 verify: reconstruction error monotonically decreases as rank grows."""
    torch.manual_seed(0)
    G = torch.randn(48, 64)
    errs = []
    for r in [4, 8, 16, 32]:
        proj = make_projector_from_grad(G, mode=mode, rank=r)
        rec = proj.reconstruct(proj.project(G))
        errs.append(_rel_err(G, rec))
    for i in range(len(errs) - 1):
        # Allow tiny numerical wiggle.
        assert errs[i] >= errs[i + 1] - 1e-6, f"non-monotone errors {errs}"


def test_rank_clamped_to_min_dim():
    """Requesting rank > min(out,in) silently clamps to min(out,in)."""
    G = torch.randn(8, 32)
    proj = make_projector_from_grad(G, mode="two_sided", rank=64)
    assert proj.rank == 8


def test_refresh_in_place():
    """refresh_ should update P/Q without changing rank/dims."""
    torch.manual_seed(0)
    G1 = torch.randn(16, 24)
    proj = make_projector_from_grad(G1, mode="two_sided", rank=8)
    P_old = proj.P.clone()
    Q_old = proj.Q.clone()

    G2 = torch.randn(16, 24)
    proj.refresh_(G2)
    assert proj.rank == 8
    assert proj.P.shape == P_old.shape
    assert proj.Q.shape == Q_old.shape
    # New bases should generally differ from the old ones.
    assert not torch.allclose(proj.P, P_old)


def test_validation_errors():
    G = torch.randn(8, 16)
    P, Q = update_projection_basis(G, mode="two_sided", rank=4)
    with pytest.raises(ValueError, match="rank must be positive"):
        LayerProjector(mode="two_sided", rank=0, out_dim=8, in_dim=16, P=P, Q=Q)
    with pytest.raises(ValueError, match="two_sided mode requires"):
        LayerProjector(mode="two_sided", rank=4, out_dim=8, in_dim=16, P=P, Q=None)
    with pytest.raises(ValueError, match="grad must be 2D"):
        update_projection_basis(torch.randn(2, 3, 4), mode="left", rank=2)
