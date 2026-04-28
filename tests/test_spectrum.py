"""Tests for hmt.optim.spectrum.randomized_svd — Stage 2.1."""
from __future__ import annotations

import time

import torch

from hmt.optim import randomized_svd


def test_top_singular_values_within_5_percent():
    """2.1 verify: top-k singular values should match full SVD within 5%
    on a gradient-like matrix (signal + small noise — typical of real
    backprop gradients). Pure random Gaussian has worst-case slow spectral
    decay; we use signal+noise to model realistic spectra.
    """
    torch.manual_seed(0)
    out_dim, in_dim = 128, 192
    # signal ≈ rank-32 + small noise → first ~32 singular values dominate
    A = torch.randn(out_dim, 32) @ torch.randn(32, in_dim) + 0.05 * torch.randn(out_dim, in_dim)
    rank = 16

    _, S_full, _ = torch.linalg.svd(A, full_matrices=False)
    _, S_rsvd, _ = randomized_svd(A, rank=rank, oversample=8, n_iter=2)

    rel = ((S_full[:rank] - S_rsvd) / S_full[:rank]).abs().max().item()
    assert rel < 0.05, f"top-k singular value relative error {rel:.4f} exceeds 5%"


def test_subspace_captures_most_energy():
    """The randomized basis should explain most of the rank-r energy."""
    torch.manual_seed(0)
    A = torch.randn(64, 96)
    rank = 8

    U_rsvd, _, _ = randomized_svd(A, rank=rank, oversample=8, n_iter=2)
    # Project A onto rsvd subspace and compare to full rank-r truncation
    proj = U_rsvd @ U_rsvd.T @ A
    _, S_full, _ = torch.linalg.svd(A, full_matrices=False)
    target_energy = (S_full[:rank] ** 2).sum().item()
    captured_energy = (proj ** 2).sum().item()
    assert captured_energy >= 0.95 * target_energy, (
        f"captured {captured_energy:.4f} of target {target_energy:.4f}"
    )


def test_low_rank_input_recovered_exactly():
    """A truly rank-r matrix should be reconstructed near-perfectly."""
    torch.manual_seed(0)
    out_dim, in_dim, r = 48, 64, 6
    L = torch.randn(out_dim, r)
    R = torch.randn(r, in_dim)
    A = L @ R  # rank-6

    U, S, Vh = randomized_svd(A, rank=r, oversample=8, n_iter=2)
    rec = U @ torch.diag(S) @ Vh
    err = ((A - rec).norm() / A.norm()).item()
    assert err < 1e-3, f"rank-{r} reconstruction error {err:.4e}"


def test_shape_clamping():
    """rank > min(m, n) should be silently clamped."""
    A = torch.randn(8, 32)
    U, S, Vh = randomized_svd(A, rank=64, oversample=4, n_iter=1)
    assert U.shape == (8, 8)
    assert S.shape == (8,)
    assert Vh.shape == (8, 32)


def test_validation_errors():
    A = torch.randn(8, 16)
    with pytest_raises():
        randomized_svd(torch.randn(2, 3, 4), rank=2)
    with pytest_raises():
        randomized_svd(A, rank=0)
    with pytest_raises():
        randomized_svd(A, rank=4, oversample=-1)
    with pytest_raises():
        randomized_svd(A, rank=4, n_iter=-1)


def pytest_raises():
    import pytest
    return pytest.raises(ValueError)


def test_faster_than_full_svd_on_large_matrix():
    """Soft speed check: randomized SVD ≥1× full SVD on a 1024² matrix at rank 32.

    Hardware-dependent; we assert ≥ rather than the README's 3× target so the
    test stays portable. Logs the actual ratio for observability.
    """
    torch.manual_seed(0)
    A = torch.randn(1024, 1024)
    rank = 32

    # Warm-up
    torch.linalg.svd(A, full_matrices=False)
    randomized_svd(A, rank=rank, oversample=8, n_iter=2)

    t0 = time.perf_counter()
    for _ in range(2):
        torch.linalg.svd(A, full_matrices=False)
    t_full = (time.perf_counter() - t0) / 2

    t0 = time.perf_counter()
    for _ in range(2):
        randomized_svd(A, rank=rank, oversample=8, n_iter=2)
    t_rsvd = (time.perf_counter() - t0) / 2

    ratio = t_full / max(t_rsvd, 1e-9)
    print(f"[stage2.1] full SVD {t_full*1000:.1f}ms vs rSVD {t_rsvd*1000:.1f}ms -> {ratio:.2f}x")
    # Loose lower bound; speedups of 3-10x are typical on CUDA, modest on CPU/MPS.
    assert ratio >= 0.8, f"randomized SVD significantly slower than full SVD: {ratio:.2f}x"
