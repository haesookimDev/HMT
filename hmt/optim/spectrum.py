"""Randomized SVD for fast top-k projection-basis estimation.

Implements Halko-Martinsson-Tropp randomized SVD with optional power iterations.
Designed as a drop-in alternative to ``torch.linalg.svd`` for the case where
only the top-k singular components are needed and ``k << min(m, n)``.

On CUDA this avoids the cubic full-SVD cost. On MPS, ``torch.linalg.svd`` and
``torch.linalg.qr`` may fall back to CPU; randomized SVD still helps because
the SVD inside operates on the smaller ``[sketch, n]`` matrix.
"""
from __future__ import annotations

from typing import Optional

import torch


@torch.no_grad()
def randomized_svd(
    A: torch.Tensor,
    *,
    rank: int,
    oversample: int = 8,
    n_iter: int = 2,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate top-``rank`` SVD of ``A`` via random sketching.

    Args:
        A: 2D tensor ``[m, n]``. Computation runs in fp32 internally.
        rank: target rank ``k``. Effective rank is ``min(rank, m, n)``.
        oversample: extra sketch columns for accuracy (HMT default: 8).
        n_iter: number of power iterations (0 = no power iterations,
            2 is a good default for moderately decaying spectra).
        generator: optional ``torch.Generator`` for the random sketch.

    Returns:
        ``(U, S, Vh)`` with shapes ``[m, k]``, ``[k]``, ``[k, n]`` in fp32.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {tuple(A.shape)}")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if oversample < 0:
        raise ValueError(f"oversample must be non-negative, got {oversample}")
    if n_iter < 0:
        raise ValueError(f"n_iter must be non-negative, got {n_iter}")

    A_fp32 = A.detach().float()
    m, n = A_fp32.shape
    k = min(rank, m, n)
    sketch = min(k + oversample, n, m)

    if generator is None:
        omega = torch.randn(n, sketch, dtype=A_fp32.dtype, device=A_fp32.device)
    else:
        omega = torch.randn(
            n, sketch,
            dtype=A_fp32.dtype, device=A_fp32.device, generator=generator,
        )

    # Range finder: Y captures span of top singular directions.
    Y = A_fp32 @ omega
    for _ in range(n_iter):
        Q, _ = torch.linalg.qr(Y)
        Z = A_fp32.T @ Q
        Q2, _ = torch.linalg.qr(Z)
        Y = A_fp32 @ Q2

    Q, _ = torch.linalg.qr(Y)        # [m, sketch]
    B = Q.T @ A_fp32                 # [sketch, n]
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub                        # [m, sketch]

    return U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k, :].contiguous()
