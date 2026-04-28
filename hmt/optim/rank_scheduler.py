"""Energy-based per-layer rank selection.

For a gradient with singular values ``σ_1 ≥ σ_2 ≥ … ≥ σ_n``, the cumulative
energy at rank ``r`` is::

    E(r) = (Σ_{i=1..r} σ_i^2) / (Σ_{i=1..n} σ_i^2)

``EnergyRankScheduler`` returns the smallest rank from a candidate set whose
energy meets a threshold (``τ``). Layers with concentrated spectra get small
ranks; layers with broad spectra get larger ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class EnergyRankScheduler:
    candidates: Sequence[int] = field(default_factory=lambda: (32, 64, 128, 256))
    threshold: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        if any(r <= 0 for r in self.candidates):
            raise ValueError(f"all candidates must be positive: {self.candidates}")
        # Cache as a sorted tuple so iteration order is deterministic.
        object.__setattr__(self, "candidates", tuple(sorted(self.candidates)))

    def select_rank(self, singular_values: torch.Tensor) -> int:
        """Return the smallest candidate ``r`` with ``E(r) >= threshold``.

        Falls through to ``min(max(candidates), len(singular_values))`` if no
        candidate meets the threshold (e.g., flat / non-decaying spectrum).
        """
        if singular_values.numel() == 0:
            return self.candidates[-1]
        s2 = singular_values.detach().to(torch.float32) ** 2
        total = s2.sum().item()
        if total <= 0.0:
            return self.candidates[-1]
        energy = torch.cumsum(s2, dim=0) / total
        n = energy.numel()
        for r in self.candidates:
            if r <= n and energy[r - 1].item() >= self.threshold:
                return r
        return min(self.candidates[-1], n)
