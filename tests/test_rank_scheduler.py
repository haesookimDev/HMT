"""Tests for hmt.optim.rank_scheduler.EnergyRankScheduler — Stage 2.2."""
from __future__ import annotations

import pytest
import torch

from hmt.optim import EnergyRankScheduler


def test_picks_smallest_meeting_threshold():
    """Spectrum: top-5 ones, then 100 small. Energy at 8 ≈ 0.998. Pick 8."""
    s = torch.cat([torch.ones(5), torch.full((100,), 0.01)])
    sched = EnergyRankScheduler(candidates=(4, 8, 16, 32), threshold=0.95)
    # candidate 4: energy[3] = 4 / (5 + 100*0.0001) ≈ 0.798 -> fail
    # candidate 8: energy[7] = (5 + 3*0.0001) / (5 + 0.01) ≈ 0.998 -> pass
    assert sched.select_rank(s) == 8


def test_low_rank_spectrum_picks_smallest_candidate():
    """Effective rank 4: only 4 nonzero singular values. Threshold met at rank 4."""
    s = torch.tensor([2.0, 1.5, 1.2, 1.0] + [0.0] * 60)
    sched = EnergyRankScheduler(candidates=(4, 8, 16), threshold=0.99)
    assert sched.select_rank(s) == 4


def test_falls_through_to_max_when_no_candidate_meets():
    """Flat spectrum, high threshold -> no candidate works, fall through."""
    s = torch.ones(64)
    sched = EnergyRankScheduler(candidates=(4, 8, 16), threshold=0.99)
    # energy[15] = 16/64 = 0.25 < 0.99 for all candidates
    assert sched.select_rank(s) == 16


def test_short_spectrum_clamped():
    """When all candidates exceed n, return min(max_candidate, n)."""
    s = torch.tensor([1.0, 0.5, 0.1])
    sched = EnergyRankScheduler(candidates=(4, 8), threshold=0.5)
    assert sched.select_rank(s) == 3


def test_zero_or_empty_spectrum_returns_largest_candidate():
    sched = EnergyRankScheduler(candidates=(4, 8, 16), threshold=0.9)
    assert sched.select_rank(torch.tensor([])) == 16
    assert sched.select_rank(torch.zeros(20)) == 16


def test_candidates_get_sorted():
    sched = EnergyRankScheduler(candidates=(64, 16, 256, 32), threshold=0.9)
    assert tuple(sched.candidates) == (16, 32, 64, 256)


def test_validation_errors():
    with pytest.raises(ValueError):
        EnergyRankScheduler(candidates=(4, 8), threshold=0.0)
    with pytest.raises(ValueError):
        EnergyRankScheduler(candidates=(4, 8), threshold=1.5)
    with pytest.raises(ValueError):
        EnergyRankScheduler(candidates=())
    with pytest.raises(ValueError):
        EnergyRankScheduler(candidates=(0, 8))


def test_attention_vs_mlp_spectrum_differential():
    """README expectation: layers with concentrated spectra (e.g. attention)
    pick smaller ranks than layers with broad spectra (e.g. MLP).

    Synthetic: 'attention-like' spectrum decays fast; 'mlp-like' decays slowly.
    """
    sched = EnergyRankScheduler(candidates=(32, 64, 128, 256), threshold=0.9)
    n = 256
    # Attention: top-30 carry most energy
    attn = torch.cat([torch.linspace(1.0, 0.1, 30), torch.full((n - 30,), 0.01)])
    # MLP: slower decay
    mlp = torch.linspace(1.0, 0.05, n)
    r_attn = sched.select_rank(attn)
    r_mlp = sched.select_rank(mlp)
    assert r_attn < r_mlp, f"expected attn rank < mlp rank, got {r_attn} vs {r_mlp}"
