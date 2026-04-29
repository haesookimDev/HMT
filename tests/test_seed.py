"""Tests for hmt.utils.seed.seed_everything — Cross-cutting C.2."""
from __future__ import annotations

import random as _random

import numpy as np
import pytest
import torch

from hmt.utils import seed_everything


def test_torch_cpu_reproducible():
    seed_everything(42)
    a = torch.randn(16)
    seed_everything(42)
    b = torch.randn(16)
    assert torch.equal(a, b)


def test_numpy_reproducible():
    seed_everything(7)
    a = np.random.randn(8)
    seed_everything(7)
    b = np.random.randn(8)
    assert np.array_equal(a, b)


def test_python_random_reproducible():
    seed_everything(123)
    a = [_random.random() for _ in range(8)]
    seed_everything(123)
    b = [_random.random() for _ in range(8)]
    assert a == b


def test_different_seeds_produce_different_streams():
    seed_everything(1)
    a = torch.randn(16)
    seed_everything(2)
    b = torch.randn(16)
    assert not torch.equal(a, b)


def test_validation_error():
    with pytest.raises(TypeError):
        seed_everything("42")  # type: ignore[arg-type]
