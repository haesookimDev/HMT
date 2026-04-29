"""Reproducibility helper — seeds Python random, numpy, and torch (CPU/CUDA/MPS)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed all random sources used in HMT training.

    Args:
        seed: integer seed for Python random, numpy, torch CPU and any
            available accelerator (CUDA / MPS).
        deterministic: if True, also enable PyTorch's deterministic-algo flag
            and set ``CUBLAS_WORKSPACE_CONFIG`` so that the same hardware run
            twice with the same seed produces identical results. Costs some
            throughput; some ops have no deterministic implementation and
            will warn rather than fail (``warn_only=True``).
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        # mps.manual_seed is available from PyTorch 2.x.
        try:
            torch.mps.manual_seed(seed)
        except (AttributeError, RuntimeError):
            pass

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
