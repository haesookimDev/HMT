"""Lightweight profiler for VRAM, throughput, and step time.

Stage 0 metric collection: peak/avg GPU memory, tokens/sec, step time.
On CUDA: uses torch.cuda.max_memory_allocated.
On non-CUDA (CPU/MPS): GPU memory metrics report 0; throughput still works.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch


def _is_cuda() -> bool:
    return torch.cuda.is_available()


def reset_peak_memory() -> None:
    if _is_cuda():
        torch.cuda.reset_peak_memory_stats()


def gpu_memory_bytes() -> tuple[int, int]:
    """Return (current_allocated, peak_allocated) in bytes. (0, 0) if no CUDA."""
    if not _is_cuda():
        return 0, 0
    return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()


@dataclass
class StepStats:
    step: int
    loss: float
    step_time_s: float
    tokens_per_s: float
    cur_mem_mb: float
    peak_mem_mb: float


@dataclass
class TrainingProfiler:
    window: int = 50  # rolling window for tokens/sec smoothing
    _step_times: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _tokens: deque[int] = field(default_factory=lambda: deque(maxlen=50))
    _t0: Optional[float] = None
    _peak_mem_run: int = 0

    def __post_init__(self) -> None:
        self._step_times = deque(maxlen=self.window)
        self._tokens = deque(maxlen=self.window)

    def start_step(self) -> None:
        if _is_cuda():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def end_step(self, *, step: int, loss: float, tokens: int) -> StepStats:
        if self._t0 is None:
            raise RuntimeError("end_step() called before start_step()")
        if _is_cuda():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self._t0 = None
        self._step_times.append(dt)
        self._tokens.append(tokens)

        cur, peak = gpu_memory_bytes()
        self._peak_mem_run = max(self._peak_mem_run, peak)

        total_tokens = sum(self._tokens)
        total_time = sum(self._step_times) or 1e-9
        tps = total_tokens / total_time

        return StepStats(
            step=step,
            loss=loss,
            step_time_s=dt,
            tokens_per_s=tps,
            cur_mem_mb=cur / (1024 ** 2),
            peak_mem_mb=peak / (1024 ** 2),
        )

    def run_peak_mb(self) -> float:
        return self._peak_mem_run / (1024 ** 2)


def format_step_stats(s: StepStats) -> str:
    return (
        f"step {s.step:>6d} | loss {s.loss:7.4f} | "
        f"step {s.step_time_s*1000:6.1f} ms | tok/s {s.tokens_per_s:8.1f} | "
        f"cur {s.cur_mem_mb:7.1f} MB | peak {s.peak_mem_mb:7.1f} MB"
    )
