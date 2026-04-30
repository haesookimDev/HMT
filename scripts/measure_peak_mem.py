"""Peak VRAM measurement for Stage 1.6 (optimizer state) and Stage 3.5
(activation compression). Runs on a synthetic stacked-Linear model so the
measurement is independent of HF model weight loading.

Usage::

    uv run python scripts/measure_peak_mem.py
    uv run python scripts/measure_peak_mem.py --hidden 1024 --depth 24 --rank 64

Reports peak `torch.cuda.max_memory_allocated()` for each scenario,
isolated by `torch.cuda.reset_peak_memory_stats()` between runs.
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass

import torch

from hmt.autograd.compressed_linear import patch_model_int8_linear
from hmt.memory.policy import ActivationPolicy, ActivationRule
from hmt.optim import LowRankAdamW, make_projector_from_grad


@dataclass
class MemSample:
    label: str
    peak_mb: float
    state_elems: int = 0
    note: str = ""


def _reset(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


class StackedLinear(torch.nn.Module):
    """A stack of Linear layers with the same hidden size -- coarse stand-in
    for transformer MLP/attention projections for memory profiling.

    With ``with_relu=True`` (default) ReLU sits between Linears and saves its
    own BF16 input for backward, which means the same activation is stored
    twice when CompressedLinear wraps the next Linear (once BF16 inside ReLU,
    once INT8 inside CompressedLinear). Set ``with_relu=False`` to isolate the
    pure CompressedLinear vs Linear comparison.
    """

    def __init__(self, hidden: int, depth: int, with_relu: bool = True):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, hidden, bias=False) for _ in range(depth)]
        )
        self.with_relu = with_relu

    def forward(self, x):
        if self.with_relu:
            for layer in self.layers:
                x = torch.nn.functional.relu(layer(x))
        else:
            for layer in self.layers:
                x = layer(x)
        return x


def _state_elems(opt: torch.optim.Optimizer) -> int:
    n = 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                n += v.numel()
    return n


def measure_optimizer_state(
    device, hidden: int, depth: int, batch: int, seq: int, rank: int, dtype
) -> list[MemSample]:
    samples: list[MemSample] = []

    def _build_model_and_x():
        torch.manual_seed(0)
        m = StackedLinear(hidden, depth).to(device=device, dtype=dtype)
        x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
        target = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
        return m, x, target

    # --- 1. Dense AdamW -----------------------------------------------------
    _reset(device)
    m, x, target = _build_model_and_x()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):  # 3 steps so AdamW state (m, v) is fully allocated
        loss = ((m(x) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    samples.append(
        MemSample(
            label="adamw_dense",
            peak_mb=_peak_mb(device),
            state_elems=_state_elems(opt),
            note=f"hidden={hidden} depth={depth} dtype={dtype}",
        )
    )
    del m, opt, x, target

    # --- 2. LowRankAdamW two_sided rank=r ----------------------------------
    _reset(device)
    m, x, target = _build_model_and_x()
    opt = LowRankAdamW(m.parameters(), lr=1e-3)

    # Seed projectors after a first backward.
    loss = ((m(x) - target) ** 2).mean()
    opt.zero_grad(); loss.backward()
    for layer in m.layers:
        proj = make_projector_from_grad(
            layer.weight.grad, mode="two_sided", rank=rank,
        )
        opt.attach_projector(layer.weight, proj)
    opt.step()
    for _ in range(2):
        loss = ((m(x) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    samples.append(
        MemSample(
            label=f"lowrank_two_sided_r{rank}",
            peak_mb=_peak_mb(device),
            state_elems=_state_elems(opt),
            note=f"hidden={hidden} depth={depth} rank={rank}",
        )
    )
    del m, opt, x, target

    return samples


def measure_activation_memory(
    device, hidden: int, depth: int, batch: int, seq: int, dtype, block_size: int,
    with_relu: bool = True,
) -> list[MemSample]:
    samples: list[MemSample] = []

    def _build():
        torch.manual_seed(0)
        m = StackedLinear(hidden, depth, with_relu=with_relu).to(device=device, dtype=dtype)
        x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
        target = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
        return m, x, target

    # --- A. Dense Linear (baseline activation memory) ----------------------
    _reset(device)
    m, x, target = _build()
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    samples.append(
        MemSample(
            label="dense_linear",
            peak_mb=_peak_mb(device),
            note=f"hidden={hidden} depth={depth} batch={batch} seq={seq} dtype={dtype}",
        )
    )
    del m, x, target

    # --- B. CompressedLinear (INT8 activation cache) -----------------------
    _reset(device)
    m, x, target = _build()
    policy = ActivationPolicy(
        rules=[ActivationRule(pattern=r"^layers\.", action="compress_int8")],
        default="keep",
        block_size=block_size,
    )
    patched = patch_model_int8_linear(m, policy)
    assert len(patched) == depth, f"expected {depth} patches, got {len(patched)}"
    loss = ((m(x) - target) ** 2).mean()
    loss.backward()
    samples.append(
        MemSample(
            label="compressed_int8_linear",
            peak_mb=_peak_mb(device),
            note=f"hidden={hidden} depth={depth} batch={batch} seq={seq} block={block_size}",
        )
    )

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="outputs/peak_mem/summary.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(
        f"device={device} | dtype={dtype} | hidden={args.hidden} depth={args.depth} "
        f"batch={args.batch} seq={args.seq} rank={args.rank} block={args.block_size}"
    )

    print("\n## Stage 1.6 -- optimizer state peak memory")
    s1 = measure_optimizer_state(
        device, args.hidden, args.depth, args.batch, args.seq, args.rank, dtype,
    )
    for s in s1:
        print(f"  {s.label:30s}  peak={s.peak_mb:8.2f} MB  state_elems={s.state_elems:>12,}")

    if len(s1) >= 2:
        peak_ratio = s1[0].peak_mb / max(s1[1].peak_mb, 1e-9)
        state_ratio = s1[0].state_elems / max(s1[1].state_elems, 1)
        print(
            f"  -> low-rank peak is {peak_ratio:.2f}x smaller than dense AdamW peak;"
            f" state element count is {state_ratio:.1f}x smaller"
        )

    print("\n## Stage 3.5 -- activation memory peak (forward+backward)")
    print("  -- with ReLU between layers (pessimistic: ReLU also saves BF16 copy)")
    s2_relu = measure_activation_memory(
        device, args.hidden, args.depth, args.batch, args.seq, dtype, args.block_size,
        with_relu=True,
    )
    for s in s2_relu:
        print(f"  {s.label:30s}  peak={s.peak_mb:8.2f} MB")
    if len(s2_relu) >= 2:
        peak_ratio = s2_relu[0].peak_mb / max(s2_relu[1].peak_mb, 1e-9)
        print(f"  -> INT8-cached peak is {peak_ratio:.2f}x dense (>1 means INT8 wins)")

    print("  -- pure Linear stack (no ReLU; isolates CompressedLinear's effect)")
    s2_pure = measure_activation_memory(
        device, args.hidden, args.depth, args.batch, args.seq, dtype, args.block_size,
        with_relu=False,
    )
    for s in s2_pure:
        print(f"  {s.label:30s}  peak={s.peak_mb:8.2f} MB")
    if len(s2_pure) >= 2:
        peak_ratio = s2_pure[0].peak_mb / max(s2_pure[1].peak_mb, 1e-9)
        print(f"  -> INT8-cached peak is {peak_ratio:.2f}x dense (>1 means INT8 wins)")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "hidden": args.hidden,
        "depth": args.depth,
        "batch": args.batch,
        "seq": args.seq,
        "rank": args.rank,
        "block_size": args.block_size,
        "stage_1_6": [asdict(s) for s in s1],
        "stage_3_5_with_relu": [asdict(s) for s in s2_relu],
        "stage_3_5_pure_linear": [asdict(s) for s in s2_pure],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
