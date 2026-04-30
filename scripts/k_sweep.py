"""F.1 verification: K-sweep comparing m/v alignment vs no-alignment.

Runs a small synthetic regression with a stack of Linears, varying the basis
refresh interval K and comparing final loss with ``align_state=True`` vs
``False``. Alignment should never hurt; at small K (frequent refresh) it
should preserve EMA history that the no-align path destroys.

Usage::

    uv run python scripts/k_sweep.py
    uv run python scripts/k_sweep.py --steps 400 --rank 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hmt.optim import LowRankAdamW, make_projector_from_grad
from hmt.optim.setup import refresh_projectors_from_grads


def _run(align: bool, K: int, steps: int, rank: int, seed: int) -> float:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    hidden = 32
    X = torch.randn(128, hidden, generator=g)
    W_true = torch.randn(hidden, hidden, generator=g) * 0.5
    Y = X @ W_true.T + 0.05 * torch.randn(128, hidden, generator=g)

    torch.manual_seed(seed + 1)
    layers = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden, bias=False),
    )

    o = LowRankAdamW(layers.parameters(), lr=5e-3, betas=(0.9, 0.95))

    # First backward to seed projectors.
    loss = ((layers(X) - Y) ** 2).mean()
    o.zero_grad(); loss.backward()
    targets: list[tuple[str, torch.nn.Parameter]] = []
    for name, p in layers.named_parameters():
        if p.ndim == 2:
            proj = make_projector_from_grad(p.grad, mode="two_sided", rank=rank)
            o.attach_projector(p, proj)
            targets.append((name, p))
    o.step()

    losses: list[float] = []
    for step in range(steps):
        loss = ((layers(X) - Y) ** 2).mean()
        o.zero_grad(); loss.backward()
        if step > 0 and step % K == 0:
            refresh_projectors_from_grads(o, targets, align_state=align)
        o.step()
        losses.append(loss.item())
    return float(sum(losses[-10:]) / 10.0)  # last-10 average — denoises


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 50, 100])
    ap.add_argument("--out", type=str, default="outputs/k_sweep/summary.json")
    args = ap.parse_args()

    print(f"steps={args.steps} rank={args.rank} seeds={args.seeds} Ks={args.ks}")
    print(f"{'K':>4} | {'aligned (mean)':>15} | {'unaligned (mean)':>17} | {'delta':>10}")
    print("-" * 56)

    results = []
    for K in args.ks:
        a = [_run(align=True,  K=K, steps=args.steps, rank=args.rank, seed=s) for s in args.seeds]
        u = [_run(align=False, K=K, steps=args.steps, rank=args.rank, seed=s) for s in args.seeds]
        a_mean = sum(a) / len(a)
        u_mean = sum(u) / len(u)
        delta = u_mean - a_mean  # positive = alignment wins
        print(f"{K:>4} | {a_mean:>15.4f} | {u_mean:>17.4f} | {delta:>+10.4f}")
        results.append({"K": K, "aligned_mean": a_mean, "unaligned_mean": u_mean,
                        "delta": delta, "aligned": a, "unaligned": u})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "steps": args.steps, "rank": args.rank,
        "seeds": args.seeds, "ks": args.ks,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
