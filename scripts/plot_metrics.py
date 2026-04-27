"""Plot training and eval metrics from one or more output directories.

Each directory is expected to contain a ``metrics.jsonl`` file produced by
``train_baseline.py``. Each line is a JSON record tagged with ``event`` of
``"train"`` or ``"eval"``.

Usage:
    uv run python scripts/plot_metrics.py outputs/baseline_adamw
    uv run python scripts/plot_metrics.py outputs/baseline_adamw outputs/hmt_stage1
    uv run python scripts/plot_metrics.py outputs/* --out comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(jsonl_path: Path) -> tuple[list[dict], list[dict]]:
    train, evalrec = [], []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            (evalrec if rec.get("event") == "eval" else train).append(rec)
    return train, evalrec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="Output dirs containing metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG (default: <first dir>/plots.png)")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_loss, ax_ppl, ax_tps, ax_mem = axes.flatten()

    plotted = 0
    for d in args.dirs:
        path = d / "metrics.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        train, evalrec = load_metrics(path)
        label = d.name

        if train:
            steps = [r["step"] for r in train]
            ax_loss.plot(steps, [r["loss"] for r in train], label=label)
            ax_tps.plot(steps, [r["tokens_per_s"] for r in train], label=label)
            ax_mem.plot(steps, [r["peak_mem_mb"] for r in train], label=label)
        if evalrec:
            esteps = [r["step"] for r in evalrec]
            ax_ppl.plot(esteps, [r["eval_ppl"] for r in evalrec], marker="o", label=label)
        plotted += 1

    if plotted == 0:
        raise SystemExit("No metrics.jsonl found in any of the given dirs.")

    ax_loss.set_title("Train loss"); ax_loss.set_xlabel("step"); ax_loss.set_ylabel("CE loss")
    ax_ppl.set_title("Eval perplexity"); ax_ppl.set_xlabel("step"); ax_ppl.set_ylabel("PPL")
    ax_tps.set_title("Throughput"); ax_tps.set_xlabel("step"); ax_tps.set_ylabel("tokens / sec")
    ax_mem.set_title("Peak GPU memory"); ax_mem.set_xlabel("step"); ax_mem.set_ylabel("MB")
    for ax in (ax_loss, ax_ppl, ax_tps, ax_mem):
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out = args.out or args.dirs[0] / "plots.png"
    plt.savefig(out, dpi=120)
    print(f"[done] saved {out}")


if __name__ == "__main__":
    main()
