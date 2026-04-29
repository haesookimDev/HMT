"""Stage 7.3: experiment matrix runner.

Runs ``train_baseline.py`` on each given config in sequence. Aggregates the
final-step metrics (last train record + last eval record) into a single
``summary.json`` for direct comparison.

Usage:
    uv run python scripts/run_matrix.py \\
        configs/baseline_adamw.yaml \\
        configs/hmt_stage1.yaml \\
        configs/hmt_stage2.yaml \\
        configs/hmt_stage3.yaml \\
        --max-steps 200 \\
        --out outputs/matrix_summary.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _last_records(metrics_path: Path) -> tuple[dict | None, dict | None]:
    last_train = None
    last_eval = None
    if not metrics_path.exists():
        return None, None
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "eval":
                last_eval = rec
            else:
                last_train = rec
    return last_train, last_eval


def _output_dir_for(config: Path) -> Path:
    """Best-effort: read `logging.output_dir` from yaml without parsing it
    fully. Fallback to ``outputs/<stem>``.
    """
    try:
        from omegaconf import OmegaConf  # local import; only needed when run
        cfg = OmegaConf.load(config)
        return Path(cfg.logging.output_dir)
    except Exception:
        return Path("outputs") / config.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="YAML config files")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/matrix_summary.json"),
        help="Aggregated summary destination",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Python interpreter for child processes",
    )
    args = parser.parse_args()

    runs: list[dict] = []
    for cfg_path in args.configs:
        if not cfg_path.exists():
            print(f"[skip] {cfg_path} not found")
            continue
        cmd = [args.python, "train_baseline.py", "--config", str(cfg_path)]
        if args.max_steps is not None:
            cmd.append(f"training.max_steps={args.max_steps}")

        print(f"\n[run] {cfg_path}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[error] {cfg_path} exited {result.returncode}; recording partial result")

        out_dir = _output_dir_for(cfg_path)
        last_train, last_eval = _last_records(out_dir / "metrics.jsonl")
        ranks_path = out_dir / "ranks.json"
        ranks_summary = None
        if ranks_path.exists():
            data = json.loads(ranks_path.read_text())
            ranks_summary = {
                "avg": data.get("avg"),
                "min": data.get("min"),
                "max": data.get("max"),
                "n_layers": len(data.get("ranks", {})),
            }

        runs.append({
            "config": str(cfg_path),
            "output_dir": str(out_dir),
            "exit_code": result.returncode,
            "last_train": last_train,
            "last_eval": last_eval,
            "ranks": ranks_summary,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": runs}, indent=2))
    print(f"\n[done] summary written to {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
