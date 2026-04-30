"""Cross-cutting C.3: pluggable metric logger.

A single ``log(step, metrics, event)`` interface fronting one or more
backends (JSONL, TensorBoard, W&B). Selected by ``cfg.logging.backends``
(string or list); defaults to ``["jsonl"]`` for backwards compatibility
with the pre-abstraction trainer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Protocol


class MetricLogger(Protocol):
    def log(self, step: int, metrics: dict[str, Any], *, event: str = "train") -> None: ...
    def close(self) -> None: ...


class JsonlLogger:
    """Append-mode JSONL — one record per call. Matches the original
    ``train_baseline.py`` output format byte-for-byte."""

    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8")

    def log(self, step: int, metrics: dict[str, Any], *, event: str = "train") -> None:
        rec = {"event": event, "step": int(step), **metrics}
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


class TensorBoardLogger:
    """Writes scalar metrics under ``<event>/<key>``. Non-scalar values are
    silently skipped (TensorBoard scalars only accept numeric)."""

    def __init__(self, log_dir: Path):
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._w = SummaryWriter(log_dir=str(log_dir))

    def log(self, step: int, metrics: dict[str, Any], *, event: str = "train") -> None:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v == v:  # filter NaN
                self._w.add_scalar(f"{event}/{k}", v, int(step))

    def close(self) -> None:
        self._w.close()


class WandBLogger:
    """Mirrors metrics into a W&B run. Requires `wandb login` to have been
    run beforehand (or ``WANDB_MODE=offline``)."""

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        mode: str | None = None,
    ):
        import wandb
        self._wandb = wandb
        self._run = wandb.init(
            project=project, name=run_name, config=config, mode=mode, reinit=True,
        )

    def log(self, step: int, metrics: dict[str, Any], *, event: str = "train") -> None:
        prefixed = {f"{event}/{k}": v for k, v in metrics.items() if v is not None}
        self._wandb.log(prefixed, step=int(step))

    def close(self) -> None:
        self._wandb.finish()


class MultiLogger:
    """Broadcasts every ``log()`` call to a list of underlying loggers."""

    def __init__(self, loggers: Iterable[MetricLogger]):
        self._loggers: list[MetricLogger] = list(loggers)

    def log(self, step: int, metrics: dict[str, Any], *, event: str = "train") -> None:
        for lg in self._loggers:
            lg.log(step, metrics, event=event)

    def close(self) -> None:
        for lg in self._loggers:
            lg.close()


def build_logger(logging_cfg, output_dir: Path | str) -> MetricLogger:
    """Construct a logger from an OmegaConf-like ``logging`` block.

    Recognised fields::

        logging:
          output_dir: ./outputs/<run>
          log_interval: 10
          backends: [jsonl, tensorboard]  # or "jsonl"; default ["jsonl"]
          wandb:
            project: hmt
            run_name: my_run
            mode: online                  # online | offline | disabled

    The default ``["jsonl"]`` writes ``<output_dir>/metrics.jsonl`` exactly
    as before.
    """
    output_dir = Path(output_dir)
    backends_raw = (
        logging_cfg.get("backends", ["jsonl"]) if hasattr(logging_cfg, "get")
        else logging_cfg.backends if "backends" in logging_cfg
        else ["jsonl"]
    )
    if isinstance(backends_raw, str):
        backends = [backends_raw]
    else:
        backends = [str(b) for b in backends_raw]

    loggers: list[MetricLogger] = []
    for b in backends:
        if b == "jsonl":
            loggers.append(JsonlLogger(output_dir / "metrics.jsonl"))
        elif b == "tensorboard":
            loggers.append(TensorBoardLogger(output_dir / "tb"))
        elif b == "wandb":
            wb = (
                logging_cfg.get("wandb", {}) if hasattr(logging_cfg, "get")
                else (logging_cfg.wandb if "wandb" in logging_cfg else {})
            )
            loggers.append(WandBLogger(
                project=str(wb.get("project", "hmt") if hasattr(wb, "get") else "hmt"),
                run_name=str(wb.get("run_name")) if hasattr(wb, "get") and wb.get("run_name") else None,
                config=dict(wb.get("config", {})) if hasattr(wb, "get") and wb.get("config") else None,
                mode=str(wb.get("mode")) if hasattr(wb, "get") and wb.get("mode") else None,
            ))
        else:
            raise ValueError(f"unknown logging backend: {b!r}. Expected: jsonl | tensorboard | wandb")

    if len(loggers) == 1:
        return loggers[0]
    return MultiLogger(loggers)
