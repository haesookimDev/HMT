"""Tests for hmt.utils.logger -- Cross-cutting C.3."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from hmt.utils import (
    JsonlLogger,
    MultiLogger,
    TensorBoardLogger,
    build_logger,
)


def test_jsonl_logger_writes_one_line_per_record(tmp_path: Path):
    log_path = tmp_path / "metrics.jsonl"
    lg = JsonlLogger(log_path)
    lg.log(1, {"loss": 1.5, "lr": 1e-3}, event="train")
    lg.log(2, {"eval_ppl": 12.3}, event="eval")
    lg.close()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 2

    rec0 = json.loads(lines[0])
    assert rec0 == {"event": "train", "step": 1, "loss": 1.5, "lr": 1e-3}

    rec1 = json.loads(lines[1])
    assert rec1 == {"event": "eval", "step": 2, "eval_ppl": 12.3}


def test_jsonl_logger_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "metrics.jsonl"
    lg = JsonlLogger(nested)
    lg.log(1, {"x": 1.0})
    lg.close()
    assert nested.exists()


def test_jsonl_logger_idempotent_close(tmp_path: Path):
    lg = JsonlLogger(tmp_path / "m.jsonl")
    lg.close()
    lg.close()  # must not raise


def test_multi_logger_dispatches_to_all(tmp_path: Path):
    a_path = tmp_path / "a.jsonl"
    b_path = tmp_path / "b.jsonl"
    multi = MultiLogger([JsonlLogger(a_path), JsonlLogger(b_path)])
    multi.log(5, {"loss": 0.42})
    multi.close()
    assert a_path.read_text() == b_path.read_text()
    rec = json.loads(a_path.read_text())
    assert rec["step"] == 5 and rec["loss"] == 0.42


def test_build_logger_default_is_jsonl_only(tmp_path: Path):
    cfg = OmegaConf.create({"output_dir": str(tmp_path), "log_interval": 10})
    lg = build_logger(cfg, tmp_path)
    assert isinstance(lg, JsonlLogger)
    lg.log(1, {"loss": 0.5})
    lg.close()
    assert (tmp_path / "metrics.jsonl").exists()


def test_build_logger_string_backend(tmp_path: Path):
    cfg = OmegaConf.create({"backends": "jsonl"})
    lg = build_logger(cfg, tmp_path)
    assert isinstance(lg, JsonlLogger)
    lg.close()


def test_build_logger_unknown_backend_raises(tmp_path: Path):
    cfg = OmegaConf.create({"backends": ["nonexistent"]})
    with pytest.raises(ValueError, match="unknown logging backend"):
        build_logger(cfg, tmp_path)


def test_build_logger_jsonl_plus_tensorboard(tmp_path: Path):
    cfg = OmegaConf.create({"backends": ["jsonl", "tensorboard"]})
    lg = build_logger(cfg, tmp_path)
    assert isinstance(lg, MultiLogger)
    lg.log(1, {"loss": 1.0})
    lg.close()
    assert (tmp_path / "metrics.jsonl").exists()
    # TB writes a tfevents file with a unique suffix; existence of any file
    # in tb/ is sufficient.
    tb_dir = tmp_path / "tb"
    assert tb_dir.exists()
    assert any(p.is_file() for p in tb_dir.iterdir())


def test_tensorboard_logger_skips_non_scalar(tmp_path: Path):
    """Strings and None values must not crash the TB writer."""
    lg = TensorBoardLogger(tmp_path / "tb")
    lg.log(1, {"loss": 1.5, "device_name": "RTX 4070", "missing": None})
    lg.close()
    assert any(p.is_file() for p in (tmp_path / "tb").iterdir())
