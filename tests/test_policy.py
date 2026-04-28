"""Tests for hmt.memory.policy — Stage 3.3."""
from __future__ import annotations

import pytest

from hmt.memory.policy import ActivationPolicy, ActivationRule


def test_first_match_wins():
    policy = ActivationPolicy(
        rules=[
            ActivationRule(pattern="mlp", action="compress_int8"),
            ActivationRule(pattern="mlp\\.dense_h_to_4h", action="keep"),
        ],
        default="keep",
    )
    # First rule matches both, so dense_h_to_4h still gets compress_int8.
    assert policy.select("layers.0.mlp.dense_h_to_4h") == "compress_int8"


def test_default_action_when_no_rule_matches():
    policy = ActivationPolicy(
        rules=[ActivationRule(pattern="attention", action="compress_int8")],
        default="keep",
    )
    assert policy.select("layers.0.mlp.fc") == "keep"


def test_rules_evaluated_in_order():
    policy = ActivationPolicy(
        rules=[
            ActivationRule(pattern="layernorm", action="keep"),
            ActivationRule(pattern="layers", action="compress_int8"),
        ],
        default="keep",
    )
    # Specific (layernorm) wins because it appears first.
    assert policy.select("layers.0.layernorm.weight") == "keep"
    assert policy.select("layers.0.attention.dense") == "compress_int8"


def test_from_config_dict_like():
    cfg = {
        "block_size": 128,
        "default": "keep",
        "rules": [
            {"pattern": "mlp\\.dense_h_to_4h", "action": "compress_int8"},
            {"pattern": "mlp\\.dense_4h_to_h", "action": "compress_int8"},
        ],
    }
    policy = ActivationPolicy.from_config(cfg)
    assert policy.block_size == 128
    assert policy.default == "keep"
    assert len(policy.rules) == 2
    assert policy.select("layers.3.mlp.dense_h_to_4h") == "compress_int8"
    assert policy.select("layers.3.attention.dense") == "keep"


def test_filter_returns_names_with_action():
    policy = ActivationPolicy(
        rules=[ActivationRule(pattern="mlp", action="compress_int8")],
        default="keep",
    )
    names = ["layers.0.attn.q", "layers.0.mlp.up", "layers.0.mlp.down", "lm_head"]
    assert policy.filter(names, "compress_int8") == ["layers.0.mlp.up", "layers.0.mlp.down"]
    assert policy.filter(names, "keep") == ["layers.0.attn.q", "lm_head"]


def test_validation_errors():
    with pytest.raises(ValueError, match="unknown action"):
        ActivationRule(pattern="x", action="garbage")
    with pytest.raises(ValueError, match="unknown default"):
        ActivationPolicy(rules=[], default="invalid")
    with pytest.raises(ValueError, match="block_size"):
        ActivationPolicy(rules=[], block_size=0)
    # Bad regex caught early
    import re as _re
    with pytest.raises(_re.error):
        ActivationRule(pattern="(unclosed", action="keep")
