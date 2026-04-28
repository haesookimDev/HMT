"""Per-module-type activation memory policy.

Selects an action for each ``nn.Linear`` (or compatible) submodule by its
qualified name. Actions::

  - "keep"          : no compression (default)
  - "compress_int8" : block-wise INT8 quantization (Stage 3.1)
  - "compress_fp8"  : reserved (Stage 3+, not implemented)
  - "recompute"     : reserved (Stage 3+, not implemented)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

ActivationAction = Literal["keep", "compress_int8", "compress_fp8", "recompute"]
_VALID_ACTIONS: tuple[ActivationAction, ...] = ("keep", "compress_int8", "compress_fp8", "recompute")


@dataclass
class ActivationRule:
    pattern: str
    action: ActivationAction

    def __post_init__(self) -> None:
        if self.action not in _VALID_ACTIONS:
            raise ValueError(f"unknown action '{self.action}', must be one of {_VALID_ACTIONS}")
        # Validate regex up-front so config errors fail fast.
        re.compile(self.pattern)


@dataclass
class ActivationPolicy:
    rules: list[ActivationRule] = field(default_factory=list)
    default: ActivationAction = "keep"
    block_size: int = 256

    def __post_init__(self) -> None:
        if self.default not in _VALID_ACTIONS:
            raise ValueError(f"unknown default '{self.default}'")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

    def select(self, module_name: str) -> ActivationAction:
        for rule in self.rules:
            if re.search(rule.pattern, module_name):
                return rule.action
        return self.default

    @classmethod
    def from_config(cls, cfg) -> "ActivationPolicy":
        """Construct from an OmegaConf/DictConfig-like object.

        Expected structure::

            block_size: 256
            default: keep
            rules:
              - pattern: "..."
                action: compress_int8
        """
        rules_cfg = cfg.get("rules", []) if hasattr(cfg, "get") else cfg["rules"]
        rules = [ActivationRule(pattern=str(r["pattern"]), action=str(r["action"])) for r in rules_cfg]
        default = str(cfg.get("default", "keep")) if hasattr(cfg, "get") else "keep"
        block_size = int(cfg.get("block_size", 256)) if hasattr(cfg, "get") else 256
        return cls(rules=rules, default=default, block_size=block_size)

    def filter(self, names: Iterable[str], action: ActivationAction) -> list[str]:
        return [n for n in names if self.select(n) == action]
