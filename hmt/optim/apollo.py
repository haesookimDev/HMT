"""APOLLO-style approximate gradient scaling.

Replaces AdamW's per-element second moment ``v`` with a coarse-grained
approximation:

  - ``scaling="tensor"``  — ``v`` is a scalar (one per parameter tensor)
  - ``scaling="channel"`` — ``v`` is a vector along dim 0 (one entry per
    output channel of a 2D weight; for 1D params it is the per-element ``v``)

The first moment ``m`` keeps full parameter shape (vanilla AdamW). State
saving comes from collapsing ``v``: for a 768×768 weight,

  AdamW  : 2·768²       ≈ 1.18M floats
  Apollo : 768² + 1     ≈ 0.59M floats   (tensor mode → ~2× reduction)
  Apollo : 768² + 768   ≈ 0.59M floats   (channel mode)

Stack with Stage 1/2 ``LowRankAdamW`` for further reduction. The full
APOLLO-Mini (rank-1 auxiliary state) is reserved for Stage 6.2.
"""
from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch.optim import Optimizer

ScalingMode = Literal["tensor", "channel"]
_VALID_MODES: tuple[ScalingMode, ...] = ("tensor", "channel")


class ApolloAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        scaling: ScalingMode = "channel",
    ):
        if scaling not in _VALID_MODES:
            raise ValueError(f"unknown scaling '{scaling}'; expected one of {_VALID_MODES}")
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, scaling=scaling)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            scaling = group["scaling"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ApolloAdamW does not support sparse gradients")

                state = self.state[p]
                self._ensure_state(state, p, scaling)
                state["step"] += 1
                t = state["step"]

                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)

                grad_f = p.grad.detach().to(torch.float32)

                m = state["exp_avg"]
                m.mul_(beta1).add_(grad_f, alpha=1.0 - beta1)

                v = state["exp_avg_sq"]
                if scaling == "tensor":
                    g_sq = grad_f.pow(2).mean()  # scalar
                else:  # channel
                    if grad_f.ndim <= 1:
                        g_sq = grad_f.pow(2)
                    else:
                        reduce_dims = tuple(range(1, grad_f.ndim))
                        g_sq = grad_f.pow(2).mean(dim=reduce_dims)  # [dim 0]
                v.mul_(beta2).add_(g_sq, alpha=1.0 - beta2)

                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                m_hat = m / bias1
                v_hat = v / bias2

                if scaling == "tensor" or grad_f.ndim <= 1:
                    denom = v_hat.sqrt() + eps
                else:
                    # broadcast [out] -> [out, 1, 1, ...]
                    broadcast_shape = (v_hat.shape[0],) + (1,) * (grad_f.ndim - 1)
                    denom = v_hat.view(broadcast_shape).sqrt() + eps

                update = m_hat / denom
                p.data.add_(update.to(p.dtype), alpha=-lr)

        return loss

    @staticmethod
    def _ensure_state(state, p, scaling: ScalingMode) -> None:
        if "step" in state:
            return
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        if scaling == "tensor":
            state["exp_avg_sq"] = torch.zeros((), dtype=torch.float32, device=p.device)
        else:  # channel
            shape = (p.shape[0],) if p.ndim >= 1 else ()
            state["exp_avg_sq"] = torch.zeros(shape, dtype=torch.float32, device=p.device)
