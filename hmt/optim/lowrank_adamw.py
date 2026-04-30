"""LowRankAdamW: AdamW with optional per-parameter low-rank gradient projection.

Parameters with an attached ``LayerProjector`` keep optimizer state (m, v) in
the low-rank shape; their step is::

    Ĝ        = projector.project(p.grad)
    m, v     ← EMAs of Ĝ in low-rank space
    Û        = AdamW(m, v) in low-rank space
    p ← p - lr * (wd * p + projector.reconstruct(Û))

Parameters without a projector follow vanilla decoupled AdamW.

State is always kept in float32 regardless of parameter dtype, matching
PyTorch's reference AdamW behavior. Memory savings come from the *shape* of
the state (low-rank), not its precision. Stage 6 (APOLLO) may revisit this.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Optimizer

from hmt.optim.projector import LayerProjector, ProjectionMode


class LowRankAdamW(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        projectors: Optional[dict[int, LayerProjector]] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._projectors: dict[int, LayerProjector] = projectors or {}

    @property
    def projectors(self) -> dict[int, LayerProjector]:
        return self._projectors

    def attach_projector(self, param: torch.Tensor, projector: LayerProjector) -> None:
        """Register a projector for ``param``. Resets that param's optimizer
        state (the low-rank state shape differs from the dense default)."""
        self._projectors[id(param)] = projector
        if param in self.state:
            self.state[param].clear()

    @torch.no_grad()
    def realign_state(
        self,
        param: torch.Tensor,
        P_old: Optional[torch.Tensor],
        Q_old: Optional[torch.Tensor],
        P_new: Optional[torch.Tensor],
        Q_new: Optional[torch.Tensor],
        mode: ProjectionMode,
    ) -> None:
        """Rotate ``(m, v)`` from the (P_old, Q_old) low-rank basis into the
        (P_new, Q_new) basis after a projector refresh. ``m`` rotates linearly
        (basis-change of a vector); ``v`` is approximated by squaring the
        rotation matrix element-wise — keeps signs non-negative and is exact
        when the new basis equals the old one. Without this realignment a
        small K (frequent refresh) leaves stale optimizer state that points
        in directions no longer represented in the new basis."""
        state = self.state.get(param)
        if state is None or "exp_avg" not in state:
            return  # state not initialized yet (no step taken)
        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        f32 = torch.float32

        if mode == "two_sided":
            if P_old is None or Q_old is None or P_new is None or Q_new is None:
                return
            L = (P_new.to(f32).T @ P_old.to(f32))   # [r_new, r_old]
            R = (Q_old.to(f32).T @ Q_new.to(f32))   # [r_old, r_new]
            state["exp_avg"] = (L @ m @ R).contiguous()
            state["exp_avg_sq"] = ((L * L) @ v @ (R * R)).contiguous()
        elif mode == "left":
            if P_old is None or P_new is None:
                return
            L = (P_new.to(f32).T @ P_old.to(f32))
            state["exp_avg"] = (L @ m).contiguous()
            state["exp_avg_sq"] = ((L * L) @ v).contiguous()
        elif mode == "right":
            if Q_old is None or Q_new is None:
                return
            R = (Q_old.to(f32).T @ Q_new.to(f32))
            state["exp_avg"] = (m @ R).contiguous()
            state["exp_avg_sq"] = (v @ (R * R)).contiguous()
        else:
            raise ValueError(f"unknown projection mode: {mode}")

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
            for p in group["params"]:
                if p.grad is None:
                    continue

                projector = self._projectors.get(id(p))
                # F.5: HF Embedding(sparse=True) yields a sparse COO grad.
                # LowRankAdamW is an AdamW variant — it has no native sparse
                # path (torch.optim.SparseAdam is the right tool for that),
                # so we densify and force the dense AdamW branch. We also
                # drop any projector for this param: SVD over a sparse grad
                # would densify it anyway and the savings rationale (rank<<dim)
                # doesn't fit embeddings.
                if p.grad.is_sparse:
                    grad = p.grad.to_dense()
                    projector_for_step = None
                else:
                    grad = p.grad
                    projector_for_step = projector

                state = self.state[p]
                self._ensure_state(state, p, projector_for_step)
                state["step"] += 1
                t = state["step"]

                # Decoupled weight decay: applied to the parameter itself (matches
                # torch.optim.AdamW ordering).
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)

                if projector_for_step is None:
                    update = self._adamw_update(
                        state["exp_avg"], state["exp_avg_sq"], grad,
                        beta1, beta2, eps, t,
                    )
                    p.data.add_(update.to(p.dtype), alpha=-lr)
                else:
                    low_grad = projector_for_step.project(grad)
                    low_update = self._adamw_update(
                        state["exp_avg"], state["exp_avg_sq"], low_grad,
                        beta1, beta2, eps, t,
                    )
                    full_update = projector_for_step.reconstruct(low_update.to(p.dtype))
                    p.data.add_(full_update, alpha=-lr)

        return loss

    @staticmethod
    def _ensure_state(state, p, projector):
        if "step" in state:
            return
        state["step"] = 0
        shape = p.shape if projector is None else projector.low_rank_shape
        state["exp_avg"] = torch.zeros(shape, dtype=torch.float32, device=p.device)
        state["exp_avg_sq"] = torch.zeros(shape, dtype=torch.float32, device=p.device)

    @staticmethod
    def _adamw_update(
        m: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta1: float,
        beta2: float,
        eps: float,
        t: int,
    ) -> torch.Tensor:
        gf = g.detach().to(torch.float32)
        m.mul_(beta1).add_(gf, alpha=1.0 - beta1)
        v.mul_(beta2).addcmul_(gf, gf, value=1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** t
        bias_correction2 = 1.0 - beta2 ** t
        m_hat = m / bias_correction1
        v_hat = v / bias_correction2
        return m_hat / (v_hat.sqrt() + eps)
