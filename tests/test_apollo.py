"""Tests for hmt.optim.apollo.ApolloAdamW — Stage 6.1."""
from __future__ import annotations

import pytest
import torch

from hmt.optim import ApolloAdamW


def _toy_regression(seed: int, in_dim: int = 24, out_dim: int = 16, n: int = 64):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    W = torch.randn(out_dim, in_dim, generator=g) * 0.5
    Y = X @ W.T + 0.05 * torch.randn(n, out_dim, generator=g)
    return X, Y


def _make_linear(in_dim: int, out_dim: int, seed: int) -> torch.nn.Linear:
    g = torch.Generator().manual_seed(seed)
    m = torch.nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.randn(out_dim, in_dim, generator=g) * 0.1)
    return m


def test_tensor_scaling_state_is_scalar():
    p = torch.nn.Parameter(torch.randn(64, 96))
    p.grad = torch.randn(64, 96)
    o = ApolloAdamW([p], lr=1e-3, scaling="tensor")
    o.step()
    state = o.state[p]
    assert state["exp_avg"].shape == p.shape       # m: full
    assert state["exp_avg_sq"].numel() == 1        # v: scalar


def test_channel_scaling_state_is_per_row_vector():
    p = torch.nn.Parameter(torch.randn(64, 96))
    p.grad = torch.randn(64, 96)
    o = ApolloAdamW([p], lr=1e-3, scaling="channel")
    o.step()
    state = o.state[p]
    assert state["exp_avg"].shape == p.shape
    assert tuple(state["exp_avg_sq"].shape) == (64,)


def test_state_size_smaller_than_torch_adamw():
    """Stage 6.1 verify: Apollo (tensor or channel) state < AdamW state."""
    out_dim = in_dim = 768

    p_ref = torch.nn.Parameter(torch.randn(out_dim, in_dim))
    p_ref.grad = torch.randn(out_dim, in_dim)
    o_ref = torch.optim.AdamW([p_ref], lr=1e-3)
    o_ref.step()
    ref = o_ref.state[p_ref]
    ref_elems = ref["exp_avg"].numel() + ref["exp_avg_sq"].numel()

    for mode in ("tensor", "channel"):
        p = torch.nn.Parameter(torch.randn(out_dim, in_dim))
        p.grad = torch.randn(out_dim, in_dim)
        o = ApolloAdamW([p], lr=1e-3, scaling=mode)
        o.step()
        st = o.state[p]
        elems = st["exp_avg"].numel() + st["exp_avg_sq"].numel()
        ratio = ref_elems / elems
        assert ratio > 1.9, f"{mode}: expected ~2× state reduction, got {ratio:.2f}×"


def test_tensor_scaling_converges_on_toy_regression():
    X, Y = _toy_regression(seed=0)
    in_dim, out_dim = X.shape[1], Y.shape[1]
    m = _make_linear(in_dim, out_dim, seed=1)
    o = ApolloAdamW(m.parameters(), lr=1e-2, scaling="tensor")

    initial = ((m(X) - Y) ** 2).mean().item()
    for _ in range(100):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()
    assert loss.item() < initial * 0.5


def test_channel_scaling_converges_on_toy_regression():
    X, Y = _toy_regression(seed=0)
    in_dim, out_dim = X.shape[1], Y.shape[1]
    m = _make_linear(in_dim, out_dim, seed=1)
    o = ApolloAdamW(m.parameters(), lr=1e-2, scaling="channel")

    initial = ((m(X) - Y) ** 2).mean().item()
    for _ in range(100):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()
    assert loss.item() < initial * 0.5


def test_handles_1d_parameter():
    """Bias-like 1D params: tensor mode → scalar v; channel mode → per-elem v."""
    p = torch.nn.Parameter(torch.randn(32))
    p.grad = torch.randn(32)
    for mode, expected_numel in [("tensor", 1), ("channel", 32)]:
        p2 = torch.nn.Parameter(p.detach().clone())
        p2.grad = p.grad.clone()
        o = ApolloAdamW([p2], lr=1e-3, scaling=mode)
        o.step()
        assert o.state[p2]["exp_avg_sq"].numel() == expected_numel


def test_decoupled_weight_decay_shrinks_weight():
    """Sanity: with non-zero wd, weight magnitude decreases each step."""
    p = torch.nn.Parameter(torch.ones(8, 8) * 1.0)
    p.grad = torch.zeros(8, 8)  # zero grad to isolate wd effect
    o = ApolloAdamW([p], lr=0.1, weight_decay=0.5, scaling="channel")
    initial = p.data.abs().mean().item()
    for _ in range(5):
        o.step()
    assert p.data.abs().mean().item() < initial


def test_validation_errors():
    p = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="unknown scaling"):
        ApolloAdamW([p], lr=1e-3, scaling="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid lr"):
        ApolloAdamW([p], lr=-1e-3, scaling="tensor")
    with pytest.raises(ValueError, match="beta1"):
        ApolloAdamW([p], lr=1e-3, betas=(1.5, 0.999), scaling="tensor")
