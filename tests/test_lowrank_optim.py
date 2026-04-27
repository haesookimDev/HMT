"""Tests for hmt.optim.lowrank_adamw — Stage 1.3."""
from __future__ import annotations

import torch

from hmt.optim import LowRankAdamW, make_projector_from_grad


def _toy_regression(seed: int, out_dim: int = 16, in_dim: int = 24, n: int = 128):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    W_true = torch.randn(out_dim, in_dim, generator=g) * 0.5
    Y = X @ W_true.T + 0.1 * torch.randn(n, out_dim, generator=g)
    return X, Y


def _make_linear(in_dim: int, out_dim: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    m = torch.nn.Linear(in_dim, out_dim, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.randn(out_dim, in_dim, generator=g) * 0.1)
    return m


def test_dense_path_matches_torch_adamw():
    """With no projectors attached, LowRankAdamW must equal torch.optim.AdamW."""
    X, Y = _toy_regression(seed=0)
    in_dim, out_dim = X.shape[1], Y.shape[1]

    m_ref = _make_linear(in_dim, out_dim, seed=1)
    o_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    m_ours = _make_linear(in_dim, out_dim, seed=1)
    o_ours = LowRankAdamW(m_ours.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

    for _ in range(40):
        loss_ref = ((m_ref(X) - Y) ** 2).mean()
        m_ref.zero_grad(); loss_ref.backward(); o_ref.step()

        loss_ours = ((m_ours(X) - Y) ** 2).mean()
        m_ours.zero_grad(); loss_ours.backward(); o_ours.step()

        assert abs(loss_ref.item() - loss_ours.item()) < 1e-5
    assert torch.allclose(m_ref.weight, m_ours.weight, atol=1e-5)


def test_dense_path_decoupled_weight_decay_matches_torch():
    X, Y = _toy_regression(seed=0)
    in_dim, out_dim = X.shape[1], Y.shape[1]
    wd = 0.05

    m_ref = _make_linear(in_dim, out_dim, seed=1)
    o_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=wd)
    m_ours = _make_linear(in_dim, out_dim, seed=1)
    o_ours = LowRankAdamW(m_ours.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=wd)

    for _ in range(20):
        for m, o in [(m_ref, o_ref), (m_ours, o_ours)]:
            loss = ((m(X) - Y) ** 2).mean()
            m.zero_grad(); loss.backward(); o.step()
    assert torch.allclose(m_ref.weight, m_ours.weight, atol=1e-5)


def test_lowrank_left_mode_converges():
    """Left-projection at full rank (= out_dim) should reach near-zero loss."""
    X, Y = _toy_regression(seed=0)
    in_dim, out_dim = X.shape[1], Y.shape[1]

    m = _make_linear(in_dim, out_dim, seed=1)

    # Warm-up: one backward to seed the projector basis from a real gradient.
    loss = ((m(X) - Y) ** 2).mean()
    m.zero_grad(); loss.backward()
    proj = make_projector_from_grad(m.weight.grad, mode="left", rank=out_dim)
    initial_loss = loss.item()

    o = LowRankAdamW(
        m.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0,
        projectors={id(m.weight): proj},
    )
    o.step()  # consume the warm-up gradient
    m.zero_grad()

    losses = []
    for _ in range(120):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()
        losses.append(loss.item())

    final = losses[-1]
    assert final < initial_loss * 0.2, (
        f"left projection did not converge: {initial_loss:.4f} -> {final:.4f}"
    )


def test_lowrank_two_sided_converges_on_square_problem():
    """On a square (out_dim == in_dim) problem at rank = out_dim, both P and Q
    are square orthonormal so P P^T = Q Q^T = I and reconstruction is exact —
    convergence should match the dense AdamW path."""
    X, Y = _toy_regression(seed=0, in_dim=16, out_dim=16)
    in_dim, out_dim = X.shape[1], Y.shape[1]

    m = _make_linear(in_dim, out_dim, seed=1)
    loss = ((m(X) - Y) ** 2).mean()
    m.zero_grad(); loss.backward()
    proj = make_projector_from_grad(m.weight.grad, mode="two_sided", rank=out_dim)
    initial_loss = loss.item()

    o = LowRankAdamW(
        m.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0,
        projectors={id(m.weight): proj},
    )
    o.step()
    m.zero_grad()

    losses = []
    for _ in range(120):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()
        losses.append(loss.item())

    # AdamW's element-wise scaling lives in the rotated frame, so per-step
    # trajectory differs from dense AdamW even at full rank. Assert only that
    # progress is monotone-ish and substantial.
    assert losses[-1] < initial_loss * 0.5
    assert losses[-1] < losses[20]


def test_lowrank_two_sided_rectangular_makes_progress_without_refresh():
    """On a rectangular problem, fixed-basis two-sided cannot reach the dense
    optimum (Q Q^T is a strict projection), but should still reduce loss
    substantially. Stage 2 basis refresh is what closes the gap."""
    X, Y = _toy_regression(seed=0, in_dim=24, out_dim=16)
    in_dim, out_dim = X.shape[1], Y.shape[1]

    m = _make_linear(in_dim, out_dim, seed=1)
    loss = ((m(X) - Y) ** 2).mean()
    m.zero_grad(); loss.backward()
    proj = make_projector_from_grad(m.weight.grad, mode="two_sided", rank=min(out_dim, in_dim))
    initial_loss = loss.item()

    o = LowRankAdamW(
        m.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0,
        projectors={id(m.weight): proj},
    )
    o.step()
    m.zero_grad()

    for _ in range(120):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()

    # Substantial decrease without claiming full convergence.
    assert loss.item() < initial_loss * 0.6


def test_attach_projector_resets_state_shape():
    """attach_projector after a few dense steps must reset state to low-rank shape."""
    m = torch.nn.Linear(8, 6, bias=False)
    o = LowRankAdamW(m.parameters(), lr=1e-3)
    x = torch.randn(4, 8)
    target = torch.randn(4, 6)

    # Take two dense steps so state is allocated at full param shape.
    for _ in range(2):
        loss = ((m(x) - target) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()
    assert o.state[m.weight]["exp_avg"].shape == m.weight.shape

    # Now attach and verify the next step uses low-rank state.
    loss = ((m(x) - target) ** 2).mean()
    m.zero_grad(); loss.backward()
    proj = make_projector_from_grad(m.weight.grad, mode="two_sided", rank=4)
    o.attach_projector(m.weight, proj)
    o.step()
    assert tuple(o.state[m.weight]["exp_avg"].shape) == (4, 4)
