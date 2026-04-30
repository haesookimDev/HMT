"""Tests for hmt.optim.lowrank_adamw — Stage 1.3 + F.1 realign_state."""
from __future__ import annotations

import torch

from hmt.optim import LowRankAdamW, make_projector_from_grad
from hmt.optim.setup import refresh_projectors_from_grads


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


def test_lowrank_state_strictly_smaller_than_dense():
    """1.6 verify (proxy on macOS): per-layer optimizer state element count
    for two-sided rank-r is r²/(out·in) of the dense state. Reports the
    reduction ratio for the run's log."""
    out_dim, in_dim, rank = 768, 768, 64
    w = torch.nn.Parameter(torch.randn(out_dim, in_dim))
    w.grad = torch.randn(out_dim, in_dim)

    o = LowRankAdamW([w], lr=1e-3)
    proj = make_projector_from_grad(w.grad, mode="two_sided", rank=rank)
    o.attach_projector(w, proj)
    o.step()

    state = o.state[w]
    lr_elems = state["exp_avg"].numel() + state["exp_avg_sq"].numel()
    dense_elems = 2 * out_dim * in_dim
    ratio = dense_elems / lr_elems

    # two_sided rank=64 on 768x768: dense=1,179,648  vs  low-rank=8,192  -> 144x.
    assert lr_elems == 2 * rank * rank
    assert ratio > 100, f"expected >100x state reduction, got {ratio:.1f}x"
    print(f"[stage1.6] dense state {dense_elems} elem  vs  low-rank {lr_elems} elem  ({ratio:.1f}x)")


def test_dummy_mlp_lowrank_makes_progress():
    """1.6 end-to-end: a tiny MLP trains with LowRankAdamW (full-rank two-sided)
    on a synthetic regression and reduces loss substantially. Catches integration
    breakage between projector + optimizer + nn.Module."""
    torch.manual_seed(0)
    h = 16

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(h, h * 4, bias=False)
            self.l2 = torch.nn.Linear(h * 4, h, bias=False)

        def forward(self, x):
            return self.l2(torch.nn.functional.relu(self.l1(x)))

    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, h, generator=g)
    # Learnable target: a fixed (random) linear map of X with mild noise.
    W_target = torch.randn(h, h, generator=g) * 0.5
    Y = (X @ W_target) + 0.05 * torch.randn(64, h, generator=g)

    torch.manual_seed(1)
    m = TinyMLP()
    o = LowRankAdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95))

    # First backward to seed projectors at full rank for each Linear weight.
    loss0 = ((m(X) - Y) ** 2).mean()
    m.zero_grad(); loss0.backward()
    for p in m.parameters():
        if p.ndim == 2:
            proj = make_projector_from_grad(p.grad, mode="two_sided", rank=min(p.shape))
            o.attach_projector(p, proj)
    o.step()

    initial = loss0.item()
    for _ in range(60):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward(); o.step()

    assert loss.item() < initial * 0.6, f"no progress: {initial:.4f} -> {loss.item():.4f}"


def test_realign_state_identity_basis_change_is_noop():
    """When P_new = P_old and Q_new = Q_old, realign_state must leave m, v
    bit-identical (the rotation is identity)."""
    torch.manual_seed(0)
    out_dim, in_dim, rank = 12, 16, 4
    w = torch.nn.Parameter(torch.randn(out_dim, in_dim))
    w.grad = torch.randn(out_dim, in_dim)

    o = LowRankAdamW([w], lr=1e-3)
    proj = make_projector_from_grad(w.grad, mode="two_sided", rank=rank)
    o.attach_projector(w, proj)
    o.step()

    m_before = o.state[w]["exp_avg"].clone()
    v_before = o.state[w]["exp_avg_sq"].clone()

    P_old, Q_old = proj.P.clone(), proj.Q.clone()
    o.realign_state(w, P_old, Q_old, proj.P, proj.Q, proj.mode)

    assert torch.allclose(o.state[w]["exp_avg"], m_before, atol=1e-5)
    assert torch.allclose(o.state[w]["exp_avg_sq"], v_before, atol=1e-5)


def test_realign_state_under_known_orthonormal_rotation():
    """If we choose new bases as a rotation of the old, m must transform as
    L @ m @ R where L = P_new.T @ P_old, R = Q_old.T @ Q_new."""
    torch.manual_seed(1)
    out_dim, in_dim, rank = 8, 8, 4
    w = torch.nn.Parameter(torch.randn(out_dim, in_dim))
    w.grad = torch.randn(out_dim, in_dim)

    o = LowRankAdamW([w], lr=1e-3)
    proj_old = make_projector_from_grad(w.grad, mode="two_sided", rank=rank)
    o.attach_projector(w, proj_old)
    o.step()
    m_old = o.state[w]["exp_avg"].clone()

    # Build a different orthonormal basis with the same shape as P_old, Q_old.
    P_new, _ = torch.linalg.qr(torch.randn(out_dim, rank))
    Q_new, _ = torch.linalg.qr(torch.randn(in_dim, rank))

    expected_m = (P_new.T @ proj_old.P) @ m_old @ (proj_old.Q.T @ Q_new)
    o.realign_state(w, proj_old.P, proj_old.Q, P_new, Q_new, "two_sided")
    assert torch.allclose(o.state[w]["exp_avg"], expected_m, atol=1e-5)


def test_refresh_with_align_does_not_diverge():
    """Frequent refresh (K=4) with alignment must keep training stable —
    no NaN, no explosion, and substantial loss reduction. Quantitative
    K-sweep showing alignment beats no-alignment lives in
    ``scripts/k_sweep.py`` since the effect is small at toy scales and
    requires longer runs to be unambiguous."""
    torch.manual_seed(0)
    in_dim, out_dim, rank = 24, 16, 8
    g = torch.Generator().manual_seed(0)
    X = torch.randn(64, in_dim, generator=g)
    W_true = torch.randn(out_dim, in_dim, generator=g) * 0.5
    Y = X @ W_true.T + 0.05 * torch.randn(64, out_dim, generator=g)

    torch.manual_seed(1)
    m = torch.nn.Linear(in_dim, out_dim, bias=False)
    o = LowRankAdamW(m.parameters(), lr=5e-3)

    loss0 = ((m(X) - Y) ** 2).mean()
    m.zero_grad(); loss0.backward()
    proj = make_projector_from_grad(m.weight.grad, mode="two_sided", rank=rank)
    o.attach_projector(m.weight, proj)
    o.step()

    targets = [("w", m.weight)]
    initial = loss0.item()
    final = initial
    for step in range(80):
        loss = ((m(X) - Y) ** 2).mean()
        m.zero_grad(); loss.backward()
        if step > 0 and step % 4 == 0:
            refresh_projectors_from_grads(o, targets, align_state=True)
        o.step()
        final = loss.item()
        assert not torch.isnan(torch.tensor(final)), f"NaN at step {step}"
        assert final < 10 * initial, f"explosion at step {step}: {final} >> {initial}"

    assert final < initial * 0.85, f"no progress: {initial:.4f} -> {final:.4f}"


def test_realign_state_left_mode():
    """left mode: state shape [r, in_dim]; rotation is L @ m only."""
    torch.manual_seed(2)
    out_dim, in_dim, rank = 8, 12, 4
    w = torch.nn.Parameter(torch.randn(out_dim, in_dim))
    w.grad = torch.randn(out_dim, in_dim)
    o = LowRankAdamW([w], lr=1e-3)
    proj = make_projector_from_grad(w.grad, mode="left", rank=rank)
    o.attach_projector(w, proj)
    o.step()
    m_old = o.state[w]["exp_avg"].clone()

    P_new, _ = torch.linalg.qr(torch.randn(out_dim, rank))
    expected = (P_new.T @ proj.P) @ m_old
    o.realign_state(w, proj.P, None, P_new, None, "left")
    assert torch.allclose(o.state[w]["exp_avg"], expected, atol=1e-5)


def test_realign_state_no_op_before_first_step():
    """Calling realign_state on a param whose state is uninitialized must
    not raise and must not allocate state."""
    w = torch.nn.Parameter(torch.randn(8, 12))
    o = LowRankAdamW([w], lr=1e-3)
    P_old = torch.eye(8, 4)
    Q_old = torch.eye(12, 4)
    P_new = torch.eye(8, 4)
    Q_new = torch.eye(12, 4)
    o.realign_state(w, P_old, Q_old, P_new, Q_new, "two_sided")
    assert w not in o.state or "exp_avg" not in o.state.get(w, {})


def test_sparse_grad_falls_back_to_dense_path():
    """F.5: HF Embedding(sparse=True) produces a sparse COO gradient. The
    optimizer must accept it (densify) and update the embedding correctly,
    matching what dense AdamW would produce on the densified gradient."""
    vocab, dim = 64, 32

    def _make_emb_with_sparse_grad(seed: int):
        g = torch.Generator().manual_seed(seed)
        emb = torch.nn.Embedding(vocab, dim, sparse=True)
        with torch.no_grad():
            emb.weight.copy_(torch.randn(vocab, dim, generator=g))
        idx = torch.tensor([3, 7, 7, 31])
        out = emb(idx).sum()
        out.backward()
        assert emb.weight.grad is not None and emb.weight.grad.is_sparse
        return emb

    # Reference: torch.optim.AdamW on the densified gradient.
    emb_ref = _make_emb_with_sparse_grad(seed=42)
    dense_grad = emb_ref.weight.grad.to_dense().clone()
    emb_ref.weight.grad = dense_grad
    o_ref = torch.optim.AdamW(emb_ref.parameters(), lr=1e-2,
                              betas=(0.9, 0.95), weight_decay=0.0)
    o_ref.step()

    # Ours: LowRankAdamW on the original sparse gradient.
    emb_ours = _make_emb_with_sparse_grad(seed=42)
    o_ours = LowRankAdamW(emb_ours.parameters(), lr=1e-2,
                          betas=(0.9, 0.95), weight_decay=0.0)
    o_ours.step()

    assert torch.allclose(emb_ref.weight, emb_ours.weight, atol=1e-5)


def test_sparse_grad_drops_projector_for_that_param():
    """If a projector was attached and then a sparse grad arrives, the
    optimizer must use the dense path (projector ignored) and not crash."""
    vocab, dim = 32, 16
    torch.manual_seed(0)
    emb = torch.nn.Embedding(vocab, dim, sparse=True)
    out = emb(torch.tensor([1, 2, 3])).sum()
    out.backward()
    sparse_grad = emb.weight.grad
    assert sparse_grad.is_sparse

    # Attach a projector built from a fake DENSE grad (so attach succeeds),
    # then call step() — the sparse grad on .grad must trigger fallback.
    fake_dense = torch.randn(vocab, dim)
    proj = make_projector_from_grad(fake_dense, mode="two_sided", rank=4)

    o = LowRankAdamW([emb.weight], lr=1e-3)
    o.attach_projector(emb.weight, proj)
    o.step()  # Must not raise.

    # State shape must be the dense shape (fallback path), not low-rank.
    assert tuple(o.state[emb.weight]["exp_avg"].shape) == (vocab, dim)


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
