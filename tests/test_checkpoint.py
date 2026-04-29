"""Tests for hmt.memory.checkpoint — Stage 7.2."""
from __future__ import annotations

import torch

from hmt.memory import load_checkpoint, save_checkpoint
from hmt.optim import LowRankAdamW, attach_projectors_from_grads, select_target_params


class _TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 32, bias=False)
        self.l2 = torch.nn.Linear(32, 16, bias=False)

    def forward(self, x):
        return self.l2(torch.nn.functional.relu(self.l1(x)))


def _make_data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(8, 16, generator=g)
    y = torch.randn(8, 16, generator=g)
    return x, y


def test_save_load_roundtrip_dense_optimizer(tmp_path):
    torch.manual_seed(0)
    m1 = _TinyMLP()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3, betas=(0.9, 0.95))
    x, y = _make_data()
    for _ in range(3):
        loss = ((m1(x) - y) ** 2).mean()
        m1.zero_grad(); loss.backward(); o1.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, step=3, model=m1, optimizer=o1, config_name="test")

    torch.manual_seed(99)  # different seed; load must override
    m2 = _TinyMLP()
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.95))
    meta = load_checkpoint(path, model=m2, optimizer=o2)

    assert meta.step == 3
    assert meta.config_name == "test"
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_resume_continues_loss_trajectory(tmp_path):
    """Run 3 then 3 contiguously vs run 3 → save → load → run 3.
    Final parameters must match exactly."""
    x, y = _make_data()

    def make():
        torch.manual_seed(7)
        return _TinyMLP()

    # Continuous reference
    m_ref = make()
    o_ref = torch.optim.AdamW(m_ref.parameters(), lr=1e-3, betas=(0.9, 0.95))
    for _ in range(6):
        loss = ((m_ref(x) - y) ** 2).mean()
        m_ref.zero_grad(); loss.backward(); o_ref.step()

    # Run 3 + save
    m_a = make()
    o_a = torch.optim.AdamW(m_a.parameters(), lr=1e-3, betas=(0.9, 0.95))
    for _ in range(3):
        loss = ((m_a(x) - y) ** 2).mean()
        m_a.zero_grad(); loss.backward(); o_a.step()
    path = tmp_path / "mid.pt"
    save_checkpoint(path, step=3, model=m_a, optimizer=o_a)

    # Fresh process simulation: load + run 3 more
    m_b = make()
    o_b = torch.optim.AdamW(m_b.parameters(), lr=1e-3, betas=(0.9, 0.95))
    load_checkpoint(path, model=m_b, optimizer=o_b)
    for _ in range(3):
        loss = ((m_b(x) - y) ** 2).mean()
        m_b.zero_grad(); loss.backward(); o_b.step()

    # Same final params
    for p_ref, p_b in zip(m_ref.parameters(), m_b.parameters()):
        assert torch.allclose(p_ref, p_b, atol=1e-6, rtol=1e-6)


def test_lowrank_optimizer_projectors_round_trip(tmp_path):
    torch.manual_seed(0)
    m1 = _TinyMLP()
    o1 = LowRankAdamW(m1.parameters(), lr=1e-2, betas=(0.9, 0.95))

    x, y = _make_data()
    # Warm-up backward to seed gradients, then attach projectors.
    loss = ((m1(x) - y) ** 2).mean()
    m1.zero_grad(); loss.backward()

    targets = select_target_params(m1, pattern="^l[12]$")
    attached = attach_projectors_from_grads(o1, targets, mode="two_sided", rank=8)
    assert len(attached) == 2

    # A few steps to populate state.
    for _ in range(3):
        loss = ((m1(x) - y) ** 2).mean()
        m1.zero_grad(); loss.backward(); o1.step()

    path = tmp_path / "lr.pt"
    save_checkpoint(path, step=3, model=m1, optimizer=o1)

    # Fresh model + fresh optimizer — must restore projectors.
    torch.manual_seed(99)
    m2 = _TinyMLP()
    o2 = LowRankAdamW(m2.parameters(), lr=1e-2, betas=(0.9, 0.95))
    load_checkpoint(path, model=m2, optimizer=o2)

    # Both projectors back, with matching mode/rank/P/Q.
    assert len(o2.projectors) == 2
    for module_name in ("l1", "l2"):
        p1 = getattr(m1, module_name).weight
        p2 = getattr(m2, module_name).weight
        proj1 = o1.projectors[id(p1)]
        proj2 = o2.projectors[id(p2)]
        assert proj1.mode == proj2.mode
        assert proj1.rank == proj2.rank
        assert torch.allclose(proj1.P.cpu(), proj2.P.cpu())
        assert torch.allclose(proj1.Q.cpu(), proj2.Q.cpu())

    # Continue training: one step on each, parameters should match.
    loss = ((m1(x) - y) ** 2).mean()
    m1.zero_grad(); loss.backward(); o1.step()
    loss = ((m2(x) - y) ** 2).mean()
    m2.zero_grad(); loss.backward(); o2.step()

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6, rtol=1e-6)


def test_rng_state_restored(tmp_path):
    torch.manual_seed(42)
    m = _TinyMLP()
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = tmp_path / "rng.pt"
    save_checkpoint(path, step=0, model=m, optimizer=o)

    # Consume some randomness, then restore.
    _ = torch.randn(8)
    a = torch.randn(4)

    load_checkpoint(path, model=m, optimizer=o)
    b = torch.randn(4)
    # After load, RNG returns to the state at save time → next draw differs from `a`.
    assert not torch.equal(a, b)
