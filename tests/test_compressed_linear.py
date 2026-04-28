"""Tests for hmt.autograd.compressed_linear — Stage 3.2 + 3.4."""
from __future__ import annotations

import torch

from hmt.autograd import CompressedLinear, CompressedLinearFunction, patch_model_int8_linear
from hmt.memory.policy import ActivationPolicy, ActivationRule


def _close(a, b, *, rtol: float = 1e-2, atol: float = 1e-2) -> bool:
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


def _close_frob(a, b, *, rtol: float = 0.02) -> bool:
    """Frobenius relative-error check, robust to per-element noise from int8.
    INT8 absmax adds ~0.7% Frobenius noise; allow up to `rtol`."""
    a_f = a.float(); b_f = b.float()
    denom = max(b_f.norm().item(), 1e-9)
    return ((a_f - b_f).norm().item() / denom) < rtol


def test_forward_matches_nn_linear_exactly():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    x = torch.randn(2, 8, 64)
    y_ref = lin(x)
    y_compressed = CompressedLinearFunction.apply(x, lin.weight, lin.bias, 256)
    # Forward arithmetic is identical (compression only affects saved-for-backward).
    assert torch.allclose(y_ref, y_compressed)


def test_backward_grads_close_to_nn_linear_with_bias():
    """3.2 verify: grads close to nn.Linear in Frobenius rel-err.

    grad_x is independent of x quantization (depends only on weight) → exact.
    grad_w aggregates `grad_y @ x`, so int8 noise in `x` propagates linearly.
    Acceptable bound is the int8 noise floor (~1–2% Frobenius rel err).
    """
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=True)

    x_ref = torch.randn(4, 16, 64, requires_grad=True)
    y_ref = lin(x_ref)
    y_ref.sum().backward()
    g_x_ref = x_ref.grad.clone()
    g_w_ref = lin.weight.grad.clone()
    g_b_ref = lin.bias.grad.clone()

    lin.weight.grad = None
    lin.bias.grad = None
    x_c = x_ref.detach().clone().requires_grad_(True)
    y_c = CompressedLinearFunction.apply(x_c, lin.weight, lin.bias, 256)
    y_c.sum().backward()

    assert torch.allclose(g_x_ref, x_c.grad)              # exact
    assert _close_frob(lin.weight.grad, g_w_ref, rtol=0.02)  # int8 noise floor
    assert torch.allclose(g_b_ref, lin.bias.grad)         # bias independent of x


def test_backward_grads_close_without_bias():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False)

    x_ref = torch.randn(4, 16, 64, requires_grad=True)
    y_ref = lin(x_ref)
    y_ref.sum().backward()
    g_x_ref = x_ref.grad.clone()
    g_w_ref = lin.weight.grad.clone()

    lin.weight.grad = None
    x_c = x_ref.detach().clone().requires_grad_(True)
    y_c = CompressedLinearFunction.apply(x_c, lin.weight, lin.bias, 256)
    y_c.sum().backward()

    assert torch.allclose(g_x_ref, x_c.grad)
    assert _close_frob(lin.weight.grad, g_w_ref, rtol=0.02)


def test_compressed_linear_module_drop_in():
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64)
    cmod = CompressedLinear(lin, block_size=256)
    x = torch.randn(2, 8, 128)

    y_ref = lin(x)
    y_c = cmod(x)
    assert torch.allclose(y_ref, y_c)
    # Parameter identity preserved (optimizer sees the same tensors).
    assert cmod.weight is lin.weight
    assert cmod.bias is lin.bias


def test_patch_model_replaces_only_targeted_modules():
    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(32, 64)   # should NOT be compressed
            self.mlp_up = torch.nn.Linear(64, 128)   # should be compressed
            self.mlp_down = torch.nn.Linear(128, 32)  # should be compressed
            self.act = torch.nn.GELU()

        def forward(self, x):
            return self.mlp_down(self.act(self.mlp_up(self.act(self.fc1(x)))))

    model = Net()
    policy = ActivationPolicy(
        rules=[
            ActivationRule(pattern="mlp_up", action="compress_int8"),
            ActivationRule(pattern="mlp_down", action="compress_int8"),
        ],
        default="keep",
    )
    patched = patch_model_int8_linear(model, policy)
    assert sorted(patched) == ["mlp_down", "mlp_up"]
    assert isinstance(model.fc1, torch.nn.Linear)
    assert not isinstance(model.fc1, CompressedLinear)
    assert isinstance(model.mlp_up, CompressedLinear)
    assert isinstance(model.mlp_down, CompressedLinear)


def test_patched_model_forward_output_unchanged():
    """3.4 verify: forward output matches pre-patch within numerical tolerance."""
    torch.manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(32, 64)
            self.fc2 = torch.nn.Linear(64, 32)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    model = Net()
    x = torch.randn(2, 8, 32)
    y_pre = model(x).detach().clone()

    policy = ActivationPolicy(
        rules=[ActivationRule(pattern="fc1", action="compress_int8"),
               ActivationRule(pattern="fc2", action="compress_int8")],
        default="keep",
    )
    patch_model_int8_linear(model, policy)
    y_post = model(x).detach()
    # Forward is unaffected by compression (only saved-for-backward path differs).
    assert torch.allclose(y_pre, y_post)


def test_patched_model_end_to_end_train_step_matches():
    """One optimizer step on patched model should produce parameters within
    tolerance of the unpatched reference."""
    torch.manual_seed(0)

    def make_model():
        torch.manual_seed(42)
        m = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
        )
        return m

    x = torch.randn(8, 32)
    target = torch.randn(8, 32)

    m_ref = make_model()
    o_ref = torch.optim.SGD(m_ref.parameters(), lr=1e-2)
    loss = ((m_ref(x) - target) ** 2).mean()
    o_ref.zero_grad(); loss.backward(); o_ref.step()

    m_c = make_model()
    policy = ActivationPolicy(
        rules=[ActivationRule(pattern="0", action="compress_int8"),
               ActivationRule(pattern="2", action="compress_int8")],
        default="keep",
    )
    patch_model_int8_linear(m_c, policy)
    o_c = torch.optim.SGD(m_c.parameters(), lr=1e-2)
    loss = ((m_c(x) - target) ** 2).mean()
    o_c.zero_grad(); loss.backward(); o_c.step()

    for p_ref, p_c in zip(m_ref.parameters(), m_c.parameters()):
        # After one SGD step, the patched-model parameter differs only by the
        # int8 quant noise on grads — bounded in Frobenius rel err.
        assert _close_frob(p_c.data, p_ref.data, rtol=0.02)
