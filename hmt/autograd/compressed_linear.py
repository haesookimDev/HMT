"""``nn.Linear`` drop-in that stores its activation in INT8 for backward.

Forward computes ``y = x W^T + b`` exactly (no quantization on the math),
but saves a block-wise INT8 packed copy of ``x`` for the backward pass
instead of the full BF16/FP32 ``x``. Decompresses on-the-fly in backward.

Trade-off: saves ~50% of activation memory at the cost of some quantization
noise on ``grad_w`` and a small dequantization compute on backward.
"""
from __future__ import annotations


import torch

from hmt.memory.activation_compress import (
    PackedInt8,
    compress_blockwise_int8,
    decompress_blockwise_int8,
)
from hmt.memory.policy import ActivationPolicy


class CompressedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, block_size: int):
        y = torch.nn.functional.linear(x, weight, bias)
        packed = compress_blockwise_int8(x, block_size=block_size)
        # save_for_backward only takes Tensors; non-tensor metadata goes on ctx.
        if bias is None:
            ctx.save_for_backward(weight, packed.q, packed.scale)
        else:
            ctx.save_for_backward(weight, packed.q, packed.scale, bias)
        ctx.has_bias = bias is not None
        ctx.orig_shape = packed.orig_shape
        ctx.padded_last = packed.padded_last
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        if ctx.has_bias:
            weight, q, scale, _bias = ctx.saved_tensors
        else:
            weight, q, scale = ctx.saved_tensors

        x = decompress_blockwise_int8(
            PackedInt8(
                q=q, scale=scale,
                orig_shape=ctx.orig_shape, padded_last=ctx.padded_last,
            ),
            dtype=ctx.x_dtype,
        )
        # Standard linear backward.
        grad_x = grad_y @ weight if weight is not None else None
        # grad_w aggregates over all leading dims.
        grad_w = (
            grad_y.reshape(-1, grad_y.shape[-1]).T
            @ x.reshape(-1, x.shape[-1])
        )
        grad_b = None
        if ctx.has_bias:
            grad_b = grad_y.reshape(-1, grad_y.shape[-1]).sum(dim=0)
        return grad_x, grad_w, grad_b, None  # None for non-tensor block_size


class CompressedLinear(torch.nn.Module):
    """Drop-in replacement for ``nn.Linear`` with INT8-compressed activation memory.

    Reuses the original linear's ``weight`` and ``bias`` Parameters by
    reference (no copy); attaching this module after optimizer construction
    is safe because parameter identity is preserved.
    """

    def __init__(self, linear: torch.nn.Linear, block_size: int = 256):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return CompressedLinearFunction.apply(x, self.weight, self.bias, self.block_size)

    def extra_repr(self) -> str:
        bias_str = "True" if self.bias is not None else "False"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={bias_str}, block_size={self.block_size}, compressed=int8"
        )


def _set_submodule(root: torch.nn.Module, qualified: str, new_module: torch.nn.Module) -> None:
    parts = qualified.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def patch_model_int8_linear(
    model: torch.nn.Module,
    policy: ActivationPolicy,
) -> list[str]:
    """Replace ``nn.Linear`` submodules with ``CompressedLinear`` per policy.

    Returns the list of patched qualified names (for logging).
    Modifies ``model`` in place. Parameter identity is preserved.
    """
    targets: list[tuple[str, torch.nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and policy.select(name) == "compress_int8":
            targets.append((name, mod))

    patched: list[str] = []
    for name, mod in targets:
        new_mod = CompressedLinear(mod, block_size=policy.block_size)
        _set_submodule(model, name, new_mod)
        patched.append(name)
    return patched
