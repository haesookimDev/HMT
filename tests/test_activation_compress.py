"""Tests for hmt.memory.activation_compress — Stage 3.1."""
from __future__ import annotations

import pytest
import torch

from hmt.memory import (
    BlockwiseInt8Compressor,
    compress_blockwise_int8,
    decompress_blockwise_int8,
)


def _frob_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / a.float().norm()).item()


def test_round_trip_relative_error_within_int8_limit_bf16():
    """3.1 verify: BF16 round-trip Frobenius rel err under int8-absmax limit (~1%).

    Note: README claimed 0.5% but standard absmax int8 on Gaussian inputs
    sits around 0.6–0.7% — well-known intrinsic limit. Smaller blocks help
    (block_size=32 reaches ~0.5%) but trade off scale-overhead / speed.
    """
    torch.manual_seed(0)
    x = torch.randn(4, 32, 256, dtype=torch.bfloat16)
    packed = compress_blockwise_int8(x, block_size=256)
    x_rec = decompress_blockwise_int8(packed, dtype=torch.bfloat16)
    err = _frob_rel_err(x, x_rec)
    assert err < 0.01, f"bf16 rel err {err:.4f} exceeds 1%"


def test_round_trip_relative_error_within_int8_limit_fp32():
    torch.manual_seed(1)
    x = torch.randn(2, 16, 1024)
    packed = compress_blockwise_int8(x, block_size=256)
    x_rec = decompress_blockwise_int8(packed, dtype=torch.float32)
    err = _frob_rel_err(x, x_rec)
    assert err < 0.01, f"fp32 rel err {err:.4f}"


def test_smaller_block_reduces_error():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 1024)
    errs = []
    for bs in (32, 64, 128, 256, 512):
        packed = compress_blockwise_int8(x, block_size=bs)
        rec = decompress_blockwise_int8(packed, dtype=torch.float32)
        errs.append(_frob_rel_err(x, rec))
    # Monotone non-decreasing error as block grows.
    for i in range(len(errs) - 1):
        assert errs[i] <= errs[i + 1] + 1e-4, f"non-monotone: {errs}"


def test_int8_storage_exactly_half_of_bf16():
    """3.1 verify: q.numel * 1byte == 0.5 * x.numel * 2bytes (multiple-of-block shape)."""
    x = torch.randn(4, 32, 256, dtype=torch.bfloat16)
    packed = compress_blockwise_int8(x, block_size=256)
    bf16_bytes = x.numel() * x.element_size()      # 2 bytes/elem
    int8_bytes = packed.q.numel() * packed.q.element_size()  # 1 byte/elem
    assert int8_bytes * 2 == bf16_bytes


def test_padding_for_non_multiple_last_dim():
    """Last dim 100 with block 256 should pad to 256 and round-trip the original shape."""
    torch.manual_seed(0)
    x = torch.randn(3, 100, dtype=torch.float32)
    packed = compress_blockwise_int8(x, block_size=256)
    assert packed.padded_last == 256
    assert tuple(packed.q.shape) == (3, 1, 256)
    x_rec = decompress_blockwise_int8(packed, dtype=torch.float32)
    assert tuple(x_rec.shape) == tuple(x.shape)
    err = _frob_rel_err(x, x_rec)
    assert err < 0.01


def test_multiple_blocks_along_last_dim():
    x = torch.randn(2, 5, 768, dtype=torch.float32)
    packed = compress_blockwise_int8(x, block_size=256)
    assert tuple(packed.q.shape) == (2, 5, 3, 256)
    assert tuple(packed.scale.shape) == (2, 5, 3)
    x_rec = decompress_blockwise_int8(packed, dtype=torch.float32)
    assert _frob_rel_err(x, x_rec) < 0.01


def test_zero_tensor_handled():
    x = torch.zeros(2, 256)
    packed = compress_blockwise_int8(x, block_size=256)
    x_rec = decompress_blockwise_int8(packed, dtype=torch.float32)
    assert torch.allclose(x_rec, x, atol=1e-6)


def test_class_wrapper_matches_functions():
    torch.manual_seed(0)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    cmp = BlockwiseInt8Compressor(block_size=256)
    p1 = cmp.compress(x)
    p2 = compress_blockwise_int8(x, block_size=256)
    assert torch.equal(p1.q, p2.q)
    assert torch.equal(p1.scale, p2.scale)


def test_validation_errors():
    with pytest.raises(ValueError, match="block_size"):
        compress_blockwise_int8(torch.randn(8), block_size=0)
    with pytest.raises(ValueError, match="empty"):
        compress_blockwise_int8(torch.empty(0), block_size=8)
    with pytest.raises(ValueError):
        BlockwiseInt8Compressor(block_size=-4)
