"""Block-wise INT8 activation quantization.

Quantizes along the **last** dim in blocks of ``block_size``. Each block has
its own fp16 absmax-based scale ``s = max(|x|) / 127``. INT8 storage is 1/2
of BF16 plus ~0.4% scale overhead at block_size=256.

Usage::

    packed = compress_blockwise_int8(x, block_size=256)
    x_rec  = decompress_blockwise_int8(packed, dtype=x.dtype)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PackedInt8:
    q: torch.Tensor              # [..., n_blocks, block_size], int8
    scale: torch.Tensor          # [..., n_blocks], fp16
    orig_shape: Tuple[int, ...]
    padded_last: int             # last-dim size post-padding (multiple of block_size)


@torch.no_grad()
def compress_blockwise_int8(x: torch.Tensor, *, block_size: int = 256) -> PackedInt8:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if x.numel() == 0:
        raise ValueError("cannot compress an empty tensor")

    orig_shape = tuple(int(s) for s in x.shape)
    last = orig_shape[-1]
    rem = last % block_size
    if rem != 0:
        pad = block_size - rem
        x = torch.nn.functional.pad(x, (0, pad), value=0)
    padded_last = x.shape[-1]
    n_blocks = padded_last // block_size

    x_blocked = x.reshape(*x.shape[:-1], n_blocks, block_size)
    # fp32 for the absmax / division to avoid bf16 underflow on small blocks.
    x_f = x_blocked.detach().float()
    scale = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = (x_f / scale).round().clamp(-128, 127).to(torch.int8)

    return PackedInt8(
        q=q.contiguous(),
        scale=scale.squeeze(-1).to(torch.float16).contiguous(),
        orig_shape=orig_shape,
        padded_last=padded_last,
    )


def decompress_blockwise_int8(
    packed: PackedInt8,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    q = packed.q.to(torch.float32)
    scale = packed.scale.to(torch.float32).unsqueeze(-1)
    x = q * scale  # [..., n_blocks, block_size]
    new_shape = packed.orig_shape[:-1] + (packed.padded_last,)
    x = x.reshape(*new_shape)
    if packed.padded_last != packed.orig_shape[-1]:
        x = x[..., : packed.orig_shape[-1]]
    if dtype is not None and dtype != torch.float32:
        x = x.to(dtype)
    return x


class BlockwiseInt8Compressor:
    """Stateful wrapper for ``compress_blockwise_int8`` / ``decompress_blockwise_int8``."""

    def __init__(self, block_size: int = 256):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.block_size = block_size

    def compress(self, x: torch.Tensor) -> PackedInt8:
        return compress_blockwise_int8(x, block_size=self.block_size)

    def decompress(
        self,
        packed: PackedInt8,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return decompress_blockwise_int8(packed, dtype=dtype)
