# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MXFP4 MLA latent cache rows: Triton quantize-store and dequantize-gather.

One token's row of an MLA layer is ``kv_lora_rank`` latent values (keys and
values share them) and ``rope_dim`` RoPE key values. An MXFP4 row stores, in
bytes::

    [ kv_lora_rank / 2 packed E2M1 | kv_lora_rank / 32 UE8M0 | rope_dim BF16 ]

(400 bytes for Kimi-K3.1's 512 + 64, against 576 for an FP8 row). The latent
encoding is bit-exact with ``mxfp4_reference.mxfp4_quantize_reference`` in
``"ceil"`` mode; the RoPE part is copied as BF16.

``dequant_gather_mla_latent_mxfp4`` reads rows by slot into BF16 scratch
rows, the fallback that lets an unchanged BF16 MLA kernel read the cache.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.quantization._triton_mxfp4 import (
    _e2m1_decode,
    _e2m1_encode,
    _ue8m0_exponent,
)

__all__ = [
    "dequant_gather_mla_latent_mxfp4",
    "mla_mxfp4_row_bytes",
    "quantize_store_mla_latent_mxfp4",
]

_GROUP = 32


def mla_mxfp4_row_bytes(kv_lora_rank: int, rope_dim: int) -> int:
    """Bytes of one MXFP4 MLA latent row."""
    if kv_lora_rank % (2 * _GROUP):
        raise ValueError(
            f"MXFP4 latent rows need kv_lora_rank % 64 == 0, got {kv_lora_rank}"
        )
    return kv_lora_rank // 2 + kv_lora_rank // _GROUP + 2 * rope_dim


@triton.jit
def _quantize_store_mla_latent_mxfp4_kernel(
    kv_ptr,  # [T, RANK] latent rows
    pe_ptr,  # [T, ROPE] RoPE key rows
    cache_ptr,  # u8 cache rows
    slots_ptr,  # [T] destination slots
    mask_ptr,  # [T] bool write mask (read only when HAS_MASK)
    kv_stride,
    pe_stride,
    page_size,
    page_stride,  # u8 elements
    row_stride,  # u8 elements
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    if HAS_MASK:
        if tl.load(mask_ptr + tok) == 0:
            return
    slot = tl.load(slots_ptr + tok).to(tl.int64)
    row = (
        cache_ptr + (slot // page_size) * page_stride + (slot % page_size) * row_stride
    )
    groups: tl.constexpr = RANK // 32

    offs = tl.arange(0, RANK)
    x = tl.load(kv_ptr + tok * kv_stride + offs).to(tl.float32)
    x = tl.reshape(x, (groups, 32))
    amax = tl.max(tl.abs(x), 1)
    # Floor amax at 6 * 2**-126 so the scale stays a normal FP32 number.
    scaled_max = tl.div_rn(tl.maximum(amax, 6.0 * 1.1754943508222875e-38), 6.0)
    exponent = tl.minimum(_ue8m0_exponent(scaled_max), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = _e2m1_encode(tl.div_rn(x, scale[:, None])).to(tl.int32)
    pairs = tl.reshape(codes, (RANK // 2, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), 1).to(tl.uint8)
    tl.store(row + tl.arange(0, RANK // 2), packed)
    tl.store(row + RANK // 2 + tl.arange(0, groups), exponent.to(tl.uint8))

    pe = tl.load(pe_ptr + tok * pe_stride + tl.arange(0, ROPE)).to(tl.bfloat16)
    pe_row = (row + RANK // 2 + groups).to(tl.pointer_type(tl.bfloat16))
    tl.store(pe_row + tl.arange(0, ROPE), pe)


def _row_geometry(cache: torch.Tensor, row_bytes: int) -> tuple[int, int, int]:
    """(page_size, page_stride, row_stride) of a u8 ``[slots, 1, row_bytes]``
    or ``[pages, page_size, 1, row_bytes]`` cache view."""
    if cache.dtype != torch.uint8 or cache.shape[-1] != row_bytes:
        raise ValueError(
            f"MXFP4 latent cache must be uint8 rows of {row_bytes} bytes, got "
            f"{tuple(cache.shape)} {cache.dtype}"
        )
    if cache.stride(-1) != 1:
        raise ValueError("MXFP4 latent cache rows must be contiguous")
    if cache.dim() == 3:
        return 1, cache.stride(0), cache.stride(0)
    if cache.dim() == 4:
        return cache.shape[1], cache.stride(0), cache.stride(1)
    raise ValueError(f"unsupported MXFP4 latent cache view {tuple(cache.shape)}")


def quantize_store_mla_latent_mxfp4(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    cache: torch.Tensor,
    slots: torch.Tensor,
    write_mask: torch.Tensor | None = None,
) -> None:
    """Quantize ``[T, kv_lora_rank]`` latent rows to MXFP4 and store them with
    their ``[T, rope_dim]`` RoPE rows (as BF16) at ``slots``.

    Args:
        kv_c: Latent rows; the trailing stride must be 1.
        k_pe: RoPE key rows; the trailing stride must be 1.
        cache: uint8 ``[slots, 1, row_bytes]`` or ``[pages, page_size, 1,
            row_bytes]`` view, ``row_bytes = mla_mxfp4_row_bytes(...)``.
        slots: ``[T]`` destination slots (page * page_size + row).
        write_mask: optional ``[T]`` bool; rows where it is False are not
            written.
    """
    kv_c = kv_c.reshape(kv_c.shape[0], -1)
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    t, rank = kv_c.shape
    rope = k_pe.shape[1]
    if k_pe.shape[0] != t or slots.shape != (t,):
        raise ValueError("kv_c, k_pe and slots must have one row per token")
    if rope & (rope - 1):
        raise ValueError(f"rope_dim must be a power of two, got {rope}")
    page_size, page_stride, row_stride = _row_geometry(
        cache, mla_mxfp4_row_bytes(rank, rope)
    )
    if t == 0:
        return
    assert kv_c.stride(-1) == 1 and k_pe.stride(-1) == 1
    _quantize_store_mla_latent_mxfp4_kernel[(t,)](
        kv_c,
        k_pe,
        cache,
        slots,
        slots if write_mask is None else write_mask,
        kv_c.stride(0),
        k_pe.stride(0),
        page_size,
        page_stride,
        row_stride,
        RANK=rank,
        ROPE=rope,
        HAS_MASK=write_mask is not None,
    )


@triton.jit
def _dequant_gather_mla_latent_mxfp4_kernel(
    cache_ptr,
    slots_ptr,  # [N] source slots, < 0 reads as a zero row
    out_ptr,  # [N, RANK + ROPE]
    out_stride,
    page_size,
    page_stride,
    row_stride,
    RANK: tl.constexpr,
    ROPE: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + i).to(tl.int64)
    valid = slot >= 0
    slot = tl.maximum(slot, 0)
    row = (
        cache_ptr + (slot // page_size) * page_stride + (slot % page_size) * row_stride
    )
    groups: tl.constexpr = RANK // 32

    packed = tl.load(row + tl.arange(0, RANK // 2), mask=valid, other=0).to(tl.int32)
    codes = tl.join(packed & 0xF, packed >> 4)  # [RANK/2, 2]: low nibble first
    values = tl.reshape(_e2m1_decode(codes), (groups, 32))
    exponent = tl.load(row + RANK // 2 + tl.arange(0, groups), mask=valid, other=0)
    scale = (exponent.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    latent = tl.reshape(values * scale[:, None], (RANK,))
    out = out_ptr + i * out_stride
    tl.store(out + tl.arange(0, RANK), latent.to(out_ptr.dtype.element_ty))

    pe_row = (row + RANK // 2 + groups).to(tl.pointer_type(tl.bfloat16))
    pe = tl.load(pe_row + tl.arange(0, ROPE), mask=valid, other=0.0)
    tl.store(out + RANK + tl.arange(0, ROPE), pe.to(out_ptr.dtype.element_ty))


def dequant_gather_mla_latent_mxfp4(
    cache: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor,
    *,
    kv_lora_rank: int,
) -> None:
    """Dequantize the MXFP4 latent rows at ``slots`` into ``out``.

    Args:
        cache: the uint8 view ``quantize_store_mla_latent_mxfp4`` writes.
        slots: ``[N]`` source slots; negative slots give zero rows (padding
            of sparse top-k index lists).
        out: ``[N, kv_lora_rank + rope_dim]`` BF16 or FP32 rows (trailing
            stride 1), latent first.
        kv_lora_rank: latent width.
    """
    n = slots.shape[0]
    out2 = out.reshape(n, -1)
    rope = out2.shape[1] - kv_lora_rank
    page_size, page_stride, row_stride = _row_geometry(
        cache, mla_mxfp4_row_bytes(kv_lora_rank, rope)
    )
    if n == 0:
        return
    assert out2.stride(-1) == 1
    _dequant_gather_mla_latent_mxfp4_kernel[(n,)](
        cache,
        slots,
        out2,
        out2.stride(0),
        page_size,
        page_stride,
        row_stride,
        RANK=kv_lora_rank,
        ROPE=rope,
    )
