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

"""Torch reference codec for MXFP4 KV-cache rows.

A row is split into groups of 32 values along its last axis. Each group stores
one UE8M0 scale (a biased power-of-two exponent byte) and 32 E2M1 codes packed
two per byte, the even element in the low nibble. The encoding matches the
Triton index-row codec (``ops/quantization/_triton_mxfp4.py``):

* ``"ceil"`` exponent: ``2**ceil(log2(max(amax, 6 * 2**-126) / 6))``, so the
  group maximum never clips.
* ``"mse"`` exponent: the one of ``ceil`` and ``ceil - 1`` with the smaller
  squared error over the group (``ceil - 1`` saturates the largest values at
  6 times the scale); ties keep ``ceil``.
* E2M1 rounding is to the nearest value, ties to the even code.

These functions are the bit-exact reference for the KV quantize/dequantize
kernels and the numerics of the simulated MXFP4 KV cache. They are not fast.
"""

from __future__ import annotations

import functools
import math
from typing import Literal

import torch

MXFP4_GROUP_SIZE = 32
E2M1_MAX = 6.0
# The smallest amax whose ceil exponent is a normal FP32 scale.
_MIN_SCALED_AMAX = 2.0**-126

ExponentMode = Literal["ceil", "mse"]


def _e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (sign in bit 3) of FP32 values already divided by the scale."""
    a = scaled.abs().clamp(max=E2M1_MAX)
    code = (
        (a > 0.25).to(torch.int32)
        + (a >= 0.75).to(torch.int32)
        + (a > 1.25).to(torch.int32)
        + (a >= 1.75).to(torch.int32)
        + (a > 2.5).to(torch.int32)
        + (a >= 3.5).to(torch.int32)
        + (a > 5.0).to(torch.int32)
    )
    sign = torch.signbit(scaled).to(torch.int32) << 3
    return (code | sign).to(torch.uint8)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    """FP32 values of E2M1 codes (uint8, one code per element)."""
    # Arithmetic rather than a table lookup: no host-to-device copy, so it
    # also runs inside a captured graph.
    exponent = ((codes >> 1) & 3).float()
    mantissa = (codes & 1).float()
    magnitude = torch.where(
        exponent == 0, 0.5 * mantissa, torch.exp2(exponent - 1) * (1 + 0.5 * mantissa)
    )
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def _ceil_exponent(amax: torch.Tensor) -> torch.Tensor:
    """Biased UE8M0 exponent of ``2**ceil(log2(amax / 6))`` as int32.

    Computed from the FP32 bits, as the Triton codec does, so it is exact.
    """
    scaled = torch.clamp(amax, min=E2M1_MAX * _MIN_SCALED_AMAX) / E2M1_MAX
    bits = scaled.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return exponent.clamp(max=254)


def _scale_of(exponent: torch.Tensor) -> torch.Tensor:
    return (exponent.to(torch.int32) << 23).view(torch.float32)


def mxfp4_quantize_reference(
    x: torch.Tensor,
    *,
    exponent_mode: ExponentMode = "ceil",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to MXFP4 along its last axis.

    Args:
        x: ``[..., D]`` floating-point rows, ``D`` a multiple of 64.
        exponent_mode: ``"ceil"`` or ``"mse"`` (see the module docstring).

    Returns:
        ``(packed, scales)``: uint8 ``[..., D // 2]`` E2M1 pairs and uint8
        ``[..., D // 32]`` biased UE8M0 exponents.
    """
    d = x.shape[-1]
    if d % (2 * MXFP4_GROUP_SIZE):
        raise ValueError(f"MXFP4 rows need a multiple of 64 values, got {d}")
    groups = x.float().reshape(*x.shape[:-1], d // MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE)
    exponent = _ceil_exponent(groups.abs().amax(dim=-1))
    codes = _e2m1_codes(groups / _scale_of(exponent)[..., None])
    if exponent_mode == "mse":
        lower = (exponent - 1).clamp(min=1)
        lower_codes = _e2m1_codes(groups / _scale_of(lower)[..., None])

        def _error(c: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
            return (e2m1_decode(c) * _scale_of(e)[..., None] - groups).square().sum(-1)

        take_lower = _error(lower_codes, lower) < _error(codes, exponent)
        exponent = torch.where(take_lower, lower, exponent)
        codes = torch.where(take_lower[..., None], lower_codes, codes)
    elif exponent_mode != "ceil":
        raise ValueError(f"unknown MXFP4 exponent mode {exponent_mode!r}")
    codes = codes.reshape(*x.shape[:-1], d // 2, 2)
    packed = codes[..., 0] | (codes[..., 1] << 4)
    return packed, exponent.to(torch.uint8)


def mxfp4_dequantize_reference(
    packed: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of :func:`mxfp4_quantize_reference`: ``[..., 2 * packed_D]``."""
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1)
    d = packed.shape[-1] * 2
    values = e2m1_decode(codes).reshape(
        *packed.shape[:-1], d // MXFP4_GROUP_SIZE, MXFP4_GROUP_SIZE
    )
    values = values * _scale_of(scales.to(torch.int32))[..., None]
    return values.reshape(*packed.shape[:-1], d).to(dtype)


@functools.lru_cache(maxsize=8)
def _hadamard(n: int, device: torch.device) -> torch.Tensor:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def hadamard_matrix(n: int, device: torch.device | str) -> torch.Tensor:
    """The orthonormal ``n x n`` Sylvester Hadamard matrix (symmetric).

    Built on ``device`` and cached per device, so no host copy is needed.
    """
    return _hadamard(n, torch.device(device))


def mxfp4_fake_quantize(
    x: torch.Tensor,
    *,
    exponent_mode: ExponentMode = "ceil",
    hadamard: bool = False,
) -> torch.Tensor:
    """``x`` rounded through MXFP4 along its last axis, in ``x``'s dtype.

    With ``hadamard`` the rows are rotated by the orthonormal Hadamard matrix
    of the row width before quantization and rotated back after; for an MLA
    latent the rotation can be folded into the absorbed key and value
    projections, so it costs nothing at run time.
    """
    rows = x.float()
    if hadamard:
        h = hadamard_matrix(rows.shape[-1], rows.device)
        rows = rows @ h
    packed, scales = mxfp4_quantize_reference(rows, exponent_mode=exponent_mode)
    out = mxfp4_dequantize_reference(packed, scales)
    if hadamard:
        out = out @ h
    return out.to(x.dtype)
