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

"""MXFP4 MLA latent quantize-store / dequantize-gather against the reference."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.mla_mxfp4 import (
    dequant_gather_mla_latent_mxfp4,
    mla_mxfp4_row_bytes,
    quantize_store_mla_latent_mxfp4,
)
from tokenspeed_kernel.ops.kvcache.mxfp4_reference import (
    mxfp4_fake_quantize,
    mxfp4_quantize_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU.")

_RANK, _ROPE = 512, 64


def _rows(t: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kv = torch.randn(t, _RANK, device="cuda") * torch.rand(t, 1, device="cuda") * 8
    kv[:, ::53] *= 30  # outlier channels
    kv[0, :32] = 0  # a zero group clamps to the smallest exponent
    kv[1, 32:64] = 1e-30  # a group below the normal floor
    pe = torch.randn(t, _ROPE, device="cuda", dtype=torch.bfloat16)
    return kv.to(dtype), pe


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("page_size", [0, 64])
def test_store_is_bit_exact_with_reference(dtype, page_size):
    torch.manual_seed(0)
    t, num_slots = 37, 256
    row_bytes = mla_mxfp4_row_bytes(_RANK, _ROPE)
    assert row_bytes == 400
    kv, pe = _rows(t, dtype)
    slots = torch.randperm(num_slots, device="cuda")[:t]
    flat = torch.full((num_slots, 1, row_bytes), 0xAB, dtype=torch.uint8, device="cuda")
    view = flat.view(num_slots // page_size, page_size, 1, -1) if page_size else flat
    mask = torch.ones(t, dtype=torch.bool, device="cuda")
    mask[5] = False
    quantize_store_mla_latent_mxfp4(kv, pe, view, slots, mask)

    packed, scales = mxfp4_quantize_reference(kv.float())
    written = slots[mask]
    got = flat[written, 0]
    assert torch.equal(got[:, : _RANK // 2], packed[mask])
    assert torch.equal(got[:, _RANK // 2 : _RANK // 2 + _RANK // 32], scales[mask])
    assert torch.equal(
        got[:, _RANK // 2 + _RANK // 32 :].view(torch.bfloat16), pe[mask]
    )
    untouched = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    untouched[written] = False
    assert torch.all(flat[untouched] == 0xAB)


@pytest.mark.parametrize("page_size", [0, 64])
def test_gather_round_trip(page_size):
    torch.manual_seed(1)
    t, num_slots = 64, 512
    kv, pe = _rows(t, torch.bfloat16)
    slots = torch.randperm(num_slots, device="cuda")[:t]
    flat = torch.zeros(num_slots, 1, 400, dtype=torch.uint8, device="cuda")
    view = flat.view(num_slots // page_size, page_size, 1, -1) if page_size else flat
    quantize_store_mla_latent_mxfp4(kv, pe, view, slots)

    order = torch.randperm(t, device="cuda")
    read = torch.cat((slots[order], torch.tensor([-1], device="cuda")))
    out = torch.full((t + 1, _RANK + _ROPE), 7.0, dtype=torch.bfloat16, device="cuda")
    dequant_gather_mla_latent_mxfp4(view, read, out, kv_lora_rank=_RANK)
    expected = mxfp4_fake_quantize(kv[order].float()).to(torch.bfloat16)
    assert torch.equal(out[:t, :_RANK], expected)
    assert torch.equal(out[:t, _RANK:], pe[order])
    assert torch.all(out[t] == 0)


def test_rejects_bad_geometry():
    cache = torch.zeros(8, 1, 576, dtype=torch.uint8, device="cuda")
    kv, pe = _rows(2, torch.bfloat16)
    with pytest.raises(ValueError):
        quantize_store_mla_latent_mxfp4(kv, pe, cache, torch.arange(2, device="cuda"))
    with pytest.raises(ValueError):
        mla_mxfp4_row_bytes(96, 64)
