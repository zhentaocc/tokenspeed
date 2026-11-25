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


"""The MXFP4 MHA KV pool (``--kv-cache-dtype mxfp4``).

The pool sizes 34-byte rows per head (head_dim 64: 32 E2M1 bytes + 2 UE8M0
scales), hands the attention prologue a BF16 staging target, and the
attention backend quantizes the staged rows into the fused pages at its
kernel page size. Checks the sizing, the staging target and that store path
against the torch codec, at a kernel page smaller than the pool page.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import (
    mxfp4_row_bytes,
    resolve_dtype,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

HEADS = 2
HEAD_DIM = 64
ROW = HEAD_DIM // 2 + HEAD_DIM // 32
LAYERS = 2


def _make_pool(page_size: int = 128, size: int = 512):
    from test.runtime.cache_pool_test_utils import make_arena, make_mha_memory_plan

    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP4,
    )

    arena = make_arena(
        make_mha_memory_plan(
            size=size,
            prefix_granularity=page_size,
            layer_num=LAYERS,
            kv_heads=HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            mxfp4=True,
        )
    )
    return MHATokenToKVPoolMXFP4(
        arena,
        dtype=torch.bfloat16,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        rank=0,
    )


def test_row_bytes_and_dtype():
    assert mxfp4_row_bytes(64) == 34
    assert mxfp4_row_bytes(128) == 68
    assert resolve_dtype("mxfp4") == torch.uint8


def test_size_accounting():
    pool = _make_pool()
    k_size, v_size = pool.get_kv_size_bytes()
    slots = pool.arena.size + pool.arena.prefix_granularity
    assert k_size == v_size == slots * HEADS * ROW * LAYERS


def test_write_target_is_bf16_staging():
    pool = _make_pool()
    slots = torch.randperm(pool.arena.size, device="cuda")[:40]
    target = pool.kv_write_target(0, slots, None)
    assert target.k_cache.dtype == torch.bfloat16
    assert target.k_cache.shape[1:] == (HEADS, HEAD_DIM)
    assert torch.equal(target.slots, torch.arange(40, device="cuda"))
    assert target.slots.dtype == slots.dtype
    with pytest.raises(NotImplementedError):
        pool.set_kv_buffer(SimpleNamespace(layer_id=0), slots, None, None)


@pytest.mark.parametrize("kernel_page", [64, 128])
def test_staged_store_matches_codec(kernel_page):
    from tokenspeed_kernel.ops.attention.mxfp4_kv import mx4_store_kv
    from tokenspeed_kernel.ops.kvcache.mxfp4_reference import mxfp4_quantize_reference

    torch.manual_seed(0)
    pool = _make_pool(page_size=128)
    t = 96
    k = torch.randn(t, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 3
    v = torch.randn(t, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    loc = torch.randperm(pool.arena.size, device="cuda")[:t]

    # The prologue's write into the staging target.
    target = pool.kv_write_target(1, loc, None)
    target.k_cache[target.slots] = k
    target.v_cache[target.slots] = v

    # The backend's store, at its kernel page size.
    shape = (-1, kernel_page, HEADS, ROW)
    k_pages = pool.get_key_buffer(1).view(shape)
    v_pages = pool.get_value_buffer(1).view(shape)
    k_rows, v_rows = pool.staged_rows(t)
    mx4_store_kv(k_rows, v_rows, loc, k_pages, None, v_pages, None)

    for x, pages in ((k, k_pages), (v, v_pages)):
        data, scales = mxfp4_quantize_reference(x)
        page, row = loc // kernel_page, loc % kernel_page
        flat = pages.view(pages.shape[0], -1)
        n = kernel_page * HEADS
        got_data = flat[:, : n * (HEAD_DIM // 2)].view(
            -1, kernel_page, HEADS, HEAD_DIM // 2
        )
        got_sf = flat[:, n * (HEAD_DIM // 2) :].view(
            -1, kernel_page, HEADS, HEAD_DIM // 32
        )
        assert torch.equal(got_data[page, row], data)
        assert torch.equal(got_sf[page, row], scales)
    # The other layer is untouched.
    assert not pool.get_key_buffer(0).any()
