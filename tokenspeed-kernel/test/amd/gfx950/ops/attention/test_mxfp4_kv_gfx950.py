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

"""MXFP4 paged KV attention (tokenspeed_kernel.ops.attention.mxfp4_kv) on gfx950:
the fused-page store against the torch codec, and decode / extend against
attention over the dequantized cache."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.mxfp4_reference import (
    mxfp4_dequantize_reference,
    mxfp4_quantize_reference,
)
from utils import is_cdna4

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="needs AMD gfx950")

# Muse Spark at TP8, per rank.
HQ, H, D, P = 16, 2, 64, 64
ROW = D // 2 + D // 32


def _mx():
    from tokenspeed_kernel.ops.attention import mxfp4_kv

    return mxfp4_kv


def _fuse(data, scales):
    """Split [pages, P, H, D//2] + [pages, P, H, D//32] planes -> fused pages."""
    n = data.shape[0]
    return torch.cat([data.reshape(n, -1), scales.reshape(n, -1)], 1).view(n, P, H, ROW)


def _cache(bs, ctx, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages_per = math.ceil(ctx / P)
    npages = bs * pages_per + 4
    pt = torch.randperm(npages, device="cuda", generator=g)[: bs * pages_per]
    pt = pt.view(bs, pages_per).int()
    k = torch.randn(npages, P, H, D, device="cuda", generator=g, dtype=torch.bfloat16)
    v = torch.randn(npages, P, H, D, device="cuda", generator=g, dtype=torch.bfloat16)
    k[..., 5] *= 4  # an outlier channel
    kd, ks = mxfp4_quantize_reference(k)
    vd, vs = mxfp4_quantize_reference(v)
    kq = mxfp4_dequantize_reference(kd, ks)
    vq = mxfp4_dequantize_reference(vd, vs)
    return pt, _fuse(kd, ks), _fuse(vd, vs), kq, vq


def _ref(q, kf, vf, pt, seqlens, qlens, window_left, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    row = 0
    for b, qlen in enumerate(qlens):
        n = int(seqlens[b])
        pages = pt[b, : math.ceil(n / P)].long()
        kk = kf[pages].reshape(-1, H, D)[:n].repeat_interleave(HQ // H, dim=1)
        vv = vf[pages].reshape(-1, H, D)[:n].repeat_interleave(HQ // H, dim=1)
        for t in range(qlen):
            end = n - (qlen - 1 - t)
            lo = max(0, end - (window_left + 1)) if window_left >= 0 else 0
            s = torch.einsum("hd,nhd->hn", q[row].float(), kk[lo:end]) * scale
            out[row] = torch.einsum("hn,nhd->hd", s.softmax(-1), vv[lo:end])
            row += 1
    return out


def _rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


def test_store_matches_codec():
    mx = _mx()
    t = 300
    k = torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16) * 3
    v = torch.randn(t, H, D, device="cuda", dtype=torch.bfloat16)
    slots = torch.randperm(16 * P, device="cuda")[:t]
    kc = torch.zeros(16, P, H, ROW, dtype=torch.uint8, device="cuda")
    vc = torch.zeros_like(kc)
    mx.mx4_store_kv(k, v, slots, kc, None, vc, None)
    for x, cache in ((k, kc), (v, vc)):
        data, scales = mxfp4_quantize_reference(x)
        want = torch.zeros(16 * P, H, D // 2, dtype=torch.uint8, device="cuda")
        want_sf = torch.zeros(16 * P, H, D // 32, dtype=torch.uint8, device="cuda")
        want[slots.long()] = data
        want_sf[slots.long()] = scales
        assert torch.equal(
            cache, _fuse(want.view(16, P, H, -1), want_sf.view(16, P, H, -1))
        )


@pytest.mark.parametrize("fmt", ["w4a16ds", "w4a4"])
@pytest.mark.parametrize(
    "bs,ctx,qlen,window_left", [(3, 1000, 1, -1), (3, 3000, 1, 2047), (2, 700, 4, -1)]
)
def test_decode(fmt, bs, ctx, qlen, window_left):
    mx = _mx()
    pt, kc, vc, kq, vq = _cache(bs, ctx, seed=bs * ctx + qlen)
    seqlens = torch.randint(ctx // 2, ctx + 1, (bs,), device="cuda", dtype=torch.int32)
    q = torch.randn(bs * qlen, HQ, D, device="cuda", dtype=torch.bfloat16)
    out = mx.mx4_decode(
        q,
        kc,
        vc,
        pt,
        seqlens,
        fmt=mx.FMT_NAMES[fmt],
        window_left=window_left,
        qlen=qlen,
    )
    ref = _ref(q, kq, vq, pt, seqlens, [qlen] * bs, window_left, D**-0.5)
    # w4a4 also rounds the query to MXFP4 (measured 0.14-0.17 here).
    assert _rel(out, ref) < (1e-2 if fmt == "w4a16ds" else 2e-1)


@pytest.mark.parametrize("fmt", ["w4a16ds", "w4a4"])
@pytest.mark.parametrize(
    "prefix,qlens,window_left", [(200, (150, 37), -1), (2100, (300,), 2047)]
)
def test_extend(fmt, prefix, qlens, window_left):
    mx = _mx()
    bs = len(qlens)
    ctx = prefix + max(qlens)
    pt, kc, vc, kq, vq = _cache(bs, ctx, seed=prefix)
    seqlens = torch.tensor(
        [prefix + n for n in qlens], device="cuda", dtype=torch.int32
    )
    cu = torch.tensor(
        [0, *torch.tensor(qlens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    q = torch.randn(sum(qlens), HQ, D, device="cuda", dtype=torch.bfloat16)
    out = mx.mx4_extend(
        q,
        kc,
        vc,
        pt,
        seqlens,
        cu,
        max(qlens),
        fmt=mx.FMT_NAMES[fmt],
        window_left=window_left,
        ppi=2,
    )
    ref = _ref(q, kq, vq, pt, seqlens, qlens, window_left, D**-0.5)
    assert _rel(out, ref) < (1e-2 if fmt == "w4a16ds" else 2e-1)


def test_dequant_pages_matches_codec():
    mx = _mx()
    pt, kc, vc, kq, vq = _cache(1, 20 * P, seed=7)
    ids = pt[0, ::3].contiguous()
    k_out = torch.empty(ids.shape[0], P, H, D, dtype=torch.bfloat16, device="cuda")
    v_out = torch.empty_like(k_out)
    mx.mx4_dequant_pages(kc, vc, ids, k_out, v_out)
    assert torch.equal(k_out, kq[ids.long()].to(torch.bfloat16))
    assert torch.equal(v_out, vq[ids.long()].to(torch.bfloat16))
