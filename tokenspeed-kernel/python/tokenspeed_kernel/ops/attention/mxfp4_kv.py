"""MXFP4 paged KV cache for GQA/MHA attention (prototype, gfx950).

Cache layout per layer (page_size P, H kv heads, head_dim D):
  k_data/v_data uint8 [pages, P, H, D // 2]  E2M1 pairs, even element in the low nibble
  k_sf/v_sf     uint8 [pages, P, H, D // 32] UE8M0 exponents (one per 32 along D)

The encoding is bit-exact with mxfp4_reference.mxfp4_quantize_reference("ceil").

Attention kernels take a KV format constexpr:
  FMT_BF16  : bf16 cache (baseline, same kernel structure)
  FMT_FP8   : fp8 e4m3 cache, upcast to bf16 (w8a16)
  FMT_A16   : MXFP4 cache, K and V dequantized to bf16 in registers, bf16 MFMA (w4a16)
  FMT_A16DS : MXFP4 cache, QK via tl.dot_scaled(bf16 x e2m1), V dequantized (w4a16)
  FMT_A4    : MXFP4 cache, Q quantized to MXFP4 in-kernel, QK via scaled FP4 MFMA,
              V dequantized, PV bf16 MFMA (w4a4 on QK)
  FMT_A4P8  : as FMT_A4, plus PV via scaled MFMA: P (times the V group scale) in
              MXFP8 e4m3 x V raw E2M1 (w4a4 QK + w4a8 PV)
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.quantization._triton_mxfp4 import (
    _e2m1_decode,
    _e2m1_encode,
    _ue8m0_exponent,
)

FMT_BF16 = tl.constexpr(0)
FMT_FP8 = tl.constexpr(1)
FMT_A16 = tl.constexpr(2)
FMT_A16DS = tl.constexpr(3)
FMT_A4 = tl.constexpr(4)
FMT_A4P8 = tl.constexpr(5)
FMT_NAMES = {"bf16": 0, "fp8": 1, "w4a16": 2, "w4a16ds": 3, "w4a4": 4, "w4a4p8": 5}
# tuned on MI355X for Muse Spark TP8 (hq16/h2/hd64/page64):
# (ppi, num_stages, split_mult, min_block_m). w4a16 (inline-asm K) ppi4+ns1+SWA
# miscompiles, so it is pinned to ppi2.
DECODE_CFG = {
    0: (2, 2, 2, 16),
    1: (2, 2, 2, 16),
    2: (2, 2, 8, 16),
    3: (4, 2, 4, 16),
    4: (2, 2, 4, 32),
    5: (2, 1, 4, 16),
}


def is_mx(fmt: int) -> bool:
    return fmt >= 2


# --------------------------------------------------------------------------- #
# quantize-store
# --------------------------------------------------------------------------- #


@triton.jit
def _mx4_encode_rows(x, ROWS: tl.constexpr, D: tl.constexpr):
    """fp32 [ROWS, D] -> (packed u8 [ROWS, D//2], exponent i32 [ROWS, D//32])."""
    g = tl.reshape(x, (ROWS, D // 32, 32))
    amax = tl.max(tl.abs(g), 2)
    scaled_max = tl.div_rn(tl.maximum(amax, 6.0 * 1.1754943508222875e-38), 6.0)
    exponent = tl.minimum(_ue8m0_exponent(scaled_max), 254)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = _e2m1_encode(tl.div_rn(g, scale[:, :, None])).to(tl.int32)
    pairs = tl.reshape(codes, (ROWS, D // 2, 2))
    lo, hi = tl.split(pairs)
    packed = (lo | (hi << 4)).to(tl.uint8)
    return packed, exponent


@triton.jit
def _mx4_store_kernel(
    k_ptr,
    v_ptr,
    slots_ptr,
    kd_ptr,
    ks_ptr,
    vd_ptr,
    vs_ptr,
    k_stride,
    v_stride,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    KPS: tl.constexpr,
    SPS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + t).to(tl.int64)
    if slot < 0:
        return
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    for which in tl.static_range(2):
        if which == 0:
            x = tl.load(k_ptr + t * k_stride + offs_h[:, None] * D + offs_d[None, :])
        else:
            x = tl.load(v_ptr + t * v_stride + offs_h[:, None] * D + offs_d[None, :])
        packed, exponent = _mx4_encode_rows(x.to(tl.float32), H, D)
        row = (slot % PAGE) * H + offs_h[:, None]
        od = (slot // PAGE) * KPS + row * (D // 2) + tl.arange(0, D // 2)[None, :]
        os_ = (slot // PAGE) * SPS + row * (D // 32) + tl.arange(0, D // 32)[None, :]
        if which == 0:
            tl.store(kd_ptr + od, packed)
            tl.store(ks_ptr + os_, exponent.to(tl.uint8))
        else:
            tl.store(vd_ptr + od, packed)
            tl.store(vs_ptr + os_, exponent.to(tl.uint8))


def mx4_planes(cache, d, sf=None):
    """(data, scales, page_stride_data, page_stride_scales, page) of one cache.

    Split layout: ``cache`` u8 [pages, P, H, D//2] plus ``sf`` u8 [pages, P, H, D//32].
    Fused layout (``sf`` None, last dim D//2 + D//32): each page holds its
    [P, H, D//2] data block followed by its [P, H, D//32] scale block.
    """
    page, h = cache.shape[1], cache.shape[2]
    if sf is not None:
        return cache, sf, page * h * (d // 2), page * h * (d // 32), page
    assert cache.shape[-1] == d // 2 + d // 32 and cache.is_contiguous()
    stride = page * h * (d // 2 + d // 32)
    flat = cache.view(-1)
    return flat, flat[page * h * (d // 2) :], stride, stride, page


def mx4_store_kv(k, v, slots, k_data, k_sf, v_data, v_sf):
    """k, v [T, H, D] (bf16), slots [T] (page * P + row; < 0 skipped).

    ``k_sf``/``v_sf`` None selects the fused page layout (see mx4_planes).
    """
    t, h, d = k.shape
    if t == 0:
        return
    assert k.stride(-1) == 1 and k.stride(-2) == d
    assert v.stride(-1) == 1 and v.stride(-2) == d
    kd, ks, kps, sps, page = mx4_planes(k_data, d, k_sf)
    vd, vs, _, _, _ = mx4_planes(v_data, d, v_sf)
    _mx4_store_kernel[(t,)](
        k,
        v,
        slots,
        kd,
        ks,
        vd,
        vs,
        k.stride(0),
        v.stride(0),
        H=h,
        D=d,
        PAGE=page,
        KPS=kps,
        SPS=sps,
    )


def _kv_planes(k_cache, v_cache, k_sf, v_sf, fmt, d):
    if not is_mx(fmt):
        page, h = k_cache.shape[1], k_cache.shape[2]
        return k_cache, k_cache, v_cache, v_cache, page * h * d, 0
    kd, ks, kps, sps, _ = mx4_planes(k_cache, d, k_sf)
    vd, vs, _, _, _ = mx4_planes(v_cache, d, v_sf)
    return kd, ks, vd, vs, kps, sps


# --------------------------------------------------------------------------- #
# tile helpers
# --------------------------------------------------------------------------- #


@triton.jit
def _mx4_dequant(packed, sf, N: tl.constexpr, D: tl.constexpr):
    """u8 [N, D//2] + u8 [N, D//32] -> bf16 [N, D]."""
    p = packed.to(tl.int32)
    codes = tl.join(p & 0xF, p >> 4)  # [N, D//2, 2]
    vals = tl.reshape(_e2m1_decode(codes), (N, D // 32, 32))
    scale = (sf.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    vals = vals * scale[:, :, None]
    return tl.reshape(vals, (N, D)).to(tl.bfloat16)


@triton.jit
def _mx4_dequant_hw(packed, sf, N: tl.constexpr, D: tl.constexpr):
    """u8 [N, D//2] + u8 [N, D//32] -> bf16 [N, D] via v_cvt_scalef32_pk_bf16_fp4."""
    scale = (sf.to(tl.int32) << 23).to(tl.float32, bitcast=True)  # [N, G]
    scale = tl.reshape(
        tl.broadcast_to(scale[:, :, None], (N, D // 32, 16)), (N, D // 2)
    )
    x = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2",
        "=v,v,v",
        [packed.to(tl.int32), scale],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    lo = (x & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = ((x >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    return tl.reshape(tl.join(lo, hi), (N, D))


@triton.jit
def _qk(
    q,
    q4,
    qs,
    k_ptrs,
    ks_ptrs,
    kmask,
    FMT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """[M, BLOCK_N] fp32 logits of the query tile against one key tile."""
    if FMT == FMT_BF16:
        k = tl.load(k_ptrs)
        return tl.dot(q, tl.trans(k))
    elif FMT == FMT_FP8:
        k = tl.load(k_ptrs)
        return tl.dot(q, tl.trans(k.to(tl.bfloat16)))
    else:
        kp = tl.load(k_ptrs)
        ks = tl.load(ks_ptrs)
        if FMT == FMT_A16:
            k = _mx4_dequant_hw(kp, ks, BLOCK_N, D)
            return tl.dot(q, tl.trans(k))
        elif FMT == FMT_A16DS:
            return tl.dot_scaled(q, None, "bf16", tl.trans(kp), ks, "e2m1")
        else:
            return tl.dot_scaled(q4, qs, "e2m1", tl.trans(kp), ks, "e2m1")


@triton.jit
def _e2m1_to_e4m3(packed, N: tl.constexpr, D: tl.constexpr):
    """u8 [N, D//2] E2M1 pairs -> float8e4nv [N, D] of the same (unscaled) values."""
    p = packed.to(tl.int32)
    c = tl.reshape(tl.join(p & 0xF, p >> 4), (N, D))
    e = (c >> 1) & 3
    m = c & 1
    mag = tl.where(e == 0, m * 0x30, ((e + 6) << 3) | (m << 2))
    return (mag | ((c & 8) << 4)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _pv8(p, s, v8, acc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pg = tl.reshape(p * s[None, :], (BLOCK_M, BLOCK_N // 32, 32))
    pmax = tl.max(pg, 2)
    pe = _ue8m0_exponent(tl.maximum(pmax, 1e-30) / 448.0)
    pe = tl.minimum(tl.maximum(pe, 1), 254)
    pscale = (pe << 23).to(tl.float32, bitcast=True)
    p8 = tl.reshape(pg / pscale[:, :, None], (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
    return tl.dot_scaled(p8, pe.to(tl.uint8), "e4m3", v8, None, "e4m3", acc)


@triton.jit
def _pv(
    p,
    acc,
    v_ptrs,
    vs_ptrs,
    vmask,
    FMT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    if FMT == FMT_BF16:
        v = tl.load(v_ptrs)
        return tl.dot(p.to(tl.bfloat16), v, acc)
    elif FMT == FMT_FP8:
        v = tl.load(v_ptrs)
        return tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc)
    else:
        vp = tl.load(v_ptrs)
        vs = tl.load(vs_ptrs)
        if FMT != FMT_A4P8:
            v = _mx4_dequant_hw(vp, vs, BLOCK_N, D)
            return tl.dot(p.to(tl.bfloat16), v, acc)
        else:
            # out[:, g] = sum_n (p[:, n] * s[n, g]) * v4[n, g]: the products are
            # MXFP8 e4m3 (one scale per 32 keys), v4 the raw E2M1 codes (unit
            # scale). head_dim 64 = two groups.
            tl.static_assert(D == 64)
            scale = (vs.to(tl.int32) << 23).to(tl.float32, bitcast=True)  # [N, 2]
            s0, s1 = tl.split(scale)
            v8 = _e2m1_to_e4m3(vp, BLOCK_N, D)  # [N, D] raw codes as e4m3
            v0, v1 = tl.split(tl.permute(tl.reshape(v8, (BLOCK_N, 2, 32)), (0, 2, 1)))
            a0, a1 = tl.split(tl.permute(tl.reshape(acc, (BLOCK_M, 2, 32)), (0, 2, 1)))
            a0 = _pv8(p, s0, v0, a0, BLOCK_M, BLOCK_N)
            a1 = _pv8(p, s1, v1, a1, BLOCK_M, BLOCK_N)
            return tl.reshape(tl.permute(tl.join(a0, a1), (0, 2, 1)), (BLOCK_M, D))


# --------------------------------------------------------------------------- #
# decode (split KV) -- rows are (query position, head of the KV group)
# --------------------------------------------------------------------------- #


@triton.jit
def _decode_stage1(
    q_ptr,
    k_ptr,
    ks_ptr,
    v_ptr,
    vs_ptr,
    page_table,
    seqlens,
    mid_o,
    mid_lse,
    sm_scale_log2,
    pt_stride,
    num_splits,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    QLEN: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FMT: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    DK: tl.constexpr,
    PPI: tl.constexpr,
    KPS: tl.constexpr,
    SPS: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    GROUP: tl.constexpr = HQ // H
    ROWS: tl.constexpr = QLEN * GROUP
    BLOCK_N: tl.constexpr = PAGE * PPI
    rows = tl.arange(0, BLOCK_M)
    row_ok = rows < ROWS
    qpos = rows // GROUP
    head = kvh * GROUP + rows % GROUP
    token = b.to(tl.int64) * QLEN + qpos
    offs_d = tl.arange(0, D)

    seq_len = tl.load(seqlens + b)
    row_end = seq_len - (QLEN - 1 - qpos)
    first_end = tl.maximum(seq_len - (QLEN - 1), 0)
    if WINDOW_LEFT >= 0:
        kv_lo = tl.maximum(first_end - (WINDOW_LEFT + 1), 0)
        row_lo = row_end - (WINDOW_LEFT + 1)
    else:
        kv_lo = 0
        row_lo = row_end * 0
    first_page = kv_lo // PAGE
    end_page = tl.cdiv(seq_len, PAGE)
    pps = tl.cdiv(end_page - first_page, num_splits)
    p0 = first_page + split * pps
    p1 = tl.minimum(p0 + pps, end_page)

    q = tl.load(
        q_ptr + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    q4 = q
    qs = q
    if FMT >= FMT_A4:
        q4, qe = _mx4_encode_rows(q.to(tl.float32), BLOCK_M, D)
        qs = qe.to(tl.uint8)

    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, DK)
    offs_s = tl.arange(0, D // 32)
    offs_pg = offs_n // PAGE
    end_key = p1 * PAGE
    for pg in range(p0, p1, PPI):
        pgs = pg + offs_pg
        phys = tl.load(page_table + b * pt_stride + pgs, mask=pgs < p1, other=0).to(
            tl.int64
        )
        rowoff = ((offs_n % PAGE) * H + kvh)[:, None]
        dbase = phys[:, None] * KPS + rowoff * DK + offs_k[None, :]
        sbase = phys[:, None] * SPS + rowoff * (D // 32) + offs_s[None, :]
        k_ptrs = k_ptr + dbase
        v_ptrs = v_ptr + dbase
        ks_ptrs = ks_ptr + sbase
        vs_ptrs = vs_ptr + sbase
        qk = _qk(q, q4, qs, k_ptrs, ks_ptrs, None, FMT, BLOCK_N, D) * sm_scale_log2
        keys = pg * PAGE + offs_n
        rend = tl.minimum(row_end, end_key)
        vis = (keys[None, :] < rend[:, None]) & (keys[None, :] >= row_lo[:, None])
        qk = tl.where(vis, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = _pv(p, acc, v_ptrs, vs_ptrs, None, FMT, BLOCK_N, D, BLOCK_M)
        m_i = m_new
    has = l_i > 0
    o = tl.where(has[:, None], acc / tl.where(has, l_i, 1.0)[:, None], 0.0)
    lse = tl.where(has, m_i + tl.log2(tl.where(has, l_i, 1.0)), float("-inf"))
    base = (token * HQ + head) * MAX_SPLITS + split
    tl.store(mid_o + base[:, None] * D + offs_d[None, :], o, mask=row_ok[:, None])
    tl.store(mid_lse + base, lse, mask=row_ok)


@triton.jit
def _decode_stage2(
    mid_o,
    mid_lse,
    out,
    num_splits,
    HQ: tl.constexpr,
    D: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_s = tl.arange(0, MAX_SPLITS)
    smask = offs_s < num_splits
    base = (t * HQ + h) * MAX_SPLITS
    lse = tl.load(mid_lse + base + offs_s, mask=smask, other=float("-inf"))
    m = tl.max(lse, 0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m)
    o = tl.load(
        mid_o + (base + offs_s)[:, None] * D + tl.arange(0, D)[None, :],
        mask=smask[:, None],
        other=0.0,
    )
    s = tl.sum(w, 0)
    res = tl.sum(o * w[:, None], 0) / tl.where(s > 0, s, 1.0)
    tl.store(out + (t * HQ + h) * D + tl.arange(0, D), res.to(out.dtype.element_ty))


_WS = {}


def decode_splits(bs: int, h: int, pages: int, sm: int = 256, mult: int = 2) -> int:
    target = mult * sm
    s = max(1, min(64, triton.cdiv(target, bs * h), pages))
    return s


def mx4_decode(
    q,
    k_cache,
    v_cache,
    page_table,
    seqlens,
    *,
    fmt,
    window_left=-1,
    qlen=1,
    k_sf=None,
    v_sf=None,
    num_splits=None,
    max_pages=None,
    softmax_scale=None,
    num_warps=4,
    num_stages=None,
    waves_per_eu=2,
    ppi=None,
    split_mult=None,
    min_block_m=None,
):
    """q [bs*qlen, HQ, D] bf16; caches [pages, P, H, D or D//2]."""
    t, hq, d = q.shape
    bs = t // qlen
    c_ppi, c_ns, c_sm, c_mbm = DECODE_CFG[fmt]
    ppi = c_ppi if ppi is None else ppi
    num_stages = c_ns if num_stages is None else num_stages
    split_mult = c_sm if split_mult is None else split_mult
    min_block_m = c_mbm if min_block_m is None else min_block_m
    page = k_cache.shape[1]
    h = k_cache.shape[2]
    if max_pages is None:
        max_pages = page_table.shape[1]
    if window_left >= 0:
        max_pages = min(max_pages, triton.cdiv(window_left + 1, page) + 1)
    if num_splits is None:
        num_splits = decode_splits(bs, h, triton.cdiv(max_pages, ppi), mult=split_mult)
    max_splits = triton.next_power_of_2(num_splits)
    key = (t, hq, d, max_splits, q.device)
    if key not in _WS:
        _WS[key] = (
            torch.empty(t * hq * max_splits * d, dtype=torch.float32, device=q.device),
            torch.empty(t * hq * max_splits, dtype=torch.float32, device=q.device),
        )
    mid_o, mid_lse = _WS[key]
    out = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    rows = qlen * (hq // h)
    block_m = max(min_block_m, triton.next_power_of_2(rows))
    kd, ks, vd, vs, kps, sps = _kv_planes(k_cache, v_cache, k_sf, v_sf, fmt, d)
    global _LAST
    _LAST = _decode_stage1[(bs, h, num_splits)](
        q,
        kd,
        ks,
        vd,
        vs,
        page_table,
        seqlens,
        mid_o,
        mid_lse,
        scale * 1.4426950408889634,
        page_table.stride(0),
        num_splits,
        HQ=hq,
        H=h,
        D=d,
        PAGE=page,
        QLEN=qlen,
        WINDOW_LEFT=window_left,
        BLOCK_M=block_m,
        FMT=fmt,
        MAX_SPLITS=max_splits,
        DK=d // 2 if is_mx(fmt) else d,
        PPI=ppi,
        KPS=kps,
        SPS=sps,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    _decode_stage2[(t, hq)](
        mid_o, mid_lse, out, num_splits, HQ=hq, D=d, MAX_SPLITS=max_splits, num_warps=1
    )
    return out


# --------------------------------------------------------------------------- #
# extend (causal suffix of the cached keys); rows are (token, head of group)
# --------------------------------------------------------------------------- #


@triton.jit
def _extend_kernel(
    q_ptr,
    k_ptr,
    ks_ptr,
    v_ptr,
    vs_ptr,
    page_table,
    seqlens,
    cu_q,
    out,
    sm_scale_log2,
    pt_stride,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    FMT: tl.constexpr,
    DK: tl.constexpr,
    PPI: tl.constexpr,
    KPS: tl.constexpr,
    SPS: tl.constexpr,
):
    qblk = tl.program_id(0)
    kvh = tl.program_id(1)
    b = tl.program_id(2)
    GROUP: tl.constexpr = HQ // H
    BLOCK_M: tl.constexpr = BLOCK_Q * GROUP
    BLOCK_N: tl.constexpr = PAGE * PPI
    q_start = tl.load(cu_q + b)
    q_len = tl.load(cu_q + b + 1) - q_start
    if qblk * BLOCK_Q >= q_len:
        return
    seq_len = tl.load(seqlens + b)
    prefix = seq_len - q_len
    rows = tl.arange(0, BLOCK_M)
    qi = qblk * BLOCK_Q + rows // GROUP
    row_ok = qi < q_len
    head = kvh * GROUP + rows % GROUP
    token = (q_start + qi).to(tl.int64)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    q4 = q
    qs = q
    if FMT >= FMT_A4:
        q4, qe = _mx4_encode_rows(q.to(tl.float32), BLOCK_M, D)
        qs = qe.to(tl.uint8)
    row_end = prefix + qi + 1  # exclusive
    last_end = prefix + tl.minimum(qblk * BLOCK_Q + BLOCK_Q, q_len)
    if WINDOW_LEFT >= 0:
        row_lo = row_end - (WINDOW_LEFT + 1)
        first = tl.maximum(prefix + qblk * BLOCK_Q + 1 - (WINDOW_LEFT + 1), 0)
    else:
        row_lo = row_end * 0
        first = 0
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, DK)
    offs_s = tl.arange(0, D // 32)
    offs_pg = offs_n // PAGE
    pg_end = tl.cdiv(last_end, PAGE)
    for pg in range(first // PAGE, pg_end, PPI):
        pgs = pg + offs_pg
        phys = tl.load(page_table + b * pt_stride + pgs, mask=pgs < pg_end, other=0).to(
            tl.int64
        )
        rowoff = ((offs_n % PAGE) * H + kvh)[:, None]
        dbase = phys[:, None] * KPS + rowoff * DK + offs_k[None, :]
        sbase = phys[:, None] * SPS + rowoff * (D // 32) + offs_s[None, :]
        k_ptrs = k_ptr + dbase
        v_ptrs = v_ptr + dbase
        ks_ptrs = ks_ptr + sbase
        vs_ptrs = vs_ptr + sbase
        qk = _qk(q, q4, qs, k_ptrs, ks_ptrs, None, FMT, BLOCK_N, D) * sm_scale_log2
        keys = pg * PAGE + offs_n
        vis = (keys[None, :] < row_end[:, None]) & (keys[None, :] >= row_lo[:, None])
        qk = tl.where(vis, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = _pv(p, acc, v_ptrs, vs_ptrs, None, FMT, BLOCK_N, D, BLOCK_M)
        m_i = m_new
    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(
        out + token[:, None] * (HQ * D) + head[:, None] * D + offs_d[None, :],
        o.to(out.dtype.element_ty),
        mask=row_ok[:, None],
    )


def mx4_extend(
    q,
    k_cache,
    v_cache,
    page_table,
    seqlens,
    cu_q,
    max_q,
    *,
    fmt,
    window_left=-1,
    k_sf=None,
    v_sf=None,
    softmax_scale=None,
    block_q=16,
    num_warps=4,
    num_stages=1,
    waves_per_eu=2,
    ppi=1,
):
    t, hq, d = q.shape
    bs = seqlens.shape[0]
    page = k_cache.shape[1]
    h = k_cache.shape[2]
    out = torch.empty_like(q)
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    kd, ks, vd, vs, kps, sps = _kv_planes(k_cache, v_cache, k_sf, v_sf, fmt, d)
    grid = (triton.cdiv(max_q, block_q), h, bs)
    _extend_kernel[grid](
        q,
        kd,
        ks,
        vd,
        vs,
        page_table,
        seqlens,
        cu_q,
        out,
        scale * 1.4426950408889634,
        page_table.stride(0),
        HQ=hq,
        H=h,
        D=d,
        PAGE=page,
        WINDOW_LEFT=window_left,
        BLOCK_Q=block_q,
        FMT=fmt,
        DK=d // 2 if is_mx(fmt) else d,
        PPI=ppi,
        KPS=kps,
        SPS=sps,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    return out
