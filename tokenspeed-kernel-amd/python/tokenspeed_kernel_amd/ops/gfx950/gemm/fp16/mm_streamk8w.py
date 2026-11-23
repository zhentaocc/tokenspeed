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

"""Experimental 8-wave hybrid Stream-K dense16 Gluon GEMM for gfx950.

``C = A @ W.T`` for skinny ``A [M, K]`` (M = 128..512) against a large
``W [N, K]``. The per-CU schedule is the 8-wave warp-pipelined MFMA/LDS
quadrant tile of largem.py (the gfx950-gluon-tutorials ``v1_sliceMN_BK64_nS2``
kernel), generalized to half tiles of 64 or 128 rows and columns (tile
``2 * BMH x 2 * BNH``) and to ``NS`` = 2-4 LDS buffers (``NS = 0`` keeps
largem's two-buffer pair schedule).

Grid: one persistent program per CU. The first ``SK_TILES`` output tiles
(the ragged last round) are done Stream-K style, the rest data-parallel. The
Stream-K tiles are split evenly over the XCDs (program ``i`` runs on XCD
``i % NUM_XCDS``); on one XCD their ``CHUNK * BLOCK_K`` K chunks form one
iteration space cut into equal contiguous ranges, one per program. A segment
that covers a whole tile stores C; any other one stores an FP32 partial and
arrives on the tile's counter. All pieces of a split tile, and their
partials, stay inside one XCD, so the partials only go through that XCD's L2.
After its data-parallel tiles a program helps reduce the split tiles it took
part in: the ``P`` pieces of a tile wait for all ``P`` partials, then each
sums every ``P``-th block over the partials in program order
(deterministic) and stores C. The last one out resets the counter.

The pieces spin on each other, so all ``GRID`` programs must be resident at
once: the launcher keeps ``GRID`` at most the CU count, and the LDS per
program keeps it at one program per CU.
"""

from __future__ import annotations

import os

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_3stage import (
    LDS_BYTES,
    _gload_layout,
    _launch_metadata,
    _shared_layout,
)

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

NUM_WARPS = 8
BLOCK_K = 64
# Module globals read inside jit code must be constexpr.
# Stream-K unit: CHUNK K steps, so every segment has at least NS steps.
CHUNK = 4
_NW = gl.constexpr(NUM_WARPS)
_BK = gl.constexpr(BLOCK_K)
_CH = gl.constexpr(CHUNK)


@gluon.jit
def _sk8w_tile(
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_ptr,
    b_ptr,
    m0,
    n0,
    k0,
    last_row,
    last_col,
    stride_am,
    stride_bn,
    iterMax,
    BMH: gl.constexpr,
    BNH: gl.constexpr,
    BK: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
):
    """Accumulate one ``2*BMH x 2*BNH`` tile over ``iterMax`` (even, >= 2) K steps.

    largem_mfma_lds_tile's schedule with layouts built for any half size of
    64 or 128. With ``iterMax == 2`` the main loop is skipped and the drain's
    wait counts still match the eight prologue groups.
    """
    NW: gl.constexpr = WARPS_M * WARPS_N
    gLoadA: gl.constexpr = _gload_layout(BMH, BK, NW, False)
    gLoadB: gl.constexpr = _gload_layout(BNH, BK, NW, True)
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )

    offs_am = gl.arange(0, BMH, gl.SliceLayout(1, gLoadA))
    offs_ak = gl.arange(0, BK, gl.SliceLayout(0, gLoadA))
    offs_bn = gl.arange(0, BNH, gl.SliceLayout(0, gLoadB))
    offs_bk = gl.arange(0, BK, gl.SliceLayout(1, gLoadB))

    a_base = a_ptr + m0.to(gl.int64) * stride_am + k0
    b_base = b_ptr + n0.to(gl.int64) * stride_bn + k0

    rows_top = gl.minimum(offs_am, last_row)
    rows_bot = gl.minimum(offs_am + BMH, last_row)
    cols_left = gl.minimum(offs_bn, last_col)
    cols_right = gl.minimum(offs_bn + BNH, last_col)

    a_top_offsets = rows_top[:, None] * stride_am + offs_ak[None, :]
    a_bot_offsets = rows_bot[:, None] * stride_am + offs_ak[None, :]
    b_left_offsets = offs_bk[:, None] + cols_left[None, :] * stride_bn
    b_right_offsets = offs_bk[:, None] + cols_right[None, :] * stride_bn

    a_top_offsets_next = a_top_offsets + BK
    a_bot_offsets_next = a_bot_offsets + BK
    b_left_offsets_next = b_left_offsets + BK
    b_right_offsets_next = b_right_offsets + BK

    acc_tl = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)

    # Every wave must be done reading the previous segment's LDS buffers
    # before this prologue overwrites them.
    gl.barrier()
    async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
    async_copy.commit_group()

    async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_offsets_next)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_top_offsets_next)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_bot.index(1), a_base, a_bot_offsets_next)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemB_right.index(1), b_base, b_right_offsets_next)
    async_copy.commit_group()

    a_base += BK * 2
    b_base += BK * 2

    async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotB)
    a_top = smemA_top.index(0).load(dotA)

    for _ in tl.range(0, iterMax - 2, 2):
        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(0).load(dotA)
            async_copy.buffer_load_to_shared(
                smemB_left.index(0), b_base, b_left_offsets
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(0).load(dotB)
            async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(1).load(dotB)
            async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(1).load(dotA)
            async_copy.buffer_load_to_shared(
                smemB_right.index(0), b_base, b_right_offsets
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(1).load(dotA)
            async_copy.buffer_load_to_shared(
                smemB_left.index(1), b_base, b_left_offsets_next
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(1).load(dotB)
            async_copy.buffer_load_to_shared(
                smemA_top.index(1), a_base, a_top_offsets_next
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(0).load(dotB)
            async_copy.buffer_load_to_shared(
                smemA_bot.index(1), a_base, a_bot_offsets_next
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(0).load(dotA)
            async_copy.buffer_load_to_shared(
                smemB_right.index(1), b_base, b_right_offsets_next
            )
            async_copy.commit_group()
            a_base += BK * 2
            b_base += BK * 2

    acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
    async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotA)

    acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
    async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotB)

    acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
    async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotB)

    acc_br = cdna4.mfma(a_bot, b_right, acc_br)
    async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotA)

    acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
    async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotA)

    acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
    async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotB)

    acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
    acc_br = cdna4.mfma(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _sk8w_ld(smem, slot, base, offsets):
    async_copy.buffer_load_to_shared(smem.index(slot), base, offsets)
    async_copy.commit_group()


@gluon.jit
def _sk8w_tile_ns(
    smemA_top,
    smemA_bot,
    smemB_left,
    smemB_right,
    a_ptr,
    b_ptr,
    m0,
    n0,
    k0,
    last_row,
    last_col,
    stride_am,
    stride_bn,
    iterMax,
    BMH: gl.constexpr,
    BNH: gl.constexpr,
    BK: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NS: gl.constexpr,
):
    """Accumulate one ``2*BMH x 2*BNH`` tile over ``iterMax`` (>= NS) K steps.

    The quadrant schedule of _sk8w_tile with ``NS`` LDS buffers: step ``s``
    lives in slot ``s % NS`` and its loads are issued ``NS`` steps ahead, so
    up to ``NS - 1`` steps of loads stay in flight behind the MFMAs. Every
    step is 4 commit groups (B left, A top, A bottom, B right).
    """
    NW: gl.constexpr = WARPS_M * WARPS_N
    gLoadA: gl.constexpr = _gload_layout(BMH, BK, NW, False)
    gLoadB: gl.constexpr = _gload_layout(BNH, BK, NW, True)
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dotA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    WAIT: gl.constexpr = 4 * NS - 3

    offs_am = gl.arange(0, BMH, gl.SliceLayout(1, gLoadA))
    offs_ak = gl.arange(0, BK, gl.SliceLayout(0, gLoadA))
    offs_bn = gl.arange(0, BNH, gl.SliceLayout(0, gLoadB))
    offs_bk = gl.arange(0, BK, gl.SliceLayout(1, gLoadB))

    a_base = a_ptr + m0.to(gl.int64) * stride_am + k0
    b_base = b_ptr + n0.to(gl.int64) * stride_bn + k0

    rows_top = gl.minimum(offs_am, last_row)
    rows_bot = gl.minimum(offs_am + BMH, last_row)
    cols_left = gl.minimum(offs_bn, last_col)
    cols_right = gl.minimum(offs_bn + BNH, last_col)
    a_top_offsets = rows_top[:, None] * stride_am + offs_ak[None, :]
    a_bot_offsets = rows_bot[:, None] * stride_am + offs_ak[None, :]
    b_left_offsets = offs_bk[:, None] + cols_left[None, :] * stride_bn
    b_right_offsets = offs_bk[:, None] + cols_right[None, :] * stride_bn

    acc_tl = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BMH, BNH), gl.float32, mfmaLayout)

    # Every wave must be done reading the previous segment's LDS buffers
    # before this prologue overwrites them.
    gl.barrier()
    for j in gl.static_range(NS):
        _sk8w_ld(smemB_left, j, b_base + j * BK, b_left_offsets)
        _sk8w_ld(smemA_top, j, a_base + j * BK, a_top_offsets)
        _sk8w_ld(smemA_bot, j, a_base + j * BK, a_bot_offsets)
        _sk8w_ld(smemB_right, j, b_base + j * BK, b_right_offsets)
    a_base += NS * BK
    b_base += NS * BK

    async_copy.wait_group(4 * NS - 2)
    b_left = smemB_left.index(0).load(dotB)
    a_top = smemA_top.index(0).load(dotA)

    for s in range(0, iterMax - NS):
        cur = s % NS
        nxt = (s + 1) % NS
        async_copy.wait_group(WAIT)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(cur).load(dotA)
            _sk8w_ld(smemB_left, cur, b_base, b_left_offsets)

        async_copy.wait_group(WAIT)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(cur).load(dotB)
            _sk8w_ld(smemA_top, cur, a_base, a_top_offsets)

        async_copy.wait_group(WAIT)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(nxt).load(dotB)
            _sk8w_ld(smemA_bot, cur, a_base, a_bot_offsets)

        async_copy.wait_group(WAIT)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(nxt).load(dotA)
            _sk8w_ld(smemB_right, cur, b_base, b_right_offsets)
            a_base += BK
            b_base += BK

    # Drain the last NS steps: no more loads, the waits count down.
    for j in gl.static_range(NS):
        cur = (iterMax - NS + j) % NS
        acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
        async_copy.wait_group(4 * (NS - j) - 3)
        a_bot = smemA_bot.index(cur).load(dotA)
        acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
        async_copy.wait_group(4 * (NS - j) - 4)
        b_right = smemB_right.index(cur).load(dotB)
        acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
        if j < NS - 1:
            nxt = (cur + 1) % NS
            async_copy.wait_group(4 * (NS - j) - 5)
            b_left = smemB_left.index(nxt).load(dotB)
            async_copy.wait_group(4 * (NS - j) - 6)
            a_top = smemA_top.index(nxt).load(dotA)
        acc_br = cdna4.mfma(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _sk8w_any_tile(smemA_top, smemA_bot, smemB_left, smemB_right, a_ptr, b_ptr, m0, n0, k0, last_row, last_col, stride_am, stride_bn, iterMax, BMH: gl.constexpr, BNH: gl.constexpr, BK: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr, NS: gl.constexpr,):  # fmt: skip
    """NS == 0: largem's two-buffer pair schedule; else the NS-buffer one."""
    if NS == 0:
        return _sk8w_tile(
            smemA_top, smemA_bot, smemB_left, smemB_right, a_ptr, b_ptr, m0, n0,
            k0, last_row, last_col, stride_am, stride_bn, iterMax,
            BMH, BNH, BK, WARPS_M, WARPS_N,
        )  # fmt: skip
    else:
        return _sk8w_tile_ns(
            smemA_top, smemA_bot, smemB_left, smemB_right, a_ptr, b_ptr, m0, n0,
            k0, last_row, last_col, stride_am, stride_bn, iterMax,
            BMH, BNH, BK, WARPS_M, WARPS_N, NS,
        )  # fmt: skip


@gluon.jit
def _sk8w_store_quad(
    acc,
    c_ptr,
    part_ptr,
    m0,
    n0,
    M,
    N,
    stride_cm,
    QM: gl.constexpr,
    QN: gl.constexpr,
    BMH: gl.constexpr,
    BNH: gl.constexpr,
    TO_PARTIAL: gl.constexpr,
):
    """Store quadrant (QM, QN) straight from the MFMA layout.

    TO_PARTIAL: FP32 into the row-major ``[2*BMH, 2*BNH]`` partial at
    ``part_ptr`` (read back through the same XCD's L2); otherwise cast and
    store into C with the M/N mask.
    """
    layout: gl.constexpr = acc.type.layout
    offs_m = gl.arange(0, BMH, gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BNH, gl.SliceLayout(0, layout))
    if TO_PARTIAL:
        cdna4.buffer_store(
            stored_value=acc,
            ptr=part_ptr,
            offsets=(QM * BMH + offs_m)[:, None] * (2 * BNH)
            + (QN * BNH + offs_n)[None, :],
        )
    else:
        rm = m0 + QM * BMH
        cn = n0 + QN * BNH
        cdna4.buffer_store(
            stored_value=acc.to(c_ptr.dtype.element_ty),
            ptr=c_ptr + rm.to(gl.int64) * stride_cm + cn,
            offsets=offs_m[:, None] * stride_cm + offs_n[None, :],
            mask=((rm + offs_m) < M)[:, None] & ((cn + offs_n) < N)[None, :],
        )


@gluon.jit
def _sk8w_sleep():
    gl.inline_asm_elementwise(
        asm="s_sleep 8",
        constraints="=v,0",
        args=[gl.arange(0, 64, gl.BlockedLayout([1], [64], [_NW], [0]))],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _sk8w_owner(x, G, GRID: gl.constexpr):
    """Program whose range [p*G//GRID, (p+1)*G//GRID) holds chunk ``x``."""
    return ((x + 1) * GRID + G - 1) // G - 1


@gluon.jit
def _sk8w_reduce(
    c_ptr,
    part_ptr,
    cnt_ptr,
    t,
    tl_,
    lid,
    xbase,
    num_pid_m,
    M,
    N,
    G,
    stride_cm,
    KP: gl.constexpr,
    PX: gl.constexpr,
    BMH: gl.constexpr,
    BNH: gl.constexpr,
):
    """Help reduce split tile ``t`` (XCD-local index ``tl_``).

    Wait for all partials, then sum every P-th block. ``lid`` is this
    program's index on its XCD and ``xbase`` the XCD's first program.
    """
    BM: gl.constexpr = 2 * BMH
    BN: gl.constexpr = 2 * BNH
    TILE: gl.constexpr = BM * BN
    FIX: gl.constexpr = 4 * 64 * _NW
    NBLK: gl.constexpr = TILE // FIX
    fixL: gl.constexpr = gl.BlockedLayout([4], [64], [_NW], [0])
    RB: gl.constexpr = 8
    p0 = _sk8w_owner(tl_ * KP, G, PX)
    p1 = _sk8w_owner(tl_ * KP + KP - 1, G, PX)
    P = p1 - p0 + 1
    arrived = gl.atomic_add(cnt_ptr + t, 0, sem="relaxed", scope="gpu")
    while arrived < P:
        _sk8w_sleep()
        arrived = gl.atomic_add(cnt_ptr + t, 0, sem="relaxed", scope="gpu")
    pid_m = t % num_pid_m
    pid_n = t // num_pid_m
    m0 = pid_m * BM
    n0 = pid_n * BN
    # Partial slot of program q in this tile: 2q if the tile holds q's range
    # start, else 2q + 1 (then it is q's last segment; only p0 can be that).
    start_p0 = (p0 * G) // PX
    slot0 = 2 * (xbase + p0) + (start_p0 < tl_ * KP).to(gl.int32)
    offs = gl.arange(0, FIX, fixL)
    c_base = c_ptr + m0.to(gl.int64) * stride_cm + n0
    for blk in range(lid - p0, NBLK, P):
        e = blk * FIX + offs
        out = cdna4.buffer_load(
            ptr=part_ptr + slot0.to(gl.int64) * TILE, offsets=e, cache=".cg"
        )
        # RB partials per batch so their loads are in flight together.
        for q in range(p0 + 1, p1 + 1, RB):
            for i in gl.static_range(RB):
                out += cdna4.buffer_load(
                    ptr=part_ptr + (2 * (xbase + q + i)).to(gl.int64) * TILE,
                    offsets=e, mask=(offs < FIX) & (q + i <= p1), other=0.0,
                    cache=".cg",
                )  # fmt: skip
        er = e // BN
        ec = e % BN
        cdna4.buffer_store(
            stored_value=out.to(c_ptr.dtype.element_ty),
            ptr=c_base,
            offsets=er * stride_cm + ec,
            mask=(m0 + er < M) & (n0 + ec < N),
        )
    # Arrival 2P - 1: every piece has left the wait above.
    done = gl.atomic_add(cnt_ptr + t, 1, sem="relaxed", scope="gpu")
    if done == 2 * P - 1:
        gl.store(cnt_ptr + t, 0)


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("M", "SK_TILES"))
def gluon_mm_a16w16_streamk8w_gfx950(
    a_ptr,
    b_ptr,
    c_ptr,
    part_ptr,
    cnt_ptr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_am,
    stride_bn: gl.constexpr,
    stride_cm,
    SK_TILES,
    BMH: gl.constexpr,
    BNH: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    GRID: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NS: gl.constexpr,
):
    BK: gl.constexpr = _BK
    BM: gl.constexpr = 2 * BMH
    BN: gl.constexpr = 2 * BNH
    CH: gl.constexpr = _CH
    KP: gl.constexpr = K // (CH * BK)
    NBUF: gl.constexpr = 2 if NS == 0 else NS
    gl.static_assert(K % (CH * BK) == 0, "K must be a multiple of CHUNK * BLOCK_K")
    gl.static_assert(GRID % NUM_XCDS == 0, "GRID must be a multiple of NUM_XCDS")
    NW: gl.constexpr = WARPS_M * WARPS_N
    gl.static_assert(NW == _NW, "8 waves")

    sharedA: gl.constexpr = _shared_layout(BMH, BK, NW, False)
    sharedB: gl.constexpr = _shared_layout(BNH, BK, NW, True)
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NBUF, BMH, BK], sharedA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NBUF, BMH, BK], sharedA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NBUF, BK, BNH], sharedB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NBUF, BK, BNH], sharedB
    )

    pid = gl.program_id(0)
    # Program i runs on XCD i % NUM_XCDS. Each XCD owns PX consecutive
    # virtual ids and a contiguous share of the Stream-K tiles, so all pieces
    # of a split tile, and their partials, stay in that XCD's L2.
    PX: gl.constexpr = GRID // NUM_XCDS
    xcd = pid % NUM_XCDS
    lid = pid // NUM_XCDS
    xbase = xcd * PX
    vid = xbase + lid
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n: gl.constexpr = (N + BN - 1) // BN
    T = num_pid_m * num_pid_n
    # Stream-K part: this XCD's tiles [tb, te) of [0, SK_TILES) form one
    # chunk space cut into PX ranges.
    tb = (xcd * SK_TILES) // NUM_XCDS
    te = ((xcd + 1) * SK_TILES) // NUM_XCDS
    G = (te - tb) * KP
    start = (lid * G) // PX
    end = ((lid + 1) * G) // PX

    x = start
    while x < end:
        t = tb + x // KP
        ks = x - (t - tb) * KP
        ke = gl.minimum(end - (t - tb) * KP, KP)
        # Tiles in M-fastest order: the M tiles of one N tile are neighbours
        # and share the weight columns in L2.
        pid_m = t % num_pid_m
        pid_n = t // num_pid_m
        m0 = pid_m * BM
        n0 = pid_n * BN
        acc_tl, acc_bl, acc_tr, acc_br = _sk8w_any_tile(
            smemA_top, smemA_bot, smemB_left, smemB_right, a_ptr, b_ptr,
            m0, n0, ks * (CH * BK), M - 1 - m0, N - 1 - n0, stride_am, stride_bn,
            (ke - ks) * CH, BMH, BNH, BK, WARPS_M, WARPS_N, NS,
        )  # fmt: skip
        if (ks == 0) & (ke == KP):
            _sk8w_store_quad(acc_tl, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 0, 0, BMH, BNH, False)  # fmt: skip
            _sk8w_store_quad(acc_bl, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 1, 0, BMH, BNH, False)  # fmt: skip
            _sk8w_store_quad(acc_tr, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 0, 1, BMH, BNH, False)  # fmt: skip
            _sk8w_store_quad(acc_br, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 1, 1, BMH, BNH, False)  # fmt: skip
        else:
            slot = 2 * vid + (x != start).to(gl.int32)
            p = part_ptr + slot.to(gl.int64) * (BM * BN)
            _sk8w_store_quad(acc_tl, c_ptr, p, m0, n0, M, N, stride_cm, 0, 0, BMH, BNH, True)  # fmt: skip
            _sk8w_store_quad(acc_bl, c_ptr, p, m0, n0, M, N, stride_cm, 1, 0, BMH, BNH, True)  # fmt: skip
            _sk8w_store_quad(acc_tr, c_ptr, p, m0, n0, M, N, stride_cm, 0, 1, BMH, BNH, True)  # fmt: skip
            _sk8w_store_quad(acc_br, c_ptr, p, m0, n0, M, N, stride_cm, 1, 1, BMH, BNH, True)  # fmt: skip
            # Publish: this wave's stores done, then every wave's, then arrive.
            gl.inline_asm_elementwise(
                asm="s_waitcnt vmcnt(0)",
                constraints="=v,0",
                args=[gl.arange(0, 64, gl.BlockedLayout([1], [64], [NW], [0]))],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.barrier()
            gl.atomic_add(cnt_ptr + t, 1, sem="relaxed", scope="gpu")
        x = (t - tb) * KP + ke

    # Data-parallel part: whole tiles [SK_TILES, T), one per program per round,
    # while the other pieces of this program's split tiles finish.
    for t in range(SK_TILES + vid, T, GRID):
        pid_m = t % num_pid_m
        pid_n = t // num_pid_m
        m0 = pid_m * BM
        n0 = pid_n * BN
        acc_tl, acc_bl, acc_tr, acc_br = _sk8w_any_tile(
            smemA_top, smemA_bot, smemB_left, smemB_right, a_ptr, b_ptr,
            m0, n0, 0, M - 1 - m0, N - 1 - n0, stride_am, stride_bn,
            CH * KP, BMH, BNH, BK, WARPS_M, WARPS_N, NS,
        )  # fmt: skip
        _sk8w_store_quad(acc_tl, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 0, 0, BMH, BNH, False)  # fmt: skip
        _sk8w_store_quad(acc_bl, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 1, 0, BMH, BNH, False)  # fmt: skip
        _sk8w_store_quad(acc_tr, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 0, 1, BMH, BNH, False)  # fmt: skip
        _sk8w_store_quad(acc_br, c_ptr, part_ptr, m0, n0, M, N, stride_cm, 1, 1, BMH, BNH, False)  # fmt: skip

    # Reduce the split tiles this program took part in: its first and last.
    if start < end:
        t_first = start // KP
        t_last = (end - 1) // KP
        first_split = _sk8w_owner(t_first * KP, G, PX) != _sk8w_owner(
            t_first * KP + KP - 1, G, PX
        )
        if first_split:
            _sk8w_reduce(
                c_ptr, part_ptr, cnt_ptr, tb + t_first, t_first, lid, xbase,
                num_pid_m, M, N, G, stride_cm, KP, PX, BMH, BNH,
            )  # fmt: skip
        if t_last != t_first:
            last_split = _sk8w_owner(t_last * KP, G, PX) != _sk8w_owner(
                t_last * KP + KP - 1, G, PX
            )
            if last_split:
                _sk8w_reduce(
                    c_ptr, part_ptr, cnt_ptr, tb + t_last, t_last, lid, xbase,
                    num_pid_m, M, N, G, stride_cm, KP, PX, BMH, BNH,
                )  # fmt: skip


# Opt-in: route the measured winning shapes through this kernel.
STREAMK8W_ENV = "TOKENSPEED_GLUON_STREAMK8W"
_ENABLED = os.environ.get(STREAMK8W_ENV, "0") == "1"
_GRID = 256
# (N, K) -> (min M, max M, config): the M ranges where the kernel beat tuned
# hipBLASLt on MI355X (cold weights, HIP graph; research/gluon_decode_gemm).
_CONFIGS = {
    # Qwen3.8-27B gate_up: 256x256 tiles, 256 data-parallel + 16 Stream-K tiles.
    (34816, 5120): ((481, 512, (128, 128, 4, 2, _GRID, 8, 1, 2)),),
}

_scratch: dict = {}
_cus: dict = {}


def _num_cus(device: torch.device) -> int:
    if device not in _cus:
        _cus[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _cus[device]


def select_gluon_mm_a16w16_streamk8w_gfx950(M: int, N: int, K: int) -> tuple | None:
    """The measured config for ``(M, N, K)``, or None to leave it to hipBLASLt."""
    for lo, hi, cfg in _CONFIGS.get((N, K), ()):
        if lo <= M <= hi:
            return cfg
    return None


def supports_gluon_mm_a16w16_streamk8w_gfx950(M: int, N: int, K: int) -> bool:
    """Whether the opt-in (``TOKENSPEED_GLUON_STREAMK8W=1``) kernel takes (M, N, K)."""
    return _ENABLED and select_gluon_mm_a16w16_streamk8w_gfx950(M, N, K) is not None


_MAX_COUNTERS = 4096


def gluon_mm_a16w16_streamk8w_num_counters_gfx950(M: int, N: int, K: int) -> int:
    """Counter elements a launch for (M, N, K) needs (one per Stream-K tile)."""
    cfg = select_gluon_mm_a16w16_streamk8w_gfx950(M, N, K)
    return max(sk_tiles(cfg, M, N, K), 1)


def sk_tiles(cfg: tuple, M: int, N: int, K: int) -> int:
    """Tiles handled by Stream-K for config ``(BMH, BNH, WM, WN, GRID, XCD, SK, NS)``.

    SK 0: none (whole tiles in rounds of GRID); 1: the last partial round
    (``T % GRID`` tiles, all T when T < GRID); 2: that plus one full round.
    """
    bmh, bnh, _, _, grid, _, sk = cfg[:7]
    tiles = triton.cdiv(M, 2 * bmh) * triton.cdiv(N, 2 * bnh)
    if sk == 0:
        return 0
    rem = tiles % grid if tiles >= grid else tiles
    if sk == 2 and rem and tiles >= rem + grid:
        rem += grid
    return rem


def config_fits(cfg: tuple, M: int, N: int, K: int) -> bool:
    """Whether a (BMH, BNH, WM, WN, GRID, XCD, SK, NS) config is legal for (M, N, K)."""
    bmh, bnh, wm, wn, grid, xcd, sk, ns = cfg[:8]
    nbuf = 2 if ns == 0 else ns
    if ns not in (0, 2, 3, 4) or ns > CHUNK:
        return False
    # LDS: NBUF x (A top + bottom + B left + right) x BLOCK_K bf16, padded 16/512.
    if nbuf * (2 * bmh + 2 * bnh) * BLOCK_K * 2 * 33 // 32 > LDS_BYTES:
        return False
    if wm * wn != NUM_WARPS or bmh not in (64, 128) or bnh not in (64, 128):
        return False
    if bmh % (16 * wm) or bnh % (16 * wn):
        return False
    if K % (CHUNK * BLOCK_K) or grid % xcd or sk not in (0, 1, 2):
        return False
    # Every program of an XCD with Stream-K tiles needs a non-empty range.
    nsk = sk_tiles(cfg, M, N, K)
    kp = K // (CHUNK * BLOCK_K)
    for x in range(xcd):
        nt = (x + 1) * nsk // xcd - x * nsk // xcd
        if nt and nt * kp < grid // xcd:
            return False
    return True


def gluon_mm_a16w16_streamk8w_gfx950_mm(
    A: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor | None = None,
    counters: torch.Tensor | None = None,
) -> torch.Tensor:
    """``A @ W.T`` in bf16 with the measured config for this shape."""
    M, K = A.shape
    N = W.shape[0]
    cfg = select_gluon_mm_a16w16_streamk8w_gfx950(M, N, K)
    if cfg is None:
        raise ValueError(f"no streamk8w config for M={M} N={N} K={K}")
    if out is None:
        out = torch.empty(M, N, device=A.device, dtype=A.dtype)
    return launch_gluon_mm_a16w16_streamk8w_gfx950(A, W, out, cfg, counters)


def launch_gluon_mm_a16w16_streamk8w_gfx950(
    A: torch.Tensor,
    W: torch.Tensor,
    C: torch.Tensor,
    cfg: tuple,
    counters: torch.Tensor | None = None,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ W[N, K].T with config (BMH, BNH, WM, WN, GRID, XCD, SK, NS).

    ``counters``: a zeroed int32 buffer of at least ``num_counters`` elements;
    the launch leaves it zeroed. Without it a cached per-device one is used.
    """
    bmh, bnh, wm, wn, grid, xcd, _, ns = cfg[:8]
    M, K = A.shape
    N = W.shape[0]
    assert A.stride(1) == 1 and W.stride(1) == 1 and C.stride(1) == 1
    assert grid <= _num_cus(A.device), f"GRID {grid} exceeds the CU count"
    assert config_fits(cfg, M, N, K), cfg
    nsk = sk_tiles(cfg, M, N, K)
    key = (A.device, grid, bmh * bnh)
    ent = _scratch.get(key)
    if ent is None:
        ent = (
            torch.empty(2 * grid * 4 * bmh * bnh, device=A.device, dtype=torch.float32),
            torch.zeros(_MAX_COUNTERS, device=A.device, dtype=torch.int32),
        )
        _scratch[key] = ent
    part, cnt = ent
    if counters is not None:
        cnt = counters
    assert cnt.numel() >= max(nsk, 1)
    gluon_mm_a16w16_streamk8w_gfx950[(grid,)](
        A,
        W,
        C,
        part,
        cnt,
        M,
        N,
        K,
        A.stride(0),
        W.stride(0),
        C.stride(0),
        nsk,
        BMH=bmh,
        BNH=bnh,
        WARPS_M=wm,
        WARPS_N=wn,
        GRID=grid,
        NUM_XCDS=xcd,
        NS=ns,
        num_warps=NUM_WARPS,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    return C
