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

"""Research FlyDSL BF16 GEMM for gfx950, C[M, N] = A[M, K] @ W[N, K].T.

Derived from FlyDSL 0.5.3 ``kernels/gemm/gemm_a16w16_gfx950.py`` (Apache-2.0,
the non-HTI kernel, "nt" layout only). Changes:

- Partial last direct-to-LDS load round: a tile whose ``rows * block_k / 8``
  16-byte vectors is a multiple of 64 but not of the thread count loads the
  remainder with the first waves only. This lets 144-, 160- and 224-wide N
  tiles build with 4 or 8 waves (stock FlyDSL needs an exact multiple).
- Persistent grid (``grid > 0``): each workgroup loops over output tiles
  ``pid, pid + grid, ...``, so two or more tiles run back to back per CU.
- XCD remap (``xcd_remap``): workgroup pid runs on XCD pid % 8, so each XCD
  gets a contiguous run of tiles in M-fastest order and the M tiles that share
  a weight tile share that XCD's L2.
- In-kernel split-K reuses FlyDSL's protocol (k-slice 0 zeroes C, all slices
  atomically add BF16 pairs), with the semaphore indexed by tile.

Needs ``flydsl`` (pip) and a FlyDSL 0.5.3 source checkout on ``sys.path`` for
``kernels.gemm.gemm_a16w16_gfx950_utils``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from kernels.gemm.gemm_a16w16_gfx950_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    SPLIT_K_SEMAPHORE_MAX_LEN,
    BlockSwizzle,
    SplitKProtocol,
    get_wave_lds_offset,
    make_lds_layout,
    run_cached,
    swizzled_col_idx,
)

LDS_BYTES = 160 * 1024
VEC = GFX950_DMA_BYTES // 2  # bf16 elements per 16-byte load


@fx.struct
class FlyGemmParam:
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    is_split_k: fx.Constexpr[bool]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    grid: fx.Constexpr[int]
    xcd_remap: fx.Constexpr[bool]
    block_threads: fx.Constexpr[int]
    ldg_a_full: fx.Constexpr[int]
    ldg_a_rem: fx.Constexpr[int]
    ldg_b_full: fx.Constexpr[int]
    ldg_b_rem: fx.Constexpr[int]
    vec_out: fx.Constexpr[int]


def make_param(
    block_m,
    block_n,
    block_k=64,
    stages=2,
    split_k=1,
    m_waves=2,
    n_waves=2,
    group_m=0,
    grid=0,
    xcd_remap=False,
):
    """Validate a config; raise ValueError if it cannot build."""
    threads = m_waves * n_waves * GFX950_WAVE_SIZE
    if m_waves * n_waves > 16 or stages < 2:
        raise ValueError("waves <= 16 and stages >= 2")
    if block_m % (16 * m_waves) or block_n % (16 * n_waves):
        raise ValueError("block_m/block_n must split into 16-wide MFMA tiles per wave")
    if block_k % 64:
        # The (3, 3, 3) LDS swizzle spans 64-element rows; stock FlyDSL 0.5.3
        # also returns wrong results at block_k=32 (relerr 1.5).
        raise ValueError("block_k must be a multiple of 64")
    if (
        stages * (block_m + block_n) * block_k * 2 > LDS_BYTES
        or block_m * block_n * 2 > LDS_BYTES
    ):
        raise ValueError("LDS")
    out = []
    for rows in (block_m, block_n):
        vecs = rows * block_k // VEC
        if vecs % GFX950_WAVE_SIZE:
            raise ValueError(f"{rows}x{block_k} is not a whole number of wave loads")
        out += [vecs // threads, vecs % threads]
    if (stages - 2) * (out[0] + out[2] + 2) >= 63:
        raise ValueError("vmcnt")
    vec_out = 4 if split_k > 1 else 8
    if block_n % vec_out:
        raise ValueError("vec_out")
    if split_k > 1 and block_m * block_n % (threads * vec_out):
        raise ValueError("split-K zero-fill")
    return FlyGemmParam(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        is_split_k=split_k > 1,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        grid=grid,
        xcd_remap=xcd_remap,
        block_threads=threads,
        ldg_a_full=out[0],
        ldg_a_rem=out[1],
        ldg_b_full=out[2],
        ldg_b_rem=out[3],
        vec_out=vec_out,
    )


def check_shape(m, n, k, cfg):
    """Return the param for (m, n, k, cfg dict) or raise ValueError."""
    p = make_param(**cfg)
    split_k = cfg.get("split_k", 1)
    working_k = -(-k // split_k)
    working_k = -(-working_k // VEC) * VEC
    last = k - (split_k - 1) * working_k
    if working_k % p.block_k or last <= 0 or last % p.block_k:
        raise ValueError("K tail")
    if min(working_k, last) // p.block_k < p.stages - 1:
        raise ValueError("too few K tiles")
    if n % p.vec_out:
        raise ValueError("N % vec")
    tiles = -(-m // p.block_m) * -(-n // p.block_n)
    if split_k > 1 and tiles > SPLIT_K_SEMAPHORE_MAX_LEN:
        raise ValueError("split-K semaphore")
    return p


@flyc.kernel
def fly_gemm_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    split_k: fx.Int32,
    working_k: fx.Int32,
    a_stride: fx.Int32,
    b_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    p: FlyGemmParam,
):
    block_m = p.block_m
    block_n = p.block_n
    block_k = p.block_k
    stages = p.stages
    threads = p.block_threads
    vec_out = p.vec_out
    is_split_k = p.is_split_k
    elem = fx.BFloat16

    tid = fx.thread_idx.x
    num_pid_m = (m + block_m - 1) // block_m
    num_pid_n = (n + block_n - 1) // block_n
    num_tiles = num_pid_m * num_pid_n
    swz = BlockSwizzle(NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=p.group_m)
    ks_idx = fx.block_idx.y
    ks_begin = ks_idx * working_k
    ks_end = ks_begin + working_k
    ks_end = (ks_end < k).select(ks_end, k)
    k_tiles = (ks_end - ks_begin) // block_k

    @fx.struct
    class ABStorage:
        a: fx.Array[elem, stages * block_m * block_k, 16]
        b: fx.Array[elem, stages * block_n * block_k, 16]

    @fx.union
    class Storage:
        ab: ABStorage
        c: fx.Array[elem, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(Storage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    wave_off = get_wave_lds_offset(tid, GFX950_DMA_BYTES)

    uni_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem)
    buf_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(buf_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(buf_atom, tiled_mma).get_slice(tid)
    a_lay = make_lds_layout(block_m, block_k)
    b_lay = make_lds_layout(block_n, block_k)
    c_lay = fx.make_layout((block_m, block_n), (block_n, 1))
    sA = fx.make_view(smem_a, a_lay)
    sB = fx.make_view(smem_b, b_lay)
    sC = fx.make_view(smem_c, c_lay)
    gC0 = fx.flat_divide(out_buf, (block_m, block_n))[None, None, 0, 0]
    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC0)
    frag_A_rt = thr_copy_A.retile(frag_A)
    frag_B_rt = thr_copy_B.retile(frag_B)
    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
    cRow = thr_mma.partition_C(row_coords)
    cCol = thr_mma.partition_C(col_coords)

    def _load_to_lds(
        lds_base,
        src_base,
        lds_layout,
        outer_bound,
        outer_offset,
        stride,
        k_off,
        tid,
        wave_off,
        full,
        rem,
        p,
    ):
        """Direct-to-LDS async copy of a (rows, block_k) tile; K-contiguous source."""
        threads = p.block_threads
        x_threads = p.block_k // VEC
        lds_ptr = lds_base + fx.Int32(wave_off) // 2
        lay = fx.make_layout(VEC, 1)
        atom = fx.make_copy_atom(fx.rocdl.cdna4.BufferLoadAsyncLDS128b(), 128)
        iters = full + (1 if rem else 0)
        for i in range_constexpr(iters):
            global_tid = threads * i + tid
            outer = global_tid // x_threads
            k_local = global_tid % x_threads * VEC
            gk = k_off + swizzled_col_idx(outer, k_local, lds_layout, p.block_k)
            go = outer_offset + outer
            go = (go < outer_bound).select(go, 0)
            src = fx.make_view(src_base + go * stride + gk, lay)
            dst = fx.make_view(lds_ptr, lay)
            if const_expr(i < full):
                rocdl.sched_barrier(0)
                fx.copy_atom_call(atom, src, dst)
                rocdl.sched_barrier(0)
            else:
                # Wave-uniform: rem is a multiple of the wave size.
                if tid < rem:
                    fx.copy_atom_call(atom, src, dst)
                rocdl.sched_barrier(0)
            if const_expr(i < iters - 1):
                lds_ptr = lds_ptr + threads * VEC

    def load_a(k_tile, stage, m_off):
        _load_to_lds(
            smem_a + stage * block_m * block_k,
            fx.get_iter(a_buf),
            a_lay,
            m,
            m_off,
            a_stride,
            ks_begin + k_tile * block_k,
            tid,
            wave_off,
            p.ldg_a_full,
            p.ldg_a_rem,
            p,
        )

    def load_b(k_tile, stage, n_off):
        _load_to_lds(
            smem_b + stage * block_n * block_k,
            fx.get_iter(b_buf),
            b_lay,
            n,
            n_off,
            b_stride,
            ks_begin + k_tile * block_k,
            tid,
            wave_off,
            p.ldg_b_full,
            p.ldg_b_rem,
            p,
        )

    def compute_stage(stage):
        tA = thr_copy_A.partition_S(
            fx.make_view(smem_a + stage * block_m * block_k, a_lay)
        )
        tB = thr_copy_B.partition_S(
            fx.make_view(smem_b + stage * block_n * block_k, b_lay)
        )
        for kk in range_constexpr(block_k // 32):
            fx.copy(uni_atom, tB[None, None, kk], frag_B_rt[None, None, kk])
            fx.copy(uni_atom, tA[None, None, kk], frag_A_rt[None, None, kk])
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, kk],
                frag_B[None, None, kk],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

    def run_tile(tile, bid_m, bid_n):
        m_off = bid_m * block_m
        n_off = bid_n * block_n
        if const_expr(is_split_k):
            sk = SplitKProtocol(block_m, block_n, vec_out, 2, threads, False)
            sk.init(
                semaphore,
                signal,
                out,
                None,
                tid,
                ks_idx,
                m,
                n,
                m_off,
                n_off,
                elem,
                tile,
                n,
            )
            sk.zero_c()
        frag_C.fill(0.0)
        for s in range_constexpr(stages - 1):
            load_b(s, s, n_off)
            load_a(s, s, m_off)
            rocdl.asyncmark()
        rocdl.sched_barrier(0)
        main_end = k_tiles - (stages - 1)
        for kt in range(0, main_end, 1):
            cur = kt % stages
            wr = (cur + stages - 1) % stages
            rocdl.wait_asyncmark(stages - 2)
            rocdl.s_barrier()
            load_b(kt + (stages - 1), wr, n_off)
            load_a(kt + (stages - 1), wr, m_off)
            rocdl.asyncmark()
            compute_stage(cur)
            rocdl.sched_barrier(0)
        cur = main_end % stages
        for s in range_constexpr(0, stages - 1):
            rocdl.wait_asyncmark(stages - 2 - s)
            rocdl.s_barrier()
            compute_stage(cur)
            cur = (cur + 1) % stages

        frag_out = fx.make_fragment_like(frag_C, elem)
        frag_out.store(frag_C.load().to(elem))
        gpu.barrier()
        for i in range_constexpr(fx.size(frag_out.shape).unpack()):
            sC[fx.get_scalar(cRow[i]), fx.get_scalar(cCol[i])] = frag_out[i]
        if const_expr(is_split_k):
            sk.wait_until_initialized()
        else:
            gpu.barrier()
        x_thr = block_n // vec_out
        nvec = block_m * block_n // vec_out
        for i in range_constexpr((nvec + threads - 1) // threads):
            vi = threads * i + tid
            if vi < nvec:
                lr = vi // x_thr
                lc = vi % x_thr * vec_out
                gr = m_off + lr
                gc = n_off + lc
                if (gr < m) and (gc < n):
                    c_vec = fx.ptr_load(
                        smem_c + lr * block_n + lc,
                        result_type=fx.Vector.make_type(vec_out, elem),
                    )
                    off = gr * n + gc
                    if const_expr(is_split_k):
                        pk_atom = fx.make_copy_atom(
                            fx.rocdl.BufferAtomicPkAdd(elem), elem
                        )
                        pair_lay = fx.make_layout(2, 1)
                        pair_frag = fx.make_rmem_tensor(pair_lay, elem)
                        for j in range_constexpr(vec_out // 2):
                            pair_frag.store(
                                fx.Vector.from_elements(
                                    [c_vec[2 * j], c_vec[2 * j + 1]], elem
                                )
                            )
                            dst = fx.make_view(
                                fx.get_iter(out_buf) + off + 2 * j, pair_lay
                            )
                            fx.copy_atom_call(pk_atom, pair_frag, dst)
                    else:
                        fx.ptr_store(c_vec, fx.get_iter(out) + off)
        if const_expr(is_split_k):
            sk.finish_split(split_k)

    if const_expr(p.grid > 0):
        # M-fastest order: the workgroups of one round share weight tiles.
        for tile in range(fx.block_idx.x, num_tiles, p.grid):
            run_tile(tile, tile % num_pid_m, tile // num_pid_m)
            # sC aliases the A/B stages: drain the epilogue before the next prologue.
            gpu.barrier()
    elif const_expr(p.xcd_remap):
        # Workgroup pid runs on XCD pid % 8. Give each XCD a contiguous run of
        # tiles in M-fastest order, so the M tiles sharing a weight tile share
        # that XCD's L2.
        pid = fx.block_idx.x
        per_xcd = ((num_tiles + 7) // 8 + num_pid_m - 1) // num_pid_m * num_pid_m
        tile = pid % 8 * per_xcd + pid // 8
        if tile < num_tiles:
            run_tile(tile, tile % num_pid_m, tile // num_pid_m)
    else:
        bid_m, bid_n = swz.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
        run_tile(fx.block_idx.x, bid_m, bid_n)


@flyc.jit
def fly_gemm_launch(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    semaphore: fx.Tensor,
    signal: fx.Tensor,
    split_k: fx.Int32,
    p: FlyGemmParam,
    stream: fx.Stream = fx.Stream(None),
):
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_stride = fx.Int32(fx.get_scalar(a.stride[0]))
    b_stride = fx.Int32(fx.get_scalar(b.stride[1]))
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    tiled_mma = fx.make_tiled_mma(
        atom,
        fx.make_layout((p.m_waves, p.n_waves, 1), (p.n_waves, 1, 0)),
        fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
    )
    working_k = (k + split_k - 1) // split_k
    working_k = (working_k + VEC - 1) // VEC * VEC
    num_tiles = ((m + p.block_m - 1) // p.block_m) * ((n + p.block_n - 1) // p.block_n)
    grid_x = num_tiles if const_expr(p.grid == 0) else p.grid
    if const_expr(p.xcd_remap):
        num_pid_m = (m + p.block_m - 1) // p.block_m
        grid_x = ((num_tiles + 7) // 8 + num_pid_m - 1) // num_pid_m * num_pid_m * 8
    fly_gemm_kernel._known_block_size = [p.block_threads, 1, 1]
    fly_gemm_kernel._func.__name__ = (
        f"fly_gemm_t{p.block_m}x{p.block_n}x{p.block_k}x{p.stages}_w{p.m_waves}x{p.n_waves}"
        f"_sk{int(p.is_split_k)}_g{p.grid}_gm{p.group_m}_x{int(p.xcd_remap)}"
    )
    fly_gemm_kernel(
        out,
        a,
        b,
        semaphore,
        signal,
        m,
        n,
        k,
        split_k,
        working_k,
        a_stride,
        b_stride,
        tiled_mma,
        p,
    ).launch(grid=(grid_x, split_k, 1), block=(p.block_threads, 1, 1), stream=stream)


@functools.lru_cache(maxsize=16)
def _split_k_buffers(device):
    z = torch.zeros((SPLIT_K_SEMAPHORE_MAX_LEN,), dtype=torch.int32, device=device)
    return z, torch.zeros_like(z)


def fly_gemm(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, cfg: dict
) -> torch.Tensor:
    """out[M, N] = x[M, K] @ w[N, K].T in BF16 with config ``cfg`` (see make_param)."""
    m, k = x.shape
    n = w.shape[0]
    assert x.dtype == w.dtype == out.dtype == torch.bfloat16
    assert x.stride(1) == 1 and w.stride(1) == 1 and out.is_contiguous()
    p = check_shape(m, n, k, cfg)
    split_k = cfg.get("split_k", 1)
    sem, sig = _split_k_buffers(x.device)
    bt = w.t()  # [K, N], column-major
    args = (
        flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(x).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(bt).mark_layout_dynamic(leading_dim=0),
        sem,
        sig,
        split_k,
        p,
        torch.cuda.current_stream(),
    )
    run_cached(
        fly_gemm_launch,
        *args,
        constexpr_param=p,
        compiler=flyc.compile,
        dispatch_args=args,
    )
    return out
