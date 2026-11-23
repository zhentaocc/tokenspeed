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

"""FlyDSL a16w16 (gfx950) config sweep for the Qwen3.8-27B decode projections.

C[M,N] = x[M,K] @ W[N,K].T with cold rotating weights under HIP-graph replay
(median us per call), next to default hipBLASLt (torch.mm) in the same process.
FlyDSL ships its GEMM kernels only in its source tree, so pass a checkout:

    pip install flydsl==0.5.3
    git clone -b v0.5.3 https://github.com/ROCm/FlyDSL
    python bench_flydsl.py --flydsl-src FlyDSL --shapes gate_up --ms 256 --out fly.jsonl
"""

import argparse
import itertools
import json
import math
import statistics
import sys
import time

import torch

SHAPES = {
    "qkv": (14336, 5120),
    "o": (5120, 6144),
    "gdn_in": (16480, 5120),
    "gate_up": (34816, 5120),
    "down": (5120, 17408),
}

p = argparse.ArgumentParser()
p.add_argument("--flydsl-src", required=True, help="FlyDSL source checkout")
p.add_argument("--shapes", default="gate_up")
p.add_argument("--ms", default="256")
p.add_argument("--cfgs", default=None, help="JSON list of user_kwargs dicts")
p.add_argument("--max", type=int, default=400)
p.add_argument("--reps", type=int, default=20)
p.add_argument("--cold-bytes", type=int, default=2 * 2**30)
p.add_argument("--out", default=None)
a = p.parse_args()

sys.path.insert(0, a.flydsl_src)
from kernels.gemm.gemm_a16w16_gfx950 import (  # noqa: E402
    GEMM_A16W16_DTYPE_BF16,
    gemm_a16w16,
    make_gemm_a16w16_param_and_validate,
)


def candidates(m, n, k):
    out = []
    for bm, bn, bk, st, (mw, nw), sk, hti, gm in itertools.product(
        (64, 128, 256),
        (64, 80, 96, 128, 144, 160, 192, 256),
        (64, 128),
        (2, 3, 4),
        ((1, 4), (2, 2), (4, 1), (2, 4), (4, 2)),
        (1, 2, 3, 4),
        (False, True),
        (0, 4),
    ):
        if bm > 2 * m and bm != 64:
            continue
        tiles = math.ceil(m / bm) * math.ceil(n / bn)
        progs = tiles * sk
        if sk > 1 and tiles >= 200:
            continue
        rounds = math.ceil(progs / 256)
        fill = progs / (rounds * 256)
        if fill < 0.7 or progs > 768:
            continue
        if hti and st != 2:
            continue
        if gm and math.ceil(m / bm) == 1:
            continue
        kw = dict(
            block_m=bm,
            block_n=bn,
            block_k=bk,
            stages=st,
            split_k=sk,
            m_waves=mw,
            n_waves=nw,
            k_waves=1,
            group_m=gm,
            use_half_tile_interleaved=hti,
        )
        full = dict(
            kw,
            in_dtype_id=GEMM_A16W16_DTYPE_BF16,
            out_dtype_id=GEMM_A16W16_DTYPE_BF16,
            a_is_transposed=False,
            b_is_transposed=True,
            has_bias=False,
        )
        if make_gemm_a16w16_param_and_validate(m, n, k, full) is None:
            continue
        out.append(kw)
    return out


def time_graph(fn, x, ws, out):
    for w in ws[:2]:
        fn(x, w, out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for w in ws:
                fn(x, w, out)
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(a.reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / len(ws))
    del g
    return statistics.median(ts)


torch.manual_seed(0)
fo = open(a.out, "a") if a.out else None
for name in a.shapes.split(","):
    n, k = SHAPES[name]
    nw = max(4, -(-a.cold_bytes // (n * k * 2)))
    ws = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(nw)
    ]
    for m in map(int, a.ms.split(",")):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ ws[0].float().t()
        t_torch = time_graph(lambda x, w, o: torch.mm(x, w.t(), out=o), x, ws, out)
        print(f"{name} M={m} torch {t_torch:.2f}", flush=True)
        cfgs = json.loads(a.cfgs) if a.cfgs else candidates(m, n, k)[: a.max]
        print(f"  {len(cfgs)} configs", flush=True)
        best = []
        for kw in cfgs:
            t0 = time.time()
            try:
                fn = lambda x, w, o, kw=kw: gemm_a16w16(
                    x, w.t(), out=o, user_kwargs=kw, layout="nt"
                )
                fn(x, ws[0], out)
                torch.cuda.synchronize()
                err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                # FlyDSL 0.5.3 accepts some half-tile-interleaved configs that
                # return wrong results (e.g. block 64x256, 2x2 waves), so check each.
                if not err < 2e-2:
                    print(f"  BAD err {err:.2e} {kw}", flush=True)
                    continue
                us = time_graph(fn, x, ws, out)
            # A sweep records configs that FlyDSL rejects at compile time and moves on.
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {kw}: {str(e)[:120]}", flush=True)
                continue
            rec = dict(
                shape=name,
                m=m,
                method="flydsl",
                us=us,
                relerr=err,
                cfg=kw,
                torch_us=t_torch,
            )
            best.append(rec)
            if fo:
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
            print(
                f"  {us:8.2f} us err {err:.1e} ({time.time() - t0:.1f}s) {kw}",
                flush=True,
            )
        best.sort(key=lambda r: r["us"])
        for r in best[:5]:
            print(
                f"BEST {name} M={m} {r['us']:.2f} (torch {t_torch:.2f}) {r['cfg']}",
                flush=True,
            )
