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

"""8-wave Stream-K Gluon GEMM (mm_streamk8w.py) vs tuned hipBLASLt, same session.

C = A @ W.T, cold rotating weights, HIP-graph replay, median us/call. Every
config is checked against an FP32 torch reference first.

usage: bench_sk8w.py --shapes gate_up --rows 256,512 [--cfgs JSON] [--tuned-file CSV]
       [--rounds 3] [--out res.jsonl]
A config is (BMH, BNH, WM, WN, GRID, XCD, SK, NS); without --cfgs every legal one is swept.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import torch
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16 import mm_streamk8w as SK8

SHAPES = {
    "qkv": (14336, 5120),
    "o": (5120, 6144),
    "gdn_in": (16480, 5120),
    "gate_up": (34816, 5120),
    "down": (5120, 17408),
}


def candidates(M, N, K, grid):
    """Every legal tile/wave/Stream-K config with the deepest NS that fits LDS (and 0)."""
    out = []
    for bmh, bnh, (wm, wn), sk in itertools.product(
        (64, 128), (64, 128), ((2, 4), (4, 2)), (0, 1, 2)
    ):
        if bmh > 64 and 2 * bmh > M:
            continue
        deep = [
            ns
            for ns in (4, 3, 2)
            if SK8.config_fits((bmh, bnh, wm, wn, grid, 8, 0, ns), M, N, K)
        ]
        for ns in [0] + deep[:1]:
            c = (bmh, bnh, wm, wn, grid, 8, sk, ns)
            if not SK8.config_fits(c, M, N, K):
                continue
            if sk and SK8.sk_tiles(c, M, N, K) == SK8.sk_tiles(
                c[:6] + (sk - 1, ns), M, N, K
            ):
                continue
            out.append(c)
    return out


def time_graph(fn, ws, reps):
    for w in ws[:2]:
        fn(w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for w in ws:
                fn(w)
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / len(ws))
    del g
    return statistics.median(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="gate_up")
    p.add_argument("--rows", default="256")
    p.add_argument("--cfgs", default=None)
    p.add_argument("--grid", type=int, default=256)
    p.add_argument("--tuned-file", default=None)
    p.add_argument("--no-torch", action="store_true")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--cold-bytes", type=float, default=2.2e9)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.tuned_file:
        from tokenspeed_kernel.ops.gemm._hipblaslt.tuning import (
            load_hipblaslt_tuned_solutions,
        )

        assert load_hipblaslt_tuned_solutions(a.tuned_file), "tuned file not loaded"
    torch.manual_seed(0)
    fo = open(a.out, "a") if a.out else None
    for name in a.shapes.split(","):
        n, k = SHAPES[name] if name in SHAPES else map(int, name.split("x"))
        nw = max(4, -(-int(a.cold_bytes) // (n * k * 2)))
        ws = [
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
            for _ in range(nw)
        ]
        for m in map(int, a.rows.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            cfgs = (
                [tuple(c) for c in json.loads(a.cfgs)]
                if a.cfgs
                else candidates(m, n, k, a.grid)
            )
            ok = []
            for c in cfgs:
                if not SK8.config_fits(c, m, n, k):
                    print(f"  skip {c}", flush=True)
                    continue
                try:
                    out.zero_()
                    SK8.launch_gluon_mm_a16w16_streamk8w_gfx950(x, ws[0], out, c)
                    torch.cuda.synchronize()
                    err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
                    # second launch: counters must have been reset
                    out.zero_()
                    SK8.launch_gluon_mm_a16w16_streamk8w_gfx950(x, ws[1], out, c)
                    torch.cuda.synchronize()
                    ref1 = x.float() @ ws[1].float().t()
                    err = max(
                        err,
                        ((out.float() - ref1).abs().max() / ref1.abs().max()).item(),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  FAIL {c}: {str(e)[:300]}", flush=True)
                    continue
                if not err < 1e-2:
                    print(f"  BAD {c} err {err:.2e}", flush=True)
                    continue
                ok.append((c, err))
            for r in range(a.rounds):
                if not a.no_torch:
                    us = time_graph(lambda w: torch.mm(x, w.t(), out=out), ws, a.reps)
                    meth = "tuned" if a.tuned_file else "torch"
                    rec = dict(shape=name, m=m, method=meth, us=us, round=r)
                    print(f"{name} M={m} {meth} {us:.2f}", flush=True)
                    if fo:
                        fo.write(json.dumps(rec) + "\n")
                for c, err in ok:
                    us = time_graph(
                        lambda w, c=c: SK8.launch_gluon_mm_a16w16_streamk8w_gfx950(
                            x, w, out, c
                        ),
                        ws,
                        a.reps,
                    )
                    rec = dict(
                        shape=name, m=m, method="sk8w", cfg=list(c), us=us,
                        relerr=err, round=r,
                    )  # fmt: skip
                    print(f"{name} M={m} sk8w {us:.2f} err {err:.1e} {c}", flush=True)
                    if fo:
                        fo.write(json.dumps(rec) + "\n")
            if fo:
                fo.flush()
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
