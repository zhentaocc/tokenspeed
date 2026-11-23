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

"""Time fly_gemm configs against hipBLASLt (default or a TunableOp file).

Cold rotating weights (> LLC) under HIP-graph replay, median us per call. Every
config is checked against an FP32 torch reference before timing, and again on
the last weight after the graph replays.

    python bench.py --flydsl-src FlyDSL --shapes gate_up --ms 256 \
        --cfgs '[{"block_m":256,"block_n":144,"m_waves":4,"n_waves":1}]' \
        [--tuned-file CSV] --out res.jsonl
"""

import argparse
import json
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
p.add_argument("--flydsl-src", required=True)
p.add_argument("--shapes", default="gate_up")
p.add_argument("--ms", default="256")
p.add_argument(
    "--cfgs",
    default="[]",
    help="JSON (or @file) list of cfg dicts, or {shape: {m: [cfg...]}}",
)
p.add_argument("--tuned-file", default=None)
p.add_argument("--no-torch", action="store_true")
p.add_argument("--reps", type=int, default=20)
p.add_argument("--cold-bytes", type=float, default=2.2e9)
p.add_argument("--tol", type=float, default=2e-2)
p.add_argument("--out", default=None)
a = p.parse_args()

sys.path.insert(0, a.flydsl_src)
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fly_gemm import check_shape, fly_gemm  # noqa: E402

if a.tuned_file:
    # Same replay-only setup as tokenspeed_kernel's load_hipblaslt_tuned_solutions.
    tunable = torch.cuda.tunable
    tunable.tuning_enable(False)
    tunable.record_untuned_enable(False)
    tunable.set_filename("/dev/null", insert_device_ordinal=False)
    tunable.enable(True)
    assert tunable.read_file(a.tuned_file), "tuned file not loaded"
    print("loaded", len(tunable.get_results()), "tuned rows", flush=True)


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


def relerr(out, ref):
    return ((out.float() - ref).abs().max() / ref.abs().max()).item()


torch.manual_seed(0)
fo = open(a.out, "a") if a.out else None
cfg_arg = json.loads(open(a.cfgs[1:]).read() if a.cfgs.startswith("@") else a.cfgs)
tag = "tuned" if a.tuned_file else "torch"
for name in a.shapes.split(","):
    n, k = SHAPES[name]
    nw = max(4, int(-(-a.cold_bytes // (n * k * 2))))
    ws = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(nw)
    ]
    for m in map(int, a.ms.split(",")):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ref0 = x.float() @ ws[0].float().t()
        ref_last = x.float() @ ws[-1].float().t()
        rec = dict(shape=name, m=m)
        if not a.no_torch:
            us = time_graph(lambda x, w, o: torch.mm(x, w.t(), out=o), x, ws, out)
            print(f"{name} M={m} {tag} {us:.2f}", flush=True)
            if fo:
                fo.write(json.dumps(dict(rec, method=tag, us=us)) + "\n")
        cfgs = (
            cfg_arg
            if isinstance(cfg_arg, list)
            else cfg_arg.get(name, {}).get(str(m), [])
        )
        for cfg in cfgs:
            t0 = time.time()
            try:
                check_shape(m, n, k, cfg)
            except ValueError as e:
                print(f"  SKIP {cfg}: {e}", flush=True)
                continue
            try:
                fn = lambda x, w, o, cfg=cfg: fly_gemm(x, w, o, cfg)  # noqa: E731
                out.fill_(7.0)
                fn(x, ws[0], out)
                torch.cuda.synchronize()
                err = relerr(out, ref0)
                if not err < a.tol:
                    print(f"  BAD err {err:.2e} {cfg}", flush=True)
                    continue
                us = time_graph(fn, x, ws, out)
                err2 = relerr(out, ref_last)
                if not err2 < a.tol:
                    print(f"  BAD-after-replay err {err2:.2e} {cfg}", flush=True)
                    continue
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {cfg}: {type(e).__name__} {str(e)[:300]}", flush=True)
                continue
            r = dict(rec, method="fly", us=us, relerr=max(err, err2), cfg=cfg)
            if fo:
                fo.write(json.dumps(r) + "\n")
                fo.flush()
            print(
                f"  {us:8.2f} us err {max(err, err2):.1e} ({time.time() - t0:.1f}s) {cfg}",
                flush=True,
            )
        del x, out
    del ws
    torch.cuda.empty_cache()
