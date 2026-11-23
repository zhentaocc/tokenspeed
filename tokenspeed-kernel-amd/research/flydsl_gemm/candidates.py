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

"""Print a fly_gemm config sweep, {shape: {M: [cfg, ...]}}, pruned by CU fill.

python candidates.py --shapes gate_up,down --ms 64,128,256,512 --max 120 > cfgs.json
"""

import argparse
import itertools
import json
import math

SHAPES = {
    "qkv": (14336, 5120),
    "o": (5120, 6144),
    "gdn_in": (16480, 5120),
    "gate_up": (34816, 5120),
    "down": (5120, 17408),
}
CUS = 256
LDS = 160 * 1024


def valid(bm, bn, bk, st, mw, nw, sk, m, n, k):
    threads = mw * nw * 64
    if bm % (16 * mw) or bn % (16 * nw):
        return False
    if st * (bm + bn) * bk * 2 > LDS or bm * bn * 2 > LDS:
        return False
    if (bm * bk // 8) % 64 or (bn * bk // 8) % 64:
        return False
    wk = -(-k // sk)
    wk = -(-wk // 8) * 8
    last = k - (sk - 1) * wk
    if wk % bk or last <= 0 or last % bk or min(wk, last) // bk < st - 1:
        return False
    tiles = -(-m // bm) * -(-n // bn)
    if sk > 1 and (tiles > 256 or bm * bn % (threads * 4)):
        return False
    # Per-wave MFMA tile must stay within the register file (fp32 accumulators).
    if (bm // mw) * (bn // nw) // 64 > 192:
        return False
    return True


def candidates(m, n, k):
    out = []
    for bm, bn, st, (mw, nw), sk in itertools.product(
        (64, 128, 256),
        (64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 288),
        (2, 3, 4, 5, 6),
        (
            (2, 2),
            (4, 1),
            (1, 4),
            (2, 3),
            (4, 2),
            (2, 4),
            (8, 1),
            (4, 3),
            (2, 6),
            (4, 4),
            (1, 8),
        ),
        (1, 2, 3, 4, 6, 8),
    ):
        bk = 64
        if bm > max(64, m):
            continue
        if not valid(bm, bn, bk, st, mw, nw, sk, m, n, k):
            continue
        pm, pn = -(-m // bm), -(-n // bn)
        progs = pm * pn * sk
        rounds = math.ceil(progs / CUS)
        fill = progs / (rounds * CUS)
        if fill < 0.75 or rounds > 2:
            continue
        # Useful MACs over issued MACs (M/N padding).
        eff = m * n / (pm * bm * pn * bn)
        if eff < 0.85:
            continue
        for x in (False, True) if pm > 1 and sk == 1 else (False,):
            cfg = dict(
                block_m=bm, block_n=bn, block_k=bk, stages=st, m_waves=mw, n_waves=nw
            )
            if sk > 1:
                cfg["split_k"] = sk
            if x:
                cfg["xcd_remap"] = True
            score = fill * eff / rounds
            out.append((-score, -mw * nw, cfg))
    out.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in out]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--ms", default="64,128,256,512")
    p.add_argument("--max", type=int, default=120)
    a = p.parse_args()
    res = {}
    for s in a.shapes.split(","):
        n, k = SHAPES[s]
        res[s] = {m: candidates(int(m), n, k)[: a.max] for m in a.ms.split(",")}
    print(json.dumps(res))
