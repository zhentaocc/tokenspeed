# FlyDSL BF16 GEMM for the Qwen3.8-27B verify shapes (gfx950)

Research kernel for tracker #95 step 3. It is not routed in tokenspeed.

- `fly_gemm.py`: kernel derived from FlyDSL 0.5.3 `gemm_a16w16_gfx950.py` (non-HTI, "nt"). It adds a partial last load round (144/160/224/288-wide N tiles with 4-12 waves), an XCD remap, a persistent grid and in-kernel split-K.
- `bench.py`: cold rotating weights (> LLC) under HIP-graph replay, median µs. Each config is checked against an FP32 torch reference before and after timing.
- `candidates.py`: sweep generator, pruned by CU fill and tile padding.

Setup: MI355X (m10-05), image `tokenspeed-runner-amd:rocm7.2.4-pytorch-2.14.0-gfx950`, `pip install flydsl==0.5.3`, and a FlyDSL v0.5.3 checkout for `kernels/`. hipBLASLt uses the tuned CSV `device_name=AMD_Instinct_MI355X.csv`, loaded explicitly.

## Sweep: best of about 3,300 configs vs tuned hipBLASLt (µs)

All configs passed the correctness check (relerr < 2e-2; observed 2e-3 to 7e-3).

| shape | M | tuned | best FlyDSL | ratio | config (bm x bn, waves, stages) |
|---|---|---|---|---|---|
| qkv | 64 | 33.2 | 34.5 | 1.04 | 64x112 4x1 st6 sk2 |
| qkv | 128 | 39.1 | 45.5 | 1.16 | 128x128 2x4 st4 sk2 |
| qkv | 256 | 55.7 | 58.8 | 1.06 | 128x112 4x1 st4 xcd |
| qkv | 512 | 77.9 | 86.9 | 1.12 | 128x224 2x2 st3 |
| o | 64 | 19.2 | 19.2 | 1.00 | 64x64 2x4 st6 sk3 |
| o | 128 | 22.5 | 25.4 | 1.13 | 128x64 4x1 st5 sk3 |
| o | 256 | 31.0 | 35.9 | 1.16 | 128x128 4x4 st3 sk3 |
| o | 512 | 45.8 | 50.2 | 1.10 | 128x80 4x1 st4 |
| gdn_in | 64 | 39.9 | 39.2 | 0.98 | 64x80 4x1 st5 |
| gdn_in | 128 | 46.3 | 49.7 | 1.07 | 128x160 2x2 st3 sk2 |
| gdn_in | 256 | 64.7 | 65.9 | 1.02 | 128x144 4x3 st4 xcd |
| gdn_in | 512 | 92.5 | 97.2 | 1.05 | 128x288 2x6 st3 xcd |
| gate_up | 64 | 67.0 | 68.5 | 1.02 | 64x144 4x1 st4 |
| gate_up | 128 | 82.0 | 79.3 | 0.97 | 128x144 4x1 st3 |
| gate_up | 256 | 109.6 | 109.2 | 1.00 | 256x144 4x3 st3 |
| gate_up | 512 | 185.9 | 179.9 | 0.97 | 256x288 4x3 st2 xcd |
| down | 64 | 41.3 | 38.8 | 0.94 | 64x80 4x1 st6 sk4 |
| down | 128 | 48.2 | 50.6 | 1.05 | 128x160 4x2 st4 sk8 |
| down | 256 | 62.7 | 69.6 | 1.11 | 128x160 4x2 st4 sk4 |
| down | 512 | 96.8 | 112.1 | 1.16 | 128x160 4x2 st4 sk2 |

The "best" column is a single-run minimum, so it is biased low. I re-timed the near-parity cells in 5 rounds, each in one process with tuned hipBLASLt (medians):

| shape | M | tuned | FlyDSL | result |
|---|---|---|---|---|
| gate_up | 64 | 64.8 | 69.3 | +7% (loses) |
| gate_up | 128 | 78.8 | 80.3 | +2% (loses) |
| gate_up | 256 | 101.6 | 111.8 | +10% (loses) |
| gate_up | 512 | 184.4 | 171.6 | -7% (wins) |
| down | 64 | 39.0 | 39.4 | parity |
| gdn_in | 64 | 37.1 | 39.6 | +7% (loses) |
| o | 64 | 19.2 | 19.0 | parity |

Tuned hipBLASLt varies by 5-10% between processes: gate_up M=256 measured 101-116 µs and M=512 measured 184-196 µs. FlyDSL moves with it, so only ratios taken in the same process are meaningful.

The gate_up M=512 win holds only at exactly M=512. At M=320/384/448/480, tuned is 170/140/152/177 µs against FlyDSL 167/170/176/181 µs. The tuned M=512 row is about 17 µs slower than M=480, which suggests it is a weak hipBLASLt pick rather than a FlyDSL strength.

## What did not help

- **Persistent grid (two tiles per CU in sequence).** It is 5-60% slower everywhere; for example gate_up M=256 is 119-198 µs against 109 µs. Every tile restarts its prologue, and grid=256 and grid=512 time the same.
- **Two co-resident WGs per CU (occupancy 2, LDS <= 80 KB).** These configs were in the sweep and none came out best.
- **In-kernel split-K / Stream-K.** It helps only at M=64 (o sk3, down sk4: parity with tuned). At M >= 128 the per-tile mainloop is already 5-15% slower than hipBLASLt, and splitting K cannot recover that.
- **block_k=32.** It is wrong in stock FlyDSL 0.5.3 as well: `gemm_a16w16(..., block_m=128, block_n=128, block_k=32, stages=2, m_waves=2, n_waves=2)` at 256x5120x6144 gives relerr 1.53, while block_k=64 gives 2.1e-3. The (3,3,3) LDS swizzle spans 64-element rows. `make_param` now rejects it.
