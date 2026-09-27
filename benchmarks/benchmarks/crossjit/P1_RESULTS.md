# P1 current fixed-fusion calibration — 2026-09-27

These results describe the pre-P2 fixed/precompiled implementation.  Every
CrossJIT row used `NPY_CROSSJIT_MIN_WORK=0`, is marked forced-admission, and
proved Provider execution.  They are not production profitability policy.

## Executed matrix

- float64, `CONTIGUOUS_SAME_SHAPE`;
- N = 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
  32768, 65536;
- linear regions of length 2–6, plus branched, array–scalar, and
  scalar–array patterns;
- five independent processes per baseline/CrossJIT mode, seven samples per
  process, randomized baseline/CrossJIT process ordering;
- 104/104 primary cells valid with `exec_hit > 0` in every CrossJIT process.

The evidence-driven transition scan added N = 96, 160, 192, 224, 320, 384,
448, and 768 for seven cases.  It produced another 56/56 valid cells.

Primary-grid median process-to-process CV was 1.54% for baseline and 0.44%
for CrossJIT; maxima were 27.53% and 10.45%.  The focused scan medians were
0.77% and 0.86%, with maxima of 3.08% and 2.63%.  No single-run fastest time
is used.

## Current break-even observations

The present curves are not monotonic, so a single `N*` would be misleading.
Observed regions are:

- two-node linear: close to parity below 1K; the focused scan ranged from
  0.988x to 1.040x between N=96 and N=448, then reached 1.229x at N=768;
  the primary grid was profitable at all points and reached 1.38–1.66x at
  16K–64K;
- three-node linear and branched: profitable through approximately N=96,
  then regressive from N=128/160 through most of the measured range;
- four-node linear: crosses between N=160 (1.039x) and N=192 (0.980x), and
  remains regressive through 64K;
- five- and six-node linear: cross between N=224 (1.060x/1.112x) and N=320
  (0.927x/0.956x), then remain materially regressive over the medium/large
  range; isolated 64K recoveries are not treated as a policy boundary;
- array–scalar: profitable across the primary grid (1.10–1.62x);
- scalar–array: crosses between N=160 (1.057x) and N=192 (0.937x), then
  degrades substantially.

This inverted/non-monotonic behavior is an important P1 result: the generic
fixed fused loop does not yet scale like NumPy's mature vector loops for many
long DAGs.  P1 therefore installs no threshold or production policy.

## Initial cost decomposition

Dedicated diagnostic processes measured three representative cells.  Main
timing did not enable these timers.

| case | prepare | guard lower | exec lower | Provider guard / exec | allocation / exec | metadata/setup / exec | fixed loop / exec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2, N=256 | 2.190 us | 0.370 us | 0.484 us | 0.131 us | 0.068 us | 0.156 us | 0.105 us |
| L4, N=4096 | 2.819 us | 0.448 us | 0.259 us | 0.219 us | 0.141 us | 0.585 us | 20.752 us |
| L6, N=65536 | 4.662 us | 0.996 us | 0.733 us | 0.360 us | 2.535 us | 11.751 us | 3900.984 us |

At small N, fixed setup and guard work are comparable to the loop itself.
At medium and large N, the fixed loop dominates overwhelmingly.  Its timer
includes the current per-node/per-chunk dispatch, which is not separately
observable without distorting the loop.  Output allocation and Host/EXT
residual are secondary in the measured cells.  Preparation/lowering costs are
one-time costs for the current plan rather than per-execution costs.

Machine-readable primary and transition CSV/JSON files are in `results/`.
