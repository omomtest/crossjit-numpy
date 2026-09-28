# G12 Generic Signature Tuning

Date: 2026-09-28

This is generic signature microbenchmark calibration, not NPBench evaluation.
No benchmark/workload identity or observed workload frequency was used for a
selection threshold or production weighting.

## Method

```text
dtype/layout: float64 / CONTIGUOUS_SAME_SHAPE
element grid: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
              8192, 16384, 32768, 65536
independent processes: 3 per implementation/cell
samples per process: 7
implementations: ordinary NumPy, generated kernel, generic fixed fallback
artifact threshold: >=1.05x, relevant-path CV <=0.15
```

The measurement override `NPY_CROSSJIT_MIN_WORK=0` and the diagnostic generic
path override were used only inside this calibration.  They are not production
policy.

## Artifact decisions

```text
measured structural signatures: 8
generated candidates:           6
fallback-only controls:         2
AUTO_SELECTED:                  6
AUTO_REJECTED:                  2
FORCED:                         0
DISABLED:                       0
```

Generated-path GM across the 13-point grid:

```text
linear L2:  1.574x
linear L3:  1.998x
linear L4:  2.087x
linear L5:  2.452x
linear L6:  2.308x
branched L3: 1.856x
```

The initial three-process run conservatively rejected linear L4 at N=65536
because its generated CV exceeded the bound.  The pre-G14 confidence audit
remeasured only that ambiguous boundary using five independent processes and
seven samples per process.  It measured 2.224x generated speedup with 1.82%
generated CV.  Because the adjacent N=32768 and N=65536 boundary measurements
now agree, the final bounded L4 range N=32769..65536 is GENERATED.  The other
decisions were clearly separated and were not rerun.  No envelope extrapolates
above N=65536 or below N=16.

The array-scalar generic fallback was measured profitable across N=16..65536.
The known scalar-array fallback regression is retained: only N=16..128 is
generic-profitable; its grid-wide GM is 0.590x and its speedup falls to 0.299x.
N>=129 therefore rejects to original NumPy under the frozen envelope.

## Code size

```text
generated kernels:                  6
generated C source:              5366 bytes
generated object file:           3944 bytes
generated object .text:          1002 bytes
mean object .text/kernel:          167 bytes
final DecisionDB:                80425 bytes
```

Frozen files:

```text
results/g12/g12_candidate_tuning.json
results/g12/g12_decision_db.json
results/g13/g12_linear4_n65536_confidence_remeasure.json
```

Frozen hashes:

```text
DecisionDB:
    292c9e1e24134c6c46eb4783bc01224ddf75e0a0303ce9dc9540dd4ec416b1b4
generated C:
    2db708e93ad9be443293ac64e5ae3ad199b831d619f49cfd813968a4eccabca6
generated H:
    caa48fc7f51359d6edb6965819f80af40681037cf606ba11ead5730aaef4f581
generated metadata with DecisionDB selection:
    da8fe6c37c7f530cda0d9b69ba0ee61c6cfa44af44917a098bbe813ee2e19b63
```

Frozen versions/environment:

```text
Generator:             0.2.0
ElementwiseOptimizer:  1
CBackend:              1
artifact schema:       1
Provider ABI:          3
compiler:              gcc 13.3.0
target:                x86_64-linux-gnu
target features:       cx16,popcnt,sahf,sse,sse2,sse3,sse4.1,sse4.2,ssse3
```

The production build now consumes the DecisionDB for AUTO materialization.
Formal G14 NPBench evaluation has not started.
