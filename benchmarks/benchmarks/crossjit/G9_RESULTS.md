# G9 generated-execution P1 rerun — 2026-09-27

G9 repeated the complete P1 float64 `CONTIGUOUS_SAME_SHAPE` primary grid
and its focused transition grid after build-time generated kernels were wired
into the NumPy Provider registry.  Admission was forced with
`NPY_CROSSJIT_MIN_WORK=0`; this is measurement, not production policy.

## Method

- primary: eight cases, L2-L6 plus branched, array-scalar, and scalar-array;
- N = 16 through 65536 on the original 13-point grid;
- transition: seven cases on N = 96, 160, 192, 224, 320, 384, 448, 768;
- five independent baseline/CrossJIT process pairs, seven samples per cell;
- randomized mode order, correctness checking enabled;
- detailed timing disabled for primary results and enabled only in separate
  cost-decomposition processes.

All 104/104 primary and 56/56 transition cells were correct and recorded
actual Provider execution.  The six generated array-array signatures account
for 78 primary cells.  The two scalar signatures account for 26 registry
misses and use the retained generic fallback.

## Primary result

| signature | backend | GM baseline/generated | minimum | maximum | GM P1-generic/generated |
| --- | --- | ---: | ---: | ---: | ---: |
| L2 | generated | 1.572x | 1.175x | 2.091x | 1.293x |
| L3 | generated | 1.987x | 1.460x | 3.366x | 2.727x |
| L4 | generated | 2.122x | 1.672x | 3.140x | 2.680x |
| L5 | generated | 2.376x | 1.776x | 3.392x | 2.727x |
| L6 | generated | 2.333x | 1.940x | 2.964x | 2.705x |
| branched | generated | 1.875x | 1.438x | 2.693x | 2.794x |
| array-scalar | generic fallback | 1.441x | 1.161x | 1.620x | 1.015x |
| scalar-array | generic fallback | 0.578x | 0.271x | 1.518x | 0.991x |

The geometric mean across all 104 cells is 1.659x.  Across the 78 generated
cells it is 2.025x, and every generated cell is profitable on this forced
matrix.  The scalar-array regression remains visible because that signature
is not generated and still exercises the generic fallback.

Representative cells:

| case | N | baseline | P1 generic CrossJIT | generated CrossJIT | baseline/generated | P1/generated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | 256 | 553.15 ns | 534.06 ns | 451.89 ns | 1.224x | 1.182x |
| L4 | 4096 | 6230.72 ns | 10231.57 ns | 2154.40 ns | 2.892x | 4.749x |
| L6 | 65536 | 127001.40 ns | 196620.67 ns | 51241.00 ns | 2.479x | 3.837x |

G6 independently established identical normalized assembly shape and
near-1.00 generated/oracle loop ratios.  The remaining difference between
the standalone oracle loop and G9 end-to-end time is expected Provider guard,
allocation, and callback work.  Detailed instrumentation itself perturbs
large-loop timing, so primary timing—not the diagnostic loop timer—is the
performance authority.

## Decision

```text
G9 = PASS
```

Generated execution recovers the G0/G6 static-kernel behavior and eliminates
the P1 medium/large L4-L6 generic-loop regression.  No production threshold
or selection rule is changed by G9.  Demand-driven signature discovery is the
next milestone.

Machine-readable data:

```text
results/g9_generated_float64_contiguous.json
results/g9_generated_float64_contiguous.csv
results/g9_generated_float64_contiguous_transition.json
results/g9_generated_float64_contiguous_transition.csv
```
