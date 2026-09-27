# CrossJIT Generator G0 static-C feasibility — 2026-09-27

Decision:

```text
G0 = PASS
```

The benchmark-only compiler-visible loops materially reduce the P1 L4/L6
generic fixed-loop bottleneck, preserve the tested value/FPE behavior, and do
so reproducibly in independent processes.  Full Generator implementation may
proceed.

## Selected P1 signatures

All cases use `float64`, `CONTIGUOUS_SAME_SHAPE`, distinct read-only input
arrays, one fresh output, and the complete P1 primary grid:

```text
N = 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536
```

The exact operation sequences are:

```text
L2: add, mul
    (a + b) * c

L4: add, mul, sub, true_divide
    ((a + b) * c - d) / e

L6: add, mul, sub, true_divide, add, mul
    (((a + b) * c - d) / e + f) * g
```

P1 ordinary-NumPy and generic-CrossJIT values come unchanged from
`results/p1_float64_contiguous.json`: five independent processes, seven timed
samples per process, correctness enabled, and forced CrossJIT execution proven
for every row.  The static oracle uses five new independent processes and
seven timed samples per process.  It uses a preallocated output so its primary
comparison is against P1's measured `fixed_fused_loop_ns`, not against a full
Python call.

The already-present P3 result file is included as a secondary end-to-end
confirmation because it measures an integrated handwritten static path with
the same original P1 Python harness.  It was pre-existing user work and is not
used as the sole G0 proof.

## Representative measurements

Times are medians in nanoseconds per region execution.

| case | N | ordinary NumPy | P1 generic CrossJIT | P1 generic loop diagnostic | benchmark-only static loop | generic-loop / oracle | integrated static CrossJIT | integrated speedup vs P1 generic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | 256 | 543.50 | 534.06 | 104.72 | 43.66 | 2.40x | 453.05 | 1.18x |
| L4 | 4096 | 5919.06 | 10231.57 | 20752.48 | 1547.45 | 13.41x | 2148.90 | 4.76x |
| L6 | 65536 | 138189.23 | 196620.67 | 3900983.56 | 52657.80 | 74.08x | 51277.97 | 3.83x |

The opt-in P1 and integrated-static loop timers perturb execution and can be
larger than their corresponding non-diagnostic end-to-end timings.  They are
used only for like-for-like bottleneck classification.  The non-diagnostic
end-to-end rows independently confirm the conclusion: L4 changes from a
`0.579x` regression against ordinary NumPy to a `2.788x` speedup, and L6 from
`0.703x` to `2.468x`.  L2 remains healthy (`1.217x` integrated-static speedup
against ordinary NumPy).

Across the five independent static-oracle processes at the representative
cells:

```text
L2/N=256:
    37.62–47.47 ns, median 43.66 ns

L4/N=4096:
    1546.95–1548.05 ns, median 1547.45 ns

L6/N=65536:
    48.23–70.66 us, median 52.66 us
```

The hybrid-core machine produces visible L6 process-to-process throughput
variation, but even the slowest oracle process remains more than 55x faster
than the P1 diagnostic generic-loop cost.  The conclusion is not dependent on
one best process.

## Correctness and FP policy

The oracle:

- preserves the exact P1 operation order using explicit temporaries;
- is compiled with `-O3 -fno-strict-aliasing -ffp-contract=off` and the same
  SSE/SSE2/SSE3/SSSE3/SSE4 feature floor recorded by the NumPy build;
- does not use `-ffast-math` or any other unsafe FP relaxation;
- checks bitwise finite and signed-zero results, treats corresponding NaNs as
  equal, and compares `FE_DIVBYZERO`, `FE_INVALID`, `FE_OVERFLOW`, and
  `FE_UNDERFLOW` against an independent scalar operation-order reference;
- passes both the ordinary finite fixtures and NaN/Inf/signed-zero/zero-divisor
  fixtures for every measured process.

## Compiler evidence

GCC 13.3 reports:

```text
L2 loop vectorized using 16 byte vectors
L4 loop vectorized using 16 byte vectors
L6 loop vectorized using 16 byte vectors
```

Assembly contains one compiler-visible loop per oracle with ordered packed
double operations:

```text
L2: addpd, mulpd
L4: addpd, mulpd, subpd, divpd
L6: addpd, mulpd, subpd, divpd, addpd, mulpd
```

There is no opcode switch, semantic-node traversal, or generic slot indexing
in any oracle loop.

The separately compiled generic fallback reports missed vectorization because
of nested loops, dynamically indexed slots, and unsuitable strided loads.  Its
assembly retains dtype dispatch and branch-heavy generic loop structure.  The
evidence does not identify alias uncertainty, required FP/FPE semantics, or
register pressure as the primary blocker: the same strict FP policy and input
footprint vectorize once dynamic semantic/slot control is removed.  At large N
the successful static loop may become limited by arithmetic division and
memory/cache bandwidth, but those are residual throughput limits rather than
the P1 dispatch bottleneck.

Evidence files:

```text
g0-build/g0_vectorization.txt
g0-build/g0_static_oracle.s
g0-build/generic_vectorization.txt
g0-build/generic_jit_kernel.s
results/g0_generator_feasibility.json
results/g0_generator_feasibility.csv
```

## Gate rationale

G0 asks whether compiler-visible signature-specialized C can solve the
measured fixed-loop problem before building Recipe/IR/DecisionDB machinery.
The answer is yes:

1. L4/L6 loop cost drops far beyond measurement noise.
2. Independent end-to-end static execution recovers and exceeds ordinary
   NumPy performance for both regressive signatures.
3. Correctness and tested FPE flags remain aligned with the ordered reference.
4. Compiler diagnostics directly explain the change: straight-line static
   loops vectorize, while the generic node/slot/stride machinery does not.

Next gate: G1, establish the generic `crossjit-generator` repository.
