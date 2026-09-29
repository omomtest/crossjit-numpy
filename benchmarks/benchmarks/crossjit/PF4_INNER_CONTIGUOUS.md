# PF4 inner-contiguous generated execution

PF4 is a **PASS**.  The production path is descriptor-driven: NumPy projects
Provider-owned runtime layout facts into a versioned `IterationDescriptor`,
the Generator combines that descriptor with `ElementwiseIR`, and the C backend
emits an outer row loop containing a compiler-visible inner element loop.  No
benchmark or workload identity is present in a runtime signature or policy
lookup.

## Identity and policy separation

Historical `OptimizationSignature` serialization and hashes are unchanged.
The new `ExecutionDecisionDB` is keyed by the complete
`ExecutionSpecialization` hash plus the frozen environment hash.  It therefore
cannot reuse a contiguous measurement to admit an inner-contiguous execution.
Missing element counts, missing specializations, and stale environments select
original NumPy.

The two PF4 artifacts exercise the same backend mechanism with distinct DAGs:

| source | operations | structural hash | execution-specialization hash |
|---|---|---|---|
| observed PF1 | subtract, multiply | `6f833ba31a2de4c89d5d74211c3471b921c7df38e1b7b77b5cb2446fd47400f6` | `b433e05a2c7b0202efe5e9e00da69fba20287373d433b6841efa07e79ec421ce` |
| validation seed | add, multiply | `943ddb58171c2969b54c4bb3cc97bb7410e526b5c170c818c8371ea4b6ef7bcf` | `52c5ff1a8250e7f05a60f113878e438604952067bb3890fe00ed7d01254dac05` |

Generator PF4 profitability commit:
`c66c57c770f06faa9cb02dc6062cf2abb5ea5d47`.

## Correctness and execution evidence

The installed integration suite verifies the exact registry/signature and
specialization match, production-policy execution for both DAGs, exact
unmeasured-specialization rejection, inner stride guards and contiguous-layout
mismatch fallback, NaN/infinity/signed-zero results, FPE soft miss and NumPy
replay, input refcounts, one output allocation per execution, zero internal
ndarray allocations, one Host materialization, and one virtual intermediate.

```text
Generator tests:                       86 passed
NumPy generation/build-tool tests:     15 passed
installed NumPy CrossJIT tests:         31 passed
oracle parity cells:                    10/10 correct
generated/oracle runtime ratio range:   0.9866x--1.0061x
generated/oracle runtime ratio GM:      0.9972x
```

GCC reports both PF4 generated inner loops vectorized with 16-byte vectors.
The retained generated source shows a runtime `outer_count` loop, row pointers
derived from guarded outer strides, and a straight-line `inner_extent` loop.

## Generic signature measurements

The calibration used five fresh processes and seven samples per process for
ordinary NumPy, forced generic CrossJIT, and the generated specialization.
It contains no NPBench timing or workload identity.

At the observed PF1 extent `(outer_count, inner_extent) = (59, 59)`:

```text
ordinary NumPy:              4753.789 ns
generic CrossJIT fallback:   1626.444 ns
generated inner-contiguous:  1160.352 ns
speedup over NumPy:             4.097x
speedup over generic:           1.402x
CV (ordinary/generic/generated): 0.0025 / 0.0050 / 0.0124
```

Across the nine measured shapes, the observed artifact is 3.162x--4.097x
faster than ordinary NumPy and 1.034x--1.402x faster than the generic
fallback.  The validation artifact is 3.180x--4.415x faster than ordinary
NumPy and 1.301x--1.931x faster than generic.  One observed cell (`16384`
elements) exceeded the 0.15 noise ceiling (CV 0.1556), so its persisted policy
is deliberately `ORIGINAL`; all unmeasured counts also remain `ORIGINAL`.

## Historical contiguous regression gate

The original PF3 protocol was repeated over all eight cases and thirteen
element counts (five processes, seven samples):

```text
semantic cells:                          104/104 correct
production-policy Provider cells:         95/104
generated cells:                          78
generic cells:                            17
intentional original-policy cells:         9
generated speedup geometric mean:       2.128x
old/new generated runtime geometric mean: 1.005x
```

The same nine scalar-array cells remain intentional original-policy choices;
there is no material contiguous regression.

## Evidence

```text
results/profiled/pf4_inner_contiguous/execution_specialization_tuning.json
    SHA-256 99557ef3d8d7a57dca37f16d8996bb5806fe6bbe0be3aed75b418924f51e5114
results/profiled/pf4_inner_contiguous/execution_decision_db.json
    SHA-256 09a516bc3b87965debea5d859fff929ca0b2d2d448c6191e458def01f1e160aa
results/profiled/pf4_inner_contiguous/oracle_parity.json
    SHA-256 2d863652786f24322d5668c5c0a4f573c7bb1785d6e913331206eb495f13c959
results/profiled/pf4_inner_contiguous/pf4_contiguous_regression.json
    SHA-256 f47a50a85ea27685e99b9a596e91d68dae44c29054aafe35c064e890108d9d05
```

Limitations are explicit: PF4 admits only exact measured element counts for
the two all-contiguous outer/inner specializations.  Scalar, zero-stride,
fixed-stride, and general-strided descriptors remain outside this policy.
CPython stayed frozen at
`fdc22d746c4017c3e28e7041f04bb197eee329b1`.
