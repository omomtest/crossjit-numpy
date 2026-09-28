# GEN-AUTO production freeze

Date: 2026-09-29

This checkpoint is the workload-independent CrossJIT Generator reference
configuration. NPBench diagnostic telemetry supplied only the bounded set of
observed structural signatures. Artifact admission and runtime profitability
were decided by generic structural-signature microbenchmarks; workload names,
occurrence frequencies, and NPBench timings were not selection inputs.

## Repositories

```text
CPython (frozen):       fdc22d746c4017c3e28e7041f04bb197eee329b1
NumPy pre-freeze base:  a7e692e4e5c9890150d02ac9e69bfca033b9842a
Generator:              e6773b09104aa5939d948c48e6341a4e49f0353f
```

The final NumPy freeze commit is recorded in the top-level execution plan and
by the `crossjit-gen-auto-freeze` tag after this file is committed.

## Diagnostic discovery and tuning

Pre-tuning 53-workload diagnostic:

```text
53/53 valid
664 recognized nodes
81 prepared regions
24 distinct observed structural signatures
76 supported-node preparations reached generated lookup
5 unsupported-node rejections
```

Of the 24 observed signatures, the v1 optimizer rejected 13
`general_strided` and 3 `inner_contiguous` signatures. Eight contiguous
signatures were structurally accepted. Five were exactly reconstructible by
the generic tuning harness and all five were `AUTO_SELECTED`; the remaining
three are retained as explicit reconstruction failures rather than silently
dropped.

Final production diagnostic:

```text
53/53 valid
664 recognized nodes
81 prepared regions
20 accepted preparations
2,044,300 generated execution hits
0 generic fixed execution hits
56 original-Host unprofitable fallbacks
5 unsupported-node rejections
24 observed structural signatures
```

The pre-tuning and final diagnostics are preserved separately under
`results/g14/diagnostic_pre_tuning/` and `results/g14/diagnostic/`.

## Frozen artifact set

```text
generated kernels:       11
runtime policies:        13
observed candidates:      8
unsupported observations:16
validation seeds:         6

generated C bytes:       15383
generated H bytes:        3411
generated metadata bytes:132689
generated object bytes:  11952
generated object text:    3290
```

SHA-256:

```text
DecisionDB
6edf4bcd08b9d68d070acdda890e8bf8769ce2059e113cc3a979fc7a3f242e05

observed-signature profile
af2fb7474190a10d34805d0d2e420fbc25177117a351181d3bae465e9466ee25

generated C
5a5bc62c9aed9944408ce29761f89ebd81151f64879549088371b09419dd2857

generated H
d9703961dff4dea5f3e409120a07a63949cc2dd2362b1e0c0433e849a06f09f4

generated registry/metadata
3c1feaecac1be71413588d85a6ce8c43b7875b9f64d37d3e4ed266fb4d7ca958

pre-tuning diagnostic
85826ef680bf92e4e69fc2f4f1755fef7c9f83ca169c91f63a9c4dafbcd4e2c1

final diagnostic
576e59f44b29614b2be9a36a56ebf1ec7c9b735f8cf9580ca6543db8f218e5ad
```

## Validation

```text
Generator tests:                         60 passed
NumPy recipe/lowering/generation tests: 14 passed
installed NumPy CrossJIT tests:          27 passed
fresh NumPy build:                       481/481 targets built
fresh/current C, H, metadata:            byte-for-byte identical
final diagnostic:                        53/53 valid
CPython:                                 clean and unchanged
```

No formal NPBench timing was performed for this freeze.
