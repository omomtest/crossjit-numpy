# G13 Production Policy

> Status: Evaluation evidence / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../../../../crossjit_v3.md), [crossjit_v3_plan.md](../../../../crossjit_v3_plan.md), [crossjit_v3_goal.md](../../../../crossjit_v3_goal.md).
> If this document conflicts with those files, the normative specification takes precedence.

Date: 2026-09-28

Formal NPBench evaluation has not started.

The recovery confidence audit remeasured the sole high-variance G12 boundary
cell, linear L4 at N=65536, with five independent processes and seven samples
per process.  The stable 2.224x generated result extends that bounded runtime
range through N=65536.  No NPBench measurement informed this change.

## Runtime pipeline

```text
Host-proven region
    -> NumPy C_P specialization
    -> structural OptimizationSignature + concrete element count
    -> frozen generated production-policy table
        -> exact selected artifact + GENERATED range
             -> generated kernel
        -> exact signature + GENERIC_FALLBACK range
             -> generic fixed backend
        -> stale/missing signature, ORIGINAL range, or unmeasured N
             -> Provider UNPROFITABLE -> original NumPy execution
```

Generated-artifact selection and runtime profitability are separate.  A
registry miss no longer activates the generic backend automatically.

The historical global `elements * (nodes - 1) >= 16384` rule is no longer a
production decision.  `NPY_CROSSJIT_MIN_WORK`, diagnostic forced fallback,
and diagnostic generated-disable overrides remain test/microbenchmark-only.

## Production smoke (forced admission off)

```text
selected generated signature, N=32:
    generated execution (4002 recorded hits)

remeasured generated linear L4 signature, N=65536:
    generated execution (4002 recorded hits)

selected generated signature, unmeasured N=8:
    Provider UNPROFITABLE, original NumPy, zero Provider execution

array-scalar registry miss, measured-profitable N=256:
    generic fixed fallback (4002 recorded hits), zero generated hits

scalar-array registry miss, known-regressive N=256:
    Provider UNPROFITABLE, original NumPy, zero generic hits
```

Evidence:

```text
results/g13/g13_production_policy_smoke.json
SHA-256 24b5683113b203b8ef82f6fdb630dd6f11e06f801023e52b1009ad4be1c27a38
```

The production smoke explicitly verifies that all forced-admission variables
are absent.  It is a policy diagnostic, not formal suite evaluation.

## Stale/disabled behavior

The build generator reconstructs the exact DecisionDB environment before
materializing AUTO entries.  A compiler-version mismatch produces zero
kernels and zero production policies; no stale record is selected.  DISABLE
continues to override FORCE/AUTO.  Disabled records authorize neither
generated nor generic execution; an AUTO_REJECTED record can authorize only a
separately measured generic range.

## Final generated artifact impact

The six kernel bodies are unchanged; G13 adds the eight-record compiled
production policy table and structural lookup.

```text
generated C source:                 9617 bytes
generated object file:              7816 bytes
generated object .text:             1002 bytes
compiled production policy table:   2880 bytes
generated registry table:            144 bytes
policy lookup symbol:                 693 bytes
DecisionDB JSON:                    80425 bytes
```

Final build-artifact hashes:

```text
generated C:
    5648f27d8b167c196378fe7563150ac7c5b72b3394c1691bff7bbbd0ba41d313
generated H:
    0db4593fe185c2f3e3b03de218314c7ff997d9055cb8a95afdfdf37a80ce5d08
generated metadata:
    a91088411c5e70322ac37211e6965085b11a04b45d97479830fb4a59025c8d57
DecisionDB:
    292c9e1e24134c6c46eb4783bc01224ddf75e0a0303ce9dc9540dd4ec416b1b4
```

CPython remains unchanged at
`fdc22d746c4017c3e28e7041f04bb197eee329b1`.  There is no LLVM dependency,
runtime compiler, or benchmark/workload identity in the signature or policy.
