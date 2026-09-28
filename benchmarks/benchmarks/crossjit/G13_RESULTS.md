# G13 Production Policy

Date: 2026-09-28

Formal NPBench evaluation has not started.

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
SHA-256 55ad3dc2ff6656e377b19c4dcfbe455242a9c3d649760dc7a9f661c6ba935dda
```

The production smoke explicitly verifies that all forced-admission variables
are absent.  It is a policy diagnostic, not formal suite evaluation.

## Stale/disabled behavior

The build generator reconstructs the exact DecisionDB environment before
materializing AUTO entries.  A compiler-version mismatch produces zero
kernels and zero production policies; no stale record is selected.  DISABLE
continues to override FORCE/AUTO, and a disabled/rejected artifact cannot
authorize generated execution.

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
DecisionDB JSON:                    80302 bytes
```

Final build-artifact hashes:

```text
generated C:
    e9fb7c527a1c9cb62c3eb96a8770be222a45c3a170b77b6efac174d8caa881cc
generated H:
    0db4593fe185c2f3e3b03de218314c7ff997d9055cb8a95afdfdf37a80ce5d08
generated metadata:
    1e84bd3e30c370ee9d39e780b9f35f1b0fa8c6ff181d89c92cf0d04478ce82ba
DecisionDB:
    e91373c3ab1b035ed558542e58d64df91a58e838a405e5d12f1e6beeb14ae33c
```

CPython remains unchanged at
`fdc22d746c4017c3e28e7041f04bb197eee329b1`.  There is no LLVM dependency,
runtime compiler, or benchmark/workload identity in the signature or policy.
