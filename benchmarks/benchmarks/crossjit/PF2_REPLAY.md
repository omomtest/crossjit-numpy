# PF2 Canonical IR Replay

> Status: Evaluation evidence / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../../../../crossjit_v3.md), [crossjit_v3_plan.md](../../../../crossjit_v3_plan.md), [crossjit_v3_goal.md](../../../../crossjit_v3_goal.md).
> If this document conflicts with those files, the normative specification takes precedence.

PF2 replaces the GEN-AUTO tuning harness's Python-source reconstruction and
parameter-permutation search with a persistent, workload-neutral replay
record.  Replay records contain canonical `ElementwiseIR`, logical input-slot
access, operation dependency slots, output slots, Provider specialization,
iteration facts, and materialization assumptions.  They reject benchmark,
workload, source-file, Python-object, and ndarray-object identity.

The synthetic harness decodes the persisted record before use, creates arrays
and scalar storage in logical-slot order, and independently compares:

```text
ordinary NumPy semantic evaluation
generated straight-line C implementation
generic fixed C implementation
```

Synthetic replay calibrates candidates and validates generated code.  It does
not re-establish Host legality; runtime integration correctness continues to
come from end-to-end NumPy/CPython tests.

## Result

```text
optimizer-accepted observed signatures: 8
reproducibly tunable by canonical replay: 8
failures: 0
old source-reconstruction failures recovered: 3
measurement cells: 40
formal NPBench timing: no
```

Recovered signature hashes:

```text
73213d44f41e2bbc6d6169962c2ff1f77caed9009359b5ccb42028b46bbb7367
b31365f25de3abbde35b95ed55297c98ad6bd11205268dbb015e1ea62fa78deb
c79aa989763762989d820e80042404be2e3f84872aaa2575557bcf8357907d94
```

Correctness includes ordinary finite inputs and fixtures containing NaN, Inf,
negative Inf, and signed zero.  The generic fixed comparison is a standalone
kernel-level interpreter and is used only where meaningful for calibration.

Evidence:

```text
results/profiled/pf2_replay/pf2_replay_manifest.json
results/profiled/pf2_replay/pf2_replay_results.json
results/profiled/pf2_replay/pf2_replay_kernels.c
results/profiled/pf2_replay/pf2_replay_kernels.h
```

SHA-256 at the PF2 checkpoint:

```text
replay manifest:
c0fd5310ad7234f447b9c45509e89d8a4932309a311d0dadeaeb462a2fe5034c
replay result:
1e4bd45e7b202c72a0abbc5f732577e02264dc040399820dd2d37507e1138008
```
