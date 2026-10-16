# PF5 scalar and zero-stride operands

> Status: Evaluation evidence / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../../../../crossjit_v3.md), [crossjit_v3_plan.md](../../../../crossjit_v3_plan.md), [crossjit_v3_goal.md](../../../../crossjit_v3_goal.md).
> If this document conflicts with those files, the normative specification takes precedence.

PF5 is complete. The C backend now derives scalar and inner-broadcast access
from the same versioned `IterationDescriptor` used by PF4. Scalar and
zero-stride operands use invariant `[0]` loads inside the compiler-visible
inner loop; contiguous operands retain `[i]` loads. Runtime matching includes
access kind and broadcast relation, and the Provider remains responsible for
guarded concrete outer strides.

Generator commit:
`53693ef8fd2662895ed39e279a158ea8a60fecaf`.

## Scope derived from PF1

PF1 recorded 24 scalar profile variants, 5,690,648 dynamic opportunities, and
no zero-stride workload region. PF5 admits only the descriptor classes the C
backend can express without fixed/general striding:

```text
PF1 scalar variants projected to generated artifacts: 14
    flat:                                             7
    outer/inner:                                      7
dynamic opportunities in those 14 variants:   3,382,463
    existing flat-policy opportunities:        1,966,290
    outer/inner opportunities:                  1,416,173
```

Five replayable two-node outer/inner scalar variants account for 1,210,455
PF1 dynamic opportunities and received new measurements. Two three-node
outer/inner variants (205,718 opportunities) could not be reconstructed by
the generic signature harness without failing Host physical proof; their
persisted envelopes are empty `AUTO_REJECTED` records. The remaining PF1
scalar variants require fixed or general striding and remain unsupported.

Because PF1 observed no zero-stride region, PF5 uses one explicit
`add,multiply` validation specialization with access
`ZERO_STRIDE,CONTIGUOUS,CONTIGUOUS` and `BROADCAST_INNER`. This validates and
measures the abstraction without claiming workload coverage.

## Measurements

Generic structural-signature measurements used five independent processes and
seven samples per process. Each cell compares ordinary NumPy, the existing
generic CrossJIT fallback, and generated execution across nine outer/inner
shapes. All 54 cells were correct.

```text
five observed scalar specializations:
    speedup over ordinary NumPy:   2.614x--4.614x
    speedup over generic fallback: 1.145x--8.328x
    maximum independent-run CV:    0.117

zero-stride validation specialization:
    speedup over ordinary NumPy:   3.201x--7.322x
    speedup over generic fallback: 1.299x--2.420x
    maximum independent-run CV:    0.127
```

All 54 measured exact points clear the 1.05 speedup and 0.15 noise gates and
select generated execution. Unmeasured element counts, the two unreplayable
three-node scalar specializations, alternate broadcast relations, and stale
environments select original NumPy. Flat scalar policy remains governed by
the historical generic-signature DecisionDB; PF5 does not globally force it.

Production runtime tests prove scalar and inner-zero-stride registry hits at a
measured point, exact results, input refcount stability, one output allocation,
zero intermediate ndarray allocations, and one Host materialization. An
outer-broadcast descriptor mismatch and an unmeasured scalar extent both
remain on original NumPy. The complete installed suite is 32/32 passing.

Direct GCC inspection reports every generated outer/inner scalar and
zero-stride inner loop vectorized with 16-byte vectors.

## Contiguous compatibility

The 104-cell historical gate was repeated after attaching both PF4 and PF5
execution DecisionDBs:

```text
semantic cells:                          104/104 correct
production-policy Provider cells:         95/104
generated cells:                          78
generic cells:                            17
intentional original-policy cells:         9
PF4/PF5 generated runtime geometric mean:  0.9999x
```

The path split is unchanged and there is no material contiguous regression.

## Evidence

```text
results/profiled/pf5_scalar_zero_stride/execution_specialization_tuning.json
    SHA-256 f39ff11f06b07eba24fbd2636f6ccc620f97fd0212c03e6fb41bcd6281d30435
results/profiled/pf5_scalar_zero_stride/execution_decision_db.json
    SHA-256 0ffbe9f093fb08c722a64fd7fe79fdda441bae7f5867a43e5ec331e576b41176
results/profiled/pf5_scalar_zero_stride/pf5_contiguous_regression.json
    SHA-256 5d4cb6895ff12f374b0eae31ae5dacfac017079f6bc594f679feb401fa6014f4
```

PF5 deliberately does not add matmul, in-place operations, fixed affine
stride execution, truly-general striding, alternate zero-stride broadcast
relations, new Host semantics, or runtime compilation. CPython remains frozen
at `fdc22d746c4017c3e28e7041f04bb197eee329b1`.
