# PF1 profile-guided coverage taxonomy

> Status: Evaluation evidence / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../../../../crossjit_v3.md), [crossjit_v3_plan.md](../../../../crossjit_v3_plan.md), [crossjit_v3_goal.md](../../../../crossjit_v3_goal.md).
> If this document conflicts with those files, the normative specification takes precedence.

Date: 2026-09-29

This is a coverage diagnostic, not a timing result. The run used the NPBench S
workload set with `repeat=100` and forced the already-existing generic Provider
executor so execution opportunities could be counted for layouts that the
GEN-AUTO production policy conservatively rejects. All 53 workloads completed.

Canonical runtime signatures still exclude workload identity. Workload names
appear only in the profile report's occurrence mapping. The 24 GEN-AUTO
canonical signatures split into 32 profile variants after access-kind and
refined-layout classification.

## Coverage before expansion

```text
canonical observed signatures:            24
refined profile variants:                  32
optimizer-supported profile variants:      12
signature coverage:                     37.50%

dynamic execution opportunities:    6,505,405
optimizer-supported opportunities:   2,827,920
dynamic coverage:                       43.47%

recognized nodes:                          664
Provider prepare calls:                      84
forced diagnostic accepted preparations:    79
unsupported-node rejections:                  5
```

The prepare count is 84 rather than the GEN-AUTO diagnostic's 81 because the
Host prefix-retry path exposes three additional shorter candidates when the
diagnostic admits all structurally supported layouts.

## Refined layout classes

| Class | Variants | Prepare count | Dynamic opportunities | Workloads |
| --- | ---: | ---: | ---: | ---: |
| `FULL_CONTIGUOUS` | 0 | 0 | 0 | 0 |
| `INNER_CONTIGUOUS` | 1 | 2 | 102,859 | 1 |
| `ZERO_STRIDE_BROADCAST` | 0 | 0 | 0 | 0 |
| `SCALAR_OPERAND` | 24 | 65 | 5,690,648 | 7 |
| `FIXED_AFFINE_STRIDE` | 3 | 8 | 710,648 | 1 |
| `TRULY_GENERAL_STRIDED` | 4 | 4 | 1,250 | 1 |

`SCALAR_OPERAND` is a primary region class; its per-operand access vector
retains whether each non-scalar input is contiguous, fixed-stride, or general.
Consequently PF5 scalar support alone will not recover scalar regions whose
array operands still require PF7-style fixed stride.

The existing optimizer supports 12 scalar variants whose array operands are
otherwise `contiguous_same_shape`. The remaining 20 variants are rejected by
the GEN-AUTO layout restriction.

## Unsupported nodes

Five rejected preparations reduce to three contextual records:

| Operation | Occurrences | Workload | Context | Atomic Recipe fit |
| --- | ---: | --- | --- | --- |
| in-place `numpy.add` | 1 | `seidel_2d` | float64, full contiguous, L2 | no: visible mutation |
| in-place `numpy.add` | 1 | `nbody` | float64, scalar operand, L2 | no: visible mutation |
| `numpy.matmul` | 3 | `nbody` | float64, scalar operand, L3 | no: non-elementwise |

These are not candidates for PF8 elementwise atomic-recipe expansion. The
in-place operations violate the current pure/fresh Recipe contract and matmul
is outside the elementwise domain.

## Priority evidence

The profile strongly prioritizes operand semantics plus fixed affine stride:

```text
scalar-primary opportunities:        5,690,648 (87.47%)
fixed-affine-primary opportunities:    710,648 (10.92%)
inner-contiguous-primary:              102,859 ( 1.58%)
truly-general-strided:                   1,250 ( 0.02%)
```

PF4 and PF5 remain required by the phase specification. PF6 must decide PF7
from the residual counts after those implementations. The current evidence
already rules out arbitrary general-strided generation as a priority.

Machine-readable report:

```text
benchmarks/benchmarks/crossjit/results/profiled/pf1_coverage/coverage_report.json
```
