# PF6 coverage remeasurement and Generator/NumPy freeze

Date: 2026-09-30

PF6 is the persisted **coverage remeasurement** milestone. It rebuilds the
PROFILED candidate universe from the final diagnostic profile, repeats the
same 53-workload correctness/coverage diagnostic, and decides optional PF7
fixed-affine and PF8 operation expansion from residual evidence. It does not
add an execution class and it does not begin formal NPBench timing.

## Diagnostic result

The exact PF1 workload list was used (`spmv`, which was added to the suite
later, was explicitly excluded). Both passes completed 53/53 workloads with
`repeat=100`:

```text
forced-generic opportunity pass:
    canonical structural signatures:       24
    refined profile variants:               32
    Provider accepted preparations:         79 / 84 calls
    dynamic execution opportunities: 6,505,405
    generated executions:                    0
    generic diagnostic executions:   6,505,405

frozen production-policy pass:
    Provider accepted preparations:         26 / 81 calls
    generated refined variants:              9
    generated executions:            2,698,694
    generic executions:                      0
    original-NumPy opportunities:    3,806,711
    unprofitable preparation rejects:       50
    unsupported-node rejects:                5
```

Generated dynamic coverage is therefore 41.48%. Relative to the GEN-AUTO
final diagnostic, generated executions increased from 2,044,300 to 2,698,694
(+654,394, +32.01%) and accepted preparations increased from 20 to 26. The
opportunity universe and canonical structural identities are unchanged from
PF1; PF4/PF5 added independent execution specializations and did not change
`OptimizationSignature`.

The final structural observation file contains the same 24 signature
identities and now carries the final 79 accepted preparations and 6,505,405
execution opportunities. Rebuilding all artifacts from it and the PF6
opportunity report is byte-for-byte identical to the installed PF5 artifacts.

## Coverage freeze audit

Primary layout classes are mutually exclusive. The additional operand-bearing
counts for fixed/general stride overlap the scalar-primary row and are shown
only to quantify residual access demand.

| Class | Dynamic count (frequency) | Implemented? | Generated? | Profitability measured? | Production decision |
| --- | ---: | --- | --- | --- | --- |
| full contiguous | 0 PF1 refined opportunities | yes | yes | yes | retain frozen GEN-AUTO exact-point policy; 104-cell historical matrix remains the regression gate |
| inner contiguous | 102,859 (1.58%) | yes | yes | yes | generated for the measured specialization/count, otherwise original NumPy |
| scalar | 5,690,648 (87.48%) | partial: 14/24 refined variants | yes; 8 variants executed in PF6 | yes | selected exact specializations/counts generated; rejected/unmeasured/fixed/general residuals use original NumPy |
| zero stride | 0 observed | yes | yes, validation specialization | yes | bounded validation policy only; alternate/unmeasured relations use original NumPy |
| fixed affine | 710,648 primary (10.92%); 3,017,583 operand-bearing incl. scalar | no | no | no | **DEFER**: concentrated in `adi`/`channel_flow` at 59/98-element fixtures; no fixed-stride static-oracle or generic-signature benefit justifies a new lowering/policy family before freeze |
| truly general stride | 1,250 primary (0.02%); 2,500 operand-bearing incl. scalar | no | no | no | **UNSUPPORTED**: four primary variants in one workload do not justify general stride machinery |
| unsupported semantic operations | 5 preparations | no | no | no | **UNSUPPORTED**: two in-place adds violate pure/fresh recipes; three matmul nodes are non-elementwise |

PF7 is deferred, not implicitly authorized by frequency. Fixed-stride benefit
has not been measured, so its profile frequency may not be converted into a
production policy. PF8 is unsupported for the observed residual: there is no
missing pure elementwise atomic operation in the five semantic rejects.
Float32/integer, in-place, matmul, and new Host semantics remain outside the
observed/justified scope.

The current adapter covers the execution classes required to demonstrate the
research abstraction: frozen full-contiguous execution, descriptor-driven
inner-contiguous execution, scalar access, and zero-stride access, with exact
specialization-specific profitability and conservative original-NumPy
fallback. Residual fixed/general striding is explicitly recorded rather than
forcing a Cartesian expansion.

```text
GENERATOR_SCHEMA_FREEZE
NUMPY_ADAPTER_COVERAGE_FREEZE
```

Only correctness bugs may change the frozen Recipe schema, ElementwiseIR
schema, ExecutionSpecialization schema, ElementwiseOptimizer interface, C
backend interface, runtime registry format, OptimizationSignature format, or
DecisionDB format.

## Production-freeze preparation

No PF6 execution specialization was introduced, so no existing decision was
retuned. All GEN-AUTO, PF4, and PF5 measurements and exact environment keys
were retained. Missing/stale/unmeasured/unprofitable cases continue to select
original NumPy; the production smoke also preserves the separately measured
generic fallback for its historical profitable structural case.

```text
generated kernels:                          28
    GEN-AUTO structural kernels:            11
    profiled execution specializations:     17
structural production policies:             13 (11 selected, 2 rejected)
execution DecisionDB records:               10 (8 selected, 2 rejected)
generated C source:                     36,419 bytes
generated header:                        8,188 bytes
generated metadata:                    285,578 bytes
generated object:                       26,104 bytes
generated object .text:                  9,434 bytes
DecisionDB inputs, total:               212,138 bytes
```

Relative to the 11-kernel GEN-AUTO freeze, the object increased from 11,952
to 26,104 bytes (+14,152), `.text` increased from 3,290 to 9,434 bytes
(+6,144), and generated C increased from 15,383 to 36,419 bytes (+21,036).
The increase is the bounded 17-specialization profile universe, not an
operation/layout Cartesian product.

Artifact hashes:

```text
PF6 observations:       79392ff49af211be6f3585a6ca6f66f099beeb2d73c1b1f03cb20fa021e40a7c
opportunity report:     27c4890144379302289aca642b2b34b08e68011a67a63f5731455698528041f2
production report:      7af950377d77912bbe8488be18caf0700706e7d9b733c479cc5e94976771c100
GEN-AUTO DecisionDB:    6edf4bcd08b9d68d070acdda890e8bf8769ce2059e113cc3a979fc7a3f242e05
PF4 execution DB:       09a516bc3b87965debea5d859fff929ca0b2d2d448c6191e458def01f1e160aa
PF5 execution DB:       0ffbe9f093fb08c722a64fd7fe79fdda441bae7f5867a43e5ec331e576b41176
generated C:            f9cf6a7ee874f20d8f46bc840f58436abe3c990f67ebfb52faf16ef286088e27
generated H:            16adc1393ac4f853343033fa87f5f351c0b7c797351300ed114ed72e26be98de
generated metadata:     51b7f468ca0705d98038708373f69a60f904e45dacc8a46efa0c59e90e7894fb
generated object:       b4286506f770aa766780b7e6643bcb435ecc2be0e889bdb645c950a1d7abd9c7
```

Freeze source hashes:

```text
Generator schema/interface set:
    a7602c182cabef94135124f4118a6651abcc9d80feb1e81eb6e7b14958410a6f
NumPy adapter/registry set:
    0f198cb16d33c5fde780fb35e3abe68a8797d29fd8a85d1e383b1ad486dedf8d
```

## Validation

```text
Generator tests:                         87 passed
NumPy generation/build-tool tests:       17 passed
installed NumPy CrossJIT tests:          32 passed
PF6 opportunity diagnostic:              53/53 valid
PF6 production diagnostic:               53/53 valid
fresh NumPy build:                        481/481 actions
fresh/current C/H/metadata/object:        byte-for-byte identical
historical contiguous semantic cells:    104/104 correct
historical Provider-active cells:         95/104
generated cells:                          78
generic cells:                            17
intentional original-policy cells:         9
generated-cell speedup GM:              2.093x
PF6/PF5 generated runtime ratio:        0.9967x (no material regression)
production policy smoke:                  pass
CPython:                                  clean at frozen HEAD
```

Formal NPBench timing has not begun.
