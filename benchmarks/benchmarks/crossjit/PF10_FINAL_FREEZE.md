# PF10 final GEN-PROFILED production freeze

> Status: Frozen implementation report / non-normative.
>
> The only normative CrossJIT v3 specification is defined by
> [crossjit_v3.md](../../../../crossjit_v3.md), [crossjit_v3_plan.md](../../../../crossjit_v3_plan.md), [crossjit_v3_goal.md](../../../../crossjit_v3_goal.md).
> If this document conflicts with those files, the normative specification takes precedence.

Date: 2026-09-30

PF10 is a **PASS**.  It freezes the already-converged PF9 production
selection without changing the candidate universe, any DecisionDB, or any
runtime profitability envelope.  PF11/PF12, PF7/PF8, Host work, new
dtype/layout support, LLVM, and runtime compilation were not started.

## PF9 production selection retained by PF10

```text
PF6 generated artifacts:                    28
selected production artifacts:               9
rejected PF6 artifacts:                     19
expected generated executions:       2,698,694
total dynamic opportunities:          6,505,405
expected generated dynamic coverage:    41.4839%
```

The final rule retains measured artifacts with nonzero frozen-profile
opportunity and a valid authorized envelope, excludes zero-frequency
validation artifacts, and rejects stale, unmeasured, or `AUTO_REJECTED`
identities.  PF6 does not contain enough physical-stride evidence to assert
structural/refined dominance, so PF9 does not invent such a claim.  Runtime
DecisionDB contents and profitability envelopes remain byte-for-byte frozen.

## Regression gates

```text
Generator tests:                              87 passed
NumPy generation/build-tool tests:            20 passed
installed NumPy CrossJIT tests:                32 passed
historical contiguous semantics:             104/104 correct
historical Provider-active cells:              95/104
historical generated / generic / original:     78 / 17 / 9
PF4 inner-contiguous regression gate:          pass
PF5 scalar/zero-stride regression gate:        pass
production diagnostic correctness:             53/53
production-policy smoke:                       pass
fresh release build:                           481/481 actions
fresh/current C/H/metadata/object:             byte-for-byte identical
CPython:                                       clean and unchanged
```

The production diagnostic retained the required execution split:

```text
Provider-accepted preparations:                   26
generated executions:                      2,698,694
generic fallback executions:                       0
unprofitable rejections:                           50
unsupported-node rejections:                       5
```

The only PF10 diagnostic code fix extends the recipe-hash attribution catalog
to production-policy signatures when PF9 has pruned a validation kernel.  The
diagnostic script and its tests are not Meson generation inputs.  A fresh
build reproduced the existing registry, kernel set, C, header, metadata, and
object bytes exactly; all three DecisionDB input hashes also remained exact.
Thus the fix changes attribution/reporting only, not production
profitability, runtime admission, or execution semantics.

## Frozen schemas, environment, and hashes

```text
crossjit-generator package version:             0.2.0
Recipe schema:                                       1
ElementwiseIR schema:                                1
IterationDescriptor schema:                          1
ExecutionSpecialization schema:                      1
ElementwiseOptimizer version:                        1
C Backend version:                                   1
OptimizationSignature schema:                        1
OptimizationSignature provider ABI:                  3
OptimizationSignature canonical format: sorted compact ASCII JSON + SHA-256
DecisionDB schema:                                   1
ExecutionDecisionDB schema:                          1
artifact format version:                             1

numpy.add recipe:
    e7cbfdf3a3d2747bbda1c949a3c7eeb707bd3551c0f99d15949e844f89014f33
numpy.multiply recipe:
    46f22c2a16bd650edd3b6955d40a75cf43fdc26513eec4d15245133267b39f9b
numpy.subtract recipe:
    6ca557bca5b492181b23f0a76c3b21bf057b3a8afee58ceed13aa8d1fa99f404
numpy.true_divide recipe:
    0820937da9f6cf4eacc31fbbd7099cf563a812aaba51ee4ee5bf68e279db2697
recipe set:
    55daa90da80f3dad7f9b53380e42817e1b209ba88ac181ef11cc7493c53e087f

structural DecisionDB:
    6edf4bcd08b9d68d070acdda890e8bf8769ce2059e113cc3a979fc7a3f242e05
PF4 ExecutionDecisionDB:
    09a516bc3b87965debea5d859fff929ca0b2d2d448c6191e458def01f1e160aa
PF5 ExecutionDecisionDB:
    0ffbe9f093fb08c722a64fd7fe79fdda441bae7f5867a43e5ec331e576b41176
production selection manifest:
    0c47c85fd321e6bed46dcf8551c97f78f2fd6ed8d3b2b7423b8ba3e0998a85c7
candidate universe:
    dc7dbe6fe56fb7ee9d51299ed1a954de70c6dee28929f1c748974a1abe153848
generated C:
    3f911134f7273efac61e225888b98ca8e09ab9dbdccadb1591ddf43fac6c4977
generated header:
    cd37427bdd20791909be00e88798c75515b0bc7ceb2fec532576347b3249c715
generated metadata:
    a316f5275832d5486e239f20b6446d6d5ad363048ced0bcd37c3a423943036c2
generated object:
    9e50246fc521bc454ba6ec6afaf993c9865bd26ac91d014e34f5a18332260adf
```

Compiler/environment identity:

```text
compiler:          gcc 13.3.0
target triple:     x86_64-linux-gnu
target features:   cx16,popcnt,sahf,sse,sse2,sse3,sse4.1,sse4.2,ssse3
compiler flags:    -std=c11 -O3 -fno-strict-aliasing -msse -msse2 -msse3
                   -mssse3 -msse4.1 -msse4.2 -mpopcnt -msahf -mcx16 -fPIC
object format:     ELF64 x86-64
```

## Final production artifact summary

```text
selected production artifacts:                  9
rejected PF6 artifacts:                         19
generated source size:                      16,534 bytes
header size:                                 4,311 bytes
metadata size:                             169,233 bytes
object size:                                13,984 bytes
.text size:                                  3,762 bytes
DecisionDB inputs:                         212,138 bytes
expected generated dynamic coverage:        41.4839%
```

The authoritative repository checkpoints are the annotated freeze tags
`FINAL_GENERATOR_FREEZE`, `FINAL_NUMPY_ADAPTER_FREEZE`, and
`FINAL_PRODUCTION_POLICY_FREEZE`.  Their resolved commit hashes are recorded
in the workspace plans after the final NumPy commit is created.

