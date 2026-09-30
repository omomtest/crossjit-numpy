# PF9 profile-guided production artifact selection

Date: 2026-09-30

PF9 selects which already-frozen PF6 generated artifacts are present in the
production binary.  It does not change the structural or execution
DecisionDBs, their element-count envelopes, the Recipe/IR/optimizer/backend
schemas, or the runtime fallback semantics.

## Selection boundary and rule

The deterministic input universe is the 30 frozen candidate identities: 13
structural DecisionDB records plus 17 refined execution specializations.  Two
structural records are explicit `AUTO_REJECTED` non-artifacts, so the PF6
generated-artifact universe contains 28 entries.

PF9 uses a rule-based Pareto filter rather than fitted score weights.  Current
measurements do not justify numerical values for `lambda` or `mu`, and fitting
them merely to increase coverage would be arbitrary.  In canonical identity
order:

1. developer `DISABLE` rejects; developer `FORCE` can select only an exact,
   valid frozen measurement;
2. `AUTO_REJECTED`, stale, and unmeasured identities reject;
3. `AUTO_SELECTED` artifacts must cover at least one frozen-profile execution
   already authorized as `GENERATED` by the unchanged runtime DecisionDB;
4. PF6 aggregate profiles do not encode enough physical-stride information to
   prove dominance between structural flat and refined outer/inner artifacts;
   when both measured identities have nonzero opportunity and authorize an
   observed count, both are conservatively retained rather than assigning
   arbitrary overlap weights;
5. a redundant unmeasured specialization rejects because it has no exact
   profitability evidence;
6. zero-frequency validation seeds remain available to validation generation
   but are excluded from the production binary.

Thus every selected artifact contributes measured, frozen-policy execution
coverage; every rejected measured artifact contributes no expected generated
execution, is validation-only, or is dominated by an already measured
artifact.  Benchmark name, workload identity, source path, and end-to-end
NPBench timing are absent from the selection manifest and are rejected as
inputs.

Build-time artifact selection remains separate from runtime profitability:

```text
PF9 manifest:
    candidate -> artifact present / absent

unchanged DecisionDBs:
    exact candidate + runtime element/count class ->
        GENERATED / GENERIC_FALLBACK / ORIGINAL
```

An artifact being present never admits an unmeasured count.  An artifact being
absent converts a would-be generated registry lookup into conservative
`ORIGINAL`; it does not authorize the generic executor unless the existing
structural policy separately says `GENERIC_FALLBACK`.

## Complete frozen-candidate audit

Hashes are abbreviated only in this human-readable table; the canonical JSON
contains full structural signatures, full execution specializations, complete
profitability points, provenance, confidence/noise, costs, and reasons.
`sm` is the frozen scalar mask.  Cost is generated-function source bytes /
symbol `.text` bytes / canonical kernel-metadata bytes.  Confidence and CV are
copied from the existing generic-signature DecisionDB records; zero confidence
and CV above 0.15 remain visible and their exact runtime `ORIGINAL` points are
not weakened.

| Structural signature | Execution specialization | Frequency | Frozen profitability envelope | Confidence / max CV | Cost | PF9 | Reason |
| --- | --- | ---: | --- | --- | ---: | --- | --- |
| `072e79dc` multiply>subtract sm=1 | `FLAT:SCALAR,CONTIGUOUS,CONTIGUOUS:70e64364` | 50,844 | unmeasured | 0 / - | 497/114/4,425 | reject | unmeasured duplicate; measured structural flat artifact supplies the frozen executions |
| `072e79dc` multiply>subtract sm=1 | `FLAT:SCALAR,CONTIGUOUS,CONTIGUOUS:d04d0aff` | 50,844 | GENERATED 13; N=16..65,536 | 0.0261 / 0.1461 | 497/114/3,204 | **select** | exact `AUTO_SELECTED` artifact covers frozen-policy executions |
| `2bd3ab14` multiply>add sm=2 | `FLAT:CONTIGUOUS,SCALAR,CONTIGUOUS:5f9a51d5` | 1,638,407 | GENERATED 12, ORIGINAL 1; N=16..65,536 | 0 / 0.1578 | 497/110/3,194 | **select** | measured structural artifact is first in the frozen registry and covers all authorized executions; noisy point remains ORIGINAL |
| `2bd3ab14` multiply>add sm=2 | `OUTER_INNER:CONTIGUOUS,SCALAR,CONTIGUOUS:6e827dfb` | 551,535 | GENERATED 9 exact points; includes N=3,481 | 0.9216 / 0.0118 | 939/264/9,246 | **select** | exact measured refinement has nonzero opportunity and authorizes its observed count; stride-level dominance is not proven |
| `2bd3ab14` multiply>add sm=2 | `FLAT:CONTIGUOUS,SCALAR,CONTIGUOUS:74901c17` | 1,086,872 | unmeasured | 0 / - | 497/110/4,415 | reject | unmeasured duplicate; measured structural flat artifact supplies the frozen executions |
| `2cdb7c75` add>multiply>subtract>true_divide>add>multiply sm=0 | `FLAT:CONTIGUOUSx7:dba88b09` | 0 | GENERATED 13; N=16..65,536 | 0.8737 / 0.0189 | 953/210/4,974 | reject | measured validation artifact has zero production frequency/execution |
| `3efa05fa` multiply>multiply sm=4 | `FLAT:CONTIGUOUS,CONTIGUOUS,SCALAR:6ff59b93` | 1,518 | GENERATED 13; N=16..65,536 | 0.2121 / 0.1182 | 497/114/3,204 | **select** | measured structural artifact is first in the frozen registry and covers the authorized executions |
| `3efa05fa` multiply>multiply sm=4 | `OUTER_INNER:CONTIGUOUS,CONTIGUOUS,SCALAR:73b3dbdc` | 1,518 | GENERATED 9 exact points; observed N=625 absent | 0.6423 / 0.0537 | 939/268/9,236 | reject | the exact refined envelope does not authorize the frozen observed count; structural policy supplies it |
| `44db0187` add>multiply sm=0 | `OUTER_INNER:ZERO_STRIDE,CONTIGUOUS,CONTIGUOUS:0a2f077d` | 0 | GENERATED 9; N=16..65,536 | 0.1535 / 0.1270 | 939/264/9,238 | reject | zero-frequency PF5 validation seed; retained outside production generation |
| `44db0187` add>multiply sm=0 | `FLAT:CONTIGUOUSx3:9de787e5` | 0 | GENERATED 13; N=16..65,536 | 0.1484 / 0.1277 | 497/122/3,199 | reject | measured validation artifact has zero production frequency/execution |
| `472e6ded` subtract>multiply sm=4 | `FLAT:CONTIGUOUS,CONTIGUOUS,SCALAR:b5d30d43` | 302,687 | GENERATED 13; N=16..65,536 | 0.7070 / 0.0439 | 497/114/3,204 | **select** | measured structural artifact is first in the frozen registry and covers all authorized executions |
| `472e6ded` subtract>multiply sm=4 | `OUTER_INNER:CONTIGUOUS,CONTIGUOUS,SCALAR:ca832d86` | 102,859 | GENERATED 9 exact points; includes N=3,481 | 0.2187 / 0.1172 | 939/268/9,226 | **select** | exact measured refinement has nonzero opportunity and authorizes its observed count; stride-level dominance is not proven |
| `472e6ded` subtract>multiply sm=4 | `FLAT:CONTIGUOUS,CONTIGUOUS,SCALAR:e4991220` | 199,828 | unmeasured | 0 / - | 497/114/4,425 | reject | unmeasured duplicate; measured structural flat artifact supplies the frozen executions |
| `5113defd` multiply>subtract sm=1 | `OUTER_INNER:SCALAR,CONTIGUOUS,CONTIGUOUS:ed92e3d2` | 551,535 | GENERATED 9; N=16..65,536 | 0.2454 / 0.1132 | 939/276/9,216 | **select** | exact measured scalar specialization covers frozen-policy executions |
| `6f833ba3` subtract>multiply sm=0 | `OUTER_INNER:CONTIGUOUSx3:b433e05a` | 102,859 | GENERATED 8, ORIGINAL 1; N=16..65,536 | 0 / 0.1556 | 939/300/9,263 | **select** | exact measured inner-contiguous specialization; noisy point remains ORIGINAL |
| `73213d44` true_divide>multiply>subtract sm=6 | `FLAT:CONTIGUOUS,SCALAR,SCALAR,CONTIGUOUS:70a9a3ea` | 199,828 | unmeasured | 0 / - | 611/142/5,032 | reject | no exact specialization measurement |
| `73213d44` true_divide>multiply>subtract sm=6 | `OUTER_INNER:CONTIGUOUS,SCALAR,SCALAR,CONTIGUOUS:df121102` | 102,859 | AUTO_REJECTED, empty | 0 / - | 1,115/341/5,353 | reject | preserves frozen `AUTO_REJECTED` outcome |
| `823cfbe9` add>multiply>subtract>true_divide>add sm=0 | `FLAT:CONTIGUOUSx6:4c5b7886` | 0 | GENERATED 13; N=16..65,536 | 0.6141 / 0.0579 | 839/185/4,526 | reject | measured validation artifact has zero production frequency/execution |
| `943ddb58` add>multiply sm=0 | `OUTER_INNER:CONTIGUOUSx3:52c5ff1a` | 0 | GENERATED 9; N=16..65,536 | 0.7146 / 0.0428 | 939/300/9,229 | reject | zero-frequency PF4 validation seed; retained outside production generation |
| `9aa00e56` multiply>multiply>add sm=0 | `FLAT:CONTIGUOUSx4:e24c76d9` | 0 | GENERATED 13; N=16..65,536 | 0.2704 / 0.1094 | 611/140/3,647 | reject | measured validation artifact has zero production frequency/execution |
| `ae54de74` add>multiply sm=2 | none | 0 | GENERIC_FALLBACK 13; N=16..65,536 | 0.6127 / 0.0581 | 0/0/0 | reject | preserves structural `AUTO_REJECTED`; generic runtime envelope is unchanged |
| `b31365f2` add>multiply sm=4 | `FLAT:CONTIGUOUS,CONTIGUOUS,SCALAR:2adb9569` | 172,356 | unmeasured | 0 / - | 497/114/4,415 | reject | no exact specialization measurement |
| `be93e9a6` multiply>add sm=1 | `FLAT:SCALAR,CONTIGUOUS,CONTIGUOUS:2044ce95` | 50,844 | GENERATED 13; N=16..65,536 | 0.7941 / 0.0309 | 497/110/3,194 | **select** | exact `AUTO_SELECTED` artifact covers frozen-policy executions |
| `be93e9a6` multiply>add sm=1 | `FLAT:SCALAR,CONTIGUOUS,CONTIGUOUS:c618a37b` | 50,844 | unmeasured | 0 / - | 497/110/4,415 | reject | unmeasured duplicate; measured structural flat artifact supplies the frozen executions |
| `c79aa989` true_divide>add>multiply sm=10 | `OUTER_INNER:CONTIGUOUS,SCALAR,CONTIGUOUS,SCALAR:b22fb92a` | 102,859 | AUTO_REJECTED, empty | 0 / - | 1,114/337/5,344 | reject | preserves frozen `AUTO_REJECTED` outcome |
| `c79aa989` true_divide>add>multiply sm=10 | `FLAT:CONTIGUOUS,SCALAR,CONTIGUOUS,SCALAR:d4cd3d82` | 205,718 | unmeasured | 0 / - | 611/138/5,023 | reject | no exact specialization measurement |
| `edd3cd98` add>multiply>subtract>true_divide sm=0 | `FLAT:CONTIGUOUSx5:01664d40` | 0 | GENERATED 13; N=16..65,536 | 0.5245 / 0.0713 | 725/162/4,095 | reject | measured validation artifact has zero production frequency/execution |
| `f33adf67` add>multiply>subtract sm=0 | `FLAT:CONTIGUOUSx4:4bc4c5c1` | 0 | GENERATED 13; N=16..65,536 | 0.2192 / 0.1171 | 611/138/3,645 | reject | measured validation artifact has zero production frequency/execution |
| `fcd7cb47` add>multiply sm=4 | `OUTER_INNER:CONTIGUOUS,CONTIGUOUS,SCALAR:4520d699` | 3,008 | GENERATED 9; N=16..65,536 | 0.2334 / 0.1150 | 939/268/9,214 | reject | no frozen-profile count is authorized as GENERATED by this exact envelope |
| `ffa53a6c` multiply>add sm=1 | none | 0 | GENERIC_FALLBACK 4, ORIGINAL 9; N=16..65,536 | 0 / 0.2096 | 0/0/0 | reject | preserves structural `AUTO_REJECTED`; regressive runtime points remain ORIGINAL |

## PF9 result

```text
candidate identities audited:                 30
PF6 generated artifacts:                      28
selected production artifacts/kernels:         9
rejected PF6 artifacts:                       19
explicit rejected non-artifact candidates:     2

frozen dynamic opportunities:          6,505,405
expected generated executions:         2,698,694
expected generated dynamic coverage:      41.4839%

generated C / H / metadata:       16,534 / 4,311 / 169,233 bytes
generated object / text column:        13,984 / 3,762 bytes
DecisionDB inputs:                            212,138 bytes (unchanged)
```

Relative to PF6, production removes 19 kernels, 19,885 C bytes, 3,877 header
bytes, 116,345 metadata bytes, 12,120 object bytes, and 5,672 bytes from the
object `size` text column.  Expected generated execution coverage is unchanged
because every removed artifact had no distinct execution authorized by the
frozen production profile and runtime policies.

Canonical hashes:

```text
candidate-universe identity:
    dc7dbe6fe56fb7ee9d51299ed1a954de70c6dee28929f1c748974a1abe153848
production selection manifest:
    0c47c85fd321e6bed46dcf8551c97f78f2fd6ed8d3b2b7423b8ba3e0998a85c7
generated C:
    3f911134f7273efac61e225888b98ca8e09ab9dbdccadb1591ddf43fac6c4977
generated H:
    cd37427bdd20791909be00e88798c75515b0bc7ceb2fec532576347b3249c715
generated metadata:
    a316f5275832d5486e239f20b6446d6d5ad363048ced0bcd37c3a423943036c2
generated object:
    9e50246fc521bc454ba6ec6afaf993c9865bd26ac91d014e34f5a18332260adf
```

Validation at the PF9 checkpoint:

```text
selection regeneration:                 byte-for-byte identical
Generator tests:                         87 passed
NumPy generation/build-tool tests:       20 passed
installed NumPy CrossJIT tests:          32 passed
production-policy smoke:                 pass
runtime profitability DB content:        unchanged
schema/layout/semantic coverage:         unchanged
```

PF9 is a **PASS**.  PF10 performs the fresh-build and complete diagnostic
freeze; PF11/PF12 formal evaluation remains forbidden.
