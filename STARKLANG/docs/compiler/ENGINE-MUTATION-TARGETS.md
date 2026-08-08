# Ranked mutation targets

**Packet:** `WP-ENGINE-INDEPENDENCE.md` EI5, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. Inputs: EI1 register, EI2 evidence audit, EI3 rustc inventory, EI4 risk profiles.

**Status: EI5 COMPLETE. AS8's mutation lane is unblocked.**

AS8 assigns its own trial IDs (`AS8-MUT-NNN`) and **must reference the `ESF-`/`RA-`/`EV-`
identifiers below**. It may not introduce a semantic classification independent of the register.

---

## Batch 0 — the harness self-test, which runs before any real batch

**CD-392's evidence invariant applies to the mutation harness itself.** Before any result below is
believed, the harness must be shown to distinguish success from failure **in both directions**:

| | Mutation | Required outcome | Why this direction matters |
| --- | --- | --- | --- |
| `MUT-SELFTEST-LIVE` | `types::is_integer` — invert one arm so `Int32` reports non-integer | **KILLED** | Proves the harness detects a real semantic disturbance |
| `MUT-SELFTEST-NOOP` | rename a local variable in `mir::drop_plan::plan_for`; reorder two independent `let` bindings | **SURVIVES** | Proves the harness is not merely failing whenever source changes |

**If `MUT-SELFTEST-NOOP` is killed, every kill in this document is uninterpretable** — the suite is
detecting edits, not defects. That failure looks identical to success in a kill-rate table, which
is why it is Batch 0 and not an appendix.

## The prediction this document makes

EI4 states it and EI5 tests it:

> A mutation in a front-end authority that all three engines inherit **should survive every
> differential suite in the tree.**

Batches 1 and 2 are designed so that a *survivor is the expected result*. A kill there is either
good news about an unrecognised control, or evidence the harness is wrong — and Batch 0 is what
tells the two apart.

## Ranked batches

Ranking follows the packet's priority rule; tags are the frozen four.

### Batch 1 — high-risk invisible shared authorities *(priority 1)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `ESF-COPY-001` `typecheck::traits::copy_eligible_types` | `SHARED_AUTHORITY` | Admit a nominal that also has a `Drop` impl (Copy+Drop is forbidden by 03); separately, drop the all-fields-Copy requirement | `three_engine_differential`, `mir_differential`, `copy_canon_matrix`, `c6-corpus` | **None expected.** `copy_canon_matrix` is enumerated from the implementation (EI2 `EV-COPY-MATRIX`) | **Immediate AS8 residual and a DEV candidate.** A `critical` authority with no control that survives mutation is a test gap, and the gap is the finding |
| `ESF-DROP-001` `typecheck::traits::nominals_with_destructor` | `SHARED_AUTHORITY` | Omit one nominal carrying a destructor | `three_engine_differential`, `operand_move_inventory`, `dev146_resource_borrow_weakening`, `c6-corpus` | **None.** EI1 records no control; `borrowck` is a fourth consumer, not a control | As above. `critical` |

### Batch 2 — shared type and representation predicates *(priority 2)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `ESF-COPY-002` `mir::TypeContext::is_copy` | `SHARED_AUTHORITY` | `MirTy::Ref { mutable: true }` reports `Copy` | `mir_differential`, `three_engine_differential`, `copy_canon_matrix`, `dev146_*` | **HIR engine** — it classifies over `Ty`, not `MirTy`, so it should disagree | If it survives, the hir↔mir differential is not exercising the case; residual against `EV-DIFF-*` coverage rather than against the rule |
| `ESF-TYPE-001` `typecheck::types::unit_or_tuple` | `SHARED_AUTHORITY` | Return `Ty::Tuple(vec![])` instead of `Primitive::Unit` — the exact pre-TYPE-PRIM-001 defect | `spec fixture conformance`, `conformance`, `three_engine_differential` | **`EV-SPEC-FIXTURES`** — spec-derived, the strongest control in the tree | A survivor would mean the spec fixtures do not cover TYPE-PRIM-001's own rule, which would be a notable gap given the rule has a fixture history |

### Batch 3 — generic compatibility and trait tables *(priority 3)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `ESF-TRAIT-001` `typecheck::traits::core_trait_contract` | `SHARED_AUTHORITY` | Change one Core trait method's declared receiver (`&self` → `self`) and, separately, its return type | `copy_canon_matrix`, `conformance`, `gate4a_prelude_traits`, `three_engine_differential` | **None expected** — `copy_canon_matrix` enumerates *from* `core_method_signature` | Residual + DEV candidate. `high` |

### Batch 4 — provider and resource mappings *(priority 4)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `ESF-PROV-001` `mir::provider_sig::signature` | `SHARED_AUTHORITY` | Swap two parameter positions in one provider signature | `C7.8 provider metadata/unit/resource/loopback`, `mir/verify` | **`EV-PROVIDER-LOOP`** — external, live peers. Genuinely independent | **Two engines only** (EI2-R2). A kill here is real evidence; a survivor means the loopback suite does not cover that signature |
| `ESF-RES-001` `mir::HostResourceNominal` | `SHARED_AUTHORITY` | Classify one host resource as `Copy`-eligible — the A11/CD-234 shape the code comments warn about explicitly | `dev146_resource_borrow_weakening`, `c788_resource_lifecycle`, `a10_provider_resource`, `a11_host_resource` | External loopback + the resource-lifecycle suites | The code carries an explicit warning that a wildcard here classified a resource `Copy` with silent consequences — a survivor would mean that warning is unenforced |

### Batch 5 — canonicalisation helpers *(priority 5)*

Covered by `ESF-TYPE-001` in Batch 2. `types::is_integer` and `types::strip_ref` are secondary
targets with the same shape and lower risk.

### Batch 6 — error-category mappings *(priority 6)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `ESF-TRAP-001` `mir::TrapCategory` assignment | `SHARED_AUTHORITY` | Assign the *wrong existing category* at one trap site — overflow reported as division-by-zero | `three_engine_differential` (`oracle_category`), `c6-corpus` (`expected_trap_category`), `cd139_float_division` | **None.** EI2-R3: the same enum is the implementation's vocabulary, the differential's expectation and the corpus manifest's | Expected survivor. Confirms EI2-R3 and becomes an immediate residual |

### Batch 7 — rustc-sensitive lowering decisions *(priority 7)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `RA-OVERFLOW` `emit_checked_expr` | `BACKEND_ASSUMPTION` | Emit the unchecked operator for one arithmetic op — i.e. re-delegate to rustc | `three_engine_differential`, `c6-corpus` trap cases, `gate5_*` | **HIR + MIR engines** — they trap independently of the generated Rust | Should be killed. A survivor means release-profile trapping rests on `overflow-checks` after all, contradicting EI3's finding that it is "recorded rather than relied upon" |
| `RA-SHIFT` the shift check | `BACKEND_ASSUMPTION` | Use Rust's `checked_shl` directly — which validates only the shift count | same, plus any shift-specific fixtures | HIR + MIR | This is the documented divergence; a survivor means no test distinguishes STARK's shift rule from Rust's |
| `RA-DROP` `drop_plan::array_order` | `BACKEND_ASSUMPTION` | Reverse array destruction order | `three_engine_differential`, drop-order fixtures | **HIR engine only** — it walks a different structure | A survivor means mir↔native inheritance (EI1 `ESF-DROP-002`) is unguarded by the one control that exists |

### Batch 8 — correlated evidence generators *(priority 8)*

| Trial target | Tag | Recommended mutation | Selected tests | Expected independent control | Survivor consequence |
| --- | --- | --- | --- | --- | --- |
| `EV-COPY-MATRIX` | `EVIDENCE_SHARED` | Mutate the *implementation* the matrix enumerates from, leaving the matrix untouched | `copy_canon_matrix` | — | Tests EI2's claim directly: if the matrix still passes, it is a transcription, not a control. **This is a mutation of the evidence, not of the compiler** |
| `EV-CORPUS-C6` generated corpus | `EVIDENCE_SHARED` | Mutate an authority the corpus's `expected_trap_category` depends on | `c6_generated_corpus`, `c6_corpus_manifest` | manifest and generator hashes | Distinguishes "the corpus detects semantic change" from "the corpus detects corpus change" |

## Reporting requirements

Per the packet's acceptance criteria, and binding on AS8:

```text
kill rates reported SEPARATELY BY TAG — never pooled
    SHARED_AUTHORITY | ENGINE_LOCAL | BACKEND_ASSUMPTION | EVIDENCE_SHARED

a pooled kill rate is misleading here BY CONSTRUCTION: SHARED_AUTHORITY mutations are PREDICTED
to survive, so mixing them with BACKEND_ASSUMPTION mutations (predicted killed) produces a
middling number that describes neither.

every shared-authority survivor -> immediate AS8 residual or DEV candidate, not a footnote

every KILL on a SHARED_AUTHORITY target -> confirm the killing test is INDEPENDENTLY DERIVED
before crediting it. The packet is explicit that a killed shared-authority mutation is not
automatically sufficient, and EI2 found that most in-tree semantic evidence is CROSS_ENGINE_DERIVED
```

## Trial record format

```text
AS8-MUT-NNN
target authority   ESF-COPY-001
EI5 rank           batch 1, priority 1
tag                SHARED_AUTHORITY
independent control  none expected (EV-COPY-MATRIX is IMPLEMENTATION_GENERATED)
mutation           <exact source alteration>
expected killer    <suite/test, or "none — survivor predicted">
actual result      KILLED | SURVIVED
killer independence  SPEC_DERIVED | EXTERNALLY_DERIVED | CROSS_ENGINE_DERIVED | ...
consequence        none | residual | DEV-NNN candidate
```

`killer independence` is a required field, not optional: a kill credited to
`CROSS_ENGINE_DERIVED` evidence is a weaker result than a kill credited to `SPEC_DERIVED`, and
recording them identically would erase the distinction this whole packet exists to draw.
