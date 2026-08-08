# AS8-DA — duplicated authorities

**Owner ruling, 2026-08-09.** These findings are **orthogonal to the shared-fate register and do
not enter EI0's category vocabulary, which stays frozen.**

> EI0's categories answer **what kind of semantic authority** something is — predicate, type table,
> normalisation, lowering, ABI mapping. Duplication answers a different question: **how many
> implementations of that semantic fact exist, and what relationship do they have.** A
> `DUPLICATED_AUTHORITY` category would mix two dimensions and break the taxonomy.

Shared fate and uncontrolled duplication are **opposite failure modes**:

```text
SHARED FATE          one implementation, many consumers   agreement proves nothing
DUPLICATION          many implementations, one rule       copies drift, and the second
                                                          copy is nobody's control
```

## The correction that shapes this file

The first instinct on finding five duplicates was to consolidate them. **That is wrong by default,
and the owner ruling says why: a verifier can derive its value precisely from implementing the same
rule independently.** Replacing both copies with one shared helper removes drift and *creates
shared fate* — the verifier would then be unable to detect a wrong shared predicate. It converts a
detectable problem into an undetectable one, which is the exact trade CD-065 records for
`mir_ty_is_copy` and the reason the register exists.

So duplication is not a defect to be removed on sight. It is a **relationship to be classified**,
and the classifier is paired one-sided mutation:

```text
mutate implementation A only  ->  killed?   YES   independent redundancy is doing real work: KEEP
                                            NO    A is unguarded
mutate implementation B only  ->  killed?   YES   useful cross-check
                                            NO    B can drift silently

BOTH SURVIVE  ->  architectural residual: one authority, or an explicit cross-check
BOTH KILLED   ->  the strongest outcome. Two independently implemented tables that check
                  each other are a BETTER design than one shared helper
```

## Register

| ID | Semantic rule | Implementation A | Implementation B | Intended relationship | One-sided mutation A | One-sided mutation B | External / control evidence | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `AS8-DA-001` | Which primitives are integers (`ESF-TYPE-001` family; 03 numeric semantics) | `typecheck::types::is_integer` — 14 call sites | `typecheck::types::is_integer_primitive` — 1 call site | **None intended.** Both `pub(super)`, same module, byte-identical, same bottom layer of the AS7 DAG. No independence value is available at one layer of one module | `AS8-MUT-018` | `AS8-MUT-024` | — | **Consolidate AFTER AS8**, per owner ruling, unless a trial exposes a live defect. Not a DEV: identical copies cannot disagree |
| `AS8-DA-002` | Which `RuntimeFn`s are Vec operations | `mir::interp::is_vec_runtime` | `mir::verify::is_vec_runtime_fn` | **Plausibly deliberate.** `verify.rs` exists to CHECK the lowering `interp.rs` executes; an independent table is what lets it disagree | `AS8-MUT-026` | `AS8-MUT-027` | `mir_verify`, `mir_differential` | **PENDING TRIAL — do not consolidate** |
| `AS8-DA-003` | Which `RuntimeFn`s are Box operations | `mir::interp::is_box_runtime` | `mir::verify::is_box_runtime_fn` | as `AS8-DA-002` | `AS8-MUT-028` | `AS8-MUT-029` | as above | **PENDING TRIAL — do not consolidate** |
| `AS8-DA-004` | Which `RuntimeFn`s are Slice operations | `mir::interp::is_slice_runtime` | `mir::verify::is_slice_runtime_fn` | as `AS8-DA-002` | `AS8-MUT-030` | `AS8-MUT-031` | as above | **PENDING TRIAL — do not consolidate** |
| `AS8-DA-006` | Does this type need drop glue (`ESF-DROP-002`; A11 §5, 05) | `mir::drop_rule::requires_drop_glue_with`, reached from `lower::ty_requires_drop_glue` — **precise** | `mir::verify::may_need_drop` — **deliberately conservative** | **Deliberate, asymmetric, and documented.** The verifier over-approximates on purpose so it can reject a missing drop without reimplementing the precise rule. AS4 added `may_need_drop_for_inventory`, a test-only window, expressly to measure the conservative rule against the precise one | — | `AS8-MUT-037` | AS4 drop-rule matrix; `a11_host_resource`, `c788_resource_lifecycle` | **KEEP. The positive exemplar.** Two implementations, an explicit intended relationship, and a measurement harness for the gap between them |
| `AS8-DA-005` | `ScalarTy` → STARK primitive spelling | `provider_synth::scalar_src` | `provider_derive::scalar_name` | **Unclear.** Two stages of provider generation naming the same mapping; neither is a check on the other | `AS8-MUT-032` | `AS8-MUT-033` | `a10_provider_bind`, `c788_starkc_build` | **PENDING TRIAL** |

## `AS8-DA-006` is the proof that this is a lower bound

`AS8-DA-006` was **not found by the scanner.** The two implementations have different names and
different bodies — one precise, one conservative — so a textual matcher cannot see them. It was
found by reading `may_need_drop`, whose own source comment says so outright:

> *"the SIXTH `MirTy` catch-all to swallow this variant — and the second copy of 'does this need
> dropping', after `lower::ty_requires_drop_glue`. **Two implementations of one rule, each corrected
> separately**: lowering stopped emitting the `Drop`, and when that was fixed the verifier rejected
> the `Drop` it now emitted."*

That is a recorded instance of the two copies **actually disagreeing**, in production, historically.
It is also the case that most argues for the owner ruling against reflexive consolidation: the pair
is deliberate, the asymmetry is the point, and AS4 built a measurement window for the gap. The right
disposition is KEEP, and it would have been invisible to a policy of "consolidate what the scanner
finds".

## This is a lower bound, not an inventory

`starkc/scripts/as8-duplicate-authorities.py` matches **byte-identical bodies after whitespace
normalisation**, over 2,532 `fn` definitions. A rule reimplemented with different names, a
different match order, or an equivalent-but-not-identical expression is **invisible to it**. A
clean report is not evidence of no duplication, and this table is not an inventory of the
compiler's duplicated rules — it is the subset a textual matcher can see.

The scanner was also **wrong on its first run**, in a way worth recording because it is the same
class of error the whole packet has been about: it reported sixteen "identical" bodies inside
`extensions/tensor/check.rs` and four in `mir/drop_rule.rs`. Both were artefacts — a trait method
DECLARATION has no body, so a naive brace matcher runs on to the next `{` in the file. Fixed by
stopping signature scanning at a `;` at paren depth zero. **A measurement is not evidence until it
has been checked against something it should not find.**

## Disposition rules (owner ruling, 2026-08-09)

Duplicated implementations **do not earn DEV numbers merely for existing.** They are
architecture-assurance candidates.

```text
one-sided trial shows a copy has ALREADY DRIFTED   ->  DEV immediately
copies identical and correct but unguarded         ->  architecture debt / follow-up
copies identical, one-sided mutation reliably      ->  KEEP BOTH; record the redundancy as
    killed                                             a control, not as debt
```
