# Engine shared-fate register

**Packet:** `WP-ENGINE-INDEPENDENCE.md`, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. **AS0 remains closed.**

**Status: EI0 COMPLETE — vocabulary frozen. EI1 COMPLETE — first register published, 10 entries.**

---

# EI0 — Frozen classification vocabulary

**Frozen 2026-08-09, before any authority was examined.** That ordering is the point of EI0: a
vocabulary chosen after looking at results can be shaped, consciously or not, to make the results
come out independent. Nothing below may be widened, narrowed or renamed during EI1–EI6; a
classification that does not fit is a **residual**, recorded as such.

## 0.1 Dependency classification

```text
DIRECT      the engine calls or reads the authority directly
INDIRECT    the engine consumes IR, metadata, normalized types, or generated structures
            produced by the authority
NONE        no relevant dependency found
UNKNOWN     insufficient evidence; resolve, or register as a residual
```

## 0.2 Differential visibility

```text
VISIBLE             a defect should affect one engine independently and create disagreement
PARTIALLY_VISIBLE   some defect classes diverge while others are inherited
INVISIBLE           the relevant engines inherit the same authority or output
UNKNOWN             visibility has not been established
```

## 0.3 Evidence independence

```text
SPEC_DERIVED                strongest
EXTERNALLY_DERIVED          strongest
HAND_AUTHORED               strongest when independently authored
CROSS_ENGINE_DERIVED        correlated unless an independent control exists
IMPLEMENTATION_GENERATED    correlated unless an independent control exists
SHARED_FIXTURE_GENERATOR    correlated unless an independent control exists
UNKNOWN
```

## 0.4 Authority category

```text
SHARED_PREDICATE        SHARED_TYPE_TABLE       SHARED_NORMALIZATION
SHARED_LOWERING         SHARED_IR               SHARED_PROVIDER_SCHEMA
SHARED_ABI_MAPPING      SHARED_ERROR_MAPPING    SHARED_TEST_ORACLE
RUSTC_ASSUMPTION        ENGINE_LOCAL
```

## Acceptance criteria, and how each is met

| Criterion | How EI0 meets it |
| --- | --- |
| classifications are frozen | Recorded here before any authority was examined; the register below was empty at the moment of freezing |
| `UNKNOWN` is not silently treated as independent | **Binding rule:** an `UNKNOWN` in any of the four axes propagates to the entry's risk as `UNKNOWN`, never as `NONE`, `VISIBLE` or independent. An entry carrying any `UNKNOWN` cannot be cited as evidence of independence in EI2 or EI4, and is a **residual** until resolved |
| direct and indirect dependencies are distinguished | Two separate values in 0.1, and EI1 records **both** per engine rather than collapsing to a boolean |
| evidence-source correlation is included | 0.3 is a required column of every EI1 entry, not an EI2-only concern — so correlation is visible at registration time rather than discovered during the audit |

## The rule EI0 exists to prevent being bent

> A shared authority whose defects are **INVISIBLE** to differential comparison, and whose only
> supporting evidence is **CROSS_ENGINE_DERIVED**, is **not** independently evidenced — regardless
> of how many engines agree.

Agreement between engines that inherit the same authority is not corroboration. Recording that as a
frozen rule now is what stops it being argued away against a specific entry later.

## Evidence invariant inherited from CD-392

> **No evidence mechanism may support a claim until its ability to distinguish success from failure
> has itself been demonstrated.**

Binding on every artefact this packet produces and on the AS8 work that consumes them.

---

# EI1 — Shared-fate register

**In progress.** Entries are appended below as authorities are identified. Each carries an `ESF-`
identifier which AS8 trials must reference; AS8 assigns its own trial IDs but **may not invent a
semantic classification independent of this register**.

| Field | Meaning |
| --- | --- |
| `id` | `ESF-<AREA>-<NNN>`, stable once assigned |
| `authority` | the shared thing: a predicate, table, normalization, lowering, IR, schema, mapping or oracle |
| `category` | one of 0.4 |
| `hir` / `mir` / `native` | dependency classification per engine, from 0.1 |
| `visibility` | from 0.2 |
| `evidence` | from 0.3 — the strongest source currently supporting it |
| `control` | an independent control, if one exists; otherwise `none` |

Ten authorities, each measured against the three engines rather than assumed. `hir` is
`src/interp.rs`, `mir` is `src/mir/interp.rs` and its supporting modules, `native` is
`src/backend/generated_rust/`.

## The finding, before the table

**Six of ten authorities are INVISIBLE to differential comparison across all three engines**, which
means three-engine agreement about them corroborates nothing:

```text
ESF-COPY-001   nominal Copy eligibility     one front-end computation, all three consume it
ESF-DROP-001   destructor eligibility       likewise
ESF-TRAP-001   trap categorisation          one enum, all three match on it
ESF-RES-001    HostResource typing          one MirTy carrier, all three consume it
ESF-TYPE-001   Unit/() canonicalisation     canonicalised once, before any engine sees it
ESF-TRAIT-001  Core trait contracts         one table, consumed through resolved calls
```

**Two more are INVISIBLE between MIR and native specifically** — the pair whose agreement the
differential suites most often cite:

```text
ESF-COPY-002   structural Copy over MirTy   the backend DELEGATES to TypeContext::is_copy
ESF-DROP-002   drop plan and order          the backend is an APPLICATION of mir::drop_plan
```

One is `PARTIALLY_VISIBLE` (`ESF-NUM-001`) and one is `UNKNOWN` (`ESF-PROV-001`).

```text
visibility   INVISIBLE 6 | INVISIBLE_MIR_NATIVE 2 | PARTIALLY_VISIBLE 1 | UNKNOWN 1
risk         critical 2  | high 4                 | medium 3           | UNKNOWN 1
```

Two of those three carry a documented history of exactly this defect: **CD-065** records that the
backend's `mir_ty_is_copy` "had been written out here identically" before being consolidated, and
**CD-062** records the same for destruction order. The consolidation was correct — duplicated rules
diverge — but it converts a *visible* disagreement into an *invisible* shared fate, and that trade
is what this register exists to make explicit rather than accidental.

## Register

| ID | Semantic fact | Category | hir | mir | native | rustc | Visibility | Evidence | Independent evidence | Residual | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ESF-COPY-001` | Which nominal types are `Copy`-eligible | `SHARED_PREDICATE` | DIRECT — `interp.rs:2031` calls `typecheck::copy_eligible_types` | INDIRECT — via the eligibility set threaded into `TypeContext` | INDIRECT — same set, via `TypeContext` | none | **INVISIBLE** | `SPEC_DERIVED` (03 Copy/Drop rules) + `HAND_AUTHORED` (`copy_canon_matrix`) | `copy_canon_matrix` enumerates from the checker's own arms — see residual | **The matrix is enumerated FROM the implementation, so it is `IMPLEMENTATION_GENERATED` for the eligibility question even though hand-authored.** No spec-derived control isolates a wrong eligibility set | **critical** |
| `ESF-COPY-002` | Structural `Copy` over a `MirTy` | `SHARED_PREDICATE` | NONE — the HIR engine classifies over `Ty`, not `MirTy` | DIRECT — `TypeContext::is_copy`, `mir::mir_ty_is_copy` | DIRECT — `emit_types::mir_ty_is_copy` is a one-line delegate to `types.is_copy` | none | **INVISIBLE (mir↔native)**, PARTIALLY_VISIBLE vs hir | `CROSS_ENGINE_DERIVED` | three-engine differential can still separate hir from the pair | mir/native agreement here is **not** independent; CD-065 consolidated a prior duplicate | **high** |
| `ESF-DROP-001` | Which nominals have a destructor | `SHARED_PREDICATE` | DIRECT — `interp.rs:2032` | INDIRECT — via lowering | INDIRECT — via lowering | none | **INVISIBLE** | `SPEC_DERIVED` (03/05) | `borrowck.rs` consumes the same authority at three sites — a fourth consumer, not a control | no control derives destructor eligibility independently of the front end | **critical** |
| `ESF-DROP-002` | Drop plan: order, array direction, variant payloads | `SHARED_LOWERING` | NONE — the HIR engine has its own destruction walk | DIRECT — `mir::drop_plan::plan_for` | DIRECT — `emit_bodies` calls `drop_plan::plan_for`; CD-062 records it as an APPLICATION of the canonical plan | none | **INVISIBLE (mir↔native)** | `CROSS_ENGINE_DERIVED` | hir's independent walk is the only control | if the canonical plan is wrong, mir and native agree while being wrong together | **high** |
| `ESF-TRAP-001` | Trap categorisation | `SHARED_ERROR_MAPPING` | DIRECT — `mir::TrapCategory`, 32 uses in `interp.rs` | DIRECT | DIRECT — `emit_program`, `emit_bodies` | none | **INVISIBLE** | `SPEC_DERIVED` (traps always trap, in every build mode) | conformance fixtures assert trap *occurrence*; category naming is inherited | a mis-categorised trap is invisible to all three engines | **high** |
| `ESF-RES-001` | Host resource typing and identity | `SHARED_TYPE_TABLE` | DIRECT — `interp.rs` | DIRECT — `HostResourceNominal`/`HostResourceTy` in `mir/mod.rs` | DIRECT — `emit_types`, `emit_bodies` | none | **INVISIBLE** | `EXTERNALLY_DERIVED` — provider loopback tests exercise real peers | `C7.8 provider metadata/unit/resource/loopback` on three platforms | resource *typing* is shared even though behaviour is externally tested | **medium** |
| `ESF-PROV-001` | Provider call signatures | `SHARED_PROVIDER_SCHEMA` | UNKNOWN — not measured | DIRECT — `mir::provider_sig::signature` | INDIRECT — `emit_provider` | `RUSTC_ASSUMPTION` on the ABI boundary — EI3 | UNKNOWN | `EXTERNALLY_DERIVED` (loopback) | `mir/verify.rs` is the only in-tree consumer of `provider_sig::signature` | **hir dependency UNMEASURED — per EI0 this is UNKNOWN, not NONE, and cannot be cited as independence** | **UNKNOWN** |
| `ESF-TYPE-001` | `Unit` and `()` are one type (TYPE-PRIM-001) | `SHARED_NORMALIZATION` | INDIRECT — consumes the canonicalised `Ty` | INDIRECT — `mir/lower.rs:1719` records MIR has one empty-tuple spelling | INDIRECT | none | **INVISIBLE** | `SPEC_DERIVED` (03 TYPE-PRIM-001) | spec fixtures | canonicalisation happens once, in `typecheck::types::unit_or_tuple`; no engine can disagree | **medium** |
| `ESF-NUM-001` | Integer literal range and suffix conversion | `SHARED_PREDICATE` | INDIRECT — through the checker's published types | INDIRECT | INDIRECT | `RUSTC_ASSUMPTION` on generated integer types — EI3 | PARTIALLY_VISIBLE | `SPEC_DERIVED` (03 numeric semantics) | `literal::primitive_int_range_contains` has two consumers, both front-end | overflow *behaviour* is engine-local and visible; the *range table* is shared | **medium** |
| `ESF-TRAIT-001` | Core trait contracts: receiver, params, return | `SHARED_TYPE_TABLE` | INDIRECT — through resolved calls | INDIRECT | INDIRECT | none | **INVISIBLE** | `SPEC_DERIVED` (06) + `HAND_AUTHORED` | `copy_canon_matrix` enumerates from `core_method_signature`'s arms | same residual as `ESF-COPY-001`: the matrix is derived from the implementation it checks | **high** |

## Residual summary

```text
ESF-PROV-001   hir dependency UNMEASURED -> UNKNOWN, risk UNKNOWN. Blocks any independence claim
               about provider signatures until resolved. EI2 must close this.
ESF-COPY-001   copy_canon_matrix is enumerated FROM the checker's own signature arms. It is a
ESF-TRAIT-001  strong control for DRIFT and a weak one for a WRONG RULE, because a wrong rule is
               enumerated faithfully. Both need a spec-derived control or the residual stands.
ESF-COPY-002   mir<->native agreement is inherited, not corroborating. The hir engine is the only
ESF-DROP-002   independent control for these two, and for ESF-DROP-002 it walks a different
               structure — so its agreement is weaker than a matching implementation would be.
```

**Eight of ten entries are INVISIBLE to some engine pair, and six of those to all three.** That is
the shared-fate result EI1 exists to produce, and it is the input EI4 ranks and EI5
turns into mutation targets — a mutation in an INVISIBLE authority is exactly the mutation a
three-engine differential cannot catch.

## Method note

Every dependency cell above was established by reading the call graph, not by assuming from module
names. Two assumptions were tested and refuted in the process: the native backend appeared to have
its own `mir_ty_is_copy` (different signature, same name) but delegates in one line; and the drop
plan appeared duplicated in `emit_bodies` but is a documented application of `mir::drop_plan`. Both
would have been recorded as `PARTIALLY_VISIBLE` had the file names been trusted, understating the
shared fate.

`ESF-PROV-001`'s `hir` cell is left `UNKNOWN` rather than guessed, per EI0's binding rule that
`UNKNOWN` never resolves silently to `NONE`.
