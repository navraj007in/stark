# Engine shared-fate register

**Packet:** `WP-ENGINE-INDEPENDENCE.md`, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. **AS0 remains closed.**

**RECONCILED WITH AC4, 2026-08-12** — see the AC4 reconciliation section before the method note.
**Eleven entries became sixteen**; one binding rule added; EI0's frozen vocabulary unchanged. No
existing row's visibility classification changed.

**Status: EI0 COMPLETE — vocabulary frozen. EI1 COMPLETE — **11 entries** after the `ESF-TRAP-001`
split. EI2 COMPLETE — see `ENGINE-EVIDENCE-INDEPENDENCE.md`; `ESF-PROV-001`'s `UNKNOWN` is closed
there and this file's row is superseded by the JSON register's updated entry.**

> **AMENDED 2026-08-09 by AS8 mutation trials — `AS8-MUTATION-FINDINGS.md`.** Two rows were wrong
> and are corrected below, both by measurement rather than re-reading:
> **`ESF-TRAP-001` was one entry and is two** (vocabulary is invisible; per-site assignment is
> not — MUT-007 was caught by the HIR oracle), and **`ESF-COPY-001` has an independent control**
> the audit missed, `c61f_structural_copy` (MUT-009/010/011 all killed by it). The original rows
> are preserved struck through in the JSON register's history field.

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

**Six of eleven authorities are INVISIBLE to differential comparison across all three engines**, which
means three-engine agreement about them corroborates nothing:

```text
ESF-COPY-001   nominal Copy eligibility     one front-end computation, all three consume it
ESF-DROP-001   destructor eligibility       likewise
ESF-TRAP-001a  trap category VOCABULARY     one enum, all three match on it
               (ESF-TRAP-001b, per-site ASSIGNMENT, is PARTIALLY_VISIBLE — AS8)
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
BEFORE AS8   visibility   INVISIBLE 6 | INVISIBLE_MIR_NATIVE 2 | PARTIALLY_VISIBLE 1 | UNKNOWN 1
             risk         critical 2  | high 4                 | medium 3           | UNKNOWN 1

AFTER  AS8   visibility   INVISIBLE 6 | INVISIBLE_MIR_NATIVE 3 | PARTIALLY_VISIBLE 2
             risk         critical 1  | high 7                 | medium 3

             11 entries. Counts are DERIVED FROM engine-shared-fate.json, which the header
             already declares authoritative over this file's ESF-PROV-001 row — EI2 closed its
             UNKNOWN there and the prose was never updated, so the two disagreed until AS8.
             ESF-TRAP-001 became 001a + 001b
             ESF-COPY-001  critical -> high    a measured control (MUT-009/010/011)
             ESF-TRAP-001b high     -> medium  a measured oracle  (MUT-007)
             ESF-TYPE-001  medium   -> HIGH    its recorded control does not control it
                                               (MUT-013 survived with the fixtures selected)
```

Two of those three carry a documented history of exactly this defect: **CD-065** records that the
backend's `mir_ty_is_copy` "had been written out here identically" before being consolidated, and
**CD-062** records the same for destruction order. The consolidation was correct — duplicated rules
diverge — but it converts a *visible* disagreement into an *invisible* shared fate, and that trade
is what this register exists to make explicit rather than accidental.

## Register

| ID | Semantic fact | Category | hir | mir | native | rustc | Visibility | Evidence | Independent evidence | Residual | Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ESF-COPY-001` | Which nominal types are `Copy`-eligible | `SHARED_PREDICATE` | DIRECT — `interp.rs:2031` calls `typecheck::copy_eligible_types` | INDIRECT — via the eligibility set threaded into `TypeContext` | INDIRECT — same set, via `TypeContext` | none | **INVISIBLE to the engines** — and that is not the whole picture: the control is a FRONT-END test, not an engine | `SPEC_DERIVED` (03 Copy/Drop rules) + **`HAND_AUTHORED` (`c61f_structural_copy`, 13 tests)** | **`c61f_structural_copy` IS an independent control** — it pins the negative surface by behaviour (reuse after move is E0100), not by enumerating the checker's arms. MUT-009/010/011 each killed by it, one of them by a single named test | `copy_canon_matrix` remains `IMPLEMENTATION_GENERATED` and is a DRIFT DETECTOR, not evidence for the rule (MUT-003 survived it). **AS8-R1: a wrong Copy rule with no drop consequence is invisible to every DIFFERENTIAL suite** — MUT-005 and MUT-006 survived with zero killers | **high** (was critical) |
| `ESF-COPY-002` | Structural `Copy` over a `MirTy` | `SHARED_PREDICATE` | NONE — the HIR engine classifies over `Ty`, not `MirTy` | DIRECT — `TypeContext::is_copy`, `mir::mir_ty_is_copy` | DIRECT — `emit_types::mir_ty_is_copy` is a one-line delegate to `types.is_copy` | none | **INVISIBLE (mir↔native)**, PARTIALLY_VISIBLE vs hir | `CROSS_ENGINE_DERIVED` | ~~three-engine differential can still separate hir from the pair~~ **AS8-R6: IN PRINCIPLE, NOT IN PRACTICE.** AS8-MUT-012 made `&mut` report `Copy` over `MirTy` and survived `mir_differential`, `three_engine_differential` and `c61f_structural_copy` with zero killers — no case duplicates a `&mut` and observes the consequence | mir/native agreement here is **not** independent; CD-065 consolidated a prior duplicate. The hir control is unexercised, so it is a COVERAGE gap, not an authority gap | **high** |
| `ESF-DROP-001` | Which nominals have a destructor | `SHARED_PREDICATE` | DIRECT — `interp.rs:2032` | INDIRECT — via lowering | INDIRECT — via lowering | none | **INVISIBLE** | `SPEC_DERIVED` (03/05) | `borrowck.rs` consumes the same authority at three sites — a fourth consumer, not a control | no control derives destructor eligibility independently of the front end | **critical** |
| `ESF-DROP-002` | Drop plan: order, array direction, variant payloads | `SHARED_LOWERING` | NONE — the HIR engine has its own destruction walk | DIRECT — `mir::drop_plan::plan_for` | DIRECT — `emit_bodies` calls `drop_plan::plan_for`; CD-062 records it as an APPLICATION of the canonical plan | none | **INVISIBLE (mir↔native)** | `CROSS_ENGINE_DERIVED` | hir's independent walk is the only control | if the canonical plan is wrong, mir and native agree while being wrong together | **high** |
| `ESF-TRAP-001a` | Trap category **vocabulary** — the `TrapCategory` enum itself | `SHARED_ERROR_MAPPING` | DIRECT | DIRECT | DIRECT | none | **INVISIBLE** | `SPEC_DERIVED` (traps always trap, in every build mode) | **none, and none is constructible** | If the enum names the wrong concept or omits one, every engine AND the corpus manifest are wrong together. **This cannot be posed as a source mutation at all** (MUT-008 is the honest no-op that marks the boundary). AS8-R2 | **high** |
| `ESF-TRAP-001b` | Trap category **assignment** at each trap site | `SHARED_ERROR_MAPPING` | DIRECT — **28 assignment sites in `interp.rs`, all 10 categories** | DIRECT — **30 sites across `mir/lower.rs` + `mir/interp.rs`, all 10 categories** | DIRECT — 3 sites; the remainder inherited from the runtime | none | **PARTIALLY_VISIBLE** | `CROSS_ENGINE_DERIVED` — `oracle_category` | **the HIR oracle is a real control here.** The same operation is categorised twice, INDEPENDENTLY, in two files. MUT-007 changed division-by-zero on the MIR path only and was killed by 4 tests: *"MIR IntegerOverflow vs oracle message 'division by zero'"* | A categorisation changed IDENTICALLY in both files is still invisible; only one-sided error is caught | **medium** (was high) |
| `ESF-RES-001` | Host resource typing and identity | `SHARED_TYPE_TABLE` | DIRECT — `interp.rs` | DIRECT — `HostResourceNominal`/`HostResourceTy` in `mir/mod.rs` | DIRECT — `emit_types`, `emit_bodies` | none | **INVISIBLE** | `EXTERNALLY_DERIVED` — provider loopback tests exercise real peers | `C7.8 provider metadata/unit/resource/loopback` on three platforms | resource *typing* is shared even though behaviour is externally tested | **medium** |
| `ESF-PROV-001` | Provider call signatures | `SHARED_PROVIDER_SCHEMA` | UNKNOWN — not measured | DIRECT — `mir::provider_sig::signature` | INDIRECT — `emit_provider` | `RUSTC_ASSUMPTION` on the ABI boundary — EI3 | UNKNOWN | `EXTERNALLY_DERIVED` (loopback) | `mir/verify.rs` is the only in-tree consumer of `provider_sig::signature` | **hir dependency UNMEASURED — per EI0 this is UNKNOWN, not NONE, and cannot be cited as independence** | **UNKNOWN** |
| `ESF-TYPE-001` | `Unit` and `()` are one type (TYPE-PRIM-001) | `SHARED_NORMALIZATION` | INDIRECT — consumes the canonicalised `Ty` | INDIRECT — `mir/lower.rs:1719` records MIR has one empty-tuple spelling | INDIRECT | none | **INVISIBLE** | `SPEC_DERIVED` (03 TYPE-PRIM-001) | ~~spec fixtures~~ **NONE — AS8-R5, measured.** AS8-MUT-013 reverted the canonicalisation with `conformance` (which runs the spec fixtures) in the selection, and it passed. The fixtures classify by what the front end ACCEPTS AND REJECTS; this rule is about type IDENTITY and breaking it makes no program fail to compile | canonicalisation happens once, in `typecheck::types::unit_or_tuple`; no engine can disagree, and no fixture does either | **high** (was medium) |
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

> **Counts in this paragraph are EI1's, over its own eleven entries, and are left as written.** AC4
> added five: three are INVISIBLE to some engine pair, one is PARTIALLY_VISIBLE
> (`ESF-LOWER-001` — hir is an independent oracle) and one is `ENGINE_LOCAL` and outside this count
> entirely (`ESF-VERIFY-001`). **Computed from the JSON, 12 of 16 entries are now INVISIBLE to some
> engine pair.**
>
> **A PRE-EXISTING DIVERGENCE, found while computing that and NOT resolved here.** The JSON gives
> `ESF-PROV-001` visibility `INVISIBLE_MIR_NATIVE`; the prose row below gives it `UNKNOWN`. That one
> cell is the whole difference between this paragraph's *"eight of eleven"* and the JSON's nine.
>
> It is left alone deliberately. AC4's reconciliation pass is **reconcile, not improve**, and this
> is not AC4's finding to settle: EI2 recorded `ESF-PROV-001`'s hir dependency as UNMEASURED, and
> EI0's binding rule says `UNKNOWN` never resolves silently — including in the direction that would
> make the register look more complete. **Whoever settles it should measure the hir cell, not pick
> the more convenient of two existing answers.**

**Eight of eleven entries are INVISIBLE to some engine pair, and six of those to all three.** That is
the shared-fate result EI1 exists to produce, and it is the input EI4 ranks and EI5
turns into mutation targets — a mutation in an INVISIBLE authority is exactly the mutation a
three-engine differential cannot catch.

## AC4 reconciliation (WP-ARCH-CLOSE, 2026-08-12)

**`engine-shared-fate.json` was updated first and is authoritative; this section reflects it.** The
prose register already states that the JSON wins where they disagree, and allowing a fresh
divergence here would repeat the problem the packet exists to eliminate.

**EI0's frozen vocabulary is UNCHANGED.** AC4 did not show the vocabulary was wrong — it showed the
**inventory** and the **evidence attached to it** were incomplete. Five entries are added using the
existing terms; nothing is widened, narrowed or renamed.

**Reconcile, not improve.** No test or semantic repair was made in this pass. The four open gaps
(F3–F6) are recorded in the state AC4 found them, deliberately, so the register describes the
evidence that exists rather than waiting for it to be made green.

### Entries added — 11 to 16

| id | fact | category | visibility | residual |
| --- | --- | --- | --- | --- |
| `ESF-TRAIT-002` | Whether a type **satisfies a bound at all** — distinct from `ESF-TRAIT-001`, which is the Core trait *signature* contract (receiver/params/return) | `SHARED_PREDICATE` | INVISIBLE | **AC4-F3 — REPAIRED 2026-08-12.** Was: three of five live arms (`Ty::Ref` forwarding, `Ty::Core` incl. the Iterator list, `Ty::Param` discharge) had **no executing test**, so arm-level mutations survived as *unreachable* rather than undetected. `ac4_bound_arms` (8 cases) now executes all five and kills all four mutations. Narrower residual remains: evidence is `HAND_AUTHORED` and the nine-bound × five-arm matrix is not exhaustively enumerated. **high → medium** |
| `ESF-DROP-003` | Whether destroying a value of a given `MirTy` **runs anything** | `SHARED_PREDICATE` | INVISIBLE mir↔native; potentially visible against hir | **AC4-F4.** Struct-recursion, `HostResource` and tuple arms all mutation-killed; a **built-in owning type** killed only incidentally. The comparator observes drops by a frame a case emits from its own `Drop` impl, and a built-in has none — the event is **structurally unobservable**, not merely inherited. **PARTIALLY REPAIRED 2026-08-12**: `ac4_builtin_destruction` controls the lowering DECISION (a String local, and a String-owning struct, must each produce a Drop). The CONSEQUENCE — an actual leak — is still unobservable: the Miri lane runs with `-Zmiri-ignore-leaks`, and a leak harness around generated binaries is a work packet. **high → medium** |
| `ESF-VERIFY-001` | MIR verification rules — **negative** enforcement | `ENGINE_LOCAL` | N/A — inventories assurance, not shared fate | **AC4-F5 — REPAIRED 2026-08-12.** Was: MIR-0035 mutated and survived, MIR-0029/0037 census-only. All 36 rule ids are now named by a test and the three gaps have malformed-MIR cases, each mutation-verified to die. Structural residual remains: a rule's positive path is always green, so **every new rule needs a malformed-body case at introduction** or the gap reopens silently. **medium → low** |
| `ESF-RES-002` | Resource **close/release selection** | `SHARED_LOWERING` | INVISIBLE mir↔native | **AC4-F6.** Real control exists and is required CI — live TLS acquire/use/release. starkc's own suite reaches `select_closes` **zero** times. Residual is **feedback latency, not absence of evidence**. **medium** |
| `ESF-LOWER-001` | Observable **evaluation order** introduced by lowering (CD-007) | `SHARED_LOWERING` | **PARTIALLY_VISIBLE** | **AC4-F7, RESOLVED.** Control `cd007_evaluation_order`; falsifier inverts RHS-before-LHS; observed **HIR/MIR DISAGREEMENT on stdout_bytes**. **low** |

### `ESF-VERIFY-001` is `ENGINE_LOCAL`, and that matters

MIR verification is **deliberately an independent checker**, so classifying MIR-0035 as a shared
semantic defect would be wrong. The finding is narrower and sharper:

```text
the verifier exists independently          -- unchanged
its NEGATIVE enforcement evidence is incomplete
    MIR-0035   demonstrated untested (mutated, survived)
    MIR-0029   census-only
    MIR-0037   census-only
```

It is a row rather than a loose residual because **AC6 must later justify the public phrase
"independently verified MIR"**, and a row gives that claim a machine-readable sensor.

### `ESF-LOWER-001` is the campaign's strongest positive result

```text
HIR semantic execution
        ≠
MIR lowering implementation
        ↓
observable DISAGREEMENT when lowering is wrong
```

Inverting CD-007 fails as `HIR/MIR DISAGREEMENT on stdout_bytes`. **A shared-fate defect cannot
produce that shape** — engines that inherit one answer agree while being wrong together. This is
materially stronger evidence than four configurations producing the same output, and it is what the
differential architecture actually buys.

### F1 and F2 get NO entry, deliberately

```text
F1/F2   no ESF entry
        reason: no live semantic authority remained after deletion
        disposition: dead construction, not shared fate
```

The bound-specialisation signature was deleted rather than controlled. **A shared-fate register is
for live semantic authorities**, and keeping ghosts in it because AC4 found them would make the
register a history of the campaign rather than a description of the compiler. Their history is in
`AC4-ADVERSARIAL-CAMPAIGN.md` §2.3.

### Rows AC4 did NOT change

```text
ESF-COPY-001/002, ESF-DROP-001/002, ESF-TRAP-001a/b, ESF-TYPE-001,
ESF-NUM-001, ESF-TRAIT-001, ESF-RES-001, ESF-PROV-001

unexamined by AC4 and unchanged. ESF-PROV-001's hir cell remains UNKNOWN: AC4 measured the
resource LIFECYCLE (F6), not the signature dependency EI2 left open, and UNKNOWN never resolves
silently to NONE.
```

### New binding rule

```text
a SURVIVED mutation is not evidence of shared fate or of a missing control until target
reachability has been demonstrated; if mutated and unmutated behaviour agree unexpectedly,
challenge the measuring path before classifying the authority
```

Justified by repeated findings rather than by principle: **F2, F3, F5 and F6 each depended on
challenging a survival**, and the namespace campaign additionally exposed a `--no-fail-fast`
instrumentation defect that made `killer_count` a lower bound.

---

## Method note

Every dependency cell above was established by reading the call graph, not by assuming from module
names. Two assumptions were tested and refuted in the process: the native backend appeared to have
its own `mir_ty_is_copy` (different signature, same name) but delegates in one line; and the drop
plan appeared duplicated in `emit_bodies` but is a documented application of `mir::drop_plan`. Both
would have been recorded as `PARTIALLY_VISIBLE` had the file names been trusted, understating the
shared fate.

`ESF-PROV-001`'s `hir` cell is left `UNKNOWN` rather than guessed, per EI0's binding rule that
`UNKNOWN` never resolves silently to `NONE`.
