# Campaign A — exit report

**Gate:** `WP-ARCHITECTURE-STABILIZATION.md` §5, "Campaign A exit gate".
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-08. **Head:** `2858dc7`.
**Status:** **CAMPAIGN A — PASS** (2026-08-08). Every exit condition is satisfied, including
repository-backed CI: `f55bcc4` completed **both workflows with zero failing jobs**, enumerated by
name. The final adversarial audit is `CAMPAIGN-A-FINAL-ADVERSARIAL-AUDIT.md`; Sprint 4's closure is
`SPRINT-4-CLOSURE.md`.

AS3 #1–#5 **PASS**. DEV-121 **CLOSED**. DEV-197 class **CLOSED**. AS4's seven scoped properties
**PASS**, with three deferments unchanged and explicitly restated (§4.0a) rather than promoted.

**What the audit did to the earlier candidacy.** AS3 #2, #3 and #5 were PASS-CANDIDATE pending
falsification. The audit found **seven further defects** (DEV-203 … DEV-209) and repaired all of
them; the criteria are recorded PASS on the repaired head rather than on the head that was
audited. That the audit found defects is not an argument against closure — every one was a
violation of a rule the architecture already stated, repaired at an authority that already existed,
with no new representation, type rule, execution funnel or place system.

> Campaign A passes only when AS0, AS1a, AS2, AS1b, AS3 and AS4 are complete and owner-reviewed. The
> exit report must classify each criterion PASS, FAIL, DEFERRED-BY-DECISION or NOT-APPLICABLE and
> include command-level evidence.

**Owner rulings applied (2026-08-08).** The first draft marked four criteria PARTIAL and asked for
rulings. All four are now classified in the gate's own vocabulary:

| Criterion | Ruling | Classification |
| --- | --- | --- |
| AS3 #3 — total type→`Value` enforcement | HOLD, then promote once the evidence exists | **PASS** |
| AS3 #5 — DEV-121 class closure | closure premature; **reopen DEV-121** | **PASS** (DEV-121 **CLOSED**) |
| AS3 #2 — environment installation | DEV-197 disproved the universal claim; requalify | **PASS** |
| AS4 #1 — one authority per type property | accept as scoped | **PASS** (§4.1) |
| AS4 #3 — new variants force revisit | accept as scoped, evidence closure required | **PASS** (§4.2) |
| AS4 #4 — three-engine adversaries | deferred for structurally unavailable lanes | **DEFERRED-BY-DECISION** |
| RB0 Q1 | approve the behavioural correction | landed as **CD-388** |

"Every type property" is read as **every property in the approved AS4/RB0 inventory**, not every
type-related helper in the compiler — consistent with AS0's inventory being adopted as AS4's
opening inventory and with the programme's no-broad-cleanup rule.

---

## 1. Packet status

| Packet | Criteria | Verdict | Record |
| --- | :---: | --- | --- |
| AS0 — baseline, inventories, characterization | 7 | **PASS** | `AS-SPRINT1-CLOSEOUT.md`, `AS0-RB0-PREDICATE-INVENTORY.md` §7 |
| AS1a — canonical package source identity | 5 | **PASS** | `AS-SPRINT1-CLOSEOUT.md` |
| AS2 — one semantic pipeline | 5 | **PASS** | `AS-SPRINT1-CLOSEOUT.md` |
| AS1b — source-aware semantic metadata | 5 | **PASS** | `AS-SPRINT2-CLOSEOUT.md` |
| AS3 — total callable-use metadata | 5 | **2 PASS, 3 FAIL** | §3, §3.1 |
| AS4 — one authority per type property | 5 | **4 PASS, 1 DEFERRED-BY-DECISION** | §4 |

AS5 is Campaign **B** and is not a gate condition; it is complete (`AS-SPRINT2-CLOSEOUT.md`).

## 2. AS0, AS1a, AS2, AS1b — carried forward

All four were classified criterion-by-criterion in the sprint closeouts and are not re-litigated
here. AS0 exited when its items 6 and 7 landed and item 10 was deferred by owner decision to AS8/C10
(`AS0-RB0-PREDICATE-INVENTORY.md` §7).

## 3. AS3 — total callable-use metadata and oracle representation enforcement

| # | Criterion | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Every executable user-callable use has exactly one record; duplicates and omissions fail an invariant test | **PASS** | `as3_callable_use_exactness` (9 tests). Expectations derived from HIR shape + `expr_types`, never from the table under test. Mutation-tested 4 ways: disabling the operator publisher fails 4/9, the bound publisher 7/9, the Display walk 1, a double publication 1 |
| 2 | Implicit and explicit dispatch install the checker-selected generic environment in the HIR oracle | **PASS** | Requalified by **omission**, not by asserting a table has an entry — see §3.2. Seven dispatch-class controls (D1–D7) each remove the environment at the single installation point and require the run to fail; each witness answers `size_of::<T>()`, so the instantiation is load-bearing rather than incidental. Three structural pins hold the shape for future edits. The requalification found **DEV-202** |
| 3 | The total type-to-`Value` relation is enforced at parameters, returns, receiver boundaries, bindings **and typed mutation without exemptions** | **PASS** | 12 of 12 `RepBoundary` variants `Wired`, pinned executably by `dev121_boundary_inventory`; four-class producer-mutation evidence proves each wire forces its defect class; the one remaining missing-metadata escape (inside `bind_typed_local`) is closed. See §3.1a |
| 4 | The frozen corpus and all engine comparisons remain green | **PASS** | On `36b3dfc`: `cargo test` 209 targets / 2 743 tests / 0 failures, `mir_differential` 132, `three_engine_differential` 109, external sample suite **39/39 at the pinned commit**, `stark-url` 20/20. **Two corpus edits are recorded, not silent** — see §3.3 |
| 5 | DEV-121 closes only with a class-level evidence statement, not one regression case | **PASS** (DEV-121 **CLOSED**) | The 2026-08-07 closure was **premature and is withdrawn** (owner ruling). It proved every known view-producing intrinsic is exercised and that the narrow `INV-VALUE-REP-001` runs at four binding positions — real evidence, but not the class, which is defined by the total relation that #3 shows is unwired. The producer audit is retained as **defence-in-depth**, not authority. DEV-121 **REOPENED** |

### 3.0 The dependency DEV-197 exposed

Wiring the **first** value boundary found two defects that no test had seen, and the reason is
structural rather than incidental:

```text
CallableUse / environment publication
             |
   environment installation
             |
      callee body executes
             |
 typed boundary consumes the result
             |
   ONLY HERE can a wrong environment become observable
```

Both DEV-197 sites — `Res::AssociatedFn` and the function-value path — ran bodies with `T` unbound.
Neither produced a wrong answer: both fixtures were `identity`-shaped, so the missing environment
could not change the output. **That makes them stronger architecture evidence, not weaker.** They
show why ordinary differential testing can be green while the interpreter is internally
inconsistent: wrong metadata becomes observable only when another semantic consumer asks a question
that depends on it, and until this packet nothing did.

It is also why AS3 #2 cannot stand as PASS on its previous evidence, and why #2 and #3 requalify
together.

### 3.3 AS3 #4 — the corpus moved twice, and both edits are recorded

A frozen corpus that changes silently is worthless as evidence, so both edits this campaign made to
it are stated here with the defect that forced them.

| Case | Edit | Forced by |
| --- | --- | --- |
| `examples/gate3/05_core_min.stark` | `println(values[0..2])` → `println(&values[0..2])` | **DEV-206** |
| `stark-samples pkg/05-data-modelling` | **none — the compiler was repaired instead** | **DEV-209** |

The Gate 3 example encoded an invalid program that was accepted only because `Display` eligibility
had the `[T]` / `&[T]` polarity reversed. Correcting the example is the last step of the chain, not
a weakening of the corpus:

```text
frozen example
      ↓ new invariant proves it depended on invalid acceptance
DEV-206 established
      ↓ checker repaired toward the existing language rule
example corrected to the valid spelling
      ↓
whole frozen corpus requalified — 39/39
```

The sample-suite case is the opposite and more important precedent: it was **valid** STARK that the
oracle could not run, and the ruling was to repair the compiler and leave the application untouched.
Rewriting it would have converted "an application exposed a missing capability" into "an application
learned a compiler workaround" — the behaviour a stabilised architecture exists to prevent.

Requalification after both: external sample suite **39/39** at pinned `b3b28e757f38d691e...`,
`stark-url` **20/20**, `mir_differential` 132, `three_engine_differential` 109.

### 3.2 AS3 #2 — requalified by omission (2026-08-08)

The criterion was recorded PASS once before, on tests that asserted the environment table had an
entry. DEV-197 is what that missed: **nine** dispatch sites installed no environment at all and
every test passed, because the bodies involved never mentioned their own generic parameters. An
environment that is never consulted cannot be observed to be absent.

So the requalification proves the claim by **removing** the environment and requiring failure.

**The eight properties, and where each is evidenced.**

| # | Property | Evidence |
| --- | --- | --- |
| P1 | One body-entry authority | `p1_exactly_one_production_body_executor` — `eval_block(callable.body)` occurs exactly once; `p1_the_raw_executor_is_called_only_by_the_authority` |
| P2 | Environment is an explicit state, never absent | `p2_every_invocation_environment_variant_is_explicit` — exhaustive over `InvocationEnv`, so a new variant fails to compile until it is described |
| P3 | Published dispatch consumes the published environment | D1, D2, D3 |
| P4 | Bound dispatch is atomic — body and environment from one specialiser call | D5 |
| P5 | Function values install their captured bindings | D6 |
| P6 | Environment dominates the typed call boundaries | `p6_typed_boundaries_run_while_the_environment_is_active` (behavioural: a `&W<T>` receiver would fail on a *correct* program if read first) plus the structural pin that the install precedes the body and the guard is bound, not dropped |
| P7 | Missing environment metadata fails loudly | every control asserts `FailureClass::InternalInvariant` — never Empty, a skip, a default, or a reconstruction |
| P8 | Nested calls install and restore | D7 and `p8_a_callees_environment_does_not_outlive_it` |

**The seven dispatch-class controls.** Each removes the environment at the single installation
point, and each asserts three things — that the witness passes unmutated *with the answer its
instantiation determines*, that the mutation was **reached**, and that the run then fails as an
internal invariant.

| Class | Witness | Unmutated answer |
| --- | --- | --- |
| D1 free generic function | `width<T>(x: T)` | `8` |
| D2 generic associated function | `S::width<T>` | `8` |
| D3 generic inherent method | `s.width<T>` | `8` |
| D4 operator into a generic impl | `impl<T> Eq for W<T>` | `1` |
| D5 bound trait dispatch | `impl<T> Sz for P<T>` through `S: Sz` | `8` |
| D6 function value | `let f: fn(Float64) -> Int32 = width` | `8` |
| D7 nested generic calls | `outer<Float64>` calling `inner<Int32>` | `848` |

Two choices in that table are deliberate, and both are corrections of how DEV-197 hid:

- **Every witness answers `size_of::<T>()`.** DEV-197's original two defects were invisible because
  both bodies returned their argument unchanged, so an unbound `T` changed no answer. A control
  with that property would reproduce the blindness it is testing for.
- **D7 uses contrasting instantiations** — `T = Float64`, `U = Int32`, answer `848`. A restoration
  bug answers `844`. Identical bindings would let a stale frame pass by coincidence.

**Requiring the mutation to be REACHED is not a formality.** A control whose witness never touches
the installer would detect nothing and look like a pass — which is the failure mode of the evidence
this requalification replaces.

**What it found: DEV-202.** `call_method` chose the environment, **installed it**, and then passed
it to `call_user_method`, which routes through the authority — which installs it again. The
redundancy is not the problem; the scope is. The outer guard was live while the *caller's* receiver
place was still being resolved and materialized, so caller-side work ran under the callee's
instantiation — the same scope error P6 exists to prevent, in the other direction. It produced no
wrong answer, because both installations push identical bindings. That is precisely the class AS3
#2 was reopened to find: a defect of the architecture claim rather than of any current output, and
the reason the pin counts installation points instead of checking that a table is populated.

### 3.1a AS3 #3 — progress since this report (updated 2026-08-08)

The classification above stands at **FAIL** until every `RepBoundary` is wired. This subsection
records how far the closure packet has got, so "nearly done" cannot accumulate around the gate the
way it did before the 2026-08-07 premature closure.

| Packet | State | What it established |
| --- | --- | --- |
| 1 — one body executor | **done** | `eval_block(callable.body)` appears in exactly one place. Receiver materialization moved into the authority, so a destructor's `self` is a genuine `&mut Self` and needs no `Drop`-shaped exemption from the receiver boundary |
| 2 — `Receiver`, `Parameter`, `Propagation` | **done** | 4 of 11 boundaries wired, all against one lookup of `callable_types[body]` |
| 3–7 — bindings, writes, aggregates, expression results | **not started** | 7 boundaries remain `Unwired` |

The executable inventory is `starkc/tests/dev121_boundary_inventory.rs`; its progress pin asserts
the exact wired set, so this table cannot drift from the code.

**Three defects were found by wiring these four boundaries**, none of which any test observed
before — DEV-198 (the published callee *selection* was the one table field never grounded, so a
bound method's inferred generic argument reached the runtime as an inference variable), DEV-199
(an associated-type projection `T::Item` was unresolvable at a value boundary because the checker's
`assoc_projections` table was never published), and DEV-200 (`&mut [T]` refused the slice-view
representation that `&[T]` accepts). That rate — three real defects for four boundaries — is the
argument for finishing the remaining seven rather than accepting partial wiring.

### 3.1 AS3 #3 — FAIL, and the first draft understated it

The first draft said four binding positions were covered and named struct fields, indexed slots and
non-binding values as the remainder. **That was wrong in the reader's favour**, and the reviewer was
right to push. Two distinct functions exist and only the narrow one runs:

| | Function | What it checks | Production callers |
| --- | --- | --- | ---: |
| the **total relation** the criterion names | `interp::check_value_for_ty(ty, value, span, RepBoundary)` | the full §6 type→`Value` relation | **0** |
| its only production wrapper | `interp::check_local_value` | ditto, against a local's declared type | **0** — marked `#[allow(dead_code)]` |
| what actually runs | `interp::check_value_representation(local, value, span)` | INV-VALUE-REP-001 only: a `&[T]`/`&str` binding must not hold owned `Vec`/`String` storage | **5** |

So the criterion's subject — *the total relation* — is **enforced nowhere in production**. What runs
at those five sites is one direction of one pairing.

`RepBoundary` names eleven boundaries. Every construction of one outside its own `as_str` is inside
`mod tests`: the enforcement path does not even take a `RepBoundary` parameter.

**The remaining surface, per the reviewer's correction and this census:**

```text
return / propagated return        RepBoundary::Return, ::Propagation
match binding                     RepBoundary::MatchBinding
ordinary typed assignment         RepBoundary::Assignment
field write                       RepBoundary::FieldWrite
indexed / element write           RepBoundary::ElementWrite
aggregate-field construction      RepBoundary::AggregateField
inline values consumed without a local binding
struct fields and indexed slots (place-oriented, no local to key on)
```

This is the gap DEV-121 came from: the oracle carrying a runtime representation inconsistent with
its static type, and answering confidently. Closing it is one bounded packet — the **place/value
boundary closure** — whose exit is an exact inventory proving every `RepBoundary` is either wired or
structurally impossible.

**Not a design gap.** `check_value_for_ty` is written, exhaustive on `Ty` (its own comment refuses a
wildcard "because it would swallow any `Ty` variant added later"), and unit-tested. The work is
wiring it to the eleven boundaries and deciding the place-oriented ones.

**Boundary 4's structural exit**, beyond the criteria: `fn find_method`, `fn find_method_pass` and
`fn find_impl_fn` **do not exist**. All 12 callers consume published selections
(`AS3-SELECTOR-CENSUS.md`).

## 4. AS4 — one authority for semantic type properties

| # | Criterion | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Every type property has one documented meaning and authority | **PASS** (scoped) | §4.1 — seven-property mapping, each to one authority |
| 2 | Near-neighbour predicates with different meanings are named so they cannot be substituted | **PASS** (for the three audited) | `requires_drop_glue` / `may_need_drop` / `ty_has_user_destructor_guarded`; `stores_a_reference` / `mentions_a_reference` |
| 3 | Adding a type/representation variant forces every applicable authority and evidence matrix to be revisited | **PASS** (scoped) | §4.2 — exhaustiveness table; every scoped authority is wildcard-free |
| 4 | Resource, iterator, reference, generic-drop and partial-move adversaries pass across HIR, MIR and native | **DEFERRED-BY-DECISION** (unavailable lanes only) | §4.3 |
| 5 | Any behavioural correction receives its own decision record; AS4 does not disguise one | **PASS** | CD-386 (DEV-188), CD-387 (DEV-195). The drop-authority merge and the reference merge were verified behaviour-neutral *before* implementation. RB0 Q1's recommended change is **deferred to its own CD** rather than folded in |

### 3.4 Sprint 4 Phase 3 — convergence qualification (2026-08-08)

Run against the tree at `f55bcc4`, after DEV-210, DEV-211 and DEV-212.

| Suite | Result |
| --- | --- |
| `cargo test --lib` | 558 |
| `mir_differential` | 132 |
| `native_c6_1_ownership` | 24 |
| `as4_hostile_combinations` | 12 |
| `as4_destructor_authority` | 6 |
| **first-party package qualification** | **53 packages, 0 failures** |
| **external sample suite** @ pinned `b3b28e757f38d691e…` | **39/39** |
| `fmt`, `clippy --workspace --all-features --all-targets -D warnings` | clean |

**The two application suites are the ones that mattered here**, because DEV-211 introduced a new
*rejection* — moving a component out of a `Drop` nominal. A new refusal cannot be qualified by
compiler tests alone: it changes what programs compile, and the question is whether any real one
relied on the old acceptance. Blast radius was measured before implementing (no first-party package
uses `impl Drop`; three sample files do), and re-measured after: 53 packages and 39 sample cases
unchanged.

**Repository-backed:** `04f0391` (DEV-210, DEV-211) completed **success** on both workflows.
`d047e13` was red — `MIR-0007` in `mir_differential` and `native_c6_1_ownership` — from an abandoned
edit left in the general match path when DEV-212's fix moved to the enum path; `f55bcc4` removes it
and is the head under test.

### 4.0 Sprint 4 re-census (2026-08-08) — and the family that was not consolidated

AS4's property inventory was re-audited from **production structure**, not from its tests, because
Sprint 3's own lesson is that a passing suite is not evidence a second answer is absent.

| # | Property | Authority | Verdict |
| ---: | --- | --- | --- |
| 1 | Copy eligibility | `is_copy_type_with` (checker) / `mir_ty_is_copy` (MIR) | **PASS** — borrowck, typecheck, lowering and the backend all delegate |
| 2 | Runtime drop glue | `drop_rule::requires_drop_glue_with` | **PASS** — verifier, lowering and the free fn all delegate |
| 3 | User-defined destruction | `nominals_with_destructor` | **PASS as of this sprint** — see below |
| 4 | Stored-reference containment | `reference_rule::stores_a_reference` | **PASS** |
| 5 | Borrow-lifetime carrying | `emit_types::mentions_a_reference` | **PASS** — a near neighbour of #4, not a duplicate: they differ only on `FnPtr`, and the difference is live and tested |
| 6 | User-nominal containment | `lower::ty_mentions_user_nominal` | **PASS** — single predicate, exhaustive, two consumers |
| 7 | Runtime representation | `value_matches_ty` | **PASS** — consolidated in Sprint 3; the value-context question derives from it rather than duplicating it |

**Property 3 had three answers, and the borrow checker's was materially weaker.** It identified
`Drop` by asking whether the written trait name `.ends_with("Drop")` — so `impl MyDrop for S` gave
`S` a destructor it did not have, and a legal partial move out of one of its fields was refused.
**Valid Core rejected on spelling** (DEV-210), the CD-379 identity class in a new place. The repair
was not a better string test: `copy_eligible_types` already computed the set by resolved identity
and kept it private, so the checker was already answering correctly and the borrow checker had
written a second, weaker answer beside it.

Consolidating it exposed two further defects, both in consumers rather than in the authority:

- **DEV-211** — moving a component out of an owned `Drop` nominal was accepted, which
  OWN-PARTIAL-001 prohibits; the checker had the rule for struct fields and never applied it to a
  matched component.
- **DEV-212** — a `match` skipped a `Drop` nominal's own destructor even when nothing moved out, in
  **both** engines. Repaired in HIR and MIR.

**Three wrong turns on DEV-212 are recorded in the ledger**, and they share one mistake: reaching
for a predicate that was *nearby* rather than establishing which types reach the site.
`ty_has_user_drop` ("contains a destructor anywhere, including a nested payload") in place of
`type_has_drop_impl` ("this nominal declares one") fired the whole-value branch for
`Option<Droppable>` and cost fourteen `MIR-0007` failures. That is the same shape as DEV-210, one
sprint later — which is the argument for AS4's second requirement, that an authority's **name say
which question it answers**, not merely that one authority exist.

### 4.0a The approved deferments, stated rather than promoted

A deferment is not a PASS with a softer word, and Sprint 4 did not convert one. Each is restated
here with the condition that would reopen it.

| Deferment | Status | Reopens when |
| --- | --- | --- |
| **Generic user `Drop` in the HIR oracle** | DEFERRED-BY-DECISION | Destruction reaches `drop_value` with a `Value` and recovers the nominal through `nominal_item`, so `Wrapper<String>` and `Wrapper<Int32>` are indistinguishable — the type arguments that selected the impl are gone. The oracle **refuses** rather than guesses (`OracleLimitation::GenericDrop`), classified `internal` so the differential harness cannot read it as a language outcome. MIR and native retain the arguments and execute it correctly. Reopens if Campaign A, or a first-party package, needs it |
| **Native drop glue for `Vec`/`Box` of a custom-destruction element** | DEFERRED-BY-DECISION | The proper solution is generated deterministic drop-glue helpers keyed by concrete `MirTy`, with recursive helpers **named** rather than recursively expanded. Not implemented merely to clear a documented deferment. Sprint 4 confirmed HIR and MIR still agree for these shapes (`a_vec_of_droppables_agrees_between_hir_and_mir`), so the deferment remains native-only and has not widened |
| **AS4 #4 — three-engine adversaries on structurally unavailable lanes** | DEFERRED-BY-DECISION | Unchanged from the 2026-08-08 ruling |

**Sprint 4 checked that neither of the first two had silently widened**, which is the risk a
deferment carries: the hostile suite exercises `Box` of a droppable and `Vec` of a droppable through
HIR and MIR and requires agreement, and a generic wrapper answering `Drop` **per instantiation**
(`W<Int32>` needs no destruction, `W<R>` destroys its field exactly once). A deferment that had
crept into the other engines would fail there rather than in a future user's program.

### 4.1 AS4 #1 — the seven scoped properties, each with one authority

Scope is the approved AS4/RB0 inventory, per the owner ruling, not every type-related helper.

| # | Property | The question it answers | Authority | Language |
| ---: | --- | --- | --- | --- |
| 1 | Copy classification | *may a value of this type be duplicated rather than moved?* | `mir::mir_ty_is_copy` (+3 wrappers) / `typecheck::is_copy_with_impls` | `MirTy` / `Ty` |
| 2 | runtime drop glue | *does destroying a value run anything?* | `mir::drop_rule::requires_drop_glue_with` + `core_requires_drop_glue` | `MirTy` |
| 3 | user-defined destruction | *does a user `impl Drop` govern this type?* | `lower::ty_has_user_destructor_guarded` | `MirTy` |
| 4 | stored-reference containment | *does a value of this type STORE a reference?* | `mir::reference_rule::stores_a_reference` | `MirTy` |
| 5 | borrow-lifetime carrying | *is a reference named anywhere, INCLUDING in a signature?* | `emit_types::mentions_a_reference` | `MirTy` |
| 6 | user-nominal containment | *does this type mention a user nominal?* | `lower::ty_mentions_user_nominal` | `MirTy` |
| 7 | runtime representation | *may this `Value` represent this `Ty`?* | `interp::check_value_for_ty` | `Ty` × `Value` |

Two languages for Copy is not a duplicate: the checker reasons over `Ty` before MIR exists. The
duplicate that DID exist — `borrowck::is_copy_type`, a second `Ty` implementation kept aligned by
hand — was merged (`AS4-COPY-RULE-AUDIT.md`).

Properties 2 and 5 were three implementations between them at AS0; both are now single, with the
genuine near neighbours (#3, #5, and `verify::may_need_drop`) **named so they cannot be
substituted**. Property 7's authority exists and is correct; it is **not wired**, which is AS3 #3's
FAIL, not an AS4 duplicate.

### 4.2 AS4 #3 — exhaustiveness: which enum addition breaks compilation, and where

| Property | Authority | Wildcard? | A new variant breaks compilation at |
| --- | --- | :---: | --- |
| Copy (`MirTy`) | `mir::mir_ty_is_copy` | none | `MirTy` — every scalar listed individually |
| Copy (`Ty`) | `typecheck::is_copy_with_impls` | none | `Ty` |
| drop glue | `drop_rule::requires_drop_glue_with` | none | `MirTy` |
| drop glue, Core | `drop_rule::core_requires_drop_glue` | none | `CoreType` — nested exhaustive match |
| user destructor | `lower::ty_has_user_destructor_guarded` | none | `MirTy` |
| stores reference | `reference_rule::stores_a_reference` | none | `MirTy` |
| mentions reference | `emit_types::mentions_a_reference` | none | `MirTy` |
| mentions user nominal | `lower::ty_mentions_user_nominal` | none | `MirTy` |
| conservative drop | `verify::may_need_drop` | none | `MirTy` |
| type→`Value` relation | `interp::value_matches_ty` | none **on `Ty`** | `Ty` |

**One row was fixed to make this table true.** `ty_has_user_destructor_guarded` recovered its type
arguments through a nested re-destructure ending in `_ => unreachable!()`. Unreachable by
construction, but it reads as a property-bearing wildcard to every future audit, so `args` is now
bound in the outer pattern and the arm is gone.

**One row needs its caveat stated.** `value_matches_ty` is exhaustive on `Ty` — its own comment
refuses a wildcard "because it would swallow any `Ty` variant added later" — and carries four inner
`_ => false` arms on the **`Value`** side. Those default to *not permitted*, the safe direction for
a permission relation: a new `Value` variant is rejected loudly rather than silently admitted.

### 4.3 AS4 #4 — DEFERRED-BY-DECISION, per lane

```text
generic Drop
  HIR lane   DEFERRED — the oracle refuses it (A3c-D/DEV-176): `drop_value` receives a
             `Value` whose type arguments are gone
  MIR/native QUALIFIED — both retain the arguments and execute it
  compensation: a NON-generic destructor through generic containers, which attacks the
  same shared recursion (`a_non_generic_droppable_inside_a_generic_nominal_is_dropped`,
  `two_instantiations_of_one_generic_differ_in_droppability`)

host resources
  interpreter lanes DEFERRED — capability packages have no host access under `stark run`
  native/provider    QUALIFIED — `c788_resource_lifecycle`, `c788_lifecycle_e2e`,
                     `a11_host_resource`

Vec<droppable>
  native lane DEFERRED — named backend limitation `destructor-in-runtime-collection`
  HIR/MIR     QUALIFIED — the oracle runs it, MIR lowers and verifies it
  refusal explicitly pinned by `a_vec_of_droppables_is_deferred_by_the_native_backend`
```

**The pin was an evidence defect until this revision.** That test asserted checker acceptance, HIR
execution and MIR lowering, then stopped — while its name claimed a native deferral it never
exercised. Reviewer finding, in this packet's own suite. It now calls `emit_native_debug` and
asserts the named refusal, and fails loudly if native ever accepts the program.

A bounded follow-up, `WP-NATIVE-DROP-COLLECTIONS`, would remove the third deferral by making the
native backend a complete consumer of the existing `DropPlan` through generated per-type drop-glue
helpers. It is **not** Campaign A work.

**RB0's open questions, both closed:**

| | Question | Outcome |
| --- | --- | --- |
| Q1 | iterator drop-glue asymmetry | **Resolved and corrected.** The asymmetry was historical; `VecIter`, `KeysIter` and `Iter` now join `CharsIter` at `requires_drop_glue = false` under **CD-388**. The pin written to demand a CD fired exactly as designed when the change was made |
| Q2 | `FnPtr` reference disagreement | **Answered** — storage vs signature are different questions; the identical pair merged, the near neighbour renamed (`AS4-REFERENCE-RULE.md`) |

## 5. Defects found and closed during Campaign A

Recorded because the count is the argument for the campaign, not decoration. Every one was found by
removing a duplicate authority or a fallback — none by reading code.

| | Defect | Severity |
| --- | --- | --- |
| DEV-183 | `TRAIT-COHERENCE-001` never enforced | over-acceptance |
| DEV-187 | bound specialisation missed generic impls | engine divergence |
| DEV-188 | trait-method generics dropped at bound call sites | **methods uncallable** |
| DEV-189 | MIR passed the bare nominal head | engine divergence |
| DEV-190/191 | `self.m()` in trait defaults, and operators on bounded parameters, published nothing | silent name scans |
| DEV-192 | `==` through an `Eq` bound used structural comparison | **wrong answers from the reference engine** |
| DEV-193 | direct calls published `FunctionValue` | false metadata |
| DEV-194 | trait default via a non-`Static` route lost its `Self` binding | **runtime failure** |
| DEV-195 | `Vec<CharsIter>::clear()` accepted, run, then refused | **valid program uncompilable** |
| DEV-196 | legacy Core `File` drop classification | characterized; not a live defect |

DEV-192 is the one that justifies the method: no differential suite could see it, because every
fixture's `eq` agreed with structural equality.

## 6. Command-level evidence

```bash
cd starkc
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --lib                                    # 538
cargo test --test as3_callable_use_exactness        # 9
cargo test --test as3_fallback_removal              # 9
cargo test --test as3_display_plan                  # 17
cargo test --test as4_property_adversaries          # 12
cargo test --test as4_reference_rule                # 5
cargo test --test as4_vecclear_divergence           # 5
cargo test --test dev121_view_producer_audit        # 7
cargo test --test dev188_bound_method_generics      # 8
cargo test --test mir_differential                  # 132
cargo test --test three_engine_differential         # 109
cargo test --test mir_verify                        # 51

# external sample suite, pinned at b3b28e75
python3 verify.py --stark <stark> --starkc <starkc>  # 39/39

# the structural exit criterion
grep -c "fn find_method\|fn find_impl_fn" src/interp.rs src/mir/lower.rs   # 0
```

## 7. What this report does NOT establish

- **CI on the head commit.** `2858dc7` was pushed minutes before this report and CI was still
  running. Every result above is local plus the last green CI run on an earlier commit. The gate
  should not be signed until CI is green on the head.
- **Windows and Linux lanes** for the Campaign A work are CI-only; nothing here was run on them.
- **AS4 criterion 1's "every"** — see §4.

## 8. Remaining work before PASS

The owner's rulings resolved three of the four open criteria. **One substantive item remains.**

| # | Item | State |
| ---: | --- | --- |
| 1 | **AS3 #3 — the place/value boundary closure** | **BLOCKING.** One bounded packet: wire `check_value_for_ty` to the eleven `RepBoundary` positions, and produce an exact inventory proving each is wired or structurally impossible. §3.1 |
| 2 | CI green on the branch head | outstanding — see §7 |

Landed since the first draft, in the ruled sequence:

```text
CD-388                        VecIter/KeysIter/Iter join CharsIter at no-drop-glue      DONE
AS4 #1 seven-property table                                                             DONE  §4.1
AS4 #3 exhaustiveness table (+1 wildcard removed)                                       DONE  §4.2
Vec<droppable> native test now genuinely reaches native emission                        DONE  §4.3
exit classifications updated                                                            DONE
```

Not Campaign A work, recorded so it is not lost: `WP-NATIVE-DROP-COLLECTIONS` would remove the
native `Vec`/`Box` destructor deferral by generating per-type drop-glue helpers from the existing
`DropPlan`. The semantic design is already done; the backend is an incomplete consumer of it.

## 9. Recommendation

Campaign A's substance is done: one semantic pipeline, one source identity, total callable metadata
with **both selector functions deleted**, and single authorities for every type property in the
approved inventory. Ten defects were found and closed, two of them producing wrong answers or
uncompilable valid programs — every one surfaced by removing a duplicate authority or a fallback,
none by reading code.

**The gate is held, and correctly so.** AS3 #3 fails on its own terms, and the reviewer's push is
what established that the gap was larger than this report first said: the total type→`Value`
relation is not merely partly wired, it is **enforced nowhere in production**. That is exactly the
defect class DEV-121 came from, and letting it through before structured concurrency would leave the
oracle able to give another confident wrong answer.

Recommended sequence:

```text
1. close AS3 #3 — the place/value boundary packet
2. update this report's AS3 #3 classification to PASS
3. obtain green CI on the final branch head
4. declare Campaign A PASS
5. lift the structured-concurrency binding restriction
```

Until step 4, the binding restriction stays in force.
