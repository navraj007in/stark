# Campaign A — exit report

**Gate:** `WP-ARCHITECTURE-STABILIZATION.md` §5, "Campaign A exit gate".
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-08. **Head:** `2858dc7`.
**Status:** **CANDIDATE-PASS. Gate held for one substantive reason: AS3 #3 FAILS.**
Owner rulings of 2026-08-08 applied throughout.

> Campaign A passes only when AS0, AS1a, AS2, AS1b, AS3 and AS4 are complete and owner-reviewed. The
> exit report must classify each criterion PASS, FAIL, DEFERRED-BY-DECISION or NOT-APPLICABLE and
> include command-level evidence.

**Owner rulings applied (2026-08-08).** The first draft marked four criteria PARTIAL and asked for
rulings. All four are now classified in the gate's own vocabulary:

| Criterion | Ruling | Classification |
| --- | --- | --- |
| AS3 #3 — total type→`Value` enforcement | **HOLD** | **FAIL** |
| AS3 #5 — DEV-121 class closure | closure premature; **reopen DEV-121** | **FAIL** |
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
| AS3 — total callable-use metadata | 5 | **3 PASS, 2 FAIL** | §3, §3.1 |
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
| 2 | Implicit and explicit dispatch install the checker-selected generic environment in the HIR oracle | **PASS** | `push_resolved_env` installs the specialiser's bindings; `as3_fallback_removal::dev194_*`. DEV-194 was exactly this criterion failing, found by CI |
| 3 | The total type-to-`Value` relation is enforced at parameters, returns, receiver boundaries, bindings **and typed mutation without exemptions** | **FAIL** | See §3.1. The remainder is larger than the first draft stated |
| 4 | The frozen corpus and all engine comparisons remain green | **PASS (pending CI on head)** | Locally: `mir_differential` 132, `three_engine_differential` 109, `c6_generated_corpus`, `c6_metamorphic`, external sample suite 39/39. See §7 |
| 5 | DEV-121 closes only with a class-level evidence statement, not one regression case | **FAIL** | The 2026-08-07 closure was **premature and is withdrawn** (owner ruling). It proved every known view-producing intrinsic is exercised and that the narrow `INV-VALUE-REP-001` runs at four binding positions — real evidence, but not the class, which is defined by the total relation that #3 shows is unwired. The producer audit is retained as **defence-in-depth**, not authority. DEV-121 **REOPENED** |

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
