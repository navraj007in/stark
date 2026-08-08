# Campaign A — exit report

**Gate:** `WP-ARCHITECTURE-STABILIZATION.md` §5, "Campaign A exit gate".
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-08. **Head:** `2858dc7`.
**Status:** **CANDIDATE-PASS with four criteria not met.** Owner review required; this report does
not declare the gate passed.

> Campaign A passes only when AS0, AS1a, AS2, AS1b, AS3 and AS4 are complete and owner-reviewed. The
> exit report must classify each criterion PASS, FAIL, DEFERRED-BY-DECISION or NOT-APPLICABLE and
> include command-level evidence.

**A note on the classification vocabulary.** The gate names four classes. Four criteria below are
genuinely *partly* met — the work is real, the remainder is named, and the remainder was scoped by
me rather than deferred by the owner. Forcing those to PASS would be false and forcing them to FAIL
would be misleading, so they are marked **PARTIAL**, following the precedent set by
`AS-SPRINT1-CLOSEOUT.md` and `AS-SPRINT2-CLOSEOUT.md`, which used the same fifth class. **Each
PARTIAL is a decision the owner must make: accept as scoped, or hold the gate.** They are listed
together in §8 so none is buried.

---

## 1. Packet status

| Packet | Criteria | Verdict | Record |
| --- | :---: | --- | --- |
| AS0 — baseline, inventories, characterization | 7 | **PASS** | `AS-SPRINT1-CLOSEOUT.md`, `AS0-RB0-PREDICATE-INVENTORY.md` §7 |
| AS1a — canonical package source identity | 5 | **PASS** | `AS-SPRINT1-CLOSEOUT.md` |
| AS2 — one semantic pipeline | 5 | **PASS** | `AS-SPRINT1-CLOSEOUT.md` |
| AS1b — source-aware semantic metadata | 5 | **PASS** | `AS-SPRINT2-CLOSEOUT.md` |
| AS3 — total callable-use metadata | 5 | **4 PASS, 1 PARTIAL** | §3 |
| AS4 — one authority per type property | 5 | **2 PASS, 3 PARTIAL** | §4 |

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
| 3 | The total type-to-`Value` relation is enforced at parameters, returns, receiver boundaries, bindings **and typed mutation without exemptions** | **PARTIAL** | `INV-VALUE-REP-001` now runs at `let`, loop item, call parameter and method receiver (was `let` only). **Not covered:** struct fields, indexed slots, and values that never bind to a local. Those need a place-oriented check, stated in DEV-121's closure rather than implied. "Without exemptions" is therefore not met |
| 4 | The frozen corpus and all engine comparisons remain green | **PASS (pending CI on head)** | Locally: `mir_differential` 132, `three_engine_differential` 109, `c6_generated_corpus`, `c6_metamorphic`, external sample suite 39/39. See §7 |
| 5 | DEV-121 closes only with a class-level evidence statement, not one regression case | **PASS** | `KNOWN-DEVIATIONS.md` DEV-121 class closure: blind spot closed, extension load-bearing **measured** (broken producer + no-`let` binding: `TRAP` with the invariant vs `OK "3\n3\n"` without), inventory enforced by `every_view_returning_intrinsic_is_classified` |

**Boundary 4's structural exit**, beyond the criteria: `fn find_method`, `fn find_method_pass` and
`fn find_impl_fn` **do not exist**. All 12 callers consume published selections
(`AS3-SELECTOR-CENSUS.md`).

## 4. AS4 — one authority for semantic type properties

| # | Criterion | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | Every type property has one documented meaning and authority | **PARTIAL** | Done for **drop** (`mir::drop_rule`), **reference** (`mir::reference_rule`), **Copy** (`Ty` duplicate merged into `typecheck::is_copy_type_with`). `mentions_user_nominal` was already single. **Not done:** no exhaustive enumeration of *every* type property exists, so "every" is unproven |
| 2 | Near-neighbour predicates with different meanings are named so they cannot be substituted | **PASS** (for the three audited) | `requires_drop_glue` / `may_need_drop` / `ty_has_user_destructor_guarded`; `stores_a_reference` / `mentions_a_reference` |
| 3 | Adding a type/representation variant forces every applicable authority and evidence matrix to be revisited | **PARTIAL** | Drop, reference and Copy predicates are all wildcard-free, so a new `MirTy`/`CoreType`/`Ty` variant breaks compilation at the authority. Unaudited for properties outside those three |
| 4 | Resource, iterator, reference, generic-drop and partial-move adversaries pass across HIR, MIR and native | **PARTIAL** | `as4_property_adversaries` (12) + `as4_reference_rule` (5), three engines, mutation-tested. **Three families structurally cannot span three engines**, each recorded with its reason: generic `Drop` (oracle refuses it, A3c-D), host resources (interpreters have no host access), `Vec` of droppables (native defers `destructor-in-runtime-collection`) |
| 5 | Any behavioural correction receives its own decision record; AS4 does not disguise one | **PASS** | CD-386 (DEV-188), CD-387 (DEV-195). The drop-authority merge and the reference merge were verified behaviour-neutral *before* implementation. RB0 Q1's recommended change is **deferred to its own CD** rather than folded in |

**RB0's open questions, both closed:**

| | Question | Outcome |
| --- | --- | --- |
| Q1 | iterator drop-glue asymmetry | **Resolved with evidence** — every constructible iterator is a borrowed cursor owning nothing, with no Rust `Drop` and a `Noop` drop plan, so the asymmetry is historical. Change deferred to a CD (`AS4-RB0-Q1-ITERATORS.md`) |
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

## 8. The four decisions the owner must make

Collected so none is buried in a table.

1. **AS3 criterion 3** — the type-to-`Value` relation is enforced at four binding positions but not
   at struct fields, indexed slots, or values that never bind. Accept as scoped, or hold?
2. **AS4 criterion 1** — three properties have single authorities; "every type property" is
   unproven because no enumeration exists. Accept, or require the enumeration?
3. **AS4 criterion 3** — wildcard-freedom holds for the three audited properties only.
4. **AS4 criterion 4** — three adversary families cannot span three engines for structural reasons.
   Accept the recorded limits, or require the engines to change?

And one recommendation carried out of AS4: **RB0 Q1** proposes classifying `VecIter`, `KeysIter` and
`Iter` as requiring no drop glue, consistent with CD-387. It is behavioural and awaits its own CD.

## 9. Recommendation

Campaign A's substance is done: one semantic pipeline, one source identity, total callable metadata
with both selector functions deleted, and single authorities for the three type properties that had
duplicates. Ten defects were found and closed, two of them producing wrong answers or uncompilable
valid programs.

I recommend **CANDIDATE-PASS**, converting to PASS when (a) CI is green on the head commit and
(b) the owner rules on the four PARTIALs in §8. The binding gate on structured-concurrency
compiler/runtime work should remain in force until that ruling.
