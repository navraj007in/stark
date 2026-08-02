# WP-DEV-134-139 — Final Report

**Status:** COMPLETE for all six defect repairs and all infrastructure tasks
**Programme:** CD-335 … CD-342
**Owner:** Claude
**Date:** 2026-08-02

Required by WP-DEV-134-139 §17. Section numbering follows §17's eleven items.

---

## 1. Commit hashes

| CD | Commit | Subject |
| --- | --- | --- |
| CD-334 | `b513f5e` | six defects filed from an external sample suite (pre-programme) |
| CD-335 | `d83acd8` | DEV-134 — `?` relates its operand to the return type |
| CD-336 | `f35ec95` | DEV-137 — condition-only borrows end at the branch boundary |
| CD-337 | `949aa4f` | DEV-136 — only branches that reach a join contribute to it |
| CD-338 | `309f7ee` | DEV-135 — a field is one place however many times it is written |
| CD-339 | `f69614e` | DEV-139 — a method body reads the impl's bounds |
| CD-340 | `53c1cce` | DEV-138 — closed as a DEV-121 instance |
| CD-341 | `0f23991` | external sample suite published, pinned, gated in CI |
| CD-342 | `e7be5c2` | layer audit becomes an enforcing gate; six findings numbered |

External suite: `navraj007in/stark-samples`, pinned at `b3b28e757f38d691e7309f168d1209e28ac459af`.

---

## 2. Defect-by-defect root cause

Every one was a single wrong line or a single missing consultation. **None required a design
change**, and the work package's largest estimate — DEV-135 as "full field-sensitive move paths" —
was wrong by roughly two orders of magnitude.

| DEV | Root cause |
| --- | --- |
| 134 | The `Try` arm asked "is the return type `?`-capable?" and "is the operand `?`-capable?" as INDEPENDENT questions and never compared them. |
| 137 | `active_borrows` is scoped by block end and statement end. A condition is neither, so nothing ever popped its temporaries. |
| 136 | Move state merged the `if`/`match`'s syntactic CHILDREN rather than the join's reaching PREDECESSORS. |
| 135 | `Projection::Field(name.lo, name.hi)` identified a field by the SPAN it was written at, so two mentions of one field never compared equal. |
| 139 | Two bound lookups each kept their own copy of the parameter search, and both read `current_fn_generics` alone — half the environment. |
| 138 | `SplitIter::next` returned `Value::String` (owned) for an item declared `&str`. |

**Four of six were wider than filed**, and in every case the extra half was found by the repair's
own must-pass tests rather than by the reproducer:

- DEV-134 also covered `Option`/`Result` cross-family propagation.
- DEV-137 also covered `if` conditions, found by the growing-vector re-evaluation case.
- DEV-139's trait-bound half is DEFERRED and survived the first fix intact.
- DEV-135's inventory (§5.3) proved parent poisoning unacceptable, so §5.2's two-stage model
  collapsed to neither stage.

---

## 3. Exact repair

| DEV | Repair |
| --- | --- |
| 134 | `check_try_compatibility` in `typecheck.rs`, deferred until inference settles. Requires same constructor and, for `Result`, an equal error type. |
| 137 | `Borrowck::check_condition` — snapshot borrow depth, check, truncate. Used by `While` and `If`. |
| 136 | `block_diverges`/`expr_diverges`; only reaching branches contribute to a join. |
| 135 | `Projection::Field(String)` / `TupleField(String)`, holding the resolved name. |
| 139 | `param_declares_bound` (both lookups) and `current_generic_env` (the deferred capture). |
| 138 | `SplitIter::next` yields `Value::Str`, which `value_is_copy` reports as Copy. |

---

## 4. Rejected alternatives

- **DEV-134: applying `From` at the propagation site.** `03-Type-System.md` does not scope a
  conversion there, so it would be new semantics, not a repair. Rejection is the conservative
  half. Whether Core v1 *should* gain conversion at `?` remains open and unowned.
- **DEV-134: widening `types_equal` with a `Ty::Param` arm.** It has no `Ty::Param` arm at all, so
  two occurrences of one parameter compare unequal — a real latent gap, caught by this repair's own
  negative control. Rejected as out of scope: its callers are coherence/overlap paths where the gap
  has no symptom, and changing a shared primitive during a release-critical repair is the wrong
  trade. The structural walk now takes the behaviour as a PARAMETER instead, written once.
- **DEV-137: clearing the borrow set at the loop header.** Would pass the reproducer and be
  unsound — a borrow created *before* the loop must survive. The truncate is depth-based for
  exactly this reason.
- **DEV-137: generalising to "loop and branch headers".** `match` scrutinees and `for` iterators
  must KEEP their borrows; PAT-BIND-001 binds payloads by reference into the scrutinee. Two
  negative controls fail if anyone tries.
- **DEV-136: treating `loop` without a reachable `break` as diverging.** Needs reachability
  analysis the checker lacks, and guessing lands on the unsound side.
- **DEV-135: parent poisoning (the WP's own §5.4 stage-one).** Ruled out by the §5.3 inventory —
  see §9 below.
- **DEV-138: treating it as an independent defect.** The §9.2 matrix put it inside DEV-121.

---

## 5. New invariants and canonical authorities

Each of these replaced a duplicated or half-consulted rule with one authority:

| Authority | Replaces |
| --- | --- |
| `check_condition` | inline borrow handling in two loop/branch arms |
| `param_declares_bound` | two independent copies of the parameter-bound search |
| `current_generic_env` | a capture that read half the environment |
| `types_equal_inner(.., params_equal_by_name)` | would-be second copy of the structural walk |
| `manifest.json` + `verify.py` | `run-all.sh`'s bare pass/fail |
| `layer_audit.rs` registered dispositions | an unconditional printout |

The recurring motive is DEV-128/DEV-130: *the rule was written twice and the copies drifted.*

---

## 6. Tests added

| File | Cases |
| --- | --- |
| `dev134_try_error_type.rs` | 16 (7 reject, 9 accept) |
| `dev135_field_move_paths.rs` | 16 (6 reject, 10 accept) |
| `dev136_terminating_path_moves.rs` | 14 (5 reject, 9 accept) |
| `dev137_while_condition_borrows.rs` | 16 (4 reject, 12 accept) |
| `dev138_iterator_item_representation.rs` | 10 |
| `dev139_impl_generic_bounds.rs` | 16 (6 reject, 10 accept) |
| **Total** | **88 in-tree cases** |

Plus 39 external-suite cases in `manifest.json`, and `layer_audit.rs` converted from 0 assertions
to 20 registered dispositions.

For four of the six, the **accepts outnumber the rejects** — deliberate. Each repair either widened
what the compiler accepts or widened an environment it consults, so over-acceptance was the risk
and the must-pass set is where that risk is pinned.

---

## 7. Commands and results

Per-commit local evidence (targeted suites, package qualification, external suite) is recorded in
each CD entry in `COMPILER-STATE.md` and in each commit message. Aggregate points:

- **Full workspace, milestone 1** (CD-335): cargo exit 0, 149 suites, 2137 passed, 0 failed.
- **Full workspace, milestone 2** (CD-337): cargo exit 0, 151 suites, 2167 passed, 0 failed.
- **Ten first-party packages**: qualified at every commit, exit 0.
- **External suite**: 34/34 via `run-all.sh` at every commit; 39/39 via `verify.py` from CD-341.
- `cargo fmt --all -- --check`: clean at every commit.

Milestones 3 and 4 were **dropped by owner ruling** mid-programme after measurement: each run costs
~17 minutes and duplicates a gate CI already enforces. See §11.

---

## 8. Package qualification result

All ten first-party packages qualify at every commit
(`qualify-first-party-packages.py`, exit 0): `stark-json`, `stark-url`, `stark-base64`,
`stark-hex`, `stark-uuid`, `stark-form`, their consumers, and the P1 REST workload path.

---

## 9. External sample-suite result

39/39 cases match the manifest. The suite is now `navraj007in/stark-samples`, pinned by SHA in CI,
run against built release artifacts rather than `cargo run`.

**All six `defects/` reproducers now do the OPPOSITE of what their file headers describe** — the
header documents the defect, the manifest documents the fix. That inversion is checked: an
unexpected PASS fails `verify.py`, because a reproducer that silently starts working means an
expectation went stale rather than that the suite is healthy.

**The §5.3 inventory result, recorded because it changed the programme.** Parent poisoning was
ruled out on evidence, not preference: sibling-after-partial-move is asserted as REQUIRED by
`gate2-valid/18_partial_moves.stark`, `mir_verify`, `mir_differential`,
`three_engine_differential`, `native_c5_3_aggregates_enums`, and the C6 corpus. One of those tests'
own doc comments already rejected the approach: *"under a whole-local approximation the sibling
read would find a dead slot and abort."* All 16 `let x = y.field;` sites in first-party packages
read `Copy` scalars, so no shipped package depended on the behaviour either way.

---

## 10. Residual open limitations

**Nothing from CD-334 remains open.** What follows was found *during* the programme and is
registered rather than repaired.

- **DEV-121 stays OPEN, and its blind spot is now named.** INV-VALUE-REP-001 checks **`let`
  bindings**; a for-loop binding is not a `let`, so no loop item is covered. Both known instances
  — `String::bytes()` (CD-305) and `String::split()` (CD-340) — were reachable through a loop item
  and both were found by a user-facing program rather than by the invariant. Extending it to loop
  bindings and call arguments is what closes the class; finding a third instance by hand is not.
- **DEV-140 … DEV-145** — six reachable lowering refusals, registered at CD-342, unscheduled.
  Disposition is per-site, not global: raise into semantic analysis (E0105) or teach lowering the
  construct (DEV-132/133). CD-294 is the precedent for why raising is not always cheap.
- **DEV-083** remains OPEN. Adjacent to DEV-139 but a different mechanism — impl-head *matching*,
  not impl-head *bounds being read*.
- **`types_equal` has no `Ty::Param` arm.** Symptomless in its current callers; unowned; gets a
  number if a symptom appears.
- **`?` conversion semantics** — a language-design question, not a defect. Unowned.
- **CD-269 is cited in five documents but absent from `COMPILER-STATE.md`.** A pre-existing ledger
  gap, noted at CD-338 and not silently patched.

---

## 11. Release recommendation

**Recommend release, conditional on the CI runs for CD-340/341/342 completing green.**

WP §15's gate, item by item:

| Requirement | State |
| --- | --- |
| DEV-134 closed | yes (CD-335) |
| DEV-137 closed | yes (CD-336) |
| DEV-136 closed | yes (CD-337) |
| DEV-135a closed, or DEV-135b complete because inventory proved poisoning unacceptable | **second branch** — inventory ruled poisoning out; the precision it would have built already existed (CD-338) |
| aggregate CI green | **partially verified — see below** |
| external suite runs from its own pinned repository in CI | yes (CD-341) |
| no unregistered soundness defect from CD-334 | yes — all six closed |

DEV-139 and DEV-138, which §15 wants closed before external users, are both closed.

### The honest qualification on CI

CI is the **sole** workspace authority since CD-337 dropped local workspace runs. As of writing:

```
d83acd8  CD-335  CI success
f35ec95  CD-336  CI success
949aa4f  CD-337  CI NOT GREEN  -- failed clippy::collapsible_match, then superseded
309f7ee  CD-338  CI success    -- carries the clippy repair
f69614e  CD-339  CI success
53c1cce  CD-340  in progress
0f23991  CD-341  in progress
e7be5c2  CD-342  queued
```

**CD-337 never went green.** It failed `clippy::collapsible_match`; the fix landed in CD-338, and
every commit from there is green, so DEV-136's code is transitively covered by green runs. It is
stated rather than smoothed over because "aggregate CI green" is a release-gate item.

**Root cause of the miss, which matters more than the lint.** The repo pins `channel = "stable"`;
CI's stable resolves to **1.97.0** and this machine's had gone stale at **1.93.0**. Every "clippy
clean" reported before CD-338 was against an OLDER lint set than CI's. The gate is now
`cargo +1.97.0 clippy`. This is disproportionately important precisely because CD-337 made CI the
sole workspace authority — a local gate that silently differs from CI undermines that arrangement.

**Release should not be declared until CD-340, CD-341 and CD-342 report green.** CD-342 in
particular adds a new required job to `ci-complete`, and CD-341 adds another; neither has yet been
observed passing in CI.

---

## Appendix — process failures in this programme

Recorded because the closure law in §17 asks what future programs are prevented, and the same
question applies to method. Three measurement errors, none in the compiler, all mine:

1. **A `head -40` truncated a workspace run** and made `head`'s exit code look like cargo's. The
   run covered under a third of the suite and was reported as progress. Fix: capture the tool's own
   exit code explicitly; validate suite counts against the known target count.
2. **A `grep '^test result'` counted tests NAMED `result_*`** as suite summaries, producing a
   phantom "3 failures". Fix: anchor on `^test result: `.
3. **A `cargo build` for the next defect landed mid-workspace-run**, and ~49 test files invoke the
   compiler through `CARGO_BIN_EXE`, so an unknown number of suites ran against a binary carrying
   an unrelated change. That milestone was discarded and re-run from a verified single-defect tree.
   Fix, now written into the evidence policy: while a workspace run is in flight, nothing else
   touches cargo.

A fourth, different in kind: an empty log was read as a passing gate. `verify.py` and every
watcher now key on an explicit exit marker rather than the absence of error lines.
