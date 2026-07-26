# WP-C6.5 — Full Differential and Generated Corpus

**Track:** Gate C6 (all of C6 is Claude-owned)
**Status:** `PARTIAL` — C6.5-0 (re-pin, inventory, coverage matrix) complete; C6.5-1 at the plan's
commit-2 boundary (comparator extracted, no observation-shape change yet). Two findings raised and
dispositioned by the owner: **C65-F1** (the comparator was forked 23 ways — CD-148) and **C65-F2 /
DEV-111** (the entry contract diverged in all three engines; MIR fixed, native escalated — CD-149).
**Authority:** `starkc/docs/WP-C6-ENTRY.md` §§38–45 (tracked, normative); inherited scope from
`WP-C2.12`.
**Execution plan:** `WP-C6.5-Full-Differential-Generated-Corpus-Execution-Plan.md` (repo root,
untracked owner draft). Section references below of the form §N.M cite that plan; §§38–45 cite the
entry document.
**Matrix:** `C6-CORPUS-COVERAGE-MATRIX.md` (this directory).
**Predecessor:** WP-C6.4, `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS` (CD-146). Its matrix row 24
is the handoff this package must satisfy.

---

## 0. Baseline packet (§3.1, §7.1)

| Item | Value |
| --- | --- |
| Baseline commit | `b0d7a72` — the plan's planning baseline `61008f6` had advanced by six commits and is superseded, per §3.1 |
| Tracked worktree | clean |
| Untracked | 5 owner side files, none in the qualified path, left alone |
| CI at baseline | green — run 30192715611, all 11 jobs across linux-x64, macos-arm64, windows-x64 |
| Host | macOS 26.5.2, arm64, `aarch64-apple-darwin` (Tier 1) |
| rustc / cargo | 1.93.0 (254b59607 2026-01-19) / 1.93.0 (083ac5135 2025-12-15) |
| Python | 3.14.4 |
| `MIR_VERSION` | 0.1 |
| `MIR_RUNTIME_SURFACE` | 0.1-A9 |
| `BACKEND_VERSION` | 0.1 |
| `RUNTIME_VERSION` | 0.1 |
| Layout contract | `stark-64-v1`, version 1, compiler revision 1 |
| Frozen exec corpus | v1.4.0 — **inherited, preserved, not absorbed** (§3.2) |
| C6.4 status | `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`, accepted CD-146 |

### 0.1 V0 baseline measurements (§5.1)

| Command | Result | Ignored | Self-skipped |
| --- | --- | --- | --- |
| `cargo test --test exec_snapshots` | 4 passed | 0 | 0 |
| `cargo test --test mir_differential` | 132 passed | 0 | 0 |
| `cargo test --test three_engine_differential` | 88 passed | 0 | 0 |
| `cargo test --test c64_platform_matrix` | 15 passed | 0 | 0 |
| `cargo fmt --all -- --check` | clean | — | — |

Per §5.1 the full workspace was not re-run locally: CI already carries stronger exact-commit
evidence for `b0d7a72` (three platforms, `--all-targets --all-features`), and repeating a weaker
single-platform version of it is not evidence.

---

## 1. Inherited asset inventory (§3.2, §3.3)

### 1.1 Frozen execution corpus — preserved as a distinct artifact

`starkc/tests/exec_snapshots/`, corpus v1.4.0, hash-locked: **26 primary cases**, 25 snapshots,
**7 metamorphic pairs** (14 files). It is referenced by the C6 matrix by case ID and is **not**
absorbed, rewritten or relabelled as the C6 generated corpus. The new corpus gets its own manifest,
generator version and lock (§9.5: "Do not reuse `exec_snapshots/corpus.lock`").

The 7 inherited metamorphic pairs already cover seven of the twelve required families:

| Inherited pair | Family |
| --- | --- |
| `alpha_base` / `alpha_renamed` | M01 alpha-renaming |
| `scopes_base` / `scopes_wrapped` | M02 harmless scope insertion |
| `generics_explicit` / `generics_inferred` | M03 explicit vs inferred generics |
| `trait_call_qualified` / `trait_call_operator` | M04 qualified vs unqualified trait call |
| `field_init_shorthand` / `field_init_explicit` | M05 shorthand vs explicit fields |
| `pattern_nested_match` / `pattern_sequential_match` | M06 equivalent pattern decomposition |
| `match_order_ascending` / `match_order_scrambled` | M07 non-overlapping arm reorder |

M08–M12 (relocation, dependency reorder, helper extraction, function value vs direct call,
equivalent loop forms) have **no** inherited pair. Each family also needs a second group (§13.2).

### 1.2 Differential harnesses

- `exec_snapshots.rs` — HIR snapshots over the frozen corpus, 4 tests.
- `mir_differential.rs` — HIR vs verified MIR, 132 tests.
- `three_engine_differential.rs` — HIR/MIR/native, 88 tests at the baseline (89 after CD-148 adds
  the O13 case): **83 from a `three_engine_test!`
  macro** plus 5 explicit tests. The comparator is `three_engine()`, `compare_outcomes()`,
  `parse_native_trap()`, `agree_completing()`, `agree_trapping()`.

---

## 2. Finding C65-F1 — the comparator is already forked 23 ways *(scope-affecting, raise before C6.5-1)*

§3.3 says "do not build a fourth unrelated comparator" and §8.2 frames C6.5-1 as a **mechanical
extraction** of the existing three-engine normalization out of one file. The tree does not match
that premise, and the difference changes the size of the phase.

**Measured at `b0d7a72`: 23 test files run all three engines, each with its own comparison logic.**

```
c62b_f2_specific_instance   c62b_f5_impl_bounds        c62b_f6_self_normalisation
c62c_associated_types       c62d_operator_coretrait    c63a_string
c63b_trapping_ops           c63b_vec_box               c63c_iterators
c63d_map_key_identity       c63e_float32               c63e_formatting
cd139_float_division        native_c5_4_workspace      native_c6_1_ownership
native_c6_2_generics_traits native_c61f_aggregates     native_c61f_b2_weakening
native_c61f_b3_stored_refs  native_c61f_nominals       native_c61f_reborrow
native_c61f_ret_refs        three_engine_differential
```

They share a shape without sharing code — each has a local `run_case`-style helper asserting HIR
status, then MIR status, then HIR/MIR output equality, then native. Nothing calls
`three_engine_differential.rs`'s comparator; it is one of twenty-three, not the authority.

**Why this matters rather than being tidiness.** Every C6.3/C6.4 claim about collections, strings,
formatting, iterators, ownership and generics rests on one of these local helpers. They were each
written to the standard of their own work package, so what any given one checks is whatever that
package needed — and the union of twenty-three ad hoc definitions of "the engines agree" is not a
definition. C6.5's required claim (§2) is precisely that the three engines produce the same
**normative observations**; that claim cannot be made from twenty-three private notions of
observation. Concretely, none of them observes the §39 shape: no stderr bytes, no returned
observation, no explicit Drop log.

**Consequence for C6.5-1.** §8.2's "mechanical extraction, no behaviour change, one shared-file
lease" is right in spirit and wrong in size. The honest options:

1. **Extract the `three_engine_differential.rs` comparator, adopt it there, and migrate the other
   22 suites incrementally** as C6.5 touches each category. Smallest first commit; leaves forks
   alive meanwhile.
2. **Extract and migrate all 23 in one slice.** Largest single commit in C6 so far; every C6.1–C6.3
   suite changes at once, and a regression in the shared helper would surface everywhere
   simultaneously — which is an argument both for and against.
3. **Extract, and require new C6.5 cases to use it, leaving inherited suites untouched.** Cheapest,
   and the one that leaves the required claim resting on evidence the shared comparator never saw.

I recommend **(1)**, and I record the choice here rather than making it silently: the coverage
matrix already cites which suite supplies each row's evidence, so migration order follows the matrix
rather than file convenience. This is flagged for the owner before C6.5-1 begins because it changes
the phase's cost, not merely its shape.

**Owner disposition — CD-148, 2026-07-26: option (1).** Extract, adopt in
`three_engine_differential.rs`, migrate the remaining 22 suites in coverage-matrix order as C6.5
touches each category. The forks stay alive in the interim and are all retired before closure; a
suite still on its own helper is not evidence for the required claim until it is migrated, and §22's
closure checklist is read that way.

---

## 3. Phase C6.5-0 exit (§7.5)

| Exit condition | State |
| --- | --- |
| matrix covers all §40 categories | 133 rows at phase exit; **not met as stated** — the entry contract (PROC-MAIN-001/PROC-EXIT-001) had no row, added as K15–K17 under CD-149, now 136. See §6.5 |
| every row has a normative citation or justified non-Core classification | yes |
| current evidence linked by exact test/case ID | yes — rows cite frozen-corpus case IDs, `three_engine_test!` case names, and C6.x suite names |
| every gap has an execution disposition | yes — one of §7.4's nine classifications |
| no category silently omitted | **no, as it turned out** — the entry contract was omitted, found by running it (§6) |

**Recorded now, so it is not discovered late:** the matrix's dispositions are what the rest of the
package executes against. `ADD-HANDWRITTEN` rows are the C6.5-3 worklist, `ADD-GENERATED` the
C6.5-4 template targets, `ADD-METAMORPHIC` the C6.5-6 families, and `ADD-MUTATION-WITNESS` the
C6.5-7 witnesses. A row moved to `NOT-APPLICABLE-NON-CORE` after this point needs the §4.3
justification recorded, not an edit.

---
## 4. Phase C6.5-1 — comparator extraction (§8, commit 2)

**Done, mechanically, at the plan's §19 commit 2 boundary: no observation-shape change.**

`starkc/tests/support/differential.rs` is now the comparator authority. It carries — verbatim, made
`pub` — the engine runners (`run_hir`, `run_mir`, `run_native`), the normalisation
(`oracle_category`, `runtime_category`, `parse_native_trap`), the comparator (`compare_outcomes`),
the case entry points (`three_engine`, `agree_completing`, `agree_trapping`,
`agree_trapping_with_message`) and the `three_engine_test!` macro. `three_engine_differential.rs`
keeps its case declarations and the comparator's own negative tests, and nothing else.

Consumers include it with `#[macro_use] mod support;` — the repo's existing `tests/common/mod.rs`
convention. The macro refers to the module by absolute path, so a migrating suite needs that one
line rather than an import list.

| Check | Result |
| --- | --- |
| `cargo test --test three_engine_differential` at the extraction commit `c789e4b` | 88 passed, 0 failed, **0 ignored**, 0 self-skipped — identical to the V0 baseline, which is the point of a mechanical move |
| the same command after §5's O13 case lands | 89 passed, 0 failed, 0 ignored, 0 self-skipped |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --tests` | clean |
| Behaviour change | none — the diff moves code and adds `pub` |

**Not yet done in this phase:** §8.3's full observation shape, §8.5's exhaustive trap-stderr
normalizer, §8.7's returned-observation protocol, §8.8's Drop-log protocol and §8.10's seventeen
comparator unit tests. Those are commit 3, deliberately separated so that a later disagreement is
attributable to the extension and not to the move. The 22 unmigrated suites remain forked.

---

## 5. Matrix amendments under CD-148

Both of the matrix's owner-flagged rows moved, in opposite directions. Neither was settled by
reading the ledger.

**O13 (non-Copy array iteration) — `BLOCKED` → `EXISTING-EVIDENCE`.** The row inherited CD-038's
"narrowed, not closed": a runtime loop index names no `ConstIndex`, and reading by copy would double
free. CD-038 also recorded what would close it — "unrolling or runtime-indexed drop flags" — and
**WP-C6.1d took the unrolling option** (CD-084 G2, closing DEV-090). Two ledger records, the older
one inherited. Settled by execution: `o13_non_copy_array_by_value_iteration_agrees` pins stdout to
`"idid\n"` independently of the engines, so a wrong Drop schedule fails even under unanimous
agreement. All three engines produce it. **This is the shape §3.6 exists to prevent and it was
pointing at the wrong row** — worth noting, because the matrix's other 132 dispositions were built
the same way, from records rather than from runs, and C6.5-5's replay is what re-derives them.

**V19 (`HashSet<T>`) — `NOT-APPLICABLE-NON-CORE` → `BLOCKED-BY-OTHER-C6-WP`.** §4.3(1) requires
genuine absence from normative Core v1. `HashSet` is specified in 06-Standard-Library and named in
`std-full`; row V18 covers `HashMap`, equally `std-full`, as existing evidence; and CD-142's own
words call the exclusion "a lowering gap like C6.3c's adapters" — precisely the reason §4.3's
closing line forbids. `c63d_map_key_identity::hashset_is_hir_only` pins the boundary and says
outright *"if it now lowers, promote it to a three-engine case"*. It is a C6 blocker held for a
lowering package, not a corpus exclusion.

The blocker count is unchanged at one. Which row it is, is not.

---

## 6. Finding C65-F2 — the entry/exit contract diverges in all three engines (DEV-111)

Found while building §8.3's `stderr_bytes` field, by asking what each engine does with a program
whose `main` returns something. Recorded here **before any compiler change**, per §18.3.

### 6.1 The §18.1 record

| Field | Value |
| --- | --- |
| case_id | `entry_exit__01..04` (retained as `starkc/tests/c65_entry_exit_contract.rs`) |
| seed / generator_version / template / dimensions | n/a — hand-written probe, not generated |
| category | packages/environment (entry contract) and traps |
| normative rules | **PROC-MAIN-001**, **PROC-EXIT-001** (07-Modules-and-Packages, "Executable and target contract") |
| first differing field | `exit_status`, then `stderr` |
| platform | macOS 26.5.2 arm64, `aarch64-apple-darwin` |
| commit | `b7e804a` |
| reproduction | `cargo test --test c65_entry_exit_contract` |

| Program | PROC-EXIT-001 requires | HIR oracle | MIR | native |
| --- | --- | --- | --- | --- |
| `main -> Result<Unit, String>` returning `Err("boom")` | status 1, `boom\n` on stderr | status 1 + stderr — correct | **status 0, no stderr** | **build refused** |
| `main -> Int32 { 3 }` | status 3 | status 3 — correct | **status 0** | **build refused** |
| `main -> Int32 { 300 }` | trap `invalid-exit-status` | traps — correct | **completes, status 0** | **build refused** |
| `main()` returning `Unit` | status 0 | correct | correct | correct |

### 6.2 Authority (§18.2)

**MIR is wrong**, and the HIR oracle is right. `run_program` matches `Ok(_)` on the entry call and
hardcodes `status: 0`, so the entry's return value is discarded; `MirExecution` has no `stderr`
field at all. Two of the three rows are wrong output and the third is a **missed trap** — §18.4's
first and second high-priority classes.

**Native is a Gate C6 blocker, not a C6.5 defect.** `Unsupported("the entry instance must return
Unit to become Rust's fn main()")` refuses a program PROC-MAIN-001 declares a legal executable
target. That is precisely "a C5-style unsupported profile remaining for normative executable Core",
which `WP-C6-ENTRY.md` §3 lists as **required result 6** for closing Gate C6. Escalated by owner
decision rather than built inside a corpus package.

### 6.3 Two escalations this finding produces

1. **The `invalid-exit-status` trap has no category.** PROC-EXIT-001 requires a language trap for an
   out-of-range status. The nine `TrapCategory` values contain nothing for it, the HIR oracle raises
   it as an uncategorised `RuntimeError`, and `oracle_category` therefore cannot normalise it — the
   comparator would fail on the *message*, not the semantics. Adding a category is a **CE3**: trap
   identity is one of the contracts WP-C6.0 froze. Until it is decided, MIR raises a loud
   `Internal` error there rather than silently completing with status 0.
2. **The Unit value was unwritable — DEV-112, and NOT a governance question.** The checker rejected
   `let x: Unit = ();` with E0001 *"type mismatch: expected 'Unit', found '()'"*, and `Ok({})` fails
   at lowering, so PROC-EXIT-001's `Ok(Unit)` clause could not be expressed in source at all. I first
   recorded this as a spec-vs-implementation conflict needing a decision. **That was wrong, and the
   correction matters:** TYPE-PRIM-001 states outright that *"`Unit` and `()` are two spellings of
   the same single-inhabitant type"*, and 03-Type-System repeats it in the tuple rules ("`()` is
   `Unit`"). The specification is unambiguous, so this was a plain conformance bug in the checker.
   Fixed under CD-150 — see §6.4.

### 6.4 Disposition — CD-149 and CD-150

**MIR fixed.** `run_program` now derives termination from the value `main` returned
(`entry_termination`, `mir/interp.rs`), and `MirExecution` gained the `stderr` field the oracle's
`Execution` has carried since Phase 4E. `Int32` → that status, `Ok(Int32)` → that status,
`Err(message)` → status 1 with `message` + LF on stderr. Not a contract change: `MirExecution`
appears nowhere in `mir.md` — the same test CD-084 applied to `FnKey` — and no MIR shape,
`RuntimeFn` or runtime-surface version moved.

**Native escalated**, per §18.5's stop-and-escalate list: a backend that can emit a non-`Unit` entry
is a feature build, and it belongs to a decision of its own rather than to a corpus package.

**The trap CE3 is bundled with the native entry work (CD-150).** The backend increment that emits a
non-`Unit` entry has to emit the `invalid-exit-status` trap anyway, so the `mir.md` amendment, the
implementation and the three-engine evidence are one package rather than three. Nothing is lost by
waiting: the case is pinned by a test that fails the day either half lands. Meanwhile MIR fails
loudly there rather than completing with status 0.

**DEV-112 fixed (CD-150) — `()` now typechecks as `Unit`.** Canonicalised at construction in all
three engines rather than taught to `unify` as an equivalence, so the two spellings are *one type* as
TYPE-PRIM-001 says, and `Ty::Tuple([])` is no longer constructible from source: `unit_or_tuple` in
the checker, `Constant::Unit` in `mir/lower.rs`, `Value::Unit` in the oracle. The fix had to reach
all three — fixing only the checker produced `MIR-0004 "aggregate Tuple assigned to incompatible type
Unit"`, and fixing checker + lowering left the oracle's `Ok(Tuple([]))` failing
`main_result_to_status`. Each engine's disagreement surfaced as its own failure, which is the
argument for the entry-contract cases running both interpreters rather than one.

**Retained** as `starkc/tests/c65_entry_exit_contract.rs`, **8 tests**: five two-engine cases
comparing every PROC-EXIT-001 field against the normative answer stated independently — including
`Ok(Unit)`, the clause DEV-112 had made unreachable — plus the `Unit`-literal case, and two boundary
tests that pin the remaining escalations *and name the condition that retires each*. If native starts
accepting a non-`Unit` entry, or the trap gains a category, the corresponding test fails and says
what to do. A boundary test that silently keeps passing after its boundary moves is how O13 became
stale.

### 6.5 What it says about the matrix

**PROC-MAIN-001 and PROC-EXIT-001 appear in none of the 133 rows.** Exit status is covered only as
X12 (exit 101 after a trap); normal nonzero statuses, the `Err` stderr write, and the entry-signature
set are absent. The §7.5 exit condition "no category silently omitted" was therefore not met when
phase 0 was declared complete. Rows **K15–K17** are added for the entry contract, and the omission
is recorded rather than quietly backfilled: two of the matrix's inherited dispositions have now
failed on contact with an actual run (O13, and this), which is the argument for C6.5-5's replay
re-deriving all of them.

---

## 7. What comes next

§19's commit 3 — the §8.3 observation model, the §8.5 trap-stderr normalizer, the §8.7 returned-value
and §8.8 Drop-log protocols, and the §8.10 comparator unit tests that prove each new field is
load-bearing. Migration of the 22 remaining forked suites then proceeds in matrix order under
CD-148's option (1).

Status remains `PARTIAL` until the comparator, matrix, manifest, generator, replay, metamorphic and
mutation requirements are all complete (§23). C6.5 has **no** valid "candidate complete but corpus
blocked" endpoint — producing the corpus is its central deliverable.
