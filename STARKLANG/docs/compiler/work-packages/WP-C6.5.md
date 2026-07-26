# WP-C6.5 — Full Differential and Generated Corpus

**Track:** Gate C6 (all of C6 is Claude-owned)
**Status:** `PARTIAL` — C6.5-0 (re-pin, inventory, matrix) and **C6.5-2 (manifest, layout, lock)**
complete; **C6.5-1 complete** (comparator extracted and extended to the full §39 observation shape,
commits 2 and 3) except for migrating the 22 still-forked suites, which proceeds in matrix order.
**C6.5-3 PARTIAL** (§10.3 sentinels done; per-row witnesses, trap balance and package breadth
outstanding — §8.1); **C6.5-4 complete** (deterministic generator, §10.1 lists its two residuals);
**C6.5-5 complete** (full replay with evidence, classifications, timeouts, sharding — §11.1 lists its
residuals); **C6.5-6 PARTIAL** (20 metamorphic groups over ten families; M08/M09 blocked on package
graphs — §12.1); **C6.5-7 complete** (all 16 mutations detected); **C6.5-8 PARTIAL** (package cases, staged package
replay, relocation and reorder measured; **DEV-113** and **DEV-114** found — §14.1); **C6.5-9
MACHINERY ONLY** (Tier-1 jobs, §16.2 identity, the §16.4 comparator and §20.7's controls; row 24
deliberately NOT flipped — §15.1). Corpus `0.5.0`: 131 cases, replay **131/131 AGREEMENT**. Two findings raised and
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
## 4. Phase C6.5-1 — the comparator authority (§8, commits 2 and 3)

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

### 4.1 Commit 3 — the §39 observation model

**Done.** `Outcome { stdout, exit }` is replaced by the full shape, and every field participates in
equality:

```text
Completed { stdout_bytes, stderr_bytes, exit_status, returned_observation, drop_log }
Trapped   { category, source_file, line, column, message_class, stdout_before_trap,
            stderr_observation, exit_status, drop_log_before_trap }
```

| §8 requirement | How it is met |
| --- | --- |
| §8.4 bytes, not host strings | native output stays `Vec<u8>`; interpreter `String` channels convert with no line-ending translation; only protocol frames are decoded as text |
| §8.5 trap stderr normalised | *parsed* from native stderr, *constructed* for the interpreters from `stark_runtime::trap`'s own category table — the same source the native ABI prints from. Exhaustive over `TrapCategory` by an exhaustive `match`, so a tenth category fails to compile until mapped |
| §8.6 message classes | `CategoryOnly` / `UserMessageExact` / `RuntimeCompatibility`; a runtime-version mismatch is raised as a harness failure, never classified as a program trap |
| §8.7 returned observation | `fn probe() -> T` plus a generated wrapper emitting `@@stark-ret:<tag>:<rendered>@@`; frame stripped from normative stdout; a probe that also prints fails |
| §8.8 Drop log | user `Drop` impls emit `@@stark-drop:<identity>@@`; frames extracted in order, sequence assigned by position, stripped from stdout; duplicate identities and mid-line frames are hard failures |
| §8.9 invariants | `first_difference` names the field, so a failure says *which* normative dimension disagreed rather than dumping two structs |
| §8.10 comparator tests | **18**, one per listed dimension |

**Two deliberate deviations from the plan's sketch, both recorded rather than silent.**

1. **The sentinel is `@@`, not `##`.** A case source is a Rust raw string in the test file, and `"##`
   terminates `r#"…"#` — with `##` every drop-observing case would have to remember `r###"`. The
   sentinel is arbitrary; the friction would not have been.
2. **Return frames are marker-delimited, not length-delimited.** §8.7 asks for a length prefix.
   Core v1 source cannot compute the byte length of an arbitrary `Display` rendering, so the frame is
   delimited by reserved markers and the probe is *required* to emit no other stdout —
   `agree_returning` asserts that, so the ambiguity the length prefix was there to prevent fails
   loudly instead.

**Evidence:** `three_engine_differential` **109 passed / 0 failed / 0 ignored / 0 self-skipped** (was
89: +18 comparator tests, +2 framed-probe cases, +1 Drop-log-before-trap case, O13 converted to the
protocol). `fmt` clean. The 18 comparator tests are what make the new fields load-bearing rather than
merely present: each perturbs exactly one field of an otherwise-agreeing triple and requires the
comparator to reject it, naming both the disagreeing pair and the field.

Three of them are worth calling out because they cover the failure modes stdout comparison cannot
see: **Drop reversal** (same identities, same count, only the order differs), **pre-trap Drop change**
(DROP-ABORT-001 says destructors do not run after a trap, so the retained log is itself an
observation), and **internal MIR error** — which runs a real `fn main() -> Int32 { 300 }`, DEV-111's
escalated case, and requires the harness to fail loudly rather than report a completion.

**Not yet done in this phase:** migration of the 22 remaining forked suites, which proceeds in
coverage-matrix order under CD-148's option (1). Until each is migrated its C6.2/C6.3 evidence still
rests on its own local notion of agreement.

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

## 7. Phase C6.5-2 — corpus manifest, layout and lock (§9, commit 4)

**Done.** `starkc/tests/c6-corpus/` exists with the §9.1 layout, a strict manifest, a generated lock,
and 28 tests — 3 on the real corpus, 25 proving the validator refuses what §9.3 says it must.

**Parser choice (§9.4).** Option 2: a deliberately small strict reader for the manifest subset
(`tests/support/corpus.rs`). Option 1 was checked first and does not apply — the workspace has no
TOML parser to reuse, and §9.4 forbids adding a network-fetched dependency to parse a test manifest.
The subset is `[[case]]` headers plus `key = "string" / ["a", "b"] / true`, with **unknown keys
rejected**: a parser that skips what it does not understand turns a typo'd attribute into an
attribute nobody checks.

**The corpus is seeded, not empty.** Six `retained` cases — DEV-111 and DEV-112's entry-contract
sources — because §18.3 requires a retained case to remain a permanent regression, and because a lock
that has never hashed a real file proves nothing about the lock. `c65_entry_exit_contract.rs` now
reads them with `include_str!`, so the corpus source and the expectation cannot drift: editing a case
changes its hash in `corpus.lock` and the assertion that pins its observation, in one change.

Two entry-contract programs are deliberately not cases, and the README says why: the out-of-range
status (no replayable observation until the CE3 lands) and the pre-DEV-112 `()` rejection (history,
not a case).

**Quarantine is unwritable where §4.4 forbids it.** The validator accepts only three reason classes —
`non-core-feature`, `external-artifact`, `environment` — each requiring a `CD-###` authority. There
is no spelling for "the engines disagree", "wrong output", "wrong Drop order" or "native refuses an
accepted program": those are C6 blockers that keep the gate open, and the test
`semantic_quarantine_rejected` proves the door is shut rather than merely discouraged.

**Lock integrity (§9.5).** Per-source SHA-256 plus manifest and generator hashes and five counts;
`generate.py --lock` writes it and `--check` is what CI asks. The generator hashes ITSELF into the
lock, so changing how the corpus is produced invalidates it. The version assertion in
`c6_corpus_manifest.rs` is the deliberate speed bump: regenerating is easy, and doing it without a
`corpus_version` bump fails, so no edit can quietly redefine the baseline later claims are measured
against.

| Check | Result |
| --- | --- |
| `cargo test --test c6_corpus_manifest` | 28 passed, 0 failed, 0 ignored |
| `cargo test --test c65_entry_exit_contract` | 8 passed (now reading corpus sources) |
| `python3 tests/c6-corpus/generate.py --check` | `corpus.lock is current` |
| `cargo fmt --all -- --check` | clean |

`corpus_version` **0.1.0**; `generator_version` 0.1.0; case_count 6 (0 handwritten, 0 generated,
6 retained, 0 metamorphic groups). The generated corpus §11 requires — ≥64 cases across ≥10
templates — is still entirely unbuilt; this phase built the thing that will hold it.

---

## 8. Phase C6.5-3 — hand-written completion (§10, commit 5) — **PARTIAL**

**Done: the thirteen §10.3 sentinels, and a bridge that runs every corpus case.**

`corpus_version` **0.2.0** — 19 cases: 13 handwritten sentinels, 6 retained. Each sentinel is built so
the *likely wrong* implementation fails it, which §10.3 states as the bar ("a case that would still
pass under the likely wrong implementation is insufficient"):

| Sentinel | The wrong implementation it catches |
| --- | --- |
| `Eq` always true | structural key comparison in a `HashMap` — CD-133's live defect |
| reverse `Ord` | comparing fields directly instead of through the user's `cmp` |
| constant `Hash`, distinct `Eq` | treating equal hashes as equal keys; hash/sorted iteration order |
| `Display` unlike layout | a structural/debug rendering fallback |
| `Clone` changing a marker | clone as a structural copy |
| non-zero `Default` | zero-initialisation |
| two generic instances | monomorphising once and reusing the body |
| two trait impls | picking the first matching impl |
| two function-value targets | resolving an indirect call statically |
| slice mutation through a view | copying elements into the view (§18.4's "slice copy instead of view") |
| insertion order ≠ sorted order | sorting, or iterating hash buckets |
| Drop identities | declaration-order, omitted or duplicated destruction |
| `Float32` rendering | carrying f32 arithmetic at f64 width — DEV-109's defect |

**Every sentinel pins its observation in the manifest** (`expected_stdout` / `expected_drop_log`), and
a test enforces that it does. This is the phase's central point: a wrong implementation is usually
wrong in *all three engines at once*, and those agree perfectly. Three-engine agreement alone would
pass every sentinel above. The pinning is what converts agreement into evidence — and it is not
theoretical: the `Float32` sentinel failed on first run against a wrong expectation of mine, which is
the mechanism working.

**`c6_corpus_cases.rs`** runs each case on the engines its manifest entry declares — three-engine
where native builds it, two-engine for the DEV-111 entry cases the native backend refuses. Deliberately
NOT §12's replay harness (commit 7, which adds admission classification, timeouts, sharding, filters
and the evidence schema): it exists now so no case is added in a state where nothing runs it.

Two surface findings while writing the cases, both recorded rather than worked around:

1. **`T::assoc()` through a type parameter does not resolve** — `E0200 "undefined variable 'T::tag'"`.
   TRAIT-ASSOC-001 speaks of `T::Item` as a projection for associated *types*; whether an associated
   *function* is callable through a parameter in Core v1 is a spec question, so the sentinel was
   rewritten onto a `&T` receiver and this is flagged, not assumed out of scope.
2. **No implicit array→slice coercion** — `&mut xs[0..2]` is the normative way to take a view. Correct
   as specified; recorded because the first draft assumed otherwise.

### 8.1 What §10 still owes

| §10 requirement | State |
| --- | --- |
| §10.2 a hand-written witness per matrix row | **not started** — 13 sentinels cover a fraction of 136 rows |
| §10.4 completion/trap balance, one direct case per admitted trap category | **not started** — no trap case is in the corpus yet |
| §10.5 package breadth (multi-file, dependency, re-export, relocation, offline, installed runtime, Unicode/spaced paths) | **not started** — every case is currently single-file |
| §10.3 "same filename in different package locations" | **not started** — needs a package graph, so it lands with §10.5 |
| §10.6 phase exit | **not met** |

C6.5-3 is `PARTIAL`. The sentinels were done first because they are the requirement nothing else in
the plan substitutes for, and because the roll-up named "adversarial sentinels: 0" as an outstanding
gap.

---

## 9. Finding C65-F3 — the matrix cited 69 invented rule IDs (CD-154)

Found while choosing citations for the §10.3 sentinels: checking the matrix's rule IDs against the
spec showed that **69 of the 84 distinct identifiers it cited do not exist in any specification
document** — 100 occurrences across ~130 rows. `OWN-DROP-001`, `FN-VALUE-001`, `MAP-001`,
`TRAP-ABORT-001`, `CTRL-IF-001`, `PAT-WILD-001`, `VEC-001`, `SLICE-001`, `REF-001`: all
plausible-looking, all fabricated. The real identifiers are `DROP-EXACT-001`, `TYPE-FN-001`,
`STD-HASH-001`, `DROP-ABORT-001`, `EXEC-EVAL-001`, `SYN-PATTERN-001`, `DROP-COLLECTION-001`,
`REF-SLICE-001`, `REF-IDENTITY-001`.

**This is the worst of the three phase-0 failures, and it is a different kind.** O13's disposition was
a wrong judgement inherited from a stale ledger entry; the missing entry-contract rows were an
omission. This was invented content presented as grounding — and §7.5's exit condition *"every row has
a normative citation or justified non-Core classification"* was recorded as met, because nothing
checked whether the citations resolved to anything. A fabricated citation is worse than a blank one: a
reader who follows it finds nothing, and every reader who does not follow it assumes someone did.

**Repaired.** All 136 rows re-cited against the spec's actual rules, each chosen for what the rule
says rather than for what its name resembles — `break`/`continue`/`return`/`?` to EXEC-CFLOW-001 (they
are one rule about normal control transfer), Drop order to DROP-ORDER-001 and Drop-once to
DROP-EXACT-001, every trap row to TRAP-CATEGORY-001 with DROP-ABORT-001 where the claim is
about post-trap cleanup, `Box`/`Option`/`Result` payload destruction to DROP-ORDER-001's own bullet
list. Two substring collisions the mechanical pass introduced (`PRIM-TRAIT-001` → `PRIM-TRAIT-DEF-001`,
`TEXT-ITER-001` → `TEXT-EXEC-FOR-001`) were caught by re-verifying every ID afterwards rather than
trusting the edit.

**Guarded, so it cannot recur silently.** Two tests:
`every_rule_id_the_matrix_cites_exists_in_the_spec` reads the matrix and fails on any ID the spec does
not define, and the corpus validator applies the same rule to each case's `normative_rules` —
`a_manifest_citing_an_invented_rule_is_rejected` proves that check refuses rather than merely runs.
The authority set is parsed from the numbered source documents only; the generated `STARK-Core-v1.md`
is excluded, so a stale compilation cannot validate an ID the sources no longer define.

**Elsewhere, audited and reported rather than quietly fixed.** The same pattern exists at smaller
scale in closed-gate records: `WP-C3-ENTRY.md` (7 invented IDs, including `STD-ITER-001`,
`STD-OPTION-001`, `STD-VEC-001`), `WP-C1.3.md` (1), `WP-C1.6.md` (2). The `CORE-Q-0##` references in
the WP-C2.x documents are a separate question-numbering scheme, not spec rules, and are fine.
Rewriting closed-gate documents is a governance decision, not a C6.5 edit, so those are named here and
left for the owner.

---

## 10. Phase C6.5-4 — the deterministic generator (§11, commit 6)

**Done.** `corpus_version` **0.3.0** — 89 cases: **70 generated** across **15 templates**, 13
handwritten sentinels, 6 retained. §11.4's floor (≥64 cases, ≥10 templates, completion *and* trap,
full provenance per case) is met and asserted by a test rather than counted by hand.

**Selection (§11.2).** Dimension tuples are enumerated in sorted order, ranked by
`SHA-256(generator_version | seed | template_id | canonical_dimensions)`, truncated to a per-template
budget of 5, and the case ID is the template plus a digest prefix. Nothing host-dependent enters
identity — no filesystem order, PID, timestamp, absolute path or Python-version-dependent
representation. The dimension tuple is canonicalised by an explicit function rather than by `repr`,
which is stable in practice but not contractually.

**Expectations come from the template, not from an engine.** Each generated case carries its
`expected_stdout` (or `expected_drop_log`, or `expected_trap_category`) computed by the template's own
semantic model. This is the same principle as the sentinels and the reason both exist: the corpus
claims the three engines agree with the *specification*, and an expectation read back from one engine
could only ever show that the engines agree with each other.

**Registry (§11.5): 15 of the 20 named families.** T01 arithmetic, T02 comparison/branch, T03 bounded
loop, T04 match/pattern, T05 struct projection, T06 enum payload move, T07 generic instances, T08
trait dispatch, T09 function values, T10 Option/Result with `?`, T11 String/Vec flow, T12 collection
order, T15 Drop order via the §8.8 protocol, T16 every admitted trap category, T20 composite
formatting. **Absent and recorded rather than stubbed** — `--list-templates` prints them with reasons —
T13 (borrow/reborrow/reference return) and T14 (partial move/reinit) are handwritten today; T17/T18/T19
need package graphs, which arrive with §15.

**Valid by construction (§11.7), not by trial.** Each template's dimension space excludes tuples that
would produce an invalid or accidentally-trapping program — unsigned subtraction that would go
negative is filtered out, for instance, because overflow traps and T01 is a completion template. No
case was found by generating and discarding failures. All 70 pass on all three engines.

**Determinism proven by running it (§11.10), 8 tests:** same seed twice byte-identical; relocation to a
different (and deeper) output root identical; pre-existing junk in the output directory irrelevant; a
different seed reselects but stays reproducible and keeps the same count; a *generator version* change
reselects — which is what makes "a version change requires corpus-version review" enforceable rather
than advisory; no absolute path anywhere in the generated corpus; `--check` byte-identical.

**Two parser bugs the generated data found in my own tooling**, both fixed: the manifest list parser
split on `,` and so tore `expected_stdout = ["[1, 2, 3]"]` in half — a rendered array is legitimate
data — and the lock generator referenced a constant I had renamed. The first is worth naming: it was
found by real data, not by review.

### 10.1 What §11 still owes

| §11 requirement | State |
| --- | --- |
| §11.11 retained-case workflow tested with a synthetic failure | **not done** — retention is documented and the retained cases exist, but the `cases/retained/<DEV-ID>/original/` + `reduced/` flow has not been exercised |
| T13/T14 templates | absent; the shapes are covered by handwritten cases |
| T17/T18/T19 package templates | blocked on §15 package graphs |
| package dimensions in §11.4's dimension list | absent for the same reason |

---

## 11. Phase C6.5-5 — the full three-engine replay (§12, commit 7)

**Done.** `starkc/tests/c6_generated_corpus.rs` — the plan's named entry point — loads and validates
the manifest, verifies the lock, enumerates cases in case-ID order, runs each on the engines it
declares, compares observations field by field, checks them against the manifest's own expectations,
and writes §21 evidence. The C6.5-3 bridge (`c6_corpus_cases.rs`) is retired: it ran cases but
produced no evidence, applied no timeout, and could not be narrowed.

**Full-corpus result, this commit:** 89 cases, **89 AGREEMENT, 0 failed**, `full_evidence: true`,
`result: PASS`, in 99s.

Three properties a plain loop would not have:

1. **Failures are classified** (§12.2's ten admissions plus `TIMEOUT`), and the report says outright
   when a classification is a **C6 blocker** — "an accepted Core case refused by MIR/native is a
   blocker" is a rule that only bites if refusal and disagreement look different in the output.
2. **A filtered run cannot be filed as closure evidence** (§12.6). Every narrowing is recorded and the
   summary's `result` reads `PARTIAL-FILTERED`. Sharding counts as narrowing: a shard is complete
   evidence *for the shard*, and evidence for the corpus only once every shard merges.
3. **A timeout is a failure, not a skip** (§12.4). Per-case work runs on a worker thread against a
   120s budget with a 3600s whole-replay ceiling. A hung native binary fails its case with the budget
   named rather than stalling the run — the shape CD-127's infinite-loop miscompile took. The worker
   thread is abandoned rather than killed, which is deliberate and recorded: a leaked thread in a
   failing run is a smaller problem than an unattributable hang.

**Sharding (§12.7)** is content-addressed — `u64(SHA-256(case_id)[0..8]) % total` — not index-based,
so adding one case moves only the cases whose digests demand it instead of reshuffling every shard.
The partition claims are checked over the real corpus at six shard counts: every case in exactly one
shard, none omitted, none duplicated.

**Determinism (§12.8)** is proven by replaying a shard twice and comparing observation hashes. The
hash is over an explicit canonical rendering of the observation, not `Debug` output — `Debug` is
stable in practice but not by contract, and an evidence hash that moved with a Rust release would
invalidate stored records for no semantic reason.

### 11.1 What §12 still owes

| §12 requirement | State |
| --- | --- |
| §12.1 step 4, "construct the package graph" | single-source only; multi-file cases arrive with §15 |
| §12.5 generated-crate path in the divergence report | not emitted — the build directory is removed after each case unless `C6_KEEP_TEMP` is set, and wiring that through the runner is §15 work |
| `C6_KEEP_TEMP` | parsed and recorded, not yet honoured by the native runner |
| §12.7 shard summary merging | shards produce independent summaries; the deterministic merge is CI work (commit 11) |

---

## 12. Phase C6.5-6 — metamorphic families (§13, commit 8) — **PARTIAL**

**Done: 20 groups over ten of the twelve families, 40 member cases.** Corpus `0.4.0`, 129 cases
total; the full replay is **129/129 AGREEMENT**.

| Family | Transformation | Precondition that makes it semantics-preserving |
| --- | --- | --- |
| M01 | alpha-renaming | locals only, no shadowing, no collision with a standard item (NAME-SHADOW-001) |
| M02 | scope insertion | **no `Drop` type in the base** — an inner block ends earlier, so with `Drop` the destruction point moves (DROP-ORDER-001) and a *correct* compiler would disagree |
| M03 | explicit vs inferred generics | the explicit argument is what inference selects (TYPE-GENERIC-001) |
| M04 | qualified vs unqualified trait call | exactly one impl in scope (TRAIT-ASSOC-001) |
| M05 | shorthand vs explicit fields | field names equal local names, declaration order unchanged (EXEC-AGG-001) |
| M06 | nested vs sequential pattern | `Copy` scrutinee, evaluated once either way (PAT-OWN-001) |
| M07 | non-overlapping arm reorder | distinct variants, **no catch-all** (PAT-USEFUL-001, §13.5) |
| M10 | helper extraction | extracted expression is side-effect-free and evaluated once (EXEC-ONCE-001) |
| M11 | direct call vs function value | non-capturing value reaching the same function (TYPE-FN-001) |
| M12 | `while` vs range `for` | body owns and drops nothing, counter is `Copy` (§13.6, DROP-LOOP-001) |

**The preconditions are constraints, not commentary.** Scope insertion is *refused* over a
`Drop`-bearing base by an assertion in the generator, because that transformation is not
semantics-preserving there — the pair would fail against a correct compiler, and "fix it by relaxing
the comparison" is exactly what §13.7 forbids. Arm reordering asserts there is no wildcard.

**Two fake pairs my own generator produced, both caught by its own guard.** `add()` asserts the
transformed source differs from the base, and it fired twice: M12/g2, where a post-hoc
`.replace("total + i", …)` broke the transform's anchor so it returned the input unchanged, and — a
different symptom of the same habit — M05/g2, where a blind `.replace("3", "8")` turned `Int32` into
`Int82`. Both are the same root cause: **generating variants by substring surgery over source.** Every
base is now a parameterised builder. Worth recording because an identity-transform pair passes
trivially and looks like evidence.

**§13.4's comparison is per engine, then across engines:** `HIR(base) == HIR(transformed)`,
`MIR(base) == MIR(transformed)`, `native(base) == native(transformed)`, with three-engine agreement
for both members coming from the §12 replay, which runs metamorphic members as ordinary cases. A
divergence report names the engine, the first differing field, and the precondition — because §13.7
requires normative analysis to decide whether a diverging pair is a compiler defect or an invalid
transformation, and the precondition is where that analysis starts.

### 12.1 The floor is not met, and a test says so

§13.2 requires **24 groups / 48 members across all twelve families**. This delivers 20/40 across ten.
**M08 (workspace relocation) and M09 (dependency declaration reorder) transform a package graph**, and
every corpus case is single-file until §15 — a single-file "relocation" pair would prove nothing about
relocation, so they are absent rather than approximated.

`the_metamorphic_floor_is_reported_honestly` asserts the present state *and* that it is below the
floor, so when M08/M09 become buildable the test fails and demands the expectation be raised. A
shortfall recorded only in prose is a shortfall that gets forgotten.

---

## 13. Phase C6.5-7 — mutation controls (§14, commit 9)

**Done: all sixteen §14.3 mutations detected**, each against a real passing corpus case, each
rejected by the production comparator naming the intended field.

This is the phase that decides whether anything before it means much. Every other phase shows the
corpus and the comparator **agree**; a suite of passing tests cannot distinguish "the engines match"
from "the harness cannot tell them apart". §14 is the negative control for the whole package.

| | Mutation | Witness | Field |
| --- | --- | --- | --- |
| MU01 | wrong arithmetic | a generated T01 case | `stdout_bytes` |
| MU02/03 | wrong trap line / category | a generated T16 overflow trap | `trap line` / `trap category` |
| MU04/05/06/07 | omitted / duplicated / reversed Drop, copied move | the three-event Drop sentinel | `drop_log` |
| MU08/09/10 | wrong generic instance / trait impl / function-value target | the three dispatch sentinels | `stdout_bytes` |
| MU11 | changed collection order | insertion order 30/10/20 | `stdout_bytes` |
| MU12 | slice view became a copy | the view-mutation sentinel | `stdout_bytes` |
| MU13 | `Float32` rendered as `Float64` | the width sentinel | `stdout_bytes` |
| MU14 | generated-Rust path replacing source path | the trap witness, mutated to `src/main.rs` | `trap source_file` |
| MU15/16 | missing output / wrong exit normalisation | the arithmetic witness | `stdout_bytes` / `exit_status` |

**Three rules keep it from being theatre**, each enforced rather than intended: the witness must
agree *before* mutation (a detection on an already-failing case proves nothing); the mutation must
actually change the observation (asserted — the same identity-transform trap the metamorphic
generator hit twice); and nothing is simulated by asserting `false` — the comparator under test is
`compare_observations`, the function the replay itself uses.

**Routing controls (§14.5).** For the two routing-sensitive mutations, mutating an observation is not
enough: it shows the comparator would catch a wrong answer, not that a wrong *route* produces one. So
MU09 and MU12 additionally run the wrong route as a **real program** — calling the other trait impl,
and passing an array by value instead of taking a view — and assert its observation differs from the
witness. Without those, both mutations would rest on my assertion that the sentinel discriminates.

**One gap, recorded rather than papered over.** `returned_observation` has no corpus witness: the
§8.7 framed-probe cases live in `three_engine_differential.rs`, not the corpus. That field's
sensitivity is proven against a constructed pair instead, which is comparator evidence rather than
corpus evidence, and the distinction is stated in the test.

Evidence: `target/c6.5-evidence/mutations.json` in the §21.3 schema.

---

## 14. Phase C6.5-8 — package breadth (§15, commit 10) — **PARTIAL**

**Done: the corpus has package cases, the replay builds package graphs, and §15.2/§15.3 are
measured.** Corpus `0.5.0`, 131 cases, replay **131/131 AGREEMENT**.

Two package cases, chosen to cover several §15.1 shapes per graph rather than one shape per case:

- `pkg__module_root_and_module` — root package + module (cross-file resolution in one package);
- `pkg__workspace_three_packages` — `app → logic → model`, covering a dependency-to-dependency call,
  a **re-exported** symbol, a generic instantiated **across a package boundary**, a function value
  crossing two boundaries, and a **`Drop` type from the leaf package destroyed in the root** —
  observed through the §8.8 protocol.

**The replay now stages package cases** to a scratch directory before compiling. Resolution writes
`stark.lock` into the root package, so compiling in place would dirty the corpus and break its own
lock; concurrent cases sharing a root would also race on that file. `C6_KEEP_TEMP` is honoured here,
which is what makes a failing package case reproducible by hand.

### 14.1 Two defects this phase found

**DEV-113 — a package build puts absolute paths in trap provenance.** §15.2 requires that no
absolute path enter semantic identity and that trap source names stay logical. For a package graph
they do not: file identity is the filesystem path, so the same workspace staged at two locations
reports different trap provenance. **Consequence: a trapping package case cannot join the corpus**,
because its observation would depend on where the repository was checked out. A second half of the
same finding: the HIR oracle attributes every trap to the ROOT file whatever file trapped, while MIR
attributes it correctly — so a dependency-trap case would make the two engines disagree about *which
file*. §15.1's last shape ("source provenance in dependency trap") is therefore **not covered**, and
both halves are pinned by tests that retire when the behaviour changes.

**DEV-114 — canonical package symbols are nondeterministic for a diamond graph.** In `app → {logic,
model}` with `logic → model`, the same function is `model::leaf@[]` in one process and
`logic::model::leaf@[]` in the next: same sources, same manifests, same declaration order. Six
consecutive runs produced both. The prefix is assigned by whichever traversal path reaches the
package first, and that traversal follows a per-process-seeded hash map. Canonical symbols are the
identity that reaches the backend, so two builds of one workspace can produce differently-named code
— against PKG-IDENTITY-001 and CD-108's deterministic identity. **Escalated, not fixed**: choosing
the canonical name for a package reachable by several paths is a compiler decision.

**And a methodological note I am recording against myself.** My first version of the reorder
experiment reused the relocation helper, which compiles *before* rewriting the manifest and leaves a
`stark.lock` behind. The stale lock made the result depend on run order, and I briefly wrote it up as
"symbols depend on declaration order" — a plausible, wrong finding. The clean experiment showed
order-independence and process-nondeterminism instead. A contaminated experiment that produces a
believable defect is worse than no experiment.

### 14.2 What §15 still owes

| §15 requirement | State |
| --- | --- |
| §15.1 dependency-trap provenance | **blocked by DEV-113** |
| §15.1 trait impl across a package boundary | not covered |
| §15.2 relocation | covered for symbols and observations; **provenance fails (DEV-113)** |
| §15.3 dependency reorder | observation-stable; symbol comparison blocked by DEV-114 |
| §15.4 `--locked --offline` via the CLI | not run; the library path is checked (relative paths only, lock written), and the generated crate is already covered by `c64_platform_matrix` |
| §15.5 installed-runtime cases | not covered here — `c63_closure_evidence` holds that evidence |
| M08/M09 as corpus metamorphic groups | **still absent.** They are covered as harness-level checks, which does not raise the §13.2 group count: a corpus case has one source set, and these transformations act on where that set lives |

---

## 15. Phase C6.5-9 — Tier-1 machinery and the C6.4 handoff (§16, commit 11) — **MACHINERY ONLY**

**Built, not yet evidenced.** This commit adds the jobs, the records and the comparator; the Tier-1
*claim* needs CI to produce two records at one commit, which happens after this push. Nothing here
asserts that both targets agree — that would be the exact substitution §16.5 forbids.

**CI jobs (§16.1).** `c65-corpus` (macos-arm64, linux-x64) runs corpus integrity, the full replay,
metamorphic pairs and package breadth, and uploads a platform-named record. `c65-mutation-controls`
runs the sixteen mutations as its own job, so a run where they were *skipped* looks different from one
where they passed. `c65-tier1-comparison` compares the two corpus records, with `if: always()` for the
same reason the C6.4 comparison has it: a skipped comparison is an absence a reader must interpret,
not a report.

**The record now carries §16.2's identity, measured** — target triple from `rustc -vV`, OS,
architecture, toolchain versions, MIR/backend/runtime versions, and `dirty_worktree` from `git
status`. Measured in the process that produced the record, because a record whose triple came from
its caller proves nothing about the machine that ran the corpus.

**The comparator (§16.4)** requires same commit, corpus version, generator version, seed, manifest
and generator hashes, identical counts, both records `PASS` and **full evidence**, clean worktrees,
two *different* Tier-1 triples, and — the strongest clause — identical **per-case observation
hashes**. Two records can agree on every count while having observed different bytes; that clause is
what makes the difference visible. Platform metadata expected to differ is reported and never treated
as disagreement.

**§20.7's twelve controls exist as tests** (`c6_tier1_controls`, 13 tests): same platform twice,
different commit, corpus version, seed and manifest hash, dirty worktree, filtered run, a skip, a
failure, a missing record, a case present on only one target, a differing per-case observation — and
one valid pair that must be *accepted*, without which "rejects everything" would pass the other
twelve.

**C6.4 handoff (§16.5).** The qualification harness now runs the five C6.5 corpus commands and
**measures** `generated_corpus_version` and `generated_corpus_case_count` from the corpus lock, with
`generated_corpus_status` derived from whether those steps actually passed in that run — `NOT-RUN`,
`PARTIAL`, `FAIL` or `PASS`. `compare-c64-evidence.py`'s expectation inverts from `BLOCKED-BY-C6.5` to
`PASS` with a nonzero case count: a record still reporting the old status now means the corpus steps
did not execute, which is a missing observation rather than a legacy state to tolerate.

### 15.1 What §16 still owes

| §16 requirement | State |
| --- | --- |
| §16.1 sharded jobs and `c65-${platform}-merge` | not built — the full replay is ~90s per target, so sharding is not yet needed; the shard *machinery* exists and is tested |
| §16.4 an actual Tier-1 comparison of two real records | **pending the first CI run at this commit** |
| §16.5 row 24 flipped to `PASS` | **not done, deliberately** — it flips when two records exist and agree, not when the machinery to produce them lands |
| §16.6 evidence import | commit 12 |

---

## 16. What comes next

§19's commit 12 — evidence and closure records (§16.6, §22): import the CI-produced Tier-1 records
rather than regenerating them locally, verify the §16.6 checklist against them, flip C6.4 row 24, and
work the §22 closure checklist. Outstanding from earlier phases: §8.1, §10.1, §11.1, §12.1, §14.2 and
§15.1 above. Migration of the 22 forked suites proceeds
alongside it in matrix order under CD-148's option (1); each migrated suite is a step toward the
required claim resting on one comparator rather than twenty-three.

Status remains `PARTIAL` until the comparator, matrix, manifest, generator, replay, metamorphic and
mutation requirements are all complete (§23). C6.5 has **no** valid "candidate complete but corpus
blocked" endpoint — producing the corpus is its central deliverable.
