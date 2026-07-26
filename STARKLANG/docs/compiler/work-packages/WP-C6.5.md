# WP-C6.5 — Full Differential and Generated Corpus

**Track:** Gate C6 (all of C6 is Claude-owned)
**Status:** `PARTIAL` — C6.5-0 (re-pin, inventory, coverage matrix) complete; C6.5-1 at the plan's
commit-2 boundary (comparator extracted, no observation-shape change yet).
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
| matrix covers all §40 categories | yes — 8 groups, 133 rows, `C6-CORPUS-COVERAGE-MATRIX.md` |
| every row has a normative citation or justified non-Core classification | yes |
| current evidence linked by exact test/case ID | yes — rows cite frozen-corpus case IDs, `three_engine_test!` case names, and C6.x suite names |
| every gap has an execution disposition | yes — one of §7.4's nine classifications |
| no category silently omitted | yes |

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


## 6. What comes next

§19's commit 3 — the §8.3 observation model, the §8.5 trap-stderr normalizer, the §8.7 returned-value
and §8.8 Drop-log protocols, and the §8.10 comparator unit tests that prove each new field is
load-bearing. Migration of the 22 remaining forked suites then proceeds in matrix order under
CD-148's option (1).

Status remains `PARTIAL` until the comparator, matrix, manifest, generator, replay, metamorphic and
mutation requirements are all complete (§23). C6.5 has **no** valid "candidate complete but corpus
blocked" endpoint — producing the corpus is its central deliverable.
