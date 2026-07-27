# WP-C6.5 §17 — adversarial closure reviews

**Scope:** the eight review passes A–H, run as *closure* reviews: each question is answered against
the artifacts, not from memory, and each finding names the closure condition it affects.

**Rule followed here (owner directive, 2026-07-27):** every finding is recorded *before* any
corrective change. Nothing in this document has been fixed by writing it down.

**Consequence that shapes the dispositions.** The Tier-1 evidence imported at CD-161 was produced at
commit `8a23772`, and C6.4's re-qualification rule invalidates a record once anything under
`starkc/src`, `starkc/tests` or `starkc/scripts` changes. So a fix to any finding below **costs a
fresh Tier-1 qualification run**. Findings are therefore batched by whether they touch the qualified
path, and none has been applied yet.

---

## Findings register

Severity: **HIGH** blocks a closure claim as written · **MEDIUM** weakens a claim or leaves a
required capability unevidenced · **LOW** precision, hygiene or documentation.

| # | Sev | Review | Finding | Closure condition affected |
| --- | --- | --- | --- | --- |
| R-01 | HIGH | D, F | The corpus covers **5 of 9 admitted trap categories** | §10.4, §22.1 |
| R-02 | HIGH | B | **23 three-engine suites still use private comparators**; C65-F1 is not discharged — **CLOSED CD-165**: all 23 migrated; 289 tests, 0 failed, 0 skipped | §22.3, the required claim (§2) |
| R-03 | MEDIUM | F | Mutation controls cover **7 of 15 comparator fields** — **CLOSED CD-166**: 23 mutations, all 15 fields, enumeration machine-checked | §22.5 |
| R-04 | MEDIUM | E | Metamorphic floor unmet: 20 groups / 40 members vs 24 / 48 — **CLOSED CD-167**: 24/48, twelve families | §22.4 |
| R-05 | MEDIUM | E | **DEV-114 blocks M08/M09 outright**, not merely the floor — **CLOSED CD-167**: DEV-114's fix is what made M09 comparable; both families built | §22.4 |
| R-06 | MEDIUM | H | Shared-file **lease protocol not followed** for two leased files — **CLOSED a690bd8**; leases for the batch taken in advance in 921a2f9 | §22.8 |
| R-07 | MEDIUM | D | **36 of 136 matrix rows** have a corpus case; nothing validates row citations — **CLOSED CD-171**: all 136 rows carry one machine-checked disposition; 0 UNATTRIBUTED | §22.1, §10.2 |
| R-08 | MEDIUM | C, E | Retention (§11.11) and divergence-retention (§13.7) **never exercised** — **CLOSED CD-172**: both driven, §11.11 by the real DEV-117 retention and §13.7 by a synthetic controlled divergence | §22.2, §22.4 |
| R-09 | MEDIUM | C | `MAX_LOOP_ITERATIONS` **declared but never enforced** | §11.8, §22.2 |
| R-10 | LOW | A | `stderr_observation` equality is **tautological between HIR and MIR** — **CLOSED a690bd8**: the claim is narrowed to what the field can actually witness | precision of §22.3 |
| R-11 | LOW | C | No case-ID **collision check** in the generator | §22.2 |
| R-12 | LOW | G | Summary records skip/ignore **counts, not identities** (§16.3) — **CLOSED CD-172**: `skipped_cases` and `engine_skips` name identities; the old count was a literal `0` | §22.7 |
| R-13 | LOW | D | Metamorphic family IDs are written into `subcategories`, a **matrix-row field** | §22.1 |

### R-01 — the corpus covers five of nine admitted trap categories · HIGH

**Claim affected.** §10.4: "Include successful completion and every admitted trap category. Each trap
category needs at least one direct case." The WP's own §16 report said "7 of 9"; the corpus contains
**5**.

**Evidence.** `grep expected_trap_category tests/c6-corpus/*.toml` returns exactly `CastFailure`,
`IndexOutOfBounds`, `IntegerOverflow`, `InvalidShift`, `Panic`. `ALL_CATEGORIES` admits nine.
`DivideByZero` and `AssertFailure` are **in T16's dimension space but were dropped by the
per-template budget of 5**; `UnwrapNone` and `UnwrapErr` were never in the space.

**Why it happened, and why it is a design fault rather than a slip.** The per-template budget exists
so a large dimension space does not explode the corpus. For T16 the dimension space *is* the coverage
requirement — each tuple is a distinct normative trap category — so truncating it silently deletes
coverage. A budget that can delete a required category is the wrong mechanism for that template.

**Recommended disposition.** Make T16 exhaustive (budget must not apply where the dimension space is
itself the claim), and add `UnwrapNone`/`UnwrapErr` cases. Touches the qualified path → new corpus
version, regeneration, re-qualification. Until then, no C6.5 claim may state "every admitted trap
category is covered", and the WP's §16 report line is corrected by this document.

### R-02 — twenty-three suites still use private comparators · HIGH

**Claim affected.** The required claim (§2): the three engines produce the same *normative
observations*. C65-F1 recorded that 23 files each had a private notion of agreement; CD-148 chose
incremental migration. **Zero have been migrated.**

**Evidence.** 23 test files run all three engines without `mod support`:
`c62b_f2_specific_instance`, `c62b_f5_impl_bounds`, `c62b_f6_self_normalisation`,
`c62c_associated_types`, `c62d_operator_coretrait`, `c63a_string`, `c63b_trapping_ops`,
`c63b_vec_box`, `c63c_iterators`, `c63d_map_key_identity`, `c63e_float32`, `c63e_formatting`,
`cd139_float_division`, `native_c5_4_workspace`, `native_c6_1_ownership`,
`native_c6_2_generics_traits`, `native_c61f_aggregates`, `native_c61f_b2_weakening`,
`native_c61f_b3_stored_refs`, `native_c61f_nominals`, `native_c61f_reborrow`, `native_c61f_ret_refs`,
and **`c65_entry_exit_contract`** — which C6.5 itself added, while the finding about forks was open.
None observes the §39 shape (`grep '@@stark'` over them returns nothing).

**Consequence for closure.** Every matrix row whose evidence column names one of those suites is
evidence produced by a comparator the C6.5 authority has never seen. That is most of the
`EXISTING-EVIDENCE` rows. **No closure claim may cite them** until they are migrated.

**RESOLUTION (CD-165, batch).** All 23 are migrated. Each kept its own `agree`-shaped wrapper — the
case bodies name it, and the doc headers explain what each case is for — but the wrapper is now one
line delegating to `support::differential`. What the migration changed, beyond provenance:

- The private helpers asserted `status == 0` on each engine **separately**. That is not a comparison:
  three engines each exiting 0 while printing three different things all passed. Agreement is now
  field-by-field over the §39 observation — stdout, Drop log, return frame, trap category, line,
  column, message class.
- They returned early when `rustc` was missing, silently dropping to a one-engine smoke test.
  `agree_completing_available_engines` compares the two interpreters instead: a missing toolchain
  removes an ENGINE, not the comparison.
- `c63b_trapping_ops::traps_at` checked the native stderr for a category **substring** and never
  asked MIR or the oracle which category they raised. `cd139_float_division::traps` asserted only
  "the oracle produced no output" and "MIR returned `Err`". Both now pin category *and* line
  through `agree_trapping`.
- `native_c5_4_workspace` ran **two** engines under comments reading "Engine 1 / Engine 2", in a
  file named `native_*`, and built its root `SourceFile` from the absolute checkout path — the exact
  provenance defect DEV-113 fixed. It now goes through `front_end_package` and runs all three.

Where a suite pinned its expected stdout independently (`c63a_string`, `c63d_map_key_identity`,
`c63e_float32`, `cd139_float_division`) the pin is preserved via `agree_completing_with_stdout`.
Where it did not — `c63c_iterators` and `c63e_formatting` took the HIR oracle's own output as the
expectation — the engine-agreement check is real but cannot notice all three rendering the same
wrong thing; those headers now say so rather than implying a pin they do not have.

**Evidence.** One clean run of all 23 migrated suites at `4bd6675`: **289 passed, 0 failed,
0 ignored, 0 skipped**, exit 0. The zero-skip count matters as much as the zero-failure count — it
means every suite ran its native engine, so this is a three-engine result rather than the two-engine
fallback. No divergence surfaced anywhere: nothing had been hiding behind the weaker check. CI at
`2f1929a` is green across all 15 jobs including `windows-x64`.

**Original recommended disposition** (retained for the record). Migrate in matrix order as CD-148
directed; start with the suites the matrix cites most (`c63a_string`, `c63b_*`, `c63c_iterators`,
`c63d_map_key_identity`,
`native_c61f_*`). `c65_entry_exit_contract` should be migrated first as a matter of consistency —
C6.5 should not be the source of a new fork.

### R-03 — mutation controls cover seven of fifteen comparator fields · MEDIUM

**Claim affected.** §22.5 / the owner's verification ask 6 ("mutation controls cover every
observation field"). They do not.

**Evidence.** `first_difference` distinguishes 15 fields. §14 mutations exercise:
`stdout_bytes` (MU01, 08–13, 15), `exit_status` (MU16), `drop_log` (MU04–07), `trap category` (MU03),
`trap line` (MU02), `trap source_file` (MU14), and `trap message_class` / `returned_observation` via
two extra tests. **Never mutated from a corpus witness:** `stderr_bytes`, `trap column`,
`stdout_before_trap`, `stderr_observation`, `trap exit_status`, `drop_log_before_trap`, and
`completion versus trap`.

Several of those *are* covered by §8.10's comparator unit tests, but that is different evidence:
§8.10 proves the comparator distinguishes two constructed values; §14 proves a **real witness** can
expose the defect class. Conflating them would overstate the sensitivity claim.

**Recommended disposition.** Add witness-based mutations for the seven uncovered fields. `stderr_bytes`
needs the `Err`-completion entry case, which is two-engine (DEV-111) — so that one is bounded by an
open blocker and should be recorded as such rather than skipped silently.

### R-04 — the metamorphic floor is unmet · MEDIUM

**Claim affected.** §13.2 / §22.4: 24 groups, 48 members, all twelve families. Delivered: 20 groups,
40 members, ten families. Already recorded at CD-157 and asserted by
`the_metamorphic_floor_is_reported_honestly`; restated here because it is a closure blocker, not a
note.

### R-05 — DEV-114 blocks M08/M09 outright, not merely the floor · MEDIUM

**Claim affected.** §22.4, and the shape of the remaining work.

**Evidence and reasoning.** M08 (workspace relocation) and M09 (dependency reorder) are metamorphic
claims over a package graph. DEV-114 shows canonical package symbols are **nondeterministic across
processes** for a diamond graph. A relocation or reorder pair over such a graph would therefore
produce different symbols run to run — the pair would be unstable for a reason unrelated to the
transformation, and "retain both sources and open a defect" (§13.7) would fire on the harness rather
than on the compiler.

This was not previously stated: CD-157 recorded M08/M09 as blocked on *package graphs*, and CD-159
recorded DEV-114 separately. The link is that **fixing DEV-114 is a precondition for the metamorphic
floor**, not just for symbol hygiene.

### R-06 — the shared-file lease protocol was not followed · MEDIUM

**Claim affected.** §22.8 governance.

**Evidence.** `C6-FILE-OWNERSHIP.md` §2 lists `starkc/tests/three_engine_differential.rs` and
`starkc/src/mir/lower.rs` as shared files requiring a lease recorded in
`C6-INTEGRATION-LEDGER.md`. C6.5 edited `three_engine_differential.rs` substantially (CD-148 commits
2 and 3) and `mir/lower.rs` (CD-150, the `()`-to-`Unit` lowering). **The ledger contains no lease
entry for either**; its last content is the C6.0 barrier. The plan's §8.2 called commit 2 "a
shared-file lease commit" and the WP repeated it, but the lease was never *recorded* where the
protocol says leases live.

**Mitigating fact, not an excuse.** C6 is described in `WP-C6.5.md` as entirely Claude-owned, and no
concurrent track edited those files, so no conflict occurred. The protocol still says a lease is a
coordination *record*; an unrecorded lease is indistinguishable from none.

**Recommended disposition.** Add retrospective lease entries to the integration ledger naming the
commits, base SHAs and the tests run. Docs-only, so it does not disturb the Tier-1 evidence.

### R-07 — thirty-six of 136 matrix rows have corpus evidence, and row citations are unvalidated · MEDIUM

**Claim affected.** §10.2 ("every coverage-matrix row needs at least one hand-written witness"),
§22.1, and the owner's verification ask 1.

**Evidence.** Corpus cases cite 46 distinct IDs in `subcategories`; ten of those are metamorphic
family IDs (`M01`…`M12`, see R-13), leaving **36 real matrix rows**. The remaining 100 rows are
covered — where they are covered at all — by the suites in R-02.

**Second half, and the sharper problem.** Nothing validates that a `subcategories` entry is a real
matrix row. The ten family IDs prove it: they passed validation while naming rows that do not exist.
A typo'd or invented row ID would claim coverage no case provides, which is the same failure mode as
CD-154's fabricated rule citations — caught there, uncaught here.

**Third half, found while fixing the second (CD-165).** The matrix's `→T##` arrows — "a generator
template covers this row too" — were never read by anything either. **36 of the 136 rows carried one
that was false**: 16 named a template in `MISSING_TEMPLATES`, which generates nothing at all
(T13, T14, T17, T19), and 20 named a real template whose cases never cited that row. Same shape as
CD-154 again: a forward-looking claim written into an evidence document and never revisited.

**RESOLUTION (CD-165, batch).**

- The `subcategories` validator now rejects any entry that is not a real matrix row, with a negative
  control (`a_manifest_citing_a_nonexistent_matrix_row_is_rejected`). This is what would have caught
  the ten family IDs.
- Every remaining `→T##` arrow is machine-checked by
  `every_template_arrow_in_the_matrix_is_backed_by_generated_cases`.
- The 36 false arrows were resolved **by checking the emitted sources**, not by assertion. Eleven
  were genuinely earned — the template's cases really do exercise the row — and those rows are now
  cited by the template that exercises them (E01, E08, E17, C08, P05, V01, V02, V08, V21, D03, D11).
  Nine were not earned and the arrow was **removed**, leaving the row on its existing evidence
  (E05, C05, C09, C10, P03, V17, V24, D04, D07). Sixteen point at deferred templates and now read
  `T## DEFERRED` in prose instead of pointing at a template that will never run.
- Real matrix-row coverage by corpus cases is now measurable and honest: **47 of 136** at 0.7.0,
  rising with the citations above. The remainder rest on the R-02 suites, which are now migrated —
  which is what makes citing them legitimate.

**Still open.** The row-by-row coverage gap itself: a majority of rows have no hand-written corpus
witness and rest on migrated suite evidence. §10.2's stronger reading — a corpus witness per row —
is not met.

### R-08 — retention and divergence-retention have never been exercised · MEDIUM

**Claim affected.** §11.11, §13.7, §22.2, §22.4.

**Evidence.** `cases/retained/` holds the six DEV-111/112 entry-contract cases, which were retained
*by hand*. The §11.11 workflow — `cases/retained/<DEV-ID>/original/` plus `reduced/`, with seed,
template, dimensions and hashes recorded — has never run, and §11.13 requires it to be "tested with a
synthetic failure". §13.7's divergence path (retain both sources, open a defect, do not rewrite the
pair) has likewise never fired, because no pair has diverged.

A capability with no evidence is a capability that will be discovered to be broken at the moment it
is first needed — which is during a real defect.

### R-09 — a declared bound is not enforced · MEDIUM

**Claim affected.** §11.8 ("set and test hard bounds"), §22.2.

**Evidence.** `generate.py` declares `MAX_LOOP_ITERATIONS = 16` and never reads it. `MAX_SOURCE_BYTES`
and `MAX_FUNCTIONS_PER_CASE` *are* enforced in `enforce_bounds`. Loop iterations are small today
because every template's dimensions make them small, which is exactly how an unenforced bound stays
invisible until a template changes.

### R-10 — `stderr_observation` equality is tautological between the interpreters · LOW

**Claim affected.** Precision of the comparator claim (§22.3), not its correctness.

**Evidence.** `constructed_trap_stderr` builds `category_text` from
`runtime_category(category).message()`. HIR and MIR both take that path, so comparing their
`stderr_observation` adds nothing beyond the category comparison already performed. The field is
independent evidence only where one side is **parsed** from the native binary's real stderr.

Not a defect — §8.5 explicitly says to construct it for the interpreters — but the C6.5 report should
not describe stderr as compared "three ways".

### R-11 — no case-ID collision check in the generator · LOW

**Claim affected.** §20.2 ("case IDs collision-free"), §22.2.

**Evidence.** `select()` derives `gen__<template>__<digest[:8]>` and appends without checking for a
prior identical ID. A collision (32 bits of digest) would make the second case overwrite the first
file, and the manifest would carry two entries with one ID; the corpus validator's duplicate-ID rule
catches that, but late and with a misleading symptom.

### R-12 — the summary records counts, not identities · LOW

**Claim affected.** §16.3 ("record full identities, not only counts"), §22.7.

**Evidence.** `summary.json` carries `skipped_count`/`quarantined_count` as numbers. Identities exist
in `per-case.json` (every case with its result), so the information is recoverable — but the summary
alone, which is what a reader checks first, does not satisfy §16.3 as written.

### R-13 — family IDs are stored in a matrix-row field · LOW

**Claim affected.** §22.1 bookkeeping; the direct cause of R-07's overcount.

**Evidence.** Metamorphic members carry `subcategories = ["M01"]`, but `subcategories` is documented
as "matrix row IDs this case is evidence for". The family already has its own field
(`metamorphic_family`). Ten IDs in the coverage count are therefore not rows.

---

## Review A — semantic authority

| # | Question | Verdict |
| --- | --- | --- |
| 1 | Comparator compares only normative observations? | **Yes** — 15 normative fields; raw engine diagnostics never enter equality. See R-10 for a precision caveat |
| 2 | HIR treated as oracle without overriding normative drafting? | **Yes** — corpus expectations come from the spec and the templates' own model, never from an engine |
| 3 | Can engine majority decide correctness? | **No** — comparison is pairwise and any difference fails; expectations are pinned independently |
| 4 | Can generated wrapper code alter semantics? | **No** — the §8.7 wrapper is appended *after* the case source, so user line numbers and provenance are unchanged |
| 5 | Can observation framing change source provenance? | **No** — frames are emitted by user code through ordinary `print` |
| 6 | Can Drop logging change Drop timing? | **No** — the frame is printed inside the user's own `Drop::drop`, at the moment the drop happens |
| 7 | Can Cargo or host text enter equality? | **No** — only the generated binary's own stdout/stderr is captured; build output is separate |
| 8 | Panic messages compared exactly where normative? | **Yes** — `TrapMessageClass::UserMessageExact` compares the string |
| 9 | Category-only traps not overfitted to prose? | **Yes** — prose matching is oracle-side only and an unknown message is a hard failure |
| 10 | Can a non-Core exclusion hide an admitted feature? | **No** — §4.3 is enforced by the validator; V19 was reclassified when its reason failed that test |
| 11 | Can a semantic quarantine let C6 close? | **No** — the three allowed reason classes cannot express one |
| 12 | Can generated-Rust host behaviour substitute for STARK semantics? | **Largely no** — float rendering goes through `stark_runtime::format::canonical_float32/64` rather than bare host `Display`. Verified by `c63e_float32`, migrated to the shared comparator in CD-165 |

## Review B — comparator architecture

| # | Question | Verdict |
| --- | --- | --- |
| 1 | One comparator implementation? | **Yes, since CD-165** — one authority, and all 23 suites delegate to it (**R-02 closed**) |
| 2 | Do existing and new suites call the same code? | **No** — 7 files use `mod support`; 23 do not |
| 3 | Observation fields byte-precise? | **Yes** — `Vec<u8>` throughout; no lossy conversion in equality |
| 4 | Trap categories exhaustive? | **Yes** — `runtime_category`'s match is exhaustive, so a tenth category fails to compile until mapped |
| 5 | Unknown renderings hard failures? | **Yes** — `parse_native_trap` panics; proven by `unknown_native_trap_rendering_fails` |
| 6 | Internal errors distinct from program traps? | **Yes** — `MirRunError::Internal` becomes a harness failure classified `MIR-INTERNAL-FAILURE` |
| 7 | Drop and return protocols validated? | **Yes** — malformed, mid-line and duplicate-identity frames are hard errors |
| 8 | Can malformed framing be ignored? | **No** — demonstrated live: the package workspace case failed until its frame occupied a whole line |

## Review C — generator correctness

| # | Question | Verdict |
| --- | --- | --- |
| 1 | Deterministic across clean runs? | **Yes** — same seed byte-identical, proven by running the generator twice |
| 2 | Selection independent of host/library order? | **Yes** — pre-existing files and output location do not change a byte |
| 3 | Case IDs stable and collision-checked? | **Stable yes; collision-checked only downstream** (**R-11**) |
| 4 | Programs correct by construction? | **Yes** — dimension spaces exclude invalid tuples; all 70 pass on three engines |
| 5 | Compiler rejection treated as a generator defect? | **Yes** — a rejection surfaces as `LOWERING-REFUSAL` and fails the replay |
| 6 | Source/MIR/runtime sizes bounded? | **Partly** — source bytes and function count enforced; MIR/runtime size not bounded (**R-09** for the unenforced loop bound) |
| 7 | Can generation recurse or loop unboundedly? | **No in practice, unguarded in principle** (**R-09**) |
| 8 | Absolute paths, timestamps or PIDs in output? | **No** — asserted over every generated file and the generated manifest |
| 9 | Seed/version attached to every case? | **Yes** — and the validator rejects a generated case without all three |
| 10 | Can a changed template fail to bump metadata? | **No** — `templates_sha256` and `metamorphic_sha256` are in the lock, so a template edit invalidates it |
| 11 | Arbitrary fuzzing mislabeled as semantic generation? | **No** — enumeration over declared dimensions, no token mutation |
| 12 | Can the generator silently drop failed candidates? | **No** — there is no generate-and-discard loop. Dimension spaces are filtered *declaratively* (e.g. unsigned subtraction that would trap), which is a different thing and is stated as such |

## Review D — coverage completeness

| # | Question | Verdict |
| --- | --- | --- |
| 1 | Does every §40 category appear? | **Yes** — 8 groups, 136 rows |
| 2 | Meaningful subcategories per category? | **Yes** |
| 3 | Completion and trap paths balanced? | **Weak** — 5 trap cases against 126 completions (**R-01**) |
| 4 | All trap categories covered? | **No — 5 of 9** (**R-01**) |
| 5 | Ownership/Drop edge paths covered? | **Partly** — reverse order, per-iteration, cross-package and pre-trap logs; partial moves and reinit via suites migrated to the shared comparator in CD-165 (R-02) |
| 6 | Trait/generic/function-value sentinels adversarial? | **Yes** — distinct sentinel values per instance, with routing controls proving the wrong route is observable |
| 7 | Collection order and slice aliasing observable? | **Yes** — insertion order distinct from sorted order; view mutation visible in the owner |
| 8 | Package and dependency shapes covered? | **Partly** — root+module and a 3-package chain; no cross-package trait impl, no dependency-trap provenance (DEV-113) |
| 9 | Relocation and dependency reorder covered? | **As harness checks, not corpus groups** (R-04, R-05) |
| 10 | Files/sets/resources covered or normatively non-Core? | **Files** non-Core (V20); **HashSet** is a recorded blocker (V19), not an exclusion |
| 11 | Every required row has hand-written evidence? | **Every row has a checked disposition** (R-07 closed, CD-171): corpus case, exact comparator-backed test identity, environment test, NOT-APPLICABLE with a reason, or BLOCKED with a DEV and owner |
| 12 | Every generatable row has generated evidence? | **No** — 15 templates map to a subset of rows; no row-to-template map exists |

## Review E — metamorphic adequacy

| # | Question | Verdict |
| --- | --- | --- |
| 1 | All 12 families present? | **Yes — 12** (R-04 closed, CD-167): M08 relocation and M09 dependency reorder |
| 2 | More than one non-trivial pair per family? | **Yes** — two independent groups each, differing in value or path, and an identity-transform guard rejects a pair that changed nothing |
| 3 | Preconditions explicit? | **Yes** — recorded per group and *enforced* (scope insertion refuses a `Drop` base; reorder refuses a catch-all) |
| 4 | Reordered arms genuinely non-overlapping? | **Yes** — distinct enum variants, asserted absence of a wildcard |
| 5 | Loop forms genuinely equivalent for ownership/Drop? | **Yes** — asserted no owning value and no `Drop` in the body |
| 6 | Pair equality checked inside every engine? | **Yes** — HIR, MIR and native separately |
| 7 | Cross-engine equality for every member? | **Yes** — via the §12 replay, which runs members as ordinary cases |
| 8 | Can a transformation avoid a defect? | **Guarded** — the identity assertion caught two of my own fake pairs (CD-157) |
| 9 | Divergent pairs retained? | **Exercised** (R-08 closed, CD-172) — §13.7's first-differing-field identification is driven by a constructed divergence, and the retention layout by DEV-117 |
| 10 | Package transformations deterministic? | **Yes** — DEV-114 fixed in bac13f5; M09's pairs pin canonical symbols across a reorder, which is the regression check (R-05 closed) |

## Review F — mutation sensitivity

| # | Question | Verdict |
| --- | --- | --- |
| 1 | Every §43 mutation has a witness? | **Yes for all 16**; `returned_observation` has no *corpus* witness and is proven against a constructed pair, recorded as such |
| 2 | Unmodified witness passes first? | **Yes** — asserted before every mutation |
| 3 | Mutation affects the intended observation? | **Yes** — asserted to change the observation, else vacuous |
| 4 | Comparator identifies the intended field? | **Yes** — each control requires the field name in the rejection |
| 5 | Drop omission/duplication/reversal distinct? | **Yes** — three different mutations; all correctly report `drop_log` |
| 6 | Wrong generic/trait/function targets distinct? | **Yes** — three sentinels with different values |
| 7 | Slice-copy mutation proves aliasing sensitivity? | **Yes** — plus a source-level control running the wrong route |
| 8 | Float32 widening uses a discriminating value? | **Yes** — `0.1f32` vs its widened rendering; a naive literal comparison would have passed under the defect |
| 9 | Source-path replacement detects generated-Rust leakage? | **Yes** — MU14 substitutes `src/main.rs` |
| 10 | Can missing output or wrong exit normalisation survive? | **No** — MU15/MU16 |
| — | *Coverage of the field set* | **15 of 15 fields** (R-03 closed, CD-166) — 23 mutations; `every_comparator_field_has_a_mutation_control` reads `COMPARATOR_FIELDS` beside `first_difference`, so a new field without a control fails |

## Review G — evidence and Tier-1

| # | Question | Verdict |
| --- | --- | --- |
| 1 | Both records from the same exact commit? | **Yes** — `8a23772` |
| 2 | Two different Tier-1 targets? | **Yes** — and a same-triple pair is rejected by test |
| 3 | Corpus, generator, seed and hashes identical? | **Yes** |
| 4 | Every required case and shard present? | **Yes** — 131 both sides, no duplicates, none one-sided; unsharded |
| 5 | Skips, ignores, timeouts visible by identity? | **Yes** (R-12 closed, CD-172) — `skipped_cases` and `engine_skips` carry identities; a filtered run reports 152 named cases where the field used to read a literal `0` |
| 6 | Artifacts preserved on failure? | **Yes** — `if: always()` on every upload |
| 7 | Can a filtered run be mistaken for full qualification? | **No** — `full_evidence` plus `PARTIAL-FILTERED`, both tested |
| 8 | Per-case observations compared, not only totals? | **Yes** — identical observation hashes for all 131, and a differing-hash pair is rejected by test |
| 9 | Is C6.4 row 24 backed by the same corpus evidence? | **Yes** — same commit, and the harness measures version and count from the lock |
| 10 | Were old records reused incorrectly? | **No** — the `4844702` records were superseded by a fresh run, not amended |
| 11 | Does Windows remain separate? | **Yes** — gap probe only, never a qualification record |
| 12 | Can missing evidence render agreement? | **No** — a missing record is an explicit disagreement, by test |

## Review H — scope and governance

| # | Question | Verdict |
| --- | --- | --- |
| 1 | Did C6.5 add a language feature? | **No** — DEV-112 made `()` typecheck as `Unit`, which TYPE-PRIM-001 already required; DEV-111 fixed MIR's entry termination against PROC-EXIT-001 |
| 2 | Did it alter MIR/runtime contracts without CE approval? | **No** — `MirExecution` appears nowhere in `mir.md`; no MIR shape, `RuntimeFn` or surface version moved |
| 3 | Did it edit another track's files without a lease? | **Protocol not followed** (**R-06**) — no concurrent track was affected, but no lease was recorded |
| 4 | Did it fix unrelated defects without recording scope? | **No** — DEV-111/112 were exposed by required cases (§18.5) and recorded with their evidence |
| 5 | Did it introduce a network dependency? | **No** — the manifest parser was hand-written precisely to avoid one |
| 6 | Did it claim release replay before C7? | **No** — debug native only |
| 7 | Did it lower corpus breadth for CI convenience? | **No** — sharding exists but is unused because the full replay is ~90s per target. But see **R-01**: breadth *was* lost to a budget mechanism, for reasons of generator design rather than CI time |
| 8 | Did it quarantine an admitted semantic failure? | **No** — the corpus contains no quarantine, and the validator cannot express one |
| 9 | Are all new deviations recorded? | **Yes** — DEV-111, DEV-112 (fixed), DEV-113, DEV-114 (open), each with pinned tests |
| 10 | Is WP-C2.12 closure explicit? | **Yes, as of the owner's directive** — recorded as a separate governance closure, not folded into C6.5 |

---

## What these reviews change about the C6.5 status

The status remains **`PARTIAL`**, and the reviews add three blocking items to the list that were not
previously visible:

1. **R-01** — a coverage claim in the WP's own report was wrong (7 of 9 → 5 of 9).
2. **R-02** — no closure claim may rest on the 23 forked suites, which is most of the matrix's
   `EXISTING-EVIDENCE` rows. This is the single largest gap between "the corpus agrees" and "C6 has
   one definition of agreement".
3. **R-07** — 36 of 136 rows have corpus evidence, and until row citations are validated the
   coverage count itself is unverified.

None of R-01…R-13 was acted on when this document was written. The owner's disposition (2026-07-27,
CD-163) then ordered the work:

**Landed immediately, because they do not touch the qualified path:**

- **R-06** — retrospective lease entries and a process correction are recorded in
  `C6-INTEGRATION-LEDGER.md`. Status: **CLOSED (record)**; the underlying protocol violation stands as
  history.
- **R-10** — `WP-C6.5.md` §16 now states what `stderr_observation` proves: parsed on the native side,
  constructed for both interpreters, therefore *not* "compared three ways". Status: **CLOSED**.
- **R-01's wording** — the report line "7 of 9 admitted trap categories" is corrected to **5 of 9**
  with the reason. The *coverage gap itself* remains **OPEN** and is batch item 1.

**Deferred to the consolidated batch, with reasons:**

- **R-12** — the owner allowed this now "provided this remains outside the qualified execution path".
  It is not: the summary writer is `starkc/tests/c6_generated_corpus.rs`, and the C6.4
  re-qualification rule invalidates the `8a23772` records on any change under `starkc/tests`. Moving
  it into the batch is the only way to record identities without discarding the Tier-1 evidence
  before the packets are dispositioned. Status: **OPEN, batched**.
- **R-01 (coverage), R-02, R-03, R-04, R-05, R-07, R-08, R-09, R-11, R-13** — all touch the qualified
  path. Status: **OPEN, batched**, in the order the owner set.
