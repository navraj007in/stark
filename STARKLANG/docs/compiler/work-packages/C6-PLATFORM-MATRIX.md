# C6-PLATFORM-MATRIX — WP-C6.4 Tier-1 qualification matrix

**Owner:** WP-C6.4 (`WP-C6.4.md`)
**Authority:** `starkc/docs/WP-C6-ENTRY.md` §§32–37
**Frozen:** 2026-07-26, at `5d2c85d` (Gate C6, immediately after WP-C6.3 closed via CD-142)
**Status:** rows 1–23 **MET** on records taken at `4844702` under the strengthened comparator
(CI 30192449131, all 11 jobs green, TIER-1 AGREEMENT). The earlier records from `61008f6` were
discarded, not carried forward — the comparator that now guards this matrix refuses them. Row 24 is
**PASS as of `8a23772` (CD-161)** — the C6.5 corpus replayed on both
Tier-1 targets with identical per-case observations; row 25 is REPORT-ONLY with G1 and G3 closed.

## How to read this file

The required column set (execution plan §7.2) is split across two tables so it stays readable:

- **Table A — requirements.** area, required observation, per-platform command, host assumption,
  expected result, implementation status.
- **Table B — evidence.** actual result per Tier-1 target, artifact path, exact commit, deviation,
  closure status. Table B is filled from CI artifacts, never by hand — see
  `starkc/docs/compiler/evidence/c6.4/README.md`.

Status vocabulary:

| Status | Meaning |
| --- | --- |
| `IMPLEMENTED` | the check exists and runs; awaiting a platform record |
| `PRE-EXISTING` | the check already existed before C6.4 and is claimed as-is |
| `BLOCKED` | cannot be satisfied within C6.4; the blocker is named |
| `REPORT-ONLY` | Tier-2 disposition; never gates a Tier-1 claim |

Unless a row says otherwise, the macOS-arm64 and Linux-x64 commands are **identical** — that is the
point of a cross-platform harness, and a row needing two different commands is itself a finding.
All commands run from `starkc/`.

---

## Table A — requirements

| # | Area | Required observation | Tier-1 command (both) | Windows | Host assumption | Expected result | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | host identification | the host triple is measured, not assumed | `cargo test --test c64_platform_matrix target_preflight_this_host` | same | `rustc -vV` `host:` field | host is a named target | IMPLEMENTED |
| 2 | selected-target identification | host and selected target are separate recorded values | `… target_preflight_accepts_both_tier1` + `… portability_build_manifest_records_host_and_selected_target_separately` | same | none — both come from `src/target.rs` | `host_triple` and `target_triple` both present in `build.json` | IMPLEMENTED |
| 3 | supported-target acceptance | both Tier-1 triples classify as tier-1 with `stark-64-v1` and no suffix | `… target_preflight_accepts_both_tier1_targets_with_the_declared_contract_and_suffix` | tier-2, `.exe` | none | 2 targets, tier-1 | IMPLEMENTED |
| 4 | unsupported-target rejection | an unnamed triple is refused **before** Cargo/linking | `… target_preflight_rejects_unknown_targets_of_either_width`; unit `backend::generated_rust::build::tests::preflight_rejects_an_unsupported_host_before_anything_is_generated` | same | none | `UnsupportedByStark`, no crate emitted | IMPLEMENTED |
| 5 | missing-toolchain rejection | distinct class from an unsupported target | `… target_preflight_separates_an_unsupported_target_from_a_missing_toolchain` | same | injectable probe; no toolchain is uninstalled | two different classes and two different messages | IMPLEMENTED |
| 6 | layout selection | the contract comes from the target and is checked, not defaulted | `… portability_build_manifest_records_host_and_selected_target_separately`; `layout::contract_for` | same | none (was: unconditional `stark-64-v1`) | `stark-64-v1` v1, recorded | IMPLEMENTED |
| 7 | executable suffix | the suffix comes from the **target**, not the compiler's host | unit `generated_binary_filename_is_platform_aware`; every native suite | `.exe` | none (was: `std::env::consts::EXE_SUFFIX`) | `stark_program` / `stark_program.exe` | IMPLEMENTED |
| 8 | runtime metadata | tier, pointer width, contract and both triples recorded in `build.json` | `… portability_build_manifest_records_host_and_selected_target_separately` | same | none | 5 target fields present | IMPLEMENTED |
| 9 | compiler/runtime compatibility | a version mismatch is rejected before user code runs | `cargo test --test c63_closure_evidence` | same | none | mismatch detected, user code never runs | PRE-EXISTING |
| 10 | stdout bytes | exact bytes, including Unicode | `… platform_stdout_is_exact_bytes_including_unicode_and_line_termination` | same | none — `output.rs` is byte-oriented | `"héllo wörld\nno newline"` exactly | IMPLEMENTED |
| 11 | stderr bytes | trap category text on stderr | `… platform_trap_reports_category_provenance_and_exit_status` | same | none | `error: runtime trap: index out of bounds` | IMPLEMENTED |
| 12 | line termination | `\n` on every platform; never CRLF | same as row 10 (explicit `\r` assertion) | same | none — Rust does no text-mode translation | no `\r` in stdout | IMPLEMENTED |
| 13 | trap category | the category is the same on both targets | same as row 11 | same | none | `IndexOutOfBounds` | IMPLEMENTED |
| 14 | trap provenance | the user's source location, plus exit 101 and the flushed pre-trap prefix | same as row 11 | same | none | `--> trapsite.stark:4:11`, exit 101, stdout `before` | IMPLEMENTED |
| 15 | filesystem paths | build and run under a path containing spaces | `… portability_builds_and_runs_under_paths_containing_spaces` | same | none | exit 0, `42\n` | IMPLEMENTED |
| 16 | Unicode paths | build and run under a non-ASCII path, install prefix included | `… portability_builds_and_runs_under_paths_containing_unicode` | same | none | exit 0, `7\n` | IMPLEMENTED |
| 17 | temporary directories | `std::env::temp_dir`, PID + counter, no shared root | all `portability_*`/`platform_*` rows | same | one survivor outside the matrix (gate-7 fixture, `/tmp`) | no collisions under parallel tests | IMPLEMENTED |
| 18 | manifest escaping | the generated `Cargo.toml` is TOML, not Rust `Debug` | unit `manifest_paths_are_escaped_to_toml_rules_not_rust_debug_rules`; `a_generated_manifest_with_an_adversarial_runtime_path_stays_one_well_formed_line`; rows 15–16 end-to-end | same, incl. drive prefixes | none (was: `{:?}`) | control chars become `\u00XX`; backslashes and quotes escape | IMPLEMENTED |
| 19 | installed runtime | a build against an installed runtime cannot silently fall back to the checkout | `cargo test --test c63_closure_evidence`; `… portability_installed_runtime_requirement_refuses_the_checkout_fallback` | same | the fallback is compiled in (`CARGO_MANIFEST_DIR`) and now switchable off | discovery FAILS under `STARK_REQUIRE_INSTALLED_RUNTIME=1` with no install present | IMPLEMENTED |
| 20 | offline build | the generated crate builds `--locked --offline`, with a lock and no registry source | `… portability_generated_crate_is_locked_and_network_free`; every native suite | same | none | lock present, no `source =`, no `checksum =` | IMPLEMENTED |
| 21 | frozen workspace | the C5 multi-package workspace builds and runs | `cargo test --test native_c5_4_workspace`; CI `release-package-smoke` | same | none | green | PRE-EXISTING |
| 22 | three-engine suite | HIR/MIR/native agreement against real native stdout | `cargo test --test three_engine_differential` | same | none | green | PRE-EXISTING |
| 23 | determinism rerun | two runs in separate **processes** agree on build key and generated source | `run-c64-qualification.py` (runs `determinism_` twice) | same | none | `determinism_result: match` | IMPLEMENTED |
| 24 | generated corpus | the deterministic generated corpus runs on both Tier-1 targets | `cargo test --test c6_generated_corpus` (+ `c6_metamorphic`, `c6_mutation`, `c6_package`, `c6_corpus_manifest`, `c6_corpus_generator`) | same | none | 131 cases, 131 AGREEMENT, 0 failed, 0 skipped, identical per-case observation hashes on both targets | **PASS** |
| 25 | Windows disposition | a classified gap report exists | CI `c64-windows-gap` probe | `cargo test --test c64_platform_matrix` | — | four classified gaps (G1, G3 CLOSED; G2, G4 open), none semantic | REPORT-ONLY |

### Row 24 — closed at `8a23772` (CD-161)

The deterministic generated corpus now exists — `starkc/tests/c6-corpus/`, `corpus_version` 0.5.0,
131 cases (70 generated across 15 templates, 55 hand-written including 40 metamorphic members and 13
adversarial sentinels, 6 retained) — and it was replayed through every engine each case declares on
**both Tier-1 targets at one commit**, by CI, with the records downloaded rather than regenerated.

It remains a **different artifact** from `tests/exec_snapshots/` (the frozen execution corpus, now
v1.4.0), which was preserved rather than absorbed; rows 21–22 still cover that one.

What closes this row is not that two jobs went green. It is that
`compare-c65-evidence.py` compared the two records and found: the same commit, corpus version,
generator version, seed, manifest and generator hashes; identical counts; both records `PASS` and in
FULL evidence mode; clean worktrees; two different Tier-1 triples; and **identical per-case
observation hashes for all 131 cases**. Two records can agree on every total while having observed
different bytes — that last clause is the one that makes this a semantic claim rather than a
scheduling one.

Both C6.4 evidence records at this commit now carry `generated_corpus_status: PASS`,
`generated_corpus_version: 0.5.0`, `generated_corpus_case_count: 131`, measured from the corpus lock
by the qualification harness rather than supplied to it.

---

## Table B — evidence

Filled from `starkc/docs/compiler/evidence/c6.4/*.json`, produced on the runner that ran the
commands and downloaded, not regenerated. Rows are grouped where one command establishes several:
splitting a single `c64_platform_matrix` run across eight rows would report the same observation
eight times and read as eight independent confirmations.

| # | Area | macOS-arm64 actual | Linux-x64 actual | Artifact | Exact commit | Deviation | Closure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1–8 | target preflight and metadata | `c64_platform_matrix` 15/15 | 15/15 | `evidence/c6.4/{macos-arm64,linux-x64}.json` | `8a23772` | none | MET |
| 9 | compiler/runtime compatibility | `c63_closure_evidence` 2/2 | 2/2 | same | `8a23772` | none | MET |
| 10–14 | output bytes, line termination, traps | `c64_platform_matrix` 15/15; `three_engine_differential` 88/88 | identical | same | `8a23772` | none | MET |
| 15–18 | paths, temp dirs, manifest escaping | within `c64_platform_matrix` 15/15 | identical | same | `8a23772` | none | MET |
| 19–20 | installed runtime, locked offline build | `c63_closure_evidence` 2/2, `c64_platform_matrix`, **and the release smoke's positive + negative pair** | identical | same, plus `release package smoke` | `8a23772` | none | MET |
| 21–22 | frozen workspace, three-engine | `workspace` 1560/1560 (2 classified ignores); `mir_differential` 132; `exec_snapshots` 4; `three_engine_differential` 109 | identical | same | `8a23772` | none | MET |
| 23 | determinism rerun | `match` | `match` | same | `8a23772` | none | MET |
| 24 | generated corpus | 131 cases, 131 AGREEMENT | identical, same observation hashes | `evidence/c6.5/{macos-arm64,linux-x64}-{summary,per-case}.json`, `c65-tier1-agreement.md` | `8a23772` | none | MET |
| 25 | Windows disposition | n/a | n/a | `evidence/c6.4/windows-x64-gap-report.md` | `8a23772` (probe green again) | G1, G3 CLOSED; G2, G4 open, none semantic | REPORT-ONLY |

**Records REFRESHED at `8a23772`, CI run 30221728539, all 15 jobs green** — the C6.5 corpus commands
are part of the qualification command set now, so these records are not the `4844702` ones with a row
appended: they are a fresh run of the whole matrix at the commit where the corpus landed. Both Tier-1
targets: 0 failed, 2 ignored (both classified, both tensor-track), 0 unclassified, 0 self-skipped,
determinism `match`, no deviations, `generated_corpus_status: PASS`. `qualification-summary.md`
reports TIER-1 AGREEMENT on identical per-command counts; `c65-tier1-agreement.md` reports TIER-1
CORPUS AGREEMENT on identical per-case observations. Both were re-verified locally against the
downloaded records rather than read off a green job.

The earlier `4844702` records are superseded: C6.4's own re-qualification rule invalidates a record
once the qualified path changes, and C6.5 changed `starkc/tests` extensively.

Tier-1 agreement is not a column here: it is a separate artifact,
`evidence/c6.4/qualification-summary.md`, produced by `scripts/compare-c64-evidence.py`. It
requires the two records to be for the two *different* Tier-1 targets, at the same commit, with the
same compiler/MIR/runtime/backend/layout versions and the same per-command observations. Two green
jobs are not agreement.

---

## Host assumptions found and disposed of

The §34 audit's findings, with where each was answered. Full statements in `WP-C6.4.md` §2.

| ID | Assumption | Disposition |
| --- | --- | --- |
| F1 | no target classification existed; the rustc host *was* the target | `src/target.rs`; rows 1–5 |
| F2 | `stark-64-v1` inherited by any triple | contract now comes from the named target and is checked against the request; row 6 |
| F3 | `i as usize >= v.len()` truncates on a 32-bit target — a missing trap, not just a wrong number | `narrow_index` compares in `u64`; preflight admits only 64-bit targets. Both, independently |
| F4 | the source-checkout runtime fallback is compiled in and unswitchable | `STARK_REQUIRE_INSTALLED_RUNTIME=1`; row 19 |
| F5 | generated crate built `--offline` but never `--locked`, with no lock | lock emitted from the linked runtime's own version; `--locked` added; row 20 |
| F6 | generated `Cargo.toml` escaped paths with Rust `Debug` | `toml_basic_string`; row 18 |
| F7 | executable suffix from the compiler's host | from the selected target; row 7 |
| F8 | (none) output bytes were already host-independent | recorded as safe; rows 10–12 observe it anyway |
| F9 | `/tmp` in the gate-7 comparator fixture | out of matrix; recorded in the Windows gap report as G4 |
| F10 | §8.3's error classification was half-absent | `TargetError` + `BackendDiagnostic::TargetRejected` + `BuildCommandError::TargetRejected`; rows 4–5 |
| B-1 | the Python scripts each carried their own tier table, and `build-release.py` classified Windows by substring | ONE description, `target-matrix.json`, read through `scripts/target_matrix.py` by every Python consumer and pinned to `src/target.rs` in BOTH directions by `target_matrix_json_matches_the_compiler`. Packaging derives suffix, archive format and installers from the exact entry and refuses unknown triples. G3 closed |
| R1 | the installed-runtime switch was proven in a unit test, not on the installed CLI | the CI release smoke runs under `STARK_REQUIRE_INSTALLED_RUNTIME=1` on all three platforms, plus a negative step that removes the installed runtime, leaves the checkout in place, and requires the build to fail; row 19 |
| R2 | a failed qualification job skipped the comparison | `if: always()` on `c64-tier1-comparison`; a missing or unreadable record is reported as a disagreement, never as a skip |
| R3 | the comparator could reach agreement from incomplete records | per-record validation before comparison: required metadata, self-consistent platform identity, full command set, corpus state, determinism; 43 fixture tests |
| R4 | ignored-test identities were truncated to the last `::` component | complete libtest names, stored in a list, with the named count required to equal Cargo's ignored count |
| R5 | two documentation claims were stale or overstated | float division cites NUM-FLOAT-OP-001 and CD-139 (CD-006 superseded); the `cfg`-absence claim is stated as reduced risk, with equivalence established by the cross-platform observations |
