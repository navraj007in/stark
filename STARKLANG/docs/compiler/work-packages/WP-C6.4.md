# WP-C6.4 — Tier-1 Platform Matrix

**Track:** Gate C6 (all of C6 is Claude-owned)
**Status:** COMPLETE for everything C6.4 can reach, including the owner's second review round
(R1–R5, §2). **Both Tier-1 records exist and agree at `4844702`** (CI run 30192449131, all 11 jobs
green), taken under the strengthened comparator. Matrix rows 1–23 MET, row 25 REPORT-ONLY with G1
and G3 closed. **Row 24 is now PASS** (CD-161, `8a23772`).
Closure status: was **`CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`**, **accepted by the owner
2026-07-26 (CD-146)**. Not `CLOSED`, and no decision could have made it so — row 24 needs an
artifact that does not exist yet (§1.2, §5).
**Authority:** `starkc/docs/WP-C6-ENTRY.md` §§32–37 (tracked, normative), §5 (fixed decisions and
escalation classes), §6 (scope boundaries), §30 (runtime compatibility), §48 (validation at the
closure commit).
**Execution plan:** `WP-C6.4-Tier1-Platform-Matrix-Execution-Plan.md` (repo root, untracked owner
draft). Where it and `WP-C6-ENTRY.md` differ the entry document wins; on every point checked they
agree, so section references below (§8.x, §9.x, §10.x) cite the plan and §§32–37 cite the entry.
**Matrix:** `C6-PLATFORM-MATRIX.md` (this directory).
**Evidence:** `starkc/docs/compiler/evidence/c6.4/`.

---

## 0. Baseline pin (§7.1)

| Item | Value |
| --- | --- |
| Baseline commit | `5d2c85d` — WP-C6.3 CLOSED (CD-142), the commit this package opens on |
| Worktree (tracked) | clean at baseline |
| Audit host OS | macOS 26.5.2 (build 25F84), Darwin 25.5.0 |
| Architecture | arm64 |
| Host triple | `aarch64-apple-darwin` (Tier 1) |
| rustc | 1.93.0 (254b59607 2026-01-19), LLVM 21.1.8 |
| cargo | 1.93.0 (083ac5135 2025-12-15) |
| Python | 3.14.4 |
| Toolchain pin | `starkc/rust-toolchain.toml` → channel `stable`, components rustfmt + clippy |
| Layout contract | `stark-64-v1`, `layout_contract_version` 1 |
| Runtime version | `stark_runtime::version::RUNTIME_VERSION` = `0.1` |
| MIR | `MIR_VERSION` 0.1, `MIR_RUNTIME_SURFACE` 0.1-A9 |
| Backend | `BACKEND_VERSION` 0.1 |
| CI at baseline | green — the `1ef4e8b` run (Actions 30188909346) was all 7 jobs across linux-x64, macos-arm64, windows-x64 |

### 0.1 Why there are no measured V0 counts here

The execution plan's Tier V0 asks for a measured local baseline. It is deliberately not recorded,
for the reason `COMPILER-STATE.md` CD-142 gives at length: a local full suite here is ~60 minutes,
single-platform, and narrower in flags than the 4m44s three-platform CI run. Local measurement
would be a strictly worse number obtained at 12× the cost. The baseline this package is pinned to
is CI's, and its own qualification evidence comes from CI by construction (§4).

---

## 1. Entry conditions and sequencing

### 1.1 WP-C6.3 is closed

C6.3 closed at `5d2c85d` (CD-142) on a full `--workspace --all-targets --all-features` run across
all three platforms — the confirming run the earlier PARTIAL status was waiting for. C6.4 therefore
opens on an admitted runtime rather than a provisional one, and the concern recorded when C6.4 was
first opened (that Tier-1 evidence would describe a runtime whose loop emission was unconfirmed) no
longer applies.

### 1.2 Generated corpus (§7.3) — **row 24 CLOSED at `8a23772` (CD-161)**

**The blocker is discharged.** WP-C6.5 built the corpus (`starkc/tests/c6-corpus/`, `corpus_version`
0.5.0, 131 cases) and CI replayed it on both Tier-1 targets at one commit; `compare-c65-evidence.py`
found identical per-case observation hashes for all 131. Both C6.4 records at that commit carry
`generated_corpus_status: PASS`, `generated_corpus_version: 0.5.0`,
`generated_corpus_case_count: 131`, measured by the harness from the corpus lock rather than
supplied to it. Evidence: `starkc/docs/compiler/evidence/c6.5/`.

**Row 24 was the only thing standing between this package and `CLOSED`**, so C6.4's ceiling is no
longer `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`; the closure decision is the owner's to record.

The original disposition, kept because it is what the row was built against:

There is no deterministic **generated** corpus in the repository.
`starkc/tests/exec_snapshots/corpus.lock` is the **frozen execution corpus** (v1.2.0, 23 cases) — a
different artifact, owned by C3/C4 and already green. The `WP-C6.5` chapter of `WP-C6-ENTRY.md` (§§38–45; §41 is the deterministic
generator) owns it.

Fixed now so it cannot be quietly ticked later:

- the frozen corpus and `mir_differential` run as their own matrix rows and count as themselves;
- the generated-corpus row (matrix row 24) is `BLOCKED-BY-C6.5-CORPUS` from the outset;
- C6.4 does not implement C6.5's generator;
- every evidence record carries `generated_corpus_status: BLOCKED-BY-C6.5`, so the state is
  asserted rather than merely absent;
- after C6.5 lands, its corpus is re-run through the C6.4 harness on both Tier-1 targets, and only
  then does row 24 close;
- the best closure status available until then is `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`.

### 1.3 Windows is already green in CI, and that is not qualification

CI runs fmt, clippy, the full suite, `exec_snapshots`, the release-package tests and an
install-and-run smoke on `windows-latest`, and all of it is green. Per §36 and Review D question 12,
that is **not** a Tier-2 disposition and must not be read as Tier-1 qualification. What the green
run does and does not establish is set out explicitly in
`starkc/docs/compiler/evidence/c6.4/windows-x64-gap-report.md` §1.

---

## 2. Findings register, with dispositions

Ten findings (F1–F10) from the §34 audit. Classifications: **load-bearing** (affects a normative
observation or a required rejection), **contract-dependent** (safe only while an unenforced
assumption holds), **evidence** (the claim was unproven, not wrong), **test-only**.

### F1 — No target classification existed *(load-bearing, P2)* → FIXED

`backend/generated_rust/build.rs` derived the build's target from the `host:` field of `rustc -vV`,
and that was the compiler's entire notion of a target: no tier, no supported-target list, no
rejection, no host/target distinction. §33 requires all four, and §8.2 names host/target
interchangeability as prohibited. Nothing rejected an unsupported target, so a rustc or linker
failure would necessarily have been the first detector — a §14 stop condition.

**Fixed** by `starkc/src/target.rs`: one table of named targets, one `classify`, one `select`, one
`preflight`. Every other site now asks it.

### F2 — `stark-64-v1` was inherited by every triple *(load-bearing, P2)* → FIXED

`NativeBuildOptions::default` set `target_contract` to `"stark-64-v1"` unconditionally.
`layout::contract_for` correctly rejected an unknown *contract name*, but nothing mapped a *triple*
to a contract, so every triple — named or not, 64-bit or not — got `stark-64-v1`. §8.2's
prohibition is "do not allow an unknown triple to inherit `stark-64-v1` merely because it is
64-bit"; it was inherited without even that check.

**Fixed** two ways, because the requested contract is still an input: the target *declares* its
contract in the target table, and `build_and_link` rejects a build whose requested contract is not
the one the selected target declares (`TargetError::LayoutContractMismatch`). Resolving a valid
contract name is not the same as resolving the right one.

### F3 — `Vec` bounds checks truncated the index on a non-64-bit target *(contract-dependent, P1 if reachable)* → FIXED

`stark-runtime/src/vec.rs` checked bounds as `i as usize >= v.len()` with `i: u64`, at four sites:
`index_get` (the trapping `v[i]`), `remove`, `get_ref`, `get_mut_ref`. On a 32-bit target the cast
truncates first: `v[0x1_0000_0000]` on a one-element vector narrows to `0`, passes the check, and
returns element 0 instead of trapping `IndexOutOfBounds` — a silent wrong value where the language
requires a trap.

Unreachable on Tier 1 (both targets are 64-bit), and it is listed precisely because it is the class
§34 asks the audit to expose: a hidden host-width assumption that becomes load-bearing the moment
F2 stops being hypothetical.

**Fixed on both axes, independently.** `narrow_index` compares in `u64` and narrows only after the
range check, which is correct at every pointer width; and preflight admits only named targets, all
of which are 64-bit, pinned by
`target::tests::every_named_target_is_64_bit_and_uses_the_declared_contract`. Two independent
guards, because the one that has to be remembered is the one that gets forgotten.

### F4 — Installed-runtime independence was asserted, not proven *(evidence, P3)* → FIXED

`native_toolchain::discover_runtime` prefers the installed layout and then falls back to
`env!("CARGO_MANIFEST_DIR")/stark-runtime` — the source checkout as it existed when the compiler was
built, baked into the binary, with no way to disable it. CI's release-package smoke asserts the
installed runtime *file exists*; it never established that `stark build` *used* it. With the
runner's checkout still present and still matching the compiled-in path, a broken install layout
would fall back silently and the job would stay green. §10.6 states plainly that "a pass while the
source checkout remains an accidental dependency is invalid."

**Fixed** by `STARK_REQUIRE_INSTALLED_RUNTIME=1`
(`native_toolchain::REQUIRE_INSTALLED_RUNTIME_VAR`): with it set, discovery fails rather than
reaching the checkout, and the checkout is not even added to the attempted list. Pinned by
`portability_installed_runtime_requirement_refuses_the_checkout_fallback`, which asserts both
directions — the fallback is still the default, and the switch really removes it.

**And, after R1, by the real installed path.** The unit test above proves the switch works; it says
nothing about the binary users install. The CI release smoke now runs `stark run` and
`stark build --locked --offline` under the switch on all three platforms, followed by a negative
check: with the installed runtime moved aside and the checkout still present at the compiled-in
path, the same build must fail. Those two together are what make the installed-runtime claim an
observation rather than an inference.

### F5 — The generated crate was built `--offline` but never `--locked` *(evidence, P3)* → FIXED

The generated-crate Cargo invocation passed `--offline` and no `--locked`, and no `Cargo.lock` was
written. The `--locked`/`--offline` flags that already existed on `stark build` govern the **STARK
package graph** (`stark.lock`), a different artifact from the generated Cargo crate's resolution.

**Fixed** by emitting a `Cargo.lock` for the generated crate and passing `--locked`. The lock's
runtime version is read from the runtime crate actually being linked
(`read_runtime_version`), not hardcoded: the installed runtime is a different file from the one the
compiler was built beside, and a lock that disagrees with the crate it locks is one Cargo rejects.
The graph is two path-only packages with no `source` and no `checksum`, which is what makes
"offline" provable rather than warm-cache-dependent.

### F6 — The generated `Cargo.toml` escaped paths with Rust's `Debug` *(load-bearing on adversarial paths, P4)* → FIXED

`generated_cargo_toml` wrote `path = {:?}` — Rust's `Debug` used as if it were TOML quoting. The
two agree on the cases that occur constantly (a backslash becomes `\\`, a quote becomes `\"`, so
Windows paths happened to come out right) and disagree where hand-rolled escapes always disagree:
`Debug` renders a control character as `\u{7}` and a non-UTF-8 byte as `\xNN`, and TOML accepts
neither spelling.

**Fixed** by `toml_basic_string`, which escapes to TOML's own rules (`\u00XX`, four hex digits).
Pinned by a unit test over seven path shapes and by the end-to-end spaces/Unicode build rows.

### F7 — The executable suffix came from the host *(safe today, unblocked by F1)* → FIXED

The artifact name was built with `std::env::consts::EXE_SUFFIX`, the **compiler's** suffix. Correct
while host and target are identical — which is every build today — and wrong the instant a selected
target exists. **Fixed**: the suffix comes from `TargetSpec::executable_suffix`.

### F8 — Output bytes and line termination were already host-independent *(safe — recorded, then observed)*

`stark-runtime/src/output.rs` is byte-oriented and writes `b"\n"` explicitly; nothing consults a
host line-ending convention, and Rust performs no text-mode translation on Windows. The trap ABI
flushes stdout before aborting (CD-120 Contract B), so a mid-render trap's prefix is the same bytes
everywhere. No change required — but "audited safe" is not "observed equal", so matrix rows 10–12
assert the exact bytes and explicitly assert the absence of `\r`.

### F9 — `/tmp` hardcoded in the Gate-7 comparator fixture *(test-only, P5)* → RECORDED, out of scope

`starkc/tests/fixtures/gate7/rust-comparator/run.py` writes to `/tmp` at four sites. Gate-7 tensor
comparator, outside the C6 required matrix. Carried in the Windows gap report as G4. Everything
inside the C6 matrix uses `std::env::temp_dir()`.

### F10 — §8.3's error classification was half-absent *(load-bearing, P2)* → FIXED

| Required class | Before | Now |
| --- | --- | --- |
| `SupportedAndAvailable` | implicit, no type | `Ok(TargetSelection)`, carrying the resolved spec |
| `SupportedButToolchainMissing` | partial — rustc/cargo absence only | `TargetError::SupportedButToolchainMissing`, via an injectable probe |
| `UnsupportedByStark` | absent | `TargetError::UnsupportedByStark`, naming the Tier-1 targets |
| `HostOrTargetMetadataMismatch` | absent | `TargetError::HostOrTargetMetadataMismatch`, raised by `preflight` |
| `RuntimeCompatibilityMismatch` | present (`stark_runtime::version::check`) | unchanged — the runtime crate's own authority (§9.2) |
| `LinkerOrExternalToolFailure` | conflated with compile errors | still `BackendDiagnostic::BuildFailed`; see §6 |

Plus `TargetError::LayoutContractMismatch` — §8.3's list is a minimum, and a build that records one
target while measuring another deserves to say which two names disagreed.

The classes reach the CLI intact: `BackendDiagnostic::TargetRejected` →
`BuildCommandError::TargetRejected` → a diagnostic that names the supported Tier-1 targets for an
unsupported target and says "install the toolchain" for a missing one. Neither is reported as a
program error, and neither is `Unsupported` (which means "this backend increment does not lower
that construct").

### Second review round (owner directive, 2026-07-26) — R1–R5

The findings above are from my own §34 audit. The owner's review of the delivered package found
five more. All are fixed; each is stated as the defect, not as the fix.

**R1 — the installed-runtime proof was a unit test, not the real path.** `STARK_REQUIRE_INSTALLED_
RUNTIME=1` was proven to disable the checkout fallback *in a unit test*, while the actual release
smoke — the one that installs a package and runs the installed `stark` — did not set it. So the
thing shipped to users was still the unproven path. Fixed on all three platforms, and paired with
the negative half that makes the positive half mean something: with the installed runtime moved
aside and the checkout still present at the compiled-in path, `stark build` must FAIL. If it
succeeded, the passing build above proved nothing about which runtime it used.

**R2 — a failed qualification job silently skipped the comparison.** `c64-tier1-comparison`
depended on `c64-qualification` without `if: always()`, so a matrix failure skipped the comparison
entirely. §10.4 forbids workflow-level skipping standing in for an explicit TIER-1 DISAGREEMENT —
and a skipped job is worse than a failing one, because it reads as "not applicable" rather than
"not established". The comparison now runs after success, failure or cancellation, downloads
whatever exists, and reports a missing or unreadable record *as* a disagreement with a named cause.

**R3 — the comparator could reach agreement from incomplete evidence.** Two records that both
omitted a field agreed on it. The comparator now validates each record *before* comparing: required
metadata present and non-blank, self-consistent platform identity (`selected_target_triple ==
host_triple`, tier-1, 64-bit, declared layout contract), positive layout contract version and
compiler layout revision, every command of the fixed qualification set present and passing, corpus
status exactly `BLOCKED-BY-C6.5` with zero cases, determinism matched with non-blank hashes, and no
deviation, dirty worktree, quick mode, unclassified ignore or self-skipped test. Only then does it
compare — including per-command exit codes, all four counts, normative argv, and the **full
identities** of ignored and unclassified-ignored tests.

**R4 — ignored-test identities were truncated to their last `::` component.** Two modules can each
hold a `basic_case`; collapsing them would let a classified ignore vouch for an unrelated
unclassified one, and would make two records with different ignores compare equal. Complete libtest
names are now stored and compared, kept in a list so the count survives, and the harness fails when
the number of named ignores does not equal Cargo's reported ignored count — an unattributed ignore
cannot have been classified.

**R5 — two documentation statements were stale or overstated.** Review A said float division
follows CD-006; CD-006 was **superseded** by NUM-FLOAT-OP-001 and CD-139 (succession of normative
authority, not a reversal on the merits). And Review A(4) claimed the absence of `cfg` in the
runtime *proved* the platforms could not diverge. It does not: identical source can still diverge
through the host toolchain, LLVM, libc or floating-point behaviour beneath it. The accurate claim —
no target-conditional semantic implementation, therefore reduced divergence risk, with actual
equivalence established by the cross-platform observations — now appears in both this document and
`COMPILER-STATE.md`.

---

## 3. What was built

### 3.1 New

| Path | Purpose |
| --- | --- |
| `starkc/src/target.rs` | the canonical classifier: `Tier`, `TargetSpec`, `classify`, `select`, `preflight`, `TargetAvailability` (15 unit tests) |
| `starkc/tests/c64_platform_matrix.rs` | the permanent C6.4 suite: 15 tests across `target_preflight_*`, `portability_*`, `platform_*`, `determinism_*` |
| `starkc/scripts/run-c64-qualification.py` | the §10 qualification harness — one cross-platform entry point, one evidence pair |
| `starkc/scripts/compare-c64-evidence.py` | the §10.4 comparison: per-record validation, then comparison; two records → one Tier-1 agreement claim, or a non-zero exit |
| `starkc/scripts/target_matrix.py` | the one Python reader for the target matrix; every script asks it instead of carrying a table |
| `starkc/scripts/test_c64_scripts.py` | 43 fixture-driven tests for the harness, the comparator and the matrix reader |
| `starkc/target-matrix.json` | the repository-owned machine-readable target description, pinned to `src/target.rs` in both directions |
| `starkc/docs/compiler/evidence/c6.4/README.md` | what lands here, how, and why it is empty until CI fills it |
| `starkc/docs/compiler/evidence/c6.4/windows-x64-gap-report.md` | §36's Tier-2 disposition: four classified gaps (G1–G4), none semantic |
| `STARKLANG/docs/compiler/work-packages/C6-PLATFORM-MATRIX.md` | the frozen matrix: 25 rows, requirements and evidence tables |

### 3.2 Changed

| Path | Change |
| --- | --- |
| `src/backend/generated_rust/build.rs` | preflight before emission; contract checked against the target; `Cargo.lock` emitted; `--locked` added; TOML escaping; suffix from the target; manifest gains `host_triple`, `target_tier`, `target_pointer_width` |
| `src/backend/generated_rust/mod.rs` | `BackendDiagnostic::TargetRejected` |
| `src/native_build.rs` | `BuildCommandError::TargetRejected` |
| `src/bin/stark.rs` | the rejection diagnostic, per class |
| `src/native_toolchain.rs` | `STARK_REQUIRE_INSTALLED_RUNTIME` |
| `src/lib.rs` | `pub mod target` |
| `stark-runtime/src/vec.rs` | `narrow_index` — width-independent bounds checks at four sites |
| `.github/workflows/ci.yml` | `c64-qualification` (2 Tier-1 jobs), `c64-tier1-comparison` (`if: always()`), `c64-windows-gap`; the release smoke now runs under `STARK_REQUIRE_INSTALLED_RUNTIME=1` and is paired with a negative check on both platform branches |
| `starkc/scripts/build-release.py` | exact named-target lookup; suffix, archive format and installer pair come from the matrix entry, and an unknown triple is refused |
| `starkc/scripts/test_build_release.py` | four classification tests, including the substring trap `sparc64-windows-unknown` |

### 3.3 Deliberately not done

- **No `--target` flag.** C7 owns target selection and cross-compilation (§33's closing line). The
  `requested` parameter on `select`/`preflight` exists so the tests can drive every branch; the CLI
  passes `None`.
- **`BuildVersions` not extended.** Host/target metadata went into the compiler-side `build.json`,
  not into the record embedded in the binary: that record is the runtime crate's shared type with
  its own separately versioned surface (§9.2), and a binary that can only be a host build has
  nothing to disambiguate. Extending it would have bumped a runtime surface version to record a
  field with no reader.
- **No C6.5 generator.** §1.2.

---

## 4. Evidence status

**Both Tier-1 records exist and agree, at `4844702`** — CI run 30192449131, all 11 jobs green.

| | macOS-arm64 | Linux-x64 |
| --- | --- | --- |
| overall | PASS | PASS |
| passed | 1705 | 1705 |
| failed | 0 | 0 |
| ignored | 2, both classified | 2, both classified |
| unclassified ignores | none | none |
| self-skipped | 0 | 0 |
| deviations | none | none |
| determinism rerun | `match` | `match` |
| pointer width / layout | 64 / `stark-64-v1` v1 rev 1 | 64 / `stark-64-v1` v1 rev 1 |

Per-command counts are identical: `c64_platform_matrix` 15, `three_engine_differential` 88,
`mir_differential` 132, `exec_snapshots` 4, `c63_closure_evidence` 2, `conformance` 3, `workspace`
1461 (with the 2 classified ignores). `qualification-summary.md` reports **TIER-1 AGREEMENT**, and
the same verdict was reproduced locally by running the comparator against the downloaded records —
so the claim does not rest solely on a CI job having exited zero.

**Why the records name `4844702` while HEAD is later.** A record cannot describe the commit that
adds it — the file must exist before it can be committed. The commit that landed these (`eb3d27b`)
changes documentation, the state file and the evidence files themselves, and **no source, no test
and no script** (`git diff --stat 4844702..eb3d27b`), so the qualified artefact is unchanged. The
standing rule, not a caveat about this one: **any later commit touching `starkc/src`,
`starkc/tests`, `starkc/scripts`, `starkc/target-matrix.json` or `stark-runtime` invalidates these
records and requires a fresh qualification run.** That rule is what forced this round's retake —
R1–R5 touched the scripts, so the `61008f6` records could not stand even though they had passed.

**The earlier records were discarded, not carried forward.** `61008f6` produced two passing,
agreeing records; R1–R5 then strengthened the harness and comparator, and those records lack
`target_pointer_width`, `layout_contract_version`, `compiler_layout_revision` and `required_steps`
and carry truncated ignore identities. Run against the current comparator they are **refused**.
Keeping them would have claimed qualification from evidence this gate rejects.

### 4.1 The first CI run (`8d894e8`, run 30190825336) — and what it caught

Nine of eleven jobs green, including every pre-existing job on all three platforms. Two results
matter:

**The Windows gap probe passed, 14/14** — the first run of the C6.4 suite on a platform outside the
claim. Exact stdout bytes with no CRLF, identical trap category and `file:line:column` provenance,
exit 101, the flushed pre-trap prefix, `--locked --offline` under Windows Cargo, and builds under
spaced and Unicode paths with the runtime installed inside them. G1 in the gap report closes as
`portable`. This is evidence about the *shared runtime*, not about Windows.

**Both Tier-1 qualification jobs failed, and failed correctly.** The harness reported
`workspace: 2 test(s) ignored in a required command`. Those two ignores are pre-existing, opt-in
tensor-track tests needing external artifacts (`imports_and_verifies_checksum_pinned_reference_model`,
`real_inference_agrees_with_reference`) — legitimately outside a Core-runtime matrix. The defect was
in the harness, not the tree: §10.4 permits an ignored test "unless explicitly classified outside
the required matrix", and I had built the refusal without building the classification.

Fixed by naming them. `CLASSIFIED_IGNORES` is a **closed list with a reason per entry**, not a
count: counting would let a new ignore silently replace a retired one, which is exactly how a
required observation goes missing without anyone deciding it should. The harness now parses
`test <name> ... ignored` lines, fails on any name not on the list, *also* fails if a nonzero
ignored count cannot be attributed to names at all, and records `classified_ignores` (with
reasons) and `unclassified_ignores` in the evidence. The comparison gate treats
`unclassified_ignores` as a must-match field.

Worth stating plainly: the harness's first act was to fail a run I expected to pass, for a reason
that was real. That is the behaviour §19 asks for.

What has been run locally, and what it covers:

| Command | Result |
| --- | --- |
| `cargo test --lib -p starkc` | 463 passed (of which `target::` 15 and `backend::generated_rust::build` 16, 7 of them new) |
| `cargo test --lib -p stark-runtime` | 23 passed |
| `cargo test --test c64_platform_matrix` | 14 passed |
| `cargo test --test native_c5_2b_locals` | 2 passed — the first real proof that `--locked` plus the emitted lock builds under real Cargo |
| `cargo test --test native_build_cli` | 9 passed (2 updated: they pinned the old Cargo argv and now assert `build --locked --offline`) |
| `cargo test --test c63_closure_evidence` | 2 passed — installed runtime, offline build, version-mismatch detection |
| `cargo test --test native_c5_1b_skeleton --test native_c5_3_aggregates_enums` | 20 passed — the suites that assert the `build.json` shape |
| `cargo fmt --all -- --check` | clean |
| `run-c64-qualification.py --only …` | harness validated end to end; determinism probe reported `match` across two processes; correctly reported FAIL for a dirty worktree and a filtered run |
| `compare-c64-evidence.py` | validated on synthetic records: agreement exits 0; a diverging per-command count and two records from the same platform each exit 1 |

The full suite, strict clippy and the exhaustive cross-platform net are CI's, per CD-142.

---

## 4.5 Review passes (§12)

Performed against the tree, not from memory. Each answer names the check.

### Review A — semantic authority

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does any target-specific path redefine STARK semantics? | No. `grep -rn "cfg(target_" src` returns three hits, all in `open_in_browser` (`stark.rs`) — which opens generated documentation in a browser and touches no program semantics. |
| 2 | Does host integer width affect Core values? | No, and this was the one place it did. `size_of`/`align_of` answer from the declared contract (`layout.rs`, CD-067), and F3 removed the last host-width dependence in the runtime's checked-index surface. |
| 3 | Does host Rust behaviour replace a STARK rule? | No. Float width is carried per CD-140; **float division follows NUM-FLOAT-OP-001 and CD-139 — CD-006 is superseded by succession of normative authority**, not reversed on its merits (CD-139 records the succession); and every checked operation traps through the STARK trap ABI rather than Rust's own panic. |
| 4 | Can macOS and Linux select different semantic runtime paths? | **The runtime contains no target-conditional semantic implementation, reducing divergence risk.** `grep -rn "cfg(" stark-runtime/src`, excluding `cfg(test)`, returns nothing, so no target-conditional branch exists to diverge on. That is a risk reduction, not a proof: identical source can still diverge through the host toolchain, LLVM, libc, or floating-point behaviour beneath it. **Actual Tier-1 equivalence is established by the exact cross-platform qualification observations** — the per-command counts and byte-exact expectations compared by `compare-c64-evidence.py`, not by the absence of `cfg`. |
| 5 | Can a runtime mismatch execute user code? | No. `emit_program` emits `version::check` as the first statement of `fn main`, before the entry block; a mismatch prints and `exit(1)`s. |
| 6 | Does any normalisation hide output, trap, Drop or provenance differences? | The three-engine harness normalises trap *messages* to categories (deliberate — no canonical message table exists), but the C6.4 rows assert exact bytes and exact `file:line:column` with the same expectation on every platform, so agreement is enforced by a shared expectation rather than by a comparison that could be loosened. |
| 7 | Does a platform error get misreported as a STARK trap? | No. Exit 101 is reachable only from `trap::abort`/`abort_with_message`. A rustc failure, a missing artifact or a linker failure is `BackendDiagnostic::BuildFailed`, and an unsupported target is now `TargetRejected`. |
| 8 | Can a Tier-2 workaround alter Tier-1 semantics? | No Windows-specific branch exists in the runtime or the backend to carry such a workaround (see 4), and none was added: Windows passes the C6.4 matrix on the same code Tier-1 runs. Were one ever added, only the Tier-1 observations would settle whether it changed anything. |

### Review B — target and compatibility architecture

| # | Question | Answer |
| --- | --- | --- |
| 1 | Is target classification centralised? | Yes, in both languages after R5's companion fix: `src/target.rs` in Rust, and `target-matrix.json` read through `scripts/target_matrix.py` in Python, pinned to each other in both directions. No substring matching remains. |
| 2 | Are host and selected target distinct? | Yes — `TargetSelection` carries both, and `build.json` records both. |
| 3 | Is layout selection versioned? | Yes — `TargetLayout::identity` carries the contract name, contract version and compiler revision, all three recorded in the manifest and in the build key. |
| 4 | Are compiler/MIR/runtime/backend versions checked? | Recorded, all of them; *checked* at run time only for `runtime_version`, which is the runtime crate's own authority under §9.2. Unchanged by this package. |
| 5 | Does unsupported-target rejection occur early? | Yes — from the `rustc -vV` probe, before emission and before Cargo. Pinned by a unit test that drives it from a synthetic transcript. |
| 6 | Is missing toolchain distinguished? | Yes — a separate `TargetError` variant with different prose, pinned by two tests. |
| 7 | Can an unknown target inherit a supported layout? | No — `classify` is an exact-match lookup, and the contract is checked against the target's declaration. |
| 8 | Can installed-runtime discovery fall back to the checkout? | By default yes (unchanged, deliberate); under `STARK_REQUIRE_INSTALLED_RUNTIME=1` no, and the checkout is not even attempted. Note the dev-only `NativeToolchainOptions::development()` still points at the checkout by construction — that is the direct-backend-test entry point, not the CLI path. |

**Finding B-1 — fully resolved after R1–R5, not merely checked.** The Python scripts cannot call
`src/target.rs`, so each carried its own tier table, and `build-release.py` classified Windows with
the substring test `"windows" in target` — wrong in two directions, since it would classify an
unknown triple containing the word AND would package a triple the compiler does not name at all.

There is now **one** description, `starkc/target-matrix.json`, read by every Python consumer through
`scripts/target_matrix.py`, and pinned to `src/target.rs` in **both** directions by
`target_matrix_json_matches_the_compiler`. Checking one direction catches half the drift — the half
noticed first. Packaging derives executable suffix, archive format and installer pair from the exact
entry and raises `UnknownTarget` otherwise; `test_build_release.py` pins that with four cases,
including the substring trap `sparc64-windows-unknown`. G3 in the Windows gap report is closed by
this.

### Review D — evidence and CI

| # | Question | Answer |
| --- | --- | --- |
| 1 | Do both Tier-1 jobs run against the same commit? | Yes — each is given `$GITHUB_SHA` and compares it with `git rev-parse HEAD`; the comparison then requires both records to name the same commit. |
| 2 | Are exact test counts recorded? | Yes, per command and in aggregate. |
| 3 | Are ignored/skipped tests visible? | Yes — by **complete libtest name** (R4), split into `classified_ignores` (with reasons) and `unclassified_ignores`, and the count must equal the number of named ignores. §4.1. |
| 4 | Does a failed command stop qualification? | Every step runs (so one failure does not hide the rest), and any failure makes `overall_result: FAIL` and exits non-zero. After R2 a failed qualification job no longer skips the comparison: it runs under `if: always()` and reports the absent record as a disagreement. |
| 5 | Are artifacts uploaded even on failure? | Yes — `if: always()` on all three uploads. Demonstrated: the failed run above still produced its evidence. |
| 6 | Can a partial matrix be mistaken for a complete claim? | No — `--only` and `--quick` each record a deviation naming themselves as not a qualification claim, and the comparison rejects any record with `quick_mode` set. |
| 7 | Is the generated corpus nonempty? | **Yes, as of CD-161**: 131 cases at `corpus_version` 0.5.0, replayed on both Tier-1 targets with identical per-case observations. Every record now carries `generated_corpus_status: PASS`. §1.2. (Originally: it did not exist, and every record asserted `BLOCKED-BY-C6.5`.) |
| 8 | Is determinism actually a second run? | Yes — a second **process**, compared on a printed build key and generated-source hash. Verified locally: `match`. |
| 9 | Is installed-runtime execution outside the checkout? | Yes, and proven on the real path after R1: `c63_closure_evidence` builds against a copied runtime; the CI release smoke runs the installed `stark` under `STARK_REQUIRE_INSTALLED_RUNTIME=1`; and a negative step removes the installed runtime, leaves the checkout in place, and requires the build to fail. |
| 10 | Is offline operation actively proved? | `--locked --offline` with an emitted lock over a path-only graph with no `source`/`checksum` — nothing in the graph *can* reach a registry. |
| 11 | Are platform observations compared rather than listed? | Yes, and after R3 each record is *validated* before the two are compared, so agreement is unreachable from incomplete evidence. 43 fixture tests in `scripts/test_c64_scripts.py` cover both the acceptance and every refusal. |
| 12 | Can Windows green be mistaken for Tier-1 qualification? | Structurally no: Windows produces a *gap-probe log*, never an evidence record; the comparison requires the two records to be the two different Tier-1 triples; and the gap report opens by stating what Windows green does and does not prove. |

### Review E — adversarial probes

| Probe | Where | Result |
| --- | --- | --- |
| unknown but plausible 64-bit target | `target_preflight_rejects_unknown_targets_of_either_width` (musl, aarch64-linux) | rejected |
| unknown 32-bit target | same test (`i686`) | rejected |
| supported target, unavailable toolchain | `NoToolchain` probe | distinct class |
| workspace path with spaces / with Unicode | two `portability_*` tests | build and run |
| install prefix with spaces / with Unicode | same two tests (the runtime is installed inside the awkward path) | build and run |
| generated manifest with a Windows-style path | `manifest_paths_are_escaped_to_toml_rules…` | `C:\Users\…` escapes correctly |
| executable path with spaces | the spaced-path test runs the produced binary | exit 0 |
| nonzero program status | `platform_trap_reports_…` | exit 101 |
| trap with exact source provenance | same | `trapsite.stark:4:11` |
| output with Unicode and a final newline | `platform_stdout_is_exact_bytes…` | exact bytes, no `\r` |
| repeated clean build | `determinism_two_clean_builds_agree…` | identical key and source |
| runtime metadata mismatch | `c63_closure_evidence::a_runtime_version_mismatch_is_detected` | rejected before user code |
| source checkout unavailable during the installed-runtime test | `portability_installed_runtime_requirement_refuses_the_checkout_fallback`, **and** the CI negative step on all three platforms | discovery fails; the real installed CLI fails too |
| offline build with no reachable registry | `portability_generated_crate_is_locked_and_network_free` | path-only graph |
| an intentionally skipped required test fails qualification | **CI run 30190825336** | proved by a real failure, not a simulation (§4.1) |
| file-not-found mapping | — | **not probed.** `std-full` file operations are excluded from C6.3 and absent from every engine, so there is no mapping to probe |

---

## 5. What closure requires

1. ~~let `c64-qualification` produce both Tier-1 records at one commit~~ — **done** at `4844702`;
2. ~~`c64-tier1-comparison` reports TIER-1 AGREEMENT~~ — **done**, and reproduced locally;
3. ~~commit the two records plus `qualification-summary.md`, and fill Table B~~ — **done**;
4. ~~read the `c64-windows-gap` probe and resolve G1~~ — **done**: 14/14 on Windows, G1 closed as
   `portable` (§4.1). G3 is also closed, by the target-matrix work;
5. ~~record the owner's closure decision~~ — **done**: accepted 2026-07-26, CD-146.

All five steps are done. The status is `CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`, accepted. `CLOSED` is not available and will not be until C6.5's
generated corpus exists and replays through this harness on both Tier-1 targets (§1.2) — a decision
about C6.5's schedule, not about whether C6.4 did its work.

### 5.1 Closure checklist (§19), as it now stands

| Group | State |
| --- | --- |
| baseline and matrix | commit pinned, versions recorded, matrix complete (25 rows), corpus sequencing resolved |
| target preflight | host and selected target identified separately; Tier-1 accepted; unsupported rejected before Cargo; missing toolchain distinguished; layout and suffix selected from the target; metadata recorded; mismatch rejects before user code |
| portability | all eleven §34 categories audited; ten findings (F1–F10) dispositioned, eight fixed; no substring target matching remains anywhere |
| evidence | complete — harness, comparator and CI wiring tested by 43 fixture tests; both Tier-1 records committed at `4844702` with no deviations and no unclassified ignores |
| Tier-1 agreement | **established** at `4844702`, and reproduced locally against the downloaded records |
| Windows | real run inspected, gap report complete, G1 and G3 closed, G2 and G4 open and classified, none semantic |
| reviews and records | A, B, D, E complete (§4.5) and corrected by R1–R5; C is the §2 register; ledger and state updated; **owner decision recorded (CD-146)** |

---

## 6. Escalation watch

Nothing in this package required CE1–CE9, and nothing in it changed a normative semantic. Two
items are flagged in advance rather than discovered later:

- **CE4** would be required if a second named layout contract were ever needed (adding a 32-bit
  target is the obvious trigger), or if F3's fix had changed trap semantics rather than preserving
  them at every width. It did not: the same condition traps, computed without narrowing.
- **CE9** would be required for a change to install or runtime-discovery *behaviour* beyond gating
  an existing fallback. `STARK_REQUIRE_INSTALLED_RUNTIME` is opt-in and default-off, so the
  shipped behaviour is unchanged.

One item is deliberately **left open** rather than fixed: `LinkerOrExternalToolFailure` is still
conflated with generated-crate compile errors inside `BackendDiagnostic::BuildFailed` (F10's last
row). Separating them means classifying rustc's own output, which is a different kind of work from
target classification and has no bearing on any Tier-1 observation. Recorded here so it is a known
gap rather than an oversight.
