# WP-C6.4 — Tier-1 Platform Matrix

**Track:** Gate C6 (all of C6 is Claude-owned)
**Status:** IMPLEMENTATION COMPLETE — awaiting the Tier-1 platform runs.
Recommended closure status: **`CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`**, and not before both
Tier-1 records exist (§5).
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

### 1.2 Generated corpus (§7.3) — disposition: BLOCKED-BY-C6.5

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

---

## 3. What was built

### 3.1 New

| Path | Purpose |
| --- | --- |
| `starkc/src/target.rs` | the canonical classifier: `Tier`, `TargetSpec`, `classify`, `select`, `preflight`, `TargetAvailability` (15 unit tests) |
| `starkc/tests/c64_platform_matrix.rs` | the permanent C6.4 suite: 14 tests across `target_preflight_*`, `portability_*`, `platform_*`, `determinism_*` |
| `starkc/scripts/run-c64-qualification.py` | the §10 qualification harness — one cross-platform entry point, one evidence pair |
| `starkc/scripts/compare-c64-evidence.py` | the §10.4 comparison: two records → one Tier-1 agreement claim, or a non-zero exit |
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
| `.github/workflows/ci.yml` | `c64-qualification` (2 Tier-1 jobs), `c64-tier1-comparison`, `c64-windows-gap` |

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

**No Tier-1 platform record exists yet.** Every "actual result" cell in `C6-PLATFORM-MATRIX.md`
Table B is empty, and `starkc/docs/compiler/evidence/c6.4/` holds no `.json` record. This is the
required state, not an omission: §35 says no real platform run means no platform claim, and a
locally simulated record would defeat the only purpose those files have.

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

## 5. What closure requires

Ordered, with nothing else outstanding:

1. push, and let `c64-qualification` produce both Tier-1 records at one commit;
2. `c64-tier1-comparison` reports TIER-1 AGREEMENT;
3. commit the two records plus `qualification-summary.md` into
   `starkc/docs/compiler/evidence/c6.4/`, and fill `C6-PLATFORM-MATRIX.md` Table B from them;
4. read the `c64-windows-gap` probe and resolve G1 in the gap report;
5. record the owner's closure decision.

Until (1) and (2), the honest status is `NOT-YET`. With them, and with row 24 still blocked, it is
`CANDIDATE-COMPLETE-BLOCKED-BY-C6.5-CORPUS`. `CLOSED` is not available while the generated corpus
does not exist.

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
