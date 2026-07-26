# C6.4 Windows Tier-2 gap report — `x86_64-pc-windows-msvc`

**Status:** OPEN — probe added, first Windows run of the C6.4 suite not yet observed.
**Authority:** `WP-C6-ENTRY.md` §36; execution plan §11.
**Baseline:** the last fully green CI run is `1ef4e8b` (Actions run 30188909346, all 7 jobs,
including `windows-x64`), recorded in `COMPILER-STATE.md` CD-142.

Windows is **Tier 2**. Its incompleteness does not block a Tier-1 C6 claim. A *semantic*
divergence in shared code is a real defect regardless of tier, and is not dismissible as "Windows
is Tier 2" (§36, execution plan §11.2).

---

## 1. What Windows CI proves today, exactly

This matters because Review D question 12 is precisely "can Windows green status be mistaken for
Tier-1 qualification". Stating what the green run covers is what makes the answer checkable.

| Job | Command | What it establishes on Windows |
| --- | --- | --- |
| `build-and-test (windows-x64)` | `cargo fmt --all -- --check` | formatting |
| | `cargo clippy --workspace --all-targets --all-features -- -D warnings` | lints, all features |
| | `cargo test --workspace --all-targets --all-features` | the whole suite, including every `native_*` suite — so the generated-Rust backend really does build and run programs on Windows |
| | `cargo test --test exec_snapshots` | the frozen execution corpus |
| | `python scripts/test_build_release.py` | package structure and both installers, including the `.exe`/`.zip`/`.ps1` Windows shapes |
| `release-package-smoke (windows-x64)` | `install.ps1` → `stark.exe --help` → `stark run` → `stark build --locked --offline` → `app.exe` | an installed Windows toolchain builds and runs a STARK workspace end to end |

**What it does not establish.** None of it is a C6.4 evidence record: no measured platform
identity, no counts attached to a commit, no determinism rerun, no comparison against a Tier-1
record, and no assertion that Windows observations *equal* Tier-1 observations. Green means "the
suite passes there", not "Windows agrees with macOS and Linux".

---

## 2. Gaps

Every row carries exactly one primary classification from §11.1's list.

### G1 — The C6.4 platform suite has never run on Windows

| | |
| --- | --- |
| Area | platform matrix |
| Exact command | `cargo test --test c64_platform_matrix` |
| Exact commit | added in this work package; not yet run on `windows-latest` |
| Actual result | **not yet observed** |
| Expected Tier-1 result | 14 passed, 0 failed, 0 ignored |
| Classification | **harness adaptation** (provisional — it becomes `portable` or a real defect once observed) |
| User impact | none directly; it is an evidence gap |
| Shared-code impact | the suite asserts exact stdout bytes, `\n` (never CRLF), trap category, `file:line:column` provenance and exit 101 — all shared runtime code, so a Windows failure here would be a shared defect, not a Windows workaround |
| Bounded fix estimate | unknown until the probe runs |
| C6 blocker? | **no** (Tier 2) |
| Owner/gate | C6.4 |
| Evidence | CI job `c64-windows-gap`, artifact `c64-windows-gap-probe` |

The probe is `continue-on-error` by design: it must gather Windows facts without giving a Tier-2
platform a veto over a Tier-1 claim.

### G2 — Two different installer paths assert the same thing

| | |
| --- | --- |
| Area | installed-runtime qualification |
| Exact command | `.github/workflows/ci.yml` → `release-package-smoke`, bash branch vs `pwsh` branch |
| Actual result | both green at `1ef4e8b` |
| Classification | **harness adaptation** |
| User impact | none |
| Shared-code impact | none — the divergence is in the CI script, not in the product |
| Bounded fix estimate | moderate; folding both into one Python driver is the §9.11 preference but is not required for a Tier-1 claim |
| C6 blocker? | no |
| Owner/gate | C6.4c or later |

The two branches make equivalent assertions in different languages, so a change to one can drift
from the other silently. Recorded rather than fixed: it is evidence-shaping, not semantics.

### G3 — Triple matching is duplicated in the packaging script

| | |
| --- | --- |
| Area | target classification |
| Exact location | `starkc/scripts/build-release.py:147` — `windows = "windows" in target` |
| Classification | **portable** (works today) |
| Shared-code impact | it is a second, weaker copy of the classification `src/target.rs` now owns: a substring test, not a named-target lookup. A triple that merely contains `windows` would be misclassified, and a Windows triple the compiler does not name would still be packaged |
| Bounded fix estimate | small — have the packaging script take the suffix/installer shape from a compiler-emitted target description rather than re-deriving it |
| C6 blocker? | no |
| Owner/gate | C6.4c or C7 (which owns the user-facing target feature anyway) |

Recorded because §8.2 explicitly forbids duplicating triple matching across CLI, builder, backend,
tests and scripts, and this is the one remaining copy after C6.4a centralised the Rust side.

### G4 — `/tmp` is hardcoded in the Gate-7 comparator fixture

| | |
| --- | --- |
| Area | temporary directories |
| Exact location | `starkc/tests/fixtures/gate7/rust-comparator/run.py:45,121,124,145` |
| Classification | **harness adaptation** |
| Shared-code impact | none — Gate-7 tensor comparator, outside the C6 required matrix |
| C6 blocker? | no |
| Owner/gate | the tensor track |

Everything inside the C6 matrix uses `std::env::temp_dir()`; this is the one survivor and it is
out of scope.

---

## 3. Unsupported-target behaviour on Windows

`x86_64-pc-windows-msvc` is a **named** target (`src/target.rs`), so preflight admits it, selects
`stark-64-v1`, and selects the `.exe` suffix from the target rather than from the compiler's host.
An *unnamed* triple is rejected on Windows by exactly the same code path as everywhere else, before
Cargo runs — §11.2's "unsupported Windows target requests must fail clearly" is satisfied by the
shared classifier, not by a Windows-specific branch.

---

## 4. Disposition

Windows is **Tier-2 with a bounded gap list**: one evidence gap awaiting its first probe run (G1)
and three harness/portability items (G2–G4), none of them semantic, none blocking a Tier-1 claim.
No Windows-specific workaround has been introduced, so §11.3's condition "does not weaken Tier-1
evidence" holds trivially.

This report is updated — not rewritten — when the `c64-windows-gap` probe first reports.
