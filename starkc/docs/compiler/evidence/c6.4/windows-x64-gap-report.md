# C6.4 Windows Tier-2 gap report — `x86_64-pc-windows-msvc`

**Status:** OPEN — two bounded gaps (G2, G4), both harness, neither semantic. **G1 CLOSED**: the
C6.4 suite passes on Windows, 14/14. **G3 CLOSED**: the packaging script no longer classifies
Windows by substring.
**Authority:** `WP-C6-ENTRY.md` §36; execution plan §11.
**Baseline:** CI run 30190825336 at `8d894e8` — the first run of the C6.4 suite on Windows. Its
`build-and-test (windows-x64)` and `release package smoke (windows-x64)` jobs are green at the same
commit, as they were at `1ef4e8b` (`COMPILER-STATE.md` CD-142).

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

### G1 — The C6.4 platform suite on Windows — **CLOSED, portable**

| | |
| --- | --- |
| Area | platform matrix |
| Exact command | `cargo test --test c64_platform_matrix -- --nocapture` |
| Exact commit | `8d894e8`, CI run 30190825336, job `C6.4 windows tier-2 gap probe` |
| Actual result | **14 passed, 0 failed, 0 ignored** (2.84s) |
| Expected Tier-1 result | the same 14 |
| Classification | **portable** |
| User impact | none |
| Shared-code impact | none — and this is the substantive part of the result |
| C6 blocker? | no |
| Owner/gate | closed in C6.4 |

What the pass actually establishes, since these are the assertions most likely to break on a
platform with different conventions:

- `platform_stdout_is_exact_bytes_including_unicode_and_line_termination` — Windows produced the
  same bytes, and the explicit "no `\r`" assertion held. Nothing translates STARK's newline;
- `platform_trap_reports_category_provenance_and_exit_status` — same trap category, same
  `trapsite.stark:4:11` provenance, exit 101, and the pre-trap stdout prefix flushed (CD-120
  Contract B);
- `portability_builds_and_runs_under_paths_containing_spaces` / `…_unicode` — real builds under
  both, with the runtime install prefix inside the awkward path, so the TOML escaping (F6) holds
  against Windows-shaped paths;
- `portability_generated_crate_is_locked_and_network_free` — `--locked --offline` builds under
  Windows Cargo with the emitted lock;
- `target_preflight_classifies_windows_tier2_and_intel_mac_tier3` — the `.exe` suffix comes from
  the target table, and the running host classified itself as tier-2.

The probe stays `continue-on-error`: it gathers Windows facts without giving a Tier-2 platform a
veto over a Tier-1 claim. That it currently passes does not change the tier.

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

### G3 — Triple matching duplicated in the packaging script — **CLOSED**

| | |
| --- | --- |
| Area | target classification |
| Was | `starkc/scripts/build-release.py` — `windows = "windows" in target` |
| Classification | **portable**, and fixed rather than tolerated |
| Shared-code impact | it was a second, weaker copy of what `src/target.rs` owns: a substring test rather than a named-target lookup. Two failure modes, not one — a triple merely *containing* `windows` would be misclassified, and a triple the compiler does not name at all would still be packaged, producing an artifact nothing can qualify |
| C6 blocker? | no |
| Owner/gate | closed in C6.4 |

**Fixed** by `starkc/target-matrix.json`, read through `scripts/target_matrix.py` and pinned to
`src/target.rs` in both directions by `target_matrix_json_matches_the_compiler`. Packaging looks up
the exact triple and takes the executable suffix, archive format and installer pair from that entry;
an unknown triple raises `UnknownTarget` instead of being packaged. `test_build_release.py` pins
both failure modes, including the substring trap `sparc64-windows-unknown` — a triple that contains
the word and is not a target STARK names.

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

Windows is **Tier-2 with a bounded gap list**: G1 closed as `portable` on its first probe run, G3
closed by the target-matrix work, and two harness items (G2, G4) remain, neither semantic, neither
blocking a Tier-1 claim. No Windows-specific workaround has been introduced, so §11.3's condition "does not weaken
Tier-1 evidence" holds trivially — there is nothing Windows-specific to weaken it with.

The stronger reading, worth stating because it was not guaranteed: every observation the C6.4
matrix defines is **already platform-neutral on a third platform**, one that is not part of the
claim and had never run this suite. That is evidence about the shared runtime, not about Windows.

Updated, not rewritten, as further probe runs report.
