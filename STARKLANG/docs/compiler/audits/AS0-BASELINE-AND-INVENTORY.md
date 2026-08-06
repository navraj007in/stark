# AS0 — Baseline, reproduction and authority inventory

**Status:** IN PROGRESS. Reproductions and the pipeline/JSON inventories are complete; the
characterization matrix and three delegated inventories are outstanding (§7).
**Date opened:** 2026-08-06.
**Owning packet:** `WP-ARCHITECTURE-STABILIZATION.md` §5, AS0.
**Claim under test:** every later packet begins from a reproducible defect or a complete authority
inventory, not from a module-size impression.

---

## 1. Work-item status

| # | AS0 work item | Status |
| ---: | --- | --- |
| 1 | reproduce package-root source duplication | **DONE** §2 |
| 2 | reproduce the provenance half separately | **DONE** §2 |
| 3 | two-checkout comparison: source maps, MIR dumps, build keys | **DONE** §2.3 — closed by AS1a |
| 4 | inventory pipeline assemblies by entry point | **DONE** §3 |
| 5 | bounded characterization matrix | **DONE** §3.2 |
| 6 | inventory explicit and implicit callable execution sites | **OUTSTANDING** §7 |
| 7 | adopt the `WP-C7.8-RB0` predicate inventory | **OUTSTANDING** §7 |
| 8 | inventory JSON parsers, serializers, RFC 8259 deviations | **DONE** — `AS0-MANIFEST-STRICTNESS-AUDIT.md` |
| 9 | record performance/size baselines | **PARTIAL** §5 |
| 10 | execute the `WP-ENGINE-INDEPENDENCE.md` AS0 scope | **OUTSTANDING** §7 |
| 11 | record and run the pinned `stark-samples` suite | **OUTSTANDING** §7 |

AS0 does not exit until every row is DONE or explicitly deferred by decision.

## 2. Reproductions

Committed as `starkc/tests/as0_source_identity.rs`, three tests, all passing. **They assert the
current, defective behaviour on purpose.** AS0 reproduces; AS1a fixes; AS1a's checkpoint is these
tests failing and being flipped to the corrected assertion. A test asserting the intended behaviour
and carrying `#[ignore]` would prove nothing in the meantime.

Each assertion is labelled DEFECT or INVARIANT in the source.

### 2.1 Two source records for one physical file — CONFIRMED

A package analysis produces **two** `SourceRecord`s for the single entry file: one named by its
absolute checkout path, one by its logical `<package>/<path>` name.

`analyze_project`'s `ProjectInput::Package` arm builds `root_file` from
`package.entry.to_string_lossy()` (`src/analysis.rs:450`), while `parse_package_graph` names the
same file logically via `logical_package_path` (`src/parser.rs:207`). `build_source_map` sees two
names and interns two records.

### 2.2 The real entry is a module with no package — CONFIRMED

Worse than a duplicate record, and not previously stated:

- the record classified `Root` is the **phantom absolute one**;
- the logical entry is classified `Module`;
- **every** package file carries `package: None`.

`build_source_map` attributes a package by testing whether a source *name* starts with the package
entry's absolute parent (`src/analysis.rs:589-594`). After DEV-113 made package files logically
named, that predicate can never match. **The package-attribution branch is dead code on the package
path**, and the `Root`/`Module` split is decided against the phantom.

### 2.3 Relocation changes the source map — CONFIRMED (partial item)

Identical sources staged at two different absolute roots do not observe identically: the source
maps differ. The test also pins the INVARIANT that the *logical* names are already
relocation-stable — DEV-113 got that right — which localises the whole defect to one construction
site rather than to the naming scheme.

**Closed by AS1a.** The MIR file-table half is now pinned by
`the_mir_file_table_is_logical_and_relocation_stable`: no absolute path reaches `program.files`, and
the table is identical from two independent roots. Since `build_key_input` writes those names
verbatim into the build key's `[sources]` section
(`src/backend/generated_rust/build.rs:625-631`), the key no longer varies with the checkout
location. The pre-fix consequence was a checkout-dependent cache key, not miscompilation — no span
referred to `FileId(0)`, so nothing reached generated code, and the metamorphic tests that pin
canonical symbols and trap provenance were correct throughout and still pass.

## 3. Pipeline inventory — exact set

Every production site that assembles parse → resolve → typecheck. Test-module call sites are
excluded and were checked individually for `#[cfg(test)]` rather than by file.

**Through the driver (`analyze_project`) — 5:**

| Entry point | Site |
| --- | --- |
| `stark build` (package, ± provider overlays) | `src/native_build.rs:710,712` |
| `starkc` analysis path | `src/main.rs:562` |
| deployment analysis | `src/deploy/mod.rs:100` |
| LSP (package and single-file) | `src/lsp/server.rs:314,317` |
| documentation generation | `src/doc_gen/mod.rs:80` |

**Bypassing the driver — 6:**

| Entry point | Site | Owning fn |
| --- | --- | --- |
| `stark check` / `stark run` (package) | `src/bin/stark.rs:184-193` | `main` |
| `stark test` | `src/bin/stark.rs:1454` | `cmd_test` |
| `stark` standalone program | `src/bin/stark.rs:1614-1619` | `run_standalone_program` |
| `starkc run` | `src/main.rs:626-629` | `cmd_run` |
| `starkide` compile | `src/bin/starkide.rs:689` | `Ide::compile` |
| ONNX signature verification (resolve-only) | `src/onnx/verifier.rs:152` | `extract_declaration` |

**Eleven production assemblies, six of which bypass the shared pipeline.**

Two corrections to earlier working assumptions, which is the point of an exact set:

- The count is **six, not four**. `starkide` and the ONNX verifier were missed by a search that
  looked only at `main.rs` and `bin/stark.rs`.
- **There are three shipped binaries, not two** — `starkc`, `stark` and `starkide`, all
  auto-discovered from `src/main.rs` and `src/bin/`. AS2's migration list must name `starkide`
  explicitly or it will be left behind.

The ONNX verifier is a *partial* assembly (resolve without typecheck) and should be classified
separately in AS2: it may not need the full session.

### 3.1 The assemblies are not behaviourally identical

Consolidation therefore *chooses* a behaviour. Without a captured baseline, "the entry points now
agree" is satisfied equally well by every assembly having silently changed. §3.2 is that baseline.

### 3.2 Characterization matrix

Committed as `starkc/tests/as0_characterization.rs` with its baseline at
`starkc/tests/as0-characterization/BASELINE.txt`. Entry points are driven as real subprocesses;
status, stdout and stderr are pinned with temp paths scrubbed to `<TMP>` and timings to `<TIMING>`.
Verified deterministic across consecutive runs. Regenerate deliberately with
`STARK_UPDATE_CHARACTERIZATION=1`; committing a regenerated baseline asserts every diff was
intended.

Rows: valid package, invalid root source, invalid dependency source (for package entry points);
valid and invalid single file (for file entry points). Assemblies that have no non-interactive
surface are recorded `NOT-APPLICABLE` with the reason — `stark build` (needs a host toolchain;
covered by `native_build_cli`), `starkide` (interactive TUI), LSP (stdio JSON-RPC), deploy/doc_gen
(single-file driver callers), and ONNX verification (partial assembly: resolve without typecheck).

**Divergences found. These are findings for AS2 to resolve consciously, not defects to fix here.**

**D1 — a resolve error suppresses every type error.** The fixture has two errors: an undefined
variable (E0200, resolve) and a `Bool`/integer mismatch (typecheck). Only the resolve error is ever
reported, by every package entry point, because they return at the first phase that produced errors
rather than continuing. "Ordered diagnostics" therefore pins a list of length one. AS2's exit
criterion 2 compares ordered diagnostic structures across entry points; it should not be read as
evidence that multi-phase diagnostics work.

**D2 — success reporting differs across the three package commands.** On the same valid package:
`stark check` prints `probe: OK`; `stark run` prints **nothing at all**; `stark test` prints a test
summary. Three commands, three conventions.

**D3 — failure summaries differ between the package and single-file paths.** Package commands end
with `<package>: package compilation failed`; `starkc check` ends with `<file>: 1 error(s)`;
`starkc run` prints the diagnostic and **no summary line at all**. A tool parsing compiler output
has to special-case each.

**D4 — naming is split by input kind, correctly, and AS2 must preserve it.** Package sources are
logically named (`probe/src/main.stark`, and a dependency's error attributes to `lib/src/lib.stark`
with its own package). Single-file sources are path-named. That is `SourceFile::name`'s documented
contract — a single-file compile has no package, so the path is not identity-bearing there.
Unifying these under one driver would be a regression, not a consolidation.

**D5 — dependency attribution is correct at the CLI surface.** An error inside a dependency reports
against the dependency's own logical file and line, not the root's. This is the property AS1a's
provenance fix protects, now visible in shipped output rather than only in a unit test.

## 4. JSON inventory

Complete, recorded separately in `AS0-MANIFEST-STRICTNESS-AUDIT.md`. Headline: two in-tree parsers
that disagree with each other on 9 of 12 constructs; `package.rs` conforms on 3/12,
`lsp/protocol.rs` on 7/12; one silent-corruption correctness defect (F1) eligible for live-defect
pre-emption.

## 5. Baselines

Recorded 2026-08-06 on macOS (darwin 25.5.0), after the Cranelift retirement.

| Metric | Value | Command |
| --- | --- | --- |
| clean `cargo check --all-targets` | **8s** (was 21s) | isolated `CARGO_TARGET_DIR`, `rm -rf` first |
| clean check target-dir size | **463M** (was 529M) | `du -sh` |
| `cargo test --lib` | **523 passed, 0.26s** | warm |
| lockfile packages | **12** total, **9** third-party | `grep -c '^\[\[package\]\]' Cargo.lock |
| `starkc` binary | 3.0M | release, pre-retirement build |
| `stark` binary | 4.4M | release, pre-retirement build |
| `starkide` binary | 2.2M | release, pre-retirement build |

Binary sizes predate the Cranelift retirement but are unaffected by it: those crates were
`[dev-dependencies]` and never linked into a shipped binary. The lockfile count, by contrast, is
post-retirement and is where the change shows.

**Outstanding baselines:** native build time (end-to-end `stark build` of a provider-backed
package) and LSP change latency. Both need a measurement harness rather than a single command.

## 6. Third-party dependency surface

Nine crates, all reachable from `sha2`: `block-buffer`, `cfg-if`, `cpufeatures`, `crypto-common`,
`digest`, `hybrid-array`, `libc`, `sha2`, `typenum`. The compiler's own direct dependency is `sha2`
alone, plus the two workspace members. This is worth restating because AS5 may propose adopting a
vetted JSON library, and the baseline it would be measured against is *nine transitive crates
total*.

## 7. Outstanding work

| Item | Why it is not done yet | Size |
| --- | --- | --- |
| callable execution-site inventory (item 6) | feeds AS3, not AS1a/AS2 | large |
| `WP-C7.8-RB0` predicate inventory (item 7) | delegated to that packet; adopt, do not duplicate | medium |
| `WP-ENGINE-INDEPENDENCE.md` AS0 scope (item 10) | separate approved subpacket with its own record | medium |
| pinned `stark-samples` run (item 11) | needs the suite pinned by commit hash first | small |
| native build time / LSP latency (item 9) | need a harness | small |

**AS1a is complete** — its two dependencies were discharged in §2 and the packet has landed.

**AS2 is now unblocked.** The characterization matrix it depends on is committed (§3.2), with five
recorded divergences it must resolve consciously rather than silently. The remaining outstanding
items feed AS3, AS5, AS8 and C10 — none of them gates AS2.
