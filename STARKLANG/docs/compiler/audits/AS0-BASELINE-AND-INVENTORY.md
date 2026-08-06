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
| 3 | two-checkout comparison: source maps, MIR dumps, build keys | **PARTIAL** §2.3 — source maps done |
| 4 | inventory pipeline assemblies by entry point | **DONE** §3 |
| 5 | bounded characterization matrix | **OUTSTANDING** §7 |
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

**Not yet done for this item:** the MIR file-table and native build-key halves. The data flow is
established by reading — `analysis.root_file` is passed to `lower_program_with_providers`
(`src/native_build.rs:749`), becomes `FileId(0)` in `ProgramMeta::build`
(`src/mir/lower.rs:260-263`), and `build_key_input` writes `file.name` verbatim into the `[sources]`
section of the build key (`src/backend/generated_rust/build.rs:625-631`) — but it is not yet pinned
by a test. The consequence is a checkout-dependent cache key, not miscompilation: no span refers to
`FileId(0)`, so nothing reaches generated code. Existing metamorphic tests already pin canonical
symbols and trap provenance and are unaffected.

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

Established by reading, and the reason AS2 needs a characterization baseline before it consolidates:

- `src/bin/stark.rs:172` builds its root `SourceFile` with the absolute entry path and **no**
  `disk_path`; `analyze_project` sets no `disk_path` either but its overlay arm reads overlaid
  content while its plain arm reads from disk.
- Consolidation therefore *chooses* a behaviour. Without a captured baseline, "the entry points now
  agree" is satisfied equally well by every assembly having silently changed.

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
| characterization matrix (item 5) | the largest AS0 deliverable; 5 row types across 11 assemblies | large |
| callable execution-site inventory (item 6) | feeds AS3, not AS1a/AS2 | large |
| `WP-C7.8-RB0` predicate inventory (item 7) | delegated to that packet; adopt, do not duplicate | medium |
| `WP-ENGINE-INDEPENDENCE.md` AS0 scope (item 10) | separate approved subpacket with its own record | medium |
| pinned `stark-samples` run (item 11) | needs the suite pinned by commit hash first | small |
| native build time / LSP latency (item 9) | need a harness | small |

Nothing outstanding blocks **AS1a**: its two dependencies — the duplicate-identity and
wrong-provenance reproductions — are discharged in §2. The characterization matrix blocks **AS2**,
not AS1a.
