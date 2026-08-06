# Sprint 1 — Closeout Report

**Sprint:** 1 of 4 — Cranelift retirement, AS0, AS1a, AS2
**Programme:** `WP-ARCHITECTURE-STABILIZATION.md`
**Branch:** `wp-arch-stability/sprint-1`, on `develop@5b4db65`
**Date:** 2026-08-06
**Status:** **PASS.** Sprint 1 closes. AS1a and AS2 are complete; AS0 remains partial by
design and does not exit.

Every criterion below is classified **PASS**, **FAIL**, **PARTIAL**, **DEFERRED-BY-DECISION** or
**NOT-APPLICABLE**, with the command or artefact that supports it. A criterion with no evidence is
marked as such rather than assumed from a green suite.

---

## 1. What landed

Nine commits:

| Commit | Packet |
| --- | --- |
| `c5bfc91` | Cranelift retirement (opening item) |
| `19ae858` | Manifest strictness audit (opening item) |
| `3fa9550` | AS0 — reproductions and pipeline inventory |
| `34f604f` | AS1a — one physical source, one logical identity |
| `d8c3c3f` | AS0 — characterization matrix |
| `517ee69` | AS2 — one compiler session |
| `1928391` | Windows path-separator defect in the characterization matrix |
| `536e1ef` | Approval-boundary record |
| `7012080` | AS0 items 9 and 11 — baselines and pinned samples |

20 files, +1948 / −1637.

## 2. Opening items

### Cranelift retirement

| Audit condition | Result |
| --- | --- |
| no `src/` reference to cranelift or `target-lexicon` | **PASS** — 0 hits |
| `[dev-dependencies]` only, shipped surface unchanged | **PASS** — charter §1.10 |
| sole consumer is the disposable WP-C3.3 spike, removal pre-authorised | **PASS** — `WP-C3.3-direct-cranelift.md:162` |

Clean `cargo check --all-targets`: **21 s → 8 s**, **529 MB → 463 MB**. Four dangling documentation
pointers corrected, not the two anticipated.

### Manifest strictness audit

**PASS**, and it changed AS5's classification. Recorded in `AS0-MANIFEST-STRICTNESS-AUDIT.md`. The
corpus is strict-clean (108/108), so no repository migration; but the two in-tree parsers disagree
with each other on 9 of 12 constructs, and F1 — the LSP parser decoding valid surrogate pairs to the
empty string — was a live correctness defect, taken as DEV-182/CD-384 under §3 pre-emption and
merged to `develop`.

## 3. Packet exit criteria

### AS0 — baseline, reproduction and authority inventory: **PARTIAL** (8 of 11 work items)

| # | Criterion | Result |
| ---: | --- | --- |
| 1 | duplicate-identity and wrong-provenance halves each reproduced | **PASS** — `as0_source_identity.rs` |
| 2 | driver, callable, predicate and JSON inventories exact-set checked | **PARTIAL** — driver and JSON done; callable and predicate inventories outstanding |
| 3 | characterization baseline committed; divergences named | **PASS** — `as0-characterization/BASELINE.txt`, five divergences D1–D5 |
| 4 | all three manifest-parser deltas recorded, AS5 classified | **PASS** |
| 5 | performance commands and raw results recorded | **PASS** — §5 of the inventory |
| 6 | pinned samples-suite result recorded | **PASS** — 39/39, pin `b3b28e75` |
| 7 | each later packet has a bounded ownership set and rollback point | **PASS** |

**AS0 does not exit.** Three work items remain — callable execution-site inventory (feeds AS3), the
`WP-C7.8-RB0` predicate inventory (AS4), and the `WP-ENGINE-INDEPENDENCE` AS0 scope (AS8). None
gates Sprint 2; Campaign A's exit gate does require them.

### AS1a — canonical package source identity: **PASS** (5 of 5)

| # | Criterion | Result | Evidence |
| ---: | --- | --- | --- |
| 1 | one physical root produces one `SourceRecord` | **PASS** | `one_physical_file_produces_one_source_record` |
| 2 | logical entry is sole `Root`; modules carry package provenance | **PASS** | `the_logical_entry_is_the_sole_root_and_every_file_carries_its_package` |
| 3 | relocation preserves source maps and MIR file tables | **PASS** | `relocating_an_identical_package_preserves_its_source_map`, `the_mir_file_table_is_logical_and_relocation_stable` |
| 4 | no absolute checkout path in reproducible identity | **PASS** | asserted in both tests above |
| 5 | package, overlay and native-build paths share the helper | **PASS** | `an_overlay_changes_the_entry_content_but_never_its_identity`; `Package::entry_source_file` is the sole constructor |

### AS2 — one compiler session: **PASS with one PARTIAL** (4 of 5, one partial)

| # | Criterion | Result | Evidence |
| ---: | --- | --- | --- |
| 1 | no production entry point assembles the pipeline outside the driver | **PASS** | `as2_one_pipeline.rs`, exact set with a non-decorative allowlist |
| 2 | same invalid package → same ordered diagnostics across CLI, package CLI, test runner **and LSP** | **PARTIAL** | the characterization matrix covers compiler CLI, package CLI and test runner. **LSP is recorded NOT-APPLICABLE** — it is a stdio JSON-RPC session with no non-interactive surface in this harness. The LSP goes through the same `analyze_project` it always did, so no regression is implied, but the criterion's LSP clause is **not** evidenced here |
| 3 | Core/tensor options per-session, sequential and parallel | **PASS** | `c91_sequential_and_parallel_analyses_do_not_share_extension_state`, 5 passed |
| 4 | provider-backed packages share one analysis for check and build | **PASS** | `c78_capability_declaration` 9, `c783_env_e2e` 3; CI `first-party package qualification` ×3 platforms |
| 5 | existing suites remain green | **PASS** | §4 |

**Criterion 2's LSP clause is the one thing Sprint 1 does not demonstrate.** It is recorded as a
gap rather than waved through: closing it needs a stdio-driving harness, which is AS8's territory
and is noted there.

## 4. Tier-3 evidence

| Requirement | Result |
| --- | --- |
| `cargo fmt --check` | **PASS** — clean |
| `cargo clippy --all-targets -- -D warnings` | **PASS** — clean; it caught `CompileFailure` returning ~2 KB by value as the `Err` half of every `check()`, now boxed |
| full Rust suite through CI | **PASS** — run `31111837210` on `7012080`, **24 of 24 jobs green** |
| Core positive and negative fixture conformance | **PASS** — CI `spec fixture conformance` |
| HIR/MIR/native debug/native release differential rows | **PASS** — `three_engine_differential` 109 passed locally; CI `C6.5 corpus tier-1 agreement`, `C6.4 tier-1 agreement`, `C6.5 corpus replay` ×2, `C6.4 tier-1 qualification` ×2 |
| tensor/extension tests | **PASS** — `c91_extension_isolation` 5 passed |
| deterministic outputs executed twice | **PASS** — characterization baseline verified identical across two consecutive runs |
| package/provider qualification | **PASS** — CI `first-party package qualification` on linux-x64, macos-arm64, windows-x64 |
| pinned external samples suite | **PASS** — 39/39 |
| focused tests failing before the repair | **PASS** — AS0's three characterization tests asserted the defect and failed when AS1a fixed it; DEV-182's five surrogate tests failed before its repair |
| updated deviations, `COMPILER-STATE.md`, architecture docs | **PASS** — CD-384, CD-385, DEV-012 narrowed, approval boundary recorded |

Local scoped evidence: `--lib` 531 · `as0_source_identity` 5 · `as0_characterization` 1 ·
`as2_one_pipeline` 3 · `three_engine_differential` 109 · `c6_metamorphic` 4 · `c6_package` 6 ·
`native_c5_4_workspace` 6 · `native_build_cli` 9 · `conformance` 3 · `exec_snapshots` 4 ·
`c91_extension_isolation` 5 · `c78_capability_declaration` 9 · `c783_env_e2e` 3. All 0 failed.

## 5. What Sprint 1 found that it was not looking for

1. **The provenance half of the source-identity defect was worse than recorded.** The phantom
   absolute record was the *only* `Root`, and every package file carried `package: None` — the
   attribution branch had been dead code since DEV-113.
2. **Six bypassing assemblies, not four; three shipped binaries, not two.** `starkide` had its own
   private pipeline that two targeted searches missed. Only an exact set found it.
3. **A resolve error suppresses every type error** (D1), so "ordered diagnostics" currently pins a
   list of length one — which is what AS2's criterion 2 is actually comparing.
4. **LSP change latency is quadratic in package size**, and the isolation places the quadratic in the
   front end rather than the LSP. At 4000 functions a keystroke costs a second. This reframes AS8.
5. **DEV-182 passed WP-C8.7's protocol validation**, because protocol tests compare verdicts and
   this was a value defect. Recorded as a standing limit on C8's evidence in `GATE-C8-CLOSURE.md` §4.

## 6. Defects introduced by this sprint, and caught

Recorded because a closeout that lists only successes is not evidence.

| Defect | Found by | Fixed |
| --- | --- | --- |
| Characterization matrix baked the host path separator into its baseline | **CI, Windows lane only** — three local runs and two other platforms were green | `1928391`, baseline unchanged |
| `as2_one_pipeline`'s first `cfg(test)` stripper counted braces, defeated by string literals in `build.rs` | its own first run | replaced before commit |
| Its second stripper produced four false positives from column-zero string content | its own run | replaced with an exact TEST_ONLY set |
| `CompileFailure` returned ~2 KB by value | `clippy` | boxed |
| First LSP latency reading was not credible (0.5 ms) | scaling the fixture before recording | harness validated, then measured |

## 7. CI record

Run `31111837210` (`CI`) and `31111840650` (`C7.8 Native Capabilities`) on `7012080`.
**24 of 24 jobs green**, on all three Tier-1 platforms:

`fmt, clippy, test` ×3 · `first-party package qualification` ×3 · `C7 P1 REST workload` ×3 ·
`release package smoke` ×3 · `C6.4 tier-1 qualification` ×2 · `C6.5 corpus replay` ×2 ·
`C6.4 tier-1 agreement` · `C6.5 corpus tier-1 agreement` · `C6.5 mutation controls` ·
`C6.4 windows tier-2 gap probe` · `DEV-160 raw slot primitives under Miri` ·
`spec fixture conformance` · `External sample suite (pinned)` · `CI complete`.

**The Windows lane matters here specifically.** The previous attempt (`523b2d8`) was red on
`fmt, clippy, test (windows-x64)` alone, from a defect in this sprint's own characterization test.
It is green now, on the fix that left the baseline file unmodified.

## 8. Verdict

> **Sprint 1 PASSES and closes.**

Delivered: the Cranelift retirement, the manifest strictness audit, **AS1a** (5/5) and **AS2**
(4 PASS, 1 PARTIAL). **AS0 is deliberately partial** — 8 of 11 work items — and does **not** exit;
its three remaining items feed AS3, AS4 and AS8, and Campaign A's exit gate requires them.

Two things the sprint does **not** claim:

1. **AS2's criterion 2 is not fully evidenced.** The LSP clause has no harness; the matrix records
   it NOT-APPLICABLE. No regression is implied — the LSP uses the same `analyze_project` it always
   did — but the criterion is not demonstrated, and closing it belongs to AS8.
2. **AS0 has not exited**, so Campaign A remains open regardless of this closeout.

Sprint 2 (AS1b, then AS5) is unblocked: its gate was this closeout, and AS5's separate C8 dependency
was settled by CD-385.
