# Sprint 2 — Closeout Report

**Sprint:** 2 of 4 — AS1b, AS5
**Programme:** `WP-ARCHITECTURE-STABILIZATION.md`
**Branch:** `wp-arch-stability/sprint-2`
**Date:** 2026-08-07
**Status:** **CANDIDATE-PASS — awaiting CI.** The implementation is complete and AS1b and AS5 are
both closed; the Tier-3 gate is **not** discharged until the main CI workflow completes green on
`1616738`/`659fa02`. At the time of writing the Native Capabilities lane is green on both and the
main lane is still running.

The truthful state:

```text
Sprint 2 implementation       COMPLETE
AS1b                          COMPLETE
AS5                           COMPLETE
Tier-3 closeout               CANDIDATE-PASS / awaiting CI
Sprint 3 implementation       RESERVED until that CI turns green
```

This header said "PASS. Sprint 2 closes." while §7 and §8 said closure was conditional — the
document contradicted itself in the direction of claiming more than the evidence supported. It is
corrected here rather than left for a reviewer to catch, and it will be changed to an unconditional
PASS by an evidence-only commit recording the green run.

Every criterion below is classified **PASS**, **FAIL**, **PARTIAL**, **DEFERRED-BY-DECISION** or
**NOT-APPLICABLE**, with the command or artefact that supports it. A criterion with no evidence is
marked as such rather than assumed from a green suite.

---

## 1. What landed

Fifteen commits on top of Sprint 1, in two packets:

| Commit | Packet |
| --- | --- |
| `eec23ee` | AS1b opening analysis — the packet's groundwork claim does not survive contact |
| `470d5ff` | AS1b-i — `SourceId` allocated at load, by one thing |
| `93d107e` | AS1b re-sized; ii split into checkpoints |
| `87aa7a8` | AS1b-ii-b — `item_files` was a rival source authority |
| `8dfc449` | AS1b-ii-a/c — a span names its source |
| `1b4cbb0` | `Span::to` could manufacture a wrong-source span in release |
| `f80651e` | AS1b-ii-d(1) — a name is not a location |
| `a6f1aa5` | AS1b-ii-d(2) — a diagnostic's source is its span's |
| `c38dff0` | AS1b-ii-e — the MIR and native boundaries |
| `a6107fb` | AS1b-iii — MIR source identity collapse; MIR 0.3 → 0.4 |
| `577add3` | AS1b closeout corrections; AS5 opening analysis |
| `81c46cc` | AS5-a — DEV-184 |
| `d97fb75` | AS5-b/c/d/e/f — one JSON authority; DEV-185 |
| `96b5cbb` | AS5-g — compatibility identity; independent JSON oracle |
| `1616738` | Windows test defect; CE9 record; DEV-186 |

170 files, +8386 / −3659, excluding the 319-file vendored JSON corpus.

---

## 2. Packet exit criteria

### AS1b — span source identity

| # | Criterion | Result |
| ---: | --- | --- |
| 1 | Dependency diagnostics and traps resolve against the dependency's file and line table | **PASS** — `as1b_span_provenance` (compile-time, HIR runtime), `as1b_mir_native_provenance` (MIR engine, built native binary) |
| 2 | No AST/HIR/MIR/query diagnostic path accepts a bare byte range | **PASS** — `Span` has no two-argument constructor; `only_the_registry_mints_source_ids` |
| 3 | Span-to-location resolution total in compile-time and runtime paths | **PASS** — `resolve_span` returns `ResolvedLocation`, not `Result`; `SpanResolutionError` deleted |
| 4 | Diagnostic JSON remains deterministic | **PASS** — `diagnostic_transport`; ids now come from spans rather than a name round-trip that could panic |
| 5 | Ambient-file guessing removed only on migration evidence; item-to-file metadata kept on its own purpose | **PASS** — `item_files` → `item_sources`, retained for module semantics (ii-b), not span reads |

### AS5 — protocol, manifest and version-surface contracts

| # | Criterion | Result |
| ---: | --- | --- |
| 1 | Production code contains one JSON parser and one escaping authority | **PASS** — two parsers and four escapers before; `crate::json` after |
| 2 | A standard JSON test corpus and project-specific malformed cases pass | **PASS** — JSONTestSuite (MIT, vendored): 95/95 must-accept, 188/188 must-reject, 35 implementation-defined verdicts pinned; plus the project corpus's 12-construct table |
| 3 | C8's LSP protocol baseline stays green under the shared authority | **PASS** — `conformance_report`, `diagnostic_transport`, `--lib` LSP tests |
| 4 | A runtime/MIR surface change cannot compile or pass tests without updating its compatibility identity | **PASS** — `as5_compatibility_identity`, mutation-tested in both directions |
| 5 | Security-sensitive parsing decisions receive CE9 review | **PASS** — three decisions recorded in `AS5-OPENING-ANALYSIS.md` §6; all owner-approved |

---

## 3. What the sprint found that it was not looking for

Four defects, none of which was the packet's stated subject. This is the pattern Sprint 1 also
showed, and it is the argument for the programme.

| Defect | Found by | Severity |
| --- | --- | --- |
| **DEV-183** — TRAIT-COHERENCE-001's cross-package clause had **never been enforced** | AS1b-ii-d removing the ambient file the orphan rule's disk probe read | A normative rule silently inert; one first-party violation |
| **DEV-184** — three of four JSON escapers emitted invalid JSON | AS5's opening inventory of the *emit* side | `stark doctor --json` produced documents no parser accepts; raw controls on the LSP wire |
| **DEV-185** — every JSON number decoded to `f64` | AS5's review of the shared data model | A JSON-RPC request id could **change value** between arriving and being answered |
| **DEV-186** — LSP `Content-Length` allocated unbounded | CE9 review of the nesting limit | Availability, on a socket-facing surface. **OPEN**, registered not fixed |

DEV-183 is the one worth dwelling on. The orphan rule decided "same package" by walking file paths
**on disk during type checking**. After AS1a gave sources logical names, that probe returned `None`
for every file, so every type looked local and the rule could not fire. Before AS1a it fired only by
an asymmetry — the root file carried an absolute path while every other file carried a logical one.
It was never a comparison. A source-identity packet found a coherence defect because both were the
same underlying problem: something answering "where did this come from?" by guessing.

---

## 4. Defects introduced by this sprint, and caught

Three, all mine, all caught before or by CI rather than by a user.

| Defect | Caught by | Note |
| --- | --- | --- |
| `Span::to` was a `debug_assert_eq!`, so a **release** compiler would silently join spans across files | Owner review of `8dfc449` | DEV-122's failure class relocated to span composition, inside the packet meant to eliminate it |
| The orphan-rule regression from `source_package` | CI (`first-party package qualification`, 3 platforms) | Turned out to be DEV-183, a real defect the change exposed |
| A TAB in a directory name — legal POSIX, `InvalidFilename` on Windows | CI (`fmt, clippy, test (windows-x64)`) | A macOS-only local run cannot see it; the repaired test is portable *and* stronger |

The first is the one that matters for process: a `debug_assert!` in an invariant that only holds in
debug builds is not an invariant, and no test would have found it because tests run in debug.

---

## 5. Corrections to claims made during the sprint

Recorded because the commits are history and cannot be amended.

| Claim | Correction |
| --- | --- |
| "no `Arc<SourceFile>` anywhere in production `interp`" | Too strong. The accurate claim: *no production interpreter source-text lookup uses an ambient `SourceFile`* |
| "holding a `RegisteredSource` is proof this compilation registered it" | Proof it was **registered rather than fabricated**; not proof of *which* registry |
| "`freeze` is the only way to build a `SourceTable`" | `From` and `Default` also did. `From` deleted; `Default` documented as the empty-`Hir` placeholder |
| AS5 §5's recommendation to pin `MIR_VERSION` as a set of variant names | Wrong — it would not have moved for AS1b-iii. Superseded in place |

---

## 6. Deliberately not done

| Item | Why |
| --- | --- |
| LSP string request ids | Protocol surface, not JSON authority. Recorded under DEV-185 |
| LSP `Content-Length` bound | Transport framing, not parser resource safety. **DEV-186**, open |
| Configurable JSON depth limit | No consumer needs deeper JSON; a `parse_with_limits()` API would be policy surface without a demand |
| Arbitrary-precision JSON numbers | The shared layer preserves what the document said; each consumer states the type it requires |
| AS0's three remaining items (AS3/AS4/AS8 scope) | Sprint 3 and 4 work; AS0 remains partial by design |

---

## 7. CI record

| Commit | CI | Native |
| --- | --- | --- |
| `a6107fb` (AS1b-iii) | **success** | success |
| `577add3` (AS1b closeout) | **success** | success |
| `81c46cc` (AS5-a) | failure — Windows test defect | success |
| `d97fb75` (AS5-b..f) | failure — same defect, inherited | success |
| `96b5cbb` (AS5-g) | *(superseded by `1616738`)* | — |
| `1616738` (fix + records) | *pending at time of writing* | — |

The two failures are one defect, described in §4. Sprint 2 does not close until `1616738` is green
on all three platforms; that is the only outstanding item.

---

## 8. Verdict

**CANDIDATE-PASS.** Every packet criterion in §2 is met and every defect in §4 is closed or
registered. The Tier-3 gate discharges when the main CI workflow is green on `1616738`/`659fa02`,
and not before — see §7.

AS1b eliminated a defect class rather than guarding it. Nine separate mechanisms for answering
"which file is this?" — the parser's, the checker's, the item table's, the callable's, the
diagnostic's, the runtime error's, MIR's `FileId`, the backend's, and the interpreter's per-frame
swap — collapsed into one: `Span.source`, resolved through an immutable `SourceTable` that HIR and
MIR carry and cannot add to.

AS5 did the same for JSON: two grammars and four escapers became one authority, validated against a
corpus the parser had never seen.

The programme's claim is that semantic-authority fragmentation is the recurrent risk. Sprint 2 found
four defects while consolidating two authorities, and three of the four were invisible precisely
because a duplicate had been quietly answering for the original.
