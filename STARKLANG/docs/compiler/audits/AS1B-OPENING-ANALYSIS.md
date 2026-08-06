# AS1b — opening analysis

**Packet:** AS1b, executing `WP-SPAN-SOURCEID.md` through AS2's single pipeline.
**Sprint:** 2 of 4. **Branch:** `wp-arch-stability/sprint-2`, built on Sprint 1.
**Date:** 2026-08-06.
**Status:** analysis complete; **implementation not started, and a decision is requested first.**

`WP-SPAN-SOURCEID.md` §6 names the risk precisely: "the change is easy to make *compile* while
threading a plausible-but-wrong `SourceId` at some sites, reproducing DEV-122 with better types." So
the first work is establishing what correct looks like and what the change actually costs — not
editing 61 construction sites.

Two findings. The first is a design constraint the packet does not resolve. The second changes how
the packet should be justified.

---

## F1 — `SourceId` is allocated after parsing, and spans are created during lexing

`Span { source: SourceId, lo, hi }` cannot be built where spans are built today.

| Fact | Evidence |
| --- | --- |
| `SourceId` is allocated in exactly one place | `analysis.rs:578`, `SourceId(map.files.len() as u32)` inside `build_source_map` |
| `build_source_map` runs **after** parse → resolve → typecheck | `analysis.rs:515` |
| The lexer, parser and AST have **zero** references to `SourceId` | `grep -c SourceId src/lexer.rs src/parser.rs src/ast.rs` → 0, 0, 0 |

So the identity a span needs does not exist at the moment the span is created. This is the
load-bearing decision in AS1b, and `WP-SPAN-SOURCEID.md` §3 passes over it — it says `SourceId`
"already exists … which is a substantial part of the groundwork", which is true of the *type* and
not of its *allocation timing*.

### Proposed correction

Move allocation to file-load time and demote `SourceMap` from allocator to view:

```text
today      parse ──► resolve ──► typecheck ──► build_source_map ALLOCATES SourceIds
proposed   SourceRegistry ALLOCATES on load ──► parse ──► … ──► SourceMap is a VIEW over it
```

- A `SourceRegistry` owns `Arc<SourceFile>`s and hands out a `SourceId` when a file is first loaded.
- The parser takes `&mut SourceRegistry`; `load_submodules_recursive` registers each child as it
  discovers it, which is already where child `SourceFile`s are constructed.
- `SourceMap` keeps provenance and lookup but stops assigning ids.
- `Ast::item_files` (`ItemId -> Arc<SourceFile>`) becomes `ItemId -> SourceId`, or disappears —
  once a span carries its source, per-item file metadata has no separate job. `WP-SPAN-SOURCEID.md`
  §4 already flags the equivalent question for `Diagnostic::file`.

This composes with AS2 rather than fighting it: `CompilerSession` is the natural owner of the
registry, and it did not exist when this packet was filed.

**Cost.** This is a wider change than "add a field to `Span`". The registry has to be threaded
through parsing, and two long-lived maps change shape. That is worth knowing before the packet is
scheduled, not after.

## F2 — the defect class is real and has bitten twice, but no live instance is currently reachable

`DEV-122` is OPEN, guarded, with two recorded instances: CD-302 (the test runner sliced a
dependency's span against the root file and panicked) and CD-306 (a fault inside `stark-mime`
reported at `stark-mime-consumer/src/main.stark:31:1` — line 31 of a 21-line file, in the wrong
package). Its user-impact note is specific: on DEV-121 the wrong file sent an investigation to the
wrong shape entirely, and a reproducer built from that description passed while the real fault was
three call frames away.

**I could not construct a live instance today.** Probes run against release binaries from this
branch:

| Probe | Result |
| --- | --- |
| Trap in a dependency, at a line number **past the end** of the root file | `lib/src/lib.stark:6:5`, correct file, correct line, correct source text |
| Trap inside a **cross-package generic**, instantiated from the app | `lib/src/lib.stark:10:5`, correct — the historic weak spot for provenance |
| Same program through the **native** path | `lib/src/lib.stark:10:5`, correct |

Compile-time dependency diagnostics were separately confirmed correct by Sprint 1's characterization
matrix (finding D5).

**What that does and does not mean.** It does *not* mean the class is closed. CD-309's guard
suppresses only spans whose offsets fall out of range for the file they are measured against; a
wrong-source span whose offsets are *coincidentally* in range still renders a confident wrong
location, and by construction I cannot produce one from ordinary source input — an in-range
wrong-source span requires an internal inconsistency, not a program. Absence of a reproduction here
is weak evidence, and the guard is part of why.

**But it does change the justification.** AS1b is a **hardening packet that eliminates a defect
class**, not a repair of something users are hitting now. Its value is that DEV-122 recurred once
already after being fixed, and the current mechanism detects rather than prevents. That is a
legitimate reason to do it. It is not the same reason as "this is broken today", and the packet
should not be scheduled as though it were.

---

## Decision requested

AS1b as specified requires the registry redesign in F1. Three options:

1. **Proceed as specified.** Registry first, then `Span` gains its `SourceId`, then the guard and
   `SpanResolutionError` are deleted per acceptance criteria 2 and 4. Widest change; ends the class.
2. **Split it.** Land the `SourceRegistry` and load-time allocation as its own reviewable change —
   which is useful on its own, since it removes the last place where identity is assigned late —
   then decide on the `Span` field separately with the registry already in hand.
3. **Re-order Sprint 2.** Take AS5 first. Its work is now well-specified by the manifest audit, its
   C8 dependency is settled, and F2 says AS1b is hardening rather than urgent.

**Recommendation: option 2.** It converts one wide change with a warned-about failure mode into two,
the first of which is mechanically checkable (ids allocated once, at load, nowhere else) and the
second of which is the behavioural one. It also puts the registry under `CompilerSession` where AS2
just consolidated the pipeline, so the second half gets a single place to thread rather than six.

Acceptance criterion 3 — "a diagnostic whose span belongs to a dependency renders against that
dependency, in both compile-time and runtime paths" — should be committed as a test **before** any
of it, in whichever option is chosen. The probes in F2 are that test's content; they currently pass,
which makes them a regression guard for the refactor rather than a reproduction.
