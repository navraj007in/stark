# AS1b — opening analysis

**Packet:** AS1b, executing `WP-SPAN-SOURCEID.md` through AS2's single pipeline.
**Sprint:** 2 of 4. **Branch:** `wp-arch-stability/sprint-2`, built on Sprint 1.
**Date:** 2026-08-06.
**Status:** AS1b-i **complete** (`470d5ff`). AS1b-ii **in progress** — the decision was taken
(option 2, split; `RegisteredSource` handle), and §4 below records the scope correction the work
produced.

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


---

## 4. Scope correction (2026-08-07)

**AS1b is larger than `WP-SPAN-SOURCEID.md` estimated, and larger than this analysis estimated.**
Recorded here as a correction, not as an argument to abandon the invariant.

### What the packet said, and what is true

| Claim | Reality |
| --- | --- |
| `SourceId` "already exists … a substantial part of the groundwork" | true of the type; its **allocation ran after the front end**. AS1b-i, a whole packet |
| "`Span` gains a `SourceId`; construction sites supply it" | 61 sites by my first count. **75** — the first grep missed multi-line `Span {` literals |
| Risk is "threading a plausible-but-wrong `SourceId`" | real, and the `RegisteredSource` handle now makes it unrepresentable rather than discouraged |
| — not mentioned at all — | **~20 sites are a `Span { lo: 0, hi: 0 }` "no location" sentinel.** Making spans carry a source forces a modelling decision the packet never raised |
| — not mentioned at all — | **`item_files` must migrate from `Arc<SourceFile>` to `SourceId`** (38 references), because the interpreter re-points at a declaring file per item (DEV-069, DEV-088) and needs registered identity to do it |
| — not mentioned at all — | **64 test files** call `interp::run` or `lower_program` with an `Arc<SourceFile>` |

### The `RegisteredSource` decision

`SourceId` is **not** stored on `SourceFile`: a file is bytes with a name and is reusable across
sessions, while an id is registry-local and means nothing outside the compilation that minted it.
Identity is carried by `RegisteredSource { id, file }` — private fields, no public constructor, only
`SourceRegistry::intern` builds one. Holding one is proof this compilation registered it. There is
no sentinel and no fabricated id anywhere in the change.

Components that scan rather than compile — format-literal scanning, documentation extraction —
derive identity from the source-bearing span or AST they are handed, not from a separately threaded
parameter. The standalone syntax highlighter owns a one-file registry, so its id is real and
registered rather than invented.

### AS1b-ii, split into local checkpoints

Each is independently reviewable and leaves a compiling tree.

| # | Checkpoint | State |
| ---: | --- | --- |
| ii-a | `Span` carries `SourceId`; `RegisteredSource`; lexer, parser, analysis, formatter, doc-gen, format-scanning, tensor, deploy, MIR lowering threaded | **in progress** — 21 lib errors remain, all in the interpreter and its callers |
| ii-b | `item_files: ItemId -> SourceId`; the interpreter takes the registry and drops per-frame `Arc` swapping | not started — **this is what ii-a is currently blocked on** |
| ii-c | Test-call migration (64 files) | not started |
| ii-d | `resolve_span` total; delete `SpanResolutionError`; remove DEV-122's interim guard (acceptance 2 and 4) | not started |
| ii-e | Dependency-file behavioural tests at each AST/HIR/MIR/runtime boundary | `as1b_span_provenance.rs` covers compile-time and HIR runtime; MIR and native boundaries outstanding |

### Two findings the work produced

**Synthetic spans block acceptance criterion 4.** Dependency `mod` items get spans at
`0x8000_0000+`, deliberately outside any real file, keyed in `ast.synthetic_spans`. Those are
precisely the out-of-range case CD-309's guard catches — so the guard cannot simply be deleted in
ii-d until synthetic spans are handled. The packet assumes deletion is free.

**The per-frame file swap is the duplication this change removes.** `interp.rs` records "the file
that DECLARES this body" per frame (DEV-069) and swaps it per constant (DEV-088), because spans
cannot say which file they index. Once they can, that machinery has nothing left to decide — which
is the clearest statement of why the invariant is worth finishing.
