# WP-SPAN-SOURCEID — a span that knows its own source

**Status:** FILED, not started.
**Filed by:** CD-309, which committed to filing it and did not.
**Owning track:** compiler (Gate C-series governance, `COMPILER-CHARTER.md` §1.6).
**Prerequisite deviations:** DEV-122 (closed by an interim guard, class open).

---

## 1. The problem, stated as what actually happened

`Span` is a byte range: `{ lo, hi }`. Nothing in it says *which source those offsets index*. The
answer is supplied at render time by whoever happens to be holding a `SourceFile`.

`SourceFile::line_col` **clamps** an out-of-range offset to end-of-file. It cannot fail. So a span
measured against the wrong source does not produce an error — it produces a well-formed, plausible,
**wrong** location. DEV-122's observed shape: a 21-line consumer was told that a fault lay at line
31 of itself, because a dependency's span had been resolved against the root file's line table.

A reader has no way to distinguish that from a real location. This is worse than no location: it
names a specific place to go and look, and the place is wrong.

## 2. What CD-309 actually shipped, and why it is not this

`diag::resolve_span` is now the single checked path from a span to a location, returning
`Result<ResolvedLocation, SpanResolutionError>` over three conditions — inverted (`hi < lo`), past
the end of this source, and a column outside its own line (what a stale or foreign line table looks
like from inside). On `Err` the location is suppressed and an unmistakably internal message naming
DEV-122 is rendered.

Two properties it holds deliberately: it never panics (a diagnostic path that can abort turns a
reportable defect into a lost one), and it never falls back to another source (substituting a
different file is precisely how a dependency's span came to be rendered against the root).

**That is a detector, not a fix.** It converts a convincing wrong answer into a visible absence of
an answer. The span still does not know its source; the renderer still guesses, and the guard only
catches guesses that happen to land out of range. A wrong-source span whose offsets are *coincidentally*
in range for the file it is measured against still renders a confident wrong location, and always
will, because nothing in the data distinguishes the two cases.

## 3. The correction

Every span carries a `SourceId`, and resolution is **total by construction** — there is no failure
mode to guard because there is no ambiguity to resolve.

```rust
struct Span { source: SourceId, lo: u32, hi: u32 }
```

`SourceId` already exists (`crate::analysis::SourceId`, with `SourceMap` and `SourceProvenance`),
which is a substantial part of the groundwork and the reason this is a work package rather than a
research question.

## 4. Scope

**In:**
- `Span` gains a `SourceId`; construction sites supply it.
- `resolve_span` takes a `SourceMap` and cannot fail. `SpanResolutionError` and its rendered text
  are deleted, not retained "just in case" — a total function with a residual error arm is an
  invitation to reintroduce the guess.
- Both rendering paths (compile-time diagnostics, runtime trap reporting) resolve through the map.
  The runtime path is where DEV-122 actually bit, and a change that fixed only the compile-time
  path would repeat CD-309's original near-miss.
- `Diagnostic::file` and `RelatedDiagnostic::file` become derivable rather than carried, or are
  justified in writing if they stay.

**Out:**
- Span interning, compression, or any size optimisation. `Span` grows by four bytes; if that turns
  out to matter, it is a separate measured change and not an argument against correctness.
- Multi-span diagnostics, span merging, macro provenance.

## 5. Acceptance

1. `Span` carries a `SourceId` and every construction site names it.
2. Span→location resolution has no error path.
3. A test proves the original defect is now unrepresentable rather than detected: a diagnostic
   whose span belongs to a dependency renders against **that dependency**, with its own line
   numbers, in both the compile-time and runtime paths.
4. DEV-122's interim guard is removed in the same change. Leaving it would leave two mechanisms for
   one property, which is the exact shape of DEV-128 (`is_copy` written twice, fixed twice, drifted
   anyway).
5. Three-engine agreement on trap-location reporting is unchanged.

## 6. Risk

The mechanical part is wide — `Span` is constructed in the lexer, parser, resolver, checker, and
lowering — but each site has a source in hand already, because it had to in order to produce the
offsets. The real risk is the opposite of the usual one: the change is easy to make *compile* while
threading a plausible-but-wrong `SourceId` at some sites, reproducing DEV-122 with better types. So
acceptance criterion 3 is a behavioural test on a real dependency, not a type-level argument.
