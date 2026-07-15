# Gate 1 Exit Report — Core Front End

Date: 2026-07-15. Scope: ROADMAP.md Gate 1 ("Core front end"), executed as
PLAN.md WP1.1–WP1.5.

## Verdict

Gate 1 is **complete**. All three roadmap exit criteria are met:

1. *Every classified parsing fixture has a deterministic expected result* —
   all 121 spec fixtures are triaged in
   `STARKLANG/tests/spec-fixtures/manifest.toml` and enforced by
   `starkc/tests/conformance.rs`, which is required-green in CI.
2. *Valid Core examples parse without implementation-specific grammar
   changes* — conformance is 91/91 checked verdicts (67 parse-pass,
   18 semantic-error, 2 parse-fail, 4 lex-pass; 30 notation skipped), and
   every parser behavior beyond the original grammar text was resolved by
   amending the normative spec (see the T10 ledger), not by private grammar
   extensions.
3. *Discovered specification ambiguities are resolved in the normative
   source, with generated combined documents kept in sync* — ten spec
   defects were found and fixed in `docs/spec/` during Gate 1;
   `STARK-Core-v1.{md,html,pdf}` regenerated each time.

## What exists

- `starkc` (stable Rust, zero dependencies, `unsafe` forbidden):
  `source` (spans, line/col), `diag` (normative renderer), `lexer`
  (full `01-Lexical-Grammar.md`), `ast` (arena + typed IDs, stable dump),
  `parser` (full `02-Syntax-Grammar.md`, 16-level precedence, two modes:
  `Program` and harness-only `Snippet`), CLI
  (`starkc parse [--snippet] [--dump]`, `starkc lex`).
- Tests (69): 58 unit (lexer 23, parser 27, source/diag 8), conformance 3,
  diagnostic-format goldens 3, robustness/pseudo-fuzz 4, AST snapshots 1
  (covering 15 representative fixtures), plus 3 CI jobs
  (fmt+clippy+test, fixture-extraction sync, manifest conformance).

## T10 ledger — spec defects found and fixed during Gate 1

Implementation-first found these; each fix landed in the normative source
with artifacts regenerated and fixtures re-triaged in the same commit.

WP1.3 (triage, commit `88036c7`):
1. Three unparseable `{ ... }` ellipsis bodies in real examples
   (03 associated types, 04 `?` operator, 05 move example).
2. Same-scope `let` redeclaration in 03's logical-operators example,
   contradicting 04's own redeclaration rule.
3. Error-code table missing codes its own examples demonstrate — added
   E0005–E0007, E0204, E0400/E0401 (new mutability/initialization
   category), E0500 (new trait category).
4. "Unreachable code" listed as error E0300 while 04's example calls it a
   warning — reclassified as warning W0005; E0300 intentionally unassigned.

WP1.4 (parser, commit `956cecf`):
5. `Statement` could not represent block-formed expression statements —
   04's own `if condition { z = 42; }` example was unparseable. Added
   `Statement ::= BlockExpression ';'?` with the greedy-statement rule and
   the trailing-expression exception.
6. `(Int32)` was a one-element tuple type — `Type` now mirrors
   `TupleLiteral`: `(T)` is grouping, `(T,)` the 1-tuple.
7. `String::from(s)` was formally unparseable (`String` lexes as a keyword,
   `PathSegment` admitted only identifiers) — `PathSegment` now admits
   `PrimitiveType` as a leading segment.

WP1.5 (this review):
8. Parsing Notes now cover the full `>`-splitting family (`>>`, `>>=`,
   `>=`) rather than only `>>`, and the `pair.0.1` FLOAT-token split —
   both were implemented; now they are specified.

(Items 1–8 count ten defects: item 1 covers three blocks. WP1.2, the lexer,
found zero — the lexical grammar survived first contact.)

Also fixed in WP1.5, implementation-side: the diagnostic renderer misaligned
the line-number row's `|` with the gutter rows by one column, which shifted
every caret one column right of its span. Caught by the new end-to-end
format goldens (`tests/diag_format.rs`).

## Documented implementation limits (per 02 "Conformance")

- Nesting depth is bounded at 200 across expression/type/pattern/block/item
  recursion; deeper input produces the diagnostic "this code is nested too
  deeply to parse" instead of a stack overflow.
- Doc comments (`/** */`) are skipped as ordinary comments; they become
  trivia when tooling needs them (post-Gate-1).
- Parse and lex diagnostics carry no `E`-codes: 04's code taxonomy starts
  at semantic analysis and assigns none to syntax errors (see open
  questions).

## Deviations from the spec

None known. Every intentional parser behavior is now normative text.

## Open questions handed to Gate 2

1. **Syntax-error codes.** 04 defines no code range for lexical/parse
   errors. Decide: reserve a range (e.g. E0600–E0699) or keep syntax errors
   uncoded by design.
2. **Semantic-error fixture policy.** The manifest records expected codes
   for the 18 `semantic-error` fixtures, but standalone extraction leaves
   some with undefined helper names (`condition`, `get_index`, ...) that
   will *also* produce E02xx name-resolution errors. The Gate 2 harness
   needs a containment rule ("the listed codes must appear; extraction
   artifacts noted in the manifest are tolerated") — the `note` fields
   already flag every such fixture, plus the future-feature ones
   (`Rc`/`RefCell`, gate2-revisit).
3. **Pattern grouping.** `(p)` is a one-element tuple *pattern* per the
   grammar, while `(e)`/`(T)` are grouping in expression/type position.
   Harmless to parsing; Gate 2's exhaustiveness work should decide whether
   patterns get the mirror rule.
4. **Negative literal patterns.** `Pattern ::= Literal` admits no leading
   `-`, so `match x { -1 => ... }` is unparseable. Deliberate for Core v1?
   If not, it is a one-line grammar fix plus checker support.
5. **`self` outside methods.** The parser accepts `self` as a leading path
   segment anywhere (a free `fn` using `self` parses); rejecting it is name
   resolution's job (E0200 family).
6. **Statement-greedy consequences.** `if c { } - 1;` is an `if` statement
   followed by a syntax error (spec-mandated greedy rule). Gate 2 should
   confirm the diagnostic quality is acceptable or add a targeted hint.
