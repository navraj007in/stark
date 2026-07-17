# WP8.5 — Documentation Generator Implementation

**Status:** Complete
**Module:** `starkc/src/doc_gen/`

## Overview

`stark doc [--open] [--output <dir>]` extracts `///` doc comments from
every public item in a package, renders them as Markdown with real
STARK-syntax-highlighted code blocks, and emits a static HTML site plus a
client-side search index — no external dependencies (no Markdown crate, no
syntax-highlighting library, no template engine), matching this project's
existing convention.

## Doc comments have no home in the AST/HIR — same root problem as WP8.2

Confirmed again here (first found in WP8.2): the lexer discards comments
entirely; there's no `doc: Option<String>` field on any AST/HIR item node.
Doc extraction reuses the exact same trick the formatter uses —
`lexer::tokenize_with_comments`'s separately-collected trivia,
re-associated with AST nodes by source position — rather than touching the
grammar. `extract.rs`'s `leading_doc_comment` walks backward from an
item's start position through the (already position-sorted) comment list,
accepting a `///` line only if it's on the line immediately above the
previous one accepted (or the item itself), so a blank line or an
unrelated comment correctly breaks the association.

## Three real bugs found and fixed during implementation

1. **Signatures included the whole item body.** Slicing `node.span`
   directly for a `struct`/`enum`/`trait`/`model` signature includes the
   full `{ ... }` body (fields, variants, etc.), not just the header.
   Fixed with `header_span`: a token-stream scan (not a raw-byte scan, so
   braces inside string literals or comments can't be mistaken for a body
   opener) that finds the first depth-0 `{` or `;` after the item's start.
   `fn` signatures use `FnSig::span` directly instead (already excludes
   the body).

2. **Duplicated `pub` in struct/enum/trait/model signatures.** Once (1)
   was fixed with `header_span(tokens, node.span)`, `node.span.lo` already
   *starts* at `pub` (unlike `FnSig::span`, which starts at `fn` and
   excludes the visibility keyword) — so manually prepending `"pub "` for
   every kind produced `"pub pub struct Point"`. Fixed: only the `Fn` case
   (and merged `impl` methods, which also use `FnSig::span`) prepend `pub`
   manually; everything else already has it from its own span.

3. **Doc examples were only compile-checked, not run — so `assert_eq`
   failures were silently accepted.** The plan's own reference example
   (`assert_eq(add(2, 3), 5);`) is a *runtime* assertion. Parsing +
   resolving + typechecking `assert_eq(add(2, 2), 999)` succeeds fine
   (`assert_eq<T>(a: T, b: T)` is generic — both sides are `Int32`,
   nothing type-checks against the literal values) — the mismatch only
   surfaces when the code actually *runs* and traps. An initial version
   of `validate_example` stopped after typecheck and reported this as a
   passing example; caught by testing against the plan's own broken-value
   variant of its own canonical example. Fixed by reusing
   `interp::run_item` (built in WP8.3 for the test runner) to actually
   execute the synthetic `fn __doc_example__() { <example> }` after
   typecheck succeeds, treating a `RuntimeError` (trap/panic) as a
   validation failure too — matching `cargo test --doc` semantics, not
   just `cargo doc`'s.

## Examples validate with their file's other definitions in scope

A doc example commonly calls the very item it documents without
redefining it (again, the plan's own `add` example). Examples are
therefore validated per source file: `validate_examples(examples,
file_source)` appends each example as a synthetic function to that file's
*own* source text before parse/resolve/typecheck/run, so `add` (or
anything else the file defines) is already in scope. This is why example
validation happens per-file in `stark doc`'s CLI loop (`bin/stark.rs`)
before items get merged across files for site generation — a merged
multi-file item list has no single "this file's source" to validate
against.

## Architecture

- `extract.rs` — `DocItem`/`ItemDocKind`, the position-based `///`
  association, and struct-field/enum-variant/trait-item/`impl`-method
  extraction (impl blocks' public methods merge into their matching
  struct/enum's `members` by type name, not rendered as separate pages).
- `highlight.rs` — tokenizes with `lexer::tokenize_with_comments` (the
  real lexer, not a regex approximation) and wraps each token/comment in a
  `<span class="tok-*">`, reconstructing whitespace gaps as escaped plain
  text.
- `markdown.rs` — a deliberately small Markdown subset (headings,
  paragraphs, `` `inline code` ``, fenced ` ```lang ` blocks routed
  through `highlight.rs` when `lang == "stark"`, `*`/`-` bullet lists) —
  everything the plan's own example doc comment uses, nothing more, so
  content outside that subset renders as plain escaped text rather than
  being misinterpreted by a partial parser.
- `html.rs` — page templates as plain string-building (no template
  engine): one `<module>/<item>/index.html` per page-level item (struct/
  enum/trait/fn/const/type/model — matching the plan's `std/option/
  index.html` layout exactly), a package `index.html` grouped by module,
  and a `search.html` with a small vanilla-JS substring/fuzzy matcher.
  Light/dark theme via `prefers-color-scheme`.
- `search.rs` — flat `search.json` array (name, kind, module, signature,
  url) for every page-level item and member, hand-rolled JSON encoding
  (matching this project's convention elsewhere — no serde).
- `mod.rs` — `generate_from_items` (site + search only) and
  `validate_examples` (compile-and-run doc examples with file context) as
  two separate entry points, not one combined call, since validation needs
  per-file source text a merged multi-file item list doesn't have.

## CLI (`stark doc`, in `bin/stark.rs`)

Walks the package root for every `.stark` file (same `collect_stark_files`
helper `stark fmt` uses), parses each independently, extracts + validates
that file's examples with its own source as context, and merges all
files' items before writing one combined site. A file that doesn't parse
aborts the whole run with a clear error (consistent with `stark fmt`'s
CORE — an item's doc pages must reflect real, buildable source). Exits
non-zero if any doc example fails (compile or runtime), printing which
item's example failed and why.

**Known scope limit, disclosed:** each `.stark` file is parsed
independently (mirroring `stark fmt`'s existing single-file-per-parse
design — see that WP's doc for why). Items declared directly or in an
inline `pub mod name { ... }` block get correct module-path nesting;
items in a file only reachable via an external `mod name;` declaration are
still documented (every `.stark` file under the package root is walked
regardless), but without inheriting the referencing module's path prefix,
since that requires resolving the module graph across files, which
single-file parsing doesn't do.

## Testing

`starkc/tests/doc_gen.rs` — 21 tests covering: extraction (doc comment
association including the blank-line/non-doc-comment edge cases,
public/private filtering for items/fields/impl-methods/mod contents,
struct/enum/impl signature correctness, nested-module path tracking),
Markdown rendering (headings/lists/inline code, `stark`-highlighted vs.
plain fenced blocks), syntax highlighting (HTML escaping, keyword/type/
string token classification), example validation (the passing case, the
runtime-assertion-failure case, the undefined-reference case), and full
site generation to a real temp directory (file existence, `index.html`
content, `search.json` content, and relative-link depth correctness for a
nested module's page).

Manually verified end-to-end against a real on-disk package: `stark doc`,
`--output <dir>`, `--open`-path construction, nested `pub mod`, enum
variants with tuple types, `impl` method merging, and both the pass and
fail paths for doc-example validation (including the specific "compiles
but wrong at runtime" case that motivated fix #3 above).

383 tests pass across the whole workspace; zero new clippy warnings;
`cargo fmt --check` clean.

## Not built (plan lists these; scoped out)

- **Fuzzy matching** in `search.html`'s JS is substring-based (matches
  the plan's own "full-text search over API names" and "namespace
  filtering" — module is shown per result — literally, but not literal
  Levenshtein-style fuzzy matching). Deferred; substring search covers the
  common case and 1000+ items are trivially fast to filter client-side at
  this scale either way.
- **Cross-reference hyperlinks** *within* doc-comment prose (e.g. auto-
  linking a mention of `Point` inside another item's doc text to
  `Point`'s page) are not implemented — the Markdown renderer doesn't
  attempt name resolution. Signatures still link naturally via the item
  index/search; prose cross-references would need a second pass matching
  extracted item names against doc text, deferred as a real but
  non-blocking enhancement.
