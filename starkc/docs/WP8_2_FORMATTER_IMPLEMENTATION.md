# WP8.2 — Formatter Implementation

**Status:** Complete
**Module:** `starkc/src/formatter/`

## Overview

`stark fmt` / `textDocument/formatting` (LSP) format Core v1 source by
walking the parsed AST — never the raw source text — and re-attaching
comment trivia by source position. Comments are formatted (spacing
normalized), not stripped: the lexer now collects them separately from the
token stream rather than discarding them (`lexer::tokenize_with_comments`,
`Comment`/`CommentKind`), since the parser's AST has no comment storage of
its own.

## Why parenthesization needed real work

The AST does not represent grouping parentheses — `02-Syntax-Grammar.md`:
`(expr)` parses to the inner expression, with precedence baked into tree
shape by the parser's 16-level table. Printing the tree back to text has to
re-derive, from tree shape alone, exactly where the original source needed
parens, or the printed output can silently reparse to a *different* tree
than the one printed. `formatter/precedence.rs` mirrors the parser's levels
and associativity 1:1 for this. It also mirrors the parser's struct-literal
restriction (`if`/`while`/`for`/`match` condition position): if a condition's
"head" subexpression is a bare struct literal, it gets wrapped in one outer
paren (simplest always-correct fix, not necessarily minimal).

## Architecture

- `formatter/precedence.rs` — expression precedence/associativity table,
  `needs_parens`, `head_is_struct_lit`.
- `formatter/comments.rs` — `CommentStream`, a position-ordered cursor over
  collected trivia.
- `formatter/printer.rs` — the AST walk. 4-space indent; delimited lists
  (params, args, struct/enum fields, generics, ...) render flat if they fit
  in 100 columns, else one element per line with a trailing comma; `{ }`
  lists (struct fields/literals) get inner padding spaces, `( [` don't;
  `use` groups are flattened to one leaf path per statement, sorted.
- `formatter/mod.rs` — `format_file(file, options) -> Result<String,
  Vec<Diagnostic>>`. Refuses to run on a file that doesn't parse cleanly
  (an AST-based formatter has no text to fall back on for the parts it
  couldn't build a tree for). Each file is formatted independently of its
  package (its own top-level items only; `mod name;` is not recursively
  expanded), so formatting is well-defined per file regardless of package
  resolution elsewhere.

## Comment attachment

Full leading/trailing attachment (blank-line-preserving) at: top-level
items, block statements, match arms. A comment on the same source line as
the end of a node renders trailing that node; otherwise it renders on its
own line above the next node, with a blank line preserved iff the source
had one. Comments in positions not specifically tracked (struct/enum/
trait/impl body lists, rare mid-expression positions) are still guaranteed
to be emitted via an end-of-file drain — never silently dropped, though
they may be relocated to the nearest statement/item boundary. This is a
disclosed v1 scope cut, not a correctness gap: no comment is ever deleted.

## CLI and LSP integration

- `stark fmt [--check] [<file.stark>]` (`starkc/src/bin/stark.rs`): no path
  formats every `.stark` file under the package root (directory walk from
  the manifest's parent, skipping `target`/`node_modules`/`.git`); a path
  formats just that file. `--check` reports non-canonical files without
  writing, exit 1 if any differ.
- `textDocument/formatting` (`starkc/src/lsp/server.rs`): formats the live
  (possibly unsaved) buffer and returns a single full-document `TextEdit`,
  or `null` if the buffer doesn't currently parse.

## Testing

- `starkc/src/lexer.rs`: 4 new unit tests for comment trivia collection
  (kinds, nesting, doc-comment detection, token-stream non-interference).
- `starkc/src/formatter/{mod,printer,precedence,comments}.rs`: unit tests
  covering idempotency under precedence/parens, comment preservation, the
  struct-literal condition guard, `use` flattening, and rejection of
  unparseable input.
- `starkc/tests/formatter.rs`: golden-file cases (struct/impl, enum/match,
  generics, long-call line-breaking) plus a corpus sweep over every
  `.stark` fixture in the repo. 112 real files parse cleanly as standalone
  programs and are confirmed idempotent (`format(format(x)) == format(x)`)
  and structure-preserving (same item count before/after, with `use`
  flattening correctly counted by leaf path rather than raw item count).

All 195 lib tests + formatter integration tests + full existing suite pass;
zero clippy warnings in new code; `cargo fmt --check` clean for all touched
files.

## Known v1 limitations (disclosed, not silent)

- Comments strictly inside an expression's interior (not at a statement/
  item/arm boundary) may be relocated to the nearest such boundary rather
  than staying at their exact sub-expression position — never dropped.
- Multi-line block comments (`/* ... */`) are preserved verbatim;
  continuation-line reindentation is not attempted.
- `struct`/`enum`/`trait`/`impl` body lists don't get full leading-comment
  attachment (only the end-of-file safety net); top-level items, block
  statements, and match arms do.
- Generic-parameter lists (`<T: Bound>`) are never soft-wrapped (assumed
  short in practice); everything else respects the 100-column rule.

## Next steps

- WP8.3 (Test Framework) can build on `format_file` for any golden-output
  comparisons it needs.
- WP8.4 (VS Code Extension) wires `stark.format` to `stark fmt` and/or the
  LSP `textDocument/formatting` request already implemented here.
