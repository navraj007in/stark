# starkc

Compiler for the STARK Core v1 language. Rust, stable toolchain, currently at
**WP1.4 done (parser)** — WP1.5 (diagnostics polish + Gate 1 exit review)
remains.

- Language definition: [`../STARKLANG/docs/spec/`](../STARKLANG/docs/spec/)
  (normative), single file: `STARK-Core-v1.md`
- Delivery gates: [`../STARKLANG/docs/ROADMAP.md`](../STARKLANG/docs/ROADMAP.md)
- Engineering plan: [`../STARKLANG/docs/PLAN.md`](../STARKLANG/docs/PLAN.md)
  (standing decisions T1–T12, work packages)
- Conformance corpus: [`../STARKLANG/tests/spec-fixtures/`](../STARKLANG/tests/spec-fixtures/)

## Use

```bash
cargo run -- parse file.stark             # parse a program, report diagnostics
cargo run -- parse --snippet --dump f.stark  # block-body mode, print the AST
cargo run -- lex file.stark               # dump the token stream
cargo test                                # everything, incl. the 121-fixture
                                          # conformance suite and pseudo-fuzz
UPDATE_SNAPSHOTS=1 cargo test --test snapshots   # regenerate AST snapshots
```

## Layout

| Module | Contents | Status |
| --- | --- | --- |
| `source` | `SourceFile`, `Span`, line/column mapping | done (WP1.1) |
| `diag` | diagnostics in the spec's normative render format | done (WP1.1) |
| `lexer` | `01-Lexical-Grammar.md` | done (WP1.2) |
| `tests/conformance` | manifest-driven fixture harness | done (WP1.3) |
| `ast` | arena + ID node tree (PLAN.md T6) + dump | done (WP1.4) |
| `parser` | recursive descent, 16-level precedence (`02-Syntax-Grammar.md`) | done (WP1.4) |
| `tests/snapshots` | AST snapshots for 15 representative fixtures | done (WP1.4) |
| `tests/robustness` | deterministic pseudo-fuzz (no panics/hangs) | done (WP1.4) |

Architecture target: `Source → Tokens → AST → HIR → typed HIR → backend`;
Gate 1 covers through the AST. Spec defects found during implementation
follow the T10 protocol: spec fix + regenerated artifacts + fixture re-triage
+ compiler change in one commit.
