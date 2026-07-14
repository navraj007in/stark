# starkc

Compiler for the STARK Core v1 language. Rust, stable toolchain, currently at
**WP1.1 (scaffold)** of Gate 1.

- Language definition: [`../STARKLANG/docs/spec/`](../STARKLANG/docs/spec/)
  (normative), single file: `STARK-Core-v1.md`
- Delivery gates: [`../STARKLANG/docs/ROADMAP.md`](../STARKLANG/docs/ROADMAP.md)
- Engineering plan: [`../STARKLANG/docs/PLAN.md`](../STARKLANG/docs/PLAN.md)
  (standing decisions T1–T12, work packages)
- Conformance corpus: [`../STARKLANG/tests/spec-fixtures/`](../STARKLANG/tests/spec-fixtures/)

## Use

```bash
cargo run -- parse file.stark   # currently reports a WP1.1 stub diagnostic
cargo test
```

## Layout

| Module | Contents | Status |
| --- | --- | --- |
| `source` | `SourceFile`, `Span`, line/column mapping | done (WP1.1) |
| `diag` | diagnostics in the spec's normative render format | done (WP1.1) |
| `lexer` | `01-Lexical-Grammar.md` | WP1.2 |
| `ast` | arena + ID node tree (PLAN.md T6) | WP1.4 |
| `parser` | recursive descent + Pratt (`02-Syntax-Grammar.md`) | WP1.4 |

Architecture target: `Source → Tokens → AST → HIR → typed HIR → backend`;
Gate 1 covers through the AST. Spec defects found during implementation
follow the T10 protocol: spec fix + regenerated artifacts + fixture re-triage
+ compiler change in one commit.
