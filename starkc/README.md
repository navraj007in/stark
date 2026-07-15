# starkc

Compiler for the STARK Core v1 language. Rust, stable toolchain.
**Gate 2 (Core semantic checker) is complete** — see
[`docs/gate2-exit.md`](docs/gate2-exit.md). The next delivery milestone is
Gate 3's minimal execution path.

- Language definition: [`../STARKLANG/docs/spec/`](../STARKLANG/docs/spec/)
  (normative), single file: `STARK-Core-v1.md`
- Delivery gates: [`../STARKLANG/docs/ROADMAP.md`](../STARKLANG/docs/ROADMAP.md)
- Engineering plan: [`../STARKLANG/docs/PLAN.md`](../STARKLANG/docs/PLAN.md)
  (standing decisions T1–T12, work packages)
- Conformance corpus: [`../STARKLANG/tests/spec-fixtures/`](../STARKLANG/tests/spec-fixtures/)

## Use

```bash
cargo run -- parse file.stark             # parse a program, report diagnostics
cargo run -- check file.stark             # resolve, type-check, and borrow-check
cargo run -- parse --snippet --dump f.stark  # block-body mode, print the AST
cargo run -- lex file.stark               # dump the token stream
cargo test                                # everything, incl. the 121-fixture
                                          # conformance suite and pseudo-fuzz
UPDATE_SNAPSHOTS=1 cargo test --test snapshots   # regenerate AST snapshots
```

## Terminal IDE

`starkide` is a dependency-free, Turbo C++ inspired terminal workbench. It
provides the classic blue/white interface, keyboard-driven drop-down menus, a
source editor, file open/save dialogs, and a compiler messages pane wired to
the complete Gate 2 semantic pipeline.

```bash
cargo run --bin starkide                         # new buffer
cargo run --bin starkide -- ../Practice/Basics/hello.st
```

Use **F10** for menus, **F2** to save, **F9** to compile, and **Ctrl+Q** to
quit. The Run command currently performs compilation and explains that
execution is unavailable until the STARK VM/backend lands.

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
| `tests/diag_format` | end-to-end diagnostic-format goldens | done (WP1.5) |
| `docs/gate1-exit.md` | Gate 1 exit report (T10 ledger, open questions) | done (WP1.5) |
| `hir`, `resolve` | HIR lowering, lexical/module names, imports, visibility | done (M2.1) |
| `typecheck`, `flow` | inference, control flow, generics, traits, coherence | done (M2.2–M2.3) |
| `borrowck` | moves, partial moves, reinitialization, lexical borrows | done (M2.4) |
| `tests/gate2-valid` | 26 end-to-end valid semantic programs | done (M2.5) |
| `docs/gate2-exit.md` | Gate 2 exit evidence and deferrals | done (M2.5) |

Architecture target: `Source → Tokens → AST → HIR → typed HIR → backend`;
Gate 2 covers through typed HIR plus ownership/borrow validation. Spec defects found during implementation
follow the T10 protocol: spec fix + regenerated artifacts + fixture re-triage
+ compiler change in one commit.
