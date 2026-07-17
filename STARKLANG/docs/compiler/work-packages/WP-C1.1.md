# WP-C1.1 — Lexical and Syntax Requalification

Gate: C1 (Core v1 Conformance Closure). Extracted verbatim in scope from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Re-run and strengthen:

- every classified specification fixture;
- reserved-token handling;
- literal boundary and overflow lexing;
- nested comments and raw strings;
- generic-closing-token splitting;
- parser recovery progress guarantees;
- recursion/depth limits;
- AST span integrity;
- multi-file module parsing;
- extension syntax rejected in Core-only mode and accepted only when explicitly enabled.

Add deterministic property/fuzz tests for no panic, no hang, bounded resource use, and stable
diagnostics on malformed input.

## Done when

Every lexical/syntax rule has a coverage entry and executable evidence; no known syntax
deviation remains unrecorded.

## Scope-control answers (Charter §2.6, recorded before implementation)

- **Exact compiler claim tested:** that `lexer.rs`/`parser.rs` conform to
  `01-Lexical-Grammar.md`/`02-Syntax-Grammar.md`, with every classified fixture passing and the
  ten checklist items above holding under both positive and adversarial input.
- **Later mechanism that would make the result impossible to attribute:** letting semantic
  concerns (name resolution, typing) leak into this WP's tests, or skipping the negative-evidence
  half of any rule.
- **Strongest existing comparator:** the existing 121-fixture corpus
  (`STARKLANG/tests/spec-fixtures/manifest.toml`, Gate 1 evidence: 91/91 checked verdicts green)
  plus `starkc/tests/{conformance,robustness,diag_format,snapshots}.rs`. This WP strengthens
  that baseline; it does not replace it wholesale.
- **Negative result that would stop this WP/gate:** a real lexer/parser bug — a valid Core
  program rejected, or an invalid one silently accepted, or a panic/hang on malformed input.

## Input packet (per Charter §2.1)

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file + `01-Lexical-Grammar.md`,
`02-Syntax-Grammar.md`, `lexer.rs`, `parser.rs`, `ast.rs`, `ast_dump.rs`,
`tests/{conformance,robustness,diag_format,snapshots}.rs`, `STARKLANG/tests/spec-fixtures/`.

## Execution log

**Closed 2026-07-17.** Full dated evidence, files touched, and decisions:
`COMPILER-STATE.md` session record `### WP-C1.1`.

Summary: all 10 checklist items reviewed with cited evidence; 6 strengthened with new tests (12
tests across 5 files, including a new `starkc/tests/span_integrity.rs`); 2 confirmed already
well-covered (parser recovery, extension gating); 1 real production bug found and fixed
(DEV-014); 1 real gap found and deferred with owner assigned (DEV-015, literal overflow — a
design-scope question, not a testing gap). `cargo test --workspace --all-targets --all-features`:
395 passed / 0 failed / 2 ignored (up from 383/0/2). `cargo fmt --check` clean. Clippy clean on
all touched files (pre-existing unrelated repo-wide clippy debt recorded as DEV-016).
`check-conformance.py` clean after coverage-database citation updates.

**Done when, evaluated:** "every lexical/syntax rule has a coverage entry and executable
evidence" — met; all 27 LEX/SYN rules already had entries (WP-C0.3), and the 7 most directly
exercised by this WP now have corrected, stronger evidence citations. "No known syntax deviation
remains unrecorded" — met; every gap found (DEV-014 through DEV-018) is recorded in
`starkc/docs/conformance/KNOWN-DEVIATIONS.md` with owner and disposition, none left silent.

Next: WP-C1.2 (name resolution, modules, and visibility).
