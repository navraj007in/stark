# WP8.3 — Test Framework Implementation

**Status:** Complete
**Module:** `starkc/src/test_runner/`

## Overview

`stark test` discovers and runs three kinds of tests:

1. **Unit tests** — `fn test_*()` functions anywhere in the package's own
   module tree (top-level or nested in `mod`), each run as its own
   interpreter entry point.
2. **Integration tests** — every `.stark` file under `tests/`, each a
   standalone program (its own `fn main()`), pass/fail determined by
   whether it panics.
3. **Examples** — every `.stark` file under `examples/`, same execution
   model as integration tests, reported separately.

## Why naming convention instead of `#[test]`

The plan's examples use `#[test]`/`#[ignore]` attributes. Core v1's
grammar has **no attribute syntax at all** — confirmed no `#` handling in
the lexer, no attribute node in the AST, nothing in
`01-Lexical-Grammar.md`/`02-Syntax-Grammar.md`. Surfaced to the user before
building (same pattern as WP8.2's comment-trivia finding); they chose the
naming-convention fallback the plan's own risk table already lists:

- `fn test_*()` — a test. Must take no parameters and have no receiver.
- `fn test_ignored_*()` — discovered and counted, but skipped unless
  `--ignored` is passed.

Display names keep the full function name including the `test_` marker
(`test module1::test_basic ... ok`), matching the plan's own output
example — only *discovery* strips the prefix to decide "is this a test."

See `docs/PHASE8_GRAMMAR_GAPS.md` for the full gap writeup and what a real
`#[test]` attribute would need in the grammar later.

## Architecture

- `test_runner::discover_tests(hir, root_file) -> Vec<TestCase>` — walks
  `Root::Program`, recursing into `ItemKind::Mod` bodies, collecting
  zero-arg receiverless `fn test_*` items. Handles the same
  `hir.item_files`-vs-`root_file` split every other multi-file-aware piece
  of the compiler needs (an item's source text comes from
  `hir.item_files.get(id)` if it was loaded from a submodule file, else
  from the package's entry file).
- `test_runner::run_test(hir, root_file, tables, test) -> TestResult` —
  runs one test via a new `interp::run_item(hir, file, tables, item)`
  entry point (parallel to the existing `interp::run`, which is hardcoded
  to call `main`; `run_item` calls an arbitrary zero-arg `ItemId`). A
  `RuntimeError` (trap/panic) is captured as `Outcome::Failed`, never
  propagated — one test failing never aborts the run.
- `test_runner::filter_by_name(tests, filter) -> Vec<&TestCase>` —
  substring match, matching `cargo test`'s single-test-name behavior.
- `stark test [name] [--ignored] [--show-output]` (`bin/stark.rs`) —
  loads the package the same way `stark check`/`build`/`run` already do,
  runs unit tests via the above, then walks `tests/` and `examples/` as
  standalone-program suites via the existing single-file
  parse→resolve→typecheck→`interp::run` pipeline.

## New stdlib builtins: `assert_eq`, `assert_ne`

The plan assumes `assert`, `assert_eq`, `assert_ne`, `assert_matches` all
exist; only `assert(cond: Bool)` did. Added `assert_eq`/`assert_ne` as new
interpreter builtins (generic `fn assert_eq<T>(a: T, b: T)` via a shared
fresh type variable, same pattern `swap`/`replace`/`take` use), with a
panic message showing both sides on failure:

```
assertion failed: `(left == right)`
  left: `4`
 right: `5`
```

`assert_matches(value, pattern)` was **not** added — patterns aren't
runtime values in Core v1, so it can't be a plain function call without
new syntax (macros or a dedicated statement form). See
`docs/PHASE8_GRAMMAR_GAPS.md` for the full reasoning and the recommended
`match`-based idiom in the meantime.

## Output format

Matches the plan's format:

```
running 4 tests

test test_add_basic ... ok
test test_add_zero ... ok
test test_ignored_slow_thing ... ignored
test geometry::test_area ... ok

test result: ok. 3 passed; 0 failed; 1 ignored; 0ms total

running 1 tests

test basic.stark ... ok

test result: ok. 1 passed; 0 failed

running 1 examples

example hello.stark ... ok

example result: ok. 1 passed; 0 failed
```

Failures include a `failures:` section with the panic message per test,
followed by a name list, matching `cargo test`'s layout.

## `--seed` omitted

The plan's `stark test -- --seed 42` is for deterministic test randomness.
Core v1 has no randomness primitives anywhere (grammar or stdlib), so
there is nothing to seed — omitted rather than shipped as a no-op flag.
Test order is already fully deterministic (depth-first source order).

## Testing

- `starkc/src/test_runner/mod.rs`: 2 unit tests for the `test_`/`ignored_`
  naming-convention parsing itself.
- `starkc/tests/test_framework.rs`: 12 integration tests covering
  discovery (top-level, nested `mod`, params excluded, bare `test`
  excluded, ignored-but-discovered), execution (pass/fail/captured output,
  one test's failure not affecting another), `assert_eq`/`assert_ne`
  pass/fail behavior and message content, and name filtering.
- Manually verified end-to-end against a real on-disk package (unit tests,
  integration tests, examples, `--ignored`, name filtering, all-pass exit
  0 / any-fail exit 1) and `mod`-nested test discovery.

349 tests pass across the whole workspace (all lib unit tests + every
integration test file); zero new clippy warnings; `cargo fmt --check`
clean.

## Known limitations

- No `#[ignore]`-equivalent for `examples/` files (only unit tests support
  the ignored convention) — every example under `examples/` always runs.
- `assert_matches` doesn't exist (see above) — no plain-function
  replacement is possible without new syntax.
- `--seed`/randomness: N/A, nothing in the language to seed yet.
- Integration-test/example failure messages are a single-line summary
  (`N parse/resolve/typecheck error(s)`, or the runtime panic message) —
  not full rendered diagnostics with source spans, unlike `stark check`'s
  output. Adequate for pass/fail triage; would need plumbing the full
  `Diagnostic` list through `run_standalone_suite` for source-level detail.

## Next steps

WP8.4 (VS Code Extension) can wire `stark.test` to `stark test` and,
longer-term, surface individual test pass/fail inline via the LSP the way
`cargo test`-integration extensions do — not attempted here (WP8.3 is CLI
only, matching the plan's own WP8.1/WP8.3 split).
