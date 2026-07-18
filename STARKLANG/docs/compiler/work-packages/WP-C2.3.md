# WP-C2.3 — Shared Project-Analysis Entry Point

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation). Extracted in scope
from `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

Status: **Completed 2026-07-18.** Compiler and tooling consumers now share
`analysis::analyze_project(ProjectInput, LanguageOptions) -> ProjectAnalysis`. See
`COMPILER-STATE.md` for verification evidence and the next work package.

## Scope

The shared analysis result owns the source map, optional package graph, AST, optional HIR,
resolution tables, optional type tables, diagnostics, symbol index, source provenance, enabled
extension set, and session-scoped opaque query handles. Partial AST and diagnostics remain
available when a later semantic stage cannot run.

The following equivalent semantic consumers use the shared entry point:

- `starkc check`, including snippet mode;
- `stark check`, `stark build`, and `stark run` package compilation;
- LSP document compilation and caching;
- documentation example validation;
- deployment pipeline validation.

The formatter and `starkc parse` intentionally retain parser-only paths: their contract is
syntax transformation/inspection, not semantic project analysis. Parser, resolver, and
type-checker unit tests may also call individual stages to test those stages in isolation.

## Stable identity contract

`SourceId` and `QueryHandle` values are meaningful only within their owning
`ProjectAnalysis`. Query handles encode an entity category and a private session/slot identity;
consumers cannot extract or persist raw arena indices. `ProjectAnalysis::owns_handle` rejects a
handle from another analysis session. Position and symbol query behavior is deliberately
deferred to WP-C2.4.

## Scope-control answers

- **Exact compiler claim tested:** all equivalent semantic consumers can obtain one coherent,
  owned parse/resolve/typecheck result with source provenance and session-stable identities.
- **Later mechanism that would make the result impossible to attribute:** implementing C2.4
  queries separately in each consumer, or exposing HIR arena indexes as protocol identities.
- **Strongest existing comparator:** the pre-existing full compiler suite plus focused shared
  analysis tests for successful ownership, partial failure results, symbol indexing, and
  cross-session handle rejection.
- **Negative result that would stop this WP/gate:** a required consumer having an input model
  that cannot be represented without changing language semantics or weakening diagnostics.

## Execution log

See `COMPILER-STATE.md` session record `### WP-C2.3`.
