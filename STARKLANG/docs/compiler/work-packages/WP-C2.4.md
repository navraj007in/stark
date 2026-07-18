# WP-C2.4 — Position and Symbol Query Infrastructure

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation). Extracted in scope
from `STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

Status: **Completed 2026-07-18.** `ProjectAnalysis` now exposes compiler-owned, session-stable
position, symbol, definition, reference, type, signature, enclosing-context, and symbol-list
queries. See `COMPILER-STATE.md` for verification evidence and the next work package.

## Scope delivered

- innermost syntax and HIR node at a source byte position;
- item/local symbol at a definition or resolved use position;
- definition and deterministic reference locations;
- inferred expression/local types and source-like item/type rendering;
- enclosing item, module, source, and package provenance;
- public item enumeration;
- document and workspace symbol enumeration.

All public identities are opaque `QueryHandle` values tied to one `ProjectAnalysis`. Syntax and
HIR handles have explicit domains, but neither exposes its arena slot. Foreign-session handles
are rejected by every query. Query indexing walks the owned AST/HIR graph with source identity
propagated from module/item ownership, including external module files; tools do not reconstruct
compiler semantics or guess file ownership.

Compiler-provided prelude entities do not have physical definition locations. Member
expressions for which the current HIR carries no resolved member identity return no symbol
instead of guessing from text; arena-backed items and locals are fully indexed, and variant or
trait-member `Res` values resolve to their owning arena item.

## Scope-control answers

- **Exact compiler claim tested:** a successful `ProjectAnalysis` supplies stable, source-aware
  navigation and type information for resolved item/local symbols in single- and multi-file
  projects.
- **Later mechanism that would make the result impossible to attribute:** rebuilding indexes
  independently in LSP/native consumers or serializing raw AST/HIR arena numbers.
- **Strongest existing comparator:** focused single-file query coverage plus a real external
  module loaded from disk and queried across its definition/reference boundary.
- **Negative result that would stop this WP/gate:** a query requiring name/type resolution not
  represented by HIR; such cases return no symbol rather than a potentially incorrect result.

## Execution log

See `COMPILER-STATE.md` session record `### WP-C2.4`.
