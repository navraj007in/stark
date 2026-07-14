//! AST for `02-Syntax-Grammar.md`.
//!
//! Per PLAN.md T6: arena-allocated nodes referenced by typed IDs
//! (`ExprId`, `ItemId`, ...); every node carries a `Span`; no Rust
//! references or lifetimes in the tree. Types/ownership facts attach in
//! side tables keyed by these IDs from Gate 2 onward.

// Node definitions land with the parser in WP1.4 (types first, then
// expressions, statements, items, patterns).
