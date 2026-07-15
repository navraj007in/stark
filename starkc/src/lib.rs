//! starkc — compiler for the STARK Core v1 language.
//!
//! Normative spec: `STARKLANG/docs/spec/` (single-file compilation:
//! `STARK-Core-v1.md`). Delivery is governed by `STARKLANG/docs/ROADMAP.md`
//! and executed per `STARKLANG/docs/PLAN.md`.
//!
//! Pipeline (target architecture per PLAN.md):
//! `Source -> Tokens -> AST -> HIR -> typed HIR -> backend`
//! Gate 1 implements Source -> Tokens -> AST.

pub mod ast;
pub mod ast_dump;
pub mod borrowck;
pub mod diag;
pub mod flow;
pub mod hir;
pub mod lexer;
pub mod parser;
pub mod resolve;
pub mod source;
pub mod typecheck;
