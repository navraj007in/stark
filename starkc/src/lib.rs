//! starkc — compiler for the STARK Core v1 language.
//!
//! Normative spec: `STARKLANG/docs/spec/` (single-file compilation:
//! `STARK-Core-v1.md`). Delivery is governed by `STARKLANG/docs/ROADMAP.md`
//! and executed per `STARKLANG/docs/PLAN.md`.
//!
//! Pipeline (target architecture per PLAN.md):
//! `Source -> Tokens -> AST -> HIR -> typed HIR -> backend`
//! Gates 1–3 implement Source -> Tokens -> AST -> HIR -> typed HIR ->
//! interpreter, including the `core-min` runtime.

pub mod ast;
pub mod ast_dump;
pub mod borrowck;
pub mod deploy;
pub mod diag;
pub mod doc_gen;
pub mod extensions;
pub mod flow;
pub mod formatter;
pub mod hir;
pub mod interp;
pub mod lexer;
pub mod literal;
pub mod lsp;
pub mod onnx;
pub mod options;
pub mod package;
pub mod parser;
pub mod resolve;
pub mod source;
pub mod test_runner;
pub mod typecheck;
