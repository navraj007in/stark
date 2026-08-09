//! starkc — compiler for the STARK Core v1 language.
//!
//! **Normative spec:** `STARKLANG/docs/spec/` — the numbered source documents 00–07 and the
//! approved `CORE-V1-*.md` chapters. `STARK-Core-v1.md` is a GENERATED compilation of those and is
//! never edited directly.
//!
//! **Governance.** Compiler work is governed by `COMPILER-STATE.md` (repo root) — read its
//! `# Current position` block first — under `STARKLANG/docs/compiler/COMPILER-CHARTER.md` and
//! `COMPILER-ROADMAP.md`. Forward planning for the platform is the repo-root `ROADMAP.md`.
//! `STARKLANG/docs/ROADMAP.md` and `PLAN.md` are HISTORICAL RECORDS of the closed Gate 1–7
//! sequence, not forward plans; this doc comment cited them as live until AS8 (2026-08-09).
//!
//! # Pipeline
//!
//! ```text
//! source -> lexer -> parser -> ast -> resolve -> hir -> typecheck -> typed HIR
//!                                                                      |
//!                          +-------------------------------------------+
//!                          |                    |                      |
//!                       interp              mir::lower             (flow, borrowck
//!                    HIR interpreter            |                   run inside typecheck)
//!                     THE ORACLE                v
//!                                          mir::verify
//!                                               |
//!                                  +------------+------------+
//!                                  v                         v
//!                            mir::interp            backend::generated_rust
//!                          THE LOWERING GATE          -> rustc -> THE PRODUCT
//! ```
//!
//! The three execution paths have **deliberately different roles and are not independent
//! reimplementations**: they share one front end, and some semantic rules are decided once so the
//! paths cannot drift. Which rules those are, and what can and cannot corroborate them, is recorded
//! in `STARKLANG/docs/compiler/ENGINE-SHARED-FATE-REGISTER.md`. Native debug and native release are
//! one engine at two optimisation levels — four CONFIGURATIONS, three engines.
//!
//! # Entry points
//!
//! ```text
//! analysis::analyze_project   the root entry: one package or one file -> ProjectAnalysis
//! session::CompilerSession    package-graph driven compilation
//! typecheck::analyze          type/flow/borrow checking over HIR
//! mir::lower / mir::verify    lowering and its verifier (an INDEPENDENT check, by design)
//! backend::generated_rust     Rust emission for the native path
//! ```
//!
//! # Module ownership
//!
//! `typecheck/` is eleven modules with an executable, cycle-free dependency graph enforced by
//! `tests/as7_module_dependencies.rs` (AS7; corrected under CD-393):
//!
//! ```text
//! types <- state <- infer <- traits <- convert <- {bounds, trait_contracts}
//!       <- patterns/body <- items <- mod
//! ```
//!
//! `extensions::tensor` holds the tensor extension's semantic authority behind the `TensorCheckCtx`
//! service trait (AS6). The boundary is **one-directional**: the extension consumes checked types
//! and never re-enters Core expression checking.

pub mod analysis;
pub mod ast;
pub mod ast_dump;
pub mod backend;
pub mod borrowck;
pub mod bound_dispatch;
pub mod build_cache;
pub mod deploy;
pub mod diag;
pub mod doc_gen;
pub mod extensions;
pub mod flow;
pub mod format_syntax;
pub mod formatter;
pub mod hir;
pub mod interp;
pub mod json;
pub mod layout;
pub mod lexer;
pub mod literal;
pub mod lsp;
pub mod mir;
pub mod native_build;
pub mod native_toolchain;
pub mod onnx;
pub mod options;
pub mod package;
pub mod parser;
pub mod provider_abi;
pub mod provider_bind;
pub mod provider_derive;
pub mod provider_manifest;
pub mod provider_registry;
pub mod provider_resolve;
pub mod provider_synth;
pub mod resolve;
pub mod session;
pub mod source;
pub mod source_extensions;
pub mod target;
pub mod test_runner;
pub mod typecheck;
