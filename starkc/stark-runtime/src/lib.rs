//! WP-C5.1b — minimal runtime skeleton consumed by generated-Rust native binaries.
//!
//! Scope per `WP-C5-ENTRY.md` §9: only what the C5 MVP uses. This crate is independently
//! versioned from MIR (`RUNTIME_VERSION` below is the "native runtime ABI version" in §9.2's
//! version-identity record, distinct from `MIR_VERSION`/`MIR_RUNTIME_SURFACE` in
//! `starkc::mir`). A generated crate that links a runtime whose `RUNTIME_VERSION` its compiler
//! build does not expect must fail before user code runs (§9.2) -- `version::check` implements
//! that gate.
//!
//! What's real in C5.1b: `output` (stdout/stderr byte submission) and `version` (the
//! compatibility check). What's a placeholder: `trap` (category vocabulary only -- the actual
//! abort path is C5.2e's deliverable, since C5.1b's proof program never traps) and `value`/
//! `provider_abi` (empty modules marking the C6.3/C5.1c module boundary from §9, populated when
//! move/Drop lowering and the Native Provider ABI validator land).

pub mod output;
pub mod provider_abi;
pub mod slot;
pub mod string;
pub mod trap;
pub mod value;
pub mod version;
