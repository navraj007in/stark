//! Re-export shim for the Native Provider ABI v0.1 metadata types and validator, which moved to
//! the crate root (`crate::provider_abi`) in WP-C7.8.2a.
//!
//! **Why the move.** A10 (`mir-amendment-A10-provider-invocation.md`) puts a validated
//! `FunctionDecl` inside MIR itself, and `crate::mir` is deliberately backend-independent — it
//! imports nothing from `crate::backend`, because MIR is what backends *consume*. Leaving the
//! declaration types under `backend::` would have inverted that dependency for every consumer of
//! `Callee::Provider`.
//!
//! **Why the shim.** `stark-time/native/src/lib.rs` compiles against
//! `starkc::backend::provider_abi::*`, and WP-C7.8.1 Packet 1's exit condition requires that crate
//! to work with **no semantic or ABI-facing source change**. Keeping this path resolving means it
//! needs no change at all, which is the stronger form of that evidence.

pub use crate::provider_abi::*;
