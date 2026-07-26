//! Shared support code for the C6 differential test binaries.
//!
//! Included by each consuming test binary with a plain `mod support;` at its root — the repo's
//! existing convention for shared test scaffolding (`tests/common/mod.rs`). Files under
//! `tests/support/` are NOT compiled as test binaries themselves, so nothing here runs on its own;
//! it is compiled once per including binary.
//!
//! `#![allow(dead_code)]` because each consumer uses a different subset: a binary that only needs
//! the comparator should not have to import the engine runners to silence warnings.

#![allow(dead_code)]

pub mod corpus;
pub mod differential;
