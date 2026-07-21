//! WP-C5.1 -- native backend (Gate C5, `WP-C5-ENTRY.md`). `generated_rust` is the sole
//! production backend (CD-026: `SELECT-GENERATED` -- generated Rust consuming verified MIR;
//! direct Cranelift stays a C7-gated migration option, not implemented here). `version` records
//! the §9.2 version-identity fields this compiler build contributes to every generated crate.

pub mod generated_rust;
pub mod version;
