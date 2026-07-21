//! WP-C5.2e — the native trap ABI (§13.1/§13.2). Every checked operation the backend lowers
//! (`starkc/src/backend/generated_rust/emit_bodies.rs`) resolves its source location at
//! COMPILE TIME (`SourceFile::line_col`, already available to the backend from
//! `MirProgram::files`) and bakes the file path/line/column into the generated call site as
//! literals, rather than a runtime span-ID lookup table indirection (§13.1's design allows for
//! deduplicating span data behind compact IDs for large programs; baking literals is simpler and
//! exactly as correct for a debug-profile MVP, and can be revisited if generated-binary size
//! from repeated string literals ever becomes a real problem, which it plausibly is not at MVP
//! scale). This module only needs to format and abort, not resolve anything.
//!
//! Exit code 101 on trap matches `stark run`'s own established convention exactly
//! (`starkc/src/bin/stark.rs`: `ExitCode::from(if error.is_trap { 101 } else { 1 })`) --
//! reusing the existing convention rather than inventing a new one.

/// Mirrors `starkc::mir::TrapCategory`. A native copy rather than a dependency on `starkc`:
/// the runtime crate must not depend on the compiler crate (it ships with generated binaries,
/// the compiler does not). The generated-Rust backend's `emit_bodies.rs` relies on these two
/// enums sharing identical variant names (it interpolates `starkc::mir::TrapCategory`'s `Debug`
/// output directly as a `stark_runtime::trap::TrapCategory::` path segment) -- keep them in
/// lockstep if either is ever extended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrapCategory {
    IntegerOverflow,
    DivideByZero,
    IndexOutOfBounds,
    CastFailure,
    Panic,
    UnwrapNone,
    UnwrapErr,
    AssertFailure,
    InvalidShift,
}

impl TrapCategory {
    /// Not claimed to match the HIR interpreter's own ad hoc per-call-site message strings
    /// (`starkc/src/interp.rs`) byte-for-byte -- no such canonical table exists there to match;
    /// the differential comparator (`WP-C5-ENTRY.md` §15.1) checks trap CATEGORY plus source
    /// file/line, not stderr text. This exists for a native binary's stderr to be readable, not
    /// to satisfy a byte-equality contract.
    fn message(self) -> &'static str {
        match self {
            TrapCategory::IntegerOverflow => "integer overflow",
            TrapCategory::DivideByZero => "division by zero",
            TrapCategory::IndexOutOfBounds => "index out of bounds",
            TrapCategory::CastFailure => "cast failure",
            TrapCategory::Panic => "explicit panic",
            TrapCategory::UnwrapNone => "called unwrap on a `None` value",
            TrapCategory::UnwrapErr => "called unwrap on an `Err` value",
            TrapCategory::AssertFailure => "assertion failed",
            TrapCategory::InvalidShift => "invalid shift amount",
        }
    }
}

/// The native trap ABI: reports category and source location on stderr, then aborts with the
/// established trap exit code. No pending Drop glue runs (§7.7) -- there is none to run yet in
/// C5's scope (every locally-declared type so far is `Copy`); the generated crate's
/// `panic = "abort"` profile means nothing downstream of `std::process::exit` ever runs anyway,
/// so this needs no unwind-suppression of its own.
pub fn abort(category: TrapCategory, file: &str, line: u32, column: u32) -> ! {
    eprintln!("error: runtime trap: {}", category.message());
    eprintln!("  --> {file}:{line}:{column}");
    std::process::exit(101);
}
