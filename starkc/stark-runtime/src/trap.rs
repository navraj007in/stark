//! Trap category vocabulary, plus a MINIMAL abort path WP-C5.2c needs to prove checked
//! operations actually trap (an "overflow adds and silently continues" native binary would
//! violate STARK's always-trap semantics, so *some* abort has to exist now). This is
//! deliberately NOT the final trap ABI: §13.1's source-map/span-ID lookup and §13.2's
//! canonical trap record/stderr format are WP-C5.2e's deliverable. `abort_minimal` reports only
//! the category, on stderr, in a format that is expected to change once WP-C5.2e lands --
//! callers must not depend on its exact text or exit code as a stable contract.

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

/// A placeholder abort: reports the trap category on stderr and terminates the process.
/// WP-C5.2e replaces (not wraps) this with the real trap ABI -- a source span resolved through
/// the §13.1 source map, the canonical §13.2 stderr format, and a settled exit-code contract.
/// No pending Drop glue runs, matching §7.7 -- there is none to run yet in WP-C5.2c's scope
/// (every locally-declared type so far is `Copy`), but the "abort, don't unwind" shape is
/// already correct: the generated crate's `panic = "abort"` profile (WP-C5-ENTRY.md §7.7) means
/// nothing downstream of `std::process::exit` ever runs.
pub fn abort_minimal(category: TrapCategory) -> ! {
    eprintln!("stark: trap ({category:?}) -- WP-C5.2e will replace this with the real trap ABI");
    std::process::exit(1);
}
