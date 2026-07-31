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
    /// PROC-EXIT-001's out-of-range exit status (CD-150 CE3, MIR amendment A6).
    InvalidExitStatus,
}

impl TrapCategory {
    /// Not claimed to match the HIR interpreter's own ad hoc per-call-site message strings
    /// (`starkc/src/interp.rs`) byte-for-byte -- no such canonical table exists there to match;
    /// the differential comparator (`WP-C5-ENTRY.md` §15.1) checks trap CATEGORY plus source
    /// file/line, not stderr text. This exists for a native binary's stderr to be readable, not
    /// to satisfy a byte-equality contract.
    ///
    /// Public so the three-engine differential harness (`tests/three_engine_differential.rs`)
    /// can normalise a native binary's stderr back into a category against THIS table rather
    /// than a second copy of it in a test file, which would drift the day a category's wording
    /// changed.
    /// The category's STRUCTURAL name — the variant identifier, not its prose.
    ///
    /// WP-C7.9: the machine-readable trap record and the comparator both key on this, so trap
    /// identity survives any rewording of [`Self::message`]. Deliberately not `Debug`: a derived
    /// format is a debugging convenience that nothing promises to keep stable, and this is a
    /// protocol field.
    pub fn name(self) -> &'static str {
        match self {
            TrapCategory::IntegerOverflow => "IntegerOverflow",
            TrapCategory::DivideByZero => "DivideByZero",
            TrapCategory::IndexOutOfBounds => "IndexOutOfBounds",
            TrapCategory::CastFailure => "CastFailure",
            TrapCategory::Panic => "Panic",
            TrapCategory::UnwrapNone => "UnwrapNone",
            TrapCategory::UnwrapErr => "UnwrapErr",
            TrapCategory::AssertFailure => "AssertFailure",
            TrapCategory::InvalidShift => "InvalidShift",
            TrapCategory::InvalidExitStatus => "InvalidExitStatus",
        }
    }

    pub fn message(self) -> &'static str {
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
            TrapCategory::InvalidExitStatus => "invalid exit status",
        }
    }
}

/// The native trap ABI: reports category and source location on stderr, then terminates with the
/// established trap exit code.
///
/// **No destructor runs, and that is a property of `exit` rather than of the build profile**
/// (DROP-ABORT-001). `std::process::exit` terminates without unwinding, so live locals are never
/// dropped no matter what `panic` strategy the generated crate is built with. The C7.0 baseline
/// (CD-184) confirmed this is what makes trap semantics profile-independent, and it is why a
/// release profile can be added without changing trap behaviour.
///
/// The previous wording justified the same conclusion by saying there was no Drop glue to run
/// because "every locally-declared type so far is `Copy`". That was true in C5's scope and has been
/// false since WP-C6.1: Drop-bearing locals are ordinary now. The reasoning was replaced rather than
/// the sentence patched, because the old one would have become dangerous if believed -- it implied
/// the guarantee came from the absence of destructors rather than from not unwinding.
pub fn abort(category: TrapCategory, file: &str, line: u32, column: u32) -> ! {
    // CD-120 Contract B: emit any buffered pre-trap output before aborting, so the observable
    // stdout prefix matches the HIR/MIR interpreters (which retain their captured prefix).
    crate::output::flush_stdout();
    // WP-C7.9 Packet D: and the program's own stderr prefix, for the same reason and one more —
    // the diagnostic below goes to the same stream, so an unflushed `eprint` prefix would appear
    // after the trap record instead of before it.
    crate::output::flush_stderr();
    emit_trap_record(category, None, file, line, column);
    std::process::exit(101);
}

/// The separator that lets a differential runner tell a program's stderr from the runtime's own
/// trap diagnostic, which share one host stream (WP-C7.9 Packet D).
///
/// A fixed delimiter would be forgeable: a STARK program can print any bytes it likes, so a case
/// could produce something the comparator mistook for a trap record — accidentally or in a test
/// designed to check exactly that. The runner therefore generates a fresh random token per run and
/// passes it in this variable; a program that does not know the token cannot reproduce the record.
///
/// **When the variable is absent — every real invocation — output is unchanged**: production CLI
/// formatting is what a user sees, and this protocol exists only for the harness.
pub const TRAP_TOKEN_VAR: &str = "STARK_DIFFERENTIAL_TRAP_TOKEN";

/// Writes the trap diagnostic: one machine-readable record under the harness protocol, or the
/// ordinary human-readable form otherwise.
fn emit_trap_record(
    category: TrapCategory,
    message: Option<&str>,
    file: &str,
    line: u32,
    column: u32,
) {
    match std::env::var(TRAP_TOKEN_VAR) {
        Ok(token) if !token.is_empty() => {
            // One line, one record, machine-readable. `message` is last because it is the only
            // field that can contain arbitrary user text.
            eprintln!(
                "{token} category={} file={file} line={line} column={column} message={}",
                category.name(),
                message.unwrap_or("")
            );
        }
        _ => {
            eprintln!("error: runtime trap: {}", category.message());
            eprintln!("  --> {file}:{line}:{column}");
            if let Some(message) = message {
                eprintln!("  {message}");
            }
        }
    }
}

/// WP-C6.3e: a trap carrying a user MESSAGE — `panic(msg)` and a failed `assert*`. The category
/// header and `-->` location stay in the same shape as [`abort`] (so the same stderr parser reads
/// the category and provenance); the resolved `&str` message is reported on its own line.
pub fn abort_with_message(
    category: TrapCategory,
    message: &str,
    file: &str,
    line: u32,
    column: u32,
) -> ! {
    // CD-120 Contract B: flush buffered pre-trap output before aborting (see [`abort`]).
    crate::output::flush_stdout();
    crate::output::flush_stderr();
    emit_trap_record(category, Some(message), file, line, column);
    std::process::exit(101);
}
