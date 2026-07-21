//! Trap category vocabulary only. The abort path (§7.7/§13.2: submit the trap record, run no
//! pending Drop glue, no unwind, terminate through this module) is WP-C5.2e's deliverable --
//! C5.1b's proof program (`fn main() { }`) never traps, so there is nothing yet to prove here.
//! Kept in its own module now (rather than added later) so `stark-runtime`'s public shape
//! matches §9's file layout from the start.

/// Mirrors `starkc::mir::TrapCategory`. A native copy rather than a dependency on `starkc`:
/// the runtime crate must not depend on the compiler crate (it ships with generated binaries,
/// the compiler does not).
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
