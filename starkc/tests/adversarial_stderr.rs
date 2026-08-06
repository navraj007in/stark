//! WP-C7.9 Packet D — `eprint`/`eprintln` as a real, compared channel.
//!
//! **What was wrong.** `eprint` and `eprintln` are normative standard-library operations, and no
//! engine below the front end performed them:
//!
//! - the HIR oracle wrote them straight to the *host process's* stderr, so they never reached
//!   `Execution.stderr` and landed in the Rust test runner's own output instead;
//! - MIR had no lowering for them at all, so any program using them was refused;
//! - the native backend therefore never saw them and emitted nothing;
//! - and the comparator's `stderr_bytes` compared empty-to-empty for every program that did not
//!   return `Err` from `main`.
//!
//! Put together: a test could "agree" across three engines while the operation it was testing was
//! never performed by two of them. That is the failure mode this file exists to make impossible —
//! every case below pins the exact stderr bytes independently, so *nobody wrote anything* fails
//! instead of passing.
//!
//! **The trap cases need one more thing.** A program's stderr and the runtime's trap diagnostic
//! share one host stream. The differential runner therefore passes a fresh random token per run and
//! the runtime emits exactly one record carrying it; everything before that record is the program's
//! own stderr. Production output is unchanged — the token is absent for every real invocation.

mod support;

use starkc::mir::TrapCategory;
use support::differential::{agree_completing_with_streams, agree_trapping_with_streams};

// ------------------------------------------------------------------ the basic channel --

/// `eprintln` writes the text and one newline; `eprint` writes the text alone. Both reach the
/// captured channel in every engine.
#[test]
fn eprint_and_eprintln_write_to_stderr() {
    agree_completing_with_streams(
        "eprintln_basic",
        "fn main() { eprintln(\"first\"); eprint(\"second\"); }",
        "",
        "first\nsecond",
    );
}

/// Repeated writes preserve their own order.
#[test]
fn repeated_stderr_writes_preserve_order() {
    agree_completing_with_streams(
        "stderr_order",
        "fn main() { eprintln(\"a\"); eprintln(\"b\"); eprintln(\"c\"); }",
        "",
        "a\nb\nc\n",
    );
}

/// Two streams, each preserving its own order. They are separate streams, so the comparison is
/// per-stream — interleaving between them is a host property that PROC-STREAM-001 does not fix, and
/// pinning it would be pinning something the specification does not promise.
#[test]
fn stdout_and_stderr_are_separate_ordered_streams() {
    agree_completing_with_streams(
        "both_streams",
        "fn main() { println(\"out1\"); eprintln(\"err1\"); println(\"out2\"); eprintln(\"err2\"); }",
        "out1\nout2\n",
        "err1\nerr2\n",
    );
}

// --------------------------------------------------------------------- value rendering --

/// **The `&str`-only restriction is gone (DEV-174, CD-381) — these are the cases it asked for.**
///
/// WP-C7.9 recorded that `eprint`/`eprintln` accepted only `&str` while `println(42)`,
/// `println(some_string)` and `println(p)` for a user `Display` type all type-checked. It said the
/// lowering Packet D added was already fully type-directed, so **widening the signature would
/// require no lowering work, only a signature change and cases** — and that the test pinning the
/// restriction would fail the day that happened, which was "the right moment to add them".
///
/// That day is CD-381. 06-Standard-Library declares `fn eprintln<T: Display>(value: T)` and
/// PRINT-DISPLAY-001 names all four output functions together, so the restriction was a
/// conformance gap rather than a design. The prediction held exactly: the fix was the signature
/// plus the shared `Display` check, and no lowering changed.
///
/// The three shapes below are the three the old test rejected, now proven to RENDER — byte for
/// byte, on the stderr channel, through the same runtime operations the stdout path uses.
#[test]
fn the_eprint_family_accepts_every_display_value() {
    support::differential::agree_completing_with_streams(
        "eprint_int",
        "fn main() { eprintln(42); }",
        "",
        "42\n",
    );
    support::differential::agree_completing_with_streams(
        "eprint_owned_string",
        "fn main() { let s: String = String::from(\"owned\"); eprintln(s); }",
        "",
        "owned\n",
    );
    support::differential::agree_completing_with_streams(
        "eprint_display",
        "struct P { v: Int32 }\n\
         impl Display for P { fn fmt(&self) -> String { String::from(\"CUSTOM\") } }\n\
         fn main() { let p = P { v: 1 }; eprintln(p); }",
        "",
        "CUSTOM\n",
    );
}

/// Widening the signature did NOT widen it to anything: a type with no `Display` is still refused
/// on stderr, for the same reason and with the same diagnostic as on stdout. Both pairs now go
/// through one deferred `Display` check, so they cannot drift apart again.
#[test]
fn the_eprint_family_still_requires_display() {
    support::differential::rejects_at_typecheck(
        "eprint_no_display.stark",
        "struct Hidden { v: Int32 }\n\
         fn main() { let h = Hidden { v: 1 }; eprintln(h); }",
        "E0500",
    );
}

/// What the channel DOES carry today, proven byte for byte: `&str` literals, and the `as_str` of an
/// owned `String`. Both go through the same runtime operation the stdout path uses.
#[test]
fn str_values_render_identically_on_both_channels() {
    agree_completing_with_streams(
        "stderr_str_rendering",
        "fn main() { let s: String = String::from(\"owned\"); \
         eprintln(s.as_str()); eprintln(\"borrowed\"); \
         println(s.as_str()); println(\"borrowed\"); }",
        "owned\nborrowed\n",
        "owned\nborrowed\n",
    );
}

// ------------------------------------------------------------------- around a trap --

/// **Stderr written before a trap is retained and compared** — and it is the program's stderr, not
/// the runtime's trap diagnostic. Before Packet D neither half was observable: the bytes were never
/// captured, and there was no way to tell them from the diagnostic if they had been.
#[test]
fn stderr_before_a_trap_is_retained() {
    agree_trapping_with_streams(
        "stderr_then_trap",
        "fn main() { println(\"out\"); eprintln(\"about to fail\"); let z: Int32 = 0; println(1 / z); }",
        TrapCategory::DivideByZero,
        1,
        "out\n",
        "about to fail\n",
    );
}

/// An unterminated `eprint` prefix survives the trap. The runtime flushes stderr before writing its
/// diagnostic, so a buffered prefix cannot be lost the way an unflushed stdout prefix once was
/// (CD-120 Contract B, now enforced on both streams).
#[test]
fn an_unterminated_stderr_prefix_survives_a_trap() {
    agree_trapping_with_streams(
        "stderr_prefix_trap",
        "fn main() { eprint(\"partial\"); let z: Int32 = 0; println(1 / z); }",
        TrapCategory::DivideByZero,
        1,
        "",
        "partial",
    );
}

/// A trap with no program stderr at all: the diagnostic must not leak into the program's channel.
/// This is the case that fails if the nonce protocol is dropped or mis-parsed — the runtime's own
/// message would be attributed to the program.
#[test]
fn a_trap_alone_leaves_the_program_stderr_empty() {
    agree_trapping_with_streams(
        "trap_only",
        "fn main() { let z: Int32 = 0; println(1 / z); }",
        TrapCategory::DivideByZero,
        1,
        "",
        "",
    );
}

/// A `panic` carries user text on the runtime's channel, and it still does not become program
/// stderr. The message is compared as the trap's own field.
#[test]
fn a_panic_message_is_not_program_stderr() {
    agree_trapping_with_streams(
        "panic_not_stderr",
        "fn main() { eprintln(\"mine\"); panic(\"theirs\"); }",
        TrapCategory::Panic,
        1,
        "",
        "mine\n",
    );
}

// -------------------------------------------------------- the entrypoint's own stderr --

/// PROC-EXIT-001's `Err(message)` write follows the program's own stderr rather than replacing it.
/// Both halves of the channel are present, in that order.
#[test]
fn program_stderr_precedes_the_entrypoint_error() {
    // Not `agree_completing_with_streams`: an `Err` entrypoint completes with status 1, and that
    // helper requires status 0. The observation is still compared across engines by `three_engine`;
    // only the assertions are stated here.
    let observation = support::differential::three_engine(
        "err_completion",
        "fn main() -> Result<Unit, String> { eprintln(\"progress\"); Err(String::from(\"failed\")) }",
    );
    match observation {
        support::differential::Observation::Completed(done) => {
            assert_eq!(done.exit_status, 1, "an Err entrypoint returns status 1");
            assert_eq!(
                String::from_utf8_lossy(&done.stderr_bytes),
                "progress\nfailed\n",
                "the program's own stderr comes first, then PROC-EXIT-001's message"
            );
        }
        other => panic!("expected completion, got {other:#?}"),
    }
}
