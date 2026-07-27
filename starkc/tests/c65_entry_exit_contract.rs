//! WP-C6.5 finding **C65-F2 / DEV-111** — the executable entry contract, PROC-MAIN-001 and
//! PROC-EXIT-001 (07-Modules-and-Packages, "Executable and target contract").
//!
//! These are the retained cases from the divergence, kept per §18.3. What they found, at `b7e804a`:
//!
//! | program | PROC-EXIT-001 | HIR oracle | MIR | native |
//! | --- | --- | --- | --- | --- |
//! | `-> Result<Unit, String>` → `Err("boom")` | status 1, `boom\n` on stderr | correct | status 0, no stderr | build refused |
//! | `-> Int32 { 3 }` | status 3 | correct | status 0 | build refused |
//! | `-> Int32 { 300 }` | trap `invalid-exit-status` | correct | completes, status 0 | build refused |
//! | `main()` | status 0 | correct | correct | correct |
//!
//! MIR discarded the entry's return value outright (`run_program` matched `Ok(_)` and hardcoded
//! `status: 0`) and had no stderr channel at all. That half is FIXED — `entry_termination` in
//! `mir/interp.rs` — and every case below now compares the two interpreters field by field, with
//! the normative answer stated independently so two engines agreeing on a wrong status still fails.
//!
//! **DEV-112, fixed here too (CD-150).** The same investigation found that `()` did not typecheck as
//! `Unit` — the checker gave the empty tuple its own type, so `let x: Unit = ()` failed and **no
//! value of type `Unit` could be written at all**, leaving PROC-EXIT-001's `Ok(Unit)` clause
//! unreachable from source. TYPE-PRIM-001 settles it outright — *"`Unit` and `()` are two spellings
//! of the same single-inhabitant type"* — so this was a conformance bug, not a spec conflict, and
//! all three engines now canonicalise: `unit_or_tuple` in the checker, `Constant::Unit` in lowering,
//! `Value::Unit` in the oracle.
//!
//! **Both escalations are discharged (CD-164, the CD-150 CE3).** The tenth trap category
//! `InvalidExitStatus` exists (MIR amendment A7) and the native backend emits all four
//! PROC-MAIN-001 entry signatures, so every case here now runs on **three engines** through the
//! shared comparator — this suite is no longer one of C65-F1's private forks.
//!
//! Provenance for `invalid-exit-status` is the ENTRY FILE at 1:1, defined rather than discovered:
//! the contract is violated by the entry's result, not by an expression, so there is no
//! sub-expression the three engines could agree to blame.

mod support;

use support::differential::{three_engine, CompletionObservation, Observation, TrapMessageClass};

const UNIT_ENTRY: &str = include_str!("c6-corpus/cases/retained/entry_exit__01_unit_entry.stark");
const INT32_STATUS: &str =
    include_str!("c6-corpus/cases/retained/entry_exit__02_int32_status.stark");
const ERR_STDERR: &str = include_str!("c6-corpus/cases/retained/entry_exit__03_err_stderr.stark");
const MAX_STATUS: &str = include_str!("c6-corpus/cases/retained/entry_exit__04_max_status.stark");
const OK_UNIT_ENTRY: &str =
    include_str!("c6-corpus/cases/retained/entry_exit__05_ok_unit_entry.stark");
const UNIT_LITERAL: &str =
    include_str!("c6-corpus/cases/retained/entry_exit__06_unit_literal.stark");

/// Runs a case through ALL THREE engines using the shared comparator authority, then returns the
/// agreed observation for the caller to check against PROC-EXIT-001. Agreement is necessary, not
/// sufficient: every assertion below states the rule's answer independently, which is what caught
/// DEV-111 when MIR and a `Unit` entry both reported status 0 for different reasons.
fn entry_observation(name: &str, source: &str) -> Observation {
    three_engine(name, source)
}

fn completion(name: &str, source: &str) -> CompletionObservation {
    match entry_observation(name, source) {
        Observation::Completed(done) => done,
        other => panic!("{name}: expected completion, got {other:#?}"),
    }
}

/// PROC-EXIT-001: "Normal `Unit` ... return status 0."
#[test]
fn unit_entry_completes_with_status_zero() {
    let observed = completion("entry_unit", UNIT_ENTRY);
    assert_eq!(observed.stdout_bytes, b"x");
    assert_eq!(observed.exit_status, 0);
    assert!(observed.stderr_bytes.is_empty());
}

/// PROC-EXIT-001: "`Int32` ... must be in `0..=255` and return that status."
///
/// The pre-DEV-111 MIR reported 0 here. 0 is also what a `Unit` entry reports, so a comparator that
/// only checked HIR-vs-MIR *agreement* on the `Unit` case would never have noticed.
#[test]
fn int32_entry_returns_its_value_as_the_exit_status() {
    let observed = completion("entry_int32", INT32_STATUS);
    assert_eq!(observed.exit_status, 3);
    assert!(observed.stdout_bytes.is_empty() && observed.stderr_bytes.is_empty());
}

/// PROC-EXIT-001: "`Err(message)` writes `message` plus LF to stderr and returns status 1."
#[test]
fn err_entry_writes_the_message_to_stderr_and_returns_status_one() {
    let observed = completion("entry_err", ERR_STDERR);
    assert_eq!(observed.exit_status, 1);
    // The LF is normative and load-bearing: the rule says "message plus LF".
    assert_eq!(observed.stderr_bytes, b"boom\n");
    assert!(observed.stdout_bytes.is_empty());
}

/// Status 255 is the last in-range value; 256 is the first out-of-range one. Pinning the boundary
/// keeps the `u8` conversion honest rather than trusting that some value in the middle works.
#[test]
fn the_last_in_range_exit_status_is_not_a_trap() {
    let observed = completion("entry_max", MAX_STATUS);
    assert_eq!(observed.exit_status, 255);
}

/// **CD-150 CE3, discharged.** An out-of-range exit status traps as `InvalidExitStatus` in all three
/// engines, at the defined provenance.
///
/// This was the escalated case: PROC-EXIT-001 requires a trap, and no `TrapCategory` covered it, so
/// MIR raised a loud `Internal` error rather than completing with a wrong status. The category now
/// exists (MIR amendment A7) and the trap is comparable.
#[test]
fn an_out_of_range_exit_status_traps_in_all_three_engines() {
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    match entry_observation("entry_out_of_range", "fn main() -> Int32 {\n    300\n}\n") {
        Observation::Trapped(trap) => {
            assert_eq!(trap.category, starkc::mir::TrapCategory::InvalidExitStatus);
            assert_eq!(trap.exit_status, 101, "TRAP-ABORT-001");
            assert_eq!(
                trap.message_class,
                TrapMessageClass::CategoryOnly,
                "the offending value is not normative text, so it is not compared"
            );
            // Defined provenance: the entry file at 1:1 (MIR amendment A7).
            assert_eq!((trap.line, trap.column), (1, 1));
        }
        other => panic!("expected a trap, got {other:#?}"),
    }
}

/// **CD-150 CE3, discharged.** The native backend emits every PROC-MAIN-001 entry signature.
///
/// It previously refused all three non-`Unit` forms with `Unsupported`, which `WP-C6-ENTRY.md` §3
/// called "a C5-style unsupported profile remaining for normative executable Core" — a Gate C6
/// blocker. The entry is now emitted as an ordinary function and `fn main()` applies the exit
/// contract to its result.
#[test]
fn every_admitted_entry_signature_builds_and_runs_natively() {
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // Each of PROC-MAIN-001's four, through all three engines. `Unit` is covered by the case above;
    // these are the three that were refused.
    let int32 = completion("entry_native_int32", INT32_STATUS);
    assert_eq!(int32.exit_status, 3);
    let ok_unit = completion("entry_native_ok_unit", OK_UNIT_ENTRY);
    assert_eq!(ok_unit.exit_status, 0);
    let err = completion("entry_native_err", ERR_STDERR);
    assert_eq!(err.exit_status, 1);
    assert_eq!(err.stderr_bytes, b"boom\n");
}

/// **DEV-112, fixed under CD-150.** `()` typechecks as `Unit`, as TYPE-PRIM-001 requires — the
/// checker used to give the empty tuple its own type that unified with nothing, so no value of type
/// `Unit` could be written at all. Both spellings, in type and value position, on three engines.
#[test]
fn the_unit_value_literal_typechecks_as_unit() {
    if !support::differential::rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let observed = completion("entry_unit_literal", UNIT_LITERAL);
    assert_eq!(observed.stdout_bytes, b"ok");
    assert_eq!(observed.exit_status, 0);
}

/// PROC-EXIT-001: "Normal `Unit` and **`Ok(Unit)`** return status 0."
///
/// This is the branch DEV-112 made unreachable: the rule gives `Ok(Unit)` its own clause, and until
/// `()` typechecked as `Unit` there was no way to construct the value, so a `Result<Unit, String>`
/// entry could only ever return `Err`.
#[test]
fn ok_unit_entry_completes_with_status_zero() {
    let observed = completion("entry_ok_unit", OK_UNIT_ENTRY);
    assert_eq!(observed.exit_status, 0);
    assert!(observed.stdout_bytes.is_empty() && observed.stderr_bytes.is_empty());
}
