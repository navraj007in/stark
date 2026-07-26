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
//! Two things remain pinned as boundaries rather than fixed, each with the condition that retires it:
//!
//! 1. **Native refuses every non-`Unit` entry** (`Unsupported: the entry instance must return Unit`).
//!    PROC-MAIN-001 makes those signatures legal executable targets, so this is "a C5-style
//!    unsupported profile remaining for normative executable Core" — `WP-C6-ENTRY.md` §3 required
//!    result 6, i.e. a Gate C6 blocker. Escalated by owner decision, not built inside a corpus
//!    package. When it lands, promote these to three-engine cases.
//! 2. **`invalid-exit-status` has no `TrapCategory`.** PROC-EXIT-001 requires a language trap for an
//!    out-of-range status; the nine categories contain nothing for it and adding one is a CE3
//!    (WP-C6.0 froze trap identity). Owner decision CD-150: **bundle that amendment with the native
//!    entry work**, since the backend that emits a non-`Unit` entry has to emit this trap anyway.
//!    Until then MIR fails loudly there instead of completing with a wrong status.

use starkc::backend::generated_rust::{emit_native_debug, BackendDiagnostic, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::{run_program, MirFailure, MirRunError};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// What both interpreters report for one program: exactly the fields PROC-EXIT-001 specifies.
#[derive(Debug, PartialEq, Eq)]
struct Termination {
    stdout: String,
    status: u8,
    stderr: String,
}

struct Front {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: starkc::typecheck::TypeTables,
}

fn front_end(name: &str, source: &str) -> Result<Front, Vec<String>> {
    let file = Arc::new(SourceFile::new(name, source.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message))
        .collect();
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(Front {
        hir,
        file,
        tables: checked.tables,
    })
}

/// Runs one program through BOTH interpreters, requires them to agree on every PROC-EXIT-001 field,
/// and returns the agreed termination for the caller to check against the rule. Agreement alone is
/// not the assertion — that is what let this contract stay broken while both `Int32` and `Err`
/// silently reported 0 in MIR.
fn both_interpreters(name: &str, source: &str) -> Termination {
    let front = front_end(name, source).unwrap_or_else(|e| panic!("{name}: typecheck: {e:?}"));
    let program = lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("{name}: lowering: {} @ {:?}", e.what, e.span));

    let hir = interp::run_with_partial_output(&front.hir, front.file.clone(), &front.tables)
        .map(|exec| Termination {
            stdout: exec.output,
            status: exec.status,
            stderr: exec.stderr,
        })
        .unwrap_or_else(|(e, partial)| {
            panic!(
                "{name}: oracle did not complete: {} (partial {partial:?})",
                e.message
            )
        });

    let verified = verify_program(&program)
        .unwrap_or_else(|errors| panic!("{name}: verifier rejected MIR:\n{errors:#?}"));
    let mir = match run_program(verified) {
        Ok(exec) => Termination {
            stdout: exec.output,
            status: exec.status,
            stderr: exec.stderr,
        },
        Err(MirFailure { error, output }) => {
            panic!("{name}: MIR did not complete: {error:?} (partial {output:?})")
        }
    };

    assert_eq!(
        hir, mir,
        "{name}: HIR/MIR disagreement on the entry contract"
    );
    hir
}

/// PROC-EXIT-001: "Normal `Unit` ... return status 0."
#[test]
fn unit_entry_completes_with_status_zero() {
    let observed = both_interpreters(
        "entry_exit__01.stark",
        "fn main() {\n    print(\"x\");\n}\n",
    );
    assert_eq!(
        observed,
        Termination {
            stdout: "x".to_string(),
            status: 0,
            stderr: String::new(),
        }
    );
}

/// PROC-EXIT-001: "`Int32` ... must be in `0..=255` and return that status."
///
/// The pre-DEV-111 MIR reported 0 here. 0 is also what a `Unit` entry reports, so a comparator that
/// only checked HIR-vs-MIR *agreement* on the `Unit` case would never have noticed.
#[test]
fn int32_entry_returns_its_value_as_the_exit_status() {
    let observed = both_interpreters("entry_exit__02.stark", "fn main() -> Int32 {\n    3\n}\n");
    assert_eq!(
        observed,
        Termination {
            stdout: String::new(),
            status: 3,
            stderr: String::new(),
        }
    );
}

/// PROC-EXIT-001: "`Err(message)` writes `message` plus LF to stderr and returns status 1."
#[test]
fn err_entry_writes_the_message_to_stderr_and_returns_status_one() {
    let observed = both_interpreters(
        "entry_exit__03.stark",
        "fn main() -> Result<Unit, String> {\n    Err(String::from(\"boom\"))\n}\n",
    );
    assert_eq!(
        observed,
        Termination {
            stdout: String::new(),
            status: 1,
            // The LF is normative and load-bearing: the rule says "message plus LF".
            stderr: "boom\n".to_string(),
        }
    );
}

/// Status 255 is the last in-range value; 256 is the first out-of-range one. Pinning the boundary
/// keeps the `u8` conversion honest rather than trusting that some value in the middle works.
#[test]
fn the_last_in_range_exit_status_is_not_a_trap() {
    let observed = both_interpreters("entry_exit__04.stark", "fn main() -> Int32 {\n    255\n}\n");
    assert_eq!(observed.status, 255);
}

/// **Escalation 2, pinned.** PROC-EXIT-001 requires an `invalid-exit-status` TRAP here. The oracle
/// raises one; MIR cannot, because the trap has no `TrapCategory` and adding one is a CE3. So MIR
/// fails loudly and deterministically instead — a wrong answer would be status 0, which is what it
/// used to report. This test retires when the category exists: at that point both engines trap and
/// this becomes an ordinary trap-parity case.
#[test]
fn an_out_of_range_exit_status_is_refused_by_both_engines_pending_a_trap_category() {
    let name = "entry_exit__05.stark";
    let source = "fn main() -> Int32 {\n    300\n}\n";
    let front = front_end(name, source).expect("accepted by the front end");
    let program = lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("{name}: lowering: {} @ {:?}", e.what, e.span));

    // The oracle: a trap, worded `invalid-exit-status` and carrying no category.
    let (error, _) = interp::run_with_partial_output(&front.hir, front.file.clone(), &front.tables)
        .expect_err("the oracle must trap on an out-of-range exit status");
    assert!(error.is_trap, "the oracle must classify this as a trap");
    assert!(
        error.message.contains("invalid-exit-status"),
        "unexpected oracle message: {}",
        error.message
    );
    assert!(
        error.trap_category.is_none(),
        "if this trap has gained a category, the CE3 is decided — make both engines trap and \
         delete this test in favour of a trap-parity case"
    );

    // MIR: a loud internal error naming the escalation, NOT a completion.
    let verified = verify_program(&program).expect("verifier accepts it");
    match run_program(verified) {
        Err(MirFailure {
            error: MirRunError::Internal(message),
            ..
        }) => assert!(
            message.contains("invalid-exit-status") && message.contains("DEV-111"),
            "unexpected MIR internal error: {message}"
        ),
        Ok(exec) => panic!(
            "MIR completed with status {} — PROC-EXIT-001 requires a trap, and completing here is \
             exactly the DEV-111 defect",
            exec.status
        ),
        Err(other) => panic!("unexpected MIR failure: {other:?}"),
    }
}

/// **Escalation 1, pinned.** The native backend refuses every non-`Unit` entry, though
/// PROC-MAIN-001 admits four entry types. Recorded as a Gate C6 blocker (`WP-C6-ENTRY.md` §3
/// required result 6), which is why the cases above are two-engine.
///
/// If this test starts failing because the build SUCCEEDS, that is the good outcome: promote the
/// entry-contract cases to three-engine and close matrix rows K15–K17.
#[test]
fn native_refuses_every_non_unit_entry_signature() {
    for (name, source) in [
        ("entry_exit__02.stark", "fn main() -> Int32 {\n    3\n}\n"),
        (
            "entry_exit__03.stark",
            "fn main() -> Result<Unit, String> {\n    Err(String::from(\"boom\"))\n}\n",
        ),
    ] {
        let front = front_end(name, source).expect("accepted by the front end");
        let program = lower_program(&front.hir, &front.tables, front.file.clone())
            .unwrap_or_else(|e| panic!("{name}: lowering: {} @ {:?}", e.what, e.span));
        let verified = verify_program(&program).expect("verifier accepts it");
        let target_dir = std::env::temp_dir().join(format!(
            "stark_c65_entry_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&target_dir);
        let result = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: target_dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        );
        let _ = std::fs::remove_dir_all(&target_dir);
        match result {
            Err(BackendDiagnostic::Unsupported(message)) => assert!(
                message.contains("entry instance must return Unit"),
                "{name}: unexpected refusal: {message}"
            ),
            Err(other) => panic!("{name}: unexpected native error: {other:?}"),
            Ok(_) => panic!(
                "{name}: native now ACCEPTS a non-Unit entry — DEV-111's native half is closed. \
                 Promote the entry-contract cases to three-engine and close matrix rows K15-K17."
            ),
        }
    }
}

/// **DEV-112, fixed under CD-150.** `()` now typechecks as `Unit`, as TYPE-PRIM-001 requires — the
/// checker used to give the empty tuple its own type that unified with nothing, so no value of type
/// `Unit` could be written at all. Both spellings, in type and value position.
#[test]
fn the_unit_value_literal_typechecks_as_unit() {
    let observed = both_interpreters(
        "entry_exit__06.stark",
        "fn main() {\n    let x: Unit = ();\n    let y: () = ();\n    print(\"ok\");\n}\n",
    );
    assert_eq!(observed.stdout, "ok");
    assert_eq!(observed.status, 0);
}

/// PROC-EXIT-001: "Normal `Unit` and **`Ok(Unit)`** return status 0."
///
/// This is the branch DEV-112 made unreachable: the rule gives `Ok(Unit)` its own clause, and until
/// `()` typechecked as `Unit` there was no way to construct the value, so a `Result<Unit, String>`
/// entry could only ever return `Err`.
#[test]
fn ok_unit_entry_completes_with_status_zero() {
    let observed = both_interpreters(
        "entry_exit__07.stark",
        "fn main() -> Result<Unit, String> {\n    Ok(())\n}\n",
    );
    assert_eq!(
        observed,
        Termination {
            stdout: String::new(),
            status: 0,
            stderr: String::new(),
        }
    );
}
