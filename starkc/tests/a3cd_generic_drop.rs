//! **A3c-D — a generic `Drop` implementation is refused by the HIR oracle, not guessed at.**
//!
//! Destruction reaches `drop_value` with a `Value` and recovers the nominal through
//! `nominal_item`, so `Wrapper<String>` and `Wrapper<Int32>` are indistinguishable: the type
//! arguments that selected the impl are gone. Every way of proceeding is a guess — an empty generic
//! frame, inference from runtime fields, or scanning impls and hoping — and a destructor is the
//! last place to guess, because running one with the wrong bindings corrupts silently.
//!
//! **This is a recorded limitation, not a repair.** Threading a concrete `Ty` through 44
//! `drop_value` call sites, or retaining type arguments in `Value`, is disproportionate to 0 `Drop`
//! impls in the first-party packages and 2 generic-`Drop` fixtures in the whole corpus. MIR and
//! native retain the arguments and execute it correctly, which is why the refusal is classified as
//! an oracle defect rather than a language outcome: telling the differential harness the PROGRAM is
//! at fault would be false.

use starkc::interp::{self, FailureClass};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn run(source: &str) -> interp::ExecutionOutcome {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "the program must type-check: {errors:?}");
    interp::run_capturing(&hir, file, &checked.tables)
}

/// **The control, and it carries most of the weight.** An ordinary non-generic `Drop` must still
/// run — a refusal that fired on every destructor would satisfy the test below and break the
/// language.
#[test]
fn a_non_generic_drop_still_executes() {
    let outcome = run("\
struct Res {
    id: Int32,
}

impl Drop for Res {
    fn drop(&mut self) {
        println(\"released\");
    }
}

fn main() {
    let r = Res { id: 1 };
    println(\"before\");
}
");
    assert!(outcome.result.is_ok(), "{:?}", outcome.result);
    assert_eq!(
        outcome.output, "before\nreleased\n",
        "a non-generic destructor runs at scope end as always"
    );
}

/// A generic `Drop` is refused rather than run with unbound parameters.
#[test]
fn a_generic_drop_is_refused() {
    let outcome = run("\
struct Wrapper<T> {
    value: T,
}

impl<T> Drop for Wrapper<T> {
    fn drop(&mut self) {
        println(\"dropped\");
    }
}

fn main() {
    let w = Wrapper { value: 1 };
    println(\"before\");
}
");
    let error = outcome
        .result
        .expect_err("a generic Drop cannot be executed by the oracle");
    assert!(
        error.message.contains("generic `Drop`"),
        "{}",
        error.message
    );
}

/// The refusal is an ORACLE defect, not a language outcome: MIR and native execute this correctly,
/// so classifying it a trap would tell the differential harness the program is at fault.
#[test]
fn the_refusal_is_an_internal_invariant_not_a_trap() {
    let outcome = run("\
struct Wrapper<T> {
    value: T,
}

impl<T> Drop for Wrapper<T> {
    fn drop(&mut self) {
        println(\"dropped\");
    }
}

fn main() {
    let w = Wrapper { value: 1 };
}
");
    let error = outcome.result.expect_err("refused");
    assert_eq!(error.class, FailureClass::InternalInvariant);
    assert_eq!(error.trap_category, None);
    assert!(!error.is_trap());
}

/// **No side effect may occur before the refusal.** A partially run destructor is worse than none:
/// it leaves the program in a state neither the author nor the engine expects. The destructor here
/// prints, so its absence from stdout is the proof it never began.
#[test]
fn the_destructor_body_does_not_partially_execute() {
    let outcome = run("\
struct Wrapper<T> {
    value: T,
}

impl<T> Drop for Wrapper<T> {
    fn drop(&mut self) {
        println(\"SIDE-EFFECT\");
    }
}

fn main() {
    println(\"before\");
    let w = Wrapper { value: 1 };
}
");
    assert!(outcome.result.is_err());
    assert!(
        !outcome.output.contains("SIDE-EFFECT"),
        "the destructor body must not have begun: {:?}",
        outcome.output
    );
    assert_eq!(outcome.output, "before\n");
}
