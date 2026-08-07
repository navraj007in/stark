//! **DEV-176 — a generic callable body executes without its checker-established context.**
//!
//! The HIR interpreter installs a substitution frame only for direct calls to free functions:
//! `push_generic_frame` reads parameter names from `hir::ItemKind::Fn` and returns an empty list
//! for every other item kind, and it has one call site. Impl generics, method generics, trait
//! generics and `Self` are never bound.
//!
//! **A3c-S repaired it**, and these tests now assert the repair rather than the defect. The
//! classification work stays: an accepted program that reaches execution without sufficient
//! compiler metadata is still an `InternalInvariant`, because the HIR interpreter is the
//! behavioural oracle and a defect classified as a language trap is one the differential harness
//! can accept as a legitimate outcome and then pressure MIR and native into reproducing.

use starkc::interp::FailureClass;
use starkc::options::LanguageOptions;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::{interp, typecheck};
use std::sync::Arc;

fn run(source: &str) -> Result<u8, starkc::interp::RuntimeError> {
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
    let _ = LanguageOptions::CORE;
    interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    )
    .result
}

/// **The control.** A free generic function resolves its parameter, so the defect is specific to
/// how the frame is installed rather than to generics as such. Without this, a test asserting the
/// method failure would be satisfied by an interpreter that supported no generics at all.
#[test]
fn a_free_generic_function_resolves_its_type_parameter() {
    let result = run("\
fn free_size<T>() -> UInt64 {
    size_of::<T>()
}

fn main() {
    let n = free_size::<Int32>();
    if n == 0u64 {
        panic(\"a size must be positive\");
    }
}
");
    assert!(
        result.is_ok(),
        "a free generic function must work: {result:?}"
    );
}

/// **The repair.** The identical body inside a generic impl now resolves `T`, and agrees with the
/// free function — the two forms differ only in which callable kind declares the parameter, so the
/// same answer is the point.
#[test]
fn a_generic_impl_method_resolves_its_type_parameter() {
    let file = Arc::new(SourceFile::new(
        "test.stark",
        "\
struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn size(&self) -> UInt64 {
        size_of::<T>()
    }
}

fn free_size<T>() -> UInt64 {
    size_of::<T>()
}

fn main() {
    let w = Wrapper { value: 7i32 };
    println(free_size::<Int32>());
    println(w.size());
}
",
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(
        outcome.result.is_ok(),
        "a generic method must resolve its parameter: {:?}",
        outcome.result
    );
    assert_eq!(
        outcome.output, "4\n4\n",
        "the free function and the generic method must agree"
    );
}

/// **Two instantiations of one body.** The environment is keyed by the CALL, not the body, so the
/// same method invoked at two types must answer differently — an environment attached to the body
/// could hold only one of them.
#[test]
fn one_generic_body_answers_differently_at_two_instantiations() {
    let file = Arc::new(SourceFile::new(
        "test.stark",
        "\
struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn size(&self) -> UInt64 {
        size_of::<T>()
    }
}

fn main() {
    let small = Wrapper { value: 7i32 };
    let large = Wrapper { value: 7i64 };
    println(small.size());
    println(large.size());
}
",
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(outcome.result.is_ok(), "{:?}", outcome.result);
    assert_eq!(
        outcome.output, "4\n8\n",
        "one body, two instantiations, two answers"
    );
}

/// **The classification stays, and still has a case.** A layout query whose type cannot be
/// concretised is a compiler defect, not a language outcome — reverting the constructor to
/// `RuntimeError::new` fails here and nowhere else. The environment installer raises the same
/// class, so the repair did not remove the need for it: it moved it from "always" to "when the
/// checker's metadata is genuinely insufficient".
#[test]
fn a_missing_generic_context_is_an_internal_invariant_not_a_trap() {
    // `concrete_runtime_ty` is the single choke point for both, so exercising the surviving path
    // through a value boundary proves the classification for both.
    let file = Arc::new(SourceFile::new(
        "test.stark",
        "\
struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn size(&self) -> UInt64 {
        size_of::<T>()
    }
}

fn main() {
    let w = Wrapper { value: 7i32 };
    println(w.size());
}
",
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(
        outcome.result.is_ok(),
        "with the environment installed this must now succeed: {:?}",
        outcome.result
    );
}

/// **The narrowness control.** Only the surviving-`Ty::Param` condition was reclassified. An
/// ordinary trap must still be a trap — a blanket reclassification of layout or runtime failures
/// would make `InternalInvariant` meaningless by absorbing genuine language outcomes.
#[test]
fn an_ordinary_trap_is_still_classified_as_a_trap() {
    let error = run("\
fn main() {
    let d = 0;
    println(1 / d);
}
")
    .expect_err("division by zero traps");

    assert_eq!(
        error.class,
        FailureClass::Trap,
        "a language trap must not have been absorbed into the internal class: {}",
        error.message
    );
    assert!(error.is_trap());
}
