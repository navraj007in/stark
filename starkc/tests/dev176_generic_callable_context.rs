//! **DEV-176 — a generic callable body executes without its checker-established context.**
//!
//! The HIR interpreter installs a substitution frame only for direct calls to free functions:
//! `push_generic_frame` reads parameter names from `hir::ItemKind::Fn` and returns an empty list
//! for every other item kind, and it has one call site. Impl generics, method generics, trait
//! generics and `Self` are never bound.
//!
//! These tests pin the defect's CURRENT shape so the repair (A3c-S) has something to flip, and pin
//! its classification now, which is separable and independently valuable: the HIR interpreter is
//! the behavioural oracle, so an oracle defect classified as a language trap is one the
//! differential harness can accept as a legitimate program outcome and then pressure MIR and
//! native into reproducing.

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
    interp::run_capturing(&hir, file, &checked.tables).result
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

/// The reported defect: the identical body inside a generic impl cannot resolve `T`.
#[test]
fn a_generic_impl_method_cannot_resolve_its_type_parameter() {
    let error = run("\
struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn size(&self) -> UInt64 {
        size_of::<T>()
    }
}

fn main() {
    let w = Wrapper { value: 7 };
    println(w.size());
}
")
    .expect_err("DEV-176: an impl generic is never bound");

    assert!(
        error.message.contains("unsubstituted generic parameter"),
        "{}",
        error.message
    );
}

/// **The classification, which is the part repaired ahead of the defect itself.**
///
/// A valid program reaching the oracle without sufficient compiler substitution metadata is a
/// compiler defect. Reverting this to `RuntimeError::new` fails here and nowhere else, so the
/// classification is load-bearing rather than decorative.
#[test]
fn the_missing_generic_context_is_an_internal_invariant_not_a_trap() {
    let error = run("\
struct Wrapper<T> {
    value: T,
}

impl<T> Wrapper<T> {
    fn size(&self) -> UInt64 {
        size_of::<T>()
    }
}

fn main() {
    let w = Wrapper { value: 7 };
    println(w.size());
}
")
    .expect_err("DEV-176 fires");

    assert_eq!(
        error.class,
        FailureClass::InternalInvariant,
        "an accepted program failing for want of compiler metadata is a defect, not a language \
         outcome: {}",
        error.message
    );
    assert_eq!(error.trap_category, None);
    assert!(!error.is_trap());
    assert!(error.message.contains("DEV-176"), "{}", error.message);
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
