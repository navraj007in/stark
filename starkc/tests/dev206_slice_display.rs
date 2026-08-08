//! **DEV-206 — `Display` accepted an unsized slice place and rejected its borrowed slice view.**
//!
//! The polarity was reversed:
//!
//! ```text
//! [T]      Display accepted     WRONG — unsized, never a standalone value (§6.6)
//! &[T]     Display rejected     WRONG — the form that can actually exist at runtime
//! ```
//!
//! So `println(v[0..2])` type-checked and then had **no valid runtime representation**: the
//! representation relation refuses `Ty::Slice` at every boundary, precisely because a bare `[T]`
//! cannot cross one. The contradiction was invisible until `RepBoundary::ExpressionResult` began
//! reading published types, and it surfaced as a Gate 3 example failing on CI.
//!
//! **Two repairs were attempted and withdrawn before this one**, and both are worth recording,
//! because each would have removed the symptom by deleting a rule that was stated on purpose:
//!
//! - widening the relation so `Ty::Slice` accepts `Value::Slice` — that conflates the unsized
//!   pointee type with a runtime view, and weakens exactly the distinction DEV-121 protects;
//! - publishing `&[T]` for a range index — the indexing expression *is* a place of unsized type,
//!   and borrowing it is what produces the reference. That change made `&v[0..2]` a double
//!   reference and broke five differential cases, which is expected rather than evidence.
//!
//! The defect was in neither. It was in `Display` eligibility, and that is where it is fixed.

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Type-checks `source`, returning its error diagnostics.
fn errors(source: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    typecheck::analyze(&hir)
        .diagnostics
        .into_iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| d.message)
        .collect()
}

/// Runs `source`, requiring it to type-check and execute, and returns its output.
fn output(source: &str) -> String {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir);
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errs.is_empty(), "must type-check: {errs:?}");
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(outcome.result.is_ok(), "{:?}", outcome.result);
    outcome.output
}

const VEC: &str = "let mut v: Vec<Int32> = Vec::new(); v.push(40); v.push(2); ";

/// **The rejection half.** A bare `[T]` is unsized and is not a value, so it cannot be printed.
#[test]
fn a_bare_slice_place_is_not_displayable() {
    let errs = errors(&format!("fn main() {{ {VEC}println(v[0..2]); }}"));
    assert_eq!(
        errs.len(),
        1,
        "expected exactly one rejection, got {errs:?}"
    );
    assert!(
        errs[0].contains("cannot be printed"),
        "the rejection must be the Display one, not an unrelated error: {}",
        errs[0]
    );
    assert!(
        errs[0].contains("[Int32]"),
        "the diagnostic must name the unsized type, so the reader knows to add `&`: {}",
        errs[0]
    );
}

/// **The acceptance half**, and the value it prints.
#[test]
fn a_borrowed_slice_view_is_displayable() {
    assert_eq!(
        output(&format!("fn main() {{ {VEC}println(&v[0..2]); }}")),
        "[40, 2]\n"
    );
}

/// Through a binding, which is the spelling a reader is most likely to write.
#[test]
fn a_bound_slice_view_is_displayable() {
    assert_eq!(
        output(&format!(
            "fn main() {{ {VEC}let s = &v[0..2]; println(s); }}"
        )),
        "[40, 2]\n"
    );
}

/// **The control that stops this being "every slice is now printable".**
///
/// Element eligibility is still required and still recursive: a slice of a nominal with no
/// `Display` implementation is rejected. Without this, the repair could have been a blanket
/// acceptance wearing the shape of a fix.
#[test]
fn a_slice_of_non_display_elements_is_still_rejected() {
    let errs = errors(
        "struct X { n: Int32 } \
         fn main() { let mut v: Vec<X> = Vec::new(); v.push(X { n: 1 }); println(&v[0..1]); }",
    );
    assert_eq!(
        errs.len(),
        1,
        "expected exactly one rejection, got {errs:?}"
    );
    assert!(
        errs[0].contains("cannot be printed"),
        "a non-`Display` element must still be refused: {}",
        errs[0]
    );
}

/// A slice of a nominal that DOES implement `Display` is accepted — the element rule is a real
/// condition, not a blanket refusal of nominals.
#[test]
fn a_slice_of_display_elements_is_accepted() {
    let out = output(
        "struct X { n: Int32 } \
         impl Display for X { fn fmt(&self) -> String { String::from(\"x\") } } \
         fn main() { let mut v: Vec<X> = Vec::new(); v.push(X { n: 1 }); println(&v[0..1]); }",
    );
    assert_eq!(out, "[x]\n");
}

/// An array slice, the other base a range index accepts.
#[test]
fn a_borrowed_array_slice_is_displayable() {
    assert_eq!(
        output("fn main() { let a: [Int32; 3] = [1, 2, 3]; println(&a[0..2]); }"),
        "[1, 2]\n"
    );
}

/// **A sized array is still printable without a borrow**, which is the half of the old shared arm
/// that was correct: an array IS a value, and separating the two is the whole repair.
#[test]
fn a_sized_array_remains_displayable_by_value() {
    assert_eq!(
        output("fn main() { let a: [Int32; 3] = [1, 2, 3]; println(a); }"),
        "[1, 2, 3]\n"
    );
}
