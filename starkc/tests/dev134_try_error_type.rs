//! **DEV-134: `?` must relate its operand to the enclosing function's return type.**
//!
//! Before this repair the `Try` arm asked two INDEPENDENT questions — "is the enclosing return
//! type `Result` or `Option`?" and "is the operand `Result` or `Option`?" — and never related the
//! two. So both of these were accepted:
//!
//! - `Result<_, Low>?` inside a function returning `Result<_, High>`, with no `From` impl
//!   required, present, or applied;
//! - `Option<_>?` inside a function returning `Result<_, _>`, and the reverse.
//!
//! Both propagate a value whose variant tag belongs to a different type than the one the caller
//! matches on. That is type confusion, not a diagnostic gap: the caller's `match` IS exhaustive
//! over its own type and still falls through, which the HIR oracle reports as "non-exhaustive
//! match reached" — a symptom several layers downstream of the cause.
//!
//! # The ruling this encodes
//!
//! `?` requires EXACT error-type compatibility. Implicit `From`-based propagation is deliberately
//! NOT part of this repair: `03-Type-System.md` does not scope a conversion at the propagation
//! site, so adding one would be new semantics rather than a repair. Rejection is the conservative
//! half. Case `from_impl_present_still_rejected` pins that specifically, so a future session
//! cannot mistake the absence of conversion for an oversight.
//!
//! # Why these negative controls
//!
//! The risk here is OVER-rejection, not under-rejection — a `?` check that is too eager breaks
//! every correct propagation in the compiler's own provider layer and in ten first-party
//! packages. So the must-pass set is deliberately larger than the must-reject set, and covers
//! generic error types, nested/helper propagation, and chained `?` in one body.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Returns the first error diagnostic as `"CODE message"`, or `None` when the program checks.
fn check(src: &str, tag: &str) -> Option<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
}

fn expect_reject(src: &str, tag: &str) -> String {
    match check(src, tag) {
        Some(diagnostic) => {
            assert!(
                diagnostic.starts_with("E0006"),
                "{tag}: expected E0006, got: {diagnostic}"
            );
            diagnostic
        }
        None => panic!("{tag}: expected rejection, but the program checked clean"),
    }
}

fn expect_accept(src: &str, tag: &str) {
    if let Some(diagnostic) = check(src, tag) {
        panic!("{tag}: expected acceptance, got: {diagnostic}");
    }
}

// ------------------------------------------------------------------ must reject --

/// The CD-334 reproducer: two different fieldless enums, no `From` impl anywhere.
#[test]
fn different_fieldless_enums_are_rejected() {
    let diagnostic = expect_reject(
        "enum Low { Bad }\n\
         enum High { Other }\n\
         fn low() -> Result<Int32, Low> { Err(Low::Bad) }\n\
         fn viaq() -> Result<Int32, High> { let value = low()?; Ok(value) }\n\
         fn main() { match viaq() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "fieldless",
    );
    assert!(
        diagnostic.contains("Low") && diagnostic.contains("High"),
        "the diagnostic must name BOTH error types, or it cannot be acted on: {diagnostic}"
    );
}

/// Payload-carrying variants: the layouts differ, so this is the worse-consequence shape.
#[test]
fn different_payload_carrying_enums_are_rejected() {
    expect_reject(
        "enum Low { Bad(Int32) }\n\
         enum High { Other(String) }\n\
         fn low() -> Result<Int32, Low> { Err(Low::Bad(1)) }\n\
         fn viaq() -> Result<Int32, High> { let value = low()?; Ok(value) }\n\
         fn main() { match viaq() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "payload",
    );
}

/// **An `impl From<Low> for High` does NOT license the propagation.** This is the ruling, pinned:
/// the repair rejects rather than converting, and the presence of a conversion changes nothing.
#[test]
fn from_impl_present_still_rejected() {
    expect_reject(
        "enum Low { Bad }\n\
         enum High { Wrapped(Low) }\n\
         impl From<Low> for High { fn from(l: Low) -> High { High::Wrapped(l) } }\n\
         fn low() -> Result<Int32, Low> { Err(Low::Bad) }\n\
         fn viaq() -> Result<Int32, High> { let value = low()?; Ok(value) }\n\
         fn main() { match viaq() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "fromimpl",
    );
}

/// `Option` propagated out of a `Result` function. The same mechanism and the same repair —
/// the constructor half of the defect, which is easy to overlook because the operand and the
/// return type are each individually `?`-capable.
#[test]
fn option_propagated_from_a_result_function_is_rejected() {
    let diagnostic = expect_reject(
        "enum E { A }\n\
         fn opt() -> Option<Int32> { None }\n\
         fn f() -> Result<Int32, E> { let v = opt()?; Ok(v) }\n\
         fn main() { match f() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "optinresult",
    );
    assert!(
        diagnostic.contains("Option") && diagnostic.contains("Result"),
        "both constructors must be named: {diagnostic}"
    );
}

/// ...and the reverse direction.
#[test]
fn result_propagated_from_an_option_function_is_rejected() {
    expect_reject(
        "enum E { A }\n\
         fn res() -> Result<Int32, E> { Err(E::A) }\n\
         fn g() -> Option<Int32> { let v = res()?; Some(v) }\n\
         fn main() { match g() { Some(v) => println(v), None => println(\"n\") } }\n",
        "resultinoption",
    );
}

/// The mismatch inside a GENERIC body, where the error types are concrete but the success type
/// is a parameter. The check must see through the generic to the error position.
#[test]
fn mismatch_inside_a_generic_body_is_rejected() {
    expect_reject(
        "enum Low { Bad }\n\
         enum High { Other }\n\
         fn low<T>(value: T) -> Result<T, Low> { Ok(value) }\n\
         fn viaq<T>(value: T) -> Result<T, High> { let inner = low(value)?; Ok(inner) }\n\
         fn main() { match viaq(1) { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "genericbody",
    );
}

/// A helper one level down: the `?` is not in the function whose signature is wrong at a glance.
#[test]
fn mismatch_through_a_nested_helper_is_rejected() {
    expect_reject(
        "enum Low { Bad }\n\
         enum High { Other }\n\
         fn deepest() -> Result<Int32, Low> { Err(Low::Bad) }\n\
         fn middle() -> Result<Int32, Low> { let v = deepest()?; Ok(v) }\n\
         fn outer() -> Result<Int32, High> { let v = middle()?; Ok(v) }\n\
         fn main() { match outer() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "nested",
    );
}

// -------------------------------------------------------------------- must pass --

/// Identical error types — the overwhelmingly common case, and the one that must not regress.
#[test]
fn identical_error_types_are_accepted() {
    expect_accept(
        "enum E { A }\n\
         fn low() -> Result<Int32, E> { Err(E::A) }\n\
         fn same() -> Result<Int32, E> { let v = low()?; Ok(v) }\n\
         fn main() { match same() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "identical",
    );
}

/// `Option` into `Option`. There is no payload on `None`, so there is nothing to relate beyond
/// the constructor.
#[test]
fn option_into_option_is_accepted() {
    expect_accept(
        "fn opt() -> Option<Int32> { None }\n\
         fn same() -> Option<Int32> { let v = opt()?; Some(v) }\n\
         fn main() { match same() { Some(v) => println(v), None => println(\"n\") } }\n",
        "optoption",
    );
}

/// **Different SUCCESS types are legal.** `?` relates the error position only; the success value
/// is extracted and may be reshaped freely by the surrounding expression. A check that compared
/// whole types instead of error types would wrongly reject this.
#[test]
fn different_success_types_are_accepted() {
    expect_accept(
        "enum E { A }\n\
         fn small() -> Result<Int32, E> { Ok(1) }\n\
         fn widen() -> Result<Int64, E> { let v = small()?; Ok(v as Int64) }\n\
         fn main() { match widen() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "successtypes",
    );
}

/// A GENERIC error type that is identical on both sides. Equality must be structural, not
/// nominal-only, or every generic propagation breaks.
#[test]
fn identical_generic_error_types_are_accepted() {
    expect_accept(
        "enum E<T> { A(T) }\n\
         fn low() -> Result<Int32, E<Int32>> { Ok(1) }\n\
         fn same() -> Result<Int32, E<Int32>> { let v = low()?; Ok(v) }\n\
         fn main() { match same() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "genericerror",
    );
}

/// Several `?` in one body, all matching. Chained propagation is the shape the REST workload and
/// the first-party packages actually use.
#[test]
fn chained_matching_propagation_is_accepted() {
    expect_accept(
        "enum E { A }\n\
         fn a() -> Result<Int32, E> { Ok(1) }\n\
         fn b() -> Result<Int32, E> { Ok(2) }\n\
         fn c() -> Result<Int32, E> { let x = a()?; let y = b()?; Ok(x + y) }\n\
         fn main() { match c() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "chained",
    );
}

/// A `String` error type, which is `Ty::Core` rather than `Ty::Enum` — a different equality path.
#[test]
fn identical_core_error_types_are_accepted() {
    expect_accept(
        "fn low() -> Result<Int32, String> { Ok(1) }\n\
         fn same() -> Result<Int32, String> { let v = low()?; Ok(v) }\n\
         fn main() { match same() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "coreerror",
    );
}

/// The generic error type is the FUNCTION's own parameter on both sides.
#[test]
fn error_type_as_a_generic_parameter_is_accepted() {
    expect_accept(
        "fn low<E>(e: E) -> Result<Int32, E> { Err(e) }\n\
         fn same<E>(e: E) -> Result<Int32, E> { let v = low(e)?; Ok(v) }\n\
         fn main() { match same(1) { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "paramerror",
    );
}

/// **A pre-existing E0006 must not be doubled.** A `?` in a function returning neither `Result`
/// nor `Option` was already reported; the new check must stay silent there rather than adding a
/// second diagnostic for the same mistake.
#[test]
fn non_result_return_reports_once_not_twice() {
    let file = Arc::new(SourceFile::new(
        "single.stark".to_string(),
        "enum E { A }\n\
         fn low() -> Result<Int32, E> { Err(E::A) }\n\
         fn bad() -> Int32 { let v = low()?; v }\n\
         fn main() { println(bad()); }\n"
            .to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir);
    let try_errors = checked
        .diagnostics
        .iter()
        .filter(|d| d.code.as_deref() == Some("E0006"))
        .count();
    assert_eq!(
        try_errors, 1,
        "expected exactly one E0006 for a `?` in a non-Result function, got {try_errors}: {:?}",
        checked.diagnostics
    );
}

/// The spelling the diagnostic advises must itself compile. A help message that recommends an
/// uncompilable shape is worse than no help at all.
#[test]
fn the_advised_explicit_conversion_compiles() {
    expect_accept(
        "enum Low { Bad }\n\
         enum High { Wrapped(Low) }\n\
         impl From<Low> for High { fn from(l: Low) -> High { High::Wrapped(l) } }\n\
         fn low() -> Result<Int32, Low> { Err(Low::Bad) }\n\
         fn viaq() -> Result<Int32, High> {\n\
         \x20   let value: Int32;\n\
         \x20   match low() {\n\
         \x20       Ok(inner) => { value = inner; }\n\
         \x20       Err(inner) => { return Err(High::from(inner)); }\n\
         \x20   }\n\
         \x20   Ok(value)\n\
         }\n\
         fn main() { match viaq() { Ok(v) => println(v), Err(_e) => println(\"e\") } }\n",
        "advised",
    );
}
