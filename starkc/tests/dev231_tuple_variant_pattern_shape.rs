//! **DEV-231: a tuple-variant pattern must match its constructor's shape and arity.**
//!
//! Four patterns compiled clean and then silently never matched — the DEV-222 failure mode, in
//! type checking rather than resolution. With a wildcard arm each took the wildcard; without one
//! the MATCH was reported non-exhaustive, pointing away from the pattern that was actually wrong.
//!
//! Three mechanisms, all in `hir::PatKind::TupleVariant`:
//!
//! 1. A variant with no tuple fields yielded `None` and the check was skipped entirely, so
//!    `Colour::Red(_v)` on a payload-less variant passed.
//! 2. `pats.iter().zip(tys)` truncated, so `Shape::Line(_a, _b)` on a one-field variant checked
//!    the overlap and ignored the rest. Too FEW sub-patterns passed the same way.
//! 3. A resolution that was neither `Res::Variant` nor `Res::Builtin` reached neither branch and
//!    fell out as `Ty::Error` with nothing reported — `Thing(_v)` for a named-field struct,
//!    `LIMIT(_v)` for a constant.
//!
//! The arm's own comments already recorded DEV-205, an earlier instance of the same class fixed
//! one table entry at a time. The general check is what was missing.
//!
//! # Why these negative controls
//!
//! The repair adds rejections inside the arm every tuple-variant pattern goes through, so the risk
//! is refusing valid ones. Correct arity, generic variants, and the builtin constructors
//! (`Some`/`Ok`/`Err`, which take the other branch entirely) are all pinned below — as is
//! `Rec::One`, a bare path naming a struct variant, which SYN-PATTERN-001 makes legal and which an
//! over-eager repair could plausibly have caught.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

const PRELUDE: &str = "\
enum Colour { Red, Green }
enum Shape { Dot, Line(Int64) }
enum Rec { One { n: Int64 }, Two }
enum Slot<T> { Empty, Filled(T) }
struct Thing { value: Int64 }
const LIMIT: Int64 = 3i64;
";

fn errors(body: &str, tag: &str) -> Vec<String> {
    let src = format!("{PRELUDE}fn main() {{\n{body}\n}}\n");
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file);
    let mut out: Vec<String> = rd
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect();
    let checked = typecheck::analyze(&hir);
    out.extend(
        checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message)),
    );
    out
}

#[test]
fn a_unit_variant_takes_no_pattern_arguments() {
    let errs = errors(
        "    let c = Colour::Red;\n    match c { Colour::Red(_v) => {}, _ => {} }",
        "dev231_unit_variant",
    );
    assert!(
        !errs.is_empty(),
        "`Colour::Red` carries no payload, so `Colour::Red(_v)` must be rejected: {errs:?}"
    );
}

#[test]
fn too_many_sub_patterns_are_rejected() {
    let errs = errors(
        "    let s = Shape::Line(1i64);\n    match s { Shape::Line(_a, _b) => {}, _ => {} }",
        "dev231_arity_high",
    );
    assert!(
        errs.iter().any(|e| e.contains("field")),
        "the arity mismatch must be named: {errs:?}"
    );
}

#[test]
fn too_few_sub_patterns_are_rejected() {
    let errs = errors(
        "    let s = Shape::Line(1i64);\n    match s { Shape::Line() => {}, _ => {} }",
        "dev231_arity_low",
    );
    assert!(
        !errs.is_empty(),
        "`zip` truncated in both directions, so too few must be rejected as well: {errs:?}"
    );
}

#[test]
fn a_named_field_struct_is_not_a_tuple_constructor() {
    let errs = errors(
        "    let t = Thing { value: 1i64 };\n    match t { Thing(_v) => {}, _ => {} }",
        "dev231_struct_call",
    );
    assert!(
        !errs.is_empty(),
        "`Thing` has named fields and is not a tuple constructor: {errs:?}"
    );
}

#[test]
fn a_constant_is_not_a_tuple_constructor() {
    let errs = errors(
        "    let n = 3i64;\n    match n { LIMIT(_v) => {}, _ => {} }",
        "dev231_const_call",
    );
    assert!(
        !errs.is_empty(),
        "a constant is not a constructor at all: {errs:?}"
    );
}

// -- controls -----------------------------------------------------------------------------------

#[test]
fn correct_arity_still_type_checks() {
    assert_eq!(
        errors(
            "    let s = Shape::Line(1i64);\n\
             \x20   match s { Shape::Line(_n) => {}, Shape::Dot => {} }",
            "dev231_control_arity"
        ),
        Vec::<String>::new(),
        "a tuple variant bound with exactly its own arity must keep working"
    );
}

#[test]
fn a_generic_variant_still_type_checks() {
    assert_eq!(
        errors(
            "    let s: Slot<Int64> = Slot::Filled(7i64);\n\
             \x20   match s { Slot::Filled(_v) => {}, Slot::Empty => {} }",
            "dev231_control_generic"
        ),
        Vec::<String>::new(),
        "the repair must not disturb generic variant instantiation"
    );
}

#[test]
fn builtin_constructors_still_type_check() {
    assert_eq!(
        errors(
            "    let o: Option<Int64> = Some(1i64);\n\
             \x20   match o { Some(_v) => {}, None => {} }",
            "dev231_control_builtin"
        ),
        Vec::<String>::new(),
        "`Some`/`None` take the builtin branch and must be unaffected"
    );
}

/// SYN-PATTERN-001: "Multi-segment `Path` patterns always match by value", and Core v1 has no rest
/// patterns — so a bare path is the ONLY way to match a struct variant without binding its fields.
/// The audit expected this to be rejected, out of Rust intuition, and the specification disagreed.
#[test]
fn a_bare_path_still_matches_a_struct_variant() {
    assert_eq!(
        errors(
            "    let r = Rec::One { n: 1i64 };\n    match r { Rec::One => {}, Rec::Two => {} }",
            "dev231_control_bare_path"
        ),
        Vec::<String>::new(),
        "a bare path naming a struct variant is legal and must not be caught by this repair"
    );
}
