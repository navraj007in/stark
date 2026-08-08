//! **Campaign A final audit §7 — the published types a boundary reads must be real types.**
//!
//! Every wired boundary reads a checker-published type and compares a value against it. That is
//! only meaningful if the published type is an *answer*. `Ty::Error` is the checker's marker for
//! "I could not decide", and a program that type-checks with **no diagnostics** while publishing
//! `Ty::Error` for a reachable expression is a checker defect in the most literal sense: it has
//! accepted the program and recorded that it does not know what one of its expressions means.
//!
//! DEV-205 was exactly that, and it is why this file exists rather than a test of one repaired
//! case. `IOError::Other(msg)` was missing from the builtin-variant arm of the pattern checker, so
//! its sub-pattern was never checked: the binding got no `local_types` entry, and every use of it
//! was typed `Ty::Error`. The program ran and printed the right answer. Nothing found it for as
//! long as nothing read the tables — the DEV-121 shape, relocated into the checker.
//!
//! The forcing property below is general, not a regression pin: **a clean-checking program
//! publishes no `Ty::Error`**. Any future construct that the checker accepts without understanding
//! fails here, whether or not it happens to execute correctly.

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck::{self, Ty};
use std::sync::Arc;

/// Type-checks `source`, requiring no error diagnostics, and returns the published tables.
fn tables_of(source: &str) -> typecheck::TypeTables {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir);
    let errors: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(errors.is_empty(), "typecheck: {errors:?}");
    checked.tables
}

/// Whether `ty` mentions `Ty::Error` anywhere, including inside aggregates and references — an
/// error nested in `Option<?>` is the same defect as a bare one.
fn mentions_error(ty: &Ty) -> bool {
    match ty {
        Ty::Error => true,
        Ty::Ref { inner, .. } => mentions_error(inner),
        Ty::Struct(_, args) | Ty::Enum(_, args) | Ty::Core(_, args) => {
            args.iter().any(mentions_error)
        }
        Ty::Tuple(elems) => elems.iter().any(mentions_error),
        Ty::Array(elem, _) | Ty::Slice(elem) | Ty::Range(elem) => mentions_error(elem),
        Ty::Fn { params, ret } => params.iter().any(mentions_error) || mentions_error(ret),
        _ => false,
    }
}

/// The witnesses. Deliberately ordinary Core v1 programs, one per construct family that carries a
/// payload the checker has to reason about — the families where "accepted but not understood" can
/// hide behind a correct answer.
const WITNESSES: &[(&str, &str)] = &[
    (
        "IOError::Other payload binding — DEV-205",
        "fn main() { let e = IOError::Other(String::from(\"custom\")); \
         match e { IOError::Other(msg) => println(msg), _ => println(\"other\") } }",
    ),
    (
        "IOError unit variants",
        "fn main() { let e = IOError::NotFound; \
         match e { IOError::NotFound => println(\"nf\"), _ => println(\"other\") } }",
    ),
    (
        "Option payload binding",
        "fn main() { let o: Option<Int32> = Some(7); \
         match o { Some(n) => println(n), None => println(0) } }",
    ),
    (
        "Result payload bindings, both sides",
        "fn main() { let r: Result<Int32, String> = Ok(7); \
         match r { Ok(n) => println(n), Err(m) => println(m) } }",
    ),
    (
        "user enum tuple variant",
        "enum E { A(Int32), B } fn main() { let e = E::A(3); \
         match e { E::A(n) => println(n), E::B => println(0) } }",
    ),
    (
        "user enum struct variant",
        "enum E { A { n: Int32 }, B } fn main() { let e = E::A { n: 3 }; \
         match e { E::A { n } => println(n), E::B => println(0) } }",
    ),
    (
        // The BORROWED form. This witness was written while `println(v[0..2])` was still
        // accepted, and DEV-206's repair — which rejects the unsized place — is what made it
        // stale. Kept as a witness rather than deleted: a range index is exactly the construct
        // whose published types this file exists to check, and `&[Int32]` is its valid spelling.
        "borrowed range index producing a slice view — DEV-206",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
         println(&v[0..2]); }",
    ),
    (
        "generic function and a function value",
        "fn width<T>(x: T) -> Int32 { size_of::<T>() as Int32 } \
         fn main() { let f: fn(Float64) -> Int32 = width; println(f(1.5)); }",
    ),
];

/// **The general property.** A program the checker accepts must not carry `Ty::Error` in any
/// published expression type.
#[test]
fn a_clean_program_publishes_no_error_type_for_any_expression() {
    for (name, source) in WITNESSES {
        let tables = tables_of(source);
        let bad: Vec<_> = tables
            .expr_types
            .iter()
            .filter(|(_, ty)| mentions_error(ty))
            .collect();
        assert!(
            bad.is_empty(),
            "{name}: the checker accepted this program with no diagnostics while publishing \
             `Ty::Error` for {} expression(s). Every boundary that reads such a type compares a \
             value against a non-answer — and the program still runs, which is why this must be \
             caught here rather than by a failing execution.",
            bad.len()
        );
    }
}

/// The same property for locals. A binding with no published type is what made the typed-local
/// funnel's permissive lookup survivable for so long; a binding published as `Ty::Error` is the
/// same defect wearing an answer's clothes.
#[test]
fn a_clean_program_publishes_no_error_type_for_any_local() {
    for (name, source) in WITNESSES {
        let tables = tables_of(source);
        let bad: Vec<_> = tables
            .local_types
            .iter()
            .filter(|(_, ty)| mentions_error(ty))
            .collect();
        assert!(
            bad.is_empty(),
            "{name}: {} local(s) published `Ty::Error`",
            bad.len()
        );
    }
}

/// **The control.** `mentions_error` must actually be able to see an error — including a nested
/// one, which is the case a shallow check would miss.
#[test]
fn the_error_detector_detects_errors() {
    assert!(mentions_error(&Ty::Error));
    assert!(mentions_error(&Ty::Slice(Box::new(Ty::Error))));
    assert!(mentions_error(&Ty::Ref {
        mutable: false,
        inner: Box::new(Ty::Tuple(vec![Ty::Error])),
    }));
    assert!(!mentions_error(&Ty::Primitive(
        starkc::ast::Primitive::Int32
    )));
}
