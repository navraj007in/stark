//! **Campaign A forcing property — a place-only type may not escape into a value context.**
//!
//! DEV-121 asks: *given a valid runtime type `T`, does the value `V` represent it?*
//! DEV-206 asked the question one step upstream: *should `T` have been allowed to reach a value
//! boundary at all?*
//!
//! The two are complementary, not duplicate. Publishing `[T]` for `v[0..2]` is **correct** — it is
//! a place of unsized type, and `&v[0..2]` is the legal place→reference conversion. The defect was
//! letting the bare place escape into a value context such as `println(...)`.
//!
//! ```text
//! expr_types[expr] = [T]
//!         │
//!         ├── &expr                  ✓ legal place → reference conversion
//!         ├── assignment / projection ✓ legal place use
//!         └── println(expr)          ✗ value required; [T] has no representation
//! ```
//!
//! # What this file found, and why it is a property rather than a new diagnostic
//!
//! The rule turned out to be **already enforced in every value context reachable in Core v1** — by
//! four *different* rules, each with its own diagnostic: unification for a user call, an unsized-
//! local rule for `let`, `Display` eligibility for `print`, and the interpolation check. What was
//! missing was anything tying them together, and DEV-206 is the proof that mattered: one of those
//! four had the rule backwards for two years' worth of examples, and nothing compared it against
//! the others or against the representation relation.
//!
//! A general checker diagnostic was written first and **withdrawn**: it could not fire, because
//! each specific rule rejects first. Shipping an unreachable diagnostic would have been
//! speculative machinery, and the audit's own scope rule forbids new abstractions without an
//! identified bypass. So the forcing function is this property — every value context is listed,
//! each is required to reject the place-only form and accept the reference form, and a new value
//! context that forgets is caught here rather than by a user.
//!
//! The predicate the property consults, `interp::ty_is_runtime_representable`, is **derived from
//! the canonical relation** by probing it with every `ValueKind` — not a second list of
//! runtime-representable types, which is exactly the duplicate authority this campaign removed.

use starkc::interp::ty_is_runtime_representable;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck::{self, Ty};
use std::sync::Arc;

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

const VEC: &str = "let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); ";

/// Every context that consumes an expression **by value**, with the place-only form and the
/// reference form that is its legal spelling.
///
/// A new value context belongs in this table. That is the entire point: DEV-206 was one of these
/// rows having the rule backwards, and nothing existed that would have compared it to the others.
const VALUE_CONTEXTS: &[(&str, &str, &str)] = &[
    (
        "print argument",
        "fn main() { VEC println(v[0..2]); }",
        "fn main() { VEC println(&v[0..2]); }",
    ),
    (
        "interpolation field",
        "fn main() { VEC println(f\"{v[0..2]}\"); }",
        "fn main() { VEC println(f\"{&v[0..2]}\"); }",
    ),
    (
        "let initialiser",
        "fn main() { VEC let s = v[0..2]; println(&s); }",
        "fn main() { VEC let s = &v[0..2]; println(s); }",
    ),
    (
        "user function argument",
        "fn take(x: Int32) {} fn main() { VEC take(v[0..2]); }",
        "fn take(x: &[Int32]) {} fn main() { VEC take(&v[0..2]); }",
    ),
    (
        "aggregate field",
        "struct H { s: Int32 } fn main() { VEC let h = H { s: v[0..2] }; }",
        "struct H { s: Int32 } fn main() { VEC let h = H { s: v[0] }; }",
    ),
];

fn expand(template: &str) -> String {
    template.replace("VEC ", VEC)
}

/// **The property.** No value context accepts a place-only type.
#[test]
fn no_value_context_accepts_a_place_only_type() {
    for (name, rejected, _) in VALUE_CONTEXTS {
        let errs = errors(&expand(rejected));
        assert!(
            !errs.is_empty(),
            "{name}: a place-only `[Int32]` was ACCEPTED as a value. It has no runtime \
             representation, so this program would reach a boundary that must refuse it — or, \
             worse, a renderer that improvises. Every other value context rejects it."
        );
    }
}

/// **The control.** The reference form — the legal place→reference conversion — is accepted
/// everywhere. Without this the property above would be satisfied by rejecting slices entirely.
#[test]
fn every_value_context_accepts_the_reference_form() {
    for (name, _, accepted) in VALUE_CONTEXTS {
        let errs = errors(&expand(accepted));
        assert!(
            errs.is_empty(),
            "{name}: the reference form must be accepted, or the rule is 'no slices' rather than \
             'a slice is observed through a reference'. Got {errs:?}"
        );
    }
}

/// **The class, not just slices.** `str` is the other unsized type §6.6 names, and the same rule
/// governs it: `String` and `&str` are values, bare `str` is not.
#[test]
fn the_rule_is_about_unsized_types_not_only_slices() {
    let copy_items = std::collections::HashSet::new();
    assert!(
        !ty_is_runtime_representable(&Ty::Primitive(starkc::ast::Primitive::Str), &copy_items),
        "bare `str` is unsized and is not a value"
    );
    assert!(
        ty_is_runtime_representable(
            &Ty::Ref {
                mutable: false,
                inner: Box::new(Ty::Primitive(starkc::ast::Primitive::Str)),
            },
            &copy_items
        ),
        "`&str` is the value form"
    );
}

/// The predicate is derived from the relation, so its answers must line up with what the relation
/// permits — including the positive direction, or "representable" could be a constant `false`.
#[test]
fn the_predicate_agrees_with_the_relation_in_both_directions() {
    let copy_items = std::collections::HashSet::new();
    let elem = || Box::new(Ty::Primitive(starkc::ast::Primitive::Int32));

    // Place-only: no representation satisfies it.
    assert!(!ty_is_runtime_representable(
        &Ty::Slice(elem()),
        &copy_items
    ));

    // Value forms: a representation exists.
    assert!(ty_is_runtime_representable(
        &Ty::Ref {
            mutable: false,
            inner: Box::new(Ty::Slice(elem())),
        },
        &copy_items
    ));
    assert!(ty_is_runtime_representable(
        &Ty::Array(elem(), 3),
        &copy_items
    ));
    assert!(ty_is_runtime_representable(
        &Ty::Primitive(starkc::ast::Primitive::Int32),
        &copy_items
    ));
    assert!(ty_is_runtime_representable(
        &Ty::Tuple(vec![Ty::Primitive(starkc::ast::Primitive::Bool)]),
        &copy_items
    ));
}
