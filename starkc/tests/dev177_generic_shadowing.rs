//! **DEV-177 — NAME-SHADOW-001 enforced: a generic parameter may not duplicate another in scope.**
//!
//! 04-Semantic-Analysis.md: "Generic parameters may not duplicate another generic parameter or an
//! item-level `Self`; a nested item introduces fresh item scopes."
//!
//! The rule existed and was unenforced. `impl<T> W<T> { fn choose<T>(..) }` both checked and RAN,
//! binding two distinct types to one name in one signature — it produced an answer at all only
//! because DEV-176 means the impl binding is never consulted.
//!
//! **Both directions are tested, and the accepting half is what keeps this honest.** A check that
//! rejected every reused name would satisfy every rejection below while breaking sibling methods,
//! unrelated functions, and any impl whose method introduces a fresh parameter. Scope here means
//! INHERITED, not merely reused.

use starkc::diag::{Diagnostic, Severity};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn diagnostics(source: &str) -> Vec<Diagnostic> {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    let mut out: Vec<Diagnostic> = parse_diags
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    out.extend(
        resolve_diags
            .into_iter()
            .filter(|d| d.severity == Severity::Error),
    );
    out.extend(
        typecheck::analyze(&hir)
            .diagnostics
            .into_iter()
            .filter(|d| d.severity == Severity::Error),
    );
    out
}

fn assert_rejected(tag: &str, source: &str) {
    let diags = diagnostics(source);
    assert!(
        diags.iter().any(|d| {
            d.code.as_deref() == Some("E0204") && d.message.contains("duplicates another generic")
        }),
        "{tag} must be rejected by NAME-SHADOW-001, got {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

fn assert_accepted(tag: &str, source: &str) {
    let diags = diagnostics(source);
    assert!(
        diags.is_empty(),
        "{tag} is legal and must not be rejected, got {:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

// ------------------------------------------------------------------------------------- REJECT --

/// The simplest form, and it was accepted: one list declaring the same name twice.
#[test]
fn one_generic_list_may_not_declare_a_name_twice() {
    assert_rejected(
        "fn<T, T>",
        "fn dup<T, T>() -> Int32 {\n    1\n}\n\nfn main() {}\n",
    );
}

/// **The reported reproducer.** The impl's `T` is in scope for the method, so the method may not
/// redeclare it.
#[test]
fn a_method_may_not_redeclare_its_impls_generic() {
    assert_rejected(
        "impl<T> + method<T>",
        "struct W<T> {\n    v: T,\n}\n\nimpl<T> W<T> {\n    fn choose<T>(self, value: T) -> T {\n        value\n    }\n}\n\nfn main() {}\n",
    );
}

/// The same rule one item kind over: a trait's generic is in scope for its default bodies.
#[test]
fn a_trait_default_may_not_redeclare_its_traits_generic() {
    assert_rejected(
        "trait<T> + default<T>",
        "trait C<T> {\n    fn choose<T>(&self, value: T) -> T {\n        value\n    }\n}\n\nfn main() {}\n",
    );
}

/// A generic named `Self` is refused by the PARSER, not by this check. Pinned so the coverage does
/// not silently move: if the parser ever accepted it, this fails and the type check must take the
/// case over.
#[test]
fn a_generic_parameter_may_not_be_named_self() {
    let diags = diagnostics("fn named<Self>() -> Int32 {\n    1\n}\n\nfn main() {}\n");
    assert!(
        diags
            .iter()
            .any(|d| d.message.contains("expected a generic parameter name")),
        "{:?}",
        diags.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

// ------------------------------------------------------------------------------------- ACCEPT --

/// **Sibling methods do not share a scope.** Two methods each introducing `U` is legal, and a check
/// built on "has this name appeared anywhere in this impl" would break it.
#[test]
fn sibling_methods_may_each_declare_the_same_new_name() {
    assert_accepted(
        "two methods declaring U",
        "struct W<T> {\n    v: T,\n}\n\nimpl<T> W<T> {\n    fn first<U>(&self, value: U) -> U {\n        value\n    }\n    fn second<U>(&self, value: U) -> U {\n        value\n    }\n}\n\nfn main() {}\n",
    );
}

/// Unrelated items reuse names freely; nothing is inherited between them.
#[test]
fn unrelated_functions_may_each_declare_t() {
    assert_accepted(
        "two free functions declaring T",
        "fn a<T>(v: T) -> T {\n    v\n}\n\nfn b<T>(v: T) -> T {\n    v\n}\n\nfn main() {}\n",
    );
}

/// A method introducing a name its impl did not declare is the ordinary generic method, and must
/// stay legal.
#[test]
fn a_method_may_introduce_a_name_its_impl_does_not_declare() {
    assert_accepted(
        "impl<T> + method<U>",
        "struct W<T> {\n    v: T,\n}\n\nimpl<T> W<T> {\n    fn m<U>(&self, value: U) -> U {\n        value\n    }\n}\n\nfn main() {}\n",
    );
}

/// The diagnostic must name the parameter and point at the earlier declaration — a bare "duplicate"
/// on a two-parameter list leaves the author guessing which is which.
#[test]
fn the_diagnostic_names_the_parameter_and_points_at_the_first_declaration() {
    let diags = diagnostics(
        "struct W<T> {\n    v: T,\n}\n\nimpl<T> W<T> {\n    fn choose<T>(self, value: T) -> T {\n        value\n    }\n}\n\nfn main() {}\n",
    );
    let found = diags
        .iter()
        .find(|d| d.code.as_deref() == Some("E0204"))
        .expect("the shadowing diagnostic");
    assert!(found.message.contains('T'), "{}", found.message);
    assert!(
        found
            .related
            .iter()
            .any(|r| r.message.contains("first declared here")),
        "the earlier declaration must be identified"
    );
}
