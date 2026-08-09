//! **WP-VALUE-REP-TOTAL A3b — every executable callable body has a published signature.**
//!
//! `TypeTables::fn_types` is keyed by `ItemId` and covers free functions only, because `hir::FnDef`
//! carries no `ItemId`. An inherent method, a trait implementation method, an associated function,
//! `Drop::drop` and a trait default body therefore all had signatures the checker computed and
//! nothing could look up — which is why `Callable::ret` could only ever have been `None` for them.
//!
//! `callable_types` is keyed by `BlockId`, the identity execution actually has: a `Callable` already
//! carries its selected body, so no name lookup or reconstructed identity is involved.
//!
//! **The coverage test is exact-set, not by example.** A test that checked "the method is present"
//! would pass against a table missing three of six classes. The set of bodies the HIR declares and
//! the set of keys the checker published must be equal in both directions — a missing body is an
//! unvalidatable callable, and an extra key is a signature attached to something that never runs.

use starkc::hir::{self, BlockId, Hir};
use starkc::options::LanguageOptions;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Every class `check_fn_def` sees, in one program, plus a bodyless trait method that must NOT be
/// published.
const EVERY_CALLABLE_CLASS: &str = "\
struct Res {
    id: Int32,
}

trait Shape {
    // Bodyless: no `BlockId`, so structurally impossible to publish.
    fn area(&self) -> Int32;

    // A default body IS executable and must be published.
    fn describe(&self) -> Int32 {
        7
    }
}

impl Shape for Res {
    fn area(&self) -> Int32 {
        self.id
    }
}

impl Res {
    // An inherent method.
    fn value(&self) -> Int32 {
        self.id
    }

    // An associated function: no receiver.
    fn make(id: Int32) -> Res {
        Res { id: id }
    }
}

impl Drop for Res {
    fn drop(&mut self) {
        println(\"gone\");
    }
}

fn free(n: Int32) -> Int32 {
    n
}

fn main() {
    let r = Res::make(1);
    println(free(r.value()));
    println(r.area());
    println(r.describe());
}
";

fn analyse(source: &str) -> (Hir, typecheck::TypeCheckResult) {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir);
    let _ = LanguageOptions::CORE;
    (hir, checked)
}

/// Every body the HIR declares as executable, gathered from the same three item shapes
/// `check_fn_def` is reached through.
fn executable_bodies(hir: &Hir) -> BTreeSet<u32> {
    let mut bodies = BTreeSet::new();
    for index in 0..hir.items.len() {
        match &hir.item(starkc::hir::ItemId(index as u32)).kind {
            hir::ItemKind::Fn(def) => {
                bodies.insert(def.body.0);
            }
            hir::ItemKind::Impl { items, .. } => {
                for item in items {
                    if let hir::ImplItem::Fn { def, .. } = item {
                        bodies.insert(def.body.0);
                    }
                }
            }
            hir::ItemKind::Trait { items, .. } => {
                for item in items {
                    // `body: Some(_)` is what excludes a bodyless declaration — structurally, not
                    // by a filter that could be forgotten.
                    if let hir::TraitItem::Method {
                        body: Some(body), ..
                    } = item
                    {
                        bodies.insert(body.0);
                    }
                }
            }
            _ => {}
        }
    }
    bodies
}

#[test]
fn every_executable_body_has_exactly_one_published_signature() {
    let (hir, checked) = analyse(EVERY_CALLABLE_CLASS);
    let expected = executable_bodies(&hir);
    let actual: BTreeSet<u32> = checked
        .tables
        .callable_types
        .keys()
        .map(|body| body.0)
        .collect();

    let missing: Vec<u32> = expected.difference(&actual).copied().collect();
    let extra: Vec<u32> = actual.difference(&expected).copied().collect();
    assert!(
        missing.is_empty(),
        "bodies with no published signature (unvalidatable callables): {missing:?}"
    );
    assert!(
        extra.is_empty(),
        "published signatures for bodies that are not executable: {extra:?}"
    );
    assert_eq!(expected, actual);
}

/// The bodyless trait method must be absent. Asserting the count of published entries against the
/// count of executable bodies is what proves it: if `area`'s declaration were somehow published it
/// would appear as an `extra` above, and if `describe`'s default were dropped it would appear as
/// `missing`.
#[test]
fn a_bodyless_trait_declaration_publishes_nothing() {
    let (hir, checked) = analyse(EVERY_CALLABLE_CLASS);
    let bodies = executable_bodies(&hir);
    assert_eq!(
        checked.tables.callable_types.len(),
        bodies.len(),
        "a bodyless declaration has no BlockId and cannot be published"
    );
    // Six executable bodies: free, inherent method, associated fn, trait impl method, Drop::drop,
    // trait default, plus main.
    assert_eq!(
        bodies.len(),
        7,
        "the fixture covers every class exactly once"
    );
}

/// The published signature must be the CHECKER's, not a re-derivation: receiver, parameters and
/// return together, for a method whose three parts differ from each other.
#[test]
fn a_methods_receiver_parameters_and_return_are_all_published() {
    let (hir, checked) = analyse(
        "\
struct Counter {
    n: Int32,
}

impl Counter {
    fn add(&mut self, delta: Int32) -> Int32 {
        self.n = self.n + delta;
        self.n
    }
}

fn main() {}
",
    );
    let body = hir
        .items
        .iter()
        .enumerate()
        .find_map(
            |(index, _)| match &hir.item(starkc::hir::ItemId(index as u32)).kind {
                hir::ItemKind::Impl { items, .. } => items.iter().find_map(|item| match item {
                    hir::ImplItem::Fn { def, .. } => Some(def.body),
                    _ => None,
                }),
                _ => None,
            },
        )
        .expect("the fixture declares an impl method");

    let sig = checked
        .tables
        .callable_types
        .get(&BlockId(body.0))
        .expect("an inherent method must be published");
    assert!(
        matches!(
            sig.receiver,
            Some(starkc::typecheck::Ty::Ref { mutable: true, .. })
        ),
        "a `&mut self` receiver must be published as a mutable reference: {:?}",
        sig.receiver
    );
    assert_eq!(sig.params.len(), 1, "the receiver is not a parameter");
    assert!(
        matches!(sig.ret, starkc::typecheck::Ty::Primitive(_)),
        "{:?}",
        sig.ret
    );
}

/// An associated function has no receiver. Publishing `Some(..)` for one would make A4 look for a
/// receiver value that never arrives.
#[test]
fn an_associated_function_publishes_no_receiver() {
    let (hir, checked) = analyse(
        "\
struct Res {
    id: Int32,
}

impl Res {
    fn make(id: Int32) -> Res {
        Res { id: id }
    }
}

fn main() {}
",
    );
    let body = (0..hir.items.len())
        .find_map(
            |index| match &hir.item(starkc::hir::ItemId(index as u32)).kind {
                hir::ItemKind::Impl { items, .. } => items.iter().find_map(|item| match item {
                    hir::ImplItem::Fn { def, .. } => Some(def.body),
                    _ => None,
                }),
                _ => None,
            },
        )
        .expect("the fixture declares an associated function");

    let sig = checked
        .tables
        .callable_types
        .get(&BlockId(body.0))
        .expect("an associated function must be published");
    assert!(sig.receiver.is_none());
    assert_eq!(sig.params.len(), 1);
}
