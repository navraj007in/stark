//! **DEV-139: a method body sees the impl's generic bounds, not only its own.**
//!
//! `impl<T: Ord> Pair<T> { fn larger(&self) -> &T { if self.a > self.b { .. } } }` was refused
//! E0500 "type 'T' does not satisfy operator trait 'Ord'", while the identical comparison in a
//! free `fn largest<T: Ord>` was accepted. The bound was declared; nothing looked at it.
//!
//! # Mechanism
//!
//! WP-C6.2b-F5 had already brought impl-head generics into scope for method bodies, via
//! `current_impl_generics`. The two bound questions — operator desugaring
//! (`ty_satisfies_operator_bound`) and trait-bound satisfaction (`satisfies_bound`) — each kept
//! their own copy of the parameter lookup, and both consulted `current_fn_generics` ALONE. The
//! repair assembles the environment in ONE place, `param_declares_bound`, which both now call.
//! Writing it once is deliberate: DEV-128 and DEV-130 are both "the rule was written twice and
//! the copies drifted", and this was already two copies that agreed only by coincidence.
//!
//! # Why the negatives matter
//!
//! Making an environment WIDER risks discharging obligations that were never declared. The
//! must-reject half pins that an operator with no bound, the wrong bound, and a bound sitting on
//! a different parameter are all still refused.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn check(src: &str, tag: &str) -> Option<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
}

fn expect_accept(src: &str, tag: &str) {
    if let Some(diagnostic) = check(src, tag) {
        panic!("{tag}: expected acceptance, got: {diagnostic}");
    }
}

fn expect_reject(src: &str, tag: &str) -> String {
    match check(src, tag) {
        Some(diagnostic) => {
            assert!(
                diagnostic.starts_with("E0500"),
                "{tag}: expected E0500, got: {diagnostic}"
            );
            diagnostic
        }
        None => panic!("{tag}: expected rejection, but the program checked clean"),
    }
}

fn run(src: &str, tag: &str) -> String {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error),
        "{tag}: check: {:?}",
        checked.diagnostics
    );
    match starkc::interp::run(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    ) {
        Ok(execution) => execution.output,
        Err(error) => panic!("{tag}: runtime error: {}", error.message),
    }
}

const PAIR: &str = "struct Pair<T> { a: T, b: T }\n";

// -------------------------------------------------------------------- must pass --

/// The CD-334 reproducer: `Ord` from the impl head, used by `>`.
#[test]
fn an_impl_level_ord_bound_reaches_the_greater_than_operator() {
    expect_accept(
        &format!(
            "{PAIR}\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn larger(&self) -> &T {{\n\
             \x20       if self.a > self.b {{ &self.a }} else {{ &self.b }}\n\
             \x20   }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 4, b: 9 }}; println(*p.larger()); }}\n"
        ),
        "implord",
    );
}

/// ...and the method must agree with the free function that was always accepted.
#[test]
fn the_method_and_the_free_function_agree() {
    let out = run(
        &format!(
            "{PAIR}\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn larger(&self) -> &T {{\n\
             \x20       if self.a > self.b {{ &self.a }} else {{ &self.b }}\n\
             \x20   }}\n\
             }}\n\
             fn largest<T: Ord>(a: T, b: T) -> T {{ if a > b {{ a }} else {{ b }} }}\n\
             fn main() {{\n\
             \x20   let p = Pair {{ a: 4, b: 9 }};\n\
             \x20   println(*p.larger());\n\
             \x20   println(largest(4, 9));\n\
             }}\n"
        ),
        "agree",
    );
    assert_eq!(
        out.trim(),
        "9\n9",
        "both routes must produce the same answer"
    );
}

/// All four ordering operators, not just `>`.
#[test]
fn all_ordering_operators_see_the_impl_bound() {
    expect_accept(
        &format!(
            "{PAIR}\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn relations(&self) -> Bool {{\n\
             \x20       self.a < self.b && self.a <= self.b\n\
             \x20           && self.b > self.a && self.b >= self.a\n\
             \x20   }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.relations()); }}\n"
        ),
        "allord",
    );
}

/// `Eq` from the impl head, used by `==` and `!=`.
#[test]
fn an_impl_level_eq_bound_reaches_the_equality_operators() {
    expect_accept(
        &format!(
            "{PAIR}\
             impl<T: Eq> Pair<T> {{\n\
             \x20   fn same(&self) -> Bool {{ self.a == self.b }}\n\
             \x20   fn different(&self) -> Bool {{ self.a != self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.same()); println(p.different()); }}\n"
        ),
        "impleq",
    );
}

/// `Num` from the impl head, used by arithmetic.
#[test]
fn an_impl_level_num_bound_reaches_arithmetic() {
    expect_accept(
        &format!(
            "{PAIR}\
             impl<T: Num> Pair<T> {{\n\
             \x20   fn total(&self) -> T {{ self.a + self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 3, b: 4 }}; println(p.total()); }}\n"
        ),
        "implnum",
    );
}

/// A TRAIT-bound obligation, not an operator: the other of the two lookups that now share
/// `param_declares_bound`. `largest` requires `T: Ord`, and the impl's bound must discharge it.
#[test]
fn an_impl_level_bound_discharges_a_callee_obligation() {
    expect_accept(
        &format!(
            "{PAIR}\
             fn largest<T: Ord>(a: T, b: T) -> T {{ if a > b {{ a }} else {{ b }} }}\n\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn best(self) -> T {{ largest(self.a, self.b) }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 4, b: 9 }}; println(p.best()); }}\n"
        ),
        "calleeobligation",
    );
}

/// **The impl and the method each contribute a bound**, and both must be visible at once.
#[test]
fn impl_and_method_bounds_are_both_in_scope() {
    expect_accept(
        &format!(
            "{PAIR}\
             fn largest<X: Ord>(a: X, b: X) -> X {{ if a > b {{ a }} else {{ b }} }}\n\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn pick<U: Num>(self, extra: U) -> U {{\n\
             \x20       let _best = largest(self.a, self.b);\n\
             \x20       extra + extra\n\
             \x20   }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 4, b: 9 }}; println(p.pick(5)); }}\n"
        ),
        "bothlevels",
    );
}

/// A nested generic nominal type instantiated with the bounded parameter.
#[test]
fn a_nested_generic_nominal_sees_the_impl_bound() {
    expect_accept(
        "struct Wrapper<T> { value: T }\n\
         struct Holder<T> { inner: Wrapper<T>, other: Wrapper<T> }\n\
         impl<T: Ord> Holder<T> {\n\
         \x20   fn ordered(&self) -> Bool { self.inner.value < self.other.value }\n\
         }\n\
         fn main() {\n\
         \x20   let h = Holder {\n\
         \x20       inner: Wrapper { value: 1 },\n\
         \x20       other: Wrapper { value: 2 },\n\
         \x20   };\n\
         \x20   println(h.ordered());\n\
         }\n",
        "nestednominal",
    );
}

/// A TRAIT impl's generics, not just an inherent impl's — the environment is installed for both.
#[test]
fn a_trait_impl_method_sees_the_impl_bound() {
    expect_accept(
        &format!(
            "{PAIR}\
             trait Best {{ fn best(&self) -> Bool; }}\n\
             impl<T: Ord> Best for Pair<T> {{\n\
             \x20   fn best(&self) -> Bool {{ self.a < self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.best()); }}\n"
        ),
        "traitimpl",
    );
}

/// Two bounds on one impl parameter; both are reachable.
#[test]
fn multiple_bounds_on_one_impl_parameter_are_all_visible() {
    expect_accept(
        &format!(
            "{PAIR}\
             impl<T: Ord + Num> Pair<T> {{\n\
             \x20   fn spread(&self) -> Bool {{ self.a < self.b }}\n\
             \x20   fn total(&self) -> T {{ self.a + self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.spread()); println(p.total()); }}\n"
        ),
        "multibound",
    );
}

// ------------------------------------------------------------------ must reject --

/// **No bound at all**: widening the environment must not invent obligations.
#[test]
fn an_operator_without_any_bound_is_still_rejected() {
    let diagnostic = expect_reject(
        &format!(
            "{PAIR}\
             impl<T> Pair<T> {{\n\
             \x20   fn larger(&self) -> Bool {{ self.a > self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.larger()); }}\n"
        ),
        "nobound",
    );
    assert!(
        diagnostic.contains("Ord"),
        "the diagnostic must name the missing bound: {diagnostic}"
    );
}

/// **The wrong bound**: `Eq` does not license ordering.
#[test]
fn an_eq_bound_does_not_license_ordering() {
    expect_reject(
        &format!(
            "{PAIR}\
             impl<T: Eq> Pair<T> {{\n\
             \x20   fn larger(&self) -> Bool {{ self.a > self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.larger()); }}\n"
        ),
        "wrongbound",
    );
}

/// ...and `Ord` does not license arithmetic.
#[test]
fn an_ord_bound_does_not_license_arithmetic() {
    expect_reject(
        &format!(
            "{PAIR}\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn total(&self) -> T {{ self.a + self.b }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.total()); }}\n"
        ),
        "ordnotnum",
    );
}

/// **The bound is on a DIFFERENT parameter.** This is the control that pins the lookup still
/// matches on parameter name rather than merely finding some bound somewhere in scope.
#[test]
fn a_bound_on_an_unrelated_parameter_does_not_apply() {
    expect_reject(
        "struct Two<A, B> { first: A, second: B }\n\
         impl<A: Ord, B> Two<A, B> {\n\
         \x20   fn compare_seconds(&self, other: &Two<A, B>) -> Bool {\n\
         \x20       self.second > other.second\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let x = Two { first: 1, second: 2 };\n\
         \x20   let y = Two { first: 3, second: 4 };\n\
         \x20   println(x.compare_seconds(&y));\n\
         }\n",
        "otherparam",
    );
}

/// A method-level parameter with no bound is not rescued by the impl having one on a different
/// name.
#[test]
fn an_unbounded_method_parameter_is_still_rejected() {
    expect_reject(
        &format!(
            "{PAIR}\
             impl<T: Ord> Pair<T> {{\n\
             \x20   fn compare<U>(&self, left: U, right: U) -> Bool {{ left > right }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 1, b: 2 }}; println(p.compare(1, 2)); }}\n"
        ),
        "unboundedmethodparam",
    );
}

/// An unsatisfied callee obligation is still reported when neither level declares the bound.
#[test]
fn an_undischarged_callee_obligation_is_still_rejected() {
    expect_reject(
        &format!(
            "{PAIR}\
             fn largest<X: Ord>(a: X, b: X) -> X {{ if a > b {{ a }} else {{ b }} }}\n\
             impl<T> Pair<T> {{\n\
             \x20   fn best(self) -> T {{ largest(self.a, self.b) }}\n\
             }}\n\
             fn main() {{ let p = Pair {{ a: 4, b: 9 }}; println(p.best()); }}\n"
        ),
        "undischarged",
    );
}
