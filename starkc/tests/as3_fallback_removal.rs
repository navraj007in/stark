//! **AS3 Boundary 4 — the MIR method and operator paths have no name-scanning fallback.**
//!
//! `lower_method_call`, `lower_user_eq` and `lower_user_ord` each ended in a call to
//! `find_impl_fn(nominal, name, ...)`: a scan of every impl on the receiver's nominal for a member
//! with the right NAME, re-deciding what the checker had already decided. Those are gone. MIR now
//! consumes the published `CallableUse` and nothing else.
//!
//! **What deleting them exposed.** The fallback was covering three publication gaps, each the same
//! shape — a receiver that is parametric at check time, so no `Static` selection can exist:
//!
//! | | Shape | Published before |
//! | --- | --- | --- |
//! | (a) | `x.m()` where the receiver's type is a bounded parameter | a `Bound` use (step 2) |
//! | (b) | `self.m()` inside a trait DEFAULT body — DEV-190 | **nothing** |
//! | (c) | `a == b` / `a < b` where the operand is a bounded parameter — DEV-191 | **nothing** |
//!
//! The census is the evidence: instrumented, the fallback fired ~60 times across the differential
//! and trait suites; after `Static` was consumed, twice; after (b), never; (c) surfaced only when
//! the arm was actually deleted, because it lives in a suite the earlier mutation evidence did not
//! cover.
//!
//! **Why these tests are about lowering succeeding, not about output.** With no fallback, a missing
//! publication is not a silent divergence any more — MIR lowering fails outright with "method not
//! found" or "`==` on a user type without an `Eq` impl". So each fixture below is a program that
//! cannot lower unless the checker published the right use. That is the structural guarantee the
//! deletion buys, and it is stronger than any assertion this file could make on its own.

mod support;

use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use starkc::typecheck::{CalleeSelection, DispatchProvenance};
use std::sync::Arc;
use support::differential::agree_completing_with_stdout;

fn program(source: &str) -> starkc::session::CheckedProgram {
    let file = Arc::new(SourceFile::new("test.stark", source));
    match CompilerSession::for_source(file, LanguageOptions::CORE).check() {
        Ok(program) => program,
        Err(failure) => panic!("must compile:\n{}", failure.render()),
    }
}

/// Every available engine, agreeing, with the expected stdout. MIR lowering is the load-bearing
/// step: with the name scan gone, a missing publication makes it fail outright rather than diverge.
fn agree(tag: &str, source: &str, expected: &str) {
    agree_completing_with_stdout(tag, source, expected);
}

const TRAIT_DEFAULT: &str = "trait Describe {\n\
                             \x20   fn id(&self) -> Int32;\n\
                             \x20   fn twice(&self) -> Int32 { self.id() * 2 }\n\
                             }\n\
                             struct Tagged<T> { v: T, n: Int32 }\n\
                             impl<T> Describe for Tagged<T> {\n    fn id(&self) -> Int32 { self.n }\n}\n";

#[test]
fn dev190_a_trait_default_calling_self_publishes_a_bound_use() {
    let source = format!(
        "{TRAIT_DEFAULT}fn main() {{ let g = Tagged {{ v: 9, n: 77 }}; println(g.twice()); }}\n"
    );
    agree("as3_dev190_trait_default", &source, "154\n");

    // `self.id()` is late-bound: the trait is known inside the default body, the body is not.
    let program = program(&source);
    let bound: Vec<&str> = program
        .tables()
        .callable_uses
        .iter()
        .filter(|u| matches!(u.provenance, DispatchProvenance::Bound { .. }))
        .filter_map(|u| match &u.selection {
            CalleeSelection::Bound { member, .. } => Some(member.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        bound.contains(&"id"),
        "`self.id()` inside `twice`'s default body must publish a Bound use, got {bound:?}"
    );
}

#[test]
fn dev191_an_operator_on_a_bounded_parameter_publishes_a_bound_use() {
    // The fixture from `over_acceptance_audit`, which is where deleting the `eq` fallback failed.
    let source = "struct P { v: Int32 }\n\
                  impl Eq for P {\n    fn eq(&self, other: &P) -> Bool { self.v == other.v }\n}\n\
                  fn same<T: Eq>(a: T, b: T) -> Bool { a == b }\n\
                  fn main() {\n\
                  \x20   let p: P = P { v: 1 };\n\
                  \x20   let q: P = P { v: 1 };\n\
                  \x20   let r: P = P { v: 2 };\n\
                  \x20   println(same(p, q));\n\
                  \x20   println(same(P { v: 3 }, r));\n}\n";
    agree("as3_dev191_bound_eq", source, "true\nfalse\n");

    let program = program(source);
    let has_bound_eq = program.tables().callable_uses.iter().any(|u| {
        matches!(
            u.provenance,
            DispatchProvenance::Bound {
                trait_: starkc::hir::BoundTrait::Core(starkc::hir::CoreTrait::Eq)
            }
        )
    });
    assert!(
        has_bound_eq,
        "`a == b` on a `T: Eq` parameter must publish a Bound use against Eq"
    );
}

#[test]
fn dev191_ordered_comparison_on_a_bounded_parameter_lowers() {
    let source = "struct P { v: Int32 }\n\
                  impl Ord for P {\n\
                  \x20   fn cmp(&self, other: &P) -> Ordering { self.v.cmp(&other.v) }\n}\n\
                  fn less<T: Ord>(a: T, b: T) -> Bool { a < b }\n\
                  fn main() {\n\
                  \x20   println(less(P { v: 1 }, P { v: 2 }));\n\
                  \x20   println(less(P { v: 2 }, P { v: 1 }));\n}\n";
    agree("as3_dev191_bound_ord", source, "true\nfalse\n");
}

#[test]
fn arithmetic_on_a_num_bounded_parameter_still_publishes_nothing() {
    // The negative half of DEV-191, and the reason the publisher checks the bound rather than
    // publishing for every `Ty::Param`. `Num` is compiler-known and primitives-only: there is no
    // user body for a call site to name, so publishing one would be an invention.
    let source = "fn total<T: Num>(a: T, b: T) -> T { a + b }\n\
                  fn main() { println(total(2, 3)); }\n";
    agree("as3_num_bound", source, "5\n");

    let program = program(source);
    let num_uses = program
        .tables()
        .callable_uses
        .iter()
        .filter(|u| {
            matches!(
                u.provenance,
                DispatchProvenance::Bound {
                    trait_: starkc::hir::BoundTrait::Core(starkc::hir::CoreTrait::Num)
                }
            )
        })
        .count();
    assert_eq!(
        num_uses, 0,
        "arithmetic through a `Num` bound names no user callable"
    );
}

#[test]
fn a_generic_impls_method_lowers_without_a_name_scan() {
    // The `Static` half: an ordinary method call on a generic nominal, which is what ~60 of the
    // instrumented fallback hits were. It lowers only because `static_selected_key` reads the
    // published body.
    let source = "struct Stack<T> { items: Vec<T> }\n\
                  impl<T> Stack<T> {\n\
                  \x20   fn make() -> Stack<T> { Stack { items: Vec::new() } }\n\
                  \x20   fn push_item(&mut self, v: T) { self.items.push(v); }\n\
                  \x20   fn size(&self) -> UInt64 { self.items.len() }\n}\n\
                  fn main() {\n\
                  \x20   let mut s: Stack<Int32> = Stack::make();\n\
                  \x20   s.push_item(4);\n\
                  \x20   println(s.size());\n\
                  \x20   let mut t: Stack<String> = Stack::make();\n\
                  \x20   t.push_item(String::from(\"hi\"));\n\
                  \x20   println(t.size());\n}\n";
    agree("as3_generic_impl_method", source, "1\n1\n");
}

#[test]
fn two_traits_declaring_the_same_method_name_select_by_the_published_use() {
    // The defect the scan permitted, now impossible by construction rather than by filtering: with
    // no name scan there is nothing left to disambiguate.
    let source = "trait Loud { fn speak(&self) -> String; }\n\
                  trait Soft { fn speak(&self) -> String; }\n\
                  struct S { v: Int32 }\n\
                  impl Loud for S {\n    fn speak(&self) -> String { String::from(\"LOUD\") }\n}\n\
                  impl Soft for S {\n    fn speak(&self) -> String { String::from(\"soft\") }\n}\n\
                  fn shout<T: Loud>(x: T) -> String { x.speak() }\n\
                  fn hush<T: Soft>(x: T) -> String { x.speak() }\n\
                  fn main() {\n\
                  \x20   println(shout(S { v: 1 }));\n\
                  \x20   println(hush(S { v: 1 }));\n}\n";
    agree("as3_same_name_two_traits", source, "LOUD\nsoft\n");
}

#[test]
fn dev192_a_bound_equality_runs_the_users_impl_not_structural_comparison() {
    // **The sharp version of DEV-191/192.** Every earlier fixture had a user `eq` that AGREED with
    // structural comparison, so falling through to `Value` equality produced the right answer and
    // the defect was invisible. Here `eq` compares only `id` and ignores `tag`, so the two
    // algorithms disagree: structural says false, the user's impl says true.
    //
    // Before the repair the HIR oracle printed `false` here — silently substituting structural
    // equality for a type that declares its own. That is a wrong answer, not a missing feature.
    let source = "struct Rec { id: Int32, tag: Int32 }\n\
                  impl Eq for Rec {\n    fn eq(&self, other: &Rec) -> Bool { self.id == other.id }\n}\n\
                  fn same<T: Eq>(a: T, b: T) -> Bool { a == b }\n\
                  fn main() {\n\
                  \x20   println(same(Rec { id: 1, tag: 10 }, Rec { id: 1, tag: 99 }));\n\
                  \x20   println(same(Rec { id: 1, tag: 10 }, Rec { id: 2, tag: 10 }));\n}\n";
    // Structural comparison would print "false\ntrue"; the user's impl prints the opposite.
    agree("as3_dev192_bound_eq_custom", source, "true\nfalse\n");
}

#[test]
fn dev192_a_bound_ordering_runs_the_users_impl() {
    // The `Ord` counterpart: `cmp` deliberately REVERSES the natural order, so an implementation
    // that ignored it and compared some other way would be caught rather than agreed with.
    let source = "struct Rev { v: Int32 }\n\
                  impl Ord for Rev {\n\
                  \x20   fn cmp(&self, other: &Rev) -> Ordering { other.v.cmp(&self.v) }\n}\n\
                  fn less<T: Ord>(a: T, b: T) -> Bool { a < b }\n\
                  fn main() {\n\
                  \x20   println(less(Rev { v: 1 }, Rev { v: 2 }));\n\
                  \x20   println(less(Rev { v: 2 }, Rev { v: 1 }));\n}\n";
    // Reversed: 1 < 2 is false under this `cmp`, and 2 < 1 is true.
    agree("as3_dev192_bound_ord_reversed", source, "false\ntrue\n");
}

#[test]
fn dev194_a_trait_default_reached_through_a_bound_gets_its_self_binding() {
    // **The `pkg/07-traits` regression, and the sharpest argument for deleting fallbacks.**
    //
    // Two Bound calls nest:
    //
    //   announce(&s)  ->  item.shout()   Bound, resolves to the trait DEFAULT body
    //                     self.name()    Bound, inside that body, needs `Self = Server`
    //
    // A `Bound` call's environment cannot be published — the body is chosen only once `Self` is
    // concrete, which happens at run time. `specialize_bound_callable` returns those bindings and
    // the interpreter was discarding them, so the default body ran with NO `Self` at all and the
    // inner `self.name()` resolved nothing.
    //
    // **The name scan had been hiding this.** It found `name` on the runtime value's nominal
    // without needing an environment, so the missing binding was invisible for as long as a
    // fallback existed to paper over it. Deleting the fallback did not cause the defect; it
    // revealed one, in the external sample suite rather than in any unit test — which is why the
    // sample suite is a gate and not a formality.
    let source = "trait Describe {\n\
                  \x20   fn name(&self) -> String;\n\
                  \x20   fn shout(&self) -> String {\n\
                  \x20       let mut out = String::from(\"<\");\n\
                  \x20       out.push_str(self.name().as_str());\n\
                  \x20       out\n\
                  \x20   }\n\
                  }\n\
                  struct Server { host: String }\n\
                  struct Job { id: Int32 }\n\
                  impl Describe for Server {\n    fn name(&self) -> String { String::from(\"server\") }\n}\n\
                  impl Describe for Job {\n\
                  \x20   fn name(&self) -> String { String::from(\"job\") }\n\
                  \x20   fn shout(&self) -> String { String::from(\"JOB!\") }\n\
                  }\n\
                  fn announce<D: Describe>(item: &D) -> String { item.shout() }\n\
                  fn main() {\n\
                  \x20   let s: Server = Server { host: String::from(\"h\") };\n\
                  \x20   let j: Job = Job { id: 1 };\n\
                  \x20   println(announce(&s));\n\
                  \x20   println(announce(&j));\n\
                  \x20   println(s.shout());\n}\n";
    // Two implementors, one taking the default and one overriding it, so a resolution that ignored
    // `Self` and picked "the first impl declaring the name" would print the same text twice.
    agree(
        "as3_dev194_bound_trait_default_self",
        source,
        "<server\nJOB!\n<server\n",
    );
}
