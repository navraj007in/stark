//! **DEV-150 / OWN-BORROW-002: argument evaluation does not provide two-phase borrow semantics.**
//!
//! The ruling, now normative in `03-Type-System.md`:
//!
//! > A call may not create an exclusive borrow of a place while another argument in the same call
//! > reads from or borrows an overlapping place. Such reads must be evaluated into locals before
//! > the exclusive borrow is created.
//!
//! ```stark
//! f(&mut x, x.field);        // refused
//! let field = x.field;       // hoist
//! f(&mut x, field);          // accepted
//! ```
//!
//! # What was wrong
//!
//! The rule existed but fired only for a LOCAL base. One indirection away it silently stopped:
//!
//! ```stark
//! bump(&mut h, h.limit);                            // refused
//! fn forward(h: &mut H) { bump(h, h.limit); }       // ACCEPTED, ran, did not build
//! ```
//!
//! Passing a `&mut`-typed place REBORROWS, which registers no active borrow, so the following read
//! saw nothing to conflict with. The HIR oracle executed it correctly and the native backend emitted
//! Rust that rustc refused with E0503 — accepted-but-unbuildable, and a rule whose meaning changed
//! depending on how the base was reached.
//!
//! # Why rejection rather than sequencing
//!
//! Accepting the indirect case would have required accepting the local case too, which is
//! two-phase borrows: evaluation-order machinery and a real semantics commitment. Uniform rejection
//! keeps ONE backend-neutral rule that every engine satisfies by construction, and stays reversible
//! if STARK later adopts two-phase borrows deliberately. This suite therefore pins the rejection as
//! the specified behaviour, not as an implementation artifact.
//!
//! # What replaced what
//!
//! `dev150_argument_conflict_through_reference.rs` pinned the INCONSISTENCY while the ruling was
//! open — it asserted that a local base is refused, a reference base accepted, and that the two
//! disagree. Its own doc required it to be rewritten around whichever ruling landed, and both of
//! its "the two disagree" tests went red the moment they agreed. This file is that rewrite, and it
//! records which way they agree: uniform REJECTION, hoisting required.
//!
//! # Uniformity is the property under test
//!
//! The defect was not "this program is wrongly accepted" — it was "the rule is not uniform". So the
//! negatives below vary the BASE (local, through `&mut`, field projection, index, receiver) and the
//! ORDER, holding the conflict constant. Each must give the same answer.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Front end only — this is a checker rule, and the whole point is that nothing reaches the later
/// engines. Returns every error, so a duplicate report is visible rather than hidden.
fn check(src: &str, tag: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    typecheck::analyze(&hir, file.clone())
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect()
}

fn expect_refused(src: &str, tag: &str) {
    let errors = check(src, tag);
    assert!(
        !errors.is_empty(),
        "{tag}: an overlapping read in the same call must be refused (OWN-BORROW-002)"
    );
    assert!(
        errors.iter().any(|e| e.starts_with("E0101")),
        "{tag}: expected the borrow-conflict code E0101, got: {errors:?}"
    );
    // One mistake, one diagnostic. Two reports in different words read as two problems and send a
    // reader looking for a second cause that does not exist.
    assert_eq!(
        errors.len(),
        1,
        "{tag}: expected exactly one diagnostic, got {}: {errors:?}",
        errors.len()
    );
}

fn expect_accepted(src: &str, tag: &str) {
    let errors = check(src, tag);
    assert!(
        errors.is_empty(),
        "{tag}: this does not overlap and must be accepted: {errors:?}"
    );
}

const HOLDER: &str = "struct H { limit: UInt64, seen: UInt64 }\n\
                      fn bump(h: &mut H, by: UInt64) { h.seen = h.seen + by; }\n";

// ------------------------------------------------------------------- negatives --

/// A LOCAL base. This was already refused; it is here so the suite states the whole rule rather
/// than only the half that changed.
#[test]
fn a_local_base_is_refused() {
    expect_refused(
        &format!(
            "{HOLDER}fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   bump(&mut h, h.limit);\n\
             }}\n"
        ),
        "localbase",
    );
}

/// **The reproducer.** A base reached through a `&mut` parameter — the case that was ACCEPTED, ran
/// on the oracle, and could not be built.
#[test]
fn a_base_reached_through_a_mutable_reference_is_refused() {
    expect_refused(
        &format!(
            "{HOLDER}fn forward(h: &mut H) {{ bump(h, h.limit); }}\n\
             fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   forward(&mut h);\n\
             }}\n"
        ),
        "throughref",
    );
}

/// **Order-independence.** `f(x.field, &mut x)` is the same conflict as `f(&mut x, x.field)`. A
/// check that fell out of the left-to-right walk could only ever catch the borrow-first order, so
/// this is what forces the rule to be a pass over the whole argument list.
#[test]
fn the_read_before_the_borrow_is_refused() {
    expect_refused(
        "struct H { limit: UInt64, seen: UInt64 }\n\
         fn bump(by: UInt64, h: &mut H) { h.seen = h.seen + by; }\n\
         fn main() {\n\
         \x20   let mut h = H { limit: 3u64, seen: 0u64 };\n\
         \x20   bump(h.limit, &mut h);\n\
         }\n",
        "readfirst",
    );
}

/// A read buried inside arithmetic still counts — the collector must reach places nested in an
/// argument expression, not only an argument that IS a place.
#[test]
fn a_read_nested_in_an_expression_is_refused() {
    expect_refused(
        &format!(
            "{HOLDER}fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   bump(&mut h, h.limit + 1u64);\n\
             }}\n"
        ),
        "nestedread",
    );
}

/// A read inside a NESTED CALL in the argument list.
#[test]
fn a_read_inside_a_nested_call_is_refused() {
    expect_refused(
        &format!(
            "{HOLDER}fn double(v: UInt64) -> UInt64 {{ v + v }}\n\
             fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   bump(&mut h, double(h.limit));\n\
             }}\n"
        ),
        "nestedcall",
    );
}

/// A nested FIELD projection that overlaps the borrowed one: `&mut o.i` against `o.i.v`. Overlap is
/// prefix-based, so a borrow of a parent conflicts with a read of a child.
#[test]
fn an_overlapping_field_projection_is_refused() {
    let errors = check(
        "struct Inner { v: UInt64 }\n\
         struct Outer { i: Inner }\n\
         fn f(x: &mut Inner, y: UInt64) { x.v = y; }\n\
         fn main() {\n\
         \x20   let mut o = Outer { i: Inner { v: 1u64 } };\n\
         \x20   f(&mut o.i, o.i.v);\n\
         }\n",
        "fieldoverlap",
    );
    assert!(
        errors.iter().any(|e| e.starts_with("E0101")),
        "a read of a child of the exclusively borrowed place must be refused: {errors:?}"
    );
}

/// A SHARED borrow of the same place in the same call. The rule says "reads from or borrows", so
/// `f(&mut x, &x)` is refused just as a plain read is.
#[test]
fn a_shared_borrow_of_the_borrowed_place_is_refused() {
    let errors = check(
        "struct H { a: UInt64 }\n\
         fn f(x: &mut H, y: &H) -> UInt64 { x.a + y.a }\n\
         fn main() {\n\
         \x20   let mut h = H { a: 1u64 };\n\
         \x20   println(f(&mut h, &h));\n\
         }\n",
        "sharedandmut",
    );
    assert!(
        errors.iter().any(|e| e.starts_with("E0101")),
        "a shared borrow overlapping an exclusive one in the same call must be refused: {errors:?}"
    );
}

/// A METHOD RECEIVER is an argument for this purpose: `v.push(v.len())` is the same conflict as
/// `push(&mut v, len(&v))`. Whichever check reports it, the ANSWER must be the same — that is what
/// uniformity means here.
#[test]
fn a_method_receiver_conflict_is_refused() {
    let errors = check(
        "fn main() {\n\
         \x20   let mut v: Vec<UInt64> = Vec::new();\n\
         \x20   v.push(v.len());\n\
         }\n",
        "receiver",
    );
    assert!(
        errors.iter().any(|e| e.starts_with("E0101")),
        "a receiver borrowed exclusively while an argument reads it must be refused: {errors:?}"
    );
}

// ------------------------------------------------------------------- positives --

/// **The hoisted form the ruling prescribes.** If this ever stops compiling the rule has no escape
/// hatch, and a rule with no escape hatch is a rule that gets reverted.
#[test]
fn the_hoisted_form_is_accepted() {
    expect_accepted(
        &format!(
            "{HOLDER}fn forward(h: &mut H) {{\n\
             \x20   let limit = h.limit;\n\
             \x20   bump(h, limit);\n\
             }}\n\
             fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   forward(&mut h);\n\
             }}\n"
        ),
        "hoisted",
    );
}

/// The hoisted form for a LOCAL base.
#[test]
fn the_hoisted_form_over_a_local_is_accepted() {
    expect_accepted(
        &format!(
            "{HOLDER}fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   let limit = h.limit;\n\
             \x20   bump(&mut h, limit);\n\
             }}\n"
        ),
        "hoistedlocal",
    );
}

/// **Non-overlap: different locals.** The rule must not fire merely because a call has a `&mut`
/// argument and some other argument reads something.
#[test]
fn a_read_of_a_different_local_is_accepted() {
    expect_accepted(
        &format!(
            "{HOLDER}fn main() {{\n\
             \x20   let mut a = H {{ limit: 1u64, seen: 0u64 }};\n\
             \x20   let b = H {{ limit: 2u64, seen: 0u64 }};\n\
             \x20   bump(&mut a, b.limit);\n\
             }}\n"
        ),
        "differentlocal",
    );
}

/// A literal argument alongside an exclusive borrow: nothing is read at all.
#[test]
fn a_literal_argument_is_accepted() {
    expect_accepted(
        &format!(
            "{HOLDER}fn main() {{\n\
             \x20   let mut h = H {{ limit: 3u64, seen: 0u64 }};\n\
             \x20   bump(&mut h, 1u64);\n\
             }}\n"
        ),
        "literalarg",
    );
}

/// Successive calls each taking `&mut` of the same place. The borrow lives for the CALL and no
/// longer (03 "References and Lifetimes" rule 4), so `f(&mut x); f(&mut x);` stays legal — if this
/// broke, the rule would have silently extended borrow lifetimes.
#[test]
fn successive_exclusive_borrows_are_accepted() {
    expect_accepted(
        "struct H { a: UInt64 }\n\
         fn f(h: &mut H) { h.a = h.a + 1u64; }\n\
         fn main() {\n\
         \x20   let mut h = H { a: 1u64 };\n\
         \x20   f(&mut h);\n\
         \x20   f(&mut h);\n\
         \x20   println(h.a);\n\
         }\n",
        "successive",
    );
}

/// Reborrowing the same `&mut` parameter across successive calls — the DEV-147 shape, which must
/// keep working now that a reborrow registers as an exclusive borrow for its call.
#[test]
fn successive_reborrows_of_a_parameter_are_accepted() {
    expect_accepted(
        "struct H { a: UInt64 }\n\
         fn f(h: &mut H) { h.a = h.a + 1u64; }\n\
         fn twice(h: &mut H) {\n\
         \x20   f(h);\n\
         \x20   f(h);\n\
         }\n\
         fn main() {\n\
         \x20   let mut h = H { a: 1u64 };\n\
         \x20   twice(&mut h);\n\
         \x20   println(h.a);\n\
         }\n",
        "successivereborrow",
    );
}

/// Two shared reads of the same place in one call. No exclusive borrow exists, so there is nothing
/// to conflict with — the rule is about exclusivity, not about repetition.
#[test]
fn two_shared_reads_of_one_place_are_accepted() {
    expect_accepted(
        "struct H { a: UInt64 }\n\
         fn f(x: UInt64, y: UInt64) -> UInt64 { x + y }\n\
         fn main() {\n\
         \x20   let h = H { a: 1u64 };\n\
         \x20   println(f(h.a, h.a));\n\
         }\n",
        "twoshared",
    );
}
