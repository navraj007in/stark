//! **DEV-154: borrow conflicts are compared by PLACE, not by root local.**
//!
//! `03-Type-System.md` OWN-BORROW-001 says, and has always said:
//!
//! > Disjoint field projections do not overlap.
//!
//! Every comparison in `borrowck` tested `b.local == local`, so a borrow of `p.a` blocked a read of
//! `p.b` — refusing a valid program, and contradicting the spec the checker exists to enforce.
//! `places_overlap` has been field-precise since DEV-135; these comparisons simply never used it.
//!
//! # How it surfaced
//!
//! CD-357 added a second conflict check for call arguments, and that one WAS place-granular. It
//! correctly declined to fire on `f(&mut p.a, p.b)` — and the older local-granular check reported it
//! anyway. Two checks in the same area disagreeing about granularity is what made a long-standing
//! over-rejection visible. It was filed rather than bundled into CD-357, because loosening a borrow
//! check must not ride along with a ruling that tightens one.
//!
//! # Why the negatives are the important half
//!
//! This repair makes the checker accept MORE. Every test below that expects a refusal is load-
//! bearing: if the place comparison were too generous, the checker would admit real aliasing —
//! two exclusive borrows of one field, a read through a parent while a child is exclusively
//! borrowed, or a move out of storage something still points into. The prefix rule in
//! `places_overlap` is what makes parent/child overlap while siblings do not, and that asymmetry is
//! pinned from both sides here.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

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

fn expect_accepted(src: &str, tag: &str) {
    let errors = check(src, tag);
    assert!(
        errors.is_empty(),
        "{tag}: disjoint places must not conflict (OWN-BORROW-001): {errors:?}"
    );
}

fn expect_refused(src: &str, tag: &str) {
    let errors = check(src, tag);
    assert!(
        errors.iter().any(|e| e.starts_with("E0101")),
        "{tag}: this genuinely overlaps and must stay refused, got: {errors:?}"
    );
}

const PAIR: &str = "struct P { a: UInt64, b: UInt64 }\n\
                    fn f(x: &mut UInt64, y: UInt64) { *x = *x + y; }\n";

// ------------------------------------------------------------------- accepted --

/// **The reproducer.** A borrow of one field beside a read of its sibling.
#[test]
fn a_borrow_of_one_field_permits_reading_a_sibling() {
    expect_accepted(
        &format!(
            "{PAIR}fn main() {{\n\
             \x20   let mut p = P {{ a: 1u64, b: 2u64 }};\n\
             \x20   f(&mut p.a, p.b);\n\
             }}\n"
        ),
        "siblingread",
    );
}

/// Two EXCLUSIVE borrows of disjoint fields in one call. This is the strongest form of the
/// permission: `&mut p.a` and `&mut p.b` are both live at once and must not conflict.
#[test]
fn two_exclusive_borrows_of_disjoint_fields_are_accepted() {
    expect_accepted(
        "struct P { a: UInt64, b: UInt64 }\n\
         fn f(x: &mut UInt64, y: &mut UInt64) { *x = *y; }\n\
         fn main() {\n\
         \x20   let mut p = P { a: 1u64, b: 2u64 };\n\
         \x20   f(&mut p.a, &mut p.b);\n\
         }\n",
        "disjointmut",
    );
}

/// Disjoint fields borrowed across separate statements, held simultaneously by `let`. Borrows bound
/// with `let` are lexically scoped to end-of-block, so both are live at the second call.
#[test]
fn disjoint_field_borrows_held_together_are_accepted() {
    expect_accepted(
        "struct P { a: UInt64, b: UInt64 }\n\
         fn use_it(x: &mut UInt64) { *x = *x + 1u64; }\n\
         fn main() {\n\
         \x20   let mut p = P { a: 1u64, b: 2u64 };\n\
         \x20   let first = &mut p.a;\n\
         \x20   use_it(first);\n\
         \x20   let second = &mut p.b;\n\
         \x20   use_it(second);\n\
         }\n",
        "heldtogether",
    );
}

/// Nested disjointness: `o.i.v` against `o.j.v`. The prefix rule has to diverge at the first
/// differing projection, not only at the root.
#[test]
fn nested_disjoint_projections_are_accepted() {
    expect_accepted(
        "struct I { v: UInt64 }\n\
         struct O { i: I, j: I }\n\
         fn f(x: &mut UInt64, y: UInt64) { *x = *x + y; }\n\
         fn main() {\n\
         \x20   let mut o = O { i: I { v: 1u64 }, j: I { v: 2u64 } };\n\
         \x20   f(&mut o.i.v, o.j.v);\n\
         }\n",
        "nesteddisjoint",
    );
}

// ------------------------------------------------------------------- refused --

/// The SAME field, read while exclusively borrowed. The permission must not extend to identity.
#[test]
fn a_read_of_the_borrowed_field_is_still_refused() {
    expect_refused(
        &format!(
            "{PAIR}fn main() {{\n\
             \x20   let mut p = P {{ a: 1u64, b: 2u64 }};\n\
             \x20   f(&mut p.a, p.a);\n\
             }}\n"
        ),
        "samefield",
    );
}

/// Two exclusive borrows of the SAME field. Admitting this would be a real aliasing hole.
#[test]
fn two_exclusive_borrows_of_one_field_are_still_refused() {
    expect_refused(
        "struct P { a: UInt64, b: UInt64 }\n\
         fn f(x: &mut UInt64, y: &mut UInt64) { *x = *y; }\n\
         fn main() {\n\
         \x20   let mut p = P { a: 1u64, b: 2u64 };\n\
         \x20   f(&mut p.a, &mut p.a);\n\
         }\n",
        "samefieldmut",
    );
}

/// A PARENT borrowed exclusively while a CHILD is read. Prefix overlap must still fire downward.
#[test]
fn reading_a_child_of_an_exclusively_borrowed_parent_is_refused() {
    expect_refused(
        "struct I { v: UInt64 }\n\
         struct O { i: I }\n\
         fn f(x: &mut I, y: UInt64) { x.v = y; }\n\
         fn main() {\n\
         \x20   let mut o = O { i: I { v: 1u64 } };\n\
         \x20   f(&mut o.i, o.i.v);\n\
         }\n",
        "parentchild",
    );
}

/// The WHOLE local borrowed exclusively while a field is read. Prefix overlap must fire from an
/// empty projection path, which is the case a naive "compare the projections" rule gets wrong.
#[test]
fn reading_a_field_of_an_exclusively_borrowed_local_is_refused() {
    expect_refused(
        "struct P { a: UInt64, b: UInt64 }\n\
         fn f(x: &mut P, y: UInt64) { x.a = y; }\n\
         fn main() {\n\
         \x20   let mut p = P { a: 1u64, b: 2u64 };\n\
         \x20   f(&mut p, p.b);\n\
         }\n",
        "wholelocal",
    );
}

/// **A move out of borrowed storage.** This check is deliberately stricter than the read rule — it
/// rejects under ANY live borrow, shared included — because moving invalidates the storage a live
/// view still points into. Making the comparison place-granular must not have weakened that.
#[test]
fn a_move_under_a_live_borrow_is_still_refused() {
    expect_refused(
        "fn main() {\n\
         \x20   let mut v: Vec<UInt64> = Vec::new();\n\
         \x20   let r = &mut v;\n\
         \x20   let w = v;\n\
         \x20   println(w.len());\n\
         }\n",
        "moveunderborrow",
    );
}

/// Assigning to a place that is exclusively borrowed.
#[test]
fn assigning_to_a_borrowed_place_is_still_refused() {
    expect_refused(
        "struct P { a: UInt64, b: UInt64 }\n\
         fn main() {\n\
         \x20   let mut p = P { a: 1u64, b: 2u64 };\n\
         \x20   let r = &mut p.a;\n\
         \x20   p.a = 5u64;\n\
         \x20   *r = 6u64;\n\
         }\n",
        "assignborrowed",
    );
}
