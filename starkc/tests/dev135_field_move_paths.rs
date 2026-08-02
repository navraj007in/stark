//! **DEV-135: a struct field is one place, however many times it is written.**
//!
//! The move set was ALREADY field-precise — `Place` carries projections, `moved_places` is a set
//! of them, and `places_overlap` does prefix matching, so moving `pair.left` correctly left
//! `pair.right` live. What was broken was field IDENTITY:
//!
//! ```text
//! Projection::Field(name.lo, name.hi)   // the SPAN's byte offsets
//! ```
//!
//! Two mentions of the same field are at different offsets, so `owner.handle` on one line and
//! `owner.handle` on the next were two DIFFERENT projections that `places_overlap` correctly
//! reported as disjoint. The second move was invisible to the front end, and the HIR oracle then
//! failed at run time with "internal compiler error: use of moved or invalid field" — the wrong
//! category for a user-authored program, and several layers too late.
//!
//! Storing the resolved NAME makes two mentions of one field the same place. Same class as
//! DEV-122: identity taken from a span rather than from what the span denotes.
//!
//! # Why the positives matter as much as the negatives here
//!
//! The WP's stage-one option was "parent poisoning" — mark the whole parent unavailable once any
//! field moves. The inventory ruled it out: sibling-after-partial-move is asserted as REQUIRED by
//! `gate2-valid/18_partial_moves.stark`, `mir_verify`, `mir_differential`,
//! `three_engine_differential`, `native_c5_3_aggregates_enums`, and the C6 corpus. So every
//! sibling/nested case below is a MUST-PASS, not a documented limitation, and this file is the
//! evidence that the precise repair did not become the blunt one.

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
                diagnostic.starts_with("E0100"),
                "{tag}: expected E0100, got: {diagnostic}"
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
    match starkc::interp::run(&hir, file, &checked.tables) {
        Ok(execution) => execution.output,
        Err(error) => panic!("{tag}: runtime error: {}", error.message),
    }
}

// ------------------------------------------------------------------ must reject --

/// The CD-334 reproducer: the same non-`Copy` field moved out twice.
#[test]
fn the_same_field_moved_twice_is_rejected() {
    let diagnostic = expect_reject(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   let first = owner.handle;\n\
         \x20   let second = owner.handle;\n\
         \x20   println(first.name.as_str());\n\
         \x20   println(second.name.as_str());\n\
         }\n",
        "twice",
    );
    assert!(
        diagnostic.contains("owner.handle"),
        "the diagnostic must name the field path, not just the local: {diagnostic}"
    );
}

/// A NESTED field moved twice. The projection chain must compare element-wise.
#[test]
fn the_same_nested_field_moved_twice_is_rejected() {
    expect_reject(
        "struct Leaf { name: String }\n\
         struct Middle { leaf: Leaf }\n\
         struct Root { middle: Middle }\n\
         fn main() {\n\
         \x20   let root = Root { middle: Middle { leaf: Leaf { name: String::from(\"x\") } } };\n\
         \x20   let first = root.middle.leaf;\n\
         \x20   let second = root.middle.leaf;\n\
         \x20   println(first.name.as_str());\n\
         \x20   println(second.name.as_str());\n\
         }\n",
        "nestedtwice",
    );
}

/// Moving the WHOLE parent after a field has been moved out of it. `places_overlap`'s prefix rule
/// is what catches this: `[]` is a prefix of `[handle]`.
#[test]
fn moving_the_parent_after_a_field_move_is_rejected() {
    expect_reject(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn take(owner: Owner) {}\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   let stolen = owner.handle;\n\
         \x20   take(owner);\n\
         \x20   println(stolen.name.as_str());\n\
         }\n",
        "parentafterfield",
    );
}

/// The reverse order: a field read after the whole parent was moved. `[handle]` has `[]` as a
/// prefix, so the same rule covers both directions.
#[test]
fn reading_a_field_after_the_parent_moved_is_rejected() {
    expect_reject(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn take(owner: Owner) {}\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   take(owner);\n\
         \x20   let stolen = owner.handle;\n\
         \x20   println(stolen.name.as_str());\n\
         }\n",
        "fieldafterparent",
    );
}

/// A tuple element moved twice — `TupleField` had the identical span-identity bug.
#[test]
fn the_same_tuple_element_moved_twice_is_rejected() {
    expect_reject(
        "fn main() {\n\
         \x20   let pair = (String::from(\"a\"), String::from(\"b\"));\n\
         \x20   let first = pair.0;\n\
         \x20   let second = pair.0;\n\
         \x20   println(first.as_str());\n\
         \x20   println(second.as_str());\n\
         }\n",
        "tupletwice",
    );
}

/// Borrowing a field after moving it out.
#[test]
fn borrowing_a_field_after_moving_it_is_rejected() {
    expect_reject(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn observe(marker: &Marker) -> UInt64 { marker.name.len() }\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   let stolen = owner.handle;\n\
         \x20   println(observe(&owner.handle));\n\
         \x20   println(stolen.name.as_str());\n\
         }\n",
        "borrowaftermove",
    );
}

// -------------------------------------------------------------------- must pass --

/// **The load-bearing shape.** Move one field, then use its SIBLING. This is
/// `gate2-valid/18_partial_moves.stark` in miniature, and it is why parent poisoning was
/// rejected: if this ever fails, the repair has become the blunt instrument the inventory ruled
/// out.
#[test]
fn moving_one_field_leaves_its_sibling_usable() {
    expect_accept(
        "struct Text { value: String }\n\
         struct Pair { left: Text, right: Text }\n\
         fn consume(value: Text) {}\n\
         fn main() {\n\
         \x20   let pair = Pair {\n\
         \x20       left: Text { value: String::from(\"a\") },\n\
         \x20       right: Text { value: String::from(\"b\") },\n\
         \x20   };\n\
         \x20   consume(pair.left);\n\
         \x20   consume(pair.right);\n\
         }\n",
        "sibling",
    );
}

/// ...and it must EXECUTE with each value destroyed exactly once.
#[test]
fn sibling_fields_are_each_destroyed_exactly_once() {
    let out = run(
        "struct Text { value: String }\n\
         impl Drop for Text {\n\
         \x20   fn drop(&mut self) { println(self.value.as_str()); }\n\
         }\n\
         struct Pair { left: Text, right: Text }\n\
         fn consume(value: Text) {}\n\
         fn main() {\n\
         \x20   let pair = Pair {\n\
         \x20       left: Text { value: String::from(\"a\") },\n\
         \x20       right: Text { value: String::from(\"b\") },\n\
         \x20   };\n\
         \x20   consume(pair.left);\n\
         \x20   consume(pair.right);\n\
         \x20   println(\"done\");\n\
         }\n",
        "siblingdrop",
    );
    assert_eq!(
        out.trim(),
        "a\nb\ndone",
        "each field is dropped once, inside the `consume` that took it"
    );
}

/// Nested siblings: moving `root.middle.leaf` must leave `root.middle.other` usable.
#[test]
fn moving_a_nested_field_leaves_its_nested_sibling_usable() {
    expect_accept(
        "struct Leaf { name: String }\n\
         struct Middle { leaf: Leaf, other: Leaf }\n\
         struct Root { middle: Middle }\n\
         fn consume(value: Leaf) {}\n\
         fn main() {\n\
         \x20   let root = Root {\n\
         \x20       middle: Middle {\n\
         \x20           leaf: Leaf { name: String::from(\"a\") },\n\
         \x20           other: Leaf { name: String::from(\"b\") },\n\
         \x20       },\n\
         \x20   };\n\
         \x20   consume(root.middle.leaf);\n\
         \x20   consume(root.middle.other);\n\
         }\n",
        "nestedsibling",
    );
}

/// Tuple siblings.
#[test]
fn moving_one_tuple_element_leaves_the_other_usable() {
    expect_accept(
        "fn consume(value: String) {}\n\
         fn main() {\n\
         \x20   let pair = (String::from(\"a\"), String::from(\"b\"));\n\
         \x20   consume(pair.0);\n\
         \x20   consume(pair.1);\n\
         }\n",
        "tuplesibling",
    );
}

/// A `Copy` field is not moved at all, so it stays readable however many times it is read. Pins
/// that the identity fix did not accidentally start treating copies as moves.
#[test]
fn a_copy_field_is_reusable() {
    expect_accept(
        "struct Cursor { offset: UInt64, line: UInt64 }\n\
         fn main() {\n\
         \x20   let cursor = Cursor { offset: 3u64, line: 1u64 };\n\
         \x20   let a = cursor.offset;\n\
         \x20   let b = cursor.offset;\n\
         \x20   let c = cursor.line;\n\
         \x20   println(a + b + c);\n\
         }\n",
        "copyfield",
    );
}

/// Borrowing a field repeatedly is not a move either.
#[test]
fn borrowing_a_field_repeatedly_is_accepted() {
    expect_accept(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn observe(marker: &Marker) -> UInt64 { marker.name.len() }\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   println(observe(&owner.handle));\n\
         \x20   println(observe(&owner.handle));\n\
         }\n",
        "repeatborrow",
    );
}

/// A field of a struct whose own type implements `Drop` still cannot be partially moved — that
/// rule predates this repair (`gate2_valid`'s partial-move-with-Drop case) and must be untouched.
#[test]
fn partial_move_out_of_a_drop_type_is_still_rejected() {
    expect_reject(
        "struct Marker { name: String }\n\
         struct Holder { m: Marker }\n\
         impl Drop for Holder {\n\
         \x20   fn drop(&mut self) { println(\"dropping holder\"); }\n\
         }\n\
         fn main() {\n\
         \x20   let h = Holder { m: Marker { name: String::from(\"x\") } };\n\
         \x20   let stolen = h.m;\n\
         \x20   println(stolen.name.as_str());\n\
         }\n",
        "droppartial",
    );
}

/// Two DIFFERENT locals of the same struct type do not share a move set — the place is keyed on
/// the local as well as the projection.
#[test]
fn the_same_field_of_two_different_locals_is_independent() {
    expect_accept(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn consume(value: Marker) {}\n\
         fn main() {\n\
         \x20   let first = Owner { handle: Marker { name: String::from(\"a\") } };\n\
         \x20   let second = Owner { handle: Marker { name: String::from(\"b\") } };\n\
         \x20   consume(first.handle);\n\
         \x20   consume(second.handle);\n\
         }\n",
        "twolocals",
    );
}

/// A field moved on a branch that TERMINATES does not poison the join — DEV-136's rule composed
/// with DEV-135's field precision, which is the interaction the work package called out.
#[test]
fn a_field_moved_on_a_terminating_branch_does_not_poison_the_join() {
    expect_accept(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn consume(value: Marker) {}\n\
         fn build(flag: Bool, owner: Owner) -> UInt64 {\n\
         \x20   if flag {\n\
         \x20       consume(owner.handle);\n\
         \x20       return 0u64;\n\
         \x20   }\n\
         \x20   owner.handle.name.len()\n\
         }\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   println(build(false, owner));\n\
         }\n",
        "dev136interaction",
    );
}

/// ...and a field moved on a REACHABLE branch still poisons it.
#[test]
fn a_field_moved_on_a_reachable_branch_is_still_rejected() {
    expect_reject(
        "struct Marker { name: String }\n\
         struct Owner { handle: Marker }\n\
         fn consume(value: Marker) {}\n\
         fn build(flag: Bool, owner: Owner) -> UInt64 {\n\
         \x20   if flag {\n\
         \x20       consume(owner.handle);\n\
         \x20   }\n\
         \x20   owner.handle.name.len()\n\
         }\n\
         fn main() {\n\
         \x20   let owner = Owner { handle: Marker { name: String::from(\"x\") } };\n\
         \x20   println(build(false, owner));\n\
         }\n",
        "reachablefield",
    );
}
