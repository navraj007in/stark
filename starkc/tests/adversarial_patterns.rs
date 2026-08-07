//! WP-C7.9 Packet C — PAT-BIND-001 in the reference interpreter, adversarially.
//!
//! `ce1_borrowed_payload_binding.rs` owns the rule's own matrix. This file covers what that matrix
//! could not: the shapes where binding by clone instead of by reference is **observable**, and the
//! ownership consequences that a clone would silently get right for reads and wrong for everything
//! else.
//!
//! The distinction matters because the original defect hid for a whole work package. Seventeen of
//! the rule's nineteen cases agreed across all three engines while the oracle was cloning, since
//! `s.len()` reads the same from a clone as from a reference. What a clone cannot do:
//!
//! - be dereferenced (`match *binding`), which is how nesting composes;
//! - leave the referent still owned and droppable exactly once;
//! - let two matches in a row observe the same storage.
//!
//! Each case here is chosen to fail under a clone-binding implementation.

mod support;

use support::differential::{agree_completing_with_drops, agree_completing_with_stdout};

const TYPES: &str = "enum Holder { Empty, Val(String) }\n\
                     struct Wrap { h: Holder }\n\
                     struct Pair { a: String, b: Int32 }\n";

fn agree(tag: &str, body: &str, expected_stdout: &str) {
    agree_completing_with_stdout(tag, &format!("{TYPES}{body}\n"), expected_stdout);
}

// --------------------------------------------------------- borrowed components, by shape --

/// A named struct field, matched through a reference, binds by reference — and the binding is used
/// as one.
#[test]
fn struct_field_through_a_reference_binds_by_reference() {
    agree(
        "struct_field",
        "fn peek(w: &Wrap) -> Int32 { match *w { Wrap { h } => match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, }, } }\n\
         fn main() { let w = Wrap { h: Holder::Val(String::from(\"abc\")) }; println(peek(&w)); }",
        "3\n",
    );
}

/// A tuple element. The rule is uniform over pattern forms, and a tuple's storage is projected the
/// same way a struct's is.
#[test]
fn tuple_element_through_a_reference_binds_by_reference() {
    agree(
        "tuple_element",
        "fn peek(t: &(String, Int32)) -> Int32 { match *t { (s, n) => s.len() as Int32 + n, } }\n\
         fn main() { let t = (String::from(\"abcd\"), 6); println(peek(&t)); }",
        "10\n",
    );
}

/// A `Copy` element in the same tuple still binds by value — proven by using it arithmetically,
/// which a reference could not do without a deref.
#[test]
fn copy_and_non_copy_components_bind_differently_in_one_pattern() {
    agree(
        "mixed_components",
        "fn peek(p: &Pair) -> Int32 { match *p { Pair { a, b } => a.len() as Int32 * b, } }\n\
         fn main() { let p = Pair { a: String::from(\"xy\"), b: 5 }; println(peek(&p)); }",
        "10\n",
    );
}

/// A variant payload reached through a field read whose base is a reference — the `w.h` form,
/// rather than an explicit `*w`. PAT-BIND-001 names both, and the interpreter's predicate must
/// agree with the type checker's on each.
#[test]
fn field_read_through_a_reference_base_is_a_borrowed_read() {
    agree(
        "field_base",
        "fn peek(w: &Wrap) -> Int32 { match w.h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let w = Wrap { h: Holder::Val(String::from(\"abcdef\")) }; println(peek(&w)); }",
        "6\n",
    );
}

/// Matching through an exclusive reference produces *shared* bindings — the deliberate floor. The
/// program reads, which is permitted; mutation through the binding is rejected by the front end and
/// pinned in `ce1_borrowed_payload_binding.rs`.
#[test]
fn matching_through_an_exclusive_reference_reads_the_referent() {
    agree(
        "through_mut",
        "fn peek(h: &mut Holder) -> Int32 { match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let mut h = Holder::Val(String::from(\"abcdefg\")); println(peek(&mut h)); }",
        "7\n",
    );
}

// ------------------------------------------------------------- what a clone would get wrong --

/// **Repeated inspection.** A borrowed match consumes nothing, so the same value can be matched
/// again and again and still read the same. A binding that had moved the payload out would fail the
/// second call, and one that cloned would pass — this case is here to bracket the third
/// possibility: that the projection went stale.
#[test]
fn a_borrowed_match_can_be_repeated() {
    agree(
        "repeat",
        "fn peek(h: &Holder) -> Int32 { match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let h = Holder::Val(String::from(\"abc\")); println(peek(&h) + peek(&h) + peek(&h)); }",
        "9\n",
    );
}

/// **A borrowed match over a user-`Drop` type is refused before MIR** — recorded, not asserted
/// away.
///
/// The case this replaces tried to prove that a borrowed match leaves the referent to be destroyed
/// exactly once by its owner. It cannot be written: lowering refuses `match *r` where the matched
/// type has a user `Drop` impl, with "front-end move-out-of-borrow gap". That is an
/// accepted-in-HIR / refused-in-MIR surface of the same family as the `HashMap` Drop-bearing entry
/// rows, whose refusal point is governed by **CE4 (CD-132)** — a ruled decision this work package
/// does not reopen.
///
/// So the boundary is pinned here instead, where someone revisiting CD-132 will find it, and the
/// non-consumption property it was meant to prove is covered by
/// `a_borrowed_match_can_be_repeated` (the referent survives and reads the same) and
/// `an_owned_scrutinee_still_moves_and_drops` (the owned path still destroys exactly once).
#[test]
fn a_borrowed_match_over_a_drop_type_is_refused_before_mir() {
    let source = "struct Loud { tag: Int32 }\n\
                  impl Drop for Loud { fn drop(&mut self) { } }\n\
                  enum Box2 { Full(Loud), Empty }\n\
                  fn peek(b: &Box2) -> Int32 { match *b { Box2::Full(l) => l.tag, Box2::Empty => 0, } }\n\
                  fn main() { let b = Box2::Full(Loud { tag: 4 }); println(peek(&b)); }\n";
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(
        "drop_through_reference.stark".to_string(),
        source.to_string(),
    ));
    let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = starkc::typecheck::analyze(&hir);
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error),
        "the front end still accepts this program"
    );
    let lowered = starkc::mir::lower::lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    );
    assert!(
        lowered.is_err(),
        "lowering now accepts a borrowed match over a user-Drop type. If that is intended, this \
         surface has left the accepted-but-unlowerable class and should become a three-engine \
         case with a drop log — see WP-C7.9-ACCEPTED-SURFACE-AUDIT.md."
    );
}

/// **An owned scrutinee is untouched by the rule.** It still moves, and its unbound components are
/// still destroyed exactly once — the `drop_unbound` path that the borrowed source must skip and
/// the owned source must keep. Getting this backwards is the most likely regression from Packet C,
/// so it is pinned as a drop log rather than as stdout.
#[test]
fn an_owned_scrutinee_still_moves_and_drops() {
    agree_completing_with_drops(
        "owned_unaffected",
        "struct Loud { tag: Int32 }\n\
         impl Drop for Loud { fn drop(&mut self) { println(\"@@stark-drop:loud@@\"); } }\n\
         enum Box2 { Full(Loud), Empty }\n\
         fn consume(b: Box2) -> Int32 { match b { Box2::Full(l) => l.tag, Box2::Empty => 0, } }\n\
         fn main() { let b = Box2::Full(Loud { tag: 9 }); println(consume(b)); }\n",
        &["loud"],
    );
}

/// The mode is decided per `match`, from that match's own scrutinee, and is **not inherited**. The
/// inner match here is over an owned temporary even though the outer one was borrowed.
#[test]
fn the_mode_is_not_inherited_by_an_inner_match() {
    agree(
        "not_inherited",
        "fn peek(w: &Wrap) -> Int32 { match *w { Wrap { h } => match *h { Holder::Val(s) => match s.len() { n => n as Int32, }, Holder::Empty => 0, }, } }\n\
         fn main() { let w = Wrap { h: Holder::Val(String::from(\"ab\")) }; println(peek(&w)); }",
        "2\n",
    );
}

/// A generic payload: the rule is decided by whether the component is `Copy`, which for a type
/// parameter is settled at the instantiation the program actually uses.
#[test]
fn a_generic_payload_follows_the_same_rule() {
    agree(
        "generic_payload",
        "enum One<T> { Only(T) }\n\
         fn peek(o: &One<String>) -> Int32 { match *o { One::Only(v) => v.len() as Int32, } }\n\
         fn main() { let o = One::Only(String::from(\"abcd\")); println(peek(&o)); }",
        "4\n",
    );
}

/// Deep nesting: three levels of borrowed projection, each deciding its own mode, ending at a
/// non-`Copy` leaf. A stale or re-rooted place shows up here rather than in the one-level cases.
#[test]
fn three_levels_of_borrowed_projection() {
    agree(
        "deep_nesting",
        "struct Outer { w: Wrap }\n\
         fn peek(o: &Outer) -> Int32 { match *o { Outer { w } => match *w { Wrap { h } => match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, }, }, } }\n\
         fn main() { let o = Outer { w: Wrap { h: Holder::Val(String::from(\"abcdefgh\")) } }; println(peek(&o)); }",
        "8\n",
    );
}
