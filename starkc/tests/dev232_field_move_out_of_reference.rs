//! **DEV-232: a non-`Copy` field may not be moved out of a shared reference.**
//!
//! ```ignore
//! struct T { v: String }
//! fn steal(t: &T) -> String { t.v }   // moves the String out of a SHARED reference
//! ```
//!
//! `stark check` accepted this. The interpreter then raised `internal compiler error: use of moved
//! or invalid field`, and `stark build` refused with `move out of the non-slot place Place {
//! local: LocalId(1), projection: [Deref, Field(0)] }`. Three engines, three behaviours, and none
//! of them a diagnostic a user could act on: a function that only *borrows* its argument destroyed
//! the caller's value.
//!
//! The rule already existed in two places. `*t` — the whole referent — was correctly rejected by
//! `check_owned_value`, and DEV-072 had implemented the same prohibition for PATTERN bindings
//! (`reject_moves_out_of_borrow`). What was missing was expression position: a field read through
//! the reference. The repair reuses DEV-072's own classifier, so the pattern and expression cases
//! cannot drift apart.
//!
//! Mirror of DEV-224, which is worth stating because the two look identical and are not. That one
//! looked like an illegal move and turned out to be a legitimate *borrow* lowering had not
//! implemented, so it was fixed by implementing the borrow. This one is a genuine move, so it is
//! fixed by rejecting it. PAT-BIND-001 separates them: a pattern binding through a reference
//! borrows; a field read in expression position moves.
//!
//! # Why these negative controls
//!
//! The repair adds a rejection to every field read, so the risk is refusing ordinary ones. A
//! `Copy` field moves nothing and must still be readable; a method call on a non-`Copy` field is a
//! receiver auto-borrow, not a move; and moving a field out of a struct you OWN is exactly what
//! ownership permits. All three are pinned, and all 65 first-party packages and consumers still
//! check clean.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn errors(src: &str, tag: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file);
    let mut out: Vec<String> = rd
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect();
    let checked = typecheck::analyze(&hir);
    out.extend(
        checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message)),
    );
    out
}

#[test]
fn a_non_copy_field_cannot_be_moved_out_of_a_shared_reference() {
    let errs = errors(
        "\
struct T { v: String }
fn steal(t: &T) -> String { t.v }
fn main() { let t = T { v: String::from(\"x\") }; println(steal(&t).as_str()); }
",
        "dev232_field_move",
    );
    assert!(
        errs.iter().any(|e| e.contains("E0100")),
        "moving a String field out of `&T` must be rejected at check time, not left to an ICE: \
         {errs:?}"
    );
}

#[test]
fn the_same_holds_through_a_mutable_reference() {
    let errs = errors(
        "\
struct T { v: String }
fn steal(t: &mut T) -> String { t.v }
fn main() { let mut t = T { v: String::from(\"x\") }; println(steal(&mut t).as_str()); }
",
        "dev232_mut_field_move",
    );
    assert!(
        !errs.is_empty(),
        "ownership cannot be moved out of a mutable reference either: {errs:?}"
    );
}

#[test]
fn a_nested_field_is_also_covered() {
    let errs = errors(
        "\
struct Inner { v: String }
struct Outer { i: Inner }
fn steal(o: &Outer) -> String { o.i.v }
fn main() {
    let o = Outer { i: Inner { v: String::from(\"x\") } };
    println(steal(&o).as_str());
}
",
        "dev232_nested_field",
    );
    assert!(
        !errs.is_empty(),
        "the reference is crossed once at the base, however deep the chain: {errs:?}"
    );
}

// -- controls -----------------------------------------------------------------------------------

#[test]
fn a_copy_field_is_still_readable_through_a_reference() {
    assert_eq!(
        errors(
            "\
struct T { v: Int64 }
fn peek(t: &T) -> Int64 { t.v }
fn main() { let t = T { v: 7i64 }; println(peek(&t)); println(t.v); }
",
            "dev232_control_copy"
        ),
        Vec::<String>::new(),
        "a Copy read moves nothing, so this must stay legal"
    );
}

#[test]
fn a_method_call_on_a_non_copy_field_is_not_a_move() {
    assert_eq!(
        errors(
            "\
struct T { v: String }
fn peek(t: &T) -> UInt64 { t.v.len() }
fn main() { let t = T { v: String::from(\"hello\") }; println(peek(&t)); }
",
            "dev232_control_method"
        ),
        Vec::<String>::new(),
        "a method receiver auto-borrows; it does not take ownership of the field"
    );
}

#[test]
fn moving_a_field_out_of_an_owned_struct_is_still_legal() {
    assert_eq!(
        errors(
            "\
struct T { v: String }
fn consume(t: T) -> String { t.v }
fn main() { let t = T { v: String::from(\"owned\") }; println(consume(t).as_str()); }
",
            "dev232_control_owned"
        ),
        Vec::<String>::new(),
        "owning the struct is exactly what permits moving the field out of it"
    );
}
