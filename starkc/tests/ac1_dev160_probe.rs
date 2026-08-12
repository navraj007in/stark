//! **WP-ARCH-CLOSE AC1 — the DEV-160 architecture probe, baseline.**
//!
//! AC1 uses DEV-160 as a live probe of whether the existing borrow architecture can absorb a
//! substantial semantic capability extension without architectural violence. This file is the
//! probe's **baseline**: it pins what the compiler does today, so that a later repair is measured
//! against a recorded starting point rather than against a memory of one.
//!
//! It asserts no capability. Every test here records a REFUSAL and requires it to be STARK's own,
//! by name, carrying the sub-id from the owner's 2026-08-03 taxonomy. When AC1's repair lands, the
//! cases that gain the capability move to assertions that they BUILD AND RUN, and the ones that
//! remain refused keep their guard here.
//!
//! **Why a baseline is worth its own file.** The first attempt at this repair (recorded in
//! `KNOWN-DEVIATIONS.md` under DEV-160) over-refused `stark_http_client::follow` and was caught by
//! a package build rather than by a test. The failure mode is not "the reproducer stops working" —
//! it is "an unrelated program that used to build stops building". A baseline that records the
//! accepted set is the control for that.

mod support;

use starkc::backend::generated_rust::emit_program;
use starkc::backend::version::build_versions;
use starkc::layout::TargetLayout;
use starkc::mir::lower::lower_program;

/// Emits the program, returning the backend's refusal text, or `None` if it emitted.
///
/// **Lowering must succeed.** These are programs STARK accepts — the whole point of the deviation
/// is that a valid program is refused by the backend, so a lowering failure here would mean the
/// probe stopped testing what it was written for.
fn refusal_for(name: &str, source: &str) -> Option<String> {
    let front = support::differential::front_end(name, source);
    let program =
        lower_program(&front.hir, &front.tables, front.file.clone()).unwrap_or_else(|e| {
            panic!(
                "{name} must LOWER; it is the BACKEND that refuses it: {} @ {:?}",
                e.what, e.span
            )
        });
    let versions = build_versions(
        "0.0.0-test".to_string(),
        "x86_64-unknown-linux-gnu".to_string(),
        starkc::backend::generated_rust::Profile::Debug,
    );
    match emit_program::emit(&program, &versions, &TargetLayout::stark64_v1()) {
        Err(refusal) => Some(format!("{refusal:?}")),
        Ok(_) => None,
    }
}

/// **The DEV-160b shape, exactly as reported: a borrow reaching the call through an earlier block.**
///
/// `r.url.as_str()` is a call, and a call ends a MIR block — so the `&str` it returns is produced in
/// one block and consumed in the next, beside a move out of a sibling field. The thunk absorbs
/// evaluation within the call's own block only, so it cannot take this over.
///
/// This is the capability half of DEV-160 and the program is valid STARK. Today it is refused; the
/// test asserts the refusal is STARK's own and says which sub-id owns it, because the alternative —
/// the state before the rustc leak was sealed — was `E0502` inside generated code the user never
/// wrote.
#[test]
fn cross_block_shared_borrow_beside_a_sibling_move_is_refused_by_name() {
    let refusal = refusal_for(
        "ac1_crossblock_shared.stark",
        r#"
struct Req { url: String, body: String }

fn send(u: &str, b: String) -> UInt64 {
    u.len() + b.len()
}

fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let n = send(r.url.as_str(), r.body);
    if n != 5u64 { panic("bad"); }
    println("OK");
}
"#,
    )
    .expect(
        "the native backend must refuse this today; emitting it produces E0502 in generated code",
    );

    assert!(
        refusal.contains("DEV-160b"),
        "the refusal must name the sub-id, not the family: three shapes are deferred and a reader \
         hitting one needs to know which. Got: {refusal}"
    );
    assert!(
        refusal.contains("through an earlier block"),
        "the refusal must describe the program's shape rather than the backend's inconvenience: \
         {refusal}"
    );
}

/// The same shape with a trailing read of another field, which the ledger records as the second
/// leak that became a named refusal. Kept distinct because it exercises a different absorption
/// question: the aggregate is still live after the call.
#[test]
fn cross_block_borrow_with_a_trailing_field_read_is_refused_by_name() {
    let refusal = refusal_for(
        "ac1_crossblock_trailing.stark",
        r#"
struct Req { url: String, body: String, tag: UInt32 }

fn send(u: &str, b: String) -> UInt64 {
    u.len() + b.len()
}

fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de"), tag: 7u32 };
    let n = send(r.url.as_str(), r.body);
    if n != 5u64 { panic("bad"); }
    if r.tag != 7u32 { panic("tag"); }
    println("OK");
}
"#,
    )
    .expect("the native backend must refuse this today");

    assert!(
        refusal.contains("DEV-160b") || refusal.contains("DEV-160d"),
        "a cross-block borrow, or a borrow outliving the call: either is a named deferral, but it \
         must be named. Got: {refusal}"
    );
}

/// **The control that caught the first repair attempt, in miniature.**
///
/// `follow` binds the fields to locals FIRST and then calls. After `let url = builder.url;` the
/// move has transferred ownership, so borrowing `url` does not borrow `builder` — and a provenance
/// rule that propagated through moves made this look like a conflict and refused a program that
/// had always built.
///
/// **This test must keep BUILDING through any AC1 repair.** It is the over-refusal guard, and it is
/// the shape a real first-party package uses.
#[test]
fn fields_bound_to_locals_first_still_emit() {
    let refusal = refusal_for(
        "ac1_fields_to_locals.stark",
        r#"
struct Req { url: String, body: String }

fn send(u: &str, b: String) -> UInt64 {
    u.len() + b.len()
}

fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let url = r.url;
    let body = r.body;
    let n = send(url.as_str(), body);
    if n != 5u64 { panic("bad"); }
    println("OK");
}
"#,
    );
    assert!(
        refusal.is_none(),
        "a move SEVERS provenance: after `let url = r.url;` borrowing `url` does not borrow `r`. \
         Refusing this is the over-refusal that broke `stark_http_client::follow` on the first \
         repair attempt. Got: {refusal:?}"
    );
}

/// The same-block shape DEV-160a already closed, kept here as the positive control: the thunk
/// mechanism works, so a cross-block refusal is about absorption reach and not about the mechanism
/// being broken.
#[test]
fn the_same_block_shape_still_emits() {
    let refusal = refusal_for(
        "ac1_same_block.stark",
        r#"
struct Req { url: String, body: String }

fn send(u: &String, b: String) -> UInt64 {
    u.len() + b.len()
}

fn main() {
    let r = Req { url: String::from("abc"), body: String::from("de") };
    let n = send(&r.url, r.body);
    if n != 5u64 { panic("bad"); }
    println("OK");
}
"#,
    );
    assert!(
        refusal.is_none(),
        "DEV-160a is CLOSED: a same-block disjoint borrow and sibling move must emit. {refusal:?}"
    );
}
