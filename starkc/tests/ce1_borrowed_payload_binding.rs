//! **CE1 — binding a non-`Copy` payload matched through a shared reference.**
//!
//! The rule this file pins:
//!
//! > When a `match` scrutinee is a place read **through a reference**, a binding to a non-`Copy`
//! > component binds by **shared reference** (`&T`) rather than by move. An owned scrutinee is
//! > unaffected and still binds by value.
//!
//! # Why this is a language decision and not a test file
//!
//! The behaviour arrived inside `b99514d`, a commit titled "stark-json: complete parser encoder
//! native path", and it silently deleted an `E0101` refusal that `gate2_valid` pinned. It is not a
//! JSON fix: it decides how every recursive enum is inspected, and without it every such type needs
//! destructive matching or hand-written accessors. It is retained deliberately (CE1), not because it
//! happened to land.
//!
//! # What the probe corrected about the normative text
//!
//! `04-Semantic-Analysis.md` §5 already said "For `&T`/`&mut T` scrutinees, bindings receive
//! shared/exclusive reference projections and never move the referent" — which looks like this rule
//! already being normative. **It is not**: a `&T` scrutinee cannot be matched with variant patterns
//! at all. `match r { Holder::Val(s) => .. }` where `r: &Holder` is rejected with a type mismatch,
//! because a pattern must name the scrutinee's normalized nominal type. The sentence describes a
//! form the language does not accept, so it governed nothing.
//!
//! The form that actually occurs is a **deref or a field read through a reference** — `match *self`,
//! `match *r`, `match w.h` where `w: &Wrap`. That is what this file pins and what the amended spec
//! rule names.

mod support;

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

const TYPES: &str = "enum Holder { Empty, Val(String) }\n\
                     struct Wrap { h: Holder }\n";

fn diagnostics(tag: &str, body: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new(tag.to_string(), format!("{TYPES}{body}\n")));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    let mut out: Vec<String> = rd
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message))
        .collect();
    if out.is_empty() {
        out = typecheck::check(&hir)
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message))
            .collect();
    }
    out
}

fn accepted(tag: &str, body: &str) {
    let errs = diagnostics(tag, body);
    assert!(errs.is_empty(), "{tag}: expected acceptance, got {errs:?}");
}

fn rejected(tag: &str, body: &str) {
    let errs = diagnostics(tag, body);
    assert!(
        !errs.is_empty(),
        "{tag}: expected a refusal, but the program was accepted"
    );
}

/// Accepted AND observed: HIR, MIR and native agree on stdout and exit status.
///
/// Acceptance alone would prove only that the type checker changed its mind. These cases prove the
/// three engines agree about what a by-reference binding *does*, which is where a binding-mode
/// change would actually go wrong.
fn agrees(tag: &str, body: &str) {
    accepted(tag, body);
    support::differential::agree_completing_available_engines(tag, &format!("{TYPES}{body}\n"));
}

// ----------------------------------------------------------------- positive --

/// The shape that motivated the change, and the one `gate2_valid` used to reject.
#[test]
fn a_non_copy_payload_through_a_deref_binds_by_reference() {
    agrees(
        "deref_self",
        "impl Holder { fn peek(&self) -> Int32 { match *self { Holder::Val(s) => 1, Holder::Empty => 0, } } }\n\
         fn main() { let h = Holder::Val(String::from(\"x\")); println(h.peek()); }",
    );
}

/// The binding really is a `&String`: a `&self` method call on it resolves, and the value it reads
/// is the referent's, not a copy's. If the binding were by value this program would be a move out
/// of a borrow; if it were a dangling projection the length would not be 2.
#[test]
fn the_binding_behaves_as_a_shared_reference() {
    agrees(
        "used_by_ref",
        "impl Holder { fn peek(&self) -> Int32 { match *self { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } } }\n\
         fn main() { let h = Holder::Val(String::from(\"xy\")); println(h.peek()); }",
    );
}

/// **Repeated inspection does not consume.** The decisive property: the same value is matched twice
/// and remains usable, which is exactly what the old `E0101` refusal made impossible.
#[test]
fn repeated_inspection_does_not_consume_the_value() {
    agrees(
        "repeated",
        "impl Holder { fn peek(&self) -> Int32 { match *self { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } } }\n\
         fn main() { let h = Holder::Val(String::from(\"xy\")); println(h.peek()); println(h.peek()); println(h.peek()); }",
    );
}

/// A deref of an ordinary `&T` parameter, not just `&self`.
#[test]
fn a_deref_of_a_reference_parameter_binds_by_reference() {
    agrees(
        "deref_param",
        "fn peek(r: &Holder) -> Int32 { match *r { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let h = Holder::Val(String::from(\"xyz\")); println(peek(&h)); }",
    );
}

/// A field read through a reference base — no explicit deref in the scrutinee at all.
#[test]
fn a_field_read_through_a_reference_binds_by_reference() {
    agrees(
        "field_through_ref",
        "fn peek(w: &Wrap) -> Int32 { match w.h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let w = Wrap { h: Holder::Val(String::from(\"xyzw\")) }; println(peek(&w)); }",
    );
}

/// **The rule is not enum-specific.** A struct pattern through a reference binds its non-`Copy`
/// fields the same way — recorded because "does this apply only to enum payloads" was an open
/// question, and the answer is no.
#[test]
fn a_struct_pattern_through_a_reference_binds_by_reference() {
    agrees(
        "struct_pattern",
        "struct Named { text: String, n: Int32 }\n\
         fn peek(v: &Named) -> Int32 { match *v { Named { text, n } => text.len() as Int32 + n, } }\n\
         fn main() { let v = Named { text: String::from(\"ab\"), n: 1 }; println(peek(&v)); }",
    );
}

/// And a tuple pattern. The mode is threaded through every sub-pattern form, so destructuring
/// depth does not change it.
#[test]
fn a_tuple_pattern_through_a_reference_binds_by_reference() {
    agrees(
        "tuple_pattern",
        "fn peek(t: &(String, Int32)) -> Int32 { match *t { (s, n) => s.len() as Int32 + n, } }\n\
         fn main() { let t = (String::from(\"abc\"), 2); println(peek(&t)); }",
    );
}

/// **Nesting needs an explicit deref at each level, and this is the precedence rule.**
///
/// The outer match binds `h: &Holder`. A binding of reference type is *not* itself matchable
/// against variant patterns (see the negative case below), so the inner match must deref it:
/// `match *h`. With that, the mode propagates and the inner payload binds `&String` in turn.
///
/// So the rule does not compose implicitly. Each match decides its own mode from its own scrutinee;
/// a by-reference binding does not carry "matched by reference" into the next match.
///
/// **Executed on every engine since WP-C7.9 Packet C.** It was front-end-only while the reference
/// interpreter could not execute it: see `the_oracle_now_binds_by_reference_like_the_other_engines`
/// below, which is the case that used to pin that divergence.
#[test]
fn nested_matching_composes_through_an_explicit_deref() {
    agrees(
        "nested",
        "fn peek(w: &Wrap) -> Int32 { match *w { Wrap { h } => match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, }, } }\n\
         fn main() { let w = Wrap { h: Holder::Val(String::from(\"abcde\")) }; println(peek(&w)); }",
    );
}

/// The counterpart: matching a by-reference *binding* directly is rejected, for the same reason a
/// `&T` parameter is. Pinned so that the deref in the case above reads as required rather than
/// stylistic.
#[test]
fn a_by_reference_binding_is_not_itself_matchable() {
    rejected(
        "nested_no_deref",
        "fn peek(w: &Wrap) -> Int32 { match *w { Wrap { h } => match h { Holder::Val(s) => 1, Holder::Empty => 0, }, } }\n\
         fn main() { let w = Wrap { h: Holder::Empty }; println(peek(&w)); }",
    );
}

/// **A `Copy` payload is unaffected**, stated explicitly rather than left to inference: it binds by
/// value, because `Copy` reads never move and wrapping them in a reference would change arithmetic
/// into a deref for no benefit.
#[test]
fn a_copy_payload_through_a_reference_still_binds_by_value() {
    agrees(
        "copy_payload",
        "enum N { Some(Int32), None }\n\
         fn peek(r: &N) -> Int32 { match *r { N::Some(v) => v + 1, N::None => 0, } }\n\
         fn main() { let n = N::Some(4); println(peek(&n)); }",
    );
}

/// **An owned scrutinee is unchanged.** The one case that would silently break every existing
/// program if the mode leaked: a by-value match must still move its payload out and be free to
/// consume it.
#[test]
fn an_owned_scrutinee_still_binds_by_value() {
    agrees(
        "owned",
        "fn eat(s: String) -> Int32 { s.len() as Int32 }\n\
         fn take(h: Holder) -> Int32 { match h { Holder::Val(s) => eat(s), Holder::Empty => 0, } }\n\
         fn main() { let h = Holder::Val(String::from(\"xy\")); println(take(h)); }",
    );
}

/// A generic payload behaves the same way — the mode is decided by the scrutinee, not by whether
/// the payload type is known concretely at the pattern.
#[test]
fn a_generic_payload_through_a_reference_binds_by_reference() {
    agrees(
        "generic",
        "enum Box2<T> { Full(T), Empty }\n\
         fn peek(r: &Box2<String>) -> Int32 { match *r { Box2::Full(s) => s.len() as Int32, Box2::Empty => 0, } }\n\
         fn main() { let b: Box2<String> = Box2::Full(String::from(\"abc\")); println(peek(&b)); }",
    );
}

/// **A `Box`-recursive enum still cannot be walked through a reference**, and this is the limit the
/// rule does *not* reach.
///
/// `Tree::Node(inner)` binds `inner: &Box<Tree>`. A recursive call wants `&Tree`, and there is no
/// coercion from `&Box<Tree>` to `&Tree` — Core v1 has no auto-deref through `Box` in argument
/// position. So the motivating use case, walking a recursive value without consuming it, works one
/// level deep and stops at the indirection every Core v1 recursive enum needs.
///
/// Recorded as a refusal rather than left out, because "recursive enums can now be inspected" is the
/// claim this rule most invites and it is not yet true in general. Not a regression: before the
/// rule, this program could not be written at all.
#[test]
fn a_box_recursive_enum_is_not_yet_walkable_through_a_reference() {
    rejected(
        "recursive",
        "enum Tree { Leaf(String), Node(Box<Tree>) }\n\
         fn depth(t: &Tree) -> Int32 { match *t { Tree::Leaf(s) => s.len() as Int32, Tree::Node(inner) => 1 + depth(inner), } }\n\
         fn main() { let t = Tree::Node(Box::new(Tree::Leaf(String::from(\"ab\")))); println(depth(&t)); }",
    );
}

// ----------------------------------- the divergence this file used to record (CLOSED) --

/// **The divergence is closed (WP-C7.9 Packet C), and this case is what closed it.**
///
/// The history, because it is the reason the case exists rather than a footnote: `b99514d` changed
/// the type checker, MIR already matched by reference (DEV-070) and so happened to follow, and the
/// HIR oracle was not touched — `match_pattern` bound `value.clone()` unconditionally, and its
/// scrutinee was itself a clone of the referent, so a binding was a *value* where the other two
/// engines had a `&T`.
///
/// For read-only use the two are observationally identical, which is why every `agrees` case above
/// passed and why this went unnoticed for a whole work package. They parted company the moment the
/// binding was used *as* a reference: dereferencing it failed in the oracle with `cannot
/// dereference non-reference`, because there was no reference there to dereference. CD-267 pinned
/// that as a refusal-to-execute rather than leaving it as a mystery failure, and escalated it,
/// because closing it meant teaching the oracle to match through a place.
///
/// Packet C did that (`PatternSource` / `match_pattern_borrowed`), so the case is now required to
/// AGREE. It is kept rather than deleted: the program that proved the divergence is the right
/// program to prove its absence.
#[test]
fn the_oracle_now_binds_by_reference_like_the_other_engines() {
    // The same program the divergence was pinned on, now required to AGREE rather than to fail.
    // The nested `match *h` is the operation that used to be impossible: `h` is a by-reference
    // binding, so dereferencing it needs a real reference, and the oracle had bound a value.
    agrees(
        "oracle_divergence",
        "fn peek(w: &Wrap) -> Int32 { match *w { Wrap { h } => match *h { Holder::Val(s) => 1, Holder::Empty => 0, }, } }\n\
         fn main() { let w = Wrap { h: Holder::Empty }; println(peek(&w)); }",
    );
}

/// The same nesting, reaching a payload that exists — so the borrowed projection is followed to a
/// real referent rather than only through the `Empty` arm.
#[test]
fn nested_borrowed_projection_reaches_a_live_payload() {
    agrees(
        "oracle_divergence_live",
        "fn peek(w: &Wrap) -> Int32 { match *w { Wrap { h } => match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, }, } }\n\
         fn main() { let w = Wrap { h: Holder::Val(String::from(\"abcde\")) }; println(peek(&w)); }",
    );
}

/// The referent is the ORIGINAL storage, not a copy of it: the value is still readable after the
/// match, and it is the same value the match saw. A binding that pointed at a temporary clone
/// would pass every read-only case above and fail this one.
#[test]
fn the_borrowed_binding_observes_the_original_storage() {
    agrees(
        "same_storage",
        "fn peek(h: &Holder) -> Int32 { match *h { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let h = Holder::Val(String::from(\"abcd\")); let first = peek(&h); let second = peek(&h); println(first + second); }",
    );
}

// ----------------------------------------------------------------- negative --

/// **Moving the referenced payload out is still rejected.** The refusal moved from `E0101` (a
/// borrow-check error) to a type mismatch — the binding simply *is* a `&String`, so passing it where
/// a `String` is wanted does not type. Recorded as a diagnostic-quality regression rather than a
/// soundness one: the program is still refused, with a less specific message.
#[test]
fn moving_the_borrowed_payload_out_is_still_rejected() {
    rejected(
        "move_out",
        "fn eat(s: String) -> Int32 { 1 }\n\
         impl Holder { fn peek(&self) -> Int32 { match *self { Holder::Val(s) => eat(s), Holder::Empty => 0, } } }\n\
         fn main() { let h = Holder::Val(String::from(\"x\")); println(h.peek()); }",
    );
}

/// Mutating through a shared-reference binding is rejected: the binding is `&String`, never
/// `&mut String`, so no mutating method resolves on it.
#[test]
fn mutation_through_a_shared_binding_is_rejected() {
    rejected(
        "mutate",
        "impl Holder { fn bump(&self) -> Int32 { match *self { Holder::Val(s) => { s.push_str(\"!\"); 1 } Holder::Empty => 0, } } }\n\
         fn main() { let h = Holder::Val(String::from(\"x\")); println(h.bump()); }",
    );
}

/// A `&T` scrutinee matched directly against variant patterns is rejected — the form the old spec
/// sentence described. It is kept as a test because the amended rule is written in terms of the
/// form that *does* occur, and this pins that the other one still does not.
#[test]
fn a_reference_typed_scrutinee_is_not_matched_against_variant_patterns() {
    rejected(
        "ref_scrutinee",
        "fn peek(r: &Holder) -> Int32 { match r { Holder::Val(s) => 1, Holder::Empty => 0, } }\n\
         fn main() { let h = Holder::Val(String::from(\"x\")); println(peek(&h)); }",
    );
}

/// **`&mut` is not inferred into an exclusive binding.** Matching through an exclusive reference
/// binds *shared* references, so a mutating method on the binding does not resolve.
///
/// This is a deliberate conservative floor, not an oversight, and it is pinned so that granting
/// exclusive bindings later is a decision someone makes rather than a behaviour that appears. What
/// it costs today: in-place mutation of a matched payload is not expressible.
#[test]
fn matching_through_an_exclusive_reference_still_binds_shared() {
    rejected(
        "mut_scrutinee_no_mutation",
        "fn bump(r: &mut Holder) -> Int32 { match *r { Holder::Val(s) => { s.push_str(\"!\"); 1 } Holder::Empty => 0, } }\n\
         fn main() { let mut h = Holder::Val(String::from(\"x\")); println(bump(&mut h)); }",
    );
}

/// Reading through an exclusive reference is accepted, though — it is only the exclusivity that is
/// withheld, not the match itself.
#[test]
fn reading_through_an_exclusive_reference_is_accepted() {
    agrees(
        "mut_scrutinee_read",
        "fn peek(r: &mut Holder) -> Int32 { match *r { Holder::Val(s) => s.len() as Int32, Holder::Empty => 0, } }\n\
         fn main() { let mut h = Holder::Val(String::from(\"xy\")); println(peek(&mut h)); }",
    );
}
