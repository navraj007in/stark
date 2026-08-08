//! **AS3 Packet 1/2 — the receiver is MATERIALIZED by the invocation authority, and then checked.**
//!
//! `Drop::drop(&mut self)` publishes a receiver of `&mut Self`; destruction holds an **owned**
//! value. The old destructor executor bound that owned value straight to `self`, so the body's
//! `self` was a `Value::Struct` where the published type said `Ty::Ref { mutable: true }`. Nothing
//! observed it because no boundary read the receiver — the DEV-121 shape exactly.
//!
//! Collapsing the destructor into the one invocation authority made the collision unavoidable, and
//! only three repairs existed. Two are wrong: exempting `Drop` from the receiver boundary punches a
//! hole in the invariant at the hardest place to reason about, and letting `&mut T` accept an owned
//! `T` deletes the very distinction DEV-121 exists to enforce. The third is materialization: the
//! authority moves the owned value into temporary backing storage in the CALLER's frame and binds
//! `self` to a genuine `Value::Ref` into it.
//!
//! **What makes these tests evidence rather than decoration.** Since AS3 Packet 2, every receiver
//! is read against `callable_types[body].receiver` — the receiver *as the body binds it*. A
//! destructor whose `self` were still an owned value would now fail with an
//! `InternalInvariant` at `RepBoundary::Receiver`. So each destructor case below is a live
//! assertion that materialization happened, not merely that destruction still works; and the
//! `&self` / `&mut self` / by-value cases assert the other three materialization rules against the
//! same boundary.

use starkc::interp::{self, FailureClass};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn run(source: &str) -> interp::ExecutionOutcome {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "the program must type-check: {errors:?}");
    interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    )
}

/// `ok(source)` — runs and insists the program neither trapped nor tripped an invariant, then
/// returns its output. A `RepBoundary::Receiver` failure surfaces here as an `Err`, which is what
/// each case below is really testing for.
fn ok(source: &str) -> String {
    let outcome = run(source);
    let result = outcome.result;
    assert!(
        result.is_ok(),
        "the receiver boundary rejected a well-formed program: {:?}",
        result
    );
    outcome.output
}

/// **The destructor case — the one the materialization exists for.**
///
/// `&mut Self` is published; an owned value is what destruction holds. If the authority handed the
/// owned value in, this fails at `RepBoundary::Receiver` before `drop` prints anything.
#[test]
fn a_destructor_receiver_satisfies_the_published_mut_self() {
    let output = ok("\
struct Res {
    id: Int32,
}

impl Drop for Res {
    fn drop(&mut self) {
        println(\"released\");
    }
}

fn main() {
    let r = Res { id: 1 };
    println(\"before\");
}
");
    assert_eq!(output, "before\nreleased\n");
}

/// **Materialization is backing storage, not a copy.** The destructor mutates `self`, and the
/// recursive field destruction that follows must see the mutated value — which is only possible if
/// `self` referenced storage the authority can read back afterwards.
#[test]
fn a_destructor_mutation_is_visible_to_the_field_destruction_that_follows() {
    let output = ok("\
struct Inner {
    tag: Int32,
}

impl Drop for Inner {
    fn drop(&mut self) {
        println(self.tag);
    }
}

struct Outer {
    inner: Inner,
}

impl Drop for Outer {
    fn drop(&mut self) {
        self.inner.tag = 99;
        println(\"outer\");
    }
}

fn main() {
    let o = Outer { inner: Inner { tag: 1 } };
}
");
    // `Outer::drop` runs first, sets the field to 99, and only then does the recursive field
    // destruction reach `Inner::drop`. Seeing `1` here would mean the destructor mutated a copy.
    assert!(
        output.contains("99"),
        "field destruction saw the pre-drop value, so `self` was not backed by real storage: \
         {output:?}"
    );
}

/// A destructor that reads through `self` gets the value that was actually being destroyed, not
/// a default — the read-back path has a live value in it.
#[test]
fn a_destructor_reads_its_own_fields_through_the_reference() {
    let output = ok("\
struct Tagged {
    tag: Int32,
}

impl Drop for Tagged {
    fn drop(&mut self) {
        println(self.tag);
    }
}

fn main() {
    let t = Tagged { tag: 7 };
}
");
    assert!(output.contains('7'), "{output:?}");
}

/// **`&mut self` — the ordinary method form of the same rule.** The receiver binds a reference to
/// the caller's place, so the mutation is observable after the call returns.
#[test]
fn a_mut_self_method_receiver_is_a_reference_to_the_callers_place() {
    let output = ok("\
struct Counter {
    n: Int32,
}

impl Counter {
    fn bump(&mut self) {
        self.n = self.n + 1;
    }
}

fn main() {
    let mut c = Counter { n: 0 };
    c.bump();
    c.bump();
    println(c.n);
}
");
    assert!(output.contains('2'), "{output:?}");
}

/// **`&self`** — a shared borrow, published as `&Self`. It must NOT consume the caller's place:
/// the receiver is still usable afterwards.
#[test]
fn a_shared_self_receiver_does_not_consume_the_callers_place() {
    let output = ok("\
struct Holder {
    n: Int32,
}

impl Holder {
    fn peek(&self) -> Int32 {
        self.n
    }
}

fn main() {
    let h = Holder { n: 5 };
    println(h.peek());
    println(h.peek());
}
");
    assert_eq!(output, "5\n5\n", "{output:?}");
}

/// **By-value `self`** — published as `Self`, and materialized by CONSUMING the resolved place
/// (DEV-034). A non-`Copy` field moving out of the receiver is the case that distinguishes a move
/// from a clone.
#[test]
fn a_by_value_self_receiver_consumes_the_resolved_place() {
    let output = ok("\
struct Owned {
    text: String,
}

impl Owned {
    fn into_text(self) -> String {
        self.text
    }
}

fn main() {
    let o = Owned { text: String::from(\"moved\") };
    println(o.into_text());
}
");
    assert_eq!(output, "moved\n");
}

/// **The whole point, stated as one assertion.** A destructor and an ordinary `&mut self` method
/// now go through the *same* receiver boundary with the *same* rule; there is no `Drop`-shaped
/// exception anywhere in the authority. Running both in one program is the cheapest way to pin
/// that: an exemption reintroduced for either would show up as an `InternalInvariant`.
#[test]
fn destructors_and_methods_share_one_receiver_rule() {
    let outcome = run("\
struct Res {
    n: Int32,
}

impl Res {
    fn bump(&mut self) {
        self.n = self.n + 1;
    }
    fn peek(&self) -> Int32 {
        self.n
    }
}

impl Drop for Res {
    fn drop(&mut self) {
        println(self.n);
    }
}

fn main() {
    let mut r = Res { n: 0 };
    r.bump();
    println(r.peek());
}
");
    assert!(
        outcome.result.is_ok(),
        "one of the four materialization rules regressed: {:?}",
        outcome.result
    );
    assert_ne!(
        outcome.result.as_ref().err().map(|e| e.class),
        Some(FailureClass::InternalInvariant),
        "a receiver-boundary invariant fired on a well-formed program"
    );
    assert_eq!(outcome.output, "1\n1\n", "{:?}", outcome.output);
}
