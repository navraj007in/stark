//! **DEV-209 — a prelude `Option`/`Result` payload is a PLACE, like every other component.**
//!
//! PAT-BIND-001 is uniform: when a scrutinee is read through a reference, a binding to a non-`Copy`
//! component receives `&C`, borrowing the component **in place**, and the referent is never moved.
//! The rule is stated over variant payloads, struct fields and tuple elements alike.
//!
//! The HIR oracle stored a prelude payload as `Box<Value>`, which no `Projection` could name — so
//! the borrowed matcher fell back to the owned rule and **moved out of a borrow**. The checker
//! published `&String`; the oracle bound `String`. MIR executed the same program correctly, so the
//! oracle was the outlier, and the specialised representation was what failed to preserve a
//! language-level place.
//!
//! ```text
//! Some / Ok / Err
//!  └── payload slot
//!       ├── Some(Value)   live
//!       └── None          moved out
//! ```
//!
//! That is the same slot model user-enum payloads already used. This is not a second place system;
//! it removes an exception from the existing one.
//!
//! **Found by `RepBoundary::MatchBinding`** — on `stark-url`, a first-party package, whose three
//! failing tests are left unchanged as application witnesses. Changing valid code to avoid a
//! compiler defect would turn "an application exposed a missing capability" into "an application
//! learned a workaround".

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn outcome(source: &str) -> interp::ExecutionOutcome {
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
        .map(|d| d.message.clone())
        .collect();
    assert!(errors.is_empty(), "typecheck: {errors:?}");
    interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    )
}

fn output(source: &str) -> String {
    let outcome = outcome(source);
    assert!(outcome.result.is_ok(), "{:?}", outcome.result);
    outcome.output
}

// ---------------------------------------------------------------------- borrow semantics --

/// The exact shape `stark-url` hit, and the one the `MatchBinding` boundary reported.
#[test]
fn a_borrowed_option_payload_binds_by_reference() {
    assert_eq!(
        output(
            "fn main() { let o: Option<String> = Some(String::from(\"x\")); let r = &o; \
             match *r { Some(s) => println(s), None => println(\"n\") } }"
        ),
        "x\n"
    );
}

#[test]
fn a_borrowed_result_ok_payload_binds_by_reference() {
    assert_eq!(
        output(
            "fn main() { let r: Result<String, Int32> = Ok(String::from(\"x\")); let b = &r; \
             match *b { Ok(s) => println(s), Err(_e) => println(\"e\") } }"
        ),
        "x\n"
    );
}

#[test]
fn a_borrowed_result_err_payload_binds_by_reference() {
    assert_eq!(
        output(
            "fn main() { let r: Result<Int32, String> = Err(String::from(\"x\")); let b = &r; \
             match *b { Ok(_n) => println(\"o\"), Err(e) => println(e) } }"
        ),
        "x\n"
    );
}

/// **The referent is never moved** — the reason the rule exists. The original is still usable
/// after the match, which a by-value binding would have made impossible.
///
/// The borrow is confined to a nested block because a `let`-bound borrow is lexically scoped to
/// end-of-block in Core v1: leaving `r` live would make the second match a move-while-borrowed,
/// which the borrow checker rejects before any of this could be observed.
#[test]
fn the_borrowed_payload_is_not_moved_out() {
    assert_eq!(
        output(
            "fn main() { let value = Some(String::from(\"abc\")); \
             { let r = &value; match *r { Some(s) => println(s.as_str()), None => println(\"n\") } } \
             match value { Some(t) => println(t), None => println(\"n\") } }"
        ),
        "abc\nabc\n",
        "the second match proves the first did not consume the payload"
    );
}

/// A `Copy` payload still binds BY VALUE — the rule is about what cannot be taken from a referent,
/// not about references for their own sake.
#[test]
fn a_copy_payload_still_binds_by_value() {
    assert_eq!(
        output(
            "fn main() { let o: Option<Int32> = Some(7); let r = &o; \
             match *r { Some(n) => println(n + 1), None => println(0) } }"
        ),
        "8\n"
    );
}

/// PAT-BIND-001's floor: an exclusive source still gives a SHARED binding for a non-`Copy`
/// component, so the match cannot move out of borrowed storage.
#[test]
fn an_exclusive_source_still_binds_shared() {
    assert_eq!(
        output(
            "fn main() { let mut o: Option<String> = Some(String::from(\"x\")); let r = &mut o; \
             match *r { Some(s) => println(s), None => println(\"n\") } }"
        ),
        "x\n"
    );
}

// ------------------------------------------------------------------ prelude/user parity --

/// **The parity control.** DEV-205 and DEV-209 both arose because the specialised prelude path was
/// less complete than the user-enum path. This asserts the two agree for the same shapes, which is
/// more useful than another one-off regression case.
#[test]
fn prelude_and_user_enum_payloads_bind_identically() {
    let user_noncopy = output(
        "enum E { A(String), B } fn main() { let e = E::A(String::from(\"x\")); let r = &e; \
         match *r { E::A(s) => println(s), E::B => println(\"b\") } }",
    );
    let prelude_noncopy = output(
        "fn main() { let o: Option<String> = Some(String::from(\"x\")); let r = &o; \
         match *r { Some(s) => println(s), None => println(\"n\") } }",
    );
    assert_eq!(user_noncopy, prelude_noncopy, "non-Copy payload");

    let user_copy = output(
        "enum E { A(Int32), B } fn main() { let e = E::A(7); let r = &e; \
         match *r { E::A(n) => println(n + 1), E::B => println(0) } }",
    );
    let prelude_copy = output(
        "fn main() { let o: Option<Int32> = Some(7); let r = &o; \
         match *r { Some(n) => println(n + 1), None => println(0) } }",
    );
    assert_eq!(user_copy, prelude_copy, "Copy payload");
}

// ------------------------------------------------------------------------- consumption --

/// The operations that legitimately EMPTY the slot: `Some(v) -> None` is how a move is
/// represented, and only intentionally consuming code may do it.
#[test]
fn consuming_operations_still_work() {
    assert_eq!(
        output("fn main() { let o: Option<Int32> = Some(7); println(o.unwrap()); }"),
        "7\n"
    );
    assert_eq!(
        output("fn main() { let o: Option<Int32> = None; println(o.unwrap_or(3)); }"),
        "3\n"
    );
    assert_eq!(
        output("fn main() { let r: Result<Int32, Int32> = Ok(7); println(r.unwrap()); }"),
        "7\n"
    );
    assert_eq!(
        output(
            "fn double(n: Int32) -> Int32 { n * 2 } \
             fn main() { let o: Option<Int32> = Some(7); \
             match o.map(double) { Some(n) => println(n), None => println(0) } }"
        ),
        "14\n"
    );
    assert_eq!(
        output(
            "fn get() -> Result<Int32, Int32> { Ok(7) } \
             fn run() -> Result<Int32, Int32> { let n = get()?; Ok(n + 1) } \
             fn main() { match run() { Ok(n) => println(n), Err(_e) => println(0) } }"
        ),
        "8\n"
    );
}

/// An OWNED match still moves the payload out, which is the other half of the slot model.
#[test]
fn an_owned_match_still_moves_the_payload() {
    assert_eq!(
        output(
            "fn main() { let o: Option<String> = Some(String::from(\"x\")); \
             match o { Some(s) => println(s), None => println(\"n\") } }"
        ),
        "x\n"
    );
}

// ------------------------------------------------------------------------- observation --

/// Rendering reads the payload without consuming it, so it must find a live slot.
#[test]
fn display_reads_a_live_payload() {
    assert_eq!(
        output("fn main() { let o: Option<Int32> = Some(7); println(o); }"),
        "Some(7)\n"
    );
    assert_eq!(
        output("fn main() { let o: Option<Int32> = None; println(o); }"),
        "None\n"
    );
    assert_eq!(
        output("fn main() { let r: Result<Int32, Int32> = Ok(7); println(r); }"),
        "Ok(7)\n"
    );
    assert_eq!(
        output("fn main() { let r: Result<Int32, Int32> = Err(9); println(r); }"),
        "Err(9)\n"
    );
}

// --------------------------------------------------------------------------- lifecycle --

/// A non-`Copy` payload is destroyed exactly once when its container goes out of scope — the
/// property most at risk from a representation change that adds an inner slot.
#[test]
fn a_payload_is_destroyed_exactly_once() {
    assert_eq!(
        output(
            "struct R { id: Int32 } \
             impl Drop for R { fn drop(&mut self) { println(\"released\"); } } \
             fn main() { let o: Option<R> = Some(R { id: 1 }); println(\"before\"); }"
        ),
        "before\nreleased\n"
    );
}

/// A payload MOVED out by an owned match is destroyed once — by its new owner, not twice.
#[test]
fn a_moved_payload_is_not_destroyed_twice() {
    assert_eq!(
        output(
            "struct R { id: Int32 } \
             impl Drop for R { fn drop(&mut self) { println(\"released\"); } } \
             fn main() { let o: Option<R> = Some(R { id: 1 }); \
             match o { Some(r) => println(\"took\"), None => println(\"none\") } \
             println(\"end\"); }"
        ),
        "took\nreleased\nend\n"
    );
}

/// A payload that a BORROWED match merely inspected is still destroyed once, by its original
/// owner — the case the old by-value fallback got wrong in the other direction.
#[test]
fn a_borrowed_payload_is_still_destroyed_by_its_owner() {
    assert_eq!(
        output(
            "struct R { id: Int32 } \
             impl Drop for R { fn drop(&mut self) { println(\"released\"); } } \
             fn main() { let o: Option<R> = Some(R { id: 1 }); let b = &o; \
             match *b { Some(r) => println(\"saw\"), None => println(\"none\") } \
             println(\"end\"); }"
        ),
        "saw\nend\nreleased\n"
    );
}
