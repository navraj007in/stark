//! **DEV-224: a catch-all binding on a non-`Copy` scrutinee read through a reference BORROWS it.**
//!
//! `match *a { Attr::Flag => .., other => .. }` for `a: &Attr` type-checked, ran under the
//! interpreter, and then failed at `stark build` with "native build does not yet support this
//! program: binding a non-Copy scrutinee through a shared reference". Three engines, and only the
//! lowering disagreed.
//!
//! The lowering's reasoning was that binding the whole value would move out of the borrow. It
//! would — but the binding is not by value. `04-Semantic-Analysis.md` PAT-BIND-001: when the
//! scrutinee is read through a reference, "a binding to a non-`Copy` component receives type `&C`
//! for component type `C`, borrowing the component in place; the referent is never moved." The
//! whole value is a component like any other. The type checker had always applied that rule and
//! the interpreter had always executed it; MIR simply had not implemented it, and bailed out
//! instead.
//!
//! This was also filed wrongly at first — as "an enum carrying a non-`Copy` payload cannot be
//! matched through a shared reference, even with `_` patterns". That was measured from a probe
//! holding two functions, with the failure attributed to the wrong one. The REVISED ledger entry
//! records the correction.
//!
//! # Why these negative controls
//!
//! The repair adds a borrow where there was a refusal, so the risks are (a) borrowing when the
//! value should be moved, and (b) the referent not surviving the match. Both are pinned: the
//! consuming form must still move and drop, and the borrowed scrutinee must still be usable after
//! the match that borrowed it.

mod support;

use starkc::interp;
use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn run(src: &str, tag: &str) -> String {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag}: check: {errs:?}");
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(outcome.result.is_ok(), "{tag}: run: {:?}", outcome.result);
    outcome.output
}

/// **The test that bites.** The refusal lived in MIR lowering, so an interpreter-only test passes
/// against the defect -- the interpreter always executed this correctly. Lower to MIR and verify.
fn lower(src: &str, tag: &str) -> Result<(), String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error),
        "{tag}: the program must type-check: {:?}",
        checked.diagnostics
    );
    let program = lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    )
    .map_err(|e| format!("LOWER: {}", e.what))?;
    starkc::mir::verify::verify_program(&program)
        .map(|_| ())
        .map_err(|errors| format!("VERIFY: {errors:?}"))
}

#[test]
fn the_borrowing_catch_all_lowers_to_mir() {
    let r = lower(
        "\
enum Attr { Flag, Text(String) }
fn tag(a: &Attr) -> Int64 { match *a { Attr::Flag => 0i64, _other => 1i64 } }
fn main() {
    let v = Attr::Text(String::from(\"hi\"));
    println(tag(&v));
}
",
        "dev224_lowers",
    );
    assert!(
        r.is_ok(),
        "a catch-all binding on a non-Copy scrutinee read through a reference borrows it, and \
         must lower: {r:?}"
    );
}

#[test]
fn the_consuming_catch_all_still_lowers() {
    let r = lower(
        "\
enum Attr { Flag, Text(String) }
fn main() {
    let v = Attr::Text(String::from(\"hi\"));
    match v { Attr::Flag => println(\"flag\"), _other => println(\"moved\") }
}
",
        "dev224_lowers_consuming",
    );
    assert!(r.is_ok(), "the consuming form must keep lowering: {r:?}");
}

/// The binding has reference type, which is what makes it legal: it is passed to a `&Attr`
/// parameter. If PAT-BIND-001 were not applied here this would not type-check at all.
#[test]
fn the_catch_all_binding_has_reference_type() {
    let out = run(
        "\
enum Attr { Flag, Text(String) }
fn tag(a: &Attr) -> Int64 { match *a { Attr::Flag => 0i64, other => tag2(other) } }
fn tag2(a: &Attr) -> Int64 { match *a { Attr::Flag => 0i64, _ => 1i64 } }
fn main() {
    let v = Attr::Text(String::from(\"hi\"));
    println(tag(&v));
}
",
        "dev224_ref_type",
    );
    assert!(out.contains('1'), "{out:?}");
}

/// The referent is never moved, so the value survives the match that borrowed it — twice — and is
/// still owned by its binding afterwards.
#[test]
fn the_referent_survives_the_match_that_borrowed_it() {
    let out = run(
        "\
enum Attr { Flag, Text(String) }
fn tag(a: &Attr) -> Int64 { match *a { Attr::Flag => 0i64, _other => 1i64 } }
fn main() {
    let v = Attr::Text(String::from(\"hello\"));
    println(tag(&v));
    println(tag(&v));
    match v { Attr::Text(s) => println(s.as_str()), Attr::Flag => println(\"flag\") }
}
",
        "dev224_referent_survives",
    );
    assert!(
        out.contains("hello"),
        "the payload must still be owned and readable after two borrowing matches: {out:?}"
    );
}

/// The control that carries the most weight: a CONSUMING match must still move the scrutinee into
/// the binding and drop it at arm end. A repair that borrowed in both modes would satisfy the
/// tests above and break ownership.
#[test]
fn a_consuming_catch_all_still_moves_and_drops() {
    let out = run(
        "\
struct Noisy { tag: Int64 }
impl Drop for Noisy { fn drop(&mut self) { println(\"dropped\"); } }
enum Holder { Empty, Full(Noisy) }
fn main() {
    let h = Holder::Full(Noisy { tag: 1i64 });
    match h { Holder::Empty => println(\"empty\"), _other => println(\"moved\") }
    println(\"after\");
}
",
        "dev224_consuming",
    );
    assert!(out.contains("moved"), "the consuming arm must run: {out:?}");
    assert!(
        out.contains("dropped"),
        "a consuming catch-all owns the scrutinee and must drop it at arm end: {out:?}"
    );
}

/// A `Copy` scrutinee is unaffected by PAT-BIND-001 and must still bind by value.
#[test]
fn a_copy_scrutinee_still_binds_by_value() {
    let out = run(
        "\
enum Flag { On, Off }
fn pick(f: &Flag) -> Int64 { match *f { Flag::On => 1i64, _other => 0i64 } }
fn main() { println(pick(&Flag::Off)); }
",
        "dev224_copy_scrutinee",
    );
    assert!(out.contains('0'), "{out:?}");
}
