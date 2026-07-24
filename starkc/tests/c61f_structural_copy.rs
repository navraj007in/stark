//! WP-C6.1g-a — structural Copy (OWN-COPY-001, amended). A recursively-`Copy`, non-`Drop`,
//! non-owning nominal is `Copy` **without an explicit `impl Copy`**; the predicate is shared by the
//! type checker, move checker, MIR, HIR interpreter, and native backend (a divergence there is the
//! DEV-072 class). These fixtures pin the positive and negative surface, per the owner's bar.
//!
//! Positive: an all-`Copy`-field struct, nested, generic-at-Copy, and borrow-carrying nominals are
//! `Copy` — a value stays usable after `let q = p;`.
//! Negative: `String`, `Box`, `Vec`, `&mut`, a `Drop`-bearing nominal, and a mixed
//! Copy/non-Copy nominal stay `Move` — reuse after move is E0100.

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// `let q = a; use(a);` — accepted iff `a`'s type is `Copy`. Runs the front end only (the Copy
/// predicate is a front-end decision; native execution of the Copy cases is covered elsewhere).
fn reuse_after_copy_is_accepted(
    tag: &str,
    decls: &str,
    pre: &str,
    ty: &str,
    ctor: &str,
    use_expr: &str,
) {
    let src = format!(
        "{decls}fn main() {{ {pre}let a: {ty} = {ctor}; let _q = a; let _ = {use_expr}; }}"
    );
    let errs = errors(tag, &src);
    assert!(
        errs.is_empty(),
        "{tag}: `{ty}` must be Copy (reuse after copy accepted); got {errs:?}"
    );
}

/// `take(a); take(a);` — the second use must be E0100 (moved). Confirms the type stays `Move`.
fn reuse_after_move_is_rejected(tag: &str, decls: &str, pre: &str, ty: &str, ctor: &str) {
    let src = format!(
        "{decls}fn take(v: {ty}) {{}}\nfn main() {{ {pre}let a: {ty} = {ctor}; take(a); take(a); }}"
    );
    let codes = error_codes(tag, &src);
    assert!(
        codes.iter().any(|c| c == "E0100"),
        "{tag}: `{ty}` must stay Move (second use E0100); got {codes:?}"
    );
}

fn analyze(tag: &str, src: &str) -> typecheck::TypeCheckResult {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    typecheck::analyze(&hir, file)
}
fn errors(tag: &str, src: &str) -> Vec<String> {
    analyze(tag, src)
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect()
}
fn error_codes(tag: &str, src: &str) -> Vec<String> {
    analyze(tag, src)
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .filter_map(|d| d.code.clone())
        .collect()
}

// ------------------------------------------------------------------- positive --

#[test]
fn c61g_primitive_field_struct_is_copy() {
    reuse_after_copy_is_accepted(
        "point",
        "struct Pt { x: Int32, y: Int32 }\n",
        "",
        "Pt",
        "Pt { x: 1, y: 2 }",
        "a.x",
    );
}

#[test]
fn c61g_nested_all_copy_struct_is_copy() {
    reuse_after_copy_is_accepted(
        "nested",
        "struct Inner { v: Int32 }\nstruct Outer { i: Inner, n: Int32 }\n",
        "",
        "Outer",
        "Outer { i: Inner { v: 1 }, n: 2 }",
        "a.n",
    );
}

#[test]
fn c61g_generic_at_a_copy_argument_is_copy() {
    reuse_after_copy_is_accepted(
        "gen_copy",
        "struct H<T> { r: T }\n",
        "",
        "H<Int32>",
        "H { r: 5 }",
        "a.r",
    );
}

#[test]
fn c61g_borrow_carrying_nominal_is_copy() {
    // A shared reference is Copy, so a nominal whose only field is one is structurally Copy.
    reuse_after_copy_is_accepted(
        "gen_ref",
        "struct H<T> { r: T }\n",
        "let x: Int32 = 7; ",
        "H<&Int32>",
        "H { r: &x }",
        "a.r",
    );
}

#[test]
fn c61g_all_copy_field_enum_is_copy() {
    reuse_after_copy_is_accepted(
        "enum_copy",
        "enum E { A, B(Int32) }\n",
        "",
        "E",
        "E::B(3)",
        "0",
    );
}

// ------------------------------------------------------------------- negative --

#[test]
fn c61g_string_field_stays_move() {
    reuse_after_move_is_rejected(
        "string",
        "struct S { s: String }\n",
        "",
        "S",
        "S { s: String::new() }",
    );
}

#[test]
fn c61g_vec_field_stays_move() {
    reuse_after_move_is_rejected(
        "vec",
        "struct S { v: Vec<Int32> }\n",
        "",
        "S",
        "S { v: Vec::new() }",
    );
}

#[test]
fn c61g_box_field_stays_move() {
    reuse_after_move_is_rejected(
        "box",
        "struct S { b: Box<Int32> }\n",
        "",
        "S",
        "S { b: Box::new(1) }",
    );
}

#[test]
fn c61g_mutable_reference_field_stays_move() {
    // Instantiate a generic at `&mut P`: exclusive references are never Copy, so the nominal is Move.
    reuse_after_move_is_rejected(
        "mut_ref",
        "struct P { v: Int32 }\nstruct H<T> { r: T }\n",
        "let mut p = P { v: 7 }; ",
        "H<&mut P>",
        "H { r: &mut p }",
    );
}

#[test]
fn c61g_drop_bearing_nominal_stays_move() {
    reuse_after_move_is_rejected(
        "drop",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) {} }\n",
        "",
        "D",
        "D { v: 1 }",
    );
}

#[test]
fn c61g_mixed_copy_and_non_copy_fields_stays_move() {
    reuse_after_move_is_rejected(
        "mixed",
        "struct M { n: Int32, s: String }\n",
        "",
        "M",
        "M { n: 1, s: String::new() }",
    );
}
