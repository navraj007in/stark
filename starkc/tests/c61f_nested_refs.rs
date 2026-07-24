//! WP-C6.1f-b4/b5 — nested reference syntax, and scoping the return-escape check.
//!
//! **b4.** `&&T` as a type and `**x` as an expression were unspellable: the lexer emits `&&` and
//! `**` as single tokens (logical AND, exponentiation) and the parser never split them in type or
//! prefix position. The **normative grammar already permits both** — `Type ::= '&' Type` and
//! `UnaryExpression ::= '*' UnaryExpression` — so the parser was refusing what the spec admits.
//! Nested references are also required by TYPE-METHOD-002, whose auto-dereference "repeatedly
//! removes one leading `&`/`&mut`", which presupposes they can be written. `&&mut T` binds the
//! `mut` to the INNER reference: a shared reference to a mutable one.
//!
//! **b5.** `check_return_escape` ran on EVERY block tail, so `&p` as an `if`-branch tail was
//! reported as E0103 "cannot return reference to local stack variable" — a wrong diagnosis (nothing
//! is returned) and an over-rejection that fired even when both branches borrowed the SAME owner.
//! OWN-CARRY-001 explicitly contemplates the shape: a control-flow merge "carries the union of
//! possible source referents". The check now runs once on the FUNCTION BODY's tail (plus every
//! `return`), which is where it belongs — `borrowed_local` recurses through `Block`/`If`/`Match`,
//! so a reference that really does reach the return is still found.
//!
//! The escape negatives below are load-bearing: b5 REMOVES a rejection, and an earlier attempt that
//! dropped the check entirely (rather than relocating it) silently disabled escape detection for
//! ordinary returns. Four lib tests caught it; these keep it caught.

use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Accepted by the front end, and agreeing across the HIR oracle and the MIR interpreter.
///
/// Deliberately NOT a native assertion: two of these shapes still hit generated-code limits
/// (`E0716` for a nested reference bound to a local, `E0502` for a merged borrow bound to one),
/// both in the `ValueSlot`-versus-Rust-borrow-region family tracked in `C6-REFERENCE-MATRIX.md`.
/// What b4/b5 fixed is the front end and MIR, and that is what is pinned here.
fn accepted(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(format!("nr_{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} must PARSE: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag} typecheck: {errs:?}");
    let hir_exec = interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
}

fn rejected_e0103(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("nr_neg_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file);
    let codes: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .filter_map(|d| d.code.clone())
        .collect();
    assert!(
        codes.iter().any(|c| c == "E0103"),
        "{tag}: a reference to a local that reaches the return must be E0103; got {codes:?}"
    );
}

const P: &str = "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n";

// ------------------------------------------------------------------------ b4 --

#[test]
fn c61f_b4_nested_reference_type_is_spellable() {
    accepted(
        "double_ref_param",
        &format!(
            "{P}fn f(r: &&P) -> Int32 {{ r.get() }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let q = &p; assert_eq(f(&q), 3); }}"
        ),
    );
    accepted(
        "double_ref_local",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let a = &p; let b: &&P = &a; \
                  assert_eq(b.get(), 3); }}"
        ),
    );
}

#[test]
fn c61f_b4_shared_reference_to_a_mutable_one() {
    // `&&mut T` is `&(&mut T)` — the `mut` binds to the inner reference.
    accepted(
        "ref_to_mut_ref",
        &format!(
            "{P}impl P {{ fn bump(&mut self) {{ self.v = self.v + 1; }} }}\n\
                  fn f(r: &&mut P) -> Int32 {{ r.get() }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; let a = &mut p; let b = &a; \
                  assert_eq(f(b), 3); }}"
        ),
    );
}

#[test]
fn c61f_b4_double_dereference_is_spellable() {
    accepted(
        "double_deref",
        "fn main() { let x: Int32 = 5; let a = &x; let b = &a; assert_eq(**b, 5); }",
    );
    accepted(
        "double_deref_param",
        "fn f(r: &&Int32) -> Int32 { **r }\n\
         fn main() { let x: Int32 = 5; let a = &x; assert_eq(f(&a), 5); }",
    );
    accepted(
        "double_ref_expression",
        "fn f(r: &&Int32) -> Int32 { **r }\nfn main() { let x: Int32 = 5; assert_eq(f(&&x), 5); }",
    );
}

#[test]
fn c61f_b4_binary_and_and_power_are_unaffected() {
    // The same tokens the parser now splits in prefix position must still parse as BINARY
    // operators, which is why they were single tokens in the first place.
    accepted(
        "binary_and",
        "fn main() { let a: Bool = true; let b: Bool = false; assert(a && !b); }",
    );
    accepted(
        "binary_pow",
        "fn main() { let a: Int32 = 2; assert_eq(a ** 3, 8); }",
    );
    accepted(
        "binary_and_with_refs",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; assert(*(&x) < *(&y) && x < y); }",
    );
}

// ------------------------------------------------------------------------ b5 --

#[test]
fn c61f_b5_a_borrow_from_a_branch_is_not_a_return() {
    accepted(
        "if_branch_two_owners",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let q = P {{ v: 9 }}; \
                  let r = if p.v > 0 {{ &p }} else {{ &q }}; assert_eq(r.get(), 3); }}"
        ),
    );
    // Used to be rejected even though BOTH branches borrow the same owner.
    accepted(
        "if_branch_same_owner",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; \
                  let r = if p.v > 0 {{ &p }} else {{ &p }}; assert_eq(r.get(), 3); }}"
        ),
    );
    accepted(
        "match_branch",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let q = P {{ v: 9 }}; \
                  let r = match p.v {{ 0 => &q, _ => &p }}; assert_eq(r.get(), 3); }}"
        ),
    );
    accepted(
        "nested_block",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let r = {{ &p }}; assert_eq(r.get(), 3); }}"
        ),
    );
}

#[test]
fn c61f_b5_genuine_escapes_are_still_rejected() {
    rejected_e0103(
        "return_local",
        "struct P { v: Int32 }\nfn f() -> &P { let p = P { v: 3 }; &p }\nfn main() { let q = f(); }",
    );
    rejected_e0103(
        "return_local_field",
        "struct P { v: Int32 }\nfn f() -> &Int32 { let p = P { v: 3 }; &p.v }\n\
         fn main() { let q = f(); }",
    );
    // Through an `if` branch that really does reach the return.
    rejected_e0103(
        "return_local_via_if",
        "struct P { v: Int32 }\n\
         fn f(c: Bool, o: &P) -> &P { if c { let p = P { v: 3 }; &p } else { o } }\n\
         fn main() { let z = P { v: 1 }; let q = f(true, &z); }",
    );
    // Through a nested block that is the function's tail.
    rejected_e0103(
        "return_local_via_block",
        "struct P { v: Int32 }\nfn f() -> &P { { let p = P { v: 3 }; &p } }\n\
         fn main() { let q = f(); }",
    );
    // Through a `return` STATEMENT rather than a tail — a separate code path.
    rejected_e0103(
        "return_statement",
        "struct P { v: Int32 }\n\
         fn f(c: Bool, o: &P) -> &P { if c { let p = P { v: 3 }; return &p; } o }\n\
         fn main() { let z = P { v: 1 }; let q = f(true, &z); }",
    );
    // A borrow-carrying aggregate returned from a local (OWN-CARRY-001 is structural).
    rejected_e0103(
        "return_local_in_tuple",
        "fn f() -> (&Int32,) { let x = 1; (&x,) }\nfn main() { let t = f(); }",
    );
}
