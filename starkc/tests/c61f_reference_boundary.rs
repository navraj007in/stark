//! WP-C6.1f-a — the reference-boundary negative corpus.
//!
//! Six refusals that are **conformant Core v1 behaviour** and must survive WP-C6.1f. They are
//! locked here at C6.1f-a, before any implementation begins, because a package whose job is to
//! widen reference support is exactly the context in which a correct rejection gets "fixed".
//!
//! Case 6 is the load-bearing one. Core v1 borrows bound with `let` are **lexically scoped to
//! end-of-block** (03-Type-System, "References and Lifetimes") — there is no NLL. Rust's NLL
//! *accepts* that program. So any implementation that lets generated Rust adjudicate borrow
//! validity will silently start accepting it, which `WP-C6.1f.md` §2 item 10 forbids.
//!
//! A failure here is not necessarily a regression in the checker: it may equally mean a test was
//! written against Rust intuition. Re-read the normative rule before changing either side.

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Assert the program is refused by the front end with `code`.
fn rejected_with(tag: &str, code: &str, src: &str) {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(
        pd.is_empty(),
        "{tag}: must reach the checker, got parse errors {pd:?}"
    );
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(
        rd.is_empty(),
        "{tag}: must reach the checker, got resolve errors {rd:?}"
    );
    let checked = typecheck::analyze(&hir, file);
    let codes: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .filter_map(|d| d.code.clone())
        .collect();
    assert!(
        codes.iter().any(|c| c == code),
        "{tag}: expected {code}; got {codes:?}. This refusal is conformant Core v1 — confirm the \
         normative rule before changing it (WP-C6.1f.md §2 item 10)."
    );
}

const P: &str = "struct P { v: Int32 }\n\
                 impl P { fn get(&self) -> Int32 { self.v } \
                 fn bump(&mut self) { self.v = self.v + 1; } }\n";

#[test]
fn c61f_two_mutable_borrows_of_one_owner_are_refused() {
    rejected_with(
        "two_mut",
        "E0101",
        &format!("{P}fn main() {{ let mut p = P {{ v: 3 }}; let a = &mut p; let b = &mut p; a.bump(); b.bump(); }}"),
    );
}

#[test]
fn c61f_shared_borrow_while_mutable_is_live_is_refused() {
    rejected_with(
        "shared_while_mut",
        "E0101",
        &format!("{P}fn main() {{ let mut p = P {{ v: 3 }}; let a = &mut p; let b = &p; a.bump(); assert_eq(b.get(), 4); }}"),
    );
}

#[test]
fn c61f_returning_a_reference_to_a_local_is_refused() {
    rejected_with(
        "return_local_ref",
        "E0103",
        "struct P { v: Int32 }\nfn f() -> &P { let p = P { v: 3 }; &p }\nfn main() { let q = f(); }",
    );
}

#[test]
fn c61f_moving_an_owner_while_borrowed_is_refused() {
    rejected_with(
        "move_while_borrowed",
        "E0101",
        &format!("{P}fn take(p: P) -> Int32 {{ p.v }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let r = &p; let n = take(p); assert_eq(r.get(), 3); }}"),
    );
}

#[test]
fn c61f_moving_a_drop_bearing_owner_while_borrowed_is_refused() {
    rejected_with(
        "move_drop_while_borrowed",
        "E0101",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         fn take(d: D) -> Int32 { d.v }\n\
         fn main() { let d = D { v: 1 }; let r = &d; let n = take(d); assert_eq(r.v, 1); }",
    );
}

#[test]
fn c61f_no_nll_owner_unusable_while_a_mutable_borrow_is_live() {
    // THE load-bearing case. Rust's NLL ends `r`'s borrow after its last use and accepts this;
    // Core v1 keeps it live to end-of-block, so `p.get()` conflicts. C6.1f must not "fix" this.
    rejected_with(
        "no_nll",
        "E0101",
        &format!("{P}fn main() {{ let mut p = P {{ v: 3 }}; let r = &mut p; r.bump(); assert_eq(p.get(), 4); }}"),
    );
}
