//! WP-C6.1f — returning a reference. The last of the five ephemeral-lane checks with real
//! semantics behind it: OWN-RETURN-001's provenance and shortest-input-lifetime rules.
//!
//! Provenance is the front end's: OWN-RETURN-001 rules 2/3 reject (E0103) a returned reference not
//! derived from a reference parameter, so the backend does not re-check it — it just emits.
//!
//! Two backend mechanisms make the emission compile:
//!   * **Definite assignment.** A reference that is a `Call` destination or an `if`/`match` join
//!     result is written in one basic block and read in another; the generated block-dispatch
//!     `loop { match … }` hides that from rustc, which rejects it E0381. Such a reference temporary
//!     — detected as spanning more than one block — is `Option<&T>`-backed, exactly as a stored
//!     user reference is (b3). Parameters (initialised at entry) and same-block ephemeral temporaries
//!     stay bare.
//!   * **Lifetimes.** `fn pick(a: &T, b: &T) -> &T` needs an explicit lifetime (E0106) once there
//!     are two or more reference parameters; a single shared `'a` on every reference parameter and
//!     the return soundly encodes the *shortest of all inputs* (03 rule 3). Zero or one reference
//!     parameter is handled by Rust's own elision.
//!
//! The shared-`'a` encoding is conservative: for `pick(a, b) -> a` STARK's shortest is `a`'s
//! lifetime alone, but `'a` also ties it to `b`. Sound — it never accepts a program STARK rejects —
//! though it can reject a valid one whose return provably derives from a longer-lived subset.

mod support;

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Delegates to the shared comparator (R-02). Was a private helper that ran three engines and
/// asserted `status == 0` on each separately -- which let three engines each exit 0 while printing
/// three different things.
fn agree(tag: &str, src: &str) {
    support::differential::agree_completing_available_engines(tag, src);
}

/// A returned reference not derived from a parameter must stay rejected (E0103, OWN-RETURN-001).
fn rejected_e0103(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("ret_neg_{tag}.stark"),
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
        "{tag}: expected E0103, got {codes:?}"
    );
}

const P: &str = "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c61f_return_a_parameter_reference() {
    agree(
        "ident",
        &format!(
            "{P}fn f(r: &P) -> &P {{ r }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let q = f(&p); assert_eq(q.get(), 3); }}"
        ),
    );
}

#[test]
fn c61f_return_a_reference_to_a_parameter_field() {
    agree(
        "field",
        "struct P { v: Int32 }\nfn f(r: &P) -> &Int32 { &r.v }\n\
         fn main() { let p = P { v: 3 }; assert_eq(*f(&p), 3); }",
    );
    agree(
        "method_returns_field_ref",
        "struct W { v: Int32 }\nimpl W { fn get(&self) -> &Int32 { &self.v } }\n\
         fn main() { let w = W { v: 5 }; assert_eq(*w.get(), 5); }",
    );
}

#[test]
fn c61f_return_a_mutable_reference() {
    agree(
        "mut",
        &format!("{P}fn f(r: &mut P) -> &mut P {{ r }}\n\
                  fn main() {{ let mut p = P {{ v: 3 }}; let q = f(&mut p); assert_eq(q.get(), 3); }}"),
    );
}

#[test]
fn c61f_return_with_two_reference_params_uses_a_shared_lifetime() {
    // E0106 without the shared 'a: two reference params leave the output lifetime ambiguous.
    agree(
        "two_params",
        "struct P { v: Int32 }\nfn pick(a: &P, b: &P) -> &P { a }\n\
         fn main() { let p = P { v: 3 }; let q = P { v: 9 }; assert_eq(pick(&p, &q).v, 3); }",
    );
}

#[test]
fn c61f_return_a_reference_from_an_if_expression() {
    // The if-join result is a cross-block reference temp — Option-backed like a Call destination.
    agree(
        "if_branch",
        "struct P { v: Int32 }\nfn pick(a: &P, b: &P) -> &P { if a.v > 0 { a } else { b } }\n\
         fn main() { let p = P { v: 3 }; let q = P { v: 9 }; assert_eq(pick(&p, &q).v, 3); }",
    );
}

#[test]
fn c61f_return_a_reference_through_a_call_chain() {
    agree(
        "chain",
        "struct P { v: Int32 }\nfn id(r: &P) -> &P { r }\nfn f(r: &P) -> &P { id(r) }\n\
         fn main() { let p = P { v: 3 }; assert_eq(f(&p).v, 3); }",
    );
}

#[test]
fn c61f_project_directly_through_a_returned_reference() {
    // `f(&p).field` / `f(&p).method()` with no intervening `let`: the call result is materialised
    // into a temp and projected through it.
    agree(
        "project_field",
        &format!(
            "{P}fn f(r: &P) -> &P {{ r }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; assert_eq(f(&p).v, 3); }}"
        ),
    );
    agree(
        "project_method",
        &format!(
            "{P}fn f(r: &P) -> &P {{ r }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; assert_eq(f(&p).get(), 3); }}"
        ),
    );
}

#[test]
fn c61f_returning_a_reference_to_a_local_is_still_rejected() {
    rejected_e0103(
        "local",
        "struct P { v: Int32 }\nfn f() -> &P { let p = P { v: 3 }; &p }\nfn main() { let q = f(); }",
    );
    rejected_e0103(
        "local_field",
        "struct P { v: Int32 }\nfn f() -> &Int32 { let p = P { v: 3 }; &p.v }\nfn main() { let q = f(); }",
    );
}
