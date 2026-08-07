//! WP-C6.2b-F5 — a bound on an impl head is visible in the impl's method bodies.
//!
//! WP-C6-ENTRY §2 carry-forward. A method call on a bounded generic *function* parameter already
//! resolved through the bound (`fn f<T: Sh>(t: T) { t.a() }`), but a bound written on the IMPL head
//! (`impl<T: Sh> W<T> { fn go(&self) { self.v.a() } }`) was invisible in the body — E0302 "method
//! 'a' not found for type 'T'". `typecheck` now tracks `current_impl_generics` alongside
//! `current_fn_generics` and consults both when resolving a method on a `Ty::Param` receiver.

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

/// HIR + MIR + (if rustc) native all complete at exit 0.
const SH: &str = "trait Sh { fn a(&self) -> Int32; }\nstruct S { n: Int32 }\n\
                  impl Sh for S { fn a(&self) -> Int32 { self.n } }\n";

#[test]
fn c62b_f5_impl_head_bound_is_visible_in_a_method_body() {
    agree(
        "impl_head_bound",
        &format!(
            "{SH}struct W<T> {{ v: T }}\n\
             impl<T: Sh> W<T> {{ fn go(&self) -> Int32 {{ self.v.a() }} }}\n\
             fn main() {{ let w = W {{ v: S {{ n: 4 }} }}; assert_eq(w.go(), 4); }}"
        ),
    );
}

#[test]
fn c62b_f5_impl_head_bound_with_a_by_value_receiver() {
    agree(
        "impl_head_byvalue",
        &format!(
            "{SH}struct W<T> {{ v: T }}\n\
             impl<T: Sh> W<T> {{ fn go(self) -> Int32 {{ self.v.a() }} }}\n\
             fn main() {{ let w = W {{ v: S {{ n: 5 }} }}; assert_eq(w.go(), 5); }}"
        ),
    );
}

#[test]
fn c62b_f5_method_and_impl_generics_both_in_scope() {
    // The method has its own generic too; both the impl-head bound and the method generic resolve.
    agree(
        "both_generics",
        &format!(
            "{SH}struct W<T> {{ v: T }}\n\
             impl<T: Sh> W<T> {{ fn combine<U>(&self, x: U) -> U {{ let _ = self.v.a(); x }} }}\n\
             fn main() {{ let w = W {{ v: S {{ n: 4 }} }}; let n: Int32 = 9; \
             assert_eq(w.combine(n), 9); }}"
        ),
    );
}

#[test]
fn c62b_f5_an_unbounded_impl_param_still_rejects_the_method() {
    // Without the bound, the method genuinely does not exist -- must stay rejected (no over-accept).
    let src = format!(
        "{SH}struct W<T> {{ v: T }}\n\
         impl<T> W<T> {{ fn go(&self) -> Int32 {{ self.v.a() }} }}\n\
         fn main() {{ let w = W {{ v: S {{ n: 4 }} }}; assert_eq(w.go(), 4); }}"
    );
    let file = Arc::new(SourceFile::new("f5_neg.stark".to_string(), src));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir);
    assert!(
        checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error),
        "an unbounded impl parameter has no method `a`; must stay rejected"
    );
}
