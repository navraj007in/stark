//! WP-C6.2b-F2 — a trait/inherent impl on a SPECIFIC generic instantiation matches an inferred
//! receiver.
//!
//! `impl Get for W<Int32>` was not matched for `let w = W { v: 7 }; w.get()` (E0302, receiver typed
//! `W<_infer>`). It was NOT that specific-instance impls are unsupported — `let w: W<Int32> = ...`
//! already worked — but that the receiver's int-literal argument (`7`) was not defaulted to `Int32`
//! before method resolution. `default_int_literals_deep` now defaults literals INSIDE the receiver
//! type (03 solving step 5), so `W<_infer>` becomes `W<Int32>` and the concrete-instance impl
//! matches. A wrong instance (`W<Bool>`) still has no matching impl and stays rejected.

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

fn rejected(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("f2_neg_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir);
    assert!(
        checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error),
        "{tag}: must stay rejected"
    );
}

const GET: &str = "trait Get { fn get(&self) -> Int32; }\nstruct W<T> { v: T }\n\
                   impl Get for W<Int32> { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c62b_f2_specific_trait_impl_matches_inferred_receiver() {
    agree(
        "inferred",
        &format!("{GET}fn main() {{ let w = W {{ v: 7 }}; assert_eq(w.get(), 7); }}"),
    );
}

#[test]
fn c62b_f2_specific_trait_impl_matches_annotated_receiver() {
    agree(
        "annotated",
        &format!("{GET}fn main() {{ let w: W<Int32> = W {{ v: 7 }}; assert_eq(w.get(), 7); }}"),
    );
}

#[test]
fn c62b_f2_specific_inherent_impl_matches_inferred_receiver() {
    agree(
        "inherent",
        "struct W<T> { v: T }\nimpl W<Int32> { fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let w = W { v: 7 }; assert_eq(w.get(), 7); }",
    );
}

#[test]
fn c62b_f2_nested_instance_argument_is_defaulted() {
    // The literal is one level deeper: W<Pair<Int32>>-ish through a nested field.
    agree(
        "nested",
        "trait Get { fn get(&self) -> Int32; }\nstruct Inner { v: Int32 }\nstruct W<T> { i: T }\n\
         impl Get for W<Inner> { fn get(&self) -> Int32 { self.i.v } }\n\
         fn main() { let w = W { i: Inner { v: 7 } }; assert_eq(w.get(), 7); }",
    );
}

#[test]
fn c62b_f2_a_different_instance_stays_rejected() {
    // No `impl Get for W<Bool>`, so the call must not resolve (no over-accept from the fix).
    rejected(
        "wrong_instance",
        &format!("{GET}fn main() {{ let w: W<Bool> = W {{ v: true }}; let _ = w.get(); }}"),
    );
}
