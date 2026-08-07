//! WP-C6.1f "aggregates" — borrow-carrying tuples and arrays.
//!
//! OWN-CARRY-001 makes borrow provenance **structural**: it flows through tuples, generic
//! arguments and enum payloads. So a tuple or array of references is ordinary Core v1, not an
//! escape hatch. (Declared reference *fields* remain forbidden — 03 rule 1 — and the front end
//! rejects them as E0001; `a_declared_reference_field_is_still_rejected` pins that.)
//!
//! What made these emit is one observation: the property that matters is **carries a borrow**, not
//! **is a reference**. A `Copy` aggregate of references (`(&T, &T)`, `[&T; N]`) is not slot-backed,
//! so it would be declared via `default_value_expr` — which cannot fabricate a reference, one level
//! down for exactly the reason it cannot fabricate one directly. Such locals are therefore
//! initialisation-deferred like a bare reference: `Option<T> = None` when they must cross basic
//! blocks, bare-uninitialised when same-block.
//!
//! Borrow-carrying **nominals** (`Option<&T>`, a user generic at a reference) are covered by
//! `native_c61f_nominals.rs`: generated nominals now carry lifetime parameters, so most of them
//! work; two shapes remain refused before rustc.

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

const P: &str = "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c61f_tuple_of_references() {
    agree(
        "two_refs",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; \
         let t: (&Int32, &Int32) = (&x, &y); assert_eq(*t.0 + *t.1, 3); }",
    );
    agree(
        "struct_refs",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let q = P {{ v: 4 }}; \
                  let t: (&P, &P) = (&p, &q); assert_eq(t.0.get() + t.1.get(), 7); }}"
        ),
    );
    // Mixed: only one element carries a borrow.
    agree(
        "one_ref",
        "fn main() { let x: Int32 = 1; let t: (&Int32, Int32) = (&x, 5); \
         assert_eq(*t.0 + t.1, 6); }",
    );
}

#[test]
fn c61f_array_of_references() {
    agree(
        "array",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; let a: [&Int32; 2] = [&x, &y]; \
         assert_eq(*a[0] + *a[1], 3); }",
    );
}

#[test]
fn c61f_nested_borrow_carrying_tuple() {
    agree(
        "nested",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; \
         let t: ((&Int32, &Int32), Int32) = ((&x, &y), 9); assert_eq(*(t.0).0 + t.1, 10); }",
    );
}

#[test]
fn c61f_borrow_carrying_tuple_crosses_basic_blocks() {
    // The Option-backed path: written in one dispatch-loop arm, read in another.
    agree(
        "across_block",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let q = P {{ v: 4 }}; \
                  let t: (&P, &P) = (&p, &q); \
                  let n = if t.0.get() > 1 {{ t.1.get() }} else {{ 0 }}; assert_eq(n, 4); }}"
        ),
    );
}

#[test]
fn c61f_tuple_of_references_to_drop_bearing_values() {
    // Borrowing does not disturb the owners' destructors.
    agree(
        "drop_refs",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         fn main() { let d = D { v: 1 }; let e = D { v: 2 }; \
         let t: (&D, &D) = (&d, &e); assert_eq(t.0.v + t.1.v, 3); }",
    );
}

#[test]
fn a_declared_reference_field_is_still_rejected() {
    // 03 rule 1: struct/enum/tuple-struct declarations MUST NOT write a reference field type.
    // Supporting borrow-carrying tuples must not have opened this.
    let src = "struct H { r: &Int32 }\nfn main() { let x: Int32 = 1; let h = H { r: &x }; }";
    let file = Arc::new(SourceFile::new("declared_ref_field.stark", src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    let (hir, rd) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir);
    let rejected = !pd.is_empty()
        || !rd.is_empty()
        || checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error);
    assert!(
        rejected,
        "a declared reference field must be rejected (03 rule 1)"
    );
}
