//! WP-C6.3c — native ITERATORS (§26).
//!
//! The §26 matrix, proven three-engine (HIR == MIR == native stdout). Iteration splits into two
//! lowering families, and only the second needs backend work:
//!
//! - **Counting loops.** `for i in a..b` and `for x in <array>` lower to an index loop with the
//!   ordinary `CheckIndex` proof discipline — no iterator object exists at runtime, so these were
//!   already native. A **user `Iterator` impl** is likewise ordinary static calls to the user's
//!   `next`.
//! - **Runtime iterator objects.** `v.iter()` and `s.chars()` lower to `*IterNew`/`*IterNext`
//!   runtime calls over a live cursor VALUE that borrows its source. These are what WP-C6.3c adds
//!   natively. (`m.keys()` takes the same shape but lands with C6.3d, alongside HashMap itself.)
//!
//! Order, early termination and `for`-vs-explicit-`next` equivalence are asserted inside the STARK
//! programs themselves (via `assert_eq`) and by comparing printed output, so a case that agreed on
//! the WRONG order would still fail.
//!
//! **The closure boundary.** Every §26 row that MIR can lower is native and proven here. The rows
//! that remain stop BEFORE MIR — the front end rejects them (slice iteration; there is no `iter_mut`
//! surface at all) or lowering refuses them (`map`/`filter`/`collect`/`count`, by-value `Vec`
//! iteration). Those are LOWERING gaps, not native ones: the MIR interpreter cannot run them either,
//! so no native/interpreter divergence exists for the backend to close. They are pinned by the
//! negative tests at the bottom of this file rather than left as prose, and closing them is a
//! front-end/MIR work package. `HashMap`/`HashSet` iteration lands with C6.3d.

mod support;

use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Delegates to the shared comparator (R-02). The private version it replaces took the HIR oracle's
/// own stdout as the expectation and checked the others against it; that comparison is preserved
/// and widened to every observation field, but see this file's header on what it does NOT pin.
fn agree_out(tag: &str, src: &str) {
    support::differential::agree_completing_available_engines(tag, src);
}

// ---- Counting-loop family: no runtime iterator object (already native before C6.3c) ----

#[test]
fn range_for_loop() {
    agree_out(
        "range",
        "fn main() { let mut s: Int32 = 0; for i in 0..4 { s = s + i; print(i); } assert_eq(s, 6); println(\"\"); }",
    );
}

#[test]
fn array_for_loop_order() {
    agree_out(
        "array",
        "fn main() { let a: [Int32; 3] = [10, 20, 30]; let mut s: Int32 = 0; for x in a { s = s + x; print(x); } assert_eq(s, 60); println(\"\"); }",
    );
}

/// A user `Iterator` impl — ordinary static calls to the user's `next`, and `for` must equal
/// explicit `next()` iteration.
#[test]
fn user_iterator_impl() {
    agree_out(
        "useriter",
        "struct Countdown { n: Int32 }\n\
         impl Iterator for Countdown {\n\
           type Item = Int32;\n\
           fn next(&mut self) -> Option<Int32> { if self.n == 0 { None } else { self.n = self.n - 1; Some(self.n) } }\n\
         }\n\
         fn main() { let mut c = Countdown { n: 3 }; for x in c { print(x); } println(\"\"); }",
    );
}

// ---- Runtime-iterator-object family: what WP-C6.3c adds natively ----

/// `v.iter()` — by-reference Vec iteration (`VecIterNew`/`VecIterNext`, yielding `Option<&T>`).
#[test]
fn vec_iter_shared() {
    agree_out(
        "veciter",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let mut s: Int32 = 0; for x in v.iter() { s = s + *x; print(*x); } assert_eq(s, 6); println(\"\"); }",
    );
}

/// Early termination: `break` mid-iteration leaves the iterator (and the borrow) unfinished.
#[test]
fn vec_iter_early_break() {
    agree_out(
        "veciterbreak",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let mut seen: Int32 = 0; for x in v.iter() { if *x == 2 { break; } seen = seen + 1; } \
         assert_eq(seen, 1); }",
    );
}

/// An empty source yields nothing — the `None`-on-first-`next` path.
#[test]
fn vec_iter_empty() {
    agree_out(
        "veciterempty",
        "fn main() { let v: Vec<Int32> = Vec::new(); let mut n: Int32 = 0; for x in v.iter() { n = n + 1; } assert_eq(n, 0); }",
    );
}

/// `s.chars()` — character iteration over a `str`.
#[test]
fn chars_iter() {
    agree_out(
        "chars",
        "fn main() { let mut n: Int32 = 0; for c in \"abc\".chars() { n = n + 1; print(c); } assert_eq(n, 3); println(\"\"); }",
    );
}

/// `chars()` over an owned `String`.
#[test]
fn chars_iter_over_string() {
    agree_out(
        "charsstring",
        "fn main() { let s: String = String::from(\"hey\"); let mut n: Int32 = 0; for c in s.chars() { n = n + 1; print(c); } assert_eq(n, 3); println(\"\"); }",
    );
}

// ---- The §26 rows that are NOT native gaps ----
//
// Everything below stops BEFORE MIR: the front end rejects it, or lowering refuses it. None of it is
// a native/interpreter divergence — the MIR interpreter cannot run these either, so there is nothing
// for the differential to compare and nothing the backend could fix. They are recorded here as
// executable evidence of exactly where each row stops, so the boundary cannot drift unnoticed and so
// a future lowering package has its starting point. `HashMap`/`HashSet` iteration is C6.3d.

/// The program type-checks but LOWERING refuses it — an HIR-only shape.
fn hir_only(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c63c_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errs.is_empty(),
        "{tag}: expected it to type-check, got {errs:?}"
    );
    // The HIR interpreter DOES run it — which is what makes this a lowering gap rather than an
    // unimplemented language feature.
    interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag}: HIR should run it: {}", e.message));
    assert!(
        lower_program(&hir, &checked.tables, file).is_err(),
        "{tag}: lowering is expected to refuse this; if it now lowers, make it a three-engine case"
    );
}

/// The front end rejects it outright — the language has no such form.
fn rejected_by_front_end(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c63c_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    assert!(
        checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error),
        "{tag}: expected a front-end rejection"
    );
}

/// `for x in <slice>` is not an iterable form: the for-loop rejects `&[T]`. Slice iteration is a
/// FRONT-END feature, not a backend one.
#[test]
fn slice_iteration_is_not_a_language_form() {
    rejected_by_front_end(
        "sliceiter",
        "fn main() { let a: [Int32; 3] = [1,2,3]; let s: &[Int32] = &a[0..2]; \
         let mut n: Int32 = 0; for x in s { n = n + *x; } assert_eq(n, 3); }",
    );
}

/// By-VALUE `Vec` iteration runs in the HIR interpreter but is not lowered.
#[test]
fn vec_by_value_iteration_is_hir_only() {
    hir_only(
        "vecbyvalue",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
         let mut n: Int32 = 0; for x in v { n = n + x; } assert_eq(n, 3); }",
    );
}

/// `map` needs a MIR representation for `MapIter` (it has none — a C4.5-era gap).
#[test]
fn map_adapter_is_hir_only() {
    hir_only(
        "mapadapter",
        "fn double(x: &Int32) -> Int32 { *x * 2 }\n\
         fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let mut n: Int32 = 0; for x in it.map(double) { n = n + x; } assert_eq(n, 2); }",
    );
}

/// `count`/`collect` are method calls on a non-nominal (core) receiver, which lowering does not do.
#[test]
fn count_and_collect_are_hir_only() {
    hir_only(
        "countadapter",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let c = it.count(); assert_eq(c, 1u64); }",
    );
    hir_only(
        "collectadapter",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let w: Vec<Int32> = it.collect(); assert_eq(w.len(), 1u64); }",
    );
}
