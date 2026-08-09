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

/// **WP-C7.9 Packet E: these are front-end REJECTIONS now, not HIR-only shapes.**
///
/// This helper used to assert the opposite: that the program type-checked, that the reference
/// interpreter ran it, and that lowering then refused it. That combination is the defect Packet E
/// closed — a program the language accepted and no compiler could build — so the assertion is
/// inverted rather than deleted. The cases are the same programs; what changed is where they stop.
fn refused_by_front_end(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("c63c_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errs.iter().any(|d| d.code.as_deref() == Some("E0105")),
        "{tag}: expected an E0105 refusal at type checking, got {errs:?}"
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
    let checked = typecheck::analyze(&hir);
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

/// By-VALUE `Vec` iteration: refused at type checking (E0105). It used to type-check and run in
/// the oracle while no compiler could build it.
#[test]
fn vec_by_value_iteration_is_refused() {
    refused_by_front_end(
        "vecbyvalue",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
         let mut n: Int32 = 0; for x in v { n = n + x; } assert_eq(n, 3); }",
    );
}

/// `map` has no MIR representation for `MapIter`, so it is refused rather than accepted (E0105).
#[test]
fn map_adapter_is_refused() {
    refused_by_front_end(
        "mapadapter",
        "fn double(x: &Int32) -> Int32 { *x * 2 }\n\
         fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let mut n: Int32 = 0; for x in it.map(double) { n = n + x; } assert_eq(n, 2); }",
    );
}

/// `count`/`collect` are method calls on a non-nominal (core) receiver, which lowering does not
/// perform — so they are refused at type checking rather than accepted (E0105).
#[test]
fn count_and_collect_are_refused() {
    refused_by_front_end(
        "countadapter",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let c = it.count(); assert_eq(c, 1u64); }",
    );
    refused_by_front_end(
        "collectadapter",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let w: Vec<Int32> = it.collect(); assert_eq(w.len(), 1u64); }",
    );
}

// ---- CD-293: `for x in &v`, the spelling that used to be refused ----

/// **`for x in &v` iterates the borrow, identically to `v.iter()`.**
///
/// This was E0001 "for-loop requires an iterable value, found '&Vec<Int32>'" — an unhelpful
/// refusal, because the value is iterable and the borrow is precisely what Vec iteration wants.
/// It lowers to the same `VecIterNew`/`VecIterNext` cursor, so the item is `&T` and `*x` reads it.
///
/// Three-engine, because the two spellings must agree in the HIR oracle as well as through MIR.
#[test]
fn vec_for_over_borrow() {
    agree_out(
        "vecforborrow",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let mut s: Int32 = 0; for x in &v { s = s + *x; print(*x); } assert_eq(s, 6); println(\"\"); }",
    );
}

/// The two spellings are the same iteration, asserted inside the program rather than by comparing
/// two runs: sum via `&v` and sum via `v.iter()` must be equal in one execution.
#[test]
fn vec_for_over_borrow_matches_iter() {
    agree_out(
        "vecforborroweq",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(4); v.push(5); v.push(6); \
         let mut a: Int32 = 0; for x in &v { a = a + *x; } \
         let mut b: Int32 = 0; for x in v.iter() { b = b + *x; } \
         assert_eq(a, b); assert_eq(a, 15); }",
    );
}

/// **The case that motivated the change: a NON-`Copy` element.**
///
/// `v[i]` is refused for such an element (`VecIndexGet` requires `Copy`), and borrowing the indexed
/// place does not help — so iteration is the only way to read one in place. When `&v` was also
/// refused, a `Vec<String>` was reachable by exactly one spelling out of three.
#[test]
fn vec_for_over_borrow_non_copy_element() {
    agree_out(
        "vecforborrownoncopy",
        "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"a\")); v.push(String::from(\"bb\")); \
         let mut n: UInt64 = 0u64; for s in &v { n = n + s.len(); } assert_eq(n, 3u64); }",
    );
}

/// `break` through the borrow form leaves the cursor unfinished, as it does through `.iter()`.
#[test]
fn vec_for_over_borrow_early_break() {
    agree_out(
        "vecforborrowbreak",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); \
         let mut seen: Int32 = 0; for x in &v { if *x == 2 { break; } seen = seen + 1; } \
         assert_eq(seen, 1); }",
    );
}

/// Empty source: `None` on the first `next`, no iterations.
#[test]
fn vec_for_over_borrow_empty() {
    agree_out(
        "vecforborrowempty",
        "fn main() { let v: Vec<Int32> = Vec::new(); let mut n: Int32 = 0; for x in &v { n = n + 1; } assert_eq(n, 0); }",
    );
}

/// **`v[i]` on a non-`Copy` element is refused BY THE BORROW CHECKER, and always was.**
///
/// CD-293 added E0106 for this in semantic analysis, on the belief that it was otherwise an
/// accepted-but-unbuildable program caught only at MIR verification. CD-294 reverted it after it
/// broke three working programs. **This test records why the whole exercise was unnecessary**:
/// E0100 already refuses it, at the right layer, with a label naming the fix —
/// "cannot move a non-Copy value out of an indexed place / use an ownership-transferring
/// collection method instead".
///
/// And E0100 is right where E0106 was wrong. It fires on the MOVE — the semantic event — not on
/// the syntax `v[i]`. That is precisely why `v[i].push(x)`, `v[i] = e`, `&v[i]` and an
/// auto-borrowed comparison operand all pass it and all failed E0106: none of them moves anything.
///
/// The premise of the original change was never checked against the compiler's actual behaviour.
/// It was reasoned from a MIR message (`MIR-0016 VecIndexGet requires a Copy element type`) that a
/// source program cannot in fact reach by this route.
#[test]
fn vec_index_non_copy_is_refused_by_the_borrow_checker() {
    let src = "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"a\")); \
               let s = v[0u64]; println(s.len()); }";
    let file = Arc::new(SourceFile::new("vecindexnoncopy.stark", src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{rd:?}");
    let checked = typecheck::analyze(&hir);
    let diag = checked
        .diagnostics
        .iter()
        .find(|d| d.code.as_deref() == Some("E0100"))
        .unwrap_or_else(|| {
            panic!(
                "moving a non-Copy element out of a Vec index must be refused in semantic \
                 analysis; got {:?}",
                checked.diagnostics
            )
        });
    assert_eq!(diag.severity, Severity::Error);
    assert!(
        diag.message.contains("indexed place"),
        "the message must name what is wrong: {}",
        diag.message
    );
    assert!(
        !diag.label.is_empty(),
        "the diagnostic must carry a label naming the way out"
    );
}

/// A `Copy` element still indexes normally — the refusal is about ownership, not about `Vec`.
#[test]
fn vec_index_copy_element_still_works() {
    agree_out(
        "vecindexcopy",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(7); v.push(8); \
         assert_eq(v[1u64], 8); }",
    );
}

// ---- CD-293: `&v[i]` — borrowing a Vec element ----

/// **`&v[i]` borrows the element instead of reading it by value.**
///
/// A Vec is an opaque runtime type, not a projectable place, so this could not be written at all
/// while `&a[i]` on an ARRAY always could. With `v[i]` by value refused for an owning element
/// (E0106, correctly — it would move out of the Vec), the borrow form is what makes such an
/// element reachable by index at all.
#[test]
fn vec_index_borrow_non_copy() {
    agree_out(
        "vecindexborrow",
        "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"alpha\")); \
         v.push(String::from(\"be\")); let s = &v[1u64]; assert_eq(s.len(), 2u64); }",
    );
}

/// The borrow reads the same element the by-value form would, for a `Copy` element where both are
/// legal — so the two forms cannot disagree about WHICH element.
#[test]
fn vec_index_borrow_matches_value_read() {
    agree_out(
        "vecindexborroweq",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(10); v.push(20); v.push(30); \
         let r = &v[2u64]; assert_eq(*r, v[2u64]); assert_eq(*r, 30); }",
    );
}

/// **Out of bounds traps, in the same category as `v[i]`.** The `None` arm of `VecGetRef` IS the
/// out-of-bounds case; routing it to a trap is what keeps the borrow form's observable behaviour
/// identical to the by-value form's rather than quietly yielding an `Option`.
#[test]
fn vec_index_borrow_out_of_bounds_traps() {
    support::differential::agree_trapping_available_engines(
        "vecindexborrowoob",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let r = &v[5u64]; println(*r); }",
        starkc::mir::TrapCategory::IndexOutOfBounds,
        1,
    );
}

// ---- CD-303: PAT-BIND-001 diagnostic enforcement ----

/// **A reference-typed scrutinee with constructor patterns is REJECTED, and the message says how.**
///
/// PAT-BIND-001: "a struct/variant path must name the scrutinee's normalized nominal type, and `&T`
/// is not a nominal type, so `match r { E::V(x) => .. }` for `r: &E` is a type error. This is why
/// the rule is stated over the place read, not over the scrutinee's type."
///
/// It was not enforced. `Ty::Ref` fell through every classifier, and the result was the worst
/// available combination: the exhaustiveness check demanded a wildcard (E0303) on a match that
/// already covered every variant, and the wildcard added to satisfy it then absorbed every case at
/// run time — a function returning the wildcard's answer for every input, silently.
///
/// The diagnostic names the fix rather than the symptom, because the symptom (E0303) pointed at
/// the wrong problem and its obvious remedy made things worse.
#[test]
fn ref_scrutinee_with_constructor_patterns_is_rejected() {
    let src = "enum E { A(UInt64), B } \
               fn classify(e: &E) -> UInt64 { match e { E::A(n) => n, E::B => 900u64 } } \
               fn main() { let v = E::A(7u64); println(classify(&v)); }";
    let file = Arc::new(SourceFile::new("refscrut.stark", src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{rd:?}");
    let checked = typecheck::analyze(&hir);
    let diag = checked
        .diagnostics
        .iter()
        .find(|d| d.message.contains("reference-typed scrutinee"))
        .unwrap_or_else(|| {
            panic!(
                "a reference-typed scrutinee with constructor patterns must be rejected; got {:?}",
                checked.diagnostics
            )
        });
    assert_eq!(diag.severity, Severity::Error);
    assert!(
        diag.helps.iter().any(|h| h.contains("match *r")),
        "the diagnostic must recommend dereferencing: {:?}",
        diag.helps
    );
}

/// A wildcard or plain binding names no constructor, so it is NOT what the rule forbids and stays
/// accepted. Without this, the check above would be free to over-reject.
#[test]
fn ref_scrutinee_without_constructor_patterns_is_accepted() {
    let src = "enum E { A(UInt64), B } \
               fn is_something(e: &E) -> UInt64 { match e { _ => 1u64 } } \
               fn main() { let v = E::B; println(is_something(&v)); }";
    let file = Arc::new(SourceFile::new("refscrutwild.stark", src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{rd:?}");
    let checked = typecheck::analyze(&hir);
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.message.contains("reference-typed scrutinee")),
        "a wildcard arm names no constructor and must not be rejected: {:?}",
        checked.diagnostics
    );
}

/// **`match *r` is the working form, and it selects the right arm in every engine.**
///
/// The positive half of the rule. Three-engine, because the runtime behaviour is the part that was
/// wrong: with the old code no arm matched, so this program would have trapped or taken a wildcard.
#[test]
fn deref_scrutinee_match_selects_the_right_arm() {
    agree_out(
        "derefscrutarm",
        "enum E { A(UInt64), B, C } \
         fn classify(e: &E) -> UInt64 { match *e { E::A(n) => n, E::B => 900u64, E::C => 901u64 } } \
         fn main() { let a = E::A(7u64); let b = E::B; let c = E::C; \
         assert_eq(classify(&a), 7u64); assert_eq(classify(&b), 900u64); assert_eq(classify(&c), 901u64); }",
    );
}

/// `match *r` covering every variant is exhaustive and needs no wildcard — the deref form was never
/// the thing E0303 was complaining about, and must not start complaining now.
#[test]
fn deref_scrutinee_match_is_exhaustive_without_a_wildcard() {
    agree_out(
        "derefscrutexh",
        "enum E { A(UInt64), B, C } \
         fn classify(e: &E) -> UInt64 { match *e { E::A(n) => n, E::B => 900u64, E::C => 901u64 } } \
         fn main() { let v = E::C; assert_eq(classify(&v), 901u64); }",
    );
}

/// **PAT-BIND-001's binding rule, pinned by use.** A `Copy` payload binds BY VALUE (`n` is a
/// `UInt64`, added directly); a non-`Copy` payload binds BY REFERENCE (`s` is a `&String`, read in
/// place and never moved out of the referent). Rust would bind both by reference; that difference
/// is deliberate, and treating Rust's ergonomics as a separate proposal is why this test states the
/// current rule explicitly rather than leaving it to be inferred.
#[test]
fn deref_scrutinee_binds_copy_by_value_and_owning_by_reference() {
    agree_out(
        "derefscrutbind",
        "enum E { N(UInt64), S(String) } \
         fn describe(e: &E) -> UInt64 { match *e { E::N(n) => n + 1u64, E::S(s) => s.len() } } \
         fn main() { let n = E::N(41u64); assert_eq(describe(&n), 42u64); \
         let s = E::S(String::from(\"abcd\")); assert_eq(describe(&s), 4u64); }",
    );
}

// ---- CD-305: a shared slice view is Copy however it was produced ----

/// **`String::bytes()` returned a value that was not `Copy`, so passing it consumed the caller's
/// binding.** Accepted-but-traps: the checker allowed it, MIR emitted `copy` for the argument, and
/// only the HIR interpreter destroyed the local — "use of unavailable value" at run time.
///
/// `bytes()` is `&[UInt8]`, a shared reference, which is `Copy`. It shared an implementation arm
/// with `into_bytes()` — `Vec<UInt8>`, genuinely owned — and both produced `Value::Vec`. So two
/// representations claimed to be `&[UInt8]` and only one obeyed the ownership contract.
///
/// This is DEV-087 in a second producer: that fix classified `Value::Slice` as `Copy` after the
/// identical symptom, but `bytes()` never produced a `Value::Slice` to classify.
///
/// Three engines, because the failure existed in exactly one of them.
#[test]
fn bytes_view_survives_being_passed_to_a_function() {
    agree_out(
        "bytesviewtwice",
        "fn use_len(value: &[UInt8]) -> UInt64 { value.len() } \
         fn main() { let input = String::from(\"abcd\"); let bytes = input.bytes(); \
         let a = use_len(bytes); let b = use_len(bytes); assert_eq(a + b, 8u64); }",
    );
}

/// Passing then indexing — the discriminator that showed the CALL was what consumed the view,
/// rather than "two calls" being the problem.
#[test]
fn bytes_view_is_indexable_after_being_passed() {
    agree_out(
        "bytesviewthenindex",
        "fn use_len(value: &[UInt8]) -> UInt64 { value.len() } \
         fn main() { let input = String::from(\"abcd\"); let bytes = input.bytes(); \
         let a = use_len(bytes); let b = bytes[0u64]; assert_eq(a, 4u64); assert_eq(b, 97u8); }",
    );
}

/// The same view reached through `as_str()`, which is how `stark-percent` spells it.
#[test]
fn as_str_bytes_view_survives_being_passed() {
    agree_out(
        "asstrbytesview",
        "fn use_len(value: &[UInt8]) -> UInt64 { value.len() } \
         fn main() { let input = String::from(\"abcd\"); let bytes = input.as_str().bytes(); \
         let a = use_len(bytes); let b = use_len(bytes); assert_eq(a + b, 8u64); }",
    );
}

/// A slice built the ordinary way already worked; pinned so the two producers cannot diverge again.
#[test]
fn array_slice_survives_being_passed() {
    agree_out(
        "arrayslicetwice",
        "fn use_len(value: &[UInt8]) -> UInt64 { value.len() } \
         fn main() { let array: [UInt8; 4] = [97u8, 98u8, 99u8, 100u8]; \
         let slice = &array[0u64..4u64]; \
         let a = use_len(slice); let b = use_len(slice); assert_eq(a + b, 8u64); }",
    );
}

/// Aliasing the view and using both — a `Copy` value may be bound again without consuming the
/// original.
#[test]
fn bytes_view_may_be_aliased() {
    agree_out(
        "bytesviewalias",
        "fn use_len(value: &[UInt8]) -> UInt64 { value.len() } \
         fn main() { let input = String::from(\"abcd\"); let bytes = input.bytes(); \
         let alias = bytes; let a = use_len(bytes); let b = use_len(alias); \
         assert_eq(a + b, 8u64); }",
    );
}

/// **The control.** Without this, a repair that made every call argument a `Copy` read would pass
/// all five tests above while silently destroying move semantics. `into_bytes()` is an OWNED
/// `Vec<UInt8>`: passing it consumes it, and a second use is a compile-time move error.
#[test]
fn owned_vec_argument_still_moves() {
    let src = "fn consume(v: Vec<UInt8>) -> UInt64 { v.len() } \
               fn main() { let input = String::from(\"abcd\"); let owned = input.into_bytes(); \
               let a = consume(owned); let b = consume(owned); println(a + b); }";
    let file = Arc::new(SourceFile::new("ownedmove.stark", src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{rd:?}");
    let checked = typecheck::analyze(&hir);
    assert!(
        checked
            .diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0100")),
        "passing an owned Vec must still move it: {:?}",
        checked.diagnostics
    );
}
