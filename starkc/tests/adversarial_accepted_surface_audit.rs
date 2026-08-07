//! WP-C7.9 Packet E — the accepted-surface audit, as an executable invariant.
//!
//! **The invariant.**
//!
//! ```text
//! type checking accepts a program  ⇒  MIR lowering can build it
//! ```
//!
//! A program that fails that implication is accepted by the language and executable by exactly one
//! engine — the reference interpreter — which makes "the engines agree" unanswerable for it. Five
//! such surfaces existed (by-value `Vec` iteration and the `Iterator` combinators), each pinned by a
//! test that asserted the split as if it were a property rather than a defect.
//!
//! This file replaces those assertions with the invariant itself. Every source below is checked two
//! ways:
//!
//! 1. it is refused at type checking, with `E0105`; and
//! 2. **the guard** — for the whole audit list, acceptance and lowerability are required to agree.
//!    A future change that re-admits one of these forms without also lowering it fails here rather
//!    than reintroducing the split silently.
//!
//! The guard is the part that matters. The individual refusals could be updated by anyone
//! implementing one of these features; the implication must hold no matter which way they go.

mod support;

use starkc::diag::Severity;
use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Every surface the audit found. Kept in one list so the guard and the refusal cases cannot
/// disagree about what is being audited.
const AUDITED: &[(&str, &str)] = &[
    (
        "vec_by_value_iteration",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
         let mut n: Int32 = 0; for x in v { n = n + x; } assert_eq(n, 3); }",
    ),
    (
        "iterator_map",
        "fn double(x: &Int32) -> Int32 { *x * 2 }\n\
         fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let mut n: Int32 = 0; for x in it.map(double) { n = n + x; } assert_eq(n, 2); }",
    ),
    (
        "iterator_filter",
        "fn keep(x: &&Int32) -> Bool { true }\n\
         fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let mut n: Int32 = 0; for x in it.filter(keep) { n = n + *x; } assert_eq(n, 1); }",
    ),
    (
        "iterator_count",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let c = it.count(); assert_eq(c, 1u64); }",
    ),
    (
        "iterator_collect",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let w: Vec<Int32> = it.collect(); assert_eq(w.len(), 1u64); }",
    ),
    (
        "iterator_fold",
        "fn add(acc: Int32, x: &Int32) -> Int32 { acc + *x }\n\
         fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let total = it.fold(0, add); assert_eq(total, 1); }",
    ),
    (
        "iterator_any",
        "fn positive(x: &Int32) -> Bool { *x > 0 }\n\
         fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); let mut it = v.iter(); \
         let found = it.any(positive); assert(found); }",
    ),
];

struct Checked {
    accepted: bool,
    codes: Vec<String>,
    lowers: bool,
}

fn check(tag: &str, src: &str) -> Checked {
    let file = Arc::new(SourceFile::new(
        format!("audit_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<&starkc::diag::Diagnostic> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    let accepted = errors.is_empty();
    let codes = errors
        .iter()
        .filter_map(|d| d.code.clone())
        .collect::<Vec<_>>();
    // Only meaningful when the program was accepted; lowering a rejected program is not a
    // supported operation and its result says nothing.
    let lowers = accepted
        && lower_program(
            &hir,
            &checked.tables,
            hir.source_named(&file.name).expect("registered"),
        )
        .is_ok();
    Checked {
        accepted,
        codes,
        lowers,
    }
}

/// **The guard.** Acceptance implies lowerability, for every audited surface.
///
/// This is the assertion that survives whichever way a future work package resolves these
/// features. Implement `map` in MIR and this still passes; re-admit `map` in the type checker
/// without implementing it and this fails.
#[test]
fn every_accepted_audited_program_can_be_lowered() {
    for (tag, src) in AUDITED {
        let result = check(tag, src);
        assert!(
            !result.accepted || result.lowers,
            "{tag}: type checking ACCEPTED this program and MIR lowering refused it. That is the \
             accepted-but-unexecutable split WP-C7.9 Packet E closed: either lower the form or \
             refuse it in the front end, but do not accept what no compiler can build."
        );
    }
}

/// Today, each audited surface is refused — with the code that says so.
#[test]
fn every_audited_surface_is_refused_with_e0105() {
    for (tag, src) in AUDITED {
        let result = check(tag, src);
        assert!(
            !result.accepted,
            "{tag}: expected a front-end refusal; if this form is now implemented, move it out of \
             AUDITED and give it three-engine cases"
        );
        assert!(
            result.codes.iter().any(|c| c == "E0105"),
            "{tag}: expected E0105, got {:?}",
            result.codes
        );
    }
}

/// The supported neighbours must keep working. A refusal that also caught `v.iter()` in a `for`
/// loop would break every program in the corpus, so the boundary is pinned from both sides.
#[test]
fn borrowed_iteration_is_unaffected() {
    for (tag, src) in [
        (
            "vec_iter_for",
            "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
             let mut n: Int32 = 0; for x in v.iter() { n = n + *x; } assert_eq(n, 3); }",
        ),
        (
            "range_for",
            "fn main() { let mut n: Int32 = 0; for i in 0..3 { n = n + i; } assert_eq(n, 3); }",
        ),
        (
            "array_for",
            "fn main() { let a: [Int32; 3] = [1, 2, 3]; let mut n: Int32 = 0; \
             for x in a { n = n + x; } assert_eq(n, 6); }",
        ),
        (
            "chars_for",
            "fn main() { let s: String = String::from(\"hey\"); let mut n: Int32 = 0; \
             for c in s.chars() { n = n + 1; } assert_eq(n, 3); }",
        ),
    ] {
        let result = check(tag, src);
        assert!(
            result.accepted,
            "{tag}: a supported iteration form was refused: {:?}",
            result.codes
        );
        assert!(result.lowers, "{tag}: a supported form failed to lower");
    }
}
