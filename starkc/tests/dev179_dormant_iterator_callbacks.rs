//! **DEV-179 (DORMANT) — the gate that fires when iterator adapters become reachable.**
//!
//! `Value::MapIter` and `Value::FilterIter` retain only the callback's `ItemId`. When stepped, they
//! reconstruct a function value with **empty bindings**, so a generic callback would execute
//! without the environment the checker selected for it — the same defect class as DEV-178, reached
//! through deferred iteration instead of an ordinary indirect call.
//!
//! It is unreachable today: Core v1 rejects `map`/`filter` at the front end with E0105. That
//! governs the defect's URGENCY, not whether the implementation contains it.
//!
//! **This test exists to fail at the right moment.** The implementation looks complete, so on the
//! day E0105 is lifted it would activate silently with an empty environment — and whoever lifts it
//! will be working in the front end, not in `interp.rs`, where the comments are. When this test
//! fails, the fix is not to delete it: store a complete `FunctionValue` inside `MapIter`/
//! `FilterIter` rather than reconstructing one, then replace this with a test that a generic
//! callback receives its instantiation.
//!
//! DEV-174 is the precedent. A recorded limitation with a test that fails when it is lifted turned
//! a rediscovery into a one-commit repair.

use starkc::diag::Severity;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn errors(source: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    let mut out: Vec<String> = parse_diags
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message))
        .collect();
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    out.extend(
        resolve_diags
            .into_iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message)),
    );
    out.extend(
        typecheck::analyze(&hir, file)
            .diagnostics
            .into_iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message)),
    );
    out
}

/// A GENERIC callback is the case that would misbehave, so it is the one gated.
#[test]
fn a_generic_map_callback_is_still_refused_by_e0105() {
    let diags = errors(
        "\
fn identity<T>(x: T) -> T {
    x
}

fn main() {
    let v: Vec<Int32> = Vec::new();
    let m = v.iter().map(identity);
}
",
    );
    assert!(
        diags.iter().any(|d| d.starts_with("E0105")),
        "DEV-179: `map` is reachable — MapIter must now retain the callback's captured \
         FunctionValue instead of reconstructing it with empty bindings. See the DEV-179 ledger \
         entry before deleting this test. Diagnostics: {diags:?}"
    );
}

/// `filter` reconstructs its predicate the same way, so it is gated separately — lifting one
/// adapter without the other is a plausible increment.
#[test]
fn a_generic_filter_callback_is_still_refused_by_e0105() {
    let diags = errors(
        "\
fn keep<T>(x: &T) -> Bool {
    true
}

fn main() {
    let v: Vec<Int32> = Vec::new();
    let f = v.iter().filter(keep);
}
",
    );
    assert!(
        diags.iter().any(|d| d.starts_with("E0105")),
        "DEV-179: `filter` is reachable — FilterIter must now retain its predicate's captured \
         FunctionValue. See the DEV-179 ledger entry before deleting this test. \
         Diagnostics: {diags:?}"
    );
}
