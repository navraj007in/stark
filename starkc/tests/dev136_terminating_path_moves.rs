//! **DEV-136: a move on a path that terminates does not poison the paths that do not.**
//!
//! Ownership state is merged at a control-flow join from its PREDECESSORS. A branch that ends in
//! `return`/`break`/`continue`/`panic` is not a predecessor of that join, so the moves it made
//! must not appear in the merged state. Before this repair every branch contributed
//! unconditionally, so
//!
//! ```stark
//! if flag { return out; }
//! out.push('a');
//! ```
//!
//! reported E0100 "use of moved value" on a path where the move provably had not happened.
//!
//! # The direction of conservatism, which is the whole safety argument
//!
//! `block_diverges`/`expr_diverges` answer "does this definitely NOT reach the join?". Answering
//! `true` wrongly would DROP a real move from the join and accept a use-after-move — unsound.
//! Answering `false` wrongly merely preserves the old false positive. So the predicate reports
//! `true` only on evidence, and anything unrecognised falls through to `false`. The must-reject
//! half of this file is therefore the important half: it pins that a branch which CAN reach the
//! join still contributes its moves.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn check(src: &str, tag: &str) -> Option<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
}

fn expect_accept(src: &str, tag: &str) {
    if let Some(diagnostic) = check(src, tag) {
        panic!("{tag}: expected acceptance, got: {diagnostic}");
    }
}

fn expect_reject(src: &str, tag: &str) {
    match check(src, tag) {
        Some(diagnostic) => assert!(
            diagnostic.starts_with("E0100"),
            "{tag}: expected E0100, got: {diagnostic}"
        ),
        None => panic!("{tag}: expected rejection, but the program checked clean"),
    }
}

fn run(src: &str, tag: &str) -> String {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error),
        "{tag}: check: {:?}",
        checked.diagnostics
    );
    match starkc::interp::run(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    ) {
        Ok(execution) => execution.output,
        Err(error) => panic!("{tag}: runtime error: {}", error.message),
    }
}

// -------------------------------------------------------------------- must pass --

/// The CD-334 reproducer.
#[test]
fn move_then_return_on_one_branch_leaves_the_fallthrough_live() {
    expect_accept(
        "fn build(flag: Bool) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   if flag {\n\
         \x20       return out;\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(false).as_str()); }\n",
        "returnbranch",
    );
}

/// ...and it must EXECUTE, on both sides of the branch.
#[test]
fn the_reproducer_executes_on_both_paths() {
    let out = run(
        "fn build(flag: Bool) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   if flag {\n\
         \x20       return out;\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() {\n\
         \x20   println(build(false).as_str());\n\
         \x20   println(build(true).len());\n\
         }\n",
        "bothpaths",
    );
    assert_eq!(out.trim(), "a\n0", "false -> \"a\", true -> empty string");
}

/// `panic` is `!`, so the branch cannot reach the join either.
#[test]
fn move_then_panic_leaves_the_fallthrough_live() {
    expect_accept(
        "fn take(value: String) {}\n\
         fn build(flag: Bool) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   if flag {\n\
         \x20       take(out);\n\
         \x20       panic(\"gone\");\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(false).as_str()); }\n",
        "panicbranch",
    );
}

/// Nested: the terminating branch is one level down.
#[test]
fn a_nested_early_return_does_not_poison_the_outer_join() {
    expect_accept(
        "fn build(a: Bool, b: Bool) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   if a {\n\
         \x20       if b {\n\
         \x20           return out;\n\
         \x20       } else {\n\
         \x20           return out;\n\
         \x20       }\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(false, false).as_str()); }\n",
        "nestedreturn",
    );
}

/// An early `return` in one arm of a `match`; the other arm reaches the join.
#[test]
fn an_early_return_in_one_match_arm_does_not_poison_the_join() {
    expect_accept(
        "fn build(value: Option<Int32>) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   match value {\n\
         \x20       Some(_n) => {\n\
         \x20           return out;\n\
         \x20       }\n\
         \x20       None => {}\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(None).as_str()); }\n",
        "matcharm",
    );
}

/// Several terminating branches in sequence.
#[test]
fn multiple_terminating_branches_all_stay_out_of_the_join() {
    expect_accept(
        "fn build(a: Bool, b: Bool) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   if a {\n\
         \x20       return out;\n\
         \x20   }\n\
         \x20   if b {\n\
         \x20       return out;\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(false, false).as_str()); }\n",
        "multiple",
    );
}

/// `break` out of a loop is a terminating edge relative to the loop body's own join.
#[test]
fn a_move_then_break_does_not_poison_the_rest_of_the_body() {
    expect_accept(
        "fn take(value: String) {}\n\
         fn main() {\n\
         \x20   let mut i = 0;\n\
         \x20   while i < 3 {\n\
         \x20       let mut out = String::new();\n\
         \x20       if i == 1 {\n\
         \x20           take(out);\n\
         \x20           break;\n\
         \x20       }\n\
         \x20       out.push('a');\n\
         \x20       println(out.as_str());\n\
         \x20       i = i + 1;\n\
         \x20   }\n\
         }\n",
        "breakedge",
    );
}

/// `continue` likewise.
#[test]
fn a_move_then_continue_does_not_poison_the_rest_of_the_body() {
    expect_accept(
        "fn take(value: String) {}\n\
         fn main() {\n\
         \x20   let mut i = 0;\n\
         \x20   while i < 3 {\n\
         \x20       i = i + 1;\n\
         \x20       let mut out = String::new();\n\
         \x20       if i == 1 {\n\
         \x20           take(out);\n\
         \x20           continue;\n\
         \x20       }\n\
         \x20       out.push('a');\n\
         \x20       println(out.as_str());\n\
         \x20   }\n\
         }\n",
        "continueedge",
    );
}

/// A value with a destructor, so the fix has to be right about drop obligations too and not
/// merely about the diagnostic.
#[test]
fn a_droppable_value_survives_a_terminating_branch() {
    let out = run(
        "struct Guard { label: String }\n\
         impl Drop for Guard {\n\
         \x20   fn drop(&mut self) { println(self.label.as_str()); }\n\
         }\n\
         fn take(guard: Guard) { println(\"taken\"); }\n\
         fn build(flag: Bool) {\n\
         \x20   let guard = Guard { label: String::from(\"kept\") };\n\
         \x20   if flag {\n\
         \x20       take(guard);\n\
         \x20       return;\n\
         \x20   }\n\
         \x20   println(\"fallthrough\");\n\
         }\n\
         fn main() { build(false); build(true); }\n",
        "droppable",
    );
    // false path: prints "fallthrough", then `guard` drops at end of scope -> "kept".
    // true path:  `take` prints "taken", then drops the guard IT now owns      -> "kept".
    // Exactly one destructor run per constructed value, on both paths.
    assert_eq!(
        out.trim(),
        "fallthrough\nkept\ntaken\nkept",
        "each `Guard` must be destroyed exactly once, whichever path ran"
    );
}

// ------------------------------------------------------------------ must reject --

/// **The central negative control.** The branch CAN reach the join, so its move still counts.
/// If this ever passes, the predicate has started reporting divergence it cannot prove and the
/// repair has become unsound.
#[test]
fn a_move_on_a_reachable_branch_is_still_rejected() {
    expect_reject(
        "fn take(value: String) {}\n\
         fn main() {\n\
         \x20   let mut out = String::new();\n\
         \x20   let flag = true;\n\
         \x20   if flag {\n\
         \x20       take(out);\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         }\n",
        "reachable",
    );
}

/// Two reachable branches, moved on one: still a maybe-move, still rejected.
#[test]
fn a_move_on_one_of_two_reachable_branches_is_still_rejected() {
    expect_reject(
        "fn take(value: String) {}\n\
         fn main() {\n\
         \x20   let mut out = String::new();\n\
         \x20   let flag = true;\n\
         \x20   if flag {\n\
         \x20       take(out);\n\
         \x20   } else {\n\
         \x20       out.push('b');\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         }\n",
        "onereachable",
    );
}

/// A `match` arm that reaches the join still contributes its move.
#[test]
fn a_move_in_a_reachable_match_arm_is_still_rejected() {
    expect_reject(
        "fn take(value: String) {}\n\
         fn main() {\n\
         \x20   let mut out = String::new();\n\
         \x20   let value: Option<Int32> = None;\n\
         \x20   match value {\n\
         \x20       Some(_n) => {\n\
         \x20           take(out);\n\
         \x20       }\n\
         \x20       None => {}\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         }\n",
        "reachablearm",
    );
}

/// The branch returns, but the move happens BEFORE the `if` — so it is on every path and must
/// still be rejected. Pins that the repair excludes a terminating branch's OWN moves, not moves
/// that merely precede one.
#[test]
fn a_move_before_a_terminating_branch_is_still_rejected() {
    expect_reject(
        "fn take(value: String) {}\n\
         fn build(flag: Bool) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   take(out);\n\
         \x20   if flag {\n\
         \x20       return String::new();\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(false).as_str()); }\n",
        "movebefore",
    );
}

/// A `match` where EVERY arm terminates: the join is unreachable, but a move made before the
/// `match` must still be remembered rather than silently dropped by the empty merge.
#[test]
fn a_move_before_an_all_diverging_match_is_still_rejected() {
    expect_reject(
        "fn take(value: String) {}\n\
         fn build(value: Option<Int32>) -> String {\n\
         \x20   let mut out = String::new();\n\
         \x20   take(out);\n\
         \x20   match value {\n\
         \x20       Some(_n) => {\n\
         \x20           return String::new();\n\
         \x20       }\n\
         \x20       None => {\n\
         \x20           return String::new();\n\
         \x20       }\n\
         \x20   }\n\
         \x20   out.push('a');\n\
         \x20   out\n\
         }\n\
         fn main() { println(build(None).as_str()); }\n",
        "alldiverge",
    );
}
