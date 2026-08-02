//! **DEV-137: a borrow created solely to evaluate a `while` condition ends before the body.**
//!
//! `03-Type-System.md` "References and Lifetimes": a temporary borrow ends with its statement. The
//! auto-borrow a method call takes of its receiver (TYPE-METHOD-002) is a temporary, so the one
//! `values.len()` takes in a loop condition must not survive into the body.
//!
//! # The mechanism, so a future reader does not re-derive it
//!
//! `Borrowck::active_borrows` is a stack scoped by exactly two mechanisms: `check_block` truncates
//! to its entry depth at block end, and `check_stmt` truncates after each expression statement. A
//! `while` condition is NEITHER — it is an expression evaluated outside any statement of its own.
//! The `While` arm pushed the condition's borrows and then called `check_block(body)`, which
//! records its own entry depth AFTER that push and therefore cannot pop it. The repair snapshots
//! the depth BEFORE the condition and truncates to it before entering the body.
//!
//! # Why the negative controls are the interesting half
//!
//! The risk is ending borrows that are still genuinely live. A borrow created BEFORE the loop sits
//! at a shallower stack depth than the snapshot, so the truncate cannot reach it — and
//! `borrow_predating_the_loop_stays_live` is the test that proves the boundary is depth-based
//! rather than "clear everything at the loop header". `for x in &v` is deliberately untouched for
//! the same reason: its iterator borrow MUST span the body.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Returns the first error diagnostic as `"CODE message"`, or `None` when the program checks.
fn check(src: &str, tag: &str) -> Option<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
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

fn expect_reject(src: &str, tag: &str) -> String {
    match check(src, tag) {
        Some(diagnostic) => {
            assert!(
                diagnostic.starts_with("E0101"),
                "{tag}: expected E0101, got: {diagnostic}"
            );
            diagnostic
        }
        None => panic!("{tag}: expected rejection, but the program checked clean"),
    }
}

/// Runs the program through the HIR oracle and returns its stdout.
fn run(src: &str, tag: &str) -> String {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error),
        "{tag}: check: {:?}",
        checked.diagnostics
    );
    match starkc::interp::run(&hir, file, &checked.tables) {
        Ok(execution) => execution.output,
        Err(error) => panic!("{tag}: runtime error: {}", error.message),
    }
}

// -------------------------------------------------------------------- must pass --

/// The CD-334 reproducer: the ordinary indexed loop.
#[test]
fn condition_receiver_borrow_ends_before_the_body() {
    expect_accept(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   values.push(2);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < values.len() {\n\
         \x20       values[i] = 5;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(values[0] + values[1]);\n\
         }\n",
        "indexedloop",
    );
}

/// ...and it must actually EXECUTE correctly, not merely check. A borrow-region change that
/// accidentally altered evaluation order would pass a checker-only test.
#[test]
fn the_indexed_loop_executes_correctly() {
    let out = run(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   values.push(2);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < values.len() {\n\
         \x20       values[i] = 5;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(values[0] + values[1]);\n\
         }\n",
        "indexedexec",
    );
    assert_eq!(out.trim(), "10", "both elements must have been written");
}

/// **The condition is re-evaluated every iteration.** This is what makes hoisting `len()` a
/// SEMANTIC change rather than a stylistic one, and it is why the repair had to be a region fix.
/// The loop grows the vector it is measuring; a hoisted length would stop at the original bound.
#[test]
fn a_growing_vector_re_evaluates_its_condition() {
    let out = run(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < values.len() {\n\
         \x20       if values.len() < 5u64 {\n\
         \x20           values.push(1);\n\
         \x20       }\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(values.len());\n\
         }\n",
        "growing",
    );
    assert_eq!(
        out.trim(),
        "5",
        "the condition must see each newly pushed element"
    );
}

/// A `&mut` PARAMETER, which is the shape every mutating helper has. Same code path — nothing
/// about the receiver being a parameter mattered; what mattered was `len()` in the condition.
#[test]
fn a_mutable_parameter_receiver_is_accepted() {
    expect_accept(
        "fn fill(values: &mut Vec<Int32>, value: Int32) {\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < values.len() {\n\
         \x20       values[i] = value;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<Int32> = Vec::new();\n\
         \x20   v.push(1);\n\
         \x20   fill(&mut v, 9);\n\
         \x20   println(v[0]);\n\
         }\n",
        "mutparam",
    );
}

/// The condition borrows through a FIELD and the body mutates the field's owner.
#[test]
fn a_field_receiver_borrow_ends_before_the_body() {
    expect_accept(
        "struct Buffer { data: Vec<Int32> }\n\
         fn main() {\n\
         \x20   let mut buffer = Buffer { data: Vec::new() };\n\
         \x20   buffer.data.push(1);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < buffer.data.len() {\n\
         \x20       buffer.data[i] = 7;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(buffer.data[0]);\n\
         }\n",
        "fieldrecv",
    );
}

/// **Multiple temporary borrows in one condition**, of two different receivers, both mutated in
/// the body. Truncation must cover every borrow the condition created, not just the last.
#[test]
fn multiple_condition_temporaries_are_all_released() {
    expect_accept(
        "fn main() {\n\
         \x20   let mut a: Vec<Int32> = Vec::new();\n\
         \x20   let mut b: Vec<Int32> = Vec::new();\n\
         \x20   a.push(1);\n\
         \x20   b.push(2);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < a.len() && i < b.len() {\n\
         \x20       a[i] = 8;\n\
         \x20       b[i] = 9;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(a[0] + b[0]);\n\
         }\n",
        "multitemp",
    );
}

/// **DEV-137 x DEV-132.** An indexed READ in the condition takes a `VecGetRef`-backed borrow of
/// the element's place; it must end at the same boundary a method receiver's does, so the body can
/// legally write to the same vector.
#[test]
fn an_indexed_place_borrow_in_the_condition_ends_before_the_body() {
    let out = run(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(0);\n\
         \x20   values.push(0);\n\
         \x20   while values[0u64] < 3 {\n\
         \x20       values[0u64] = values[0u64] + 1;\n\
         \x20       values[1u64] = values[1u64] + 10;\n\
         \x20   }\n\
         \x20   println(values[0] + values[1]);\n\
         }\n",
        "vecgetref",
    );
    assert_eq!(out.trim(), "33", "3 iterations: 3 + 30");
}

/// Nested loops over independent receivers, each with its own condition borrow.
#[test]
fn nested_loops_with_independent_receivers_are_accepted() {
    expect_accept(
        "fn main() {\n\
         \x20   let mut outer: Vec<Int32> = Vec::new();\n\
         \x20   let mut inner: Vec<Int32> = Vec::new();\n\
         \x20   outer.push(1);\n\
         \x20   inner.push(2);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < outer.len() {\n\
         \x20       let mut j = 0u64;\n\
         \x20       while j < inner.len() {\n\
         \x20           inner[j] = 4;\n\
         \x20           j = j + 1u64;\n\
         \x20       }\n\
         \x20       outer[i] = 3;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(outer[0] + inner[0]);\n\
         }\n",
        "nested",
    );
}

/// A condition that borrows a receiver the body does NOT mutate must remain fine — the repair
/// must not disturb the ordinary case.
#[test]
fn a_condition_borrow_without_body_mutation_is_unaffected() {
    expect_accept(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   let mut i = 0u64;\n\
         \x20   let mut total = 0;\n\
         \x20   while i < values.len() {\n\
         \x20       total = total + values[i];\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(total);\n\
         }\n",
        "readonly",
    );
}

/// **The same defect lived in `if` conditions**, and the growing-vector case above is what
/// exposed it: `if values.len() < 5u64 { values.push(1); }` was refused for exactly the same
/// reason. A condition is a condition whether it guards a loop or a branch, so the rule is
/// written once (`check_condition`) and used by both rather than special-cased for `while`.
#[test]
fn an_if_condition_receiver_borrow_ends_before_its_block() {
    expect_accept(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   if values.len() < 5u64 {\n\
         \x20       values.push(2);\n\
         \x20   }\n\
         \x20   println(values.len());\n\
         }\n",
        "ifcond",
    );
}

/// ...including the `else` branch, which is a different block off the same condition.
#[test]
fn an_if_condition_borrow_ends_before_the_else_block() {
    expect_accept(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   if values.len() > 5u64 {\n\
         \x20       values.push(2);\n\
         \x20   } else {\n\
         \x20       values.push(3);\n\
         \x20   }\n\
         \x20   println(values.len());\n\
         }\n",
        "elsecond",
    );
}

// ------------------------------------------------------------------ must reject --

/// **The boundary control.** A borrow bound with `let` BEFORE the loop is lexically scoped to the
/// end of its block, so it is still live in the body and the mutation must stay refused. It sits
/// at a shallower stack depth than the condition snapshot, which is precisely why a depth-based
/// truncate cannot reach it. If this test ever passes, the repair has become "clear all borrows at
/// the loop header" and has stopped modelling anything.
#[test]
fn borrow_predating_the_loop_stays_live() {
    expect_reject(
        "fn observe(values: &Vec<Int32>) -> UInt64 { values.len() }\n\
         fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   let view = &values;\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < 1u64 {\n\
         \x20       values[0u64] = 5;\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   println(observe(view));\n\
         }\n",
        "predating",
    );
}

/// The same shape with the pre-existing borrow held across a `loop` rather than a `while`, so the
/// rejection cannot be an artefact of the `While` arm specifically.
#[test]
fn borrow_predating_a_loop_expression_stays_live() {
    expect_reject(
        "fn observe(values: &Vec<Int32>) -> UInt64 { values.len() }\n\
         fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   let view = &values;\n\
         \x20   loop {\n\
         \x20       values[0u64] = 5;\n\
         \x20       break;\n\
         \x20   }\n\
         \x20   println(observe(view));\n\
         }\n",
        "predatingloop",
    );
}

/// A borrow created inside the BODY and still live when the body mutates through the owner must
/// stay rejected. The repair narrows the condition's region only; the body's own regions are
/// untouched.
#[test]
fn a_body_local_borrow_still_conflicts() {
    expect_reject(
        "fn observe(values: &Vec<Int32>) -> UInt64 { values.len() }\n\
         fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < 1u64 {\n\
         \x20       let view = &values;\n\
         \x20       values[0u64] = 5;\n\
         \x20       println(observe(view));\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         }\n",
        "bodylocal",
    );
}

/// **A `match` scrutinee's borrow must KEEP spanning the arms.** PAT-BIND-001 binds a non-Copy
/// payload BY REFERENCE into the scrutinee, so an arm that mutates the scrutinee's owner while a
/// binding still references it is a genuine conflict. Truncating a scrutinee the way a condition
/// is truncated would hand out references to storage the checker had stopped tracking.
///
/// `Match` is not routed through `check_condition` by construction — this test is what would fail
/// if someone later "generalised" the repair to every operand that precedes a block.
#[test]
fn a_match_scrutinee_borrow_still_spans_the_arms() {
    expect_reject(
        "enum Holder { Full(String), Empty }\n\
         fn main() {\n\
         \x20   let mut holder = Holder::Full(String::from(\"x\"));\n\
         \x20   let view = &holder;\n\
         \x20   match *view {\n\
         \x20       Holder::Full(text) => {\n\
         \x20           holder = Holder::Empty;\n\
         \x20           println(text.as_str());\n\
         \x20       }\n\
         \x20       Holder::Empty => {}\n\
         \x20   }\n\
         }\n",
        "matchscrutinee",
    );
}

/// **`for x in &v` must keep its iterator borrow across the body.** This is the control that stops
/// the repair from being generalised to "loop headers" as a category: the `For` arm's borrow is
/// not a condition temporary, it is live for the whole loop.
#[test]
fn for_loop_iterator_borrow_still_spans_the_body() {
    expect_reject(
        "fn main() {\n\
         \x20   let mut values: Vec<Int32> = Vec::new();\n\
         \x20   values.push(1);\n\
         \x20   for value in &values {\n\
         \x20       values[0u64] = *value;\n\
         \x20   }\n\
         }\n",
        "foriter",
    );
}
