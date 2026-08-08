//! DEV-119 — a borrowed iterator's cursor must not outlive its loop.
//!
//! Found by the WP-C6.6 audit (CD-179) while verifying `HashMap::remove`/`clear`, and it is the
//! kind of defect that audit was built to expose: `HashMap::keys`, `Vec::iter` and `HashSet::iter`
//! all COUNTED as executable, because a minimal call lowers and verifies. What did not work was the
//! ordinary usage shape — finish the loop, then mutate the collection:
//!
//! ```text
//! HIR:     accepts and runs
//! MIR:     accepts and runs
//! native:  E0502, cannot borrow as mutable because it is also borrowed as immutable
//! ```
//!
//! A genuine three-engine disagreement, and therefore NOT carriable the way DEV-118 is: that one is
//! an enforcement omission every engine shares, this one is a program the semantic authority runs
//! and the backend refuses to build.
//!
//! **Scope was wider than first reported.** The initial note said "iterating a HashMap's keys".
//! Reproducing it with the fix reverted shows `HashMap::keys`, `HashSet::iter` AND `Vec::iter` all
//! failing; only `String::chars` escaped, because it yields `Char` BY VALUE and so holds no borrow.
//! The defect was never about maps — it was in the shared `for`-loop lowering, which is where the
//! fix belongs and where it went.
//!
//! **The fix** (Codex, reviewed and verified here): the cursor is given its own scope that ends at
//! the loop's exit block, so its slot drop — and with it the shared borrow — is emitted before any
//! following statement. Normal exhaustion and `break` both converge on that exit; `continue` keeps
//! the cursor live, which is what it must do.
//!
//! **The distinction this suite pins**, because the fix would be worse than the defect if it
//! relaxed the borrow rules instead of tightening the cursor's lifetime:
//!
//! ```text
//! iterator no longer live after the loop   ->  mutation MUST succeed
//! reference derived from it still live     ->  mutation MUST remain rejected
//! ```

mod support;

use support::differential::{agree_completing_with_stdout, rustc_available};

/// The loop is over, so the borrow is over: mutating the source must build and run.
fn mutation_after_the_loop_is_allowed(tag: &str, source: &str, expected: &str) {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    agree_completing_with_stdout(tag, source, expected);
}

/// The borrow is still live, so the mutation must be REFUSED — and refused for the RIGHT REASON.
///
/// The first version of this helper accepted any parse, resolve or type error as proof. That is too
/// weak to be evidence: a source with a typo would have "passed" while proving nothing about borrow
/// checking, and these three cases are the entire safety argument for the DEV-119 fix. So the shape
/// is pinned end to end — the program must PARSE and RESOLVE cleanly, and then be rejected by the
/// borrow checker specifically, with `E0101`.
fn mutation_while_borrowed_is_refused(tag: &str, source: &str) {
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(
        format!("dev119_{tag}.stark"),
        source.to_string(),
    ));
    let (ast, parse_diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(
        parse_diags.is_empty(),
        "{tag}: the source must PARSE — a parse error would make this test pass for the wrong \
         reason: {parse_diags:?}"
    );
    let (hir, resolve_diags) = starkc::resolve::resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "{tag}: the source must RESOLVE — an unresolved name would make this test pass for the \
         wrong reason: {resolve_diags:?}"
    );
    let checked = starkc::typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(
        !errors.is_empty(),
        "{tag}: mutating a collection while it is still borrowed was ACCEPTED — the DEV-119 fix \
         must shorten the CURSOR's lifetime, never relax the borrow rules"
    );
    assert!(
        errors.iter().any(|d| d.code.as_deref() == Some("E0101")),
        "{tag}: rejected, but NOT by the borrow checker. The expected diagnostic is E0101 \
         (a conflict with an active borrow); got: {:?}",
        errors
            .iter()
            .map(|d| (d.code.clone(), d.message.clone()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn exhausted_map_keys_then_insert() {
    mutation_after_the_loop_is_allowed(
        "dev119_keys_insert",
        "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 10); \
         for k in m.keys() { print(*k); } println(\"\"); m.insert(2, 20); println(m.len()); }",
        "1\n2\n",
    );
}

#[test]
fn exhausted_map_keys_then_remove() {
    mutation_after_the_loop_is_allowed(
        "dev119_keys_remove",
        "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 10); \
         for k in m.keys() { print(*k); } println(\"\"); let r: Option<Int32> = m.remove(&1); \
         println(m.len()); }",
        "1\n0\n",
    );
}

#[test]
fn exhausted_map_keys_then_clear() {
    mutation_after_the_loop_is_allowed(
        "dev119_keys_clear",
        "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 10); \
         for k in m.keys() { print(*k); } println(\"\"); m.clear(); println(m.len()); }",
        "1\n0\n",
    );
}

/// `break` leaves the cursor UNEXHAUSTED, and it still has to be dropped at the exit block.
#[test]
fn break_then_mutate() {
    mutation_after_the_loop_is_allowed(
        "dev119_break",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
         for x in v.iter() { print(*x); break; } println(\"\"); v.push(3); println(v.len()); }",
        "1\n3\n",
    );
}

/// `continue` must NOT drop the cursor — it goes round again.
#[test]
fn continue_then_mutate() {
    mutation_after_the_loop_is_allowed(
        "dev119_continue",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); \
         for x in v.iter() { if *x == 1 { continue; } print(*x); } println(\"\"); \
         v.push(3); println(v.len()); }",
        "2\n3\n",
    );
}

#[test]
fn exhausted_set_iter_then_remove() {
    mutation_after_the_loop_is_allowed(
        "dev119_set",
        "fn main() { let mut s: HashSet<Int32> = HashSet::new(); s.insert(1); \
         for x in s.iter() { print(*x); } println(\"\"); let r: Bool = s.remove(&1); \
         println(s.len()); }",
        "1\n0\n",
    );
}

#[test]
fn exhausted_vec_iter_then_push() {
    mutation_after_the_loop_is_allowed(
        "dev119_vec",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); \
         for x in v.iter() { print(*x); } println(\"\"); v.push(2); println(v.len()); }",
        "1\n2\n",
    );
}

/// Nested loops over the same source: the INNER cursor's scope must close without taking the
/// outer's with it.
#[test]
fn nested_loops_then_mutate() {
    mutation_after_the_loop_is_allowed(
        "dev119_nested",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); \
         for a in v.iter() { for b in v.iter() { print(*b); } } println(\"\"); \
         v.push(2); println(v.len()); }",
        "1\n2\n",
    );
}

/// `String::chars` yields `Char` BY VALUE, so it never held a borrow and never had the defect.
/// Kept so the family is covered uniformly rather than only where it happened to break.
#[test]
fn exhausted_chars_then_push() {
    mutation_after_the_loop_is_allowed(
        "dev119_chars",
        "fn main() { let mut s: String = String::from(\"a\"); for c in s.chars() { print(c); } \
         println(\"\"); s.push('b'); println(s.len()); }",
        "a\n2\n",
    );
}

// ---------------------------------------------------------------- safety --

/// A reference the iterator YIELDED, still live after the loop. The collection must stay frozen.
#[test]
fn a_held_yielded_reference_still_blocks_mutation() {
    mutation_while_borrowed_is_refused(
        "held_ref",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); \
         let mut held: &Int32 = &v[0u64]; for x in v.iter() { held = x; } v.push(2); \
         print(*held); }",
    );
}

#[test]
fn mutating_inside_the_loop_is_still_refused() {
    mutation_while_borrowed_is_refused(
        "inside_vec",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); \
         for x in v.iter() { v.push(2); } }",
    );
}

#[test]
fn mutating_a_map_inside_its_keys_loop_is_still_refused() {
    mutation_while_borrowed_is_refused(
        "inside_map",
        "fn main() { let mut m: HashMap<Int32, Int32> = HashMap::new(); m.insert(1, 10); \
         for k in m.keys() { m.insert(2, 20); } }",
    );
}
