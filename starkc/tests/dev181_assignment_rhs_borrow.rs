//! **DEV-181 — a borrow taken by an assignment's own right-hand side must not block it.**
//!
//! `n = n.deeper()` was refused: the `Assign` arm called `check_expr(rhs)`, which pushed the
//! receiver auto-borrow from `n.deeper()`, then ran the write check with that borrow still on the
//! stack. Same mechanism as DEV-137 — nothing popped it between the two halves of one assignment.
//!
//! **The repair is gated, and the gate is the point.** DEV-137's `check_condition` truncates
//! unconditionally, which is safe for a condition because its value is consumed by the branch. An
//! assignment is different: the RHS's borrow is sometimes the assigned VALUE. `n.deeper()` yields an
//! owned `Node` whose temporary's borrow dies with it; `r = &v.field` yields a reference whose
//! borrow must survive. Truncating that would hand out a reference the checker had stopped
//! tracking.
//!
//! So the accepting tests below prove the idiom works, and the rejecting ones prove the checker did
//! not simply get weaker — which is the only thing that distinguishes this repair from a hole.

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
    let checked = typecheck::analyze(&hir);
    out.extend(
        checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message)),
    );
    out.extend(
        starkc::borrowck::check(
            &hir,
            &checked.tables.expr_types,
            &checked.tables.local_types,
        )
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message)),
    );
    out
}

// ------------------------------------------------------------------------------------- ACCEPT --

/// **The reported idiom.** Updating a value through a method that returns a new one, with no
/// hoisting workaround available — the statement previously had to be split in two.
#[test]
fn a_value_may_be_reassigned_from_a_method_on_itself() {
    let diags = errors(
        "\
struct N {
    v: Int32,
}

impl N {
    fn next(&self) -> N {
        N { v: self.v + 1 }
    }
}

fn main() {
    let mut n = N { v: 0 };
    n = n.next();
    println(n.v);
}
",
    );
    assert!(diags.is_empty(), "{diags:?}");
}

/// The same shape through the standard library, where the RHS borrows the target to build an owned
/// replacement.
#[test]
fn a_string_may_be_reassigned_from_a_borrow_of_itself() {
    let diags = errors(
        "\
fn main() {
    let mut s = String::from(\"  hi  \");
    s = String::from(s.trim());
    println(s.as_str());
}
",
    );
    assert!(diags.is_empty(), "{diags:?}");
}

/// In a loop, which is where the idiom is most useful and where DEV-137's sibling defect lived.
#[test]
fn the_idiom_works_inside_a_loop() {
    let diags = errors(
        "\
struct N {
    v: Int32,
}

impl N {
    fn doubled(&self) -> N {
        N { v: self.v * 2 }
    }
}

fn main() {
    let mut n = N { v: 1 };
    let mut i = 0;
    while i < 3 {
        n = n.doubled();
        i = i + 1;
    }
    println(n.v);
}
",
    );
    assert!(diags.is_empty(), "{diags:?}");
}

// ------------------------------------------------------------------------------------- REJECT --

/// **The control that matters.** A borrow taken BEFORE the assignment and still live must still
/// block it — the repair releases only the assignment's own temporaries.
#[test]
fn an_outstanding_borrow_still_blocks_an_assignment() {
    let diags = errors(
        "\
struct W {
    a: Int32,
}

fn main() {
    let mut w = W { a: 1 };
    let r = &w.a;
    w = W { a: 5 };
    println(*r);
}
",
    );
    assert!(
        diags.iter().any(|d| d.starts_with("E0101")),
        "a live borrow must still block the assignment: {diags:?}"
    );
}

/// A reference-valued RHS: the borrow IS the value being stored, so it must survive and must still
/// conflict. This is the case an ungated truncation would have broken.
#[test]
fn a_reference_valued_assignment_keeps_its_borrow() {
    let diags = errors(
        "\
struct W {
    a: Int32,
    b: Int32,
}

fn main() {
    let mut w = W { a: 1, b: 2 };
    let mut r: &Int32 = &w.a;
    r = &w.b;
    w = W { a: 9, b: 9 };
    println(*r);
}
",
    );
    assert!(
        diags.iter().any(|d| d.starts_with("E0101")),
        "a stored reference must still be tracked: {diags:?}"
    );
}
