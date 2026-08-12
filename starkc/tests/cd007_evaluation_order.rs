//! **CD-007's evaluation order, made observable.**
//!
//! The module header of `mir/lower.rs` says evaluation order is *"preserved **structurally**"* — by
//! the order in which operands are lowered into temporaries. A structural guarantee is one no type
//! rule and no verifier check restates, so **if it is wrong, only execution can tell**, and only
//! then if a case exists where the order is observable.
//!
//! WP-ARCH-CLOSE AC4 found that no such case existed for assignment. `AC4-MUT-LOW-002` inverted
//! CD-007's RHS-before-LHS-place rule and **survived** `--lib`, `three_engine_differential`,
//! `mir_differential` and `conformance` — not because the code is unreached (every assignment
//! lowers through it) but because every assignment in the suite has an inert left-hand side.
//! `a = f()` cannot observe an ordering; `a[idx()] = val()` can.
//!
//! These cases exist to be the falsifier. Each is checked across all four engine configurations,
//! and the disagreement a broken lowering produces is HIR-versus-MIR: the HIR oracle evaluates in
//! its own order, so a lowering that diverges shows up as engines disagreeing rather than as a
//! wrong-looking answer everyone shares.

mod support;

/// **The assignment rule: RHS before the LHS place.**
///
/// Both sides print. Inverting the lowering order — the mutation that survived — makes this fail
/// with `HIR/MIR DISAGREEMENT on stdout_bytes`, which is exactly the shape a shared-fate defect
/// could never produce.
#[test]
fn an_assignment_evaluates_its_rhs_before_the_lhs_place() {
    support::differential::agree_completing_with_stdout(
        "cd007_assign_order",
        r#"
fn idx() -> UInt64 { println("idx"); 0u64 }
fn val() -> Int32 { println("rhs"); 7i32 }
fn main() {
    let mut a: [Int32; 2] = [0, 0];
    a[idx()] = val();
    if a[0u64] != 7i32 { panic("the assignment did not land"); }
}
"#,
        "rhs\nidx\n",
    );
}

/// **Short-circuiting, observed through effects rather than through the result.**
///
/// `AC4-MUT-LOW-001` — making both `&&` arms jump to the RHS block — was killed, but by a single
/// test. A language rule this basic deserves a case that states it directly: the right-hand side
/// must not run at all when the left decides the answer.
#[test]
fn a_false_left_operand_means_the_right_never_runs() {
    support::differential::agree_completing_with_stdout(
        "cd007_and_short",
        r#"
fn loud() -> Bool { println("rhs ran"); true }
fn main() {
    if false && loud() { panic("unreachable"); }
    println("done");
}
"#,
        "done\n",
    );
}

/// The `||` mirror: a true left operand must not evaluate the right.
#[test]
fn a_true_left_operand_of_or_means_the_right_never_runs() {
    support::differential::agree_completing_with_stdout(
        "cd007_or_short",
        r#"
fn loud() -> Bool { println("rhs ran"); false }
fn main() {
    if true || loud() { println("taken"); }
}
"#,
        "taken\n",
    );
}

/// **Short-circuiting is what makes a guarded index safe**, and this states that as a trap
/// property rather than an output one: if `&&` evaluated both sides, this program would trap on an
/// out-of-bounds read instead of completing.
#[test]
fn a_guarded_index_is_safe_only_because_the_right_side_is_skipped() {
    support::differential::agree_completing_with_stdout(
        "cd007_guarded_index",
        r#"
fn main() {
    let a: [Int32; 2] = [1, 2];
    let i: UInt64 = 5u64;
    if i < 2u64 && a[i] == 1i32 { println("in range"); } else { println("guarded"); }
}
"#,
        "guarded\n",
    );
}
