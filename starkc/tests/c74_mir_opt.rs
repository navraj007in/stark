//! WP-C7.4 — the baseline MIR optimisations, checked against unoptimised MIR.
//!
//! Gate C7 fixes an order of authority: unoptimised MIR outranks optimised MIR. So the shape of
//! every test here is the same and is not negotiable — run the program twice, once as lowered and
//! once optimised, and require the two §39 observations to be **identical**: stdout, stderr, exit
//! status, returned value, trap category, trap location, and the drop log.
//!
//! Two properties are asserted alongside agreement, and both matter:
//!
//! 1. **The optimised program still VERIFIES.** An optimiser that produced ill-formed MIR but
//!    happened to run correctly under the interpreter would pass an agreement-only test and then
//!    fail in a backend. `mir::verify` is the same gate the build path applies.
//! 2. **The pass actually FIRED.** A test that only checks agreement is satisfied by an optimiser
//!    that does nothing at all — which is the most likely way for this suite to rot. Each case
//!    below states which counter must be non-zero, so a pass that silently stops working fails
//!    here rather than quietly costing performance.
//!
//! The negative cases are as important as the positive ones. Floating-point arithmetic and
//! `CheckIndex` must NOT fold, and a trap must survive folding with its category and line intact.

mod support;

use starkc::mir::opt::{optimise, OptStats};
use starkc::mir::verify::verify_program;
use starkc::mir::MirProgram;
use support::differential::{
    canonical_form, first_difference, front_end, run_mir, run_native, rustc_available, Observation,
};

use starkc::mir::lower::lower_program;

fn lower(tag: &str, source: &str) -> MirProgram {
    let front = front_end(tag, source);
    match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(error) => panic!("{tag}: lowering failed: {}", error.what),
    }
}

/// Run `source` as lowered and as optimised, require identical observations, and return the stats
/// so the caller can assert which pass fired.
fn agree(tag: &str, source: &str) -> (OptStats, Observation) {
    let plain = lower(tag, source);
    verify_program(&plain)
        .unwrap_or_else(|errors| panic!("{tag}: UNOPTIMISED MIR failed to verify: {errors:?}"));
    let plain_observation = run_mir(tag, &plain);

    let mut optimised = plain.clone();
    let stats = optimise(&mut optimised);
    verify_program(&optimised).unwrap_or_else(|errors| {
        panic!("{tag}: OPTIMISED MIR failed to verify — the optimiser produced ill-formed MIR, which is a defect in the pass, never a reason to relax the verifier: {errors:?}")
    });
    let optimised_observation = run_mir(tag, &optimised);

    if let Some(field) = first_difference(&plain_observation, &optimised_observation) {
        panic!(
            "{tag}: optimised MIR observes differently from unoptimised MIR on `{field}`.\n\
             Unoptimised MIR is the higher authority, so this is an optimiser defect.\n\
             --- unoptimised ---\n{}\n--- optimised ---\n{}",
            canonical_form(&plain_observation),
            canonical_form(&optimised_observation)
        );
    }
    (stats, optimised_observation)
}

/// The same agreement, extended to the native backend — the optimised program must also compile and
/// observe identically once compiled, not merely under the MIR interpreter.
fn agree_including_native(tag: &str, source: &str) -> OptStats {
    let (stats, mir_observation) = agree(tag, source);
    if !rustc_available() {
        eprintln!("SKIP native half of {tag}: no rustc");
        return stats;
    }
    let plain = lower(tag, source);
    let mut optimised = plain.clone();
    optimise(&mut optimised);
    let native_plain = run_native(tag, "plain", &plain);
    let native_optimised = run_native(tag, "opt", &optimised);
    for (label, other) in [
        ("unoptimised native", &native_plain),
        ("optimised native", &native_optimised),
    ] {
        if let Some(field) = first_difference(&mir_observation, other) {
            panic!(
                "{tag}: {label} disagrees with optimised MIR on `{field}`.\n--- mir ---\n{}\n--- {label} ---\n{}",
                canonical_form(&mir_observation),
                canonical_form(other)
            );
        }
    }
    stats
}

// ------------------------------------------------------------- constant folding --

/// Integer arithmetic on literals folds, and the folded program computes the same answer.
#[test]
fn integer_arithmetic_on_constants_folds() {
    let stats = agree_including_native(
        "c74_int_fold",
        r#"
fn main() {
    let a: Int32 = 6 * 7 + 2 - 2;
    print(a);
}
"#,
    );
    assert!(
        stats.checked_folded > 0,
        "checked integer arithmetic on constants should fold: {stats:?}"
    );
}

/// Comparisons and boolean operators fold through the same evaluator the interpreter uses.
#[test]
fn comparisons_and_boolean_operators_fold() {
    let stats = agree_including_native(
        "c74_bool_fold",
        r#"
fn main() {
    let a: Bool = 3 < 4;
    let b: Bool = !a;
    if b { print("unexpected"); } else { print("ok"); }
}
"#,
    );
    assert!(
        stats.rvalues_folded > 0,
        "a comparison and a `!` on constants should fold: {stats:?}"
    );
}

// ----------------------------------------------------------------- trap semantics --

/// **The central trap case.** A division by zero written entirely in constants must still trap, with
/// the same category and the same line, after folding. An optimiser that evaluated `1/0` into a
/// value would delete an abort the language requires in every build mode.
#[test]
fn constant_division_by_zero_still_traps_at_the_same_place() {
    let (stats, observation) = agree(
        "c74_div_zero",
        r#"
fn main() {
    let d: Int32 = 0;
    let a: Int32 = 10 / d;
    print(a);
}
"#,
    );
    assert!(
        stats.checked_trapped > 0,
        "a constant division by zero should be PROVEN to trap, not folded to a value: {stats:?}"
    );
    let Observation::Trapped(trap) = observation else {
        panic!("a constant division by zero must still trap after folding, got {observation:?}");
    };
    assert_eq!(
        trap.category,
        starkc::mir::TrapCategory::DivideByZero,
        "the folded trap must keep the original category"
    );
    assert_eq!(
        trap.line, 4,
        "the folded trap must keep the original source line"
    );
}

/// Overflow that is provable at compile time is still a runtime trap, not a compile error and not a
/// wrapped value.
#[test]
fn constant_overflow_still_traps() {
    let (stats, observation) = agree(
        "c74_overflow",
        r#"
fn main() {
    let big: Int32 = 2147483647;
    let one: Int32 = 1;
    let sum: Int32 = big + one;
    print(sum);
}
"#,
    );
    assert!(
        stats.checked_trapped > 0,
        "should prove the trap: {stats:?}"
    );
    let Observation::Trapped(trap) = observation else {
        panic!("constant overflow must trap, got {observation:?}");
    };
    assert_eq!(trap.category, starkc::mir::TrapCategory::IntegerOverflow);
}

/// A shift with a constant out-of-range count reports `InvalidShift`, not the terminator's own
/// category. The interpreter applies that override; a fold that forgot it would report
/// `IntegerOverflow` for a program the interpreter calls `InvalidShift`.
#[test]
fn folded_shift_keeps_the_invalid_shift_override() {
    let (stats, observation) = agree(
        "c74_shift",
        r#"
fn main() {
    let value: Int32 = 1;
    let count: Int32 = 64;
    let shifted: Int32 = value << count;
    print(shifted);
}
"#,
    );
    assert!(
        stats.checked_trapped > 0,
        "should prove the trap: {stats:?}"
    );
    let Observation::Trapped(trap) = observation else {
        panic!("an out-of-range shift must trap, got {observation:?}");
    };
    assert_eq!(
        trap.category,
        starkc::mir::TrapCategory::InvalidShift,
        "the category override must survive folding"
    );
}

/// Statements before a folded trap still run. This is what makes a folded trap a trap rather than a
/// compile-time rejection: the output written before it is part of the observation.
#[test]
fn output_before_a_folded_trap_is_preserved() {
    let (_, observation) = agree(
        "c74_trap_ordering",
        r#"
fn main() {
    print("before");
    let d: Int32 = 0;
    let a: Int32 = 1 / d;
    print(a);
}
"#,
    );
    let Observation::Trapped(trap) = observation else {
        panic!("expected a trap, got {observation:?}");
    };
    assert_eq!(
        trap.stdout_before_trap,
        b"before".to_vec(),
        "everything printed before a folded trap must still appear"
    );
}

// ------------------------------------------------- branch folding and dead blocks --

/// A constant condition collapses to a `Goto`, which makes the untaken arm unreachable, which
/// dead-block elimination then removes. The two passes are only useful together, so they are
/// asserted together.
#[test]
fn a_constant_condition_removes_the_untaken_arm() {
    let stats = agree_including_native(
        "c74_branch",
        r#"
fn main() {
    let take: Bool = true;
    if take { print("taken"); } else { print("untaken"); }
}
"#,
    );
    assert!(
        stats.branches_folded > 0,
        "a constant condition should fold to a Goto: {stats:?}"
    );
    assert!(
        stats.blocks_removed > 0,
        "the untaken arm should then become unreachable and be removed: {stats:?}"
    );
}

// -------------------------------------------------------- what must NOT be folded --

/// **Float arithmetic must not fold.** The interpreter computes in `f64` and a backend may compute
/// a `Float32` expression in `f32`; folding with the interpreter's answer would make the native
/// result depend on whether an operand happened to be a literal. Asserted as a hard zero rather
/// than left to the module comment, so a later "improvement" that admits floats has to confront it.
#[test]
fn float_arithmetic_is_not_folded() {
    let plain = lower(
        "c74_float",
        r#"
fn main() {
    let a: Float64 = 1.0;
    let b: Float64 = 3.0;
    let c: Float64 = a / b;
    print(c);
}
"#,
    );
    let mut optimised = plain.clone();
    let stats = optimise(&mut optimised);
    assert_eq!(
        stats.checked_folded, 0,
        "float division must not be folded — see the f32/f64 note in mir::opt: {stats:?}"
    );
    assert_eq!(
        stats.rvalues_folded, 0,
        "no float rvalue may be folded: {stats:?}"
    );
}

/// Indexing must keep its bounds check. `CheckIndex` produces an opaque proof token rather than a
/// value, and folding it away would break the discipline the verifier enforces on `Index`.
#[test]
fn a_constant_index_keeps_its_bounds_check() {
    let stats = agree_including_native(
        "c74_index",
        r#"
fn main() {
    let xs: [Int32; 3] = [10, 20, 30];
    let i: Int32 = 1;
    print(xs[i]);
}
"#,
    );
    let _ = stats;
    let plain = lower(
        "c74_index_shape",
        r#"
fn main() {
    let xs: [Int32; 3] = [10, 20, 30];
    let i: Int32 = 1;
    print(xs[i]);
}
"#,
    );
    let mut optimised = plain.clone();
    optimise(&mut optimised);
    assert!(
        mentions_check_index(&optimised),
        "the bounds check must survive optimisation"
    );
}

fn mentions_check_index(program: &MirProgram) -> bool {
    program.bodies.iter().any(|body| {
        body.blocks.iter().any(|block| {
            matches!(
                &block.terminator.0,
                starkc::mir::Terminator::Checked {
                    op: starkc::mir::CheckedOp::CheckIndex,
                    ..
                }
            )
        })
    })
}

/// An out-of-bounds constant index still traps at run time rather than being folded away, and it
/// reports `IndexOutOfBounds`.
#[test]
fn a_constant_out_of_bounds_index_still_traps() {
    let (_, observation) = agree(
        "c74_index_oob",
        r#"
fn main() {
    let xs: [Int32; 3] = [10, 20, 30];
    let i: Int32 = 7;
    print(xs[i]);
}
"#,
    );
    let Observation::Trapped(trap) = observation else {
        panic!("an out-of-bounds index must trap, got {observation:?}");
    };
    assert_eq!(trap.category, starkc::mir::TrapCategory::IndexOutOfBounds);
}

// ----------------------------------------------------------- drop-log preservation --

/// **The drop log is observable output**, so an optimiser that removed a "useless" local would
/// change the program's meaning. Destructor order and count must be identical either way.
#[test]
fn drop_order_and_count_are_unchanged() {
    let stats = agree_including_native(
        "c74_drops",
        "struct Loud { id: Int32 }\n\
         impl Drop for Loud {\n    fn drop(&mut self) {\n        print(\"@@stark-drop:Loud#\");\n\
         print(self.id);\n        println(\"@@\");\n    }\n}\n\
         fn main() {\n    let unused: Bool = true;\n    let a: Loud = Loud { id: 1 };\n\
         let b: Loud = Loud { id: 2 };\n    println(\"body\");\n}\n",
    );
    // `unused` is a constant local that nothing reads. It may be propagated, but the two `Noisy`
    // values must still be dropped, in order — which `agree` has already checked byte for byte.
    let _ = stats;
}

// ------------------------------------------------------------------- idempotence --

/// Optimising an already-optimised program must change nothing. A pass that kept finding new work
/// on its own output would either loop to the round bound on every build or, worse, be rewriting
/// something it should have left alone.
#[test]
fn the_optimiser_reaches_a_fixpoint() {
    for (tag, source) in [
        (
            "c74_fix_arith",
            "fn main() { let a: Int32 = 2 + 3 * 4; print(a); }",
        ),
        (
            "c74_fix_branch",
            "fn main() { let t: Bool = false; if t { print(\"a\"); } else { print(\"b\"); } }",
        ),
    ] {
        let mut program = lower(tag, source);
        let first = optimise(&mut program);
        assert!(
            first.total() > 0,
            "{tag}: nothing to optimise, so this test would pass vacuously"
        );
        let second = optimise(&mut program);
        assert_eq!(
            second.total(),
            0,
            "{tag}: a second run found more work, so the first did not reach a fixpoint: {second:?}"
        );
    }
}

/// Optimisation must be deterministic: the same source twice produces byte-identical MIR. The build
/// key is computed from the optimised program, so a nondeterministic pass would give one source two
/// cache entries and defeat WP-C7.3.
#[test]
fn optimisation_is_deterministic() {
    let source = r#"
fn main() {
    let a: Int32 = 6 * 7;
    let t: Bool = a == 42;
    if t { print("yes"); } else { print("no"); }
}
"#;
    let mut first = lower("c74_determinism", source);
    let mut second = lower("c74_determinism", source);
    optimise(&mut first);
    optimise(&mut second);
    assert_eq!(
        first.dump(),
        second.dump(),
        "optimised MIR must be identical across runs, or the C7.3 build key is unstable"
    );
}
