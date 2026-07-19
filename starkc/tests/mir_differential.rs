//! WP-C4.4 — HIR vs MIR differential validation.
//!
//! THE Gate C4 comparator (roadmap WP-C4.4): for each frozen workload the scalar core can
//! lower, `HIR interpreter output/failure == MIR interpreter output/failure`. The HIR
//! interpreter is the semantic oracle (charter §1.6 rule 6); the MIR pipeline under test is
//! lower → verify → execute. Success cases compare stdout byte-for-byte and exit status; trap
//! cases require BOTH engines to trap, with the MIR trap category consistent with the oracle's
//! trap message.
//!
//! Corpus cases come from the frozen `exec_snapshots` corpus (`corpus_version = 1.0.0`);
//! inline programs cover scalar shapes the corpus reaches only via constructs the scalar core
//! defers to C4.5 (fn values, Option/Result, structs without methods).

use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::{run_program, MirRunError};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::TrapCategory;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

struct Front {
    hir: starkc::hir::Hir,
    file: Arc<SourceFile>,
    tables: starkc::typecheck::TypeTables,
}

fn front_end(name: &str, source: String) -> Front {
    let file = Arc::new(SourceFile::new(name, source));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: typecheck: {errors:?}");
    Front {
        hir,
        file,
        tables: checked.tables,
    }
}

/// The message fragment the HIR oracle uses for each MIR trap category (the observable
/// comparator's trap-category correspondence).
fn oracle_fragment(category: TrapCategory) -> &'static str {
    match category {
        TrapCategory::IntegerOverflow => "integer overflow",
        TrapCategory::DivideByZero => "division by zero",
        TrapCategory::IndexOutOfBounds => "out of bounds",
        TrapCategory::CastFailure => "cast",
        TrapCategory::Panic => "panic",
        TrapCategory::UnwrapNone => "unwrap",
        TrapCategory::UnwrapErr => "unwrap",
        TrapCategory::AssertFailure => "assert",
    }
}

fn differential(name: &str, source: String) {
    let front = front_end(name, source);

    // Oracle.
    let oracle = interp::run(&front.hir, front.file.clone(), &front.tables);

    // MIR pipeline: lower -> verify -> execute.
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
    };
    if let Err(errors) = verify_program(&program) {
        panic!("{name}: verifier rejected lowered MIR:\n{errors:#?}");
    }
    let mir_result = run_program(&program);

    match (oracle, mir_result) {
        (Ok(oracle_exec), Ok(mir_exec)) => {
            assert_eq!(
                mir_exec.output, oracle_exec.output,
                "{name}: DIFFERENTIAL STDOUT MISMATCH\n--- HIR oracle ---\n{}\n--- MIR ---\n{}",
                oracle_exec.output, mir_exec.output
            );
            assert_eq!(
                mir_exec.status, oracle_exec.status,
                "{name}: exit status mismatch"
            );
        }
        (Err(oracle_err), Err(MirRunError::Trap { category })) => {
            assert!(
                oracle_err.is_trap,
                "{name}: oracle errored non-trap ({}) but MIR trapped {category:?}",
                oracle_err.message
            );
            assert!(
                oracle_err.message.contains(oracle_fragment(category)),
                "{name}: trap category mismatch — MIR {category:?} vs oracle message {:?}",
                oracle_err.message
            );
        }
        (Ok(exec), Err(mir_err)) => panic!(
            "{name}: oracle succeeded (stdout {:?}) but MIR failed: {mir_err:?}",
            exec.output
        ),
        (Err(oracle_err), Ok(mir_exec)) => panic!(
            "{name}: oracle trapped ({}) but MIR succeeded with stdout {:?} — A MISSED TRAP",
            oracle_err.message, mir_exec.output
        ),
        (Err(oracle_err), Err(MirRunError::Internal(message))) => panic!(
            "{name}: MIR internal error ({message}) while oracle said: {}",
            oracle_err.message
        ),
    }
}

#[test]
fn frozen_corpus_scalar_cases_agree() {
    for name in [
        "expr_stmt__01_arithmetic_and_precedence",
        "expr_stmt__03_loops_break_continue",
        "primitive__01_integer_widths_and_overflow_traps",
        "primitive__02_integer_overflow_traps",
        "struct_enum_trait__02_enum_and_pattern_match",
    ] {
        let path = corpus_dir().join(format!("{name}.stark"));
        let source = std::fs::read_to_string(&path).unwrap();
        differential(&path.to_string_lossy(), source);
    }
}

#[test]
fn function_values_agree() {
    differential(
        "fnval.stark",
        "fn double(x: Int32) -> Int32 { x * 2 } \
         fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 { f(v) } \
         fn main() { \
             let f: fn(Int32) -> Int32 = double; \
             println(f(21)); \
             println(apply(double, 5)); \
             println(apply(f, 7)); \
             println(f(f(10))); \
         }"
        .to_string(),
    );
}

#[test]
fn option_result_agree() {
    differential(
        "opt.stark",
        "fn half(n: Int32) -> Option<Int32> { \
             if n % 2 == 0 { Some(n / 2) } else { None } \
         } \
         fn main() { \
             match half(10) { \
                 Some(v) => println(v), \
                 None => println(0 - 1), \
             } \
             match half(7) { \
                 Some(v) => println(v), \
                 None => println(0 - 1), \
             } \
             let r: Result<Int32, Bool> = Ok(4); \
             match r { \
                 Ok(v) => println(v), \
                 Err(flag) => println(flag), \
             } \
         }"
        .to_string(),
    );
}

#[test]
fn structs_and_tuples_agree() {
    differential(
        "structs.stark",
        "struct Point { x: Int32, y: Int32 } \
         fn make(x: Int32, y: Int32) -> Point { Point { x: x, y: y } } \
         fn main() { \
             let p = make(3, 4); \
             println(p.x * p.x + p.y * p.y); \
             let t = (1, true); \
             println(t.0); \
             println(t.1); \
         }"
        .to_string(),
    );
}

#[test]
fn division_by_zero_trap_agrees() {
    differential(
        "divzero.stark",
        "fn main() { \
             let a = 10; \
             let b = 0; \
             println(a / b); \
         }"
        .to_string(),
    );
}

#[test]
fn mid_output_trap_agrees_on_failure_not_output() {
    // A trap after partial output: both engines must trap; the comparator's contract is
    // failure-equality (the oracle's Execution is consumed whole, so partial stdout before a
    // trap is not observable through interp::run's error path — category equality is).
    differential(
        "midtrap.stark",
        "fn main() { \
             println(1); \
             let big: Int8 = 120i8; \
             let other: Int8 = 100i8; \
             println(big + other); \
         }"
        .to_string(),
    );
}

#[test]
fn recursion_and_calls_agree() {
    differential(
        "fib.stark",
        "fn fib(n: Int32) -> Int32 { \
             if n < 2 { n } else { fib(n - 1) + fib(n - 2) } \
         } \
         fn main() { \
             let mut i = 0; \
             while i < 10 { \
                 println(fib(i)); \
                 i = i + 1; \
             } \
         }"
        .to_string(),
    );
}
