//! WP-C5.2c bring-up proof: real arithmetic (with overflow trapping), comparisons, `if`/`else`,
//! and a `while` loop -- not just straight-line locals (C5.2b) -- compile and run natively.
//! Direct/indirect calls and printed output are WP-C5.2d; this file stays narrow to the C5.2c
//! proof, matching the scope discipline of `native_c5_1b_skeleton.rs`/`native_c5_2b_locals.rs`.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::process::Command;
use std::sync::Arc;

fn rustc_available() -> bool {
    Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn compile_and_run(source: &str, tag: &str) -> std::process::Output {
    let file = Arc::new(SourceFile::new(
        format!("c5_2c_{tag}.stark"),
        source.to_string(),
    ));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "{tag} parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "{tag} resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let type_errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(type_errors.is_empty(), "{tag} typecheck: {type_errors:?}");

    let mir_program = match lower_program(&hir, &checked.tables, file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{tag} must lower: {} @ {:?}", e.what, e.span),
    };
    let verified = match verify_program(&mir_program) {
        Ok(v) => v,
        Err(errors) => panic!("{tag}'s MIR must verify: {errors:?}"),
    };

    let target_dir = std::env::temp_dir().join(format!("stark_c5_2c_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let options = NativeBuildOptions {
        target_dir: target_dir.clone(),
        target_contract: "stark-64-v1".to_string(),
    };
    let artifact = emit_native_debug(&verified, &options)
        .unwrap_or_else(|e| panic!("{tag} native build failed: {e:?}"));

    let run = Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    let _ = std::fs::remove_dir_all(&target_dir);
    run
}

#[test]
fn checked_arithmetic_and_comparisons_succeed_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // Every result is OBSERVED, via `assert_eq` in the STARK program itself. Computing these
    // values and checking only for exit 0 would pass against a backend that returned zero from
    // every operation -- the test would prove the program COMPILES, not that it computes. Native
    // `println` is still out of scope (WP-C5.3), so a failed `assert_eq` -- which `mir::lower`
    // emits as a message-less `Terminator::Trap`, supported natively as of WP-C5.2e -- is the
    // observation channel: exit 0 now means every assertion below held.
    let source = r#"
fn main() {
    let a: Int32 = 10;
    let b: Int32 = 3;
    assert_eq(a + b, 13);
    assert_eq(a - b, 7);
    assert_eq(a * b, 30);
    assert_eq(a / b, 3);
    assert_eq(a % b, 1);
    assert_eq(a << 2, 40);
    assert_eq(a >> 1, 5);
    assert_eq(a & b, 2);
    assert_eq(a | b, 11);
    assert_eq(a ^ b, 9);
    assert_eq(-a, -10);
    assert(a > b);
    assert(!(a == b));
    assert(a != b);
    assert(a >= b);
    let f: Float64 = 3.5;
    let g: Float64 = 1.5;
    assert_eq(f + g, 5.0);
    assert_eq(f - g, 2.0);
    assert_eq(f * g, 5.25);
    assert_eq(f / g, 2.3333333333333335);
    assert_eq(-f, -3.5);
    assert(f > g);
}
"#;
    let run = compile_and_run(source, "arith");
    assert_eq!(
        run.status.code(),
        Some(0),
        "every in-program assertion must hold; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// The negative control for every `assert_eq`-based test in this file and in
/// `native_c5_2d_calls.rs`. Without it, "exit 0" is ambiguous: a backend that lowered assertions
/// to nothing at all would also exit 0, and every assertion above would be decorative. This
/// proves a FALSE assertion really does reach the trap ABI and fail the process.
#[test]
fn a_false_assertion_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 10;
    assert_eq(a + 1, 12);
}
"#;
    let run = compile_and_run(source, "assert_control");
    assert_eq!(
        run.status.code(),
        Some(101),
        "a false assertion must trap (exit 101), not pass silently"
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("assertion failed"),
        "stderr must name the assertion failure: {stderr}"
    );
}

#[test]
fn integer_overflow_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 2147483647;
    let b: Int32 = 1;
    let c: Int32 = a + b;
}
"#;
    let run = compile_and_run(source, "overflow");
    // WP-C5.2e settled the exact trap exit code (101, matching `stark run`'s own convention);
    // tightened from a loose `assert_ne!` now that the real contract exists.
    assert_eq!(
        run.status.code(),
        Some(101),
        "overflow must trap (exit 101), not silently wrap or succeed"
    );
}

#[test]
fn division_by_zero_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 10;
    let b: Int32 = 0;
    let c: Int32 = a / b;
}
"#;
    let run = compile_and_run(source, "divzero");
    assert_eq!(
        run.status.code(),
        Some(101),
        "division by zero must trap (exit 101)"
    );
}

#[test]
fn if_else_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // Both arms are exercised and the SELECTED one is observed -- a backend that always took the
    // then-branch (or always the else-branch) would exit 0 under a bare compile-and-run check.
    let source = r#"
fn main() {
    let a: Int32 = 5;
    let mut b: Int32 = 0;
    if a > 3 {
        b = 1;
    } else {
        b = 2;
    }
    assert_eq(b, 1);

    let mut c: Int32 = 0;
    if a > 10 {
        c = 1;
    } else {
        c = 2;
    }
    assert_eq(c, 2);
}
"#;
    let run = compile_and_run(source, "ifelse");
    assert_eq!(
        run.status.code(),
        Some(0),
        "the branch actually taken must match the condition; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn while_loop_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // The final counter AND an accumulator are observed: the counter alone would be satisfied by
    // a loop that ran the right number of times but executed a wrong body, and the accumulator
    // alone would be satisfied by the wrong trip count with compensating arithmetic.
    let source = r#"
fn main() {
    let mut i: Int32 = 0;
    let mut total: Int32 = 0;
    while i < 5 {
        i = i + 1;
        total = total + i;
    }
    assert_eq(i, 5);
    assert_eq(total, 15);

    // A loop whose condition is false on entry must run its body zero times.
    let mut never: Int32 = 0;
    while never > 10 {
        never = never + 1;
    }
    assert_eq(never, 0);
}
"#;
    let run = compile_and_run(source, "whileloop");
    assert_eq!(
        run.status.code(),
        Some(0),
        "loop trip count and body effects must both be correct; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// The 64-bit float→integer cast boundary, NATIVELY -- the third engine for the boundary cases
/// `mir_differential.rs` pins for the HIR oracle and the MIR interpreter. The native backend
/// shared the MIR interpreter's defect: both compared the truncated value against `max as f64`,
/// which rounds UP at 64-bit widths (`u64::MAX as f64` is 2^64), so exactly 2^64 passed the guard
/// and Rust's saturating `as` clamped it to `u64::MAX` instead of trapping.
#[test]
fn float_to_int_cast_boundaries_agree_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // Every value here is exactly representable as f64, so the literals are exact: 2^64 - 2048
    // is the greatest f64 below 2^64, and 2^63 - 1024 the greatest below 2^63.
    //
    // Two pre-existing FRONT-END limitations shape how the expectations are written; neither is
    // related to the cast bounds under test, and neither is worked around in a way that weakens
    // the check:
    //
    //  1. No integer literal above `Int64::MAX` is expressible at all -- an unsuffixed literal
    //     types as `Int64` first, so even `let x: UInt64 = 18446744073709549568;` is rejected as
    //     "integer literal out of range for 'Int64'". The near-2^64 result is therefore pinned by
    //     halving it: 2^64 - 2048 is even, and its half (2^63 - 1024) IS below `Int64::MAX`, so
    //     the exact converted value is still fully determined.
    //  2. `Int64::MIN` has no literal spelling (`9223372036854775808` overflows `Int64` before
    //     the unary minus applies), so it is built as `-9223372036854775807 - 1`.
    //
    // Expected values are also bound through an annotated `let` rather than written inline as
    // `assert_eq`'s second argument, because an unsuffixed literal in argument position types
    // independently of the first argument and defaults to `Int64`.
    let source = r#"
fn main() {
    let below_u64: Float64 = 18446744073709549568.0;
    let converted_u64: UInt64 = below_u64 as UInt64;
    let expect_half: UInt64 = 9223372036854774784;
    assert_eq(converted_u64 / 2, expect_half);

    let below_i64: Float64 = 9223372036854774784.0;
    let expect_i64: Int64 = 9223372036854774784;
    assert_eq(below_i64 as Int64, expect_i64);

    let at_i64_min: Float64 = -9223372036854775808.0;
    let expect_min: Int64 = -9223372036854775807 - 1;
    assert_eq(at_i64_min as Int64, expect_min);

    // Truncation toward zero, then the range check -- not a trap for merely having a fraction.
    let fractional: Float64 = 42.9;
    assert_eq(fractional as Int32, 42);
    let negative_fraction: Float64 = -42.9;
    assert_eq(negative_fraction as Int32, -42);
}
"#;
    let run = compile_and_run(source, "cast_ok");
    assert_eq!(
        run.status.code(),
        Some(0),
        "in-range casts at the 64-bit boundary must convert; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// Exactly 2^64 is one past `UInt64::MAX` and must TRAP natively rather than saturate. The
/// primary native regression test for the rounded-bound defect.
#[test]
fn float_to_uint64_at_two_pow_64_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let at_bound: Float64 = 18446744073709551616.0;
    let converted: UInt64 = at_bound as UInt64;
}
"#;
    let run = compile_and_run(source, "cast_u64_trap");
    assert_eq!(
        run.status.code(),
        Some(101),
        "2^64 must trap, not saturate to UInt64::MAX"
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("cast failure"),
        "stderr must name the cast failure: {stderr}"
    );
}

/// The signed twin: exactly 2^63 is one past `Int64::MAX`.
#[test]
fn float_to_int64_at_two_pow_63_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let at_bound: Float64 = 9223372036854775808.0;
    let converted: Int64 = at_bound as Int64;
}
"#;
    let run = compile_and_run(source, "cast_i64_trap");
    assert_eq!(
        run.status.code(),
        Some(101),
        "2^63 must trap, not saturate to Int64::MAX"
    );
}
