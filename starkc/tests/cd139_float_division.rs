//! CD-139 / DEV-110 — floating division and remainder are TOTAL; integer division still traps.
//!
//! **NUM-INT-DIV-001:** "Integer division by zero and remainder by zero trap."
//! **NUM-FLOAT-OP-001:** "Floating division by zero does not trap: it produces the IEEE infinity or
//! NaN result." Floating `%` "with a zero divisor, infinite dividend, or NaN operand produces NaN."
//!
//! Those two rules are adjacent in `CORE-V1-ABSTRACT-MACHINE.md` (lines 544 and 570) and split the
//! integer and floating cases deliberately. MIR used to trap on BOTH, under **CD-006** — an owner
//! ruling that arbitrated a since-deleted sentence in `03-Type-System.md` ("Division or modulo by
//! zero is a runtime error and MUST trap") nine hours before WP-C2.9 replaced it with the paired
//! rules above. The HIR oracle never had a float trap, so the engines disagreed on a program both
//! accepted: HIR yielded `inf`, MIR trapped `DivideByZero`. The owner ruled CD-006 **superseded by
//! succession of authority, not reversed on its merits**.
//!
//! Lowering now emits `MirBinOp::FloatDiv`/`FloatRem` (a narrow additive MIR amendment, A6) instead
//! of `CheckedOp::FloatDiv`/`FloatRem`. The checked variants remain, deprecated and unreachable:
//! keeping a TOTAL operation in the checked family would preserve the enum shape while corrupting
//! its contract — a primitive declared trapping that is guaranteed never to trap.
//!
//! Half of this file exists to guard the OVER-correction. "Division by zero no longer traps" is
//! true of floats and false of integers, and a fix that reads the headline instead of the rule
//! would silently make integer division total too. Those cases must still trap, in every engine.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct Compiled {
    program: starkc::mir::MirProgram,
    hir_output: Option<String>,
}

fn front_end(tag: &str, src: &str) -> Compiled {
    let file = Arc::new(SourceFile::new(
        format!("cd139_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag} typecheck: {errs:?}");
    let hir_output = interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .map(|e| e.output)
        .ok();
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    Compiled {
        program,
        hir_output,
    }
}

/// All three engines run to completion and print exactly `expect`.
fn agree(tag: &str, src: &str, expect: &str) {
    let Compiled {
        program,
        hir_output,
    } = front_end(tag, src);
    assert_eq!(
        hir_output.as_deref(),
        Some(expect),
        "{tag}: HIR output (the oracle)"
    );

    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir =
        run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR must not trap: {:?}", f.error));
    assert_eq!(mir.output, expect, "{tag}: MIR output");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_cd139_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("{tag} native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(run.status.success(), "{tag}: native must exit 0, not trap");
        assert_eq!(
            String::from_utf8_lossy(&run.stdout),
            expect,
            "{tag}: native output"
        );
    }
}

/// Every engine TRAPS. Used for the integer cases NUM-INT-DIV-001 still governs.
fn traps(tag: &str, src: &str) {
    let Compiled {
        program,
        hir_output,
    } = front_end(tag, src);
    assert!(
        hir_output.is_none(),
        "{tag}: HIR must trap, got output {hir_output:?}"
    );

    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    assert!(run_program(verified).is_err(), "{tag}: MIR must trap");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_cd139t_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("{tag} native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(run.status.code(), Some(101), "{tag}: native must abort");
        let stderr = String::from_utf8_lossy(&run.stderr);
        // `TrapCategory::DivideByZero` renders as this message (`mir_differential.rs`'s
        // category table); the category NAME never reaches stderr.
        assert!(
            stderr.contains("division by zero"),
            "{tag}: native must report a division-by-zero trap: {stderr}"
        );
    }
}

// ---- Floating division: TOTAL (NUM-FLOAT-OP-001) ----

/// The case DEV-110 was found on. HIR yielded `inf`, MIR trapped.
#[test]
fn float64_division_by_zero_yields_signed_infinity() {
    agree(
        "f64_div_zero",
        "fn main() { let z: Float64 = 0.0; println(1.0 / z); println(-1.0 / z); }",
        "inf\n-inf\n",
    );
}

/// `0.0 / 0.0` is the NaN case — an indeterminate form, not an infinity.
#[test]
fn float64_zero_over_zero_is_nan() {
    agree(
        "f64_zero_over_zero",
        "fn main() { let z: Float64 = 0.0; println(z / z); }",
        "NaN\n",
    );
}

/// The divisor's SIGN selects the infinity's sign: IEEE division by negative zero is not the same
/// as by positive zero, which a "return infinity on zero divisor" shortcut would miss.
#[test]
fn negative_zero_divisor_flips_the_infinity() {
    agree(
        "f64_neg_zero_divisor",
        "fn main() { let nz: Float64 = -0.0; println(1.0 / nz); println(-1.0 / nz); }",
        "-inf\ninf\n",
    );
}

/// NUM-FLOAT-OP-001 for `%`: "zero divisor, infinite dividend, or NaN operand produces NaN".
/// All three producers, since each reaches the NaN by a different route.
#[test]
fn float64_remainder_produces_nan() {
    agree(
        "f64_rem_nan",
        "fn main() { let z: Float64 = 0.0; let inf: Float64 = 1.0 / z; \
         println(5.0 % z); println(inf % 2.0); println((z / z) % 2.0); }",
        "NaN\nNaN\nNaN\n",
    );
}

/// An ordinary remainder still computes: making the operation total must not make it vacuous.
#[test]
fn float64_remainder_still_computes() {
    agree(
        "f64_rem_ok",
        "fn main() { let a: Float64 = 7.5; let b: Float64 = 2.0; println(a % b); \
         println(-7.5 % b); }",
        "1.5\n-1.5\n",
    );
}

/// `Float32` takes the same path — the rule is stated for binary32 and binary64 alike.
#[test]
fn float32_division_by_zero_yields_infinity() {
    agree(
        "f32_div_zero",
        "fn main() { let z: Float32 = 0.0f32; let o: Float32 = 1.0f32; \
         println(o / z); println(-o / z); println(z / z); }",
        "inf\n-inf\nNaN\n",
    );
}

/// NaN PROPAGATES through ordinary arithmetic rather than being confined to the operation that
/// created it — the property that makes a total division usable instead of merely non-trapping.
#[test]
fn nan_propagates_through_arithmetic() {
    agree(
        "nan_propagates",
        "fn main() { let z: Float64 = 0.0; let n: Float64 = z / z; \
         println(n + 1.0); println(n * 2.0); println(n - n); }",
        "NaN\nNaN\nNaN\n",
    );
}

/// `inf - inf` — an indeterminate form that was UNREACHABLE before CD-139, because constructing an
/// infinity required a division by zero and that trapped. DEV-105's evidence file had to skip the
/// NaN cases for exactly this reason.
#[test]
fn infinity_minus_infinity_is_nan() {
    agree(
        "inf_minus_inf",
        "fn main() { let z: Float64 = 0.0; let inf: Float64 = 1.0 / z; \
         println(inf - inf); println(inf + inf); println(inf * 0.0); }",
        "NaN\ninf\nNaN\n",
    );
}

/// Comparisons against NaN are all false, `!=` excepted — so a total division does not quietly
/// break ordering. (NaN is unordered, not "some large value".)
#[test]
fn nan_comparisons_are_unordered() {
    agree(
        "nan_compare",
        "fn main() { let z: Float64 = 0.0; let n: Float64 = z / z; \
         println(n == n); println(n != n); println(n < 1.0); println(n > 1.0); }",
        "false\ntrue\nfalse\nfalse\n",
    );
}

// ---- Integer division: STILL TRAPS (NUM-INT-DIV-001). The over-correction guard. ----

/// NUM-INT-DIV-001 is untouched by CD-139. This is the case a fix applied to "division by zero"
/// generally — rather than to the floating rule specifically — would silently break.
#[test]
fn integer_division_by_zero_still_traps() {
    traps(
        "int_div_zero",
        "fn main() { let z: Int32 = 0; println(1 / z); }",
    );
}

#[test]
fn integer_remainder_by_zero_still_traps() {
    traps(
        "int_rem_zero",
        "fn main() { let z: Int32 = 0; println(1 % z); }",
    );
}

/// Unsigned integers take the same path as signed ones.
#[test]
fn unsigned_division_by_zero_still_traps() {
    traps(
        "uint_div_zero",
        "fn main() { let z: UInt32 = 0u32; let a: UInt32 = 1u32; println(a / z); }",
    );
}

// ---- Amendment shape ----

/// A6 is ADDITIVE: the deprecated `CheckedOp::FloatDiv`/`FloatRem` still exist, and lowering no
/// longer emits them. Both halves matter — the first keeps the amendment additive, the second is
/// the actual semantic change, and only the second is observable from a program's behaviour.
#[test]
fn lowering_no_longer_emits_the_checked_float_ops() {
    use starkc::mir::{CheckedOp, Terminator};
    let Compiled { program, .. } = front_end(
        "shape",
        "fn main() { let z: Float64 = 0.0; let a: Float64 = 1.0; \
         println(a / z); println(a % z); println(a * z); }",
    );
    for body in &program.bodies {
        for (i, block) in body.blocks.iter().enumerate() {
            if let Terminator::Checked { op, .. } = &block.terminator.0 {
                assert!(
                    !matches!(op, CheckedOp::FloatDiv | CheckedOp::FloatRem),
                    "{}: bb{i} still emits the deprecated checked float op {op:?}",
                    body.instance.symbol
                );
            }
        }
    }
}
