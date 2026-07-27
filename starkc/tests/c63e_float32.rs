//! WP-C6.3e / DEV-105 — `Float32` renders at its DECLARED width (0.1-A9).
//!
//! PRINT-DISPLAY-001: a finite float uses "the fewest significant decimal digits that parse back to
//! the same **declared** IEEE value". For a `Float32` that value is the f32, so `0.1f32` prints
//! `0.1` — not `0.10000000149011612`, which is the shortest round-trip of the f64 it becomes when
//! widened.
//!
//! Before 0.1-A9 there was no width-preserving print operation: `Float32` was cast to `Float64` and
//! printed through `PrintFloat64`. The engines then disagreed — HIR rounds to f32 and formats at f32
//! (`0.1`), MIR happened to agree because its constant never actually narrows, and NATIVE held a real
//! f32, widened it, and printed the long form. `PrintFloat32`/`PrintlnFloat32` carry the declared
//! width in the OPERATION's identity: the verifier requires a `Float32` operand, the MIR interpreter
//! narrows its f64 storage at that boundary, and the backend calls an `f32` runtime function. All
//! three then route through the one `canonical_float32`.

mod support;

use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Delegates to the shared comparator (R-02), keeping this suite's independent stdout pin.
fn agree(tag: &str, src: &str, expect: &str) {
    support::differential::agree_completing_with_stdout(tag, src, expect);
}

/// All three engines print exactly `expect`.
// ---- The scalar case DEV-105 was named for ----

#[test]
fn println_float32_uses_the_declared_width() {
    agree(
        "scalar",
        "fn main() { let x: Float32 = 0.1f32; println(x); }",
        "0.1\n",
    );
}

#[test]
fn print_float32_without_newline() {
    agree(
        "scalar_print",
        "fn main() { let x: Float32 = 0.1f32; print(x); }",
        "0.1",
    );
}

/// The case that makes width substitution VISIBLE. `0.1f32` widened to f64 is exactly
/// `0.10000000149011612`, so an engine printing the f64 shortest form of a `Float32` would differ here
/// and nowhere else — a value like `2.5` is exact in both widths and cannot detect the substitution.
///
/// The two values are written as INDEPENDENT literals rather than derived by a cast, deliberately:
/// `0.1f32 as Float64` exposes a *different* defect (DEV-109 — MIR does not round a `Float32` to f32
/// precision, so the cast is a no-op there while HIR rounds), and this test is about DISPLAY width,
/// not cast semantics. Conflating them would leave one failure masking the other.
#[test]
fn a_value_whose_f32_and_f64_renderings_differ() {
    agree(
        "visible_diff",
        "fn main() { let x: Float32 = 0.1f32; let y: Float64 = 0.10000000149011612; \
         println(x); println(y); }",
        "0.1\n0.10000000149011612\n",
    );
}

// ---- Composite contexts: the refusal DEV-105 forced is gone ----

#[test]
fn float32_in_a_tuple() {
    agree(
        "tuple",
        "fn main() { println((0.1f32, 1.0f32)); }",
        "(0.1, 1.0)\n",
    );
}

#[test]
fn float32_in_an_array() {
    agree(
        "array",
        "fn main() { let a: [Float32; 2] = [0.1f32, 0.2f32]; println(a); }",
        "[0.1, 0.2]\n",
    );
}

#[test]
fn float32_in_an_option() {
    agree(
        "option",
        "fn main() { let o: Option<Float32> = Some(0.1f32); println(o); }",
        "Some(0.1)\n",
    );
}

#[test]
fn float32_in_a_result() {
    agree(
        "result",
        "fn main() { let r: Result<Float32, Int32> = Ok(0.1f32); println(r); }",
        "Ok(0.1)\n",
    );
}

#[test]
fn float32_in_a_vec() {
    agree(
        "vec",
        "fn main() { let mut v: Vec<Float32> = Vec::new(); v.push(0.1f32); v.push(0.25f32); \
         println(v); }",
        "[0.1, 0.25]\n",
    );
}

// ---- The IEEE edges ----

#[test]
fn negative_zero_keeps_its_sign() {
    agree(
        "negzero",
        "fn main() { let x: Float32 = -0.0f32; println(x); }",
        "-0.0\n",
    );
}

/// Infinities render at the declared width. `-inf` is produced by negation, which is exact.
///
/// **This case was originally written with NaN and division-by-zero construction EXCLUDED**, because
/// both were blocked by defects outside Display: DEV-110 (MIR trapped on float division by zero,
/// under a superseded decision) and DEV-109 (`Float32` arithmetic was carried in f64 and rounded
/// only at display, so an overflowing product stayed finite at `3.4e39` and `inf - inf` gave `0.0`).
/// Both are now closed — CD-139 and CD-140 — and the cases they blocked are covered at the end of
/// this file. This one is kept as written: it reaches an infinity by OVERFLOW rather than by
/// division, which is the only route that exercises the rounding CD-140 added.
#[test]
fn infinities_render_at_the_declared_width() {
    agree(
        "edges",
        "fn main() { let big: Float32 = 3.4028235e38f32; let inf: Float32 = big * 10.0f32; \
         let ninf: Float32 = -inf; println(inf); println(ninf); }",
        "inf\n-inf\n",
    );
}

/// The largest finite f32 and the smallest positive subnormal — the extremes of the format, where a
/// renderer working at the wrong width produces visibly different digits.
#[test]
fn max_finite_and_min_subnormal() {
    agree(
        "extremes",
        "fn main() { let big: Float32 = 3.4028235e38f32; let tiny: Float32 = 1e-45f32; \
         println(big); println(tiny); }",
        "3.4028235e38\n1e-45\n",
    );
}

// ---------------------------------------------------------------------------------------------
// MUTATION AND VERSION EVIDENCE — that the width is carried by the OPERATION, and that the surface
// revision it lives in is enforced.
// ---------------------------------------------------------------------------------------------

fn lower_only(tag: &str, src: &str) -> starkc::mir::MirProgram {
    let file = Arc::new(SourceFile::new(
        format!("c63ef32_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    lower_program(&hir, &checked.tables, file).unwrap_or_else(|e| panic!("{tag} lower: {}", e.what))
}

/// **The mutation DEV-105 is defined by.** Rewriting `PrintlnFloat32` back to `PrintlnFloat64` —
/// the pre-A9 lowering, exactly — must be REJECTED, and by VERIFICATION rather than by a downstream
/// output comparison.
///
/// That is the whole point of giving the operation its own identity instead of a widening
/// convention. Under the old scheme the two prints were the same operation applied to a widened
/// operand, so nothing in the IR recorded that a `Float32` had been printed and nothing could check
/// it: the loss of width was invisible until an engine rendered the value, and only NATIVE rendered
/// it differently. Now the declared width is a typed operand, so the substitution is a well-formedness
/// error a single engine catches on its own — no differential run required.
#[test]
fn substituting_the_float64_print_is_rejected_by_verification() {
    use starkc::mir::{Callee, RuntimeFn, Terminator};
    let mut program = lower_only(
        "mutwidth",
        "fn main() { let x: Float32 = 0.1f32; println(x); }",
    );
    assert!(
        verify_program(&program).is_ok(),
        "the unmutated program must verify, or this mutation proves nothing"
    );

    let mut swapped = 0;
    for body in &mut program.bodies {
        for block in &mut body.blocks {
            if let Terminator::Call {
                callee: Callee::Runtime(rt @ RuntimeFn::PrintlnFloat32),
                ..
            } = &mut block.terminator.0
            {
                *rt = RuntimeFn::PrintlnFloat64;
                swapped += 1;
            }
        }
    }
    assert_eq!(
        swapped, 1,
        "the program must contain exactly one Float32 print"
    );

    let errors = match verify_program(&program) {
        Err(e) => e,
        Ok(_) => panic!("printing a Float32 through the Float64 print must fail"),
    };
    assert!(
        !errors.is_empty(),
        "verification must name the operand-type mismatch"
    );
}

/// **Surface-revision enforcement.** `PrintFloat32`/`PrintlnFloat32` are additive, so they are a
/// RUNTIME SURFACE revision (0.1-A8 → 0.1-A9), not a MIR shape version change. An A9 build must
/// accept an A9 program and reject an A8 one — and reject it at the §A1 gate, BEFORE consuming any
/// body, because an A8 consumer given an A9 program would meet a `RuntimeFn` it cannot lower.
#[test]
fn the_a9_surface_is_accepted_and_a8_is_rejected() {
    let mut program = lower_only(
        "version",
        "fn main() { let x: Float32 = 0.1f32; println(x); }",
    );
    assert_eq!(
        program.runtime_surface,
        starkc::mir::MIR_RUNTIME_SURFACE,
        "lowering must stamp the surface this build produces"
    );
    assert_eq!(
        starkc::mir::MIR_RUNTIME_SURFACE,
        "0.1-A9",
        "CE3 set the surface to 0.1-A9"
    );
    assert!(
        verify_program(&program).is_ok(),
        "an A9 program must verify"
    );

    program.runtime_surface = "0.1-A8".to_string();
    let errors = match verify_program(&program) {
        Err(e) => e,
        Ok(_) => panic!("an A8 program must be rejected by an A9 build"),
    };
    assert_eq!(
        errors.len(),
        1,
        "the surface gate rejects before any body: {errors:?}"
    );
    assert_eq!(errors[0].code, "MIR-0017", "expected the surface gate");
    assert!(
        errors[0].message.contains("0.1-A8") && errors[0].message.contains("0.1-A9"),
        "the mismatch must name both surfaces: {}",
        errors[0].message
    );
}

// ---------------------------------------------------------------------------------------------
// CD-140 / DEV-109 — `Float32` VALUE semantics, not just `Float32` rendering.
//
// DEV-105 gave `Float32` a print operation that respects the declared width. That fixed DISPLAY.
// It did not make the VALUE binary32: both interpreters carry every float in an f64, so a
// `Float32` local could hold a value no f32 can represent, and only the printer rounded it. The
// value and its rendering disagreed — a number that PRINTS as `inf` while arithmetic still treats
// it as finite is worse than one that prints wrongly, because the display looks right.
//
// NUM-FLOAT-FORMAT-001 requires IEEE binary32 for `Float32`, and NUM-FLOAT-REPRO-001 requires the
// same result bits for the same declared type, inputs, and sequence of operations. Both are about
// VALUES. CD-140 rounds at three points: a `Float32` literal is the nearest binary32
// (NUM-FLOAT-LIT-001 converts a decimal literal directly to the DESTINATION format), an
// integer-to-`Float32` cast rounds once (NUM-FLOAT-CONV-001), and any assignment to a `Float32`
// destination rounds — which is where arithmetic lands. The HIR oracle already did all three.
//
// These cases were UNREACHABLE until CD-139: constructing an `inf` or a `NaN` needs a division by
// zero, and that trapped in MIR.
// ---------------------------------------------------------------------------------------------

/// **The literal.** `0.1f32` denotes the f32 nearest 0.1, whose exact f64 value is
/// `0.10000000149011612`. Widening it must expose those digits. MIR used to store the f64 nearest
/// 0.1 and hand it back unchanged, so the cast was a no-op there and a real rounding in HIR.
#[test]
fn widening_a_float32_literal_shows_it_was_narrowed() {
    agree(
        "widen_literal",
        "fn main() { let x: Float32 = 0.1f32; let y: Float64 = x as Float64; println(y); }",
        "0.10000000149011612\n",
    );
}

/// **Arithmetic.** Each step rounds to binary32, so the widened result carries f32 rounding at
/// every step rather than f64 rounding throughout with one narrowing at the end.
#[test]
fn float32_arithmetic_rounds_at_every_step() {
    agree(
        "arith_rounds",
        "fn main() { let a: Float32 = 0.1f32; let b: Float32 = 0.2f32; let c: Float32 = a + b; \
         println(c as Float64); }",
        "0.30000001192092896\n",
    );
}

/// **Overflow becomes a real infinity, not a large finite number.** This is the case that exposed
/// DEV-109: the product exceeds binary32's range, so it must BE `inf` — and `inf - inf` must
/// therefore be `NaN`. Before CD-140 the value stayed finite (`3.4e39`) and merely PRINTED as
/// `inf`, so the subtraction quietly gave `0.0`.
#[test]
fn float32_overflow_produces_a_real_infinity() {
    agree(
        "overflow_inf",
        "fn main() { let big: Float32 = 3.4028235e38f32; let inf: Float32 = big * 10.0f32; \
         println(inf); println(inf - inf); println(inf as Float64); }",
        "inf\nNaN\ninf\n",
    );
}

/// **Underflow to zero** is the other end of the same property: a product below binary32's
/// smallest subnormal is exactly zero, not a tiny f64.
#[test]
fn float32_underflow_reaches_zero() {
    agree(
        "underflow",
        "fn main() { let tiny: Float32 = 1e-45f32; let t: Float32 = tiny * 0.01f32; \
         println(t); println(t as Float64); }",
        "0.0\n0.0\n",
    );
}

/// **Integer-to-`Float32` conversion rounds once.** `16777217` is the first integer binary32
/// cannot represent (2^24 + 1); it must round to `16777216.0`. Sharing the `Float64` conversion
/// path would have preserved the odd value.
#[test]
fn integer_to_float32_rounds_to_binary32() {
    agree(
        "int_to_f32",
        "fn main() { let n: Int32 = 16777217; let f: Float32 = n as Float32; \
         println(f); println(f as Float64); }",
        "16777216.0\n16777216.0\n",
    );
}

/// **A `Float32` division by zero** — the CD-139 rule at the narrower width, and the shortest route
/// to an infinity that does not depend on overflow.
#[test]
fn float32_division_by_zero_is_a_real_infinity() {
    agree(
        "f32_div_zero",
        "fn main() { let z: Float32 = 0.0f32; let o: Float32 = 1.0f32; let i: Float32 = o / z; \
         println(i); println(i - i); println(i as Float64); }",
        "inf\nNaN\ninf\n",
    );
}

/// **NaN survives a widening cast** rather than becoming a finite f64 — and stays unordered.
#[test]
fn float32_nan_widens_to_nan() {
    agree(
        "f32_nan_widen",
        "fn main() { let z: Float32 = 0.0f32; let n: Float32 = z / z; \
         println(n); println(n as Float64); println(n == n); }",
        "NaN\nNaN\nfalse\n",
    );
}

/// **Accumulation in a loop**, where a missing per-step rounding compounds instead of cancelling.
/// Ten additions of `0.1f32` diverge visibly from the f64 computation of the same sum.
#[test]
fn float32_accumulation_rounds_each_iteration() {
    agree(
        "accumulate",
        "fn main() { let mut acc: Float32 = 0.0f32; let mut i: Int32 = 0; \
         while i < 10 { acc = acc + 0.1f32; i = i + 1; } println(acc as Float64); }",
        "1.0000001192092896\n",
    );
}
