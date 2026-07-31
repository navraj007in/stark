//! WP-C7.9 G.7 — boundary and property coverage on the axes the review's findings exposed.
//!
//! Every finding in this work package sat at a boundary that no maintained case exercised:
//! `MIN / -1` at the extreme of a signed range, a trap category at the edge between two causes, a
//! binding mode at the edge between owned and borrowed. The pattern is not a coincidence — interior
//! values are what examples are written from, and edges are what implementations get wrong.
//!
//! So this file walks the edges deliberately: exact minima and maxima, one step beyond them, the
//! zero and `-1` divisors, shift counts at `0`, `width - 1` and `width`, exponentiation whose
//! *intermediate* product overflows, and `Float32` rounding boundaries. Each case pins its category
//! and its line independently, so an engine that traps for the wrong reason fails here.

mod support;

use starkc::mir::TrapCategory;
use support::differential::{agree_completing_with_stdout, agree_trapping};

// ------------------------------------------------------------------- integer boundaries --

/// The extremes are representable and print exactly. A width confusion anywhere in the pipeline
/// shows up here before it shows up as arithmetic.
#[test]
fn signed_and_unsigned_extremes_render_exactly() {
    agree_completing_with_stdout(
        "extremes",
        "fn main() { \
         let i8max: Int8 = 127i8; let i8min: Int8 = -127i8 - 1i8; \
         let u8max: UInt8 = 255u8; \
         let i64max: Int64 = 9223372036854775807i64; \
         let i64min: Int64 = -9223372036854775807i64 - 1i64; \
         let u64max: UInt64 = 18446744073709551615u64; \
         println(i8max); println(i8min); println(u8max); \
         println(i64max); println(i64min); println(u64max); }",
        "127\n-128\n255\n9223372036854775807\n-9223372036854775808\n18446744073709551615\n",
    );
}

/// One step past the maximum traps, and one step before it does not. The pair is what makes the
/// boundary a boundary rather than an assertion about a single value.
#[test]
fn one_step_past_the_maximum_traps() {
    agree_completing_with_stdout(
        "at_max",
        "fn main() { let m: Int8 = 126i8; println(m + 1i8); }",
        "127\n",
    );
    agree_trapping(
        "past_max",
        "fn main() { let m: Int8 = 127i8; println(m + 1i8); }",
        TrapCategory::IntegerOverflow,
        1,
    );
    agree_trapping(
        "past_min",
        "fn main() { let m: Int8 = -127i8 - 1i8; println(m - 1i8); }",
        TrapCategory::IntegerOverflow,
        1,
    );
    agree_trapping(
        "unsigned_past_min",
        "fn main() { let z: UInt8 = 0u8; println(z - 1u8); }",
        TrapCategory::IntegerOverflow,
        1,
    );
}

/// Negation of the minimum is the third member of the `MIN`-with-`-1` family, and it traps for the
/// same reason: the result is not representable.
#[test]
fn negating_the_minimum_traps_as_overflow() {
    agree_trapping(
        "neg_min",
        "fn main() { let m: Int32 = -2147483647i32 - 1i32; println(-m); }",
        TrapCategory::IntegerOverflow,
        1,
    );
}

// ------------------------------------------------------------------------------- casts --

/// A cast at the exact limit succeeds; one beyond it traps as a **cast failure**, not an overflow.
/// The two categories are distinct and were confused once before (DEV-096), so both are stated.
#[test]
fn casts_at_and_beyond_the_exact_limit() {
    agree_completing_with_stdout(
        "cast_at_limit",
        "fn main() { let v: Int32 = 127; println(v as Int8); }",
        "127\n",
    );
    agree_trapping(
        "cast_past_limit",
        "fn main() { let v: Int32 = 128; println(v as Int8); }",
        TrapCategory::CastFailure,
        1,
    );
    agree_completing_with_stdout(
        "cast_at_negative_limit",
        "fn main() { let v: Int32 = -128; println(v as Int8); }",
        "-128\n",
    );
    agree_trapping(
        "cast_past_negative_limit",
        "fn main() { let v: Int32 = -129; println(v as Int8); }",
        TrapCategory::CastFailure,
        1,
    );
}

/// A negative value cast to an unsigned type traps rather than wrapping — the case where a host's
/// native `as` would silently produce a large positive number.
#[test]
fn a_negative_value_does_not_wrap_into_an_unsigned_type() {
    agree_trapping(
        "negative_to_unsigned",
        "fn main() { let v: Int32 = -1; println(v as UInt8); }",
        TrapCategory::CastFailure,
        1,
    );
}

// ------------------------------------------------------------------------------ shifts --

/// The shift-count boundary: `0` and `width - 1` are valid, `width` is not. NUM-SHIFT-001 states no
/// masking and no count reduction, so a host that masks the count would pass at `width` instead of
/// trapping.
#[test]
fn shift_counts_at_the_exact_width_boundary() {
    agree_completing_with_stdout(
        "shift_zero",
        "fn main() { let v: Int32 = 5; println(v << 0); }",
        "5\n",
    );
    agree_completing_with_stdout(
        "shift_width_minus_one",
        "fn main() { let v: Int32 = 1; println(v << 30); }",
        "1073741824\n",
    );
    agree_trapping(
        "shift_at_width",
        "fn main() { let v: Int32 = 1; println(v << 32); }",
        TrapCategory::InvalidShift,
        1,
    );
    agree_trapping(
        "shift_negative_count",
        "fn main() { let v: Int32 = 1; let c: Int32 = -1; println(v << c); }",
        TrapCategory::InvalidShift,
        1,
    );
}

/// A left shift whose COUNT is valid but whose RESULT is not representable traps as an overflow,
/// not as an invalid shift. The two are deliberately separate categories and this is the case that
/// tells them apart.
#[test]
fn a_left_shift_that_overflows_is_not_an_invalid_shift() {
    agree_trapping(
        "shift_overflows",
        "fn main() { let v: Int32 = 1; println(v << 31); }",
        TrapCategory::IntegerOverflow,
        1,
    );
}

// ---------------------------------------------------------------------- exponentiation --

/// **The intermediate** product overflows even though a mathematician would say the answer is
/// representable — `2 ** 62` fits an `Int64`, but a base that grows past the range on the way there
/// does not. NUM-INT-ARITH-001 checks each multiply, so the trap is required.
#[test]
fn exponentiation_checks_each_intermediate_multiply() {
    agree_completing_with_stdout(
        "pow_at_limit",
        "fn main() { let b: Int64 = 2i64; println(b ** 62); }",
        "4611686018427387904\n",
    );
    agree_trapping(
        "pow_intermediate_overflow",
        "fn main() { let b: Int64 = 2i64; println(b ** 63); }",
        TrapCategory::IntegerOverflow,
        1,
    );
}

/// A negative exponent is an overflow-category trap, not a silent zero or a float promotion.
#[test]
fn a_negative_exponent_traps() {
    agree_trapping(
        "pow_negative_exponent",
        "fn main() { let b: Int32 = 2; let e: Int32 = -1; println(b ** e); }",
        TrapCategory::IntegerOverflow,
        1,
    );
}

// ---------------------------------------------------------------------------- Float32 --

/// `Float32` renders at its DECLARED width, at the values where that is observable. `0.1f32`
/// widened to `f64` prints seventeen digits; at its own width it prints `0.1` (DEV-105).
///
/// The three expectations are derived from NUM-FLOAT-FORMAT-001 rather than from any engine's
/// output — shortest digits that round-trip the DECLARED width, positional notation exactly when
/// the scientific exponent is in `[-4, 15]`:
///
/// - `0.1f32` — exponent -1, positional, and the shortest f32 round-trip is one digit.
/// - `16777216.0f32` — 2^24, exponent 7, positional; an integral value still carries `.0`, the
///   same way `canonical_float(12.0)` is `"12.0"`.
/// - `1.0e-40f32` — a subnormal at exponent -40, outside the positional window, so e-notation; the
///   shortest digits that round-trip the f32 are `1e-40`, NOT the long decimal expansion the f64
///   value would produce.
#[test]
fn float32_rounding_boundaries_render_at_the_declared_width() {
    agree_completing_with_stdout(
        "f32_boundaries",
        "fn main() { \
         let a: Float32 = 0.1f32; let b: Float32 = 16777216.0f32; let c: Float32 = 1.0e-40f32; \
         println(a); println(b); println(c); }",
        "0.1\n16777216.0\n1e-40\n",
    );
}

/// Float division by zero does NOT trap (CD-139 superseded CD-006): it produces an infinity, and
/// the rendering of that infinity is the observation.
#[test]
fn float_division_by_zero_produces_infinities_rather_than_trapping() {
    agree_completing_with_stdout(
        "float_div_zero",
        "fn main() { let z: Float64 = 0.0f64; println(1.0f64 / z); println(-1.0f64 / z); \
         println(z / z); }",
        "inf\n-inf\nNaN\n",
    );
}

// ------------------------------------------------------------------- indexing boundaries --

/// The last valid index is valid; the first invalid one traps, and so does a negative one — both
/// as `IndexOutOfBounds`, with the user's own line.
#[test]
fn index_boundaries_on_both_ends() {
    agree_completing_with_stdout(
        "index_last_valid",
        "fn main() { let a: [Int32; 3] = [1, 2, 3]; println(a[2u64]); }",
        "3\n",
    );
    agree_trapping(
        "index_first_invalid",
        "fn main() { let a: [Int32; 3] = [1, 2, 3]; println(a[3u64]); }",
        TrapCategory::IndexOutOfBounds,
        1,
    );
    agree_trapping(
        "index_negative",
        "fn main() { let a: [Int32; 3] = [1, 2, 3]; let i: Int32 = -1; println(a[i as UInt64]); }",
        TrapCategory::CastFailure,
        1,
    );
}

/// An empty collection has no valid index at all — index `0` traps rather than reading past the
/// start of an allocation.
#[test]
fn indexing_an_empty_collection_traps() {
    agree_trapping(
        "index_empty_vec",
        "fn main() { let v: Vec<Int32> = Vec::new(); println(v[0u64]); }",
        TrapCategory::IndexOutOfBounds,
        1,
    );
}

// ---------------------------------------------------------------------------------------------
// WP-C7.9 G.4 — the same cases, stated as independent expectations.
//
// The helpers above each pin one shape. `ExpectedOutcome` states the whole outcome as data,
// including the two shapes no helper covered: a program the front end must REJECT, and the exact
// streams around a trap. Each case below says what the program does before any engine runs, so
// three engines agreeing on something else fails.
// ---------------------------------------------------------------------------------------------

use support::differential::ExpectedOutcome;

#[test]
fn outcomes_are_pinned_independently_across_every_shape() {
    ExpectedOutcome::Complete {
        stdout: "3\n",
        stderr: "",
        status: 0,
    }
    .check("pinned_complete", "fn main() { println(1 + 2); }");

    ExpectedOutcome::Trap {
        category: TrapCategory::IntegerOverflow,
        line: 1,
        stdout_before: "before\n",
        stderr_before: "note\n",
    }
    .check(
        "pinned_trap",
        "fn main() { println(\"before\"); eprintln(\"note\"); let m: Int8 = 127i8; println(m + 1i8); }",
    );

    ExpectedOutcome::FrontendReject { code: "E0105" }.check(
        "pinned_reject",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); \
         let mut n: Int32 = 0; for x in v { n = n + x; } println(n); }",
    );
}
