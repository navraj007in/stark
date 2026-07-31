//! WP-C7.9 Packet A — integer division and remainder trap identity, adversarially.
//!
//! Two findings are closed here, and they share one root cause.
//!
//! - **F2 — `MIN % -1` did not trap at all below the oracle.** MIR and the generated-Rust backend
//!   both evaluate on an `i128` carrier and then range-filter the result. The mathematical
//!   remainder of `MIN % -1` is `0`, which is *in range*, so no trap fired and the program
//!   **completed with a value** while the HIR oracle trapped. A completion where the specification
//!   requires a trap is the most severe divergence class this project defines.
//! - **F1 — `MIN / -1` trapped with the wrong identity.** It trapped in all three engines, but
//!   lowering attaches one static category per operator (`Div`/`Rem` → `DivideByZero`), so MIR and
//!   native reported a *division by zero* for an operation whose divisor was `-1`.
//!
//! NUM-INT-DIV-001 is explicit that both pairs trap "because the intermediate quotient is not
//! representable" — an overflow. MIR amendment A13 therefore lets the checked evaluation override
//! the terminator's default category for this cause, exactly as a bad shift count already did.
//!
//! **Why these cases needed writing by hand.** Every case here is one an ordinary agreement test
//! could not have caught: for F2 two of the three engines agreed *with each other* on the wrong
//! answer, and for F1 no maintained case exercised `MIN / -1` at all. Each expectation below is
//! pinned against the specification independently — `agree_trapping` states the category and the
//! line, `agree_completing_with_stdout` states the bytes — so unanimity on a wrong answer fails.

mod support;

use starkc::mir::TrapCategory;

/// The four widths, with a source prelude that constructs each type's minimum.
///
/// The minimum is not directly writable as a literal at any width (the magnitude exceeds the
/// positive range, so `-128i8` lexes as a negation of an out-of-range `128i8`), which is exactly
/// why `MIN / -1` had no maintained coverage. Each case builds it as `-(MAX) - 1`, which is
/// in-range at every step.
struct Width {
    /// The STARK type name.
    ty: &'static str,
    /// The literal suffix.
    suffix: &'static str,
    /// `MAX` negated — the largest-magnitude negative literal that is writable.
    neg_max: &'static str,
    /// `MIN / -2`, as it must print.
    half: &'static str,
    /// `MIN`, as it must print.
    min: &'static str,
}

const WIDTHS: &[Width] = &[
    Width {
        ty: "Int8",
        suffix: "i8",
        neg_max: "-127",
        half: "64",
        min: "-128",
    },
    Width {
        ty: "Int16",
        suffix: "i16",
        neg_max: "-32767",
        half: "16384",
        min: "-32768",
    },
    Width {
        ty: "Int32",
        suffix: "i32",
        neg_max: "-2147483647",
        half: "1073741824",
        min: "-2147483648",
    },
    Width {
        ty: "Int64",
        suffix: "i64",
        neg_max: "-9223372036854775807",
        half: "4611686018427387904",
        min: "-9223372036854775808",
    },
];

/// `let min: T = <-MAX> - 1;` — the minimum, built without an unwritable literal.
fn min_prelude(w: &Width) -> String {
    format!(
        "let base: {ty} = {neg_max}{sfx}; let min: {ty} = base - 1{sfx}; ",
        ty = w.ty,
        neg_max = w.neg_max,
        sfx = w.suffix
    )
}

// ---------------------------------------------------------------- F1 / F2: the trap identity --

/// `MIN / -1` traps `IntegerOverflow` — **not** `DivideByZero`, which is what MIR and native
/// reported before Packet A. The category is stated here, so the pre-fix behaviour fails this test
/// even though all three engines agreed on it.
#[test]
fn signed_min_div_negative_one_traps_as_overflow() {
    for w in WIDTHS {
        let source = format!(
            "fn main() {{ {prelude}println(min / -1{sfx}); }}",
            prelude = min_prelude(w),
            sfx = w.suffix
        );
        support::differential::agree_trapping(
            &format!("min_div_neg1_{}", w.ty),
            &source,
            TrapCategory::IntegerOverflow,
            1,
        );
    }
}

/// `MIN % -1` traps `IntegerOverflow`. Before Packet A this **completed and printed `0`** in MIR
/// and native while the oracle trapped: the mathematical remainder is representable, so the
/// range filter never saw a failure.
#[test]
fn signed_min_rem_negative_one_traps_as_overflow() {
    for w in WIDTHS {
        let source = format!(
            "fn main() {{ {prelude}println(min % -1{sfx}); }}",
            prelude = min_prelude(w),
            sfx = w.suffix
        );
        support::differential::agree_trapping(
            &format!("min_rem_neg1_{}", w.ty),
            &source,
            TrapCategory::IntegerOverflow,
            1,
        );
    }
}

// ------------------------------------------------------- the neighbours that must NOT change --

/// `MIN / 1` and `MIN % 1` are ordinary operations. A fix that trapped on "the dividend is the
/// minimum" rather than on the pair would break these.
#[test]
fn signed_min_divided_by_one_is_unaffected() {
    for w in WIDTHS {
        let source = format!(
            "fn main() {{ {prelude}println(min / 1{sfx}); println(min % 1{sfx}); }}",
            prelude = min_prelude(w),
            sfx = w.suffix
        );
        support::differential::agree_completing_with_stdout(
            &format!("min_div_one_{}", w.ty),
            &source,
            &format!("{}\n0\n", w.min),
        );
    }
}

/// `MIN / -2` and `MIN % -2` complete. A fix that trapped on "the divisor is negative" would break
/// these, and the quotient is the one value a width-confusion bug would get wrong.
#[test]
fn signed_min_divided_by_negative_two_is_unaffected() {
    for w in WIDTHS {
        let source = format!(
            "fn main() {{ {prelude}println(min / -2{sfx}); println(min % -2{sfx}); }}",
            prelude = min_prelude(w),
            sfx = w.suffix
        );
        support::differential::agree_completing_with_stdout(
            &format!("min_div_neg_two_{}", w.ty),
            &source,
            &format!("{}\n0\n", w.half),
        );
    }
}

/// A non-minimum dividend with divisor `-1` completes: `7 / -1` is `-7`, not a trap. The guard is
/// on the *pair*, not on either operand alone.
#[test]
fn ordinary_dividend_by_negative_one_is_unaffected() {
    for w in WIDTHS {
        let source = format!(
            "fn main() {{ println(7{sfx} / -1{sfx}); println(7{sfx} % -1{sfx}); }}",
            sfx = w.suffix
        );
        support::differential::agree_completing_with_stdout(
            &format!("seven_div_neg1_{}", w.ty),
            &source,
            "-7\n0\n",
        );
    }
}

// ------------------------------------------------------------- the other cause: zero divisors --

/// A zero divisor keeps `DivideByZero`. This is the half of the terminator's contract Packet A must
/// *not* disturb: the override applies to one cause, and the default still covers the other.
#[test]
fn zero_divisor_still_traps_as_divide_by_zero() {
    for w in WIDTHS {
        let div = format!(
            "fn main() {{ let z: {ty} = 0{sfx}; println(5{sfx} / z); }}",
            ty = w.ty,
            sfx = w.suffix
        );
        support::differential::agree_trapping(
            &format!("div_zero_{}", w.ty),
            &div,
            TrapCategory::DivideByZero,
            1,
        );

        let rem = format!(
            "fn main() {{ let z: {ty} = 0{sfx}; println(5{sfx} % z); }}",
            ty = w.ty,
            sfx = w.suffix
        );
        support::differential::agree_trapping(
            &format!("rem_zero_{}", w.ty),
            &rem,
            TrapCategory::DivideByZero,
            1,
        );
    }
}

/// Unsigned types have no negative minimum, so neither the pair nor the override can arise; a zero
/// divisor is their only division failure. Included because the guard reads the destination type's
/// signedness, and a guard that read the `i128` carrier instead would misfire here.
#[test]
fn unsigned_division_is_untouched() {
    support::differential::agree_completing_with_stdout(
        "unsigned_div",
        "fn main() { let a: UInt8 = 200u8; println(a / 3u8); println(a % 3u8); }",
        "66\n2\n",
    );
    support::differential::agree_trapping(
        "unsigned_div_zero",
        "fn main() { let z: UInt8 = 0u8; println(200u8 / z); }",
        TrapCategory::DivideByZero,
        1,
    );
}

// --------------------------------------------------------------------- compound assignment --

/// `/=` and `%=` lower through the same checked terminator and must classify identically. A fix
/// applied only to the binary-expression path would leave these reporting `DivideByZero`.
#[test]
fn compound_assignment_traps_as_overflow() {
    let div = format!(
        "fn main() {{ {prelude}let mut acc: Int32 = min; acc /= -1i32; println(acc); }}",
        prelude = min_prelude(&WIDTHS[2])
    );
    support::differential::agree_trapping(
        "compound_div_assign",
        &div,
        TrapCategory::IntegerOverflow,
        1,
    );

    let rem = format!(
        "fn main() {{ {prelude}let mut acc: Int32 = min; acc %= -1i32; println(acc); }}",
        prelude = min_prelude(&WIDTHS[2])
    );
    support::differential::agree_trapping(
        "compound_rem_assign",
        &rem,
        TrapCategory::IntegerOverflow,
        1,
    );
}

// ------------------------------------------------------------------------------ provenance --

/// The trap is blamed on the user's operation, on the right line, in a multi-line program — not on
/// the function, the entry point, or the last line of the file. Provenance is compared as a field,
/// so a backend that reported the correct category at the wrong place fails here.
#[test]
fn overflow_trap_is_blamed_on_the_operating_line() {
    let source = "fn divide(a: Int32, b: Int32) -> Int32 {\n    \
                  let q: Int32 = a / b;\n    \
                  q\n\
                  }\n\
                  fn main() {\n    \
                  let base: Int32 = -2147483647i32;\n    \
                  let min: Int32 = base - 1i32;\n    \
                  println(divide(min, -1i32));\n\
                  }\n";
    support::differential::agree_trapping("blame_line", source, TrapCategory::IntegerOverflow, 2);
}

// ----------------------------------------------------- exhaustive Int8 × Int8 property evidence --

/// Every non-trapping `Int8 ÷ Int8` pair, checked against an independent host oracle.
///
/// **Why batched.** There are 65 536 ordered pairs. Compiling one native binary per pair is not a
/// test, it is a build farm; instead one STARK program folds every quotient and remainder into a
/// rolling modular accumulator and prints it once, so the whole space costs a single native
/// compilation. The accumulator is order-sensitive and modulus-mixed, so a single wrong quotient
/// anywhere in the space changes the printed digit string.
///
/// **Why it is independent evidence.** The expected accumulator is computed here in Rust from
/// `i8`'s own semantics — not read back from any STARK engine. Three engines agreeing on a wrong
/// division table fails this test, which is the property ordinary agreement testing cannot supply.
///
/// The two trapping shapes are excluded by the same conditions the specification states, and are
/// covered exhaustively by [`every_int8_trapping_pair_traps`] instead.
#[test]
fn every_non_trapping_int8_pair_matches_an_independent_oracle() {
    const MODULUS: i64 = 1_000_000_007;
    const MIXER: i64 = 31;

    let mut expected: i64 = 0;
    for a in i8::MIN..=i8::MAX {
        for b in i8::MIN..=i8::MAX {
            if b == 0 {
                continue;
            }
            if a == i8::MIN && b == -1 {
                continue;
            }
            let quotient = i64::from(a.wrapping_div(b));
            let remainder = i64::from(a.wrapping_rem(b));
            // `%`, not `rem_euclid`: STARK's remainder truncates toward zero and takes the sign of
            // the dividend (NUM-INT-DIV-001), which is Rust's `%` and is NOT `rem_euclid`. The
            // accumulator therefore ranges over negatives too, and an oracle that normalised them
            // away would disagree with a correct compiler.
            expected = (expected * MIXER + quotient) % MODULUS;
            expected = (expected * MIXER + remainder) % MODULUS;
        }
    }

    // `wrapping_div` is exact here: the only pair where it differs from the true quotient is
    // `MIN / -1`, which the loop above skips. Stated rather than assumed, because a silent wrap
    // would make the oracle agree with a broken compiler.
    assert_eq!(
        i8::MIN.wrapping_div(-1),
        i8::MIN,
        "the excluded pair is the only wrapping case"
    );

    let source = format!(
        "fn main() {{\n    \
         let mut acc: Int64 = 0i64;\n    \
         let mut a: Int32 = -128i32;\n    \
         while a <= 127i32 {{\n        \
         let mut b: Int32 = -128i32;\n        \
         while b <= 127i32 {{\n            \
         if b != 0i32 {{\n                \
         let skip: Bool = a == -128i32;\n                \
         if skip {{\n                    \
         if b != -1i32 {{\n                        \
         acc = fold(acc, a, b);\n                    \
         }}\n                \
         }} else {{\n                    \
         acc = fold(acc, a, b);\n                \
         }}\n            \
         }}\n            \
         b = b + 1i32;\n        \
         }}\n        \
         a = a + 1i32;\n    \
         }}\n    \
         println(acc);\n\
         }}\n\
         fn fold(acc: Int64, a: Int32, b: Int32) -> Int64 {{\n    \
         let x: Int8 = a as Int8;\n    \
         let y: Int8 = b as Int8;\n    \
         let q: Int64 = (x / y) as Int64;\n    \
         let r: Int64 = (x % y) as Int64;\n    \
         let first: Int64 = (acc * {MIXER}i64 + q) % {MODULUS}i64;\n    \
         (first * {MIXER}i64 + r) % {MODULUS}i64\n\
         }}\n"
    );

    support::differential::agree_completing_with_stdout(
        "int8_division_table",
        &source,
        &format!("{expected}\n"),
    );
}

/// Every trapping `Int8` pair — all 256 zero-divisor dividends for `/` and for `%`, plus the two
/// `MIN op -1` cases — trapping with the right category.
///
/// **Engine scope, stated rather than implied.** A trap ends the program, so these cannot be
/// batched the way the completing pairs were: 514 cases means 514 programs. They run on the two
/// interpreters, where a case costs a lowering rather than a native build. The native engine is not
/// skipped for this space — it is covered by the four-engine cases above, which exercise both
/// categories at all four widths. This is a deliberate split of an infeasible cross-product, not an
/// engine quietly dropping out of a comparison.
#[test]
fn every_int8_trapping_pair_traps() {
    for a in i8::MIN..=i8::MAX {
        // `-128i8` is not writable — it lexes as a negation of an out-of-range `128i8`, which is
        // the same fact the `Width` table above records. The minimum dividend is therefore built
        // the same way every other case builds it.
        let dividend = if a == i8::MIN {
            "(-127i8 - 1i8)".to_string()
        } else {
            format!("{a}i8")
        };
        let div = format!("fn main() {{ let z: Int8 = 0i8; println({dividend} / z); }}");
        support::differential::interpreters_agree_trapping(
            &format!("int8_div_zero_{a}"),
            &div,
            TrapCategory::DivideByZero,
            1,
        );

        let rem = format!("fn main() {{ let z: Int8 = 0i8; println({dividend} % z); }}");
        support::differential::interpreters_agree_trapping(
            &format!("int8_rem_zero_{a}"),
            &rem,
            TrapCategory::DivideByZero,
            1,
        );
    }

    let prelude = min_prelude(&WIDTHS[0]);
    support::differential::interpreters_agree_trapping(
        "int8_min_div_neg1",
        &format!("fn main() {{ {prelude}println(min / -1i8); }}"),
        TrapCategory::IntegerOverflow,
        1,
    );
    support::differential::interpreters_agree_trapping(
        "int8_min_rem_neg1",
        &format!("fn main() {{ {prelude}println(min % -1i8); }}"),
        TrapCategory::IntegerOverflow,
        1,
    );
}
