//! Specification-derived tests for `interp::canonical_float` — the SHARED Float64 formatter
//! (HIR oracle and MIR runtime both call it, by design: one algorithm, no drift).
//!
//! Because it is shared, the HIR/MIR differential suite is structurally BLIND to defects in
//! it — both engines would print the same wrong text (review caveat, CD-029). These tests are
//! the compensating control: golden cases derived from the frozen numeric contract's rules
//! (shortest round-trip digits; positional notation exactly when the scientific exponent is
//! in [-4, 15], else e-notation; fixed NaN/inf/-0.0 spellings), plus a round-trip property.

use starkc::interp::canonical_float;

#[test]
fn special_values_have_fixed_spellings() {
    assert_eq!(canonical_float(f64::NAN), "NaN");
    assert_eq!(canonical_float(f64::INFINITY), "inf");
    assert_eq!(canonical_float(f64::NEG_INFINITY), "-inf");
    assert_eq!(canonical_float(0.0), "0.0");
    assert_eq!(canonical_float(-0.0), "-0.0");
}

#[test]
fn integral_and_simple_fractions_are_positional() {
    assert_eq!(canonical_float(12.0), "12.0");
    assert_eq!(canonical_float(-3.5), "-3.5");
    assert_eq!(canonical_float(0.1), "0.1");
    assert_eq!(canonical_float(0.3), "0.3");
    assert_eq!(canonical_float(1.0), "1.0");
    assert_eq!(canonical_float(-1.0), "-1.0");
    assert_eq!(canonical_float(100.25), "100.25");
}

#[test]
fn exponent_boundaries_switch_notation_exactly_at_the_contract_limits() {
    // Scientific exponent 15 is the last positional magnitude; 16 switches to e-notation.
    assert_eq!(canonical_float(1e15), "1000000000000000.0");
    assert_eq!(canonical_float(1e16), "1e16");
    assert_eq!(canonical_float(1.5e16), "1.5e16");
    // Scientific exponent -4 is the last positional small magnitude; -5 switches.
    assert_eq!(canonical_float(0.0001), "0.0001");
    assert_eq!(canonical_float(0.00001), "1e-5");
}

#[test]
fn shortest_round_trip_digits() {
    assert_eq!(canonical_float(1.0 / 3.0), "0.3333333333333333");
    assert_eq!(canonical_float(0.1 + 0.2), "0.30000000000000004");
}

#[test]
fn extreme_finite_and_subnormal_values() {
    assert_eq!(canonical_float(f64::MAX), "1.7976931348623157e308");
    assert_eq!(canonical_float(f64::MIN), "-1.7976931348623157e308");
    assert_eq!(
        canonical_float(f64::MIN_POSITIVE),
        "2.2250738585072014e-308"
    );
    // Smallest positive subnormal.
    assert_eq!(canonical_float(5e-324), "5e-324");
}

#[test]
fn rendering_round_trips_for_finite_values() {
    let cases: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        12.0,
        -3.5,
        0.1,
        0.3,
        1.0 / 3.0,
        0.1 + 0.2,
        1e15,
        1e16,
        1.5e16,
        0.0001,
        0.00001,
        123456789.123456,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        5e-324,
        2.5e-10,
        9.007199254740993e15, // 2^53 + 1 territory: shortest digits must still round-trip
    ];
    for &value in cases {
        let rendered = canonical_float(value);
        let parsed: f64 = rendered
            .parse()
            .unwrap_or_else(|_| panic!("rendered text {rendered:?} must parse as f64"));
        assert_eq!(
            parsed.to_bits(),
            value.to_bits(),
            "round-trip failed: {value:?} rendered as {rendered:?} parsed back as {parsed:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// WP-C7.9 G.5 — mutation coverage: proof that the tests above can FAIL.
//
// The compensating-control argument has a hole in it until this exists. `canonical_float` is shared
// by the oracle and the MIR interpreter deliberately, so the differential cannot see a defect in
// it — these golden cases are the only thing that can. But golden cases only compensate if they are
// *sensitive*: a table of expectations that a wrong algorithm would also satisfy is not a control,
// it is decoration.
//
// So each mutant below is a deliberately wrong renderer of the kind that could plausibly be written
// — Rust's own `Display`, a fixed-precision format, dropped signed zero, the notation switch off by
// one — and each is required to FAIL the same expectation table the real implementation passes. A
// mutant that survives means the table has a blind spot at exactly that property.
// ---------------------------------------------------------------------------------------------

/// The specification-derived expectations, in one place so the real implementation and every mutant
/// are judged against the same table.
const EXPECTATIONS: &[(f64, &str)] = &[
    (f64::INFINITY, "inf"),
    (f64::NEG_INFINITY, "-inf"),
    (0.0, "0.0"),
    (-0.0, "-0.0"),
    (12.0, "12.0"),
    (-3.5, "-3.5"),
    (0.1, "0.1"),
    (1.0 / 3.0, "0.3333333333333333"),
    (1e15, "1000000000000000.0"),
    (1e16, "1e16"),
    (0.0001, "0.0001"),
    (0.00001, "1e-5"),
    (5e-324, "5e-324"),
];

fn satisfies(render: impl Fn(f64) -> String) -> bool {
    // NaN is checked separately: it is the one case where `==` on the input is useless.
    if render(f64::NAN) != "NaN" {
        return false;
    }
    EXPECTATIONS
        .iter()
        .all(|(value, expected)| render(*value) == *expected)
}

/// The real implementation satisfies the table — the baseline every mutant is measured against.
#[test]
fn the_shared_formatter_satisfies_the_table() {
    assert!(
        satisfies(canonical_float),
        "canonical_float no longer satisfies its own specification table"
    );
}

/// Rust's `Display`. The obvious wrong answer, and the one the codebase would drift toward if the
/// shared formatter were ever "simplified": it renders `12` for `12.0`, `inf` for infinity but
/// `NaN` for NaN, and `0.00001` positionally where the contract switches to e-notation.
#[test]
fn rust_display_does_not_satisfy_the_table() {
    assert!(
        !satisfies(|v: f64| format!("{v}")),
        "Rust's Display satisfies the table — the table cannot be distinguishing STARK's \
         formatting rules from Rust's"
    );
}

/// Fixed precision. Round-trips nothing and loses the shortest-digits property.
#[test]
fn fixed_precision_does_not_satisfy_the_table() {
    assert!(
        !satisfies(|v: f64| format!("{v:.6}")),
        "a fixed-precision renderer satisfies the table — the shortest-round-trip property is \
         not being tested"
    );
}

/// Signed zero dropped. A single case in the table catches it, and this proves that case is load
/// bearing rather than incidental.
#[test]
fn dropping_signed_zero_does_not_satisfy_the_table() {
    assert!(
        !satisfies(|v: f64| {
            if v == 0.0 {
                "0.0".to_string()
            } else {
                canonical_float(v)
            }
        }),
        "dropping the sign of -0.0 satisfies the table — signed zero is not being tested"
    );
}

/// The notation switch, off by one at the upper boundary. This is the mutation most likely to occur
/// for real, because the boundary is a `>=` versus `>` decision in the renderer.
#[test]
fn an_off_by_one_notation_boundary_does_not_satisfy_the_table() {
    assert!(
        !satisfies(|v: f64| {
            if v == 1e15 {
                "1e15".to_string()
            } else {
                canonical_float(v)
            }
        }),
        "switching to e-notation one magnitude early satisfies the table — the positional/\
         scientific boundary is not being tested at its exact limit"
    );
}

/// Subnormals mishandled: flushed to zero, as a naive implementation might.
#[test]
fn flushing_subnormals_to_zero_does_not_satisfy_the_table() {
    assert!(
        !satisfies(|v: f64| {
            if v != 0.0 && v.abs() < f64::MIN_POSITIVE {
                "0.0".to_string()
            } else {
                canonical_float(v)
            }
        }),
        "flushing subnormals to zero satisfies the table — subnormal rendering is not being tested"
    );
}
