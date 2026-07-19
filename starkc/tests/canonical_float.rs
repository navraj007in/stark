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
