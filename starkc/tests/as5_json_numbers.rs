//! AS5-b — DEV-185: the JSON layer's `Number(f64)` does not preserve the value the input denotes.
//!
//! DEV-182's governing lesson, stated in `GATE-C8-CLOSURE.md` §4 and this packet's dependencies:
//! **parsing successfully is insufficient; the returned value has to be the value the input
//! denotes.** A model that decodes every JSON number to `f64` loses that before any consumer sees
//! it, and `as_i64()`'s `n as i64` cast then converts silently rather than refusing.
//!
//! Written to FAIL before the repair. The three cases are ordered by how much they cost:
//!
//! 1. an integer beyond `f64`'s exact range **changes value**, and on the LSP surface that value is
//!    a request identifier — correlation identity between a request and its response;
//! 2. a fractional number is silently truncated to an integer rather than refused;
//! 3. `JsonValue::Number` can hold NaN or infinity, whose textual forms are not JSON numbers, so
//!    the type can represent a document that cannot be serialized.

use starkc::lsp::protocol::{parse_json, JsonValue};

/// 2^53 + 1: the smallest positive integer `f64` cannot represent. It is a perfectly ordinary
/// JSON-RPC request id — 64-bit ids are what a client with a counter or a snowflake produces.
const BEYOND_F64: &str = "9007199254740993";

#[test]
fn a_request_id_beyond_f64_survives_a_round_trip() {
    let parsed = parse_json(&format!("{{\"id\":{BEYOND_F64}}}")).expect("valid JSON");
    let id = parsed.get("id").expect("id present");
    assert_eq!(
        id.to_string(),
        BEYOND_F64,
        "the id changed value between parsing and re-emission; a JSON-RPC response would \
         correlate to a request that was never sent"
    );
}

#[test]
fn a_fractional_number_is_not_an_integer() {
    let parsed = parse_json("{\"id\":1.5}").expect("valid JSON");
    let id = parsed.get("id").expect("id present");
    assert_eq!(
        id.as_i64(),
        None,
        "1.5 was accepted as the integer {:?}; a request id must be refused, not truncated",
        id.as_i64()
    );
}

#[test]
fn a_number_is_exactly_what_the_input_said() {
    // Each of these is valid RFC 8259 and each must round-trip unchanged. `-0`, the exponent forms
    // and the trailing-zero form all denote values a lossy decode would normalise away.
    for text in ["0", "-0", "1.5", "1e3", "1E-3", "1.50", BEYOND_F64] {
        let parsed = parse_json(text).unwrap_or_else(|| panic!("{text} is valid JSON"));
        assert_eq!(
            parsed.to_string(),
            text,
            "{text} did not survive a parse/emit round trip"
        );
    }
}

#[test]
fn a_number_too_large_for_f64_still_parses() {
    // RFC 8259 §6 sets no range limit; a parser may not refuse `1e400` because a binary64 cannot
    // hold it. What a CONSUMER does about it is the consumer's decision, made explicitly.
    let parsed = parse_json("1e400").expect("1e400 is valid JSON");
    assert_eq!(parsed.to_string(), "1e400");
    assert_eq!(
        parsed.as_f64(),
        None,
        "as_f64 must refuse a value it cannot represent rather than return infinity"
    );
}

#[test]
fn malformed_numbers_are_refused() {
    // ASCII digits only, and the grammar literally: no leading zero, no bare `+`, no naked
    // decimal point on either side.
    for text in ["01", "+1", "1.", ".1", "1e", "1e+", "--1", "0x10", "１"] {
        assert!(
            parse_json(text).is_none(),
            "{text} is not a JSON number and must be refused"
        );
    }
}

#[test]
fn the_value_model_cannot_hold_a_non_json_number() {
    // NaN and the infinities have no JSON textual form. A model that can hold them can hold a
    // document that cannot be serialized, and the failure surfaces at emit time in whatever code
    // path happened to build it.
    assert!(
        JsonValue::number_from_f64(f64::NAN).is_none(),
        "NaN is not a JSON number"
    );
    assert!(
        JsonValue::number_from_f64(f64::INFINITY).is_none(),
        "infinity is not a JSON number"
    );
    assert!(
        JsonValue::number_from_f64(f64::NEG_INFINITY).is_none(),
        "negative infinity is not a JSON number"
    );
    let finite = JsonValue::number_from_f64(1.5).expect("1.5 is a JSON number");
    assert_eq!(finite.to_string(), "1.5");
}
