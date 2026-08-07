//! AS5-d — the RFC 8259 conformance corpus for the compiler's one JSON authority.
//!
//! Exit criterion 2 says "a standard JSON test corpus and project-specific malformed cases pass".
//! `AS0-MANIFEST-STRICTNESS-AUDIT.md` §5 makes the twelve-construct table the project-specific
//! minimum, *because it already distinguished the two parsers this replaced* — a corpus that both
//! old parsers passed would have proved nothing about the divergence being fixed.
//!
//! Two of the twelve are CE9 decisions rather than plain conformance, taken explicitly:
//!
//! - **trailing commas and leading-zero numbers in a manifest are rejected.** A manifest is a
//!   durable configuration contract; accepting non-JSON syntax is compatibility debt for no
//!   benefit. Every first-party manifest is already strict-clean (audit §5), so this narrows what
//!   third-party manifests may contain rather than requiring a repository migration.
//! - **trailing input after a complete JSON-RPC value is rejected.** A frame carries exactly one
//!   value; accepting `{...} garbage` weakens framing and hides malformed clients. C8's protocol
//!   baseline already expected the rejection.

use starkc::json::{self, JsonValue};

fn parses(text: &str) -> bool {
    json::parse(text).is_ok()
}

fn string_of(text: &str) -> String {
    match json::parse(text) {
        Ok(JsonValue::String(s)) => s,
        other => panic!("expected a string from {text:?}, got {other:?}"),
    }
}

/// The audit's table, read as a specification. Every row is a construct on which the two previous
/// parsers disagreed with RFC 8259, with each other, or both.
#[test]
fn the_twelve_construct_table_matches_rfc_8259() {
    // 1, 2 — trailing commas. `package.rs` accepted both.
    assert!(!parses(r#"{"a":1,}"#), "1: trailing comma in object");
    assert!(!parses("[1,]"), "2: trailing comma in array");

    // 3 — trailing input. `lsp/protocol.rs` accepted it.
    assert!(!parses("{} garbage"), "3: trailing input after a value");
    assert!(parses("  {}  \t\r\n"), "3: trailing WHITESPACE is fine");

    // 4 — raw control character in a string. Both accepted it.
    let raw = format!("\"a{}b\"", char::from(1u8));
    assert!(!parses(&raw), "4: raw U+0001 in a string");

    // 5 — a plain BMP escape. `package.rs` had no `'u'` arm at all.
    //
    // These are built from `BS` rather than written inline: a file containing the DECODED character
    // would assert nothing about escape handling, which is the mistake this test made once already.
    const BS: char = '\\';
    let esc = |body: &str| format!("\"{body}\"");
    assert_eq!(string_of(&esc(&format!("{BS}u0041"))), "A", "5: u0041");

    // 6 — a valid surrogate pair. DEV-182: the LSP parser produced the EMPTY STRING here, and
    // passed C8's protocol validation while doing it.
    assert_eq!(
        string_of(&esc(&format!("{BS}ud83d{BS}ude00"))),
        "\u{1F600}",
        "6: valid surrogate pair"
    );

    // 7 — an unpaired surrogate. Refused, not substituted.
    assert!(
        !parses(&esc(&format!("{BS}ud83d"))),
        "7: unpaired high surrogate"
    );
    assert!(
        !parses(&esc(&format!("{BS}udc00"))),
        "7: unpaired low surrogate"
    );
    assert!(
        !parses(&esc(&format!("{BS}ud83dA"))),
        "7: high surrogate followed by a non-surrogate"
    );

    // 8 — `\u0000` is a legal ESCAPE and denotes U+0000, even though the raw character is not.
    assert_eq!(
        string_of(&esc(&format!("{BS}u0000"))),
        "\0",
        "8: escaped NUL"
    );

    // 9 — duplicate keys: last wins. The one place RFC 8259 says SHOULD, kept deliberately.
    let value = json::parse(r#"{"a":1,"a":2}"#).expect("valid");
    assert_eq!(
        value.get("a").and_then(|v| v.as_i64()),
        Some(2),
        "9: last wins"
    );

    // 10 — leading-zero numbers. Both parsers accepted them.
    assert!(!parses("01"), "10: leading zero");
    assert!(!parses("-01"), "10: negative leading zero");
    assert!(parses("0"), "10: a bare zero is fine");

    // 11, 12 — exponents. `package.rs` rejected all of them.
    assert_eq!(
        json::parse("1e3").unwrap().as_f64(),
        Some(1000.0),
        "11: exponent"
    );
    assert_eq!(
        json::parse("1.5e-3").unwrap().as_f64(),
        Some(0.0015),
        "12: negative exponent"
    );
}

/// RFC 8259 §2: whitespace is space, tab, LF and CR — and nothing else.
///
/// Both previous parsers used `char::is_whitespace`, which also accepts U+00A0, U+2028, U+3000 and
/// the rest of Unicode's whitespace, so they skipped bytes the grammar does not permit anywhere.
#[test]
fn only_the_four_json_whitespace_characters_are_whitespace() {
    for ws in [" ", "\t", "\n", "\r", " \t\r\n "] {
        assert!(parses(&format!("{ws}1{ws}")), "{ws:?} is JSON whitespace");
    }
    for not_ws in [
        "\u{a0}", "\u{2028}", "\u{3000}", "\u{feff}", "\u{b}", "\u{c}",
    ] {
        assert!(
            !parses(&format!("{not_ws}1")),
            "{not_ws:?} is not JSON whitespace and must not be skipped"
        );
    }
}

/// Digits are ASCII `0`–`9`. `char::is_numeric` — which the old number scanner reached for —
/// admits Devanagari, fullwidth and other numeric characters into the scan.
#[test]
fn digits_are_ascii_only() {
    for text in ["１", "١", "1٢3", "٣"] {
        assert!(!parses(text), "{text:?} must not scan as a number");
    }
}

/// Structural refusals a hand-edited manifest actually produces.
#[test]
fn project_specific_malformed_cases_are_refused() {
    let cases = [
        ("", "empty input"),
        ("   ", "whitespace only"),
        ("{", "unterminated object"),
        ("[", "unterminated array"),
        (r#"{"a"}"#, "key with no value"),
        (r#"{"a":}"#, "missing value"),
        (r#"{a:1}"#, "unquoted key"),
        (r#"{'a':1}"#, "single-quoted key"),
        (r#""unterminated"#, "unterminated string"),
        (r#""bad \q escape""#, "unknown escape"),
        (r#""\u00""#, "short \\u escape"),
        (r#""\uZZZZ""#, "non-hex \\u escape"),
        ("tru", "truncated literal"),
        ("NaN", "NaN is not JSON"),
        ("Infinity", "Infinity is not JSON"),
        ("[1 2]", "missing comma"),
        ("{} {}", "two values"),
        ("// comment\n{}", "comments are not JSON"),
    ];
    for (text, what) in cases {
        assert!(!parses(text), "{what}: {text:?} must be refused");
    }
}

/// Nesting is bounded. RFC 8259 §9 permits a limit, and a recursive-descent parser without one
/// turns `[[[[[…` into a stack overflow — an abort rather than a diagnostic, on input that arrives
/// from a file or a socket.
#[test]
fn nesting_is_bounded_rather_than_overflowing_the_stack() {
    let deep = "[".repeat(json::MAX_DEPTH + 10) + &"]".repeat(json::MAX_DEPTH + 10);
    let error = json::parse(&deep).expect_err("must be refused, not overflow");
    assert!(
        error.message.contains("nesting"),
        "expected a nesting-depth refusal, got: {error}"
    );

    let ok = "[".repeat(json::MAX_DEPTH - 1) + &"]".repeat(json::MAX_DEPTH - 1);
    assert!(parses(&ok), "depth below the limit must still parse");
}

/// Every escape is emitted such that the result parses back to the same string.
#[test]
fn escaping_round_trips_through_the_parser() {
    let mut adversarial = String::from("quote\" backslash\\ solidus/ ");
    for byte in 0u8..0x20 {
        adversarial.push(char::from(byte));
    }
    adversarial.push_str(" 😀 é \u{2028}");

    let quoted = json::quote(&adversarial);
    assert!(
        !quoted.chars().any(|c| (c as u32) < 0x20),
        "the emitted string still contains a raw control character"
    );
    assert_eq!(
        string_of(&quoted),
        adversarial,
        "a value must survive quote → parse unchanged"
    );
}

/// A whole document round-trips, including nesting and every value kind.
#[test]
fn a_document_round_trips() {
    let text = r#"{"array":[1,"two",true,null,{"nested":-0.5}],"exp":1e3,"big":9007199254740993}"#;
    let parsed = json::parse(text).expect("valid");
    let reemitted = parsed.to_string();
    let reparsed = json::parse(&reemitted).expect("re-emitted text must parse");
    assert_eq!(parsed, reparsed, "a round trip must not change the value");
    assert!(
        reemitted.contains("9007199254740993"),
        "the large integer must survive re-emission: {reemitted}"
    );
}

/// The `Display` order is deterministic. `HashMap` iteration is not stable across runs, and the LSP
/// transport's previous serializer inherited that — two runs could emit the same document
/// differently, which defeats byte-comparison of any JSON evidence.
#[test]
fn object_serialization_is_deterministic() {
    let text = r#"{"zeta":1,"alpha":2,"mid":3,"beta":4}"#;
    let first = json::parse(text).unwrap().to_string();
    for _ in 0..16 {
        assert_eq!(
            json::parse(text).unwrap().to_string(),
            first,
            "object key order must not vary between runs"
        );
    }
    assert!(
        first.find("\"alpha\"") < first.find("\"beta\""),
        "keys should be emitted in sorted order: {first}"
    );
}
