//! LSP protocol message types and minimal JSON support (no external dependencies).

use std::collections::HashMap;

/// **AS5-c: the LSP transport no longer owns a JSON implementation.**
///
/// It had its own `JsonValue`, parser and escaper. The parser agreed with `package.rs`'s on 3 of 12
/// constructs — they were two grammars, not one that drifted — and it accepted trailing input after
/// a complete value, so a truncated or concatenated frame parsed as its first value and the
/// remainder vanished. The escaper left 29 C0 control characters raw on the wire (DEV-184), and the
/// value model decoded every number to `f64` (DEV-185).
///
/// The message types below are the protocol-specific model and stay here. Parsing, the value type
/// and escaping come from `crate::json`.
pub use crate::json::{JsonNumber, JsonValue};

/// Parse one complete JSON value.
///
/// AS5 CE9 decision: **trailing non-whitespace input after the value is rejected.** A JSON-RPC frame
/// contains exactly one JSON value; accepting `{"jsonrpc":"2.0",...} garbage` weakens framing and
/// hides malformed clients. C8's protocol baseline already expected this rejection, so the
/// implementation now matches the contract it was written against.
///
/// Returns `Option` for source compatibility with the call sites; `crate::json::parse` carries the
/// message and byte offset for callers that want them.
pub fn parse_json(s: &str) -> Option<JsonValue> {
    crate::json::parse(s).ok()
}

/// LSP message type.
#[derive(Debug, Clone)]
pub enum Message {
    Request(Request),
    Response(Response),
    Notification(Notification),
}

#[derive(Debug, Clone)]
pub struct Request {
    pub id: i64,
    pub method: String,
    pub params: JsonValue,
}

#[derive(Debug, Clone)]
pub struct Response {
    pub id: i64,
    pub result: Option<JsonValue>,
    pub error: Option<ResponseError>,
}

#[derive(Debug, Clone)]
pub struct ResponseError {
    pub code: i32,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct Notification {
    pub method: String,
    pub params: JsonValue,
}

/// Parse JSON into an LSP message
pub fn parse_message(content: &str) -> Result<Message, String> {
    let value = parse_json(content).ok_or("Invalid JSON")?;

    match (
        value.get("method").and_then(|m| m.as_str()),
        value.get("result"),
        value.get("error"),
        value.get("id"),
    ) {
        (Some(method), _, _, Some(id)) => {
            // Request
            let id = id.as_i64().ok_or("Invalid request id")?;
            let params = value
                .get("params")
                .cloned()
                .unwrap_or(JsonValue::Object(HashMap::new()));

            Ok(Message::Request(Request {
                id,
                method: method.to_string(),
                params,
            }))
        }
        (Some(method), _, _, None) => {
            // Notification
            let params = value
                .get("params")
                .cloned()
                .unwrap_or(JsonValue::Object(HashMap::new()));

            Ok(Message::Notification(Notification {
                method: method.to_string(),
                params,
            }))
        }
        (None, _, _, Some(id)) => {
            // Response
            let id = id.as_i64().ok_or("Invalid response id")?;
            let result = value.get("result").cloned();
            let error = value.get("error").map(|e| ResponseError {
                code: e.get("code").and_then(|c| c.as_i64()).unwrap_or(-1) as i32,
                message: e
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error")
                    .to_string(),
            });

            Ok(Message::Response(Response { id, result, error }))
        }
        _ => Err("Invalid message format".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_string() {
        let val = parse_json("\"hello\"").unwrap();
        assert_eq!(val.as_str(), Some("hello"));
    }

    #[test]
    fn test_parse_json_number() {
        let val = parse_json("42").unwrap();
        assert_eq!(val.as_i64(), Some(42));
    }

    #[test]
    fn test_parse_json_bool() {
        let val = parse_json("true").unwrap();
        assert_eq!(val.as_bool(), Some(true));
    }

    #[test]
    fn test_parse_json_null() {
        let val = parse_json("null").unwrap();
        assert_eq!(val, JsonValue::Null);
    }

    #[test]
    fn test_parse_json_object() {
        let val = parse_json(r#"{"key":"value","num":42}"#).unwrap();
        assert_eq!(val.get("key").and_then(|v| v.as_str()), Some("value"));
        assert_eq!(val.get("num").and_then(|v| v.as_i64()), Some(42));
    }

    #[test]
    fn object_encoding_is_deterministic_and_key_sorted() {
        let value = JsonValue::Object(HashMap::from([
            ("zeta".to_string(), JsonValue::number_from_i64(2)),
            ("alpha".to_string(), JsonValue::number_from_i64(1)),
        ]));
        assert_eq!(value.to_string(), r#"{"alpha":1,"zeta":2}"#);
        assert_eq!(value.to_string(), value.to_string());
    }

    #[test]
    fn test_parse_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        match parse_message(json) {
            Ok(Message::Request(req)) => {
                assert_eq!(req.id, 1);
                assert_eq!(req.method, "initialize");
            }
            _ => panic!("Failed to parse request"),
        }
    }

    /// DEV-182 — `\uXXXX` handling.
    ///
    /// Every case below FAILED before the repair. A surrogate pair produced the empty string, an
    /// unpaired surrogate was accepted as the empty string, and an invalid `\u` was silently
    /// dropped mid-string: the parser reported success and returned a value the input did not
    /// denote. Nothing rejected them, so no verdict-based check could see it.
    fn parsed_string(body: &str) -> Option<String> {
        match parse_json(&format!(
            "{{{}a{}:{}{}{}}}",
            QUOTE, QUOTE, QUOTE, body, QUOTE
        )) {
            Some(JsonValue::Object(o)) => match o.get("a") {
                Some(JsonValue::String(s)) => Some(s.clone()),
                _ => None,
            },
            _ => None,
        }
    }

    const QUOTE: char = '"';

    #[test]
    fn surrogate_pairs_decode_to_one_scalar() {
        let body = format!("{BS}ud83d{BS}ude00", BS = BACKSLASH);
        assert_eq!(
            parsed_string(&body).as_deref(),
            Some("\u{1F600}"),
            "a valid surrogate pair must decode to U+1F600, not the empty string"
        );
    }

    #[test]
    fn basic_multilingual_plane_escapes_still_decode() {
        let body = format!("{BS}u0041{BS}u00e9", BS = BACKSLASH);
        assert_eq!(parsed_string(&body).as_deref(), Some("A\u{e9}"));
    }

    #[test]
    fn an_escaped_nul_is_a_character_not_a_terminator() {
        let body = format!("x{BS}u0000y", BS = BACKSLASH);
        assert_eq!(parsed_string(&body).as_deref(), Some("x\u{0}y"));
    }

    #[test]
    fn a_lone_high_surrogate_is_rejected() {
        let body = format!("{BS}ud83d", BS = BACKSLASH);
        assert_eq!(
            parsed_string(&body),
            None,
            "an unpaired surrogate is invalid JSON and must be rejected, not swallowed"
        );
    }

    #[test]
    fn a_lone_low_surrogate_is_rejected() {
        let body = format!("{BS}ude00", BS = BACKSLASH);
        assert_eq!(parsed_string(&body), None);
    }

    #[test]
    fn a_high_surrogate_followed_by_a_non_surrogate_is_rejected() {
        let body = format!("{BS}ud83d{BS}u0041", BS = BACKSLASH);
        assert_eq!(parsed_string(&body), None);
    }

    #[test]
    fn a_malformed_hex_escape_is_rejected_rather_than_dropped() {
        let body = format!("{BS}u00zz", BS = BACKSLASH);
        assert_eq!(
            parsed_string(&body),
            None,
            "a bad escape must fail the parse, not vanish from the value"
        );
    }

    #[test]
    fn a_truncated_escape_at_end_of_input_is_rejected() {
        let body = format!("{BS}u00", BS = BACKSLASH);
        assert_eq!(parsed_string(&body), None);
    }

    const BACKSLASH: char = '\\';
}
