//! LSP protocol message types and minimal JSON support (no external dependencies).

use std::collections::HashMap;

/// Minimal JSON value type (no external serde_json dependency).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(n) => Some(*n as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(arr) => Some(arr),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(obj) => obj.get(key),
            _ => None,
        }
    }
}

impl std::fmt::Display for JsonValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{}", b),
            JsonValue::Number(n) => {
                if n.fract() == 0.0 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            JsonValue::String(s) => write!(f, "\"{}\"", escape_json_string(s)),
            JsonValue::Array(arr) => {
                let items = arr
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                write!(f, "[{}]", items)
            }
            JsonValue::Object(obj) => {
                let mut entries = obj.iter().collect::<Vec<_>>();
                entries.sort_by_key(|(left, _)| *left);
                let items = entries
                    .into_iter()
                    .map(|(k, v)| format!("\"{}\":{}", escape_json_string(k), v))
                    .collect::<Vec<_>>()
                    .join(",");
                write!(f, "{{{}}}", items)
            }
        }
    }
}

/// DEV-184: this escaped only `" \ \n \r \t`, so every other C0 control went onto the LSP wire
/// raw — invalid per RFC 8259 §7. C8's protocol validation compared verdicts rather than values,
/// which is why a lenient client reported success and the defect survived (`GATE-C8-CLOSURE.md`
/// §4). AS5-c replaces this with the shared authority; this is the repair.
fn escape_json_string(s: &str) -> String {
    let mut out = String::new();
    for character in s.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // DEV-184: every remaining C0 control. RFC 8259 §7 requires U+0000-U+001F to be
            // escaped; leaving them raw produced documents no conforming parser accepts.
            character if character <= '\u{1f}' => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    let nibble = (character as u32 >> shift) & 0xf;
                    out.push(char::from_digit(nibble, 16).expect("nibble is < 16"));
                }
            }
            character => out.push(character),
        }
    }
    out
}

/// Parse minimal JSON (enough for LSP messages)
pub fn parse_json(s: &str) -> Option<JsonValue> {
    let mut chars = s.trim().chars().peekable();
    parse_json_value(&mut chars)
}

fn parse_json_value(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<JsonValue> {
    skip_whitespace(chars);

    match chars.peek()? {
        '{' => parse_json_object(chars),
        '[' => parse_json_array(chars),
        '"' => parse_json_string(chars).map(JsonValue::String),
        't' | 'f' => parse_json_bool(chars),
        'n' => parse_json_null(chars),
        '-' | '0'..='9' => parse_json_number(chars),
        _ => None,
    }
}

fn parse_json_object(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<JsonValue> {
    chars.next()?; // consume '{'
    let mut obj = HashMap::new();

    skip_whitespace(chars);
    if chars.peek() == Some(&'}') {
        chars.next();
        return Some(JsonValue::Object(obj));
    }

    loop {
        skip_whitespace(chars);
        let key = parse_json_string(chars)?;
        skip_whitespace(chars);
        if chars.next() != Some(':') {
            return None;
        }
        skip_whitespace(chars);
        let value = parse_json_value(chars)?;
        obj.insert(key, value);

        skip_whitespace(chars);
        match chars.peek() {
            Some(',') => {
                chars.next();
            }
            Some('}') => {
                chars.next();
                break;
            }
            _ => return None,
        }
    }

    Some(JsonValue::Object(obj))
}

fn parse_json_array(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<JsonValue> {
    chars.next()?; // consume '['
    let mut arr = Vec::new();

    skip_whitespace(chars);
    if chars.peek() == Some(&']') {
        chars.next();
        return Some(JsonValue::Array(arr));
    }

    loop {
        skip_whitespace(chars);
        arr.push(parse_json_value(chars)?);

        skip_whitespace(chars);
        match chars.peek() {
            Some(',') => {
                chars.next();
            }
            Some(']') => {
                chars.next();
                break;
            }
            _ => return None,
        }
    }

    Some(JsonValue::Array(arr))
}

/// The four hex digits of a `\uXXXX` escape, with `\u` already consumed.
///
/// Returns `None` on a non-hex digit or on end of input, so a malformed escape fails the parse
/// instead of contributing nothing to a value that is then reported as successfully parsed.
fn read_hex4(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<u32> {
    let mut value = 0u32;
    for _ in 0..4 {
        value = value * 16 + chars.next()?.to_digit(16)?;
    }
    Some(value)
}

/// One `\u` escape, decoding a UTF-16 surrogate pair into the scalar it denotes (RFC 8259 §7).
///
/// **DEV-182.** This used to read four hex digits, call `char::from_u32`, and *discard the escape
/// when that failed* — which it always does for a surrogate half, since a surrogate is not a Rust
/// `char`. There was no pairing step and no error, so a valid escaped `U+1F600` parsed to the
/// **empty string** and an unpaired surrogate was accepted as the empty string. Both parsers
/// returned success; only the value was wrong, which is why no verdict-based comparison could see
/// it. An editor sending a non-BMP character in any string — a completion label, a file path, a
/// diagnostic message — lost it silently.
fn parse_unicode_escape(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<char> {
    const HIGH: std::ops::RangeInclusive<u32> = 0xD800..=0xDBFF;
    const LOW: std::ops::RangeInclusive<u32> = 0xDC00..=0xDFFF;

    let first = read_hex4(chars)?;
    if HIGH.contains(&first) {
        // A high surrogate is only meaningful paired with a low one, and the pair must be written
        // as a second `\u` escape. Anything else is invalid JSON, including a bare non-BMP
        // character or a second high surrogate.
        if chars.next()? != '\\' || chars.next()? != 'u' {
            return None;
        }
        let low = read_hex4(chars)?;
        if !LOW.contains(&low) {
            return None;
        }
        let combined = 0x10000 + ((first - 0xD800) << 10) + (low - 0xDC00);
        return char::from_u32(combined);
    }
    if LOW.contains(&first) {
        // A low surrogate with no high surrogate before it denotes nothing.
        return None;
    }
    char::from_u32(first)
}

fn parse_json_string(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<String> {
    if chars.next() != Some('"') {
        return None;
    }

    let mut result = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(result),
            '\\' => match chars.next()? {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                '/' => result.push('/'),
                'b' => result.push('\u{0008}'),
                'f' => result.push('\u{000C}'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                'u' => result.push(parse_unicode_escape(chars)?),
                _ => return None,
            },
            _ => result.push(c),
        }
    }
    None
}

fn parse_json_bool(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<JsonValue> {
    let mut word = String::new();
    while let Some(&c) = chars.peek() {
        if c.is_alphanumeric() {
            word.push(chars.next().unwrap());
        } else {
            break;
        }
    }

    match word.as_str() {
        "true" => Some(JsonValue::Bool(true)),
        "false" => Some(JsonValue::Bool(false)),
        _ => None,
    }
}

fn parse_json_null(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<JsonValue> {
    let mut word = String::new();
    for _ in 0..4 {
        if let Some(c) = chars.next() {
            word.push(c);
        }
    }

    if word == "null" {
        Some(JsonValue::Null)
    } else {
        None
    }
}

fn parse_json_number(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<JsonValue> {
    let mut num_str = String::new();

    if chars.peek() == Some(&'-') {
        num_str.push(chars.next().unwrap());
    }

    while let Some(&c) = chars.peek() {
        if c.is_numeric() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
            num_str.push(chars.next().unwrap());
        } else {
            break;
        }
    }

    num_str.parse::<f64>().ok().map(JsonValue::Number)
}

fn skip_whitespace(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
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
            ("zeta".to_string(), JsonValue::Number(2.0)),
            ("alpha".to_string(), JsonValue::Number(1.0)),
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
