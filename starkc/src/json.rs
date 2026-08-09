//! The compiler's **one** JSON authority: one grammar, one escaper (AS5).
//!
//! Before this there were two parsers and four escapers. The parsers agreed on 3 of 12 constructs
//! — they were not one grammar that drifted, they were two grammars — and three of the four
//! escapers emitted raw C0 control characters, which RFC 8259 §7 forbids, so they could produce
//! documents no conforming parser accepts. `AS0-MANIFEST-STRICTNESS-AUDIT.md` and
//! `AS5-OPENING-ANALYSIS.md` record both inventories.
//!
//! Everything here is RFC 8259. Where the previous parsers were lenient they are now strict, and
//! where they were strict about valid input they now accept it:
//!
//! | Construct | old `package.rs` | old `lsp/protocol.rs` | here |
//! | --- | --- | --- | --- |
//! | trailing comma | accepted | rejected | **rejected** |
//! | trailing input after a value | rejected | accepted | **rejected** |
//! | raw control character in a string | accepted | accepted | **rejected** |
//! | `A` | rejected | `A` | **`A`** |
//! | `😀` (valid pair) | rejected | empty string | **U+1F600** |
//! | unpaired surrogate | rejected | `""` | **rejected** |
//! | leading-zero number `01` | accepted | accepted | **rejected** |
//! | exponent `1e3` / `1.5e-3` | rejected | accepted | **accepted** |
//! | duplicate keys | last wins | last wins | **last wins** |
//!
//! Duplicate keys are the one place RFC 8259 leaves a choice ("the names within an object SHOULD be
//! unique"); last-wins is what both previous parsers did and what the audit recorded as the
//! expected behaviour, so it is kept deliberately rather than by accident.
//!
//! **Protocol-specific data models live above this layer.** `package.rs` keeps its manifest types
//! and `lsp/protocol.rs` its message types; both delegate parsing and string escaping here.

use std::collections::HashMap;
use std::fmt;

/// A JSON value. One model, shared by the manifest reader and the LSP transport — the two
/// definitions this replaced were textually identical, so nothing was reconciled to get here.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(HashMap<String, JsonValue>),
}

/// A JSON number, kept as the **exact text the document contained**.
///
/// DEV-185: this was an `f64`, so `9007199254740993` — an ordinary 64-bit request id — came back
/// as `9007199254740992`, and `as_i64` performed `n as i64`, turning `1.5` into `1`. The value the
/// input denoted was gone before any consumer could state what it needed.
///
/// The shared layer's job is to preserve what the document said. Deciding which numeric type a
/// field requires belongs to the consumer, and is an explicit conversion that can fail:
///
/// ```text
///   JSON parser ── exact JSON number ──┬── manifest consumer picks its type
///                                      └── protocol consumer picks its type
/// ```
///
/// Not arbitrary-precision arithmetic — no arithmetic at all. `raw` is always a syntactically valid
/// RFC 8259 number because the parser is the only thing that fills it in, and [`JsonNumber::from_f64`]
/// refuses NaN and the infinities, which have no JSON form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonNumber {
    raw: String,
}

impl JsonNumber {
    /// The number's exact text, as it appeared.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// The value as an `i64`, **only** if the document wrote an integer that fits.
    ///
    /// `1.5` and `1e3` return `None` rather than a truncated or re-derived integer: a field typed
    /// as an integer should refuse a document that did not write one, and `1e3` — while equal to
    /// 1000 — is a decision the caller should make deliberately if it wants it.
    pub fn as_i64(&self) -> Option<i64> {
        self.raw.parse::<i64>().ok()
    }

    /// The value as a `u64`, on the same terms.
    pub fn as_u64(&self) -> Option<u64> {
        self.raw.parse::<u64>().ok()
    }

    /// The value as a finite `f64`, or `None` if it cannot be represented.
    ///
    /// `1e400` is valid JSON and has no `f64`; returning infinity would be the same class of silent
    /// substitution this deviation is about.
    pub fn as_f64(&self) -> Option<f64> {
        self.raw
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
    }

    pub fn from_i64(value: i64) -> Self {
        JsonNumber {
            raw: value.to_string(),
        }
    }

    /// A `JsonNumber` for a finite `f64`. `None` for NaN and the infinities, which have no JSON
    /// textual form — so the value model cannot hold a document that fails to serialize.
    pub fn from_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let raw = if value == value.trunc() && value.abs() < 1e15 {
            format!("{}", value as i64)
        } else {
            format!("{value}")
        };
        Some(JsonNumber { raw })
    }
}

impl fmt::Display for JsonNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value as an `i64`, only if the document wrote an integer that fits — see
    /// [`JsonNumber::as_i64`].
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(number) => number.as_i64(),
            _ => None,
        }
    }

    /// The value as a finite `f64` — see [`JsonNumber::as_f64`].
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(number) => number.as_f64(),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<&JsonNumber> {
        match self {
            JsonValue::Number(number) => Some(number),
            _ => None,
        }
    }

    /// A number value for a finite `f64`; `None` for NaN and the infinities.
    pub fn number_from_f64(value: f64) -> Option<Self> {
        JsonNumber::from_f64(value).map(JsonValue::Number)
    }

    pub fn number_from_i64(value: i64) -> Self {
        JsonValue::Number(JsonNumber::from_i64(value))
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, JsonValue>> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(o) => o.get(key),
            _ => None,
        }
    }
}

/// Why a document was refused, and where.
///
/// `offset` is a byte offset into the input. Both previous parsers reported position-free strings
/// (`package.rs`) or nothing at all (`lsp/protocol.rs` returned `Option`), so a malformed manifest
/// said *what* was wrong but never *where*.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError {
    pub message: String,
    pub offset: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at byte {})", self.message, self.offset)
    }
}

/// Maximum nesting depth.
///
/// RFC 8259 §9 explicitly permits an implementation to set a depth limit, and a recursive-descent
/// parser without one turns `[[[[[…` into a stack overflow — an abort, not a diagnostic, on input
/// that arrives from a file or a socket. 128 is far above anything a manifest or an LSP message
/// legitimately reaches; the deepest first-party manifest nests 3.
pub const MAX_DEPTH: usize = 128;

/// Parse `input` as a complete JSON document.
///
/// "Complete" is load-bearing: trailing input after the top-level value is an error. The LSP parser
/// accepted it, which meant a truncated or concatenated message body could parse as its first value
/// and the remainder vanish silently.
pub fn parse(input: &str) -> Result<JsonValue, JsonError> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
        depth: 0,
    };
    parser.skip_whitespace();
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.pos < parser.bytes.len() {
        return Err(parser.error("trailing input after the top-level JSON value"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn error(&self, message: impl Into<String>) -> JsonError {
        JsonError {
            message: message.into(),
            offset: self.pos,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// RFC 8259 §2: only space, tab, LF and CR are whitespace. `char::is_whitespace` — which
    /// `package.rs` used — also accepts U+00A0, U+2028 and friends, so it skipped bytes the grammar
    /// does not permit anywhere.
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), JsonError> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.error(format!("expected '{}'", byte as char)))
        }
    }

    fn value(&mut self) -> Result<JsonValue, JsonError> {
        match self.peek() {
            None => Err(self.error("unexpected end of input")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(JsonValue::String),
            Some(b't') => self.literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.literal("null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(other) => Err(self.error(format!(
                "unexpected character '{}' where a value was expected",
                char::from(other).escape_debug()
            ))),
        }
    }

    fn literal(&mut self, word: &str, value: JsonValue) -> Result<JsonValue, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error(format!("expected '{word}'")))
        }
    }

    fn enter(&mut self) -> Result<(), JsonError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error(format!("nesting deeper than {MAX_DEPTH} levels")));
        }
        Ok(())
    }

    fn object(&mut self) -> Result<JsonValue, JsonError> {
        self.enter()?;
        self.pos += 1; // '{'
        let mut object = HashMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Object(object));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                // The specific case worth naming: `{"a":1,}`. `package.rs` accepted it.
                return Err(self.error("expected a string key (a trailing comma is not allowed)"));
            }
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value()?;
            // Last wins — the one point where RFC 8259 says SHOULD rather than MUST.
            object.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
        self.depth -= 1;
        Ok(JsonValue::Object(object))
    }

    fn array(&mut self) -> Result<JsonValue, JsonError> {
        self.enter()?;
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                return Err(self.error("a trailing comma is not allowed before ']'"));
            }
            items.push(self.value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
        self.depth -= 1;
        Ok(JsonValue::Array(items))
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let Some(byte) = self.peek() else {
                return Err(self.error("unterminated string"));
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    self.escape(&mut out)?;
                }
                // RFC 8259 §7: U+0000 through U+001F MUST be escaped. Both previous parsers copied
                // them through, so a manifest could carry a literal NUL or ESC in a package name.
                0x00..=0x1f => {
                    return Err(self.error(format!(
                        "raw control character U+{byte:04X} in a string; it must be escaped"
                    )));
                }
                _ => {
                    // Copy one whole UTF-8 scalar. The input is a `&str`, so it is well-formed.
                    let start = self.pos;
                    self.pos += utf8_width(byte);
                    let slice = &self.bytes[start..self.pos.min(self.bytes.len())];
                    match std::str::from_utf8(slice) {
                        Ok(text) => out.push_str(text),
                        Err(_) => return Err(self.error("malformed UTF-8 in a string")),
                    }
                }
            }
        }
    }

    /// One escape sequence, the backslash already consumed.
    fn escape(&mut self, out: &mut String) -> Result<(), JsonError> {
        let Some(byte) = self.peek() else {
            return Err(self.error("unterminated escape sequence"));
        };
        self.pos += 1;
        let simple = match byte {
            b'"' => Some('"'),
            b'\\' => Some('\\'),
            b'/' => Some('/'),
            b'b' => Some('\u{08}'),
            b'f' => Some('\u{0c}'),
            b'n' => Some('\n'),
            b'r' => Some('\r'),
            b't' => Some('\t'),
            _ => None,
        };
        if let Some(character) = simple {
            out.push(character);
            return Ok(());
        }
        if byte != b'u' {
            self.pos -= 1;
            return Err(self.error(format!(
                "unsupported escape '\\{}'",
                char::from(byte).escape_debug()
            )));
        }

        // `\uXXXX`, with surrogate pairing. DEV-182: the LSP parser decoded a valid pair to the
        // EMPTY STRING and an unpaired surrogate to `""`, both silently. `package.rs` had no `'u'`
        // arm at all, so every escape was rejected. Neither is RFC 8259.
        let first = self.hex4()?;
        let scalar = if (0xD800..0xDC00).contains(&first) {
            // A high surrogate must be followed by `\uDC00`–`\uDFFF`.
            if !self.bytes[self.pos..].starts_with(b"\\u") {
                return Err(self
                    .error("a high surrogate escape must be followed by a low surrogate escape"));
            }
            self.pos += 2;
            let second = self.hex4()?;
            if !(0xDC00..0xE000).contains(&second) {
                return Err(self.error(format!(
                    "expected a low surrogate escape after \\u{first:04X}, found \\u{second:04X}"
                )));
            }
            0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
        } else if (0xDC00..0xE000).contains(&first) {
            return Err(self.error(format!("unpaired low surrogate escape \\u{first:04X}")));
        } else {
            first
        };
        match char::from_u32(scalar) {
            Some(character) => {
                out.push(character);
                Ok(())
            }
            None => Err(self.error(format!("\\u{scalar:04X} is not a Unicode scalar value"))),
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.error("a \\u escape needs four hexadecimal digits"));
        }
        let mut value: u32 = 0;
        for offset in 0..4 {
            let byte = self.bytes[self.pos + offset];
            let digit = match byte {
                b'0'..=b'9' => u32::from(byte - b'0'),
                b'a'..=b'f' => u32::from(byte - b'a') + 10,
                b'A'..=b'F' => u32::from(byte - b'A') + 10,
                _ => {
                    return Err(self.error(format!(
                        "'{}' is not a hexadecimal digit",
                        char::from(byte).escape_debug()
                    )))
                }
            };
            value = value * 16 + digit;
        }
        self.pos += 4;
        Ok(value)
    }

    /// RFC 8259 §6: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
    ///
    /// Two corrections at once. Leading zeros (`01`) were accepted by both parsers and are not
    /// JSON. Exponents (`1e3`, `1.5e-3`) were rejected by `package.rs`, which consumed only digits
    /// and `.` and then reported `Expected ',' or '}'` pointing at the `e` — a misleading message
    /// for input that is perfectly valid.
    fn number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("a number may not have a leading zero"));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.error("expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit after the decimal point"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected a digit in the exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        // DEV-185: the TEXT is the value. Converting here — as `parse::<f64>()` did — is what lost
        // `9007199254740993`, and no consumer could recover it afterwards.
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .expect("the number grammar accepts ASCII only");
        Ok(JsonValue::Number(JsonNumber {
            raw: text.to_string(),
        }))
    }
}

fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

// -------------------------------------------------------------------------------- serializing --

/// Escape the *contents* of a JSON string — no surrounding quotes.
///
/// **The one escaping authority.** There were four, and three of them left C0 control characters
/// raw, which RFC 8259 §7 forbids: `stark doctor --json` emitted a document a standard parser
/// rejects whenever an install path contained a tab, and the LSP transport put raw controls on the
/// wire. Only `diag.rs`'s version was correct, and this is it.
pub fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    escape_into(&mut out, value);
    out
}

/// Append `value`'s escaped contents to `out`.
pub fn escape_into(out: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // Every remaining C0 control, which is what the three broken escapers missed.
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
}

/// A complete quoted, escaped JSON string.
pub fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    escape_into(&mut out, value);
    out.push('"');
    out
}

impl fmt::Display for JsonValue {
    /// Canonical serialization. Object keys are emitted in sorted order so a document round-trips
    /// deterministically — `HashMap` iteration order is not stable across runs, and the LSP
    /// transport's previous `Display` inherited that non-determinism.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(true) => write!(f, "true"),
            JsonValue::Bool(false) => write!(f, "false"),
            // Verbatim: `-0`, `1.50` and `1E-3` all denote something a re-derived form would
            // normalise away, and a round trip that changes the document is what DEV-185 was.
            JsonValue::Number(number) => write!(f, "{number}"),
            JsonValue::String(value) => write!(f, "{}", quote(value)),
            JsonValue::Array(items) => {
                write!(f, "[")?;
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            JsonValue::Object(entries) => {
                let mut keys: Vec<&String> = entries.keys().collect();
                keys.sort();
                write!(f, "{{")?;
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}:{}", quote(key), entries[key])?;
                }
                write!(f, "}}")
            }
        }
    }
}
