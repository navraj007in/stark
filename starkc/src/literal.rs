//! Shared literal-value parsing.
//!
//! `typecheck.rs` (pattern-literal comparison in `pat_subsumes`, and integer-literal magnitude
//! checking) and `interp.rs` (`eval_lit`) both need to turn a `Lit` + its source text into an
//! actual value. Before this module existed, `interp.rs` had its own private copy of this logic
//! and `typecheck.rs` had none at all -- `pat_subsumes` compared `Lit` shape only (base/suffix
//! tags, no value), so `match x { 1 => .., 2 => .. }` treated any two same-kind integer literal
//! patterns as equal regardless of value. One shared implementation avoids that kind of drift.

use crate::ast::Lit;
use crate::lexer::{Base, FloatSuffix, IntSuffix};

#[derive(Debug, Clone, PartialEq)]
pub enum LitValue {
    Int(i128),
    Float(f64),
    Str(String),
    Char(char),
    Bool(bool),
}

/// Parse a literal's value from its `Lit` shape tag and source text. Returns `None` on a
/// malformed literal (should not happen for anything that passed the lexer/parser, but this
/// stays fallible rather than panicking since it is also used for optimistic pattern comparison).
pub fn eval_lit_value(lit: Lit, text: &str) -> Option<LitValue> {
    match lit {
        Lit::Bool(value) => Some(LitValue::Bool(value)),
        Lit::Char => parse_char(text).map(LitValue::Char),
        Lit::Str { raw } => Some(LitValue::Str(parse_string(text, raw))),
        Lit::Int { base, suffix } => parse_int_literal(text, base, suffix).map(LitValue::Int),
        Lit::Float { suffix } => parse_float_literal(text, suffix).map(LitValue::Float),
    }
}

pub fn parse_int_literal(text: &str, base: Base, suffix: Option<IntSuffix>) -> Option<i128> {
    let mut digits = text.replace('_', "");
    if let Some(suffix) = suffix {
        let suffix = format!("{suffix:?}").to_ascii_lowercase();
        digits.truncate(digits.len().saturating_sub(suffix.len()));
    }
    let (digits, radix) = match base {
        Base::Dec => (digits.as_str(), 10),
        Base::Bin => (digits.trim_start_matches("0b"), 2),
        Base::Oct => (digits.trim_start_matches("0o"), 8),
        Base::Hex => (digits.trim_start_matches("0x"), 16),
    };
    i128::from_str_radix(digits, radix).ok()
}

pub fn parse_float_literal(text: &str, suffix: Option<FloatSuffix>) -> Option<f64> {
    let mut number = text.replace('_', "");
    if let Some(suffix) = suffix {
        let suffix = format!("{suffix:?}").to_ascii_lowercase();
        number.truncate(number.len().saturating_sub(suffix.len()));
    }
    number.parse::<f64>().ok()
}

pub fn parse_string(text: &str, raw: bool) -> String {
    let content = if raw {
        text.strip_prefix('r').unwrap_or(text)
    } else {
        text
    };
    let content = content
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(content);
    if raw {
        return content.to_string();
    }
    let mut result = String::new();
    let mut chars = content.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => result.push('\n'),
            Some('r') => result.push('\r'),
            Some('t') => result.push('\t'),
            Some('0') => result.push('\0'),
            Some('\\') => result.push('\\'),
            Some('"') => result.push('"'),
            Some(other) => result.push(other),
            None => {}
        }
    }
    result
}

pub fn parse_char(text: &str) -> Option<char> {
    let content = text.strip_prefix('\'')?.strip_suffix('\'')?;
    if let Some(escaped) = content.strip_prefix('\\') {
        match escaped {
            "n" => Some('\n'),
            "r" => Some('\r'),
            "t" => Some('\t'),
            "0" => Some('\0'),
            "\\" => Some('\\'),
            "'" => Some('\''),
            _ => None,
        }
    } else {
        let mut chars = content.chars();
        let value = chars.next()?;
        chars.next().is_none().then_some(value)
    }
}

/// Whether an already-parsed integer value fits in the given suffix's representable range.
pub fn int_suffix_range_contains(suffix: IntSuffix, value: i128) -> bool {
    match suffix {
        IntSuffix::I8 => i8::try_from(value).is_ok(),
        IntSuffix::I16 => i16::try_from(value).is_ok(),
        IntSuffix::I32 => i32::try_from(value).is_ok(),
        IntSuffix::I64 => i64::try_from(value).is_ok(),
        IntSuffix::U8 => u8::try_from(value).is_ok(),
        IntSuffix::U16 => u16::try_from(value).is_ok(),
        IntSuffix::U32 => u32::try_from(value).is_ok(),
        IntSuffix::U64 => u64::try_from(value).is_ok(),
    }
}
