//! WP-FMT-001 — splitting an interpolated string literal into segments, and parsing a field's
//! format specification.
//!
//! Pure functions over source text. The parser owns turning a field's expression span into an
//! `ExprId`; everything about *where the pieces are* lives here, so the scan can be tested
//! directly rather than only through whole programs.
//!
//! **Spans are absolute file offsets throughout.** A diagnostic about `{value:zz}` points at `zz`,
//! not at the literal that contains it.

use crate::ast::{FormatAlign, FormatKind, FormatSign, FormatSpec};
use crate::diag::Diagnostic;
use crate::source::{SourceFile, Span};

/// LIMIT-FMT-SEGMENTS — the most segments one interpolated literal may produce.
///
/// A bound on how much AST a single token can expand into. Named, like every other compiler
/// resource limit, so exceeding it is a diagnostic rather than an allocation.
pub const MAX_SEGMENTS: usize = 1024;

/// LIMIT-FMT-WIDTH — the largest field width a specification may request.
///
/// A static request beyond this is a mistake, not an intent: the padding alone would be a megabyte.
/// Refused at compile time, so no allocator ever sees it.
pub const MAX_WIDTH: u32 = 1_000_000;

/// LIMIT-FMT-PRECISION — the most fractional digits a specification may request.
pub const MAX_PRECISION: u32 = 10_000;

/// One piece of a scanned literal. The parser converts `Field`'s `expr_span` into an expression.
#[derive(Clone, Debug)]
pub enum RawSegment {
    Literal {
        text: String,
        span: Span,
    },
    Field {
        expr_span: Span,
        spec: FormatSpec,
        span: Span,
    },
}

/// Split the body of an `f"..."` literal, whose CONTENT (between the quotes) is `body`.
///
/// Returns the segments, or the diagnostics explaining why it could not be split. A malformed
/// literal always produces at least one diagnostic and never panics — §9.3's requirement, and the
/// reason every early return here carries a span rather than unwrapping.
pub fn scan_format_literal(
    file: &SourceFile,
    body: Span,
) -> Result<Vec<RawSegment>, Vec<Diagnostic>> {
    let src = file.src.as_bytes();
    let lo = body.lo as usize;
    let hi = (body.hi as usize).min(src.len());
    let mut diags = Vec::new();
    let mut segments: Vec<RawSegment> = Vec::new();

    let mut literal = String::new();
    let mut literal_start = lo;
    let mut i = lo;

    while i < hi {
        let b = src[i];
        match b {
            // An escape is consumed WHOLE before any brace is considered, so `\u{1F600}`'s braces
            // are part of the escape and never read as a field delimiter.
            b'\\' => {
                let start = i;
                i += 1;
                if i >= hi {
                    diags.push(
                        Diagnostic::error(
                            "unterminated escape sequence",
                            Span::new(start as u32, hi as u32),
                        )
                        .with_code("E0218"),
                    );
                    return Err(diags);
                }
                let escape = src[i];
                i += 1;
                if escape == b'u' {
                    // `\u{...}` — consume through the closing brace.
                    while i < hi && src[i] != b'}' {
                        i += 1;
                    }
                    if i < hi {
                        i += 1;
                    }
                }
                match decode_escape(&file.src[start..i]) {
                    Some(text) => literal.push_str(&text),
                    None => {
                        diags.push(
                            Diagnostic::error(
                                "invalid escape sequence",
                                Span::new(start as u32, i as u32),
                            )
                            .with_code("E0218"),
                        );
                        return Err(diags);
                    }
                }
            }
            b'{' if i + 1 < hi && src[i + 1] == b'{' => {
                literal.push('{');
                i += 2;
            }
            b'}' if i + 1 < hi && src[i + 1] == b'}' => {
                literal.push('}');
                i += 2;
            }
            b'}' => {
                diags.push(
                    Diagnostic::error(
                        "unmatched '}' in an interpolated string",
                        Span::new(i as u32, i as u32 + 1),
                    )
                    .with_code("E0218")
                    .with_label("write '}}' for a literal closing brace"),
                );
                return Err(diags);
            }
            b'{' => {
                // Flush the literal run before the field.
                if !literal.is_empty() {
                    segments.push(RawSegment::Literal {
                        text: std::mem::take(&mut literal),
                        span: Span::new(literal_start as u32, i as u32),
                    });
                }
                let field_start = i;
                let Some(field_end) = find_field_end(src, i + 1, hi) else {
                    diags.push(
                        Diagnostic::error(
                            "unterminated interpolation field",
                            Span::new(field_start as u32, hi as u32),
                        )
                        .with_code("E0218")
                        .with_label("expected a closing '}'"),
                    );
                    return Err(diags);
                };
                let inner_lo = i + 1;
                let colon = find_spec_colon(src, inner_lo, field_end);
                let expr_hi = colon.unwrap_or(field_end);
                let expr_span = Span::new(inner_lo as u32, expr_hi as u32);

                if file.src[inner_lo..expr_hi].trim().is_empty() {
                    diags.push(
                        Diagnostic::error(
                            "empty interpolation field",
                            Span::new(field_start as u32, field_end as u32 + 1),
                        )
                        .with_code("E0218")
                        .with_label("write the expression to interpolate between the braces"),
                    );
                    return Err(diags);
                }

                let spec = match colon {
                    None => FormatSpec::default(),
                    Some(colon) => {
                        let spec_span = Span::new(colon as u32 + 1, field_end as u32);
                        match parse_format_spec(&file.src[colon + 1..field_end], spec_span) {
                            Ok(spec) => spec,
                            Err(diagnostic) => {
                                diags.push(diagnostic);
                                return Err(diags);
                            }
                        }
                    }
                };

                segments.push(RawSegment::Field {
                    expr_span,
                    spec,
                    span: Span::new(field_start as u32, field_end as u32 + 1),
                });
                if segments.len() > MAX_SEGMENTS {
                    diags.push(
                        Diagnostic::error(
                            format!(
                                "interpolated string has more than {MAX_SEGMENTS} segments \
                                 (LIMIT-FMT-SEGMENTS)"
                            ),
                            body,
                        )
                        .with_code("E0218"),
                    );
                    return Err(diags);
                }
                i = field_end + 1;
                literal_start = i;
            }
            _ => {
                // Advance one whole UTF-8 scalar, so a multi-byte character is never split.
                let len = utf8_len(b);
                let end = (i + len).min(hi);
                literal.push_str(&file.src[i..end]);
                i = end;
            }
        }
    }

    if !literal.is_empty() {
        segments.push(RawSegment::Literal {
            text: literal,
            span: Span::new(literal_start as u32, hi as u32),
        });
    }
    Ok(segments)
}

fn utf8_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        // A continuation byte here means the slice started mid-scalar, which cannot happen for a
        // literal the lexer already validated. Advancing one byte keeps the scan terminating.
        _ => 1,
    }
}

fn decode_escape(text: &str) -> Option<String> {
    // Reuse the ONE cooked-string decoder rather than writing a second escape table: an escape
    // means the same thing inside an interpolated literal as in an ordinary one.
    let quoted = format!("\"{text}\"");
    if crate::literal::cooked_string_is_valid(&quoted) {
        Some(crate::literal::parse_string(&quoted, false))
    } else {
        None
    }
}

/// The index of the `}` that closes a field opened at `start`, tracking nesting.
///
/// Depth counts `(`, `[` and `{`; a `{` inside the field is a struct literal's, not the end of the
/// field, which is what makes `f"{Point { x: 1, y: 2 }}"` parse. A nested string literal is
/// skipped whole, so a brace or colon inside it is text.
fn find_field_end(src: &[u8], start: usize, hi: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = start;
    while i < hi {
        match src[i] {
            // An escape inside a field is consumed whole. A nested string literal is written
            // `\"..\"` — the f-string's own delimiter forces it — so an unescaped `"` byte here
            // would be the literal's end, and `\"` must not be read as one.
            b'\\' => i += 2,
            b'"' => i = skip_string(src, i, hi),
            b'\'' => i = skip_char_literal(src, i, hi),
            b'(' | b'[' | b'{' => {
                depth += 1;
                i += 1;
            }
            b')' | b']' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b'}' => {
                if depth == 0 {
                    return Some(i);
                }
                depth -= 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    None
}

/// The index of the `:` that separates the expression from its specification, or `None`.
///
/// Only a TOP-LEVEL, single `:` counts. A `:` at depth > 0 belongs to a struct literal's field; a
/// `::` is a path separator. Both are inside the expression.
fn find_spec_colon(src: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = start;
    while i < end {
        match src[i] {
            b'\\' => i += 2,
            b'"' => i = skip_string(src, i, end),
            b'\'' => i = skip_char_literal(src, i, end),
            b'(' | b'[' | b'{' => {
                depth += 1;
                i += 1;
            }
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                i += 1;
            }
            b':' if depth == 0 => {
                if i + 1 < end && src[i + 1] == b':' {
                    i += 2; // `::` — a path, not a specification.
                    continue;
                }
                return Some(i);
            }
            _ => i += 1,
        }
    }
    None
}

fn skip_string(src: &[u8], start: usize, hi: usize) -> usize {
    let mut i = start + 1;
    while i < hi {
        match src[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    hi
}

fn skip_char_literal(src: &[u8], start: usize, hi: usize) -> usize {
    let mut i = start + 1;
    while i < hi {
        match src[i] {
            b'\\' => i += 2,
            b'\'' => return i + 1,
            _ => i += 1,
        }
    }
    hi
}

// ------------------------------------------------------------------ the specification grammar --

/// Parse a field's specification:
///
/// ```text
/// format_spec := [ [fill] align ] [ sign ] [ "#" ] [ "0" ] [ width ] [ "." precision ] [ type ]
/// ```
///
/// Every rejection is a compile-time diagnostic. Nothing is silently ignored — an unrecognised
/// character is an error, because a specification the programmer wrote and the compiler dropped is
/// worse than one it refused.
pub fn parse_format_spec(text: &str, span: Span) -> Result<FormatSpec, Diagnostic> {
    let mut spec = FormatSpec {
        span: Some(span),
        ..FormatSpec::default()
    };
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;

    // [[fill] align] — a fill character is only a fill if an alignment follows it.
    if chars.len() >= 2 {
        if let Some(align) = align_of(chars[1]) {
            spec.fill = Some(chars[0]);
            spec.align = Some(align);
            i = 2;
        }
    }
    if spec.align.is_none() && !chars.is_empty() {
        if let Some(align) = align_of(chars[0]) {
            spec.align = Some(align);
            i = 1;
        }
    }

    // [sign]
    if i < chars.len() {
        match chars[i] {
            '+' => {
                spec.sign = Some(FormatSign::Plus);
                i += 1;
            }
            '-' => {
                spec.sign = Some(FormatSign::Minus);
                i += 1;
            }
            ' ' => {
                spec.sign = Some(FormatSign::Space);
                i += 1;
            }
            _ => {}
        }
    }

    // ["#"]
    if i < chars.len() && chars[i] == '#' {
        spec.alternate = true;
        i += 1;
    }

    // ["0"] — zero-fill shorthand. A leading `0` here is a flag; any following digits are width.
    if i < chars.len() && chars[i] == '0' {
        spec.zero_pad = true;
        i += 1;
    }

    // [width]
    let width_start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i > width_start {
        let digits: String = chars[width_start..i].iter().collect();
        let width = digits.parse::<u64>().unwrap_or(u64::MAX);
        if width > MAX_WIDTH as u64 {
            return Err(Diagnostic::error(
                format!("format width exceeds the maximum of {MAX_WIDTH} (LIMIT-FMT-WIDTH)"),
                span,
            )
            .with_code("E0218"));
        }
        spec.width = Some(width as u32);
    }

    // ["." precision]
    if i < chars.len() && chars[i] == '.' {
        i += 1;
        let precision_start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i == precision_start {
            return Err(
                Diagnostic::error("format precision requires a number after '.'", span)
                    .with_code("E0218"),
            );
        }
        let digits: String = chars[precision_start..i].iter().collect();
        let precision = digits.parse::<u64>().unwrap_or(u64::MAX);
        if precision > MAX_PRECISION as u64 {
            return Err(Diagnostic::error(
                format!(
                    "format precision exceeds the maximum of {MAX_PRECISION} \
                     (LIMIT-FMT-PRECISION)"
                ),
                span,
            )
            .with_code("E0218"));
        }
        spec.precision = Some(precision as u32);
    }

    // [type]
    if i < chars.len() {
        let kind = match chars[i] {
            'b' => Some(FormatKind::Bin),
            'o' => Some(FormatKind::Oct),
            'x' => Some(FormatKind::LowerHex),
            'X' => Some(FormatKind::UpperHex),
            'f' => Some(FormatKind::Fixed),
            _ => None,
        };
        match kind {
            Some(kind) => {
                spec.kind = Some(kind);
                i += 1;
            }
            None => {
                return Err(
                    Diagnostic::error(format!("unknown format type '{}'", chars[i]), span)
                        .with_code("E0218")
                        .with_label("expected one of 'b', 'o', 'x', 'X' or 'f'"),
                );
            }
        }
    }

    if i != chars.len() {
        let rest: String = chars[i..].iter().collect();
        return Err(Diagnostic::error(
            format!("unexpected '{rest}' in a format specification"),
            span,
        )
        .with_code("E0218"));
    }

    // An alignment with no width aligns inside a field of width zero — it does nothing. That is
    // almost always a typo, so it is refused rather than silently accepted (§16's ruling).
    if spec.align.is_some() && spec.width.is_none() {
        return Err(Diagnostic::error("alignment requires a width", span)
            .with_code("E0218")
            .with_label("write the field width after the alignment, e.g. '>10'"));
    }

    Ok(spec)
}

fn align_of(ch: char) -> Option<FormatAlign> {
    match ch {
        '<' => Some(FormatAlign::Left),
        '>' => Some(FormatAlign::Right),
        '^' => Some(FormatAlign::Center),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(text: &str) -> Result<FormatSpec, String> {
        parse_format_spec(text, Span::new(0, text.len() as u32)).map_err(|d| d.message)
    }

    fn scan(source: &str) -> Result<Vec<RawSegment>, Vec<String>> {
        // `source` is the literal BODY, as it appears between the quotes.
        let file = SourceFile::new("t", source.to_string());
        let body = Span::new(0, source.len() as u32);
        scan_format_literal(&file, body).map_err(|ds| ds.into_iter().map(|d| d.message).collect())
    }

    fn literals(segments: &[RawSegment]) -> Vec<String> {
        segments
            .iter()
            .filter_map(|s| match s {
                RawSegment::Literal { text, .. } => Some(text.clone()),
                RawSegment::Field { .. } => None,
            })
            .collect()
    }

    fn field_exprs(source: &str, segments: &[RawSegment]) -> Vec<String> {
        segments
            .iter()
            .filter_map(|s| match s {
                RawSegment::Field { expr_span, .. } => {
                    Some(source[expr_span.lo as usize..expr_span.hi as usize].to_string())
                }
                RawSegment::Literal { .. } => None,
            })
            .collect()
    }

    #[test]
    fn plain_text_is_one_literal_segment() {
        let segments = scan("plain").unwrap();
        assert_eq!(literals(&segments), vec!["plain"]);
    }

    #[test]
    fn escaped_braces_become_literal_braces() {
        let segments = scan("{{ and }}").unwrap();
        assert_eq!(literals(&segments), vec!["{ and }"]);
    }

    #[test]
    fn a_unicode_escape_does_not_open_a_field() {
        // `\u{1F600}` contains braces. A scanner that looked at bytes before escapes would read
        // `{1F600}` as an interpolation field.
        let segments = scan("emoji \\u{1F600} done").unwrap();
        assert_eq!(literals(&segments), vec!["emoji \u{1F600} done"]);
    }

    #[test]
    fn a_struct_literal_does_not_end_the_field() {
        let source = "{Point { x: 1, y: 2 }}";
        let segments = scan(source).unwrap();
        assert_eq!(field_exprs(source, &segments), vec!["Point { x: 1, y: 2 }"]);
    }

    #[test]
    fn a_path_separator_is_not_a_spec_separator() {
        let source = "{module::CONSTANT}";
        let segments = scan(source).unwrap();
        assert_eq!(field_exprs(source, &segments), vec!["module::CONSTANT"]);
    }

    #[test]
    fn a_nested_string_hides_braces_and_colons() {
        let source = "{call(\"a:b}c\")}";
        let segments = scan(source).unwrap();
        assert_eq!(field_exprs(source, &segments), vec!["call(\"a:b}c\")"]);
    }

    #[test]
    fn a_top_level_colon_starts_the_spec() {
        let source = "{value:>10}";
        let segments = scan(source).unwrap();
        assert_eq!(field_exprs(source, &segments), vec!["value"]);
        let RawSegment::Field { spec, .. } = &segments[0] else {
            panic!("expected a field");
        };
        assert_eq!(spec.width, Some(10));
        assert_eq!(spec.align, Some(FormatAlign::Right));
    }

    #[test]
    fn nested_calls_and_indexing_parse() {
        let source = "{call(a, nested(b))}{items[index]}";
        let segments = scan(source).unwrap();
        assert_eq!(
            field_exprs(source, &segments),
            vec!["call(a, nested(b))", "items[index]"]
        );
    }

    #[test]
    fn malformed_literals_are_diagnosed_not_panicked() {
        assert!(scan("{value").is_err());
        assert!(scan("value}").is_err());
        assert!(scan("{}").is_err());
        assert!(scan("{   }").is_err());
        assert!(scan("{{}").is_err());
    }

    #[test]
    fn specs_parse_their_parts() {
        let s = spec(".^12").unwrap();
        assert_eq!(s.fill, Some('.'));
        assert_eq!(s.align, Some(FormatAlign::Center));
        assert_eq!(s.width, Some(12));

        let s = spec("#010x").unwrap();
        assert!(s.alternate);
        assert!(s.zero_pad);
        assert_eq!(s.width, Some(10));
        assert_eq!(s.kind, Some(FormatKind::LowerHex));

        let s = spec("+").unwrap();
        assert_eq!(s.sign, Some(FormatSign::Plus));

        let s = spec(".2").unwrap();
        assert_eq!(s.precision, Some(2));

        let s = spec(".2f").unwrap();
        assert_eq!(s.precision, Some(2));
        assert_eq!(s.kind, Some(FormatKind::Fixed));

        let s = spec("04").unwrap();
        assert!(s.zero_pad);
        assert_eq!(s.width, Some(4));
    }

    #[test]
    fn bad_specs_are_rejected_with_a_reason() {
        assert!(spec("unknown").is_err());
        assert!(spec("ab>10").is_err());
        assert!(spec(">").is_err());
        assert!(spec(".").is_err());
        assert!(spec("999999999999999999999999999").is_err());
        assert!(spec(".99999").is_err());
    }

    #[test]
    fn a_two_character_fill_is_not_a_fill() {
        // `ab>10` would need `a` to be a fill and `b` an alignment. `b` is not an alignment, so
        // this is a malformed specification rather than a fill of `ab`.
        let message = spec("ab>10").unwrap_err();
        assert!(
            message.contains("unexpected") || message.contains("unknown"),
            "{message}"
        );
    }
}
