//! Syntax highlighting for `stark` code blocks, using the real
//! `lexer::tokenize_with_comments` rather than a regex-based approximation
//! — the lexer already exists and is authoritative for what a token is.

use crate::lexer::{tokenize_with_comments, Kw, TokenKind};
use crate::source::SourceFile;

/// Render `code` as an HTML `<code>` body with `<span class="tok-*">`
/// wrappers around each token; whitespace and text the lexer couldn't
/// tokenize are passed through HTML-escaped but unstyled.
pub fn highlight(code: &str) -> String {
    let file = SourceFile::new("example.stark", code.to_string());
    let (tokens, comments, _diags) = tokenize_with_comments(&file);

    // Merge tokens and comments into one position-ordered stream so gaps
    // between them (whitespace) render as plain escaped text.
    enum Piece<'a> {
        Token(crate::lexer::Token),
        Comment(&'a crate::lexer::Comment),
    }
    let mut pieces: Vec<Piece> = Vec::new();
    for t in &tokens {
        if t.kind != TokenKind::Eof {
            pieces.push(Piece::Token(*t));
        }
    }
    for c in &comments {
        pieces.push(Piece::Comment(c));
    }
    pieces.sort_by_key(|p| match p {
        Piece::Token(t) => t.span.lo,
        Piece::Comment(c) => c.span.lo,
    });

    let mut out = String::new();
    let mut cursor = 0u32;
    for piece in &pieces {
        let (lo, hi, class) = match piece {
            Piece::Token(t) => (t.span.lo, t.span.hi, token_class(&t.kind)),
            Piece::Comment(c) => (c.span.lo, c.span.hi, "comment"),
        };
        if lo > cursor {
            escape_into(&file.src[cursor as usize..lo as usize], &mut out);
        }
        out.push_str("<span class=\"tok-");
        out.push_str(class);
        out.push_str("\">");
        escape_into(&file.src[lo as usize..hi as usize], &mut out);
        out.push_str("</span>");
        cursor = hi.max(cursor);
    }
    if (cursor as usize) < file.src.len() {
        escape_into(&file.src[cursor as usize..], &mut out);
    }
    out
}

fn token_class(kind: &TokenKind) -> &'static str {
    match kind {
        TokenKind::Int { .. } | TokenKind::Float { .. } => "num",
        TokenKind::Str { .. } | TokenKind::CharLit => "str",
        TokenKind::Keyword(kw) if is_type_keyword(*kw) => "type",
        TokenKind::Keyword(_) => "kw",
        TokenKind::Reserved => "kw",
        TokenKind::Ident => "ident",
        TokenKind::Error => "error",
        TokenKind::Eof => "",
        _ => "op",
    }
}

fn is_type_keyword(kw: Kw) -> bool {
    matches!(
        kw,
        Kw::Int8
            | Kw::Int16
            | Kw::Int32
            | Kw::Int64
            | Kw::UInt8
            | Kw::UInt16
            | Kw::UInt32
            | Kw::UInt64
            | Kw::Float32
            | Kw::Float64
            | Kw::Bool
            | Kw::StringTy
            | Kw::CharTy
            | Kw::Unit
            | Kw::Str
    )
}

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    escape_into(s, &mut out);
    out
}

fn escape_into(s: &str, out: &mut String) {
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
}
