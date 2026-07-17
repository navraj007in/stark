//! Lexer for `01-Lexical-Grammar.md`.
//!
//! Hand-written (PLAN.md T4). Behavior required by the spec:
//! - maximal munch, longest operator first;
//! - keywords vs reserved words vs identifiers (reserved words lex as
//!   `Reserved` so the parser can say "reserved for future use");
//! - literal suffixes and the strict underscore rule (separators only
//!   *between* digits);
//! - nested block comments; raw strings without escape processing;
//! - `>>`/`>>=` lexed as single tokens; the parser splits them in
//!   generic-argument position;
//! - lexing continues after an error (an `Error` token is emitted and a
//!   diagnostic recorded) so one bad literal doesn't hide later ones.
//!
//! Tokens carry no owned text: consumers slice the source via the span.
//! Comments are not part of the token stream (the parser never sees them),
//! but `tokenize_with_comments` collects them as a separate trivia list —
//! keyed by span, ordered by position — for tooling that must not discard
//! them (the formatter, `starkc/src/formatter/`).

use crate::diag::Diagnostic;
use crate::source::{SourceFile, Span};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TokenKind {
    // Literals
    Int {
        base: Base,
        suffix: Option<IntSuffix>,
    },
    Float {
        suffix: Option<FloatSuffix>,
    },
    Str {
        raw: bool,
    },
    CharLit,
    // Names
    Ident,
    Keyword(Kw),
    /// Reserved for future use (`async`, `await`, `dyn`, ...). Never valid
    /// in a program; kept distinct for good diagnostics.
    Reserved,

    // Operators (longest-munch order documented in `operator()`)
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    AndAnd,
    OrOr,
    Bang,
    Amp,
    Pipe,
    Caret,
    Tilde,
    Shl,
    Shr,
    Eq,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    StarStarEq,
    AmpEq,
    PipeEq,
    CaretEq,
    ShlEq,
    ShrEq,
    DotDot,
    DotDotEq,
    Question,
    ColonColon,
    Dot,
    Arrow,
    FatArrow,

    // Delimiters
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semi,
    Colon,

    /// Placeholder emitted where a diagnostic was produced.
    Error,
    Eof,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Base {
    Dec,
    Hex,
    Bin,
    Oct,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum IntSuffix {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatSuffix {
    F32,
    F64,
}

/// Active keywords per `01-Lexical-Grammar.md` §1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kw {
    // Control flow
    If,
    Else,
    Match,
    For,
    While,
    Loop,
    Break,
    Continue,
    Return,
    // Declarations
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    Let,
    Mut,
    Const,
    Type,
    Use,
    Mod,
    // Types
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    StringTy,
    CharTy,
    Unit,
    Str,
    // Visibility
    Pub,
    Priv,
    // Module paths and Self type
    SelfLower,
    SelfUpper,
    Super,
    Crate,
    // Operators
    In,
    As,
    // Literals
    True,
    False,
}

fn keyword(word: &str) -> Option<Kw> {
    use Kw::*;
    Some(match word {
        "if" => If,
        "else" => Else,
        "match" => Match,
        "for" => For,
        "while" => While,
        "loop" => Loop,
        "break" => Break,
        "continue" => Continue,
        "return" => Return,
        "fn" => Fn,
        "struct" => Struct,
        "enum" => Enum,
        "trait" => Trait,
        "impl" => Impl,
        "let" => Let,
        "mut" => Mut,
        "const" => Const,
        "type" => Type,
        "use" => Use,
        "mod" => Mod,
        "Int8" => Int8,
        "Int16" => Int16,
        "Int32" => Int32,
        "Int64" => Int64,
        "UInt8" => UInt8,
        "UInt16" => UInt16,
        "UInt32" => UInt32,
        "UInt64" => UInt64,
        "Float32" => Float32,
        "Float64" => Float64,
        "Bool" => Bool,
        "String" => StringTy,
        "Char" => CharTy,
        "Unit" => Unit,
        "str" => Str,
        "pub" => Pub,
        "priv" => Priv,
        "self" => SelfLower,
        "Self" => SelfUpper,
        "super" => Super,
        "crate" => Crate,
        "in" => In,
        "as" => As,
        "true" => True,
        "false" => False,
        _ => return None,
    })
}

/// Reserved words per `01-Lexical-Grammar.md` §9.
const RESERVED: &[&str] = &[
    "async", "await", "yield", "where", "macro", "unsafe", "extern", "import", "export", "null",
    "and", "or", "not", "is", "dyn",
];

const MAX_IDENT_LEN: usize = 255;

/// Lex an entire file. Always returns a token stream ending in `Eof`;
/// errors are reported as diagnostics with `Error` placeholder tokens.
pub fn tokenize(file: &SourceFile) -> (Vec<Token>, Vec<Diagnostic>) {
    let (tokens, _comments, diags) = tokenize_with_comments(file);
    (tokens, diags)
}

/// Like [`tokenize`], but also returns every comment in the file as trivia,
/// ordered by position. The token stream is identical to `tokenize`'s (the
/// parser still never sees comments); only tooling that must preserve or
/// reformat comments (the formatter) needs this entry point.
pub fn tokenize_with_comments(file: &SourceFile) -> (Vec<Token>, Vec<Comment>, Vec<Diagnostic>) {
    let mut lexer = Lexer {
        src: file.src.as_bytes(),
        pos: 0,
        tokens: Vec::new(),
        comments: Vec::new(),
        diags: Vec::new(),
    };
    lexer.run();
    (lexer.tokens, lexer.comments, lexer.diags)
}

/// A comment, preserved as trivia for the formatter. Not part of the token
/// stream; the parser never sees these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Comment {
    pub kind: CommentKind,
    /// Full span including the delimiters (`//`, `/*` ... `*/`), excluding
    /// the trailing newline for line comments.
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommentKind {
    /// `// text`
    Line,
    /// `/// text` — outer doc comment, attaches to the item that follows.
    LineDoc,
    /// `//! text` — inner doc comment, attaches to the enclosing item.
    LineInnerDoc,
    /// `/* text */`, possibly multi-line; nests per `01-Lexical-Grammar.md` §6.
    Block,
}

struct Lexer<'src> {
    src: &'src [u8],
    pos: usize,
    tokens: Vec<Token>,
    comments: Vec<Comment>,
    diags: Vec<Diagnostic>,
}

impl Lexer<'_> {
    fn run(&mut self) {
        while self.pos < self.src.len() {
            let start = self.pos;
            let b = self.src[self.pos];
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                b'/' if self.peek(1) == Some(b'/') => self.line_comment(start),
                b'/' if self.peek(1) == Some(b'*') => self.block_comment(start),
                b'0'..=b'9' => self.number(start),
                b'"' => self.string(start),
                b'r' if self.peek(1) == Some(b'"') => self.raw_string(start),
                b'\'' => self.char_literal(start),
                b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.ident_or_keyword(start),
                _ => self.operator(start),
            }
        }
        let end = self.src.len() as u32;
        self.push(TokenKind::Eof, Span::point(end));
    }

    // --- infrastructure -------------------------------------------------

    fn peek(&self, ahead: usize) -> Option<u8> {
        self.src.get(self.pos + ahead).copied()
    }

    fn push(&mut self, kind: TokenKind, span: Span) {
        self.tokens.push(Token { kind, span });
    }

    fn span_from(&self, start: usize) -> Span {
        Span::new(start as u32, self.pos as u32)
    }

    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.diags.push(Diagnostic::error(message, span));
        self.tokens.push(Token {
            kind: TokenKind::Error,
            span,
        });
    }

    // --- comments -------------------------------------------------------

    fn line_comment(&mut self, start: usize) {
        // `///` (exactly three slashes) is an outer doc comment; `////+` is
        // an ordinary (decorative) line comment, matching common practice.
        let kind = if self.peek(2) == Some(b'/') && self.peek(3) != Some(b'/') {
            CommentKind::LineDoc
        } else if self.peek(2) == Some(b'!') {
            CommentKind::LineInnerDoc
        } else {
            CommentKind::Line
        };
        while self.pos < self.src.len() && self.src[self.pos] != b'\n' {
            self.pos += 1;
        }
        self.comments.push(Comment {
            kind,
            span: self.span_from(start),
        });
    }

    /// Block comments nest (`01-Lexical-Grammar.md` §6). `/**` doc comments
    /// are recorded as ordinary block comments for now; doc-comment
    /// semantics for block form are unspecified.
    fn block_comment(&mut self, start: usize) {
        self.pos += 2; // consume "/*"
        let mut depth = 1usize;
        while self.pos < self.src.len() {
            if self.src[self.pos] == b'/' && self.peek(1) == Some(b'*') {
                depth += 1;
                self.pos += 2;
            } else if self.src[self.pos] == b'*' && self.peek(1) == Some(b'/') {
                depth -= 1;
                self.pos += 2;
                if depth == 0 {
                    self.comments.push(Comment {
                        kind: CommentKind::Block,
                        span: self.span_from(start),
                    });
                    return;
                }
            } else {
                self.pos += 1;
            }
        }
        let span = self.span_from(start);
        self.error("Unterminated block comment", span);
    }

    // --- identifiers and keywords ----------------------------------------

    fn ident_or_keyword(&mut self, start: usize) {
        while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') = self.peek(0) {
            self.pos += 1;
        }
        let span = self.span_from(start);
        let word = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
        if word.len() > MAX_IDENT_LEN {
            self.error(
                format!("Identifier exceeds maximum length of {MAX_IDENT_LEN} characters"),
                span,
            );
            return;
        }
        let kind = if let Some(kw) = keyword(word) {
            TokenKind::Keyword(kw)
        } else if RESERVED.contains(&word) {
            TokenKind::Reserved
        } else {
            TokenKind::Ident
        };
        self.push(kind, span);
    }

    // --- numbers ----------------------------------------------------------

    /// Consume `digit ('_'? digit)*` in the given radix. The caller
    /// guarantees the current byte is a valid digit (or this reports an
    /// error and returns false, for empty prefixed literals like `0x`).
    fn digits(&mut self, is_digit: fn(u8) -> bool) -> bool {
        let mut any = false;
        loop {
            match self.peek(0) {
                Some(b) if is_digit(b) => {
                    any = true;
                    self.pos += 1;
                }
                Some(b'_') if any && self.peek(1).is_some_and(is_digit) => {
                    self.pos += 2; // underscore + following digit
                }
                _ => break,
            }
        }
        any
    }

    fn number(&mut self, start: usize) {
        let is_dec = |b: u8| b.is_ascii_digit();
        let is_hex = |b: u8| b.is_ascii_hexdigit();
        let is_bin = |b: u8| b == b'0' || b == b'1';
        let is_oct = |b: u8| (b'0'..=b'7').contains(&b);

        let base = if self.src[self.pos] == b'0' {
            match self.peek(1) {
                Some(b'x' | b'X') => Some((Base::Hex, is_hex as fn(u8) -> bool)),
                Some(b'b' | b'B') => Some((Base::Bin, is_bin as fn(u8) -> bool)),
                Some(b'o' | b'O') => Some((Base::Oct, is_oct as fn(u8) -> bool)),
                _ => None,
            }
        } else {
            None
        };

        let mut is_float = false;
        let base = match base {
            Some((base, is_digit)) => {
                self.pos += 2; // consume "0x" / "0b" / "0o"
                if !self.digits(is_digit) {
                    self.consume_number_tail();
                    let span = self.span_from(start);
                    self.error("Invalid number format", span);
                    return;
                }
                base
            }
            None => {
                self.digits(is_dec);
                // Fraction: '.' only if followed by a digit — `42.abs()` and
                // tuple access `pair.0` must not be captured by a float.
                if self.peek(0) == Some(b'.') && self.peek(1).is_some_and(is_dec) {
                    self.pos += 1;
                    self.digits(is_dec);
                    is_float = true;
                }
                // Exponent: e/E, optional sign, then required digits.
                if let Some(b'e' | b'E') = self.peek(0) {
                    let after_sign = match self.peek(1) {
                        Some(b'+' | b'-') => 2,
                        _ => 1,
                    };
                    if self.peek(after_sign).is_some_and(is_dec) {
                        self.pos += after_sign;
                        self.digits(is_dec);
                        is_float = true;
                    }
                }
                Base::Dec
            }
        };

        // Suffix or invalid trailing junk (`42abc`, `12_`, `1__2` residue).
        let tail_start = self.pos;
        self.consume_number_tail();
        let tail = std::str::from_utf8(&self.src[tail_start..self.pos]).unwrap();
        let span = self.span_from(start);

        let kind = match (tail, is_float) {
            ("", false) => TokenKind::Int { base, suffix: None },
            ("", true) => TokenKind::Float { suffix: None },
            ("i8", false) => int(base, IntSuffix::I8),
            ("i16", false) => int(base, IntSuffix::I16),
            ("i32", false) => int(base, IntSuffix::I32),
            ("i64", false) => int(base, IntSuffix::I64),
            ("u8", false) => int(base, IntSuffix::U8),
            ("u16", false) => int(base, IntSuffix::U16),
            ("u32", false) => int(base, IntSuffix::U32),
            ("u64", false) => int(base, IntSuffix::U64),
            ("f32", true) => TokenKind::Float {
                suffix: Some(FloatSuffix::F32),
            },
            ("f64", true) => TokenKind::Float {
                suffix: Some(FloatSuffix::F64),
            },
            _ => {
                self.error("Invalid number format", span);
                return;
            }
        };
        self.push(kind, span);
    }

    /// Consume identifier-like characters (and stray underscores) following
    /// a number so the whole malformed literal is one error span.
    fn consume_number_tail(&mut self) {
        while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') = self.peek(0) {
            self.pos += 1;
        }
    }

    // --- strings and chars -------------------------------------------------

    /// Validate one escape sequence after the backslash has been *peeked*
    /// (`self.pos` is on the backslash). Returns false on error (reported).
    fn escape(&mut self, literal_start: usize) -> bool {
        let esc_start = self.pos;
        self.pos += 1; // consume '\'
        match self.peek(0) {
            Some(b'n' | b't' | b'r' | b'0' | b'\\' | b'\'' | b'"') => {
                self.pos += 1;
                true
            }
            Some(b'x') => {
                self.pos += 1;
                for _ in 0..2 {
                    if self.peek(0).is_some_and(|b| b.is_ascii_hexdigit()) {
                        self.pos += 1;
                    } else {
                        let span = Span::new(esc_start as u32, self.pos as u32);
                        self.error("Invalid escape sequence: \\x expects two hex digits", span);
                        return false;
                    }
                }
                true
            }
            Some(b'u') => {
                self.pos += 1;
                if self.peek(0) != Some(b'{') {
                    let span = self.span_from(esc_start);
                    self.error("Invalid escape sequence: \\u expects '{'", span);
                    return false;
                }
                self.pos += 1;
                let digits_start = self.pos;
                while self.peek(0).is_some_and(|b| b.is_ascii_hexdigit()) {
                    self.pos += 1;
                }
                let ndigits = self.pos - digits_start;
                let ok_digits = (1..=6).contains(&ndigits);
                let value = std::str::from_utf8(&self.src[digits_start..self.pos])
                    .ok()
                    .and_then(|s| u32::from_str_radix(s, 16).ok());
                if self.peek(0) != Some(b'}') || !ok_digits {
                    let span = self.span_from(esc_start);
                    self.error(
                        "Invalid escape sequence: \\u{...} expects 1-6 hex digits",
                        span,
                    );
                    return false;
                }
                self.pos += 1; // consume '}'
                match value {
                    Some(v) if char::from_u32(v).is_some() => true,
                    _ => {
                        let span = self.span_from(esc_start);
                        self.error("Invalid unicode escape: not a Unicode scalar value", span);
                        false
                    }
                }
            }
            _ => {
                if self.peek(0).is_some() {
                    self.pos += 1;
                }
                let span = self.span_from(esc_start);
                self.error("Invalid escape sequence", span);
                // Skip to a plausible end of this literal so we don't cascade.
                let _ = literal_start;
                false
            }
        }
    }

    fn string(&mut self, start: usize) {
        self.pos += 1; // consume '"'
        let mut bad = false;
        loop {
            match self.peek(0) {
                None => {
                    let span = self.span_from(start);
                    self.error("Unterminated string literal", span);
                    return;
                }
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    if !self.escape(start) {
                        bad = true;
                    }
                }
                Some(_) => self.pos += 1,
            }
        }
        if !bad {
            let span = self.span_from(start);
            self.push(TokenKind::Str { raw: false }, span);
        }
    }

    /// `r"..."` — no escapes; ends at the first `"` (spec: `r" .*? "`).
    fn raw_string(&mut self, start: usize) {
        self.pos += 2; // consume `r"`
        loop {
            match self.peek(0) {
                None => {
                    let span = self.span_from(start);
                    self.error("Unterminated string literal", span);
                    return;
                }
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(_) => self.pos += 1,
            }
        }
        let span = self.span_from(start);
        self.push(TokenKind::Str { raw: true }, span);
    }

    fn char_literal(&mut self, start: usize) {
        self.pos += 1; // consume '\''
        match self.peek(0) {
            None => {
                let span = self.span_from(start);
                self.error("Unterminated character literal", span);
                return;
            }
            Some(b'\'') => {
                self.pos += 1;
                let span = self.span_from(start);
                self.error("Empty character literal", span);
                return;
            }
            Some(b'\\') => {
                if !self.escape(start) {
                    return;
                }
            }
            Some(_) => {
                // One character content, which may be multi-byte UTF-8.
                let rest = std::str::from_utf8(&self.src[self.pos..]).unwrap_or("");
                let ch_len = rest.chars().next().map_or(1, char::len_utf8);
                self.pos += ch_len;
            }
        }
        if self.peek(0) == Some(b'\'') {
            self.pos += 1;
            let span = self.span_from(start);
            self.push(TokenKind::CharLit, span);
        } else {
            // Consume up to the closing quote or line end for one error span.
            while let Some(b) = self.peek(0) {
                self.pos += 1;
                if b == b'\'' || b == b'\n' {
                    break;
                }
            }
            let span = self.span_from(start);
            self.error("Character literal must contain exactly one character", span);
        }
    }

    // --- operators and delimiters -------------------------------------------

    /// Maximal munch: three-byte operators, then two-byte, then one-byte.
    fn operator(&mut self, start: usize) {
        use TokenKind::*;
        let b0 = self.src[self.pos];
        let b1 = self.peek(1);
        let b2 = self.peek(2);

        let (kind, len) = match (b0, b1, b2) {
            (b'*', Some(b'*'), Some(b'=')) => (StarStarEq, 3),
            (b'<', Some(b'<'), Some(b'=')) => (ShlEq, 3),
            (b'>', Some(b'>'), Some(b'=')) => (ShrEq, 3),
            (b'.', Some(b'.'), Some(b'=')) => (DotDotEq, 3),

            (b'*', Some(b'*'), _) => (StarStar, 2),
            (b'=', Some(b'='), _) => (EqEq, 2),
            (b'!', Some(b'='), _) => (NotEq, 2),
            (b'<', Some(b'='), _) => (LtEq, 2),
            (b'>', Some(b'='), _) => (GtEq, 2),
            (b'&', Some(b'&'), _) => (AndAnd, 2),
            (b'|', Some(b'|'), _) => (OrOr, 2),
            (b'<', Some(b'<'), _) => (Shl, 2),
            (b'>', Some(b'>'), _) => (Shr, 2),
            (b'+', Some(b'='), _) => (PlusEq, 2),
            (b'-', Some(b'='), _) => (MinusEq, 2),
            (b'*', Some(b'='), _) => (StarEq, 2),
            (b'/', Some(b'='), _) => (SlashEq, 2),
            (b'%', Some(b'='), _) => (PercentEq, 2),
            (b'&', Some(b'='), _) => (AmpEq, 2),
            (b'|', Some(b'='), _) => (PipeEq, 2),
            (b'^', Some(b'='), _) => (CaretEq, 2),
            (b'.', Some(b'.'), _) => (DotDot, 2),
            (b':', Some(b':'), _) => (ColonColon, 2),
            (b'-', Some(b'>'), _) => (Arrow, 2),
            (b'=', Some(b'>'), _) => (FatArrow, 2),

            (b'+', ..) => (Plus, 1),
            (b'-', ..) => (Minus, 1),
            (b'*', ..) => (Star, 1),
            (b'/', ..) => (Slash, 1),
            (b'%', ..) => (Percent, 1),
            (b'=', ..) => (Eq, 1),
            (b'<', ..) => (Lt, 1),
            (b'>', ..) => (Gt, 1),
            (b'!', ..) => (Bang, 1),
            (b'&', ..) => (Amp, 1),
            (b'|', ..) => (Pipe, 1),
            (b'^', ..) => (Caret, 1),
            (b'~', ..) => (Tilde, 1),
            (b'.', ..) => (Dot, 1),
            (b'?', ..) => (Question, 1),
            (b'(', ..) => (LParen, 1),
            (b')', ..) => (RParen, 1),
            (b'[', ..) => (LBracket, 1),
            (b']', ..) => (RBracket, 1),
            (b'{', ..) => (LBrace, 1),
            (b'}', ..) => (RBrace, 1),
            (b',', ..) => (Comma, 1),
            (b';', ..) => (Semi, 1),
            (b':', ..) => (Colon, 1),

            _ => {
                // Skip one full UTF-8 character, not one byte.
                let rest = std::str::from_utf8(&self.src[self.pos..]).unwrap_or("");
                let ch = rest.chars().next();
                let ch_len = ch.map_or(1, char::len_utf8);
                self.pos += ch_len;
                let span = self.span_from(start);
                let display = ch.map_or(String::from("?"), |c| c.to_string());
                self.error(format!("Unexpected character '{display}'"), span);
                return;
            }
        };
        self.pos += len;
        self.push(kind, self.span_from(start));
    }
}

fn int(base: Base, suffix: IntSuffix) -> TokenKind {
    TokenKind::Int {
        base,
        suffix: Some(suffix),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> (Vec<TokenKind>, Vec<String>) {
        let file = SourceFile::new("test.stark", src);
        let (tokens, diags) = tokenize(&file);
        let kinds = tokens.iter().map(|t| t.kind).collect();
        let msgs = diags.iter().map(|d| d.message.clone()).collect();
        (kinds, msgs)
    }

    fn kinds_ok(src: &str) -> Vec<TokenKind> {
        let (kinds, msgs) = lex(src);
        assert!(
            msgs.is_empty(),
            "unexpected diagnostics for {src:?}: {msgs:?}"
        );
        kinds
    }

    fn errors(src: &str) -> Vec<String> {
        let (_, msgs) = lex(src);
        assert!(!msgs.is_empty(), "expected diagnostics for {src:?}");
        msgs
    }

    use TokenKind::*;

    // --- names ------------------------------------------------------------

    #[test]
    fn keywords_reserved_and_idents() {
        assert_eq!(
            kinds_ok("fn Self self dyn async foo _bar Int32 str"),
            vec![
                Keyword(Kw::Fn),
                Keyword(Kw::SelfUpper),
                Keyword(Kw::SelfLower),
                Reserved,
                Reserved,
                Ident,
                Ident,
                Keyword(Kw::Int32),
                Keyword(Kw::Str),
                Eof
            ]
        );
    }

    #[test]
    fn all_reserved_words_lex_as_reserved() {
        // WP-C1.1: keywords_reserved_and_idents above only exercises 2 of the 15 entries in
        // RESERVED; this covers the full table so a future edit to RESERVED can't silently drop
        // a word without a lexer-level test noticing.
        for word in [
            "async", "await", "yield", "where", "macro", "unsafe", "extern", "import", "export",
            "null", "and", "or", "not", "is", "dyn",
        ] {
            assert_eq!(
                kinds_ok(word),
                vec![Reserved, Eof],
                "'{word}' should lex as Reserved"
            );
        }
    }

    #[test]
    fn ident_length_limit() {
        let long = "x".repeat(256);
        let msgs = errors(&long);
        assert!(msgs[0].contains("maximum length"));
        assert!(kinds_ok(&"x".repeat(255)).contains(&Ident));
    }

    #[test]
    fn r_is_only_special_before_quote() {
        assert_eq!(kinds_ok("radius"), vec![Ident, Eof]);
    }

    // --- numbers -----------------------------------------------------------

    #[test]
    fn spec_integer_examples() {
        // The exact examples from 01-Lexical-Grammar.md.
        let kinds = kinds_ok("42 1_000_000 0xFF_FF 0b1010_1010 0o755 42i32 255u8");
        assert_eq!(
            kinds,
            vec![
                Int {
                    base: Base::Dec,
                    suffix: None
                },
                Int {
                    base: Base::Dec,
                    suffix: None
                },
                Int {
                    base: Base::Hex,
                    suffix: None
                },
                Int {
                    base: Base::Bin,
                    suffix: None
                },
                Int {
                    base: Base::Oct,
                    suffix: None
                },
                Int {
                    base: Base::Dec,
                    suffix: Some(IntSuffix::I32)
                },
                Int {
                    base: Base::Dec,
                    suffix: Some(IntSuffix::U8)
                },
                Eof
            ]
        );
    }

    #[test]
    fn spec_float_examples() {
        let kinds = kinds_ok("3.14 1.0e10 2.5e-3 42.0f32");
        assert_eq!(
            kinds,
            vec![
                Float { suffix: None },
                Float { suffix: None },
                Float { suffix: None },
                Float {
                    suffix: Some(FloatSuffix::F32)
                },
                Eof
            ]
        );
    }

    #[test]
    fn underscore_rules() {
        // Spec: separators only between digits; 1__2 and 12_ are invalid.
        assert!(errors("1__2")[0].contains("Invalid number"));
        assert!(errors("12_")[0].contains("Invalid number"));
        assert!(errors("0x_FF")[0].contains("Invalid number"));
    }

    #[test]
    fn invalid_suffixes_and_bodies() {
        assert!(errors("42abc")[0].contains("Invalid number"));
        assert!(errors("42f32")[0].contains("Invalid number")); // f32 is float-only
        assert!(errors("3.14i32")[0].contains("Invalid number")); // i32 is int-only
        assert!(errors("0x")[0].contains("Invalid number"));
        assert!(errors("0b12")[0].contains("Invalid number"));
    }

    #[test]
    fn dot_after_int_is_not_a_float() {
        // Tuple access / method calls on literals must survive.
        assert_eq!(
            kinds_ok("pair.0 42.abs"),
            vec![
                Ident,
                Dot,
                Int {
                    base: Base::Dec,
                    suffix: None
                },
                Int {
                    base: Base::Dec,
                    suffix: None
                },
                Dot,
                Ident,
                Eof
            ]
        );
    }

    #[test]
    fn exponent_without_digits_is_not_an_exponent() {
        // `1e` has no exponent digits: 'e' becomes an invalid suffix.
        assert!(errors("1e")[0].contains("Invalid number"));
        assert_eq!(kinds_ok("1e5"), vec![Float { suffix: None }, Eof]);
    }

    // --- strings and chars ---------------------------------------------------

    #[test]
    fn spec_string_examples() {
        let kinds = kinds_ok(
            "\"Hello, World!\" \"Line 1\\nLine 2\" \"\\x41\\x42\\x43\" \"\\u{1F600}\" r\"Raw string with \\n literal backslashes\"",
        );
        assert_eq!(
            kinds,
            vec![
                Str { raw: false },
                Str { raw: false },
                Str { raw: false },
                Str { raw: false },
                Str { raw: true },
                Eof
            ]
        );
    }

    #[test]
    fn spec_char_examples() {
        assert_eq!(
            kinds_ok("'a' '\\n' '\\x41' '\\u{1F600}'"),
            vec![CharLit, CharLit, CharLit, CharLit, Eof]
        );
    }

    #[test]
    fn string_errors() {
        assert!(errors("\"unterminated")[0].contains("Unterminated string"));
        assert!(errors("r\"unterminated")[0].contains("Unterminated string"));
        assert!(errors("\"bad \\q escape\"")[0].contains("Invalid escape"));
        assert!(errors("\"\\u{110000}\"")[0].contains("scalar value"));
        assert!(errors("\"\\x4\"")[0].contains("two hex digits"));
    }

    #[test]
    fn char_errors() {
        assert!(errors("''")[0].contains("Empty character"));
        assert!(errors("'ab'")[0].contains("exactly one character"));
        assert!(errors("'a")[0].contains("exactly one character"));
    }

    #[test]
    fn multibyte_char_content() {
        assert_eq!(kinds_ok("'\u{1F600}'"), vec![CharLit, Eof]);
    }

    // --- comments -------------------------------------------------------------

    #[test]
    fn comments_are_skipped_including_nested() {
        assert_eq!(kinds_ok("a // line\nb"), vec![Ident, Ident, Eof]);
        assert_eq!(
            kinds_ok("a /* x /* nested */ y */ b"),
            vec![Ident, Ident, Eof]
        );
        assert_eq!(kinds_ok("/** doc */ fn"), vec![Keyword(Kw::Fn), Eof]);
        assert_eq!(kinds_ok("/**/ x"), vec![Ident, Eof]);
    }

    #[test]
    fn deeply_nested_block_comments() {
        // WP-C1.1: comments_are_skipped_including_nested above only exercises 2 nesting levels.
        assert_eq!(
            kinds_ok("a /* 1 /* 2 /* 3 /* 4 */ still 3 */ still 2 */ still 1 */ b"),
            vec![Ident, Ident, Eof]
        );
        // An unterminated comment nested 3 deep must still error, not silently close early.
        assert!(errors("/* 1 /* 2 /* 3 */ still 2")[0].contains("Unterminated block comment"));
    }

    #[test]
    fn unterminated_block_comment() {
        assert!(errors("/* /* */")[0].contains("Unterminated block comment"));
    }

    fn comments(src: &str) -> Vec<(CommentKind, &str)> {
        let file = SourceFile::new("test.stark", src);
        let (_, comments, _) = tokenize_with_comments(&file);
        comments
            .iter()
            .map(|c| (c.kind, &src[c.span.lo as usize..c.span.hi as usize]))
            .collect()
    }

    #[test]
    fn comments_are_collected_as_trivia() {
        assert_eq!(
            comments("a // line\nb"),
            vec![(CommentKind::Line, "// line")]
        );
        assert_eq!(
            comments("a /* x /* nested */ y */ b"),
            vec![(CommentKind::Block, "/* x /* nested */ y */")]
        );
    }

    #[test]
    fn doc_comment_kinds_are_distinguished() {
        assert_eq!(
            comments("/// outer\nfn f() {}"),
            vec![(CommentKind::LineDoc, "/// outer")]
        );
        assert_eq!(
            comments("//! inner"),
            vec![(CommentKind::LineInnerDoc, "//! inner")]
        );
        // Four or more slashes is a decorative line comment, not doc.
        assert_eq!(
            comments("//// banner"),
            vec![(CommentKind::Line, "//// banner")]
        );
    }

    #[test]
    fn token_stream_is_unaffected_by_comment_collection() {
        let file = SourceFile::new("test.stark", "a // line\nb /* c */ d");
        let (tokens, _) = tokenize(&file);
        let kinds: Vec<_> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(kinds, vec![Ident, Ident, Ident, Eof]);
    }

    // --- operators ---------------------------------------------------------------

    #[test]
    fn maximal_munch() {
        assert_eq!(
            kinds_ok("**= ** *= *"),
            vec![StarStarEq, StarStar, StarEq, Star, Eof]
        );
        assert_eq!(kinds_ok("<<= << <= <"), vec![ShlEq, Shl, LtEq, Lt, Eof]);
        assert_eq!(kinds_ok(">>= >> >= >"), vec![ShrEq, Shr, GtEq, Gt, Eof]);
        assert_eq!(kinds_ok("..= .. ."), vec![DotDotEq, DotDot, Dot, Eof]);
        assert_eq!(kinds_ok(":: :"), vec![ColonColon, Colon, Eof]);
        assert_eq!(kinds_ok("=> == ="), vec![FatArrow, EqEq, Eq, Eof]);
        assert_eq!(kinds_ok("-> -= -"), vec![Arrow, MinusEq, Minus, Eof]);
        assert_eq!(kinds_ok("&& &= &"), vec![AndAnd, AmpEq, Amp, Eof]);
        assert_eq!(kinds_ok("|| |= |"), vec![OrOr, PipeEq, Pipe, Eof]);
    }

    #[test]
    fn shr_is_one_token_for_parser_splitting() {
        // Vec<Vec<Int32>> — the parser splits the trailing Shr.
        let kinds = kinds_ok("Vec<Vec<Int32>>");
        assert_eq!(
            kinds,
            vec![Ident, Lt, Ident, Lt, Keyword(Kw::Int32), Shr, Eof]
        );
    }

    #[test]
    fn unexpected_character() {
        let msgs = errors("let x = #;");
        assert!(msgs[0].contains("Unexpected character '#'"));
        // Lexing continued past the error:
        let (kinds, _) = lex("let x = #;");
        assert!(kinds.contains(&Semi));
    }

    #[test]
    fn adjacent_tokens_without_spaces() {
        assert_eq!(
            kinds_ok("foo(a,b)->c[0]"),
            vec![
                Ident,
                LParen,
                Ident,
                Comma,
                Ident,
                RParen,
                Arrow,
                Ident,
                LBracket,
                Int {
                    base: Base::Dec,
                    suffix: None
                },
                RBracket,
                Eof
            ]
        );
    }

    // --- error recovery ------------------------------------------------------------

    #[test]
    fn lexing_continues_after_errors() {
        let (kinds, msgs) = lex("1__2 fn 0x \"ok\"");
        assert_eq!(msgs.len(), 2);
        assert!(kinds.contains(&Keyword(Kw::Fn)));
        assert!(kinds.contains(&Str { raw: false }));
    }
}
