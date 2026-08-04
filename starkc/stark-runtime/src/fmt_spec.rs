//! WP-FMT-001 — **the one implementation of format-specification rendering.**
//!
//! Alignment, fill, width, sign, radix, alternate prefix, zero-padding and fixed float precision
//! are defined here and nowhere else. The HIR interpreter, the MIR interpreter and generated
//! native code all call these functions, so a padding or rounding rule cannot be written three
//! times and drift twice. `starkc` already depends on this crate and every native binary links
//! it — the same arrangement that keeps `x.fmt()` and `println(x)` rendering identically.
//!
//! **Nothing here parses a format string.** A specification is validated and packed at compile
//! time into a [`Spec`] word plus a fill character; at run time there is only a bitfield to read.
//!
//! **Nothing here consults a locale.** Digits are ASCII, the decimal separator is `.`, and there
//! is no grouping. That is a language guarantee, not a default.

// ---------------------------------------------------------------- the packed specification --

/// Field offsets in the specification word. The compiler packs; this unpacks. Both sides read
/// these constants, so the layout is stated once.
mod bits {
    pub const WIDTH_SHIFT: u32 = 0;
    pub const WIDTH_MASK: u64 = 0xFFFF_FFFF;
    pub const PRECISION_SHIFT: u32 = 32;
    pub const PRECISION_MASK: u64 = 0xFFFF;
    pub const HAS_PRECISION: u64 = 1 << 48;
    pub const ALIGN_SHIFT: u32 = 49;
    pub const ALIGN_MASK: u64 = 0b11;
    pub const SIGN_SHIFT: u32 = 51;
    pub const SIGN_MASK: u64 = 0b11;
    pub const ALTERNATE: u64 = 1 << 53;
    pub const ZERO_PAD: u64 = 1 << 54;
    pub const KIND_SHIFT: u32 = 55;
    pub const KIND_MASK: u64 = 0b111;
}

/// How a rendered value sits inside a wider field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Align {
    /// No alignment written. Text and every non-numeric value default to left; a numeric value
    /// with a width defaults to right, which is what makes `{n:6}` right-align like `{n:>6}`.
    Default,
    Left,
    Right,
    Center,
}

/// What prefix a non-negative number carries.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sign {
    /// `-` (or unwritten): negatives get `-`, non-negatives get nothing.
    Minus,
    /// `+`: non-negatives get `+`.
    Plus,
    /// ` `: non-negatives get a space, so columns of mixed signs line up.
    Space,
}

/// The value-family conversion a specification asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// No type character — canonical `Display`.
    Display,
    Bin,
    Oct,
    LowerHex,
    UpperHex,
    /// `f` — fixed-point. Identical in meaning to a bare `.precision`.
    Fixed,
}

/// A decoded format specification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Spec {
    pub width: u32,
    pub precision: Option<u16>,
    pub align: Align,
    pub sign: Sign,
    pub alternate: bool,
    pub zero_pad: bool,
    pub kind: Kind,
    pub fill: char,
}

impl Spec {
    /// Decode the packed word. Unknown align/sign/kind encodings cannot occur — the compiler is
    /// the only producer and it packs from validated enums — so they decode to the default rather
    /// than panicking in a running program.
    pub fn unpack(word: u64, fill: char) -> Spec {
        let align = match (word >> bits::ALIGN_SHIFT) & bits::ALIGN_MASK {
            1 => Align::Left,
            2 => Align::Right,
            3 => Align::Center,
            _ => Align::Default,
        };
        let sign = match (word >> bits::SIGN_SHIFT) & bits::SIGN_MASK {
            1 => Sign::Plus,
            2 => Sign::Space,
            _ => Sign::Minus,
        };
        let kind = match (word >> bits::KIND_SHIFT) & bits::KIND_MASK {
            1 => Kind::Bin,
            2 => Kind::Oct,
            3 => Kind::LowerHex,
            4 => Kind::UpperHex,
            5 => Kind::Fixed,
            _ => Kind::Display,
        };
        Spec {
            width: ((word >> bits::WIDTH_SHIFT) & bits::WIDTH_MASK) as u32,
            precision: if word & bits::HAS_PRECISION != 0 {
                Some(((word >> bits::PRECISION_SHIFT) & bits::PRECISION_MASK) as u16)
            } else {
                None
            },
            align,
            sign,
            alternate: word & bits::ALTERNATE != 0,
            zero_pad: word & bits::ZERO_PAD != 0,
            kind,
            fill,
        }
    }

    /// Pack a specification into the word the compiler emits. Lives here, next to `unpack`, so the
    /// two cannot disagree about a bit position.
    #[allow(clippy::too_many_arguments)]
    pub fn pack(
        width: u32,
        precision: Option<u16>,
        align: Align,
        sign: Sign,
        alternate: bool,
        zero_pad: bool,
        kind: Kind,
    ) -> u64 {
        let mut word = (width as u64) << bits::WIDTH_SHIFT;
        if let Some(precision) = precision {
            word |= bits::HAS_PRECISION;
            word |= ((precision as u64) & bits::PRECISION_MASK) << bits::PRECISION_SHIFT;
        }
        let align_bits = match align {
            Align::Default => 0u64,
            Align::Left => 1,
            Align::Right => 2,
            Align::Center => 3,
        };
        word |= align_bits << bits::ALIGN_SHIFT;
        let sign_bits = match sign {
            Sign::Minus => 0u64,
            Sign::Plus => 1,
            Sign::Space => 2,
        };
        word |= sign_bits << bits::SIGN_SHIFT;
        if alternate {
            word |= bits::ALTERNATE;
        }
        if zero_pad {
            word |= bits::ZERO_PAD;
        }
        let kind_bits = match kind {
            Kind::Display => 0u64,
            Kind::Bin => 1,
            Kind::Oct => 2,
            Kind::LowerHex => 3,
            Kind::UpperHex => 4,
            Kind::Fixed => 5,
        };
        word |= kind_bits << bits::KIND_SHIFT;
        word
    }
}

// ------------------------------------------------------------------------------- padding --

/// The number of Unicode scalar values in `text`.
///
/// **Width is measured in scalars, not bytes and not terminal cells.** A multi-byte character
/// counts once; a combining sequence counts once per scalar; East Asian wide characters are not
/// modelled. The rule is chosen for determinism: the same program pads identically on every
/// platform, which terminal-cell width cannot promise.
pub fn scalar_len(text: &str) -> usize {
    text.chars().count()
}

/// Place `text` in a field of `spec.width`, filling with `spec.fill`.
///
/// **Width never truncates.** A value already at least as wide as the field is returned unchanged —
/// silently cutting a value would turn a display concern into data loss.
///
/// When centring needs an odd number of fill characters, the extra one goes on the RIGHT.
pub fn pad(text: &str, spec: &Spec, default_align: Align) -> String {
    let len = scalar_len(text);
    let width = spec.width as usize;
    if len >= width {
        return text.to_string();
    }
    let total = width - len;
    let align = match spec.align {
        Align::Default => default_align,
        explicit => explicit,
    };
    let (left, right) = match align {
        Align::Left | Align::Default => (0, total),
        Align::Right => (total, 0),
        // The extra fill character goes right, so `{"x":^4}` is `| x  |`.
        Align::Center => (total / 2, total - total / 2),
    };
    let mut out = String::new();
    for _ in 0..left {
        out.push(spec.fill);
    }
    out.push_str(text);
    for _ in 0..right {
        out.push(spec.fill);
    }
    out
}

/// Zero-padding for a number: the fill goes AFTER the sign and any `0x`-style prefix, so the sign
/// stays at the left edge where a reader expects it (`-00042`, never `00-42`).
fn pad_numeric(sign: &str, prefix: &str, digits: &str, spec: &Spec) -> String {
    let body_len = scalar_len(sign) + scalar_len(prefix) + scalar_len(digits);
    let width = spec.width as usize;
    if !spec.zero_pad || body_len >= width {
        let joined = format_join(sign, prefix, digits);
        // A zero-padded field that is already wide enough, or one that never asked for zero
        // padding, still obeys alignment. Numbers default to RIGHT alignment in a wider field.
        return pad(&joined, spec, Align::Right);
    }
    let mut out = String::new();
    out.push_str(sign);
    out.push_str(prefix);
    for _ in 0..(width - body_len) {
        out.push('0');
    }
    out.push_str(digits);
    out
}

fn format_join(sign: &str, prefix: &str, digits: &str) -> String {
    let mut joined = String::with_capacity(sign.len() + prefix.len() + digits.len());
    joined.push_str(sign);
    joined.push_str(prefix);
    joined.push_str(digits);
    joined
}

// ------------------------------------------------------------------------------ integers --

fn sign_text(negative: bool, spec: &Spec) -> &'static str {
    if negative {
        "-"
    } else {
        match spec.sign {
            Sign::Plus => "+",
            Sign::Space => " ",
            Sign::Minus => "",
        }
    }
}

fn prefix_text(spec: &Spec) -> &'static str {
    if !spec.alternate {
        return "";
    }
    match spec.kind {
        Kind::Bin => "0b",
        Kind::Oct => "0o",
        // `0x` for BOTH hex cases: the prefix names the base, the type character chooses the
        // digit case. `0XFF` would be claiming the prefix is data.
        Kind::LowerHex | Kind::UpperHex => "0x",
        Kind::Display | Kind::Fixed => "",
    }
}

/// The magnitude of `value` in the requested base, ASCII digits, no sign and no prefix.
fn digits_of(magnitude: u128, spec: &Spec) -> String {
    let (radix, upper) = match spec.kind {
        Kind::Bin => (2u128, false),
        Kind::Oct => (8, false),
        Kind::LowerHex => (16, false),
        Kind::UpperHex => (16, true),
        Kind::Display | Kind::Fixed => (10, false),
    };
    if magnitude == 0 {
        return "0".to_string();
    }
    let mut buf: Vec<u8> = Vec::new();
    let mut n = magnitude;
    while n > 0 {
        let digit = (n % radix) as u8;
        buf.push(match digit {
            0..=9 => b'0' + digit,
            _ if upper => b'A' + (digit - 10),
            _ => b'a' + (digit - 10),
        });
        n /= radix;
    }
    buf.reverse();
    String::from_utf8(buf).unwrap_or_default()
}

/// A signed integer under `spec`.
///
/// The magnitude is taken in `u128` before any negation, so the minimum value of any width — whose
/// magnitude does not fit its own type — is rendered exactly rather than overflowing.
/// `Int64::MIN` formats as `-9223372036854775808`, not as a trap.
///
/// A negative value in a non-decimal base keeps its `-` and renders the MAGNITUDE in that base:
/// `-255` in hex is `-ff`. The host's two's-complement bit pattern is never exposed — that would
/// leak a representation STARK does not define.
pub fn fmt_int_spec(value: i64, word: u64, fill: char) -> String {
    let spec = Spec::unpack(word, fill);
    let negative = value < 0;
    let magnitude = (value as i128).unsigned_abs();
    let digits = digits_of(magnitude, &spec);
    pad_numeric(
        sign_text(negative, &spec),
        prefix_text(&spec),
        &digits,
        &spec,
    )
}

/// An unsigned integer under `spec`. Never negative, so `+`/space still apply and `-` never does.
pub fn fmt_uint_spec(value: u64, word: u64, fill: char) -> String {
    let spec = Spec::unpack(word, fill);
    let digits = digits_of(value as u128, &spec);
    pad_numeric(sign_text(false, &spec), prefix_text(&spec), &digits, &spec)
}

// -------------------------------------------------------------------------------- floats --

/// Fixed-point rendering of `value` to `precision` digits after the point.
///
/// **Rounding is round-half-to-even.** Exact halfway cases go to the even last digit, so a column
/// of rounded values does not drift upward the way round-half-away-from-zero does. The comparison
/// is made on the exact decimal expansion of the binary value, not on a re-parsed approximation.
fn fixed_digits(value: f64, precision: u16) -> String {
    // `f64`'s exact decimal expansion is finite, and Rust's `{:.*}` formatting of a finite `f64`
    // is specified to round half-to-even on that exact value. This is the one place the shared
    // core leans on the host's decimal conversion; it is a NUMERIC conversion, not trait
    // dispatch, and it is the same call in all three engines because they all reach this function.
    format!("{:.*}", precision as usize, value)
}

fn non_finite(value: f64) -> Option<&'static str> {
    if value.is_nan() {
        // Every NaN renders as `NaN`; sign and payload are not observable in STARK.
        return Some("NaN");
    }
    if value == f64::INFINITY {
        return Some("inf");
    }
    if value == f64::NEG_INFINITY {
        return Some("-inf");
    }
    None
}

/// A `Float64` under `spec`.
///
/// A non-finite value ignores precision entirely: `NaN` with `.2` is `NaN`, never `NaN.00`.
/// Padding still applies, so a column stays aligned.
pub fn fmt_float64_spec(value: f64, word: u64, fill: char) -> String {
    let spec = Spec::unpack(word, fill);
    if let Some(text) = non_finite(value) {
        return pad(text, &spec, Align::Right);
    }
    let negative = value.is_sign_negative();
    let rendered = match spec.precision {
        Some(precision) => fixed_digits(value.abs(), precision),
        None => crate::format::canonical_float(value.abs()),
    };
    pad_numeric(sign_text(negative, &spec), "", &rendered, &spec)
}

/// A `Float32` under `spec`.
///
/// The value is narrowed to `f32` before rendering, so precision and canonical digits describe the
/// `Float32` that was written — not the `f64` it would become if widened first (DEV-105).
pub fn fmt_float32_spec(value: f32, word: u64, fill: char) -> String {
    let spec = Spec::unpack(word, fill);
    if let Some(text) = non_finite(value as f64) {
        return pad(text, &spec, Align::Right);
    }
    let negative = value.is_sign_negative();
    let rendered = match spec.precision {
        Some(precision) => fixed_digits(value.abs() as f64, precision),
        None => crate::format::canonical_float32(value.abs()),
    };
    pad_numeric(sign_text(negative, &spec), "", &rendered, &spec)
}

// -------------------------------------------------------------------------------- strings --

/// Already-rendered text under `spec` — the padding-only path, used for `Display` output and for
/// every non-numeric value. Text defaults to LEFT alignment in a wider field.
pub fn fmt_pad_spec(text: &str, word: u64, fill: char) -> String {
    let spec = Spec::unpack(word, fill);
    pad(text, &spec, Align::Left)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(width: u32, align: Align, _fill: char) -> u64 {
        Spec::pack(width, None, align, Sign::Minus, false, false, Kind::Display)
    }

    #[test]
    fn pack_and_unpack_round_trip() {
        let word = Spec::pack(
            12,
            Some(3),
            Align::Center,
            Sign::Plus,
            true,
            true,
            Kind::UpperHex,
        );
        let decoded = Spec::unpack(word, '.');
        assert_eq!(decoded.width, 12);
        assert_eq!(decoded.precision, Some(3));
        assert_eq!(decoded.align, Align::Center);
        assert_eq!(decoded.sign, Sign::Plus);
        assert!(decoded.alternate);
        assert!(decoded.zero_pad);
        assert_eq!(decoded.kind, Kind::UpperHex);
        assert_eq!(decoded.fill, '.');
    }

    #[test]
    fn centring_puts_the_extra_fill_on_the_right() {
        assert_eq!(fmt_pad_spec("x", spec(4, Align::Center, ' '), ' '), " x  ");
    }

    #[test]
    fn width_never_truncates() {
        assert_eq!(
            fmt_pad_spec("abcdef", spec(3, Align::Left, ' '), ' '),
            "abcdef"
        );
    }

    #[test]
    fn width_counts_scalars_not_bytes() {
        // Four scalars, ten UTF-8 bytes. A byte-counting implementation would not pad at all.
        assert_eq!(scalar_len("日本語だ"), 4);
        assert_eq!(
            fmt_pad_spec("日本語だ", spec(6, Align::Left, '.'), '.'),
            "日本語だ.."
        );
    }

    #[test]
    fn the_sign_precedes_zero_padding() {
        let word = Spec::pack(
            6,
            None,
            Align::Default,
            Sign::Minus,
            false,
            true,
            Kind::Display,
        );
        assert_eq!(fmt_int_spec(-42, word, '0'), "-00042");
        assert_eq!(fmt_int_spec(42, word, '0'), "000042");
    }

    #[test]
    fn minimum_signed_values_do_not_overflow_their_own_width() {
        let word = Spec::pack(
            0,
            None,
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::Display,
        );
        assert_eq!(fmt_int_spec(i64::MIN, word, ' '), "-9223372036854775808");
    }

    #[test]
    fn a_negative_value_in_another_base_keeps_its_sign() {
        let word = Spec::pack(
            0,
            None,
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::LowerHex,
        );
        assert_eq!(fmt_int_spec(-255, word, ' '), "-ff");
    }

    #[test]
    fn alternate_prefix_precedes_zero_padding() {
        let word = Spec::pack(
            10,
            None,
            Align::Default,
            Sign::Minus,
            true,
            true,
            Kind::LowerHex,
        );
        assert_eq!(fmt_uint_spec(255, word, '0'), "0x000000ff");
    }

    #[test]
    fn uppercase_hex_keeps_a_lowercase_prefix() {
        let word = Spec::pack(
            0,
            None,
            Align::Default,
            Sign::Minus,
            true,
            false,
            Kind::UpperHex,
        );
        assert_eq!(fmt_uint_spec(255, word, ' '), "0xFF");
    }

    #[test]
    fn precision_rounds_half_to_even() {
        let two = |v: f64| {
            let word = Spec::pack(
                0,
                Some(2),
                Align::Default,
                Sign::Minus,
                false,
                false,
                Kind::Display,
            );
            fmt_float64_spec(v, word, ' ')
        };
        assert_eq!(two(0.75623), "0.76");
        assert_eq!(two(1.0), "1.00");
        // Exact halves in binary: .125 -> .12 (2 is even), .375 -> .38 (8 is even).
        assert_eq!(two(0.125), "0.12");
        assert_eq!(two(0.375), "0.38");
    }

    #[test]
    fn zero_precision_drops_the_point() {
        let word = Spec::pack(
            0,
            Some(0),
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::Display,
        );
        assert_eq!(fmt_float64_spec(1.0, word, ' '), "1");
    }

    #[test]
    fn negative_zero_keeps_its_sign() {
        let word = Spec::pack(
            0,
            Some(2),
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::Display,
        );
        assert_eq!(fmt_float64_spec(-0.0, word, ' '), "-0.00");
    }

    #[test]
    fn non_finite_values_ignore_precision() {
        let word = Spec::pack(
            0,
            Some(2),
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::Display,
        );
        assert_eq!(fmt_float64_spec(f64::NAN, word, ' '), "NaN");
        assert_eq!(fmt_float64_spec(f64::INFINITY, word, ' '), "inf");
        assert_eq!(fmt_float64_spec(f64::NEG_INFINITY, word, ' '), "-inf");
    }

    #[test]
    fn float32_precision_describes_the_declared_width() {
        let word = Spec::pack(
            0,
            Some(3),
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::Display,
        );
        // 0.1f32 widened to f64 is 0.10000000149011612; at the declared width it is 0.100.
        assert_eq!(fmt_float32_spec(0.1f32, word, ' '), "0.100");
    }

    #[test]
    fn numbers_default_to_right_alignment_in_a_wider_field() {
        let word = Spec::pack(
            6,
            None,
            Align::Default,
            Sign::Minus,
            false,
            false,
            Kind::Display,
        );
        assert_eq!(fmt_int_spec(42, word, ' '), "    42");
    }

    #[test]
    fn text_defaults_to_left_alignment_in_a_wider_field() {
        assert_eq!(
            fmt_pad_spec("stark", spec(8, Align::Default, ' '), ' '),
            "stark   "
        );
    }
}
