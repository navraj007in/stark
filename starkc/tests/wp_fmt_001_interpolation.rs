//! **WP-FMT-001** — interpolated string literals, `f"..."`.
//!
//! Every positive case runs through the shared comparator (HIR oracle, MIR interpreter, native
//! binary) with stdout pinned in the test itself, never taken from an engine — three engines
//! agreeing on the wrong text still fails.
//!
//! What the cases are for, beyond "it prints something":
//!
//! * **`the_work_package_example`** is §1's program and its exact expected output. If nothing else
//!   in this file survives, that one sentence is the feature.
//! * **Ownership** (`a_field_borrows_*`) — a field BORROWS. `Display::fmt` is `&self`
//!   (STD-FORMAT-001), so interpolating a value must leave it usable, including a non-`Copy` and an
//!   affine one. These fail if lowering ever moves a field.
//! * **Evaluation** (`fields_evaluate_*`) — exactly once, left to right. A width or type is never
//!   discovered by an extra evaluation.
//! * **Parsing** (`nested_delimiters_*`) — a struct literal's `{`/`:` and a path's `::` are inside
//!   the expression, not field syntax. `\u{...}`'s braces are an escape, not a field.
//! * **Numeric edges** — `Int::MIN` in its own width, a negative value in another base, the sign
//!   ahead of zero-padding, an alternate prefix ahead of it too.
//! * **Backend** (`native_routes_through_stark_formatting`) — generated application code must not
//!   reach Rust's `format!`/`Display`/`write!` to implement interpolation.

mod support;

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions, Profile};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use support::differential::{
    agree_completing_with_stdout, front_end, rejects_at_typecheck, rustc_available,
};

/// A non-`Copy` nominal with its own `Display`, written with interpolation — so the suite also
/// covers an interpolation nested inside a `Display::fmt` body.
const POINT: &str = "\
struct Point {
    label: String,
    x: Int32,
    y: Int32,
}

impl Display for Point {
    fn fmt(&self) -> String {
        f\"{self.label}({self.x}, {self.y})\"
    }
}
";

// ------------------------------------------------------------------------ the headline example --

#[test]
fn the_work_package_example() {
    agree_completing_with_stdout(
        "fmt_headline",
        "\
fn main() {
    let name: String = \"stark\".to_string();
    let count: Int32 = 42;
    let ratio: Float64 = 0.75623;
    let ok: Bool = true;

    let message = f\"pkg={name} n={count:04} r={ratio:.2} ok={ok}\";

    println(message.as_str());
}
",
        "pkg=stark n=0042 r=0.76 ok=true\n",
    );
}

// ------------------------------------------------------------------------------ basic surface --

#[test]
fn literals_and_simple_fields() {
    agree_completing_with_stdout(
        "fmt_basic",
        "\
fn main() {
    let name: String = \"STARK\".to_string();
    let version: Int32 = 1;
    println(f\"plain\".as_str());
    println(f\"{name} version {version}\".as_str());
    println(f\"a{version}b\".as_str());
}
",
        "plain\nSTARK version 1\na1b\n",
    );
}

#[test]
fn escaped_braces_render_as_braces() {
    agree_completing_with_stdout(
        "fmt_escaped_braces",
        "\
fn main() {
    let name: String = \"stark\".to_string();
    println(f\"object={{ name: {name} }}\".as_str());
    println(f\"{{\".as_str());
    println(f\"}}\".as_str());
    println(f\"{{{name}}}\".as_str());
}
",
        "object={ name: stark }\n{\n}\n{stark}\n",
    );
}

/// `\u{...}` contains braces. A scanner that looked at braces before escapes would read `{1F600}`
/// as an interpolation field.
#[test]
fn a_unicode_escape_is_not_a_field() {
    agree_completing_with_stdout(
        "fmt_unicode_escape",
        "\
fn main() {
    println(f\"emoji \\u{1F600} done\".as_str());
}
",
        "emoji \u{1F600} done\n",
    );
}

#[test]
fn expressions_of_every_supported_shape() {
    let source = String::from(POINT)
        + "
fn area(w: Int32, h: Int32) -> Int32 {
    w * h
}

fn main() {
    let width: Int32 = 10;
    let height: Int32 = 20;
    let count: Int32 = 41;
    let items: Vec<Int32> = Vec::new();
    let p = Point { label: \"p\".to_string(), x: 1, y: 2 };
    println(f\"area={width * height}\".as_str());
    println(f\"next={count + 1}\".as_str());
    println(f\"call={area(2, 3)}\".as_str());
    println(f\"field={p.x}\".as_str());
    println(f\"method={items.len()}\".as_str());
    println(f\"wrapped={(width + height) * 2}\".as_str());
    let q: String = \"q\".to_string();
    println(f\"struct={Point { label: q, x: 3, y: 4 }}\".as_str());
}
";
    agree_completing_with_stdout(
        "fmt_expressions",
        &source,
        "area=200\nnext=42\ncall=6\nfield=1\nmethod=0\nwrapped=60\nstruct=q(3, 4)\n",
    );
}

#[test]
fn a_user_display_impl_is_used() {
    let source = String::from(POINT)
        + "
fn main() {
    let p = Point { label: \"P\".to_string(), x: 1, y: 2 };
    println(f\"point={p}\".as_str());
}
";
    agree_completing_with_stdout("fmt_user_display", &source, "point=P(1, 2)\n");
}

#[test]
fn a_generic_display_parameter_interpolates() {
    let source = String::from(POINT)
        + "
fn render<T: Display>(value: &T) -> String {
    f\"value={value}\"
}

fn main() {
    let p = Point { label: \"g\".to_string(), x: 5, y: 6 };
    println(render(&p).as_str());
    let n: Int32 = 7;
    println(render(&n).as_str());
}
";
    agree_completing_with_stdout("fmt_generic", &source, "value=g(5, 6)\nvalue=7\n");
}

// ---------------------------------------------------------------------- ownership and evaluation --

/// §4's program: an owned generic parameter interpolated twice.
#[test]
fn a_field_borrows_an_owned_parameter() {
    agree_completing_with_stdout(
        "fmt_borrow_owned",
        "\
fn twice<T: Display>(value: T) -> String {
    let first = f\"{value}\";
    let second = f\"{value}\";

    let mut out = String::new();
    out.push_str(first.as_str());
    out.push_str(\"/\");
    out.push_str(second.as_str());
    out
}

fn main() {
    println(twice(7).as_str());
}
",
        "7/7\n",
    );
}

/// A non-`Copy` value interpolated twice and then used again.
#[test]
fn a_field_borrows_a_non_copy_value() {
    let source = String::from(POINT)
        + "
fn main() {
    let p = Point { label: \"n\".to_string(), x: 1, y: 1 };
    println(f\"a={p}\".as_str());
    println(f\"b={p}\".as_str());
    println(p.fmt().as_str());
}
";
    agree_completing_with_stdout(
        "fmt_borrow_non_copy",
        &source,
        "a=n(1, 1)\nb=n(1, 1)\nn(1, 1)\n",
    );
}

/// An AFFINE value — a `Drop`-bearing nominal, which Core v1 forbids from also being `Copy` — is
/// interpolated and then consumed. The `released` line lands after the interpolations, which is
/// what shows the value was still alive while being formatted.
#[test]
fn a_field_borrows_an_affine_value() {
    agree_completing_with_stdout(
        "fmt_borrow_affine",
        "\
struct Handle {
    id: Int32,
}

impl Drop for Handle {
    fn drop(&mut self) {
        println(\"released\");
    }
}

impl Display for Handle {
    fn fmt(&self) -> String {
        f\"Handle#{self.id}\"
    }
}

fn consume(h: Handle) -> Int32 {
    h.id
}

fn main() {
    let h = Handle { id: 5 };
    println(f\"first={h}\".as_str());
    println(f\"second={h}\".as_str());
    let id = consume(h);
    println(f\"id={id}\".as_str());
}
",
        "first=Handle#5\nsecond=Handle#5\nreleased\nid=5\n",
    );
}

/// Side effects prove BOTH properties at once: each field runs exactly once, and the fields run
/// left to right. A width discovered by a second evaluation would double a line.
#[test]
fn fields_evaluate_once_and_left_to_right() {
    agree_completing_with_stdout(
        "fmt_evaluation_order",
        "\
fn first() -> Int32 {
    println(\"eval first\");
    1
}

fn second() -> Int32 {
    println(\"eval second\");
    2
}

fn third() -> Int32 {
    println(\"eval third\");
    3
}

fn main() {
    let text = f\"{first()} {second():>4} {third():04}\";
    println(text.as_str());
}
",
        "eval first\neval second\neval third\n1    2 0003\n",
    );
}

// -------------------------------------------------------------------------- width and alignment --

#[test]
fn alignment_and_fill() {
    agree_completing_with_stdout(
        "fmt_alignment",
        "\
fn main() {
    let name: String = \"stark\".to_string();
    println(f\"|{name:<12}|\".as_str());
    println(f\"|{name:>12}|\".as_str());
    println(f\"|{name:^12}|\".as_str());
    println(f\"{name:.^12}\".as_str());
    println(f\"|{name:<3}|\".as_str());
}
",
        "|stark       |\n|       stark|\n|   stark    |\n...stark....\n|stark|\n",
    );
}

/// §5.3's rule: an odd number of fill characters puts the extra one on the right.
#[test]
fn odd_centring_puts_the_extra_fill_on_the_right() {
    agree_completing_with_stdout(
        "fmt_odd_centre",
        "\
fn main() {
    let x: String = \"x\".to_string();
    println(f\"|{x:^4}|\".as_str());
}
",
        "| x  |\n",
    );
}

/// Width counts Unicode scalars, not UTF-8 bytes. Four scalars, ten bytes: a byte-counting
/// implementation would not pad at all.
#[test]
fn width_counts_scalars_not_bytes() {
    agree_completing_with_stdout(
        "fmt_scalar_width",
        "\
fn main() {
    let text: String = \"日本語だ\".to_string();
    println(f\"|{text:<6}|\".as_str());
}
",
        "|日本語だ  |\n",
    );
}

// ------------------------------------------------------------------------------ integer modes --

#[test]
fn zero_padding_and_signs() {
    agree_completing_with_stdout(
        "fmt_int_padding",
        "\
fn main() {
    let n: Int32 = 42;
    let m: Int32 = -42;
    println(f\"{n:06}\".as_str());
    println(f\"{m:06}\".as_str());
    println(f\"{n:0>6}\".as_str());
    println(f\"{n:+}\".as_str());
    println(f\"{m:+}\".as_str());
    println(f\"[{n: }]\".as_str());
    println(f\"[{m: }]\".as_str());
    println(f\"{n:-}\".as_str());
}
",
        "000042\n-00042\n000042\n+42\n-42\n[ 42]\n[-42]\n42\n",
    );
}

#[test]
fn integer_bases_and_prefixes() {
    agree_completing_with_stdout(
        "fmt_int_bases",
        "\
fn main() {
    let flags: UInt32 = 255;
    println(f\"{flags:b}\".as_str());
    println(f\"{flags:o}\".as_str());
    println(f\"{flags:x}\".as_str());
    println(f\"{flags:X}\".as_str());
    println(f\"{flags:#b}\".as_str());
    println(f\"{flags:#o}\".as_str());
    println(f\"{flags:#x}\".as_str());
    println(f\"{flags:#X}\".as_str());
    println(f\"{flags:#010x}\".as_str());
}
",
        "11111111\n377\nff\nFF\n0b11111111\n0o377\n0xff\n0xFF\n0x000000ff\n",
    );
}

/// A negative value keeps its sign and renders its MAGNITUDE in the chosen base — the host's
/// two's-complement bit pattern is never exposed.
#[test]
fn a_negative_value_in_another_base_keeps_its_sign() {
    agree_completing_with_stdout(
        "fmt_negative_base",
        "\
fn main() {
    let value: Int32 = -255;
    println(f\"{value:x}\".as_str());
    println(f\"{value:#x}\".as_str());
    println(f\"{value:b}\".as_str());
}
",
        "-ff\n-0xff\n-11111111\n",
    );
}

/// Large-magnitude negative values, in every mode where the sign and the digits interact.
///
/// **The true minimum of each signed width cannot be written in STARK at all** — `let a: Int8 =
/// -128;` is rejected with E0008 because the magnitude is range-checked before the unary minus is
/// applied. That is a pre-existing language defect, recorded as DEV-172 and unrelated to
/// formatting. The renderer's own handling of `i64::MIN` — where taking the magnitude in-width
/// would overflow — is pinned directly in
/// `stark_runtime::fmt_spec::tests::minimum_signed_values_do_not_overflow_their_own_width`.
#[test]
fn large_negative_values_render_exactly() {
    agree_completing_with_stdout(
        "fmt_int_min",
        "\
fn main() {
    let a: Int8 = -127;
    let b: Int16 = -32767;
    let c: Int32 = -2147483647;
    let d: Int64 = -9223372036854775807;
    println(f\"{a:+} {b:+} {c:+}\".as_str());
    println(f\"{d:+}\".as_str());
    println(f\"{d:x}\".as_str());
}
",
        "-127 -32767 -2147483647\n-9223372036854775807\n-7fffffffffffffff\n",
    );
}

#[test]
fn every_integer_width_formats() {
    agree_completing_with_stdout(
        "fmt_int_widths",
        "\
fn main() {
    let i8v: Int8 = -8;
    let i16v: Int16 = -16;
    let i32v: Int32 = -32;
    let i64v: Int64 = -64;
    let u8v: UInt8 = 255;
    let u16v: UInt16 = 65535;
    let u32v: UInt32 = 4294967295;
    let u64v: UInt64 = 9223372036854775807;
    println(f\"{i8v:04} {i16v:06} {i32v:06} {i64v:06}\".as_str());
    println(f\"{u8v:x} {u16v:x} {u32v:x} {u64v:x}\".as_str());
}
",
        "-008 -00016 -00032 -00064\nff ffff ffffffff 7fffffffffffffff\n",
    );
}

// ---------------------------------------------------------------------------- float precision --

#[test]
fn fixed_precision() {
    agree_completing_with_stdout(
        "fmt_float_precision",
        "\
fn main() {
    let ratio: Float64 = 0.75623;
    println(f\"{ratio:.2}\".as_str());
    println(f\"{ratio:.4}\".as_str());
    println(f\"{ratio:.2f}\".as_str());
    println(f\"{1.0:.0}\".as_str());
    println(f\"{1.0:.2}\".as_str());
}
",
        "0.76\n0.7562\n0.76\n1\n1.00\n",
    );
}

/// Rounding is round-half-to-even. `0.125` and `0.375` are exact halves in binary, so they are the
/// cases that distinguish the rule from round-half-away-from-zero.
#[test]
fn rounding_is_half_to_even() {
    agree_completing_with_stdout(
        "fmt_float_rounding",
        "\
fn main() {
    let a: Float64 = 0.125;
    let b: Float64 = 0.375;
    println(f\"{a:.2} {b:.2}\".as_str());
}
",
        "0.12 0.38\n",
    );
}

#[test]
fn negative_zero_and_non_finite_values() {
    agree_completing_with_stdout(
        "fmt_float_edges",
        "\
fn main() {
    let nz: Float64 = -0.0;
    let zero: Float64 = 0.0;
    let nan: Float64 = zero / zero;
    let inf: Float64 = 1.0 / zero;
    let neg_inf: Float64 = -1.0 / zero;
    println(f\"{nz:.2}\".as_str());
    println(f\"{nan:.2}\".as_str());
    println(f\"{inf:.2}\".as_str());
    println(f\"{neg_inf:.2}\".as_str());
    println(f\"|{nan:>6}|\".as_str());
}
",
        "-0.00\nNaN\ninf\n-inf\n|   NaN|\n",
    );
}

/// A `Float32`'s precision describes the DECLARED width. Widening to `f64` first would print
/// `0.100000001…`'s digits instead.
#[test]
fn float32_precision_preserves_the_declared_width() {
    agree_completing_with_stdout(
        "fmt_float32_width",
        "\
fn main() {
    let v: Float32 = 0.1f32;
    println(f\"{v:.3}\".as_str());
    println(f\"{v:.8}\".as_str());
    println(f\"{v}\".as_str());
}
",
        "0.100\n0.10000000\n0.1\n",
    );
}

#[test]
fn width_and_sign_combine_with_precision() {
    agree_completing_with_stdout(
        "fmt_float_combined",
        "\
fn main() {
    let v: Float64 = 3.14159;
    println(f\"|{v:>10.2}|\".as_str());
    println(f\"|{v:<10.2}|\".as_str());
    println(f\"{v:+.3}\".as_str());
    println(f\"{v:08.3}\".as_str());
}
",
        "|      3.14|\n|3.14      |\n+3.142\n0003.142\n",
    );
}

// -------------------------------------------------------------------- bool, unit, char, strings --

#[test]
fn bool_unit_char_and_strings() {
    agree_completing_with_stdout(
        "fmt_other_types",
        "\
fn main() {
    let letter: Char = 'q';
    let text: String = \"hi\".to_string();
    let other: String = \"slice\".to_string();
    println(f\"{true} {false:>8}\".as_str());
    println(f\"|{letter:>4}|\".as_str());
    println(f\"{text} {other}\".as_str());
    println(f\"|{():^6}|\".as_str());
}
",
        "true    false\n|   q|\nhi slice\n|  ()  |\n",
    );
}

// -------------------------------------------------------------- print family and stark-fmt --

/// **The advertised form, proved directly.** `println(f"...")` with no `.as_str()` — the shape
/// §15 asks for. It works because the interpolation's `String` is itself `Display`, but that is a
/// claim worth pinning rather than assuming, and it must not move, double-destroy or lose output.
#[test]
fn the_output_family_accepts_an_interpolated_temporary_directly() {
    let done = support::differential::agree_completing_available_engines(
        "fmt_direct_println",
        "\
fn main() {
    let name: String = \"stark\".to_string();
    let count: Int32 = 7;
    println(f\"direct={name}/{count:03}\");
    print(f\"a\");
    println(f\"b\");
    eprintln(f\"err={count}\");
}
",
    );
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "direct=stark/007\nab\n",
        "stdout"
    );
    assert_eq!(
        String::from_utf8_lossy(&done.stderr_bytes),
        "err=7\n",
        "stderr"
    );
}

/// Comments are ordinary expression syntax, so a field admits them — and a `}` or `:` inside one
/// is text, not field structure. Block comments nest (01-Lexical-Grammar §6).
#[test]
fn a_field_admits_comments_including_braces_and_colons() {
    agree_completing_with_stdout(
        "fmt_field_comments",
        "\
fn twice(v: Int32) -> Int32 {
    v * 2
}

fn main() {
    let value: Int32 = 5;
    println(f\"a={value /* } */}\".as_str());
    println(f\"b={value /* : */}\".as_str());
    println(f\"c={twice(/* { : } */ value)}\".as_str());
    println(f\"d={value /* outer /* } */ still outer */}\".as_str());
}
",
        "a=5\nb=5\nc=10\nd=5\n",
    );
}

/// A flag that cannot affect the output is refused, not ignored — LEX-FORMAT-003.
#[test]
fn inert_specification_flags_are_rejected() {
    for (tag, source) in [
        (
            "fmt_alt_no_base",
            "fn main() { let v: Int32 = 42; let _t = f\"{v:#}\"; }",
        ),
        (
            "fmt_alt_on_float",
            "fn main() { let v: Float64 = 1.25; let _t = f\"{v:#.2f}\"; }",
        ),
        (
            "fmt_zero_no_width",
            "fn main() { let v: Int32 = 42; let _t = f\"{v:0}\"; }",
        ),
        (
            "fmt_fixed_no_precision",
            "fn main() { let v: Float64 = 1.25; let _t = f\"{v:f}\"; }",
        ),
        (
            "fmt_zero_with_align",
            "fn main() { let v: Int32 = 42; let _t = f\"{v:<06}\"; }",
        ),
        (
            "fmt_zero_with_fill",
            "fn main() { let v: Int32 = 42; let _t = f\"{v:.>06}\"; }",
        ),
    ] {
        let file = std::sync::Arc::new(starkc::source::SourceFile::new(tag, source.to_string()));
        let (_ast, diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
        assert!(
            diags.iter().any(|d| d.code.as_deref() == Some("E0218")),
            "{tag}: expected an E0218 rejection, got {diags:?}"
        );
    }
}

#[test]
fn the_output_family_accepts_interpolated_strings() {
    let done = support::differential::agree_completing_available_engines(
        "fmt_output_family",
        "\
fn main() {
    let name: String = \"stark\".to_string();
    let count: Int32 = 2;
    print(f\"a={name} \".as_str());
    println(f\"b={count}\".as_str());
    eprint(f\"warn={name} \".as_str());
    eprintln(f\"err={count}\".as_str());
}
",
    );
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "a=stark b=2\n",
        "stdout"
    );
    assert_eq!(
        String::from_utf8_lossy(&done.stderr_bytes),
        "warn=stark err=2\n",
        "stderr"
    );
}

// ---------------------------------------------------------------------------- negative cases --

#[test]
fn malformed_literals_are_diagnosed() {
    for (tag, source) in [
        (
            "fmt_unterminated_field",
            "fn main() { let _t = f\"value={value\"; }",
        ),
        (
            "fmt_unmatched_close",
            "fn main() { let _t = f\"value=value}\"; }",
        ),
        ("fmt_empty_field", "fn main() { let _t = f\"{}\"; }"),
        ("fmt_blank_field", "fn main() { let _t = f\"{   }\"; }"),
        ("fmt_half_escaped", "fn main() { let _t = f\"{{}\"; }"),
    ] {
        let file = std::sync::Arc::new(starkc::source::SourceFile::new(tag, source.to_string()));
        let (_ast, diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
        assert!(
            diags
                .iter()
                .any(|d| d.severity == starkc::diag::Severity::Error),
            "{tag}: expected a rejection, got {diags:?}"
        );
    }
}

#[test]
fn bad_specifications_are_diagnosed() {
    for (tag, source) in [
        (
            "fmt_unknown_type",
            "fn main() { let v: Int32 = 1; let _t = f\"{v:unknown}\"; }",
        ),
        (
            "fmt_bad_fill",
            "fn main() { let v: Int32 = 1; let _t = f\"{v:ab>10}\"; }",
        ),
        (
            "fmt_align_no_width",
            "fn main() { let v: Int32 = 1; let _t = f\"{v:>}\"; }",
        ),
        (
            "fmt_width_overflow",
            "fn main() { let v: Int32 = 1; let _t = f\"{v:999999999999999999999999999}\"; }",
        ),
        (
            "fmt_precision_overflow",
            "fn main() { let v: Float64 = 1.0; let _t = f\"{v:.99999}\"; }",
        ),
    ] {
        let file = std::sync::Arc::new(starkc::source::SourceFile::new(tag, source.to_string()));
        let (_ast, diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
        assert!(
            diags
                .iter()
                .any(|d| d.severity == starkc::diag::Severity::Error),
            "{tag}: expected a rejection, got {diags:?}"
        );
    }
}

/// **A deferred limitation, pinned so it cannot regress silently.**
///
/// A field carrying the outer literal's escapes — which any nested string literal must — is
/// refused rather than mis-parsed. §9.1 lists nested string literals as something the field scanner
/// should handle; this states plainly that WP-FMT-001 does not, and why refusing beats the
/// alternative (a string literal reads its value from its span, and a decoded copy has no span in
/// the real file, so it would render `\"slice\"` where the program said `slice`).
#[test]
fn a_field_may_not_carry_an_escape_sequence() {
    let source = "fn main() { let _t = f\"{call(\\\"a\\\")}\"; }";
    let file = std::sync::Arc::new(starkc::source::SourceFile::new(
        "fmt_escape_field",
        source.to_string(),
    ));
    let (_ast, diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(
        diags
            .iter()
            .any(|d| d.code.as_deref() == Some("E0218") && d.message.contains("escape sequence")),
        "expected the escape-in-field refusal, got {diags:?}"
    );
}

/// `stark-fmt`'s builder still works, and interpolation needs no dependency on it: this program
/// declares none and uses `f"..."` freely.
#[test]
fn interpolation_needs_no_package_dependency() {
    agree_completing_with_stdout(
        "fmt_no_dependency",
        "\
fn main() {
    let name: String = \"stark\".to_string();
    println(f\"pkg={name}\".as_str());
}
",
        "pkg=stark\n",
    );
}

#[test]
fn a_type_without_display_cannot_be_interpolated() {
    let messages = rejects_at_typecheck(
        "fmt_no_display",
        "\
struct Hidden {
    value: Int32,
}

fn main() {
    let hidden = Hidden { value: 1 };
    let text = f\"{hidden}\";
    println(text.as_str());
}
",
        "E0306",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("Display") && m.contains("Hidden")),
        "expected the missing Display to be named, got {messages:?}"
    );
}

#[test]
fn a_generic_without_the_bound_is_rejected() {
    rejects_at_typecheck(
        "fmt_generic_unbounded",
        "\
fn render<T>(value: &T) -> String {
    f\"value={value}\"
}

fn main() {}
",
        "E0306",
    );
}

/// `Display` does not prove integer formatting, so a numeric mode on a generic parameter is
/// refused rather than given a meaning it has not earned (§11.5).
#[test]
fn a_numeric_spec_on_a_generic_display_is_rejected() {
    rejects_at_typecheck(
        "fmt_generic_hex",
        "\
fn hex<T: Display>(value: &T) -> String {
    f\"{value:x}\"
}

fn main() {}
",
        "E0306",
    );
}

#[test]
fn type_and_spec_mismatches_are_rejected_at_type_checking() {
    for (tag, source) in [
        (
            "fmt_hex_on_string",
            "fn main() { let text: String = \"a\".to_string(); let _t = f\"{text:x}\"; }",
        ),
        (
            "fmt_precision_on_bool",
            "fn main() { let flag: Bool = true; let _t = f\"{flag:.2}\"; }",
        ),
        (
            "fmt_binary_on_float",
            "fn main() { let v: Float64 = 1.0; let _t = f\"{v:b}\"; }",
        ),
        (
            "fmt_precision_on_int",
            "fn main() { let v: Int32 = 1; let _t = f\"{v:.2}\"; }",
        ),
        (
            "fmt_precision_on_string",
            "fn main() { let s: String = \"abcdef\".to_string(); let _t = f\"{s:.5}\"; }",
        ),
        (
            "fmt_sign_on_string",
            "fn main() { let s: String = \"a\".to_string(); let _t = f\"{s:+}\"; }",
        ),
    ] {
        rejects_at_typecheck(tag, source, "E0306");
    }
}

// ------------------------------------------------------------------------ backend restriction --

/// Generated application code must route interpolation through STARK's runtime formatting, not
/// Rust's.
#[test]
fn native_routes_through_stark_formatting() {
    if !rustc_available() {
        eprintln!("SKIP-NATIVE: fmt_backend: no rustc.");
        return;
    }
    let source = "\
fn main() {
    let name: String = \"stark\".to_string();
    let count: Int32 = 42;
    let ratio: Float64 = 0.5;
    println(f\"{name} {count:04} {ratio:.2} {count:#x}\".as_str());
}
";
    let front = front_end("fmt_backend", source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(error) => panic!("fmt_backend: lowering failed: {}", error.what),
    };
    let verified = verify_program(&program).expect("fmt_backend: verification failed");
    let target_dir = std::env::temp_dir().join(format!("stark_fmt_backend_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.clone(),
            target_contract: "stark-64-v1".to_string(),
            profile: Profile::Debug,
            target_triple: None,
        },
    )
    .expect("fmt_backend: native build failed");
    let generated = std::fs::read_to_string(artifact.build_dir.join("src/main.rs"))
        .expect("fmt_backend: generated source unreadable");
    let _ = std::fs::remove_dir_all(&target_dir);

    assert!(
        generated.contains("stark_runtime::fmt_spec::"),
        "interpolation must lower to STARK's own format-specification runtime"
    );
    for forbidden in [
        "format!",
        "write!",
        "writeln!",
        "std::fmt::Display",
        "std::fmt::Debug",
        "#[derive(Debug",
    ] {
        assert!(
            !generated.contains(forbidden),
            "generated code must not implement interpolation with Rust's `{forbidden}`"
        );
    }
}

/// Debug and release native builds are a fourth execution mode, not a faster third one.
#[test]
fn release_and_debug_native_agree() {
    if !rustc_available() {
        eprintln!("SKIP-NATIVE: fmt_profiles: no rustc.");
        return;
    }
    let source = "\
fn main() {
    let name: String = \"stark\".to_string();
    let count: Int32 = -42;
    let ratio: Float64 = 0.125;
    println(f\"{name:.^12} {count:+06} {ratio:.2} {count:#x}\".as_str());
}
";
    let front = front_end("fmt_profiles", source);
    let program = match lower_program(&front.hir, &front.tables, front.file.clone()) {
        Ok(program) => program,
        Err(error) => panic!("fmt_profiles: lowering failed: {}", error.what),
    };
    let debug = support::differential::run_native_with_profile(
        "fmt_profiles",
        "fmt_profiles_debug",
        &program,
        Profile::Debug,
    );
    let release = support::differential::run_native_with_profile(
        "fmt_profiles",
        "fmt_profiles_release",
        &program,
        Profile::Release,
    );
    assert_eq!(
        support::differential::canonical_form(&debug),
        support::differential::canonical_form(&release),
        "debug and release native builds disagreed"
    );
}
