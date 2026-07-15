//! End-to-end check that parser-produced diagnostics render byte-for-byte
//! in the normative format of 04-Semantic-Analysis.md "Error Message
//! Format" (WP1.5). The unit test in `diag.rs` covers a hand-built
//! diagnostic; this covers the real pipeline: source → lexer → parser →
//! render.

use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;

fn render_all(src: &str, mode: ParseMode) -> String {
    let file = SourceFile::new("example.stark", src.to_string());
    let (_ast, diags) = parse(&file, mode);
    diags.iter().map(|d| d.render(&file)).collect()
}

#[test]
fn chained_comparison_renders_normative_format() {
    let out = render_all("let x = a < b < c;\n", ParseMode::Snippet);
    let expected = "\
Error: comparison operators cannot be chained
  --> example.stark:1:15
   |
 1 | let x = a < b < c;
   |               ^
   |
   = help: parenthesize to compare a Bool result explicitly
   = note: generic arguments in expressions use `path::<T>` (turbofish), not `path<T>`
";
    assert_eq!(out, expected);
}

#[test]
fn reserved_word_renders_with_label() {
    let out = render_all("let x = await;\n", ParseMode::Snippet);
    let expected = "\
Error: `await` is reserved for future use
  --> example.stark:1:9
   |
 1 | let x = await;
   |         ^^^^^ expected an expression
";
    assert_eq!(out, expected);
}

#[test]
fn multiple_diagnostics_from_one_file() {
    // Recovery must surface both errors, in source order.
    let out = render_all("let = 1;\nlet y = )\n", ParseMode::Snippet);
    let first = out.find("--> example.stark:1:5").expect(&out);
    let second = out.find("--> example.stark:2:9").expect(&out);
    assert!(first < second, "{out}");
}
