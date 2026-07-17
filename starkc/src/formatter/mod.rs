//! Specification-driven, idempotent source formatter (WP8.2).
//!
//! `format_file` walks the parsed [`crate::ast::Ast`] — never the raw
//! source text — and re-attaches comment trivia from
//! [`crate::lexer::tokenize_with_comments`] by source position. See
//! `printer.rs` for the formatting rules and `precedence.rs` for why
//! parenthesization must be re-derived from tree shape.
//!
//! Each file is formatted independently of its package (its own top-level
//! items only; `mod name;` declarations are not recursively expanded), so
//! formatting is well-defined per file regardless of how the package graph
//! resolves elsewhere.

pub mod comments;
pub mod precedence;
pub mod printer;

use crate::ast::Ast;
use crate::diag::{Diagnostic, Severity};
use crate::lexer::tokenize_with_comments;
use crate::options::LanguageOptions;
use crate::parser::{parse_with_options_into, ParseMode};
use crate::source::SourceFile;

/// Format `file`. Refuses to run (returning the blocking diagnostics
/// instead of guessing) if `file` does not parse cleanly — an AST-based
/// formatter has no text to fall back on for the parts it couldn't build a
/// tree for.
pub fn format_file(file: &SourceFile, options: LanguageOptions) -> Result<String, Vec<Diagnostic>> {
    let mut ast = Ast::default();
    let (root, diags) = parse_with_options_into(file, ParseMode::Program, options, &mut ast);
    if diags.iter().any(|d| d.severity == Severity::Error) {
        return Err(diags);
    }
    ast.root = root;
    let (_, comments, _) = tokenize_with_comments(file);
    Ok(printer::format(&ast, file, &comments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceFile;

    fn fmt(src: &str) -> String {
        let file = SourceFile::new("t.stark", src);
        format_file(&file, LanguageOptions::CORE).expect("should format")
    }

    fn assert_idempotent(src: &str) {
        let once = fmt(src);
        let file2 = SourceFile::new("t.stark", once.clone());
        let twice = format_file(&file2, LanguageOptions::CORE).expect("should format twice");
        assert_eq!(once, twice, "formatting is not idempotent");
    }

    #[test]
    fn formats_a_simple_function() {
        let out = fmt("fn add(a:Int32,b:Int32)->Int32{a+b}");
        assert_eq!(out, "fn add(a: Int32, b: Int32) -> Int32 {\n    a + b\n}\n");
    }

    #[test]
    fn preserves_line_comments() {
        let out = fmt("// leading\nfn f() {}\n");
        assert_eq!(out, "// leading\nfn f() {}\n");
    }

    #[test]
    fn preserves_trailing_comments() {
        let out = fmt("fn f() {\n    let x = 1; // note\n}\n");
        assert!(out.contains("// note"), "output was: {out:?}");
    }

    #[test]
    fn idempotent_on_operators_and_precedence() {
        assert_idempotent("fn f() { let x = (1 + 2) * 3 - 4 / (5 - 6); }");
        assert_idempotent("fn f() { let x = 1 + 2 * 3; }");
        assert_idempotent("fn f() { let x = a ** b ** c; }");
        assert_idempotent("fn f() { let x = -(-a); }");
        assert_idempotent("fn f() { let x = (a = b); }");
    }

    #[test]
    fn idempotent_with_comments() {
        assert_idempotent("// doc\nfn f() {\n    // body comment\n    let x = 1; // trailing\n}\n");
    }

    #[test]
    fn struct_literal_guard_in_condition() {
        let out = fmt("struct P { x: Int32 }\nfn f() { if (P { x: 1 }).x == 1 {} }\n");
        assert_idempotent("struct P { x: Int32 }\nfn f() { if (P { x: 1 }).x == 1 {} }\n");
        // The whole condition is guarded (not just the struct literal): the
        // parser's restriction propagates through the entire "head"
        // position, so a single outer paren is the simplest always-correct
        // fix, not necessarily the minimal one.
        assert!(
            out.contains("if (P { x: 1 }.x == 1)"),
            "output was: {out:?}"
        );
    }

    #[test]
    fn rejects_unparseable_input() {
        let file = SourceFile::new("t.stark", "fn f( {");
        assert!(format_file(&file, LanguageOptions::CORE).is_err());
    }

    #[test]
    fn flattens_and_sorts_use_groups() {
        let out = fmt("use std::{b, a};\n");
        assert_eq!(out, "use std::a;\nuse std::b;\n");
    }
}
