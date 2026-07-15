//! Diagnostics in the normative format of `04-Semantic-Analysis.md`:
//!
//! ```text
//! Error: [E0001] Type mismatch
//!   --> file.stark:3:9
//!    |
//!  3 | let x: String = 42;
//!    |        ^^^^^^ expected String, found Int32
//!    |
//!    = help: detailed explanation
//!    = note: additional information
//! ```
//!
//! The renderer is deliberately small and dependency-free (PLAN.md T7):
//! matching the spec's format exactly matters more than gradient underlines.

use crate::source::{SourceFile, Span};
use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    fn heading(self) -> &'static str {
        match self {
            Severity::Error => "Error",
            Severity::Warning => "Warning",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    /// Spec error code (`E0001`, `W0100`, ...). Internal/staging diagnostics
    /// may omit it.
    pub code: Option<String>,
    pub message: String,
    pub span: Span,
    /// Text shown under the caret line.
    pub label: String,
    pub helps: Vec<String>,
    pub notes: Vec<String>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            severity: Severity::Error,
            code: None,
            message: message.into(),
            span,
            label: String::new(),
            helps: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            code: None,
            message: message.into(),
            span,
            label: String::new(),
            helps: Vec::new(),
            notes: Vec::new(),
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.helps.push(help.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Render this diagnostic against its source file, in the normative
    /// format. The result always ends with a newline.
    pub fn render(&self, file: &SourceFile) -> String {
        let (line, col) = file.line_col(self.span.lo);
        let line_str = line.to_string();
        // Right-align the line number into a min-width-2 column so the `|`
        // of every row lines up and carets sit exactly under their span.
        let width = line_str.len().max(2);
        let gutter = " ".repeat(width - 1);
        let mut out = String::new();

        match &self.code {
            Some(code) => {
                let _ = writeln!(
                    out,
                    "{}: [{}] {}",
                    self.severity.heading(),
                    code,
                    self.message
                );
            }
            None => {
                let _ = writeln!(out, "{}: {}", self.severity.heading(), self.message);
            }
        }
        let _ = writeln!(out, "{gutter} --> {}:{line}:{col}", file.name);
        let _ = writeln!(out, "{gutter}  |");

        let text = file.line_text(line);
        let _ = writeln!(out, "{line_str:>width$} | {text}");

        // Caret width: span portion that falls on the first line, min 1.
        let (end_line, end_col) = file.line_col(self.span.hi);
        let width = if end_line == line && end_col > col {
            end_col - col
        } else {
            1
        };
        let carets = "^".repeat(width);
        let pad = " ".repeat(col - 1);
        if self.label.is_empty() {
            let _ = writeln!(out, "{gutter}  | {pad}{carets}");
        } else {
            let _ = writeln!(out, "{gutter}  | {pad}{carets} {}", self.label);
        }

        if !self.helps.is_empty() || !self.notes.is_empty() {
            let _ = writeln!(out, "{gutter}  |");
            for help in &self.helps {
                let _ = writeln!(out, "{gutter}  = help: {help}");
            }
            for note in &self.notes {
                let _ = writeln!(out, "{gutter}  = note: {note}");
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_normative_format() {
        let file = SourceFile::new("example.stark", "let x: String = 42;\n");
        let diag = Diagnostic::error("Type mismatch", Span::new(16, 18))
            .with_code("E0001")
            .with_label("expected String, found Int32")
            .with_help("change the annotation or the initializer")
            .with_note("no implicit numeric conversions exist");
        let rendered = diag.render(&file);
        let expected = "\
Error: [E0001] Type mismatch
  --> example.stark:1:17
   |
 1 | let x: String = 42;
   |                 ^^ expected String, found Int32
   |
   = help: change the annotation or the initializer
   = note: no implicit numeric conversions exist
";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn renders_without_code_or_extras() {
        let file = SourceFile::new("f.stark", "fn main() {}\n");
        let diag = Diagnostic::error("something went wrong", Span::point(0));
        let rendered = diag.render(&file);
        let expected = "\
Error: something went wrong
  --> f.stark:1:1
   |
 1 | fn main() {}
   | ^
";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn gutter_widens_for_multidigit_lines() {
        let src = "//\n".repeat(11) + "oops\n";
        let file = SourceFile::new("f.stark", src);
        let diag = Diagnostic::error("bad", Span::new(33, 37)).with_code("E0002");
        let rendered = diag.render(&file);
        assert!(rendered.contains("12 | oops"), "got:\n{rendered}");
        assert!(rendered.contains("   | ^^^^"), "got:\n{rendered}");
    }
}
