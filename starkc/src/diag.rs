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

use crate::analysis::{SourceId, SourceMap, SourceProvenance};
use crate::source::{SourceFile, Span};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }

    fn heading(self) -> &'static str {
        match self {
            Severity::Error => "Error",
            Severity::Warning => "Warning",
        }
    }

    fn uncategorized_code(self) -> &'static str {
        match self {
            Severity::Error => "E0000",
            Severity::Warning => "W0000",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelatedDiagnostic {
    pub message: String,
    pub span: Span,
    pub file: Arc<SourceFile>,
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
    pub file: Option<Arc<SourceFile>>,
    pub related: Vec<RelatedDiagnostic>,
    pub rule_id: Option<String>,
    pub deviation_id: Option<String>,
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
            file: None,
            related: Vec::new(),
            rule_id: None,
            deviation_id: None,
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
            file: None,
            related: Vec::new(),
            rule_id: None,
            deviation_id: None,
        }
    }

    pub fn with_file(mut self, file: Arc<SourceFile>) -> Self {
        self.file = Some(file);
        self
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

    pub fn with_related(
        mut self,
        file: Arc<SourceFile>,
        span: Span,
        message: impl Into<String>,
    ) -> Self {
        self.related.push(RelatedDiagnostic {
            message: message.into(),
            span,
            file,
        });
        self
    }

    pub fn with_rule_id(mut self, rule_id: impl Into<String>) -> Self {
        self.rule_id = Some(rule_id.into());
        self
    }

    pub fn with_deviation_id(mut self, deviation_id: impl Into<String>) -> Self {
        self.deviation_id = Some(deviation_id.into());
        self
    }

    /// Render this diagnostic against its source file, in the normative
    /// format. The result always ends with a newline.
    pub fn render(&self, default_file: &SourceFile) -> String {
        let file = self.file.as_deref().unwrap_or(default_file);
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticLocation {
    pub source: SourceId,
    pub span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticRelatedInformation {
    pub message: String,
    pub location: DiagnosticLocation,
}

/// Compiler-owned diagnostic form consumed by CLI and language-service transports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuredDiagnostic {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    pub primary: DiagnosticLocation,
    pub label: Option<String>,
    pub related: Vec<DiagnosticRelatedInformation>,
    pub notes: Vec<String>,
    pub help: Vec<String>,
    pub source_version: Option<i64>,
    pub rule_id: Option<String>,
    pub deviation_id: Option<String>,
}

/// A deterministic diagnostic snapshot for one analysis session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticBatch {
    pub schema_version: u32,
    pub sources: Vec<DiagnosticSource>,
    pub extensions: Vec<String>,
    pub diagnostics: Vec<StructuredDiagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticSource {
    pub id: SourceId,
    pub file: String,
    pub kind: DiagnosticSourceKind,
    pub package: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSourceKind {
    Root,
    Module,
}

impl DiagnosticSourceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::Module => "module",
        }
    }
}

impl DiagnosticBatch {
    pub(crate) fn from_compiler_diagnostics(
        diagnostics: &[Diagnostic],
        sources: &SourceMap,
        default_source: SourceId,
        source_versions: &HashMap<SourceId, i64>,
        extensions: Vec<String>,
    ) -> Self {
        let diagnostics = diagnostics
            .iter()
            .map(|diagnostic| {
                let primary_source = diagnostic
                    .file
                    .as_ref()
                    .map(|file| {
                        sources.id_for_name(&file.name).unwrap_or_else(|| {
                            panic!(
                                "diagnostic primary file `{}` is absent from the source map",
                                file.name
                            )
                        })
                    })
                    .unwrap_or(default_source);
                let related = diagnostic
                    .related
                    .iter()
                    .map(|related| {
                        let source = sources.id_for_name(&related.file.name).unwrap_or_else(|| {
                            panic!(
                                "diagnostic related file `{}` is absent from the source map",
                                related.file.name
                            )
                        });
                        DiagnosticRelatedInformation {
                            message: related.message.clone(),
                            location: DiagnosticLocation {
                                source,
                                span: related.span,
                            },
                        }
                    })
                    .collect();
                StructuredDiagnostic {
                    code: diagnostic
                        .code
                        .clone()
                        .unwrap_or_else(|| diagnostic.severity.uncategorized_code().to_string()),
                    severity: diagnostic.severity,
                    message: diagnostic.message.clone(),
                    primary: DiagnosticLocation {
                        source: primary_source,
                        span: diagnostic.span,
                    },
                    label: (!diagnostic.label.is_empty()).then(|| diagnostic.label.clone()),
                    related,
                    notes: diagnostic.notes.clone(),
                    help: diagnostic.helps.clone(),
                    source_version: source_versions.get(&primary_source).copied(),
                    rule_id: diagnostic.rule_id.clone(),
                    deviation_id: diagnostic.deviation_id.clone(),
                }
            })
            .collect();
        Self {
            schema_version: 1,
            sources: sources
                .files()
                .iter()
                .map(|source| {
                    let (kind, package) = match &source.provenance {
                        SourceProvenance::Root { package } => {
                            (DiagnosticSourceKind::Root, package.clone())
                        }
                        SourceProvenance::Module { package } => {
                            (DiagnosticSourceKind::Module, package.clone())
                        }
                    };
                    DiagnosticSource {
                        id: source.id,
                        file: source.file.name.clone(),
                        kind,
                        package,
                    }
                })
                .collect(),
            extensions,
            diagnostics,
        }
    }

    pub fn render(&self, sources: &SourceMap) -> String {
        let mut output = String::new();
        for diagnostic in &self.diagnostics {
            output.push_str(&diagnostic.render(sources));
        }
        output
    }

    pub fn to_json(&self, sources: &SourceMap) -> String {
        let source_records = self
            .sources
            .iter()
            .map(|source| {
                format!(
                    "{{\"sourceId\":{},\"file\":\"{}\",\"kind\":\"{}\",\"package\":{}}}",
                    source.id.as_u32(),
                    escape_json(&source.file),
                    source.kind.as_str(),
                    option_json(source.package.as_deref())
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let diagnostics = self
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.to_json(sources))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{",
                "\"schemaVersion\":{},",
                "\"tool\":\"starkc\",",
                "\"toolVersion\":\"{}\",",
                "\"extensions\":{},",
                "\"sources\":[{}],",
                "\"diagnostics\":[{}]",
                "}}"
            ),
            self.schema_version,
            env!("CARGO_PKG_VERSION"),
            string_array_json(&self.extensions),
            source_records,
            diagnostics
        )
    }
}

impl StructuredDiagnostic {
    pub fn render(&self, sources: &SourceMap) -> String {
        let primary = sources
            .get(self.primary.source)
            .expect("structured diagnostic source must exist");
        let diagnostic = Diagnostic {
            severity: self.severity,
            code: (self.code != self.severity.uncategorized_code()).then(|| self.code.clone()),
            message: self.message.clone(),
            span: self.primary.span,
            label: self.label.clone().unwrap_or_default(),
            helps: self.help.clone(),
            notes: self.notes.clone(),
            file: Some(primary.file.clone()),
            related: Vec::new(),
            rule_id: self.rule_id.clone(),
            deviation_id: self.deviation_id.clone(),
        };
        let mut rendered = diagnostic.render(&primary.file);
        for related in &self.related {
            let record = sources
                .get(related.location.source)
                .expect("related diagnostic source must exist");
            let (line, column) = record.file.line_col(related.location.span.lo);
            let _ = writeln!(
                rendered,
                "   = related: {}:{line}:{column}: {}",
                record.file.name, related.message
            );
        }
        rendered
    }

    fn to_json(&self, sources: &SourceMap) -> String {
        let primary = sources
            .get(self.primary.source)
            .expect("structured diagnostic source must exist");
        let related = self
            .related
            .iter()
            .map(|related| {
                let source = sources
                    .get(related.location.source)
                    .expect("related diagnostic source must exist");
                format!(
                    "{{\"message\":\"{}\",\"sourceId\":{},\"file\":\"{}\",\"span\":{}}}",
                    escape_json(&related.message),
                    related.location.source.as_u32(),
                    escape_json(&source.file.name),
                    span_json(related.location.span)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{",
                "\"code\":\"{}\",",
                "\"severity\":\"{}\",",
                "\"message\":\"{}\",",
                "\"primary\":{{\"sourceId\":{},\"file\":\"{}\",\"span\":{},\"label\":{}}},",
                "\"related\":[{}],",
                "\"notes\":{},",
                "\"help\":{},",
                "\"sourceVersion\":{},",
                "\"ruleId\":{},",
                "\"deviationId\":{}",
                "}}"
            ),
            escape_json(&self.code),
            self.severity.as_str(),
            escape_json(&self.message),
            self.primary.source.as_u32(),
            escape_json(&primary.file.name),
            span_json(self.primary.span),
            option_json(self.label.as_deref()),
            related,
            string_array_json(&self.notes),
            string_array_json(&self.help),
            self.source_version
                .map_or_else(|| "null".to_string(), |version| version.to_string()),
            option_json(self.rule_id.as_deref()),
            option_json(self.deviation_id.as_deref())
        )
    }
}

fn span_json(span: Span) -> String {
    format!("{{\"startByte\":{},\"endByte\":{}}}", span.lo, span.hi)
}

fn option_json(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", escape_json(value)),
    )
}

fn string_array_json(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("\"{}\"", escape_json(value)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn escape_json(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0c}' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character <= '\u{1f}' => {
                let _ = write!(escaped, "\\u{:04x}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped
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
