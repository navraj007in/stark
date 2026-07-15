use starkc::diag::{Diagnostic, Severity};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gate2-valid")
}

fn analyze(name: &str, source: String) -> Vec<Diagnostic> {
    let file = Arc::new(SourceFile::new(name.to_string(), source));
    let (ast, mut diagnostics) = parse(&file, ParseMode::Program);
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return diagnostics;
    }
    let (hir, mut resolution) = resolve(&ast, file.clone());
    diagnostics.append(&mut resolution);
    if !diagnostics.iter().any(|d| d.severity == Severity::Error) {
        diagnostics.append(&mut typecheck::check(&hir, file));
    }
    diagnostics
}

#[test]
fn all_gate2_valid_programs_pass_semantic_analysis() {
    let mut paths: Vec<_> = std::fs::read_dir(fixture_dir())
        .expect("gate2 valid fixture directory exists")
        .map(|entry| entry.expect("fixture entry is readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "stark")
        })
        .collect();
    paths.sort();
    assert!(
        paths.len() >= 20,
        "Gate 2 requires at least 20 valid programs"
    );

    let mut failures = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let source = std::fs::read_to_string(&path).expect("fixture is readable");
        let diagnostics = analyze(&name, source);
        let errors: Vec<_> = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .map(|diagnostic| {
                format!(
                    "{}: {}",
                    diagnostic.code.as_deref().unwrap_or("uncoded"),
                    diagnostic.message
                )
            })
            .collect();
        if !errors.is_empty() {
            failures.push(format!("{name}: {}", errors.join(", ")));
        }
    }
    assert!(
        failures.is_empty(),
        "valid Gate 2 programs failed:\n{}",
        failures.join("\n")
    );
}
