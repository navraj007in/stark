use starkc::options::LanguageOptions;
use starkc::parser::{parse_with_options, ParseMode};
use starkc::source::SourceFile;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Default)]
struct Case {
    name: String,
    class: String,
    source: String,
    errors: Vec<String>,
}

fn manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gate4/manifest.toml")
}

fn value(line: &str) -> String {
    line.split_once('=')
        .expect("manifest assignment")
        .1
        .trim()
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .expect("quoted manifest value")
        .to_string()
}

fn semantic_cases() -> Vec<Case> {
    let text = std::fs::read_to_string(manifest_path()).expect("Gate 4 manifest");
    let mut cases = Vec::new();
    let mut current: Option<Case> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line == "[[case]]" {
            if let Some(case) = current.take() {
                cases.push(case);
            }
            current = Some(Case::default());
        } else if line.starts_with('[') {
            if let Some(case) = current.take() {
                cases.push(case);
            }
        } else if let Some(case) = current.as_mut() {
            if line.starts_with("name =") {
                case.name = value(line);
            } else if line.starts_with("class =") {
                case.class = value(line);
            } else if line.starts_with("source =") {
                case.source = value(line);
            } else if line.starts_with("errors =") {
                case.errors = value(line).split(',').map(str::to_string).collect();
            }
        }
    }
    if let Some(case) = current {
        cases.push(case);
    }
    cases
        .into_iter()
        .filter(|case| case.class.starts_with("semantic-") || case.class == "reserved-reject")
        .collect()
}

fn diagnostics(case: &Case) -> Vec<starkc::diag::Diagnostic> {
    let options = LanguageOptions::with_tensor();
    let file = Arc::new(SourceFile::new(
        format!("{}.stark", case.name),
        case.source.clone(),
    ));
    let (ast, mut diagnostics) = parse_with_options(&file, ParseMode::Program, options);
    if diagnostics.is_empty() {
        let (hir, mut resolve_diagnostics) =
            starkc::resolve::resolve_with_options(&ast, file.clone(), options);
        diagnostics.append(&mut resolve_diagnostics);
        diagnostics.extend(starkc::typecheck::check_with_options(&hir, file, options));
    }
    diagnostics
}

#[test]
fn gate4_manifest_semantic_cases_match_compiler_verdicts() {
    let cases = semantic_cases();
    assert!(!cases.is_empty(), "Gate 4 manifest has no semantic cases");
    for case in cases {
        let diagnostics = diagnostics(&case);
        let codes = diagnostics
            .iter()
            .filter_map(|diagnostic| diagnostic.code.as_deref())
            .collect::<Vec<_>>();
        match case.class.as_str() {
            "semantic-pass" => assert!(
                diagnostics.is_empty(),
                "{} unexpectedly failed: {diagnostics:?}",
                case.name
            ),
            "semantic-error" => {
                assert!(!diagnostics.is_empty(), "{} unexpectedly passed", case.name);
                for expected in &case.errors {
                    assert!(
                        codes.contains(&expected.as_str()),
                        "{} expected {expected}, found {codes:?}",
                        case.name
                    );
                }
            }
            "reserved-reject" => assert!(
                diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message.contains("reserved")),
                "{} did not produce a focused reserved diagnostic: {diagnostics:?}",
                case.name
            ),
            other => panic!("unknown semantic class {other}"),
        }
    }
}
