//! The 01-Lexical-Grammar spec fixtures must lex without diagnostics.
//! (Full-corpus conformance arrives with the parser in WP1.3/WP1.4; the
//! lexical fixtures are in scope for WP1.2.)

use starkc::lexer::{tokenize, TokenKind};
use starkc::source::SourceFile;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../STARKLANG/tests/spec-fixtures")
}

#[test]
fn lexical_grammar_fixtures_lex_cleanly() {
    let dir = fixture_dir();
    let mut checked = 0;
    for entry in std::fs::read_dir(&dir).expect("fixture dir exists") {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.starts_with("01-Lexical-Grammar") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let file = SourceFile::new(name.clone(), src);
        let (tokens, diags) = tokenize(&file);
        assert!(
            diags.is_empty(),
            "{name}: unexpected lex diagnostics: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        assert_eq!(tokens.last().map(|t| t.kind), Some(TokenKind::Eof));
        checked += 1;
    }
    assert_eq!(checked, 4, "expected exactly four 01-Lexical fixtures");
}

#[test]
fn all_fixtures_lex_without_panicking() {
    // Robustness only: many fixtures are semantic-error examples or API
    // notation, so diagnostics are fine — but the lexer must never panic
    // and must always terminate with Eof.
    for entry in std::fs::read_dir(fixture_dir()).expect("fixture dir exists") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "stark") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).unwrap();
        let file = SourceFile::new(name, src);
        let (tokens, _diags) = tokenize(&file);
        assert_eq!(tokens.last().map(|t| t.kind), Some(TokenKind::Eof));
    }
}
