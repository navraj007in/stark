//! Spec-fixture conformance harness (PLAN.md WP1.3).
//!
//! Reads STARKLANG/tests/spec-fixtures/manifest.toml — the hand-triaged
//! verdict for every ```stark block extracted from the spec — and enforces
//! it against the compiler:
//!
//! - `manifest_is_valid_and_covers_corpus` (always on): the manifest parses,
//!   uses only known keys/verdicts, and covers exactly the extracted files.
//! - `lex_level_conformance` (always on): `lex-pass`, `parse-pass`, and
//!   `semantic-error` fixtures lex without diagnostics; every fixture lexes
//!   without panicking and ends in Eof.
//! - `spec_conformance` (#[ignore] until the WP1.4 parser lands): full
//!   verdict enforcement. Run with `cargo test --test conformance --
//!   --include-ignored`. Red today by design: parse verdicts need a parser.
//!
//! The manifest is a deliberately flat TOML subset (see its header) so this
//! harness stays dependency-free.

use starkc::lexer::{tokenize, TokenKind};
use starkc::source::SourceFile;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    ParsePass,
    ParseFail,
    SemanticError,
    LexPass,
    Notation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Program,
    Snippet,
}

#[derive(Debug)]
struct Entry {
    verdict: Verdict,
    mode: Option<Mode>,
    errors: Option<String>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../STARKLANG/tests/spec-fixtures")
}

/// Parse the manifest's flat TOML subset: `["name"]` section headers followed
/// by `key = "value"` pairs. Anything else (arrays, nesting, multi-line
/// strings) is rejected so the format cannot silently grow past this parser.
fn parse_manifest(text: &str) -> BTreeMap<String, Entry> {
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();
    let mut current: Option<String> = None;
    for (idx, raw) in text.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line
            .strip_prefix("[\"")
            .and_then(|rest| rest.strip_suffix("\"]"))
        {
            assert!(
                !entries.contains_key(name),
                "manifest line {lineno}: duplicate entry for {name}"
            );
            entries.insert(
                name.to_string(),
                Entry {
                    verdict: Verdict::Notation, // placeholder until `verdict` key
                    mode: None,
                    errors: None,
                },
            );
            current = Some(name.to_string());
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("manifest line {lineno}: expected `key = \"value\"`"));
        let key = key.trim();
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or_else(|| panic!("manifest line {lineno}: value must be a quoted string"));
        let name = current
            .as_ref()
            .unwrap_or_else(|| panic!("manifest line {lineno}: key outside any [\"...\"] entry"));
        let entry = entries.get_mut(name).unwrap();
        match key {
            "verdict" => {
                entry.verdict = match value {
                    "parse-pass" => Verdict::ParsePass,
                    "parse-fail" => Verdict::ParseFail,
                    "semantic-error" => Verdict::SemanticError,
                    "lex-pass" => Verdict::LexPass,
                    "notation" => Verdict::Notation,
                    other => panic!("manifest line {lineno}: unknown verdict {other:?}"),
                };
            }
            "mode" => {
                entry.mode = Some(match value {
                    "program" => Mode::Program,
                    "snippet" => Mode::Snippet,
                    other => panic!("manifest line {lineno}: unknown mode {other:?}"),
                });
            }
            "errors" => entry.errors = Some(value.to_string()),
            "note" => {} // free text, not machine-checked
            other => panic!("manifest line {lineno}: unknown key {other:?}"),
        }
    }
    entries
}

fn load_manifest() -> BTreeMap<String, Entry> {
    let path = fixture_dir().join("manifest.toml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let entries = parse_manifest(&text);

    // Schema invariants per entry.
    for (name, entry) in &entries {
        let needs_mode = matches!(
            entry.verdict,
            Verdict::ParsePass | Verdict::ParseFail | Verdict::SemanticError
        );
        assert_eq!(
            entry.mode.is_some(),
            needs_mode,
            "{name}: `mode` is required for parse-*/semantic-error verdicts and \
             forbidden otherwise"
        );
        assert_eq!(
            entry.errors.is_some(),
            entry.verdict == Verdict::SemanticError,
            "{name}: `errors` is required exactly for semantic-error verdicts"
        );
        if let Some(errors) = &entry.errors {
            for code in errors.split(',') {
                let ok = code.len() == 5
                    && code.starts_with('E')
                    && code[1..].chars().all(|c| c.is_ascii_digit());
                assert!(ok, "{name}: malformed error code {code:?} in {errors:?}");
            }
        }
    }
    entries
}

fn corpus_files() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(fixture_dir())
        .expect("fixture dir exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "stark"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn lex_fixture(name: &str) -> (SourceFile, Vec<TokenKind>, usize) {
    let src = std::fs::read_to_string(fixture_dir().join(name)).unwrap();
    let file = SourceFile::new(name.to_string(), src);
    let (tokens, diags) = tokenize(&file);
    let kinds = tokens.iter().map(|t| t.kind).collect();
    (file, kinds, diags.len())
}

#[test]
fn manifest_is_valid_and_covers_corpus() {
    let entries = load_manifest();
    let files = corpus_files();
    let listed: Vec<String> = entries.keys().cloned().collect();
    assert_eq!(
        listed, files,
        "manifest entries and extracted fixtures diverge — re-run \
         STARKLANG/tools/extract-spec-examples.sh and re-triage"
    );
}

#[test]
fn lex_level_conformance() {
    let entries = load_manifest();
    let mut failures = Vec::new();
    for (name, entry) in &entries {
        let (_file, kinds, diag_count) = lex_fixture(name);
        // Robustness floor for every fixture, notation included.
        assert_eq!(
            kinds.last(),
            Some(&TokenKind::Eof),
            "{name}: token stream must end in Eof"
        );
        let must_lex_clean = matches!(
            entry.verdict,
            Verdict::LexPass | Verdict::ParsePass | Verdict::SemanticError
        );
        if must_lex_clean && diag_count > 0 {
            failures.push(format!("{name}: {diag_count} lex diagnostics"));
        }
    }
    assert!(
        failures.is_empty(),
        "fixtures that must lex cleanly produced diagnostics:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "red until the WP1.4 parser lands; run via --include-ignored"]
fn spec_conformance() {
    let entries = load_manifest();
    let mut pass = 0usize;
    let mut skip = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let mut by_class: BTreeMap<&str, (usize, usize)> = BTreeMap::new(); // (pass, fail)

    for (name, entry) in &entries {
        let class = match entry.verdict {
            Verdict::ParsePass => "parse-pass",
            Verdict::ParseFail => "parse-fail",
            Verdict::SemanticError => "semantic-error",
            Verdict::LexPass => "lex-pass",
            Verdict::Notation => "notation",
        };
        if entry.verdict == Verdict::Notation {
            skip += 1;
            continue;
        }
        let (_file, _kinds, diag_count) = lex_fixture(name);
        let result: Result<(), String> = match entry.verdict {
            Verdict::LexPass => {
                if diag_count == 0 {
                    Ok(())
                } else {
                    Err(format!("{diag_count} lex diagnostics"))
                }
            }
            Verdict::ParseFail => {
                if diag_count > 0 {
                    Ok(()) // rejected by the lexer — rejection is rejection
                } else {
                    // WP1.4: parse under entry.mode and require a parse error.
                    Err("parser not implemented (WP1.4)".to_string())
                }
            }
            Verdict::ParsePass | Verdict::SemanticError => {
                if diag_count > 0 {
                    Err(format!("{diag_count} lex diagnostics"))
                } else {
                    // WP1.4: parse under entry.mode and require success.
                    // Gate 2: for semantic-error, require the codes in
                    // entry.errors from the checker.
                    Err("parser not implemented (WP1.4)".to_string())
                }
            }
            Verdict::Notation => unreachable!(),
        };
        let slot = by_class.entry(class).or_insert((0, 0));
        match result {
            Ok(()) => {
                pass += 1;
                slot.0 += 1;
            }
            Err(why) => {
                slot.1 += 1;
                failures.push(format!("{name} [{class}]: {why}"));
            }
        }
    }

    println!(
        "spec conformance: {pass} pass, {} fail, {skip} skipped (notation)",
        failures.len()
    );
    for (class, (p, f)) in &by_class {
        println!("  {class:15} {p:3} pass  {f:3} fail");
    }
    assert!(
        failures.is_empty(),
        "{} fixtures failed conformance:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
