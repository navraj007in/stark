//! AST snapshot tests for representative spec fixtures (PLAN.md WP1.4).
//!
//! Each named fixture is parsed (in its manifest mode) and the `ast::dump`
//! output is compared byte-for-byte against
//! `tests/snapshots/<fixture>.ast`. On intended AST changes, regenerate
//! with:
//!
//! ```text
//! UPDATE_SNAPSHOTS=1 cargo test --test snapshots
//! ```
//!
//! and review the diff like any other code change.

use starkc::ast;
use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;
use std::path::PathBuf;

/// (fixture, mode) — a cross-section of the grammar: ownership signatures,
/// generics and bounds, traits with associated types, impls, match/patterns,
/// block-formed statements, ranges/slices, use trees, struct literals.
const CASES: &[(&str, ParseMode)] = &[
    ("00-Core-Language-Overview__03.stark", ParseMode::Program),
    ("02-Syntax-Grammar__01.stark", ParseMode::Snippet),
    ("03-Type-System__08.stark", ParseMode::Snippet),
    ("03-Type-System__11.stark", ParseMode::Snippet),
    ("03-Type-System__13.stark", ParseMode::Program),
    ("03-Type-System__20.stark", ParseMode::Snippet),
    ("03-Type-System__31.stark", ParseMode::Program),
    ("03-Type-System__37.stark", ParseMode::Program),
    ("03-Type-System__40.stark", ParseMode::Program),
    ("04-Semantic-Analysis__12.stark", ParseMode::Program),
    ("04-Semantic-Analysis__15.stark", ParseMode::Snippet),
    ("05-Memory-Model__11.stark", ParseMode::Snippet),
    ("05-Memory-Model__20.stark", ParseMode::Snippet),
    ("06-Standard-Library__16.stark", ParseMode::Program),
    ("07-Modules-and-Packages__03.stark", ParseMode::Program),
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../STARKLANG/tests/spec-fixtures")
}

fn snapshot_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
}

#[test]
fn ast_snapshots() {
    let update = std::env::var_os("UPDATE_SNAPSHOTS").is_some();
    let mut failures = Vec::new();
    for &(name, mode) in CASES {
        let src = std::fs::read_to_string(fixture_dir().join(name)).unwrap();
        let file = SourceFile::new(name.to_string(), src);
        let (tree, diags) = parse(&file, mode);
        assert!(
            diags.is_empty(),
            "{name}: snapshot fixtures must parse cleanly: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
        let actual = ast::dump(&tree, &file);
        let snap_path = snapshot_dir().join(format!("{name}.ast"));
        if update {
            std::fs::create_dir_all(snapshot_dir()).unwrap();
            std::fs::write(&snap_path, &actual).unwrap();
            continue;
        }
        let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|_| {
            panic!(
                "missing snapshot {}; run UPDATE_SNAPSHOTS=1",
                snap_path.display()
            )
        });
        if actual != expected {
            failures.push(name);
            eprintln!("=== snapshot mismatch: {name} ===");
            for diff in diff_lines(&expected, &actual) {
                eprintln!("{diff}");
            }
        }
    }
    assert!(
        failures.is_empty(),
        "snapshot mismatches in {failures:?}; if intended, rerun with UPDATE_SNAPSHOTS=1"
    );
}

fn diff_lines(expected: &str, actual: &str) -> Vec<String> {
    let e: Vec<&str> = expected.lines().collect();
    let a: Vec<&str> = actual.lines().collect();
    let mut out = Vec::new();
    for i in 0..e.len().max(a.len()) {
        match (e.get(i), a.get(i)) {
            (Some(x), Some(y)) if x == y => {}
            (x, y) => {
                if let Some(x) = x {
                    out.push(format!("-{x}"));
                }
                if let Some(y) = y {
                    out.push(format!("+{y}"));
                }
            }
        }
    }
    out
}
