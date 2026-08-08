//! AS2 exit criterion 1 — no production entry point assembles the semantic pipeline outside the
//! driver.
//!
//! AS0 enumerated eleven production assemblies, six of which ran their own parse → resolve →
//! typecheck. AS2 migrated all six onto `CompilerSession`. This test is what keeps them there: a
//! seventh is easy to add by accident, and a search that looked only at the obvious files is
//! exactly how `starkide` and the ONNX verifier were missed the first time.
//!
//! The check is an EXACT SET, not a count. A new allowed exception must be named here with its
//! reason, which makes adding one a reviewable act rather than a silent one.

use std::path::{Path, PathBuf};

/// Calls that mean "I am running a pipeline phase myself".
const PHASE_CALLS: &[&str] = &[
    "resolve_with_options(",
    "resolve::resolve(",
    "typecheck::analyze(",
    "typecheck::analyze_with_options(",
    "analyze_with_options(",
];

/// Production files allowed to call a phase directly, and why.
///
/// `analysis.rs` IS the pipeline. `resolve.rs` calls its own `resolve_with_options` from the
/// `resolve` convenience wrapper. `onnx/verifier.rs` is a deliberate PARTIAL assembly — it resolves
/// a tensor declaration without typechecking it, so it is not a competing full pipeline (AS0 §3).
///
/// `typecheck.rs` is deliberately absent: it *defines* the phase and never calls one outside its own
/// tests.
const ALLOWED: &[&str] = &["src/analysis.rs", "src/resolve.rs", "src/onnx/verifier.rs"];

/// Files whose only phase calls sit after a `#[cfg(test)]` marker, and are therefore not scanned.
///
/// Pinned as an exact set so that "it's only in a test" stays a checked claim. If a phase call
/// appears after a test marker in a file not named here, either a new test grew one — fine, add it —
/// or production code moved below a test module, where [`production_text`] would stop seeing it.
const TEST_ONLY: &[&str] = &[
    "src/backend/generated_rust/build.rs",
    "src/interp.rs",
    "src/resolve.rs",
    "src/typecheck.rs",
];

fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            source_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Everything before the first top-level `#[cfg(test)]`. A test may build a pipeline by hand, and
/// several do.
///
/// **Deliberately not brace-counting.** The first version of this tracked `{`/`}` to find the end of
/// the test module, and `backend/generated_rust/build.rs` broke it immediately: that file emits Rust
/// source, so its string literals are full of unbalanced braces, and the counter left the test
/// module early and reported a `#[cfg(test)]`-only call as production code. Counting braces in a
/// language with string literals and raw strings needs a lexer, not a line scan.
///
/// Instead this relies on the convention the codebase actually follows — a column-zero
/// `#[cfg(test)] mod tests` is the last item in the file — and [`assert_test_modules_come_last`]
/// checks that assumption rather than trusting it.
fn production_text(text: &str) -> String {
    match text.find("\n#[cfg(test)]") {
        Some(at) => text[..at + 1].to_string(),
        None => text.to_string(),
    }
}

/// The phase calls that [`production_text`] deliberately does not scan, pinned as an exact set.
///
/// This is the honest way to state the scan's blind spot. An earlier version tried to *prove* that
/// test modules come last by scanning for column-zero items after the marker; it reported four false
/// positives immediately — a second `#[cfg(test)] mod doctor_tests`, and a line of expected
/// diagnostic text inside a multi-line string that happened to start at column zero. Line scanning
/// cannot tell code from string content. Naming the affected files instead makes the blind spot
/// small, visible and reviewed.
#[test]
fn phase_calls_below_a_test_marker_are_a_known_set() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    source_files(&src, &mut files);
    files.sort();
    let root = src.parent().unwrap();

    let mut found: Vec<String> = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap().replace("\r\n", "\n");
        let Some(at) = text.find("\n#[cfg(test)]") else {
            continue;
        };
        let below = &text[at + 1..];
        if PHASE_CALLS.iter().any(|needle| below.contains(needle)) {
            found.push(
                file.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    found.sort();

    let mut expected: Vec<String> = TEST_ONLY.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert_eq!(
        found, expected,
        "the set of files with phase calls below a #[cfg(test)] marker changed.\n\
         A new test that builds a pipeline by hand is fine — add it to TEST_ONLY.\n\
         Production code that MOVED below a test module is not: the AS2 scan would stop seeing it."
    );
}

#[test]
fn no_production_entry_point_assembles_the_pipeline_outside_the_driver() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    source_files(&src, &mut files);
    files.sort();
    assert!(
        files.len() > 20,
        "the source scan found {} files, which cannot be right",
        files.len()
    );

    let root = src.parent().unwrap();
    let mut offenders: Vec<String> = Vec::new();
    let mut allowed_seen: Vec<String> = Vec::new();

    for file in &files {
        let relative = file
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        // Normalise line endings at the read: a CRLF checkout must not change the result.
        let text = production_text(&std::fs::read_to_string(file).unwrap().replace("\r\n", "\n"));
        let hits: Vec<&str> = PHASE_CALLS
            .iter()
            .copied()
            .filter(|needle| text.contains(needle))
            .collect();
        if hits.is_empty() {
            continue;
        }
        if ALLOWED.contains(&relative.as_str()) {
            allowed_seen.push(relative);
        } else {
            offenders.push(format!("{relative}: {hits:?}"));
        }
    }

    assert!(
        offenders.is_empty(),
        "these production files assemble the semantic pipeline outside `CompilerSession`:\n  {}\n\n\
         Use `starkc::session::CompilerSession`. If the file genuinely needs a phase directly, add\n\
         it to ALLOWED in this test with the reason — deliberately, so it is reviewed.",
        offenders.join("\n  ")
    );

    // An allowlist that matches nothing is an allowlist that has silently stopped being checked.
    let mut missing: Vec<&str> = ALLOWED
        .iter()
        .copied()
        .filter(|a| !allowed_seen.iter().any(|seen| seen == a))
        .collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "ALLOWED names files that no longer call a phase directly: {missing:?}. \
         Remove them, or the allowlist is decorative."
    );
}

#[test]
fn the_three_shipped_binaries_all_route_through_the_session() {
    // AS0 found three binaries, not two: a search of `main.rs` and `bin/stark.rs` missed
    // `starkide` entirely. Naming all three here makes that omission impossible to repeat.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for binary in ["src/main.rs", "src/bin/stark.rs", "src/bin/starkide.rs"] {
        let text = std::fs::read_to_string(root.join(binary))
            .unwrap_or_else(|e| panic!("{binary} must exist: {e}"))
            .replace("\r\n", "\n");
        assert!(
            text.contains("CompilerSession"),
            "{binary} does not go through CompilerSession"
        );
    }
}
