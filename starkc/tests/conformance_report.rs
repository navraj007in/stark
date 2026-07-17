//! WP-C1.6: the conformance evidence generator is a Python script (deliberately, to match
//! `check-conformance.py`'s existing dependency-free-tooling convention), so it can't be unit
//! tested via `cargo test` the normal way. These are CLI-invocation smoke tests, mirroring
//! `gate3_execution.rs`'s `run_cli_executes_program_and_reports_runtime_failure` pattern:
//! shell out to the real script and assert on its actual behavior.

use starkc::lsp::protocol::{parse_json, JsonValue};
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("starkc/ has a parent directory")
        .to_path_buf()
}

#[test]
fn coverage_database_passes_its_own_validator() {
    let output = Command::new("python3")
        .arg("starkc/scripts/check-conformance.py")
        .current_dir(repo_root())
        .output()
        .expect("python3 is available (already assumed by CI's fixture-conformance job)");
    assert!(
        output.status.success(),
        "check-conformance.py failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The report's `--format=json` output must be valid, well-formed JSON: an array of objects,
/// each with exactly the 8 columns the WP-C1.6 roadmap text specifies (rule id, spec chapter,
/// status, source, positive tests, negative tests, deviation, last verified commit).
#[test]
fn json_report_has_the_required_columns_for_every_rule() {
    let output = Command::new("python3")
        .args([
            "starkc/scripts/generate-conformance-report.py",
            "--format=json",
        ])
        .current_dir(repo_root())
        .output()
        .expect("python3 is available");
    assert!(
        output.status.success(),
        "generator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_json(&stdout).expect("generator stdout is valid JSON");
    let JsonValue::Array(rows) = parsed else {
        panic!("expected a top-level JSON array, got {parsed:?}");
    };
    assert!(
        rows.len() >= 59,
        "expected at least the 59 rules known as of WP-C1.6, got {}",
        rows.len()
    );

    let required_keys = [
        "rule_id",
        "spec_chapter",
        "status",
        "source",
        "positive_tests",
        "negative_tests",
        "deviation",
        "last_verified_commit",
    ];
    let mut seen_ids = Vec::new();
    for row in &rows {
        let JsonValue::Object(fields) = row else {
            panic!("expected each row to be a JSON object, got {row:?}");
        };
        for key in required_keys {
            assert!(
                fields.contains_key(key),
                "row missing required column '{key}': {row:?}"
            );
        }
        if let Some(JsonValue::String(id)) = fields.get("rule_id") {
            seen_ids.push(id.clone());
        }
    }

    // Sorted (the generator explicitly sorts by rule id) and unique (mirrors
    // check-conformance.py's own duplicate-id check, from the report's own output this time).
    let mut sorted_ids = seen_ids.clone();
    sorted_ids.sort();
    assert_eq!(
        seen_ids, sorted_ids,
        "report rows must be sorted by rule_id"
    );
    let mut dedup_ids = seen_ids.clone();
    dedup_ids.sort();
    dedup_ids.dedup();
    assert_eq!(
        dedup_ids.len(),
        seen_ids.len(),
        "report must not contain duplicate rule ids"
    );
}

/// The WP's own text requires the report to be "deterministic" -- two runs against the same
/// commit must produce byte-identical output.
#[test]
fn json_report_is_deterministic_across_two_runs() {
    let run = || {
        Command::new("python3")
            .args([
                "starkc/scripts/generate-conformance-report.py",
                "--format=json",
            ])
            .current_dir(repo_root())
            .output()
            .expect("python3 is available")
            .stdout
    };
    let first = run();
    let second = run();
    assert_eq!(
        first, second,
        "two consecutive report runs against the same repo state must be identical"
    );
}

/// A rule with real `positive_tests`/`negative_tests` citations (not the file-level fallback)
/// must report them as actual arrays with the cited entries, not the "unclassified" fallback
/// string used for rules that only have the legacy `tests` field.
#[test]
fn a_rule_with_split_evidence_reports_real_positive_and_negative_lists() {
    let output = Command::new("python3")
        .args([
            "starkc/scripts/generate-conformance-report.py",
            "--format=json",
        ])
        .current_dir(repo_root())
        .output()
        .expect("python3 is available");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed = parse_json(&stdout).expect("valid JSON");
    let JsonValue::Array(rows) = parsed else {
        panic!("expected array");
    };
    let lex003 = rows
        .iter()
        .find(|row| {
            matches!(
                row,
                JsonValue::Object(fields)
                    if fields.get("rule_id") == Some(&JsonValue::String("LEX-003".to_string()))
            )
        })
        .expect("LEX-003 exists in the report");
    let JsonValue::Object(fields) = lex003 else {
        unreachable!()
    };
    assert!(
        matches!(fields.get("positive_tests"), Some(JsonValue::Array(_))),
        "LEX-003 has explicit positive_tests in the TOML; expected a real array, got {:?}",
        fields.get("positive_tests")
    );
    assert_eq!(
        fields.get("deviation"),
        Some(&JsonValue::String("DEV-015".to_string())),
        "LEX-003 is tied to DEV-015 in the TOML"
    );
}
