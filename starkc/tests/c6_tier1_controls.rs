//! WP-C6.5 §20.7 — controls on the Tier-1 corpus comparator.
//!
//! The comparator is the last thing standing between "two green jobs" and a claim that two targets
//! agree. Its own failure mode is silent acceptance, so each control below hands it a pair of
//! records that must be REJECTED and requires it to say why. The twelfth hands it a valid pair and
//! requires acceptance — without that one, "rejects everything" would pass all the others.
//!
//! Records are built here rather than captured from a run: a control needs to differ from agreement
//! in exactly one respect, and editing a real record is how you end up testing two differences at
//! once.

use std::path::{Path, PathBuf};
use std::process::Command;

fn script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/compare-c65-evidence.py")
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stark_c65tier1_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// A summary record that agrees with its partner in every respect the comparator checks.
fn summary(triple: &str, os: &str, arch: &str) -> String {
    format!(
        r#"{{
  "schema_version": "c6.5-evidence-1",
  "commit_sha": "abc123",
  "corpus_version": "0.5.0",
  "generator_version": "0.1.0",
  "seed": "c6.5-default",
  "manifest_sha256": "aaaa",
  "generator_sha256": "bbbb",
  "case_count": 2,
  "handwritten_count": 1,
  "generated_count": 1,
  "retained_count": 0,
  "metamorphic_family_count": 0,
  "metamorphic_group_count": 0,
  "mutation_count": 0,
  "passed_count": 2,
  "failed_count": 0,
  "skipped_count": 0,
  "quarantined_count": 0,
  "full_evidence": true,
  "target_triple": "{triple}",
  "os": "{os}",
  "architecture": "{arch}",
  "rustc": "rustc 1.93.0",
  "cargo": "cargo 1.93.0",
  "python": "Python 3.14.4",
  "mir_version": "0.1",
  "backend_version": "0.1",
  "runtime_version": "0.1",
  "dirty_worktree": "false",
  "result": "PASS"
}}"#
    )
}

fn per_case(second_hash: &str) -> String {
    format!(
        r#"[
  {{"case_id": "one", "result": "AGREEMENT", "observation_hash": "1111"}},
  {{"case_id": "two", "result": "AGREEMENT", "observation_hash": "{second_hash}"}}
]"#
    )
}

struct Verdict {
    accepted: bool,
    report: String,
}

fn run(dir: &Path, left: &str, right: &str, cases: Option<(&str, &str)>) -> Verdict {
    std::fs::write(dir.join("a.json"), left).expect("write a");
    std::fs::write(dir.join("b.json"), right).expect("write b");
    let mut command = Command::new("python3");
    command
        .arg(script())
        .arg(dir.join("a.json"))
        .arg(dir.join("b.json"));
    if let Some((a, b)) = cases {
        std::fs::write(dir.join("a-cases.json"), a).expect("write a cases");
        std::fs::write(dir.join("b-cases.json"), b).expect("write b cases");
        command
            .arg("--per-case")
            .arg(dir.join("a-cases.json"))
            .arg(dir.join("b-cases.json"));
    }
    let output = command.output().expect("run the comparator");
    Verdict {
        accepted: output.status.success(),
        report: String::from_utf8_lossy(&output.stdout).into_owned(),
    }
}

const MAC: (&str, &str, &str) = ("aarch64-apple-darwin", "macos", "aarch64");
const LINUX: (&str, &str, &str) = ("x86_64-unknown-linux-gnu", "linux", "x86_64");

#[track_caller]
fn rejects(tag: &str, right: String, expect: &str) {
    let dir = scratch(tag);
    let verdict = run(
        &dir,
        &summary(MAC.0, MAC.1, MAC.2),
        &right,
        Some((&per_case("2222"), &per_case("2222"))),
    );
    assert!(
        !verdict.accepted,
        "{tag}: the comparator ACCEPTED records it must reject\n{}",
        verdict.report
    );
    assert!(
        verdict.report.contains(expect),
        "{tag}: rejected for the wrong reason — wanted {expect:?}, got:\n{}",
        verdict.report
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The control that makes the other eleven meaningful.
#[test]
fn valid_tier1_agreement_accepted() {
    let dir = scratch("valid");
    let verdict = run(
        &dir,
        &summary(MAC.0, MAC.1, MAC.2),
        &summary(LINUX.0, LINUX.1, LINUX.2),
        Some((&per_case("2222"), &per_case("2222"))),
    );
    assert!(
        verdict.accepted,
        "the comparator rejected a valid Tier-1 pair:\n{}",
        verdict.report
    );
    assert!(verdict.report.contains("TIER-1 CORPUS AGREEMENT"));
    // Platform differences are reported, not treated as disagreement.
    assert!(verdict.report.contains("Platform metadata"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn same_platform_twice_rejected() {
    rejects(
        "same_platform",
        summary(MAC.0, MAC.1, MAC.2),
        "same target triple",
    );
}

#[test]
fn different_commit_rejected() {
    rejects(
        "commit",
        summary(LINUX.0, LINUX.1, LINUX.2).replace("abc123", "def456"),
        "commit_sha differs",
    );
}

#[test]
fn different_corpus_version_rejected() {
    rejects(
        "corpus_version",
        summary(LINUX.0, LINUX.1, LINUX.2).replace("\"0.5.0\"", "\"0.4.0\""),
        "corpus_version differs",
    );
}

#[test]
fn different_seed_rejected() {
    rejects(
        "seed",
        summary(LINUX.0, LINUX.1, LINUX.2).replace("c6.5-default", "other-seed"),
        "seed differs",
    );
}

#[test]
fn different_manifest_hash_rejected() {
    rejects(
        "manifest_hash",
        summary(LINUX.0, LINUX.1, LINUX.2).replace("\"aaaa\"", "\"cccc\""),
        "manifest_sha256 differs",
    );
}

#[test]
fn dirty_worktree_rejected() {
    rejects(
        "dirty",
        summary(LINUX.0, LINUX.1, LINUX.2).replace(
            "\"dirty_worktree\": \"false\"",
            "\"dirty_worktree\": \"true\"",
        ),
        "DIRTY worktree",
    );
}

#[test]
fn filtered_run_rejected() {
    rejects(
        "filtered",
        summary(LINUX.0, LINUX.1, LINUX.2)
            .replace("\"full_evidence\": true", "\"full_evidence\": false")
            .replace("\"result\": \"PASS\"", "\"result\": \"PARTIAL-FILTERED\""),
        "not FULL evidence",
    );
}

#[test]
fn a_skip_rejected() {
    rejects(
        "skip",
        summary(LINUX.0, LINUX.1, LINUX.2).replace("\"skipped_count\": 0", "\"skipped_count\": 1"),
        "skipped case",
    );
}

#[test]
fn a_failure_rejected() {
    rejects(
        "failure",
        summary(LINUX.0, LINUX.1, LINUX.2)
            .replace("\"failed_count\": 0", "\"failed_count\": 1")
            .replace("\"result\": \"PASS\"", "\"result\": \"FAIL\""),
        "failed case",
    );
}

#[test]
fn missing_evidence_file_rejected() {
    let dir = scratch("missing");
    std::fs::write(dir.join("a.json"), summary(MAC.0, MAC.1, MAC.2)).expect("write");
    let output = Command::new("python3")
        .arg(script())
        .arg(dir.join("a.json"))
        .arg(dir.join("absent.json"))
        .output()
        .expect("run");
    assert!(
        !output.status.success(),
        "a missing record was accepted as agreement"
    );
    let report = String::from_utf8_lossy(&output.stdout);
    assert!(
        report.contains("does not exist") && report.contains("missing record is not a pass"),
        "unexpected report: {report}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The strongest control: the summaries agree completely, every count matches, both say PASS — and
/// one case observed different bytes. A comparator that only read summaries would accept this.
#[test]
fn different_per_case_observation_rejected() {
    let dir = scratch("observation");
    let verdict = run(
        &dir,
        &summary(MAC.0, MAC.1, MAC.2),
        &summary(LINUX.0, LINUX.1, LINUX.2),
        Some((&per_case("2222"), &per_case("3333"))),
    );
    assert!(
        !verdict.accepted,
        "two targets observed different bytes and the comparator accepted it:\n{}",
        verdict.report
    );
    assert!(
        verdict
            .report
            .contains("SAME outcome class but DIFFERENT observation"),
        "unexpected report: {}",
        verdict.report
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A case present on one target and absent on the other — the "missing shard" shape, which at the
/// per-case level is indistinguishable from a case that silently did not run.
#[test]
fn a_case_missing_from_one_target_rejected() {
    let dir = scratch("missing_case");
    let short = r#"[{"case_id": "one", "result": "AGREEMENT", "observation_hash": "1111"}]"#;
    let verdict = run(
        &dir,
        &summary(MAC.0, MAC.1, MAC.2),
        &summary(LINUX.0, LINUX.1, LINUX.2),
        Some((&per_case("2222"), short)),
    );
    assert!(
        !verdict.accepted,
        "a case that ran on only one target was accepted:\n{}",
        verdict.report
    );
    assert!(
        verdict.report.contains("ran only on"),
        "unexpected report: {}",
        verdict.report
    );
    let _ = std::fs::remove_dir_all(&dir);
}
