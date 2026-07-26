//! WP-C6.5 §10 — every corpus case, executed.
//!
//! The bridge between the corpus (commit 4) and the comparator (commit 3): each manifest case is
//! loaded, run through the engines it declares, and checked against the expectations the manifest
//! states — `expected_outcome`, `expected_stdout`, `expected_drop_log`.
//!
//! **Why the expectations are in the manifest rather than left to three-engine agreement.** A
//! sentinel (§10.3) exists to fail under the likely WRONG implementation, and a wrong
//! implementation is usually wrong in all three engines at once — a structural `Display` fallback,
//! a sorted `HashMap` iteration, a declaration-order Drop schedule. Those agree perfectly. Stating
//! the answer independently is what turns agreement into evidence.
//!
//! This is NOT §12's replay harness (commit 7), which adds admission classification, per-case
//! timeouts, sharding, replay filters and the evidence schema. It exists now so that no case is
//! added in a state where nothing runs it: a corpus file that nothing executes is exactly the
//! evidence-shaped-but-unchecked artifact this package is meant to eliminate.

mod support;

use support::corpus::{corpus_root, load, Case};
use support::differential::{rustc_available, three_engine, two_engine, Observation};

fn source_of(case: &Case) -> String {
    assert_eq!(
        case.sources.len(),
        1,
        "{}: multi-source package cases arrive with §15; this bridge runs single-source cases",
        case.case_id
    );
    std::fs::read_to_string(corpus_root().join(&case.sources[0]))
        .unwrap_or_else(|e| panic!("{}: {e}", case.sources[0]))
}

/// Runs one case and returns the reason it failed, or `None`.
fn check(case: &Case) -> Option<String> {
    let source = source_of(case);
    let wants_native = case
        .required_engines
        .iter()
        .any(|engine| engine == "native-debug");
    let observed = if wants_native {
        three_engine(&case.case_id, &source)
    } else {
        two_engine(&case.case_id, &source)
    };

    let (stdout, drop_log, trapped) = match &observed {
        Observation::Completed(done) => (&done.stdout_bytes, &done.drop_log, false),
        Observation::Trapped(trap) => (&trap.stdout_before_trap, &trap.drop_log_before_trap, true),
    };

    let expected_trap = case.expected_outcome == "trap";
    if trapped != expected_trap {
        return Some(format!(
            "{}: manifest says `{}`, engines produced {}",
            case.case_id,
            case.expected_outcome,
            if trapped { "a trap" } else { "a completion" }
        ));
    }
    if let (Observation::Trapped(trap), Some(expected)) =
        (&observed, case.expected_trap_category.as_deref())
    {
        let actual = format!("{:?}", trap.category);
        if actual != expected {
            return Some(format!(
                "{}: expected trap category {expected}, got {actual}",
                case.case_id
            ));
        }
    }
    if !case.expected_stdout.is_empty() {
        let expected = case.expected_stdout.join("\n");
        let actual = String::from_utf8_lossy(stdout);
        if actual != expected {
            return Some(format!(
                "{}: stdout\n  expected {expected:?}\n  observed {actual:?}",
                case.case_id
            ));
        }
    }
    if !case.expected_drop_log.is_empty() {
        let actual: Vec<&str> = drop_log.iter().map(|e| e.identity.as_str()).collect();
        if actual != case.expected_drop_log {
            return Some(format!(
                "{}: Drop log\n  expected {:?}\n  observed {actual:?}",
                case.case_id, case.expected_drop_log
            ));
        }
    }
    None
}

/// Every case in the manifest, in case-ID order. Reports ALL failures rather than the first: when a
/// shared change breaks several cases, which ones it broke is the diagnosis.
#[test]
fn every_corpus_case_runs_on_the_engines_it_declares() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let (cases, _) = load();
    let failures: Vec<String> = cases.iter().filter_map(check).collect();
    assert!(
        failures.is_empty(),
        "{} of {} corpus cases failed:\n\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n\n")
    );
}

/// §10.3's sentinels are the reason this phase exists: a case that would still pass under the
/// likely wrong implementation is insufficient. This asserts they are PRESENT and PINNED — a
/// sentinel with no stated expectation is back to proving only that three engines agree.
#[test]
fn every_sentinel_pins_its_observation() {
    let (cases, _) = load();
    let sentinels: Vec<&Case> = cases
        .iter()
        .filter(|c| c.case_id.starts_with("sentinel__"))
        .collect();
    assert!(
        sentinels.len() >= 13,
        "§10.3 lists thirteen sentinel shapes; the corpus has {}",
        sentinels.len()
    );
    for case in sentinels {
        assert!(
            !case.expected_stdout.is_empty() || !case.expected_drop_log.is_empty(),
            "{}: a sentinel must pin its observation independently of the engines",
            case.case_id
        );
    }
}
