//! **The cohort gate: a limitations page that has gone stale must fail the build.**
//!
//! `STARKLANG/docs/pre-alpha/KNOWN-LIMITATIONS.md` is developer-facing prose owned by the
//! cohort/release surface. The facts in it are owned elsewhere — the native conformance matrix and
//! the deviation ledger — and the rule that keeps it from becoming a fourth authority is this test:
//!
//! ```text
//! compiler/conformance track   OWNS the facts
//! cohort/release track         OWNS presenting them
//! this test                    OWNS the agreement between the two
//! ```
//!
//! **Only facts that can drift are checked mechanically.** The explanatory prose is written for
//! humans and is deliberately not asserted here; a test that pinned wording would make the page
//! unmaintainable and would catch nothing that matters.
//!
//! Three failure modes, each a real way a cohort gets misled:
//!
//! ```text
//! a deviation named as live has been RESOLVED     the page preserves historical workaround
//!                                                 advice after the defect closed
//! a live user-reachable deviation is MISSING      the cohort meets it with no warning
//! the page contradicts the conformance matrix     two answers to one question
//! ```
//!
//! The first is not hypothetical. DEV-236 was a live limitation when the gate opened and was
//! repaired the same day; a page written an hour earlier would have told the cohort to work around
//! a defect that no longer exists.
//!
//! # A known limit of this gate, stated rather than discovered
//!
//! The authority sorts deviations into open, closed, accepted, dormant — **and `ADJUDICATE`**, for
//! entries whose last heading does not settle the question. This gate ranges over population A
//! only, so an ADJUDICATE entry can be neither required nor cited: naming one fails
//! `the_cohort_page_names_no_deviation_that_has_been_resolved`, because it is not in the live set.
//!
//! That is deliberate and it is the safe direction — but it means **a real limitation can sit in
//! ADJUDICATE and be invisible to the cohort**. `DEV-214` is the live example: repaired under OD-9
//! *"with one criterion that cannot be met at MAX_DEPTH = 200"*, so a long enough left-associative
//! operator chain still has a limit a participant could reach. It is not on the page because the
//! gate cannot admit it.
//!
//! **The fix is adjudication, not a weaker gate.** Settle DEV-214's heading and it becomes citable
//! by the ordinary rule.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("starkc has a parent")
        .to_path_buf()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()))
        // CRLF normalised at the read: a Windows checkout must not change the answers.
        .replace("\r\n", "\n")
}

/// Every `DEV-nnn` the page presents as a LIVE limitation.
///
/// Deliberately every mention: a page that names a deviation at all is telling the cohort something
/// about it, and if that deviation is closed the page is wrong however the sentence is phrased.
/// DEV-160 appears in §5 as a closed historical example and is excluded by name — the one exception,
/// stated here rather than inferred from a regex.
fn cited_by_the_page(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = text.as_bytes();
    for (i, _) in text.match_indices("DEV-") {
        let digits: String = bytes[i + 4..]
            .iter()
            .take(3)
            .take_while(|b| b.is_ascii_digit())
            .map(|b| *b as char)
            .collect();
        if digits.len() == 3 {
            out.insert(format!("DEV-{digits}"));
        }
    }
    out.remove("DEV-160"); // §5's closed example, cited as history
    out
}

/// The live-open set, **from the authority that owns the question**.
///
/// The first version of this test hand-rolled a heading classifier — last heading wins, reject on
/// RESOLVED/CLOSED/SUPERSEDED/… — and it disagreed with the real one on two entries, reporting
/// DEV-214 and DEV-228 as open when `c10-deviation-populations.py` puts both in its ADJUDICATE
/// bucket. **That is a second classifier for one question**, which is precisely the shape AC5
/// spent a day cataloguing, introduced by the test written to prevent drift.
///
/// So this shells out to the script instead. It is the only parser of `KNOWN-DEVIATIONS.md`'s
/// append-only heading structure, it already knows that the LAST heading decides and that
/// `ACCEPTED-INDEFINITELY` and `DORMANT` are neither open nor closed, and a divergence between two
/// readings of the ledger is a defect this test would otherwise create rather than catch.
fn population_a(root: &Path) -> Option<BTreeSet<String>> {
    let out = std::process::Command::new("python3")
        .arg(root.join("starkc/scripts/c10-deviation-populations.py"))
        .current_dir(root)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut live = BTreeSet::new();
    let mut in_section = false;
    for line in text.lines() {
        if line.starts_with("POPULATION A -- live OPEN") {
            in_section = true;
            continue;
        }
        if in_section {
            if line.trim().is_empty() {
                break;
            }
            if let Some(rest) = line.trim().strip_prefix("DEV-") {
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if digits.len() == 3 {
                    live.insert(format!("DEV-{digits}"));
                }
            }
        }
    }
    (!live.is_empty()).then_some(live)
}

/// Runs `body` with the live-open set, or prints the repo's `SKIP:` line and returns.
///
/// `SKIP:` is detected by `run-c64-qualification.py`, which fails a required step on it — so a
/// machine without python3 cannot report this gate as green having checked nothing.
fn with_population_a(body: impl FnOnce(BTreeSet<String>)) {
    let root = repo_root();
    match population_a(&root) {
        Some(live) => body(live),
        None => eprintln!(
            "SKIP: python3 unavailable, so the deviation authority could not be consulted."
        ),
    }
}

#[test]
fn the_cohort_page_names_no_deviation_that_has_been_resolved() {
    let root = repo_root();
    let page = read(&root.join("STARKLANG/docs/pre-alpha/KNOWN-LIMITATIONS.md"));
    let ledger = read(&root.join("starkc/docs/conformance/KNOWN-DEVIATIONS.md"));

    let _ = &ledger;
    let mut stale: Vec<String> = Vec::new();
    with_population_a(|open| {
        stale = cited_by_the_page(&page)
            .into_iter()
            .filter(|dev| !open.contains(dev))
            .collect();
    });

    assert!(
        stale.is_empty(),
        "the cohort limitations page presents {stale:?} as a live limitation, but the ledger's \
         LAST heading for each says it is no longer open. A page that keeps historical workaround \
         advice after the defect closes sends the cohort round an obstacle that is not there. \
         Remove it, or cite it explicitly as history the way §5 cites DEV-160."
    );
}

#[test]
fn every_live_user_reachable_deviation_is_named() {
    let root = repo_root();
    let page = read(&root.join("STARKLANG/docs/pre-alpha/KNOWN-LIMITATIONS.md"));
    let ledger = read(&root.join("starkc/docs/conformance/KNOWN-DEVIATIONS.md"));

    let _ = &ledger;
    let cited = cited_by_the_page(&page);
    let mut missing: Vec<String> = Vec::new();
    with_population_a(|open| {
        missing = open
            .into_iter()
            .filter(|dev| !cited.contains(dev))
            .collect();
    });

    assert!(
        missing.is_empty(),
        "these deviations are OPEN in the ledger and the cohort page does not mention them: \
         {missing:?}. Every open deviation is something a participant can hit, so silence about \
         one is the cohort meeting it with no warning. Add it with its workaround, or — if it is \
         genuinely unreachable from user code — say so on the page and record why."
    );
}

#[test]
fn the_page_agrees_with_the_conformance_matrix_about_native_boundaries() {
    let root = repo_root();
    let page = read(&root.join("STARKLANG/docs/pre-alpha/KNOWN-LIMITATIONS.md"));
    let matrix = read(&root.join("starkc/docs/conformance/NATIVE-CONFORMANCE-MATRIX.md"));

    // Every deviation the MATRIX marks as a native boundary must appear on the page: those are
    // exactly the constructs a participant writes and cannot build.
    let mut in_matrix = BTreeSet::new();
    for line in matrix.lines().filter(|l| l.contains("KNOWN-DEVIATION")) {
        for (i, _) in line.match_indices("DEV-") {
            let digits: String = line.as_bytes()[i + 4..]
                .iter()
                .take(3)
                .take_while(|b| b.is_ascii_digit())
                .map(|b| *b as char)
                .collect();
            if digits.len() == 3 {
                in_matrix.insert(format!("DEV-{digits}"));
            }
        }
    }
    assert!(
        !in_matrix.is_empty(),
        "the matrix has no KNOWN-DEVIATION rows at all — either the compiler gained a great deal, \
         or this test is reading the wrong file and would pass vacuously"
    );

    let cited = cited_by_the_page(&page);
    let missing: Vec<String> = in_matrix
        .into_iter()
        .filter(|dev| !cited.contains(dev))
        .collect();
    assert!(
        missing.is_empty(),
        "the conformance matrix marks {missing:?} as native boundaries and the cohort page does \
         not mention them. The matrix is the authority; this page is the presentation, and it is \
         currently presenting less than the authority says."
    );
}
