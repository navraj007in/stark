//! WP-C6.5 §12 — the full three-engine corpus replay.
//!
//! The plan's §12.1 entry point, and the qualification path for the corpus: load and validate the
//! manifest, verify the lock, enumerate required cases in case-ID order, run each on the engines it
//! declares, normalise and compare observations, and write machine-readable evidence.
//!
//! It replaces the C6.5-3 bridge (`c6_corpus_cases.rs`), which ran cases but produced no evidence,
//! applied no timeout, and could not be narrowed.
//!
//! **Three properties this harness has that a plain loop would not:**
//!
//! 1. **A failure is CLASSIFIED** (§12.2) — parse, resolution, typecheck, lowering refusal, MIR
//!    verify/internal, native linkage/build/runtime, or observation divergence. The distinction is
//!    load-bearing: an accepted Core case refused by MIR or native is a *blocker*, not a skip, and
//!    the classification is what makes the difference visible instead of both looking like "failed".
//! 2. **A filtered run cannot be mistaken for closure evidence** (§12.6). Every narrowing — a single
//!    case, a category, a shard — is recorded in the summary, and the summary's `result` says
//!    `PARTIAL-FILTERED` rather than `PASS`.
//! 3. **A timeout is a failure, not a skip** (§12.4). Per-case work runs on a worker thread with a
//!    deadline; exceeding it fails the case with the budget stated.
//!
//! Filters (all optional, all diagnostic):
//!
//! ```text
//! C6_CASE=<case-id>   C6_CATEGORY=<category>   C6_TEMPLATE=<T##>   C6_KIND=<kind>
//! C6_ENGINE=hir|mir|native|all     C6_SHARD_INDEX=<i> C6_SHARD_TOTAL=<n>     C6_KEEP_TEMP=1
//! C6_EVIDENCE_DIR=<dir>            (default: target/c6.5-evidence)
//! ```

mod support;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use support::corpus::{corpus_root, load, sha256_hex, shard_of, verify_lock, Case, Filters};
use support::differential::{
    canonical_form, first_difference, run_hir, run_mir, run_native, rustc_available, Observation,
};

/// §12.4. Generous enough that a slow machine is not a false failure, bounded enough that a hang is
/// a failure rather than a CI job burning its whole budget — the shape CD-127's infinite-loop
/// miscompile took.
const PER_CASE_BUDGET: Duration = Duration::from_secs(120);
const WHOLE_REPLAY_BUDGET: Duration = Duration::from_secs(3600);

/// §12.2 admission classifications. Every failure is one of these; there is no "other".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Admission {
    ParseDivergence,
    ResolutionDivergence,
    TypecheckDivergence,
    LoweringRefusal,
    MirVerifyFailure,
    MirInternalFailure,
    NativeLinkageRefusal,
    NativeBuildFailure,
    NativeRuntimeFailure,
    ObservationDivergence,
    Timeout,
}

impl Admission {
    fn as_str(self) -> &'static str {
        match self {
            Admission::ParseDivergence => "PARSE-DIVERGENCE",
            Admission::ResolutionDivergence => "RESOLUTION-DIVERGENCE",
            Admission::TypecheckDivergence => "TYPECHECK-DIVERGENCE",
            Admission::LoweringRefusal => "LOWERING-REFUSAL",
            Admission::MirVerifyFailure => "MIR-VERIFY-FAILURE",
            Admission::MirInternalFailure => "MIR-INTERNAL-FAILURE",
            Admission::NativeLinkageRefusal => "NATIVE-LINKAGE-REFUSAL",
            Admission::NativeBuildFailure => "NATIVE-BUILD-FAILURE",
            Admission::NativeRuntimeFailure => "NATIVE-RUNTIME-FAILURE",
            Admission::ObservationDivergence => "OBSERVATION-DIVERGENCE",
            Admission::Timeout => "TIMEOUT",
        }
    }

    /// §12.2: "an accepted Core case refused by MIR/native is a blocker". The classification exists
    /// so that refusal is reported as a gate-blocking fact rather than as an inconvenience.
    fn is_blocker(self) -> bool {
        matches!(
            self,
            Admission::LoweringRefusal
                | Admission::MirVerifyFailure
                | Admission::MirInternalFailure
                | Admission::NativeLinkageRefusal
                | Admission::NativeBuildFailure
        )
    }
}

/// Classifies a panic message from the engine runners into a §12.2 admission. The runners assert
/// with messages that name their stage, so the mapping is by stage rather than by guesswork; an
/// unrecognised message classifies as an observation divergence and prints in full, which is loud
/// rather than silent.
fn classify(message: &str) -> Admission {
    if message.contains(": parse:") {
        Admission::ParseDivergence
    } else if message.contains(": resolve:") {
        Admission::ResolutionDivergence
    } else if message.contains(": typecheck:") {
        Admission::TypecheckDivergence
    } else if message.contains("lowering failed") {
        Admission::LoweringRefusal
    } else if message.contains("verifier rejected") {
        Admission::MirVerifyFailure
    } else if message.contains("MIR internal error") {
        Admission::MirInternalFailure
    } else if message.contains("must return Unit") || message.contains("linkage") {
        Admission::NativeLinkageRefusal
    } else if message.contains("native build failed") {
        Admission::NativeBuildFailure
    } else if message.contains("terminated by a signal")
        || message.contains("running the generated")
    {
        Admission::NativeRuntimeFailure
    } else {
        Admission::ObservationDivergence
    }
}

struct CaseResult {
    case_id: String,
    engines: Vec<(&'static str, String)>,
    observation_hash: String,
    outcome: Result<(), (Admission, String)>,
}

fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_list(items: &[String]) -> String {
    let rendered: Vec<String> = items
        .iter()
        .map(|item| format!("\"{}\"", json_escape(item)))
        .collect();
    format!("[{}]", rendered.join(", "))
}

/// Runs one case on the engines it declares. Every engine invocation is wrapped, so an engine that
/// panics becomes a classified failure rather than aborting the replay — §12.1's "continue through
/// all cases and produce a complete summary".
fn replay(case: &Case, filters: &Filters) -> CaseResult {
    let source_path = corpus_root().join(&case.sources[0]);
    let source = match std::fs::read_to_string(&source_path) {
        Ok(text) => text,
        Err(e) => {
            return CaseResult {
                case_id: case.case_id.clone(),
                engines: Vec::new(),
                observation_hash: String::new(),
                outcome: Err((
                    Admission::ObservationDivergence,
                    format!("cannot read {}: {e}", case.sources[0]),
                )),
            }
        }
    };
    let name = format!("{}.stark", case.case_id);
    let wants = |engine: &str| {
        let requested = filters.engine.as_deref().unwrap_or("all");
        let declared = case.required_engines.iter().any(|e| e == engine);
        declared
            && (requested == "all"
                || requested == engine
                || (requested == "native" && engine == "native-debug"))
    };

    let mut engines: Vec<(&'static str, String)> = Vec::new();
    let mut observations: Vec<(&'static str, Observation)> = Vec::new();

    macro_rules! stage {
        ($label:literal, $body:expr) => {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
                Ok(value) => value,
                Err(payload) => {
                    let message = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "<non-string panic>".to_string());
                    engines.push(($label, "FAIL".to_string()));
                    return CaseResult {
                        case_id: case.case_id.clone(),
                        engines,
                        observation_hash: String::new(),
                        outcome: Err((classify(&message), message)),
                    };
                }
            }
        };
    }

    let front = stage!(
        "front-end",
        support::differential::front_end(&name, &source)
    );
    let program = stage!("front-end", {
        match starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone()) {
            Ok(program) => program,
            Err(e) => panic!("{name}: lowering failed: {} @ {:?}", e.what, e.span),
        }
    });

    if wants("hir") {
        let observed = stage!("hir", run_hir(&name, &front));
        engines.push(("hir", "PASS".to_string()));
        observations.push(("hir", observed));
    }
    if wants("mir") {
        let observed = stage!("mir", run_mir(&name, &program));
        engines.push(("mir", "PASS".to_string()));
        observations.push(("mir", observed));
    }
    if wants("native-debug") {
        let observed = stage!("native-debug", run_native(&name, &case.case_id, &program));
        engines.push(("native-debug", "PASS".to_string()));
        observations.push(("native-debug", observed));
    }

    // §12.10: compare every adjacent pair, so a three-engine run reports WHICH pair diverged.
    for window in observations.windows(2) {
        let (left_name, left) = &window[0];
        let (right_name, right) = &window[1];
        if let Some(field) = first_difference(left, right) {
            return CaseResult {
                case_id: case.case_id.clone(),
                engines,
                observation_hash: String::new(),
                outcome: Err((
                    Admission::ObservationDivergence,
                    format!(
                        "{left_name}/{right_name} disagree on {field}\n--- {left_name} ---\n{left:#?}\n\
                         --- {right_name} ---\n{right:#?}"
                    ),
                )),
            };
        }
    }

    // The manifest's own expectations: agreement is necessary, not sufficient.
    let mut expectation_failure = None;
    if let Some((_, observed)) = observations.first() {
        let (stdout, drop_log, trapped) = match observed {
            Observation::Completed(done) => (&done.stdout_bytes, &done.drop_log, false),
            Observation::Trapped(trap) => {
                (&trap.stdout_before_trap, &trap.drop_log_before_trap, true)
            }
        };
        if trapped != (case.expected_outcome == "trap") {
            expectation_failure = Some(format!(
                "manifest says `{}`, engines produced {}",
                case.expected_outcome,
                if trapped { "a trap" } else { "a completion" }
            ));
        }
        if let (Observation::Trapped(trap), Some(expected)) =
            (observed, case.expected_trap_category.as_deref())
        {
            let actual = format!("{:?}", trap.category);
            if actual != expected && expectation_failure.is_none() {
                expectation_failure =
                    Some(format!("expected trap category {expected}, got {actual}"));
            }
        }
        if !case.expected_stdout.is_empty() && expectation_failure.is_none() {
            let expected = case.expected_stdout.join("\n");
            let actual = String::from_utf8_lossy(stdout);
            if actual != expected {
                expectation_failure = Some(format!(
                    "stdout: expected {expected:?}, observed {actual:?}"
                ));
            }
        }
        if !case.expected_drop_log.is_empty() && expectation_failure.is_none() {
            let actual: Vec<&str> = drop_log.iter().map(|e| e.identity.as_str()).collect();
            if actual != case.expected_drop_log {
                expectation_failure = Some(format!(
                    "Drop log: expected {:?}, observed {actual:?}",
                    case.expected_drop_log
                ));
            }
        }
    }

    let observation_hash = observations
        .first()
        .map(|(_, observed)| sha256_hex(canonical_form(observed).as_bytes()))
        .unwrap_or_default();

    CaseResult {
        case_id: case.case_id.clone(),
        engines,
        observation_hash,
        outcome: match expectation_failure {
            Some(reason) => Err((Admission::ObservationDivergence, reason)),
            None => Ok(()),
        },
    }
}

/// §12.5's divergence report.
fn report(case: &Case, admission: Admission, detail: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("\n=== {} ===\n", case.case_id));
    out.push_str(&format!("classification  {}\n", admission.as_str()));
    if admission.is_blocker() {
        out.push_str("                 ^ an accepted Core case refused by an engine is a C6 BLOCKER (§12.2)\n");
    }
    out.push_str(&format!("category        {}\n", case.category));
    out.push_str(&format!("kind            {}\n", case.kind));
    if let Some(template) = &case.template_id {
        out.push_str(&format!(
            "generator       template {template}, seed {}, version {}\n",
            case.generator_seed.as_deref().unwrap_or("-"),
            case.generator_version.as_deref().unwrap_or("-"),
        ));
    }
    out.push_str(&format!("sources         {}\n", case.sources.join(", ")));
    out.push_str(&format!("package graph   {}\n", case.package_graph));
    out.push_str(&format!(
        "normative rules {}\n",
        case.normative_rules.join(", ")
    ));
    out.push_str(&format!(
        "reproduce       C6_CASE={} cargo test --test c6_generated_corpus\n",
        case.case_id
    ));
    out.push_str(&format!(
        "retention       cases/retained/<DEV-ID>/original/{}\n",
        case.sources[0].rsplit('/').next().unwrap_or("case.stark")
    ));
    out.push_str("detail\n");
    for line in detail.lines() {
        out.push_str(&format!("  {line}\n"));
    }
    out
}

fn evidence_dir() -> PathBuf {
    match std::env::var("C6_EVIDENCE_DIR") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/c6.5-evidence"),
    }
}

/// §12.1 / §21. The replay.
#[test]
fn the_corpus_replays_through_every_required_engine() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let started = Instant::now();
    let filters = Filters::from_env();
    let (cases, lock) = load();
    verify_lock(&corpus_root(), &lock, &cases).expect("corpus.lock must match before replay");

    let selected: Vec<&Case> = cases.iter().filter(|case| filters.selects(case)).collect();
    assert!(
        !selected.is_empty(),
        "no case matched the filters {filters:?} — a filtered run that selects nothing is a \
         mistyped filter, not a pass"
    );

    let mut results = Vec::new();
    let mut failures = Vec::new();
    for case in &selected {
        // §12.4: the per-case budget is enforced on a worker thread. A hung native binary fails the
        // case with its budget named instead of stalling the run; the thread is abandoned rather
        // than killed, which is deliberate — a leaked thread at the end of a failing test run is a
        // smaller problem than an unattributable hang.
        let (sender, receiver) = std::sync::mpsc::channel();
        let case_copy = (*case).clone();
        let filters_copy = filters.clone();
        std::thread::spawn(move || {
            let _ = sender.send(replay(&case_copy, &filters_copy));
        });
        let result = match receiver.recv_timeout(PER_CASE_BUDGET) {
            Ok(result) => result,
            Err(_) => CaseResult {
                case_id: case.case_id.clone(),
                engines: vec![("native-debug", "TIMEOUT".to_string())],
                observation_hash: String::new(),
                outcome: Err((
                    Admission::Timeout,
                    format!(
                        "exceeded the {}s per-case budget (§12.4: a timeout is a failure, not a skip)",
                        PER_CASE_BUDGET.as_secs()
                    ),
                )),
            },
        };
        if let Err((admission, detail)) = &result.outcome {
            failures.push(report(case, *admission, detail));
        }
        results.push((case, result));
        assert!(
            started.elapsed() < WHOLE_REPLAY_BUDGET,
            "the whole replay exceeded its {}s budget",
            WHOLE_REPLAY_BUDGET.as_secs()
        );
    }

    write_evidence(&results, &filters, &lock);

    assert!(
        failures.is_empty(),
        "{} of {} replayed cases failed:\n{}",
        failures.len(),
        selected.len(),
        failures.join("")
    );
}

fn write_evidence(
    results: &[(&&Case, CaseResult)],
    filters: &Filters,
    lock: &support::corpus::Lock,
) {
    let dir = evidence_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let mut per_case = Vec::new();
    for (case, result) in results {
        let engines: Vec<String> = result
            .engines
            .iter()
            .map(|(name, verdict)| format!("\"{name}\": {{\"result\": \"{verdict}\"}}"))
            .collect();
        per_case.push(format!(
            "{{\"case_id\": \"{}\", \"category\": \"{}\", \"kind\": \"{}\", \"template_id\": {}, \
             \"generator_version\": {}, \"seed\": {}, \"normative_rules\": {}, \"engines\": {{{}}}, \
             \"observation_hash\": \"{}\", \"result\": \"{}\"}}",
            json_escape(&result.case_id),
            json_escape(&case.category),
            json_escape(&case.kind),
            case.template_id
                .as_deref()
                .map(|t| format!("\"{t}\""))
                .unwrap_or_else(|| "null".to_string()),
            case.generator_version
                .as_deref()
                .map(|t| format!("\"{t}\""))
                .unwrap_or_else(|| "null".to_string()),
            case.generator_seed
                .as_deref()
                .map(|t| format!("\"{t}\""))
                .unwrap_or_else(|| "null".to_string()),
            json_list(&case.normative_rules),
            engines.join(", "),
            result.observation_hash,
            match &result.outcome {
                Ok(()) => "AGREEMENT".to_string(),
                Err((admission, _)) => admission.as_str().to_string(),
            },
        ));
    }
    let _ = std::fs::write(
        dir.join("per-case.json"),
        format!("[\n  {}\n]\n", per_case.join(",\n  ")),
    );

    let header = |key: &str| lock.headers.get(key).cloned().unwrap_or_default();
    let passed = results.iter().filter(|(_, r)| r.outcome.is_ok()).count();
    let failed = results.len() - passed;
    let count_kind = |kind: &str| results.iter().filter(|(case, _)| case.kind == kind).count();
    // §12.6: a filtered run is a diagnostic run. The summary says so, in the field a reader checks.
    let result = if failed > 0 {
        "FAIL"
    } else if filters.is_full_evidence() {
        "PASS"
    } else {
        "PARTIAL-FILTERED"
    };
    let summary = format!(
        "{{\n  \"schema_version\": \"c6.5-evidence-1\",\n  \"commit_sha\": \"{}\",\n  \
         \"corpus_version\": \"{}\",\n  \"generator_version\": \"{}\",\n  \"seed\": \"{}\",\n  \
         \"manifest_sha256\": \"{}\",\n  \"generator_sha256\": \"{}\",\n  \"case_count\": {},\n  \
         \"handwritten_count\": {},\n  \"generated_count\": {},\n  \"retained_count\": {},\n  \
         \"metamorphic_family_count\": 0,\n  \"metamorphic_group_count\": {},\n  \
         \"mutation_count\": 0,\n  \"passed_count\": {},\n  \"failed_count\": {},\n  \
         \"skipped_count\": 0,\n  \"quarantined_count\": 0,\n  \"full_evidence\": {},\n  \
         \"filters\": \"{}\",\n  \"result\": \"{}\"\n}}\n",
        commit_sha(),
        header("corpus_version"),
        header("generator_version"),
        header("generator_seed"),
        header("manifest_sha256"),
        header("generator_sha256"),
        results.len(),
        count_kind("handwritten"),
        count_kind("generated"),
        count_kind("retained"),
        header("metamorphic_group_count"),
        passed,
        failed,
        filters.is_full_evidence(),
        json_escape(&format!("{filters:?}")),
        result,
    );
    let _ = std::fs::write(dir.join("summary.json"), summary);
}

fn commit_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// §12.8: "replay deterministic twice". Run over one shard rather than the whole corpus — the
/// property is per-case observation stability, and 89 native builds twice would cost three minutes to
/// prove the same thing five cases prove. The hash is over the canonical observation form, so a case
/// whose output varied run to run (an address, a map order, a timestamp) changes it.
#[test]
fn replaying_a_shard_twice_produces_identical_observation_hashes() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let (cases, _) = load();
    let mut filters = Filters::default();
    filters.shard_index = Some(0);
    filters.shard_total = Some(16);
    let selected: Vec<&Case> = cases.iter().filter(|case| filters.selects(case)).collect();
    assert!(!selected.is_empty(), "shard 0 of 16 is empty");

    let hashes = |filters: &Filters| -> Vec<(String, String)> {
        selected
            .iter()
            .map(|case| {
                let result = replay(case, filters);
                assert!(
                    result.outcome.is_ok(),
                    "{}: {:?}",
                    case.case_id,
                    result.outcome.err()
                );
                (result.case_id, result.observation_hash)
            })
            .collect()
    };
    let first = hashes(&filters);
    let second = hashes(&filters);
    assert_eq!(
        first, second,
        "replaying the same cases twice produced different observations"
    );
    assert!(
        first.iter().all(|(_, hash)| !hash.is_empty()),
        "an observation hash was empty, so nothing was actually compared"
    );
}

// ------------------------------------------------------------- §12.7 sharding --

/// Every case lands in exactly one shard, no case is omitted, and no case is duplicated — §12.7's
/// "formal evidence" clauses, checked over the real corpus rather than argued.
#[test]
fn sharding_partitions_the_corpus_exactly() {
    let (cases, _) = load();
    for shard_total in [1u64, 2, 3, 4, 8, 16] {
        let mut seen: Vec<&str> = Vec::new();
        for index in 0..shard_total {
            let mut filters = Filters::default();
            filters.shard_index = Some(index);
            filters.shard_total = Some(shard_total);
            for case in cases.iter().filter(|case| filters.selects(case)) {
                assert!(
                    !seen.contains(&case.case_id.as_str()),
                    "{} appears in more than one shard of {shard_total}",
                    case.case_id
                );
                seen.push(&case.case_id);
            }
        }
        assert_eq!(
            seen.len(),
            cases.len(),
            "sharding by {shard_total} lost or duplicated cases"
        );
    }
}

/// The shard assignment must not depend on the process, the platform or the case's position — it is
/// a digest of the case ID and nothing else.
#[test]
fn shard_assignment_is_content_addressed_and_stable() {
    assert_eq!(
        shard_of("gen__t01__deadbeef", 4),
        shard_of("gen__t01__deadbeef", 4)
    );
    assert_eq!(
        shard_of("anything", 1),
        0,
        "a single shard holds everything"
    );
    let spread: std::collections::BTreeSet<u64> = load()
        .0
        .iter()
        .map(|case| shard_of(&case.case_id, 8))
        .collect();
    assert!(
        spread.len() > 1,
        "every case hashed to one shard of eight — the split is not distributing"
    );
}

/// §12.6: a filtered run must be distinguishable from full evidence. This is the property that keeps
/// a diagnostic run from being filed as closure evidence.
#[test]
fn a_filtered_run_is_not_full_evidence() {
    assert!(Filters::default().is_full_evidence());
    for narrowed in [
        Filters {
            case: Some("x".into()),
            ..Default::default()
        },
        Filters {
            kind: Some("generated".into()),
            ..Default::default()
        },
        Filters {
            shard_index: Some(0),
            shard_total: Some(4),
            ..Default::default()
        },
        Filters {
            engine: Some("hir".into()),
            ..Default::default()
        },
    ] {
        assert!(
            !narrowed.is_full_evidence(),
            "{narrowed:?} was treated as full evidence"
        );
    }
}

/// §10.3's sentinels must each pin their observation. Kept here rather than in the retired bridge:
/// the replay is what consumes those expectations, so the check belongs beside it.
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
