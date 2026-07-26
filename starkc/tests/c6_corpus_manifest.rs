//! WP-C6.5 §9 / §20.1 — the corpus manifest, its strict validation, and the lock.
//!
//! Two halves, and the second is the one that matters. The first proves the real corpus loads and
//! its lock matches. The second proves the validator REJECTS each thing §9.3 says it must, by
//! feeding it manifests that violate exactly one rule — because a validator whose only evidence is
//! a valid manifest is a validator nobody has watched refuse anything.
//!
//! The rejection tests use synthetic manifest text rather than temporary corpus trees wherever the
//! rule is a manifest rule, so they stay fast and cannot leave debris. The three rules that are
//! genuinely about the filesystem (missing source, unlisted source, lock hash mismatch) build a
//! scratch tree.

mod support;

use support::corpus::{
    corpus_root, load, parse_lock, parse_manifest, rule_ids_in, sha256_hex, spec_rule_ids,
    validate, verify_lock,
};

/// §9.6 governance. Changing the corpus means regenerating `corpus.lock` AND bumping the version;
/// this assertion is what makes the second half unskippable.
const EXPECTED_CORPUS_VERSION: &str = "0.4.0";

// ------------------------------------------------------------- the real corpus --

#[test]
fn the_manifest_loads_and_validates() {
    let (cases, _) = load();
    assert!(!cases.is_empty(), "the corpus must not be empty");
    // Deterministic enumeration order is a §9.3 requirement, not an accident of file order.
    let ids: Vec<&str> = cases.iter().map(|c| c.case_id.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(
        ids, sorted,
        "cases must enumerate in ascending case_id order"
    );
}

#[test]
fn the_lock_matches_the_corpus() {
    let (cases, lock) = load();
    verify_lock(&corpus_root(), &lock, &cases).expect("corpus.lock must match the corpus");
    assert_eq!(
        lock.headers.get("corpus_version").map(String::as_str),
        Some(EXPECTED_CORPUS_VERSION),
        "corpus_version changed without updating this assertion (§9.6 freeze governance)"
    );
}

/// Every field the lock is required to carry (§9.5). Checked by name so a regenerated lock that
/// silently stopped emitting one is caught here rather than by its absence going unnoticed.
#[test]
fn the_lock_carries_every_required_header() {
    let (_, lock) = load();
    for key in [
        "corpus_version",
        "generator_version",
        "generator_seed",
        "manifest_sha256",
        "generator_sha256",
        "case_count",
        "handwritten_count",
        "generated_count",
        "retained_count",
        "metamorphic_group_count",
    ] {
        assert!(
            lock.headers.contains_key(key),
            "corpus.lock is missing `{key}`"
        );
    }
}

/// **CD-154.** Every normative rule ID cited by the coverage matrix must be a rule the
/// specification actually defines.
///
/// This is the test that should have existed at phase C6.5-0. The matrix was built citing **69
/// invented identifiers out of 84** — `OWN-DROP-001`, `FN-VALUE-001`, `MAP-001`, `TRAP-ABORT-001`,
/// `CTRL-IF-001` and 64 more — and its exit condition "every row has a normative citation" passed
/// because nothing compared the citations to the spec. Fabricated grounding is worse than absent
/// grounding: a reader who follows the reference finds nothing, and everyone else assumes it was
/// checked.
#[test]
fn every_rule_id_the_matrix_cites_exists_in_the_spec() {
    let spec = spec_rule_ids();
    let matrix = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("repo root")
            .join("STARKLANG/docs/compiler/work-packages/C6-CORPUS-COVERAGE-MATRIX.md"),
    )
    .expect("the coverage matrix");
    // Only TABLE ROWS are citations. The document's own note about CD-154 has to name the
    // fabricated identifiers it is reporting (`OWN-DROP-001`, `FN-VALUE-001`, …), and a checker that
    // could not tell a citation from a description of one would force that finding to be written
    // vaguely — which is the opposite of the point.
    let cited_in_rows: std::collections::BTreeSet<String> = matrix
        .lines()
        .filter(|line| line.trim_start().starts_with('|'))
        .flat_map(rule_ids_in)
        .collect();
    let missing: Vec<String> = cited_in_rows
        .into_iter()
        .filter(|id| !spec.contains(id))
        .collect();
    assert!(
        missing.is_empty(),
        "the coverage matrix cites {} rule ID(s) that no spec document defines: {}",
        missing.len(),
        missing.join(", ")
    );
}

/// The manifest's citations are checked by `validate`; this proves that check REJECTS, rather than
/// trusting that it runs.
#[test]
fn a_manifest_citing_an_invented_rule_is_rejected() {
    rejected(
        &valid_case().replace("\"PROC-EXIT-001\"", "\"OWN-DROP-001\""),
        "is not a rule defined in",
    );
}

// ------------------------------------------------------- §20.1 rejection tests --

/// A manifest with one case, valid unless the caller perturbs it. Every rejection test below starts
/// here and changes exactly one thing, so the test's name is also the reason it fails.
fn valid_case() -> String {
    [
        "[[case]]",
        "case_id = \"a_case\"",
        "kind = \"handwritten\"",
        "category = \"traps\"",
        "sources = [\"cases/retained/entry_exit__01_unit_entry.stark\"]",
        "package_graph = \"single-file\"",
        "expected_outcome = \"completion\"",
        "required_engines = [\"hir\", \"mir\", \"native-debug\"]",
        "required_targets = [\"aarch64-apple-darwin\"]",
        "normative_rules = [\"PROC-EXIT-001\"]",
    ]
    .join("\n")
}

#[track_caller]
fn rejected(manifest: &str, expect: &str) {
    let result = parse_manifest(manifest).and_then(|cases| validate(&cases, &corpus_root()));
    match result {
        Ok(()) => panic!("the validator ACCEPTED a manifest it must reject ({expect})"),
        Err(reason) => assert!(
            reason.contains(expect),
            "rejected for the wrong reason — wanted {expect:?}, got: {reason}"
        ),
    }
}

#[test]
fn valid_manifest_loads() {
    // The control: without this, "rejects everything" would pass every test below.
    let cases = parse_manifest(&valid_case()).expect("parses");
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].case_id, "a_case");
    assert_eq!(cases[0].required_engines.len(), 3);
}

#[test]
fn unknown_key_rejected() {
    // Not on §9.3's list, but the reason that list can be trusted: an ignored key is an attribute
    // nobody checks.
    let manifest = format!("{}\nmystery_field = \"x\"", valid_case());
    assert!(
        parse_manifest(&manifest).is_err(),
        "unknown key was accepted"
    );
}

#[test]
fn duplicate_case_id_rejected() {
    let manifest = format!("{}\n{}", valid_case(), valid_case());
    rejected(&manifest, "duplicate case_id");
}

#[test]
fn missing_source_rejected() {
    let manifest = valid_case().replace(
        "cases/retained/entry_exit__01_unit_entry.stark",
        "cases/handwritten/not_written_yet.stark",
    );
    rejected(&manifest, "does not exist");
}

#[test]
fn unlisted_source_rejected() {
    // The real corpus has six sources; a manifest listing only one leaves five unclaimed.
    rejected(&valid_case(), "no case lists it");
}

#[test]
fn duplicate_source_ownership_rejected() {
    let second = valid_case().replace("a_case", "b_case");
    rejected(
        &format!("{}\n{second}", valid_case()),
        "already owned by case",
    );
}

#[test]
fn absolute_and_escaping_paths_rejected() {
    for (path, expect) in [
        ("/etc/passwd", "absolute source path"),
        ("../../../etc/passwd", "escapes the corpus root"),
    ] {
        let manifest = valid_case().replace("cases/retained/entry_exit__01_unit_entry.stark", path);
        rejected(&manifest, expect);
    }
}

/// A Windows-separated path is caught one layer EARLIER than the §9.3 path rule: the manifest string
/// subset has no backslash at all, so it never reaches `validate`. Asserted at the layer that
/// actually refuses it — the validator's `/`-separator rule stays as defence for paths that arrive
/// from anywhere other than the manifest text.
#[test]
fn backslash_separated_path_rejected_by_the_parser() {
    let manifest = valid_case().replace(
        "cases/retained/entry_exit__01_unit_entry.stark",
        "cases\\retained\\x.stark",
    );
    let reason = parse_manifest(&manifest).expect_err("a backslash path must not parse");
    assert!(
        reason.contains("backslashes are not supported"),
        "unexpected reason: {reason}"
    );
}

#[test]
fn unknown_category_rejected() {
    rejected(
        &valid_case().replace("\"traps\"", "\"vibes\""),
        "unknown category",
    );
}

#[test]
fn unknown_engine_rejected() {
    rejected(
        &valid_case().replace("\"native-debug\"", "\"native-release\""),
        "unknown required engine",
    );
}

#[test]
fn unknown_target_rejected() {
    rejected(
        &valid_case().replace(
            "\"aarch64-apple-darwin\"",
            "\"riscv64gc-unknown-linux-gnu\"",
        ),
        "unknown required target",
    );
}

#[test]
fn empty_normative_rules_rejected() {
    rejected(
        &valid_case().replace(
            "normative_rules = [\"PROC-EXIT-001\"]",
            "normative_rules = []",
        ),
        "must cite at least one normative rule",
    );
}

#[test]
fn trap_case_without_expected_category_rejected() {
    rejected(
        &valid_case().replace("\"completion\"", "\"trap\""),
        "must state `expected_trap_category`",
    );
}

#[test]
fn generated_case_missing_seed_rejected() {
    rejected(
        &valid_case().replace("\"handwritten\"", "\"generated\""),
        "must carry generator_seed",
    );
}

#[test]
fn metamorphic_member_missing_group_rejected() {
    let manifest = format!("{}\nmetamorphic_family = \"M01\"", valid_case());
    rejected(&manifest, "needs BOTH family and group");
}

/// §4.4: the disallowed reasons are not merely discouraged, they are unwritable. A quarantine whose
/// reason is "the engines disagree" cannot be spelled, because no allowed reason class covers it.
#[test]
fn semantic_quarantine_rejected() {
    let manifest = format!(
        "{}\nquarantine = \"engine-disagreement: native computes a different value (CD-999)\"",
        valid_case()
    );
    rejected(&manifest, "not an available classification");
}

#[test]
fn quarantine_without_authority_rejected() {
    let manifest = format!(
        "{}\nquarantine = \"non-core-feature: HashSet has no MIR representation\"",
        valid_case()
    );
    rejected(&manifest, "must name its deciding authority");
}

#[test]
fn unsorted_manifest_rejected() {
    let first = valid_case().replace("a_case", "z_case");
    let second = valid_case().replace("a_case", "b_case").replace(
        "cases/retained/entry_exit__01_unit_entry.stark",
        "cases/retained/entry_exit__02_int32_status.stark",
    );
    rejected(
        &format!("{first}\n{second}"),
        "not in ascending case_id order",
    );
}

// ------------------------------------------------------------- lock rejections --

/// A scratch copy of the whole corpus — every source, plus the manifest and generator. Built under
/// the harness's temp dir so a test can mutate it without touching the checked-in corpus, and copied
/// WHOLESALE rather than dir-by-dir: a partial copy makes the lock fail for the copier's reason
/// instead of the test's, which is how these three tests first went red.
fn scratch_corpus(tag: &str) -> std::path::PathBuf {
    fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).expect("scratch dir");
        for entry in std::fs::read_dir(from).expect("read corpus") {
            let path = entry.expect("entry").path();
            let name = path.file_name().expect("name").to_owned();
            if path.is_dir() {
                copy_tree(&path, &to.join(name));
            } else if name != "corpus.lock" {
                std::fs::copy(&path, to.join(name)).expect("copy corpus file");
            }
        }
    }
    let root = std::env::temp_dir().join(format!(
        "stark_c6corpus_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    copy_tree(&corpus_root(), &root);
    root
}

#[test]
fn lock_hash_mismatch_rejected() {
    let root = scratch_corpus("hash");
    let (cases, lock) = load();
    // A one-byte change to a case, with the lock untouched: the corpus edit that a hash list exists
    // to catch.
    let case = root.join("cases/retained/entry_exit__01_unit_entry.stark");
    let mutated = std::fs::read_to_string(&case)
        .unwrap()
        .replace("\"x\"", "\"y\"");
    std::fs::write(&case, mutated).unwrap();
    let reason =
        verify_lock(&root, &lock, &cases).expect_err("a mutated source must fail the lock");
    assert!(
        reason.contains("has changed"),
        "unexpected reason: {reason}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn unlisted_file_rejected() {
    let root = scratch_corpus("unlisted");
    let (cases, lock) = load();
    std::fs::write(
        root.join("cases/retained/smuggled.stark"),
        "fn main() {\n}\n",
    )
    .unwrap();
    let reason =
        verify_lock(&root, &lock, &cases).expect_err("an unlisted corpus file must fail the lock");
    assert!(
        reason.contains("corpus.lock does not list it"),
        "unexpected reason: {reason}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn count_mismatch_rejected() {
    let root = scratch_corpus("counts");
    let (cases, mut lock) = load();
    lock.headers
        .insert("retained_count".to_string(), "99".to_string());
    let reason = verify_lock(&root, &lock, &cases).expect_err("a wrong count must fail the lock");
    assert!(
        reason.contains("retained_count"),
        "unexpected reason: {reason}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn manifest_hash_mismatch_rejected() {
    let root = scratch_corpus("manifesthash");
    let (cases, lock) = load();
    let manifest = root.join("manifest.toml");
    let text = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(&manifest, format!("{text}\n# an untracked edit\n")).unwrap();
    let reason =
        verify_lock(&root, &lock, &cases).expect_err("an edited manifest must fail the lock");
    assert!(
        reason.contains("manifest_sha256"),
        "unexpected reason: {reason}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn lock_paths_are_canonically_ordered_and_slash_separated() {
    let (_, lock) = load();
    let paths: Vec<&str> = lock.files.iter().map(|(p, _)| p.as_str()).collect();
    let mut sorted = paths.clone();
    sorted.sort_unstable();
    assert_eq!(paths, sorted, "lock paths must be sorted");
    assert!(
        paths.iter().all(|p| !p.contains('\\')),
        "lock paths must use `/` separators on every platform"
    );
}

/// The hash function the lock is written with must be the one it is verified with — trivial, and
/// exactly the kind of thing that silently diverges when one side is Python and the other Rust.
#[test]
fn rust_and_python_agree_on_the_hash() {
    let root = corpus_root();
    let (_, lock) = load();
    let (path, expected) = &lock.files[0];
    let bytes = std::fs::read(root.join(path)).expect("first locked source");
    assert_eq!(&sha256_hex(&bytes), expected, "{path}");
}

/// The lock parser must reject a malformed lock rather than silently reading past it.
#[test]
fn malformed_lock_rejected() {
    assert!(parse_lock("this is not a lock line").is_err());
    assert!(parse_lock("corpus_version = 1\ncorpus_version = 2").is_err());
}
