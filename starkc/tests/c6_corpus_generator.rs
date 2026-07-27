//! WP-C6.5 §11.10 / §20.2 — the generator's determinism, proven by running it.
//!
//! The generator's value rests entirely on being reproducible: a case whose bytes depend on the
//! machine, the Python version, the filesystem order or the wall clock is not evidence, because
//! nobody else can regenerate it to check. So these tests invoke `generate.py` for real, into
//! temporary directories, and compare bytes.
//!
//! They are cheap — generation is pure text, no compiler involved — and they are the only tests here
//! that shell out. `python3` is already required by the repo's tooling (spec build, corpus lock), so
//! this adds no dependency; if it is genuinely absent the tests say so rather than passing quietly.

mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/c6-corpus")
}

fn python_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stark_c6gen_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Runs `generate.py --seed <seed> --out <dir>` and returns every generated file as
/// (relative path, bytes), sorted. Paths are relative on purpose: an absolute path in the
/// comparison would make relocation trivially "differ".
fn generate(seed: &str, out: &Path) -> Vec<(String, Vec<u8>)> {
    let status = Command::new("python3")
        .arg(corpus_dir().join("generate.py"))
        .arg("--seed")
        .arg(seed)
        .arg("--out")
        .arg(out)
        .output()
        .expect("running generate.py");
    assert!(
        status.status.success(),
        "generate.py failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let mut files = Vec::new();
    fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        for entry in std::fs::read_dir(dir).expect("read generated dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                walk(&path, base, out);
            } else {
                let rel = path
                    .strip_prefix(base)
                    .expect("relative")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((rel, std::fs::read(&path).expect("read generated file")));
            }
        }
    }
    walk(out, out, &mut files);
    files.sort();
    assert!(!files.is_empty(), "generate.py produced nothing");
    files
}

#[test]
fn the_same_seed_twice_is_byte_identical() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let first_dir = scratch("same_a");
    let second_dir = scratch("same_b");
    let first = generate("determinism-probe", &first_dir);
    let second = generate("determinism-probe", &second_dir);
    assert_eq!(
        first.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        second.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        "the same seed produced a different case set"
    );
    assert_eq!(first, second, "the same seed produced different bytes");
    let _ = std::fs::remove_dir_all(&first_dir);
    let _ = std::fs::remove_dir_all(&second_dir);
}

/// §11.10 relocation: two different output roots must produce identical LOGICAL paths and bytes.
/// A generator that leaked its output directory into a case ID, a source comment or an expectation
/// would fail here — which is the failure this catches, since the two roots differ by construction.
#[test]
fn relocation_does_not_change_the_output() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let shallow = scratch("reloc_shallow");
    let deep = scratch("reloc_deep").join("nested/one/two");
    std::fs::create_dir_all(&deep).expect("deep dir");
    let a = generate("relocation-probe", &shallow);
    let b = generate("relocation-probe", &deep);
    assert_eq!(a, b, "output depends on where it was written");
    let _ = std::fs::remove_dir_all(&shallow);
}

/// §11.10: nothing about the surrounding filesystem may influence generation. The generator
/// enumerates its own template registry, so pre-existing junk in the output directory — including
/// files that would sort before and after every real case — must not change a byte.
#[test]
fn pre_existing_files_do_not_change_the_output() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let clean = scratch("fsorder_clean");
    let dirty = scratch("fsorder_dirty");
    std::fs::create_dir_all(dirty.join("cases/generated")).expect("dirty dir");
    for name in [
        "000_first.stark",
        "zzz_last.stark",
        "gen__t01__ffffffff.stark",
    ] {
        std::fs::write(dirty.join("cases/generated").join(name), b"fn main() {}\n").expect("junk");
    }
    let expected = generate("fsorder-probe", &clean);
    let observed = generate("fsorder-probe", &dirty);
    assert_eq!(
        expected, observed,
        "generation depended on what was already in the output directory"
    );
    let _ = std::fs::remove_dir_all(&clean);
    let _ = std::fs::remove_dir_all(&dirty);
}

/// §11.10: a different seed must produce a different — but equally reproducible — selection. Both
/// halves matter. A seed that changed nothing would mean the seed is not in the identity; a seed
/// that changed the CASE COUNT would mean the budget is not being honoured.
#[test]
fn a_different_seed_selects_differently_and_still_deterministically() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let one = scratch("seed_one");
    let two = scratch("seed_two");
    let two_again = scratch("seed_two_again");
    let a = generate("seed-alpha", &one);
    let b = generate("seed-beta", &two);
    let b_again = generate("seed-beta", &two_again);
    assert_eq!(b, b_again, "the second seed is not itself deterministic");
    assert_ne!(
        a.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        b.iter().map(|(p, _)| p).collect::<Vec<_>>(),
        "two seeds selected exactly the same cases — the seed is not part of case identity"
    );
    assert_eq!(
        a.len(),
        b.len(),
        "the per-template budget is not seed-independent"
    );
    for dir in [&one, &two, &two_again] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// §11.10: the generator version participates in identity, so a version change reselects. That is
/// what makes "generator version change requires corpus version review" enforceable rather than
/// advisory — the cases themselves move.
#[test]
fn a_generator_version_change_reselects_cases() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let fake_root = scratch("version_root");
    // Every module the generator imports, not just the entry point: `generate.py` imports both
    // registries, so copying it alone produced a ModuleNotFoundError that read as "the patched
    // generator failed" rather than "the test copied too little".
    for name in ["generate.py", "templates.py", "metamorphic.py"] {
        std::fs::copy(corpus_dir().join(name), fake_root.join(name)).expect("copy generator");
    }
    std::fs::write(fake_root.join("generator-version.txt"), b"9.9.9\n").expect("version");
    let out = fake_root.join("out");
    let status = Command::new("python3")
        .arg(fake_root.join("generate.py"))
        .arg("--seed")
        .arg("version-probe")
        .arg("--out")
        .arg(&out)
        .output()
        .expect("running the patched generator");
    assert!(
        status.status.success(),
        "patched generator failed: {}",
        String::from_utf8_lossy(&status.stderr)
    );
    let baseline_dir = scratch("version_baseline");
    let baseline = generate("version-probe", &baseline_dir);
    let mut patched: Vec<String> = std::fs::read_dir(out.join("cases/generated"))
        .expect("patched output")
        .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
        .collect();
    patched.sort();
    let mut original: Vec<String> = baseline
        .iter()
        .filter(|(p, _)| p.starts_with("cases/generated/"))
        .map(|(p, _)| p.rsplit('/').next().unwrap().to_string())
        .collect();
    original.sort();
    assert_ne!(
        original, patched,
        "changing generator_version reselected nothing — the version is not part of case identity"
    );
    let _ = std::fs::remove_dir_all(&fake_root);
    let _ = std::fs::remove_dir_all(&baseline_dir);
}

/// §11.10: no absolute path may enter a generated file. A leaked build path would make the corpus
/// unreproducible on any other machine while still passing every local check — the quietest possible
/// way for this to be wrong.
#[test]
fn no_absolute_path_enters_the_generated_corpus() {
    let root = corpus_dir();
    let repo = root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("repo root")
        .to_string_lossy()
        .into_owned();
    let mut checked = 0;
    for entry in std::fs::read_dir(root.join("cases/generated")).expect("generated cases") {
        let path = entry.expect("entry").path();
        let text = std::fs::read_to_string(&path).expect("case source");
        for needle in [repo.as_str(), "/Users/", "/home/", "C:\\"] {
            assert!(
                !text.contains(needle),
                "{}: contains an absolute path fragment {needle:?}",
                path.display()
            );
        }
        checked += 1;
    }
    let manifest =
        std::fs::read_to_string(root.join("generated.manifest.toml")).expect("generated manifest");
    for needle in [repo.as_str(), "/Users/", "/home/", "C:\\"] {
        assert!(
            !manifest.contains(needle),
            "generated.manifest.toml contains an absolute path fragment {needle:?}"
        );
    }
    assert!(checked >= 64, "only {checked} generated cases on disk");
}

/// §11.9's `--check`: the checked-in generated corpus must be exactly what the generator produces
/// today. This is the test that catches a hand-edited generated case.
#[test]
fn the_checked_in_generated_corpus_is_current() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let out = Command::new("python3")
        .arg(corpus_dir().join("generate.py"))
        .arg("--check")
        .output()
        .expect("running generate.py --check");
    assert!(
        out.status.success(),
        "generate.py --check reported drift:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// **R-09 and R-11's negative controls.** The generator gained two guards — a case-ID collision
/// check and an enforced loop bound — and a guard that has never refused anything is a guard nobody
/// has watched work. `--self-test-guards` forces each failure deliberately and requires the
/// generator to reject it for the right reason.
#[test]
fn the_generator_guards_refuse_what_they_claim_to() {
    if !python_available() {
        eprintln!("SKIP: no python3 in this environment.");
        return;
    }
    let out = Command::new("python3")
        .arg(corpus_dir().join("generate.py"))
        .arg("--self-test-guards")
        .output()
        .expect("running the guard self-test");
    assert!(
        out.status.success(),
        "a generator guard did not refuse:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // Each guard is named rather than matching one summary phrase. The phrase was "both refuse",
    // and when R-04/R-05 added five more rules it still said "both" until this assertion failed —
    // which is the right outcome, but a phrase match would equally have kept passing had the
    // wording not changed while guards were REMOVED.
    let stdout = String::from_utf8_lossy(&out.stdout);
    for guard in ["collision", "loop bound", "metamorphic package-pair"] {
        assert!(
            stdout.contains(guard),
            "the self-test no longer reports the {guard} guard: {stdout}"
        );
    }
}

/// **R-01.** Every admitted trap category has a direct corpus case. The gap this closes was not a
/// missing case but a mechanism: T16's dimension space IS the coverage claim, and a per-template
/// sampling budget deleted two categories from it while the corpus still reported them covered.
#[test]
fn every_admitted_trap_category_has_a_corpus_case() {
    let (cases, _) = support::corpus::load();
    let covered: std::collections::BTreeSet<&str> = cases
        .iter()
        .filter_map(|c| c.expected_trap_category.as_deref())
        .collect();
    let admitted: Vec<String> = support::differential::ALL_CATEGORIES
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();
    let missing: Vec<&String> = admitted
        .iter()
        .filter(|name| !covered.contains(name.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "§10.4 requires a direct case per admitted trap category; missing: {missing:?}"
    );
}

/// §11.4's floor, and the template breadth behind it. Asserted against the manifest rather than the
/// generator's own report, so a generator that claimed 64 while writing 12 would fail.
#[test]
fn the_generated_corpus_meets_the_acceptance_floor() {
    let (cases, _) = support::corpus::load();
    let generated: Vec<&support::corpus::Case> =
        cases.iter().filter(|c| c.kind == "generated").collect();
    assert!(
        generated.len() >= 64,
        "§11.4 requires at least 64 generated primary cases; the corpus has {}",
        generated.len()
    );
    let templates: std::collections::BTreeSet<&str> = generated
        .iter()
        .filter_map(|c| c.template_id.as_deref())
        .collect();
    assert!(
        templates.len() >= 10,
        "§11.4 requires at least 10 distinct top-level templates; the corpus has {}: {:?}",
        templates.len(),
        templates
    );
    // Completion AND trap cases, because a corpus of only completions cannot observe a missed trap.
    assert!(
        generated.iter().any(|c| c.expected_outcome == "trap"),
        "no generated trap case"
    );
    assert!(
        generated.iter().any(|c| c.expected_outcome == "completion"),
        "no generated completion case"
    );
    // Every generated case must be reproducible from its recorded metadata.
    for case in &generated {
        assert!(
            case.generator_seed.is_some()
                && case.generator_version.is_some()
                && case.template_id.is_some(),
            "{}: generated case without full provenance",
            case.case_id
        );
    }
}
