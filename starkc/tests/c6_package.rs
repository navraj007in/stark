//! WP-C6.5 §15 — package breadth: relocation, dependency ordering, and what a package build leaks.
//!
//! §15.2 and §15.3 are metamorphic claims about a *package graph* rather than a source file: moving a
//! workspace or reordering its dependency declarations must change nothing observable. They cannot be
//! expressed as ordinary corpus cases — a corpus case has one source set, and these transformations
//! act on where that source set lives — so they are harness-level checks over the corpus's own
//! workspace case.
//!
//! **This suite found DEV-113**, and the shape of that finding is why §15.2 asks for it: a package
//! build bakes filesystem paths into semantic identity, so a workspace observed at two locations is
//! not observed identically once a trap is involved. The tests below assert what is true and pin what
//! is not, each with the condition that retires the pin.

mod support;

use std::path::{Path, PathBuf};

use support::corpus::{corpus_root, load, Case};
use support::differential::{
    first_difference, front_end_package, run_hir, run_mir, stage_dir, Observation,
};

fn workspace_case(cases: &[Case]) -> &Case {
    cases
        .iter()
        .find(|c| c.case_id == "pkg__workspace_three_packages")
        .expect("the corpus workspace case")
}

fn symbols(program: &starkc::mir::MirProgram) -> Vec<String> {
    let mut out: Vec<String> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect();
    out.sort();
    out
}

/// Stages the corpus workspace under `parent/<leaf>` and compiles it from there.
fn stage_at(parent: &Path, leaf: &str, case: &Case) -> (PathBuf, Vec<String>, Observation) {
    let root = case.package_root.as_deref().expect("package_root");
    let case_dir = root.split('/').take(3).collect::<Vec<_>>().join("/");
    let remainder = root.split('/').skip(3).collect::<Vec<_>>().join("/");
    let destination = parent.join(leaf);
    let _ = std::fs::remove_dir_all(&destination);
    std::fs::create_dir_all(&destination).expect("staging parent");
    copy_tree(&corpus_root().join(&case_dir), &destination);
    let package_root = if remainder.is_empty() {
        destination.clone()
    } else {
        destination.join(&remainder)
    };
    let (front, program) = front_end_package(&package_root);
    let observed = run_hir("relocated", &front);
    let mir = run_mir("relocated", &program);
    assert!(
        first_difference(&observed, &mir).is_none(),
        "the staged workspace disagrees between HIR and MIR before relocation is even compared"
    );
    (package_root, symbols(&program), observed)
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("dir");
    for entry in std::fs::read_dir(from).expect("read dir") {
        let path = entry.expect("entry").path();
        let name = path.file_name().expect("name").to_owned();
        if path.is_dir() {
            copy_tree(&path, &to.join(name));
        } else {
            std::fs::copy(&path, to.join(name)).expect("copy");
        }
    }
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stark_c6pkgtest_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch");
    dir
}

/// §15.2 — **M08 relocation.** The same workspace at a simple path and at a path containing spaces
/// and non-ASCII characters must produce the same package graph, the same canonical symbols, and the
/// same observation.
#[test]
fn relocation_preserves_symbols_and_observations() {
    let (cases, _) = load();
    let case = workspace_case(&cases);
    let home = scratch("reloc");
    let (simple_root, simple_symbols, simple_observed) = stage_at(&home, "plain", case);
    let (unicode_root, unicode_symbols, unicode_observed) =
        stage_at(&home, "with spaces — ünïcode", case);

    assert_ne!(simple_root, unicode_root, "the two stagings must differ");
    assert_eq!(
        simple_symbols, unicode_symbols,
        "canonical symbols changed under relocation — package identity is leaking its location"
    );
    assert!(
        first_difference(&simple_observed, &unicode_observed).is_none(),
        "the relocated workspace observed differently:\n{simple_observed:#?}\n{unicode_observed:#?}"
    );
    // §15.2: no absolute path in semantic identity. Symbols are the identity that reaches the
    // backend, so a leaked staging path here would make two identical workspaces incompatible.
    let staging_prefix = home.to_string_lossy().into_owned();
    for symbol in &simple_symbols {
        assert!(
            !symbol.contains(&staging_prefix) && !symbol.contains('/'),
            "canonical symbol {symbol:?} carries a filesystem path — package identity must be \
             logical, or two copies of one workspace are different programs"
        );
    }
    let _ = std::fs::remove_dir_all(&home);
}

/// §15.3 — **M09 dependency declaration reorder.** For a semantically order-independent graph,
/// reordering the dependency declarations must change nothing: same canonical symbols, same
/// observation.
///
/// The corpus workspace's root declares one dependency, and a one-entry map cannot be reordered, so
/// the experiment uses a diamond: the root declares BOTH `logic` and `model`, in each order, while
/// `logic` also depends on `model`.
///
/// **The staging deliberately does not compile before rewriting the manifest.** An earlier draft
/// reused the relocation helper, which compiles first and writes `stark.lock`; the stale lock made
/// the result depend on run order, and I briefly recorded that as a defect. It was an artifact of the
/// experiment, not of the compiler — the note stays because a contaminated experiment that produces a
/// plausible finding is worse than no experiment.
#[test]
fn dependency_declaration_order_does_not_leak() {
    let (cases, _) = load();
    let case = workspace_case(&cases);
    let home = scratch("reorder");
    let root_rel = case.package_root.as_deref().expect("package_root");
    let case_dir = root_rel.split('/').take(3).collect::<Vec<_>>().join("/");
    let remainder = root_rel.split('/').skip(3).collect::<Vec<_>>().join("/");

    let build = |order: [&str; 2], leaf: &str| -> (Vec<String>, Observation) {
        let destination = home.join(leaf);
        let _ = std::fs::remove_dir_all(&destination);
        copy_tree(&corpus_root().join(&case_dir), &destination);
        let root = destination.join(&remainder);
        let deps = order
            .iter()
            .map(|name| format!("\"{name}\": {{ \"path\": \"../{name}\" }}"))
            .collect::<Vec<_>>()
            .join(", ");
        std::fs::write(
            root.join("starkpkg.json"),
            format!(
                "{{\n    \"name\": \"app\",\n    \"version\": \"0.1.0\",\n    \
                 \"entry\": \"src/main.stark\",\n    \"dependencies\": {{ {deps} }}\n}}\n"
            ),
        )
        .expect("rewrite manifest");
        let (front, program) = front_end_package(&root);
        let observed = run_hir("reordered", &front);
        let mir = run_mir("reordered", &program);
        assert!(
            first_difference(&observed, &mir).is_none(),
            "{leaf}: the reordered graph disagrees between HIR and MIR"
        );
        (symbols(&program), observed)
    };

    let (logic_first, observed_a) = build(["logic", "model"], "deps_logic_first");
    let (model_first, observed_b) = build(["model", "logic"], "deps_model_first");

    // What HOLDS: the OBSERVATION is order-independent. Nothing a program can see depends on the
    // order its manifest lists dependencies in.
    assert!(
        first_difference(&observed_a, &observed_b).is_none(),
        "reordering dependency declarations changed the observation"
    );
    // Restored at CD-164: with DEV-114 fixed, symbols are order-independent, so §15.3's real claim
    // can be asserted rather than weakened to a count.
    assert_eq!(
        logic_first, model_first,
        "canonical symbols depend on dependency declaration ORDER — a package-order leak (§15.3)"
    );
    let _ = std::fs::remove_dir_all(&home);
}

/// **DEV-114 — FIXED (CD-164).** Canonical package symbols are deterministic and path-independent.
///
/// The defect: with `app → {logic, model}` and `logic → model`, the same function was
/// `model::leaf@[]` in one process and `logic::model::leaf@[]` in the next, because dependency
/// iteration walked a per-process-seeded `HashMap` and whichever path reached a package first fixed
/// the module nesting its items were seen under.
///
/// The fix is two parts, and the second is the one that matters: dependency iteration is **sorted**
/// (removing the nondeterminism), and crossing a **package boundary restarts identity** in
/// `ProgramMeta::build` (removing the path-dependence). TYPE-NOMINAL-001 defines identity as
/// "canonical package instance + module path + item name" — a dependency edge is not a module-path
/// segment — and PKG-IDENTITY-001 adds that aliases and re-exports preserve identity.
///
/// Sorting alone would have made a specification violation merely reproducible, which is why this
/// asserts the canonical FORM and not just stability.
#[test]
fn diamond_package_symbols_are_canonical_and_deterministic_dev_114() {
    let home = scratch("dev114");
    for pkg in ["app", "logic", "model"] {
        std::fs::create_dir_all(home.join(pkg).join("src")).expect("package dir");
    }
    std::fs::write(
        home.join("model/starkpkg.json"),
        "{ \"name\": \"model\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\" }",
    )
    .expect("model manifest");
    std::fs::write(
        home.join("model/src/main.stark"),
        "pub fn leaf() -> Int32 { 1 }\n",
    )
    .expect("model source");
    std::fs::write(
        home.join("logic/starkpkg.json"),
        "{ \"name\": \"logic\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\", \
         \"dependencies\": { \"model\": { \"path\": \"../model\" } } }",
    )
    .expect("logic manifest");
    std::fs::write(
        home.join("logic/src/main.stark"),
        "use model::leaf;\npub fn mid() -> Int32 { leaf() }\n",
    )
    .expect("logic source");
    std::fs::write(
        home.join("app/starkpkg.json"),
        "{ \"name\": \"app\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\", \
         \"dependencies\": { \"logic\": { \"path\": \"../logic\" }, \"model\": { \"path\": \"../model\" } } }",
    )
    .expect("app manifest");
    std::fs::write(
        home.join("app/src/main.stark"),
        "use logic::mid;\nfn main() { print(mid()); }\n",
    )
    .expect("app source");

    // Compiled repeatedly IN THIS PROCESS the answer would be stable even under the defect (one hash
    // seed per process), so the cross-process claim rests on CI running this on two platforms and on
    // the six-run probe recorded at CD-159. What this asserts is the part a single process can prove:
    // the canonical FORM, which the defect got wrong half the time.
    let (_front, program) = front_end_package(&home.join("app"));
    let leaf: Vec<String> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .filter(|s| s.contains("leaf"))
        .collect();
    assert_eq!(
        leaf,
        vec!["model::leaf@[]".to_string()],
        "a package reachable both directly and through a dependency must carry ONE canonical name"
    );
    let _ = std::fs::remove_dir_all(&home);
}

/// **DEV-113-A — FIXED (CD-164).** Package trap provenance is logical and relocation-stable.
///
/// The defect: a package build named every `SourceFile` by its filesystem path, so the same
/// workspace staged in two directories reported different trap provenance — against
/// PKG-IDENTITY-001 ("never an absolute checkout path") and §15.2 ("trap source names remain logical
/// source paths"). The fix names package files `<package>/<path within the package>` and keeps the
/// real location in a separate field used only to resolve `mod` declarations.
#[test]
fn package_trap_provenance_is_logical_and_relocation_stable_dev_113() {
    let home = scratch("dev113");
    let build = |leaf: &str| -> Observation {
        let root = home.join(leaf);
        std::fs::create_dir_all(root.join("app/src")).expect("app");
        std::fs::create_dir_all(root.join("dep/src")).expect("dep");
        std::fs::write(
            root.join("dep/starkpkg.json"),
            "{ \"name\": \"dep\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\" }",
        )
        .expect("dep manifest");
        std::fs::write(
            root.join("dep/src/main.stark"),
            "pub fn boom(v: Int32) -> Int32 {\n    let z: Int32 = 0;\n    v / z\n}\n",
        )
        .expect("dep source");
        std::fs::write(
            root.join("app/starkpkg.json"),
            "{ \"name\": \"app\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\", \
             \"dependencies\": { \"dep\": { \"path\": \"../dep\" } } }",
        )
        .expect("app manifest");
        std::fs::write(
            root.join("app/src/main.stark"),
            "use dep::boom;\nfn main() {\n    print(\"before\");\n    print(boom(7));\n}\n",
        )
        .expect("app source");
        let (front, _program) = front_end_package(&root.join("app"));
        run_hir("dev113", &front)
    };

    let first = build("plain");
    let second = build("moved — ünïcode");
    let (first_file, second_file) = match (&first, &second) {
        (Observation::Trapped(a), Observation::Trapped(b)) => {
            (a.source_file.clone(), b.source_file.clone())
        }
        other => panic!("expected both stagings to trap, got {other:#?}"),
    };
    assert_eq!(
        first_file, "dep/src/main.stark",
        "provenance must be the logical package path"
    );
    assert_eq!(
        first_file, second_file,
        "relocation changed trap provenance — the checkout path is leaking into identity"
    );
    assert!(
        first_difference(&first, &second).is_none(),
        "the relocated workspace observed differently"
    );
    let _ = std::fs::remove_dir_all(&home);
}

/// **DEV-113-B — FIXED (CD-164).** The oracle attributes a trap to the file it was raised in.
///
/// The defect: `RuntimeError` carried a span and no file, so `run_hir` blamed the entry file for
/// every trap — even though `call_callable` swaps `self.file` per callee (DEV-069) and therefore
/// always knew. The fix stamps the raising file onto the error at the innermost frame. Without it a
/// dependency-trap case makes the two engines disagree about WHICH FILE trapped, which is why no such
/// case could exist in the corpus before now.
#[test]
fn the_oracle_and_mir_agree_on_which_file_trapped_dev_113() {
    let home = scratch("dev113b");
    let root = home.join("ws");
    std::fs::create_dir_all(root.join("app/src")).expect("app");
    std::fs::create_dir_all(root.join("dep/src")).expect("dep");
    std::fs::write(
        root.join("dep/starkpkg.json"),
        "{ \"name\": \"dep\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\" }",
    )
    .expect("dep manifest");
    std::fs::write(
        root.join("dep/src/main.stark"),
        "pub fn boom(v: Int32) -> Int32 {\n    let z: Int32 = 0;\n    v / z\n}\n",
    )
    .expect("dep source");
    std::fs::write(
        root.join("app/starkpkg.json"),
        "{ \"name\": \"app\", \"version\": \"0.1.0\", \"entry\": \"src/main.stark\", \
         \"dependencies\": { \"dep\": { \"path\": \"../dep\" } } }",
    )
    .expect("app manifest");
    std::fs::write(
        root.join("app/src/main.stark"),
        "use dep::boom;\nfn main() {\n    print(\"before\");\n    print(boom(7));\n}\n",
    )
    .expect("app source");

    let (front, program) = front_end_package(&root.join("app"));
    let hir = run_hir("dev113b", &front);
    let mir = run_mir("dev113b", &program);
    assert!(
        first_difference(&hir, &mir).is_none(),
        "the oracle and MIR disagree about the dependency trap:\n{hir:#?}\n{mir:#?}"
    );
    if let Observation::Trapped(trap) = &hir {
        assert_eq!(
            trap.source_file, "dep/src/main.stark",
            "the trap must be attributed to the DEPENDENCY, not the entry file"
        );
        assert_eq!(trap.line, 3, "the division is on line 3 of the dependency");
    } else {
        panic!("expected a trap, got {hir:#?}");
    }
    let _ = std::fs::remove_dir_all(&home);
}

/// §15.4, as far as it can be claimed here: the corpus workspace resolves and builds with the lock
/// present, and staging never reaches the network because every dependency is a relative path.
/// Recorded honestly — this is weaker than `stark build --locked --offline`, which exercises the CLI
/// path rather than the library one, and is already covered for the generated crate by
/// `c64_platform_matrix::portability_generated_crate_is_locked_and_network_free`.
#[test]
fn the_corpus_workspace_resolves_offline_from_relative_paths() {
    let (cases, _) = load();
    let case = workspace_case(&cases);
    let root = case.package_root.as_deref().expect("package_root");
    let case_dir = root.split('/').take(3).collect::<Vec<_>>().join("/");
    let (staged, _) = stage_dir("offline", &corpus_root().join(&case_dir));
    for manifest in [
        "app/starkpkg.json",
        "logic/starkpkg.json",
        "model/starkpkg.json",
    ] {
        let text = std::fs::read_to_string(staged.join(manifest)).expect(manifest);
        assert!(
            !text.contains("registry") && !text.contains("http"),
            "{manifest} names a non-path source, so resolution could reach the network"
        );
    }
    let (_front, program) = front_end_package(&staged.join("app"));
    assert!(
        !program.bodies.is_empty(),
        "the staged workspace produced no bodies"
    );
    assert!(
        staged.join("app/stark.lock").is_file(),
        "resolution did not write a lock into the staged root"
    );
    let _ = std::fs::remove_dir_all(&staged);
}
