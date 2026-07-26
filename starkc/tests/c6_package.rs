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
    // The symbol sets are NOT compared here: they are unstable for this graph shape for a reason
    // that has nothing to do with ordering — see `diamond_package_symbols_are_nondeterministic_dev_114`.
    assert_eq!(
        logic_first.len(),
        model_first.len(),
        "reordering changed the number of monomorphised instances"
    );
    let _ = std::fs::remove_dir_all(&home);
}

/// **DEV-114 — canonical package symbols are NONDETERMINISTIC for a diamond dependency graph.**
///
/// Found by §15.3. In a graph where a package is reachable both directly from the root and through
/// another dependency (`app → {logic, model}`, `logic → model`), the same function is named
/// `model::leaf@[]` in one process and `logic::model::leaf@[]` in the next — same sources, same
/// manifests, same declaration order. Six consecutive runs produced both forms.
///
/// The cause is that a package's symbol prefix is assigned by whichever traversal path reaches it
/// first, and the traversal follows a hash map whose iteration order is seeded per process.
///
/// **Why it matters.** Canonical symbols are the identity that reaches the backend, so two builds of
/// one workspace can produce differently-named generated code. PKG-IDENTITY-001 requires a resolved
/// package token to be relocation-stable and CD-108 made identity deterministic; neither holds here.
/// The corpus's own workspace case is a CHAIN, not a diamond, so it is unaffected — which is why no
/// corpus case is flaky and why this needed a purpose-built graph to surface.
///
/// **Escalated, not fixed** (§18.5): choosing the canonical name for a package reachable by several
/// paths is a compiler decision — shortest path, declaration path, or package-name-only — not a
/// corpus edit. This test pins the defect as an admitted set and retires when one form is produced
/// consistently.
#[test]
fn diamond_package_symbols_are_nondeterministic_dev_114() {
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

    let (_front, program) = front_end_package(&home.join("app"));
    let leaf: Vec<String> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .filter(|s| s.contains("leaf"))
        .collect();
    assert_eq!(leaf.len(), 1, "expected one `leaf` instance, got {leaf:?}");
    // Both forms are observed across processes. A fixed implementation produces ONE of them every
    // time — at which point this test should be replaced by an equality assertion on that form.
    assert!(
        leaf[0] == "model::leaf@[]" || leaf[0] == "logic::model::leaf@[]",
        "unexpected symbol {:?} — DEV-114 may have changed shape",
        leaf[0]
    );
    let _ = std::fs::remove_dir_all(&home);
}

/// **DEV-113, pinned.** §15.2 requires that "no absolute path enters semantic identity" and that
/// "trap source names remain logical source paths". For a PACKAGE build, they do not: the file names
/// in a package graph are filesystem paths, so a trap raised inside a relocated workspace reports the
/// staging directory. The two stagings below differ only in location and produce different trap
/// provenance.
///
/// Recorded rather than worked around, because the fix is a compiler decision — package file
/// identity would have to become logical (package-relative) rather than filesystem-absolute — and
/// because it has a live consequence: a **trapping package case cannot be added to the corpus**
/// until it lands, since its observation would depend on where the corpus was checked out.
///
/// This test retires when the assertion below starts failing: that will mean provenance became
/// relocation-stable.
#[test]
fn a_package_trap_reports_an_absolute_path_dev_113() {
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
    assert!(
        first_file.starts_with('/') || first_file.contains(':'),
        "DEV-113 appears to be FIXED — package trap provenance is now {first_file:?}, not an \
         absolute path. Delete this test, add a trapping package case to the corpus, and close the \
         §15.2 deviation."
    );
    assert_ne!(
        first_file, second_file,
        "DEV-113 appears to be FIXED — the two stagings now report the same provenance"
    );
    let _ = std::fs::remove_dir_all(&home);
}

/// The second half of DEV-113: the HIR oracle attributes every trap to the ROOT file, whatever file
/// actually trapped, because `RuntimeError` carries a span but not a file. MIR attributes it
/// correctly through `SourceInfo`. So a trap inside a DEPENDENCY would be reported by the two engines
/// at different files — which is why no such case is in the corpus yet.
#[test]
fn the_oracle_attributes_a_dependency_trap_to_the_root_file_dev_113() {
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
    let field = first_difference(&hir, &mir);
    assert_eq!(
        field,
        Some("trap source_file"),
        "expected the oracle and MIR to disagree about WHICH FILE trapped (DEV-113). If this now \
         agrees, the oracle has learned per-file trap attribution: delete this test and add a \
         dependency-trap case to the corpus (§15.1's last shape)."
    );
    if let (Observation::Trapped(h), Observation::Trapped(m)) = (&hir, &mir) {
        // Separators are normalised before comparing. On Windows these paths come back MIXED —
        // `…\ws\app\src/main.stark` — because the OS builds the directory part with `\` while the
        // entry suffix is composed with a literal `/` in the compiler. That inconsistency belongs to
        // DEV-113's record; here it must not be mistaken for the attribution claim under test.
        let oracle_file = h.source_file.replace('\\', "/");
        let mir_file = m.source_file.replace('\\', "/");
        assert!(
            oracle_file.ends_with("app/src/main.stark"),
            "the oracle reported {oracle_file:?}"
        );
        assert!(
            mir_file.ends_with("dep/src/main.stark"),
            "MIR reported {mir_file:?}, which was expected to be the dependency"
        );
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
