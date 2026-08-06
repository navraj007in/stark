//! AS0 — reproductions for the package source-identity defect.
//!
//! **These tests assert the CURRENT, DEFECTIVE behaviour on purpose.** AS0's job is to reproduce,
//! not to fix; AS1a's job is to fix, and its checkpoint is these tests failing and being flipped to
//! the corrected assertion. A test that asserted the *intended* behaviour and was ignored would
//! prove nothing in the meantime and would not notice the behaviour changing for some other reason.
//!
//! Every assertion below is labelled DEFECT (pins what is wrong today) or INVARIANT (pins something
//! that must not change either way).
//!
//! Reproduces: two `SourceRecord`s for one physical entry file; the real entry classified as a
//! non-root module with no package; and both varying with the absolute checkout path.

use starkc::analysis::{analyze_project, ProjectInput, SourceProvenance};
use starkc::options::LanguageOptions;
use starkc::package::PackageGraph;

/// A two-file package staged at `root`. The entry is deliberately outside the invoking process's
/// current directory, which is the condition AS0 names.
fn stage(root: &std::path::Path, name: &str) {
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        root.join("starkpkg.json"),
        format!(r#"{{"name":"{name}","version":"0.1.0","entry":"src/main.stark"}}"#),
    )
    .unwrap();
    std::fs::write(
        src.join("main.stark"),
        "mod helper;\n\nfn main() {\n    let value: Int32 = helper::seven();\n}\n",
    )
    .unwrap();
    std::fs::write(
        src.join("helper.stark"),
        "pub fn seven() -> Int32 {\n    7\n}\n",
    )
    .unwrap();
}

fn unique_root(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let root = std::env::temp_dir().join(format!("as0_{tag}_{}_{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    root
}

fn analyse(root: &std::path::Path) -> starkc::analysis::ProjectAnalysis {
    let graph =
        PackageGraph::load_from_root_with_modes(&root.join("starkpkg.json"), false, true).unwrap();
    analyze_project(ProjectInput::package(graph), LanguageOptions::CORE)
}

#[test]
fn a_package_entry_produces_two_source_records_for_one_physical_file() {
    let root = unique_root("dup");
    stage(&root, "probe");
    let analysis = analyse(&root);

    let names: Vec<&str> = analysis
        .source_map
        .files()
        .iter()
        .map(|r| r.file.name.as_str())
        .collect();

    let absolute: Vec<&&str> = names
        .iter()
        .filter(|n| std::path::Path::new(n).is_absolute())
        .collect();
    let logical: Vec<&&str> = names.iter().filter(|n| n.starts_with("probe/")).collect();

    // DEFECT: one physical entry file, two identities — the absolute path AND the logical name.
    assert_eq!(
        absolute.len(),
        1,
        "expected exactly one absolute-named record (the phantom root): {names:?}"
    );
    assert!(
        logical.iter().any(|n| n.ends_with("main.stark")),
        "expected a logical entry record as well: {names:?}"
    );
    assert!(
        absolute[0].ends_with("main.stark"),
        "the phantom record is the entry file under its absolute path: {names:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_real_entry_is_classified_as_a_module_with_no_package() {
    let root = unique_root("prov");
    stage(&root, "probe");
    let analysis = analyse(&root);

    let mut roots = Vec::new();
    let mut modules = Vec::new();
    for record in analysis.source_map.files() {
        match &record.provenance {
            SourceProvenance::Root { package } => {
                roots.push((record.file.name.clone(), package.clone()))
            }
            SourceProvenance::Module { package } => {
                modules.push((record.file.name.clone(), package.clone()))
            }
        }
    }

    // INVARIANT: exactly one record claims Root.
    assert_eq!(roots.len(), 1, "exactly one Root record: {roots:?}");

    // DEFECT: the record claiming Root is the phantom absolute one, not the logical entry.
    assert!(
        std::path::Path::new(&roots[0].0).is_absolute(),
        "the Root record is the absolute-path phantom: {roots:?}"
    );

    // DEFECT: every real package file lands as Module with NO package attribution, because
    // build_source_map infers the package by testing whether a source NAME starts with the
    // package entry's absolute parent — which a logical name never does.
    assert!(
        !modules.is_empty(),
        "the logical package files are present as modules: {modules:?}"
    );
    assert!(
        modules.iter().all(|(_, package)| package.is_none()),
        "no package file carries package attribution: {modules:?}"
    );
    assert!(
        modules.iter().any(|(name, _)| name.starts_with("probe/")),
        "the logical entry is among the modules: {modules:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn relocating_an_identical_package_changes_its_source_map() {
    let one = unique_root("relocA");
    let two = unique_root("relocB");
    stage(&one, "probe");
    stage(&two, "probe");
    assert_ne!(one, two, "the two roots must actually differ");

    let a = analyse(&one);
    let b = analyse(&two);

    let names = |analysis: &starkc::analysis::ProjectAnalysis| -> Vec<String> {
        analysis
            .source_map
            .files()
            .iter()
            .map(|r| r.file.name.clone())
            .collect()
    };
    let (na, nb) = (names(&a), names(&b));

    // DEFECT: identical sources at two roots do not observe identically. PKG-IDENTITY-001 says a
    // package token is "never an absolute checkout path", and §15.2 requires relocation stability.
    assert_ne!(
        na, nb,
        "source maps should differ today because one entry is named by absolute path"
    );

    // INVARIANT: the LOGICAL names are already relocation-stable — DEV-113 got that right. Only the
    // phantom absolute record moves, which is what localises the defect to one construction site.
    let logical = |v: &[String]| -> Vec<String> {
        let mut v: Vec<String> = v
            .iter()
            .filter(|n| n.starts_with("probe/"))
            .cloned()
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        logical(&na),
        logical(&nb),
        "logical names must already be relocation-stable"
    );
    assert!(
        !logical(&na).is_empty(),
        "a logical-name comparison over an empty set would be vacuous"
    );

    let _ = std::fs::remove_dir_all(&one);
    let _ = std::fs::remove_dir_all(&two);
}
