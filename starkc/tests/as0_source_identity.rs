//! AS0 reproduction → AS1a regression: one physical source, one logical identity.
//!
//! **History, because it is the point of the file.** AS0 committed these tests asserting the
//! DEFECTIVE behaviour: a package analysis produced two `SourceRecord`s for one entry file (the
//! absolute checkout path and the logical `<package>/<path>` name), the phantom absolute record was
//! the only thing classified `Root`, every package file carried `package: None`, and relocating the
//! package changed its source map. AS1a fixed it, those assertions failed, and they are flipped
//! here to pin the corrected behaviour.
//!
//! Assertions are labelled by the exit criterion they discharge, so a later change that weakens one
//! is traceable to the packet that promised it.
//!
//! AS1a exit criteria: 1 one `SourceRecord` per physical root; 2 the logical entry is the sole
//! `Root` and every package module carries non-empty package provenance; 3 relocation preserves
//! logical source maps and MIR file tables; 4 no absolute checkout path in reproducible identity;
//! 5 package, package-with-overlay and native-build paths share the helper.

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

fn source_names(analysis: &starkc::analysis::ProjectAnalysis) -> Vec<String> {
    let mut names: Vec<String> = analysis
        .source_map
        .files()
        .iter()
        .map(|r| r.file.name.clone())
        .collect();
    names.sort();
    names
}

#[test]
fn one_physical_file_produces_one_source_record() {
    let root = unique_root("dup");
    stage(&root, "probe");
    let analysis = analyse(&root);
    let names = source_names(&analysis);

    // AS1a-1: two files staged, two records. Before AS1a there were three, because the entry
    // appeared twice — once logically and once by absolute path.
    assert_eq!(
        names,
        vec!["probe/src/helper.stark", "probe/src/main.stark"],
        "one record per physical file, each named logically"
    );

    // AS1a-4: no absolute checkout path participates in identity at all.
    assert!(
        !names.iter().any(|n| std::path::Path::new(n).is_absolute()),
        "no absolute path may appear in the source map: {names:?}"
    );

    // The disk path survives as loading metadata — identity is logical, resolution is physical.
    let entry = analysis
        .source_map
        .files()
        .iter()
        .find(|r| r.file.name.ends_with("main.stark"))
        .expect("entry record present");
    // Compared in canonical form: the package graph canonicalises, and on macOS `/var` is a
    // symlink to `/private/var`, so the staged path and the stored one differ textually while
    // naming the same file.
    let disk = entry
        .file
        .disk_path
        .as_deref()
        .expect("the logical entry still knows where it physically is");
    assert_eq!(
        disk.canonicalize().unwrap(),
        root.join("src").join("main.stark").canonicalize().unwrap(),
        "the disk path resolves to the staged entry file"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_logical_entry_is_the_sole_root_and_every_file_carries_its_package() {
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

    // AS1a-2: exactly one Root, and it is the LOGICAL entry carrying its package.
    assert_eq!(
        roots,
        vec![(
            "probe/src/main.stark".to_string(),
            Some("probe".to_string())
        )],
        "the logical entry is the sole Root, attributed to its package"
    );

    // AS1a-2: every module carries non-empty package provenance. Before AS1a all of them were
    // `None`, because attribution tested whether a source NAME started with the entry's absolute
    // parent — which a logical name never does, leaving that branch dead on the package path.
    assert_eq!(
        modules,
        vec![(
            "probe/src/helper.stark".to_string(),
            Some("probe".to_string())
        )],
        "package modules carry their package"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn relocating_an_identical_package_preserves_its_source_map() {
    let one = unique_root("relocA");
    let two = unique_root("relocB");
    stage(&one, "probe");
    stage(&two, "probe");
    assert_ne!(one, two, "the two roots must actually differ");

    let a = analyse(&one);
    let b = analyse(&two);

    // AS1a-3: identical sources at two independent physical roots observe identically.
    // PKG-IDENTITY-001 ("never an absolute checkout path") and §15.2 (relocation stability).
    let (na, nb) = (source_names(&a), source_names(&b));
    assert_eq!(na, nb, "source maps must not depend on the checkout path");
    assert!(
        !na.is_empty(),
        "a source-map comparison over an empty set would be vacuous"
    );

    // The provenance must match too — equal names with different attribution would still be a
    // relocation difference, and comparing names alone would not catch it.
    let provenance = |analysis: &starkc::analysis::ProjectAnalysis| -> Vec<String> {
        let mut v: Vec<String> = analysis
            .source_map
            .files()
            .iter()
            .map(|r| format!("{} {:?}", r.file.name, r.provenance))
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        provenance(&a),
        provenance(&b),
        "provenance must not depend on the checkout path"
    );

    let _ = std::fs::remove_dir_all(&one);
    let _ = std::fs::remove_dir_all(&two);
}

#[test]
fn an_overlay_changes_the_entry_content_but_never_its_identity() {
    let root = unique_root("overlay");
    stage(&root, "probe");
    let graph =
        PackageGraph::load_from_root_with_modes(&root.join("starkpkg.json"), false, true).unwrap();
    let entry = graph.packages[&graph.root_package_name].entry.clone();

    let overlaid = "mod helper;\n\nfn main() {\n    let value: Int32 = helper::seven();\n    let extra: Int32 = 1;\n}\n";
    let mut overlays = std::collections::HashMap::new();
    overlays.insert(entry.clone(), overlaid.to_string());
    let analysis = analyze_project(
        ProjectInput::package_with_overlays(graph, overlays),
        LanguageOptions::CORE,
    );

    // AS1a-5: the overlay path shares the helper, so it produces the SAME identity as the plain
    // package path — this arm used to name the entry by its absolute path too.
    assert_eq!(
        source_names(&analysis),
        vec!["probe/src/helper.stark", "probe/src/main.stark"],
        "an overlaid package has the same logical source map as a plain one"
    );

    // ...and the overlay is what decides the CONTENT.
    let entry_record = analysis
        .source_map
        .files()
        .iter()
        .find(|r| r.file.name.ends_with("main.stark"))
        .expect("entry record present");
    assert_eq!(
        entry_record.file.src, overlaid,
        "the entry's content comes from the overlay, not from disk"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_mir_file_table_is_logical_and_relocation_stable() {
    let one = unique_root("mirA");
    let two = unique_root("mirB");
    stage(&one, "probe");
    stage(&two, "probe");

    let mir_files = |root: &std::path::Path| -> Vec<String> {
        let analysis = analyse(root);
        assert!(
            !analysis.has_errors(),
            "the staged package must analyse cleanly"
        );
        let hir = analysis.hir.as_ref().expect("hir");
        let tables = analysis.type_tables.as_ref().expect("tables");
        let program = match starkc::mir::lower::lower_program(
            hir,
            tables,
            hir.source_named(&analysis.root_file.name)
                .expect("the analysis registered its root"),
        ) {
            Ok(program) => program,
            Err(error) => panic!("lowering must succeed: {}", error.what),
        };
        // AS1b-iii: the MIR source table IS the compilation's registry; there is no separate
        // MIR-local file list to check for absolute names.
        let mut names: Vec<String> = program
            .sources
            .iter()
            .map(|source| source.name.clone())
            .collect();
        names.sort();
        names
    };

    let (a, b) = (mir_files(&one), mir_files(&two));

    // AS1a-4: the MIR source table feeds the native build key's `[sources]` section verbatim
    // (backend/generated_rust/build.rs). An absolute name there made the build key depend on where
    // the checkout happened to live.
    assert!(
        !a.iter().any(|n| std::path::Path::new(n).is_absolute()),
        "no absolute path may reach the MIR source table: {a:?}"
    );
    assert!(
        !a.is_empty(),
        "an absolute-path check over an empty source table would be vacuous"
    );

    // AS1a-3: and it is the same table from either root.
    assert_eq!(
        a, b,
        "MIR source tables must not depend on the checkout path"
    );

    let _ = std::fs::remove_dir_all(&one);
    let _ = std::fs::remove_dir_all(&two);
}
