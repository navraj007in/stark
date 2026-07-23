use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

fn setup_temp_workspace(name: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(format!("temp_workspace_{}", name));
    if base.exists() {
        let _ = std::fs::remove_dir_all(&base);
    }
    std::fs::create_dir_all(&base).unwrap();
    base
}

#[test]
fn test_three_package_workspace_compiles() {
    let workspace = setup_temp_workspace("three_pkg");

    let app_dir = workspace.join("app");
    let ip_dir = workspace.join("image-pipeline");
    let util_dir = workspace.join("utilities");

    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(ip_dir.join("src")).unwrap();
    std::fs::create_dir_all(util_dir.join("src")).unwrap();

    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "image_pipeline": { "path": "../image-pipeline" },
            "utilities": { "path": "../utilities" }
        }
    }"#,
    )
    .unwrap();

    std::fs::write(
        ip_dir.join("starkpkg.json"),
        r#"{
        "name": "image_pipeline",
        "version": "0.2.0",
        "entry": "src/main.stark",
        "dependencies": {
            "utilities": { "path": "../utilities" }
        }
    }"#,
    )
    .unwrap();

    std::fs::write(
        util_dir.join("starkpkg.json"),
        r#"{
        "name": "utilities",
        "version": "0.3.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();

    std::fs::write(
        util_dir.join("src/main.stark"),
        r#"
        pub fn get_pi() -> Float64 { 3.14159 }
    "#,
    )
    .unwrap();

    std::fs::write(
        ip_dir.join("src/main.stark"),
        r#"
        use utilities::get_pi;
        pub fn process_image() -> Float64 {
            get_pi()
        }
    "#,
    )
    .unwrap();

    std::fs::write(
        app_dir.join("src/main.stark"),
        r#"
        use image_pipeline::process_image;
        use utilities::get_pi;
        
        fn main() {
            let _p = process_image();
            let _val = get_pi();
        }
    "#,
    )
    .unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();

    let options = LanguageOptions::CORE;
    let (ast, mut diags) = parse_package_graph(&graph, options);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));

    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diags.append(&mut resolution);
    assert!(diags.is_empty(), "resolution failed: {:?}", diags);

    let mut tc_diags = typecheck::check(&hir, root_file);
    diags.append(&mut tc_diags);
    assert!(diags.is_empty(), "typecheck failed: {:?}", diags);

    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_dependency_alias_is_distinct_from_canonical_package_name() {
    let workspace = setup_temp_workspace("canonical_alias");
    let app_dir = workspace.join("app");
    let tensor_dir = workspace.join("tensor-lib");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(tensor_dir.join("src")).unwrap();

    std::fs::write(
        tensor_dir.join("starkpkg.json"),
        r#"{
            "name": "tensor-lib",
            "version": "2.1.0",
            "entry": "src/main.stark"
        }"#,
    )
    .unwrap();
    std::fs::write(
        tensor_dir.join("src/main.stark"),
        "pub fn answer() -> Int32 { 42 }",
    )
    .unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
            "name": "app",
            "version": "0.1.0",
            "entry": "src/main.stark",
            "dependencies": {
                "TensorLib": {
                    "package": "tensor-lib",
                    "path": "../tensor-lib"
                }
            }
        }"#,
    )
    .unwrap();
    std::fs::write(
        app_dir.join("src/main.stark"),
        "use TensorLib::answer; fn main() { let value: Int32 = answer(); }",
    )
    .unwrap();

    let graph = PackageGraph::load_from_root(&app_dir.join("starkpkg.json")).unwrap();
    assert_eq!(graph.packages.get("TensorLib").unwrap().name, "tensor-lib");
    let (ast, mut diagnostics) = parse_package_graph(&graph, LanguageOptions::CORE);
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap(),
    ));
    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diagnostics.append(&mut resolution);
    diagnostics.append(&mut typecheck::check(&hir, root_file));
    assert!(
        diagnostics.is_empty(),
        "alias-based package import failed: {diagnostics:?}"
    );

    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_manifest_rejects_aliases_that_are_not_source_identifiers() {
    let workspace = setup_temp_workspace("invalid_alias");
    let app_dir = workspace.join("app");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
            "name": "app",
            "version": "0.1.0",
            "dependencies": {
                "tensor-lib": { "package": "tensor-lib", "version": "^2.1.0" }
            }
        }"#,
    )
    .unwrap();

    let error = PackageGraph::load_from_root(&app_dir.join("starkpkg.json")).unwrap_err();
    assert!(error.contains("non-keyword STARK identifier"), "{error}");
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_manifest_rejects_constraints_spanning_major_lines() {
    let workspace = setup_temp_workspace("multi_major_constraint");
    let app_dir = workspace.join("app");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
            "name": "app",
            "version": "0.1.0",
            "dependencies": {
                "TensorLib": {
                    "package": "tensor-lib",
                    "version": ">=1.2.0, <3.0.0"
                }
            }
        }"#,
    )
    .unwrap();

    let error = PackageGraph::load_from_root(&app_dir.join("starkpkg.json")).unwrap_err();
    assert!(error.contains("exactly one major version line"), "{error}");
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_distinct_aliases_allow_incompatible_major_lines_to_coexist() {
    let workspace = setup_temp_workspace("major_coexistence");
    let app_dir = workspace.join("app");
    let v1_dir = workspace.join("tensor-v1");
    let v2_dir = workspace.join("tensor-v2");
    for directory in [&app_dir, &v1_dir, &v2_dir] {
        std::fs::create_dir_all(directory.join("src")).unwrap();
    }
    for (directory, version, value) in [(&v1_dir, "1.9.0", 1), (&v2_dir, "2.3.0", 2)] {
        std::fs::write(
            directory.join("starkpkg.json"),
            format!(
                r#"{{
                    "name": "tensor-lib",
                    "version": "{version}",
                    "entry": "src/main.stark"
                }}"#
            ),
        )
        .unwrap();
        std::fs::write(
            directory.join("src/main.stark"),
            format!("pub fn major() -> Int32 {{ {value} }}"),
        )
        .unwrap();
    }
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
            "name": "app",
            "version": "0.1.0",
            "entry": "src/main.stark",
            "dependencies": {
                "TensorV1": { "package": "tensor-lib", "path": "../tensor-v1" },
                "TensorV2": { "package": "tensor-lib", "path": "../tensor-v2" }
            }
        }"#,
    )
    .unwrap();
    std::fs::write(
        app_dir.join("src/main.stark"),
        "fn main() { let one = TensorV1::major(); let two = TensorV2::major(); println(one + two); }",
    )
    .unwrap();

    let graph = PackageGraph::load_from_root(&app_dir.join("starkpkg.json")).unwrap();
    assert_eq!(
        graph.packages.get("TensorV1").unwrap().version,
        starkc::package::Version {
            major: 1,
            minor: 9,
            patch: 0,
        }
    );
    assert_eq!(
        graph.packages.get("TensorV2").unwrap().version,
        starkc::package::Version {
            major: 2,
            minor: 3,
            patch: 0,
        }
    );
    let (ast, mut diagnostics) = parse_package_graph(&graph, LanguageOptions::CORE);
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap(),
    ));
    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diagnostics.append(&mut resolution);
    diagnostics.append(&mut typecheck::check(&hir, root_file));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn test_missing_manifest_rejected() {
    let workspace = setup_temp_workspace("missing_manifest");
    let result = find_package_root(&workspace);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("missing manifest"));
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_invalid_entry_rejected() {
    let workspace = setup_temp_workspace("invalid_entry");
    let app_dir = workspace.join("app");
    std::fs::create_dir_all(&app_dir).unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/nonexistent.stark"
    }"#,
    )
    .unwrap();

    let result = PackageGraph::load_from_root(&app_dir.join("starkpkg.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("does not exist"));
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_dependency_cycle_rejected() {
    let workspace = setup_temp_workspace("cycle");
    let a_dir = workspace.join("a");
    let b_dir = workspace.join("b");
    std::fs::create_dir_all(a_dir.join("src")).unwrap();
    std::fs::create_dir_all(b_dir.join("src")).unwrap();

    std::fs::write(a_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(b_dir.join("src/main.stark"), "fn main() {}").unwrap();

    std::fs::write(
        a_dir.join("starkpkg.json"),
        r#"{
        "name": "a",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": { "b": { "path": "../b" } }
    }"#,
    )
    .unwrap();

    std::fs::write(
        b_dir.join("starkpkg.json"),
        r#"{
        "name": "b",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": { "a": { "path": "../a" } }
    }"#,
    )
    .unwrap();

    let result = PackageGraph::load_from_root(&a_dir.join("starkpkg.json"));
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("dependency cycle detected: a -> b -> a"));
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_duplicate_package_names_rejected() {
    let workspace = setup_temp_workspace("duplicate");
    let app_dir = workspace.join("app");
    let d1_dir = workspace.join("d1");
    let d2_dir = workspace.join("d2");
    let dup1_dir = workspace.join("dup1");
    let dup2_dir = workspace.join("dup2");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(d1_dir.join("src")).unwrap();
    std::fs::create_dir_all(d2_dir.join("src")).unwrap();
    std::fs::create_dir_all(dup1_dir.join("src")).unwrap();
    std::fs::create_dir_all(dup2_dir.join("src")).unwrap();

    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(d1_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(d2_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(dup1_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(dup2_dir.join("src/main.stark"), "fn main() {}").unwrap();

    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "d1": { "path": "../d1" },
            "d2": { "path": "../d2" }
        }
    }"#,
    )
    .unwrap();

    std::fs::write(
        d1_dir.join("starkpkg.json"),
        r#"{
        "name": "d1",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "dup": { "path": "../dup1" }
        }
    }"#,
    )
    .unwrap();

    std::fs::write(
        d2_dir.join("starkpkg.json"),
        r#"{
        "name": "d2",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "dup": { "path": "../dup2" }
        }
    }"#,
    )
    .unwrap();

    std::fs::write(
        dup1_dir.join("starkpkg.json"),
        r#"{
        "name": "dup",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();

    std::fs::write(
        dup2_dir.join("starkpkg.json"),
        r#"{
        "name": "dup",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();

    let result = PackageGraph::load_from_root(&app_dir.join("starkpkg.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("duplicate package name 'dup'"));
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_workspace_boundary_rejected() {
    let workspace = setup_temp_workspace("boundary");
    let app_dir = workspace.join("app");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();

    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "outside": { "path": "../../" }
        }
    }"#,
    )
    .unwrap();

    let result = PackageGraph::load_from_root(&app_dir.join("starkpkg.json"));
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("outside the permitted workspace"));

    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_noncanonical_root_uses_canonical_workspace_paths() {
    let workspace = setup_temp_workspace("noncanonical_root");
    let app_dir = workspace.join("app");
    let dep_dir = workspace.join("dep");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(dep_dir.join("src")).unwrap();
    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();
    std::fs::write(
        dep_dir.join("src/main.stark"),
        "pub fn value() -> Int32 { 1 }",
    )
    .unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "dep": { "path": "../dep" }
        }
    }"#,
    )
    .unwrap();
    std::fs::write(
        dep_dir.join("starkpkg.json"),
        r#"{
        "name": "dep",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();

    let noncanonical_manifest = app_dir.join("..").join("app").join("starkpkg.json");
    let graph = PackageGraph::load_from_root(&noncanonical_manifest).unwrap();
    assert_eq!(graph.workspace_root, workspace.canonicalize().unwrap());
    assert_eq!(
        graph.packages["app"].manifest_path,
        app_dir.join("starkpkg.json").canonicalize().unwrap()
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

/// WP-C1.2 (checklist item 8): `use`-ing a package that exists elsewhere in the same workspace
/// but is NOT listed in the *importing* package's own `starkpkg.json` dependencies -- distinct
/// from a dependency that fails to fetch/resolve (test_missing_manifest_rejected,
/// test_invalid_entry_rejected above) or a cycle (test_dependency_cycle_rejected). Per
/// parser.rs's `parse_package_rec`, only names in `pkg.dependencies.keys()` get a synthesized
/// `mod <dep_name> { ... }` wrapper, so an undeclared sibling degrades to an ordinary
/// unresolved-name error rather than a dedicated diagnostic -- this test pins that behavior
/// down since it had no prior test evidence of any kind.
#[test]
fn test_undeclared_dependency_import_is_rejected() {
    let workspace = setup_temp_workspace("undeclared_dep");
    let app_dir = workspace.join("app");
    let sibling_dir = workspace.join("sibling");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(sibling_dir.join("src")).unwrap();

    std::fs::write(
        sibling_dir.join("starkpkg.json"),
        r#"{
        "name": "sibling",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();
    std::fs::write(
        sibling_dir.join("src/main.stark"),
        "pub fn thing() -> Int32 { 1 }",
    )
    .unwrap();

    // Note: "sibling" is NOT listed in app's dependencies, even though it exists on disk right
    // next to app and would resolve fine if app's manifest declared it.
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();
    std::fs::write(
        app_dir.join("src/main.stark"),
        "use sibling::thing;\nfn main() { let _x = thing(); }",
    )
    .unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
    let options = LanguageOptions::CORE;
    let (ast, mut diags) = parse_package_graph(&graph, options);

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));
    let (_hir, mut resolution) = resolve(&ast, root_file);
    diags.append(&mut resolution);

    assert!(
        !diags.is_empty(),
        "expected an error for `use sibling::thing;` where 'sibling' is not a declared \
         dependency of 'app', got a clean resolve"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

/// WP-C1.2 (checklist item 9): a resolve-stage error INSIDE a dependency package's own source
/// (not the root package) must render with that dependency's filename, not the root package's.
/// This exercises the cross-package file-attribution path described in resolve.rs's
/// declare_items (deriving a synthesized dependency-wrapper module's file from its first real
/// child item, since the wrapper item itself is attributed to the *importing* package by
/// parser.rs's parse_package_rec) -- a distinct, previously-unverified risk beyond DEV-006's
/// same-package fix (see resolve_diagnostics_carry_their_own_file_not_the_caller_default in
/// resolve.rs, which only covers a single-file case).
#[test]
fn test_cross_package_diagnostic_reports_dependency_file_not_root_file() {
    let workspace = setup_temp_workspace("cross_pkg_diag");
    let app_dir = workspace.join("app");
    let dep_dir = workspace.join("dep");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(dep_dir.join("src")).unwrap();

    std::fs::write(
        dep_dir.join("starkpkg.json"),
        r#"{
        "name": "dep",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();
    // A genuine duplicate-definition error, entirely inside the dependency's own source.
    std::fs::write(
        dep_dir.join("src/main.stark"),
        "pub fn broken() -> Int32 { 1 }\npub fn broken() -> Int32 { 2 }",
    )
    .unwrap();

    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": { "dep": { "path": "../dep" } }
    }"#,
    )
    .unwrap();
    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
    let options = LanguageOptions::CORE;
    let (ast, mut diags) = parse_package_graph(&graph, options);

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));
    let (_hir, mut resolution) = resolve(&ast, root_file);
    diags.append(&mut resolution);

    let dup = diags
        .iter()
        .find(|d| d.code.as_deref() == Some("E0204"))
        .unwrap_or_else(|| panic!("expected E0204 for the duplicate 'broken' fn, got {diags:?}"));
    let dup_file = dup
        .file
        .as_ref()
        .expect("diagnostic should carry a file identity");
    assert!(
        dup_file.name.contains("dep") && dup_file.name.ends_with("main.stark"),
        "expected the diagnostic's file to be the dependency's src/main.stark, got {:?}",
        dup_file.name
    );
    assert!(
        !dup_file
            .name
            .contains(app_dir.join("src").to_string_lossy().as_ref()),
        "diagnostic incorrectly attributed to the root package's own src directory: {:?}",
        dup_file.name
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

/// WP-C1.2 (checklist item 10): coherence checking (SEM-007, orphan rule / overlapping impls)
/// is implemented in typecheck.rs via `find_package_root`, a pure filesystem walk-up from each
/// item's own file path -- entirely independent of the `PackageGraph` object built here. No
/// prior test exercised this with a REAL two-package workspace (both `resolve.rs`'s unit tests
/// and typecheck.rs's own coherence tests use a bare in-memory "test.stark" with no
/// `starkpkg.json` on disk anywhere, which the WP-C1.2 research confirmed makes
/// `find_package_root` return `None` for every existing test). This test builds a real
/// workspace to observe actual behavior, whatever it is, rather than assume it.
#[test]
fn test_cross_package_coherence_orphan_rule_with_real_packages() {
    let workspace = setup_temp_workspace("coherence");
    let app_dir = workspace.join("app");
    let dep_dir = workspace.join("dep");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(dep_dir.join("src")).unwrap();

    std::fs::write(
        dep_dir.join("starkpkg.json"),
        r#"{
        "name": "dep",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#,
    )
    .unwrap();
    // A trait AND a struct, both declared in the dependency package -- an impl of this trait for
    // this struct, from the ROOT package, is a textbook orphan-rule violation (neither the trait
    // nor the type is local to the implementing package).
    std::fs::write(
        dep_dir.join("src/main.stark"),
        "pub trait Greet { fn greet(&self) -> Int32; }\npub struct Foreign { pub v: Int32 }",
    )
    .unwrap();

    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": { "dep": { "path": "../dep" } }
    }"#,
    )
    .unwrap();
    std::fs::write(
        app_dir.join("src/main.stark"),
        "use dep::Greet;\nuse dep::Foreign;\nimpl Greet for Foreign {\n    fn greet(&self) -> Int32 { self.v }\n}\nfn main() {}",
    )
    .unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
    let options = LanguageOptions::CORE;
    let (ast, mut diags) = parse_package_graph(&graph, options);

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));
    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diags.append(&mut resolution);
    assert!(diags.is_empty(), "resolution failed: {:?}", diags);
    let mut tc_diags = typecheck::check(&hir, root_file);
    diags.append(&mut tc_diags);

    // Report, don't assume: this is exactly the untested case the WP-C1.2 research flagged.
    // Whichever way this resolves is recorded as evidence in COMPILER-STATE.md/KNOWN-DEVIATIONS
    // rather than silently accepted -- see DEV-021.
    let orphan_flagged = diags.iter().any(|d| d.code.as_deref() == Some("E0500"));
    assert!(
        orphan_flagged,
        "cross-package orphan-rule violation (impl of dep::Greet for dep::Foreign, both from a \
         dependency package, written in the root package) was NOT flagged with E0500 -- this is \
         the real, previously-unverified behavior this test exists to pin down. If this \
         assertion is failing, that confirms cross-package coherence checking does not actually \
         work end-to-end (find_package_root's cross-package attribution has a gap), which is a \
         real conformance issue to record, not a test bug to silence by relaxing this assertion."
    );

    let _ = std::fs::remove_dir_all(&workspace);
}
