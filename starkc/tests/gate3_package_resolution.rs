use starkc::options::LanguageOptions;
use starkc::package::{
    calculate_dir_sha256, discover_toolchain_package_root, find_package_root, Lockfile,
    PackageGraph, Version, VersionReq,
};
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

fn stage_toolchain_package(root: &std::path::Path, value: i32) {
    let package = root.join("math");
    let utility = root.join("utility");
    std::fs::create_dir_all(package.join("src")).unwrap();
    std::fs::create_dir_all(utility.join("src")).unwrap();
    std::fs::write(
        package.join("starkpkg.json"),
        r#"{"name":"math","version":"1.2.3","entry":"src/lib.stark","dependencies":{"utility":{"path":"../utility","version":"1.0.0"}}}"#,
    )
    .unwrap();
    std::fs::write(
        package.join("src/lib.stark"),
        format!("pub fn val() -> Int32 {{ {value} }}\n"),
    )
    .unwrap();
    std::fs::write(
        utility.join("starkpkg.json"),
        r#"{"name":"utility","version":"1.0.0","entry":"src/lib.stark"}"#,
    )
    .unwrap();
    std::fs::write(utility.join("src/lib.stark"), "pub fn helper() {}\n").unwrap();
}

fn stage_version_app(root: &std::path::Path) -> PathBuf {
    let app = root.join("app");
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::write(
        app.join("starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","dependencies":{"math":{"version":"^1.2.0"}}}"#,
    )
    .unwrap();
    std::fs::write(app.join("src/main.stark"), "fn main() {}\n").unwrap();
    app
}

#[test]
fn toolchain_package_root_is_discovered_from_archive_and_versioned_prefix_layouts() {
    let workspace = setup_temp_workspace("toolchain_discovery");
    let archive_packages = workspace.join("archive/lib/stark/packages");
    std::fs::create_dir_all(&archive_packages).unwrap();
    std::fs::create_dir_all(workspace.join("archive/bin")).unwrap();
    assert_eq!(
        discover_toolchain_package_root(Some(&workspace.join("archive/bin/stark"))).unwrap(),
        archive_packages.canonicalize().unwrap()
    );

    let prefix_packages = workspace.join("prefix/lib/stark/current/lib/stark/packages");
    std::fs::create_dir_all(&prefix_packages).unwrap();
    std::fs::create_dir_all(workspace.join("prefix/bin")).unwrap();
    assert_eq!(
        discover_toolchain_package_root(Some(&workspace.join("prefix/bin/stark"))).unwrap(),
        prefix_packages.canonicalize().unwrap()
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn toolchain_packages_resolve_offline_lock_without_install_paths() {
    let workspace = setup_temp_workspace("toolchain_root");
    let toolchain = workspace.join("installed/lib/stark/packages");
    stage_toolchain_package(&toolchain, 12);
    let app = stage_version_app(&workspace);
    let manifest = app.join("starkpkg.json");

    let graph = PackageGraph::load_from_root_with_modes_and_toolchain(
        &manifest,
        false,
        true,
        Some(&toolchain),
    )
    .unwrap();
    assert_eq!(graph.packages.get("math").unwrap().version_str(), "1.2.3");
    let lock_text = std::fs::read_to_string(app.join("stark.lock")).unwrap();
    let lock = Lockfile::parse(&lock_text).unwrap();
    assert_eq!(lock.packages["math"].source.as_deref(), Some("toolchain"));
    assert_eq!(
        lock.packages["utility"].source.as_deref(),
        Some("toolchain")
    );
    assert!(!lock_text.contains(&toolchain.display().to_string()));

    PackageGraph::load_from_root_with_modes_and_toolchain(&manifest, true, true, Some(&toolchain))
        .unwrap();
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn workspace_registry_precedes_toolchain_and_incompatible_toolchain_versions_are_named() {
    let workspace = setup_temp_workspace("toolchain_precedence");
    let toolchain = workspace.join("installed/lib/stark/packages");
    stage_toolchain_package(&toolchain, 12);
    let app = stage_version_app(&workspace);
    let registry = workspace.join("tmp/stark_registry/math/1.2.4");
    std::fs::create_dir_all(registry.join("src")).unwrap();
    std::fs::write(
        registry.join("starkpkg.json"),
        r#"{"name":"math","version":"1.2.4","entry":"src/lib.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        registry.join("src/lib.stark"),
        "pub fn val() -> Int32 { 124 }\n",
    )
    .unwrap();
    PackageGraph::load_from_root_with_modes_and_toolchain(
        &app.join("starkpkg.json"),
        false,
        false,
        Some(&toolchain),
    )
    .unwrap();
    let lock = Lockfile::parse(&std::fs::read_to_string(app.join("stark.lock")).unwrap()).unwrap();
    assert_eq!(lock.packages["math"].source.as_deref(), Some("registry"));
    assert_eq!(lock.packages["math"].version.patch, 4);

    std::fs::remove_dir_all(workspace.join("tmp")).unwrap();
    std::fs::remove_file(app.join("stark.lock")).unwrap();
    std::fs::write(
        app.join("starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","dependencies":{"math":{"version":"^2.0.0"}}}"#,
    )
    .unwrap();
    let error = PackageGraph::load_from_root_with_modes_and_toolchain(
        &app.join("starkpkg.json"),
        false,
        true,
        Some(&toolchain),
    )
    .unwrap_err();
    assert!(error.contains("^2.0.0"), "{error}");
    assert!(error.contains("1.2.3"), "{error}");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn toolchain_lock_entries_are_identical_across_install_prefixes_and_path_wins() {
    let first = setup_temp_workspace("toolchain_prefix_a");
    let second = setup_temp_workspace("toolchain_prefix_b");
    for root in [&first, &second] {
        let toolchain = root.join("prefix/lib/stark/packages");
        stage_toolchain_package(&toolchain, 12);
        let app = stage_version_app(root);
        PackageGraph::load_from_root_with_modes_and_toolchain(
            &app.join("starkpkg.json"),
            false,
            false,
            Some(&toolchain),
        )
        .unwrap();
    }
    let first_lock = std::fs::read_to_string(first.join("app/stark.lock")).unwrap();
    let second_lock = std::fs::read_to_string(second.join("app/stark.lock")).unwrap();
    assert_eq!(first_lock, second_lock);

    let path_math = first.join("path-math");
    stage_toolchain_package(&first, 99);
    std::fs::rename(first.join("math"), &path_math).unwrap();
    std::fs::write(
        first.join("app/starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","dependencies":{"math":{"path":"../path-math","version":"^1.2.0"}}}"#,
    )
    .unwrap();
    let graph = PackageGraph::load_from_root_with_modes_and_toolchain(
        &first.join("app/starkpkg.json"),
        false,
        false,
        Some(&first.join("prefix/lib/stark/packages")),
    )
    .unwrap();
    // `PackageGraph` canonicalizes every dependency path, and on Windows canonicalization adds
    // the `\\?\` verbatim prefix -- so the stored manifest path and this locally-built one are the
    // same directory in two different representations, and `starts_with` is false. Canonicalize
    // both sides before comparing. `package.rs` already documents this for the graph root; the
    // same rule applies to anything compared against it.
    let path_math_canonical = path_math.canonicalize().unwrap();
    assert!(
        graph.packages["math"]
            .manifest_path
            .starts_with(&path_math_canonical),
        "expected {} to be under {}",
        graph.packages["math"].manifest_path.display(),
        path_math_canonical.display()
    );

    let _ = std::fs::remove_dir_all(&first);
    let _ = std::fs::remove_dir_all(&second);
}

#[test]
fn toolchain_packages_contribute_the_transitive_capability_envelope() {
    let workspace = setup_temp_workspace("toolchain_capabilities");
    let app = workspace.join("app");
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::write(app.join("src/main.stark"), "fn main() {}\n").unwrap();
    std::fs::write(
        app.join("starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","capabilities":[],"dependencies":{"stark_io":{"package":"stark-io","version":"0.1.0"}}}"#,
    )
    .unwrap();
    let packages = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../packages");
    let error = PackageGraph::load_from_root_with_modes_and_toolchain(
        &app.join("starkpkg.json"),
        false,
        true,
        Some(&packages),
    )
    .unwrap_err();
    for expected in ["filesystem-read", "stark_io", "provider_api.functions"] {
        assert!(error.contains(expected), "missing {expected:?}: {error}");
    }
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_version_req_parsing_and_matching() {
    let req_any = VersionReq::parse("*").unwrap();
    assert!(req_any.matches(&Version {
        major: 1,
        minor: 2,
        patch: 3
    }));

    let req_caret = VersionReq::parse("^1.2.3").unwrap();
    assert!(req_caret.matches(&Version {
        major: 1,
        minor: 2,
        patch: 3
    }));
    assert!(req_caret.matches(&Version {
        major: 1,
        minor: 3,
        patch: 0
    }));
    assert!(!req_caret.matches(&Version {
        major: 1,
        minor: 1,
        patch: 0
    }));
    assert!(!req_caret.matches(&Version {
        major: 2,
        minor: 0,
        patch: 0
    }));

    let req_range = VersionReq::parse(">=1.2.0, <2.0.0").unwrap();
    assert!(req_range.matches(&Version {
        major: 1,
        minor: 2,
        patch: 0
    }));
    assert!(req_range.matches(&Version {
        major: 1,
        minor: 9,
        patch: 9
    }));
    assert!(!req_range.matches(&Version {
        major: 2,
        minor: 0,
        patch: 0
    }));
}

#[test]
fn test_full_reproducible_resolution_pipeline() {
    let workspace = setup_temp_workspace("reproducible");

    let reg_dir = workspace.join("tmp/stark_registry");
    let cache_dir = workspace.join("tmp/stark_cache");

    let math_1_0_0 = reg_dir.join("math/1.0.0");
    let math_1_1_0 = reg_dir.join("math/1.1.0");
    let math_1_2_0 = reg_dir.join("math/1.2.0");
    let math_2_0_0 = reg_dir.join("math/2.0.0");

    for dir in &[&math_1_0_0, &math_1_1_0, &math_1_2_0, &math_2_0_0] {
        std::fs::create_dir_all(dir.join("src")).unwrap();
    }

    std::fs::write(
        math_1_0_0.join("starkpkg.json"),
        r#"{"name": "math", "version": "1.0.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        math_1_0_0.join("src/main.stark"),
        "pub fn val() -> Int32 { 10 }",
    )
    .unwrap();

    std::fs::write(
        math_1_1_0.join("starkpkg.json"),
        r#"{"name": "math", "version": "1.1.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        math_1_1_0.join("src/main.stark"),
        "pub fn val() -> Int32 { 11 }",
    )
    .unwrap();

    std::fs::write(
        math_1_2_0.join("starkpkg.json"),
        r#"{"name": "math", "version": "1.2.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        math_1_2_0.join("src/main.stark"),
        "pub fn val() -> Int32 { 12 }",
    )
    .unwrap();

    std::fs::write(
        math_2_0_0.join("starkpkg.json"),
        r#"{"name": "math", "version": "2.0.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        math_2_0_0.join("src/main.stark"),
        "pub fn val() -> Int32 { 20 }",
    )
    .unwrap();

    let app_dir = workspace.join("app");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "math": { "version": "^1.1.0" }
        }
    }"#,
    )
    .unwrap();

    std::fs::write(
        app_dir.join("src/main.stark"),
        r#"
        use math::val;
        fn main() {
            let _x = val();
        }
    "#,
    )
    .unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root_with_modes(&manifest_path, false, false).unwrap();

    let resolved_math = graph.packages.get("math").unwrap();
    assert_eq!(
        resolved_math.version,
        Version {
            major: 1,
            minor: 2,
            patch: 0
        }
    );

    let lock_path = app_dir.join("stark.lock");
    assert!(lock_path.exists());
    let lockfile = Lockfile::parse(&std::fs::read_to_string(&lock_path).unwrap()).unwrap();
    let lock_pkg = lockfile.packages.get("math").unwrap();
    assert_eq!(
        lock_pkg.version,
        Version {
            major: 1,
            minor: 2,
            patch: 0
        }
    );

    let cached_pkg_dir = cache_dir.join("math/1.2.0");
    assert!(cached_pkg_dir.exists());
    let expected_hash = calculate_dir_sha256(&cached_pkg_dir).unwrap();
    assert_eq!(lock_pkg.sha256, expected_hash);

    let (ast, mut diags) = parse_package_graph(&graph, LanguageOptions::CORE);
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

    let mut tc_diags = typecheck::check(&hir);
    diags.append(&mut tc_diags);
    assert!(diags.is_empty(), "typecheck failed: {:?}", diags);

    let graph_locked =
        PackageGraph::load_from_root_with_modes(&manifest_path, true, false).unwrap();
    assert_eq!(
        graph_locked.packages.get("math").unwrap().version,
        Version {
            major: 1,
            minor: 2,
            patch: 0
        }
    );

    let graph_offline =
        PackageGraph::load_from_root_with_modes(&manifest_path, false, true).unwrap();
    assert_eq!(
        graph_offline.packages.get("math").unwrap().version,
        Version {
            major: 1,
            minor: 2,
            patch: 0
        }
    );

    let _ = std::fs::remove_dir_all(&cached_pkg_dir);
    let result_offline_fail = PackageGraph::load_from_root_with_modes(&manifest_path, false, true);
    assert!(result_offline_fail.is_err());
    assert!(result_offline_fail.unwrap_err().contains("offline mode"));

    std::fs::create_dir_all(cached_pkg_dir.join("src")).unwrap();
    std::fs::write(
        cached_pkg_dir.join("starkpkg.json"),
        r#"{"name": "math", "version": "1.2.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        cached_pkg_dir.join("src/main.stark"),
        "pub fn val() -> Int32 { 999 }",
    )
    .unwrap();

    let result_corrupted = PackageGraph::load_from_root_with_modes(&manifest_path, true, false);
    assert!(result_corrupted.is_err());
    assert!(result_corrupted
        .unwrap_err()
        .contains("content hash mismatch"));

    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn test_conflicting_constraints_rejected() {
    let workspace = setup_temp_workspace("conflicts");

    let reg_dir = workspace.join("tmp/stark_registry");
    let math_1_0_0 = reg_dir.join("math/1.0.0");
    let math_2_0_0 = reg_dir.join("math/2.0.0");
    std::fs::create_dir_all(math_1_0_0.join("src")).unwrap();
    std::fs::create_dir_all(math_2_0_0.join("src")).unwrap();
    std::fs::write(
        math_1_0_0.join("starkpkg.json"),
        r#"{"name": "math", "version": "1.0.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(math_1_0_0.join("src/main.stark"), "pub fn val() {}").unwrap();
    std::fs::write(
        math_2_0_0.join("starkpkg.json"),
        r#"{"name": "math", "version": "2.0.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(math_2_0_0.join("src/main.stark"), "pub fn val() {}").unwrap();

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
            "math": { "version": "^3.0.0" }
        }
    }"#,
    )
    .unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let result = PackageGraph::load_from_root(&manifest_path);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("no compatible version of 'math' found"));

    let _ = std::fs::remove_dir_all(&workspace);
}
