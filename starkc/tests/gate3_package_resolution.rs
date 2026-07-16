use std::path::PathBuf;
use std::sync::Arc;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, calculate_dir_sha256, PackageGraph, Version, VersionReq, Lockfile};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;

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
fn test_version_req_parsing_and_matching() {
    let req_any = VersionReq::parse("*").unwrap();
    assert!(req_any.matches(&Version { major: 1, minor: 2, patch: 3 }));
    
    let req_caret = VersionReq::parse("^1.2.3").unwrap();
    assert!(req_caret.matches(&Version { major: 1, minor: 2, patch: 3 }));
    assert!(req_caret.matches(&Version { major: 1, minor: 3, patch: 0 }));
    assert!(!req_caret.matches(&Version { major: 1, minor: 1, patch: 0 }));
    assert!(!req_caret.matches(&Version { major: 2, minor: 0, patch: 0 }));

    let req_range = VersionReq::parse(">=1.2.0, <2.0.0").unwrap();
    assert!(req_range.matches(&Version { major: 1, minor: 2, patch: 0 }));
    assert!(req_range.matches(&Version { major: 1, minor: 9, patch: 9 }));
    assert!(!req_range.matches(&Version { major: 2, minor: 0, patch: 0 }));
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
    
    std::fs::write(math_1_0_0.join("starkpkg.json"), r#"{"name": "math", "version": "1.0.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(math_1_0_0.join("src/main.stark"), "pub fn val() -> Int32 { 10 }").unwrap();
    
    std::fs::write(math_1_1_0.join("starkpkg.json"), r#"{"name": "math", "version": "1.1.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(math_1_1_0.join("src/main.stark"), "pub fn val() -> Int32 { 11 }").unwrap();
    
    std::fs::write(math_1_2_0.join("starkpkg.json"), r#"{"name": "math", "version": "1.2.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(math_1_2_0.join("src/main.stark"), "pub fn val() -> Int32 { 12 }").unwrap();
    
    std::fs::write(math_2_0_0.join("starkpkg.json"), r#"{"name": "math", "version": "2.0.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(math_2_0_0.join("src/main.stark"), "pub fn val() -> Int32 { 20 }").unwrap();

    let app_dir = workspace.join("app");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::write(app_dir.join("starkpkg.json"), r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "math": { "version": "^1.1.0" }
        }
    }"#).unwrap();
    
    std::fs::write(app_dir.join("src/main.stark"), r#"
        use math::val;
        fn main() {
            let _x = val();
        }
    "#).unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root_with_modes(&manifest_path, false, false).unwrap();
    
    let resolved_math = graph.packages.get("math").unwrap();
    assert_eq!(resolved_math.version, Version { major: 1, minor: 2, patch: 0 });
    
    let lock_path = app_dir.join("stark.lock");
    assert!(lock_path.exists());
    let lockfile = Lockfile::parse(&std::fs::read_to_string(&lock_path).unwrap()).unwrap();
    let lock_pkg = lockfile.packages.get("math").unwrap();
    assert_eq!(lock_pkg.version, Version { major: 1, minor: 2, patch: 0 });
    
    let cached_pkg_dir = cache_dir.join("math/1.2.0");
    assert!(cached_pkg_dir.exists());
    let expected_hash = calculate_dir_sha256(&cached_pkg_dir).unwrap();
    assert_eq!(lock_pkg.sha256, expected_hash);

    let (ast, mut diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);
    
    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(app_dir.join("src/main.stark").to_string_lossy().into_owned(), entry_src));
    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diags.append(&mut resolution);
    assert!(diags.is_empty(), "resolution failed: {:?}", diags);
    
    let mut tc_diags = typecheck::check(&hir, root_file);
    diags.append(&mut tc_diags);
    assert!(diags.is_empty(), "typecheck failed: {:?}", diags);

    let graph_locked = PackageGraph::load_from_root_with_modes(&manifest_path, true, false).unwrap();
    assert_eq!(graph_locked.packages.get("math").unwrap().version, Version { major: 1, minor: 2, patch: 0 });

    let graph_offline = PackageGraph::load_from_root_with_modes(&manifest_path, false, true).unwrap();
    assert_eq!(graph_offline.packages.get("math").unwrap().version, Version { major: 1, minor: 2, patch: 0 });

    let _ = std::fs::remove_dir_all(&cached_pkg_dir);
    let result_offline_fail = PackageGraph::load_from_root_with_modes(&manifest_path, false, true);
    assert!(result_offline_fail.is_err());
    assert!(result_offline_fail.unwrap_err().contains("offline mode"));

    std::fs::create_dir_all(cached_pkg_dir.join("src")).unwrap();
    std::fs::write(cached_pkg_dir.join("starkpkg.json"), r#"{"name": "math", "version": "1.2.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(cached_pkg_dir.join("src/main.stark"), "pub fn val() -> Int32 { 999 }").unwrap();
    
    let result_corrupted = PackageGraph::load_from_root_with_modes(&manifest_path, true, false);
    assert!(result_corrupted.is_err());
    assert!(result_corrupted.unwrap_err().contains("content hash mismatch"));

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
    std::fs::write(math_1_0_0.join("starkpkg.json"), r#"{"name": "math", "version": "1.0.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(math_1_0_0.join("src/main.stark"), "pub fn val() {}").unwrap();
    std::fs::write(math_2_0_0.join("starkpkg.json"), r#"{"name": "math", "version": "2.0.0", "entry": "src/main.stark"}"#).unwrap();
    std::fs::write(math_2_0_0.join("src/main.stark"), "pub fn val() {}").unwrap();

    let app_dir = workspace.join("app");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::write(app_dir.join("src/main.stark"), "fn main() {}").unwrap();
    
    std::fs::write(app_dir.join("starkpkg.json"), r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "math": { "version": "^3.0.0" }
        }
    }"#).unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let result = PackageGraph::load_from_root(&manifest_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("no compatible version of 'math' found"));

    let _ = std::fs::remove_dir_all(&workspace);
}
