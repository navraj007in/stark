use std::path::PathBuf;
use std::sync::Arc;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
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
fn test_three_package_workspace_compiles() {
    let workspace = setup_temp_workspace("three_pkg");
    
    let app_dir = workspace.join("app");
    let ip_dir = workspace.join("image-pipeline");
    let util_dir = workspace.join("utilities");
    
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(ip_dir.join("src")).unwrap();
    std::fs::create_dir_all(util_dir.join("src")).unwrap();
    
    std::fs::write(app_dir.join("starkpkg.json"), r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "image_pipeline": { "path": "../image-pipeline" },
            "utilities": { "path": "../utilities" }
        }
    }"#).unwrap();
    
    std::fs::write(ip_dir.join("starkpkg.json"), r#"{
        "name": "image_pipeline",
        "version": "0.2.0",
        "entry": "src/main.stark",
        "dependencies": {
            "utilities": { "path": "../utilities" }
        }
    }"#).unwrap();
    
    std::fs::write(util_dir.join("starkpkg.json"), r#"{
        "name": "utilities",
        "version": "0.3.0",
        "entry": "src/main.stark"
    }"#).unwrap();
    
    std::fs::write(util_dir.join("src/main.stark"), r#"
        pub fn get_pi() -> Float64 { 3.14159 }
    "#).unwrap();
    
    std::fs::write(ip_dir.join("src/main.stark"), r#"
        use utilities::get_pi;
        pub fn process_image() -> Float64 {
            get_pi()
        }
    "#).unwrap();
    
    std::fs::write(app_dir.join("src/main.stark"), r#"
        use image_pipeline::process_image;
        use utilities::get_pi;
        
        fn main() {
            let _p = process_image();
            let _val = get_pi();
        }
    "#).unwrap();
    
    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
    
    let options = LanguageOptions::CORE;
    let (ast, mut diags) = parse_package_graph(&graph, options);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);
    
    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(app_dir.join("src/main.stark").to_string_lossy().into_owned(), entry_src));
    
    let (hir, mut resolution) = resolve(&ast, root_file.clone());
    diags.append(&mut resolution);
    assert!(diags.is_empty(), "resolution failed: {:?}", diags);
    
    let mut tc_diags = typecheck::check(&hir, root_file);
    diags.append(&mut tc_diags);
    assert!(diags.is_empty(), "typecheck failed: {:?}", diags);
    
    let _ = std::fs::remove_dir_all(&workspace);
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
    std::fs::write(app_dir.join("starkpkg.json"), r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/nonexistent.stark"
    }"#).unwrap();
    
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
    
    std::fs::write(a_dir.join("starkpkg.json"), r#"{
        "name": "a",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": { "b": { "path": "../b" } }
    }"#).unwrap();
    
    std::fs::write(b_dir.join("starkpkg.json"), r#"{
        "name": "b",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": { "a": { "path": "../a" } }
    }"#).unwrap();
    
    let result = PackageGraph::load_from_root(&a_dir.join("starkpkg.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("dependency cycle detected: a -> b -> a"));
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
    
    std::fs::write(app_dir.join("starkpkg.json"), r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "d1": { "path": "../d1" },
            "d2": { "path": "../d2" }
        }
    }"#).unwrap();
    
    std::fs::write(d1_dir.join("starkpkg.json"), r#"{
        "name": "d1",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "dup": { "path": "../dup1" }
        }
    }"#).unwrap();
    
    std::fs::write(d2_dir.join("starkpkg.json"), r#"{
        "name": "d2",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "dup": { "path": "../dup2" }
        }
    }"#).unwrap();
    
    std::fs::write(dup1_dir.join("starkpkg.json"), r#"{
        "name": "dup",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#).unwrap();
    
    std::fs::write(dup2_dir.join("starkpkg.json"), r#"{
        "name": "dup",
        "version": "0.1.0",
        "entry": "src/main.stark"
    }"#).unwrap();
    
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
    
    std::fs::write(app_dir.join("starkpkg.json"), r#"{
        "name": "app",
        "version": "0.1.0",
        "entry": "src/main.stark",
        "dependencies": {
            "outside": { "path": "../../" }
        }
    }"#).unwrap();
    
    let result = PackageGraph::load_from_root(&app_dir.join("starkpkg.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("outside the permitted workspace"));
    
    let _ = std::fs::remove_dir_all(&workspace);
}
