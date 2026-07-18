use starkc::diag::Severity;
use starkc::options::LanguageOptions;
use starkc::package::PackageGraph;
use starkc::parser::{parse, parse_package_graph, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

fn workspace(name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(format!(
            "temp_source_extension_{name}_{}",
            std::process::id()
        ));
    if path.exists() {
        let _ = std::fs::remove_dir_all(&path);
    }
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn parse_project(path: &Path) -> Vec<starkc::diag::Diagnostic> {
    let source = std::fs::read_to_string(path).unwrap();
    let file = Arc::new(SourceFile::new(path.to_string_lossy().into_owned(), source));
    let (ast, mut diagnostics) = parse(&file, ParseMode::Program);
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == Severity::Error)
    {
        return diagnostics;
    }
    let (hir, mut resolution) = resolve(&ast, file.clone());
    diagnostics.append(&mut resolution);
    if !diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == Severity::Error)
    {
        diagnostics.append(&mut typecheck::check(&hir, file));
    }
    diagnostics
}

#[test]
fn compiler_cli_accepts_st_root_file_for_all_frontend_commands() {
    let directory = workspace("cli");
    let source = directory.join("main.st");
    std::fs::write(&source, "fn main() { println(\"short extension\"); }").unwrap();

    for command in ["lex", "parse", "check"] {
        let output = Command::new(env!("CARGO_BIN_EXE_starkc"))
            .args([command, source.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{command} rejected .st input: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let output = Command::new(env!("CARGO_BIN_EXE_starkc"))
        .args(["run", source.to_str().unwrap()])
        .output()
        .unwrap();

    assert!(output.status.success(), "{:?}", output.stderr);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "short extension\n"
    );
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn st_root_loads_flat_and_directory_st_modules() {
    let directory = workspace("modules");
    let nested = directory.join("nested");
    std::fs::create_dir_all(&nested).unwrap();
    let root = directory.join("main.st");
    std::fs::write(
        &root,
        "mod flat;\nmod nested;\nfn main() { let _ = flat::one() + nested::two(); }",
    )
    .unwrap();
    std::fs::write(directory.join("flat.st"), "pub fn one() -> Int32 { 1 }").unwrap();
    std::fs::write(nested.join("mod.st"), "pub fn two() -> Int32 { 2 }").unwrap();

    let diagnostics = parse_project(&root);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn stark_and_st_files_can_be_mixed_across_module_boundaries() {
    let directory = workspace("mixed");
    let root = directory.join("main.stark");
    std::fs::write(&root, "mod short;\nfn main() { let _ = short::value(); }").unwrap();
    let short_directory = directory.join("short");
    std::fs::create_dir_all(&short_directory).unwrap();
    std::fs::write(
        short_directory.join("mod.st"),
        "mod long;\npub fn value() -> Int32 { long::value() }",
    )
    .unwrap();
    std::fs::write(
        short_directory.join("long.stark"),
        "pub fn value() -> Int32 { 42 }",
    )
    .unwrap();

    let diagnostics = parse_project(&root);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn module_candidates_are_ambiguous_across_extensions() {
    let directory = workspace("ambiguous");
    let root = directory.join("main.st");
    std::fs::write(&root, "mod duplicate;\nfn main() {}").unwrap();
    std::fs::write(directory.join("duplicate.stark"), "pub fn long() {}").unwrap();
    std::fs::write(directory.join("duplicate.st"), "pub fn short() {}").unwrap();

    let diagnostics = parse_project(&root);
    let conflict = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code.as_deref() == Some("E0208"))
        .expect("ambiguous module candidates must report E0208");
    assert!(conflict.message.contains("duplicate.stark"));
    assert!(conflict.message.contains("duplicate.st"));
    let _ = std::fs::remove_dir_all(directory);
}

#[test]
fn package_manifest_accepts_explicit_st_entry() {
    let directory = workspace("package");
    let source_directory = directory.join("src");
    std::fs::create_dir_all(&source_directory).unwrap();
    let manifest = directory.join("starkpkg.json");
    std::fs::write(
        &manifest,
        r#"{
            "name": "short-source",
            "version": "1.0.0",
            "entry": "src/main.st"
        }"#,
    )
    .unwrap();
    std::fs::write(source_directory.join("main.st"), "fn main() {}").unwrap();

    let graph = PackageGraph::load_from_root(&manifest).unwrap();
    let (_ast, diagnostics) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let _ = std::fs::remove_dir_all(directory);
}
