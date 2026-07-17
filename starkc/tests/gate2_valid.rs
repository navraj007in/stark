use starkc::diag::{Diagnostic, Severity};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::sync::Arc;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/gate2-valid")
}

fn analyze(name: &str, source: String) -> Vec<Diagnostic> {
    let file = Arc::new(SourceFile::new(name.to_string(), source));
    let (ast, mut diagnostics) = parse(&file, ParseMode::Program);
    if diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return diagnostics;
    }
    let (hir, mut resolution) = resolve(&ast, file.clone());
    diagnostics.append(&mut resolution);
    if !diagnostics.iter().any(|d| d.severity == Severity::Error) {
        diagnostics.append(&mut typecheck::check(&hir, file));
    }
    diagnostics
}

#[test]
fn all_gate2_valid_programs_pass_semantic_analysis() {
    let mut paths: Vec<_> = std::fs::read_dir(fixture_dir())
        .expect("gate2 valid fixture directory exists")
        .map(|entry| entry.expect("fixture entry is readable").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "stark")
        })
        .collect();
    paths.sort();
    assert!(
        paths.len() >= 20,
        "Gate 2 requires at least 20 valid programs"
    );

    let mut failures = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let source = std::fs::read_to_string(&path).expect("fixture is readable");
        let diagnostics = analyze(&name, source);
        let errors: Vec<_> = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == Severity::Error)
            .map(|diagnostic| {
                format!(
                    "{}: {}",
                    diagnostic.code.as_deref().unwrap_or("uncoded"),
                    diagnostic.message
                )
            })
            .collect();
        if !errors.is_empty() {
            failures.push(format!("{name}: {}", errors.join(", ")));
        }
    }
    assert!(
        failures.is_empty(),
        "valid Gate 2 programs failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn test_multi_file_module_loading() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_multi_file_test");
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(&base_dir);
    }
    std::fs::create_dir_all(&base_dir).unwrap();
    let extra_dir = base_dir.join("extra");
    std::fs::create_dir_all(&extra_dir).unwrap();

    let main_src =
        "mod math;\nmod extra;\nfn main() { let _x = math::add(1, 2); let _y = extra::sub(3, 1); }";
    let math_src = "pub fn add(a: Int32, b: Int32) -> Int32 { a + b }";
    let extra_src = "pub fn sub(a: Int32, b: Int32) -> Int32 { a - b }";

    let main_path = base_dir.join("main.stark");
    let math_path = base_dir.join("math.stark");
    let extra_mod_path = extra_dir.join("mod.stark");

    std::fs::write(&main_path, main_src).unwrap();
    std::fs::write(&math_path, math_src).unwrap();
    std::fs::write(&extra_mod_path, extra_src).unwrap();

    let file = Arc::new(SourceFile::new(
        main_path.to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (ast, mut diags) = parse(&file, ParseMode::Program);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);

    let (hir, mut resolution) = resolve(&ast, file.clone());
    diags.append(&mut resolution);
    assert!(diags.is_empty(), "resolution failed: {:?}", diags);

    let mut tc_diags = typecheck::check(&hir, file.clone());
    diags.append(&mut tc_diags);
    assert!(diags.is_empty(), "typecheck failed: {:?}", diags);

    // Write conflict file
    let extra_file_path = base_dir.join("extra.stark");
    std::fs::write(&extra_file_path, "pub fn conflict() {}").unwrap();

    let (_ast_conflict, diags_conflict) = parse(&file, ParseMode::Program);
    // E0202 conflict error should be reported
    assert!(
        diags_conflict
            .iter()
            .any(|d| d.code.as_deref() == Some("E0202")),
        "expected E0202 conflict error, got {:?}",
        diags_conflict
    );

    // Clean up
    let _ = std::fs::remove_dir_all(&base_dir);
}

/// WP-C1.1: missing-module-file case for the `mod foo;` layout rule (checklist item 8).
/// Regression test for DEV-014 -- `load_submodules_recursive` used to suppress this diagnostic
/// for any process whose argv happened to contain "test" or "conformance" (which included every
/// real `stark test` invocation), so a genuinely missing module file was silently accepted as
/// having zero items rather than reported. See COMPILER-STATE.md DEV-014 and
/// starkc/docs/conformance/KNOWN-DEVIATIONS.md.
#[test]
fn test_missing_module_file_is_reported_not_silently_accepted() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_missing_module_file");
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(&base_dir);
    }
    std::fs::create_dir_all(&base_dir).unwrap();

    let main_src = "mod does_not_exist;\nfn main() {}";
    let main_path = base_dir.join("main.stark");
    std::fs::write(&main_path, main_src).unwrap();

    let file = Arc::new(SourceFile::new(
        main_path.to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (_ast, diags) = parse(&file, ParseMode::Program);
    assert!(
        diags.iter().any(|d| d.code.as_deref() == Some("E0202")),
        "expected E0202 'file not found for module', got {:?}",
        diags
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}

/// WP-C1.1: duplicate `mod foo;` declarations in the same file (checklist item 8). Two `Mod`
/// items with the same name are two separate item definitions in the same scope, so this is
/// correctly caught by the resolver's ordinary duplicate-definition check (E0204) at parse+
/// resolve time -- it never reaches the parser's `loaded_modules` path-based dedup at all, since
/// that dedup only fires once a *single* `mod` item is actually being loaded. Confirmed by this
/// test; the initial version of this test incorrectly assumed the file would load silently and
/// was corrected after running it against the real implementation.
#[test]
fn test_duplicate_mod_declaration_is_rejected() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_duplicate_mod");
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(&base_dir);
    }
    std::fs::create_dir_all(&base_dir).unwrap();

    let main_src = "mod math;\nmod math;\nfn main() { let _x = math::add(1, 2); }";
    let math_src = "pub fn add(a: Int32, b: Int32) -> Int32 { a + b }";
    std::fs::write(base_dir.join("main.stark"), main_src).unwrap();
    std::fs::write(base_dir.join("math.stark"), math_src).unwrap();

    let file = Arc::new(SourceFile::new(
        base_dir.join("main.stark").to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (ast, diags) = parse(&file, ParseMode::Program);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);

    let (_hir, resolution) = resolve(&ast, file.clone());
    assert!(
        resolution
            .iter()
            .any(|d| d.code.as_deref() == Some("E0204")),
        "expected E0204 duplicate definition for the repeated 'mod math;', got {:?}",
        resolution
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}

/// WP-C1.1: a circular `mod` reference (checklist item 8) -- `a.stark` declares `mod b;` and
/// `b.stark` declares `mod a;` pointing back to the entry file. Confirms `loaded_modules`
/// (a `HashSet<String>` keyed by resolved path) prevents infinite recursion rather than only
/// being incidentally safe.
#[test]
fn test_circular_mod_reference_does_not_hang() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_circular_mod");
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(&base_dir);
    }
    std::fs::create_dir_all(&base_dir).unwrap();

    let a_src = "mod b;\nfn main() {}";
    let b_src = "mod a;\npub fn helper() {}";
    std::fs::write(base_dir.join("a.stark"), a_src).unwrap();
    std::fs::write(base_dir.join("b.stark"), b_src).unwrap();

    let file = Arc::new(SourceFile::new(
        base_dir.join("a.stark").to_string_lossy().into_owned(),
        a_src.to_string(),
    ));
    // The assertion here is termination itself -- a regression that reintroduced unbounded
    // recursion would hang or stack-overflow this test process, not fail an assert.
    let (_ast, _diags) = parse(&file, ParseMode::Program);

    let _ = std::fs::remove_dir_all(&base_dir);
}
