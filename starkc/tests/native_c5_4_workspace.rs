//! WP-C5.4d — the frozen three-package reference workspace (§12/§14.4).
//!
//! One verified multi-package `MirProgram` (`app` → `logic` → `model`) is driven through all three
//! engines and must agree, then built into ONE standalone native executable that exits normally
//! because every in-program `assert`/`assert_eq` held. The fixture is checked in under
//! `tests/fixtures/c5-native-workspace/`; its canonical `Instance.symbol` set is frozen here, and
//! the freeze is relocation- and traversal-order-independent (§11.4/§13.6).

use starkc::backend::generated_rust::{emit_native_debug, linkage, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::MirProgram;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Front {
    hir: starkc::hir::Hir,
    tables: starkc::typecheck::TypeTables,
    root_file: Arc<SourceFile>,
    program: MirProgram,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/c5-native-workspace")
}

/// Front end → verified-ready MIR for the workspace rooted at `root/app`.
fn compile_workspace(root: &Path) -> Front {
    let app_dir = root.join("app");
    let manifest = find_package_root(&app_dir).expect("find app manifest");
    let graph = PackageGraph::load_from_root(&manifest).expect("load package graph");
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));
    let (hir, resolve_diags) = resolve(&ast, root_file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, root_file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck: {errors:?}");
    let program = lower_program(&hir, &checked.tables, root_file.clone())
        .unwrap_or_else(|e| panic!("workspace must lower: {}", e.what));
    Front {
        hir,
        tables: checked.tables,
        root_file,
        program,
    }
}

fn symbols(program: &MirProgram) -> Vec<String> {
    program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect()
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &to);
        } else {
            std::fs::copy(entry.path(), &to).unwrap();
        }
    }
}

#[test]
fn the_canonical_symbols_match_the_frozen_list() {
    // §12.2 step 6: the frozen canonical `Instance.symbol` set. Deterministic and
    // relocation-independent (no absolute paths appear in a symbol), so a byte comparison against
    // the checked-in list is a real freeze — it catches a dropped body, an extra body, a renamed
    // instance, or a changed monomorphisation, in one assertion.
    let front = compile_workspace(&fixture_root());
    let syms = symbols(&front.program); // already in canonical sorted order
    let frozen = std::fs::read_to_string(fixture_root().join("EXPECTED-SYMBOLS.txt")).unwrap();
    let expected: Vec<String> = frozen.lines().map(|l| l.to_string()).collect();
    assert_eq!(
        syms, expected,
        "canonical symbol set drifted from EXPECTED-SYMBOLS.txt"
    );
}

#[test]
fn hir_and_mir_agree_and_the_workspace_completes() {
    let front = compile_workspace(&fixture_root());

    // Engine 1: HIR oracle.
    let hir_exec =
        interp::run_with_partial_output(&front.hir, front.root_file.clone(), &front.tables)
            .unwrap_or_else(|(e, _)| panic!("HIR run failed: {}", e.message));
    assert_eq!(hir_exec.status, 0, "HIR must exit 0");
    assert!(hir_exec.output.is_empty(), "C5 has no stdout surface");

    // Engine 2: MIR interpreter (implies verification).
    let verified = verify_program(&front.program).expect("MIR must verify");
    let mir_exec =
        run_program(verified).unwrap_or_else(|f| panic!("MIR run failed: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "MIR must exit 0");

    // Agreement (both completed, same exit, no output).
    assert_eq!(hir_exec.status, mir_exec.status);
    assert_eq!(hir_exec.output, mir_exec.output);
}

#[test]
fn the_linked_body_set_is_complete_and_consistent() {
    let front = compile_workspace(&fixture_root());
    // Reuses the C5.4a preflight: every referenced instance resolves to exactly one body, symbols
    // strictly sorted and unique, generated names unique — i.e. no duplicate concrete body and no
    // missing referenced body (§14.4 exit).
    let index = linkage::build(&front.program).expect("workspace must link");
    assert_eq!(index.by_symbol.len(), front.program.bodies.len());

    // The workspace exercises the shapes §12.3 requires; spot-check the instances are present.
    let syms = symbols(&front.program);
    let has = |needle: &str| syms.iter().any(|s| s.contains(needle));
    assert!(has("triple"), "cross-package function value target");
    assert!(has("only_via_value"), "value-only reachability target");
    assert!(has("apply"), "higher-order function");
    assert!(has("get_triple"), "function-value-returning function");
    // Two concrete instantiations of `wrap` (app's cross-package generic call).
    let wraps = syms.iter().filter(|s| s.contains("::wrap@[")).count();
    assert_eq!(wraps, 2, "wrap instantiated at two types: {syms:?}");
    // Two concrete instantiations of model's `transform` (the `::` anchor excludes
    // `double_transform`).
    let transforms = syms.iter().filter(|s| s.contains("::transform@[")).count();
    assert_eq!(
        transforms, 2,
        "transform instantiated at two types: {syms:?}"
    );
}

#[test]
fn the_workspace_builds_one_native_executable_that_exits_normally() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let front = compile_workspace(&fixture_root());
    let verified = verify_program(&front.program).expect("verify");
    let out = std::env::temp_dir().join(format!("stark_c5_4d_ws_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: out.clone(),
            target_contract: "stark-64-v1".to_string(),
        },
    )
    .expect("workspace must build one native executable");
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run failed");
    assert!(
        run.status.success(),
        "the standalone executable must exit 0 (all asserts held); stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn relocation_does_not_change_canonical_symbols() {
    // §11.4/§13.6: the same workspace at a DIFFERENT absolute path yields byte-identical canonical
    // symbols — the backend treats `Instance.symbol` as authoritative and never mixes in paths.
    let reloc = std::env::temp_dir().join(format!("stark_c5_4d_reloc_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&reloc);
    copy_dir(&fixture_root(), &reloc);
    let moved = symbols(&compile_workspace(&reloc).program);
    let frozen: Vec<String> = std::fs::read_to_string(fixture_root().join("EXPECTED-SYMBOLS.txt"))
        .unwrap()
        .lines()
        .map(|l| l.to_string())
        .collect();
    assert_eq!(
        moved, frozen,
        "relocating the workspace must not change canonical symbols"
    );
    let _ = std::fs::remove_dir_all(&reloc);
}

#[test]
fn a_broken_assertion_traps_in_all_three_engines() {
    // §13.4 negative control: the workspace's assertions actually execute. Flip one expected value
    // and every engine must fail — otherwise "all three exit 0" would be satisfiable by a backend
    // that compiled assertions away.
    let broken = std::env::temp_dir().join(format!("stark_c5_4d_broken_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&broken);
    copy_dir(&fixture_root(), &broken);
    let app_main = broken.join("app/src/main.stark");
    let src = std::fs::read_to_string(&app_main).unwrap();
    // `f(4)` is 12; assert it equals 13 instead.
    let mutated = src.replacen("assert_eq(f(4), 12);", "assert_eq(f(4), 13);", 1);
    assert_ne!(src, mutated, "the mutation must apply");
    std::fs::write(&app_main, mutated).unwrap();

    let front = compile_workspace(&broken);

    // HIR traps.
    let hir = interp::run_with_partial_output(&front.hir, front.root_file.clone(), &front.tables);
    assert!(hir.is_err(), "HIR must trap on the false assertion");

    // MIR traps.
    let verified = verify_program(&front.program).expect("verify");
    assert!(
        run_program(verified).is_err(),
        "MIR must trap on the false assertion"
    );

    // Native traps (non-zero exit), if rustc is available.
    if rustc_available() {
        let verified = verify_program(&front.program).expect("verify");
        let out = broken.join("out");
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: out,
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .expect("broken workspace still builds");
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run failed");
        assert!(
            !run.status.success(),
            "native must trap (non-zero exit) on the false assertion"
        );
    }
    let _ = std::fs::remove_dir_all(&broken);
}
