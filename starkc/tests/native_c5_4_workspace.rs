//! WP-C5.4d — the frozen three-package reference workspace (§12/§14.4).
//!
//! **`EXPECTED-SYMBOLS.txt` was RE-PINNED on 2026-07-27 (CD-164, DEV-114): every `logic::model::*`
//! symbol became `model::*`.** TYPE-NOMINAL-001 defines identity as "canonical package instance +
//! module path + item name", so a dependency EDGE is not a module-path segment — reaching `model`
//! through `logic` must not rename its items, which is also what PKG-IDENTITY-001 means by "aliases
//! and re-exports preserve it". The previous nesting additionally made the prefix depend on which
//! path reached a package FIRST, and that walk followed a per-process-seeded `HashMap`, so a diamond
//! graph produced different symbols run to run. The fixture stays pure data (no comment lines)
//! because this test reads every line as a symbol.
//!
//! One verified multi-package `MirProgram` (`app` → `logic` → `model`) is driven through all three
//! engines and must agree, then built into ONE standalone native executable that exits normally
//! because every in-program `assert`/`assert_eq` held. The fixture is checked in under
//! `tests/fixtures/c5-native-workspace/`; its canonical `Instance.symbol` set is frozen here, and
//! the freeze is relocation- and traversal-order-independent (§11.4/§13.6).

mod support;

use starkc::backend::generated_rust::{emit_native_debug, linkage, NativeBuildOptions};
use starkc::diag::Severity;
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Front {
    program: MirProgram,
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/c5-native-workspace")
}

/// A private copy of the fixture workspace. `compile_workspace` writes a lockfile into the
/// package root it's given (§7 manifest resolution); tests must never point that at the
/// checked-in fixture directly; `cargo test` runs tests in this file concurrently, and
/// concurrent writers to the same `stark.lock` race (fatal on Windows: "os error 32").
fn isolated_fixture_root() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "stark_c5_4d_isolated_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(&fixture_root(), &dir);
    dir
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
    let program = lower_program(
        &hir,
        &checked.tables,
        // AS1b-ii: the package entry is registered logically, not by its checkout path — which is
        // the relocation invariant this file exists to measure.
        hir.source_named(&graph.packages[&graph.root_package_name].entry_logical_name())
            .expect("the parse registered the package entry"),
    )
    .unwrap_or_else(|e| panic!("workspace must lower: {}", e.what));
    Front { program }
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
    let root = isolated_fixture_root();
    let front = compile_workspace(&root);
    let syms = symbols(&front.program); // already in canonical sorted order
    let frozen = std::fs::read_to_string(fixture_root().join("EXPECTED-SYMBOLS.txt")).unwrap();
    let expected: Vec<String> = frozen.lines().map(|l| l.to_string()).collect();
    assert_eq!(
        syms, expected,
        "canonical symbol set drifted from EXPECTED-SYMBOLS.txt"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// Migrated to the shared comparator (R-02). What it replaced ran two engines and compared
/// `status` and `output` — the only two fields it had. It is now the full §39 observation, and it
/// runs the **third** engine as well: the comment "Engine 1 / Engine 2" was itself the finding, in
/// a file named `native_*`.
///
/// It also drops this suite's own front end, which built the root `SourceFile` from the ABSOLUTE
/// checkout path. DEV-113 made that a provenance defect (PKG-IDENTITY-001: never an absolute
/// checkout path), and `relocation_does_not_change_canonical_symbols` below is the test that would
/// have to catch it — so going through `front_end_package` is a correctness change, not a tidy-up.
#[test]
fn the_workspace_completes_identically_in_every_available_engine() {
    let root = isolated_fixture_root();
    let (front, program) = support::differential::front_end_package(&root.join("app"));

    let name = "c5_4_workspace";
    let hir = support::differential::run_hir(name, &front);
    let mir = support::differential::run_mir(name, &program);
    if support::differential::rustc_available() {
        let native = support::differential::run_native(name, "c5_4_ws", &program);
        if let Err(disagreement) =
            support::differential::compare_observations(name, &hir, &mir, &native)
        {
            panic!("{disagreement}");
        }
    } else if let Some(field) = support::differential::first_difference(&hir, &mir) {
        panic!("{name}: HIR/MIR DISAGREEMENT on {field}\n{hir:#?}\n{mir:#?}");
    }
    match &hir {
        support::differential::Observation::Completed(done) => {
            assert_eq!(done.exit_status, 0, "the workspace must exit 0");
            assert!(done.stdout_bytes.is_empty(), "C5 has no stdout surface");
        }
        other => panic!("the workspace must complete, got {other:#?}"),
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_linked_body_set_is_complete_and_consistent() {
    let root = isolated_fixture_root();
    let front = compile_workspace(&root);
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
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_workspace_builds_one_native_executable_that_exits_normally() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = isolated_fixture_root();
    let front = compile_workspace(&root);
    let verified = verify_program(&front.program).expect("verify");
    let out = std::env::temp_dir().join(format!("stark_c5_4d_ws_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: out.clone(),
            target_contract: "stark-64-v1".to_string(),
            ..NativeBuildOptions::default()
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
    let _ = std::fs::remove_dir_all(&root);
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

    // Through the shared comparator (R-02). The three engines must not merely each fail — they must
    // fail the SAME way. "HIR returned Err, MIR returned Err, native exited non-zero" was satisfiable
    // by three different failures, including a native build error, which is not a trap at all.
    let (front, program) = support::differential::front_end_package(&broken.join("app"));
    let name = "c5_4_workspace_broken";
    let hir = support::differential::run_hir(name, &front);
    let mir = support::differential::run_mir(name, &program);
    if support::differential::rustc_available() {
        let native = support::differential::run_native(name, "c5_4_ws_broken", &program);
        if let Err(disagreement) =
            support::differential::compare_observations(name, &hir, &mir, &native)
        {
            panic!("{disagreement}");
        }
    } else if let Some(field) = support::differential::first_difference(&hir, &mir) {
        panic!("{name}: HIR/MIR DISAGREEMENT on {field}\n{hir:#?}\n{mir:#?}");
    }
    match &hir {
        support::differential::Observation::Trapped(trap) => {
            assert_eq!(
                trap.category,
                starkc::mir::TrapCategory::AssertFailure,
                "the false assertion must raise assert-failure, not some other trap"
            );
        }
        other => panic!("the false assertion must trap, got {other:#?}"),
    }
    let _ = std::fs::remove_dir_all(&broken);
}
