//! WP-C5.4a — the linkage preflight and deterministic-symbol proof.
//!
//! Two families:
//!   * hand-built `MirProgram`s driven straight through `linkage::build`, isolating each §6.2
//!     validation and the §6.3 shared operand walker without needing a source program that hits
//!     exactly that shape;
//!   * source-driven cross-package programs, proving canonical symbols survive package boundaries
//!     and workspace relocation (§11.3/§11.4/§13.6) and that a cross-package program links and
//!     runs natively end to end.
//!
//! The backend is not a linker/resolver (§2.2): these tests assert it VALIDATES the body set it is
//! given and refuses BEFORE rustc, never that it repairs or discovers anything.

use starkc::backend::generated_rust::linkage::{self, RefKind};
use starkc::backend::generated_rust::{
    emit_native_debug, mangle, BackendDiagnostic, NativeBuildOptions,
};
use starkc::diag::Severity;
use starkc::hir::ItemId;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::{
    BasicBlock, BlockId, Callee, Constant, FileId, Instance, LocalId, MirBody, MirProgram, MirTy,
    Operand, Origin, Place, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
};
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::{SourceFile, Span};
use starkc::typecheck;
use std::path::Path;
use std::sync::Arc;

// ------------------------------------------------------------- hand-built MIR --

fn info() -> SourceInfo {
    SourceInfo {
        file: FileId(0),
        span: Span::new(0, 0),
        origin: Origin::UserCode,
    }
}

fn program(bodies: Vec<MirBody>) -> MirProgram {
    MirProgram {
        files: Vec::new(),
        bodies,
        types: TypeContext::default(),
        mir_version: "test".to_string(),
        runtime_surface: "test".to_string(),
    }
}

/// A trivial body: no params, `Unit` return, one block that returns.
fn leaf_body(symbol: &str, item: u32) -> MirBody {
    MirBody {
        instance: Instance {
            item: ItemId(item),
            type_args: Vec::new(),
            symbol: symbol.to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals: Vec::new(),
        blocks: vec![BasicBlock {
            statements: Vec::new(),
            terminator: (Terminator::Return, info()),
        }],
        entry: BlockId(0),
    }
}

/// A body that directly CALLS `callee` (block 0), then returns (block 1).
fn calling_body(symbol: &str, item: u32, callee: Instance) -> MirBody {
    let mut body = leaf_body(symbol, item);
    body.blocks = vec![
        BasicBlock {
            statements: Vec::new(),
            terminator: (
                Terminator::Call {
                    callee: Callee::Instance(callee),
                    args: Vec::new(),
                    dest: Place::local(LocalId(0)),
                    target: BlockId(1),
                },
                info(),
            ),
        },
        BasicBlock {
            statements: Vec::new(),
            terminator: (Terminator::Return, info()),
        },
    ];
    body
}

/// A body that takes `target` only as a FUNCTION VALUE (`Constant::FnPtr`), never calling it
/// directly. This is the §10.5 reachability shape in miniature.
fn fnvalue_body(symbol: &str, item: u32, target: Instance) -> MirBody {
    let mut body = leaf_body(symbol, item);
    body.blocks = vec![BasicBlock {
        statements: vec![(
            Statement::Assign(
                Place::local(LocalId(0)),
                Rvalue::Use(Operand::Const(Constant::FnPtr(target))),
            ),
            info(),
        )],
        terminator: (Terminator::Return, info()),
    }];
    body
}

fn inst(symbol: &str, item: u32) -> Instance {
    Instance {
        item: ItemId(item),
        type_args: Vec::new(),
        symbol: symbol.to_string(),
    }
}

fn unsupported_message(result: Result<linkage::LinkageIndex<'_>, BackendDiagnostic>) -> String {
    match result {
        Ok(_) => panic!("expected linkage refusal, got a valid index"),
        Err(BackendDiagnostic::Unsupported(msg)) => msg,
        Err(other) => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn a_well_formed_body_set_builds_an_index_with_every_body() {
    let prog = program(vec![leaf_body("main@[]", 0), leaf_body("z_foo@[]", 1)]);
    let index = linkage::build(&prog).expect("well-formed program must link");
    assert_eq!(index.by_symbol.len(), 2);
    assert_eq!(
        index.by_symbol["main@[]"].rust_name, "main",
        "the entry maps to Rust `main`"
    );
    assert_eq!(
        index.by_symbol["z_foo@[]"].rust_name,
        mangle::function_name_for_symbol("z_foo@[]"),
    );
}

#[test]
fn unsorted_bodies_are_refused_not_re_sorted() {
    // `z_foo@[]` before `main@[]` is out of canonical order. §6.4: the backend refuses rather than
    // silently normalising a producer defect.
    let prog = program(vec![leaf_body("z_foo@[]", 1), leaf_body("main@[]", 0)]);
    let msg = unsupported_message(linkage::build(&prog));
    assert!(msg.contains("not strictly sorted"), "{msg}");
}

#[test]
fn a_duplicate_canonical_symbol_is_refused() {
    // Two bodies with one symbol. (Adjacent duplicates trip the strict-order guard first, which is
    // itself a refusal; the point of the test is that a duplicate never links.)
    let prog = program(vec![leaf_body("main@[]", 0), leaf_body("z_dup@[]", 1), {
        let mut b = leaf_body("z_dup@[]", 2);
        b.instance.symbol = "z_dup@[]".to_string();
        b
    }]);
    let _ = unsupported_message(linkage::build(&prog));
}

#[test]
fn a_direct_call_naming_no_body_is_refused() {
    // `main` calls `z_missing@[]`, which has no body in the program.
    let prog = program(vec![calling_body("main@[]", 0, inst("z_missing@[]", 9))]);
    let msg = unsupported_message(linkage::build(&prog));
    assert!(msg.contains("names no body"), "{msg}");
    assert!(msg.contains("z_missing@[]"), "{msg}");
}

#[test]
fn a_function_constant_naming_no_body_is_refused() {
    // A function VALUE (`Constant::FnPtr`) to a missing body must be refused on the same path as a
    // direct call — that is the guarantee §10.5's reachability case rests on.
    let prog = program(vec![fnvalue_body("main@[]", 0, inst("z_missing@[]", 9))]);
    let msg = unsupported_message(linkage::build(&prog));
    assert!(msg.contains("names no body"), "{msg}");
}

#[test]
fn one_symbol_with_two_identities_is_refused() {
    // `main` references `z_foo@[]` with item 1, but the body for `z_foo@[]` is defined with item 2:
    // one canonical symbol, two identities (§5.3). This is a canonical-identity defect, refused —
    // never patched with a second package-identity scheme (§11.3).
    let prog = program(vec![
        calling_body("main@[]", 0, inst("z_foo@[]", 1)),
        leaf_body("z_foo@[]", 2),
    ]);
    let msg = unsupported_message(linkage::build(&prog));
    assert!(msg.contains("two identities"), "{msg}");
}

#[test]
fn a_program_without_an_entry_body_is_refused() {
    let prog = program(vec![leaf_body("z_foo@[]", 1)]);
    let msg = unsupported_message(linkage::build(&prog));
    assert!(msg.contains("entry body"), "{msg}");
}

#[test]
fn a_function_referenced_only_as_a_value_resolves_through_the_same_path() {
    // The §10.5 guard, at the linkage layer: `z_target` is reached ONLY through `Constant::FnPtr`,
    // never a direct call, yet it must resolve. A direct-call-only reachability assumption would
    // drop it.
    let prog = program(vec![
        fnvalue_body("main@[]", 0, inst("z_target@[]", 1)),
        leaf_body("z_target@[]", 1),
    ]);
    let index = linkage::build(&prog).expect("function-value-only reference must resolve");
    assert!(index.by_symbol.contains_key("z_target@[]"));
}

#[test]
fn the_shared_walker_sees_both_direct_and_function_value_references() {
    // §6.3: the ONE walker collects direct callees and function constants alike. A body that does
    // both must yield exactly one of each.
    let mut body = leaf_body("main@[]", 0);
    body.blocks = vec![
        BasicBlock {
            statements: vec![(
                Statement::Assign(
                    Place::local(LocalId(0)),
                    Rvalue::Use(Operand::Const(Constant::FnPtr(inst("z_val@[]", 1)))),
                ),
                info(),
            )],
            terminator: (
                Terminator::Call {
                    callee: Callee::Instance(inst("z_call@[]", 2)),
                    args: Vec::new(),
                    dest: Place::local(LocalId(0)),
                    target: BlockId(1),
                },
                info(),
            ),
        },
        BasicBlock {
            statements: Vec::new(),
            terminator: (Terminator::Return, info()),
        },
    ];

    let mut direct = Vec::new();
    let mut fnval = Vec::new();
    linkage::visit_instance_refs(&body, &mut |i, kind| match kind {
        RefKind::DirectCall => direct.push(i.symbol.clone()),
        RefKind::FnValue => fnval.push(i.symbol.clone()),
    });
    assert_eq!(direct, vec!["z_call@[]".to_string()]);
    assert_eq!(fnval, vec!["z_val@[]".to_string()]);
}

// ---------------------------------------------------------- cross-package MIR --

/// Write a two-package workspace (`lib` + `app`) whose `app` makes a cross-package call into
/// `lib`, and return the app entry manifest's discovered root. `app_main`/`lib_main` are STARK
/// source; the observation channel is `assert`/`assert_eq` (native has no stdout in C5).
fn write_workspace(root: &Path, lib_main: &str, app_main: &str) {
    let app_dir = root.join("app");
    let lib_dir = root.join("lib");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(lib_dir.join("src")).unwrap();
    std::fs::write(
        lib_dir.join("starkpkg.json"),
        r#"{"name": "lib", "version": "0.1.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(lib_dir.join("src/main.stark"), lib_main).unwrap();
    std::fs::write(
        app_dir.join("starkpkg.json"),
        r#"{
            "name": "app",
            "version": "0.1.0",
            "entry": "src/main.stark",
            "dependencies": { "lib": { "path": "../lib" } }
        }"#,
    )
    .unwrap();
    std::fs::write(app_dir.join("src/main.stark"), app_main).unwrap();
}

/// Full front end → MIR for a workspace rooted at `root/app`. Panics with a stage-tagged message
/// on any diagnostic, so a broken fixture fails loudly rather than silently skipping a dimension.
fn lower_workspace(root: &Path) -> MirProgram {
    let app_dir = root.join("app");
    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
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
    lower_program(&hir, &checked.tables, root_file)
        .unwrap_or_else(|e| panic!("workspace must lower to MIR: {}", e.what))
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("stark_c5_4a_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

const LIB: &str = "pub fn double(n: Int32) -> Int32 { n * 2 }";
const APP: &str = "use lib::double;\nfn main() { assert_eq(double(21), 42); }";

#[test]
fn a_cross_package_program_links_and_runs_natively() {
    let root = temp("crosspkg_run");
    write_workspace(&root, LIB, APP);
    let prog = lower_workspace(&root);
    let verified = verify_program(&prog).expect("must verify");

    // The linkage index accepts it, and the cross-package callee is a distinct package-qualified
    // symbol — not `main`, and present exactly once.
    let index = linkage::build(&prog).expect("cross-package program must link");
    let double = prog
        .bodies
        .iter()
        .find(|b| b.instance.symbol.contains("double"))
        .expect("the lib::double instance must be in the body set");
    assert!(
        double.instance.symbol.contains("lib"),
        "cross-package symbol should be package-qualified: {}",
        double.instance.symbol
    );
    assert_eq!(index.by_symbol.len(), prog.bodies.len());

    let target_dir = root.join("out");
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir,
            target_contract: "stark-64-v1".to_string(),
        },
    )
    .expect("cross-package program must emit and build");

    // The lib function is emitted exactly once, under its package-qualified name.
    let generated = std::fs::read_to_string(artifact.build_dir.join("src/main.rs")).unwrap();
    let name = mangle::function_name_for_symbol(&double.instance.symbol);
    assert_eq!(
        generated.matches(&format!("fn {name}(")).count(),
        1,
        "the cross-package function must be defined exactly once"
    );

    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    assert!(
        run.status.success(),
        "cross-package native run must exit 0 (asserts held); stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn workspace_relocation_does_not_change_canonical_symbols() {
    // §11.4/§13.6: relocating the entire workspace to a different absolute path must not change any
    // canonical `Instance.symbol`. The backend treats the symbol as authoritative and never mixes
    // in file-system paths (§11.3).
    let a = temp("reloc_a");
    write_workspace(&a, LIB, APP);
    let symbols_a: Vec<String> = lower_workspace(&a)
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect();

    let b = temp("reloc_b_different_absolute_path");
    write_workspace(&b, LIB, APP);
    let symbols_b: Vec<String> = lower_workspace(&b)
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect();

    assert_eq!(
        symbols_a, symbols_b,
        "canonical symbols must not depend on the workspace's absolute location"
    );
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}

#[test]
fn two_clean_lowerings_of_one_workspace_agree_on_symbols() {
    // §13.6: determinism across two clean builds of the same semantic workspace.
    let root = temp("determinism");
    write_workspace(&root, LIB, APP);
    let first: Vec<String> = lower_workspace(&root)
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect();
    let second: Vec<String> = lower_workspace(&root)
        .bodies
        .iter()
        .map(|b| b.instance.symbol.clone())
        .collect();
    assert_eq!(first, second);
    // Strictly sorted, so the list is its own canonical order.
    let mut sorted = first.clone();
    sorted.sort();
    assert_eq!(
        first, sorted,
        "bodies must already be in canonical symbol order"
    );
    let _ = std::fs::remove_dir_all(&root);
}
