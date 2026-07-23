//! DEV-101 — cross-package (cross-file) generic instantiation.
//!
//! A generic function/nominal declared in a dependency package is instantiated by a caller in
//! another package. The generic parameter and bound NAMES are declared by the callee, so their
//! spans are only meaningful against the callee's file (`item_text`); reading them against the
//! caller's file (`self.text`) left the parameter unsubstituted and made every cross-package
//! generic use fail. This is a type-checker provenance fix — no resolver/HIR/MIR/linkage/backend
//! change — so these tests drive the front end (and, for a few, the full native pipeline).

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::parse_package_graph;
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// (package name, dependency name or "", source). The first entry is the root `app`.
fn workspace(tag: &str, packages: &[(&str, &str, &str)]) -> PathBuf {
    let root = std::env::temp_dir().join(format!("stark_dev101_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (name, dep, src) in packages {
        let dir = root.join(name);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let manifest = if dep.is_empty() {
            format!(r#"{{ "name": "{name}", "version": "0.1.0", "entry": "src/main.stark" }}"#)
        } else {
            format!(
                r#"{{ "name": "{name}", "version": "0.1.0", "entry": "src/main.stark", "dependencies": {{ "{dep}": {{ "path": "../{dep}" }} }} }}"#
            )
        };
        std::fs::write(dir.join("starkpkg.json"), manifest).unwrap();
        std::fs::write(dir.join("src/main.stark"), src).unwrap();
    }
    root
}

struct Compiled {
    type_errors: Vec<String>,
    program: Option<starkc::mir::MirProgram>,
}

fn compile_app(root: &Path) -> Compiled {
    let app_dir = root.join("app");
    let manifest = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest).unwrap();
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        src,
    ));
    let (hir, resolve_diags) = resolve(&ast, root_file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, root_file.clone());
    let type_errors: Vec<String> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    let program = if type_errors.is_empty() {
        lower_program(&hir, &checked.tables, root_file).ok()
    } else {
        None
    };
    Compiled {
        type_errors,
        program,
    }
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Full native build + run; returns whether the process exited 0. Skips (returns true) with no
/// rustc, so the test still asserts the compile succeeded.
fn native_exits_zero(root: &Path, program: &starkc::mir::MirProgram) -> bool {
    if !rustc_available() {
        return true;
    }
    let verified = verify_program(program).expect("verify");
    let out = root.join("out");
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: out,
            target_contract: "stark-64-v1".to_string(),
        },
    )
    .expect("native build");
    std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run")
        .status
        .success()
}

fn assert_clean_and_runs(tag: &str, packages: &[(&str, &str, &str)]) {
    let root = workspace(tag, packages);
    let c = compile_app(&root);
    assert!(
        c.type_errors.is_empty(),
        "{tag}: type errors: {:?}",
        c.type_errors
    );
    let program = c.program.unwrap_or_else(|| panic!("{tag}: must lower"));
    assert!(
        native_exits_zero(&root, &program),
        "{tag}: native must exit 0"
    );
    let _ = std::fs::remove_dir_all(&root);
}

const ID_MODEL: &str = "pub fn id<T>(x: T) -> T { x }\n";

#[test]
fn explicit_turbofish_cross_package_generic_call() {
    assert_clean_and_runs(
        "turbofish",
        &[
            (
                "app",
                "model",
                "use model::id;\nfn main() { let a: Int32 = 5; assert_eq(id::<Int32>(a), 5); }",
            ),
            ("model", "", ID_MODEL),
        ],
    );
}

#[test]
fn inferred_cross_package_generic_call() {
    assert_clean_and_runs(
        "infer",
        &[
            (
                "app",
                "model",
                "use model::id;\nfn main() { let a: Int32 = 5; assert_eq(id(a), 5); }",
            ),
            ("model", "", ID_MODEL),
        ],
    );
}

#[test]
fn fully_qualified_cross_package_generic_call() {
    assert_clean_and_runs(
        "qualified",
        &[
            (
                "app",
                "model",
                "fn main() { let a: Int32 = 5; assert_eq(model::id::<Int32>(a), 5); }",
            ),
            ("model", "", ID_MODEL),
        ],
    );
}

#[test]
fn generic_function_coerced_to_a_concrete_function_value_cross_package() {
    assert_clean_and_runs(
        "coerce",
        &[
            (
                "app",
                "model",
                "use model::id;\nfn main() { let f: fn(Int32) -> Int32 = id; assert_eq(f(5), 5); }",
            ),
            ("model", "", ID_MODEL),
        ],
    );
}

#[test]
fn cross_package_generic_nominal_substitution() {
    assert_clean_and_runs(
        "nominal",
        &[
            (
                "app",
                "model",
                "use model::mkpair;\nuse model::first;\nfn main() { let p = mkpair(3, 4); assert_eq(first(p), 3); }",
            ),
            (
                "model",
                "",
                "pub struct Pair<T> { a: T, b: T }\n\
                 pub fn mkpair(a: Int32, b: Int32) -> Pair<Int32> { Pair { a: a, b: b } }\n\
                 pub fn first(p: Pair<Int32>) -> Int32 { p.a }\n",
            ),
        ],
    );
}

#[test]
fn cross_package_bounded_generic_satisfied() {
    assert_clean_and_runs(
        "bound_ok",
        &[
            (
                "app",
                "model",
                "use model::maxof;\nfn main() { let a: Int32 = 3; let b: Int32 = 5; assert_eq(maxof::<Int32>(a, b), 5); }",
            ),
            (
                "model",
                "",
                "pub fn maxof<T: Ord>(a: T, b: T) -> T { if a < b { b } else { a } }\n",
            ),
        ],
    );
}

#[test]
fn cross_package_bounded_generic_unsatisfied_is_rejected_with_the_real_bound_name() {
    // Before DEV-101 the bound name was read against the caller's file, producing a garbage bound
    // name (and, worse, wrongly REJECTING a satisfied bound). Now the rejection names `Ord`.
    let root = workspace(
        "bound_bad",
        &[
            (
                "app",
                "model",
                "use model::needs_ord;\nuse model::NoOrd;\nfn main() { let a: NoOrd = NoOrd { v: 1 }; let _ = needs_ord::<NoOrd>(a); }",
            ),
            (
                "model",
                "",
                "pub struct NoOrd { v: Int32 }\npub fn needs_ord<T: Ord>(x: T) -> T { x }\n",
            ),
        ],
    );
    let c = compile_app(&root);
    assert!(
        c.type_errors
            .iter()
            .any(|e| e.contains("does not satisfy trait bound 'Ord'")),
        "expected a clean `Ord` bound rejection, got: {:?}",
        c.type_errors
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn dependency_to_dependency_generic_call() {
    // `logic` (a dependency) calls `model`'s (another dependency) generic; `app` calls `logic`.
    assert_clean_and_runs(
        "dep2dep",
        &[
            (
                "app",
                "logic",
                "use logic::use_id;\nfn main() { let a: Int32 = 9; assert_eq(use_id(a), 9); }",
            ),
            (
                "logic",
                "model",
                "use model::id;\npub fn use_id(x: Int32) -> Int32 { id::<Int32>(x) }\n",
            ),
            ("model", "", ID_MODEL),
        ],
    );
}

#[test]
fn adversarial_same_offset_different_text_resolves_against_the_callee() {
    // The callee's generic parameter `T` sits at a byte offset where the CALLER's file holds
    // different text. If instantiation read the parameter name against the caller's file (the
    // DEV-101 bug) it would read that other text and fail to substitute; reading against the
    // callee's file is what makes this compile and run.
    assert_clean_and_runs(
        "adversarial",
        &[
            // Caller: whatever byte lands at the callee's `T` span, it is not `T` here.
            (
                "app",
                "model",
                "use model::id;\nfn main(){let zzzzz:Int32=1;assert_eq(id::<Int32>(zzzzz),1);}",
            ),
            ("model", "", "pub fn id<T>(x: T) -> T { x }\n"),
        ],
    );
}

// ------------------------------------------------------------------- controls --

#[test]
fn control_same_file_generic_still_works() {
    // A single-package program: the same-file path must be unaffected by the fix.
    assert_clean_and_runs(
        "ctrl_samefile",
        &[(
            "app",
            "",
            "fn id<T>(x: T) -> T { x }\nfn main() { let a: Int32 = 5; assert_eq(id::<Int32>(a), 5); }",
        )],
    );
}

#[test]
fn control_non_generic_cross_package_call_still_works() {
    assert_clean_and_runs(
        "ctrl_nongeneric",
        &[
            (
                "app",
                "model",
                "use model::inc;\nfn main() { assert_eq(inc(4), 5); }",
            ),
            ("model", "", "pub fn inc(x: Int32) -> Int32 { x + 1 }\n"),
        ],
    );
}
