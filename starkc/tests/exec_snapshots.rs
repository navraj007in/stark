//! Differential/execution snapshot corpus (WP-C2.12, Gate C2).
//!
//! Every named case is parsed, resolved, type-checked (asserting zero diagnostics -- this
//! corpus tests *execution* semantics for programs that compile cleanly; compile-time rejection
//! has its own corpus in `conformance.rs`/`gate2_valid.rs`), and executed. The full observable
//! outcome (status/stdout/stderr on normal completion, or trap category/message) is rendered
//! deterministically and compared against a golden file in `tests/exec_snapshots/<name>.snap`.
//!
//! Regenerate golden files after an intentional semantic change with:
//! ```text
//! UPDATE_SNAPSHOTS=1 cargo test --test exec_snapshots
//! ```
//! and review the diff like any other code change -- mirrors `tests/snapshots.rs`'s existing
//! AST-snapshot convention, applied to full-pipeline execution instead of parse-only output.
//!
//! Metamorphic pairs (`tests/exec_snapshots/metamorphic/`) assert two differently-shaped but
//! semantically-equivalent programs produce byte-identical execution snapshots -- the base case
//! is the oracle for its transformed sibling, no golden file needed. See `METAMORPHIC_PAIRS`.
//! `workspace_relocation_produces_identical_execution_output` covers the one metamorphic class
//! that needs a real multi-package filesystem layout rather than a single source file.
//!
//! This snapshot format is designed to run unchanged against the HIR interpreter (today), a
//! future MIR interpreter, and native debug/release builds (Gate C3+) -- the differential
//! comparator this WP and the abstract machine's `OBS-COMPARE-001` rule both require. That
//! cross-backend replay is future work; this WP builds the corpus and the single-backend
//! harness it will run under.
//!
//! ## Coverage in this initial corpus (see COMPILER-STATE.md's WP-C2.12 session record for the
//! full disposition)
//!
//! Every category named by the WP-C2.12 roadmap text has real, representative coverage:
//! expressions/statements, primitive operations (including an overflow-trap case), struct/enum/
//! generic/trait/method, ownership/drop edges, `Option`/`Result`, collections/iterators, and
//! multi-file/package execution. All seven named metamorphic-transformation classes (alpha-
//! renaming, harmless scopes, equivalent explicit/inferred generics, trait-qualified calls,
//! field shorthand/explicit initialization, equivalent pattern decompositions, equivalent
//! non-overlapping match-arm order) plus workspace relocation are covered by at least one
//! worked pair each. This is a representative *initial* corpus, not an exhaustive one -- the
//! roadmap's "generated ... coverage" half (a case generator, as opposed to this WP's
//! hand-written cases) and deeper per-category breadth remain follow-up work, tracked in
//! COMPILER-STATE.md rather than silently claimed as complete here.

use starkc::diag::Severity;
use starkc::interp;
use starkc::options::LanguageOptions;
use starkc::package::{find_package_root, PackageGraph};
use starkc::parser::{parse, parse_package_graph, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/exec_snapshots")
}

/// Every primary corpus case, one per named coverage category from the WP-C2.12 roadmap text.
/// Filenames are `<category>__<NN>_<description>.stark`; the matching golden file is the same
/// name with a `.snap` extension.
const CASES: &[&str] = &[
    "expr_stmt__01_arithmetic_and_precedence",
    "expr_stmt__02_if_else_and_block_tail",
    "expr_stmt__03_loops_break_continue",
    "expr_stmt__04_match_and_patterns",
    "primitive__01_integer_widths_and_overflow_traps",
    "primitive__02_integer_overflow_traps",
    "primitive__03_float_arithmetic_and_casts",
    "struct_enum_trait__01_struct_construction_and_methods",
    "struct_enum_trait__02_enum_and_pattern_match",
    "struct_enum_trait__03_generic_function_and_trait_bound",
    "struct_enum_trait__04_trait_default_and_override",
    "ownership_drop__01_move_and_drop_order",
    "ownership_drop__02_shared_borrow_does_not_move",
    "option_result__01_option_construction_and_match",
    "option_result__02_result_and_try_propagation",
    "collection_iter__01_vec_push_index_iterate",
    "collection_iter__02_hashmap_insert_get_iteration_order",
];

/// (base, transformed) pairs: each pair must render an *identical* execution snapshot. One pair
/// per metamorphic transformation class named in the WP-C2.12 roadmap text.
const METAMORPHIC_PAIRS: &[(&str, &str)] = &[
    // alpha-renaming
    ("metamorphic/alpha_base", "metamorphic/alpha_renamed"),
    // harmless (behavior-preserving) added scopes
    ("metamorphic/scopes_base", "metamorphic/scopes_wrapped"),
    // equivalent explicit and inferred generics
    (
        "metamorphic/generics_inferred",
        "metamorphic/generics_explicit",
    ),
    // trait-qualified calls vs. operator sugar for the same dispatch
    (
        "metamorphic/trait_call_operator",
        "metamorphic/trait_call_qualified",
    ),
    // field shorthand vs. explicit initialization
    (
        "metamorphic/field_init_shorthand",
        "metamorphic/field_init_explicit",
    ),
    // equivalent pattern decompositions (nested tuple match vs. sequential match)
    (
        "metamorphic/pattern_nested_match",
        "metamorphic/pattern_sequential_match",
    ),
    // equivalent non-overlapping match-arm order (wildcard arm stays last in both; only the
    // mutually-exclusive literal arms above it are reordered, so reordering cannot change which
    // arm a given input selects)
    (
        "metamorphic/match_order_ascending",
        "metamorphic/match_order_scrambled",
    ),
];

/// Renders the full observable outcome of running `name.stark`: normal completion (status,
/// stdout, stderr) or a language trap (message, plus whether it is a genuine runtime trap versus
/// an entrypoint-selection failure detected before execution starts). Deterministic and stable
/// across repeated runs -- required for both golden-file comparison and metamorphic-pair
/// equality. Every case is expected to compile cleanly; a parse/resolve/type diagnostic is a
/// harness bug (a case that shouldn't be in this corpus), not a captured outcome, so those fail
/// the assertion immediately rather than being rendered into the snapshot.
fn render(name: &str) -> String {
    let path = corpus_dir().join(format!("{name}.stark"));
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let file = Arc::new(SourceFile::new(path.to_string_lossy().into_owned(), source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(
        parse_diags.is_empty(),
        "{name}: parse diagnostics: {parse_diags:?}"
    );
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "{name}: resolve diagnostics: {resolve_diags:?}"
    );
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: type diagnostics: {errors:?}");
    render_execution(interp::run(&hir, file, &checked.tables))
}

fn render_execution(result: Result<interp::Execution, interp::RuntimeError>) -> String {
    match result {
        Ok(execution) => format!(
            "STATUS: {}\n--- stdout ---\n{}--- stderr ---\n{}",
            execution.status, execution.output, execution.stderr
        ),
        Err(error) => format!("TRAP (is_trap={}): {}\n", error.is_trap, error.message),
    }
}

fn snapshot_path(name: &str) -> PathBuf {
    corpus_dir().join(format!("{name}.snap"))
}

#[test]
fn exec_snapshots() {
    let update = std::env::var_os("UPDATE_SNAPSHOTS").is_some();
    let mut failures = Vec::new();
    for &name in CASES {
        let actual = render(name);
        let snap_path = snapshot_path(name);
        if update {
            std::fs::create_dir_all(snap_path.parent().unwrap()).unwrap();
            std::fs::write(&snap_path, &actual).unwrap();
            continue;
        }
        let expected = std::fs::read_to_string(&snap_path).unwrap_or_else(|_| {
            panic!(
                "missing snapshot {}; run UPDATE_SNAPSHOTS=1",
                snap_path.display()
            )
        });
        if actual != expected {
            failures.push(format!(
                "{name}:\n--- expected ---\n{expected}\n--- actual ---\n{actual}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "execution snapshot mismatches (if intended, rerun UPDATE_SNAPSHOTS=1 and review the \
         diff like any other code change):\n{}",
        failures.join("\n\n")
    );
}

#[test]
fn metamorphic_transformations_produce_identical_execution_snapshots() {
    let mut failures = Vec::new();
    for &(base, transformed) in METAMORPHIC_PAIRS {
        let base_output = render(base);
        let transformed_output = render(transformed);
        if base_output != transformed_output {
            failures.push(format!(
                "{base} vs {transformed} diverged (semantically equivalent programs must \
                 produce identical execution output):\n--- {base} ---\n{base_output}\n\
                 --- {transformed} ---\n{transformed_output}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

fn setup_temp_dir(name: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name);
    if base.exists() {
        let _ = std::fs::remove_dir_all(&base);
    }
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn copy_dir_recursive(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir_recursive(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

/// Builds a two-package workspace (a `lib` package plus an `app` package depending on it via a
/// relative path) under `root`, executes `app`, and returns the rendered execution snapshot.
fn build_and_run_relocatable_workspace(root: &Path) -> String {
    let app_dir = root.join("app");
    let lib_dir = root.join("lib");
    std::fs::create_dir_all(app_dir.join("src")).unwrap();
    std::fs::create_dir_all(lib_dir.join("src")).unwrap();

    std::fs::write(
        lib_dir.join("starkpkg.json"),
        r#"{"name": "lib", "version": "0.1.0", "entry": "src/main.stark"}"#,
    )
    .unwrap();
    std::fs::write(
        lib_dir.join("src/main.stark"),
        "pub fn double(n: Int32) -> Int32 { n * 2 }",
    )
    .unwrap();

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
    std::fs::write(
        app_dir.join("src/main.stark"),
        "use lib::double;\nfn main() { println(double(21)); }",
    )
    .unwrap();

    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(parse_diags.is_empty(), "parse: {:?}", parse_diags);

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));
    let (hir, resolve_diags) = resolve(&ast, root_file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {:?}", resolve_diags);
    let checked = typecheck::analyze(&hir, root_file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck: {:?}", errors);
    render_execution(interp::run(&hir, root_file, &checked.tables))
}

/// Metamorphic case: relocation of an entire workspace without changing manifests, lock data, or
/// logical package sources must not change execution output. Builds a two-package workspace at
/// one temp path, runs it, copies the *entire* directory tree byte-for-byte to a second,
/// differently-named temp path (manifests and source untouched -- only the absolute path
/// changes, and the dependency is declared as a relative path so it remains valid after the
/// move), runs it again from there, and asserts identical execution snapshots.
#[test]
fn workspace_relocation_produces_identical_execution_output() {
    let original = setup_temp_dir("temp_exec_snapshot_relocation_original");
    let original_output = build_and_run_relocatable_workspace(&original);

    let relocated = setup_temp_dir("temp_exec_snapshot_relocation_moved_elsewhere");
    let _ = std::fs::remove_dir_all(&relocated);
    copy_dir_recursive(&original, &relocated);

    let app_dir = relocated.join("app");
    let manifest_path = find_package_root(&app_dir).unwrap();
    let graph = PackageGraph::load_from_root(&manifest_path).unwrap();
    let (ast, parse_diags) = parse_package_graph(&graph, LanguageOptions::CORE);
    assert!(parse_diags.is_empty(), "parse: {:?}", parse_diags);

    let entry_src = std::fs::read_to_string(app_dir.join("src/main.stark")).unwrap();
    let root_file = Arc::new(SourceFile::new(
        app_dir
            .join("src/main.stark")
            .to_string_lossy()
            .into_owned(),
        entry_src,
    ));
    let (hir, resolve_diags) = resolve(&ast, root_file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {:?}", resolve_diags);
    let checked = typecheck::analyze(&hir, root_file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "typecheck: {:?}", errors);
    let relocated_output = render_execution(interp::run(&hir, root_file, &checked.tables));

    assert_eq!(
        original_output, relocated_output,
        "relocating a workspace to a different absolute path must not change execution output"
    );

    let _ = std::fs::remove_dir_all(&original);
    let _ = std::fs::remove_dir_all(&relocated);
}
