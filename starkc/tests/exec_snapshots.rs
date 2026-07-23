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

/// WP-C3-ENTRY corpus freeze (CD-025): the execution-snapshot corpus is versioned and locked
/// so that C3/C4 differential work references a stable baseline by `corpus_version`, not by
/// "current contents." `corpus.lock` records `corpus_version`, the base commit, and a SHA-256
/// per corpus file (every `.stark`/`.snap`, including `metamorphic/`). This test enforces the
/// freeze three ways: every listed file's hash must still match, no listed file may be missing,
/// and no `.stark`/`.snap` file may exist that the lock does not list. Any intentional corpus
/// change must regenerate `corpus.lock` AND bump `corpus_version` with a dated note in
/// `COMPILER-STATE.md` (see WP-C3-ENTRY.md); regenerating a `.snap` via `UPDATE_SNAPSHOTS=1`
/// without a version bump is a freeze violation this test will catch.
#[test]
fn corpus_lock_matches_frozen_snapshot() {
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }

    let dir = corpus_dir();
    let lock_text = std::fs::read_to_string(dir.join("corpus.lock"))
        .expect("corpus.lock must exist (WP-C3-ENTRY freeze)");

    // Parse: skip blank/`#` lines; `key = value` header lines; `<hash>  <path>` file lines.
    let mut version: Option<String> = None;
    let mut locked: BTreeMap<String, String> = BTreeMap::new();
    for line in lock_text.lines() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once(" = ") {
            if key == "corpus_version" {
                version = Some(value.to_string());
            }
            continue;
        }
        let (hash, path) = line
            .split_once("  ")
            .unwrap_or_else(|| panic!("malformed corpus.lock line: {line:?}"));
        assert!(
            locked.insert(path.to_string(), hash.to_string()).is_none(),
            "duplicate path in corpus.lock: {path}"
        );
    }
    // Freeze governance: this assertion is the deliberate speed bump. A corpus change must bump
    // `corpus_version`, regenerate the lock, AND update this line, each with a dated note in
    // COMPILER-STATE.md — so no `UPDATE_SNAPSHOTS=1` run can quietly redefine the baseline.
    //
    // 1.0.0 → 1.1.0 (WP-C4.7-9, 2026-07-20, CD-037): five cases for the Class-A / WP-C4.7
    // constructs, every one of which the differential suite exercised and no frozen case did.
    // 1.1.0 → 1.2.0 (post-exit-report, 2026-07-20, CD-039): completes the compact refresh to the
    // six workloads the owner specified — a MULTI-FILE case (cross-file structs, methods, trait
    // default + override, cross-file Drop, source provenance) and DEV-086's consuming array
    // pattern folded into the array/slice case. All 48 hashes from 1.0.0 remain unchanged across
    // both bumps, so the original baseline survives byte-identically and comparisons against it
    // stay valid.
    // 1.2.0 → 1.3.0 (WP-C5.3e, 2026-07-23, CD-067/CD-069): a RE-PIN rather than new coverage, and
    // the first bump that changes an existing expectation instead of adding cases. The layout
    // lines of `option_result__03_box_and_layout_queries` recorded the pre-contract placeholder —
    // every consumer answered one machine word for every type — so `size_of::<Int32>()` was 8 and
    // `align_of::<Bool>()` was 8. Under the named target contract `stark-64-v1` they are 4 and 1.
    // One file, two output lines; every other hash from 1.0.0 onward is untouched. MIR amendment
    // A4 predicted this re-pin as the cost of its option (b).
    // 1.3.0 → 1.4.0 (WP-C5.6, 2026-07-23, CD-076): adds the two exact source cases approved for
    // C5 cross-backend replay. They use only the frozen C5 native subset: one comprehensive
    // completion and one deterministic overflow trap. Existing hashes remain unchanged.
    assert_eq!(
        version.as_deref(),
        Some("1.4.0"),
        "corpus.lock corpus_version changed without updating this assertion (freeze governance)"
    );

    // Enumerate the actual corpus (every .stark/.snap, recursively), relative to the corpus dir.
    let mut actual: Vec<String> = Vec::new();
    fn collect(dir: &Path, base: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(&path, base, out);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("stark") | Some("snap")
            ) {
                out.push(
                    path.strip_prefix(base)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    collect(&dir, &dir, &mut actual);
    actual.sort();

    // No unlisted corpus file (present-but-unlisted).
    for path in &actual {
        assert!(
            locked.contains_key(path),
            "corpus file '{path}' is not listed in corpus.lock: freeze a new corpus_version \
             and regenerate the lock rather than adding files silently"
        );
    }
    // No missing listed file (listed-but-absent) and every hash matches.
    for (path, expected_hash) in &locked {
        let bytes = std::fs::read(dir.join(path))
            .unwrap_or_else(|_| panic!("corpus.lock lists '{path}' but it is absent on disk"));
        let actual_hash = hex(&Sha256::digest(&bytes));
        assert_eq!(
            &actual_hash, expected_hash,
            "corpus file '{path}' has changed since the v1.0.0 freeze; if intentional, bump \
             corpus_version and regenerate corpus.lock with a dated COMPILER-STATE.md note"
        );
    }
}

/// Every primary corpus case, one per named coverage category from the WP-C2.12 roadmap text.
/// Filenames are `<category>__<NN>_<description>.stark`; the matching golden file is the same
/// name with a `.snap` extension.
const CASES: &[&str] = &[
    // corpus_version 1.4.0 (WP-C5.6): exact sources replayed by the three-engine harness.
    "c5_native__01_supported_completion",
    "c5_native__02_supported_overflow_trap",
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
    // corpus_version 1.1.0 (WP-C4.7-9): the constructs WP-C4.6's Class-A campaign and WP-C4.7
    // added. Until now every one of them was covered by the differential suite but by NO frozen
    // case, so a future freeze would have locked in a corpus that never exercised them.
    "ownership_drop__03_discarded_values_and_nested_patterns",
    "collection_iter__03_slice_views_and_array_iteration",
    "struct_enum_trait__05_generic_methods_and_impl_heads",
    "primitive__04_bitwise_shift_pow_and_ordering",
    "option_result__03_box_and_layout_queries",
    // Multi-file: the module's own file (`helper.stark`) is a corpus FILE but not a CASE — it has
    // no `main`. The lock hashes it; only the entry appears here.
    "multi_file__01_cross_file_execution_and_provenance",
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
