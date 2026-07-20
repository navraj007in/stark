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
    // E0208 module-file conflict error should be reported
    assert!(
        diags_conflict
            .iter()
            .any(|d| d.code.as_deref() == Some("E0208")),
        "expected E0208 conflict error, got {:?}",
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
        diags.iter().any(|d| d.code.as_deref() == Some("E0208")),
        "expected E0208 'file not found for module', got {:?}",
        diags
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}

/// DEV-036 (WP-C2.12) regression: `load_submodules_recursive` used to suppress "file not found
/// for module" whenever the compiled file's path contained the substring `"spec-fixtures"` or
/// `"STARKLANG"`, or was named exactly `"test.stark"` -- a narrower heuristic DEV-014's WP-C1.1
/// fix deliberately kept for one legitimate spec fixture, but which could silently swallow a
/// genuinely missing module file for any real user project whose path happened to collide. The
/// three tests below build real projects at colliding paths and assert the diagnostic is now
/// reported regardless -- the exemption is a harness-side, exact-fixture-name opt-in
/// (`parser::parse_project_allowing_missing_modules`, called only by
/// `starkc/tests/conformance.rs` for `07-Modules-and-Packages__01.stark`), not a runtime
/// path/filename heuristic that could ever match unrelated real projects again.
fn assert_missing_module_reported_at(base_dir: &std::path::Path, entry_file_name: &str) {
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(base_dir);
    }
    std::fs::create_dir_all(base_dir).unwrap();

    let main_src = "mod does_not_exist;\nfn main() {}";
    let main_path = base_dir.join(entry_file_name);
    std::fs::write(&main_path, main_src).unwrap();

    let file = Arc::new(SourceFile::new(
        main_path.to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (_ast, diags) = parse(&file, ParseMode::Program);
    assert!(
        diags.iter().any(|d| d.code.as_deref() == Some("E0208")),
        "expected E0208 'file not found for module' for a real project at {}, got {:?} \
         (a path/filename collision with the old bypass heuristic must not suppress this)",
        main_path.display(),
        diags
    );

    let _ = std::fs::remove_dir_all(base_dir);
}

#[test]
fn test_missing_module_file_is_reported_even_when_path_contains_spec_fixtures() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/temp_missing_module_spec-fixtures_collision");
    assert_missing_module_reported_at(&base_dir, "main.stark");
}

#[test]
fn test_missing_module_file_is_reported_even_when_path_contains_starklang() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/temp_missing_module_STARKLANG_collision");
    assert_missing_module_reported_at(&base_dir, "main.stark");
}

#[test]
fn test_missing_module_file_is_reported_even_when_entry_file_is_named_test_stark() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/temp_missing_module_test_stark_collision");
    assert_missing_module_reported_at(&base_dir, "test.stark");
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

/// WP-C1.4 regression test for DEV-006 (borrowck half): a use-after-move error inside a
/// mod-loaded, non-root file used to render against the root file (`borrowck.rs`'s
/// `BorrowChecker` held a single whole-crate `self.file`, never swapped per item the way
/// resolve.rs/typecheck.rs already do). Confirms the diagnostic's file now correctly identifies
/// `moved.stark`, not `main.stark`.
#[test]
fn test_borrowck_diagnostic_in_nonroot_file_reports_correct_file() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_borrowck_provenance");
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(&base_dir);
    }
    std::fs::create_dir_all(&base_dir).unwrap();

    let main_src = "mod moved;\nfn main() {}";
    let moved_src = "pub fn broken() { let a = String::from(\"x\"); let b = a; let c = a; }";
    std::fs::write(base_dir.join("main.stark"), main_src).unwrap();
    std::fs::write(base_dir.join("moved.stark"), moved_src).unwrap();

    let file = Arc::new(SourceFile::new(
        base_dir.join("main.stark").to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (ast, diags) = parse(&file, ParseMode::Program);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "resolve failed: {:?}",
        resolve_diags
    );
    let diags = typecheck::check(&hir, file);

    let moved = diags
        .iter()
        .find(|d| d.code.as_deref() == Some("E0100"))
        .unwrap_or_else(|| panic!("expected E0100 use-of-moved-value, got {diags:?}"));
    let moved_file = moved.file.as_ref().expect("diagnostic should carry a file");
    assert!(
        moved_file.name.ends_with("moved.stark"),
        "expected the diagnostic's file to be moved.stark, got {:?}",
        moved_file.name
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}

/// WP-C1.4 regression test for DEV-006 (flow half): a "use of possibly-uninitialized variable"
/// error inside a non-root file used to render against the root file (`flow.rs`'s file
/// parameter was previously named `_file` and structurally unused). Confirms the diagnostic's
/// file now correctly identifies `uninit.stark`, not `main.stark`.
#[test]
fn test_flow_diagnostic_in_nonroot_file_reports_correct_file() {
    let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_flow_provenance");
    if base_dir.exists() {
        let _ = std::fs::remove_dir_all(&base_dir);
    }
    std::fs::create_dir_all(&base_dir).unwrap();

    let main_src = "mod uninit;\nfn main() {}";
    let uninit_src = "pub fn broken() -> Int32 { let x: Int32; x }";
    std::fs::write(base_dir.join("main.stark"), main_src).unwrap();
    std::fs::write(base_dir.join("uninit.stark"), uninit_src).unwrap();

    let file = Arc::new(SourceFile::new(
        base_dir.join("main.stark").to_string_lossy().into_owned(),
        main_src.to_string(),
    ));
    let (ast, diags) = parse(&file, ParseMode::Program);
    assert!(diags.is_empty(), "parse failed: {:?}", diags);
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "resolve failed: {:?}",
        resolve_diags
    );
    let diags = typecheck::check(&hir, file);

    let uninit = diags
        .iter()
        .find(|d| d.code.as_deref() == Some("E0401"))
        .unwrap_or_else(|| panic!("expected E0401 use-of-uninitialized, got {diags:?}"));
    let uninit_file = uninit
        .file
        .as_ref()
        .expect("diagnostic should carry a file");
    assert!(
        uninit_file.name.ends_with("uninit.stark"),
        "expected the diagnostic's file to be uninit.stark, got {:?}",
        uninit_file.name
    );

    let _ = std::fs::remove_dir_all(&base_dir);
}

/// WP-C1.4 soundness fix: moving a non-Copy value out of a reference via `*r` (e.g.
/// `let stolen = *r;`) used to be accepted by `check_expr`'s generic `Unary`/`Deref` handling,
/// which had no case recognizing a deref as a move of the pointee. For a `Drop` type this is a
/// double-free/double-destruction hazard: both the original owner and the "stolen" local would
/// run `drop` on the same logical value. `check_owned_value` (borrowck.rs) now rejects this at
/// the three "value consumed as owned" sites (let-init, return, block-tail) while leaving sound
/// patterns (comparing through a deref, reborrowing `&*r`, accessing a field/index through a
/// deref) untouched, since those route through `check_expr`'s narrower, pre-existing handling.
#[test]
fn test_deref_move_of_drop_type_out_of_a_shared_reference_is_rejected() {
    let source = "\
        struct Marker { name: String }\n\
        impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } }\n\
        fn peek(r: &Marker) -> Unit {\n\
            let stolen = *r;\n\
            println(stolen.name.as_str());\n\
        }\n\
        fn main() -> Unit {\n\
            let m = Marker { name: String::from(\"m\") };\n\
            peek(&m);\n\
            println(m.name.as_str());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("deref_move.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0100")),
        "expected E0100 (cannot move a non-Copy value out of a reference), got {diagnostics:?}"
    );
}

/// WP-C1.4 soundness fix: `let it = v.iter();` creates a shared borrow of `v` that must stay
/// live for as long as `it` is (`it` is a live aliasing view into `v`'s storage), per Core v1's
/// "borrow-carrying types" rule (03-Type-System.md). Two bugs combined to defeat this:
/// (1) `type_carries_borrow` didn't recognize iterator CoreTypes as borrow-carrying at all (their
/// only generic argument is the *element* type, not a reference, so the borrow registered by
/// evaluating `v.iter()` was truncated at the end of the `let` statement instead of persisting);
/// (2) even with the borrow correctly kept alive, `consume_place` only rejected moves that
/// conflicted with a *mutable* active borrow (`check_read_borrow_conflict` is read-oriented:
/// concurrent shared reads are sound), never a shared one -- but moving `v` out from under a live
/// shared-borrowing iterator is unsound regardless, since the move invalidates the storage the
/// iterator still aliases. Confirmed empirically before the fix: this program compiled and then
/// crashed at runtime ("use of unavailable value"). Both are now fixed: `type_carries_borrow`
/// gained an iterator-CoreType arm, and `consume_place` now rejects a move against *any* active
/// borrow of the local being moved.
#[test]
fn test_moving_a_vec_while_a_live_iterator_borrows_it_is_rejected() {
    let source = "\
        fn consume(v: Vec<Int32>) -> Unit {}\n\
        fn main() -> Unit {\n\
            let mut v: Vec<Int32> = Vec::new();\n\
            v.push(1);\n\
            v.push(2);\n\
            let mut it = v.iter();\n\
            consume(v);\n\
            match it.next() {\n\
                Some(x) => println(*x),\n\
                None => println(-1),\n\
            }\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("iter_move.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0101")),
        "expected E0101 (move conflicts with an active borrow), got {diagnostics:?}"
    );
}

/// Companion positive case for the fix above: an iterator that is fully consumed (or otherwise
/// goes out of scope) before the source collection is moved must still compile -- the borrow is
/// lexically scoped to the enclosing block, not to the whole function. Guards against an
/// overly-conservative fix that would reject all post-iteration moves of an iterated collection.
#[test]
fn test_moving_a_vec_after_its_iterator_block_ends_is_still_accepted() {
    let source = "\
        fn consume(v: Vec<Int32>) -> Unit {}\n\
        fn main() -> Unit {\n\
            let mut v: Vec<Int32> = Vec::new();\n\
            v.push(1);\n\
            v.push(2);\n\
            {\n\
                let mut it = v.iter();\n\
                match it.next() {\n\
                    Some(x) => println(*x),\n\
                    None => println(-1),\n\
                }\n\
            }\n\
            consume(v);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("iter_scoped.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "expected no errors once the iterator's block has ended, got {errors:?}"
    );
}

/// WP-C1.4 coverage for the shortest-input-lifetime rule (03-Type-System.md's worked example
/// under "References and Lifetimes", rule 3): `longest`'s returned reference must be treated as
/// derived from BOTH `x` and `y`, "regardless of which branch was taken" (the spec's own
/// wording), since the checker cannot know which branch runs without evaluating the condition.
/// This was flagged during WP-C1.4 research as a possibly-missing rule (no code appeared to
/// trace a call's reference-typed arguments to its borrow-carrying return type), but empirical
/// verification found it already sound: `check_expr`'s handling of `&expr`/ref-returning-method
/// arguments unconditionally registers a borrow on the argument's local no matter where that
/// argument expression appears (call, tuple, etc.), and `check_stmt`'s `Let` branch keeps every
/// borrow registered while evaluating the init expression alive past the `let` whenever the
/// bound type carries a borrow (`longest`'s `&str` return type does). The combination already
/// implements rule 3's "conservatively borrow every reference argument" semantics with no
/// call-argument-to-return-type tracing code needed. This test is a regression guard confirming
/// that behavior, not a bug fix.
#[test]
fn test_shortest_input_lifetime_rule_ties_a_returned_reference_to_every_ref_argument() {
    let source = "\
        fn longest(x: &str, y: &str) -> &str {\n\
            if x.len() > y.len() { x } else { y }\n\
        }\n\
        fn main() -> Unit {\n\
            let a = String::from(\"a\");\n\
            let mut b = String::from(\"bb\");\n\
            let l = longest(a.as_str(), b.as_str());\n\
            b.push_str(\"ccc\");\n\
            println(l);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("shortest_lifetime.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0101")),
        "expected E0101 (mutating `b` while `l` may still alias it through the shortest-input-\
         lifetime rule), got {diagnostics:?}"
    );
}

/// WP-C1.4 coverage (item verified correctly implemented, previously untested): moving a single
/// field out of a struct whose *own* type implements `Drop` must be rejected -- `Drop::drop`
/// takes `&mut self` and assumes every field is still present, so a partial move would leave the
/// value in a state its own destructor cannot safely observe. `consume_place`'s check keys on
/// `local_has_drop(place.local)` -- the root local's type, not the field's type -- which is the
/// correct rule (see the companion positive test below for why field-type-Drop alone must not
/// trigger this).
#[test]
fn test_partial_move_out_of_a_struct_that_itself_implements_drop_is_rejected() {
    let source = "\
        struct Marker { name: String }\n\
        struct Holder { m: Marker }\n\
        impl Drop for Holder { fn drop(&mut self) { println(\"dropping holder\"); } }\n\
        fn main() -> Unit {\n\
            let h = Holder { m: Marker { name: String::from(\"x\") } };\n\
            let stolen = h.m;\n\
            println(stolen.name.as_str());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("partial_move_drop.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0100")),
        "expected E0100 (cannot partially move a value whose type implements Drop), got \
         {diagnostics:?}"
    );
}

/// Companion positive case: `Holder` here has no `Drop` impl of its own (only its field type
/// `Marker` does). Moving the `m` field out whole is sound Rust-style partial-move behavior --
/// the remaining fields of `h` are still individually droppable, and `h` itself never has a
/// `drop()` invariant to violate. Guards against a stricter-than-intended fix that would key
/// the check on the *field's* type instead of the *containing local's* type.
#[test]
fn test_partial_move_out_of_a_struct_without_its_own_drop_impl_is_accepted() {
    let source = "\
        struct Marker { name: String }\n\
        impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } }\n\
        struct Holder { m: Marker }\n\
        fn main() -> Unit {\n\
            let h = Holder { m: Marker { name: String::from(\"x\") } };\n\
            let stolen = h.m;\n\
            println(stolen.name.as_str());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("partial_move_no_drop.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "expected no errors for a partial move out of a Drop-free container, got {errors:?}"
    );
}

/// WP-C1.4 soundness fix: `check_owned_value`'s deref-move check (see the fix above for the
/// let/return/tail case) originally left call arguments uncovered, on the theory that closing
/// that gap would need callee-signature-aware argument classification. Confirmed empirically
/// that theory was unnecessary: STARK has no argument-position auto-ref/deref-coercion, so
/// `take(*r)` only type-checks when `take`'s parameter is already the pointee type by value --
/// meaning the check can apply uniformly to every call argument with no false positives on
/// by-reference parameters (those already fail to type-check independent of this rule). Before
/// this fix, `take(*r)` for a `Drop` type compiled and double-dropped at runtime (confirmed:
/// the destructor's `println` ran twice, once from `take`'s parameter and once from the
/// original owner going out of scope).
#[test]
fn test_deref_move_as_a_call_argument_is_rejected() {
    let source = "\
        struct Marker { name: String }\n\
        impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } }\n\
        fn take(m: Marker) -> Unit { println(m.name.as_str()); }\n\
        fn peek(r: &Marker) -> Unit {\n\
            take(*r);\n\
        }\n\
        fn main() -> Unit {\n\
            let m = Marker { name: String::from(\"m\") };\n\
            peek(&m);\n\
            println(m.name.as_str());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("deref_move_arg.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0100")),
        "expected E0100 (cannot move a non-Copy value out of a reference) for a by-value call \
         argument, got {diagnostics:?}"
    );
}

/// WP-C1.4 soundness fix: the same deref-move gap as the call-argument case above, but for
/// aggregate construction -- a tuple element (`(*r, 1)`), array element, or struct-literal field
/// value (`S { field: *r }`) all build a new owner out of the sub-expression's value, and were
/// equally uncovered before `check_owned_value` was wired into their element/field loops.
/// Confirmed empirically that a tuple literal containing `*r` for a `Drop` type compiled and
/// double-dropped before this fix.
#[test]
fn test_deref_move_into_a_tuple_literal_is_rejected() {
    let source = "\
        struct Marker { name: String }\n\
        impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } }\n\
        fn peek(r: &Marker) -> Unit {\n\
            let pair = (*r, 1);\n\
            println(pair.0.name.as_str());\n\
        }\n\
        fn main() -> Unit {\n\
            let m = Marker { name: String::from(\"m\") };\n\
            peek(&m);\n\
            println(m.name.as_str());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("deref_move_tuple.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0100")),
        "expected E0100 (cannot move a non-Copy value out of a reference) for a tuple element, \
         got {diagnostics:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// WP-C1.5: control flow, patterns, constants, and numeric semantics
// ---------------------------------------------------------------------------------------------

/// `pat_subsumes` used to compare `Lit` patterns by shape only (base/suffix tags, no value),
/// so any two same-kind integer literal patterns were treated as equal regardless of value.
/// `2` here must not be flagged as redundant against `1`.
#[test]
fn test_distinct_integer_literal_patterns_are_not_flagged_unreachable() {
    let source = "\
        fn main() -> Unit {\n\
            let x: Int32 = 2;\n\
            let r = match x { 1 => 10, 2 => 20, _ => 0 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("lit_pattern_distinct.stark", source);
    assert!(
        !diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("W0006")),
        "distinct literal patterns must not be flagged unreachable, got {diagnostics:?}"
    );
}

/// Companion negative case: a genuinely duplicate literal pattern must still be flagged.
#[test]
fn test_duplicate_integer_literal_pattern_is_still_flagged_unreachable() {
    let source = "\
        fn main() -> Unit {\n\
            let x: Int32 = 2;\n\
            let r = match x { 1 => 10, 1 => 99, _ => 0 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("lit_pattern_dup.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("W0006")),
        "expected W0006 (unreachable match arm) for a genuinely duplicate literal, got \
         {diagnostics:?}"
    );
}

/// Same literal-value-blindness bug, for string literal patterns.
#[test]
fn test_distinct_string_literal_patterns_are_not_flagged_unreachable() {
    let source = "\
        fn main() -> Unit {\n\
            let x: &str = \"b\";\n\
            let r = match x { \"a\" => 1, \"b\" => 2, _ => 0 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("lit_pattern_str.stark", source);
    assert!(
        !diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("W0006")),
        "distinct string literal patterns must not be flagged unreachable, got {diagnostics:?}"
    );
}

/// DEV-015: a suffixed integer literal whose value doesn't fit its suffix's representable range
/// must be rejected at compile time (previously: `let x: UInt8 = 300u8;` compiled clean).
#[test]
fn test_suffixed_integer_literal_out_of_range_is_rejected() {
    let source = "\
        fn main() -> Unit {\n\
            let x: UInt8 = 300u8;\n\
            println(x);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("dev015_suffixed.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0008")),
        "expected E0008 (integer literal out of range), got {diagnostics:?}"
    );
}

/// Companion positive case: a suffixed literal within range is unaffected.
#[test]
fn test_suffixed_integer_literal_within_range_is_accepted() {
    let source = "\
        fn main() -> Unit {\n\
            let x: UInt8 = 200u8;\n\
            println(x);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("dev015_suffixed_ok.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

/// DEV-015: an unsuffixed integer literal that doesn't fit `Int32` must be promoted to `Int64`
/// per 03-Type-System.md:28 ("Default integer type is Int32 for literals that fit, Int64
/// otherwise"), not silently mistyped as a broken Int32 (previously: no check existed at all).
#[test]
fn test_unsuffixed_integer_literal_exceeding_int32_promotes_to_int64() {
    let source = "\
        fn main() -> Unit {\n\
            let x = 99999999999;\n\
            println(x);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("dev015_unsuffixed_big.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "a literal exceeding Int32 but fitting Int64 must be accepted (promoted to Int64), got \
         {errors:?}"
    );
}

/// Companion: a literal exceeding even Int64's range must be rejected outright.
#[test]
fn test_unsuffixed_integer_literal_exceeding_int64_is_rejected() {
    let source = "\
        fn main() -> Unit {\n\
            let x = 999999999999999999999;\n\
            println(x);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("dev015_unsuffixed_toobig.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0008")),
        "expected E0008 (integer literal out of range) for a literal exceeding Int64, got \
         {diagnostics:?}"
    );
}

/// Array-repeat-count const-eval previously parsed the *raw source text* of `count` as a bare
/// unsuffixed decimal, so a suffixed literal (`5u32`) silently computed length 0, falsely
/// rejecting every subsequent valid index into the array.
#[test]
fn test_array_repeat_count_accepts_a_suffixed_literal() {
    let source = "\
        fn main() -> Unit {\n\
            let a = [0; 5u32];\n\
            println(a[4]);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("repeat_suffixed.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

/// Same bug, for a reference to a `const` item as the repeat count.
#[test]
fn test_array_repeat_count_accepts_a_const_item_reference() {
    let source = "\
        const N: Int32 = 3;\n\
        fn main() -> Unit {\n\
            let a = [7; N];\n\
            println(a[2]);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("repeat_const.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

/// Companion negative case: a genuinely non-constant repeat count must be rejected with a clear
/// diagnostic (E0009), not silently fall back to length 0 (which previously produced a confusing
/// cascade of "index out of bounds" errors instead).
#[test]
fn test_array_repeat_count_rejects_a_non_constant_expression() {
    let source = "\
        fn main() -> Unit {\n\
            let mut n: Int32 = 3;\n\
            n = n + 1;\n\
            let a = [0; n];\n\
            println(a[0]);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("repeat_not_const.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0009")),
        "expected E0009 (array repeat count must be a compile-time constant expression), got \
         {diagnostics:?}"
    );
}

/// The `?` operator's Result/Option identity check used to be a substring search over an enum's
/// *entire declaration source text* for "Result"/"Option" -- an unrelated user enum with a
/// variant merely named `ResultVariant` satisfied it. 03-Type-System.md:590 defines `?`
/// exclusively for `Result<T,E>`/`Option<T>`; there is no user-extensible Try trait in Core v1.
#[test]
fn test_try_operator_rejects_an_unrelated_enum_with_a_result_like_name() {
    let source = "\
        enum Foo { ResultVariant(Int32), Other }\n\
        fn get() -> Foo { Foo::ResultVariant(5) }\n\
        fn caller() -> Foo {\n\
            let x = get()?;\n\
            Foo::ResultVariant(x)\n\
        }\n\
        fn main() -> Unit {}\n\
    "
    .to_string();
    let diagnostics = analyze("try_op_unrelated_enum.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0006")),
        "expected E0006 (try operator requires Result or Option), got {diagnostics:?}"
    );
}

/// Companion positive case: `?` on the real prelude `Result`/`Option` types is unaffected.
#[test]
fn test_try_operator_still_accepts_real_result_and_option() {
    let result_source = "\
        fn get(x: Int32) -> Result<Int32, String> {\n\
            if x < 0 { Err(String::from(\"negative\")) } else { Ok(x) }\n\
        }\n\
        fn caller(x: Int32) -> Result<Int32, String> {\n\
            let v = get(x)?;\n\
            Ok(v + 1)\n\
        }\n\
        fn main() -> Unit {}\n\
    "
    .to_string();
    let result_diags = analyze("try_op_result_ok.stark", result_source);
    let result_errors: Vec<_> = result_diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(result_errors.is_empty(), "got {result_errors:?}");

    let option_source = "\
        fn get(x: Int32) -> Option<Int32> {\n\
            if x < 0 { None } else { Some(x) }\n\
        }\n\
        fn caller(x: Int32) -> Option<Int32> {\n\
            let v = get(x)?;\n\
            Some(v + 1)\n\
        }\n\
        fn main() -> Unit {}\n\
    "
    .to_string();
    let option_diags = analyze("try_op_option_ok.stark", option_source);
    let option_errors: Vec<_> = option_diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(option_errors.is_empty(), "got {option_errors:?}");
}

/// Match exhaustiveness was only enforced for `Enum` and `Bool` scrutinees. `Option`/`Result`
/// resolve to `Ty::Core`, not `Ty::Enum` (see `hir::CoreType`), so they were never covered at
/// all -- `match opt { Some(v) => .. }` (missing `None`) compiled clean.
#[test]
fn test_non_exhaustive_option_match_is_rejected() {
    let source = "\
        fn main() -> Unit {\n\
            let x: Option<Int32> = None;\n\
            match x { Some(v) => println(v) }\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_option.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0303")),
        "expected E0303 (non-exhaustive pattern match), got {diagnostics:?}"
    );
}

/// Same gap, for `Result`.
#[test]
fn test_non_exhaustive_result_match_is_rejected() {
    let source = "\
        fn main() -> Unit {\n\
            let x: Result<Int32, String> = Ok(5);\n\
            match x { Ok(v) => println(v) }\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_result.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0303")),
        "expected E0303 (non-exhaustive pattern match), got {diagnostics:?}"
    );
}

/// Companion positive case: covering both Option/Result arms is still accepted.
#[test]
fn test_exhaustive_option_and_result_matches_are_accepted() {
    let option_source = "\
        fn main() -> Unit {\n\
            let x: Option<Int32> = Some(5);\n\
            let r = match x { Some(v) => v, None => 0 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let option_diags = analyze("exhaustive_option_ok.stark", option_source);
    let option_errors: Vec<_> = option_diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(option_errors.is_empty(), "got {option_errors:?}");

    let result_source = "\
        fn main() -> Unit {\n\
            let x: Result<Int32, String> = Ok(5);\n\
            let r = match x { Ok(v) => v, Err(_) => 0 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let result_diags = analyze("exhaustive_result_ok.stark", result_source);
    let result_errors: Vec<_> = result_diags
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(result_errors.is_empty(), "got {result_errors:?}");
}

/// 04-Semantic-Analysis.md: "If a match is not exhaustive, it is a compile-time error." Before
/// this WP, exhaustiveness was checked only for Enum/Bool scrutinees; every other type (Int32,
/// tuples, &str, etc.) silently accepted a non-exhaustive match and only trapped at *runtime* if
/// an unmatched value actually occurred. A real usefulness/coverage algorithm is out of this
/// WP's scope; instead any scrutinee type outside the small, exactly-enumerable domains
/// (Bool/Enum/Option/Result) now requires an explicit wildcard/binding arm.
#[test]
fn test_non_exhaustive_int32_match_without_wildcard_is_rejected() {
    let source = "\
        fn main() -> Unit {\n\
            let x: Int32 = 2;\n\
            let r = match x { 1 => 10, 2 => 20 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_int32.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0303")),
        "expected E0303 (non-exhaustive pattern match), got {diagnostics:?}"
    );
}

/// Companion positive case: a wildcard makes the same match acceptable.
#[test]
fn test_int32_match_with_wildcard_is_accepted() {
    let source = "\
        fn main() -> Unit {\n\
            let x: Int32 = 2;\n\
            let r = match x { 1 => 10, 2 => 20, _ => 0 };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_int32_ok.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

/// A `Ty::Struct` scrutinee is exempted from the general "requires wildcard" rule: a struct type
/// has exactly one shape, so a single struct-pattern arm is exhaustive over it by construction,
/// even without a trailing wildcard.
#[test]
fn test_single_arm_struct_match_without_wildcard_is_accepted() {
    let source = "\
        struct Point { x: Int32, y: Int32 }\n\
        fn main() -> Unit {\n\
            let p = Point { x: 1, y: 2 };\n\
            let r = match p { Point { x, y } => x + y };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_struct.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

/// A fully-binding tuple pattern (`(a, b)`, every component a binding) is irrefutable and must
/// count as a wildcard-equivalent arm, even though it isn't literally `Wild`/`Binding` at the
/// top level. Regression guard for the general "requires wildcard" rule above -- an earlier
/// version of this fix flagged this exact program as non-exhaustive.
#[test]
fn test_fully_binding_tuple_pattern_counts_as_exhaustive() {
    let source = "\
        fn main() -> Unit {\n\
            let pair: (Int32, Int32) = (2, 3);\n\
            match pair { (a, b) => println(a + b) }\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_tuple_binding.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

/// Companion negative case: a tuple pattern with a literal component is *not* irrefutable and
/// must still require a wildcard.
#[test]
fn test_tuple_pattern_with_a_literal_component_requires_wildcard() {
    let source = "\
        fn main() -> Unit {\n\
            let pair: (Int32, Int32) = (1, 2);\n\
            let r = match pair { (1, y) => y };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_tuple_partial.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0303")),
        "expected E0303 (non-exhaustive pattern match), got {diagnostics:?}"
    );
}

/// A struct-shaped pattern matching an *enum variant* (`Message::Named { amount }`) is not
/// irrefutable on its own -- other variants can still occur. Regression guard distinguishing
/// this from the plain-struct exemption above.
#[test]
fn test_single_enum_variant_struct_pattern_without_other_variants_is_rejected() {
    let source = "\
        enum Message { Number(Int32), Named { amount: Int32 }, Empty }\n\
        fn main() -> Unit {\n\
            let m = Message::Named { amount: 5 };\n\
            let r = match m { Message::Named { amount } => amount };\n\
            println(r);\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("exhaustive_enum_variant_struct.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0303")),
        "expected E0303 (non-exhaustive pattern match), got {diagnostics:?}"
    );
}

#[test]
fn constant_subset_is_evaluated_at_compile_time() {
    let source = "\
        const BASE: Int32 = 3;\n\
        const TOTAL: Int32 = if true { BASE * 4 } else { 1 / 0 };\n\
        const PAIR: (Int32, Int32) = (TOTAL, 2);\n\
        fn main() { println(TOTAL); }\n\
    "
    .to_string();
    let diagnostics = analyze("constant_subset.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

#[test]
fn constants_reject_runtime_forms_cycles_and_traps() {
    let cases = [
        (
            "const BAD: Result<String, IOError> = read_file(\"missing\"); fn main() {}",
            "not permitted",
        ),
        (
            "const A: Int32 = B; const B: Int32 = A; fn main() {}",
            "constant dependency cycle",
        ),
        ("const BAD: Int32 = 1 / 0; fn main() {}", "division by zero"),
    ];
    for (index, (source, expected)) in cases.iter().enumerate() {
        let diagnostics = analyze(
            &format!("invalid_constant_{index}.stark"),
            source.to_string(),
        );
        assert!(
            diagnostics.iter().any(|diagnostic| {
                diagnostic.code.as_deref() == Some("E0215") && diagnostic.message.contains(expected)
            }),
            "expected E0215 containing {expected:?}, got {diagnostics:?}"
        );
    }
}

#[test]
fn aliases_are_transparent_and_boxed_recursion_is_sized() {
    let source = "\
        type Number = Int32;\n\
        type Pair<T> = (T, T);\n\
        struct Node { value: Number, next: Option<Box<Node>> }\n\
        struct Empty {}\n\
        enum Never {}\n\
        fn sum(pair: Pair<Number>) -> Number { pair.0 + pair.1 }\n\
        fn main() { let pair: (Int32, Int32) = (2, 3); println(sum(pair)); }\n\
    "
    .to_string();
    let diagnostics = analyze("transparent_aliases.stark", source);
    let errors: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "got {errors:?}");
}

#[test]
fn alias_cycles_direct_value_recursion_and_bare_unsized_types_are_rejected() {
    let cases = [
        (
            "type A = B; type B = A; fn main() {}",
            "recursive type-alias cycle",
        ),
        ("struct Node { next: Node } fn main() {}", "infinite size"),
        (
            "struct Left { right: Right } struct Right { left: Left } fn main() {}",
            "infinite size",
        ),
        ("struct Text { value: str } fn main() {}", "unsized types"),
    ];
    for (index, (source, expected)) in cases.iter().enumerate() {
        let diagnostics = analyze(&format!("invalid_type_{index}.stark"), source.to_string());
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains(expected)),
            "expected diagnostic containing {expected:?}, got {diagnostics:?}"
        );
    }
}

/// DEV-072 (WP-C4.7-5): binding a non-`Copy` payload out of a scrutinee read THROUGH a reference
/// moves ownership out of a borrow, which the ownership rules forbid. This passed the front end
/// before — patterns were not inspected at the `match` at all — while MIR lowering refused it, so
/// the two engines disagreed about whether the program was legal. The oracle's legacy clone
/// semantics hid the unsoundness at runtime by consuming the clone rather than the referent.
#[test]
fn binding_a_non_copy_payload_through_a_reference_is_rejected() {
    let source = "\
        enum Holder { Empty, Val(String) }\n\
        impl Holder {\n\
            fn peek(&self) -> Int32 {\n\
                match *self {\n\
                    Holder::Val(s) => 1,\n\
                    Holder::Empty => 0,\n\
                }\n\
            }\n\
        }\n\
        fn main() -> Unit {\n\
            let h = Holder::Val(String::from(\"x\"));\n\
            println(h.peek());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("moveoutofborrow.stark", source);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0101")),
        "expected E0101 (move out of a borrow via a match binding), got {diagnostics:?}"
    );
}

/// The complementary POSITIVE cases, which must keep compiling: matching by reference is legal —
/// it is only *taking ownership* that is not. A wildcard binds nothing, and a `Copy` payload is
/// copied rather than moved. Without these, the fix above could be "correct" by rejecting all
/// by-reference matching, which would break far more than it fixed.
#[test]
fn matching_through_a_reference_without_taking_ownership_is_accepted() {
    let wildcard = "\
        enum Holder { Empty, Val(String) }\n\
        impl Holder {\n\
            fn peek(&self) -> Int32 {\n\
                match *self {\n\
                    Holder::Val(_) => 1,\n\
                    Holder::Empty => 0,\n\
                }\n\
            }\n\
        }\n\
        fn main() -> Unit {\n\
            let h = Holder::Val(String::from(\"x\"));\n\
            println(h.peek());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("wildcard_ok.stark", wildcard);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "a wildcard binds nothing and must stay legal, got {diagnostics:?}"
    );

    let copy_payload = "\
        enum Tag { None, Num(Int32) }\n\
        impl Tag {\n\
            fn value(&self) -> Int32 {\n\
                match *self {\n\
                    Tag::Num(n) => n,\n\
                    Tag::None => 0,\n\
                }\n\
            }\n\
        }\n\
        fn main() -> Unit {\n\
            let t = Tag::Num(9);\n\
            println(t.value());\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("copy_ok.stark", copy_payload);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "a Copy payload is copied, not moved, and must stay legal, got {diagnostics:?}"
    );
}

/// DEV-067 (WP-C4.7-7) negatives: relaxing bound checking must not weaken it. A concrete type
/// that does not implement the bound, and an UNBOUNDED parameter forwarded to a bounded one,
/// must both still be rejected — the fix discharges an obligation only from a bound the caller
/// actually declared.
#[test]
fn unsatisfied_trait_bounds_are_still_rejected() {
    let concrete = "\
        trait Speak { fn speak(&self) -> Int32; }\n\
        fn needs_speak<T: Speak>(t: T) -> Int32 { t.speak() }\n\
        struct Mute { n: Int32 }\n\
        fn main() -> Unit {\n\
            let m = Mute { n: 1 };\n\
            println(needs_speak(m));\n\
        }\n\
    "
    .to_string();
    let diagnostics = analyze("bound_concrete.stark", concrete);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0500")),
        "a concrete type without the impl must still fail the bound, got {diagnostics:?}"
    );

    let unbounded = "\
        fn needs_ord<T: Ord>(a: T, b: T) -> Bool { a < b }\n\
        fn passes_unbounded<T>(a: T, b: T) -> Bool { needs_ord(a, b) }\n\
        fn main() -> Unit { println(passes_unbounded(1, 2)); }\n\
    "
    .to_string();
    let diagnostics = analyze("bound_unbounded.stark", unbounded);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0500")),
        "an unbounded parameter must not satisfy a callee's bound, got {diagnostics:?}"
    );
}

/// DEV-071 (WP-C4.7-7): an all-three-variant `Ordering` match is exhaustive and needs no
/// wildcard — but a two-variant one is still non-exhaustive. Both directions, because the fix
/// enumerates a domain and an enumeration that is too generous silently accepts unsound matches.
#[test]
fn ordering_match_exhaustiveness_counts_all_three_variants() {
    let exhaustive = "\
        fn label(o: Ordering) -> Int32 {\n\
            match o {\n\
                Ordering::Less => 1,\n\
                Ordering::Equal => 2,\n\
                Ordering::Greater => 3,\n\
            }\n\
        }\n\
        fn main() -> Unit { println(label(3.cmp(&5))); }\n\
    "
    .to_string();
    let diagnostics = analyze("ord_exhaustive.stark", exhaustive);
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "all three Ordering variants must be exhaustive, got {diagnostics:?}"
    );

    let partial = "\
        fn label(o: Ordering) -> Int32 {\n\
            match o {\n\
                Ordering::Less => 1,\n\
                Ordering::Equal => 2,\n\
            }\n\
        }\n\
        fn main() -> Unit { println(label(3.cmp(&5))); }\n\
    "
    .to_string();
    let diagnostics = analyze("ord_partial.stark", partial);
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0303")),
        "a two-variant Ordering match must stay non-exhaustive, got {diagnostics:?}"
    );
}

/// WP-C4.7-6.1: `*box` is NOT a Core v1 construct and must stay rejected. Core v1 has no `Deref`
/// trait (it is absent from `core-min`'s essential-trait list), TYPE-METHOD-002's auto-dereference
/// removes only leading `&`/`&mut`, and the abstract machine's dereference operates on *the
/// reference*. 06 gives `Box<T>` exactly `new` and `into_inner`.
///
/// This test exists because the WP-C4.6 gate audit misclassified "`Box` deref" as a `core-min`
/// hole. It is not a hole — it is conformant behaviour, and the real gap was `Box::new`/
/// `into_inner` not reaching MIR (fixed in the same increment, surface `0.1-A7`). Pinning the
/// rejection keeps a future session from "fixing" spec-conformant behaviour.
#[test]
fn box_deref_is_rejected() {
    let source = "fn main() -> Unit { let b = Box::new(5); println(*b); }\n".to_string();
    let diagnostics = analyze("box_deref.stark", source);
    assert!(
        diagnostics.iter().any(|d| d.severity == Severity::Error),
        "*box must remain unsupported in Core v1, got {diagnostics:?}"
    );
}

/// WP-C4.7-6.3: an unsuffixed integer literal in a position with an expected integer type ADOPTS
/// that type when its value is representable. 03-Type-System's solver flows expected types inward
/// from annotations, function parameters, fields and assignment destinations, and only defaults
/// "an **unconstrained** integer literal" to `Int32`/`Int64` — step 5, after that flow. The
/// checker used to skip to the default, committing every literal to `Int32`, so `v.get(0)` failed
/// "expected 'UInt64', found 'Int32'".
///
/// This is expected-type propagation, NOT a coercion (03 confines coercions to explicit coercion
/// sites), which is why the negative cases below still hold.
#[test]
fn unsuffixed_integer_literals_adopt_the_expected_integer_type() {
    let accepted = [
        // A function parameter.
        "fn takes_u64(n: UInt64) -> UInt64 { n }\n\
         fn main() -> Unit { println(takes_u64(0)); }\n",
        // An annotated local and a struct field.
        "struct S { n: UInt64 }\n\
         fn main() -> Unit { let s = S { n: 3 }; println(s.n); let a: UInt64 = 9; println(a); }\n",
        // TYPE-INFER-001: a LATER use constrains an unannotated local, so `index` is UInt64.
        "fn main() -> Unit {\n\
             let mut v: Vec<Int32> = Vec::new();\n\
             v.push(7);\n\
             let index = 0;\n\
             match v.get(index) { Some(x) => println(*x), None => println(-1), }\n\
         }\n",
    ];
    for source in accepted {
        let diagnostics = analyze("lit_ok.stark", source.to_string());
        assert!(
            !diagnostics.iter().any(|d| d.severity == Severity::Error),
            "expected-type propagation should accept this, got {diagnostics:?}\n{source}"
        );
    }
}

/// The three negatives that keep the rule above from becoming an implicit-conversion hole. Each
/// is a DIFFERENT reason, and all three must keep failing.
#[test]
fn integer_literal_typing_negatives_still_fail() {
    // Out of the expected type's range — rejected at compile time, not truncated.
    let out_of_range = "fn takes_u8(n: UInt8) -> UInt8 { n }\n\
                        fn main() -> Unit { println(takes_u8(300)); }\n";
    let diagnostics = analyze("lit_range.stark", out_of_range.to_string());
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0008")),
        "a literal outside the expected type's range must be rejected, got {diagnostics:?}"
    );

    // An explicit suffix is authoritative: Core has no implicit numeric conversion.
    let suffixed = "fn takes_u64(n: UInt64) -> UInt64 { n }\n\
                    fn main() -> Unit { println(takes_u64(0i32)); }\n";
    let diagnostics = analyze("lit_suffix.stark", suffixed.to_string());
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0001")),
        "a suffixed literal must not adopt a different type, got {diagnostics:?}"
    );

    // A TYPED VALUE is never converted — only the literal itself is retyped.
    let typed_value = "fn takes_u64(n: UInt64) -> UInt64 { n }\n\
                       fn main() -> Unit { let x: Int32 = 5; println(takes_u64(x)); }\n";
    let diagnostics = analyze("lit_typed.stark", typed_value.to_string());
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0001")),
        "an Int32 value must not pass as UInt64, got {diagnostics:?}"
    );

    // An integer literal is integer-KINDED: it may not stand in for a non-integer type.
    let non_integer = "fn takes_bool(b: Bool) -> Bool { b }\n\
                       fn main() -> Unit { println(takes_bool(1)); }\n";
    let diagnostics = analyze("lit_kind.stark", non_integer.to_string());
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0001")),
        "an integer literal must not satisfy a Bool parameter, got {diagnostics:?}"
    );
}

/// DEV-075 (owner specification decision; PRIM-TRAIT-001): `Bool` implements `Eq` and `Hash` but
/// NOT `Ord`. Its ordered operators and `Bool::cmp` are compile-time errors — an ordering could be
/// defined, but Core v1 has no use for ordering truth values, and rejecting is clearer than
/// fixing an arbitrary order. Previously the checker ACCEPTED `false < true` and then both engines
/// failed at runtime, which is strictly worse than a diagnostic.
#[test]
fn bool_is_not_ordered() {
    for source in [
        "fn main() -> Unit { if false < true { println(1); } else { println(0); } }\n",
        "fn main() -> Unit { if false <= true { println(1); } else { println(0); } }\n",
        "fn main() -> Unit { if false > true { println(1); } else { println(0); } }\n",
        "fn main() -> Unit { if false >= true { println(1); } else { println(0); } }\n",
    ] {
        let diagnostics = analyze("bool_ord.stark", source.to_string());
        assert!(
            diagnostics.iter().any(|d| d.severity == Severity::Error),
            "ordered operators on Bool must be rejected, got {diagnostics:?}\n{source}"
        );
    }

    // `Bool::cmp` likewise.
    let cmp = "fn main() -> Unit { let o = true.cmp(&false); }\n";
    let diagnostics = analyze("bool_cmp.stark", cmp.to_string());
    assert!(
        diagnostics.iter().any(|d| d.severity == Severity::Error),
        "Bool::cmp must be rejected, got {diagnostics:?}"
    );

    // But equality stays valid — `Bool` IS `Eq`.
    let eq = "fn main() -> Unit { if true == true { println(1); } else { println(0); } }\n";
    let diagnostics = analyze("bool_eq.stark", eq.to_string());
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "Bool equality must stay valid, got {diagnostics:?}"
    );
}

/// PRIM-TRAIT-001's float row: the comparison OPERATORS remain available on primitive floats as
/// built-in IEEE operations, while the `Eq`/`Ord` TRAITS are withheld (IEEE comparison is neither
/// an equivalence relation nor a total order). The two questions are distinct for primitives, and
/// conflating them would silently break ordinary float comparison — which it did, once, while
/// this was being implemented.
#[test]
fn floats_compare_but_do_not_satisfy_ord_bounds() {
    let operators = "fn main() -> Unit {\n\
        if 1.5 < 2.5 { println(1); } else { println(0); }\n\
        if 1.5 == 1.5 { println(1); } else { println(0); }\n\
    }\n";
    let diagnostics = analyze("float_ops.stark", operators.to_string());
    assert!(
        !diagnostics.iter().any(|d| d.severity == Severity::Error),
        "built-in float comparison must stay available, got {diagnostics:?}"
    );

    let bound = "fn smallest<T: Ord>(a: T, b: T) -> T { if a < b { a } else { b } }\n\
                 fn main() -> Unit { println(smallest(1.5, 2.5)); }\n";
    let diagnostics = analyze("float_bound.stark", bound.to_string());
    assert!(
        diagnostics
            .iter()
            .any(|d| d.code.as_deref() == Some("E0500")),
        "Float64 must not satisfy a T: Ord bound, got {diagnostics:?}"
    );
}
