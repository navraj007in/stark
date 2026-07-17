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
