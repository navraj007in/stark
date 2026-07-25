//! WP-C6.3b — native Vec and Box (value surface).
//!
//! `stark_runtime::vec` and `stark_runtime::boxed` pin the STARK semantics for the owning
//! containers; a `Vec<T>`/`Box<T>` is the Rust container, non-`Copy`, slot-backed. This slice lands
//! the value operations that neither trap nor hand out an interior reference: `Vec::new`/
//! `with_capacity`/`push`/`pop`/`len`/`is_empty`/`clear`, and `Box::new`/`into_inner`. `pop` reuses
//! the Option-return bridge (CD-110).
//!
//! Buffer reclamation: `ValueSlot::drop_with` now runs `ManuallyDrop::drop` after the MIR glue, so
//! Rust's structural drop reclaims the container's buffer and its elements (recursive-safe) while
//! the glue runs only user destructors — fixing a latent leak of every owning value in a slot.
//!
//! Deferred:
//!   * A `Vec`/`Box` whose element carries a USER destructor — refused pre-rustc (running a
//!     destructor at every, possibly recursive, element is the destructor-in-runtime-collection
//!     design, a later slice).
//!   * `v.push(f(...))` where the pushed value is itself a runtime call (e.g. `Vec<String>`): the
//!     `&mut Vec` receiver borrow is held across the argument-evaluation block — the WP-C6.1g-c
//!     dispatch-loop borrow problem. HIR+MIR pass; native is deferred with the other C6.1g-c cases.
//!   * Trapping index/replace/remove, interior-ref `get`/iteration, and slices — later slices.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::diag::Severity;
use starkc::interp;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct Front {
    hir: starkc::hir::Hir,
    tables: starkc::typecheck::TypeTables,
    file: Arc<SourceFile>,
}

fn front(tag: &str, src: &str) -> Front {
    let file = Arc::new(SourceFile::new(
        format!("c63b_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag} parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag} resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag} typecheck: {errs:?}");
    Front {
        hir,
        tables: checked.tables,
        file,
    }
}

/// HIR + MIR + native must all exit 0.
fn agree(tag: &str, src: &str) {
    let f = front(tag, src);

    let hir_exec = interp::run_with_partial_output(&f.hir, f.file.clone(), &f.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");

    let program = lower_program(&f.hir, &f.tables, f.file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");

    if rustc_available() {
        let verified = verify_program(&program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c63b_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap_or_else(|e| panic!("{tag} native build: {e:?}"));
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        assert!(run.status.success(), "{tag}: native must exit 0");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn vec_new_push_len() {
    agree(
        "new_push_len",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.push(3); assert_eq(v.len(), 3); }",
    );
}

#[test]
fn vec_new_is_empty() {
    agree(
        "empty",
        "fn main() { let v: Vec<Int32> = Vec::new(); assert_eq(v.is_empty(), true); }",
    );
}

#[test]
fn vec_pop_returns_option() {
    agree(
        "pop",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(7); let x = v.pop(); assert_eq(x.unwrap_or(0), 7); }",
    );
}

#[test]
fn vec_pop_none_when_empty() {
    agree(
        "pop_none",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); let x = v.pop(); assert_eq(x.is_some(), false); }",
    );
}

#[test]
fn vec_clear_empties() {
    agree(
        "clear",
        "fn main() { let mut v: Vec<Int32> = Vec::new(); v.push(1); v.push(2); v.clear(); assert_eq(v.is_empty(), true); }",
    );
}

#[test]
fn vec_returned_across_function() {
    agree(
        "ret",
        "fn mk() -> Vec<Int32> { let mut v = Vec::new(); v.push(9); v }\n\
         fn main() { let v = mk(); assert_eq(v.len(), 1); }",
    );
}

#[test]
fn box_new_and_into_inner() {
    agree(
        "box",
        "fn main() { let b = Box::new(42); let x = b.into_inner(); assert_eq(x, 42); }",
    );
}

/// A `Box` of an owning value: the buffer is reclaimed by the slot's structural drop.
#[test]
fn box_of_string() {
    agree(
        "box_str",
        "fn main() { let b = Box::new(String::from(\"x\")); let s = b.into_inner(); assert_eq(s.len(), 1); }",
    );
}

// ---- Formerly deferred to WP-C6.1g-c; native since CD-112 (dispatch-loop linearisation). ----

/// `v.push(String::from(..))` holds the `&mut Vec` receiver borrow across the argument's own
/// runtime call. Under CD-112's linearised emission rustc sees that borrow end before the drop, so
/// it now builds and runs natively.
#[test]
fn vec_of_string_push() {
    agree(
        "vec_str",
        "fn main() { let mut v: Vec<String> = Vec::new(); v.push(String::from(\"hi\")); assert_eq(v.len(), 1); }",
    );
}
