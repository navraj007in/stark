//! WP-C6.1f "aggregates" — borrow-carrying tuples and arrays.
//!
//! OWN-CARRY-001 makes borrow provenance **structural**: it flows through tuples, generic
//! arguments and enum payloads. So a tuple or array of references is ordinary Core v1, not an
//! escape hatch. (Declared reference *fields* remain forbidden — 03 rule 1 — and the front end
//! rejects them as E0001; `a_declared_reference_field_is_still_rejected` pins that.)
//!
//! What made these emit is one observation: the property that matters is **carries a borrow**, not
//! **is a reference**. A `Copy` aggregate of references (`(&T, &T)`, `[&T; N]`) is not slot-backed,
//! so it would be declared via `default_value_expr` — which cannot fabricate a reference, one level
//! down for exactly the reason it cannot fabricate one directly. Such locals are therefore
//! initialisation-deferred like a bare reference: `Option<T> = None` when they must cross basic
//! blocks, bare-uninitialised when same-block.
//!
//! **Still refused, deliberately and before rustc:** a borrow-carrying *nominal* — `Option<&T>`, or
//! a user generic at a reference. A generated Rust struct/enum has no lifetime parameters, so a
//! reference in a field cannot be spelled and rustc would say `E0106`. Refusing it as a named STARK
//! limitation instead is the whole point of the pre-rustc boundary; see
//! `native_c5_3_aggregates_enums.rs`. Tuples and arrays need no such thing — they are structural
//! Rust types whose lifetimes rustc infers, which is exactly why they work and nominals do not.

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

fn agree(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(format!("agg_{tag}.stark"), src.to_string()));
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
    let hir_exec = interp::run_with_partial_output(&hir, file.clone(), &checked.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir_exec.status, 0, "{tag}: HIR must exit 0");
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir_exec = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir_exec.status, 0, "{tag}: MIR must exit 0");
    if !rustc_available() {
        return;
    }
    let verified = verify_program(&program).unwrap();
    let dir = std::env::temp_dir().join(format!("aggt_{tag}_{}", std::process::id()));
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
    assert!(
        run.status.success(),
        "{tag}: native must exit 0; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

const P: &str = "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c61f_tuple_of_references() {
    agree(
        "two_refs",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; \
         let t: (&Int32, &Int32) = (&x, &y); assert_eq(*t.0 + *t.1, 3); }",
    );
    agree(
        "struct_refs",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let q = P {{ v: 4 }}; \
                  let t: (&P, &P) = (&p, &q); assert_eq(t.0.get() + t.1.get(), 7); }}"
        ),
    );
    // Mixed: only one element carries a borrow.
    agree(
        "one_ref",
        "fn main() { let x: Int32 = 1; let t: (&Int32, Int32) = (&x, 5); \
         assert_eq(*t.0 + t.1, 6); }",
    );
}

#[test]
fn c61f_array_of_references() {
    agree(
        "array",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; let a: [&Int32; 2] = [&x, &y]; \
         assert_eq(*a[0] + *a[1], 3); }",
    );
}

#[test]
fn c61f_nested_borrow_carrying_tuple() {
    agree(
        "nested",
        "fn main() { let x: Int32 = 1; let y: Int32 = 2; \
         let t: ((&Int32, &Int32), Int32) = ((&x, &y), 9); assert_eq(*(t.0).0 + t.1, 10); }",
    );
}

#[test]
fn c61f_borrow_carrying_tuple_crosses_basic_blocks() {
    // The Option-backed path: written in one dispatch-loop arm, read in another.
    agree(
        "across_block",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let q = P {{ v: 4 }}; \
                  let t: (&P, &P) = (&p, &q); \
                  let n = if t.0.get() > 1 {{ t.1.get() }} else {{ 0 }}; assert_eq(n, 4); }}"
        ),
    );
}

#[test]
fn c61f_tuple_of_references_to_drop_bearing_values() {
    // Borrowing does not disturb the owners' destructors.
    agree(
        "drop_refs",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\n\
         fn main() { let d = D { v: 1 }; let e = D { v: 2 }; \
         let t: (&D, &D) = (&d, &e); assert_eq(t.0.v + t.1.v, 3); }",
    );
}

#[test]
fn a_declared_reference_field_is_still_rejected() {
    // 03 rule 1: struct/enum/tuple-struct declarations MUST NOT write a reference field type.
    // Supporting borrow-carrying tuples must not have opened this.
    let src = "struct H { r: &Int32 }\nfn main() { let x: Int32 = 1; let h = H { r: &x }; }";
    let file = Arc::new(SourceFile::new("declared_ref_field.stark", src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    let (hir, rd) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file);
    let rejected = !pd.is_empty()
        || !rd.is_empty()
        || checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error);
    assert!(
        rejected,
        "a declared reference field must be rejected (03 rule 1)"
    );
}
