//! WP-C6.1f — borrow-carrying **nominals**: `Option<&T>`, and user generics instantiated at a
//! reference.
//!
//! A generated nominal is a *declared* Rust type, so unlike a tuple it cannot borrow implicitly:
//! a reference in one of its fields needs a lifetime parameter, or rustc reports `E0106`. Generated
//! nominals therefore carry one — `Name<'a>` in the declaration, `Name<'_>` at every use site.
//! The two spellings are not interchangeable: `'_` is illegal in a field type, which has no
//! enclosing binder to infer from, so declaration and use positions are rendered separately
//! (`emit_types::LifetimePosition`).
//!
//! **Two shapes remain refused, before rustc** — both are the `ValueSlot`-versus-Rust-borrow-region
//! tension the C6.1f-a matrix flagged as this package's central design question:
//!
//! 1. A **slot-backed** (non-`Copy`) borrow-carrying nominal — a user struct or enum at a
//!    reference. Its slot's destruction needs `&mut` while the reference it stores still borrows
//!    its referent's slot immutably; Rust treats those as overlapping for the local's whole lexical
//!    region (`E0502`) even though MIR drops the borrower first. Dropping the slot is not an
//!    escape: the slot also carries MOVE liveness, and without it the move fails instead.
//! 2. A **function returning** a borrow-carrying nominal: the elided output lifetime keeps the
//!    borrow live across the referent's own slot destruction.
//!
//! Both are refused as named STARK limitations rather than allowed to surface as errors in
//! generated code, which is the pre-rustc boundary this backend is built around.

use starkc::backend::generated_rust::{emit_native_debug, BackendDiagnostic, NativeBuildOptions};
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

/// Drive a program to a native binary, asserting HIR and MIR agree on the way.
fn build(tag: &str, src: &str) -> Result<std::process::Output, BackendDiagnostic> {
    let file = Arc::new(SourceFile::new(format!("nom_{tag}.stark"), src.to_string()));
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

    let verified = verify_program(&program).unwrap();
    let dir = std::env::temp_dir().join(format!("nomt_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: dir.clone(),
            target_contract: "stark-64-v1".to_string(),
        },
    )?;
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(run)
}

fn agree(tag: &str, src: &str) {
    if !rustc_available() {
        return;
    }
    let run = build(tag, src).unwrap_or_else(|e| panic!("{tag} native build: {e:?}"));
    assert!(
        run.status.success(),
        "{tag}: native must exit 0; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// The refusal must be OURS (`Unsupported`), never a rustc error in the generated crate.
fn refused_before_rustc(tag: &str, src: &str, expected: &str) {
    if !rustc_available() {
        return;
    }
    match build(tag, src) {
        Err(BackendDiagnostic::Unsupported(message)) => assert!(
            message.contains(expected),
            "{tag}: expected a refusal mentioning {expected:?}, got: {message}"
        ),
        Err(BackendDiagnostic::BuildFailed(f)) => panic!(
            "{tag}: reached rustc and failed THERE; the backend must refuse it first:\n{}",
            f.stderr
        ),
        Err(other) => panic!("{tag}: expected Unsupported, got {other:?}"),
        Ok(_) => panic!("{tag}: this shape is not supported yet and must be refused"),
    }
}

const P: &str = "struct P { v: Int32 }\nimpl P { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c61f_option_holding_a_reference() {
    agree(
        "some",
        "fn main() { let x: Int32 = 5; let o: Option<&Int32> = Some(&x); assert_eq(*o.unwrap(), 5); }",
    );
    agree(
        "none",
        "fn main() { let o: Option<&Int32> = None; assert(o.is_none()); }",
    );
}

#[test]
fn c61f_matching_on_an_option_holding_a_reference() {
    agree(
        "match",
        &format!(
            "{P}fn main() {{ let p = P {{ v: 3 }}; let o: Option<&P> = Some(&p); \
                  match o {{ Some(r) => assert_eq(r.get(), 3), None => assert(false) }} }}"
        ),
    );
}

#[test]
fn c61f_nested_and_embedded_borrow_carrying_nominals() {
    agree(
        "nested",
        "fn main() { let x: Int32 = 5; let o: Option<Option<&Int32>> = Some(Some(&x)); \
         assert(o.is_some()); }",
    );
    agree(
        "in_tuple",
        "fn main() { let x: Int32 = 5; let t: (Option<&Int32>, Int32) = (Some(&x), 1); \
         assert(t.0.is_some()); }",
    );
}

#[test]
fn c61f_a_nominal_without_a_borrow_is_unaffected() {
    // The lifetime parameter appears only when the instance actually carries a borrow.
    agree(
        "plain_option",
        "fn main() { let o: Option<Int32> = Some(5); assert_eq(o.unwrap(), 5); }",
    );
}

#[test]
fn c61f_a_slot_backed_borrow_carrying_nominal_is_refused_before_rustc() {
    refused_before_rustc(
        "generic_struct",
        &format!(
            "{P}struct H<T> {{ r: T }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let h: H<&P> = H {{ r: &p }}; \
                  assert_eq(h.r.get(), 3); }}"
        ),
        "slot-backed borrow-carrying nominal",
    );
    refused_before_rustc(
        "user_enum",
        &format!(
            "{P}enum E<T> {{ A, B(T) }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let e: E<&P> = E::B(&p); \
                  match e {{ E::A => assert(false), E::B(r) => assert_eq(r.get(), 3) }} }}"
        ),
        "slot-backed borrow-carrying nominal",
    );
}

#[test]
fn c61f_returning_a_borrow_carrying_nominal_is_refused_before_rustc() {
    refused_before_rustc(
        "return_option_ref",
        &format!(
            "{P}fn wrap(r: &P) -> Option<&P> {{ Some(r) }}\n\
                  fn main() {{ let p = P {{ v: 3 }}; let o = wrap(&p); \
                  assert_eq(o.unwrap().get(), 3); }}"
        ),
        "returning the borrow-carrying nominal",
    );
}
