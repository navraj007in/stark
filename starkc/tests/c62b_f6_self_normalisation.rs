//! WP-C6.2b-F6 — an impl method signature may spell the concrete type where the trait writes `Self`.
//!
//! `impl Mk for G { fn make() -> G { ... } }` for `trait Mk { fn make() -> Self; }` was rejected
//! E0500 "signature incompatible", because the compatibility check keyed `Self` (in the trait) and
//! the concrete `G` (in the impl) to different strings. `Self` in an `impl … for G` IS `G`, so the
//! two spellings are equivalent. `typecheck` now keys the impl's self type in the same format a
//! path produces (`ty_signature_key`) and returns that for a `Self` mention, so `Self` and the
//! written self type compare equal. A DIFFERENT concrete type (`-> H`) still mismatches and is
//! rejected -- no over-accept.

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
    let file = Arc::new(SourceFile::new(format!("f6_{tag}.stark"), src.to_string()));
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
    let dir = std::env::temp_dir().join(format!("f6_{tag}_{}", std::process::id()));
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

fn rejected(tag: &str, src: &str) {
    let file = Arc::new(SourceFile::new(
        format!("f6_neg_{tag}.stark"),
        src.to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file);
    assert!(
        checked
            .diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error),
        "{tag}: must stay rejected"
    );
}

#[test]
fn c62b_f6_return_self_written_as_the_concrete_type() {
    agree(
        "ret_concrete",
        "trait Mk { fn make() -> Self; fn val(&self) -> Int32; }\nstruct G { v: Int32 }\n\
         impl Mk for G { fn make() -> G { G { v: 8 } } fn val(&self) -> Int32 { self.v } }\n\
         fn main() { assert_eq(G::make().val(), 8); }",
    );
}

#[test]
fn c62b_f6_return_self_written_as_self_still_works() {
    agree(
        "ret_self",
        "trait Mk { fn make() -> Self; fn val(&self) -> Int32; }\nstruct G { v: Int32 }\n\
         impl Mk for G { fn make() -> Self { G { v: 8 } } fn val(&self) -> Int32 { self.v } }\n\
         fn main() { assert_eq(G::make().val(), 8); }",
    );
}

#[test]
fn c62b_f6_param_self_written_as_the_concrete_type() {
    agree(
        "param_concrete",
        "trait Eqx { fn same(&self, o: &Self) -> Int32; }\nstruct G { v: Int32 }\n\
         impl Eqx for G { fn same(&self, o: &G) -> Int32 { self.v + o.v } }\n\
         fn main() { let a = G { v: 1 }; let b = G { v: 2 }; assert_eq(a.same(&b), 3); }",
    );
}

#[test]
fn c62b_f6_generic_self_type_normalises() {
    // The impl self type carries arguments; a `&Self` parameter written as `&W<Int32>` must agree
    // (exercises Self-normalisation on a generic instance without an associated-fn call).
    agree(
        "generic_self",
        "trait Combine { fn merge(&self, o: &Self) -> Int32; }\nstruct W<T> { v: T }\n\
         impl Combine for W<Int32> { fn merge(&self, o: &W<Int32>) -> Int32 { self.v + o.v } }\n\
         fn main() { let a: W<Int32> = W { v: 1 }; let b: W<Int32> = W { v: 2 }; \
         assert_eq(a.merge(&b), 3); }",
    );
}

#[test]
fn c62b_f6_a_different_concrete_type_stays_rejected() {
    // `-> H` where Self is G must still mismatch -- no over-accept from the normalisation.
    rejected(
        "wrong_concrete",
        "trait Mk { fn make() -> Self; }\nstruct G { v: Int32 }\nstruct H { v: Int32 }\n\
         impl Mk for G { fn make() -> H { H { v: 8 } } }\nfn main() {}",
    );
}
