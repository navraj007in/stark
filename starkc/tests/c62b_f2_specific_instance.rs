//! WP-C6.2b-F2 — a trait/inherent impl on a SPECIFIC generic instantiation matches an inferred
//! receiver.
//!
//! `impl Get for W<Int32>` was not matched for `let w = W { v: 7 }; w.get()` (E0302, receiver typed
//! `W<_infer>`). It was NOT that specific-instance impls are unsupported — `let w: W<Int32> = ...`
//! already worked — but that the receiver's int-literal argument (`7`) was not defaulted to `Int32`
//! before method resolution. `default_int_literals_deep` now defaults literals INSIDE the receiver
//! type (03 solving step 5), so `W<_infer>` becomes `W<Int32>` and the concrete-instance impl
//! matches. A wrong instance (`W<Bool>`) still has no matching impl and stays rejected.

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
    let file = Arc::new(SourceFile::new(format!("f2_{tag}.stark"), src.to_string()));
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
    let dir = std::env::temp_dir().join(format!("f2_{tag}_{}", std::process::id()));
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
        format!("f2_neg_{tag}.stark"),
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

const GET: &str = "trait Get { fn get(&self) -> Int32; }\nstruct W<T> { v: T }\n\
                   impl Get for W<Int32> { fn get(&self) -> Int32 { self.v } }\n";

#[test]
fn c62b_f2_specific_trait_impl_matches_inferred_receiver() {
    agree(
        "inferred",
        &format!("{GET}fn main() {{ let w = W {{ v: 7 }}; assert_eq(w.get(), 7); }}"),
    );
}

#[test]
fn c62b_f2_specific_trait_impl_matches_annotated_receiver() {
    agree(
        "annotated",
        &format!("{GET}fn main() {{ let w: W<Int32> = W {{ v: 7 }}; assert_eq(w.get(), 7); }}"),
    );
}

#[test]
fn c62b_f2_specific_inherent_impl_matches_inferred_receiver() {
    agree(
        "inherent",
        "struct W<T> { v: T }\nimpl W<Int32> { fn get(&self) -> Int32 { self.v } }\n\
         fn main() { let w = W { v: 7 }; assert_eq(w.get(), 7); }",
    );
}

#[test]
fn c62b_f2_nested_instance_argument_is_defaulted() {
    // The literal is one level deeper: W<Pair<Int32>>-ish through a nested field.
    agree(
        "nested",
        "trait Get { fn get(&self) -> Int32; }\nstruct Inner { v: Int32 }\nstruct W<T> { i: T }\n\
         impl Get for W<Inner> { fn get(&self) -> Int32 { self.i.v } }\n\
         fn main() { let w = W { i: Inner { v: 7 } }; assert_eq(w.get(), 7); }",
    );
}

#[test]
fn c62b_f2_a_different_instance_stays_rejected() {
    // No `impl Get for W<Bool>`, so the call must not resolve (no over-accept from the fix).
    rejected(
        "wrong_instance",
        &format!("{GET}fn main() {{ let w: W<Bool> = W {{ v: true }}; let _ = w.get(); }}"),
    );
}
