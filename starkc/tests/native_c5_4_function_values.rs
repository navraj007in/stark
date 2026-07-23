//! WP-C5.4c — function-value representation: unit and generated-source structural proofs.
//!
//! Value AGREEMENT across HIR/MIR/native lives in `three_engine_differential.rs` (the `fnval_*`
//! cases). This file covers what a value comparison structurally cannot see (§13.2):
//!   * `MirTy::FnPtr` maps to a typed Rust `fn(..) -> ..` pointer, and an unsupported signature is
//!     refused before rustc (§7.2/§5.2);
//!   * `FnPtr` is MIR-authorised `Copy` and its default local value is the ABORTING sentinel, never
//!     an arbitrary function (§7.3/§7.4/§7.6, mutation guard §13.5 #6);
//!   * exactly one sentinel per distinct signature (§7.5);
//!   * no `dyn Fn`, closure, `Box`, raw-address cast, or function registry (§7.1, guard §13.5 #7);
//!   * a function reached ONLY through a value is emitted once and never direct-called (§10.5);
//!   * the entry `main` as a function value is source-reachable AND coherent through a real native
//!     build (§8.3).

use starkc::backend::generated_rust::{
    emit_native_debug, emit_program, emit_types, mangle, NativeBuildOptions,
};
use starkc::backend::version::build_versions;
use starkc::diag::Severity;
use starkc::layout::TargetLayout;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::mir::{MirProgram, MirTy, TypeContext};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn compile(source: &str, tag: &str) -> (MirProgram, String) {
    let file = Arc::new(SourceFile::new(
        format!("c5_4c_{tag}.stark"),
        source.to_string(),
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
    let program = lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let _verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let versions = build_versions("0.0.0-test".to_string(), "test-triple".to_string());
    let generated = emit_program::emit(&program, &versions, &TargetLayout::default())
        .unwrap_or_else(|e| panic!("{tag} emit: {e:?}"))
        .main_rs;
    (program, generated)
}

fn count(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

fn fnptr(params: Vec<MirTy>, ret: MirTy) -> MirTy {
    MirTy::FnPtr {
        params,
        ret: Box::new(ret),
    }
}

// ------------------------------------------------------------------- unit tests --

#[test]
fn fnptr_type_emits_as_a_rust_function_pointer() {
    let e = |ty: &MirTy| emit_types::emit_ty(ty).unwrap();
    assert_eq!(
        e(&fnptr(vec![MirTy::Int32], MirTy::Int32)),
        "fn(i32) -> i32"
    );
    assert_eq!(e(&fnptr(vec![], MirTy::Unit)), "fn() -> ()");
    assert_eq!(
        e(&fnptr(vec![MirTy::Int32, MirTy::Bool], MirTy::Int64)),
        "fn(i32, bool) -> i64"
    );
    // A function pointer whose parameter is itself a function pointer.
    assert_eq!(
        e(&fnptr(
            vec![fnptr(vec![MirTy::Int32], MirTy::Int32)],
            MirTy::Int32
        )),
        "fn(fn(i32) -> i32) -> i32"
    );
    // Distinct signatures produce distinct Rust type text.
    assert_ne!(
        e(&fnptr(vec![MirTy::Int32], MirTy::Int32)),
        e(&fnptr(vec![MirTy::Int64], MirTy::Int32))
    );
}

#[test]
fn an_unsupported_signature_is_refused_before_rustc() {
    // §5.2/§7.2: a signature containing a type with no C5 representation must produce
    // `Unsupported` — the deterministic pre-rustc boundary, not a rustc error.
    let bad = fnptr(vec![MirTy::String], MirTy::Int32);
    assert!(
        emit_types::emit_ty(&bad).is_err(),
        "a FnPtr over an unsupported type must be Unsupported"
    );
}

#[test]
fn fnptr_is_copy_per_type_fn_001() {
    // §7.3: `FnPtr` is Copy because MIR says so (TYPE-FN-001), read through the shared
    // `TypeContext::is_copy` authority — the backend must not independently ask Rust.
    let tc = TypeContext::default();
    assert!(tc.is_copy(&fnptr(vec![MirTy::Int32], MirTy::Int32)));
    assert!(tc.is_copy(&fnptr(vec![], MirTy::Unit)));
    // §7.3: a Copy type is never slot-backed, so no `ValueSlot` is introduced for a `FnPtr` local.
    assert!(!emit_types::is_slot_backed(
        &fnptr(vec![MirTy::Int32], MirTy::Int32),
        &tc
    ));
}

#[test]
fn the_default_value_for_a_fnptr_is_the_aborting_sentinel() {
    // §7.6 + mutation guard §13.5 #6: the default is the sentinel name, NOT 0/null/transmute or a
    // non-aborting arbitrary function.
    let ty = fnptr(vec![MirTy::Int32], MirTy::Int32);
    let default =
        emit_types::default_value_expr(&ty, &TypeContext::default()).expect("FnPtr has a default");
    assert_eq!(default, mangle::fn_sentinel_name(&ty));
    assert!(default.contains("sentinel"), "{default}");
    assert_ne!(default, "0");
}

// -------------------------------------------------------------- structural tests --

#[test]
fn exactly_one_sentinel_per_distinct_signature() {
    // Two DISTINCT signatures → two sentinels.
    let (_p, two) = compile(
        r#"fn a(x: Int32) -> Int32 { x }
fn b(x: Bool) -> Bool { x }
fn main() {
    let f: fn(Int32) -> Int32 = a;
    let g: fn(Bool) -> Bool = b;
    assert_eq(f(1), 1);
    assert(g(true));
}
"#,
        "two_sigs",
    );
    assert_eq!(
        count(&two, "aborting sentinel for"),
        2,
        "one sentinel per distinct signature"
    );

    // TWO function values of the SAME signature → ONE sentinel (per-signature dedup, §7.5).
    let (_p, one) = compile(
        r#"fn a(x: Int32) -> Int32 { x }
fn b(x: Int32) -> Int32 { x + 1 }
fn main() {
    let f: fn(Int32) -> Int32 = a;
    let g: fn(Int32) -> Int32 = b;
    assert_eq(f(1) + g(1), 3);
}
"#,
        "one_sig",
    );
    assert_eq!(
        count(&one, "aborting sentinel for"),
        1,
        "identical signatures share one sentinel"
    );
}

#[test]
fn function_values_use_no_closure_trait_object_box_or_address() {
    let (_p, gen) = compile(
        r#"fn add_one(x: Int32) -> Int32 { x + 1 }
fn apply(f: fn(Int32) -> Int32, v: Int32) -> Int32 { f(v) }
fn main() {
    let f: fn(Int32) -> Int32 = add_one;
    assert_eq(apply(f, 41), 42);
}
"#,
        "no_erasure",
    );
    // §7.1: a function value is a bare typed pointer. None of the erasure/allocation mechanisms.
    for banned in [
        "dyn Fn",
        "dyn ",
        "Box<",
        "Box::",
        "|| ",
        "as usize",
        "as *const",
        "as *mut",
    ] {
        assert!(
            !gen.contains(banned),
            "generated source must not use `{banned}` for a function value:\n{gen}"
        );
    }
    // The declared parameter type is a Rust function pointer.
    assert!(
        gen.contains("fn(i32) -> i32"),
        "the FnPtr parameter should emit as a Rust function pointer type"
    );
}

#[test]
fn a_function_reached_only_as_a_value_is_defined_once_and_never_direct_called() {
    // §10.5 + mutation guard §13.5 #7: `only` is never a direct callee, only a function value.
    let (program, gen) = compile(
        r#"fn only(x: Int32) -> Int32 { x * 3 }
fn main() {
    let f: fn(Int32) -> Int32 = only;
    assert_eq(f(4), 12);
}
"#,
        "only_value",
    );
    let sym = program
        .bodies
        .iter()
        .find(|b| b.instance.symbol.contains("only"))
        .expect("the only-via-value instance must be emitted")
        .instance
        .symbol
        .clone();
    let name = mangle::function_name_for_symbol(&sym);
    // Exactly one occurrence of `name(` — the DEFINITION `fn {name}(`. A direct call would add
    // another `{name}(`; a function-value reference is `= {name};` / `(_n)(..)`, neither of which
    // contains `{name}(`.
    assert_eq!(
        count(&gen, &format!("{name}(")),
        1,
        "the target must be defined once and never direct-called"
    );
    // It IS referenced as a value (definition + assignment), so it appears at least twice overall.
    assert!(
        count(&gen, &name) >= 2,
        "the target must be referenced as a value"
    );
}

#[test]
fn the_entry_main_used_as_a_function_value_builds_natively() {
    // §8.3: `let f = main;` is valid STARK source. The probe confirmed the front end accepts it;
    // this proves the generated reference coheres with the Rust `main` wrapper through a REAL
    // native build (guarding §15.5: accepted source must not fail only in rustc).
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let (_program, gen) = compile(
        "fn main() { let f: fn() -> Unit = main; }",
        "entry_value_src",
    );
    assert!(
        gen.contains("= main"),
        "the entry function value must reference Rust `main`"
    );

    // Full build: parse → … → emit_native_debug (which runs cargo/rustc).
    let file = Arc::new(SourceFile::new(
        "entry_value.stark".to_string(),
        "fn main() { let f: fn() -> Unit = main; }".to_string(),
    ));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let program =
        lower_program(&hir, &checked.tables, file).unwrap_or_else(|e| panic!("{}", e.what));
    let verified = verify_program(&program).unwrap();
    let dir = std::env::temp_dir().join(format!("stark_c5_4c_entry_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: dir.clone(),
            target_contract: "stark-64-v1".to_string(),
        },
    )
    .expect("entry-as-value must build natively (§8.3 coherence, §15.5)");
    // It builds and (referencing but not calling `main`) exits normally.
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run failed");
    assert!(run.status.success(), "entry-as-value program must exit 0");
    let _ = std::fs::remove_dir_all(&dir);
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
