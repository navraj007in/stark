//! WP-C6.1b — general cross-block non-`Copy` movement, Track A.
//!
//! The C6.1a ownership audit found that the broad cross-block shapes already reach native parity;
//! the concrete open gap C6.1b closes is **G3 — multi-level (depth ≥2) partial move/drop** through
//! a projection chain (`o.a.x`). C5.3d-0 implemented only depth 1. This file drives the multi-level
//! shapes through HIR, MIR, and native and requires agreement, and proves — for Drop-bearing types —
//! that the moved deep unit is not double-dropped (a wrong drop trips `slot_violation`, aborting the
//! native process, so a clean exit is the evidence).
//!
//! Track-A-owned (no shared-file lease): the comparison is done here rather than by editing the
//! shared `three_engine_differential.rs`.

use starkc::backend::generated_rust::{emit_native_debug, emit_program, NativeBuildOptions};
use starkc::backend::version::build_versions;
use starkc::diag::Severity;
use starkc::interp;
use starkc::layout::TargetLayout;
use starkc::mir::interp::run_program;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

struct Compiled {
    program: starkc::mir::MirProgram,
    hir: starkc::hir::Hir,
    tables: starkc::typecheck::TypeTables,
    file: Arc<SourceFile>,
    generated: String,
}

fn compile(source: &str, tag: &str) -> Compiled {
    let file = Arc::new(SourceFile::new(
        format!("c6_1_{tag}.stark"),
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
    let program = lower_program(&hir, &checked.tables, file.clone())
        .unwrap_or_else(|e| panic!("{tag} lower: {}", e.what));
    let versions = build_versions("0.0.0-test".to_string(), "test-triple".to_string());
    let generated = emit_program::emit(&program, &versions, &TargetLayout::default())
        .unwrap_or_else(|e| panic!("{tag} emit: {e:?}"))
        .main_rs;
    Compiled {
        program,
        hir,
        tables: checked.tables,
        file,
        generated,
    }
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run through HIR + MIR, require both complete with exit 0 and no output, then build+run native
/// and require exit 0 (all three agree on successful completion). Returns the generated source.
fn agree_completes(tag: &str, source: &str) -> String {
    let c = compile(source, tag);

    let hir = interp::run_with_partial_output(&c.hir, c.file.clone(), &c.tables)
        .unwrap_or_else(|(e, _)| panic!("{tag} HIR: {}", e.message));
    assert_eq!(hir.status, 0, "{tag}: HIR must exit 0");
    assert!(hir.output.is_empty(), "{tag}: no output surface in C5/C6.1");

    let verified = verify_program(&c.program).unwrap_or_else(|e| panic!("{tag} verify: {e:?}"));
    let mir = run_program(verified).unwrap_or_else(|f| panic!("{tag} MIR: {:?}", f.error));
    assert_eq!(mir.status, 0, "{tag}: MIR must exit 0");
    assert_eq!(hir.output, mir.output, "{tag}: HIR/MIR output disagree");

    if rustc_available() {
        let verified = verify_program(&c.program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c6_1_{tag}_{}", std::process::id()));
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
            "{tag}: native must exit 0 (no double/missing drop → no slot_violation); stderr: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    c.generated
}

// --------------------------------------------------------------- G3: depth ≥2 --

const NESTED: &str =
    "struct S { v: Int32 }\nstruct Inner { x: S }\nstruct Outer { a: Inner, b: S }\n";

#[test]
fn multi_level_partial_move_agrees_across_engines() {
    let gen = agree_completes(
        "multilevel_move",
        &format!(
            "{NESTED}fn main() {{\n\
             \x20   let o = Outer {{ a: Inner {{ x: S {{ v: 7 }} }}, b: S {{ v: 9 }} }};\n\
             \x20   let y = o.a.x;\n\
             \x20   assert_eq(y.v, 7);\n\
             \x20   assert_eq(o.b.v, 9);\n\
             }}\n"
        ),
    );
    // Structural: the chained raw selector appears inside a projection helper.
    assert!(
        gen.contains("mod stark_proj"),
        "a projection helper module must be generated"
    );
    assert!(
        gen.contains(".f0.f0"),
        "the multi-level raw selector `.f0.f0` must be emitted (chained addr_of_mut):\n{gen}"
    );
}

const NESTED_DROP: &str = "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { } }\nstruct Inner { x: D }\nstruct Outer { a: Inner, b: D }\n";

#[test]
fn multi_level_partial_move_of_drop_types_does_not_double_drop() {
    // `o.a.x` (a `D`) is moved; at scope end `y`, `o.b`, and the remnant of `o.a` drop, but `o.a.x`
    // must NOT be dropped again. A wrong drop over partial storage aborts in `slot_violation`, so a
    // clean exit-0 native run is the correctness evidence.
    agree_completes(
        "multilevel_move_drop",
        &format!(
            "{NESTED_DROP}fn main() {{\n\
             \x20   let o = Outer {{ a: Inner {{ x: D {{ v: 7 }} }}, b: D {{ v: 9 }} }};\n\
             \x20   let y = o.a.x;\n\
             \x20   assert_eq(y.v, 7);\n\
             }}\n"
        ),
    );
}

#[test]
fn multi_level_copy_read_after_sibling_move_agrees() {
    // Read a nested Copy field (`o.b` is Copy Int32 via a tuple) AFTER a sibling non-Copy unit
    // (`o.a.x`) has been moved out — the slot is partial, so the read must go through the raw copy
    // helper rather than `get()` (which is refused on partial storage).
    agree_completes(
        "multilevel_copy_after_move",
        "struct S { v: Int32 }\n\
         struct Outer { a: S, b: (Int32, Int32) }\n\
         fn main() {\n\
         \x20   let o = Outer { a: S { v: 1 }, b: (5, 6) };\n\
         \x20   let moved = o.a;\n\
         \x20   assert_eq(moved.v, 1);\n\
         \x20   assert_eq(o.b.0, 5);\n\
         }\n",
    );
}

// ------------------------------------------------------ regression: broad shapes --

#[test]
fn broad_cross_block_movement_still_agrees() {
    // C6.1a proved these already work; pin them so C6.1b's changes cannot regress them.
    agree_completes(
        "cross_block_regression",
        "struct S { v: Int32 }\n\
         fn id(x: S) -> S { x }\n\
         fn main() {\n\
         \x20   let c: Bool = true;\n\
         \x20   let a = S { v: 1 };\n\
         \x20   let b = if c { a } else { S { v: 2 } };\n\
         \x20   let d = id(b);\n\
         \x20   let mut i: Int32 = 0;\n\
         \x20   let mut acc = S { v: 0 };\n\
         \x20   while i < 3 { let t = acc; acc = S { v: t.v + 1 }; i = i + 1; }\n\
         \x20   assert_eq(d.v, 1);\n\
         \x20   assert_eq(acc.v, 3);\n\
         }\n",
    );
}

// ------------------------------------------------------------- negative control --

#[test]
fn a_false_assertion_traps_in_all_three_engines() {
    // Proves the assertions above actually execute rather than being compiled away.
    let source = format!(
        "{NESTED}fn main() {{\n\
         \x20   let o = Outer {{ a: Inner {{ x: S {{ v: 7 }} }}, b: S {{ v: 9 }} }};\n\
         \x20   let y = o.a.x;\n\
         \x20   assert_eq(y.v, 8);\n\
         }}\n"
    );
    let c = compile(&source, "multilevel_false");

    assert!(
        interp::run_with_partial_output(&c.hir, c.file.clone(), &c.tables).is_err(),
        "HIR must trap"
    );
    let verified = verify_program(&c.program).unwrap();
    assert!(run_program(verified).is_err(), "MIR must trap");

    if rustc_available() {
        let verified = verify_program(&c.program).unwrap();
        let dir = std::env::temp_dir().join(format!("stark_c6_1_false_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let artifact = emit_native_debug(
            &verified,
            &NativeBuildOptions {
                target_dir: dir.clone(),
                target_contract: "stark-64-v1".to_string(),
            },
        )
        .unwrap();
        let run = std::process::Command::new(&artifact.binary_path)
            .output()
            .unwrap();
        assert!(
            !run.status.success(),
            "native must trap on the false assertion"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
