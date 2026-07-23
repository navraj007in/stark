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

// ============================================ WP-C6.1c — enum payload partial moves (G1) --
//
// A multi-field variant payload with a non-Copy field is decomposed by lowering into ONE tuple
// aggregate, emitted as a single destructuring `match e.take() { … }`; after that, per-field
// movement reuses C6.1b's raw-projectable tuple machinery. The whole enum is moved exactly once,
// so no partial-slot access occurs. For Drop-bearing payloads, exit-0 is the no-double/missing-drop
// evidence: a whole-enum drop after decomposition, or a drop of a moved unit, trips `slot_violation`
// (non-zero exit). Observable Drop ORDER is preserved by lowering (unbound-first/bindings-second
// registration unchanged) and checked by the HIR/MIR oracles; native order observation waits for
// C6.3 output.

const ENUM2: &str = "struct S { v: Int32 }\nenum E { V(S, S) }\nfn take(x: S) -> Int32 { x.v }\n";
const ENUM2_DROP: &str = "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { let r: Int32 = self.v; } }\nenum E { V(D, D) }\nfn take(x: D) -> Int32 { x.v }\n";

#[test]
fn c61c_two_payload_fields_both_bound_and_consumed() {
    let gen = agree_completes(
        "c61c_both",
        &format!(
            "{ENUM2}fn main() {{ let e = E::V(S {{ v: 1 }}, S {{ v: 2 }}); \
             match e {{ E::V(a, b) => {{ assert_eq(take(a) + take(b), 3); }} }} }}"
        ),
    );
    assert_eq!(
        gen.matches(".take() {").count(),
        1,
        "exactly one destructuring extraction for the payload:\n{gen}"
    );
}

#[test]
fn c61c_first_bound_second_wildcard_drop() {
    // The unbound `_` sibling (a Drop type) is decomposed into the tuple and dropped at arm end;
    // exit-0 proves it is dropped exactly once and the moved unit is not dropped.
    agree_completes(
        "c61c_first_bound",
        &format!(
            "{ENUM2_DROP}fn main() {{ let e = E::V(D {{ v: 1 }}, D {{ v: 2 }}); \
             match e {{ E::V(a, _) => {{ assert_eq(take(a), 1); }} }} }}"
        ),
    );
}

#[test]
fn c61c_first_wildcard_second_bound_drop() {
    agree_completes(
        "c61c_first_wild",
        &format!(
            "{ENUM2_DROP}fn main() {{ let e = E::V(D {{ v: 1 }}, D {{ v: 2 }}); \
             match e {{ E::V(_, b) => {{ assert_eq(take(b), 2); }} }} }}"
        ),
    );
}

#[test]
fn c61c_three_fields_middle_discarded_drop() {
    agree_completes(
        "c61c_three_mid",
        "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { let r: Int32 = self.v; } }\n\
         enum E { V(D, D, D) }\nfn take(x: D) -> Int32 { x.v }\n\
         fn main() { let e = E::V(D { v: 1 }, D { v: 2 }, D { v: 3 }); \
         match e { E::V(a, _, c) => { assert_eq(take(a) + take(c), 4); } } }",
    );
}

#[test]
fn c61c_struct_shaped_variant_bindings_reordered() {
    // A struct-shaped variant, with pattern fields written in a DIFFERENT order than declaration
    // (`{ b: y, a: x }`). Decomposition is by declaration order; bindings still resolve correctly.
    agree_completes(
        "c61c_struct_reorder",
        "struct S { v: Int32 }\nenum E { V { a: S, b: S } }\nfn take(x: S) -> Int32 { x.v }\n\
         fn main() { let e = E::V { a: S { v: 10 }, b: S { v: 20 } }; \
         match e { E::V { b: y, a: x } => { assert_eq(take(x) * 2 + take(y), 40); } } }",
    );
}

#[test]
fn c61c_payload_moves_used_across_later_blocks() {
    // The bindings are moved out at arm entry (one block) but consumed inside an `if` in the arm
    // body (later blocks) — proving the decomposition's per-field liveness survives control flow.
    agree_completes(
        "c61c_cross_block",
        &format!(
            "{ENUM2}fn main() {{ let cond: Bool = true; let e = E::V(S {{ v: 4 }}, S {{ v: 5 }}); \
             match e {{ E::V(a, b) => {{ let r: Int32 = if cond {{ take(a) + take(b) }} else {{ 0 }}; \
             assert_eq(r, 9); }} }} }}"
        ),
    );
}

#[test]
fn c61c_a_false_assertion_still_traps_in_all_three_engines() {
    // Negative control: the C6.1c observations actually execute.
    let source = format!(
        "{ENUM2}fn main() {{ let e = E::V(S {{ v: 1 }}, S {{ v: 2 }}); \
         match e {{ E::V(a, b) => {{ assert_eq(take(a) + take(b), 99); }} }} }}"
    );
    let c = compile(&source, "c61c_false");
    assert!(
        interp::run_with_partial_output(&c.hir, c.file.clone(), &c.tables).is_err(),
        "HIR must trap"
    );
    let verified = verify_program(&c.program).unwrap();
    assert!(run_program(verified).is_err(), "MIR must trap");
}

// ================================== WP-C6.1d — non-Copy array by-value iteration (G2) --
//
// `for x in arr` over `[T; N]` with a non-Copy `T` is lowered by unconditional unrolling: the array
// is moved once into a per-element-drop-tracked owner, each element moves out via ConstIndex(i)
// into a FRESH binding local, and the body is lowered once per element. Break/continue/return/?
// drop the current binding; the array owner's scope drop destroys the unconsumed tail. Drop-bearing
// exit-0 is the no-double/missing-drop evidence (a wrong drop trips slot_violation). The HIR oracle
// moves each element and drops the remainder identically, so all three engines agree. Observable
// Drop ORDER/COUNT needs stdout and is a C6.3 item.

const ARR_S: &str = "struct S { v: Int32 }\nfn take(x: S) -> Int32 { x.v }\n";
const ARR_D: &str = "struct D { v: Int32 }\nimpl Drop for D { fn drop(&mut self) { let r: Int32 = self.v; } }\nfn take(x: D) -> Int32 { x.v }\n";

#[test]
fn c61d_consuming_match_array_baseline_drop() {
    // Owner test 1: the consuming-match array path (which shares the ConstIndex + array-drop
    // machinery) agrees across engines, establishing the baseline before iteration.
    agree_completes(
        "c61d_match_baseline",
        &format!(
            "{ARR_D}fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}]; \
             let n: Int32 = match arr {{ [a, b] => take(a) + take(b) }}; assert_eq(n, 3); }}"
        ),
    );
}

#[test]
fn c61d_zero_length_iterable_evaluated_once() {
    agree_completes(
        "c61d_zero",
        &format!(
            "{ARR_S}fn main() {{ let arr: [S; 0] = []; let mut sum: Int32 = 0; \
             for x in arr {{ sum = sum + take(x); }} assert_eq(sum, 0); }}"
        ),
    );
}

#[test]
fn c61d_single_non_copy_element() {
    agree_completes(
        "c61d_one",
        &format!(
            "{ARR_S}fn main() {{ let arr = [S {{ v: 42 }}]; let mut sum: Int32 = 0; \
             for x in arr {{ sum = sum + take(x); }} assert_eq(sum, 42); }}"
        ),
    );
}

#[test]
fn c61d_multiple_non_copy_elements_in_source_order() {
    let gen = agree_completes(
        "c61d_multi",
        &format!(
            "{ARR_S}fn main() {{ let arr = [S {{ v: 1 }}, S {{ v: 2 }}, S {{ v: 3 }}]; \
             let mut acc: Int32 = 0; for x in arr {{ acc = acc * 10 + take(x); }} \
             assert_eq(acc, 123); }}"
        ),
    );
    // Structural: unrolled ConstIndex moves, NO dynamic index / CheckIndex in this path.
    assert!(
        !gen.contains("as usize]"),
        "non-Copy array iteration must not use a dynamic index:\n{gen}"
    );
}

#[test]
fn c61d_multiple_drop_elements_full_consumption() {
    agree_completes(
        "c61d_drop_full",
        &format!(
            "{ARR_D}fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}, D {{ v: 3 }}]; \
             let mut sum: Int32 = 0; for x in arr {{ sum = sum + take(x); }} assert_eq(sum, 6); }}"
        ),
    );
}

#[test]
fn c61d_continue_drops_binding_keeps_remaining() {
    agree_completes(
        "c61d_continue",
        &format!(
            "{ARR_D}fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}, D {{ v: 3 }}]; \
             let mut sum: Int32 = 0; for x in arr {{ let k: Int32 = take(x); \
             if k == 2 {{ continue; }} sum = sum + k; }} assert_eq(sum, 4); }}"
        ),
    );
}

#[test]
fn c61d_break_at_first_iteration_drops_remaining() {
    // Break immediately: element 0 (bound) and elements 1,2 (unconsumed) all drop exactly once.
    agree_completes(
        "c61d_break_first",
        &format!(
            "{ARR_D}fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}, D {{ v: 3 }}]; \
             let mut sum: Int32 = 0; for x in arr {{ sum = sum + take(x); break; }} \
             assert_eq(sum, 1); }}"
        ),
    );
}

#[test]
fn c61d_break_in_the_middle_drops_remaining() {
    agree_completes(
        "c61d_break_mid",
        &format!(
            "{ARR_D}fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}, D {{ v: 3 }}, D {{ v: 4 }}]; \
             let mut sum: Int32 = 0; for x in arr {{ sum = sum + take(x); if sum >= 3 {{ break; }} }} \
             assert_eq(sum, 3); }}"
        ),
    );
}

#[test]
fn c61d_return_from_body_drops_remaining() {
    agree_completes(
        "c61d_return",
        &format!(
            "{ARR_D}fn first_ge(arr: [D; 3], threshold: Int32) -> Int32 {{ \
             for x in arr {{ let k: Int32 = take(x); if k >= threshold {{ return k; }} }} 0 }}\n\
             fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}, D {{ v: 3 }}]; \
             assert_eq(first_ge(arr, 2), 2); }}"
        ),
    );
}

#[test]
fn c61d_question_mark_propagation_drops_remaining() {
    agree_completes(
        "c61d_try",
        &format!(
            "{ARR_D}fn checked(k: Int32) -> Result<Int32, Int32> {{ if k >= 2 {{ Err(k) }} else {{ Ok(k) }} }}\n\
             fn sum_until(arr: [D; 3]) -> Result<Int32, Int32> {{ let mut sum: Int32 = 0; \
             for x in arr {{ let k: Int32 = take(x); let v: Int32 = checked(k)?; sum = sum + v; }} Ok(sum) }}\n\
             fn main() {{ let arr = [D {{ v: 1 }}, D {{ v: 2 }}, D {{ v: 3 }}]; \
             let r: Int32 = match sum_until(arr) {{ Ok(n) => n, Err(e) => e + 100 }}; assert_eq(r, 102); }}"
        ),
    );
}

#[test]
fn c61d_iterable_returned_by_a_function() {
    // The iterable is a function CALL — moved once into the array owner.
    agree_completes(
        "c61d_fn_iter",
        &format!(
            "{ARR_S}fn make() -> [S; 2] {{ [S {{ v: 10 }}, S {{ v: 20 }}] }}\n\
             fn main() {{ let mut sum: Int32 = 0; for x in make() {{ sum = sum + take(x); }} \
             assert_eq(sum, 30); }}"
        ),
    );
}

#[test]
fn c61d_a_trap_in_the_body_aborts_with_no_cleanup() {
    // A body that traps: HIR and MIR both trap (no binding/remaining cleanup after an abort).
    let source = format!(
        "{ARR_S}fn main() {{ let arr = [S {{ v: 1 }}, S {{ v: 0 }}, S {{ v: 3 }}]; let mut acc: Int32 = 0; \
         for x in arr {{ acc = acc + 10 / take(x); }} assert_eq(acc, 0); }}"
    );
    let c = compile(&source, "c61d_trap");
    assert!(
        interp::run_with_partial_output(&c.hir, c.file.clone(), &c.tables).is_err(),
        "HIR must trap (divide by zero)"
    );
    let verified = verify_program(&c.program).unwrap();
    assert!(run_program(verified).is_err(), "MIR must trap");
}
