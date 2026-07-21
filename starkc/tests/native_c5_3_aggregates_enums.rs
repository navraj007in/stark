//! WP-C5.3a/C5.3b bring-up proofs that belong to the native engine alone: the generated-source
//! shape (nominal definitions for structs and enums), and the SCOPE BOUNDARY — what the backend
//! refuses, and how.
//!
//! Value agreement for aggregates lives in `three_engine_differential.rs`, which is where §14's
//! C5.3 exit condition is discharged. This file covers what a three-engine comparator
//! structurally cannot: a program that one engine must reject.

use starkc::backend::generated_rust::{emit_native_debug, BackendDiagnostic, NativeBuildOptions};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn build(source: &str, tag: &str) -> Result<(String, std::process::Output), BackendDiagnostic> {
    let file = Arc::new(SourceFile::new(
        format!("c5_3_{tag}.stark"),
        source.to_string(),
    ));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "{tag} parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "{tag} resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let type_errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(type_errors.is_empty(), "{tag} typecheck: {type_errors:?}");

    let program = match lower_program(&hir, &checked.tables, file) {
        Ok(program) => program,
        Err(e) => panic!("{tag} must lower: {} @ {:?}", e.what, e.span),
    };
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("{tag} must verify: {e:?}"));

    let target_dir = std::env::temp_dir().join(format!("stark_c5_3_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.clone(),
        },
    )?;
    let generated = std::fs::read_to_string(artifact.build_dir.join("src/main.rs")).unwrap();
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    let _ = std::fs::remove_dir_all(&target_dir);
    Ok((generated, run))
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// §6.3: one Rust definition per reachable concrete nominal instance, and no derived traits
/// beyond what MIR's own `Copy` classification calls for.
#[test]
fn each_nominal_instance_gets_exactly_one_generated_definition() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Point { x: Int32, y: Int32 }

struct Wrapper { p: Point }

fn main() {
    let a: Point = Point { x: 1, y: 2 };
    let b: Point = Point { x: 3, y: 4 };
    let w: Wrapper = Wrapper { p: Point { x: 5, y: 6 } };
    assert_eq(a.x + b.y + w.p.x, 10);
}
"#;
    let (generated, run) = build(source, "nominals").expect("must build");
    assert_eq!(
        run.status.code(),
        Some(0),
        "in-program assertions must hold"
    );

    // TWO nominals, TWO definitions -- three `Point` values do not produce three definitions.
    assert_eq!(
        generated.matches("struct stark_ty_").count(),
        2,
        "expected exactly one definition per nominal instance:\n{generated}"
    );
    // Neither struct has an `impl Copy` in STARK, so neither generated type derives anything.
    assert!(
        !generated.contains("#[derive("),
        "no STARK type here is Copy, so nothing should be derived:\n{generated}"
    );
}

/// The flagged §6.3-vs-§7.4 reading (CD-056): a STARK `impl Copy` — and ONLY that — makes the
/// generated type derive `Clone, Copy`. If the owner overrules the reading, this test is what
/// changes.
#[test]
fn a_stark_impl_copy_is_what_makes_a_generated_type_copy() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Marked { v: Int32 }

impl Copy for Marked {}

struct Unmarked { v: Int32 }

fn main() {
    let a: Marked = Marked { v: 1 };
    let b: Unmarked = Unmarked { v: 2 };
    assert_eq(a.v + b.v, 3);
}
"#;
    let (generated, run) = build(source, "copyimpl").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    assert_eq!(
        generated.matches("#[derive(Clone, Copy)]").count(),
        1,
        "exactly the marked type should derive Copy:\n{generated}"
    );
}

/// **WP-C5.3d-0's payoff.** This program was refused before the slot foundation existed: passing
/// a non-`Copy` struct by value puts the move in a different basic block from the construction,
/// and the block-dispatch loop made Rust's borrow checker reject a move verified MIR proves
/// sound. With `ValueSlot` carrying liveness explicitly, it compiles and runs.
#[test]
fn a_cross_block_non_copy_move_now_compiles_and_runs() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Point { x: Int32, y: Int32 }

fn sum(p: Point) -> Int32 {
    p.x + p.y
}

fn main() {
    let p: Point = Point { x: 3, y: 4 };
    assert_eq(p.x, 3);
    assert_eq(sum(p), 7);
}
"#;
    let (generated, run) = build(source, "crossblock").expect("slots must make this expressible");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        generated.contains("ValueSlot"),
        "the non-Copy local must be slot-backed:\n{generated}"
    );
    // Deliverable 2: no ad hoc unsafe in emitted MIR bodies -- every unsafe operation routes
    // through the reviewed runtime helpers.
    assert!(
        !generated.contains("unsafe"),
        "emitted bodies must contain no unsafe of their own:\n{generated}"
    );
}

/// The guard must not over-reject: moving a non-`Copy` value WITHIN one block is how ordinary
/// aggregate construction lowers (`_2 = aggregate ..; _1 = move _2;`) and must keep working.
#[test]
fn a_same_block_non_copy_move_still_builds() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"struct Point { x: Int32, y: Int32 }

fn main() {
    let p: Point = Point { x: 3, y: 4 };
    assert_eq(p.x + p.y, 7);
}
"#;
    let (_, run) = build(source, "sameblock").expect("same-block moves must still build");
    assert_eq!(run.status.code(), Some(0));
}

// ------------------------------------------------------------ WP-C5.3b: enums --

/// §6.3 again, for enums: one Rust definition per instance, with uniformly TUPLE variants so an
/// empty payload needs no unit-variant special case in construction, patterns, or the
/// discriminant match.
#[test]
fn an_enum_instance_gets_one_definition_with_uniform_tuple_variants() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"enum Shape { Point, Circle(Int32), Rect(Int32, Int32) }

fn main() {
    let s: Shape = Shape::Rect(3, 4);
    let a: Int32 = match s {
        Shape::Point => 0,
        Shape::Circle(r) => r,
        Shape::Rect(w, h) => w * h,
    };
    assert_eq(a, 12);
}
"#;
    let (generated, run) = build(source, "enumdef").expect("must build");
    assert_eq!(run.status.code(), Some(0));

    assert_eq!(
        generated.matches("enum stark_ty_").count(),
        1,
        "expected exactly one enum definition:\n{generated}"
    );
    // The fieldless variant is still a tuple variant -- `V0()`, not `V0`.
    assert!(
        generated.contains("V0(),"),
        "an empty payload must still be a tuple variant:\n{generated}"
    );
    assert!(
        generated.contains("V1(i32),") && generated.contains("V2(i32, i32),"),
        "payload arity must follow the type context:\n{generated}"
    );
}

/// The discriminant is recovered by MATCHING, not by a Rust integer cast: an enum with payloads
/// has no `as` conversion. Every variant is listed with no catch-all, so adding a variant cannot
/// silently fall through to a wrong index.
#[test]
fn a_discriminant_read_lists_every_variant_with_no_catch_all() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"enum Three { A, B, C(Int32) }

fn main() {
    let t: Three = Three::B;
    let v: Int32 = match t {
        Three::A => 1,
        Three::B => 2,
        Three::C(n) => n,
    };
    assert_eq(v, 2);
}
"#;
    let (generated, run) = build(source, "enumdisc").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    for variant in ["V0(..) =>", "V1(..) =>", "V2(..) =>"] {
        assert!(
            generated.contains(variant),
            "the discriminant match must name {variant}:\n{generated}"
        );
    }
}

/// V-DISC-1 makes a variant-field projection legal only after a discriminant test, so the `_`
/// arm of a payload read is provably dead. It is emitted as `unreachable!()` naming the rule --
/// the same treatment the verifier-proved dead-block path gets -- rather than as a fabricated
/// value that would silently paper over a lowering bug.
#[test]
fn a_payload_read_marks_its_impossible_arm_unreachable() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"enum Holder { Has(Int32), Empty }

fn main() {
    let h: Holder = Holder::Has(41);
    let v: Int32 = match h {
        Holder::Has(n) => n + 1,
        Holder::Empty => 0,
    };
    assert_eq(v, 42);
}
"#;
    let (generated, run) = build(source, "enumpayload").expect("must build");
    assert_eq!(run.status.code(), Some(0));
    assert!(
        generated.contains("V-DISC-1"),
        "the impossible arm should name the rule that makes it impossible:\n{generated}"
    );
}
