//! WP-C5.3a bring-up proofs that belong to the native engine alone: the generated-source shape
//! (nominal definitions), and the SCOPE BOUNDARY — what the backend refuses, and how.
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
        format!("c5_3a_{tag}.stark"),
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

    let target_dir = std::env::temp_dir().join(format!("stark_c5_3a_{tag}_{}", std::process::id()));
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

/// **The scope boundary, made a clean diagnostic instead of a rustc error.** Moving a non-`Copy`
/// value out of a local initialised in an earlier block is what WP-C5.3d's controlled storage is
/// for; until then the backend must say so itself rather than emitting code the Rust borrow
/// checker rejects with "value moved here, in previous iteration of loop".
#[test]
fn a_cross_block_non_copy_move_is_refused_as_unsupported_not_as_a_build_failure() {
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
    match build(source, "crossblock") {
        Err(BackendDiagnostic::Unsupported(message)) => {
            assert!(
                message.contains("WP-C5.3d"),
                "the diagnostic must name the package that lifts the limit: {message}"
            );
        }
        Err(other) => panic!(
            "a scope limit must be Unsupported, not {other:?} -- a BuildFailed here means rustc \
             rejected generated code, which is the failure mode this guard exists to prevent"
        ),
        Ok(_) => panic!(
            "cross-block non-Copy moves are not supported yet; if this now builds, the guard is \
             stale and should be removed along with this test"
        ),
    }
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
