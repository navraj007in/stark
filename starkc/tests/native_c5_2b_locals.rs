//! WP-C5.2b bring-up proof: a `main` with real local declarations, primitive constants of
//! several types, and a copy of one local into another -- not just C5.1b's trivial empty body --
//! compiles and runs natively. Broader coverage (arithmetic, control flow, calls) is WP-C5.2c/d;
//! this file stays narrow to the C5.2b proof, matching `native_c5_1b_skeleton.rs`'s own scope
//! discipline.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::process::Command;
use std::sync::Arc;

fn rustc_available() -> bool {
    Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn compile_and_run(source: &str, tag: &str) -> std::process::Output {
    let file = Arc::new(SourceFile::new(
        format!("c5_2b_{tag}.stark"),
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

    let mir_program = match lower_program(&hir, &checked.tables, file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("{tag} must lower: {} @ {:?}", e.what, e.span),
    };
    let verified = match verify_program(&mir_program) {
        Ok(v) => v,
        Err(errors) => panic!("{tag}'s MIR must verify: {errors:?}"),
    };

    let target_dir = std::env::temp_dir().join(format!("stark_c5_2b_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let options = NativeBuildOptions {
        target_dir: target_dir.clone(),
    };
    let artifact = emit_native_debug(&verified, &options)
        .unwrap_or_else(|e| panic!("{tag} native build failed: {e:?}"));

    let run = Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    let _ = std::fs::remove_dir_all(&target_dir);
    run
}

#[test]
fn locals_constants_and_copies_compile_and_run_natively() {
    if !rustc_available() {
        eprintln!(
            "SKIP: no rustc in this environment; WP-C5.2b's compile+run leg cannot be exercised \
             here."
        );
        return;
    }

    // Exercises: several primitive types as `let` locals, a `Bool`/`Char`/`Float64` constant
    // each, and a copy of one local's value into another (`let y = x;`) -- WP-C5.2b's actual
    // scope, not yet arithmetic or printing.
    let source = r#"
fn main() {
    let a: Int32 = 42;
    let b: Bool = true;
    let c: Char = 'A';
    let d: Float64 = 3.5;
    let e: UInt8 = 255;
    let f: Int32 = a;
}
"#;
    let run = compile_and_run(source, "locals");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(run.stdout.is_empty());
}

#[test]
fn float32_and_float64_locals_compile_and_run_natively() {
    if !rustc_available() {
        eprintln!(
            "SKIP: no rustc in this environment; WP-C5.2b's compile+run leg cannot be exercised \
             here."
        );
        return;
    }

    // Ordinary Float32/Float64 locals end to end. NaN/infinity's Rust-syntax validity is already
    // covered at the unit level in `emit_types.rs` (there is no way to construct them from
    // STARK source yet -- division is a `Checked` terminator, not a plain `Use` rvalue, so it
    // lands in WP-C5.2c).
    let source = r#"
fn main() {
    let x: Float64 = 1.0;
    let y: Float32 = 2.5f32;
}
"#;
    let run = compile_and_run(source, "floats");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}
