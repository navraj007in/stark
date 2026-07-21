//! WP-C5.2c bring-up proof: real arithmetic (with overflow trapping), comparisons, `if`/`else`,
//! and a `while` loop -- not just straight-line locals (C5.2b) -- compile and run natively.
//! Direct/indirect calls and printed output are WP-C5.2d; this file stays narrow to the C5.2c
//! proof, matching the scope discipline of `native_c5_1b_skeleton.rs`/`native_c5_2b_locals.rs`.

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
        format!("c5_2c_{tag}.stark"),
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

    let target_dir = std::env::temp_dir().join(format!("stark_c5_2c_{tag}_{}", std::process::id()));
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
fn checked_arithmetic_and_comparisons_succeed_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 10;
    let b: Int32 = 3;
    let sum: Int32 = a + b;
    let diff: Int32 = a - b;
    let prod: Int32 = a * b;
    let quot: Int32 = a / b;
    let rem: Int32 = a % b;
    let shifted: Int32 = a << 2;
    let anded: Int32 = a & b;
    let cmp: Bool = a > b;
    let eq: Bool = a == b;
    let notted: Bool = !eq;
    let f: Float64 = 3.5;
    let g: Float64 = 1.5;
    let fsum: Float64 = f + g;
    let fneg: Float64 = -f;
    let fcmp: Bool = f > g;
}
"#;
    let run = compile_and_run(source, "arith");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn integer_overflow_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 2147483647;
    let b: Int32 = 1;
    let c: Int32 = a + b;
}
"#;
    let run = compile_and_run(source, "overflow");
    assert_ne!(
        run.status.code(),
        Some(0),
        "overflow must trap (nonzero exit), not silently wrap or succeed"
    );
}

#[test]
fn division_by_zero_traps_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 10;
    let b: Int32 = 0;
    let c: Int32 = a / b;
}
"#;
    let run = compile_and_run(source, "divzero");
    assert_ne!(run.status.code(), Some(0), "division by zero must trap");
}

#[test]
fn if_else_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 5;
    let mut b: Int32 = 0;
    if a > 3 {
        b = 1;
    } else {
        b = 2;
    }
}
"#;
    let run = compile_and_run(source, "ifelse");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn while_loop_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let mut i: Int32 = 0;
    while i < 5 {
        i = i + 1;
    }
}
"#;
    let run = compile_and_run(source, "whileloop");
    assert_eq!(
        run.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}
