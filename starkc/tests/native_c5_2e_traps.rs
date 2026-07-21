//! WP-C5.2e bring-up proof: a native trap reports its category AND a correct source file/line
//! on stderr, with exit code 101 -- not just "some nonzero exit happened" (C5.2c's own trap
//! tests only checked that). This file proves the trap ABI itself; C5.2c's tests keep proving
//! that the operations that can trap produce correct values on the success path.

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

fn compile_and_run(source: &str, tag: &str) -> (std::process::Output, String) {
    let file_name = format!("c5_2e_{tag}.stark");
    let file = Arc::new(SourceFile::new(file_name.clone(), source.to_string()));
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

    let target_dir = std::env::temp_dir().join(format!("stark_c5_2e_{tag}_{}", std::process::id()));
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
    (run, file_name)
}

#[test]
fn overflow_trap_reports_category_and_exact_line() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // No leading newline in the raw string: line 1 is `fn main() {`, so the trapping statement
    // is deliberately on line 4 -- an exact, easy-to-verify expectation, not a loose one.
    let source = r#"fn main() {
    let a: Int32 = 2147483647;
    let b: Int32 = 1;
    let c: Int32 = a + b;
}
"#;
    let (run, file_name) = compile_and_run(source, "overflow_line");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("integer overflow"),
        "stderr missing category message: {stderr}"
    );
    assert!(
        stderr.contains(&format!("{file_name}:4:")),
        "stderr missing correct file:line ({file_name}:4): {stderr}"
    );
}

#[test]
fn division_by_zero_trap_reports_category() {
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
    let (run, _) = compile_and_run(source, "divzero");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("division by zero"),
        "stderr missing category message: {stderr}"
    );
}

#[test]
fn invalid_shift_trap_reports_category() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 1;
    let n: Int32 = 40;
    let b: Int32 = a << n;
}
"#;
    let (run, _) = compile_and_run(source, "invalidshift");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("invalid shift amount"),
        "stderr missing category message: {stderr}"
    );
}

#[test]
fn cast_failure_trap_reports_category() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn main() {
    let a: Int32 = 1000;
    let b: Int8 = a as Int8;
}
"#;
    let (run, _) = compile_and_run(source, "castfail");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("cast failure"),
        "stderr missing category message: {stderr}"
    );
}
