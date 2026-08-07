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
    let checked = typecheck::analyze(&hir);
    let type_errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(type_errors.is_empty(), "{tag} typecheck: {type_errors:?}");

    let mir_program = match lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    ) {
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
        target_contract: "stark-64-v1".to_string(),
        ..NativeBuildOptions::default()
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

// WP-C6.3e: `panic("...")` carries a `&str` message. It aborts with exit 101, the `explicit panic`
// category header (parseable), the source location, AND the user message on its own line.
#[test]
fn panic_with_message_reports_category_location_and_message() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = "fn main() {\n    panic(\"the sky is falling\");\n}\n";
    let (run, file_name) = compile_and_run(source, "panicmsg");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("explicit panic"),
        "stderr missing category: {stderr}"
    );
    assert!(
        stderr.contains(&format!("{file_name}:2:")),
        "stderr missing location {file_name}:2: {stderr}"
    );
    assert!(
        stderr.contains("the sky is falling"),
        "stderr missing the user message: {stderr}"
    );
}

// CD-120 Contract B (partial output on trap): output emitted BEFORE a trap survives on stdout,
// and NOTHING after the trap is emitted. `print("before")` has no trailing newline, so it sits in
// stdout's LineWriter; the trap ABI must flush it before `exit(101)` or the prefix is lost --
// which would diverge from the HIR/MIR interpreters that retain their captured prefix.
#[test]
fn output_before_trap_is_flushed_then_abort() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"fn main() {
    print("before");
    let a: Int32 = 2147483647;
    let b: Int32 = a + 1;
    print("after");
}
"#;
    let (run, _) = compile_and_run(source, "flush_before_trap");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert_eq!(
        stdout, "before",
        "the pre-trap prefix must be flushed, and nothing after the trap emitted: {stdout:?}"
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("integer overflow"),
        "stderr missing category: {stderr}"
    );
}

/// A panic reached only on a taken branch still carries its message natively.
#[test]
fn conditional_panic_with_message() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = "fn main() {\n    let x: Int32 = 5;\n    if x > 3 { panic(\"too big\"); }\n}\n";
    let (run, _) = compile_and_run(source, "condpanic");
    assert_eq!(run.status.code(), Some(101), "trap exit code must be 101");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("explicit panic"), "category: {stderr}");
    assert!(stderr.contains("too big"), "user message: {stderr}");
}
