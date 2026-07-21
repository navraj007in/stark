//! WP-C5.2d bring-up proof: a multi-function program with real parameters and direct calls --
//! not just a single entry-only body (C5.1b-C5.2c) -- compiles and runs natively. Indirect
//! calls and printed output remain out of scope; this file stays narrow to the C5.2d proof,
//! matching the scope discipline of the earlier `native_c5_*` files.

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
        format!("c5_2d_{tag}.stark"),
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

    let target_dir = std::env::temp_dir().join(format!("stark_c5_2d_{tag}_{}", std::process::id()));
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
fn direct_call_with_parameters_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // The RETURNED VALUE is observed, not merely the fact that a call compiled and the process
    // exited: `let x: Int32 = add(2, 3);` with an exit-0 check passes against a backend that
    // returns zero from every function. Native `println` is out of scope until WP-C5.3, so a
    // failed `assert_eq` (a message-less `Terminator::Trap`, supported natively as of WP-C5.2e)
    // is the observation channel -- see `native_c5_2c_operations.rs`'s negative control, which
    // proves a false assertion really does trap rather than being compiled away.
    let source = r#"
fn add(a: Int32, b: Int32) -> Int32 {
    a + b
}

fn main() {
    assert_eq(add(2, 3), 5);
    // Argument ORDER is observable only with asymmetric arguments and a non-commutative
    // operation; `add` alone could not distinguish a backend that swapped them.
    assert_eq(add(10, 1), 11);
}
"#;
    let run = compile_and_run(source, "call");
    assert_eq!(
        run.status.code(),
        Some(0),
        "the call's return value must be correct; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// Parameter ORDER and identity across a multi-parameter signature, which a commutative helper
/// cannot pin: `sub` and the three-way `pick` both change result if any two parameters are
/// transposed on the way in.
#[test]
fn parameter_order_is_preserved_across_a_direct_call() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = r#"
fn sub(a: Int32, b: Int32) -> Int32 {
    a - b
}

fn pick(first: Int32, second: Int32, third: Int32) -> Int32 {
    first * 100 + second * 10 + third
}

fn main() {
    assert_eq(sub(10, 3), 7);
    assert_eq(sub(3, 10), -7);
    assert_eq(pick(1, 2, 3), 123);
}
"#;
    let run = compile_and_run(source, "paramorder");
    assert_eq!(
        run.status.code(),
        Some(0),
        "parameters must arrive in declaration order; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn multi_parameter_helper_used_in_control_flow_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    // Exercises: a 3-parameter function, a call whose result feeds a comparison used by `if`,
    // and a second, differently-typed helper -- proving parameter-index-to-local mapping and
    // multi-function symbol naming both work, not just a single trivial call.
    let source = r#"
fn clamp(value: Int32, lo: Int32, hi: Int32) -> Int32 {
    let mut result: Int32 = value;
    if result < lo {
        result = lo;
    }
    if result > hi {
        result = hi;
    }
    result
}

fn is_positive(x: Float64) -> Bool {
    x > 0.0
}

fn main() {
    // All three of `clamp`'s paths are exercised and each result observed -- clamped high,
    // clamped low, and passed through untouched.
    assert_eq(clamp(15, 0, 10), 10);
    assert_eq(clamp(-5, 0, 10), 0);
    assert_eq(clamp(7, 0, 10), 7);

    let clamped: Int32 = clamp(15, 0, 10);
    let mut flag: Bool = false;
    if clamped == 10 {
        flag = true;
    }
    assert(flag);

    // The differently-typed helper's Bool return is observed in both directions.
    assert(is_positive(2.5));
    assert(!is_positive(-2.5));
}
"#;
    let run = compile_and_run(source, "multiparam");
    assert_eq!(
        run.status.code(),
        Some(0),
        "every helper result must be correct; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}
