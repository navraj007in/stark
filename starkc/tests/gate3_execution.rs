use starkc::diag::Severity;
use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

fn examples() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/gate3")
}

fn execute(name: &str) -> String {
    let path = examples().join(name);
    let source = std::fs::read_to_string(&path).unwrap();
    let file = Arc::new(SourceFile::new(path.to_string_lossy(), source));
    let (ast, parse_diagnostics) = parse(&file, ParseMode::Program);
    assert!(
        parse_diagnostics.is_empty(),
        "{name}: {parse_diagnostics:?}"
    );
    let (hir, resolve_diagnostics) = resolve(&ast, file.clone());
    assert!(
        resolve_diagnostics.is_empty(),
        "{name}: {resolve_diagnostics:?}"
    );
    let checked = typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: {errors:?}");
    interp::run(&hir, file, &checked.tables).unwrap().output
}

#[test]
fn gate3_examples_execute_with_expected_output() {
    let cases = [
        ("01_hello.stark", "Hello from STARK\n"),
        ("02_control_flow.stark", "120\n"),
        ("03_aggregates.stark", "42\n"),
        ("04_try.stark", "42\n"),
        ("05_core_min.stark", "Gate 3\n42\ntrue\n[40, 2]\nGate\n"),
        ("06_methods.stark", "42\n"),
        ("07_drop_order.stark", "second\nfirst\n"),
    ];
    for (name, expected) in cases {
        assert_eq!(execute(name), expected, "{name}");
    }
}

#[test]
fn file_io_returns_results_instead_of_silently_failing() {
    let path = std::env::temp_dir().join(format!("stark-gate3-{}.txt", std::process::id()));
    let escaped = path.to_string_lossy().replace('\\', "\\\\");
    let source = format!(
        "fn main() {{ write_file(\"{escaped}\", \"saved\").unwrap(); println(read_file(\"{escaped}\").unwrap()); }}"
    );
    let file = Arc::new(SourceFile::new("io-test.stark", source));
    let (ast, parse_diagnostics) = parse(&file, ParseMode::Program);
    assert!(parse_diagnostics.is_empty(), "{parse_diagnostics:?}");
    let (hir, resolve_diagnostics) = resolve(&ast, file.clone());
    assert!(resolve_diagnostics.is_empty(), "{resolve_diagnostics:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    assert!(
        checked
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.severity != Severity::Error),
        "{:?}",
        checked.diagnostics
    );
    let execution = interp::run(&hir, file, &checked.tables).unwrap();
    assert_eq!(execution.output, "saved\n");
    let _ = std::fs::remove_file(path);
}

#[test]
fn run_cli_executes_program_and_reports_runtime_failure() {
    let binary = env!("CARGO_BIN_EXE_starkc");
    let success = Command::new(binary)
        .args(["run", examples().join("01_hello.stark").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(success.status.success());
    assert_eq!(
        String::from_utf8(success.stdout).unwrap(),
        "Hello from STARK\n"
    );

    let path =
        std::env::temp_dir().join(format!("stark-runtime-error-{}.stark", std::process::id()));
    std::fs::write(&path, "fn main() { let zero = 0; println(1 / zero); }").unwrap();
    let failure = Command::new(binary)
        .args(["run", path.to_str().unwrap()])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(path);
    assert!(!failure.status.success());
    assert!(String::from_utf8(failure.stderr)
        .unwrap()
        .contains("runtime error: division by zero"));
}

/// WP-C1.4 coverage: CLAUDE.md's memory-model rule that traps and `panic` abort execution
/// without running destructors ("Traps and panic abort: destructors do not run."). A live
/// `Marker` value with a `Drop` impl is still in scope when a division-by-zero trap fires;
/// its destructor's `println` must never execute, since the trap unwinds via `?` propagation
/// straight out of `run_main` without passing through the interpreter's normal end-of-scope
/// drop-execution code. Uses the CLI (not the in-process `interp::run` test helper) because
/// `run`'s `Result<Execution, RuntimeError>` signature discards any output captured before an
/// `Err`, making the captured-stdout assertion this test needs impossible via that helper.
#[test]
fn trap_aborts_without_running_pending_destructors() {
    let binary = env!("CARGO_BIN_EXE_starkc");
    let path =
        std::env::temp_dir().join(format!("stark-abort-no-drop-{}.stark", std::process::id()));
    std::fs::write(
        &path,
        "struct Marker { name: String } \
         impl Drop for Marker { fn drop(&mut self) { println(self.name.as_str()); } } \
         fn main() { \
             let m = Marker { name: String::from(\"should-not-print\") }; \
             let x: Int32 = 1; \
             let y: Int32 = 0; \
             let z = x / y; \
             println(\"unreachable\"); \
         }",
    )
    .unwrap();
    let failure = Command::new(binary)
        .args(["run", path.to_str().unwrap()])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(path);
    assert!(!failure.status.success());
    let stdout = String::from_utf8(failure.stdout).unwrap();
    assert!(
        !stdout.contains("should-not-print"),
        "a trap must abort before running any pending destructor, got stdout: {stdout:?}"
    );
    assert!(!stdout.contains("unreachable"), "got stdout: {stdout:?}");
    assert!(String::from_utf8(failure.stderr)
        .unwrap()
        .contains("runtime error: division by zero"));
}
