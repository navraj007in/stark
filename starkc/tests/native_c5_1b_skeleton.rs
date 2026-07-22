//! WP-C5.1b bring-up proof: one verified, empty `fn main() {}` MIR program becomes a standalone
//! executable via the real pipeline (parse -> resolve -> typecheck -> lower -> verify ->
//! `backend::generated_rust::emit_native_debug` -> `cargo build` -> run). This is C5.1's exit
//! condition ("one verified empty/scalar MIR program becomes a standalone executable on both
//! pinned targets" -- `WP-C5.1.md`), proven here for the local target this test runs on; the
//! `x86_64-unknown-linux-gnu` half is proven by this same test running in CI.
//!
//! Broader native coverage (`native_scalar.rs`, `native_traps.rs`, etc., §15.2) is WP-C5.2+;
//! this file is deliberately narrow and will not grow past the C5.1b proof.

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

#[test]
fn empty_main_compiles_and_runs_natively() {
    if !rustc_available() {
        eprintln!(
            "SKIP: no rustc in this environment; WP-C5.1b's compile+run leg cannot be exercised \
             here."
        );
        return;
    }

    let source = "fn main() { }\n";
    let file = Arc::new(SourceFile::new(
        "c5_1b_empty_main.stark",
        source.to_string(),
    ));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let type_errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(type_errors.is_empty(), "typecheck: {type_errors:?}");

    let mir_program = match lower_program(&hir, &checked.tables, file.clone()) {
        Ok(program) => program,
        Err(e) => panic!("empty main must lower: {} @ {:?}", e.what, e.span),
    };
    let verified = verify_program(&mir_program).expect("empty main's MIR must verify");

    let target_dir =
        std::env::temp_dir().join(format!("stark_c5_1b_skeleton_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);
    let options = NativeBuildOptions {
        target_dir: target_dir.clone(),
        target_contract: "stark-64-v1".to_string(),
    };

    let artifact = emit_native_debug(&verified, &options).unwrap_or_else(|e| {
        panic!("WP-C5.1b native build of empty `fn main() {{ }}` failed: {e:?}")
    });
    assert!(
        artifact.binary_path.exists(),
        "reported binary path does not exist: {}",
        artifact.binary_path.display()
    );
    assert!(
        artifact.build_dir.join("build.json").exists(),
        "build-manifest schema (§11.1) was not written"
    );
    assert!(
        artifact.build_dir.join("Cargo.toml").exists(),
        "generated-crate Cargo.toml was not written"
    );

    let run = Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    assert_eq!(
        run.status.code(),
        Some(0),
        "generated binary must exit 0 for an empty, non-trapping `main`; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        run.stdout.is_empty(),
        "empty main must produce no stdout; got: {:?}",
        String::from_utf8_lossy(&run.stdout)
    );

    let _ = std::fs::remove_dir_all(&target_dir);
}
