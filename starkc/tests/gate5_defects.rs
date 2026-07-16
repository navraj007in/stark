//! Gate 5 defect corpus — diagnostic guards (plan §M5.4).
//!
//! Hermetic: runs `starkc check`/`starkc deploy` on the four defect examples
//! and asserts the exact stage and diagnostic each is claimed to produce in
//! `docs/gate5-exit.md`. This keeps the exit report's safety evidence honest —
//! if an example stops failing for the documented reason, CI fails here.

mod common;

use common::{model_path, TempDir};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/gate5")
        .join(name)
}

fn check(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_starkc"))
        .args(["check", "--extension", "tensor"])
        .arg(path)
        .output()
        .expect("run starkc check")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn bad_shape_is_a_compile_time_dimension_error() {
    let out = check(&example("bad_shape.stark"));
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("E0212"), "{err}");
    assert!(err.contains("dimension mismatch"), "{err}");
}

#[test]
fn bad_dtype_is_a_compile_time_element_type_error() {
    let out = check(&example("bad_dtype.stark"));
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("E0212"), "{err}");
    assert!(err.contains("element type mismatch"), "{err}");
}

#[test]
fn bad_device_is_a_compile_time_device_placement_error() {
    // Regression guard: this case must fail at the `model.predict` device
    // mismatch (E0212), not earlier at a malformed `refine`/`to_device`.
    let out = check(&example("bad_device.stark"));
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("E0212"), "{err}");
    assert!(err.contains("device mismatch"), "{err}");
    assert!(err.contains("Cpu") && err.contains("Cuda<0>"), "{err}");
    // The failure is at the prediction site, past preprocessing.
    assert!(
        err.contains("model.predict") || err.contains("predict"),
        "{err}"
    );
}

#[test]
fn artifact_drift_is_rejected_at_deploy_before_inference() {
    // The declaration says output [N, 999]; the artifact says [N, 1000].
    // `starkc deploy` runs Gate 4 live verification and refuses to generate.
    let dir = TempDir::new();
    let model = model_path(&dir);
    let out = Command::new(env!("CARGO_BIN_EXE_starkc"))
        .arg("deploy")
        .arg(example("artifact_drift.stark"))
        .args(["--model"])
        .arg(&model)
        .args(["--entry", "infer", "--out"])
        .arg(dir.path().join("host"))
        .output()
        .expect("run starkc deploy");
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(
        err.contains("does not match") || err.contains("999") || err.contains("1000"),
        "{err}"
    );
    // No host project was generated.
    assert!(!dir.path().join("host/Cargo.toml").exists());
}
