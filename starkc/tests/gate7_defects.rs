//! Gate 7 (G7-05) — defect-corpus guard.
//!
//! Hermetic: runs `starkc check`/`starkc deploy` on the Gate 7 defect examples
//! and asserts the exact stage and diagnostic each is claimed to produce in the
//! G7-05 evidence (`tests/results/gate7/defect-matrix.json`). The full
//! build-and-run evaluation (incl. the runtime artifact swap) lives in
//! `scripts/run-gate7-evaluation.py`; this keeps the source/deploy-stage
//! evidence from silently drifting.

mod common;

use common::{tiny_yolov2_signature_bytes, write_model_bytes, TempDir};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn defect(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/gate7/defects")
        .join(name)
}

fn check(path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_starkc"))
        .args(["check", "--extension", "tensor"])
        .arg(path)
        .output()
        .expect("run starkc check")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The seven source-level defects are rejected at compile time with `E0212`.
#[test]
fn source_defects_are_compile_time_rejected() {
    for name in [
        "d1_reshape_product.stark",
        "d2_symbol_relationship.stark",
        "d3_broadcast.stark",
        "d4_range_missing.stark",
        "d5_range_double.stark",
        "d6_dtype.stark",
        "d7_device.stark",
        "d10_range_normalize_bytes.stark",
        "d11_range_wrong_model.stark",
        "d12_range_merge.stark",
        "d13_range_erase.stark",
    ] {
        let out = check(&defect(name));
        assert_eq!(out.status.code(), Some(1), "{name} should be rejected");
        let err = stderr(&out);
        assert!(err.contains("E0212"), "{name}: expected E0212:\n{err}");
    }
}

/// Declaration/artifact drift is caught at deploy time against the real
/// signature (built in-memory here — no network, no model download).
#[test]
fn declaration_drift_is_deploy_time_rejected() {
    let dir = TempDir::new();
    let model = write_model_bytes(&dir, &tiny_yolov2_signature_bytes());
    let out_dir = dir.path().join("host");
    let out = Command::new(env!("CARGO_BIN_EXE_starkc"))
        .args(["deploy"])
        .arg(defect("d8_declaration_drift.stark"))
        .args(["--model"])
        .arg(&model)
        .args(["--entry", "infer", "--out"])
        .arg(&out_dir)
        .args(["--force"])
        .output()
        .expect("run starkc deploy");
    assert_eq!(
        out.status.code(),
        Some(1),
        "drift must be rejected at deploy"
    );
    let err = stderr(&out);
    assert!(
        err.contains("does not match the model declaration"),
        "expected declaration-drift error:\n{err}"
    );
}
