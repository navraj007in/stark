//! Gate 5 opt-in real-backend test (plan §10).
//!
//! This test is `#[ignore]`d: it downloads/links ONNX Runtime, builds the
//! generated host, and runs a Python reference, so it is NOT part of the
//! hermetic default suite. Run it explicitly with the pinned fixtures:
//!
//! ```bash
//! STARK_GATE5_MODEL=/path/resnet50-v1-7.onnx \
//! STARK_GATE5_IMAGE=/path/dog.jpg \
//! cargo test --test gate5_codegen -- --ignored --nocapture
//! ```
//!
//! Fixture checksums are verified before any expensive work; a wrong file is a
//! hard failure, not a silent pass.

use sha2::{Digest, Sha256};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// SHA-256 of the pinned ResNet50 v1.7 artifact (ONNX Model Zoo).
const MODEL_SHA256: &str = "af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c";
/// SHA-256 of the reference input image (`dog.jpg`, PyTorch Hub sample).
const IMAGE_SHA256: &str = "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a";

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir
}

fn sha256_hex(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read fixture");
    let digest = Sha256::digest(&bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Resolve a fixture from an env override or the default `tmp/` path; return
/// `None` (skip) if absent, and panic if present but checksum-wrong.
fn fixture(env_var: &str, default: PathBuf, expected_sha: &str) -> Option<PathBuf> {
    let path = env::var(env_var).map(PathBuf::from).unwrap_or(default);
    if !path.exists() {
        eprintln!("skipping: fixture not found at {}", path.display());
        return None;
    }
    let actual = sha256_hex(&path);
    assert_eq!(
        actual,
        expected_sha,
        "fixture {} has SHA-256 {actual}, expected {expected_sha}",
        path.display()
    );
    Some(path)
}

fn parse_field(stdout: &str, prefix: &str) -> Option<String> {
    stdout
        .lines()
        .find(|l| l.starts_with(prefix))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
}

#[test]
#[ignore = "opt-in: downloads/links ONNX Runtime, builds the host, runs Python"]
fn real_inference_agrees_with_reference() {
    let ws = workspace_root();
    let Some(model) = fixture(
        "STARK_GATE5_MODEL",
        ws.join("starkc/tmp/resnet50-v1-7.onnx"),
        MODEL_SHA256,
    ) else {
        return;
    };
    let Some(image) = fixture(
        "STARK_GATE5_IMAGE",
        ws.join("starkc/tmp/dog.jpg"),
        IMAGE_SHA256,
    ) else {
        return;
    };

    let starkc_dir = ws.join("starkc");
    let out_dir = env::temp_dir().join(format!("stark-gate5-host-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);

    // Generate.
    let deploy = Command::new(env!("CARGO_BIN_EXE_starkc"))
        .arg("deploy")
        .arg(starkc_dir.join("examples/gate5/valid_pipeline.stark"))
        .arg("--model")
        .arg(&model)
        .args(["--entry", "infer", "--out"])
        .arg(&out_dir)
        .arg("--force")
        .output()
        .expect("run starkc deploy");
    assert!(
        deploy.status.success(),
        "deploy failed: {}",
        String::from_utf8_lossy(&deploy.stderr)
    );

    // Build the runnable binary (locked), then run the generated runtime's own
    // unit tests. `cargo test` alone does not leave `target/release/<bin>`.
    let manifest = out_dir.join("Cargo.toml");
    let build = Command::new("cargo")
        .args(["build", "--release", "--locked", "--manifest-path"])
        .arg(&manifest)
        .status()
        .expect("cargo build on generated host");
    assert!(build.success(), "generated host build failed");
    let test = Command::new("cargo")
        .args(["test", "--release", "--locked", "--manifest-path"])
        .arg(&manifest)
        .status()
        .expect("cargo test on generated host");
    assert!(test.success(), "generated host unit tests failed");

    let binary = out_dir.join("target/release/stark-resnet50");
    let run = Command::new(&binary)
        .arg("--model")
        .arg(&model)
        .arg("--image")
        .arg(&image)
        .output()
        .expect("run generated binary");
    assert!(
        run.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let host_out = String::from_utf8_lossy(&run.stdout);
    let host_class: i32 = parse_field(&host_out, "top-1 class :")
        .expect("host class")
        .parse()
        .unwrap();
    let host_prob: f32 = parse_field(&host_out, "probability :")
        .expect("host prob")
        .parse()
        .unwrap();

    // Independent Python/ORT reference.
    let python = ws.join("starkc/tmp/venv/bin/python3");
    let script = ws.join("starkc/tests/fixtures/gate5/reference.py");
    let py = Command::new(&python)
        .arg(&script)
        .arg(&model)
        .arg(&image)
        .output()
        .expect("run python reference");
    assert!(
        py.status.success(),
        "python reference failed: {}",
        String::from_utf8_lossy(&py.stderr)
    );
    let py_out = String::from_utf8_lossy(&py.stdout);
    let py_class: i32 = parse_field(&py_out, "top-1 class :")
        .expect("py class")
        .parse()
        .unwrap();
    let py_prob: f32 = parse_field(&py_out, "probability :")
        .expect("py prob")
        .parse()
        .unwrap();

    println!("host: class={host_class} prob={host_prob}");
    println!("python: class={py_class} prob={py_prob}");

    // Categorical agreement is exact; probability agreement is within the
    // documented tolerance (Pillow vs `image` interpolation — see
    // docs/gate5-exit.md §2).
    assert_eq!(host_class, py_class, "top-1 class mismatch");
    let diff = (host_prob - py_prob).abs();
    assert!(diff <= 1e-3, "probability diff {diff} exceeds 1e-3");

    let _ = std::fs::remove_dir_all(&out_dir);
}
