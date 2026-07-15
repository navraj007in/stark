//! Gate 5 M5.2 — deterministic host emission and the `starkc deploy` CLI.
//!
//! Hermetic: exercises the emitter and project writer against a tiny in-memory
//! ONNX signature. Never builds the generated crate or downloads ONNX Runtime.

mod common;

use common::{model_path, valid_pipeline_source, TempDir};
use starkc::deploy::{emit, lower_pipeline, write_project};
use std::path::Path;
use std::process::Command;

fn program(dir: &TempDir) -> starkc::deploy::DeploymentProgram {
    let src = dir.write("pipeline.stark", valid_pipeline_source().as_bytes());
    let model = model_path(dir);
    lower_pipeline(&src, &model, "infer").unwrap_or_else(|e| panic!("{}", e.render()))
}

#[test]
fn emission_is_byte_for_byte_deterministic() {
    let dir = TempDir::new();
    let prog = program(&dir);
    let a = emit(&prog);
    let b = emit(&prog);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.path, y.path);
        assert_eq!(
            x.contents, y.contents,
            "file `{}` differs across runs",
            x.path
        );
    }
}

#[test]
fn emitted_files_are_clean_and_self_contained() {
    let dir = TempDir::new();
    let prog = program(&dir);
    let files = emit(&prog);

    let names: Vec<&str> = files.iter().map(|f| f.path).collect();
    for expected in [
        "Cargo.toml",
        "Cargo.lock",
        "src/main.rs",
        "src/runtime.rs",
        "src/generated_pipeline.rs",
        "deployment.json",
        "README.md",
    ] {
        assert!(names.contains(&expected), "missing {expected}");
    }

    for f in &files {
        assert!(
            f.contents.ends_with('\n'),
            "{} lacks trailing newline",
            f.path
        );
        assert!(
            !f.contents.ends_with("\n\n"),
            "{} has multiple trailing newlines",
            f.path
        );
        assert!(!f.contents.contains('\r'), "{} contains CR", f.path);
        // No leaked host paths from the generating machine.
        assert!(
            !f.contents.contains("/private/tmp") && !f.contents.contains("stark-gate5-"),
            "{} leaks a host path",
            f.path
        );
    }
}

#[test]
fn cargo_toml_pins_the_backend() {
    let dir = TempDir::new();
    let prog = program(&dir);
    let files = emit(&prog);
    let cargo = &files
        .iter()
        .find(|f| f.path == "Cargo.toml")
        .unwrap()
        .contents;
    assert!(cargo.contains("ort = \"=2.0.0-rc.12\""), "{cargo}");
    assert!(cargo.contains("ndarray = \"=0.17.2\""));
    assert!(cargo.contains("image = "));
    assert!(cargo.contains("sha2 = "));
    assert!(cargo.contains("rust-version = \"1.88\""));
}

#[test]
fn manifest_records_model_and_ownership() {
    let dir = TempDir::new();
    let prog = program(&dir);
    let files = emit(&prog);
    let manifest = &files
        .iter()
        .find(|f| f.path == "deployment.json")
        .unwrap()
        .contents;
    assert!(manifest.contains("\"generated_by\": \"starkc deploy\""));
    assert!(manifest.contains("\"schema\": 1"));
    assert!(manifest.contains(&prog.model.artifact_sha256));
    assert!(manifest.contains("\"type_name\": \"Resnet50V17\""));
    assert!(manifest.contains("\"data\""));
    assert!(manifest.contains("\"resnetv17_dense0_fwd\""));
}

#[test]
fn generated_pipeline_translates_the_ir() {
    let dir = TempDir::new();
    let prog = program(&dir);
    let files = emit(&prog);
    let pipeline = &files
        .iter()
        .find(|f| f.path == "src/generated_pipeline.rs")
        .unwrap()
        .contents;
    assert!(
        pipeline.contains("pub fn infer(model: &mut Model, raw: Tensor) -> Result<Tensor, String>")
    );
    assert!(pipeline.contains("pub fn preprocess(nhwc: Tensor)"));
    assert!(pipeline.contains(".refine(DType::UInt8"));
    assert!(pipeline.contains("model.predict("));
    assert!(pipeline.contains(".softmax(1)"));
    assert!(pipeline.contains(".argmax(1)"));
    assert!(pipeline.contains("0.485f32"));
    // No `unsafe` in generated code.
    assert!(!pipeline.contains("unsafe"));
}

#[test]
fn write_refuses_then_forces_and_protects_foreign_dirs() {
    let dir = TempDir::new();
    let prog = program(&dir);
    let files = emit(&prog);
    let out = dir.path().join("host");

    // Fresh write into a new dir.
    write_project(&out, &files, false).unwrap();
    assert!(out.join("src/main.rs").exists());
    assert!(out.join("deployment.json").exists());

    // Second write without --force is refused.
    let err = write_project(&out, &files, false).unwrap_err();
    assert!(err.contains("--force"), "{err}");

    // With --force it succeeds.
    write_project(&out, &files, true).unwrap();

    // A non-generated existing directory is protected regardless of --force.
    let foreign = dir.path().join("foreign");
    std::fs::create_dir_all(&foreign).unwrap();
    std::fs::write(foreign.join("important.txt"), b"keep me").unwrap();
    let err = write_project(&foreign, &files, true).unwrap_err();
    assert!(err.contains("not a starkc-generated project"), "{err}");
    assert!(
        foreign.join("important.txt").exists(),
        "foreign file destroyed"
    );
}

// ---- CLI contract ----

fn deploy_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_starkc"))
        .arg("deploy")
        .args(args)
        .output()
        .expect("run starkc deploy")
}

#[test]
fn cli_missing_arguments_exit_2() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", valid_pipeline_source().as_bytes());
    let model = model_path(&dir);
    // Missing --entry.
    let out = deploy_cli(&[
        src.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--out",
        dir.path().join("h").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn cli_success_reports_summary() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", valid_pipeline_source().as_bytes());
    let model = model_path(&dir);
    let out_dir = dir.path().join("host");
    let out = deploy_cli(&[
        src.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--entry",
        "infer",
        "--out",
        out_dir.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Generated deployment host"));
    assert!(stdout.contains("cargo build --release --locked"));
    assert!(Path::new(&out_dir).join("Cargo.toml").exists());
}

#[test]
fn cli_unsupported_pipeline_exits_1_with_diagnostic() {
    let dir = TempDir::new();
    let bad = r#"
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>;
}

fn infer(
    model: Resnet50V17,
    raw: TensorAny,
) -> Result<Tensor<Int64, [1]>, String> {
    let nhwc = raw.refine::<UInt8, [1, 224, 224, 3]>()?;
    let nchw = nhwc.permute::<[0, 3, 1, 2]>().cast::<Float32>();
    let logits = model.predict(&nchw);
    Ok(logits.softmax::<1>().argmax::<1>())
}
"#;
    // Force a lowering failure via an unknown entry.
    let src = dir.write("pipeline.stark", bad.as_bytes());
    let model = model_path(&dir);
    let out = deploy_cli(&[
        src.to_str().unwrap(),
        "--model",
        model.to_str().unwrap(),
        "--entry",
        "nope",
        "--out",
        dir.path().join("h").to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("E0601"));
}
