//! Gate 7 (G7-03) — native deployment lowering of the symbolic-shape pipeline.
//!
//! Hermetic: builds a Tiny-YOLOv2-signature ONNX artifact in memory (no
//! network) and drives `starkc::deploy` over the frozen Gate 7 pipeline plus
//! focused negative cases. Covers symbolic-dim lowering, deterministic
//! codegen, the shape-arithmetic wall, and that lowering never silently falls
//! back to an interpreter.

mod common;

use common::{tiny_yolov2_signature_bytes, write_model_bytes, TempDir};
use starkc::deploy::{emit, lower_pipeline, DeployOp, DeployTy};
use std::path::PathBuf;

fn frozen_pipeline_source() -> String {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/gate7/valid_pipeline.stark");
    std::fs::read_to_string(path).expect("read frozen pipeline")
}

fn model(dir: &TempDir) -> PathBuf {
    write_model_bytes(dir, &tiny_yolov2_signature_bytes())
}

// ---- valid symbolic lowering -------------------------------------------------

#[test]
fn frozen_pipeline_lowers_with_symbolic_reshape() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", frozen_pipeline_source().as_bytes());
    let program = lower_pipeline(&src, &model(&dir), "infer")
        .unwrap_or_else(|e| panic!("expected valid lowering:\n{}", e.render()));

    assert_eq!(program.model.type_name, "TinyYoloV2");
    assert_eq!(program.model.input.name, "image");
    assert_eq!(program.model.output.name, "grid");
    // entry + decode helper
    assert_eq!(program.functions.len(), 2);
    assert_eq!(program.functions[0].name, "infer");
    assert_eq!(program.functions[1].name, "decode");

    // Entry: refine (boundary) -> predict -> call(decode) -> Ok.
    let entry = program.entry_fn();
    assert!(entry
        .body
        .iter()
        .any(|s| matches!(s.op, DeployOp::Refine { .. })));
    assert!(entry
        .body
        .iter()
        .any(|s| matches!(s.op, DeployOp::Predict { .. })));
    assert!(entry
        .body
        .iter()
        .any(|s| matches!(s.op, DeployOp::Call { .. })));

    // decode carries the symbolic reshape.
    let decode = &program.functions[1];
    let reshape = decode
        .body
        .iter()
        .find_map(|s| match &s.op {
            DeployOp::Reshape { dims, .. } => Some(dims),
            _ => None,
        })
        .expect("decode has a reshape");
    // First dim is the symbolic batch, the rest literal (5,25,13,13).
    assert!(
        reshape[0].is_symbolic(),
        "reshape[0] should be symbolic: {reshape:?}"
    );
    assert_eq!(reshape[0].to_string(), "B");
    assert_eq!(
        reshape[1..]
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>(),
        vec!["5", "25", "13", "13"]
    );
}

#[test]
fn entry_return_type_is_symbolic_tensor() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", frozen_pipeline_source().as_bytes());
    let program =
        lower_pipeline(&src, &model(&dir), "infer").unwrap_or_else(|e| panic!("{}", e.render()));
    match &program.entry_fn().ret {
        DeployTy::Result(inner) => match &**inner {
            DeployTy::Tensor(shape) => {
                assert!(shape.dims[0].is_symbolic());
                assert_eq!(shape.to_string(), "Tensor<Float32, [B, 5, 13, 13, 25]>");
            }
            other => panic!("expected tensor, got {other}"),
        },
        other => panic!("expected Result, got {other}"),
    }
}

// ---- codegen -----------------------------------------------------------------

#[test]
fn generated_host_evaluates_symbolic_reshape() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", frozen_pipeline_source().as_bytes());
    let program =
        lower_pipeline(&src, &model(&dir), "infer").unwrap_or_else(|e| panic!("{}", e.render()));
    let files = emit(&program);
    let pipeline = files
        .iter()
        .find(|f| f.path == "src/generated_pipeline.rs")
        .map(|f| f.contents.as_str())
        .unwrap();

    // The symbolic batch is bound at the boundary and evaluated in the reshape.
    assert!(pipeline.contains("DimBind::Sym(\"B\")"), "{pipeline}");
    assert!(
        pipeline.contains("reshape(&[__env.get(\"B\")?, 5usize, 25usize, 13usize, 13usize])"),
        "{pipeline}"
    );
    // The host input is built from the pipeline's own refine contract.
    assert!(
        pipeline.contains(
            "build_host_input(image, DType::Float32, &[1usize, 3usize, 416usize, 416usize])"
        ),
        "{pipeline}"
    );
    // No interpreter is embedded.
    assert!(!pipeline.contains("interpret"), "{pipeline}");
}

#[test]
fn codegen_is_deterministic() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", frozen_pipeline_source().as_bytes());
    let program =
        lower_pipeline(&src, &model(&dir), "infer").unwrap_or_else(|e| panic!("{}", e.render()));
    let a = emit(&program);
    let b = emit(&program);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.path, y.path);
        assert_eq!(
            x.contents, y.contents,
            "non-deterministic emission for {}",
            x.path
        );
    }
}

// ---- negative cases ----------------------------------------------------------

/// A pipeline that passes the front end (product-preserving) but whose reshape
/// target uses subtraction, which the deployment backend cannot represent.
const SUBTRACTION_DIM: &str = r#"
model TinyYoloV2<B: Dim> {
    input image: Tensor<Float32, [B, 3, 416, 416]>;
    output grid: Tensor<Float32, [B, 125, 13, 13]>;
}
fn infer<B: Dim>(model: TinyYoloV2, raw: TensorAny) -> Result<Tensor<Float32, [B, 5, 25, 13, 13]>, String> {
    let image = raw.refine::<Float32, [B, 3, 416, 416]>()?;
    let grid = model.predict(&image);
    Ok(grid.reshape::<[B, 5, 25, 14 - 1, 13]>())
}
"#;

#[test]
fn unsupported_symbolic_dimension_is_rejected_at_generation() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", SUBTRACTION_DIM.as_bytes());
    let err =
        lower_pipeline(&src, &model(&dir), "infer").expect_err("subtraction dim must be rejected");
    let rendered = err.render();
    assert!(rendered.contains("E0606"), "{rendered}");
    assert!(rendered.contains("subtraction"), "{rendered}");
    // Source-mapped: the diagnostic points at the pipeline source.
    assert!(rendered.contains("pipeline.stark:"), "{rendered}");
}

/// A pipeline with control flow the deployment backend does not support. It
/// must be rejected outright — lowering never silently falls back to an
/// interpreter to execute the unsupported construct.
const CONTROL_FLOW: &str = r#"
model TinyYoloV2<B: Dim> {
    input image: Tensor<Float32, [B, 3, 416, 416]>;
    output grid: Tensor<Float32, [B, 125, 13, 13]>;
}
fn infer<B: Dim>(model: TinyYoloV2, raw: TensorAny) -> Result<Tensor<Float32, [B, 125, 13, 13]>, String> {
    let image = raw.refine::<Float32, [B, 3, 416, 416]>()?;
    let grid = if true { model.predict(&image) } else { model.predict(&image) };
    Ok(grid)
}
"#;

#[test]
fn unsupported_construct_is_rejected_without_fallback() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", CONTROL_FLOW.as_bytes());
    let rendered = lower_pipeline(&src, &model(&dir), "infer")
        .expect_err("control flow must be rejected in deployment")
        .render();
    // A deployment-lowering diagnostic (E06xx), source-mapped, and no host.
    assert!(rendered.contains("E060"), "{rendered}");
    assert!(rendered.contains("pipeline.stark:"), "{rendered}");
}
