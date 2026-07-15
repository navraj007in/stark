//! Gate 5 M5.1 — deployment IR lowering.
//!
//! Hermetic: builds a tiny ResNet50-signature ONNX artifact in memory (no
//! network, no 98 MB model) and drives `starkc::deploy::lower_pipeline`. Covers
//! valid lowering, deterministic ordering, and every rejection path.

mod common;

use common::{model_path, valid_pipeline_source as valid_source, TempDir};
use starkc::deploy::{lower_pipeline, DeployOp, DeployTy};

// ---- valid lowering ----

#[test]
fn valid_pipeline_lowers_with_expected_structure() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", valid_source().as_bytes());
    let model = model_path(&dir);

    let program = lower_pipeline(&src, &model, "infer").unwrap_or_else(|e| {
        panic!("expected valid lowering, got:\n{}", e.render());
    });

    // Model was selected and bound to the verified signature + hash.
    assert_eq!(program.entry, "infer");
    assert_eq!(program.model.type_name, "Resnet50V17");
    assert_eq!(program.model.input.name, "data");
    assert_eq!(program.model.output.name, "resnetv17_dense0_fwd");
    assert_eq!(program.model.artifact_sha256.len(), 64);
    assert!(program
        .model
        .artifact_sha256
        .chars()
        .all(|c| c.is_ascii_hexdigit()));

    // Entry first, helper discovered second.
    assert_eq!(program.functions.len(), 2);
    assert_eq!(program.functions[0].name, "infer");
    assert_eq!(program.functions[1].name, "preprocess");

    // Entry ABI recovered as the deployment contract.
    let entry = program.entry_fn();
    assert_eq!(entry.params.len(), 2);
    assert_eq!(entry.params[0].ty, DeployTy::Model);
    assert_eq!(entry.params[1].ty, DeployTy::TensorAny);

    // The bounded op set is present, including the helper call.
    let ops = &entry.body;
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Refine { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Try { .. })));
    assert!(ops
        .iter()
        .any(|s| matches!(s.op, DeployOp::Call { callee: 1, .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Full { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Concat { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Sub { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Div { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Predict { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::Softmax { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::ArgMax { .. })));
    assert!(ops.iter().any(|s| matches!(s.op, DeployOp::WrapOk { .. })));

    // `full` preserves the exact source literal.
    let full_lits: Vec<&str> = ops
        .iter()
        .filter_map(|s| match &s.op {
            DeployOp::Full { scalar, .. } => Some(scalar.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(full_lits.contains(&"0.485"));
    assert!(full_lits.contains(&"0.225"));
}

#[test]
fn lowering_is_deterministic() {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", valid_source().as_bytes());
    let model = model_path(&dir);

    let render = |e: starkc::deploy::DeployError| e.render();
    let a = lower_pipeline(&src, &model, "infer").unwrap_or_else(|e| panic!("{}", render(e)));
    let b = lower_pipeline(&src, &model, "infer").unwrap_or_else(|e| panic!("{}", render(e)));
    assert_eq!(a, b, "identical inputs must lower to an identical program");
}

// ---- rejection paths ----

fn expect_error(source: &str, entry: &str) -> String {
    let dir = TempDir::new();
    let src = dir.write("pipeline.stark", source.as_bytes());
    let model = model_path(&dir);
    match lower_pipeline(&src, &model, entry) {
        Ok(_) => panic!("expected lowering to fail"),
        Err(e) => e.render(),
    }
}

#[test]
fn unknown_entry_is_rejected() {
    let rendered = expect_error(valid_source(), "does_not_exist");
    assert!(rendered.contains("E0601"), "{rendered}");
}

#[test]
fn nonconforming_entry_abi_is_rejected() {
    // Type-checks, but returns a bare tensor and takes a concrete input tensor,
    // so it is not a deployable entry.
    let source = r#"
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>;
}

fn infer(
    model: Resnet50V17,
    nchw: Tensor<Float32, [1, 3, 224, 224]>,
) -> Tensor<Float32, [1, 1000]> {
    let logits = model.predict(&nchw);
    logits.softmax::<1>()
}
"#;
    let rendered = expect_error(source, "infer");
    assert!(rendered.contains("E0602"), "{rendered}");
}

#[test]
fn recursion_is_rejected() {
    let source = r#"
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>;
}

fn spin(t: Tensor<Float32, [1, 3, 224, 224]>) -> Tensor<Float32, [1, 3, 224, 224]> {
    spin(t)
}

fn infer(
    model: Resnet50V17,
    raw: TensorAny,
) -> Result<Tensor<Int64, [1]>, String> {
    let nhwc = raw.refine::<UInt8, [1, 224, 224, 3]>()?;
    let nchw = nhwc.permute::<[0, 3, 1, 2]>().cast::<Float32>();
    let looped = spin(nchw);
    let logits = model.predict(&looped);
    Ok(logits.softmax::<1>().argmax::<1>())
}
"#;
    let rendered = expect_error(source, "infer");
    assert!(rendered.contains("E0604"), "{rendered}");
}

#[test]
fn unsupported_control_flow_is_rejected_with_span() {
    // An `if` expression in reachable code is out of the bounded op set.
    let source = r#"
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>;
}

fn infer(
    model: Resnet50V17,
    raw: TensorAny,
) -> Result<Tensor<Int64, [1]>, String> {
    let nhwc = raw.refine::<UInt8, [1, 224, 224, 3]>()?;
    let nchw = if true {
        nhwc.permute::<[0, 3, 1, 2]>().cast::<Float32>()
    } else {
        nhwc.permute::<[0, 3, 1, 2]>().cast::<Float32>()
    };
    let logits = model.predict(&nchw);
    Ok(logits.softmax::<1>().argmax::<1>())
}
"#;
    let rendered = expect_error(source, "infer");
    assert!(rendered.contains("E0605"), "{rendered}");
    assert!(rendered.contains("`if`"), "{rendered}");
    // The diagnostic points at source, not a synthetic span.
    assert!(rendered.contains("--> "), "{rendered}");
}

#[test]
fn multiple_models_are_rejected() {
    let source = r#"
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>;
}

model OtherNet<N: Dim> {
    input x: Tensor<Float32, [N, 3, 224, 224]>;
    output y: Tensor<Float32, [N, 1000]>;
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
    let rendered = expect_error(source, "infer");
    assert!(rendered.contains("E0603"), "{rendered}");
}

#[test]
fn signature_mismatch_is_rejected() {
    // A model declaration whose port dtype disagrees with the artifact fails
    // Gate 4 live verification before the program is emitted.
    let dir = TempDir::new();
    let source = r#"
model Resnet50V17<N: Dim> {
    input data: Tensor<Int32, [N, 3, 224, 224]>;
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
    let src = dir.write("pipeline.stark", source.as_bytes());
    let model = model_path(&dir);
    let rendered = match lower_pipeline(&src, &model, "infer") {
        Ok(_) => panic!("expected verification to fail"),
        Err(e) => e.render(),
    };
    assert!(
        rendered.contains("does not match") || rendered.contains("mismatch"),
        "{rendered}"
    );
}
