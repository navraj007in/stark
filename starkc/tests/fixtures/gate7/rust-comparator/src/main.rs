// Gate 7 (G7-06) — real typed-Rust comparator host.
//
// This is an ACTUAL ONNX Runtime application (correcting the Gate 6 stub): it
// verifies the artifact with real SHA-256, creates a real `ort` session, runs
// real inference, and performs the reshape+permute with `ndarray`. The typed
// layer in `typed.rs` carries the compile-time shape/dtype/device/range
// guarantees (its cases/ measure exactly what rustc catches); this file does the
// runtime work and enforces what the type layer cannot (a dynamic output shape,
// artifact integrity).

#![allow(dead_code, unused_variables)]

use ndarray::{ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use sha2::{Digest, Sha256};
use std::path::Path;

mod typed;

// The pipeline's fixed dims (Tiny YOLOv2, batch 1). These are the const generics
// the typed layer is instantiated with.
const N: usize = 1;
const C_IN: usize = 3;
const SIDE: usize = 416;
const CH: usize = 125; // A*(5+C) = 5*(5+20)
const A: usize = 5;
const K: usize = 25; // 5 + C
const GRID: usize = 13;

fn sha256_hex(bytes: &[u8]) -> String {
    let mut out = String::new();
    for b in Sha256::digest(bytes) {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

fn load_raw_f32(path: &Path, n: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() != n * 4 {
        return Err(format!("raw input has {} bytes, want {}", bytes.len(), n * 4));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut input = None;
    let mut expected_sha = None;
    let mut dump = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--input-raw" => input = args.next(),
            "--expected-sha256" => expected_sha = args.next(),
            "--dump-output" => dump = args.next(),
            other => return Err(format!("unexpected arg `{other}`")),
        }
    }
    let model = model.ok_or("missing --model")?;
    let input = input.ok_or("missing --input-raw")?;

    // Real artifact integrity check (runtime, before the session): refuse a
    // swapped model. This is the d9 defect path — real SHA-256, not FNV.
    let bytes = std::fs::read(&model).map_err(|e| format!("read model: {e}"))?;
    let actual = sha256_hex(&bytes);
    if let Some(exp) = &expected_sha {
        if &actual != exp {
            return Err(format!(
                "ArtifactMismatch: model SHA-256 {actual} != expected {exp}; refusing to run"
            ));
        }
    }

    // Real ORT session + inference.
    let mut session = Session::builder()
        .map_err(|e| format!("session builder: {e}"))?
        .commit_from_file(&model)
        .map_err(|e| format!("load model: {e}"))?;
    // Port names are part of the generated model contract (Tiny YOLOv2).
    let in_name = "image";
    let out_name = "grid";

    let raw = load_raw_f32(Path::new(&input), N * C_IN * SIDE * SIDE)?;

    // Typed refine boundary: adopt the const-shaped typed tensor (batch checked).
    let image = typed::Tensor4::<f32, N, C_IN, SIDE, SIDE, typed::Cpu, typed::Unspecified>::refine(
        &[N, C_IN, SIDE, SIDE],
        raw.clone(),
    )?;
    let _ = &image; // the typed value carries the compile-time input contract

    let value = OrtTensor::from_array((vec![N as i64, C_IN as i64, SIDE as i64, SIDE as i64], raw))
        .map_err(|e| format!("build ORT input: {e}"))?;
    let outputs = session
        .run(ort::inputs![in_name => value])
        .map_err(|e| format!("inference: {e}"))?;
    let (out_shape, out_data) = outputs[out_name]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("extract output: {e}"))?;
    let out_shape: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();

    // The typed predict contract expects [N,125,13,13]; a runtime output that
    // disagrees (declaration/artifact drift, d8) is rejected here by the typed
    // `from_vec` element-count check.
    if out_shape != [N, CH, GRID, GRID] {
        return Err(format!(
            "model output shape {out_shape:?} != declared [{N},{CH},{GRID},{GRID}]"
        ));
    }
    let grid_ty = typed::TinyYoloV2::predict_ty::<N>(&image, out_data.to_vec())?;
    // Type-level reshape + permute (compile-time guarantees):
    let split_ty = typed::split_channels(grid_ty);
    let _perm_ty = typed::permute_grid(split_ty);

    // Real value-level reshape + permute with ndarray: [N,125,H,W] ->
    // [N,5,25,H,W] -> [N,5,H,W,25].
    let grid = ArrayD::from_shape_vec(IxDyn(&[N, CH, GRID, GRID]), out_data.to_vec())
        .map_err(|e| format!("grid: {e}"))?;
    let split = grid
        .into_shape_with_order(IxDyn(&[N, A, K, GRID, GRID]))
        .map_err(|e| format!("reshape: {e}"))?;
    let permuted = split.permuted_axes(IxDyn(&[0, 1, 3, 4, 2]));
    let permuted = permuted.as_standard_layout().to_owned();

    println!("result: f32 {:?} ({} elems)", permuted.shape(), permuted.len());
    if let Some(path) = dump {
        let mut buf = Vec::with_capacity(permuted.len() * 4);
        for &v in permuted.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, buf).map_err(|e| format!("write {path}: {e}"))?;
    }
    Ok(())
}
