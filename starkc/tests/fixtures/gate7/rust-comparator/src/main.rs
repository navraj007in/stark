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
    // The declared output channel count. Defaults to the real 125; passing a
    // different value simulates a drifted declaration (d8), rejected at runtime.
    let mut expect_channels = CH;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--input-raw" => input = args.next(),
            "--expected-sha256" => expected_sha = args.next(),
            "--dump-output" => dump = args.next(),
            "--expect-channels" => {
                expect_channels = args
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--expect-channels needs a number")?
            }
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

    // The generated contract declares `expect_channels` (125 for the real
    // artifact). A runtime output that disagrees (declaration/artifact drift,
    // d8) is rejected here — a *runtime* check, unlike STARK's deploy-time one.
    if out_shape != [N, expect_channels, GRID, GRID] {
        return Err(format!(
            "declaration/artifact drift: model output {out_shape:?} != declared \
             [{N},{expect_channels},{GRID},{GRID}]"
        ));
    }

    // The typed pipeline performs the actual reshape+permute; the dumped output
    // is produced BY the typed tensors (no parallel ndarray path).
    let grid_ty = typed::TinyYoloV2::predict_ty::<N>(&image, out_data.to_vec())?;
    let split_ty = typed::split_channels(grid_ty);
    let perm_ty = typed::permute_grid(split_ty);
    let result = perm_ty.into_data();

    println!("result: f32 [{N}, {A}, {GRID}, {GRID}, {K}] ({} elems)", result.len());
    if let Some(path) = dump {
        let mut buf = Vec::with_capacity(result.len() * 4);
        for v in &result {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, buf).map_err(|e| format!("write {path}: {e}"))?;
    }
    Ok(())
}
