//! Shared hermetic scaffolding for the Gate 5 test binaries: a tiny in-memory
//! ONNX signature builder (no network, no 98 MB model) and a temp directory.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---- minimal ONNX protobuf builder (mirrors tests/gate4_onnx.rs) ----

fn varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        bytes.push(byte);
        if value == 0 {
            return bytes;
        }
    }
}
fn key(field: u32, wire: u8) -> Vec<u8> {
    varint((u64::from(field) << 3) | u64::from(wire))
}
fn scalar(field: u32, value: u64) -> Vec<u8> {
    let mut bytes = key(field, 0);
    bytes.extend(varint(value));
    bytes
}
fn message(field: u32, payload: &[u8]) -> Vec<u8> {
    let mut bytes = key(field, 2);
    bytes.extend(varint(payload.len() as u64));
    bytes.extend(payload);
    bytes
}
fn string(field: u32, value: &str) -> Vec<u8> {
    message(field, value.as_bytes())
}

pub enum Dim<'a> {
    Static(i64),
    Named(&'a str),
}
fn dimension(d: &Dim<'_>) -> Vec<u8> {
    match d {
        Dim::Static(v) => scalar(1, *v as u64),
        Dim::Named(n) => string(2, n),
    }
}
fn tensor_type(dtype: u64, dims: &[Dim<'_>]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        shape.extend(message(1, &dimension(d)));
    }
    let mut t = scalar(1, dtype);
    t.extend(message(2, &shape));
    t
}
fn value(name: &str, dtype: u64, dims: &[Dim<'_>]) -> Vec<u8> {
    let mut info = string(1, name);
    info.extend(message(2, &message(1, &tensor_type(dtype, dims))));
    info
}
fn model_bytes(inputs: &[Vec<u8>], outputs: &[Vec<u8>]) -> Vec<u8> {
    let mut graph = Vec::new();
    for i in inputs {
        graph.extend(message(11, i));
    }
    for o in outputs {
        graph.extend(message(12, o));
    }
    message(7, &graph)
}

/// A ResNet50V17-shaped artifact: input `data` F32 [N,3,224,224],
/// output `resnetv17_dense0_fwd` F32 [N,1000].
pub fn resnet_signature_bytes() -> Vec<u8> {
    model_bytes(
        &[value(
            "data",
            1,
            &[
                Dim::Named("N"),
                Dim::Static(3),
                Dim::Static(224),
                Dim::Static(224),
            ],
        )],
        &[value(
            "resnetv17_dense0_fwd",
            1,
            &[Dim::Named("N"), Dim::Static(1000)],
        )],
    )
}

// ---- temp-file scaffolding ----

static COUNTER: AtomicUsize = AtomicUsize::new(0);

pub struct TempDir(pub PathBuf);
impl TempDir {
    pub fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("stark-gate5-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&path).unwrap();
        TempDir(path)
    }
    pub fn write(&self, name: &str, contents: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(&p, contents).unwrap();
        p
    }
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}
impl Default for TempDir {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write the tiny ResNet-signature artifact into `dir` and return its path.
pub fn model_path(dir: &TempDir) -> PathBuf {
    dir.write("model.onnx", &resnet_signature_bytes())
}

pub fn valid_pipeline_source() -> &'static str {
    include_str!("../../examples/gate5/valid_pipeline.stark")
}
