// Gate 6 (G6-05) — "strongest comparator" typed-Rust host library.
//
// This is the shared preamble `include!`d by every case in `cases/`. It models
// the STRONGEST plausible *generated* Rust host for the Gate 5 ResNet50
// pipeline: what an ONNX signature importer could emit if it pushed as much of
// the STARK tensor contract into Rust's type system as stable Rust allows.
//
// Encoded in the type system:
//   * element type  -> the generic parameter `T` (f32 vs u8) — native to rustc
//   * shape         -> per-rank const-generic dims (Tensor4<T,N,C,H,W,Dev>)
//   * device        -> a phantom device tag `Dev` (Cpu / Cuda<ID>)
//
// Honesty caveats (expanded in the memo):
//   * Encoding device in the tensor type is NOT idiomatic. A typical `ort`
//     host configures the device via execution providers at session-build time
//     — a RUNTIME concern — so an ordinary Rust host catches device placement
//     at runtime (or silently, exactly like the Python baseline). We include a
//     phantom device only to show the strongest host CAN push it to compile
//     time, at the cost of a bespoke generated API.
//   * Per-rank types (Tensor4/Tensor2) are needed because stable Rust has no
//     rank-generic const-array parameter. A generator can emit them, but every
//     new rank is new generated code.
//   * Any op with a *computed* output dim (reshape/flatten/matmul/broadcast/
//     conv stride arithmetic) needs `feature(generic_const_exprs)` (unstable).
//     `cases/limit_reshape.rs` measures that wall. Pure permutations and
//     elementwise ops (all this CV preprocessing needs) do NOT, which is why
//     the strongest host reaches parity with STARK on THIS pipeline.
//
// NB: lint allows (dead_code, unused_variables) are passed via rustc -A flags
// by run.py, because this file is `include!`d mid-file where an inner attribute
// (`#![..]`) is not permitted.

use std::marker::PhantomData;

// ---- Device tags -----------------------------------------------------------
pub struct Cpu;
pub struct Cuda<const ID: usize>;

// ---- Rank-4 tensor: element T, const dims N,C,H,W, device Dev --------------
pub struct Tensor4<T, const N: usize, const C: usize, const H: usize, const W: usize, Dev> {
    data: Vec<T>,
    _dev: PhantomData<Dev>,
}

impl<T: Copy + Default, const N: usize, const C: usize, const H: usize, const W: usize, Dev>
    Tensor4<T, N, C, H, W, Dev>
{
    pub fn zeros() -> Self {
        Self { data: vec![T::default(); N * C * H * W], _dev: PhantomData }
    }

    /// Runtime boundary (the analogue of STARK's `TensorAny.refine::<..>()`):
    /// decoded-image data has runtime dims, so re-entering the typed world is a
    /// runtime-checked `Result`. Both STARK and this host pay this at runtime.
    pub fn from_runtime(v: Vec<T>) -> Result<Self, String> {
        let want = N * C * H * W;
        if v.len() == want {
            Ok(Self { data: v, _dev: PhantomData })
        } else {
            Err(format!("shape mismatch: got {} elements, want {}", v.len(), want))
        }
    }

    /// Elementwise cast — changes the element type, preserves shape+device.
    pub fn cast<U: Copy + Default + From<T>>(self) -> Tensor4<U, N, C, H, W, Dev> {
        Tensor4 { data: self.data.into_iter().map(U::from).collect(), _dev: PhantomData }
    }

    /// Move to another device — changes only the device tag.
    pub fn to_device<D2>(self) -> Tensor4<T, N, C, H, W, D2> {
        Tensor4 { data: self.data, _dev: PhantomData }
    }
}

/// NHWC -> NCHW permutation, fully typed on STABLE Rust: the output dims are a
/// *reordering* of the input dims (no arithmetic), so plain generics suffice.
/// Input  Tensor4<T, N, H, W, C, Dev>  (NHWC)
/// Output Tensor4<T, N, C, H, W, Dev>  (NCHW)
pub fn nhwc_to_nchw<T, const N: usize, const H: usize, const W: usize, const C: usize, Dev>(
    t: Tensor4<T, N, H, W, C, Dev>,
) -> Tensor4<T, N, C, H, W, Dev> {
    Tensor4 { data: t.data, _dev: PhantomData }
}

// ---- Rank-2 tensor (model output logits) -----------------------------------
pub struct Tensor2<T, const N: usize, const M: usize, Dev> {
    data: Vec<T>,
    _dev: PhantomData<Dev>,
}

// ---- Generated model signature (from the ONNX import) ----------------------
// input  data: f32 [1,3,224,224] on Cpu
// output logits: f32 [1,1000] on Cpu
pub struct Resnet50V17;

impl Resnet50V17 {
    pub fn predict(
        &self,
        x: &Tensor4<f32, 1, 3, 224, 224, Cpu>,
    ) -> Tensor2<f32, 1, 1000, Cpu> {
        // real host calls ORT here; stubbed — G6-05 measures the type contract
        Tensor2 { data: vec![0.0; 1000], _dev: PhantomData }
    }
}

// ---- Load-time artifact integrity gate (the d4b analogue) ------------------
// Both STARK's generated host and any competent Rust host embed the expected
// artifact digest and refuse to run on mismatch. The specific hash is
// immaterial to the "when is it caught" question (load time, runtime); we use a
// std-only FNV-1a here to stay dependency-free. The real hosts use SHA-256.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn verify_artifact(model_bytes: &[u8], expected_digest: u64) -> Result<(), String> {
    let got = fnv1a(model_bytes);
    if got == expected_digest {
        Ok(())
    } else {
        Err(format!(
            "ArtifactMismatch: digest {got:#x} does not match expected {expected_digest:#x}; refusing to run inference"
        ))
    }
}
