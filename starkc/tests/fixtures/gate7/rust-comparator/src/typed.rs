// Gate 7 (G7-06) — the strongest stable-Rust typed layer for the Tiny YOLOv2
// pipeline. This is the type surface a maximal *generated* Rust host could emit:
// per-rank const-generic tensors, a phantom device tag, phantom value-range
// markers, and bespoke typed shape operations. It contains no `ort` so the
// defect cases can be type-checked quickly with `rustc`.
//
// Honesty notes (measured, not asserted, by run.py):
//   * Batch is a *const generic* `B` (or fully-runtime dims). Stable Rust cannot
//     carry a *symbolic/dynamic* batch through the type system: a batch known
//     only at runtime forces runtime-shaped tensors. We fix `B` at compile time
//     and re-check it at the refine boundary.
//   * The computed reshape `[B,125,H,W] -> [B,5,25,H,W]` is expressed as a
//     *bespoke* per-reshape function that bakes in `125 == 5*25`. A *general*
//     split whose factors are checked from the input needs `generic_const_exprs`
//     (unstable). Each distinct reshape is thus new generated code.
//
// NB: lint allows are passed via `rustc -A` by run.py (this file is `include!`d
// mid-file, where an inner `#![..]` attribute is not permitted).

use std::marker::PhantomData;

// ---- device tags -----------------------------------------------------------
pub struct Cpu;
pub struct Cuda<const ID: usize>;

// ---- value-range markers (phantom) -----------------------------------------
pub trait Range {}
pub struct Unspecified;
pub struct ByteRange;
pub struct UnitRange;
pub struct Normalized;
impl Range for Unspecified {}
impl Range for ByteRange {}
impl Range for UnitRange {}
impl Range for Normalized {}

// ---- rank-4 and rank-5 const-generic tensors -------------------------------
// element type T, const dims, phantom device Dev, phantom value-range R.
pub struct Tensor4<T, const N: usize, const C: usize, const H: usize, const W: usize, Dev, R> {
    data: Vec<T>,
    _dev: PhantomData<Dev>,
    _range: PhantomData<R>,
}

pub struct Tensor5<
    T,
    const N: usize,
    const A: usize,
    const B: usize,
    const H: usize,
    const W: usize,
    Dev,
    R,
> {
    data: Vec<T>,
    _dev: PhantomData<Dev>,
    _range: PhantomData<R>,
}

impl<T: Copy + Default, const N: usize, const C: usize, const H: usize, const W: usize, Dev, R>
    Tensor4<T, N, C, H, W, Dev, R>
{
    pub fn from_vec(data: Vec<T>) -> Result<Self, String> {
        let want = N * C * H * W;
        if data.len() != want {
            return Err(format!("shape mismatch: {} elements, want {want}", data.len()));
        }
        Ok(Self { data, _dev: PhantomData, _range: PhantomData })
    }

    /// The `refine` trust boundary: check a runtime-shaped, type-erased tensor
    /// has exactly these static dims (and re-bind the const batch), then adopt
    /// the typed form. Batch that varies at runtime is *checked* here, not typed.
    pub fn refine(actual_shape: &[usize], data: Vec<T>) -> Result<Self, String> {
        if actual_shape != [N, C, H, W] {
            return Err(format!("refine shape mismatch: {actual_shape:?} != {:?}", [N, C, H, W]));
        }
        Self::from_vec(data)
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Elementwise cast — changes T, preserves shape/device/range.
    pub fn cast<U: Copy + Default + From<T>>(self) -> Tensor4<U, N, C, H, W, Dev, R> {
        Tensor4 {
            data: self.data.into_iter().map(U::from).collect(),
            _dev: PhantomData,
            _range: PhantomData,
        }
    }
}

// ---- value-range transitions (phantom-typed) -------------------------------
impl<const N: usize, const C: usize, const H: usize, const W: usize, Dev>
    Tensor4<f32, N, C, H, W, Dev, ByteRange>
{
    pub fn scale_255(self) -> Tensor4<f32, N, C, H, W, Dev, UnitRange> {
        Tensor4 { data: self.data, _dev: PhantomData, _range: PhantomData }
    }
}
impl<const N: usize, const C: usize, const H: usize, const W: usize, Dev>
    Tensor4<f32, N, C, H, W, Dev, UnitRange>
{
    pub fn normalize(self) -> Tensor4<f32, N, C, H, W, Dev, Normalized> {
        Tensor4 { data: self.data, _dev: PhantomData, _range: PhantomData }
    }
}

// ---- the bespoke reshape: [N,125,H,W] -> [N,5,25,H,W] -----------------------
// IMPORTANT (measured honestly): rustc does NOT prove `125 == 5*25` here. Both
// tensor structs hold an unconstrained `Vec<T>`, so this body compiles no matter
// what factors the return type declares — the author/generator could write
// `-> Tensor5<.., 5, 26, ..>` and it would still compile. What rustc enforces is
// only that *callers* cannot request a different signature than this fixed one.
// A *general* split whose factors are checked against 125 needs the unstable
// `generic_const_exprs`. STARK, by contrast, proves the polynomial equality.
// The data move is the correct row-major re-view (channel axis split, no reorder).
pub fn split_channels<const N: usize, const H: usize, const W: usize, Dev, R>(
    t: Tensor4<f32, N, 125, H, W, Dev, R>,
) -> Tensor5<f32, N, 5, 25, H, W, Dev, R> {
    Tensor5 { data: t.data, _dev: PhantomData, _range: PhantomData }
}

// ---- the bespoke permute [0,1,3,4,2]: [N,A,K,H,W] -> [N,A,H,W,K] ------------
// This performs the *real* reindex so the typed value carries the actual result
// (no parallel ndarray path); the const dims drive the loop bounds.
pub fn permute_grid<const N: usize, const A: usize, const K: usize, const H: usize, const W: usize, Dev, R>(
    t: Tensor5<f32, N, A, K, H, W, Dev, R>,
) -> Tensor5<f32, N, A, H, W, K, Dev, R> {
    let src = &t.data;
    let mut out = vec![0.0f32; src.len()];
    for n in 0..N {
        for a in 0..A {
            for k in 0..K {
                for h in 0..H {
                    for w in 0..W {
                        let si = ((((n * A + a) * K + k) * H + h) * W + w) as usize;
                        let oi = ((((n * A + a) * H + h) * W + w) * K + k) as usize;
                        out[oi] = src[si];
                    }
                }
            }
        }
    }
    Tensor5 { data: out, _dev: PhantomData, _range: PhantomData }
}

impl<
        T,
        const N: usize,
        const A: usize,
        const B: usize,
        const H: usize,
        const W: usize,
        Dev,
        R,
    > Tensor5<T, N, A, B, H, W, Dev, R>
{
    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

// ---- typed broadcast_to (fixed signature; same caveat as split_channels) ----
// Broadcasts [N,1,H,W] to [N,C,H,W]. A caller requesting an incompatible target
// (different N/H/W) is a compile-time type error; but as with reshape, the
// element relationship is not *proved*, only fixed by this signature.
pub fn broadcast_channels<const N: usize, const C: usize, const H: usize, const W: usize, Dev, R>(
    t: Tensor4<f32, N, 1, H, W, Dev, R>,
) -> Tensor4<f32, N, C, H, W, Dev, R> {
    let mut out = Vec::with_capacity(N * C * H * W);
    for n in 0..N {
        for _c in 0..C {
            for hw in 0..(H * W) {
                out.push(t.data[n * H * W + hw]);
            }
        }
    }
    Tensor4 { data: out, _dev: PhantomData, _range: PhantomData }
}

// ---- elementwise add requiring matching value range ------------------------
// Two operands must share the range marker `R` (the type unifies them); mixing
// e.g. ByteRange and UnitRange is a compile-time type error.
pub fn add<const N: usize, const C: usize, const H: usize, const W: usize, Dev, R>(
    a: Tensor4<f32, N, C, H, W, Dev, R>,
    b: Tensor4<f32, N, C, H, W, Dev, R>,
) -> Tensor4<f32, N, C, H, W, Dev, R> {
    let data = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor4 { data, _dev: PhantomData, _range: PhantomData }
}

// ---- a model whose input contract requires `Normalized` --------------------
pub struct NormNet;
impl NormNet {
    pub fn predict_ty<const N: usize>(
        _input: &Tensor4<f32, N, 3, 4, 4, Cpu, Normalized>,
        out: Vec<f32>,
    ) -> Result<Tensor4<f32, N, 10, 1, 1, Cpu, Unspecified>, String> {
        Tensor4::from_vec(out)
    }
}

// ---- generated model contract ----------------------------------------------
// input  data: f32 [N,3,416,416] on Cpu; output grid: f32 [N,125,13,13] on Cpu.
pub struct TinyYoloV2;
impl TinyYoloV2 {
    // The typed signature; the *real* ORT call is in main.rs (this layer is
    // ort-free so defect cases type-check fast). N is the const batch.
    pub fn predict_ty<const N: usize>(
        _input: &Tensor4<f32, N, 3, 416, 416, Cpu, Unspecified>,
        out_data: Vec<f32>,
    ) -> Result<Tensor4<f32, N, 125, 13, 13, Cpu, Unspecified>, String> {
        Tensor4::from_vec(out_data)
    }
}
