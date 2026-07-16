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
// The factors 5 and 25 are baked in; `125 == 5*25` is not derived from the input
// by a general rule (that needs generic_const_exprs). A wrong *declared* target
// is still a compile-time type error because this function's return type is fixed.
pub fn split_channels<const N: usize, const H: usize, const W: usize, Dev, R>(
    t: Tensor4<f32, N, 125, H, W, Dev, R>,
) -> Tensor5<f32, N, 5, 25, H, W, Dev, R> {
    Tensor5 { data: t.data, _dev: PhantomData, _range: PhantomData }
}

// ---- the bespoke permute [0,1,3,4,2]: [N,5,25,H,W] -> [N,5,H,W,25] ----------
pub fn permute_grid<const N: usize, const A: usize, const K: usize, const H: usize, const W: usize, Dev, R>(
    t: Tensor5<f32, N, A, K, H, W, Dev, R>,
) -> Tensor5<f32, N, A, H, W, K, Dev, R> {
    // A real permute reindexes the data; the type-level reordering is what we
    // are measuring. (run.py checks numerical output via main.rs, which does the
    // actual reindex.)
    Tensor5 { data: t.data, _dev: PhantomData, _range: PhantomData }
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
