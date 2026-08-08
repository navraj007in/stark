//! **AS6 — the tensor extension's builtin catalogue.**
//!
//! Exit criterion 2: *"central Core modules do not contain open-ended tensor spelling tables or
//! method catalogues."* Both lived in `resolve.rs` — thirty-three `name => Builtin::Tensor*` arms
//! and a thirty-three-arm membership test — so every new tensor operation grew Core's resolver.
//!
//! Moving them here does not change behaviour and is not meant to: the resolver still consults
//! them, and `LanguageOptions` still decides whether a session may use what they name. What
//! changes is **ownership** — the list of what the extension provides now lives with the extension,
//! and Core's resolver reads one call instead of carrying the catalogue.
//!
//! The `Builtin` variants themselves remain in `hir`, which is the larger relocation AS6 still
//! has open (175 match arms across the resolver, checker, interpreter and deployment lowering).
//! That move is bounded and scoped; this one is the half that could be finished without it.

use crate::hir::Builtin;

/// The extension's spelling table: a source name to the builtin it denotes.
///
/// Returns `None` for every Core name, so the resolver's own table decides those first and this is
/// consulted only as its fallthrough. Gating is unchanged and remains `LanguageOptions`' job —
/// this answers *what the extension provides*, never *whether this session may use it*.
pub fn builtin_named(name: &str) -> Option<Builtin> {
    match name {
        "zeros" => Some(Builtin::TensorZeros),
        "ones" => Some(Builtin::TensorOnes),
        "full" => Some(Builtin::TensorFull),
        "from_vec" => Some(Builtin::TensorFromVec),
        "add" => Some(Builtin::TensorAdd),
        "sub" => Some(Builtin::TensorSub),
        "mul" => Some(Builtin::TensorMul),
        "div" => Some(Builtin::TensorDiv),
        "min" => Some(Builtin::TensorMin),
        "max" => Some(Builtin::TensorMax),
        "eq" => Some(Builtin::TensorEq),
        "ne" => Some(Builtin::TensorNe),
        "lt" => Some(Builtin::TensorLt),
        "le" => Some(Builtin::TensorLe),
        "gt" => Some(Builtin::TensorGt),
        "ge" => Some(Builtin::TensorGe),
        "broadcast_to" => Some(Builtin::TensorBroadcastTo),
        "matmul" => Some(Builtin::TensorMatMul),
        "batch_matmul" => Some(Builtin::TensorBatchMatMul),
        "concat" => Some(Builtin::TensorConcat),
        "permute" => Some(Builtin::TensorPermute),
        "reshape" => Some(Builtin::TensorReshape),
        "slice_axis" => Some(Builtin::TensorSliceAxis),
        "transpose" => Some(Builtin::TensorTranspose),
        "sum_axis" => Some(Builtin::TensorSumAxis),
        "mean_axis" => Some(Builtin::TensorMeanAxis),
        "argmax" => Some(Builtin::TensorArgMax),
        "sum" => Some(Builtin::TensorSum),
        "softmax" => Some(Builtin::TensorSoftmax),
        "cast" => Some(Builtin::TensorCast),
        "to_device" => Some(Builtin::TensorToDevice),
        "scale_255" => Some(Builtin::TensorScale255),
        "normalize" => Some(Builtin::TensorNormalize),
        _ => None,
    }
}

/// Whether `b` is one of the extension's operations.
///
/// **Exhaustive by enumeration on purpose.** A `matches!` over a named list is what makes adding a
/// tensor builtin without classifying it here a visible omission rather than a silent `false` —
/// and a `false` here is the resolver deciding a tensor operation is Core, which is the leak
/// direction AS6's two-directional evidence exists to catch.
pub fn owns_builtin(b: Builtin) -> bool {
    matches!(
        b,
        Builtin::TensorZeros
            | Builtin::TensorOnes
            | Builtin::TensorFull
            | Builtin::TensorFromVec
            | Builtin::TensorAdd
            | Builtin::TensorSub
            | Builtin::TensorMul
            | Builtin::TensorDiv
            | Builtin::TensorMin
            | Builtin::TensorMax
            | Builtin::TensorEq
            | Builtin::TensorNe
            | Builtin::TensorLt
            | Builtin::TensorLe
            | Builtin::TensorGt
            | Builtin::TensorGe
            | Builtin::TensorBroadcastTo
            | Builtin::TensorMatMul
            | Builtin::TensorBatchMatMul
            | Builtin::TensorConcat
            | Builtin::TensorPermute
            | Builtin::TensorReshape
            | Builtin::TensorSliceAxis
            | Builtin::TensorTranspose
            | Builtin::TensorSumAxis
            | Builtin::TensorMeanAxis
            | Builtin::TensorArgMax
            | Builtin::TensorSum
            | Builtin::TensorSoftmax
            | Builtin::TensorCast
            | Builtin::TensorToDevice
            | Builtin::TensorScale255
            | Builtin::TensorNormalize
    )
}
