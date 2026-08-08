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

/// **The extension's operation set (AS6 exit criterion 2).**
///
/// These were thirty-three variants of Core's `hir::Builtin`, so the Core enum *was* the
/// extension's method catalogue and every new tensor operation widened it. `Builtin` now holds one
/// tensor-owned variant — `Builtin::Tensor(TensorBuiltin)` — and this is the sealed set behind it.
///
/// **Sealing it is what creates the forcing function.** `owns_builtin`'s old `matches!` list was
/// explicit but not compiler-exhaustive: adding a `Builtin::TensorFoo` compiled and answered
/// `false`, and only the AS6 isolation tests would have noticed. Every match on `TensorBuiltin` in
/// this module is exhaustive, so a new operation now fails to compile until it is classified —
/// which is the structural mechanism the packet exists to establish.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorBuiltin {
    Zeros,
    Ones,
    Full,
    FromVec,
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    BroadcastTo,
    MatMul,
    BatchMatMul,
    Concat,
    Permute,
    Reshape,
    SliceAxis,
    Transpose,
    SumAxis,
    MeanAxis,
    ArgMax,
    Sum,
    Softmax,
    Cast,
    ToDevice,
    Scale255,
    Normalize,
}

impl TensorBuiltin {
    /// The operation's source spelling, which the checker's tensor rules key on.
    ///
    /// Exhaustive: Core used to hold this table too (a second thirty-three-arm block feeding
    /// `check_tensor_op`), which is the same criterion-2 shape as the resolver's.
    pub fn op_name(self) -> &'static str {
        match self {
            TensorBuiltin::Zeros => "zeros",
            TensorBuiltin::Ones => "ones",
            TensorBuiltin::Full => "full",
            TensorBuiltin::FromVec => "from_vec",
            TensorBuiltin::Add => "add",
            TensorBuiltin::Sub => "sub",
            TensorBuiltin::Mul => "mul",
            TensorBuiltin::Div => "div",
            TensorBuiltin::Min => "min",
            TensorBuiltin::Max => "max",
            TensorBuiltin::Eq => "eq",
            TensorBuiltin::Ne => "ne",
            TensorBuiltin::Lt => "lt",
            TensorBuiltin::Le => "le",
            TensorBuiltin::Gt => "gt",
            TensorBuiltin::Ge => "ge",
            TensorBuiltin::BroadcastTo => "broadcast_to",
            TensorBuiltin::MatMul => "matmul",
            TensorBuiltin::BatchMatMul => "batch_matmul",
            TensorBuiltin::Concat => "concat",
            TensorBuiltin::Permute => "permute",
            TensorBuiltin::Reshape => "reshape",
            TensorBuiltin::SliceAxis => "slice_axis",
            TensorBuiltin::Transpose => "transpose",
            TensorBuiltin::SumAxis => "sum_axis",
            TensorBuiltin::MeanAxis => "mean_axis",
            TensorBuiltin::ArgMax => "argmax",
            TensorBuiltin::Sum => "sum",
            TensorBuiltin::Softmax => "softmax",
            TensorBuiltin::Cast => "cast",
            TensorBuiltin::ToDevice => "to_device",
            TensorBuiltin::Scale255 => "scale_255",
            TensorBuiltin::Normalize => "normalize",
        }
    }
}

/// The extension's spelling table: a source name to the builtin it denotes.
///
/// Returns `None` for every Core name, so the resolver's own table decides those first and this is
/// consulted only as its fallthrough. Gating is unchanged and remains `LanguageOptions`' job —
/// this answers *what the extension provides*, never *whether this session may use it*.
pub fn builtin_named(name: &str) -> Option<Builtin> {
    match name {
        "zeros" => Some(Builtin::Tensor(TensorBuiltin::Zeros)),
        "ones" => Some(Builtin::Tensor(TensorBuiltin::Ones)),
        "full" => Some(Builtin::Tensor(TensorBuiltin::Full)),
        "from_vec" => Some(Builtin::Tensor(TensorBuiltin::FromVec)),
        "add" => Some(Builtin::Tensor(TensorBuiltin::Add)),
        "sub" => Some(Builtin::Tensor(TensorBuiltin::Sub)),
        "mul" => Some(Builtin::Tensor(TensorBuiltin::Mul)),
        "div" => Some(Builtin::Tensor(TensorBuiltin::Div)),
        "min" => Some(Builtin::Tensor(TensorBuiltin::Min)),
        "max" => Some(Builtin::Tensor(TensorBuiltin::Max)),
        "eq" => Some(Builtin::Tensor(TensorBuiltin::Eq)),
        "ne" => Some(Builtin::Tensor(TensorBuiltin::Ne)),
        "lt" => Some(Builtin::Tensor(TensorBuiltin::Lt)),
        "le" => Some(Builtin::Tensor(TensorBuiltin::Le)),
        "gt" => Some(Builtin::Tensor(TensorBuiltin::Gt)),
        "ge" => Some(Builtin::Tensor(TensorBuiltin::Ge)),
        "broadcast_to" => Some(Builtin::Tensor(TensorBuiltin::BroadcastTo)),
        "matmul" => Some(Builtin::Tensor(TensorBuiltin::MatMul)),
        "batch_matmul" => Some(Builtin::Tensor(TensorBuiltin::BatchMatMul)),
        "concat" => Some(Builtin::Tensor(TensorBuiltin::Concat)),
        "permute" => Some(Builtin::Tensor(TensorBuiltin::Permute)),
        "reshape" => Some(Builtin::Tensor(TensorBuiltin::Reshape)),
        "slice_axis" => Some(Builtin::Tensor(TensorBuiltin::SliceAxis)),
        "transpose" => Some(Builtin::Tensor(TensorBuiltin::Transpose)),
        "sum_axis" => Some(Builtin::Tensor(TensorBuiltin::SumAxis)),
        "mean_axis" => Some(Builtin::Tensor(TensorBuiltin::MeanAxis)),
        "argmax" => Some(Builtin::Tensor(TensorBuiltin::ArgMax)),
        "sum" => Some(Builtin::Tensor(TensorBuiltin::Sum)),
        "softmax" => Some(Builtin::Tensor(TensorBuiltin::Softmax)),
        "cast" => Some(Builtin::Tensor(TensorBuiltin::Cast)),
        "to_device" => Some(Builtin::Tensor(TensorBuiltin::ToDevice)),
        "scale_255" => Some(Builtin::Tensor(TensorBuiltin::Scale255)),
        "normalize" => Some(Builtin::Tensor(TensorBuiltin::Normalize)),
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
    matches!(b, Builtin::Tensor(_))
}
