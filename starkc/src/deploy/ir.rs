//! Deployment IR — a bounded, typed, backend-oriented representation of one
//! checked inference pipeline (Gate 5, `CLAUDE_GATE5_IMPLEMENTATION_PLAN.md`
//! §6).
//!
//! This is deliberately *not* a general STARK-to-Rust IR. It models exactly the
//! straight-line tensor/model operations the representative ResNet50 pipeline
//! needs; every other reachable construct is rejected during lowering with an
//! `E06xx` diagnostic (see the `lower` module). Each node keeps its originating
//! [`Span`] so the emitter and any later diagnostic can point back at source.

use crate::onnx::{DType as OnnxDType, ModelSignature};
use crate::source::Span;
use std::fmt;

pub use crate::extensions::tensor::types::DType;

/// SSA-style value handle, unique within a [`DeploymentFunction`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A fully-static tensor type as it appears inside the deployment pipeline.
/// The prototype requires every intermediate tensor to have a constant shape
/// (the batch dimension is fixed to 1 by `refine`); non-static shapes are a
/// lowering error, never a runtime guess.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorShape {
    pub dtype: DType,
    pub dims: Vec<u64>,
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor<{}, [", self.dtype.name())?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{d}")?;
        }
        f.write_str("]>")
    }
}

/// The type of a deployment value. Reduced from the Core/tensor `Ty`: only the
/// forms the bounded pipeline produces are representable.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeployTy {
    /// A statically-shaped tensor.
    Tensor(TensorShape),
    /// The type-erased boundary input handed in by the host (`TensorAny`).
    TensorAny,
    /// The opaque model handle (an `ort` session at runtime).
    Model,
    /// `Result<Tensor<..>, String>` — the fallible result of `refine` and the
    /// entry return type. `String` is the only error type in the prototype.
    Result(Box<DeployTy>),
}

impl fmt::Display for DeployTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeployTy::Tensor(t) => write!(f, "{t}"),
            DeployTy::TensorAny => f.write_str("TensorAny"),
            DeployTy::Model => f.write_str("Model"),
            DeployTy::Result(inner) => write!(f, "Result<{inner}, String>"),
        }
    }
}

/// A scalar constant fed to `full`. The literal source text is preserved so the
/// emitter reproduces the exact value without a lossy parse/format round-trip.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScalarLit {
    /// Digits as written in source, suffix stripped (e.g. `"0.485"`).
    pub text: String,
    pub dtype: DType,
}

/// One bounded operation. Every variant maps to a small, verified fragment of
/// generated Rust in the host emitter (M5.2); model inference itself becomes an
/// ONNX Runtime call.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DeployOp {
    /// `refine::<dtype, dims>()` on a `TensorAny`/`TensorDyn` — produces
    /// `Result<Tensor<dtype, dims>, String>`.
    Refine {
        src: ValueId,
        dtype: DType,
        dims: Vec<u64>,
    },
    /// `?` applied to a `Result` value — unwraps or early-returns the error.
    Try { src: ValueId },
    /// `permute::<perm>()`.
    Permute { src: ValueId, perm: Vec<u32> },
    /// `cast::<dtype>()` — saturating element cast.
    Cast { src: ValueId, dtype: DType },
    /// `full::<dtype, dims>(scalar)` — a broadcastable constant tensor.
    Full {
        dtype: DType,
        dims: Vec<u64>,
        scalar: ScalarLit,
    },
    /// `concat::<axis>(&lhs, &rhs)`.
    Concat {
        axis: u32,
        lhs: ValueId,
        rhs: ValueId,
    },
    /// Broadcasting elementwise `lhs.sub(&rhs)`.
    Sub { lhs: ValueId, rhs: ValueId },
    /// Broadcasting elementwise `lhs.div(&rhs)`.
    Div { lhs: ValueId, rhs: ValueId },
    /// `model.predict(&input)` — the ONNX Runtime session call.
    Predict { model: ValueId, input: ValueId },
    /// `softmax::<axis>()`.
    Softmax { src: ValueId, axis: u32 },
    /// `argmax::<axis>()`.
    ArgMax { src: ValueId, axis: u32 },
    /// A call to another lowered function (index into
    /// [`DeploymentProgram::functions`]).
    Call { callee: usize, args: Vec<ValueId> },
    /// `Ok(src)` — wraps the entry result.
    WrapOk { src: ValueId },
}

/// One straight-line binding: `result = op` at `span`, of static type `ty`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeployStmt {
    pub result: ValueId,
    pub op: DeployOp,
    pub ty: DeployTy,
    pub span: Span,
}

/// A lowered function: typed params, a straight-line body, and the value that
/// is its result (the tail expression).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeploymentFunction {
    pub name: String,
    pub params: Vec<DeployParam>,
    pub ret: DeployTy,
    pub body: Vec<DeployStmt>,
    /// The value produced as the function's result (its tail).
    pub result: ValueId,
    pub span: Span,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeployParam {
    pub name: String,
    pub ty: DeployTy,
    pub value: ValueId,
}

/// The selected model, with its port signature taken from the *verified* ONNX
/// artifact (authoritative and identical to what the runtime session exposes),
/// plus its STARK declaration name and the artifact hash bound into the build.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeployModel {
    /// STARK model type name, e.g. `Resnet50V17`.
    pub type_name: String,
    pub input: DeployPort,
    pub output: DeployPort,
    /// Lowercase hex SHA-256 of the exact artifact bytes.
    pub artifact_sha256: String,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeployPort {
    pub name: String,
    pub dtype: OnnxDType,
    /// `None` marks a dynamic (batch) dimension.
    pub dims: Vec<Option<u64>>,
}

impl DeployPort {
    pub(crate) fn from_onnx(port: &crate::onnx::Port) -> DeployPort {
        DeployPort {
            name: port.name.clone(),
            dtype: port.dtype,
            dims: port
                .dimensions
                .iter()
                .map(|d| match d {
                    crate::onnx::Dimension::Static(n) => Some(*n),
                    crate::onnx::Dimension::Dynamic { .. } => None,
                })
                .collect(),
        }
    }
}

/// The whole checked pipeline, ready for deterministic emission (M5.2).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeploymentProgram {
    pub compiler_version: String,
    pub entry: String,
    pub model: DeployModel,
    /// Reachable functions, entry first, then callees in discovery order.
    pub functions: Vec<DeploymentFunction>,
}

impl DeploymentProgram {
    pub fn entry_fn(&self) -> &DeploymentFunction {
        &self.functions[0]
    }
}

/// Build a [`DeployModel`] from a verified signature + artifact hash.
pub(crate) fn model_from_signature(
    type_name: String,
    signature: &ModelSignature,
    artifact_sha256: String,
) -> DeployModel {
    DeployModel {
        type_name,
        input: DeployPort::from_onnx(&signature.inputs[0]),
        output: DeployPort::from_onnx(&signature.outputs[0]),
        artifact_sha256,
    }
}
