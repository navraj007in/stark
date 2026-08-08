//! **AS6 packet 4B, group 1 — the tensor operation rules, moved untouched.**
//!
//! Packet 4A's inventory found these are *standalone types and a table*, not methods on
//! `TypeChecker`: the schema, dtype, device, shape and result rules, the per-operation descriptor,
//! the `TENSOR_OPS` table itself, and the broadcast failure kind. Nothing held them in Core but the
//! file they sat in — they reference no checker state and call no checker service.
//!
//! So this is the part of the extraction with no design question attached. `check_tensor_op` still
//! lives in the checker and reads `TENSOR_OPS` across the boundary; group 2 moves it, and the
//! `pub(crate)` visibility here is what lets the two land in separate commits rather than one
//! unreviewable change.

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorGenericSchema {
    None,
    DTypeAndShape,
    DTypeAndDim,
    Shape,
    Axis,
    AxisStartLen,
    DType,
    Device,
    IndexList,
}

impl TensorGenericSchema {
    pub(crate) const fn arity(self) -> usize {
        match self {
            TensorGenericSchema::None => 0,
            TensorGenericSchema::DTypeAndShape | TensorGenericSchema::DTypeAndDim => 2,
            TensorGenericSchema::Shape
            | TensorGenericSchema::Axis
            | TensorGenericSchema::DType
            | TensorGenericSchema::Device
            | TensorGenericSchema::IndexList => 1,
            TensorGenericSchema::AxisStartLen => 3,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorDTypeRule {
    Construct,
    Match,
    Compare,
    Preserve,
    ArgMax,
    Cast,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorDeviceRule {
    Fresh,
    Match,
    Preserve,
    Target,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorShapeRule {
    Construct,
    FromVec,
    Elementwise,
    BroadcastTo,
    MatMul,
    BatchMatMul,
    Concat,
    Permute,
    Reshape,
    SliceAxis,
    Transpose,
    ReduceAxis,
    FullReduce,
    Softmax,
    Cast,
    ToDevice,
    /// A value-range transition (Gate 7): identity shape/dtype, requires the
    /// receiver to already be in `from`, and produces `to`.
    RangeTransition {
        from: crate::extensions::tensor::types::ValueRange,
        to: crate::extensions::tensor::types::ValueRange,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorResultRule {
    Tensor,
    BoolTensor,
    Int64Tensor,
    FallibleTensor,
}

#[derive(Clone, Copy)]
pub(crate) struct TensorOpDescriptor {
    pub(crate) name: &'static str,
    pub(crate) arity: usize,
    pub(crate) standalone: bool,
    pub(crate) method: bool,
    pub(crate) generics: TensorGenericSchema,
    pub(crate) dtype: TensorDTypeRule,
    pub(crate) device: TensorDeviceRule,
    pub(crate) shape: TensorShapeRule,
    pub(crate) result: TensorResultRule,
}

macro_rules! tensor_op {
    ($name:literal, $arity:literal, $method:literal, $generics:expr, $dtype:expr, $device:expr, $shape:expr, $result:expr $(,)?) => {
        TensorOpDescriptor {
            name: $name,
            arity: $arity,
            standalone: true,
            method: $method,
            generics: $generics,
            dtype: $dtype,
            device: $device,
            shape: $shape,
            result: $result,
        }
    };
}

pub(crate) static TENSOR_OPS: &[TensorOpDescriptor] = &[
    tensor_op!(
        "zeros",
        0,
        false,
        TensorGenericSchema::DTypeAndShape,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::Construct,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "ones",
        0,
        false,
        TensorGenericSchema::DTypeAndShape,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::Construct,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "full",
        1,
        false,
        TensorGenericSchema::DTypeAndShape,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::Construct,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "from_vec",
        1,
        false,
        TensorGenericSchema::DTypeAndDim,
        TensorDTypeRule::Construct,
        TensorDeviceRule::Fresh,
        TensorShapeRule::FromVec,
        TensorResultRule::FallibleTensor,
    ),
    tensor_op!(
        "add",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "sub",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "mul",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "div",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "min",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "max",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "eq",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "ne",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "lt",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "le",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "gt",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "ge",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Compare,
        TensorDeviceRule::Match,
        TensorShapeRule::Elementwise,
        TensorResultRule::BoolTensor,
    ),
    tensor_op!(
        "broadcast_to",
        1,
        true,
        TensorGenericSchema::Shape,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::BroadcastTo,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "matmul",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::MatMul,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "batch_matmul",
        2,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::BatchMatMul,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "concat",
        2,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Match,
        TensorDeviceRule::Match,
        TensorShapeRule::Concat,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "permute",
        1,
        true,
        TensorGenericSchema::IndexList,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Permute,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "reshape",
        1,
        true,
        TensorGenericSchema::Shape,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Reshape,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "slice_axis",
        1,
        true,
        TensorGenericSchema::AxisStartLen,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::SliceAxis,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "transpose",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Transpose,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "sum_axis",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::ReduceAxis,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "mean_axis",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::ReduceAxis,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "argmax",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::ArgMax,
        TensorDeviceRule::Preserve,
        TensorShapeRule::ReduceAxis,
        TensorResultRule::Int64Tensor,
    ),
    tensor_op!(
        "sum",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::FullReduce,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "softmax",
        1,
        true,
        TensorGenericSchema::Axis,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Softmax,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "cast",
        1,
        true,
        TensorGenericSchema::DType,
        TensorDTypeRule::Cast,
        TensorDeviceRule::Preserve,
        TensorShapeRule::Cast,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "to_device",
        1,
        true,
        TensorGenericSchema::Device,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Target,
        TensorShapeRule::ToDevice,
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "scale_255",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::RangeTransition {
            from: crate::extensions::tensor::types::ValueRange::ByteRange,
            to: crate::extensions::tensor::types::ValueRange::UnitRange,
        },
        TensorResultRule::Tensor,
    ),
    tensor_op!(
        "normalize",
        1,
        true,
        TensorGenericSchema::None,
        TensorDTypeRule::Preserve,
        TensorDeviceRule::Preserve,
        TensorShapeRule::RangeTransition {
            from: crate::extensions::tensor::types::ValueRange::UnitRange,
            to: crate::extensions::tensor::types::ValueRange::Normalized,
        },
        TensorResultRule::Tensor,
    ),
];

/// Why an explicit `broadcast_to` failed: a rank mismatch, or a specific
/// target-aligned axis that cannot be expanded to the target dimension.
pub(crate) enum BroadcastError {
    Rank { source: usize, target: usize },
    Axis { result_axis: usize },
}
