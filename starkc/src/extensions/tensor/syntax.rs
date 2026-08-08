//! **AS6 packet 4C — the tensor extension's surface spellings.**
//!
//! Exit criterion 2: *"central Core modules do not contain open-ended tensor spelling tables or
//! method catalogues."* The audit of `parser.rs` found its tensor references fall into four
//! groups, and only one of them is a boundary violation:
//!
//! ```text
//! enablement           `tensor_enabled()` gates                      stays — that IS the mechanism
//! grammar recognition  ShapeGroupKind, the dim_* precedence family,  stays — parsing is dispatch
//!                      generic_args' three-way bracket dispatch
//! AST construction     ShapeArg, DimExprKind, ModelDef, ModelPort    stays — ast-owned node kinds
//! semantic knowledge   21 hardcoded tensor SPELLINGS at 5 sites      moves — this file
//! ```
//!
//! The parser keeps every decision about *shape*: how a `[...]` group is disambiguated, how a
//! dimension expression associates, where a model's ports may appear. What it no longer keeps is
//! the extension's **vocabulary** — which identifiers name a tensor dtype, which constructor opens
//! a shape position, which contextual keywords introduce a model, and which names `tensor` v0.1
//! has reserved for later versions.
//!
//! This is the same cut as [`super::builtins`] (the resolver's operation catalogue) and
//! [`super::rules`] (the checker's operation table), applied to the parser's surface syntax. Its
//! effect is the same: adding a reserved dtype or a second shape-carrying constructor is now an
//! edit inside `extensions/tensor/`, not in Core's `parse_type`.
//!
//! **What this deliberately does not do** is introduce a parser plugin architecture. AS6 exit
//! criterion 4 forbids a public extension API, and the packet's own test is whether a new tensor
//! syntax form requires edits in many unrelated places — it does not, and centralised syntax
//! dispatch is the right structure for a parser.

use crate::ast::{PortDir, Primitive};
use crate::extensions::tensor::types::{DType, ValueRange};

/// The contextual keyword introducing a model declaration.
///
/// Contextual, not reserved: `model` stays usable as an ordinary identifier in expression
/// position, which is why the parser pairs this with an `Ident Ident` lookahead rather than
/// treating the spelling alone as an item start.
pub(crate) const MODEL_KEYWORD: &str = "model";

/// The contextual keyword introducing a model port, and the direction it declares.
///
/// Paired with [`port_keyword`] and derived from it, so the parse and print spellings cannot
/// drift: `ast::PortDir::keyword` delegates here rather than carrying a second copy.
pub(crate) fn port_direction(name: &str) -> Option<PortDir> {
    [PortDir::Input, PortDir::Output]
        .into_iter()
        .find(|dir| port_keyword(*dir) == name)
}

/// The spelling a port direction prints as. The single source of truth for both directions of the
/// mapping — the parser matches against it and the formatter writes it.
pub(crate) fn port_keyword(dir: PortDir) -> &'static str {
    match dir {
        PortDir::Input => "input",
        PortDir::Output => "output",
    }
}

/// Element-type identifiers the extension adds to Core's primitive set (§D3).
///
/// They lex as ordinary identifiers, so in a Core-only session they are left to normal name
/// resolution and a Core program may still use them as ordinary type names. The parser therefore
/// consults this only behind its `tensor_enabled()` gate.
pub(crate) fn extension_primitive(name: &str) -> Option<Primitive> {
    match name {
        "Float16" => Some(Primitive::Float16),
        "BFloat16" => Some(Primitive::BFloat16),
        _ => None,
    }
}

/// Whether a type constructor's generic argument list opens a **shape position**.
///
/// `[T]` is valid Core slice syntax and is otherwise lexically indistinguishable from a rank-one
/// symbolic shape, so only an extension-owned constructor establishes a shape position at parse
/// time; all other bare identifiers remain Core types until a later semantic signature says
/// otherwise. Today that is exactly one constructor, and the point of naming it here is that a
/// second one would be added here rather than in Core's `parse_type`.
pub(crate) fn opens_shape_position(name: &str) -> bool {
    name == "Tensor"
}

/// Type names `tensor` v0.1 reserves for later versions, with the note to emit.
///
/// This is the open-ended spelling table exit criterion 2 names: every future dtype, layout,
/// deployment constraint or autodiff type added to the reservation list used to widen a `match` in
/// Core's `parse_type`. Reserving a name is a statement about the extension's roadmap, not about
/// Core's grammar — the forms parse identically either way.
pub(crate) fn reserved_type_note(name: &str) -> Option<&'static str> {
    match name {
        "QInt8" | "QUInt8" | "QInt16" | "Quantized" => {
            Some("quantized dtypes are reserved in `tensor` v0.1")
        }
        "NCHW" | "NHWC" | "RowMajor" | "ColumnMajor" | "TensorLayout" => {
            Some("memory layout types are reserved in `tensor` v0.1")
        }
        "PeakMemory" | "MemoryProfile" => {
            Some("peak-memory deployment constraints are reserved in `tensor` v0.1")
        }
        "Gradient" | "Grad" | "Tape" | "Autodiff" => {
            Some("training and autodiff types are reserved in `tensor` v0.1")
        }
        _ => None,
    }
}

/// Single-segment names the `tensor` extension owns: element types, tensor and device type
/// constructors, value-range states, and the `Dim`/`DType`/`Device` kinds — each with the phrase
/// naming it in a diagnostic.
///
/// The resolver uses this in both directions, which is why it is a *table* and not a predicate: in
/// a Core-only session it produces the focused "the `Tensor` type requires extension `tensor`"
/// diagnostic, and under the extension it suppresses "undefined type" for the same names. Either
/// way the list of what the extension owns is the extension's to state.
///
/// This is the second half of the resolver cut fe80129 began. That commit moved the *operation*
/// catalogue (`builtin_named`/`owns_builtin`); this is the *type-name* vocabulary, which stayed
/// behind and which AS6 exit qualification found still in `resolve.rs`.
pub(crate) fn extension_type_name(name: &str) -> Option<&'static str> {
    match name {
        "Dim" => Some("`Dim` kind"),
        "DType" => Some("`DType` kind"),
        "Device" => Some("`Device` kind"),
        "Float16" => Some("`Float16` element type"),
        "BFloat16" => Some("`BFloat16` element type"),
        "Tensor" => Some("`Tensor` type"),
        "TensorDyn" => Some("`TensorDyn` type"),
        "TensorAny" => Some("`TensorAny` type"),
        "Cpu" => Some("`Cpu` device type"),
        "Cuda" => Some("`Cuda` device type"),
        "ByteRange" => Some("`ByteRange` value range"),
        "UnitRange" => Some("`UnitRange` value range"),
        "Normalized" => Some("`Normalized` value range"),
        "Unspecified" => Some("`Unspecified` value range"),
        "ModelError" => Some("`ModelError` type"),
        _ => None,
    }
}

/// The kind a tensor generic parameter may be declared with (§3.1).
///
/// Extension-owned, and deliberately *not* Core's `GenericKind`: Core's enum also carries `Type`,
/// the ordinary case, and owning that is Core's business. This is the set of kinds the extension
/// adds, so a fourth one is added here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorParamKind {
    Dim,
    DType,
    Device,
}

/// Classify a generic parameter bound's spelling as a tensor kind.
pub(crate) fn tensor_param_kind(name: &str) -> Option<TensorParamKind> {
    match name {
        "Dim" => Some(TensorParamKind::Dim),
        "DType" => Some(TensorParamKind::DType),
        "Device" => Some(TensorParamKind::Device),
        _ => None,
    }
}

/// What a generic parameter carrying a tensor kind is allowed to look like. The phrase enumerates
/// the vocabulary, so it belongs with the vocabulary — a fourth kind must not leave Core reciting
/// a list of three.
pub(crate) const TENSOR_PARAM_KIND_EXPECTATION: &str =
    "tensor kind parameters must have exactly one of `Dim`, `DType`, or `Device` and no trait bounds";

/// A device type constructor (§8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeviceConstructor {
    /// Takes no arguments.
    Cpu,
    /// Takes an ordinal, `Cuda<N>`.
    Cuda,
}

/// Classify a single-segment type name as a device constructor.
pub(crate) fn device_constructor(name: &str) -> Option<DeviceConstructor> {
    match name {
        "Cpu" => Some(DeviceConstructor::Cpu),
        "Cuda" => Some(DeviceConstructor::Cuda),
        _ => None,
    }
}

/// The phrase naming what may appear in device position.
pub(crate) const DEVICE_EXPECTATION: &str =
    "unknown tensor device; expected `Cpu`, `Cuda<N>`, or a `Device` parameter";

/// Classify a single-segment type name as a value-range state (§9). The states are a fixed, closed
/// set; an unknown name is a tensor error.
pub(crate) fn value_range_state(name: &str) -> Option<ValueRange> {
    match name {
        "Unspecified" => Some(ValueRange::Unspecified),
        "ByteRange" => Some(ValueRange::ByteRange),
        "UnitRange" => Some(ValueRange::UnitRange),
        "Normalized" => Some(ValueRange::Normalized),
        _ => None,
    }
}

/// The phrase naming the value-range states.
pub(crate) const VALUE_RANGE_EXPECTATION: &str =
    "unknown value range; expected `ByteRange`, `UnitRange`, `Normalized`, or `Unspecified`";

/// A tensor type constructor, as written in source.
///
/// AS6 packet 4D-D: this spelling set had **three** copies — Core's `build_tensor_type`, the
/// deployment lowering's `deploy_ty_from_ast`, and the display paths. The set of type constructors
/// the extension provides is the extension's to state, once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorTypeConstructor {
    /// `TensorAny` — no arguments.
    TensorAny,
    /// `TensorDyn<T>` — element type only, shape unknown.
    TensorDyn,
    /// `Tensor<T, [..], device = D, range = R>` — two to four arguments.
    Tensor,
    /// `ModelError` — no arguments.
    ModelError,
}

impl TensorTypeConstructor {
    /// The spelling. Exhaustive, so a new constructor cannot reach a diagnostic unnamed.
    pub(crate) fn name(self) -> &'static str {
        match self {
            TensorTypeConstructor::TensorAny => "TensorAny",
            TensorTypeConstructor::TensorDyn => "TensorDyn",
            TensorTypeConstructor::Tensor => "Tensor",
            TensorTypeConstructor::ModelError => "ModelError",
        }
    }
}

/// Classify a single-segment type name as a tensor type constructor.
pub(crate) fn tensor_type_constructor(name: &str) -> Option<TensorTypeConstructor> {
    [
        TensorTypeConstructor::TensorAny,
        TensorTypeConstructor::TensorDyn,
        TensorTypeConstructor::Tensor,
        TensorTypeConstructor::ModelError,
    ]
    .into_iter()
    .find(|constructor| constructor.name() == name)
}

/// The element-type name table, in the parsing direction.
///
/// Covers Core's primitive dtype spellings as well as the two the extension adds, because a
/// `DType` is named by whichever of them appears in source. `Primitive::name` is the printing
/// direction of the same mapping and this is derived from it, so the two cannot drift.
pub(crate) fn dtype_by_name(name: &str) -> Option<DType> {
    let primitive = [
        Primitive::Int8,
        Primitive::Int16,
        Primitive::Int32,
        Primitive::Int64,
        Primitive::UInt8,
        Primitive::UInt16,
        Primitive::UInt32,
        Primitive::UInt64,
        Primitive::Float16,
        Primitive::BFloat16,
        Primitive::Float32,
        Primitive::Float64,
        Primitive::Bool,
    ]
    .into_iter()
    .find(|p| p.name() == name)?;
    dtype_of_primitive(primitive)
}

/// The `DType` a Core primitive denotes, for the primitives that are valid tensor element types.
pub(crate) fn dtype_of_primitive(primitive: Primitive) -> Option<DType> {
    Some(match primitive {
        Primitive::Int8 => DType::Int8,
        Primitive::Int16 => DType::Int16,
        Primitive::Int32 => DType::Int32,
        Primitive::Int64 => DType::Int64,
        Primitive::UInt8 => DType::UInt8,
        Primitive::UInt16 => DType::UInt16,
        Primitive::UInt32 => DType::UInt32,
        Primitive::UInt64 => DType::UInt64,
        Primitive::Float16 => DType::Float16,
        Primitive::BFloat16 => DType::BFloat16,
        Primitive::Float32 => DType::Float32,
        Primitive::Float64 => DType::Float64,
        Primitive::Bool => DType::Bool,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two element-type spellings appear in two tables — one maps them to a Core `Primitive`
    /// for the parser, the other describes them for the resolver's diagnostic. They are allowed to
    /// serve different purposes; they are not allowed to disagree about which names exist.
    #[test]
    fn every_extension_primitive_is_a_known_extension_type_name() {
        for name in ["Float16", "BFloat16"] {
            assert!(
                extension_primitive(name).is_some(),
                "{name} should map to a primitive"
            );
            assert!(
                extension_type_name(name).is_some(),
                "{name} maps to a primitive but is not listed as an extension type name"
            );
        }
    }

    /// A reserved *future* name and an *owned current* name are different claims, and a name that
    /// is both would produce two diagnostics for one identifier.
    #[test]
    fn reserved_and_owned_name_tables_are_disjoint() {
        for name in [
            "QInt8",
            "QUInt8",
            "QInt16",
            "Quantized",
            "NCHW",
            "NHWC",
            "RowMajor",
            "ColumnMajor",
            "TensorLayout",
            "PeakMemory",
            "MemoryProfile",
            "Gradient",
            "Grad",
            "Tape",
            "Autodiff",
        ] {
            assert!(
                reserved_type_note(name).is_some(),
                "{name} should be reserved"
            );
            assert!(
                extension_type_name(name).is_none(),
                "{name} is both reserved for a future version and owned today"
            );
        }
    }

    /// Every classifier's vocabulary must also be a name the extension declares it owns, or the
    /// resolver will report "undefined type" for a name the checker then classifies.
    #[test]
    fn classifier_vocabularies_are_declared_extension_names() {
        for name in ["Dim", "DType", "Device"] {
            assert!(tensor_param_kind(name).is_some());
            assert!(
                extension_type_name(name).is_some(),
                "{name} is a tensor kind but is not a declared extension name"
            );
        }
        for name in ["Cpu", "Cuda"] {
            assert!(device_constructor(name).is_some());
            assert!(
                extension_type_name(name).is_some(),
                "{name} is a device constructor but is not a declared extension name"
            );
        }
        for name in ["Unspecified", "ByteRange", "UnitRange", "Normalized"] {
            assert!(value_range_state(name).is_some());
            assert!(
                extension_type_name(name).is_some(),
                "{name} is a value-range state but is not a declared extension name"
            );
        }
    }

    /// Every tensor type constructor must also be a declared extension name, or the resolver will
    /// call it undefined and the checker will then build it.
    #[test]
    fn type_constructors_are_declared_extension_names() {
        for constructor in [
            TensorTypeConstructor::TensorAny,
            TensorTypeConstructor::TensorDyn,
            TensorTypeConstructor::Tensor,
            TensorTypeConstructor::ModelError,
        ] {
            assert_eq!(
                tensor_type_constructor(constructor.name()),
                Some(constructor)
            );
            assert!(
                extension_type_name(constructor.name()).is_some(),
                "{} is a type constructor but is not a declared extension name",
                constructor.name()
            );
        }
    }

    /// The parse and print directions of the dtype-name mapping are one table read two ways.
    #[test]
    fn dtype_names_round_trip() {
        for name in [
            "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float16",
            "BFloat16", "Float32", "Float64", "Bool",
        ] {
            assert!(dtype_by_name(name).is_some(), "{name} should name a dtype");
        }
        assert_eq!(dtype_by_name("String"), None);
        for name in ["Float16", "BFloat16"] {
            let primitive = extension_primitive(name).expect("extension primitive");
            assert_eq!(dtype_by_name(name), dtype_of_primitive(primitive));
        }
    }

    #[test]
    fn port_spellings_round_trip() {
        for dir in [PortDir::Input, PortDir::Output] {
            assert_eq!(port_direction(port_keyword(dir)), Some(dir));
        }
        assert_eq!(port_direction("inputs"), None);
    }
}
