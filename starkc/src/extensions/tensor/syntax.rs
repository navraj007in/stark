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

/// The contextual keyword introducing a model declaration.
///
/// Contextual, not reserved: `model` stays usable as an ordinary identifier in expression
/// position, which is why the parser pairs this with an `Ident Ident` lookahead rather than
/// treating the spelling alone as an item start.
pub(crate) const MODEL_KEYWORD: &str = "model";

/// The contextual keyword introducing a model port, and the direction it declares.
pub(crate) fn port_direction(name: &str) -> Option<PortDir> {
    match name {
        "input" => Some(PortDir::Input),
        "output" => Some(PortDir::Output),
        _ => None,
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
