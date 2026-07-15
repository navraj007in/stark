//! The `tensor` v0.1 extension: dimension algebra, tensor/model types, and
//! (M4.5) ONNX import. Kept isolated from the Core type representation behind a
//! registration/capability boundary so a Core-only checker never needs to know
//! about tensor constructors.

pub mod dim;
pub mod types;
