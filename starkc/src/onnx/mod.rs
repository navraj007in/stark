//! Bounded ONNX signature import and declaration verification (Gate 4 M4.5).
//!
//! Only graph input/output metadata is decoded. No graph node or initializer
//! data is interpreted or executed.

mod importer;
mod verifier;

pub use importer::{
    decode_signature, format_declaration, import_file, read_signature, DType, DecodeLimits,
    Dimension, ModelSignature, Port, DEFAULT_LIMITS,
};
pub use verifier::{verify_declaration_file, verify_declaration_source, VerificationReport};

use std::fmt;

/// A focused importer/verifier failure suitable for CLI presentation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnnxError {
    message: String,
}

impl OnnxError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for OnnxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for OnnxError {}
