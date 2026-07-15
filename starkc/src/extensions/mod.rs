//! Optional, non-Core language extensions (Gate 4+).
//!
//! Extensions are gated behind [`crate::options::LanguageOptions`] and keep
//! their types and rules out of the Core representation as far as practical.
//! The only extension today is `tensor` v0.1
//! (`STARKLANG/docs/extensions/Tensor-Model-Types.md`).

pub mod tensor;
