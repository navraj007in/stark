//! WP-C7.8.2c — the MIR signature a provider call must present, derived from its validated
//! `FunctionDecl`.
//!
//! **The physical ABI is the model.** Native Provider ABI v0.1 §11 makes `ProviderStatus` the
//! return of *every* provider function, and §6.1 maps each `AbiParam` to exactly one C parameter.
//! MIR mirrors that mapping one-for-one rather than inventing a STARK-shaped signature: the call's
//! `dest` receives the status code, and every declared parameter — including output slots — is an
//! argument.
//!
//! Deriving the signature *is* A10 §4 invariant 5 ("MIR argument types, mutability, ownership and
//! ABI shapes match `FunctionDecl`"): once the expected parameter list exists, the verifier's
//! existing arity and per-argument type checks enforce it with no separate rule.

use super::MirTy;
use crate::provider_abi::{AbiParam, ScalarTy};

/// The MIR type of a provider call's destination: ABI §11's `ProviderStatus.code`.
///
/// `UInt32` because the status is a `u32` code, not a STARK `Result`. Converting a status into a
/// STARK `Result::Err` is the *binding layer's* job (Packet 1 §1.2), and doing it here would
/// collapse the three failure channels into one at exactly the layer that must keep them apart.
pub const PROVIDER_STATUS_TY: MirTy = MirTy::UInt32;

/// An `AbiParam` form MIR cannot yet type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnmappedParam {
    /// A handle-carrying parameter. Typing one requires binding a provider `resource_type` string
    /// to a `MirTy`, which arrives with the first resource-bearing capability (C7.8.4's `File`).
    /// Until then a resource-typed provider call is refused rather than guessed at — guessing
    /// would mean inventing a MIR type for a resource whose identity the compiler does not yet
    /// know, and ABI §11.1's `resource_type` validation would have nothing to check against.
    ResourceHandle { index: usize, resource_type: String },
}

fn scalar_ty(t: ScalarTy) -> MirTy {
    match t {
        ScalarTy::U8 => MirTy::UInt8,
        ScalarTy::U16 => MirTy::UInt16,
        ScalarTy::U32 => MirTy::UInt32,
        ScalarTy::U64 => MirTy::UInt64,
        ScalarTy::I8 => MirTy::Int8,
        ScalarTy::I16 => MirTy::Int16,
        ScalarTy::I32 => MirTy::Int32,
        ScalarTy::I64 => MirTy::Int64,
        ScalarTy::Bool => MirTy::Bool,
        ScalarTy::F32 => MirTy::Float32,
        ScalarTy::F64 => MirTy::Float64,
    }
}

fn byte_slice(mutable: bool) -> MirTy {
    MirTy::Ref {
        mutable,
        inner: Box::new(MirTy::Slice(Box::new(MirTy::UInt8))),
    }
}

/// One `AbiParam`'s MIR argument type, following ABI §6.1's physical mapping exactly.
///
/// `ScalarOut` and `ScalarInOut` map to the same MIR type (`&mut T`) and are deliberately *not*
/// distinguished here — §6.1 says the same of their C forms, both `*mut T`, and records that the
/// difference is an initialisation contract (§11.1) rather than a type. That contract is a
/// verifier rule about *reads on failure paths*, not something a signature can express.
pub fn param_ty(index: usize, param: &AbiParam) -> Result<MirTy, UnmappedParam> {
    Ok(match param {
        AbiParam::ScalarIn(t) => scalar_ty(*t),
        AbiParam::ScalarOut(t) | AbiParam::ScalarInOut(t) => MirTy::Ref {
            mutable: true,
            inner: Box::new(scalar_ty(*t)),
        },
        AbiParam::BufferIn => byte_slice(false),
        AbiParam::BufferInOut => byte_slice(true),
        AbiParam::HandleBorrowed { resource_type }
        | AbiParam::HandleConsumed { resource_type }
        | AbiParam::HandleOut { resource_type } => {
            return Err(UnmappedParam::ResourceHandle {
                index,
                resource_type: resource_type.clone(),
            });
        }
    })
}

/// The full `(params, ret)` signature for a validated provider call.
pub fn signature(params: &[AbiParam]) -> Result<(Vec<MirTy>, MirTy), UnmappedParam> {
    let mut tys = Vec::with_capacity(params.len());
    for (index, p) in params.iter().enumerate() {
        tys.push(param_ty(index, p)?);
    }
    Ok((tys, PROVIDER_STATUS_TY))
}
