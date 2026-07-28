//! WP-C7.8.2d-1 — the provider binding plan.
//!
//! A compiler-owned description sitting between validated provider metadata and generated Rust.
//! Its purpose is to stop provider emission from being a single raw call expression, because that
//! shape has nowhere to put A10 §4's invariants 6, 8 and 9. The plan names the structure those
//! invariants inspect:
//!
//! ```text
//! prepare borrowed inputs
//! prepare uninitialised outputs
//! move consumed resources
//! call extern "C" provider
//! inspect ProviderStatus
//! branch:
//!     success
//!     declared recoverable error
//!     undeclared status / contract violation
//! validate resource outputs
//! read successful outputs
//! construct STARK result
//! ```
//!
//! This is **not** public MIR. MIR carries `Callee::Provider` and the validated declaration; the
//! plan is derived from that declaration for the backend's benefit and never widens what MIR
//! admits.
//!
//! Governing decisions: `WP-C7.8.1-DECISION-PACKETS.md` Packet 1 §1.2 (status channels), Packet 2
//! / A10 §4-§6.

use crate::mir::{MirTy, ProviderCallId, ValidatedProviderCall};
use crate::provider_abi::{AbiParam, ScalarTy};
use std::collections::BTreeMap;

// ------------------------------------------------------------------ registry --

/// Binds a provider's declared `resource_type` string to the MIR type that represents it.
///
/// **Empty in C7.8.2d.** The first entry arrives with C7.8.4, which will register
/// `"file" → MirTy::Core(CoreType::File, [])`. Until a resource type is bound, a call carrying one
/// is inadmissible (MIR-0024) — structurally defined, but not yet mappable to a STARK type, so
/// ABI §11.1's `resource_type` validation would have nothing to check against.
#[derive(Debug, Clone, Default)]
pub struct ResourceRegistry {
    map: BTreeMap<String, MirTy>,
}

impl ResourceRegistry {
    /// The compiler's built-in bindings. Deliberately empty until C7.8.4 registers `File`.
    pub fn builtin() -> Self {
        Self::default()
    }

    pub fn register(&mut self, resource_type: impl Into<String>, mir_type: MirTy) {
        self.map.insert(resource_type.into(), mir_type);
    }

    pub fn lookup(&self, resource_type: &str) -> Option<&MirTy> {
        self.map.get(resource_type)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// --------------------------------------------------------------------- plan --

/// A parameter the provider reads (and may write, for the in/out forms).
///
/// **Every input names its declared parameter index.** The plan is checked against the declaration
/// by index, so a reordering defect cannot hide behind matching counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderInputPlan {
    /// Copied scalar. No borrow, no lifetime obligation.
    Scalar { index: usize, ty: MirTy },
    /// Caller-initialised, caller-owned across the call (ABI §11.1). Readable after **either**
    /// status — its validity does not depend on the status code.
    ScalarInOut { index: usize, ty: MirTy },
    /// Immutable borrowed byte view, valid only for the duration of the call (ABI §9).
    BufferIn { index: usize },
    /// Mutable borrowed byte view. Caller-owned and caller-initialised, so — unlike an output slot
    /// — it is **not** `MaybeUninit` and may be inspected after either status.
    BufferInOut { index: usize },
    /// Caller retains ownership for the duration of the call (ABI §8's default).
    HandleBorrowed {
        index: usize,
        resource_type: String,
        mir_type: MirTy,
    },
    /// Ownership transfers **at call entry**, and the source place is dead from that point
    /// regardless of the returned status (ABI §8's consumed-handle error rule). There is no
    /// "the call failed, so you still own it" path: ownership returning on failure would make a
    /// handle's liveness a runtime property, and exactly-once close would stop being statically
    /// verifiable.
    HandleConsumed {
        index: usize,
        resource_type: String,
        mir_type: MirTy,
    },
}

/// A parameter the provider writes on success only.
///
/// These are the `MaybeUninit` forms. `ScalarInOut` and `BufferInOut` are deliberately **not**
/// here: ABI §11.1 makes them caller-initialised and caller-owned, so applying uninitialised-slot
/// semantics to them would be wrong in both directions — it would forbid a legitimate read and
/// imply the provider allocates storage it does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderOutputPlan {
    /// Uninitialised before the call; valid only when the status reports success.
    Scalar { index: usize, ty: MirTy },
    /// A freshly-owned handle, written on success. Its `resource_type` must be validated **before**
    /// the owning STARK value is constructed (ABI §11.1); a mismatch is a contract violation, not
    /// a recoverable error.
    Handle {
        index: usize,
        resource_type: String,
        mir_type: MirTy,
    },
}

impl ProviderInputPlan {
    pub fn index(&self) -> usize {
        match self {
            Self::Scalar { index, .. }
            | Self::ScalarInOut { index, .. }
            | Self::BufferIn { index }
            | Self::BufferInOut { index }
            | Self::HandleBorrowed { index, .. }
            | Self::HandleConsumed { index, .. } => *index,
        }
    }

    /// Whether this input requires a borrow whose backing storage stays live and immovable for the
    /// complete call (A10 §4 invariant 6).
    pub fn requires_live_borrow(&self) -> bool {
        matches!(
            self,
            Self::ScalarInOut { .. }
                | Self::BufferIn { .. }
                | Self::BufferInOut { .. }
                | Self::HandleBorrowed { .. }
        )
    }
}

impl ProviderOutputPlan {
    pub fn index(&self) -> usize {
        match self {
            Self::Scalar { index, .. } | Self::Handle { index, .. } => *index,
        }
    }
}

// ------------------------------------------------------------------- status --

/// The package's declared recoverable status vocabulary (Packet 1 §1.2).
///
/// Codes map to a package-defined error identity the compiler treats as opaque — the compiler
/// never interprets what `IOError::NotFound` *means*, only that the package declared code N to be
/// that error. An empty map is legal and says something precise: **every** nonzero status from
/// this provider is a contract violation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusBinding {
    declared: BTreeMap<u32, String>,
}

impl StatusBinding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare(&mut self, code: u32, package_error: impl Into<String>) {
        self.declared.insert(code, package_error.into());
    }

    pub fn declared_codes(&self) -> impl Iterator<Item = (&u32, &String)> {
        self.declared.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.declared.is_empty()
    }
}

/// What the generated dispatch does for one status value. The three arms are ABI §12's three
/// channels, and they must stay structurally distinct — a `_ =>` fallback to a generic package
/// error is exactly the collapse Packet 1 §1.2 forbids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusOutcome {
    /// Channel 0: validate and read outputs, construct the STARK success value.
    Success,
    /// Channel 1: a declared code becomes a package-defined `Result::Err`.
    RecoverableError { code: u32, package_error: String },
    /// Channel 2: an **undeclared** nonzero code is a contract violation — fatal, never
    /// `Result::Err`.
    ///
    /// This defends the failure mode where a provider and its package drift apart while remaining
    /// physically ABI-compatible: nothing crashes, and the meaning quietly changes.
    ContractViolation { code: u32 },
}

/// Success is `ProviderStatus.code == 0` (ABI §11).
pub const STATUS_SUCCESS: u32 = 0;

// --------------------------------------------------------------------- plan --

/// Why a call cannot be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The declaration carries a resource type no registry entry binds to a MIR type. Reported as
    /// MIR-0024. **Not** "MIR cannot support resources" — the structure is defined; this specific
    /// resource type is not yet bound.
    UnboundResourceType { index: usize, resource_type: String },
}

/// The full binding plan for one provider call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderBindingPlan {
    pub call: ProviderCallId,
    pub inputs: Vec<ProviderInputPlan>,
    pub outputs: Vec<ProviderOutputPlan>,
    pub status: StatusBinding,
}

impl ProviderBindingPlan {
    /// Classifies one status value into its ABI §12 channel.
    pub fn classify(&self, status: u32) -> StatusOutcome {
        if status == STATUS_SUCCESS {
            return StatusOutcome::Success;
        }
        match self.status.declared.get(&status) {
            Some(package_error) => StatusOutcome::RecoverableError {
                code: status,
                package_error: package_error.clone(),
            },
            None => StatusOutcome::ContractViolation { code: status },
        }
    }

    /// Every declared parameter produced exactly one plan item, and the indices cover
    /// `0..params_len` without gaps or repeats.
    ///
    /// Held as a checkable property rather than a construction comment: the plan is what emission
    /// walks, so a dropped parameter would become a call with a missing argument rather than a
    /// compile error.
    pub fn covers(&self, params_len: usize) -> bool {
        let mut seen: Vec<usize> = self
            .inputs
            .iter()
            .map(|i| i.index())
            .chain(self.outputs.iter().map(|o| o.index()))
            .collect();
        seen.sort_unstable();
        seen.len() == params_len && seen.iter().enumerate().all(|(i, got)| i == *got)
    }
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

/// Builds the plan for a validated call, classifying every declared parameter.
pub fn plan(
    id: ProviderCallId,
    call: &ValidatedProviderCall,
    registry: &ResourceRegistry,
    status: StatusBinding,
) -> Result<ProviderBindingPlan, PlanError> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for (index, param) in call.function.params.iter().enumerate() {
        let resolve = |resource_type: &String| -> Result<MirTy, PlanError> {
            registry
                .lookup(resource_type)
                .cloned()
                .ok_or_else(|| PlanError::UnboundResourceType {
                    index,
                    resource_type: resource_type.clone(),
                })
        };

        match param {
            AbiParam::ScalarIn(t) => inputs.push(ProviderInputPlan::Scalar {
                index,
                ty: scalar_ty(*t),
            }),
            AbiParam::ScalarInOut(t) => inputs.push(ProviderInputPlan::ScalarInOut {
                index,
                ty: scalar_ty(*t),
            }),
            AbiParam::BufferIn => inputs.push(ProviderInputPlan::BufferIn { index }),
            AbiParam::BufferInOut => inputs.push(ProviderInputPlan::BufferInOut { index }),
            AbiParam::ScalarOut(t) => outputs.push(ProviderOutputPlan::Scalar {
                index,
                ty: scalar_ty(*t),
            }),
            AbiParam::HandleBorrowed { resource_type } => {
                inputs.push(ProviderInputPlan::HandleBorrowed {
                    index,
                    resource_type: resource_type.clone(),
                    mir_type: resolve(resource_type)?,
                })
            }
            AbiParam::HandleConsumed { resource_type } => {
                inputs.push(ProviderInputPlan::HandleConsumed {
                    index,
                    resource_type: resource_type.clone(),
                    mir_type: resolve(resource_type)?,
                })
            }
            AbiParam::HandleOut { resource_type } => outputs.push(ProviderOutputPlan::Handle {
                index,
                resource_type: resource_type.clone(),
                mir_type: resolve(resource_type)?,
            }),
        }
    }

    Ok(ProviderBindingPlan {
        call: id,
        inputs,
        outputs,
        status,
    })
}
