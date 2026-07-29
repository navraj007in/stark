//! WP-C7.8.8 step 2 — deriving a STARK-callable signature from validated provider metadata.
//!
//! **The standing invariant this implements (CD-224):**
//!
//! > There is one authoritative callable signature: validated provider metadata. The package
//! > declaration exposes and names that callable surface but does not mirror its physical or
//! > ownership signature.
//!
//! So a manifest binding carries identity only, and everything about *shape* is computed here. The
//! alternative — a declared signature checked against the provider's — is what CD-219 proved
//! drifts: a mirrored `stark_time_unix_now` declared one out-slot where the provider declared two,
//! and metadata validation could not see it, because the wrong mirror was internally consistent.
//! Deriving means there is nothing to drift *from*.
//!
//! The derived items are the **raw binding layer** and are package-private (CD-225 §7.3). A package
//! writes its ergonomic API over them in ordinary STARK; an application calls that, never this.

use crate::provider_abi::{AbiParam, FunctionDecl, ScalarTy};
use std::collections::BTreeMap;

/// A source-level type in a derived signature.
///
/// Deliberately its own small language rather than HIR or MIR types: derivation is a property of
/// the ABI declaration, and step 3 maps these onto HIR once. Keeping it separate means the
/// derivation rule can be read, and tested, without a compiler context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DerivedTy {
    /// A copied scalar parameter.
    Scalar(ScalarTy),
    /// `&mut T` — a caller-owned in/out scalar slot (ABI §11.1).
    ScalarSlot(ScalarTy),
    /// `&[UInt8]`.
    SharedBytes,
    /// `&mut [UInt8]`.
    ExclusiveBytes,
    /// `&R` — a borrowed resource; the caller keeps it (ABI §8's default).
    SharedResource { nominal: String },
    /// `R` by value — ownership transfers at call entry and does not return on failure.
    OwnedResource { nominal: String },
}

/// One derived signature. `results` are the out-slots, in declared order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DerivedSignature {
    /// The item path this is synthesized at, e.g. `Instant::now_ns`.
    pub item_path: String,
    /// The receiver type, when the path is associated and the declaration borrows or consumes it.
    pub receiver: Option<DerivedTy>,
    /// Parameters after the receiver, in declared order.
    pub params: Vec<DerivedTy>,
    /// Out-slot components. Empty → `Result<Unit, E>`; one → `Result<T, E>`; several →
    /// `Result<(T1, …), E>`.
    pub results: Vec<DerivedTy>,
    /// The package's raw error type for this capability.
    pub error: String,
}

impl DerivedSignature {
    /// How the return type renders, for diagnostics and for step 3's HIR construction.
    pub fn render_return(&self) -> String {
        let ok = match self.results.len() {
            0 => "Unit".to_string(),
            1 => render(&self.results[0]),
            _ => format!(
                "({})",
                self.results
                    .iter()
                    .map(render)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        format!("Result<{ok}, {}>", self.error)
    }
}

fn render(t: &DerivedTy) -> String {
    match t {
        DerivedTy::Scalar(s) => scalar_name(*s).to_string(),
        DerivedTy::ScalarSlot(s) => format!("&mut {}", scalar_name(*s)),
        DerivedTy::SharedBytes => "&[UInt8]".to_string(),
        DerivedTy::ExclusiveBytes => "&mut [UInt8]".to_string(),
        DerivedTy::SharedResource { nominal } => format!("&{nominal}"),
        DerivedTy::OwnedResource { nominal } => nominal.clone(),
    }
}

fn scalar_name(s: ScalarTy) -> &'static str {
    match s {
        ScalarTy::U8 => "UInt8",
        ScalarTy::U16 => "UInt16",
        ScalarTy::U32 => "UInt32",
        ScalarTy::U64 => "UInt64",
        ScalarTy::I8 => "Int8",
        ScalarTy::I16 => "Int16",
        ScalarTy::I32 => "Int32",
        ScalarTy::I64 => "Int64",
        ScalarTy::Bool => "Bool",
        ScalarTy::F32 => "Float32",
        ScalarTy::F64 => "Float64",
    }
}

/// Why a signature could not be derived. These replace the withdrawn "declared signature disagrees"
/// diagnostic (CD-224): a package declares no signature, so disagreement is impossible, but
/// derivation itself can fail — and each way it can fail is its own message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeriveError {
    /// **13.5a** — a parameter or result names a resource the package does not bind, so the derived
    /// signature would reference a type that does not exist.
    UnboundResourceInSignature { item_path: String, resource: String },
    /// **13.5b** — an `AbiParam` form that source lowering does not admit. No form is currently in
    /// this state; the variant exists so a future ABI addition fails loudly here rather than
    /// deriving something plausible.
    UnsupportedParamForm { item_path: String, form: String },
    /// **13.5c** — the return shape cannot be derived unambiguously.
    AmbiguousResultShape { item_path: String, detail: String },
    /// **13.5d** — the derived ownership form conflicts with the API category the path implies.
    ///
    /// Associated placement is API organisation only (CD-225 §7.1), but it does carry one claim: a
    /// method on `T` operates on a `T`. A declaration whose handle parameter is a *different*
    /// resource would produce `TcpStream::read_raw(l: &TcpListener, …)`, which reads as a method on
    /// the wrong type.
    OwnershipCategoryConflict {
        item_path: String,
        declared_on: String,
        receiver_resource: String,
    },
    /// **13.5e** — the capability has no raw error type, so `Result<_, E>` has no `E`.
    MissingErrorType {
        item_path: String,
        capability: String,
    },
    /// **13.5f** — two ABI-distinct functions derive to the same package item path.
    AmbiguousItemPath {
        item_path: String,
        symbols: Vec<String>,
    },
}

/// Derives one signature.
///
/// `resource_nominal` maps a provider resource name to the package nominal bound to it — including
/// Core resources, which the compiler owns (CD-224) and a package cannot declare.
pub fn derive(
    item_path: &str,
    capability: &str,
    decl: &FunctionDecl,
    resource_nominal: &BTreeMap<String, String>,
    errors: &BTreeMap<String, String>,
) -> Result<DerivedSignature, DeriveError> {
    let error = errors
        .get(capability)
        .cloned()
        .ok_or_else(|| DeriveError::MissingErrorType {
            item_path: item_path.to_string(),
            capability: capability.to_string(),
        })?;

    let nominal_for = |resource: &str| -> Result<String, DeriveError> {
        resource_nominal.get(resource).cloned().ok_or_else(|| {
            DeriveError::UnboundResourceInSignature {
                item_path: item_path.to_string(),
                resource: resource.to_string(),
            }
        })
    };

    let mut params = Vec::new();
    let mut results = Vec::new();
    for param in &decl.params {
        match param {
            AbiParam::ScalarIn(t) => params.push(DerivedTy::Scalar(*t)),
            AbiParam::ScalarInOut(t) => params.push(DerivedTy::ScalarSlot(*t)),
            AbiParam::BufferIn => params.push(DerivedTy::SharedBytes),
            AbiParam::BufferInOut => params.push(DerivedTy::ExclusiveBytes),
            AbiParam::HandleBorrowed { resource_type } => params.push(DerivedTy::SharedResource {
                nominal: nominal_for(resource_type)?,
            }),
            AbiParam::HandleConsumed { resource_type } => params.push(DerivedTy::OwnedResource {
                nominal: nominal_for(resource_type)?,
            }),
            // Out-slots are results, never parameters -- the shape the whole derivation exists to
            // produce. A provider returns through slots; STARK returns a value.
            AbiParam::ScalarOut(t) => results.push(DerivedTy::Scalar(*t)),
            AbiParam::HandleOut { resource_type } => results.push(DerivedTy::OwnedResource {
                nominal: nominal_for(resource_type)?,
            }),
        }
    }

    // Associated placement: if the path is `T::f` and the first parameter is a handle, that handle
    // must be a `T`. This is 13.5d, and it is the check that keeps `TcpStream::read_raw` from
    // silently taking a listener.
    let mut receiver = None;
    if let Some((declared_on, _)) = item_path.split_once("::") {
        if let Some(first @ (DerivedTy::SharedResource { .. } | DerivedTy::OwnedResource { .. })) =
            params.first().cloned()
        {
            let nominal = match &first {
                DerivedTy::SharedResource { nominal } | DerivedTy::OwnedResource { nominal } => {
                    nominal.clone()
                }
                _ => unreachable!(),
            };
            if nominal != declared_on {
                return Err(DeriveError::OwnershipCategoryConflict {
                    item_path: item_path.to_string(),
                    declared_on: declared_on.to_string(),
                    receiver_resource: nominal,
                });
            }
            receiver = Some(first);
            params.remove(0);
        }
    }

    Ok(DerivedSignature {
        item_path: item_path.to_string(),
        receiver,
        params,
        results,
        error,
    })
}

/// Derives a whole package's bound surface, rejecting **13.5f** — two ABI-distinct functions
/// deriving to one item path.
///
/// Checked across the set rather than per function, because a collision is by definition not
/// visible from either side alone.
pub fn derive_all(
    bindings: &[(String, String, FunctionDecl)],
    resource_nominal: &BTreeMap<String, String>,
    errors: &BTreeMap<String, String>,
) -> Result<Vec<DerivedSignature>, Vec<DeriveError>> {
    let mut by_path: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (item_path, _, decl) in bindings {
        by_path
            .entry(item_path.as_str())
            .or_default()
            .push(decl.name.clone());
    }
    let mut failures: Vec<DeriveError> = by_path
        .iter()
        .filter(|(_, symbols)| symbols.len() > 1)
        .map(|(path, symbols)| {
            let mut symbols = symbols.clone();
            symbols.sort();
            DeriveError::AmbiguousItemPath {
                item_path: (*path).to_string(),
                symbols,
            }
        })
        .collect();

    let mut derived = Vec::new();
    for (item_path, capability, decl) in bindings {
        match derive(item_path, capability, decl, resource_nominal, errors) {
            Ok(sig) => derived.push(sig),
            Err(e) => failures.push(e),
        }
    }

    if failures.is_empty() {
        derived.sort_by(|a, b| a.item_path.cmp(&b.item_path));
        Ok(derived)
    } else {
        Err(failures)
    }
}
