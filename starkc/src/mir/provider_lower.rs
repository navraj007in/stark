//! WP-C7.8.8 step 6 — the input that lets lowering emit `Callee::Provider`.
//!
//! Until now `lower_program` hard-coded `provider_calls: Vec::new()`, so **no** STARK source could
//! reach a provider: every executing test hand-built its MIR. That was the actual critical path,
//! not TCP.
//!
//! This module carries what lowering needs and nothing else: a resolved call arena, and a map from
//! a synthesized item's qualified name to its index. Both are produced *before* lowering — provider
//! selection and metadata validation are A10 §3's pre-verification phase — so lowering performs no
//! selection and interprets no unvalidated metadata. It looks a call up and emits it.

use crate::mir::{ProviderCallId, ValidatedProviderCall};
use std::collections::BTreeMap;

/// The provider calls a program may lower, and which item names reach them.
///
/// Empty is the ordinary case and the correct default: a program that binds no provider gets an
/// empty arena, which is what every existing `lower_program` caller already produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProviderLowering {
    /// The arena, copied verbatim into `MirProgram::provider_calls`.
    pub arena: Vec<ValidatedProviderCall>,
    /// Qualified item name (`symbol_prefix` + leaf, e.g. `"monotonic_now_ns"` or
    /// `"time::now_ns"`) → arena index.
    ///
    /// Keyed by **name** rather than `ItemId` because bindings are computed from the manifest
    /// before any source is parsed, so no item ids exist yet. Lowering resolves names to ids once,
    /// up front, and the per-call-site path is an id lookup.
    pub by_item_name: BTreeMap<String, ProviderCallId>,
    /// Raw error type name → status code → variant index, from
    /// [`crate::provider_synth::SynthesizedLayer::error_variants`].
    ///
    /// Lowering resolves the type name to an `ItemId` once, then builds `Err(RawE::V)` for a
    /// declared code as an ordinary enum aggregate. Carried by name for the same reason the call
    /// bindings are: this is computed from the manifest and the provider vocabulary, before any
    /// source exists to have item ids.
    pub error_variants: BTreeMap<String, BTreeMap<u32, u32>>,
    /// Provider call id → the raw error type its `Result` uses.
    ///
    /// Kept beside the arena rather than on [`ValidatedProviderCall`]: that record is A10's
    /// *validated ABI contract*, and a package's choice of STARK error type name is not part of it.
    pub error_ty_for_call: BTreeMap<u32, String>,
}

impl ProviderLowering {
    /// Builds from a synthesized layer's binding table and a resolver that turns
    /// `(capability, symbol)` into a validated call record.
    ///
    /// The resolver is a parameter rather than a dependency so this module needs no provider-set
    /// plumbing: `provider_resolve` owns selection, and this owns only the lowering input.
    pub fn build<F>(
        bindings: &BTreeMap<String, (String, String)>,
        resolve: F,
    ) -> Result<Self, String>
    where
        F: FnMut(&str, &str) -> Result<ValidatedProviderCall, String>,
    {
        Self::build_with_errors(bindings, &BTreeMap::new(), &BTreeMap::new(), resolve)
    }

    /// `build`, plus the raw error types the derived signatures return.
    ///
    /// `error_ty_by_item` maps an item path to its raw error type name — taken from the derived
    /// signatures, so lowering and synthesis cannot disagree about which enum a call's `Err` arm
    /// constructs.
    pub fn build_with_errors<F>(
        bindings: &BTreeMap<String, (String, String)>,
        error_variants: &BTreeMap<String, BTreeMap<u32, u32>>,
        error_ty_by_item: &BTreeMap<String, String>,
        mut resolve: F,
    ) -> Result<Self, String>
    where
        F: FnMut(&str, &str) -> Result<ValidatedProviderCall, String>,
    {
        let mut arena = Vec::new();
        let mut by_item_name = BTreeMap::new();
        let mut error_ty_for_call = BTreeMap::new();
        // `bindings` is a `BTreeMap`, so iteration is name-ordered and arena indices are a
        // deterministic function of the manifest -- not of hash order. MIR is compared byte-for-byte
        // by the reproducibility suites, so this is a correctness property, not tidiness.
        for (item_path, (capability, symbol)) in bindings {
            let call = resolve(capability, symbol)?;
            let id = ProviderCallId(arena.len() as u32);
            arena.push(call);
            by_item_name.insert(leaf(item_path).to_string(), id);
            if let Some(ty) = error_ty_by_item.get(item_path) {
                error_ty_for_call.insert(id.0, ty.clone());
            }
        }
        Ok(Self {
            arena,
            by_item_name,
            error_variants: error_variants.clone(),
            error_ty_for_call,
        })
    }

    /// The `(error type name, code → variant index)` for a call, when it has a raw error type with
    /// declared recoverable statuses.
    pub fn error_mapping_for(&self, id: ProviderCallId) -> Option<(&str, &BTreeMap<u32, u32>)> {
        let ty = self.error_ty_for_call.get(&id.0)?;
        let variants = self.error_variants.get(ty)?;
        (!variants.is_empty()).then_some((ty.as_str(), variants))
    }

    /// The shared empty set, so `FnLowerer::new` can hand out a borrow without every caller
    /// owning one. A program that binds no provider is the overwhelmingly common case.
    pub fn none() -> &'static ProviderLowering {
        static NONE: std::sync::OnceLock<ProviderLowering> = std::sync::OnceLock::new();
        NONE.get_or_init(ProviderLowering::default)
    }

    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn call_for(&self, item_name: &str) -> Option<ProviderCallId> {
        self.by_item_name.get(item_name).copied()
    }
}

/// `time::now_ns` → `now_ns`. Synthesis emits free functions (its `RESOURCE_SYNTHESIS_LIMIT`
/// refuses anything else), so the leaf is the item's source name.
fn leaf(item_path: &str) -> &str {
    item_path.rsplit_once("::").map_or(item_path, |(_, n)| n)
}
