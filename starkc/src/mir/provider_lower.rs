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
    /// Provider resource name → the nominal's resolved `ItemId`, so lowering can build a
    /// `MirTy::HostResource` (CD-234/CD-235). Resolved once, after the generated source is parsed.
    pub resource_items: BTreeMap<String, crate::hir::ItemId>,
    /// Provider resource name → the package NOMINAL name bound to it.
    ///
    /// Names, not `ItemId`s, because this is known from the manifest before any source exists.
    /// Lowering resolves them once `ProgramMeta` can map an item to its file and text.
    pub resource_nominal_names: BTreeMap<String, String>,
    /// Provider resource name → the arena id of its close call.
    ///
    /// **Split from `closes` deliberately.** Selecting the close needs the provider set, which only
    /// the driver has; building the `ValidatedProviderClose` needs the resource's `MirTy`, which
    /// needs an `ItemId`, which only exists after parsing. So the driver records the pair here and
    /// lowering completes it — each stage doing the part it can actually know.
    pub pending_closes: BTreeMap<String, ProviderCallId>,
    /// Synthesized nominal `ItemId` → the `HostResource` it lowers to.
    ///
    /// **CD-234's "the binding replaces its representation at the established boundary."** The
    /// nominal is a zero-variant enum in SOURCE — that is what makes it opaque — but it must not
    /// stay one in MIR, or the same type gets two representations: `Enum(User(id))` wherever the
    /// ordinary type path saw it, and `HostResource` wherever a provider signature did. Those then
    /// fail to unify at every call boundary between them.
    pub nominal_types: BTreeMap<u32, crate::mir::MirTy>,
    /// **A11 §5: the close selected for each bound resource**, copied onto
    /// `MirProgram::provider_closes` and keyed into `TypeContext::host_resource_closes` by lowering.
    pub closes: Vec<crate::mir::ValidatedProviderClose>,
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
            resource_items: BTreeMap::new(),
            resource_nominal_names: BTreeMap::new(),
            pending_closes: BTreeMap::new(),
            nominal_types: BTreeMap::new(),
            closes: Vec::new(),
            error_variants: error_variants.clone(),
            error_ty_for_call,
        })
    }

    /// **A11 §5: selects each bound resource's close, at resolution time.**
    ///
    /// Returns the `(arena, closes)` additions. A resource with no `is_close_for` function in its
    /// provider's metadata is an ERROR, not an empty result: §5 obligation 5 says a resource reaching
    /// emission without a close is a leak the ABI cannot detect, because the provider never learns
    /// the handle was abandoned.
    pub fn select_closes<F>(&mut self, mut close_for: F) -> Result<(), String>
    where
        F: FnMut(&str) -> Result<ValidatedProviderCall, String>,
    {
        // Iterates the NOMINAL NAMES, which the manifest supplies, rather than `resource_items`,
        // which is empty until lowering resolves ids. Selecting from an empty map is how this
        // silently selected nothing.
        let resources: Vec<String> = self.resource_nominal_names.keys().cloned().collect();
        for resource in resources {
            let call = close_for(&resource)?;
            let id = ProviderCallId(self.arena.len() as u32);
            self.arena.push(call);
            self.pending_closes.insert(resource, id);
        }
        Ok(())
    }

    /// Resolves nominal NAMES to `ItemId`s and completes the close bindings.
    ///
    /// Called by lowering, where `ProgramMeta` can map an item to its file and read its name text.
    /// Returns the completed closes; `resource_items` is filled in place so `resource_ty` works for
    /// the rest of lowering.
    pub fn resolve_nominals<F>(
        &mut self,
        mut find_item: F,
    ) -> Result<Vec<crate::mir::ValidatedProviderClose>, String>
    where
        F: FnMut(&str) -> Option<crate::hir::ItemId>,
    {
        for (resource, nominal) in &self.resource_nominal_names {
            let item = find_item(nominal).ok_or_else(|| {
                format!(
                    "synthesized nominal `{nominal}` for resource `{resource}` is not among the \
                     program's items"
                )
            })?;
            self.resource_items.insert(resource.clone(), item);
        }

        // Every bound nominal gets its MIR representation recorded, so the ordinary type path can
        // replace the enum shell with the resource form. Derived from the close's provider, which is
        // the same provider the resource's own calls resolve against (A11 §5 obligation 3).
        for (resource, id) in &self.pending_closes {
            let Some(provider) = self
                .arena
                .get(id.0 as usize)
                .map(|c| c.provider.name.clone())
            else {
                continue;
            };
            if let (Some(item), Some(ty)) = (
                self.resource_items.get(resource).copied(),
                self.resource_ty(resource, &provider),
            ) {
                self.nominal_types.insert(item.0, ty);
            }
        }

        let mut closes = Vec::new();
        for (resource, id) in &self.pending_closes {
            let provider = self
                .arena
                .get(id.0 as usize)
                .map(|c| c.provider.name.clone())
                .ok_or_else(|| format!("close for `{resource}` is not in the arena"))?;
            let ty = self.resource_ty(resource, &provider).ok_or_else(|| {
                format!(
                    "resource `{resource}` has no bound nominal, so its close has nothing to close"
                )
            })?;
            closes.push(crate::mir::ValidatedProviderClose {
                resource: ty,
                close: *id,
            });
        }
        self.closes = closes.clone();
        Ok(closes)
    }

    /// The `MirTy` for a provider resource name, when its nominal has been resolved.
    ///
    /// Built here rather than stored, because the provider is a property of the call and the same
    /// nominal could in principle be reached through different providers — A11 §Q5 makes those
    /// different types deliberately.
    pub fn resource_ty(&self, resource: &str, provider: &str) -> Option<crate::mir::MirTy> {
        let item = self.resource_items.get(resource)?;
        Some(crate::mir::MirTy::host_resource(
            crate::mir::HostResourceNominal::Item(*item),
            provider,
            resource,
        ))
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
