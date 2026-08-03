//! WP-C7.8.2b — provider selection and validated call-record construction.
//!
//! Implements the binding sequence A10 §3 requires, in the order it requires:
//!
//! ```text
//! package operation
//! → capability requirement
//! → provider selection for target
//! → metadata validation
//! → FunctionDecl resolution
//! → ValidatedProviderCall allocation
//! → Callee::Provider(ProviderCallId)
//! ```
//!
//! **Everything here runs before MIR construction.** By the time verification sees a
//! `Callee::Provider`, selection has happened and metadata has passed
//! [`crate::provider_abi::validate`]. The backend never performs first-time provider selection and
//! never interprets unvalidated metadata (A10 §3, §6).
//!
//! Governing decisions: `WP-C7.8.1-DECISION-PACKETS.md` Packet 1 §1.3 (symbols) and §1.4
//! (platform selection), CD-198/CD-199; Packet 2 (A10), CD-200.

use crate::mir::{ProviderCallId, ValidatedProviderCall};
use crate::provider_abi::{self, AbiViolation, ProviderMetadata};

/// A provider's declared metadata plus where it came from.
///
/// **The origin is deliberately not a `ProviderMetadata` field.** Packet 1 §1.4 requires the
/// ambiguity diagnostic to name each conflicting provider's metadata location, but
/// `ProviderMetadata` is constructed as a struct literal by provider crates —
/// `stark-time/native/src/lib.rs` does exactly that — and Packet 1's exit condition requires that
/// crate to work with no source change. Adding a required field would have broken it. Pairing the
/// location alongside costs nothing and keeps that condition intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredProvider {
    pub metadata: ProviderMetadata,
    /// Cargo package name of the crate implementing this provider. Carried alongside the metadata
    /// for the same reason `origin` is: `ProviderMetadata` is a struct literal in provider crates,
    /// and adding a field would break every one of them.
    pub crate_name: String,
    /// The package's declared recoverable status vocabulary (Packet 1 §1.2).
    ///
    /// Lives here rather than in `ProviderMetadata` for the same reason: the ABI deliberately does
    /// not carry code meanings, because they are a *package* concern. Empty is meaningful — it
    /// says every nonzero status from this provider is a contract violation.
    pub status_binding: crate::provider_bind::StatusBinding,
    /// CD-363: where this provider's Cargo crate lives, RELATIVE to a root the caller supplies —
    /// the compiler's own root for a built-in, the manifest's directory for an external one.
    ///
    /// **A location, never part of MIR.** `crate_location`'s original doc had this right and it
    /// survives its deletion: a crate's path is a property of the machine doing the build, its name
    /// a property of the program, and keeping them apart is what lets a verified MIR artefact stay
    /// relocation-stable. Constrained to be relative and free of `..` at parse time, because for an
    /// external provider the root is the only containment there is.
    pub crate_path: String,
    /// Human-readable provenance for diagnostics, e.g. a manifest path. Never load-bearing for
    /// selection — two providers with identical origins still conflict, and one with no
    /// meaningful origin still resolves.
    pub origin: String,
}

/// Why a provider symbol is not a usable C identifier (Packet 1 §1.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolProblem {
    Empty,
    /// A C identifier may not begin with a digit.
    LeadingDigit,
    /// Any byte outside `[A-Za-z0-9_]` — whitespace, NUL, and punctuation all land here.
    IllegalByte {
        byte: u8,
        index: usize,
    },
}

/// A failure in the binding sequence. Every variant is fatal and reported **before backend
/// invocation** (`WP-C5-ENTRY.md` §3.2), never as a silent downgrade (ABI §16).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// **A11 §5 obligation 5:** the resource has no `is_close_for` function, so a handle of it could
    /// never be released. A leak the ABI cannot detect — the provider never learns the handle was
    /// abandoned — so it is refused at selection rather than discovered at runtime as a leak.
    NoCloseForResource { resource_type: String },
    /// **CD-360:** a provider declares it consumes a foreign resource, but no provider in the
    /// SELECTED set owns that resource type. Refused at selection rather than at link, so the
    /// diagnostic names the consumer, the resource and the provider it expected — a linker cannot.
    ForeignResourceUnsupplied {
        consumer: String,
        expected_provider: String,
        resource_type: String,
    },
    /// **CD-360:** more than one selected provider declares the resource type a transfer consumes.
    /// Ownership must be unambiguous: the destination's release authority is the source's identity,
    /// and two owners means two closes for one resource.
    ForeignResourceAmbiguous {
        consumer: String,
        resource_type: String,
        owners: Vec<String>,
    },
    /// **CD-360:** the resource type is supplied, but by a provider other than the one the consumer
    /// named. Resource identity is structural over `{nominal, provider, resource}`, so a matching
    /// NAME with a different owner is a different resource — accepting it would transfer ownership
    /// of something the consumer never declared it could consume.
    ForeignResourceOwnerMismatch {
        consumer: String,
        resource_type: String,
        expected_provider: String,
        actual_provider: String,
    },
    /// Two closes for one resource: two destruction paths, and choosing by order would make the
    /// binary depend on something other than what the program declared.
    AmbiguousClose {
        resource_type: String,
        functions: Vec<String>,
    },
    /// The provider's declared metadata does not satisfy ABI v0.1 (§2–§16).
    InvalidMetadata {
        provider: String,
        origin: String,
        violations: Vec<AbiViolation>,
    },
    /// Packet 1 §1.3: a declared symbol is not a valid portable C identifier. **Rejected, never
    /// sanitised** — repairing it would make the metadata name differ from the linkage name, the
    /// one thing that must never be true when the same field has to resolve under a future
    /// `dlsym`.
    InvalidSymbol {
        provider: String,
        origin: String,
        symbol: String,
        problem: SymbolProblem,
    },
    /// Packet 1 §1.3: two selected providers export the same C symbol. Unlike a prefix
    /// convention, this is the *exact* anti-collision guarantee.
    ///
    /// Remediation is a *configuration* change: remove one provider, or narrow its target
    /// declarations.
    DuplicateSymbol {
        symbol: String,
        providers: Vec<(String, String)>,
    },
    /// One provider declares the same C symbol twice.
    ///
    /// Kept distinct from [`ResolveError::DuplicateSymbol`] because the remediation is different —
    /// this is a *provider* defect, fixed in that provider's own declaration, and reporting it as
    /// a cross-provider conflict would name the same provider twice and send the reader looking
    /// for a second one that does not exist.
    DuplicateSymbolWithinProvider {
        provider: String,
        origin: String,
        symbol: String,
    },
    /// ABI §16 check 3: the program requires a capability no selected provider supplies for this
    /// target.
    CapabilityUnavailable {
        capability: String,
        target: String,
        /// Providers that declare the capability but exclude this target, so the diagnostic can
        /// say "declared, but not for your target" rather than "unknown capability".
        declared_for_other_targets: Vec<(String, Vec<String>)>,
    },
    /// Packet 1 §1.4: two providers applicable to the same capability and target.
    ///
    /// **A hard error with no priority mechanism.** Silent first-match — by package order,
    /// dependency order, or lexical precedence — would make the produced binary depend on
    /// incidental declaration ordering, and a priority field would only convert an invalid
    /// configuration into another implicit selection mechanism.
    CapabilityAmbiguous {
        capability: String,
        target: String,
        /// `(provider name, metadata origin)`, sorted, at least two.
        providers: Vec<(String, String)>,
    },
    /// The selected provider for this capability declares no such function.
    UnknownFunction {
        capability: String,
        function: String,
        provider: String,
    },
    /// A10 §4 invariant 3: the function exists but belongs to a different capability.
    FunctionCapabilityMismatch {
        function: String,
        provider: String,
        declared: String,
        requested: String,
    },
}

/// Validates a declared symbol against Packet 1 §1.3's admitted grammar `[A-Za-z_][A-Za-z0-9_]*`.
///
/// Public because MIR verification re-runs it (V-PROV-4): emission reads the *record*, not the
/// resolver's transient state, so the record itself must be checkable.
///
/// Byte-oriented rather than char-oriented on purpose: a C symbol is bytes, and a multi-byte
/// UTF-8 sequence must be rejected by its first offending byte rather than silently accepted as
/// "one character".
pub fn check_symbol(symbol: &str) -> Result<(), SymbolProblem> {
    let bytes = symbol.as_bytes();
    let Some(&first) = bytes.first() else {
        return Err(SymbolProblem::Empty);
    };
    if first.is_ascii_digit() {
        return Err(SymbolProblem::LeadingDigit);
    }
    for (index, &b) in bytes.iter().enumerate() {
        if !(b.is_ascii_alphanumeric() || b == b'_') {
            return Err(SymbolProblem::IllegalByte { byte: b, index });
        }
    }
    Ok(())
}

/// The providers selected for one build, for one target.
///
/// Construction is the only way to get one, and construction is the point every ABI §16 check
/// runs. A `ProviderSet` in hand therefore means: every member validated, every symbol is a legal
/// C identifier, no two members collide, and every required capability has exactly one supplier.
#[derive(Debug, Clone)]
pub struct ProviderSet {
    target: String,
    selected: Vec<DeclaredProvider>,
}

impl ProviderSet {
    /// Runs the full selection sequence. Returns **every** failure rather than the first, matching
    /// the MIR verifier's and the ABI validator's own convention — a build with three
    /// misconfigured providers should report three, not require three rebuilds.
    pub fn select(
        declared: Vec<DeclaredProvider>,
        target: &str,
        required_capabilities: &[String],
    ) -> Result<Self, Vec<ResolveError>> {
        let mut errors = Vec::new();

        // 1. Metadata validation (§2-§16's mechanically checkable rules, including the
        //    abi_version check that is §16's first gate) and 2. symbol grammar (Packet 1 §1.3).
        //    Both run over EVERY declared provider, not only the ones this target selects: a
        //    malformed provider is a defect to report, not a thing to skip past quietly because
        //    the current target happens not to need it.
        for p in &declared {
            if let Err(violations) = provider_abi::validate(&p.metadata) {
                errors.push(ResolveError::InvalidMetadata {
                    provider: p.metadata.identity.name.clone(),
                    origin: p.origin.clone(),
                    violations,
                });
            }
            for f in &p.metadata.functions {
                if let Err(problem) = check_symbol(&f.name) {
                    errors.push(ResolveError::InvalidSymbol {
                        provider: p.metadata.identity.name.clone(),
                        origin: p.origin.clone(),
                        symbol: f.name.clone(),
                        problem,
                    });
                }
            }
        }

        // 3. Target applicability (§16 check 2). A provider that does not admit this target is
        //    excluded from selection -- not an error in itself, only if it leaves a required
        //    capability unsupplied, which step 5 reports with that context.
        let selected: Vec<DeclaredProvider> = declared
            .iter()
            .filter(|p| p.metadata.target_triples.iter().any(|t| t == target))
            .cloned()
            .collect();

        // 3b. Packet 5's ADMISSION rule: a provider is admitted if and only if it supplies a
        //     capability some package declared. Target compatibility alone is not admission --
        //     without this filter a provider nobody asked for would be linked into the binary
        //     merely because it could have run there, which is the implicit discovery Packet 5
        //     forbids. Requiring nothing therefore selects nothing.
        let selected: Vec<DeclaredProvider> = selected
            .into_iter()
            .filter(|p| {
                p.metadata
                    .capabilities
                    .iter()
                    .any(|c| required_capabilities.contains(c))
            })
            .collect();

        // 4. Cross-provider symbol uniqueness over the SELECTED set (Packet 1 §1.3). Scoped to
        //    the selected set because two providers for different targets can never be linked
        //    into one binary, so their symbols cannot collide.
        //
        //    Intra-provider duplicates are separated out first: they are a provider defect with a
        //    different fix, and folding them in here would name one provider twice and send the
        //    reader hunting for a second one.
        for p in &selected {
            let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for f in &p.metadata.functions {
                if !seen.insert(f.name.as_str()) {
                    errors.push(ResolveError::DuplicateSymbolWithinProvider {
                        provider: p.metadata.identity.name.clone(),
                        origin: p.origin.clone(),
                        symbol: f.name.clone(),
                    });
                }
            }
        }

        let mut by_symbol: std::collections::BTreeMap<&str, Vec<(String, String)>> =
            std::collections::BTreeMap::new();
        for p in &selected {
            // Deduplicated per provider so an intra-provider repeat (reported above) does not also
            // manifest as a phantom cross-provider collision.
            let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
            for f in &p.metadata.functions {
                if seen.insert(f.name.as_str()) {
                    by_symbol
                        .entry(f.name.as_str())
                        .or_default()
                        .push((p.metadata.identity.name.clone(), p.origin.clone()));
                }
            }
        }
        for (symbol, providers) in by_symbol {
            if providers.len() > 1 {
                errors.push(ResolveError::DuplicateSymbol {
                    symbol: symbol.to_string(),
                    providers,
                });
            }
        }

        // 5. Capability supply (§16 check 3 + Packet 1 §1.4). Exactly one supplier, always.
        for capability in required_capabilities {
            let suppliers: Vec<(String, String)> = selected
                .iter()
                .filter(|p| p.metadata.capabilities.contains(capability))
                .map(|p| (p.metadata.identity.name.clone(), p.origin.clone()))
                .collect();
            match suppliers.len() {
                0 => {
                    let declared_for_other_targets = declared
                        .iter()
                        .filter(|p| p.metadata.capabilities.contains(capability))
                        .map(|p| {
                            (
                                p.metadata.identity.name.clone(),
                                p.metadata.target_triples.clone(),
                            )
                        })
                        .collect();
                    errors.push(ResolveError::CapabilityUnavailable {
                        capability: capability.clone(),
                        target: target.to_string(),
                        declared_for_other_targets,
                    });
                }
                1 => {}
                _ => {
                    let mut providers = suppliers;
                    providers.sort();
                    errors.push(ResolveError::CapabilityAmbiguous {
                        capability: capability.clone(),
                        target: target.to_string(),
                        providers,
                    });
                }
            }
        }

        // 6. CD-360: every declared foreign consumption resolves to EXACTLY ONE owning provider in
        //    the selected set, and to the provider the consumer named. This is the build-time half
        //    of the transfer rule -- `provider_abi::validate` can only check a provider against
        //    itself, so "does the thing I consume actually exist, once, and belong to whom I said"
        //    is answerable only here, where the set is.
        for consumer in &selected {
            for foreign in &consumer.metadata.foreign_resources {
                let owners: Vec<&DeclaredProvider> = selected
                    .iter()
                    .filter(|p| {
                        p.metadata
                            .resource_types
                            .iter()
                            .any(|rt| rt == &foreign.resource)
                    })
                    .collect();
                match owners.as_slice() {
                    [] => errors.push(ResolveError::ForeignResourceUnsupplied {
                        consumer: consumer.metadata.identity.name.clone(),
                        expected_provider: foreign.provider.clone(),
                        resource_type: foreign.resource.clone(),
                    }),
                    [owner] => {
                        if owner.metadata.identity.name != foreign.provider {
                            errors.push(ResolveError::ForeignResourceOwnerMismatch {
                                consumer: consumer.metadata.identity.name.clone(),
                                resource_type: foreign.resource.clone(),
                                expected_provider: foreign.provider.clone(),
                                actual_provider: owner.metadata.identity.name.clone(),
                            });
                        }
                    }
                    many => errors.push(ResolveError::ForeignResourceAmbiguous {
                        consumer: consumer.metadata.identity.name.clone(),
                        resource_type: foreign.resource.clone(),
                        owners: many
                            .iter()
                            .map(|p| p.metadata.identity.name.clone())
                            .collect(),
                    }),
                }
            }
        }

        if errors.is_empty() {
            Ok(ProviderSet {
                target: target.to_string(),
                selected,
            })
        } else {
            Err(errors)
        }
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn providers(&self) -> &[DeclaredProvider] {
        &self.selected
    }

    /// Step 4 of A10 §3: resolve one capability + function name to its validated call record.
    ///
    /// The capability must have been named in `required_capabilities` at selection time — that is
    /// what proves exactly one supplier exists. Resolving an unselected capability reports
    /// [`ResolveError::CapabilityUnavailable`] rather than searching, so a caller cannot bypass
    /// the ambiguity check by resolving directly.
    /// **A11 §5: the close function for a resource, from validated metadata.**
    ///
    /// Selected here rather than searched for at drop time, which is what lets the verifier discharge
    /// §5's five obligations before emission. A resource whose provider declares no `is_close_for`
    /// for it is an ERROR, never an empty result: obligation 5 says a resource reaching emission
    /// without a close is a leak the ABI cannot detect, because the provider never learns the handle
    /// was abandoned.
    ///
    /// Ambiguity is refused too. Two closes for one resource would be two destruction paths, and
    /// picking either by order would make the produced binary depend on something other than what
    /// the program declared — the same rule provider selection already applies.
    pub fn close_for(&self, resource_type: &str) -> Result<ValidatedProviderCall, ResolveError> {
        let mut found: Vec<(&DeclaredProvider, &crate::provider_abi::FunctionDecl)> = Vec::new();
        for provider in self.providers() {
            for function in &provider.metadata.functions {
                if function.is_close_for.as_deref() == Some(resource_type) {
                    found.push((provider, function));
                }
            }
        }
        match found.as_slice() {
            [(provider, function)] => {
                self.resolve(&function.capability, &function.name)
                    .map(|mut call| {
                        // The capability the close belongs to is the declaring function's own, which
                        // `resolve` already enforces (V-PROV-3); this only names the provider for clarity.
                        call.provider = provider.metadata.identity.clone();
                        call
                    })
            }
            [] => Err(ResolveError::NoCloseForResource {
                resource_type: resource_type.to_string(),
            }),
            many => Err(ResolveError::AmbiguousClose {
                resource_type: resource_type.to_string(),
                functions: many.iter().map(|(_, f)| f.name.clone()).collect(),
            }),
        }
    }

    pub fn resolve(
        &self,
        capability: &str,
        function: &str,
    ) -> Result<ValidatedProviderCall, ResolveError> {
        let suppliers: Vec<&DeclaredProvider> = self
            .selected
            .iter()
            .filter(|p| p.metadata.capabilities.iter().any(|c| c == capability))
            .collect();

        let provider = match suppliers.as_slice() {
            [one] => *one,
            [] => {
                return Err(ResolveError::CapabilityUnavailable {
                    capability: capability.to_string(),
                    target: self.target.clone(),
                    declared_for_other_targets: Vec::new(),
                });
            }
            many => {
                let mut providers: Vec<(String, String)> = many
                    .iter()
                    .map(|p| (p.metadata.identity.name.clone(), p.origin.clone()))
                    .collect();
                providers.sort();
                return Err(ResolveError::CapabilityAmbiguous {
                    capability: capability.to_string(),
                    target: self.target.clone(),
                    providers,
                });
            }
        };

        let Some(decl) = provider
            .metadata
            .functions
            .iter()
            .find(|f| f.name == function)
        else {
            return Err(ResolveError::UnknownFunction {
                capability: capability.to_string(),
                function: function.to_string(),
                provider: provider.metadata.identity.name.clone(),
            });
        };

        // A10 §4 invariant 3: membership is checked against the DECLARATION, not assumed from the
        // lookup succeeding. A function reachable by name from the right provider can still belong
        // to a different capability, and that mismatch must not silently widen what the capability
        // admits.
        if decl.capability != capability {
            return Err(ResolveError::FunctionCapabilityMismatch {
                function: function.to_string(),
                provider: provider.metadata.identity.name.clone(),
                declared: decl.capability.clone(),
                requested: capability.to_string(),
            });
        }

        // CD-360: a transfer's foreign resources travel with the call, carrying the OWNING
        // provider's identity and resource list — resolved HERE, where the selected set is, since
        // §6's resolution rule has already proved each one has exactly one owner in it.
        let foreign_resources = provider
            .metadata
            .foreign_resources
            .iter()
            .filter_map(|foreign| {
                let owner = self.selected.iter().find(|p| {
                    p.metadata
                        .resource_types
                        .iter()
                        .any(|rt| rt == &foreign.resource)
                })?;
                Some(crate::mir::ForeignResourceCall {
                    provider: owner.metadata.identity.name.clone(),
                    resource: foreign.resource.clone(),
                    owner_resource_types: owner.metadata.resource_types.clone(),
                })
            })
            .collect();

        Ok(ValidatedProviderCall {
            provider: provider.metadata.identity.clone(),
            capability: capability.to_string(),
            function: decl.clone(),
            target_triple: self.target.clone(),
            foreign_resources,
            provider_crate: provider.crate_name.clone(),
            provider_resource_types: provider.metadata.resource_types.clone(),
            provider_target_triples: provider.metadata.target_triples.clone(),
            // The provider's declared vocabulary, carried onto every call site resolved from it.
            // Empty is not "unknown": it means every nonzero status is a contract violation.
            status_binding: provider.status_binding.clone(),
        })
    }
}

/// Interns [`ValidatedProviderCall`] records into the program-level arena, returning the
/// [`ProviderCallId`] a `Callee::Provider` carries.
///
/// Deduplicating is what makes the arena deterministic: one provider function called from ten
/// bodies produces one record and one id, in first-resolution order, so the same program lowers to
/// the same arena regardless of body iteration order.
#[derive(Debug, Clone, Default)]
pub struct ProviderCallArena {
    calls: Vec<ValidatedProviderCall>,
}

impl ProviderCallArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, call: ValidatedProviderCall) -> ProviderCallId {
        if let Some(index) = self.calls.iter().position(|existing| *existing == call) {
            return ProviderCallId(index as u32);
        }
        let index = self.calls.len();
        self.calls.push(call);
        ProviderCallId(index as u32)
    }

    pub fn get(&self, id: ProviderCallId) -> Option<&ValidatedProviderCall> {
        self.calls.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.calls.len()
    }

    pub fn is_empty(&self) -> bool {
        self.calls.is_empty()
    }

    /// Hands the arena to `MirProgram::provider_calls`.
    pub fn into_calls(self) -> Vec<ValidatedProviderCall> {
        self.calls
    }
}
