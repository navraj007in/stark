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

        Ok(ValidatedProviderCall {
            provider: provider.metadata.identity.clone(),
            capability: capability.to_string(),
            function: decl.clone(),
            target_triple: self.target.clone(),
            provider_target_triples: provider.metadata.target_triples.clone(),
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
