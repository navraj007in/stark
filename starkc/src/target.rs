//! WP-C6.4a — target classification and build preflight.
//!
//! `WP-C6-ENTRY.md` §33 requires the build path to identify host and selected target separately,
//! accept only qualified targets, reject unsupported ones *before linking*, name the supported
//! targets in the diagnostic, distinguish an unsupported target from a missing toolchain, and
//! select the layout contract and executable naming from the **target** rather than from the host.
//!
//! # Why this is one module
//!
//! Before this package the compiler had no notion of a target at all. The generated-crate driver
//! read `host:` out of `rustc -vV` and used it as the target
//! (`backend/generated_rust/build.rs`), the layout contract `stark-64-v1` was applied
//! unconditionally to whatever that turned out to be
//! (`backend/generated_rust/mod.rs`'s default), and the executable suffix came from the
//! *compiler's own* `std::env::consts::EXE_SUFFIX`. Each of those is correct exactly while host
//! and target are the same string, and silently wrong otherwise — so they were three separate
//! places for the same assumption to survive. This module is the single place a triple is
//! interpreted; §8.2 of the execution plan states that as a design constraint ("there must be one
//! canonical target-classification function or type"), and every other site now asks it.
//!
//! # What this is NOT
//!
//! Not the user-facing `--target` feature: C7 owns target selection, cross-compilation, and
//! toolchain installation (`WP-C6-ENTRY.md` §33's closing line). C6.4 needs the *classification*
//! and the *rejection*, which is why [`select`] takes an optional requested triple that the CLI
//! does not yet supply — the seam exists so the qualification harness and the test inventory can
//! drive every branch without a flag that would promise cross-compilation the compiler cannot do.
//!
//! # Tier is a qualification claim, not an admission rule
//!
//! `WP-C6-ENTRY.md` §32 grades targets into tiers for the purpose of the **Gate C6 claim**: a
//! positive claim requires both Tier-1 targets. It does not say Tier-2/Tier-3 builds are refused,
//! and they are not — CI builds and runs the generated native path on `x86_64-pc-windows-msvc`
//! today. So admission is "this compiler names the target" and the tier travels with it into the
//! evidence record, where the distinction actually matters.

use std::fmt;

/// The Gate C6 qualification grade of a named target (`WP-C6-ENTRY.md` §32).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    /// Required for a positive Gate C6 claim. Both Tier-1 targets must produce a real run.
    One,
    /// Built where bounded; a gap report rather than a qualification claim.
    Two,
    /// Named, admitted, no C6 qualification claim required.
    Three,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::One => "tier-1",
            Tier::Two => "tier-2",
            Tier::Three => "tier-3",
        })
    }
}

/// Everything the build path may derive from a target. Deliberately a closed record rather than a
/// set of `fn(triple) -> …` helpers: a caller cannot reach one property while forgetting another,
/// and adding a target means adding one table row rather than editing several matches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TargetSpec {
    pub triple: &'static str,
    pub tier: Tier,
    /// The named layout contract (`layout::contract_for`) this target answers `size_of`/
    /// `align_of` from. Not inferred from the pointer width — LAYOUT-ABI-001 makes the contract a
    /// declared property of a *named* target, so an unnamed target has no contract at all rather
    /// than a plausible one.
    pub layout_contract: &'static str,
    /// The suffix the produced executable carries **on this target**, not on the compiler's host.
    pub executable_suffix: &'static str,
    /// The target's pointer width in bits. Recorded because the runtime's checked-index surface
    /// (`stark_runtime::vec`) takes `u64` indices and must compare them without narrowing; see
    /// `WP-C6.4.md` F3.
    pub pointer_width: u32,
}

/// Every target this compiler names. Ordered Tier-1 first so diagnostics list the targets a C6
/// claim actually depends on before the ones it does not.
const KNOWN_TARGETS: &[TargetSpec] = &[
    TargetSpec {
        triple: "aarch64-apple-darwin",
        tier: Tier::One,
        layout_contract: "stark-64-v1",
        executable_suffix: "",
        pointer_width: 64,
    },
    TargetSpec {
        triple: "x86_64-unknown-linux-gnu",
        tier: Tier::One,
        layout_contract: "stark-64-v1",
        executable_suffix: "",
        pointer_width: 64,
    },
    TargetSpec {
        triple: "x86_64-pc-windows-msvc",
        tier: Tier::Two,
        layout_contract: "stark-64-v1",
        executable_suffix: ".exe",
        pointer_width: 64,
    },
    TargetSpec {
        triple: "x86_64-apple-darwin",
        tier: Tier::Three,
        layout_contract: "stark-64-v1",
        executable_suffix: "",
        pointer_width: 64,
    },
];

/// The Tier-1 triples, in the order a diagnostic should name them.
/// The triple this compiler binary was built for, derived **without probing a Rust toolchain**.
///
/// `native_toolchain::discover` reports a host triple by running `rustc -vV`, which is correct and
/// is what a native build must use — a build needs the toolchain anyway. `stark test` does not: it
/// runs through the reference interpreter and compiles nothing. Requiring `rustc` to run
/// interpreter-only tests would make a machine without a Rust toolchain unable to test a STARK
/// package at all, and `native_build.rs`'s own ordering note records the failure mode this creates
/// — a probe reached too eagerly turns every source error into "rustc not found".
///
/// `std::env::consts` is the compiler's own build configuration, so this is exact for the binary
/// asking, which is the only thing the caller needs: the triple gates provider AVAILABILITY, and a
/// provider available on the host is what a host-run test will use.
///
/// `None` when the host is not a target this compiler knows. That is a real answer, not a failure
/// to compute one — a caller must decide whether an unknown host is fatal, and silently guessing a
/// triple would bind providers that were never declared for it.
pub fn host_triple_of_this_build() -> Option<&'static str> {
    let triple = match (std::env::consts::ARCH, std::env::consts::OS) {
        ("aarch64", "macos") => "aarch64-apple-darwin",
        ("x86_64", "macos") => "x86_64-apple-darwin",
        ("x86_64", "linux") => "x86_64-unknown-linux-gnu",
        ("x86_64", "windows") => "x86_64-pc-windows-msvc",
        _ => return None,
    };
    // Routed through `classify` rather than returned directly, so this cannot name a triple the
    // rest of the module does not know about.
    classify(triple).map(|spec| spec.triple)
}

pub fn tier1_triples() -> Vec<&'static str> {
    KNOWN_TARGETS
        .iter()
        .filter(|t| t.tier == Tier::One)
        .map(|t| t.triple)
        .collect()
}

/// Every named target, for evidence records and tests.
pub fn known_targets() -> &'static [TargetSpec] {
    KNOWN_TARGETS
}

/// Exact-match lookup. Deliberately not a prefix or "contains" match: `x86_64-unknown-linux-musl`
/// is not `x86_64-unknown-linux-gnu`, and an unknown triple must not inherit a contract by
/// resembling a known one (§8.2: "do not allow an unknown triple to inherit `stark-64-v1` merely
/// because it is 64-bit").
pub fn classify(triple: &str) -> Option<&'static TargetSpec> {
    KNOWN_TARGETS.iter().find(|t| t.triple == triple)
}

/// §8.3's classification. The user-facing prose may change; these variants are the stable contract
/// the tests and the qualification harness assert against.
///
/// `SupportedAndAvailable` is represented by `Ok(TargetSelection)` rather than a variant here — a
/// success carries the resolved spec, and giving it an error-shaped twin invites code that
/// matches on the class and then re-derives the spec.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TargetError {
    /// This compiler does not name the triple. Nothing downstream may guess a layout contract,
    /// pointer width, or executable suffix for it.
    UnsupportedByStark {
        requested: String,
        supported: Vec<&'static str>,
    },
    /// A target this compiler *does* name, whose Rust toolchain support is not installed here.
    /// Distinct from `UnsupportedByStark` because the remedy is different — install a target, not
    /// change one — and §8.5's test inventory requires the two to be separately observable.
    SupportedButToolchainMissing {
        requested: &'static str,
        host: String,
        detail: String,
    },
    /// The host and the selected target disagree in a way the build cannot honour. Today this is
    /// reachable only through the injectable seam (cross-compilation is C7), but the class exists
    /// because the *record* it guards — host and target as separate values — exists now.
    HostOrTargetMetadataMismatch { host: String, selected: String },
    /// The build was asked to answer layout queries from a contract other than the one the
    /// selected target declares. §8.3's list is a minimum, not a maximum; this is a metadata
    /// mismatch specific enough to deserve saying which two names disagreed, because a build that
    /// records one target and measures another is wrong in a way no single triple explains.
    LayoutContractMismatch {
        target: &'static str,
        declared: &'static str,
        requested: String,
    },
}

impl fmt::Display for TargetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetError::UnsupportedByStark {
                requested,
                supported,
            } => write!(
                f,
                "`{requested}` is not a target this compiler supports. Supported tier-1 targets: {}",
                supported.join(", ")
            ),
            TargetError::SupportedButToolchainMissing {
                requested,
                host,
                detail,
            } => write!(
                f,
                "`{requested}` is a supported target, but its Rust toolchain support is not \
                 available on this host (`{host}`): {detail}"
            ),
            TargetError::HostOrTargetMetadataMismatch { host, selected } => write!(
                f,
                "host `{host}` cannot produce a build for selected target `{selected}`"
            ),
            TargetError::LayoutContractMismatch {
                target,
                declared,
                requested,
            } => write!(
                f,
                "target `{target}` declares layout contract `{declared}`, but this build was \
                 asked for `{requested}`"
            ),
        }
    }
}

/// Host and selected target as two separate values, plus the resolved spec. §33 requires the two
/// to be identified separately even while they are always equal in practice — a record that
/// *cannot* distinguish them is one that will silently report the host as the target the day
/// cross-compilation exists.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TargetSelection {
    pub host_triple: String,
    pub selected: &'static TargetSpec,
}

impl TargetSelection {
    pub fn selected_triple(&self) -> &'static str {
        self.selected.triple
    }

    /// True when this build is for the machine it runs on — the only shape C6.4 qualifies, and the
    /// precondition for treating a host toolchain probe as a target toolchain probe.
    pub fn is_native_build(&self) -> bool {
        self.host_triple == self.selected.triple
    }
}

/// Whether the Rust toolchain on this host can build for a given target.
///
/// Injectable because §8.5 is explicit that "tests must not require uninstalling the developer's
/// toolchain": the missing-toolchain branch has to be reachable without breaking the machine
/// running the test.
pub trait TargetAvailability {
    fn is_available(&self, host: &str, target: &str) -> Result<(), String>;
}

/// The production probe. C7 owns cross-compilation, so the only target this compiler can actually
/// build for is the host's own — and saying so through `SupportedButToolchainMissing` is accurate
/// rather than evasive: `x86_64-unknown-linux-gnu` really is a supported STARK target whose
/// toolchain is unavailable when you ask for it from an arm64 Mac.
pub struct HostOnlyAvailability;

impl TargetAvailability for HostOnlyAvailability {
    fn is_available(&self, host: &str, target: &str) -> Result<(), String> {
        if host == target {
            Ok(())
        } else {
            Err(format!(
                "this compiler builds only for its host target; cross-compilation to `{target}` \
                 is not part of Core v1 (it belongs to the C7 target feature)"
            ))
        }
    }
}

/// Resolve the target for a build.
///
/// `requested` is `None` for every build the CLI performs today, which means "the host". It is not
/// dead scaffolding for C7: it is what lets the preflight tests drive an unsupported triple, a
/// supported-but-unavailable triple, and a host/target mismatch without a flag that would advertise
/// cross-compilation.
///
/// The host itself is validated. A build running on a triple this compiler does not name is
/// rejected here rather than proceeding to guess a layout contract for it.
pub fn select(
    host_triple: &str,
    requested: Option<&str>,
    availability: &dyn TargetAvailability,
) -> Result<TargetSelection, TargetError> {
    let wanted = requested.unwrap_or(host_triple);
    let selected = classify(wanted).ok_or_else(|| TargetError::UnsupportedByStark {
        requested: wanted.to_string(),
        supported: tier1_triples(),
    })?;

    // The host is checked too, and after the target: a request for an unsupported *target* should
    // report that target, not the host that happens to also be unnamed.
    if classify(host_triple).is_none() {
        return Err(TargetError::UnsupportedByStark {
            requested: host_triple.to_string(),
            supported: tier1_triples(),
        });
    }

    availability
        .is_available(host_triple, selected.triple)
        .map_err(|detail| TargetError::SupportedButToolchainMissing {
            requested: selected.triple,
            host: host_triple.to_string(),
            detail,
        })?;

    Ok(TargetSelection {
        host_triple: host_triple.to_string(),
        selected,
    })
}

/// The check the native build path runs before it emits anything.
///
/// [`select`] answers "is this target named, and can this toolchain reach it". `preflight` adds
/// the one constraint the *generated-crate* build imposes on top: the crate is built by invoking
/// the host's Cargo with no `--target`, so the only artifact it can produce is a host artifact.
/// A selection that says otherwise would put a target triple in the build manifest that does not
/// describe the binary next to it — metadata that lies is worse than metadata that is absent, so
/// it is refused rather than recorded.
///
/// Splitting it from `select` keeps the classification reusable by the qualification harness and
/// the evidence record, which legitimately want to classify targets they are not building for.
pub fn preflight(
    host_triple: &str,
    requested: Option<&str>,
    availability: &dyn TargetAvailability,
) -> Result<TargetSelection, TargetError> {
    let selection = select(host_triple, requested, availability)?;
    if !selection.is_native_build() {
        return Err(TargetError::HostOrTargetMetadataMismatch {
            host: selection.host_triple,
            selected: selection.selected.triple.to_string(),
        });
    }
    Ok(selection)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe that says yes to everything, so a test can isolate classification from availability.
    struct AlwaysAvailable;
    impl TargetAvailability for AlwaysAvailable {
        fn is_available(&self, _host: &str, _target: &str) -> Result<(), String> {
            Ok(())
        }
    }

    /// A probe that says no to everything — §8.5(7)'s "supported target but missing Rust
    /// target/toolchain", reachable without touching the developer's installation.
    struct NeverAvailable;
    impl TargetAvailability for NeverAvailable {
        fn is_available(&self, _host: &str, _target: &str) -> Result<(), String> {
            Err("rust-std for this target is not installed".to_string())
        }
    }

    #[test]
    fn tier1_triples_are_the_two_gate_c6_targets() {
        assert_eq!(
            tier1_triples(),
            vec!["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"]
        );
    }

    #[test]
    fn macos_arm64_is_tier1_with_no_suffix() {
        let spec = classify("aarch64-apple-darwin").unwrap();
        assert_eq!(spec.tier, Tier::One);
        assert_eq!(spec.executable_suffix, "");
        assert_eq!(spec.layout_contract, "stark-64-v1");
        assert_eq!(spec.pointer_width, 64);
    }

    #[test]
    fn linux_x64_is_tier1_with_no_suffix() {
        let spec = classify("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(spec.tier, Tier::One);
        assert_eq!(spec.executable_suffix, "");
        assert_eq!(spec.layout_contract, "stark-64-v1");
    }

    #[test]
    fn windows_x64_is_tier2_and_selects_the_exe_suffix() {
        let spec = classify("x86_64-pc-windows-msvc").unwrap();
        assert_eq!(spec.tier, Tier::Two);
        assert_eq!(spec.executable_suffix, ".exe");
    }

    #[test]
    fn intel_mac_is_tier3() {
        assert_eq!(classify("x86_64-apple-darwin").unwrap().tier, Tier::Three);
    }

    /// The §8.2 prohibition, as a test: 64-bitness is not a licence to inherit `stark-64-v1`.
    #[test]
    fn an_unknown_64_bit_triple_is_rejected_rather_than_inheriting_the_contract() {
        assert!(classify("x86_64-unknown-linux-musl").is_none());
        assert!(classify("aarch64-unknown-linux-gnu").is_none());
        let err = select("x86_64-unknown-linux-musl", None, &AlwaysAvailable).unwrap_err();
        match err {
            TargetError::UnsupportedByStark {
                ref requested,
                ref supported,
            } => {
                assert_eq!(requested, "x86_64-unknown-linux-musl");
                assert_eq!(supported, &tier1_triples());
            }
            other => panic!("expected UnsupportedByStark, got {other:?}"),
        }
        assert!(err.to_string().contains("aarch64-apple-darwin"));
        assert!(err.to_string().contains("x86_64-unknown-linux-gnu"));
    }

    #[test]
    fn an_unknown_32_bit_triple_is_rejected() {
        assert!(classify("i686-unknown-linux-gnu").is_none());
        assert!(matches!(
            select("i686-unknown-linux-gnu", None, &AlwaysAvailable),
            Err(TargetError::UnsupportedByStark { .. })
        ));
    }

    /// Every named target is 64-bit, which is what makes the runtime's `u64` index surface safe.
    /// A 32-bit target may only be added together with the width-independent bounds work
    /// (`WP-C6.4.md` F3), so this test is the tripwire for that pairing.
    #[test]
    fn every_named_target_is_64_bit_and_uses_the_declared_contract() {
        for spec in known_targets() {
            assert_eq!(spec.pointer_width, 64, "{}", spec.triple);
            assert_eq!(spec.layout_contract, "stark-64-v1", "{}", spec.triple);
        }
    }

    #[test]
    fn a_supported_target_with_no_toolchain_is_a_different_class_from_an_unsupported_one() {
        let missing = select("aarch64-apple-darwin", None, &NeverAvailable).unwrap_err();
        assert!(matches!(
            missing,
            TargetError::SupportedButToolchainMissing { .. }
        ));
        let unsupported = select("sparc64-unknown-linux-gnu", None, &AlwaysAvailable).unwrap_err();
        assert!(matches!(
            unsupported,
            TargetError::UnsupportedByStark { .. }
        ));
        assert_ne!(missing, unsupported);
        // The prose must separate them too: a reader has to know whether to install or to retarget.
        assert!(missing.to_string().contains("not available on this host"));
        assert!(unsupported
            .to_string()
            .contains("not a target this compiler"));
    }

    #[test]
    fn host_and_selected_target_are_recorded_separately() {
        let selection = select("aarch64-apple-darwin", None, &AlwaysAvailable).unwrap();
        assert_eq!(selection.host_triple, "aarch64-apple-darwin");
        assert_eq!(selection.selected_triple(), "aarch64-apple-darwin");
        assert!(selection.is_native_build());
    }

    /// Cross-compilation is C7's, and the production probe says so as a *toolchain* fact rather
    /// than by pretending the target is unsupported.
    #[test]
    fn the_production_probe_admits_only_the_host_target() {
        assert!(HostOnlyAvailability
            .is_available("aarch64-apple-darwin", "aarch64-apple-darwin")
            .is_ok());
        let err = select(
            "aarch64-apple-darwin",
            Some("x86_64-unknown-linux-gnu"),
            &HostOnlyAvailability,
        )
        .unwrap_err();
        match err {
            TargetError::SupportedButToolchainMissing { requested, .. } => {
                assert_eq!(requested, "x86_64-unknown-linux-gnu")
            }
            other => panic!("expected SupportedButToolchainMissing, got {other:?}"),
        }
    }

    /// An unnamed *host* is rejected as well — otherwise the compiler would happily build on a
    /// platform it has no contract for, as long as nothing asked about the target.
    #[test]
    fn an_unnamed_host_is_rejected_even_when_the_request_is_valid() {
        let err = select(
            "powerpc64-unknown-linux-gnu",
            Some("aarch64-apple-darwin"),
            &AlwaysAvailable,
        )
        .unwrap_err();
        match err {
            TargetError::UnsupportedByStark { requested, .. } => {
                assert_eq!(requested, "powerpc64-unknown-linux-gnu")
            }
            other => panic!("expected UnsupportedByStark for the host, got {other:?}"),
        }
    }

    /// The generated crate is built by the host's Cargo with no `--target`, so a non-host
    /// selection cannot be honoured. It is refused rather than recorded as a target triple that
    /// does not describe the binary beside it.
    #[test]
    fn preflight_refuses_a_selection_the_generated_build_cannot_produce() {
        let err = preflight(
            "aarch64-apple-darwin",
            Some("x86_64-unknown-linux-gnu"),
            &AlwaysAvailable,
        )
        .unwrap_err();
        match err {
            TargetError::HostOrTargetMetadataMismatch {
                ref host,
                ref selected,
            } => {
                assert_eq!(host, "aarch64-apple-darwin");
                assert_eq!(selected, "x86_64-unknown-linux-gnu");
            }
            other => panic!("expected HostOrTargetMetadataMismatch, got {other:?}"),
        }
        assert!(err.to_string().contains("cannot produce a build"));
    }

    #[test]
    fn preflight_accepts_a_host_build() {
        let selection = preflight("x86_64-unknown-linux-gnu", None, &AlwaysAvailable).unwrap();
        assert_eq!(selection.selected_triple(), "x86_64-unknown-linux-gnu");
        assert_eq!(selection.selected.tier, Tier::One);
    }

    /// An unsupported *target* reports the target, not the host, even when both are unnamed.
    #[test]
    fn an_unsupported_target_is_reported_before_an_unsupported_host() {
        let err = select(
            "powerpc64-unknown-linux-gnu",
            Some("mips-unknown-linux-gnu"),
            &AlwaysAvailable,
        )
        .unwrap_err();
        match err {
            TargetError::UnsupportedByStark { requested, .. } => {
                assert_eq!(requested, "mips-unknown-linux-gnu")
            }
            other => panic!("expected the target to be reported, got {other:?}"),
        }
    }
}
