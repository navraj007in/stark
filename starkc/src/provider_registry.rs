//! WP-C7.8 step 3 (CD-212) — the first-party provider set, and how a build selects from it.
//!
//! **Packet 5's admission rule, implemented.** A provider is linked if and only if some package
//! declared its capability by name. There is no discovery: nothing scans a directory, reads an
//! environment variable, or infers a provider from what a program happens to call. The set below
//! is compiled into the compiler, and a capability nobody asked for selects nothing.
//!
//! The sequence a build follows:
//!
//! ```text
//! package manifest `capabilities: [...]`
//! → this registry's declared providers
//! → ProviderSet::select(target, required)     -- validates metadata, checks targets, rejects
//!                                                ambiguity and unavailability
//! → crate name → location
//! → NativeToolchainOptions::provider_crates
//! ```
//!
//! **Metadata is mirrored here rather than read from the provider crate**, because `starkc` cannot
//! depend on a provider crate — the dependency runs the other way, since providers link into
//! programs the compiler produces. Each provider crate carries its own test asserting its metadata
//! validates against this repository's ABI validator, which is what keeps the mirror honest.

use crate::provider_resolve::DeclaredProvider;
use std::path::PathBuf;

/// Every capability the first-party provider set can supply, for diagnostics that need to say what
/// *was* available rather than only what was missing.
pub fn known_capabilities() -> Vec<String> {
    let mut caps: Vec<String> = first_party()
        .iter()
        .flat_map(|p| p.metadata.capabilities.clone())
        .collect();
    caps.sort();
    caps.dedup();
    caps
}

/// The compiler's built-in provider set.
///
/// Adding an entry here is the *entire* act of making a capability available — deliberately, so
/// that "which providers can this compiler link?" has one answer in one place rather than being a
/// property of the filesystem it runs on.
/// The manifests shipping with the compiler, embedded so a first-party provider needs no file on
/// disk and no network — but declared in exactly the format an EXTERNAL provider uses.
///
/// P0.2: "first party" is a TRUST and DEFAULT classification, not an implementation path. Before
/// this, first-party providers were Rust struct literals and external providers had no route at
/// all; keeping both would mean two resolvers, two validation paths, and capability handling that
/// could drift between them. One mechanism means a bug found in either is found in both.
const BUILT_IN_MANIFESTS: &[(&str, &str)] = &[
    (
        "stark-time-native",
        include_str!("../providers/stark-time-native.json"),
    ),
    (
        "stark-env-native",
        include_str!("../providers/stark-env-native.json"),
    ),
    (
        "stark-file-native",
        include_str!("../providers/stark-file-native.json"),
    ),
    (
        "stark-net-native",
        include_str!("../providers/stark-net-native.json"),
    ),
    (
        "stark-random-native",
        include_str!("../providers/stark-random-native.json"),
    ),
];

/// The default provider set: the built-in manifests, parsed through the same loader an external
/// manifest uses.
///
/// A parse failure here is a compiler defect, not user error — these manifests ship with the
/// binary and are covered by `p02_provider_manifest.rs`'s equivalence test — so it panics with the
/// offending provider named rather than silently returning a short set. A silently-missing provider
/// would surface as "capability unsupplied" somewhere far from the cause.
pub fn first_party() -> Vec<DeclaredProvider> {
    BUILT_IN_MANIFESTS
        .iter()
        .map(|(name, text)| {
            crate::provider_manifest::parse_provider_manifest(text, &format!("built-in:{name}"))
                .unwrap_or_else(|error| {
                    panic!("built-in provider manifest `{name}` is malformed: {error}")
                })
        })
        .collect()
}

/// Resolves a provider crate name to its location on this machine.
///
/// Locations are **not** part of the provider set above, and not part of MIR: a crate's path is a
/// property of the checkout doing the build, while its name is a property of the program. Keeping
/// them apart is what lets a verified MIR artefact stay relocation-stable while still naming the
/// providers it needs.
pub fn crate_location(crate_name: &str, repo_root: &std::path::Path) -> Option<PathBuf> {
    match crate_name {
        "stark-time-native" => Some(repo_root.join("stark-time").join("native")),
        "stark-env-native" => Some(repo_root.join("stark-env").join("native")),
        "stark-file-native" => Some(repo_root.join("stark-file").join("native")),
        "stark-net-native" => Some(repo_root.join("stark-net").join("native")),
        "stark-random-native" => Some(repo_root.join("stark-random").join("native")),
        _ => None,
    }
}
