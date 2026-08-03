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
    // HC9. The first provider that CONSUMES another's resource (CD-360): its manifest declares
    // `consumes: [{ "provider": "stark-std-net", "resource": "tcp_stream" }]`, and
    // `ProviderSet::select` refuses the build if that owner is not uniquely present in the
    // selected set.
    (
        "stark-tls-native",
        include_str!("../providers/stark-tls-native.json"),
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

/// Where a BUILT-IN provider's crate lives, resolved against `root`.
///
/// CD-363 deleted `crate_location`, whose hardcoded `match` over five names was the last piece of
/// the mechanism that made every native capability a compiler-source change. This is not that
/// returning: it is a lookup OVER the manifests, so the path data still lives in exactly one place
/// and this function cannot disagree with it. Adding a provider means adding a manifest, and
/// nothing here changes.
///
/// Built-in only, by design. An external provider's root comes from the application's declaration,
/// which is what makes it containable — see `provider_manifest::AdmittedProvider`.
pub fn built_in_crate_location(crate_name: &str, root: &std::path::Path) -> Option<PathBuf> {
    first_party()
        .into_iter()
        .find(|provider| provider.crate_name == crate_name)
        .map(|provider| root.join(provider.crate_path))
}
