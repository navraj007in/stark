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

use crate::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy};
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
pub fn first_party() -> Vec<DeclaredProvider> {
    vec![stark_time()]
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
        _ => None,
    }
}

fn stark_time() -> DeclaredProvider {
    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-time".to_string(),
                semver: (0, 1, 0),
                abi_version: crate::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["clock".to_string()],
            resource_types: vec![],
            functions: vec![
                FunctionDecl {
                    name: "stark_time_monotonic_now_ns".to_string(),
                    capability: "clock".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_time_unix_now".to_string(),
                    capability: "clock".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::I64)],
                    is_close_for: None,
                    may_block: false,
                },
            ],
        },
        crate_name: "stark-time-native".to_string(),
        origin: "stark-time/native/Cargo.toml".to_string(),
    }
}
