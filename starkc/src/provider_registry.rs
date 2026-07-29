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
use crate::provider_bind::StatusBinding;
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
    vec![stark_time(), stark_env(), stark_file()]
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
        // stark-time declares NO recoverable status. Empty is the meaningful value: every nonzero
        // status from it is a contract violation.
        status_binding: StatusBinding::new(),
    }
}

/// `stark-env` — process arguments and environment (WP-C7.8.3).
///
/// Both capabilities are **read-only** (Packet 5): there is no environment-mutating function, and
/// none may be added in C7.8.
///
/// The two-call shape — `_len` then `_fill` — is what ABI §9's borrowed buffers require. A provider
/// cannot allocate for the caller, so the caller asks how much room it needs, allocates, and passes
/// a `BufferInOut` for the provider to write into. `stark_env_var_len` additionally reports
/// presence through a `ScalarOut(Bool)`, so "absent" is distinguishable from "present and empty"
/// without a sentinel length.
fn stark_env() -> DeclaredProvider {
    // Codes 1-4 as `stark-env/native/src/lib.rs` declares them. This is the first provider with a
    // non-empty vocabulary, so it is the first place channel one is real rather than vacuous.
    let mut status = StatusBinding::new();
    status.declare(1, "ProcessError::InvalidName");
    status.declare(2, "ProcessError::InvalidEncoding");
    status.declare(3, "ProcessError::BufferTooSmall");
    status.declare(4, "ProcessError::Unsupported");

    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-env".to_string(),
                semver: (0, 1, 0),
                abi_version: crate::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["process.args".to_string(), "process.env".to_string()],
            resource_types: vec![],
            functions: vec![
                FunctionDecl {
                    name: "stark_env_args_len".to_string(),
                    capability: "process.args".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_env_args_fill".to_string(),
                    capability: "process.args".to_string(),
                    params: vec![AbiParam::BufferInOut, AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_env_var_len".to_string(),
                    capability: "process.env".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_env_var_fill".to_string(),
                    capability: "process.env".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: false,
                },
            ],
        },
        crate_name: "stark-env-native".to_string(),
        origin: "stark-env/native/Cargo.toml".to_string(),
        status_binding: status,
    }
}

/// `stark-file` — file I/O (WP-C7.8.4). The first provider with a **resource type**.
///
/// Its shape is Packet 3's close semantics made concrete. `stark_file_complete` is the recoverable
/// completion operation, taking a *borrowed* handle so it can fail and be handled; `stark_file_close`
/// is the ABI close, taking a **consumed** handle and declared `is_close_for: Some("file")`. That
/// separation is exactly ABI §13.1's rule that anything fallible and argument-bearing must be a
/// distinct call made *before* Drop — the close itself has nowhere to put a result.
///
/// Packet 4 holds: this provider supplies the Core `File` surface's needs without adding a Core
/// symbol. `read`/`write` are the byte primitives package conveniences layer over.
fn stark_file() -> DeclaredProvider {
    // Codes 1-8 as `stark-file/native/src/lib.rs` declares them. `IOError` has five variants
    // (STD-IO-001) and this vocabulary has eight, which is not a contradiction: the package binding
    // maps codes to Core's variants, and `Other(String)` is where the surplus lands. The compiler
    // treats every name here as opaque.
    let mut status = StatusBinding::new();
    status.declare(1, "IOError::NotFound");
    status.declare(2, "IOError::PermissionDenied");
    status.declare(3, "IOError::InvalidInput");
    status.declare(4, "IOError::Other(invalid encoding)");
    status.declare(5, "IOError::Other(is a directory)");
    status.declare(6, "IOError::AlreadyExists");
    status.declare(7, "IOError::Other(unsupported)");
    status.declare(8, "IOError::Other");

    let file = "file".to_string();
    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-file".to_string(),
                semver: (0, 1, 0),
                abi_version: crate::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["filesystem".to_string()],
            resource_types: vec![file.clone()],
            functions: vec![
                FunctionDecl {
                    name: "stark_file_open".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_create".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_read".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: file.clone(),
                        },
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_write".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: file.clone(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_complete".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleBorrowed {
                        resource_type: file.clone(),
                    }],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_file_close".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: file.clone(),
                    }],
                    is_close_for: Some(file),
                    may_block: true,
                },
            ],
        },
        crate_name: "stark-file-native".to_string(),
        origin: "stark-file/native/Cargo.toml".to_string(),
        status_binding: status,
    }
}
