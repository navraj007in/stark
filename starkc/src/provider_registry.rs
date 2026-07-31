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
    vec![
        stark_time(),
        stark_env(),
        stark_file(),
        stark_net(),
        stark_random(),
    ]
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
                    // TWO out-slots: seconds and nanoseconds. Mirrored wrong once (one slot), which
                    // generated a call passing a single pointer -- the provider found its second
                    // argument null and aborted, exactly as it is written to. Only execution caught
                    // it; metadata validation cannot, because a one-slot declaration is internally
                    // consistent and simply describes a different function.
                    params: vec![
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::U32),
                    ],
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
    // Codes 1-8 as `stark-file/native/src/lib.rs` declares them. WP-IO.1 bounds the package-facing
    // vocabulary to the minimal native file API and keeps undeclared codes as provider contract
    // violations.
    let mut status = StatusBinding::new();
    status.declare(1, "IOError::NotFound");
    status.declare(2, "IOError::PermissionDenied");
    status.declare(3, "IOError::InvalidInput");
    status.declare(4, "IOError::InvalidData");
    status.declare(5, "IOError::IsDirectory");
    status.declare(6, "IOError::AlreadyExists");
    status.declare(7, "IOError::Unsupported");
    status.declare(8, "IOError::Other");

    let file = "file".to_string();
    let io_file = "io_file".to_string();
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
            resource_types: vec![file.clone(), io_file.clone()],
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
                    is_close_for: Some(file.clone()),
                    may_block: true,
                },
                // --- `io_file`: the package-facing file resource ---
                //
                // `file` above is Core-owned — `ResourceRegistry::builtin()` binds it to
                // `CoreType::File` on the legacy path, and CD-224 forbids a package from claiming
                // it. That is correct and stays. But it left a STARK package with no way to own a
                // file handle at all, and the first attempt to get one worked by deleting CD-224
                // and two verifier guards, which produced the half-migrated state SELECT-C exists
                // to refuse.
                //
                // A second resource identity answers the need without touching any of that.
                // `io_file` is an ordinary A11 host resource: not in the builtin registry, so a
                // package may declare it; not `LegacyCore`, so MIR-0027 does not fire; wholly on
                // the `HostResource` path, so its close runs from a `Drop` terminator and the
                // "MIR owns the only close" rule holds without an exemption.
                //
                // Distinct SYMBOLS, not a second binding of the same ones: a `FunctionDecl` names
                // one symbol and one resource type, and the provider tags handles by type, so a
                // `file` handle reaching an `io_file` entry point aborts instead of being
                // reinterpreted.
                //
                // Core `File`'s migration off the legacy path is untouched by this and remains
                // open. It is a three-engine change; this is not a substitute for it, only a way
                // for packages to have working file IO that does not wait on it.
                FunctionDecl {
                    name: "stark_iofile_open".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: io_file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_create".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: io_file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_read".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_write".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_complete".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleBorrowed {
                        resource_type: io_file.clone(),
                    }],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_close".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: io_file.clone(),
                    }],
                    is_close_for: Some(io_file.clone()),
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_open_options".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::ScalarIn(ScalarTy::U32),
                        AbiParam::HandleOut {
                            resource_type: io_file.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_seek".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::ScalarIn(ScalarTy::U8),
                        AbiParam::ScalarIn(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_sync".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::HandleBorrowed {
                        resource_type: io_file.clone(),
                    }],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_set_len".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::ScalarIn(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_metadata".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: io_file.clone(),
                        },
                        AbiParam::ScalarOut(ScalarTy::U8),
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iopath_metadata".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U8),
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iopath_exists".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::Bool),
                        AbiParam::ScalarOut(ScalarTy::U8),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_remove".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::BufferIn],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_rename".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::BufferIn, AbiParam::BufferIn],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iofile_copy".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iodir_create".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::BufferIn],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iodir_remove".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![AbiParam::BufferIn],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_iodir_list".to_string(),
                    capability: "filesystem".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::U64),
                        AbiParam::ScalarOut(ScalarTy::Bool),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
            ],
        },
        crate_name: "stark-file-native".to_string(),
        origin: "stark-file/native/Cargo.toml".to_string(),
        status_binding: status,
    }
}

/// `stark-net` — blocking TCP (WP-C7.8.6).
///
/// **Its resource types are deliberately unbound, and that is Packet 4 working rather than a gap.**
/// `tcp_listener` and `tcp_stream` are *package* types, so binding them needs a STARK package
/// declaring them — there is no Core type to point at, and adding one would be the CE1 Core change
/// Packet 4 ruled against. Every `stark-net` function carries a handle, so none of them plans yet;
/// they report MIR-0024 naming the resource type, which is precisely the diagnostic C7.8.2d-4 built
/// for this case.
///
/// What IS live: selection, target applicability, and the status vocabulary — eleven codes, the
/// richest of the four providers, including the `bind`-specific `AddressInUse`.
///
/// Packet 5 admits inbound TCP only through an explicit `stark_tcp_listener_bind(address)`. There
/// is no default address in the declaration and none may be added: a listener is created by a
/// program calling `bind`, never as a side effect of anything else.
fn stark_net() -> DeclaredProvider {
    let mut status = StatusBinding::new();
    status.declare(1, "NetworkError::ConnectionRefused");
    status.declare(2, "NetworkError::TimedOut");
    status.declare(3, "NetworkError::NotFound");
    status.declare(4, "NetworkError::PermissionDenied");
    status.declare(5, "NetworkError::AddressInUse");
    status.declare(6, "NetworkError::InvalidInput");
    status.declare(7, "NetworkError::ConnectionReset");
    status.declare(8, "NetworkError::BrokenPipe");
    status.declare(9, "NetworkError::WouldBlock");
    status.declare(10, "NetworkError::Unsupported");
    status.declare(11, "NetworkError::Other");

    let listener = "tcp_listener".to_string();
    let stream = "tcp_stream".to_string();
    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-net".to_string(),
                semver: (0, 1, 0),
                abi_version: crate::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["tcp".to_string()],
            resource_types: vec![listener.clone(), stream.clone()],
            functions: vec![
                FunctionDecl {
                    name: "stark_tcp_listener_bind".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: listener.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_listener_accept".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: listener.clone(),
                        },
                        AbiParam::HandleOut {
                            resource_type: stream.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_connect".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: stream.clone(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_read".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: stream.clone(),
                        },
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_write".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: stream.clone(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "stark_tcp_listener_close".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: listener.clone(),
                    }],
                    is_close_for: Some(listener),
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_tcp_stream_close".to_string(),
                    capability: "tcp".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: stream.clone(),
                    }],
                    is_close_for: Some(stream),
                    may_block: false,
                },
            ],
        },
        crate_name: "stark-net-native".to_string(),
        origin: "stark-net/native/Cargo.toml".to_string(),
        status_binding: status,
    }
}

/// `stark-random` — OS-backed secure randomness.
///
/// This is deliberately function-shaped: the program supplies a mutable byte buffer, the provider
/// either fills the entire buffer or reports failure. There is no partial-success value and no
/// deterministic fallback behind the secure API.
fn stark_random() -> DeclaredProvider {
    let mut status = StatusBinding::new();
    status.declare(1, "RandomError::Unavailable");
    status.declare(2, "RandomError::LimitExceeded");
    status.declare(3, "RandomError::Other");

    DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: "stark-std-random".to_string(),
                semver: (0, 1, 0),
                abi_version: crate::provider_abi::ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["random".to_string()],
            resource_types: vec![],
            functions: vec![FunctionDecl {
                name: "stark_random_secure_fill".to_string(),
                capability: "random".to_string(),
                params: vec![AbiParam::BufferInOut],
                is_close_for: None,
                may_block: false,
            }],
        },
        crate_name: "stark-random-native".to_string(),
        origin: "stark-random/native/Cargo.toml".to_string(),
        status_binding: status,
    }
}
