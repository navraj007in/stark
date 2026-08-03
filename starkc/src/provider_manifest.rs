//! **P0.2 / WP-EXTERNAL-PROVIDERS: a provider declared by MANIFEST rather than by compiler source.**
//!
//! > A provider can be supplied outside the compiler repository without modifying compiler source,
//! > while preserving ABI validation, reproducibility, target qualification, and explicit trust
//! > policy.
//!
//! # What this changes, and what it deliberately does not
//!
//! Before this, `provider_registry::first_party()` was a hardcoded `Vec` and `crate_location` a
//! hardcoded `match` over five names. Providers were compiler-integrated extensions, not an
//! ecosystem mechanism: every native capability needed a compiler change, nobody outside the
//! repository could publish one, provider versioning was welded to compiler releases, and trust
//! policy was implicit because we wrote everything that existed.
//!
//! **Discovery changes. Loading does not.** Providers stay statically linked into the generated
//! Cargo workspace — no `dlopen`, no plugins. That is what makes the current safety model work: ABI
//! validation happens before a symbol is ever referenced, and a build either links or fails.
//!
//! **`provider_abi::validate` is unchanged.** Only its input source moves, from a hardcoded `Vec`
//! to a parsed manifest. Every existing ABI rule therefore keeps applying unaltered, which is the
//! smallest change that removes the hardcoding — and the reason this packet does not reopen any
//! question CD-360 just settled.
//!
//! # Schema
//!
//! ```json
//! {
//!   "name": "stark-postgres-native",
//!   "version": "0.1.0",
//!   "provider": {
//!     "abi": "0.1",
//!     "identity": "stark-db-postgres",
//!     "crate": "native",
//!     "crate_path": "native",
//!     "capabilities": ["stark.db.postgres"],
//!     "targets": ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"],
//!     "resources": [
//!       { "name": "pg_connection", "close": "stark_pg_connection_close" }
//!     ],
//!     "consumes": [
//!       { "provider": "stark-std-net", "resource": "tcp_stream" }
//!     ],
//!     "status": { "1": "ConnectionRefused" },
//!     "functions": [
//!       {
//!         "symbol": "stark_pg_connect",
//!         "capability": "stark.db.postgres",
//!         "may_block": true,
//!         "params": [
//!           { "form": "buffer_in" },
//!           { "form": "handle_out", "resource": "pg_connection" }
//!         ]
//!       }
//!     ]
//!   }
//! }
//! ```
//!
//! `consumes` is CD-360's `foreign_resources`: resources this provider may take ownership of but
//! does not close.

use crate::package::{parse_json, JsonValue};
use crate::provider_abi::{
    AbiParam, ForeignResource, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy,
};
use crate::provider_bind::StatusBinding;
use crate::provider_resolve::DeclaredProvider;

/// Why a provider manifest could not be read.
///
/// Every variant names the field, because a manifest is written by someone outside this repository
/// and a diagnostic that does not say which key is wrong is unusable to them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The JSON did not parse at all.
    Malformed(String),
    /// A required key is absent.
    MissingField { path: String },
    /// A key is present with the wrong shape.
    WrongType { path: String, expected: String },
    /// A parameter form this ABI version does not define.
    UnknownParamForm { function: String, form: String },
    /// A scalar type name this ABI version does not define.
    UnknownScalarType { function: String, scalar: String },
    /// A handle parameter with no `resource` key: a handle must say which resource type it carries,
    /// or §13's wrong-resource-type rule has nothing to check against.
    HandleWithoutResource { function: String, form: String },
    /// A status code that is not a non-negative integer.
    InvalidStatusCode { code: String },
    /// **A `crate_path` that escapes the root it was admitted under.**
    ///
    /// An external manifest is written by a third party, by definition not by us. An absolute path
    /// or a `..` component would let it point the build at a crate outside the directory the
    /// application approved — escaping the only containment this mechanism has. Harmless for a
    /// built-in; for the exact class of provider this packet exists to admit safely, it is the
    /// whole containment boundary.
    CratePathEscapesRoot { path: String },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Malformed(why) => {
                write!(f, "provider manifest is not valid JSON: {why}")
            }
            ManifestError::MissingField { path } => write!(f, "missing required field `{path}`"),
            ManifestError::WrongType { path, expected } => {
                write!(f, "field `{path}` must be {expected}")
            }
            ManifestError::UnknownParamForm { function, form } => {
                write!(f, "function `{function}`: unknown parameter form `{form}`")
            }
            ManifestError::UnknownScalarType { function, scalar } => {
                write!(f, "function `{function}`: unknown scalar type `{scalar}`")
            }
            ManifestError::HandleWithoutResource { function, form } => write!(
                f,
                "function `{function}`: `{form}` must name the `resource` it carries"
            ),
            ManifestError::InvalidStatusCode { code } => {
                write!(f, "status code `{code}` must be a non-negative integer")
            }
            ManifestError::CratePathEscapesRoot { path } => write!(
                f,
                "`crate_path` must be relative and must not escape the provider root: `{path}`"
            ),
        }
    }
}

fn object<'a>(
    value: &'a JsonValue,
    path: &str,
) -> Result<&'a std::collections::HashMap<String, JsonValue>, ManifestError> {
    value.as_object().ok_or_else(|| ManifestError::WrongType {
        path: path.to_string(),
        expected: "an object".to_string(),
    })
}

fn field<'a>(
    parent: &'a std::collections::HashMap<String, JsonValue>,
    key: &str,
    path: &str,
) -> Result<&'a JsonValue, ManifestError> {
    parent.get(key).ok_or_else(|| ManifestError::MissingField {
        path: format!("{path}.{key}"),
    })
}

fn string(value: &JsonValue, path: &str) -> Result<String, ManifestError> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| ManifestError::WrongType {
            path: path.to_string(),
            expected: "a string".to_string(),
        })
}

fn array<'a>(value: &'a JsonValue, path: &str) -> Result<&'a [JsonValue], ManifestError> {
    value.as_array().ok_or_else(|| ManifestError::WrongType {
        path: path.to_string(),
        expected: "an array".to_string(),
    })
}

fn string_array(value: &JsonValue, path: &str) -> Result<Vec<String>, ManifestError> {
    array(value, path)?
        .iter()
        .enumerate()
        .map(|(i, item)| string(item, &format!("{path}[{i}]")))
        .collect()
}

fn scalar_ty(name: &str, function: &str) -> Result<ScalarTy, ManifestError> {
    Ok(match name {
        "u8" => ScalarTy::U8,
        "u16" => ScalarTy::U16,
        "u32" => ScalarTy::U32,
        "u64" => ScalarTy::U64,
        "i8" => ScalarTy::I8,
        "i16" => ScalarTy::I16,
        "i32" => ScalarTy::I32,
        "i64" => ScalarTy::I64,
        "bool" => ScalarTy::Bool,
        "f32" => ScalarTy::F32,
        "f64" => ScalarTy::F64,
        other => {
            return Err(ManifestError::UnknownScalarType {
                function: function.to_string(),
                scalar: other.to_string(),
            })
        }
    })
}

fn param(value: &JsonValue, function: &str, path: &str) -> Result<AbiParam, ManifestError> {
    let map = object(value, path)?;
    let form = string(field(map, "form", path)?, &format!("{path}.form"))?;

    // A handle form must name its resource; a scalar form must name its type. Both are refused
    // rather than defaulted, because a default here would silently produce a different ABI than
    // the provider author wrote.
    let resource = || -> Result<String, ManifestError> {
        match map.get("resource") {
            Some(v) => string(v, &format!("{path}.resource")),
            None => Err(ManifestError::HandleWithoutResource {
                function: function.to_string(),
                form: form.clone(),
            }),
        }
    };
    let scalar = || -> Result<ScalarTy, ManifestError> {
        let name = string(field(map, "type", path)?, &format!("{path}.type"))?;
        scalar_ty(&name, function)
    };

    Ok(match form.as_str() {
        "buffer_in" => AbiParam::BufferIn,
        "buffer_in_out" => AbiParam::BufferInOut,
        "scalar_in" => AbiParam::ScalarIn(scalar()?),
        "scalar_out" => AbiParam::ScalarOut(scalar()?),
        "scalar_in_out" => AbiParam::ScalarInOut(scalar()?),
        "handle_borrowed" => AbiParam::HandleBorrowed {
            resource_type: resource()?,
        },
        "handle_consumed" => AbiParam::HandleConsumed {
            resource_type: resource()?,
        },
        "handle_out" => AbiParam::HandleOut {
            resource_type: resource()?,
        },
        other => {
            return Err(ManifestError::UnknownParamForm {
                function: function.to_string(),
                form: other.to_string(),
            })
        }
    })
}

/// Parse a provider manifest into the same `DeclaredProvider` a hardcoded registry entry produces.
///
/// **Deliberately does not validate the ABI.** `provider_abi::validate` and `ProviderSet::select`
/// still run over the result exactly as they do for a first-party provider — a manifest is a
/// different SOURCE for metadata, never a different standard for it. Keeping the two apart is what
/// lets `first_party()` become manifest-driven without weakening anything.
pub fn parse_provider_manifest(
    text: &str,
    origin: &str,
) -> Result<DeclaredProvider, ManifestError> {
    let root = parse_json(text).map_err(ManifestError::Malformed)?;
    let root = object(&root, "")?;

    let package_name = string(field(root, "name", "")?, "name")?;
    let version = string(field(root, "version", "")?, "version")?;
    let provider = object(field(root, "provider", "")?, "provider")?;

    let abi_version = string(field(provider, "abi", "provider")?, "provider.abi")?;
    // The provider's §2 identity defaults to the package name. Two names exist because a package
    // may ship under a distribution name while declaring a stable ABI identity, and resource
    // identity is structural over that identity (CD-360).
    let identity_name = match provider.get("identity") {
        Some(v) => string(v, "provider.identity")?,
        None => package_name.clone(),
    };
    let crate_name = match provider.get("crate") {
        Some(v) => string(v, "provider.crate")?,
        None => package_name.clone(),
    };
    // CD-363: where the provider's Cargo crate lives, RELATIVE to a root the caller supplies —
    // the compiler's own root for a built-in, the manifest's directory for an external one. One
    // rule, two roots. Defaulting to the crate name covers the common external layout, where the
    // crate sits beside its manifest.
    let crate_path = match provider.get("crate_path") {
        Some(v) => string(v, "provider.crate_path")?,
        None => crate_name.clone(),
    };
    check_contained(&crate_path)?;

    let capabilities = string_array(
        field(provider, "capabilities", "provider")?,
        "provider.capabilities",
    )?;
    let target_triples = string_array(field(provider, "targets", "provider")?, "provider.targets")?;

    let mut resource_types = Vec::new();
    if let Some(resources) = provider.get("resources") {
        for (i, entry) in array(resources, "provider.resources")?.iter().enumerate() {
            let path = format!("provider.resources[{i}]");
            let map = object(entry, &path)?;
            resource_types.push(string(field(map, "name", &path)?, &format!("{path}.name"))?);
        }
    }

    // CD-360: resources this provider may consume but does not own.
    let mut foreign_resources = Vec::new();
    if let Some(consumes) = provider.get("consumes") {
        for (i, entry) in array(consumes, "provider.consumes")?.iter().enumerate() {
            let path = format!("provider.consumes[{i}]");
            let map = object(entry, &path)?;
            foreign_resources.push(ForeignResource {
                provider: string(field(map, "provider", &path)?, &format!("{path}.provider"))?,
                resource: string(field(map, "resource", &path)?, &format!("{path}.resource"))?,
            });
        }
    }

    let mut functions = Vec::new();
    for (i, entry) in array(
        field(provider, "functions", "provider")?,
        "provider.functions",
    )?
    .iter()
    .enumerate()
    {
        let path = format!("provider.functions[{i}]");
        let map = object(entry, &path)?;
        let symbol = string(field(map, "symbol", &path)?, &format!("{path}.symbol"))?;
        let capability = string(
            field(map, "capability", &path)?,
            &format!("{path}.capability"),
        )?;
        let may_block = match map.get("may_block") {
            Some(JsonValue::Bool(b)) => *b,
            None => false,
            Some(_) => {
                return Err(ManifestError::WrongType {
                    path: format!("{path}.may_block"),
                    expected: "a boolean".to_string(),
                })
            }
        };
        // `close_for` names the resource this function releases. Absent means "not a close", which
        // is the common case and therefore the default.
        let is_close_for = match map.get("close_for") {
            Some(v) => Some(string(v, &format!("{path}.close_for"))?),
            None => None,
        };
        let mut params = Vec::new();
        for (j, p) in array(field(map, "params", &path)?, &format!("{path}.params"))?
            .iter()
            .enumerate()
        {
            params.push(param(p, &symbol, &format!("{path}.params[{j}]"))?);
        }
        functions.push(FunctionDecl {
            name: symbol,
            capability,
            params,
            is_close_for,
            may_block,
        });
    }

    // The package's recoverable status vocabulary. Empty is MEANINGFUL — it says every nonzero
    // status from this provider is a contract violation — so an absent key is not an error.
    let mut status_binding = StatusBinding::new();
    if let Some(status) = provider.get("status") {
        for (code, name) in object(status, "provider.status")? {
            let parsed = code
                .parse::<u32>()
                .map_err(|_| ManifestError::InvalidStatusCode { code: code.clone() })?;
            status_binding.declare(parsed, string(name, &format!("provider.status.{code}"))?);
        }
    }

    Ok(DeclaredProvider {
        metadata: ProviderMetadata {
            identity: ProviderIdentity {
                name: identity_name,
                semver: parse_semver(&version),
                abi_version,
            },
            target_triples,
            capabilities,
            resource_types,
            foreign_resources,
            functions,
        },
        crate_name,
        crate_path,
        status_binding,
        origin: origin.to_string(),
    })
}

/// The crate path a manifest declared, resolved against `root`.
///
/// **A LOCATION, never part of MIR.** `crate_location`'s original doc had this right and it
/// survives the migration: a crate's path is a property of the machine doing the build, its name a
/// property of the program. Keeping them apart is what lets a verified MIR artefact stay
/// relocation-stable while still naming the providers it needs.
pub fn resolve_crate_path(text: &str, root: &Path) -> Result<PathBuf, ManifestError> {
    let parsed = parse_json(text).map_err(ManifestError::Malformed)?;
    let root_obj = object(&parsed, "")?;
    let provider = object(field(root_obj, "provider", "")?, "provider")?;
    let declared = match provider.get("crate_path") {
        Some(v) => string(v, "provider.crate_path")?,
        None => match provider.get("crate") {
            Some(v) => string(v, "provider.crate")?,
            None => string(field(root_obj, "name", "")?, "name")?,
        },
    };
    check_contained(&declared)?;
    Ok(root.join(declared))
}

/// Refuse a `crate_path` that leaves the root it is resolved against.
///
/// Checked on the STRING rather than on the joined path: normalising first would let
/// `provider/../../elsewhere` cancel out into something that looks contained, and a symlink could
/// defeat a post-hoc canonicalisation anyway. Refusing the components outright is the answer that
/// does not depend on the filesystem's cooperation.
fn check_contained(path: &str) -> Result<(), ManifestError> {
    let escapes = Path::new(path).is_absolute()
        || path.starts_with('/')
        || path.starts_with('\\')
        // A Windows drive prefix (`C:\...`) is absolute even where the host is not Windows, and
        // the manifest may have been written on another platform.
        || path.chars().nth(1) == Some(':')
        || Path::new(path)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir));
    if escapes {
        return Err(ManifestError::CratePathEscapesRoot {
            path: path.to_string(),
        });
    }
    Ok(())
}

/// A lenient `major.minor.patch`. A malformed version is not an error here: `ProviderSet::select`
/// checks ABI compatibility through `abi_version`, which is the actual compatibility boundary, and
/// refusing a manifest over a cosmetic version string would be a worse diagnostic than accepting it
/// and failing on the thing that matters.
fn parse_semver(text: &str) -> (u32, u32, u32) {
    let mut parts = text.split('.').map(|p| p.parse::<u32>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

// ============================================================ external discovery ==

use std::path::{Path, PathBuf};

/// How much authority a provider was granted, and by whom.
///
/// P0.2: third-party providers execute native code in the user's build and process. They are **not**
/// ordinary STARK packages, and the manifest must not let them pretend to be. Trust is made
/// EXPLICIT rather than enforced — no sandboxing is attempted, because a partial isolation story
/// invites misplaced confidence, whereas a visible tier is honest and achievable now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderTrust {
    /// Ships with the compiler and is versioned with it.
    FirstParty,
    /// Declared by the APPLICATION with an exact version and checksum.
    ApprovedThirdParty,
    /// Path-based, development only. Never admitted to a release build.
    UntrustedLocal,
}

/// An application's request for one external provider.
///
/// Parsed from the application manifest's `providers` map:
///
/// ```json
/// "providers": {
///   "stark-postgres-native": {
///     "path": "../stark-postgres-native",
///     "version": "0.1.0",
///     "sha256": "9f86d0…"
///   }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequirement {
    pub name: String,
    pub path: PathBuf,
    pub version: String,
    /// SHA-256 of the provider's manifest file. Absent means UNTRUSTED-LOCAL: usable while
    /// developing, refused in a release build.
    pub sha256: Option<String>,
}

/// Why an external provider was not admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryError {
    /// The application asked for a provider whose manifest is not where it said.
    ManifestNotFound { provider: String, path: PathBuf },
    /// The manifest was found but could not be read as a provider manifest.
    Unreadable {
        provider: String,
        path: PathBuf,
        error: ManifestError,
    },
    /// The manifest's declared version is not the version the application pinned. Reproducibility
    /// requires the two to agree exactly — "close enough" is how a build stops being repeatable.
    VersionMismatch {
        provider: String,
        requested: String,
        found: String,
    },
    /// The manifest's SHA-256 is not the one the application pinned. The provider on disk is NOT
    /// the provider that was approved.
    ChecksumMismatch {
        provider: String,
        expected: String,
        found: String,
    },
    /// A release build asked for a provider with no checksum. Development-only trust does not
    /// survive into a release artefact.
    UnpinnedInReleaseBuild { provider: String },
    /// External providers are disabled and the application declared one anyway. Off by default is
    /// the point: enabling native third-party code must be a deliberate act.
    ExternalProvidersDisabled { provider: String },
    /// A DEPENDENCY declared providers. Only the application may activate one — a library must not
    /// be able to pull native code into a program that never asked for it.
    TransitiveActivation {
        dependency: String,
        provider: String,
    },
}

impl std::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiscoveryError::ManifestNotFound { provider, path } => write!(
                f,
                "provider `{provider}`: no manifest at {}",
                path.display()
            ),
            DiscoveryError::Unreadable {
                provider,
                path,
                error,
            } => write!(f, "provider `{provider}` at {}: {error}", path.display()),
            DiscoveryError::VersionMismatch {
                provider,
                requested,
                found,
            } => write!(
                f,
                "provider `{provider}`: pinned {requested}, found {found}"
            ),
            DiscoveryError::ChecksumMismatch {
                provider,
                expected,
                found,
            } => write!(
                f,
                "provider `{provider}`: checksum mismatch — approved {expected}, found {found}. \
                 The provider on disk is not the one that was approved."
            ),
            DiscoveryError::UnpinnedInReleaseBuild { provider } => write!(
                f,
                "provider `{provider}` has no `sha256` and cannot enter a release build; \
                 path-based providers are development-only"
            ),
            DiscoveryError::ExternalProvidersDisabled { provider } => write!(
                f,
                "provider `{provider}` is declared but external providers are disabled; \
                 enable it explicitly in the application manifest"
            ),
            DiscoveryError::TransitiveActivation {
                dependency,
                provider,
            } => write!(
                f,
                "dependency `{dependency}` declares provider `{provider}`, but only the \
                 APPLICATION may activate a provider — a library must not pull native code into a \
                 program that did not ask for it"
            ),
        }
    }
}

/// Parse an application manifest's `providers` map. Absent means none, which is the common case.
pub fn parse_provider_requirements(
    manifest_text: &str,
    manifest_dir: &Path,
) -> Result<Vec<ProviderRequirement>, ManifestError> {
    let root = parse_json(manifest_text).map_err(ManifestError::Malformed)?;
    let root = object(&root, "")?;
    let Some(providers) = root.get("providers") else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for (name, entry) in object(providers, "providers")? {
        let path = format!("providers.{name}");
        let map = object(entry, &path)?;
        out.push(ProviderRequirement {
            name: name.clone(),
            path: manifest_dir.join(string(field(map, "path", &path)?, &format!("{path}.path"))?),
            version: string(field(map, "version", &path)?, &format!("{path}.version"))?,
            sha256: match map.get("sha256") {
                Some(v) => Some(string(v, &format!("{path}.sha256"))?),
                None => None,
            },
        });
    }
    // Sorted so the selected provider set — and therefore the generated workspace — cannot depend
    // on JSON key order, the same reason `capabilities` is sorted at parse time.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Policy for admitting external providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExternalProviderPolicy {
    /// External providers are **disabled by default** — `Default` gives `false` deliberately, not
    /// incidentally. Enabling native third-party code is a
    /// deliberate act, not something a dependency graph can arrange.
    pub enabled: bool,
    /// Defaults to `false`: a build is a development build unless it says otherwise, so the
    /// stricter release rule is never applied by accident to a dev loop.
    /// A release build refuses an unpinned provider: development-only trust does not survive into
    /// a release artefact.
    pub release_build: bool,
}

/// A provider admitted for a build, with the authority under which it was admitted.
#[derive(Debug, Clone)]
pub struct AdmittedProvider {
    pub provider: DeclaredProvider,
    pub trust: ProviderTrust,
    /// The directory containing the provider's manifest — this replaces `crate_location`'s
    /// hardcoded match. A LOCATION, never part of MIR: a crate's path is a property of the machine
    /// doing the build, while its name is a property of the program, and keeping them apart is what
    /// lets a verified MIR artefact stay relocation-stable.
    pub root: PathBuf,
    /// Recorded for build metadata and reproducibility evidence.
    pub sha256: Option<String>,
}

/// Admit the external providers an application declared, under `policy`.
///
/// Built-in providers are not handled here — they are always admitted, need no pinning, and are
/// added by the caller. This function exists for the part that carries risk.
pub fn discover_external_providers(
    requirements: &[ProviderRequirement],
    policy: ExternalProviderPolicy,
) -> Result<Vec<AdmittedProvider>, Vec<DiscoveryError>> {
    let mut admitted = Vec::new();
    let mut errors = Vec::new();

    for requirement in requirements {
        if !policy.enabled {
            errors.push(DiscoveryError::ExternalProvidersDisabled {
                provider: requirement.name.clone(),
            });
            continue;
        }
        if policy.release_build && requirement.sha256.is_none() {
            errors.push(DiscoveryError::UnpinnedInReleaseBuild {
                provider: requirement.name.clone(),
            });
            continue;
        }

        let manifest_path = requirement.path.join("starkpkg.json");
        let Ok(text) = std::fs::read_to_string(&manifest_path) else {
            errors.push(DiscoveryError::ManifestNotFound {
                provider: requirement.name.clone(),
                path: manifest_path,
            });
            continue;
        };

        // The checksum covers the MANIFEST — the ABI surface, capabilities and symbol names the
        // build is approving. It is not a substitute for verifying the crate's source; it pins what
        // the compiler read, which is what determines what gets linked and what it may do.
        if let Some(expected) = &requirement.sha256 {
            let found = sha256_hex(text.as_bytes());
            if &found != expected {
                errors.push(DiscoveryError::ChecksumMismatch {
                    provider: requirement.name.clone(),
                    expected: expected.clone(),
                    found,
                });
                continue;
            }
        }

        let declared = match parse_provider_manifest(&text, &manifest_path.display().to_string()) {
            Ok(d) => d,
            Err(error) => {
                errors.push(DiscoveryError::Unreadable {
                    provider: requirement.name.clone(),
                    path: manifest_path,
                    error,
                });
                continue;
            }
        };

        let (a, b, c) = declared.metadata.identity.semver;
        let found_version = format!("{a}.{b}.{c}");
        if found_version != requirement.version {
            errors.push(DiscoveryError::VersionMismatch {
                provider: requirement.name.clone(),
                requested: requirement.version.clone(),
                found: found_version,
            });
            continue;
        }

        admitted.push(AdmittedProvider {
            provider: declared,
            trust: match requirement.sha256 {
                Some(_) => ProviderTrust::ApprovedThirdParty,
                None => ProviderTrust::UntrustedLocal,
            },
            root: requirement.path.clone(),
            sha256: requirement.sha256.clone(),
        });
    }

    if errors.is_empty() {
        Ok(admitted)
    } else {
        Err(errors)
    }
}

/// Refuse a provider declared by anything other than the application.
///
/// **No transitive activation.** A library must not be able to pull native code into a program that
/// never asked for it — which is the difference between a dependency graph and an attack surface.
pub fn reject_transitive_activation(
    dependency_name: &str,
    dependency_manifest: &str,
) -> Result<(), Vec<DiscoveryError>> {
    let Ok(root) = parse_json(dependency_manifest) else {
        return Ok(()); // a malformed dependency manifest is the package loader's error to report
    };
    let Some(root) = root.as_object() else {
        return Ok(());
    };
    let Some(providers) = root.get("providers").and_then(|p| p.as_object()) else {
        return Ok(());
    };
    let mut errors: Vec<DiscoveryError> = providers
        .keys()
        .map(|provider| DiscoveryError::TransitiveActivation {
            dependency: dependency_name.to_string(),
            provider: provider.clone(),
        })
        .collect();
    errors.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// SHA-256 of raw bytes, hex-encoded.
///
/// The checksum covers the MANIFEST, not the crate's source: it pins the ABI surface, capabilities
/// and symbol names the build approved, which is what determines what gets linked and what it may
/// do. Verifying the implementation is a supply-chain question this packet deliberately does not
/// claim to answer — see WP-EXTERNAL-PROVIDERS' non-goals.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
