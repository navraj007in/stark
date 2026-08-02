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
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestError::Malformed(why) => write!(f, "provider manifest is not valid JSON: {why}"),
            ManifestError::MissingField { path } => write!(f, "missing required field `{path}`"),
            ManifestError::WrongType { path, expected } => {
                write!(f, "field `{path}` must be {expected}")
            }
            ManifestError::UnknownParamForm { function, form } => write!(
                f,
                "function `{function}`: unknown parameter form `{form}`"
            ),
            ManifestError::UnknownScalarType { function, scalar } => write!(
                f,
                "function `{function}`: unknown scalar type `{scalar}`"
            ),
            ManifestError::HandleWithoutResource { function, form } => write!(
                f,
                "function `{function}`: `{form}` must name the `resource` it carries"
            ),
            ManifestError::InvalidStatusCode { code } => {
                write!(f, "status code `{code}` must be a non-negative integer")
            }
        }
    }
}

fn object<'a>(value: &'a JsonValue, path: &str) -> Result<&'a std::collections::HashMap<String, JsonValue>, ManifestError> {
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

fn param(
    value: &JsonValue,
    function: &str,
    path: &str,
) -> Result<AbiParam, ManifestError> {
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
        let name = string(
            field(map, "type", path)?,
            &format!("{path}.type"),
        )?;
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
pub fn parse_provider_manifest(text: &str, origin: &str) -> Result<DeclaredProvider, ManifestError> {
    let root = parse_json(text).map_err(ManifestError::Malformed)?;
    let root = object(&root, "")?;

    let package_name = string(field(root, "name", "")?, "name")?;
    let version = string(field(root, "version", "")?, "version")?;
    let provider = object(field(root, "provider", "")?, "provider")?;

    let abi_version = string(
        field(provider, "abi", "provider")?,
        "provider.abi",
    )?;
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

    let capabilities = string_array(
        field(provider, "capabilities", "provider")?,
        "provider.capabilities",
    )?;
    let target_triples = string_array(
        field(provider, "targets", "provider")?,
        "provider.targets",
    )?;

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
                provider: string(
                    field(map, "provider", &path)?,
                    &format!("{path}.provider"),
                )?,
                resource: string(
                    field(map, "resource", &path)?,
                    &format!("{path}.resource"),
                )?,
            });
        }
    }

    let mut functions = Vec::new();
    for (i, entry) in array(field(provider, "functions", "provider")?, "provider.functions")?
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
        status_binding,
        origin: origin.to_string(),
    })
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
