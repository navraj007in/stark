//! WP-C5.1c — compile-time Native Provider ABI v0.1 metadata validator
//! (`STARKLANG/docs/compiler/native-provider-abi-v0.1.md`, §17 -- PROPOSED, owner CE4 review
//! pending). Validates a provider's *declared metadata* before the compiler ever trusts it;
//! this is compile-time/build-time validation, not a runtime check against an executing
//! provider -- no provider actually executes in the C5 MVP (§10.2's implementation boundary).

/// §2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderIdentity {
    pub name: String,
    pub semver: (u32, u32, u32),
    pub abi_version: String,
}

/// §6. `AbiType` (below) is deliberately closed over no callback/function-pointer variant and
/// no internal-generated-Rust-aggregate variant, so §10 and §15's callback prohibition are
/// enforced by this type's shape, not by a runtime check here (§17: "defense in depth, not the
/// primary enforcement" -- there is nothing further to check because there is no way to
/// construct the excluded case).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub name: String,
    pub capability: String,
    pub params: Vec<AbiType>,
    pub returns: AbiType,
    /// `Some(resource_type)` if this is THE close function for that resource type (§13). At
    /// most one function per resource type may set this -- checked by [`validate`].
    pub is_close_for: Option<String>,
    pub may_block: bool,
}

/// §6/§10: the closed cross-boundary type vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    F32,
    F64,
    ResourceHandle,
    BorrowedBuffer,
    BorrowedBufferMut,
    ProviderStatus,
}

/// The full declaration a provider must present. Combines §2 (identity), §4 (target triples),
/// §5 (capabilities), §6 (functions), and the resource-type list §13's close-function rule
/// checks against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderMetadata {
    pub identity: ProviderIdentity,
    pub target_triples: Vec<String>,
    pub capabilities: Vec<String>,
    pub resource_types: Vec<String>,
    pub functions: Vec<FunctionDecl>,
}

/// One violation of a §2-§16 rule this validator can check mechanically. Every variant names
/// the document section it enforces so a validation failure is traceable back to the contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiViolation {
    /// §2/§16 point 1.
    UnsupportedAbiVersion { found: String },
    /// §4.
    NoTargetTriples,
    /// §5.
    NoCapabilities,
    /// §5: a function claims a capability the provider never declared.
    FunctionCapabilityUndeclared {
        function: String,
        capability: String,
    },
    /// §5: a declared capability has no function implementing it.
    CapabilityUnreachable { capability: String },
    /// §13: a declared resource type has no close function.
    ResourceTypeMissingClose { resource_type: String },
    /// §13: a resource type has more than one function claiming to close it.
    ResourceTypeMultipleClose {
        resource_type: String,
        functions: Vec<String>,
    },
    /// §13: a function's `is_close_for` names a resource type the provider never declared.
    CloseForUndeclaredResourceType {
        function: String,
        resource_type: String,
    },
}

/// This ABI's version (§2). A provider declaring anything else is rejected outright.
pub const ABI_VERSION: &str = "0.1";

/// Checks every mechanically checkable rule in
/// `STARKLANG/docs/compiler/native-provider-abi-v0.1.md` §2-§16. Returns every violation found,
/// not just the first, matching the MIR verifier's own "return every violation" convention
/// (`starkc/src/mir/verify.rs`) rather than failing fast on the first mismatch.
pub fn validate(metadata: &ProviderMetadata) -> Result<(), Vec<AbiViolation>> {
    let mut violations = Vec::new();

    if metadata.identity.abi_version != ABI_VERSION {
        violations.push(AbiViolation::UnsupportedAbiVersion {
            found: metadata.identity.abi_version.clone(),
        });
    }
    if metadata.target_triples.is_empty() {
        violations.push(AbiViolation::NoTargetTriples);
    }
    if metadata.capabilities.is_empty() {
        violations.push(AbiViolation::NoCapabilities);
    }

    for f in &metadata.functions {
        if !metadata.capabilities.iter().any(|c| c == &f.capability) {
            violations.push(AbiViolation::FunctionCapabilityUndeclared {
                function: f.name.clone(),
                capability: f.capability.clone(),
            });
        }
    }
    for cap in &metadata.capabilities {
        if !metadata.functions.iter().any(|f| &f.capability == cap) {
            violations.push(AbiViolation::CapabilityUnreachable {
                capability: cap.clone(),
            });
        }
    }

    for rt in &metadata.resource_types {
        let closers: Vec<&str> = metadata
            .functions
            .iter()
            .filter(|f| f.is_close_for.as_deref() == Some(rt.as_str()))
            .map(|f| f.name.as_str())
            .collect();
        match closers.len() {
            0 => violations.push(AbiViolation::ResourceTypeMissingClose {
                resource_type: rt.clone(),
            }),
            1 => {}
            _ => violations.push(AbiViolation::ResourceTypeMultipleClose {
                resource_type: rt.clone(),
                functions: closers.into_iter().map(String::from).collect(),
            }),
        }
    }
    for f in &metadata.functions {
        if let Some(rt) = &f.is_close_for {
            if !metadata.resource_types.iter().any(|d| d == rt) {
                violations.push(AbiViolation::CloseForUndeclaredResourceType {
                    function: f.name.clone(),
                    resource_type: rt.clone(),
                });
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

#[cfg(test)]
mod fixtures {
    //! §17's mock provider metadata. `example-kv` is a **fictional, illustrative** key-value
    //! store -- not a committed STARK capability and not tied to any real stdlib type. It
    //! exists only to exercise the validator against a resource handle, a paired open/close
    //! pair, and a `BorrowedBuffer`-taking function, per the document's own description.
    use super::*;

    pub fn valid_example_kv() -> ProviderMetadata {
        ProviderMetadata {
            identity: ProviderIdentity {
                name: "example-kv".to_string(),
                semver: (0, 1, 0),
                abi_version: ABI_VERSION.to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
            ],
            capabilities: vec!["kv".to_string()],
            resource_types: vec!["KvStore".to_string()],
            functions: vec![
                FunctionDecl {
                    name: "kv_open".to_string(),
                    capability: "kv".to_string(),
                    params: vec![AbiType::BorrowedBuffer],
                    returns: AbiType::ResourceHandle,
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "kv_get".to_string(),
                    capability: "kv".to_string(),
                    params: vec![AbiType::ResourceHandle, AbiType::BorrowedBuffer],
                    returns: AbiType::ProviderStatus,
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "kv_set".to_string(),
                    capability: "kv".to_string(),
                    params: vec![
                        AbiType::ResourceHandle,
                        AbiType::BorrowedBuffer,
                        AbiType::BorrowedBuffer,
                    ],
                    returns: AbiType::ProviderStatus,
                    is_close_for: None,
                    may_block: true,
                },
                FunctionDecl {
                    name: "kv_close".to_string(),
                    capability: "kv".to_string(),
                    params: vec![AbiType::ResourceHandle],
                    returns: AbiType::ProviderStatus,
                    is_close_for: Some("KvStore".to_string()),
                    may_block: false,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::valid_example_kv;
    use super::*;

    #[test]
    fn valid_mock_provider_passes() {
        assert_eq!(validate(&valid_example_kv()), Ok(()));
    }

    #[test]
    fn wrong_abi_version_is_rejected() {
        let mut m = valid_example_kv();
        m.identity.abi_version = "0.2".to_string();
        let violations = validate(&m).unwrap_err();
        assert!(violations.contains(&AbiViolation::UnsupportedAbiVersion {
            found: "0.2".to_string()
        }));
    }

    #[test]
    fn resource_type_missing_close_is_rejected() {
        let mut m = valid_example_kv();
        m.functions.retain(|f| f.name != "kv_close");
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.contains(&AbiViolation::ResourceTypeMissingClose {
                resource_type: "KvStore".to_string()
            })
        );
    }

    #[test]
    fn resource_type_with_two_close_functions_is_rejected() {
        let mut m = valid_example_kv();
        let mut second_close = m
            .functions
            .iter()
            .find(|f| f.name == "kv_close")
            .unwrap()
            .clone();
        second_close.name = "kv_close_again".to_string();
        m.functions.push(second_close);
        let violations = validate(&m).unwrap_err();
        assert!(violations
            .iter()
            .any(|v| matches!(v, AbiViolation::ResourceTypeMultipleClose { resource_type, .. } if resource_type == "KvStore")));
    }

    #[test]
    fn capability_with_no_function_is_rejected() {
        let mut m = valid_example_kv();
        m.capabilities.push("unreachable-capability".to_string());
        let violations = validate(&m).unwrap_err();
        assert!(violations.contains(&AbiViolation::CapabilityUnreachable {
            capability: "unreachable-capability".to_string()
        }));
    }

    #[test]
    fn function_with_undeclared_capability_is_rejected() {
        let mut m = valid_example_kv();
        m.functions.push(FunctionDecl {
            name: "rogue_fn".to_string(),
            capability: "undeclared".to_string(),
            params: vec![],
            returns: AbiType::ProviderStatus,
            is_close_for: None,
            may_block: false,
        });
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.contains(&AbiViolation::FunctionCapabilityUndeclared {
                function: "rogue_fn".to_string(),
                capability: "undeclared".to_string()
            })
        );
    }

    #[test]
    fn empty_target_triples_and_capabilities_are_rejected() {
        let mut m = valid_example_kv();
        m.target_triples.clear();
        m.capabilities.clear();
        m.functions.clear();
        m.resource_types.clear();
        let violations = validate(&m).unwrap_err();
        assert!(violations.contains(&AbiViolation::NoTargetTriples));
        assert!(violations.contains(&AbiViolation::NoCapabilities));
    }
}
