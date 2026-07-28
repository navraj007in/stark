//! WP-C5.1c — compile-time Native Provider ABI v0.1 metadata validator
//! (`STARKLANG/docs/compiler/native-provider-abi-v0.1.md` §17, as amended by **CE4 Amendment 1**
//! — approved 2026-07-21, CD-054). Validates a provider's *declared metadata* before the compiler
//! ever trusts it; this is compile-time/build-time validation, not a runtime check against an
//! executing provider -- no provider actually executes in the C5 MVP (§10.2's implementation
//! boundary).

/// §2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderIdentity {
    pub name: String,
    pub semver: (u32, u32, u32),
    pub abi_version: String,
}

/// §6 (amended). Note what is NOT here: a `returns` field. **The physical ABI return is always
/// `ProviderStatus`** (§11), so it is a property of the ABI rather than a per-function choice --
/// unrepresentable otherwise, which is what makes §11 enforced by construction instead of by a
/// check nobody wrote. Values a function produces travel through [`AbiParam`]'s explicit output
/// forms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDecl {
    pub name: String,
    pub capability: String,
    pub params: Vec<AbiParam>,
    /// `Some(resource_type)` if this is THE close function for that resource type (§13). At
    /// most one function per resource type may set this, and a close function's shape is
    /// tightly constrained -- both checked by [`validate`].
    pub is_close_for: Option<String>,
    pub may_block: bool,
}

/// §6/§10 (amended): the scalar widths that cross the boundary by value.
///
/// Split out of the old flat `AbiType`: these are the only members of that enum that were ever
/// *types*. The rest (`ResourceHandle`, the buffers, `ProviderStatus`) were parameter FORMS
/// wearing a type's clothing, which is exactly the conflation [`AbiParam`] removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarTy {
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
}

/// §6/§10 (amended): the closed parameter vocabulary — one variant per admitted form, with no
/// product of independent axes, so **every representable parameter is meaningful and every
/// meaningful parameter is representable**.
///
/// The rejected alternative was `Direction × AbiType`: of its fifteen combinations six were
/// meaningful, three were one case spelled three times, and the distinction that actually matters
/// -- borrowed vs. consumed handle -- was the one it could not express, because ownership is not
/// a direction.
///
/// Three properties hold structurally, with no validator rule required:
/// - a buffer can never be "owned" or "consumed" -- no variant says so;
/// - a handle's ownership is always stated (borrowed, consumed, or newly-owned-out);
/// - `ProviderStatus` is not a parameter form and not a type, so it cannot be written as a
///   parameter at all.
///
/// §15's callback prohibition and §10's no-internal-aggregate rule are likewise enforced by this
/// enum's closure rather than by a runtime check: there is no way to construct the excluded case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiParam {
    /// Copied scalar input. Physical C parameter: `T`.
    ScalarIn(ScalarTy),

    /// Scalar output slot: caller-allocated, provider writes, caller reads **only on success**
    /// (§4.7). Physical C parameter: `*mut T`.
    ScalarOut(ScalarTy),
    /// Scalar in/out slot: caller-initialised and caller-owned across the call; the provider
    /// reads and may overwrite. Physical C parameter: `*mut T` -- physically identical to
    /// `ScalarOut`, deliberately distinct here because the CONTRACT differs and the C signature
    /// cannot carry that.
    ScalarInOut(ScalarTy),

    /// Immutable borrowed buffer. A call-duration view (§9), never an ownership transfer.
    BufferIn,
    /// Mutable borrowed buffer. Caller-initialised and caller-owned across the call; the caller
    /// reads it afterward, which is the entire point of the form.
    BufferInOut,

    /// Borrowed typed resource handle -- **the default for ordinary resource operations**. The
    /// caller retains ownership; the provider must not retain the handle past return.
    HandleBorrowed { resource_type: String },
    /// Consumed typed resource handle: ownership transfers in **at call entry**, regardless of
    /// what the status reports (§4.6). Close functions, and operations that explicitly end a
    /// resource's life -- rare by design.
    HandleConsumed { resource_type: String },
    /// Newly-owned typed resource-handle output slot: on success the provider writes a handle
    /// the CALLER now solely owns and must eventually close exactly once (§13). Uninitialised
    /// before the call and valid only on success (§4.7). Physical C parameter:
    /// `*mut RawResourceHandle`.
    HandleOut { resource_type: String },
}

impl AbiParam {
    /// The declared resource type a handle-carrying parameter names, if any. Used by the
    /// validator to check every handle against `ProviderMetadata.resource_types`.
    pub fn handle_resource_type(&self) -> Option<&str> {
        match self {
            AbiParam::HandleBorrowed { resource_type }
            | AbiParam::HandleConsumed { resource_type }
            | AbiParam::HandleOut { resource_type } => Some(resource_type),
            _ => None,
        }
    }

    /// Whether this parameter yields a value back to the caller. A close function may have none
    /// (§4.4) -- its only result is the `ProviderStatus` every function returns.
    pub fn is_output(&self) -> bool {
        matches!(
            self,
            AbiParam::ScalarOut(_)
                | AbiParam::ScalarInOut(_)
                | AbiParam::BufferInOut
                | AbiParam::HandleOut { .. }
        )
    }
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
    /// §6/§13 (amendment §3.3): a handle parameter names a resource type the provider never
    /// declared, so §13's wrong-resource-type rule could not be checked against it.
    HandleResourceTypeUndeclared {
        function: String,
        resource_type: String,
    },
    /// §13 (amendment §4.4): a close function does not have the mandatory shape.
    CloseFunctionShape {
        function: String,
        resource_type: String,
        problem: CloseShapeProblem,
    },
}

/// Which clause of the close-function rule a declaration broke. Named per clause rather than
/// collapsed into one string, so a violation says what to fix.
///
/// The rule (§4.4, owner ruling CD-054): a close function takes **exactly one parameter**, a
/// `HandleConsumed` of the resource type it closes, and nothing else. The reason is
/// architectural: **MIR's `Drop(place)` terminator supplies only the resource being dropped** --
/// there is no argument list at a drop site, so a close function with a second parameter is one
/// the generated code cannot call. Anything needing arguments (a flush option, a completion
/// mode) must be a separate provider function invoked before Drop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseShapeProblem {
    /// Not exactly one parameter.
    NotExactlyOneParameter { found: usize },
    /// The single parameter is not a `HandleConsumed` -- e.g. `HandleBorrowed`, which would
    /// close a resource the caller still believes it owns.
    ParameterIsNotAConsumedHandle,
    /// The consumed handle's resource type is not the one `is_close_for` names.
    ConsumedHandleWrongResourceType { found: String },
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

    // Amendment §3.3: every handle-carrying parameter names a DECLARED resource type. Without
    // this, §13's "closing a handle from a different resource type is a contract violation" rule
    // is unenforceable at compile time -- the runtime struct carries a `resource_type` value, but
    // a value is not something a declaration can be checked against.
    for f in &metadata.functions {
        for p in &f.params {
            if let Some(rt) = p.handle_resource_type() {
                if !metadata.resource_types.iter().any(|d| d == rt) {
                    violations.push(AbiViolation::HandleResourceTypeUndeclared {
                        function: f.name.clone(),
                        resource_type: rt.to_string(),
                    });
                }
            }
        }
    }

    // Amendment §4.4: the close-function shape. Exactly one parameter, a `HandleConsumed` of the
    // declared resource type, nothing else.
    for f in &metadata.functions {
        let Some(rt) = &f.is_close_for else { continue };
        let problem = match f.params.as_slice() {
            [AbiParam::HandleConsumed { resource_type }] => {
                if resource_type == rt {
                    None
                } else {
                    Some(CloseShapeProblem::ConsumedHandleWrongResourceType {
                        found: resource_type.clone(),
                    })
                }
            }
            [_] => Some(CloseShapeProblem::ParameterIsNotAConsumedHandle),
            other => Some(CloseShapeProblem::NotExactlyOneParameter { found: other.len() }),
        };
        if let Some(problem) = problem {
            violations.push(AbiViolation::CloseFunctionShape {
                function: f.name.clone(),
                resource_type: rt.clone(),
                problem,
            });
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
    //! pair, and buffer-taking functions, per the document's own description.
    //!
    //! Rewritten under CE4 Amendment 1: the pre-amendment fixture VIOLATED the contract it was
    //! the positive example for -- `kv_open` returned a `ResourceHandle` directly (§11 forbids
    //! any direct return), `kv_get` had nowhere to put the value it retrieved, and `kv_get`/
    //! `kv_set` CONSUMED the store they operated on, so a second call was a use-after-transfer.
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
                // Opens a store: the new handle comes back through an explicit output slot,
                // never as a direct return.
                FunctionDecl {
                    name: "kv_open".to_string(),
                    capability: "kv".to_string(),
                    params: vec![
                        AbiParam::BufferIn,
                        AbiParam::HandleOut {
                            resource_type: "KvStore".to_string(),
                        },
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                // Reads a value: BORROWS the store (so it survives the call), takes the key as
                // an immutable view, fills a caller-allocated buffer, and reports how many bytes
                // it wrote through a scalar output.
                FunctionDecl {
                    name: "kv_get".to_string(),
                    capability: "kv".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: "KvStore".to_string(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::BufferInOut,
                        AbiParam::ScalarOut(ScalarTy::U64),
                    ],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "kv_set".to_string(),
                    capability: "kv".to_string(),
                    params: vec![
                        AbiParam::HandleBorrowed {
                            resource_type: "KvStore".to_string(),
                        },
                        AbiParam::BufferIn,
                        AbiParam::BufferIn,
                    ],
                    is_close_for: None,
                    may_block: true,
                },
                // The close function: exactly one parameter, the consumed handle. Nothing else
                // is callable from a MIR `Drop` terminator, which supplies only the resource.
                FunctionDecl {
                    name: "kv_close".to_string(),
                    capability: "kv".to_string(),
                    params: vec![AbiParam::HandleConsumed {
                        resource_type: "KvStore".to_string(),
                    }],
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

    /// Amendment §3.3: a handle naming a resource type the provider never declared.
    #[test]
    fn handle_naming_an_undeclared_resource_type_is_rejected() {
        let mut m = valid_example_kv();
        m.functions.push(FunctionDecl {
            name: "kv_peek".to_string(),
            capability: "kv".to_string(),
            params: vec![AbiParam::HandleBorrowed {
                resource_type: "NotDeclared".to_string(),
            }],
            is_close_for: None,
            may_block: false,
        });
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.contains(&AbiViolation::HandleResourceTypeUndeclared {
                function: "kv_peek".to_string(),
                resource_type: "NotDeclared".to_string()
            }),
            "{violations:?}"
        );
    }

    /// §4.4, first clause: a close function with a second parameter is one the generated code
    /// cannot call -- a MIR `Drop` terminator has no argument list to supply it from.
    #[test]
    fn close_function_with_an_extra_parameter_is_rejected() {
        let mut m = valid_example_kv();
        let close = m
            .functions
            .iter_mut()
            .find(|f| f.name == "kv_close")
            .unwrap();
        close.params.push(AbiParam::ScalarIn(ScalarTy::Bool));
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.contains(&AbiViolation::CloseFunctionShape {
                function: "kv_close".to_string(),
                resource_type: "KvStore".to_string(),
                problem: CloseShapeProblem::NotExactlyOneParameter { found: 2 }
            }),
            "{violations:?}"
        );
    }

    /// §4.4, and the same rule catches an output: a close that also hands back a value is a
    /// close that is also something else.
    #[test]
    fn close_function_with_an_output_is_rejected() {
        let mut m = valid_example_kv();
        let close = m
            .functions
            .iter_mut()
            .find(|f| f.name == "kv_close")
            .unwrap();
        close.params.push(AbiParam::ScalarOut(ScalarTy::U64));
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.iter().any(|v| matches!(
                v,
                AbiViolation::CloseFunctionShape {
                    problem: CloseShapeProblem::NotExactlyOneParameter { .. },
                    ..
                }
            )),
            "{violations:?}"
        );
    }

    /// §4.4, second clause: closing through a BORROWED handle would close a resource the caller
    /// still believes it owns.
    #[test]
    fn close_function_that_borrows_rather_than_consumes_is_rejected() {
        let mut m = valid_example_kv();
        let close = m
            .functions
            .iter_mut()
            .find(|f| f.name == "kv_close")
            .unwrap();
        close.params = vec![AbiParam::HandleBorrowed {
            resource_type: "KvStore".to_string(),
        }];
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.contains(&AbiViolation::CloseFunctionShape {
                function: "kv_close".to_string(),
                resource_type: "KvStore".to_string(),
                problem: CloseShapeProblem::ParameterIsNotAConsumedHandle
            }),
            "{violations:?}"
        );
    }

    /// §4.4, third clause: consuming the WRONG resource type. Both types are declared, so only
    /// the close-shape check can catch this -- exactly the case the pre-amendment model could
    /// not see, because its handles carried no declared type at all.
    #[test]
    fn close_function_consuming_the_wrong_resource_type_is_rejected() {
        let mut m = valid_example_kv();
        m.resource_types.push("KvCursor".to_string());
        m.functions.push(FunctionDecl {
            name: "cursor_close".to_string(),
            capability: "kv".to_string(),
            params: vec![AbiParam::HandleConsumed {
                resource_type: "KvStore".to_string(),
            }],
            is_close_for: Some("KvCursor".to_string()),
            may_block: false,
        });
        let violations = validate(&m).unwrap_err();
        assert!(
            violations.contains(&AbiViolation::CloseFunctionShape {
                function: "cursor_close".to_string(),
                resource_type: "KvCursor".to_string(),
                problem: CloseShapeProblem::ConsumedHandleWrongResourceType {
                    found: "KvStore".to_string()
                }
            }),
            "{violations:?}"
        );
    }

    /// The positive counterpart to the shape tests: `kv_get`/`kv_set` BORROW the store, so a
    /// program may call them repeatedly. Under the pre-amendment model they consumed it, which
    /// made the document's own example provider unusable after one read.
    #[test]
    fn ordinary_operations_borrow_rather_than_consume_their_resource() {
        let m = valid_example_kv();
        for name in ["kv_get", "kv_set"] {
            let f = m.functions.iter().find(|f| f.name == name).unwrap();
            assert!(
                f.params
                    .iter()
                    .any(|p| matches!(p, AbiParam::HandleBorrowed { .. })),
                "{name} must borrow its store"
            );
            assert!(
                !f.params
                    .iter()
                    .any(|p| matches!(p, AbiParam::HandleConsumed { .. })),
                "{name} must not consume its store"
            );
        }
    }

    /// §11, enforced by construction rather than by a check: there is no way to declare a
    /// function that returns a value directly, because `FunctionDecl` has no `returns` field and
    /// every value-bearing result is an `AbiParam` output form.
    #[test]
    fn every_value_result_is_an_explicit_output_parameter() {
        let m = valid_example_kv();
        let open = m.functions.iter().find(|f| f.name == "kv_open").unwrap();
        assert!(open.params.iter().any(|p| matches!(
            p,
            AbiParam::HandleOut {
                resource_type
            } if resource_type == "KvStore"
        )));
        let get = m.functions.iter().find(|f| f.name == "kv_get").unwrap();
        assert!(get.params.iter().any(|p| p.is_output()));
        // And a close function has none at all (§4.4).
        let close = m.functions.iter().find(|f| f.name == "kv_close").unwrap();
        assert!(!close.params.iter().any(|p| p.is_output()));
    }
}
