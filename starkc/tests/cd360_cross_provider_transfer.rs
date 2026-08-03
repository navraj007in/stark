//! **CD-360: cross-provider `HandleConsumed` is unconditionally consuming.**
//!
//! The frozen ruling:
//!
//! > A cross-provider `HandleConsumed` transfer consumes the source handle regardless of whether
//! > the provider operation succeeds or fails. Failure does not restore the source resource. The
//! > consuming provider is responsible for releasing any underlying native resource when it fails
//! > before producing the destination handle.
//!
//! `HandleConsumed<T>` therefore keeps the meaning it has always had — **ownership leaves the
//! caller unconditionally** — which is why this needs no change to drop elaboration, no
//! branch-dependent move state, and no place that is live on one result arm and dead on another.
//!
//! The STARK-facing shape:
//!
//! ```stark
//! fn connect_tls(stream: TcpStream, config: &TlsConfig) -> Result<TlsStream, TlsError>;
//! ```
//!
//! After the call `stream` is unavailable on **both** `Ok` and `Err`.
//!
//! # What this amendment actually is
//!
//! Not a new transfer mechanism. Probing established that most of the contract already existed:
//! resource identity is structural over `{nominal, provider, resource}`, `HandleOut` writes only on
//! success, close is selected per resource and a closeless resource is refused, and every function
//! returns `ProviderStatus`. What was missing was permission:
//!
//! > A provider function may reference a foreign provider's resource type only in a consuming
//! > handle position, without inheriting or redefining that resource's close operation.
//!
//! # Why the declaration is explicit
//!
//! `foreign_resources` is declared, not inferred. Treating "any handle type I did not declare" as
//! foreign would silently accept `HandleConsumed { resource_type: "tcp_strem" }` and defer the typo
//! to a link failure. Naming the owning provider keeps the check at the same three-part identity
//! the type system uses.
//!
//! # Why the negatives are the load-bearing half
//!
//! This amendment LOOSENS the ABI, and the rules it loosens are what make "exactly one owner,
//! exactly one release" true by construction. Every refusal below is a double-free or a leak that
//! would otherwise ship.

use starkc::provider_abi::{
    validate, AbiParam, AbiViolation, ForeignResource, FunctionDecl, ProviderIdentity,
    ProviderMetadata, ABI_VERSION,
};

fn tls_provider(
    foreign: Vec<ForeignResource>,
    resource_types: Vec<&str>,
    functions: Vec<FunctionDecl>,
) -> ProviderMetadata {
    ProviderMetadata {
        identity: ProviderIdentity {
            name: "stark-tls-native".to_string(),
            semver: (0, 1, 0),
            abi_version: ABI_VERSION.to_string(),
        },
        target_triples: vec!["aarch64-apple-darwin".to_string()],
        capabilities: vec!["tls".to_string()],
        resource_types: resource_types.into_iter().map(String::from).collect(),
        foreign_resources: foreign,
        functions,
    }
}

fn tcp_stream() -> ForeignResource {
    ForeignResource {
        provider: "stark-std-net".to_string(),
        resource: "tcp_stream".to_string(),
    }
}

fn consumed(rt: &str) -> AbiParam {
    AbiParam::HandleConsumed {
        resource_type: rt.to_string(),
    }
}

fn produced(rt: &str) -> AbiParam {
    AbiParam::HandleOut {
        resource_type: rt.to_string(),
    }
}

fn borrowed(rt: &str) -> AbiParam {
    AbiParam::HandleBorrowed {
        resource_type: rt.to_string(),
    }
}

fn func(name: &str, params: Vec<AbiParam>, close_for: Option<&str>) -> FunctionDecl {
    FunctionDecl {
        name: name.to_string(),
        capability: "tls".to_string(),
        params,
        is_close_for: close_for.map(String::from),
        may_block: true,
    }
}

/// The close every provider owes for each resource type it declares.
fn tls_close() -> FunctionDecl {
    func(
        "stark_tls_stream_close",
        vec![consumed("tls_stream")],
        Some("tls_stream"),
    )
}

fn violations(metadata: &ProviderMetadata) -> Vec<AbiViolation> {
    validate(metadata).err().unwrap_or_default()
}

// ------------------------------------------------------------------------- allowed --

/// **The proving declaration.** TLS consumes the net provider's `TcpStream` and produces its own
/// `TlsStream`. This is the shape HC9 needs, and before CD-360 it was refused both ways round.
#[test]
fn a_transfer_consuming_a_foreign_resource_and_producing_an_owned_one_validates() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_client_connect",
                vec![
                    consumed("tcp_stream"),
                    AbiParam::BufferIn,
                    produced("tls_stream"),
                ],
                None,
            ),
            tls_close(),
        ],
    );
    assert_eq!(
        validate(&metadata),
        Ok(()),
        "a consuming cross-provider transfer must validate — this is the whole point of CD-360"
    );
}

/// The provider still borrows and closes its OWN resource normally. The amendment must not have
/// disturbed the ordinary case.
#[test]
fn owned_resources_still_borrow_and_close_normally() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_client_connect",
                vec![consumed("tcp_stream"), produced("tls_stream")],
                None,
            ),
            func(
                "stark_tls_stream_write",
                vec![borrowed("tls_stream"), AbiParam::BufferIn],
                None,
            ),
            tls_close(),
        ],
    );
    assert_eq!(validate(&metadata), Ok(()));
}

// ------------------------------------------------------------------------- refused --

/// **Producing another provider's resource.** This would let TLS manufacture a `TcpStream` it does
/// not own and cannot close — a leak the ABI could never detect.
#[test]
fn producing_a_foreign_resource_is_refused() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_unwrap",
                vec![consumed("tcp_stream"), produced("tcp_stream")],
                None,
            ),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata).iter().any(|v| matches!(
            v,
            AbiViolation::ForeignResourceNotConsumed { resource_type, .. }
                if resource_type == "tcp_stream"
        )),
        "producing a foreign resource must be refused: {:?}",
        violations(&metadata)
    );
}

/// **Claiming the foreign resource's close.** Two closes for one resource is a double-release.
#[test]
fn declaring_a_close_for_a_foreign_resource_is_refused() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_client_connect",
                vec![consumed("tcp_stream"), produced("tls_stream")],
                None,
            ),
            func(
                "stark_tls_close_tcp",
                vec![consumed("tcp_stream")],
                Some("tcp_stream"),
            ),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata)
            .iter()
            .any(|v| matches!(v, AbiViolation::ForeignResourceClaimsClose { .. })),
        "claiming another provider's close must be refused: {:?}",
        violations(&metadata)
    );
}

/// **Borrowing a foreign resource.** Not designed: a borrow across providers has no owner story and
/// no identified caller need. Refused until separately designed.
#[test]
fn borrowing_a_foreign_resource_is_refused() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_peek",
                vec![borrowed("tcp_stream"), produced("tls_stream")],
                None,
            ),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata)
            .iter()
            .any(|v| matches!(v, AbiViolation::ForeignResourceNotConsumed { .. })),
        "a foreign BORROW must be refused: {:?}",
        violations(&metadata)
    );
}

/// **An undeclared foreign resource.** A typo must be a validation error, not a link failure — this
/// is why `foreign_resources` is explicit rather than inferred.
#[test]
fn consuming_an_undeclared_foreign_resource_is_refused() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_client_connect",
                vec![consumed("tcp_strem"), produced("tls_stream")],
                None,
            ),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata).iter().any(|v| matches!(
            v,
            AbiViolation::HandleResourceTypeUndeclared { resource_type, .. }
                if resource_type == "tcp_strem"
        )),
        "a misspelled foreign resource must be refused at validation: {:?}",
        violations(&metadata)
    );
}

/// **A transfer that produces nothing of its own** is a disguised close of somebody else's
/// resource — the exact thing the close-shape rule exists to prevent.
#[test]
fn consuming_a_foreign_resource_without_producing_an_owned_one_is_refused() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func("stark_tls_discard", vec![consumed("tcp_stream")], None),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata)
            .iter()
            .any(|v| matches!(v, AbiViolation::ForeignConsumeWithoutOwnedOutput { .. })),
        "a foreign consume with no owned output must be refused: {:?}",
        violations(&metadata)
    );
}

/// **Two foreign sources in one call.** Refused until a use case exists: two would need an ordering
/// rule for the failure path, and CD-360's failure rule is written for exactly one source.
#[test]
fn consuming_two_foreign_resources_in_one_function_is_refused() {
    let mut foreign = vec![tcp_stream()];
    foreign.push(ForeignResource {
        provider: "stark-std-net".to_string(),
        resource: "tcp_listener".to_string(),
    });
    let metadata = tls_provider(
        foreign,
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_fuse",
                vec![
                    consumed("tcp_stream"),
                    consumed("tcp_listener"),
                    produced("tls_stream"),
                ],
                None,
            ),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata)
            .iter()
            .any(|v| matches!(v, AbiViolation::MultipleForeignConsumed { .. })),
        "two foreign sources must be refused: {:?}",
        violations(&metadata)
    );
}

/// **A declared foreign resource nobody consumes.** A dead declaration grants silent permission,
/// so it is refused the way an unreachable capability already is.
#[test]
fn an_unused_foreign_declaration_is_refused() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func("stark_tls_new", vec![produced("tls_stream")], None),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata)
            .iter()
            .any(|v| matches!(v, AbiViolation::ForeignResourceUnused { .. })),
        "an unused foreign declaration must be refused: {:?}",
        violations(&metadata)
    );
}

/// **The pre-CD-360 refusals must survive for non-transfer cases.** Declaring the foreign type in
/// `resource_types` — the naive workaround — still demands a close for it here, which is what
/// prevents one resource acquiring two competing closes.
#[test]
fn declaring_a_foreign_type_as_owned_still_demands_a_close() {
    let metadata = tls_provider(
        vec![],
        vec!["tls_stream", "tcp_stream"],
        vec![
            func(
                "stark_tls_client_connect",
                vec![consumed("tcp_stream"), produced("tls_stream")],
                None,
            ),
            tls_close(),
        ],
    );
    assert!(
        violations(&metadata).iter().any(|v| matches!(
            v,
            AbiViolation::ResourceTypeMissingClose { resource_type } if resource_type == "tcp_stream"
        )),
        "claiming ownership of a foreign type must still demand a close: {:?}",
        violations(&metadata)
    );
}

/// A resource this provider genuinely owns still cannot be left closeless. CD-360 must not have
/// created a route around the close obligation.
#[test]
fn an_owned_resource_still_requires_a_close() {
    let metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![func(
            "stark_tls_client_connect",
            vec![consumed("tcp_stream"), produced("tls_stream")],
            None,
        )],
    );
    assert!(
        violations(&metadata).iter().any(|v| matches!(
            v,
            AbiViolation::ResourceTypeMissingClose { resource_type } if resource_type == "tls_stream"
        )),
        "an owned resource must still require a close: {:?}",
        violations(&metadata)
    );
}

// ------------------------------------------------------- build-time resolution --
//
// `validate` can only check a provider against ITSELF. "Does the resource I consume actually
// exist, exactly once, and belong to whom I said" is answerable only against the selected set, so
// these rules live in `ProviderSet::select` and are refused at selection rather than at link — a
// linker cannot name the consumer, the resource and the provider that was expected.

use starkc::provider_resolve::{DeclaredProvider, ProviderSet, ResolveError};

fn declared(metadata: ProviderMetadata) -> DeclaredProvider {
    DeclaredProvider {
        crate_path: "native".to_string(),
        metadata,
        crate_name: "test-native".to_string(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        origin: "test".to_string(),
    }
}

/// A provider owning `tcp_stream`, under a chosen identity so owner-mismatch is expressible.
fn net_provider(name: &str) -> ProviderMetadata {
    ProviderMetadata {
        identity: ProviderIdentity {
            name: name.to_string(),
            semver: (0, 1, 0),
            abi_version: ABI_VERSION.to_string(),
        },
        target_triples: vec!["test-triple".to_string()],
        capabilities: vec!["tcp".to_string()],
        resource_types: vec!["tcp_stream".to_string()],
        foreign_resources: Vec::new(),
        functions: vec![FunctionDecl {
            name: "stark_tcp_stream_close".to_string(),
            capability: "tcp".to_string(),
            params: vec![consumed("tcp_stream")],
            is_close_for: Some("tcp_stream".to_string()),
            may_block: false,
        }],
    }
}

fn wrap_provider() -> ProviderMetadata {
    let mut metadata = tls_provider(
        vec![tcp_stream()],
        vec!["tls_stream"],
        vec![
            func(
                "stark_tls_client_connect",
                vec![consumed("tcp_stream"), produced("tls_stream")],
                None,
            ),
            tls_close(),
        ],
    );
    metadata.target_triples = vec!["test-triple".to_string()];
    metadata
}

fn resolve_errors(providers: Vec<ProviderMetadata>) -> Vec<ResolveError> {
    ProviderSet::select(
        providers.into_iter().map(declared).collect(),
        "test-triple",
        &["tls".to_string(), "tcp".to_string()],
    )
    .err()
    .unwrap_or_default()
}

/// The consumer and its owner both selected: the transfer resolves.
#[test]
fn a_transfer_resolves_when_the_owner_is_in_the_selected_set() {
    assert_eq!(
        resolve_errors(vec![wrap_provider(), net_provider("stark-std-net")]),
        Vec::new(),
        "a declared foreign consumption whose owner is present must resolve"
    );
}

/// **Nobody owns it.** Refused at selection, naming the consumer and the expected provider —
/// information a link failure could not carry.
#[test]
fn a_foreign_resource_with_no_owner_is_refused() {
    let errors = resolve_errors(vec![wrap_provider()]);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::ForeignResourceUnsupplied { .. })),
        "an unsupplied foreign resource must be refused at selection: {errors:?}"
    );
}

/// **Two owners.** Ownership must be unambiguous: the destination's release authority derives from
/// the source's identity, and two owners means two closes for one resource.
#[test]
fn a_foreign_resource_with_two_owners_is_refused() {
    let errors = resolve_errors(vec![
        wrap_provider(),
        net_provider("stark-std-net"),
        net_provider("other-net"),
    ]);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::ForeignResourceAmbiguous { .. })),
        "two owners of one foreign resource must be refused: {errors:?}"
    );
}

/// **Right name, wrong owner.** Identity is structural over `{nominal, provider, resource}`, so a
/// matching resource NAME under a different provider is a DIFFERENT resource. Accepting it would
/// transfer ownership of something the consumer never declared it could consume.
#[test]
fn a_foreign_resource_owned_by_a_different_provider_is_refused() {
    let errors = resolve_errors(vec![wrap_provider(), net_provider("some-other-net")]);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::ForeignResourceOwnerMismatch { .. })),
        "a foreign resource owned by a provider the consumer did not name must be refused: \
         {errors:?}"
    );
}
