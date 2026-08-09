//! **HC9 — the MIR verifier must expect a transferred handle at its OWNER's identity.**
//!
//! CD-360 found the transfer rule implemented in three places and fixed each. There was a fourth,
//! and it stayed hidden because CD-360's fixture built its `ValidatedProviderCall` by hand and
//! emitted from it directly — never running the verifier over a real transfer. HC9's first native
//! build of `stark-tls-consumer` produced:
//!
//! ```text
//! MIR-0005 stark_tls::connect bb53: call argument:
//!   expected HostResource(… provider: "stark-std-tls", resource: "tcp_stream"),
//!   found    HostResource(… provider: "stark-std-net", resource: "tcp_stream")
//! ```
//!
//! The planner was right and the verifier was wrong — the worst arrangement, because the program
//! was correct and the compiler refused it. `provider_sig::owner_of` is now the single statement of
//! the rule, and both callers use it.
//!
//! These tests are at the level the defect lived at: the ownership function and the signature
//! derived from it. The end-to-end proof is `stark-tls-consumer`, which the package gate builds
//! natively and runs against a live TLS peer.

use starkc::mir::provider_sig::{owner_of, signature};
use starkc::mir::{ForeignResourceCall, MirTy};
use starkc::provider_abi::AbiParam;
use starkc::provider_bind::ResourceRegistry;

mod common;

const NET: &str = "stark-std-net";
const TLS: &str = "stark-std-tls";

fn transfer() -> Vec<ForeignResourceCall> {
    vec![ForeignResourceCall {
        provider: NET.to_string(),
        resource: "tcp_stream".to_string(),
        owner_resource_types: vec!["tcp_stream".to_string()],
    }]
}

/// A registry with both nominals, as a real program's would have: `tcp_stream` bound by the net
/// package and `tls_stream` by the TLS package, each to its own `ItemId`.
fn registry() -> ResourceRegistry {
    let mut registry = ResourceRegistry::builtin();
    registry.register_nominal("tcp_stream", starkc::hir::ItemId(7));
    registry.register_nominal("tls_stream", starkc::hir::ItemId(8));
    registry
}

/// **The rule itself.** A declared-foreign type resolves to its owner; everything else to the
/// caller.
#[test]
fn a_foreign_resource_resolves_to_its_owner_and_everything_else_to_the_caller() {
    let foreign = transfer();
    assert_eq!(owner_of("tcp_stream", TLS, &foreign), NET);
    assert_eq!(owner_of("tls_stream", TLS, &foreign), TLS);
    // With no declaration there is no transfer, so the caller owns everything it names. This is
    // what keeps the rule from silently reassigning ownership of an ordinary call.
    assert_eq!(owner_of("tcp_stream", TLS, &[]), TLS);
}

/// A name that merely RESEMBLES the declared one is not it. Matching loosely would let
/// `tcp_stream2` inherit the net provider's identity.
#[test]
fn only_the_exact_declared_resource_is_treated_as_foreign() {
    let foreign = transfer();
    assert_eq!(owner_of("tcp_stream2", TLS, &foreign), TLS);
    assert_eq!(owner_of("network-client", TLS, &foreign), TLS);
    assert_eq!(owner_of("", TLS, &foreign), TLS);
}

/// **The defect, as a regression test.** The verifier's expected signature for a transfer must
/// carry the OWNER's provider on the consumed handle and this provider's on the produced one.
/// Before the fix both said `stark-std-tls`, and every cross-provider transfer failed MIR-0005.
#[test]
fn the_verified_signature_of_a_transfer_names_the_owner_on_the_consumed_handle() {
    let params = vec![
        AbiParam::HandleConsumed {
            resource_type: "tcp_stream".to_string(),
        },
        AbiParam::BufferIn,
        AbiParam::HandleOut {
            resource_type: "tls_stream".to_string(),
        },
    ];
    let (tys, _ret) = signature(&params, &registry(), TLS, &transfer())
        .expect("a transfer must have a signature");

    assert_eq!(
        tys[0],
        MirTy::host_resource(
            starkc::mir::HostResourceNominal::Item(starkc::hir::ItemId(7)),
            NET,
            "tcp_stream",
        ),
        "the consumed handle keeps the identity it was CREATED with; deriving it from the consumer \
         hands the provider a tag naming a different resource"
    );
    assert_eq!(
        tys[2],
        MirTy::host_resource(
            starkc::mir::HostResourceNominal::Item(starkc::hir::ItemId(8)),
            TLS,
            "tls_stream",
        ),
        "the produced handle is this provider's own"
    );
    assert_ne!(tys[0], tys[2], "a transfer is a genuine type transition");
}

/// The same parameter list WITHOUT the declaration must still resolve against the caller. The fix
/// widened one case; it must not have changed the ordinary one.
#[test]
fn an_ordinary_call_is_unaffected_by_the_transfer_rule() {
    let params = vec![AbiParam::HandleBorrowed {
        resource_type: "tls_stream".to_string(),
    }];
    let (tys, _ret) = signature(&params, &registry(), TLS, &[]).expect("an ordinary call");
    assert_eq!(
        tys[0],
        MirTy::Ref {
            mutable: false,
            inner: Box::new(MirTy::host_resource(
                starkc::mir::HostResourceNominal::Item(starkc::hir::ItemId(8)),
                TLS,
                "tls_stream",
            )),
        },
        "a borrowed handle stays a shared reference at the calling provider's identity"
    );
}

/// The planner and the verifier must agree, because disagreement is the whole defect. Both derive
/// the consumed handle's type through `owner_of`; this asserts the results are equal rather than
/// merely that each looks plausible on its own.
#[test]
fn the_planner_and_the_verifier_agree_on_a_transferred_handles_type() {
    use starkc::provider_abi::{FunctionDecl, ProviderIdentity};

    let function = FunctionDecl {
        name: "stark_tls_stream_connect".to_string(),
        capability: "network-client".to_string(),
        params: vec![
            AbiParam::HandleConsumed {
                resource_type: "tcp_stream".to_string(),
            },
            AbiParam::HandleOut {
                resource_type: "tls_stream".to_string(),
            },
        ],
        is_close_for: None,
        may_block: true,
    };
    let call = starkc::mir::ValidatedProviderCall {
        provider: ProviderIdentity {
            name: TLS.to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "network-client".to_string(),
        function: function.clone(),
        target_triple: "aarch64-apple-darwin".to_string(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        foreign_resources: transfer(),
        provider_crate: "stark-tls-native".to_string(),
        provider_resource_types: vec!["tls_stream".to_string()],
        provider_target_triples: vec!["aarch64-apple-darwin".to_string()],
    };

    let plan = starkc::provider_bind::plan(
        starkc::mir::ProviderCallId(0),
        &call,
        &registry(),
        starkc::provider_bind::StatusBinding::new(),
    )
    .expect("the planner must plan a transfer");
    let (verified, _ret) =
        signature(&function.params, &registry(), TLS, &call.foreign_resources).unwrap();

    let planned = plan
        .inputs
        .iter()
        .find_map(|input| match input {
            starkc::provider_bind::ProviderInputPlan::HandleConsumed { mir_type, .. } => {
                Some(mir_type.clone())
            }
            _ => None,
        })
        .expect("the plan must carry the consumed handle");

    assert_eq!(
        planned, verified[0],
        "the planner's actual type and the verifier's expected type must be THE SAME value. They \
         were derived independently and disagreed, which refused every correct transfer"
    );
}
