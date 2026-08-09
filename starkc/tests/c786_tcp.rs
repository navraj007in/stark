//! WP-C7.8.6 — TCP: registered, selectable, and precisely blocked.
//!
//! `stark-net` is the fourth provider and the one that stops short of execution — **because of a
//! disposition, not a gap**. Packet 4 made TCP a *package* capability, so `tcp_listener` and
//! `tcp_stream` are package types. There is no Core type to bind them to, and adding one would be
//! the CE1 Core change Packet 4 ruled against. Every `stark-net` function carries a handle, so none
//! of them plans until a STARK package declares those types.
//!
//! That makes this slice's job to establish exactly three things:
//!
//! 1. selection, target applicability and the status vocabulary are live;
//! 2. resource-carrying calls are refused **naming the resource type**, not vaguely;
//! 3. the refusal disappears the moment a binding exists — proven by registering one here, so the
//!    remaining work is a package declaration rather than more compiler machinery.
//!
//! Packet 5's inbound rule is asserted structurally: a listener exists only through an explicit
//! `bind(address)`, and the declaration carries no default address for one to be created without.

use starkc::mir::ProviderCallId;
use starkc::provider_abi::{AbiParam, ScalarTy};
use starkc::provider_bind::{
    plan, PlanError, ProviderInputPlan, ProviderOutputPlan, ResourceRegistry, StatusOutcome,
    STATUS_SUCCESS,
};
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;

const LINUX: &str = "x86_64-unknown-linux-gnu";

fn tcp() -> ProviderSet {
    let providers = provider_registry::first_party()
        .into_iter()
        .filter(|provider| provider.crate_name == "stark-net-native")
        .collect();
    ProviderSet::select(
        providers,
        LINUX,
        &["network-client".to_string(), "network-listen".to_string()],
    )
    .expect("tcp must be available on a Tier-1 target")
}

/// A registry with the two TCP resource types bound, standing in for the package declaration that
/// will supply them. `UInt64` is a placeholder for whatever nominal type the package declares — the
/// point is that the machinery is complete and waiting on a *name*.
fn registry_with_tcp_types() -> ResourceRegistry {
    let mut r = ResourceRegistry::builtin();
    r.register_nominal("tcp_listener", starkc::hir::ItemId(201));
    r.register_nominal("tcp_stream", starkc::hir::ItemId(202));
    r
}

// --------------------------------------------------------------- live now --

/// Selection works: `tcp` resolves to `stark-net`, and only when something declares it.
#[test]
fn tcp_selects_stark_net_and_only_on_request() {
    let set = tcp();
    assert_eq!(set.providers().len(), 1);
    assert!(set
        .providers()
        .iter()
        .any(|provider| provider.crate_name == "stark-net-native"));

    for function in [
        "stark_tcp_listener_bind",
        "stark_tcp_listener_accept",
        "stark_tcp_stream_connect",
        "stark_tcp_stream_read",
        "stark_tcp_stream_write",
        "stark_tcp_listener_close",
        "stark_tcp_stream_close",
    ] {
        let capability = if function.contains("listener") {
            "network-listen"
        } else {
            "network-client"
        };
        set.resolve(capability, function)
            .unwrap_or_else(|e| panic!("{function} must resolve: {e:#?}"));
    }

    // Not requested, not selected.
    let others = ProviderSet::select(
        provider_registry::first_party(),
        LINUX,
        &["clock".to_string()],
    )
    .expect("selects");
    assert!(others
        .resolve("network-client", "stark_tcp_stream_connect")
        .is_err());
}

/// Eleven declared codes — the richest vocabulary of the four providers, including `AddressInUse`,
/// which only `bind` can produce. Everything else is a contract violation.
#[test]
fn the_tcp_status_vocabulary_is_channel_one() {
    let call = tcp()
        .resolve("network-client", "stark_tcp_stream_connect")
        .unwrap();
    let p = plan(
        ProviderCallId(0),
        &call,
        &registry_with_tcp_types(),
        call.status_binding.clone(),
    )
    .expect("plans once the resource types are bound");

    assert_eq!(p.classify(STATUS_SUCCESS), StatusOutcome::Success);
    for code in 1u32..=11 {
        assert!(
            matches!(p.classify(code), StatusOutcome::RecoverableError { .. }),
            "code {code} must be recoverable"
        );
    }
    for code in [12u32, 99, u32::MAX] {
        assert_eq!(p.classify(code), StatusOutcome::ContractViolation { code });
    }
}

// ------------------------------------------------- precisely blocked now --

/// **The boundary.** Every `stark-net` function carries a handle, so none plans under the
/// compiler's own registry — and the refusal names the resource type rather than saying resources
/// are unsupported.
#[test]
fn every_tcp_function_is_refused_naming_its_resource_type() {
    let set = tcp();
    for function in [
        "stark_tcp_listener_bind",
        "stark_tcp_listener_accept",
        "stark_tcp_stream_connect",
        "stark_tcp_stream_read",
        "stark_tcp_stream_write",
        "stark_tcp_listener_close",
        "stark_tcp_stream_close",
    ] {
        let capability = if function.contains("listener") {
            "network-listen"
        } else {
            "network-client"
        };
        let call = set.resolve(capability, function).unwrap();
        match plan(
            ProviderCallId(0),
            &call,
            &ResourceRegistry::builtin(),
            call.status_binding.clone(),
        ) {
            Err(PlanError::UnboundResourceType { resource_type, .. }) => {
                assert!(
                    resource_type == "tcp_listener" || resource_type == "tcp_stream",
                    "{function} named an unexpected resource type: {resource_type}"
                );
            }
            other => panic!("{function} must be refused, got {other:#?}"),
        }
    }
}

/// `file` stays bound while the TCP types do not — the refusal is per type, so C7.8.4's work is not
/// undone by C7.8.6's boundary.
#[test]
fn file_remains_bound_while_tcp_types_are_not() {
    let registry = ResourceRegistry::builtin();
    assert!(registry.lookup("file").is_some());
    assert!(registry.lookup("tcp_listener").is_none());
    assert!(registry.lookup("tcp_stream").is_none());
}

/// **The refusal is one registration away from gone.** With the two types bound, the entire TCP
/// surface plans — so what remains is a package declaring `TcpListener`/`TcpStream`, not more
/// compiler work.
#[test]
fn the_whole_tcp_surface_plans_once_its_types_are_bound() {
    let registry = registry_with_tcp_types();
    let set = tcp();

    for function in [
        "stark_tcp_listener_bind",
        "stark_tcp_listener_accept",
        "stark_tcp_stream_connect",
        "stark_tcp_stream_read",
        "stark_tcp_stream_write",
        "stark_tcp_listener_close",
        "stark_tcp_stream_close",
    ] {
        let capability = if function.contains("listener") {
            "network-listen"
        } else {
            "network-client"
        };
        let call = set.resolve(capability, function).unwrap();
        let p = plan(
            ProviderCallId(0),
            &call,
            &registry,
            call.status_binding.clone(),
        )
        .unwrap_or_else(|e| panic!("{function} must plan once bound: {e:#?}"));
        assert!(p.covers(call.function.params.len()), "{function}");
    }
}

// ------------------------------------------------------ ownership shapes --

/// `accept` is the shape no earlier capability had: it **borrows** a listener and **produces** a
/// stream, so one call both keeps a resource and creates another.
#[test]
fn accept_borrows_a_listener_and_produces_a_stream() {
    let call = tcp()
        .resolve("network-listen", "stark_tcp_listener_accept")
        .unwrap();
    let p = plan(
        ProviderCallId(0),
        &call,
        &registry_with_tcp_types(),
        call.status_binding.clone(),
    )
    .expect("plans");

    assert!(matches!(
        p.inputs.as_slice(),
        [ProviderInputPlan::HandleBorrowed { index: 0, .. }]
    ));
    assert!(matches!(
        p.outputs.as_slice(),
        [ProviderOutputPlan::Handle { index: 1, .. }]
    ));
    assert!(p.inputs[0].requires_live_borrow());
}

/// Each resource type has exactly one close, and they are distinct functions — closing a listener
/// must not be able to close a stream.
#[test]
fn each_tcp_resource_has_its_own_close() {
    let provider = provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == "stark-std-net")
        .expect("stark-net must be registered");

    for resource in ["tcp_listener", "tcp_stream"] {
        let closers: Vec<&str> = provider
            .metadata
            .functions
            .iter()
            .filter(|f| f.is_close_for.as_deref() == Some(resource))
            .map(|f| f.name.as_str())
            .collect();
        assert_eq!(closers.len(), 1, "{resource}: {closers:?}");
    }

    starkc::provider_abi::validate(&provider.metadata)
        .unwrap_or_else(|v| panic!("stark-net metadata must validate: {v:#?}"));
}

// --------------------------------------------------- Packet 5 boundary --

/// **Inbound TCP exists only through an explicit `bind(address)`.** The address is a `BufferIn` the
/// program supplies; there is no default, no implicit `0.0.0.0`, and no other function produces a
/// listener — so one cannot come into being as a side effect.
#[test]
fn a_listener_exists_only_through_an_explicit_bind() {
    let provider = provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == "stark-std-net")
        .expect("registered");

    let producers: Vec<&str> = provider
        .metadata
        .functions
        .iter()
        .filter(|f| {
            f.params.iter().any(|p| {
                matches!(p, AbiParam::HandleOut { resource_type } if resource_type == "tcp_listener")
            })
        })
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(
        producers,
        vec!["stark_tcp_listener_bind"],
        "exactly one function may produce a listener"
    );

    // And it takes the address from the program, as a buffer.
    let bind = provider
        .metadata
        .functions
        .iter()
        .find(|f| f.name == "stark_tcp_listener_bind")
        .unwrap();
    assert!(
        matches!(
            bind.params.as_slice(),
            [AbiParam::BufferIn, AbiParam::HandleOut { .. }]
        ),
        "{:#?}",
        bind.params
    );
}

/// No `stark-net` function exposes a raw descriptor or takes one — Packet 5 forbids it, and the
/// closed `AbiParam` vocabulary makes it unrepresentable, but the declaration is checked anyway so
/// a future addition cannot slip past.
#[test]
fn no_raw_descriptor_crosses_the_boundary() {
    let provider = provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == "stark-std-net")
        .expect("registered");

    for f in &provider.metadata.functions {
        let name = f.name.to_ascii_lowercase();
        for forbidden in ["fd", "descriptor", "socket_raw", "into_raw", "from_raw"] {
            assert!(
                !name.contains(forbidden),
                "{} looks like it exposes a descriptor",
                f.name
            );
        }
        // Every scalar that crosses is a byte count, never a handle in disguise.
        for p in &f.params {
            if let AbiParam::ScalarOut(t) = p {
                assert_eq!(*t, ScalarTy::U64, "{}: unexpected scalar out", f.name);
            }
        }
    }
}
