//! WP-C7.8.8 step 2 — signature derivation from validated provider metadata.
//!
//! The claim under test is CD-224's invariant: **there is one authoritative callable signature, and
//! it is the provider's.** A package names a symbol; the shape is computed. Nothing here reads a
//! declared signature, because there is none to read.
//!
//! The positive cases derive against the **real registry**, so a drift between the compiler's
//! provider metadata and what these tests expect shows up here — which is the failure CD-219 found
//! by execution, now reachable much earlier.

use starkc::provider_abi::{AbiParam, FunctionDecl, ScalarTy};
use starkc::provider_derive::{derive, derive_all, DeriveError, DerivedTy};
use starkc::provider_registry;
use std::collections::BTreeMap;

fn errors_for(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn nominals(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// A declaration straight from the compiler's registry, so these tests derive against the same
/// metadata a build would.
fn decl(provider: &str, function: &str) -> FunctionDecl {
    provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == provider)
        .unwrap_or_else(|| panic!("{provider} must be registered"))
        .metadata
        .functions
        .into_iter()
        .find(|f| f.name == function)
        .unwrap_or_else(|| panic!("{function} must be declared"))
}

// ------------------------------------------------------------------- positive --

/// One out-slot derives `Result<UInt64, E>` — no parameters, because the provider has none.
#[test]
fn one_out_slot_derives_a_single_result() {
    let sig = derive(
        "Instant::now_ns",
        "clock",
        &decl("stark-std-time", "stark_time_monotonic_now_ns"),
        &nominals(&[]),
        &errors_for(&[("clock", "RawTimeError")]),
    )
    .expect("derives");

    assert!(sig.params.is_empty());
    assert!(sig.receiver.is_none());
    assert_eq!(sig.results, vec![DerivedTy::Scalar(ScalarTy::U64)]);
    assert_eq!(sig.render_return(), "Result<UInt64, RawTimeError>");
}

/// **The case CD-219's hand-written mirror got wrong.** Two out-slots derive a tuple, in declared
/// order. Nothing here could have declared "one slot" — the shape comes from the provider.
#[test]
fn two_out_slots_derive_a_tuple_in_declared_order() {
    let sig = derive(
        "SystemTime::unix_now",
        "clock",
        &decl("stark-std-time", "stark_time_unix_now"),
        &nominals(&[]),
        &errors_for(&[("clock", "RawTimeError")]),
    )
    .expect("derives");

    assert_eq!(
        sig.results,
        vec![
            DerivedTy::Scalar(ScalarTy::I64),
            DerivedTy::Scalar(ScalarTy::U32)
        ]
    );
    assert_eq!(sig.render_return(), "Result<(Int64, UInt32), RawTimeError>");
}

/// A buffer in, two out-slots out: parameters and results separate exactly along the ABI's in/out
/// split, which is what makes §11.1's caller-owned rule visible in the source signature.
#[test]
fn buffers_are_parameters_and_out_slots_are_results() {
    let sig = derive(
        "env::var_len",
        "environment-read",
        &decl("stark-std-env", "stark_env_var_len"),
        &nominals(&[]),
        &errors_for(&[("environment-read", "RawEnvError")]),
    )
    .expect("derives");

    assert_eq!(sig.params, vec![DerivedTy::SharedBytes]);
    assert_eq!(
        sig.results,
        vec![
            DerivedTy::Scalar(ScalarTy::Bool),
            DerivedTy::Scalar(ScalarTy::U64)
        ]
    );
}

/// A `BufferInOut` stays a **parameter**, not a result: §11.1 makes it caller-initialised and
/// caller-owned, so it is not a value the call produces.
#[test]
fn an_in_out_buffer_is_a_parameter_not_a_result() {
    let sig = derive(
        "env::var_fill",
        "environment-read",
        &decl("stark-std-env", "stark_env_var_fill"),
        &nominals(&[]),
        &errors_for(&[("environment-read", "RawEnvError")]),
    )
    .expect("derives");

    assert_eq!(
        sig.params,
        vec![DerivedTy::SharedBytes, DerivedTy::ExclusiveBytes]
    );
    assert_eq!(sig.results, vec![DerivedTy::Scalar(ScalarTy::U64)]);
}

/// A `HandleOut` derives an **owned** result — the call produces a resource. Core `File` is the
/// nominal here, supplied by the compiler rather than by any package declaration (CD-224).
#[test]
fn a_handle_out_derives_an_owned_resource_result() {
    let sig = derive(
        "File::open_raw",
        "filesystem-read",
        &decl("stark-std-file", "stark_file_open"),
        &nominals(&[("file", "File")]),
        &errors_for(&[("filesystem-read", "RawIoError")]),
    )
    .expect("derives");

    assert_eq!(sig.params, vec![DerivedTy::SharedBytes]);
    assert_eq!(
        sig.results,
        vec![DerivedTy::OwnedResource {
            nominal: "File".to_string()
        }]
    );
    assert_eq!(sig.render_return(), "Result<File, RawIoError>");
}

/// A `HandleBorrowed` first parameter on an associated path becomes the **receiver**, so
/// `file.read_raw(buf)` reads as a method and the file survives the call (ABI §8's default).
#[test]
fn a_borrowed_handle_becomes_the_receiver() {
    let sig = derive(
        "File::read_raw",
        "filesystem-read",
        &decl("stark-std-file", "stark_file_read"),
        &nominals(&[("file", "File")]),
        &errors_for(&[("filesystem-read", "RawIoError")]),
    )
    .expect("derives");

    assert_eq!(
        sig.receiver,
        Some(DerivedTy::SharedResource {
            nominal: "File".to_string()
        })
    );
    assert_eq!(sig.params, vec![DerivedTy::ExclusiveBytes]);
    assert_eq!(
        sig.results,
        vec![
            DerivedTy::Scalar(ScalarTy::U64),
            DerivedTy::Scalar(ScalarTy::Bool)
        ]
    );
}

/// `accept` borrows a listener and produces a stream — the one shape where a call both keeps a
/// resource and creates another.
#[test]
fn accept_derives_a_borrowed_receiver_and_an_owned_result() {
    let sig = derive(
        "TcpListener::accept_raw",
        "network-client",
        &decl("stark-std-net", "stark_tcp_listener_accept"),
        &nominals(&[("tcp_listener", "TcpListener"), ("tcp_stream", "TcpStream")]),
        &errors_for(&[("network-client", "RawNetError")]),
    )
    .expect("derives");

    assert_eq!(
        sig.receiver,
        Some(DerivedTy::SharedResource {
            nominal: "TcpListener".to_string()
        })
    );
    assert_eq!(sig.render_return(), "Result<TcpStream, RawNetError>");
}

// ------------------------------------------------- the six derivation failures --

/// **13.5a** — a resource the package does not bind. The derived signature would name a type that
/// does not exist.
#[test]
fn an_unbound_resource_in_a_signature_is_rejected() {
    let e = derive(
        "TcpStream::connect_raw",
        "network-client",
        &decl("stark-std-net", "stark_tcp_stream_connect"),
        &nominals(&[]), // nothing bound
        &errors_for(&[("network-client", "RawNetError")]),
    )
    .expect_err("must fail");

    assert!(matches!(
        e,
        DeriveError::UnboundResourceInSignature { ref resource, .. } if resource == "tcp_stream"
    ));
}

/// **13.5d** — associated placement carries one claim: a method on `T` operates on a `T`. Binding
/// `stark_tcp_listener_accept` at `TcpStream::accept_raw` would derive a method on `TcpStream`
/// whose receiver is a `TcpListener`.
#[test]
fn a_receiver_of_the_wrong_resource_is_rejected() {
    let e = derive(
        "TcpStream::accept_raw",
        "network-client",
        &decl("stark-std-net", "stark_tcp_listener_accept"),
        &nominals(&[("tcp_listener", "TcpListener"), ("tcp_stream", "TcpStream")]),
        &errors_for(&[("network-client", "RawNetError")]),
    )
    .expect_err("must fail");

    match e {
        DeriveError::OwnershipCategoryConflict {
            declared_on,
            receiver_resource,
            ..
        } => {
            assert_eq!(declared_on, "TcpStream");
            assert_eq!(receiver_resource, "TcpListener");
        }
        other => panic!("expected OwnershipCategoryConflict, got {other:?}"),
    }
}

/// **13.5e** — no raw error type for the capability, so `Result<_, E>` has no `E`.
#[test]
fn a_missing_error_type_is_rejected() {
    let e = derive(
        "Instant::now_ns",
        "clock",
        &decl("stark-std-time", "stark_time_monotonic_now_ns"),
        &nominals(&[]),
        &errors_for(&[]),
    )
    .expect_err("must fail");

    assert!(matches!(
        e,
        DeriveError::MissingErrorType { ref capability, .. } if capability == "clock"
    ));
}

/// **13.5f** — two ABI-distinct functions deriving to one item path. Invisible from either binding
/// alone, so it is checked across the set.
#[test]
fn two_symbols_deriving_to_one_item_path_are_rejected() {
    let failures = derive_all(
        &[
            (
                "Instant::now_ns".to_string(),
                "clock".to_string(),
                decl("stark-std-time", "stark_time_monotonic_now_ns"),
            ),
            (
                "Instant::now_ns".to_string(),
                "clock".to_string(),
                decl("stark-std-time", "stark_time_unix_now"),
            ),
        ],
        &nominals(&[]),
        &errors_for(&[("clock", "RawTimeError")]),
    )
    .expect_err("must fail");

    match failures
        .iter()
        .find(|f| matches!(f, DeriveError::AmbiguousItemPath { .. }))
    {
        Some(DeriveError::AmbiguousItemPath { item_path, symbols }) => {
            assert_eq!(item_path, "Instant::now_ns");
            assert_eq!(
                symbols,
                &vec![
                    "stark_time_monotonic_now_ns".to_string(),
                    "stark_time_unix_now".to_string()
                ]
            );
        }
        other => panic!("expected AmbiguousItemPath, got {other:?}"),
    }
}

/// **13.5b/13.5c** are reachable but currently unreached: every `AbiParam` form is admitted, and
/// the result shape is total — zero, one or several out-slots always derive. The variants exist so
/// a future ABI form fails loudly here rather than deriving something plausible.
#[test]
fn every_current_abi_form_derives() {
    let all = FunctionDecl {
        name: "probe".to_string(),
        capability: "cap".to_string(),
        params: vec![
            AbiParam::ScalarIn(ScalarTy::U32),
            AbiParam::ScalarInOut(ScalarTy::I16),
            AbiParam::BufferIn,
            AbiParam::BufferInOut,
            AbiParam::HandleBorrowed {
                resource_type: "r".to_string(),
            },
            AbiParam::ScalarOut(ScalarTy::F64),
            AbiParam::HandleOut {
                resource_type: "r".to_string(),
            },
        ],
        is_close_for: None,
        may_block: false,
    };

    let sig = derive(
        "R::probe",
        "cap",
        &all,
        &nominals(&[("r", "R")]),
        &errors_for(&[("cap", "E")]),
    )
    .expect("every admitted form must derive");

    // The handle is the receiver; the rest are parameters in order; out-slots are results.
    assert!(
        sig.receiver.is_none(),
        "the handle is not first, so it is not a receiver"
    );
    assert_eq!(sig.params.len(), 5);
    assert_eq!(sig.results.len(), 2);
}

/// Derivation is a pure function of the declaration: the same metadata derives the same signature
/// regardless of binding order, so nothing about manifest ordering reaches a synthesized item.
#[test]
fn derivation_is_order_independent() {
    let a = derive_all(
        &[
            (
                "A::one".to_string(),
                "clock".to_string(),
                decl("stark-std-time", "stark_time_unix_now"),
            ),
            (
                "B::two".to_string(),
                "clock".to_string(),
                decl("stark-std-time", "stark_time_monotonic_now_ns"),
            ),
        ],
        &nominals(&[]),
        &errors_for(&[("clock", "E")]),
    )
    .expect("derives");
    let b = derive_all(
        &[
            (
                "B::two".to_string(),
                "clock".to_string(),
                decl("stark-std-time", "stark_time_monotonic_now_ns"),
            ),
            (
                "A::one".to_string(),
                "clock".to_string(),
                decl("stark-std-time", "stark_time_unix_now"),
            ),
        ],
        &nominals(&[]),
        &errors_for(&[("clock", "E")]),
    )
    .expect("derives");

    assert_eq!(a, b);
}
