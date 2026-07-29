//! WP-C7.8.4 — file I/O, and the first **bound** resource type.
//!
//! The resource framework landed in C7.8.2d-4 and was proven with a synthetic type. This slice
//! turns `File` on by *registering* it — `"file" → MirTy::Core(CoreType::File, [])` — which is the
//! claim that framework was built to make good.
//!
//! Packet 3's close semantics are what `stark-file` implements, and this file pins them:
//! `stark_file_complete` is the recoverable completion operation on a **borrowed** handle, and
//! `stark_file_close` is the ABI close on a **consumed** one. ABI §13.1 forces that separation —
//! close takes exactly one consumed handle and returns only a status, so anything fallible and
//! argument-bearing has to happen before it.
//!
//! Packet 4 holds throughout: no Core symbol is added. `stark_file_read`/`write` are the byte
//! primitives that package conveniences layer over, and `06-Standard-Library.md` is untouched.

use starkc::hir::CoreType;
use starkc::mir::{MirTy, ProviderCallId};
use starkc::provider_abi::{AbiParam, ScalarTy};
use starkc::provider_bind::{
    plan, PlanError, ProviderInputPlan, ProviderOutputPlan, ResourceRegistry, StatusOutcome,
    STATUS_SUCCESS,
};
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;

const LINUX: &str = "x86_64-unknown-linux-gnu";

fn filesystem() -> Vec<String> {
    vec!["filesystem".to_string()]
}

fn selected() -> ProviderSet {
    ProviderSet::select(provider_registry::first_party(), LINUX, &filesystem())
        .expect("filesystem must be available on a Tier-1 target")
}

fn planned(function: &str) -> starkc::provider_bind::ProviderBindingPlan {
    let call = selected()
        .resolve("filesystem", function)
        .unwrap_or_else(|e| panic!("{function} must resolve: {e:#?}"));
    plan(
        ProviderCallId(0),
        &call,
        &ResourceRegistry::builtin(),
        call.status_binding.clone(),
    )
    .unwrap_or_else(|e| panic!("{function} must plan: {e:#?}"))
}

// ------------------------------------------------------- the binding --

/// **The registration that is this slice.** `"file"` maps to the Core `File` type, so a provider
/// call carrying it now plans where C7.8.2d-4 refused it.
#[test]
fn the_file_resource_type_is_bound_to_core_file() {
    let registry = ResourceRegistry::builtin();
    assert_eq!(
        registry.lookup("file"),
        Some(&MirTy::Core(CoreType::File, Vec::new())),
        "`file` must bind to the Core File type, not an invented one"
    );
}

/// Binding is per type. `File` being admitted says nothing about any other resource type, so
/// MIR-0024 keeps discriminating rather than becoming dead.
#[test]
fn binding_file_does_not_admit_other_resource_types() {
    let registry = ResourceRegistry::builtin();
    for other in ["File", "socket", "custom-db-session", "directory"] {
        assert!(
            registry.lookup(other).is_none(),
            "{other} must remain unbound"
        );
    }
}

/// Every declared function plans now — the whole `stark-file` surface, including the three
/// resource-carrying forms that were refused a slice ago.
#[test]
fn the_whole_file_surface_plans() {
    for function in [
        "stark_file_open",
        "stark_file_create",
        "stark_file_read",
        "stark_file_write",
        "stark_file_complete",
        "stark_file_close",
    ] {
        let p = planned(function);
        assert!(
            p.covers(
                selected()
                    .resolve("filesystem", function)
                    .unwrap()
                    .function
                    .params
                    .len()
            ),
            "{function}: plan does not cover its declaration"
        );
    }
}

// ------------------------------------------- ownership forms per §8 --

/// `open` and `create` **produce** a resource: the handle is a `HandleOut`, so it is an output slot
/// validated on success only, never an input.
#[test]
fn open_and_create_produce_a_handle_as_an_output() {
    for function in ["stark_file_open", "stark_file_create"] {
        let p = planned(function);
        assert!(
            matches!(
                p.outputs.as_slice(),
                [ProviderOutputPlan::Handle {
                    index: 1,
                    type_id: 0,
                    ..
                }]
            ),
            "{function}: {:#?}",
            p.outputs
        );
        // The path is a borrowed buffer input, passed verbatim (Packet 5).
        assert!(matches!(
            p.inputs.as_slice(),
            [ProviderInputPlan::BufferIn { index: 0 }]
        ));
    }
}

/// `read`, `write` and `complete` **borrow** the handle: the caller keeps the file and can use it
/// again afterwards, which is exactly what an ordinary operation needs (ABI §8's default).
#[test]
fn ordinary_operations_borrow_the_handle() {
    for function in ["stark_file_read", "stark_file_write", "stark_file_complete"] {
        let p = planned(function);
        let handle = p
            .inputs
            .iter()
            .find(|i| matches!(i, ProviderInputPlan::HandleBorrowed { .. }))
            .unwrap_or_else(|| panic!("{function} must borrow its handle: {:#?}", p.inputs));
        assert!(handle.requires_live_borrow());
        assert!(
            !p.inputs
                .iter()
                .any(|i| matches!(i, ProviderInputPlan::HandleConsumed { .. })),
            "{function} must not consume the file"
        );
    }
}

/// **Packet 3's separation, as declared metadata rather than as prose.**
///
/// `complete` borrows and can fail recoverably; `close` consumes and is the ABI close. ABI §13.1
/// forces this: a close function takes exactly one consumed handle and no other parameter, because
/// MIR's `Drop` terminator supplies only the resource being dropped — there is no argument list at
/// a drop site. Anything fallible and argument-bearing must therefore be a separate call made
/// *before* Drop, which is precisely what `complete` is.
#[test]
fn complete_is_recoverable_and_close_consumes() {
    let set = selected();

    let complete = set.resolve("filesystem", "stark_file_complete").unwrap();
    assert!(
        complete.function.is_close_for.is_none(),
        "complete is not the close function"
    );
    assert!(matches!(
        complete.function.params.as_slice(),
        [AbiParam::HandleBorrowed { .. }]
    ));

    let close = set.resolve("filesystem", "stark_file_close").unwrap();
    assert_eq!(
        close.function.is_close_for.as_deref(),
        Some("file"),
        "close must be declared as THE close for `file`"
    );
    // §13.1: exactly one consumed handle, nothing else. An extra parameter would be one the
    // generated code could not supply.
    assert!(
        matches!(
            close.function.params.as_slice(),
            [AbiParam::HandleConsumed { .. }]
        ),
        "{:#?}",
        close.function.params
    );

    // And the plan agrees: consumed, and not a borrow.
    let p = planned("stark_file_close");
    assert!(matches!(
        p.inputs.as_slice(),
        [ProviderInputPlan::HandleConsumed { .. }]
    ));
    assert!(!p.inputs[0].requires_live_borrow());
    assert!(p.outputs.is_empty(), "close produces no output slot");
}

/// Exactly one close function exists for `file`. ABI §13 requires it, and the validator checks it —
/// this asserts the registry's own copy satisfies the rule rather than trusting the mirror.
#[test]
fn exactly_one_close_is_declared_for_file() {
    let provider = provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == "stark-std-file")
        .expect("stark-file must be registered");

    let closers: Vec<&str> = provider
        .metadata
        .functions
        .iter()
        .filter(|f| f.is_close_for.as_deref() == Some("file"))
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(closers, vec!["stark_file_close"]);

    starkc::provider_abi::validate(&provider.metadata)
        .unwrap_or_else(|v| panic!("stark-file metadata must validate: {v:#?}"));
}

// ------------------------------------------------- read/write shapes --

/// `read` reports bytes read **and** end-of-file separately, so a short read is distinguishable
/// from exhaustion without a sentinel count.
#[test]
fn read_reports_count_and_eof_separately() {
    let call = selected().resolve("filesystem", "stark_file_read").unwrap();
    assert!(
        matches!(
            call.function.params.as_slice(),
            [
                AbiParam::HandleBorrowed { .. },
                AbiParam::BufferInOut,
                AbiParam::ScalarOut(ScalarTy::U64),
                AbiParam::ScalarOut(ScalarTy::Bool),
            ]
        ),
        "{:#?}",
        call.function.params
    );
}

/// `write` reports bytes accepted, which is STD-IO-001's short-write contract reaching the provider
/// boundary intact. Packet 4 kept Core's `write` rather than substituting `write_all`, and this is
/// the primitive that makes that possible.
#[test]
fn write_reports_bytes_accepted() {
    let call = selected()
        .resolve("filesystem", "stark_file_write")
        .unwrap();
    assert!(
        matches!(
            call.function.params.as_slice(),
            [
                AbiParam::HandleBorrowed { .. },
                AbiParam::BufferIn,
                AbiParam::ScalarOut(ScalarTy::U64),
            ]
        ),
        "{:#?}",
        call.function.params
    );
}

// ----------------------------------------------------- status channels --

/// `stark-file` declares eight recoverable codes; `IOError` has five variants. That is not a
/// contradiction — the package binding maps codes onto Core's variants and `Other(String)` absorbs
/// the surplus. What matters here is that all eight are channel one and everything else is not.
#[test]
fn the_declared_file_vocabulary_is_channel_one() {
    let p = planned("stark_file_open");

    assert_eq!(p.classify(STATUS_SUCCESS), StatusOutcome::Success);
    for code in 1u32..=8 {
        assert!(
            matches!(p.classify(code), StatusOutcome::RecoverableError { .. }),
            "code {code} must be recoverable"
        );
    }
    for code in [9u32, 42, u32::MAX] {
        assert_eq!(
            p.classify(code),
            StatusOutcome::ContractViolation { code },
            "undeclared code {code} must be a contract violation"
        );
    }
}

/// A provider declaring a resource type the compiler does not bind is still refused, with the
/// resource type named. C7.8.4 admitted `file` and nothing more.
#[test]
fn an_unbound_resource_type_is_still_refused() {
    let mut call = selected().resolve("filesystem", "stark_file_open").unwrap();
    call.function.params = vec![AbiParam::HandleOut {
        resource_type: "directory".to_string(),
    }];
    call.provider_resource_types = vec!["directory".to_string()];

    assert!(
        matches!(
            plan(
                ProviderCallId(0),
                &call,
                &ResourceRegistry::builtin(),
                call.status_binding.clone()
            ),
            Err(PlanError::UnboundResourceType { .. })
        ),
        "an unbound resource type must still be refused"
    );
}
