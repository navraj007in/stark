//! WP-C7.8.2d-1 — the provider binding plan.
//!
//! Proves the structure A10 §4's invariants 6, 8 and 9 will be checked against: every declared
//! parameter is classified, output slots are separated from caller-owned in/out storage, and the
//! status dispatch keeps ABI §12's three channels distinct.
//!
//! No emission yet. These tests are about the plan being *right*, because emission walks it — a
//! parameter dropped here would become a call with a missing argument rather than a compile error.

use starkc::mir::{MirTy, ProviderCallId, ValidatedProviderCall};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ScalarTy};
use starkc::provider_bind::{
    plan, PlanError, ProviderInputPlan, ProviderOutputPlan, ResourceRegistry, StatusBinding,
    StatusOutcome, STATUS_SUCCESS,
};

const LINUX: &str = "x86_64-unknown-linux-gnu";

fn call_with(params: Vec<AbiParam>) -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: "stark-std-time".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "clock".to_string(),
        function: FunctionDecl {
            name: "stark_time_monotonic_now_ns".to_string(),
            capability: "clock".to_string(),
            params,
            is_close_for: None,
            may_block: false,
        },
        target_triple: LINUX.to_string(),
        provider_target_triples: vec![LINUX.to_string()],
    }
}

fn empty_plan(
    params: Vec<AbiParam>,
) -> Result<starkc::provider_bind::ProviderBindingPlan, PlanError> {
    plan(
        ProviderCallId(0),
        &call_with(params),
        &ResourceRegistry::builtin(),
        StatusBinding::new(),
    )
}

// ------------------------------------------------------------ classification --

/// `stark_time_monotonic_now_ns`'s real shape: one `ScalarOut(U64)`, which is an **output**, not an
/// input. This is the case C7.8.2e proves end to end.
#[test]
fn a_scalar_out_is_an_output_not_an_input() {
    let p = empty_plan(vec![AbiParam::ScalarOut(ScalarTy::U64)]).expect("plans");
    assert!(p.inputs.is_empty());
    assert_eq!(
        p.outputs,
        vec![ProviderOutputPlan::Scalar {
            index: 0,
            ty: MirTy::UInt64
        }]
    );
}

/// ABI §11.1 makes `ScalarInOut` and `BufferInOut` caller-initialised and caller-owned, so they are
/// **inputs**. Classifying them as outputs would apply `MaybeUninit` semantics to storage the
/// caller already owns — wrong in both directions: it would forbid a legitimate read after failure
/// and imply the provider allocates storage it does not.
#[test]
fn in_out_forms_are_inputs_never_maybeuninit_outputs() {
    let p = empty_plan(vec![
        AbiParam::ScalarInOut(ScalarTy::I32),
        AbiParam::BufferInOut,
    ])
    .expect("plans");

    assert!(
        p.outputs.is_empty(),
        "in/out storage must never be an output slot: {:#?}",
        p.outputs
    );
    assert_eq!(
        p.inputs,
        vec![
            ProviderInputPlan::ScalarInOut {
                index: 0,
                ty: MirTy::Int32
            },
            ProviderInputPlan::BufferInOut { index: 1 },
        ]
    );
}

/// Every declared parameter produces exactly one plan item, and the indices cover the declaration
/// without gaps or repeats — including when inputs and outputs interleave.
#[test]
fn every_declared_parameter_is_covered_exactly_once() {
    let params = vec![
        AbiParam::ScalarIn(ScalarTy::U32),
        AbiParam::ScalarOut(ScalarTy::U64),
        AbiParam::BufferIn,
        AbiParam::ScalarOut(ScalarTy::Bool),
        AbiParam::BufferInOut,
    ];
    let len = params.len();
    let p = empty_plan(params).expect("plans");

    assert_eq!(p.inputs.len() + p.outputs.len(), len);
    assert!(p.covers(len), "plan does not cover the declaration: {p:#?}");

    // Interleaving is preserved by index, so emission cannot reorder arguments by walking inputs
    // then outputs.
    assert_eq!(
        p.inputs.iter().map(|i| i.index()).collect::<Vec<_>>(),
        vec![0, 2, 4]
    );
    assert_eq!(
        p.outputs.iter().map(|o| o.index()).collect::<Vec<_>>(),
        vec![1, 3]
    );
}

/// A borrow obligation (invariant 6) attaches to exactly the forms that cross as pointers into
/// caller storage. A copied scalar has none.
#[test]
fn borrow_obligations_attach_to_the_pointer_forms_only() {
    let p = empty_plan(vec![
        AbiParam::ScalarIn(ScalarTy::U8),
        AbiParam::ScalarInOut(ScalarTy::U8),
        AbiParam::BufferIn,
        AbiParam::BufferInOut,
    ])
    .expect("plans");

    let requires: Vec<bool> = p.inputs.iter().map(|i| i.requires_live_borrow()).collect();
    assert_eq!(requires, vec![false, true, true, true]);
}

// ------------------------------------------------------------------ resources --

/// The registry is empty in C7.8.2d, so every resource-carrying form is unbound — and the error
/// names the *resource type*, not "resources are unsupported".
#[test]
fn resource_forms_are_unbound_while_the_registry_is_empty() {
    assert!(ResourceRegistry::builtin().is_empty());

    for param in [
        AbiParam::HandleBorrowed {
            resource_type: "file".to_string(),
        },
        AbiParam::HandleConsumed {
            resource_type: "file".to_string(),
        },
        AbiParam::HandleOut {
            resource_type: "file".to_string(),
        },
    ] {
        match empty_plan(vec![param]) {
            Err(PlanError::UnboundResourceType {
                index,
                resource_type,
            }) => {
                assert_eq!(index, 0);
                assert_eq!(resource_type, "file");
            }
            other => panic!("expected UnboundResourceType, got {other:#?}"),
        }
    }
}

/// A **bound** resource type plans successfully — the framework is complete, only the binding is
/// absent. This is what C7.8.4 will turn on by registering `file`, and it is proven now with a
/// synthetic type so that landing `File` is a registration rather than new machinery.
#[test]
fn a_registered_resource_type_plans() {
    let mut registry = ResourceRegistry::builtin();
    registry.register("synthetic-session", MirTy::UInt64);

    let p = plan(
        ProviderCallId(0),
        &call_with(vec![
            AbiParam::HandleConsumed {
                resource_type: "synthetic-session".to_string(),
            },
            AbiParam::HandleOut {
                resource_type: "synthetic-session".to_string(),
            },
        ]),
        &registry,
        StatusBinding::new(),
    )
    .expect("a registered resource type must plan");

    assert_eq!(
        p.inputs,
        vec![ProviderInputPlan::HandleConsumed {
            index: 0,
            resource_type: "synthetic-session".to_string(),
            mir_type: MirTy::UInt64,
        }]
    );
    assert_eq!(
        p.outputs,
        vec![ProviderOutputPlan::Handle {
            index: 1,
            resource_type: "synthetic-session".to_string(),
            mir_type: MirTy::UInt64,
        }]
    );
}

/// Registering one resource type does not admit another. MIR-0024 outlives the empty registry:
/// after C7.8.4 binds `file`, an unknown `custom-db-session` is still inadmissible.
#[test]
fn an_unregistered_type_stays_unbound_after_others_are_registered() {
    let mut registry = ResourceRegistry::builtin();
    registry.register("file", MirTy::UInt64);

    let err = plan(
        ProviderCallId(0),
        &call_with(vec![AbiParam::HandleBorrowed {
            resource_type: "custom-db-session".to_string(),
        }]),
        &registry,
        StatusBinding::new(),
    )
    .expect_err("an unregistered type must stay unbound");

    assert_eq!(
        err,
        PlanError::UnboundResourceType {
            index: 0,
            resource_type: "custom-db-session".to_string()
        }
    );
}

// --------------------------------------------------------- channel discipline --

/// ABI §12's three channels, kept structurally distinct. An **undeclared** nonzero status is a
/// contract violation, never a recoverable error — the failure mode this defends is a provider and
/// its package drifting apart while staying physically ABI-compatible.
#[test]
fn status_dispatch_keeps_the_three_channels_distinct() {
    let mut status = StatusBinding::new();
    status.declare(1, "IOError::NotFound");
    status.declare(2, "IOError::PermissionDenied");

    let p = plan(
        ProviderCallId(0),
        &call_with(vec![AbiParam::ScalarOut(ScalarTy::U64)]),
        &ResourceRegistry::builtin(),
        status,
    )
    .expect("plans");

    assert_eq!(p.classify(STATUS_SUCCESS), StatusOutcome::Success);
    assert_eq!(
        p.classify(1),
        StatusOutcome::RecoverableError {
            code: 1,
            package_error: "IOError::NotFound".to_string()
        }
    );
    assert_eq!(
        p.classify(2),
        StatusOutcome::RecoverableError {
            code: 2,
            package_error: "IOError::PermissionDenied".to_string()
        }
    );
    // Not declared -> channel two. NOT a generic `Other`.
    assert_eq!(p.classify(3), StatusOutcome::ContractViolation { code: 3 });
    assert_eq!(
        p.classify(u32::MAX),
        StatusOutcome::ContractViolation { code: u32::MAX }
    );
}

/// An empty status vocabulary is legal and says something precise: **every** nonzero status from
/// this provider is a contract violation. `stark-time` is exactly this case — neither of its
/// functions declares a recoverable error.
#[test]
fn no_declared_codes_means_every_nonzero_status_is_a_violation() {
    let p = empty_plan(vec![AbiParam::ScalarOut(ScalarTy::U64)]).expect("plans");
    assert!(p.status.is_empty());

    assert_eq!(p.classify(STATUS_SUCCESS), StatusOutcome::Success);
    for code in [1u32, 2, 42, u32::MAX] {
        assert_eq!(p.classify(code), StatusOutcome::ContractViolation { code });
    }
}

/// Declared codes iterate in **ascending numeric order**, regardless of declaration order.
///
/// This is a reproducibility property, not a tidiness one: C7.8.2d-3 generates one match arm per
/// declared code, and arms emitted in declaration order would make the generated Rust — and so the
/// produced binary — depend on the order a package happened to declare its errors. Gate C7.2
/// classified reproducibility per artefact and profile; this is one of the inputs that has to hold
/// for that classification to keep meaning anything.
#[test]
fn declared_codes_iterate_in_deterministic_order() {
    let mut a = StatusBinding::new();
    a.declare(7, "IOError::TimedOut");
    a.declare(1, "IOError::NotFound");
    a.declare(3, "IOError::PermissionDenied");

    let mut b = StatusBinding::new();
    b.declare(3, "IOError::PermissionDenied");
    b.declare(7, "IOError::TimedOut");
    b.declare(1, "IOError::NotFound");

    let order_a: Vec<(u32, String)> = a.declared_codes().map(|(c, e)| (*c, e.clone())).collect();
    let order_b: Vec<(u32, String)> = b.declared_codes().map(|(c, e)| (*c, e.clone())).collect();

    assert_eq!(
        order_a, order_b,
        "declaration order must not survive into iteration"
    );
    assert_eq!(
        order_a.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
        vec![1, 3, 7]
    );
}

/// Success is status zero, and nothing else is. Guards against a "nonzero means failure" check
/// drifting into "negative means failure" or similar.
#[test]
fn success_is_exactly_status_zero() {
    assert_eq!(STATUS_SUCCESS, 0);
    let mut status = StatusBinding::new();
    status.declare(0, "IOError::NotFound");

    let p = plan(
        ProviderCallId(0),
        &call_with(vec![AbiParam::ScalarOut(ScalarTy::U64)]),
        &ResourceRegistry::builtin(),
        status,
    )
    .expect("plans");

    // Even a package that mistakenly declares code 0 cannot turn success into an error: the
    // success test precedes the declared-code lookup.
    assert_eq!(p.classify(0), StatusOutcome::Success);
}
