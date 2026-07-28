//! WP-C7.8.2d-4 — the resource framework, proven with a **synthetic** resource type.
//!
//! Establishes A10 §4 invariant 7 structurally without claiming `File` is supported. Every case
//! here registers a test-only resource type; the compiler's own registry
//! (`ResourceRegistry::builtin()`) stays empty, so a real program carrying a resource is still
//! refused with MIR-0024. C7.8.4 turns `File` on by *registering* it, not by adding machinery.
//!
//! Invariant 7's semantic rule, which these tests pin:
//!
//! ```text
//! take owned handle from MIR place
//! mark source moved
//! construct RawResourceHandle
//! invoke provider
//! never restore source place
//! ```
//!
//! The shape it rules out is `call; if success { mark moved } else { restore }` — restoring on
//! failure would violate ABI §8 and make exactly-once close unverifiable statically.

use starkc::backend::generated_rust::emit_provider;
use starkc::mir::{self, MirProgram, MirTy, ProviderCallId, TypeContext, ValidatedProviderCall};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ScalarTy};
use starkc::provider_bind::{
    plan, PlanError, ProviderBindingPlan, ProviderInputPlan, ProviderOutputPlan, ResourceRegistry,
    StatusBinding,
};
use starkc::source::SourceFile;
use std::sync::Arc;

const LINUX: &str = "x86_64-unknown-linux-gnu";
/// A test-only resource type. Deliberately not `file`: nothing here should read as `File` support.
const SYNTH: &str = "synthetic-session";

fn synth_registry() -> ResourceRegistry {
    let mut r = ResourceRegistry::builtin();
    // The MIR type a real binding would supply. `UInt64` stands in for whatever C7.8.4 registers.
    r.register(SYNTH, MirTy::UInt64);
    r
}

fn call_with(params: Vec<AbiParam>) -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: "synthetic-provider".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "synthetic".to_string(),
        function: FunctionDecl {
            name: "synth_open".to_string(),
            capability: "synthetic".to_string(),
            params,
            is_close_for: None,
            may_block: false,
        },
        target_triple: LINUX.to_string(),
        provider_resource_types: vec!["other-first".to_string(), SYNTH.to_string()],
        provider_target_triples: vec![LINUX.to_string()],
        status_binding: StatusBinding::new(),
    }
}

fn planned(params: Vec<AbiParam>) -> (ValidatedProviderCall, ProviderBindingPlan) {
    let call = call_with(params);
    let p = plan(
        ProviderCallId(0),
        &call,
        &synth_registry(),
        StatusBinding::new(),
    )
    .expect("a registered resource type must plan");
    (call, p)
}

fn program_for(call: &ValidatedProviderCall) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("a10.stark", ""))],
        bodies: Vec::new(),
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: vec![call.clone()],
    }
}

// ------------------------------------------------------------- the type id --

/// §7: a handle's `resource_type` is "a compiler-assigned index into the provider's declared
/// resource-type list" — **not** a global id and not a provider-chosen tag. The fixture declares
/// `other-first` ahead of the synthetic type precisely so an implementation that returned 0, or a
/// registry index, would fail here.
#[test]
fn the_type_id_is_the_index_into_the_providers_own_list() {
    let (_, p) = planned(vec![AbiParam::HandleOut {
        resource_type: SYNTH.to_string(),
    }]);
    match p.outputs.as_slice() {
        [ProviderOutputPlan::Handle {
            type_id,
            resource_type,
            ..
        }] => {
            assert_eq!(resource_type, SYNTH);
            assert_eq!(*type_id, 1, "SYNTH is the second declared resource type");
        }
        other => panic!("expected one handle output, got {other:#?}"),
    }
}

/// A parameter naming a resource type the provider never declared is a *provider metadata* defect,
/// reported distinctly from the compiler simply lacking a binding — there is no id to assign, so
/// nothing could validate a returned handle.
#[test]
fn a_type_absent_from_the_providers_list_is_its_own_error() {
    let mut call = call_with(vec![AbiParam::HandleBorrowed {
        resource_type: SYNTH.to_string(),
    }]);
    call.provider_resource_types = vec!["something-else".to_string()];

    match plan(
        ProviderCallId(0),
        &call,
        &synth_registry(),
        StatusBinding::new(),
    ) {
        Err(PlanError::UndeclaredResourceType { resource_type, .. }) => {
            assert_eq!(resource_type, SYNTH);
        }
        other => panic!("expected UndeclaredResourceType, got {other:#?}"),
    }
}

// ----------------------------------------- invariant 7: consumed invalidation --

fn emit(call: &ValidatedProviderCall) -> String {
    emit_provider::emit_extern_declarations(&program_for(call)).expect("declarations")
}

/// A consumed handle crosses as `RawResourceHandle` by value, and an out handle as a pointer to
/// one. The resource *type* never appears in the C signature — it travels in the handle's own
/// field and is validated on return (§6.1, §11.1).
#[test]
fn handle_forms_declare_the_raw_abi_types() {
    let (call, _p) = planned(vec![
        AbiParam::HandleConsumed {
            resource_type: SYNTH.to_string(),
        },
        AbiParam::HandleOut {
            resource_type: SYNTH.to_string(),
        },
    ]);
    let src = emit(&call);

    assert!(
        src.contains("a0: stark_runtime::provider_abi::RawResourceHandle"),
        "{src}"
    );
    assert!(
        src.contains("a1: *mut stark_runtime::provider_abi::RawResourceHandle"),
        "{src}"
    );
    assert!(
        !src.contains(SYNTH),
        "the resource type name must not leak into the C signature:\n{src}"
    );
}

/// The plan classifies a consumed handle as an **input** and an out handle as an **output**, so
/// the out handle picks up `MaybeUninit` discipline while the consumed one does not.
#[test]
fn consumed_is_an_input_and_out_is_an_output() {
    let (_, p) = planned(vec![
        AbiParam::HandleConsumed {
            resource_type: SYNTH.to_string(),
        },
        AbiParam::HandleOut {
            resource_type: SYNTH.to_string(),
        },
    ]);

    assert!(matches!(
        p.inputs.as_slice(),
        [ProviderInputPlan::HandleConsumed { index: 0, .. }]
    ));
    assert!(matches!(
        p.outputs.as_slice(),
        [ProviderOutputPlan::Handle { index: 1, .. }]
    ));
    assert!(p.covers(2));
}

/// A borrowed handle carries **no** ownership transfer, so it is an input with a live-borrow
/// obligation — the caller keeps using it after the call (§8's default).
#[test]
fn a_borrowed_handle_keeps_ownership_and_requires_a_live_borrow() {
    let (_, p) = planned(vec![AbiParam::HandleBorrowed {
        resource_type: SYNTH.to_string(),
    }]);
    match p.inputs.as_slice() {
        [input @ ProviderInputPlan::HandleBorrowed { .. }] => {
            assert!(input.requires_live_borrow());
        }
        other => panic!("expected a borrowed handle input, got {other:#?}"),
    }
    assert!(p.outputs.is_empty());
}

/// A consumed handle is **not** a live-borrow obligation: it is gone, not lent.
#[test]
fn a_consumed_handle_is_not_a_borrow() {
    let (_, p) = planned(vec![AbiParam::HandleConsumed {
        resource_type: SYNTH.to_string(),
    }]);
    match p.inputs.as_slice() {
        [input] => assert!(
            !input.requires_live_borrow(),
            "a consumed handle is transferred, not borrowed"
        ),
        other => panic!("expected one input, got {other:#?}"),
    }
}

// ------------------------------------------------- the registry stays empty --

/// **The load-bearing negative.** The compiler's own registry binds nothing, so a real program
/// carrying a resource type is still refused. `File` is not supported by this slice, and nothing
/// here should be read as saying it is.
#[test]
fn the_builtin_registry_admits_no_resource_type() {
    assert!(ResourceRegistry::builtin().is_empty());

    for name in ["file", "File", SYNTH, "custom-db-session"] {
        let mut call = call_with(vec![AbiParam::HandleOut {
            resource_type: name.to_string(),
        }]);
        call.provider_resource_types = vec![name.to_string()];

        match plan(
            ProviderCallId(0),
            &call,
            &ResourceRegistry::builtin(),
            StatusBinding::new(),
        ) {
            Err(PlanError::UnboundResourceType { resource_type, .. }) => {
                assert_eq!(resource_type, name);
            }
            other => panic!("{name} must be unbound under the builtin registry, got {other:#?}"),
        }
    }
}

/// Registering one type admits exactly that one. This is the property that makes MIR-0024 outlive
/// the empty registry: after C7.8.4 binds `file`, an unknown type is still refused.
#[test]
fn registration_is_per_type_not_a_global_switch() {
    let registry = synth_registry();

    let mut unknown = call_with(vec![AbiParam::HandleOut {
        resource_type: "custom-db-session".to_string(),
    }]);
    unknown.provider_resource_types = vec!["custom-db-session".to_string()];

    assert!(matches!(
        plan(ProviderCallId(0), &unknown, &registry, StatusBinding::new()),
        Err(PlanError::UnboundResourceType { .. })
    ));

    // …while the registered one plans.
    let (_, _p) = planned(vec![AbiParam::HandleOut {
        resource_type: SYNTH.to_string(),
    }]);
}

/// Mixed scalar and resource parameters interleave correctly, with each classified independently
/// and every declared index covered exactly once.
#[test]
fn scalars_and_resources_interleave() {
    let (_, p) = planned(vec![
        AbiParam::ScalarIn(ScalarTy::U32),
        AbiParam::HandleConsumed {
            resource_type: SYNTH.to_string(),
        },
        AbiParam::ScalarOut(ScalarTy::U64),
        AbiParam::HandleOut {
            resource_type: SYNTH.to_string(),
        },
    ]);

    assert_eq!(
        p.inputs.iter().map(|i| i.index()).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        p.outputs.iter().map(|o| o.index()).collect::<Vec<_>>(),
        vec![2, 3]
    );
    assert!(p.covers(4));
}
