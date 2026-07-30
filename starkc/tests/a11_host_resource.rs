//! WP-C7.8.8 / A11 — `MirTy::HostResource`, and the CD-234 rule that **nothing may manufacture one**.
//!
//! CD-234 approved a synthesized **zero-variant enum** as the source-level nominal for a host
//! resource, subject to one condition: the nominal supplies source identity only, and a
//! provider-bound instance of it must never receive the backend representation or
//! default-initialisation behaviour of an ordinary zero-variant enum.
//!
//! The architectural split this pins:
//!
//! ```text
//! zero-variant enum = front-end nominal shell
//! HostResource      = MIR ownership and native representation
//! ```
//!
//! **Why the tests are shaped as refusals.** A forged handle is undetectable at the boundary:
//! `from_raw_checked` compares a handle's `resource_type` against the provider's declared list, and a
//! fabricated handle's `resource_type` is whatever the fabricator wrote. So the guarantee has to be
//! that no fabricated handle exists, and each test below is a way one could have come into being.

use starkc::backend::generated_rust::emit_types;
use starkc::hir::ItemId;
use starkc::mir::{
    self, AggKind, BasicBlock, Callee, Constant, EnumRef, LocalDecl, LocalKind, MirBody,
    MirProgram, MirTy, Operand, Place, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
};
use starkc::source::{SourceFile, Span};
use std::sync::Arc;

fn resource_ty() -> MirTy {
    MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(7)),
        "stark-std-net",
        "tcp_stream",
    )
}

fn info() -> SourceInfo {
    SourceInfo {
        file: mir::FileId(0),
        span: Span { lo: 0, hi: 0 },
        origin: mir::Origin::UserCode,
    }
}

fn place(i: u32) -> Place {
    Place {
        local: mir::LocalId(i),
        projection: Vec::new(),
    }
}

/// A one-block body whose second local is a host resource, assigned by `rvalue`.
fn body_assigning(rvalue: Rvalue) -> MirProgram {
    body_assigning_ty(rvalue, resource_ty())
}

/// `body_assigning`, with the local's type chosen — so the CD-235 Core-nominal case can be built.
fn body_assigning_ty(rvalue: Rvalue, ty: MirTy) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("r.stark", ""))],
        bodies: vec![MirBody {
            instance: mir::Instance {
                item: ItemId(0),
                type_args: Vec::new(),
                symbol: "main@[]".to_string(),
            },
            params: Vec::new(),
            ret: MirTy::Unit,
            locals: vec![
                LocalDecl {
                    ty: MirTy::Unit,
                    kind: LocalKind::Return,
                },
                LocalDecl {
                    ty: ty.clone(),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    ty,
                    kind: LocalKind::Temp,
                },
            ],
            blocks: vec![BasicBlock {
                statements: vec![(Statement::Assign(place(1), rvalue), info())],
                terminator: (Terminator::Return, info()),
            }],
            entry: mir::BlockId(0),
        }],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        resource_bindings: Vec::new(),
        provider_closes: Vec::new(),
        provider_calls: Vec::new(),
    }
}

fn verify_codes(program: &MirProgram) -> Vec<String> {
    match mir::verify::verify_program(program) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.iter().map(|e| e.code.to_string()).collect(),
    }
}

// ------------------------------------------------------- the version increment --

/// **A11 §3: `MIR_VERSION` is `0.2`.** A `MirTy` variant flows through every part of the compiler
/// that reasons about types, unlike A10's `Callee` variant which fails at one match site — so this is
/// a shape change, not a surface revision.
///
/// The increment invalidates every build key, which is the intent: a key that ignored a
/// representation change would serve a cached artifact built under different type rules.
#[test]
fn mir_version_is_0_2_for_a11() {
    assert_eq!(mir::MIR_VERSION, "0.2");
    // `MIR_RUNTIME_SURFACE` deliberately does NOT move: A11 adds no `RuntimeFn`, because a close is a
    // provider call through MIR's `Drop` terminator.
    assert_eq!(mir::MIR_RUNTIME_SURFACE, "0.1-A10");
}

// ------------------------------------------------- nothing may manufacture one --

/// **MIR-0026: an `Aggregate` may not produce a host resource.**
///
/// This is the direct forgery route. An aggregate targeting a resource type would build a handle out
/// of operands the program chose, and the ABI could not tell it from a real one.
#[test]
fn an_aggregate_cannot_manufacture_a_host_resource() {
    let program = body_assigning(Rvalue::Aggregate(AggKind::Tuple, Vec::new()));
    assert!(
        verify_codes(&program).contains(&"MIR-0026".to_string()),
        "an aggregate assigned to a host resource must be rejected"
    );
}

/// An enum-variant aggregate is refused for the same reason — and this one matters most, because the
/// SOURCE nominal really is an enum (a zero-variant one). If the enum's own construction machinery
/// could reach the resource type, CD-234's separation would be decorative.
#[test]
fn an_enum_variant_aggregate_cannot_manufacture_a_host_resource() {
    let program = body_assigning(Rvalue::Aggregate(
        AggKind::EnumVariant(EnumRef::User(ItemId(7)), 0),
        Vec::new(),
    ));
    assert!(
        verify_codes(&program).contains(&"MIR-0026".to_string()),
        "an enum-variant aggregate must not produce a host resource"
    );
}

/// A constant may not produce one.
#[test]
fn a_constant_cannot_manufacture_a_host_resource() {
    let program = body_assigning(Rvalue::Use(Operand::Const(Constant::Int(0, MirTy::UInt64))));
    assert!(
        verify_codes(&program).contains(&"MIR-0026".to_string()),
        "a constant must not produce a host resource"
    );
}

/// A discriminant read may not produce one.
#[test]
fn a_discriminant_cannot_manufacture_a_host_resource() {
    let program = body_assigning(Rvalue::Discriminant(place(2)));
    assert!(
        verify_codes(&program).contains(&"MIR-0026".to_string()),
        "a discriminant must not produce a host resource"
    );
}

/// **A COPY may not produce one either**, which is the case most easily mistaken for benign.
///
/// Duplicating a handle would give two owners of one resource and close it twice — exactly the
/// exactly-once guarantee the MIR `Drop` terminator exists to hold. A host resource is not `Copy`.
#[test]
fn a_copy_cannot_duplicate_a_host_resource() {
    let program = body_assigning(Rvalue::Use(Operand::Copy(place(2))));
    assert!(
        verify_codes(&program).contains(&"MIR-0026".to_string()),
        "copying a host resource must be rejected: two owners would close it twice"
    );
}

/// A borrow may not produce one (the type is the resource, not a reference to it).
#[test]
fn a_borrow_cannot_manufacture_a_host_resource() {
    let program = body_assigning(Rvalue::RefOf {
        mutable: false,
        place: place(2),
    });
    assert!(
        verify_codes(&program).contains(&"MIR-0026".to_string()),
        "a borrow must not produce a host resource"
    );
}

/// **A move IS admitted** — the rule is targeted, not a blanket ban. Without this the type would be
/// unusable rather than protected, and the test suite would not distinguish the two.
#[test]
fn a_move_is_the_one_admitted_rvalue() {
    let program = body_assigning(Rvalue::Use(Operand::Move(place(2))));
    assert!(
        !verify_codes(&program).contains(&"MIR-0026".to_string()),
        "moving an already-live resource must be allowed"
    );
}

// --------------------------------------- no default, no eager materialisation --

/// **CD-234's central backend requirement: a host resource has no default value.**
///
/// The `FnPtr` sentinel and the uninhabited-enum placeholder both answer "this local needs a starting
/// value" by inventing one. A host resource must not: an invented `OwnedResourceHandle` *is* a forged
/// handle. So emission refuses rather than satisfies the request.
///
/// The refusal is structural and does not rest on drop flags making it unreachable — CD-234 is
/// explicit that a placeholder-backed host-resource local is forbidden even if current drop flags
/// appear to rule it out.
#[test]
fn a_host_resource_has_no_default_value() {
    let types = TypeContext::default();
    let err = emit_types::default_value_expr(&resource_ty(), &types)
        .expect_err("a host resource must have no default value");
    let message = format!("{err:?}");
    assert!(message.contains("tcp_stream"), "{message}");
    assert!(
        message.contains("forged") || message.contains("eagerly materialised"),
        "the diagnostic must say why, not merely that it is unsupported: {message}"
    );
}

/// Codegen emits **`OwnedResourceHandle` for every host resource, whatever its nominal** (A11 §Q6).
///
/// The nominal distinction is a STARK type-system distinction, enforced before emission. At the ABI
/// boundary all handles are one shape, and their runtime distinction is the `resource_type` field
/// `from_raw_checked` validates. Static distinctness in STARK, dynamic validation at the boundary.
#[test]
fn every_host_resource_emits_as_an_owned_handle() {
    let a = emit_types::emit_ty(&resource_ty()).expect("emits");
    let b = emit_types::emit_ty(&MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(9)),
        "stark-std-file",
        "file",
    ))
    .expect("emits");

    assert_eq!(a, "stark_runtime::provider_abi::OwnedResourceHandle");
    assert_eq!(a, b, "codegen type selection must not consult the nominal");
}

/// **No generated Rust contains a placeholder resource handle.** The uninhabited-enum placeholder
/// path exists for ORDINARY zero-variant enums and must be unreachable for a resource — which holds
/// structurally, because a provider-bound nominal lowers to `HostResource` and never reaches the enum
/// machinery at all.
#[test]
fn the_uninhabited_enum_placeholder_never_backs_a_resource() {
    let types = TypeContext::default();

    // The ordinary uninhabited enum keeps its placeholder: this is the path that must remain.
    let ordinary = MirTy::Enum(EnumRef::User(ItemId(3)), Vec::new());
    // With no variant table it is not defaultable at all, which is a separate (older) refusal; the
    // property under test is that the RESOURCE path never produces a handle expression.
    let _ = emit_types::default_value_expr(&ordinary, &types);

    let resource = emit_types::default_value_expr(&resource_ty(), &types);
    assert!(
        resource.is_err(),
        "a resource must never acquire a default expression"
    );
    if let Ok(expr) = resource {
        assert!(
            !expr.contains("OwnedResourceHandle"),
            "generated Rust must contain no placeholder resource handle, got {expr}"
        );
    }
}

// ------------------------------------------------------- canonical identity --

/// A11 §Q5: two nominals bound to the same provider resource are **different types**, and the same
/// nominal through different providers is too. A11 §7's negative cases turn on telling those apart,
/// so structural equality over `(nominal, provider, resource)` is load-bearing.
#[test]
fn identity_is_structural_over_all_three_fields() {
    let base = resource_ty();

    let nom7 = mir::HostResourceNominal::Item(ItemId(7));
    let other_nominal = MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(8)),
        "stark-std-net",
        "tcp_stream",
    );
    let other_provider = MirTy::host_resource(nom7, "other-net", "tcp_stream");
    let other_resource = MirTy::host_resource(nom7, "stark-std-net", "tcp_listener");

    assert_ne!(base, other_nominal);
    assert_ne!(base, other_provider);
    assert_ne!(base, other_resource);
    assert_eq!(base, resource_ty());
}

/// The dump carries all three identities, so a MIR dump of a resource-bearing program is readable and
/// diffable rather than opaque.
#[test]
fn the_dump_carries_provider_resource_and_nominal() {
    let program = body_assigning(Rvalue::Use(Operand::Move(place(2))));
    let dumped = program.dump();
    assert!(
        dumped.contains("hostres#stark-std-net/tcp_stream"),
        "{dumped}"
    );
    assert!(
        dumped.contains("0.2"),
        "the dump must record MIR 0.2: {dumped}"
    );
}

// ----------------------------------- CD-235: the Core sequencing exception --

/// **MIR-0027: a Core nominal must not reach `HostResource` before its migration.**
///
/// Core `File` is deliberately still on `MirTy::Core(CoreType::File, [])` — the qualified path behind
/// C7.8.4's evidence. A `HostResource` naming a Core nominal would give one Core resource two
/// representations inside one program: two drop-close paths for one kind of handle, and the first
/// consumer to pick the other one closes twice.
///
/// This is the guard that makes CD-235's sequencing exception *safe* rather than merely documented.
/// It is removed by the migration step that requalifies the File evidence — not before.
#[test]
fn a_core_nominal_cannot_reach_host_resource_yet() {
    let core_resource = MirTy::host_resource(
        mir::HostResourceNominal::Core(starkc::hir::CoreType::File),
        "stark-std-file",
        "file",
    );
    let program = body_assigning_ty(Rvalue::Use(Operand::Move(place(2))), core_resource);
    assert!(
        verify_codes(&program).contains(&"MIR-0027".to_string()),
        "a Core-nominal host resource must be rejected until the migration lands"
    );
}

/// The registry keeps `file` on the legacy path, which is what CD-235 requires. If this flips
/// without the requalification, `a_core_nominal_cannot_reach_host_resource_yet` is the net that
/// catches it — but this test names the intent directly.
#[test]
fn the_registry_keeps_core_file_on_the_legacy_path() {
    let registry = starkc::provider_bind::ResourceRegistry::builtin();
    assert_eq!(
        registry.lookup("file"),
        Some(&starkc::provider_bind::ResourceBinding::LegacyCore(
            starkc::hir::CoreType::File
        )),
        "Core File must stay on MirTy::Core until its migration is requalified (CD-235)"
    );
    assert!(
        registry.partially_migrated_core().is_none(),
        "no Core type may be bound both ways in one program"
    );
}

/// A package nominal, by contrast, uses `HostResource` immediately — the whole point of CD-235's
/// split. Without this the previous two tests would be consistent with nothing working at all.
#[test]
fn a_package_nominal_uses_host_resource_immediately() {
    let mut registry = starkc::provider_bind::ResourceRegistry::builtin();
    registry.register_nominal("tcp_stream", ItemId(7));
    assert_eq!(
        registry.lookup("tcp_stream"),
        Some(&starkc::provider_bind::ResourceBinding::Nominal(ItemId(7)))
    );
}

// ------------------------------------------ A11 §5: the close, exactly once --

use starkc::mir::{ValidatedProviderCall, ValidatedProviderClose};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity};

fn close_decl(name: &str, is_close_for: Option<&str>, params: Vec<AbiParam>) -> FunctionDecl {
    FunctionDecl {
        name: name.to_string(),
        capability: "tcp".to_string(),
        params,
        is_close_for: is_close_for.map(|s| s.to_string()),
        may_block: false,
    }
}

fn call_for(decl: FunctionDecl, provider: &str) -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: provider.to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "tcp".to_string(),
        function: decl,
        target_triple: "aarch64-apple-darwin".to_string(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        provider_crate: "stark-net-native".to_string(),
        provider_resource_types: vec!["tcp_listener".to_string(), "tcp_stream".to_string()],
        provider_target_triples: vec!["aarch64-apple-darwin".to_string()],
    }
}

/// A program with the given close arena, and no bodies — the obligations are program-level.
fn program_with_closes(
    calls: Vec<ValidatedProviderCall>,
    closes: Vec<ValidatedProviderClose>,
) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("c.stark", ""))],
        bodies: Vec::new(),
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        resource_bindings: Vec::new(),
        provider_closes: closes,
        provider_calls: calls,
    }
}

/// **Obligation 4, the one a structural check misses.**
///
/// `stark_tcp_listener_close` and `stark_tcp_stream_close` have *identical shapes* — both consume one
/// handle — and differ only in which resource they name. So a listener closed by the stream's close
/// typechecks perfectly, and only the `is_close_for` comparison catches it.
#[test]
fn a_stream_close_cannot_close_a_listener() {
    let stream_close = call_for(
        close_decl(
            "stark_tcp_stream_close",
            Some("tcp_stream"),
            vec![AbiParam::HandleConsumed {
                resource_type: "tcp_stream".to_string(),
            }],
        ),
        "stark-std-net",
    );
    let listener = MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(5)),
        "stark-std-net",
        "tcp_listener",
    );

    let program = program_with_closes(
        vec![stream_close],
        vec![ValidatedProviderClose {
            resource: listener,
            close: mir::ProviderCallId(0),
        }],
    );
    let codes = verify_codes(&program);
    assert!(
        codes.contains(&"MIR-0030".to_string()),
        "the stream's close must not be accepted for a listener; got {codes:?}"
    );
}

/// Obligation 1: two closes for one resource are two destruction paths, and whichever ran second
/// would close an already-closed handle.
#[test]
fn two_closes_for_one_resource_are_rejected() {
    let decl = close_decl(
        "stark_tcp_stream_close",
        Some("tcp_stream"),
        vec![AbiParam::HandleConsumed {
            resource_type: "tcp_stream".to_string(),
        }],
    );
    let stream = MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(6)),
        "stark-std-net",
        "tcp_stream",
    );
    let program = program_with_closes(
        vec![
            call_for(decl.clone(), "stark-std-net"),
            call_for(decl, "stark-std-net"),
        ],
        vec![
            ValidatedProviderClose {
                resource: stream.clone(),
                close: mir::ProviderCallId(0),
            },
            ValidatedProviderClose {
                resource: stream,
                close: mir::ProviderCallId(1),
            },
        ],
    );
    assert!(
        verify_codes(&program).contains(&"MIR-0028".to_string()),
        "exactly one close per resource (A11 §5 obligation 1)"
    );
}

/// Obligation 3: a close from a different provider would receive a handle that provider never issued.
#[test]
fn a_close_from_another_provider_is_rejected() {
    let program = program_with_closes(
        vec![call_for(
            close_decl(
                "other_close",
                Some("tcp_stream"),
                vec![AbiParam::HandleConsumed {
                    resource_type: "tcp_stream".to_string(),
                }],
            ),
            "some-other-provider",
        )],
        vec![ValidatedProviderClose {
            resource: MirTy::host_resource(
                mir::HostResourceNominal::Item(ItemId(6)),
                "stark-std-net",
                "tcp_stream",
            ),
            close: mir::ProviderCallId(0),
        }],
    );
    assert!(
        verify_codes(&program).contains(&"MIR-0031".to_string()),
        "a close must belong to the provider that issued the handle"
    );
}

/// ABI §13.1: exactly one consumed handle and no ordinary output. A close taking extra outputs would
/// return a value derived from a resource it has just destroyed.
#[test]
fn a_close_with_the_wrong_parameter_list_is_rejected() {
    let program = program_with_closes(
        vec![call_for(
            close_decl(
                "stark_tcp_stream_close",
                Some("tcp_stream"),
                vec![
                    AbiParam::HandleConsumed {
                        resource_type: "tcp_stream".to_string(),
                    },
                    AbiParam::ScalarOut(starkc::provider_abi::ScalarTy::U64),
                ],
            ),
            "stark-std-net",
        )],
        vec![ValidatedProviderClose {
            resource: MirTy::host_resource(
                mir::HostResourceNominal::Item(ItemId(6)),
                "stark-std-net",
                "tcp_stream",
            ),
            close: mir::ProviderCallId(0),
        }],
    );
    assert!(
        verify_codes(&program).contains(&"MIR-0032".to_string()),
        "a close takes exactly one HandleConsumed and no value output (ABI §13.1)"
    );
}

/// A correctly-paired close verifies. Without this the four refusals above would be consistent with
/// rejecting everything.
#[test]
fn a_correctly_paired_close_is_accepted() {
    let program = program_with_closes(
        vec![call_for(
            close_decl(
                "stark_tcp_stream_close",
                Some("tcp_stream"),
                vec![AbiParam::HandleConsumed {
                    resource_type: "tcp_stream".to_string(),
                }],
            ),
            "stark-std-net",
        )],
        vec![ValidatedProviderClose {
            resource: MirTy::host_resource(
                mir::HostResourceNominal::Item(ItemId(6)),
                "stark-std-net",
                "tcp_stream",
            ),
            close: mir::ProviderCallId(0),
        }],
    );
    let codes = verify_codes(&program);
    assert!(
        !codes.iter().any(|c| c.starts_with("MIR-003")),
        "a correctly paired close must verify; got {codes:?}"
    );
}

/// **A11 §5 obligation 5: a resource with no close cannot be dropped.**
///
/// Planning `Noop` would be the leak itself — the provider never learns the handle was abandoned, so
/// nothing downstream could detect it. It fails at planning instead.
#[test]
fn dropping_a_resource_with_no_recorded_close_fails() {
    let types = TypeContext::default();
    let err = mir::drop_plan::plan_for(&resource_ty(), &types)
        .expect_err("a resource with no close must not plan");
    let message = format!("{err:?}");
    assert!(message.contains("tcp_stream"), "{message}");
    assert!(
        message.contains("no recorded close"),
        "the diagnostic must name the cause: {message}"
    );
}

/// With a close recorded, the plan is that close — not a destructor and not runtime glue.
#[test]
fn a_recorded_close_becomes_the_drop_plan() {
    let mut types = TypeContext::default();
    types
        .host_resource_closes
        .insert(resource_ty(), mir::ProviderCallId(3));
    let plan = mir::drop_plan::plan_for(&resource_ty(), &types).expect("plans");
    assert_eq!(plan.host_resource_close(), Some(mir::ProviderCallId(3)));
}

// --------------------------- CD-240: never Copy, therefore slot-backed --

/// **A host resource is never `Copy`, and this is a regression guard for a wildcard.**
///
/// `TypeContext::is_copy` ends in `_ => true`, so adding `MirTy::HostResource` silently classified it
/// `Copy` — with three consequences, none of which announced themselves:
///
/// 1. `is_slot_backed` became false, so the local was declared through `default_value_expr`, which
///    refuses a resource — emission failed before `Drop` was ever reached;
/// 2. `emit_drop` refuses a `Copy` type outright, so the close could not have run either;
/// 3. `Copy` is the licence to **duplicate** a handle, giving two owners of one resource and closing
///    it twice.
///
/// The arm is now explicit. This test exists because the defect produced **zero compile errors** —
/// the type checker cannot notice a new variant falling into a catch-all, so only an assertion can.
#[test]
fn a_host_resource_is_never_copy() {
    let types = TypeContext::default();
    assert!(
        !types.is_copy(&resource_ty()),
        "a host resource must never be Copy"
    );
    assert!(
        !types.is_copy(&MirTy::host_resource(
            mir::HostResourceNominal::Core(starkc::hir::CoreType::File),
            "stark-std-file",
            "file"
        )),
        "the Core nominal form must not be Copy either"
    );
}

/// Being non-`Copy` is what makes a resource **slot-backed**, and a slot is what gives it
/// `ValueSlot::dead()`. CD-234's "the slot begins dead" is then the representation itself, rather
/// than a rule some pass has to remember to apply.
#[test]
fn a_host_resource_is_slot_backed() {
    let types = TypeContext::default();
    assert!(
        starkc::backend::generated_rust::emit_types::is_slot_backed(&resource_ty(), &types),
        "a host resource must be slot-backed: that is what lets it be declared dead with no default"
    );
}

/// The two properties are not independent — the slot rule is *derived* from non-`Copy`. Pinning the
/// derivation means a future change to either one cannot quietly decouple them.
#[test]
fn slot_backing_follows_from_not_being_copy() {
    let types = TypeContext::default();
    let ty = resource_ty();
    assert_eq!(
        starkc::backend::generated_rust::emit_types::is_slot_backed(&ty, &types),
        !types.is_copy(&ty) && !matches!(ty, MirTy::Ref { .. }),
        "slot backing must remain the stated function of Copy-ness"
    );
}

/// **Rule 4 applies only to resources actually on the A11 path (CD-235).**
///
/// A close for a resource still on the legacy `MirTy::Core` representation stays directly callable:
/// no `Drop` terminator will ever close it, so forbidding the direct call would leave it forbidden
/// one way and unreachable the other. That is exactly what broke `c784_file_e2e` when rule 4 was
/// applied globally.
#[test]
fn a_legacy_core_close_may_still_be_called_directly() {
    let close = call_for(
        close_decl(
            "stark_file_close",
            Some("file"),
            vec![AbiParam::HandleConsumed {
                resource_type: "file".to_string(),
            }],
        ),
        "stark-std-file",
    );
    // No `provider_closes` entry for `file`: it is not on the A11 path.
    let program = program_calling_provider(vec![close], Vec::new());
    assert!(
        !verify_codes(&program).contains(&"MIR-0033".to_string()),
        "a legacy Core resource's close must remain directly callable until it migrates"
    );
}

/// **And the guard tightens automatically.** The same direct call becomes a violation the moment the
/// resource has a `HostResource` close binding — nothing has to be remembered and deleted at
/// migration time, which is why this is derived from the program rather than from a named exemption.
#[test]
fn the_same_close_becomes_a_violation_once_the_resource_is_migrated() {
    let close = call_for(
        close_decl(
            "stark_file_close",
            Some("file"),
            vec![AbiParam::HandleConsumed {
                resource_type: "file".to_string(),
            }],
        ),
        "stark-std-file",
    );
    let migrated = MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(11)),
        "stark-std-file",
        "file",
    );
    let program = program_calling_provider(
        vec![close],
        vec![ValidatedProviderClose {
            resource: migrated,
            close: mir::ProviderCallId(0),
        }],
    );
    assert!(
        verify_codes(&program).contains(&"MIR-0033".to_string()),
        "once a resource is on the A11 path, MIR owns the only close and a direct call is a second \
         destruction path"
    );
}

/// A body whose sole terminator is a direct `Callee::Provider` call to `calls[0]`.
fn program_calling_provider(
    calls: Vec<ValidatedProviderCall>,
    closes: Vec<ValidatedProviderClose>,
) -> MirProgram {
    let mut program = program_with_closes(calls, closes);
    program.bodies = vec![MirBody {
        instance: mir::Instance {
            item: ItemId(0),
            type_args: Vec::new(),
            symbol: "main@[]".to_string(),
        },
        params: Vec::new(),
        ret: MirTy::Unit,
        locals: vec![LocalDecl {
            ty: MirTy::Unit,
            kind: LocalKind::Return,
        }],
        blocks: vec![
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(mir::ProviderCallId(0)),
                        args: Vec::new(),
                        dest: place(0),
                        target: mir::BlockId(1),
                    },
                    info(),
                ),
            },
            BasicBlock {
                statements: Vec::new(),
                terminator: (Terminator::Return, info()),
            },
        ],
        entry: mir::BlockId(0),
    }];
    program
}
