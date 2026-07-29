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
    self, AggKind, BasicBlock, Constant, EnumRef, LocalDecl, LocalKind, MirBody, MirProgram, MirTy,
    Operand, Place, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
};
use starkc::source::{SourceFile, Span};
use std::sync::Arc;

fn resource_ty() -> MirTy {
    MirTy::host_resource(ItemId(7), "stark-std-net", "tcp_stream")
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
                    ty: resource_ty(),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    ty: resource_ty(),
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
    let b = emit_types::emit_ty(&MirTy::host_resource(ItemId(9), "stark-std-file", "file"))
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

    let other_nominal = MirTy::host_resource(ItemId(8), "stark-std-net", "tcp_stream");
    let other_provider = MirTy::host_resource(ItemId(7), "other-net", "tcp_stream");
    let other_resource = MirTy::host_resource(ItemId(7), "stark-std-net", "tcp_listener");

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
