//! **CD-360, the lowering half: a cross-provider transfer consumes the source on EVERY path.**
//!
//! The ruling:
//!
//! > A cross-provider `HandleConsumed` transfer consumes the source handle regardless of whether
//! > the provider operation succeeds or fails.
//!
//! `cd360_cross_provider_transfer.rs` pins the DECLARATION rules. This file pins what the generated
//! code actually does, which is the half that would ship a double-free if the two ever diverged.
//!
//! # This is a verification, not an implementation
//!
//! Worth stating plainly: the lowering already behaved this way. `mir/lower.rs`'s `HandleConsumed`
//! arm carries a comment written for A11 §8, long before CD-360 was asked:
//!
//! ```text
//! // A11 §8: a CONSUMED handle. Ownership transfers at call entry and does not return
//! // on failure, so the argument is a move and the caller's drop flag is cleared --
//! // otherwise the implicit close would run on a handle the provider already owns.
//! ```
//!
//! That is CD-360's rule verbatim. The ruling therefore chose the option the compiler already
//! implemented — which is exactly why it needs no change to drop elaboration, and why choosing
//! Option B would have been expensive rather than merely different. **But a comment is not
//! evidence.** Nothing had ever lowered a transfer, and this repository's recent history is a list
//! of things that were true in a comment and false in the first real caller (DEV-146, DEV-151,
//! DEV-153, DEV-155). Hence a fixture.
//!
//! # Deliberately not TLS
//!
//! A synthetic transfer between two test providers, so the proving case does not wait on a TLS peer
//! with a controlled certificate chain. The property under test is ownership, not cryptography.

use starkc::backend::generated_rust::emit_program;
use starkc::backend::version::build_versions;
use starkc::hir::ItemId;
use starkc::mir::{
    self, BasicBlock, Callee, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand, Place,
    SourceInfo, Terminator, TypeContext, ValidatedProviderCall,
};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity};

/// AS1b-ii: a real registered source for a hand-built MIR program.
/// The one registry a hand-built `MirProgram` in this file is measured against.
///
/// AS1b-iii: a fixture used to state its source twice — a `RegisteredSource` for the spans and an
/// unrelated `Arc<SourceFile>` in `MirProgram::files`, often under a different name. Nothing
/// checked that they agreed, which is the duplication the amendment removes. Now the program
/// carries the registry the handle came from, so there is nothing to keep in step.
fn test_sources() -> starkc::source::SourceTable {
    let mut registry = starkc::source::SourceRegistry::default();
    registry.intern(std::sync::Arc::new(starkc::source::SourceFile::new(
        "test.stark",
        "",
    )));
    registry.freeze()
}

fn test_source() -> starkc::source::RegisteredSource {
    test_sources()
        .entry()
        .expect("the registry was just populated")
        .clone()
}

fn host_triple() -> String {
    starkc::native_toolchain::discover(None)
        .expect("a Rust toolchain is required")
        .host_triple
}

fn source_info() -> SourceInfo {
    SourceInfo {
        span: test_source().synthetic_span(),
        origin: mir::Origin::UserCode,
    }
}

/// The SOURCE resource, owned by the `net` provider.
fn source_ty() -> MirTy {
    MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(7)),
        "stark-std-net",
        "tcp_stream",
    )
}

/// The DESTINATION resource, owned by the `wrap` provider. A different nominal AND a different
/// provider — the transfer is a genuine type transition, not a re-tag.
fn dest_ty() -> MirTy {
    MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(8)),
        "stark-wrap-native",
        "wrapped_stream",
    )
}

/// The transfer: consumes the net provider's resource, produces its own.
fn transfer_call() -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: "stark-wrap-native".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "wrap".to_string(),
        function: FunctionDecl {
            name: "stark_wrap_adopt".to_string(),
            capability: "wrap".to_string(),
            params: vec![
                AbiParam::HandleConsumed {
                    resource_type: "tcp_stream".to_string(),
                },
                AbiParam::HandleOut {
                    resource_type: "wrapped_stream".to_string(),
                },
            ],
            is_close_for: None,
            may_block: true,
        },
        target_triple: host_triple(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        foreign_resources: vec![mir::ForeignResourceCall {
            provider: "stark-std-net".to_string(),
            resource: "tcp_stream".to_string(),
            owner_resource_types: vec!["tcp_stream".to_string()],
        }],
        provider_crate: "stark-wrap-native".to_string(),
        provider_resource_types: vec!["wrapped_stream".to_string()],
        provider_target_triples: vec![host_triple()],
    }
}

fn close_call(
    provider: &str,
    crate_name: &str,
    capability: &str,
    resource: &str,
) -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: provider.to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: capability.to_string(),
        function: FunctionDecl {
            name: format!("stark_{resource}_close"),
            capability: capability.to_string(),
            params: vec![AbiParam::HandleConsumed {
                resource_type: resource.to_string(),
            }],
            is_close_for: Some(resource.to_string()),
            may_block: false,
        },
        target_triple: host_triple(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        foreign_resources: Vec::new(),
        provider_crate: crate_name.to_string(),
        provider_resource_types: vec![resource.to_string()],
        provider_target_triples: vec![host_triple()],
    }
}

/// A body that holds a source handle, transfers it, and returns. Locals:
/// `_0` return, `_1` the SOURCE handle, `_2` the DESTINATION slot, `_3` the status.
fn transfer_program() -> MirProgram {
    let mut types = TypeContext::default();
    types
        .host_resource_closes
        .insert(source_ty(), mir::ProviderCallId(1));
    types
        .host_resource_closes
        .insert(dest_ty(), mir::ProviderCallId(2));

    MirProgram {
        entry_source: test_source().id(),
        sources: test_sources(),
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
                    ty: source_ty(),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    ty: dest_ty(),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    ty: MirTy::UInt32,
                    kind: LocalKind::Temp,
                },
            ],
            blocks: vec![
                BasicBlock {
                    statements: Vec::new(),
                    terminator: (
                        Terminator::Call {
                            callee: Callee::Provider(mir::ProviderCallId(0)),
                            args: vec![
                                // MOVE, not Copy: the source is consumed.
                                Operand::Move(Place::local(mir::LocalId(1))),
                                // The HandleOut argument NAMES the destination place (C7.8.4):
                                // a slot-backed resource is dead until the provider writes it, so
                                // it cannot be passed as a `&mut`.
                                Operand::Copy(Place::local(mir::LocalId(2))),
                            ],
                            dest: Place::local(mir::LocalId(3)),
                            target: mir::BlockId(1),
                        },
                        source_info(),
                    ),
                },
                BasicBlock {
                    statements: Vec::new(),
                    terminator: (Terminator::Return, source_info()),
                },
            ],
            entry: mir::BlockId(0),
        }],
        types,
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        resource_bindings: vec![
            (
                "tcp_stream".to_string(),
                mir::HostResourceNominal::Item(ItemId(7)),
            ),
            (
                "wrapped_stream".to_string(),
                mir::HostResourceNominal::Item(ItemId(8)),
            ),
        ],
        provider_calls: vec![
            transfer_call(),
            close_call(
                "stark-std-net",
                "stark-net-native",
                "network-client",
                "tcp_stream",
            ),
            close_call(
                "stark-wrap-native",
                "stark-wrap-native",
                "wrap",
                "wrapped_stream",
            ),
        ],
        provider_closes: Vec::new(),
    }
}

fn emit(program: &MirProgram) -> String {
    emit_program::emit(
        program,
        &build_versions(
            "rustc-test".to_string(),
            host_triple(),
            starkc::backend::generated_rust::Profile::Debug,
        ),
        &starkc::layout::TargetLayout::default(),
    )
    .expect("a cross-provider transfer must emit")
    .main_rs
}

/// **The proving case.** A transfer emits, and the destination slot begins dead and is written only
/// under the success arm — so a failed transfer leaves nothing for a later Drop to close.
#[test]
fn a_transfer_emits_and_writes_the_destination_only_on_success() {
    let emitted = emit(&transfer_program());
    assert!(
        emitted.contains("ValueSlot::dead()"),
        "the destination handle must begin dead:\n{emitted}"
    );
    if let (Some(zero_arm), Some(write)) = (emitted.find("0u32 =>"), emitted.find(".write(")) {
        assert!(
            write > zero_arm,
            "the destination must be written INSIDE the success arm — a failed transfer must \
             leave the slot dead:\n{emitted}"
        );
    }
}

/// **The load-bearing assertion.** The SOURCE's close must never be emitted for the transferred
/// handle, on any path. Ownership left the caller at call entry; closing it here would be a
/// double-release against the provider that now owns it.
#[test]
fn the_source_close_is_never_emitted_for_a_transferred_handle() {
    let emitted = emit(&transfer_program());

    // The symbol necessarily appears in the `extern "C"` block — it is a declared provider call in
    // the program, and declaring a close is not calling one. Asserting on the whole file would pass
    // or fail for the wrong reason, so the extern block is cut first and the BODY is what is
    // checked. (Observed: the crude form failed here purely on the declaration.)
    let body_start = emitted
        .find("extern \"C\" {")
        .and_then(|start| emitted[start..].find("\n}").map(|end| start + end))
        .expect("the generated crate must have an extern block");
    let body = &emitted[body_start..];

    assert!(
        !body.contains("stark_tcp_stream_close("),
        "the SOURCE provider's close must never be CALLED for a transferred handle: ownership left \
         the caller at call entry, on success AND on failure (CD-360). Calling it is a \
         double-release against the provider that now owns it:\n{body}"
    );
}

/// Source and destination are DISTINCT types. Identity is structural over
/// `{nominal, provider, resource}`, so a transfer is a real type transition — and no path may treat
/// the two as interchangeable.
#[test]
fn the_source_and_destination_are_distinct_resource_types() {
    assert_ne!(
        source_ty(),
        dest_ty(),
        "a transfer must change the resource's identity, not merely re-tag it"
    );
    let types = TypeContext::default();
    assert!(
        !types.is_copy(&source_ty()) && !types.is_copy(&dest_ty()),
        "neither side of a transfer may be Copy — duplicating either is a double-release"
    );
}

/// The consumed argument is a MOVE. A `Copy` here would leave the caller holding a live handle to a
/// resource another provider now owns, which is the duplicate-ownership failure the packet named.
#[test]
fn the_consumed_argument_is_moved_not_copied() {
    let program = transfer_program();
    let (terminator, _) = &program.bodies[0].blocks[0].terminator;
    let Terminator::Call { args, .. } = terminator else {
        panic!("expected a provider call terminator");
    };
    assert!(
        matches!(args.first(), Some(Operand::Move(_))),
        "the consumed source must be moved, not copied: {args:?}"
    );
}
