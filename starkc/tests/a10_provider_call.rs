//! WP-C7.8.2a — MIR **representation** of Native Provider ABI v0.1 calls, and their refusal
//! everywhere execution or emission would otherwise occur.
//!
//! Governing decision: `WP-C7.8.1-DECISION-PACKETS.md` Packet 2 (CE3, CD-200); amendment
//! `STARKLANG/docs/compiler/mir-amendment-A10-provider-invocation.md`.
//!
//! This slice proves the halves A10 needs before any binding layer exists: MIR can **represent** a
//! provider call, and every consumer that cannot yet honour one **rejects it deterministically**
//! rather than dropping it. The nine verifier invariants land in C7.8.2c and static `extern "C"`
//! emission in C7.8.2d; until then the refusals below are the contract.
//!
//! The failure this guards is specific and silent: a provider call that is representable but
//! unhandled would emit a binary whose host side effect simply never happens.

use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
    ValidatedProviderCall,
};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ScalarTy};
use starkc::source::{SourceFile, Span};
use std::sync::Arc;

// ---------------------------------------------------------------- fixtures --

/// The `stark-time` monotonic clock function, as its provider crate already declares it
/// (`stark-time/native/src/lib.rs`). Used rather than an invented name so this fixture and
/// C7.8.2e's end-to-end test describe the same call.
fn monotonic_now_decl() -> FunctionDecl {
    FunctionDecl {
        name: "stark_time_monotonic_now_ns".to_string(),
        capability: "clock".to_string(),
        // §11: the physical return is always `ProviderStatus`, so the produced value travels
        // through an explicit output slot rather than a return type.
        params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
        is_close_for: None,
        may_block: false,
    }
}

fn validated_call() -> ValidatedProviderCall {
    ValidatedProviderCall {
        // CD-360: predates cross-provider transfer; consumes nothing foreign.
        foreign_resources: Vec::new(),
        provider: ProviderIdentity {
            name: "stark-time".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "clock".to_string(),
        function: monotonic_now_decl(),
        target_triple: "x86_64-unknown-linux-gnu".to_string(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        provider_crate: "test-provider-native".to_string(),
        provider_resource_types: Vec::new(),
        provider_target_triples: vec![
            "x86_64-unknown-linux-gnu".to_string(),
            "aarch64-apple-darwin".to_string(),
        ],
    }
}

fn info() -> SourceInfo {
    SourceInfo {
        file: mir::FileId(0),
        span: Span { lo: 0, hi: 0 },
        origin: mir::Origin::UserCode,
    }
}

/// A body presenting the ABI signature `stark_time_monotonic_now_ns` declares: one `&mut UInt64`
/// argument for its `ScalarOut(U64)` slot, and a `UInt32` destination for `ProviderStatus.code`.
///
/// Built to match rather than simplified, because A10 §4 invariant 5 is exactly the claim that
/// MIR argument types match the declaration — a fixture that sidestepped the parameter would
/// verify nothing.
fn provider_call_body(id: ProviderCallId) -> MirBody {
    MirBody {
        instance: mir::Instance {
            item: starkc::hir::ItemId(0),
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
            // The out-slot storage.
            LocalDecl {
                ty: MirTy::UInt64,
                kind: LocalKind::Temp,
            },
            // `&mut` to it -- ABI §6.1 maps `ScalarOut(T)` to `*mut T`.
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: true,
                    inner: Box::new(MirTy::UInt64),
                },
                kind: LocalKind::Temp,
            },
            // ProviderStatus.code.
            LocalDecl {
                ty: MirTy::UInt32,
                kind: LocalKind::Temp,
            },
        ],
        blocks: vec![
            BasicBlock {
                statements: vec![
                    (
                        Statement::Assign(
                            local_place(1),
                            Rvalue::Use(Operand::Const(Constant::Int(0, MirTy::UInt64))),
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            local_place(2),
                            Rvalue::RefOf {
                                mutable: true,
                                place: local_place(1),
                            },
                        ),
                        info(),
                    ),
                ],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(id),
                        args: vec![Operand::Move(local_place(2))],
                        dest: local_place(3),
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
    }
}

fn local_place(index: u32) -> Place {
    Place {
        local: mir::LocalId(index),
        projection: Vec::new(),
    }
}

fn program_with_provider_call(calls: Vec<ValidatedProviderCall>, id: ProviderCallId) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("a10.stark", ""))],
        bodies: vec![provider_call_body(id)],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: calls,
        resource_bindings: Vec::new(),
        provider_closes: Vec::new(),
    }
}

fn expect_code(program: &MirProgram, code: &str) {
    match mir::verify::verify_program(program) {
        Ok(_) => panic!("expected verifier rejection with {code}, got a clean pass"),
        Err(errors) => assert!(
            errors.iter().any(|e| e.code == code),
            "expected {code}, got: {errors:#?}"
        ),
    }
}

// ------------------------------------------------------------ representation --

/// A10 §7: the surface constant advances when the runtime surface changes.
///
/// A10 took it to `0.1-A10`; A13 (WP-C7.9 Packet D) took it to `0.1-A13` by adding the stderr half
/// of the output operations; **A14 (CD-381) takes it to `0.1-A14`**, covering twelve `RuntimeFn`
/// members added across CD-378 (`Fmt*` — `Display::fmt` on standard-library receivers) and CD-380
/// (`Fmt*Spec` — format-specification application). The pin moves with each revision on purpose: an
/// unannounced surface change fails here rather than reaching a consumer built against the old one.
///
/// **A known weakness of this guard, recorded rather than left implicit.** It pins the CONSTANT,
/// not the surface, so it fails when the constant moves — not when the surface grows without it.
/// CD-378 and CD-380 each added `RuntimeFn` members and left the constant alone; this test stayed
/// green through both, and an external review found the omission instead. A guard that could fail
/// for the right reason would have to derive something from the `RuntimeFn` set itself; a
/// hand-maintained variant count would not, because whoever adds a member updates the count in the
/// same edit.
#[test]
fn runtime_surface_is_current() {
    assert_eq!(mir::MIR_RUNTIME_SURFACE, "0.1-A14");
}

/// A10 adds **no** `RuntimeFn` member, and `Callee::Provider` is a form `RuntimeFn` cannot
/// express: a provider call carries a validated declaration, not a compiler-known opcode.
///
/// The stronger property A10 §8 requires — "no provider call represented as `RuntimeFn`" — is
/// pinned at C7.8.2e, where a real lowering exists to inspect. It is deliberately NOT asserted
/// here as a `RuntimeFn` variant count: a hand-maintained count in a test file cannot fail for
/// the reason that matters (someone adding a host capability as a runtime intrinsic), because
/// they would update the count in the same edit.
#[test]
fn provider_call_carries_a_declaration_runtime_fn_cannot() {
    let call = validated_call();
    assert_eq!(call.provider.name, "stark-time");
    assert_eq!(call.capability, "clock");
    assert_eq!(call.function.capability, call.capability);
    // The ABI shape lives on the declaration -- an output slot, because §11 makes
    // `ProviderStatus` the physical return for every provider function.
    assert!(matches!(
        call.function.params.as_slice(),
        [AbiParam::ScalarOut(ScalarTy::U64)]
    ));
}

/// The validated record carries the declaration, and the symbol is the declared name **verbatim**
/// — never routed through `mangle::sanitize_symbol` (Packet 1 §1.3). A repaired name would make
/// the metadata name differ from the linkage name, which must never be true when the same field
/// has to resolve under a future `dlsym`.
#[test]
fn symbol_is_the_declared_name_verbatim() {
    let call = validated_call();
    assert_eq!(call.symbol(), "stark_time_monotonic_now_ns");
    assert_eq!(call.symbol(), call.function.name);
}

/// `ProviderCallId` resolves through the program arena, and an out-of-range id resolves to
/// `None` rather than panicking — verification reports it (below) instead.
#[test]
fn arena_resolves_ids_and_rejects_out_of_range() {
    let program = program_with_provider_call(vec![validated_call()], ProviderCallId(0));
    assert_eq!(
        program.provider_call(ProviderCallId(0)).map(|c| c.symbol()),
        Some("stark_time_monotonic_now_ns")
    );
    assert!(program.provider_call(ProviderCallId(1)).is_none());
}

/// The dump names the provider and the verbatim symbol. A bare index would make a dump
/// unreadable without the arena beside it.
#[test]
fn dump_names_provider_and_symbol() {
    let program = program_with_provider_call(vec![validated_call()], ProviderCallId(0));
    let dump = program.dump();
    assert!(
        dump.contains("provider:stark-time:stark_time_monotonic_now_ns"),
        "dump did not name the provider call:\n{dump}"
    );
}

/// An unresolvable id still renders — as the defect, not as silence.
#[test]
fn dump_renders_unresolved_id_visibly() {
    let program = program_with_provider_call(Vec::new(), ProviderCallId(7));
    let dump = program.dump();
    assert!(
        dump.contains("provider:<unresolved #7>"),
        "an unresolved provider id must be visible in the dump:\n{dump}"
    );
}

// ----------------------------------------------------------------- refusals --

/// C7.8.2c admits a well-formed provider call. This is the positive case the blanket C7.8.2a
/// refusal (MIR-0020, now retired) stood in for: a record whose target, ABI version, capability
/// membership, symbol and parameter shapes all check out, called with matching MIR argument types
/// and a `UInt32` status destination.
#[test]
fn a_well_formed_provider_call_verifies() {
    let program = program_with_provider_call(vec![validated_call()], ProviderCallId(0));
    if let Err(errors) = mir::verify::verify_program(&program) {
        panic!("a well-formed provider call must verify, got: {errors:#?}");
    }
}

/// A dangling id is an arena-construction defect and is reported as its own error (MIR-0019),
/// never folded into a contract failure — the two have different causes and different fixes.
#[test]
fn dangling_provider_call_id_is_its_own_error() {
    let program = program_with_provider_call(Vec::new(), ProviderCallId(3));
    expect_code(&program, "MIR-0019");
}

// ------------------------------------------------------- A9 consumer rejection --

/// A10 §7 / V-SURFACE-1: a consumer pinned to `0.1-A9` must reject a `0.1-A10` program **before
/// consuming any body**. This is the intended behaviour of a versioned surface, not a regression.
///
/// Simulated in the only direction a single build can: stamping the older surface onto a program
/// this build produces exercises the same gate, in the same place, with the same code.
#[test]
fn a9_pinned_consumer_rejects_an_a10_program() {
    let mut program = program_with_provider_call(vec![validated_call()], ProviderCallId(0));
    program.runtime_surface = "0.1-A9".to_string();
    expect_code(&program, "MIR-0017");

    // The gate must fire BEFORE body verification: the body's own MIR-0020 must not appear.
    match mir::verify::verify_program(&program) {
        Ok(_) => panic!("expected rejection"),
        Err(errors) => {
            assert_eq!(
                errors.len(),
                1,
                "surface rejection must precede body verification: {errors:#?}"
            );
            assert_eq!(errors[0].code, "MIR-0017");
        }
    }
}
