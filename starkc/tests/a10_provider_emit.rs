//! WP-C7.8.2d-2 — static `extern "C"` declaration and call emission for non-resource parameters.
//!
//! Closes A10 §4 invariant 6 for the admitted forms: **every provider borrow is materialised from
//! a named live place, and its lifetime covers the complete call.** The failure this guards is a
//! pointer taken from a temporary that dies before or during the call expression — which compiles,
//! links, and is undefined behaviour at runtime, so nothing downstream would catch it.
//!
//! Status *dispatch* is C7.8.2d-3; these tests assert the raw status reaches the MIR destination
//! and is not interpreted here.

use starkc::backend::version::build_versions;
use starkc::backend::generated_rust::{emit_program, emit_provider};
use starkc::layout::TargetLayout;
use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
    ValidatedProviderCall,
};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ScalarTy};
use starkc::source::{SourceFile, Span};
use std::sync::Arc;

const LINUX: &str = "x86_64-unknown-linux-gnu";

fn call_named(symbol: &str, params: Vec<AbiParam>) -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: "stark-std-time".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "clock".to_string(),
        function: FunctionDecl {
            name: symbol.to_string(),
            capability: "clock".to_string(),
            params,
            is_close_for: None,
            may_block: false,
        },
        target_triple: LINUX.to_string(),
        provider_target_triples: vec![LINUX.to_string()],
    }
}

fn program_with(calls: Vec<ValidatedProviderCall>) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("a10.stark", ""))],
        bodies: Vec::new(),
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: calls,
    }
}

fn externs(calls: Vec<ValidatedProviderCall>) -> String {
    emit_provider::emit_extern_declarations(&program_with(calls)).expect("declarations emit")
}

// ------------------------------------------------------- extern declarations --

/// A program calling no provider emits no `extern` block at all — providers must not leave a trace
/// in the overwhelming majority of programs that use none.
#[test]
fn no_provider_calls_emits_nothing() {
    assert_eq!(externs(Vec::new()), "");
}

/// The declared symbol is emitted **verbatim**. It never passes through `mangle::sanitize_symbol`,
/// because the same name must resolve under a future `dlsym` and a repair would make the linkage
/// name differ from the metadata name.
#[test]
fn symbols_are_emitted_verbatim() {
    let src = externs(vec![call_named(
        "stark_time_monotonic_now_ns",
        vec![AbiParam::ScalarOut(ScalarTy::U64)],
    )]);
    assert!(
        src.contains("fn stark_time_monotonic_now_ns("),
        "symbol not emitted verbatim:\n{src}"
    );
    assert!(src.contains("extern \"C\" {"), "no extern block:\n{src}");
}

/// ABI §6.1's parameter table, emitted exactly. `ScalarOut` and `ScalarInOut` are both `*mut T` —
/// the C signature cannot carry their difference, which is an initialisation contract (§11.1)
/// enforced by C7.8.2d-3 rather than by a type.
#[test]
fn parameter_types_follow_the_abi_table() {
    let src = externs(vec![call_named(
        "p",
        vec![
            AbiParam::ScalarIn(ScalarTy::U32),
            AbiParam::ScalarOut(ScalarTy::U64),
            AbiParam::ScalarInOut(ScalarTy::I16),
            AbiParam::BufferIn,
            AbiParam::BufferInOut,
        ],
    )]);

    assert!(src.contains("a0: u32"), "{src}");
    assert!(src.contains("a1: *mut u64"), "{src}");
    assert!(src.contains("a2: *mut i16"), "{src}");
    assert!(
        src.contains("a3: stark_runtime::provider_abi::BorrowedBuffer"),
        "{src}"
    );
    assert!(
        src.contains("a4: stark_runtime::provider_abi::BorrowedBufferMut"),
        "{src}"
    );
}

/// Every provider function returns `ProviderStatus` (ABI §11) — never a value directly. A
/// declaration returning anything else would mean the emitter had invented a return channel.
#[test]
fn every_declaration_returns_provider_status() {
    let src = externs(vec![
        call_named("a", vec![AbiParam::ScalarOut(ScalarTy::U64)]),
        call_named("b", Vec::new()),
    ]);
    assert_eq!(
        src.matches("-> stark_runtime::provider_abi::ProviderStatus;")
            .count(),
        2,
        "{src}"
    );
}

/// Two call sites of one function produce **one** declaration, and the declaration order does not
/// depend on the order records appear — the determinism Gate C7.2's reproducibility classification
/// rests on.
#[test]
fn declarations_are_deduplicated_and_order_independent() {
    let a = call_named("zzz_last", vec![AbiParam::ScalarOut(ScalarTy::U64)]);
    let b = call_named("aaa_first", vec![AbiParam::ScalarIn(ScalarTy::U8)]);

    let forward = externs(vec![a.clone(), b.clone(), a.clone()]);
    let reverse = externs(vec![b, a]);

    assert_eq!(
        forward, reverse,
        "declaration order must not depend on record order"
    );
    assert_eq!(forward.matches("fn zzz_last(").count(), 1, "{forward}");
    assert!(
        forward.find("fn aaa_first(") < forward.find("fn zzz_last("),
        "declarations must be sorted:\n{forward}"
    );
}

/// A resource-carrying declaration is refused at emission too, not only at verification. MIR-0024
/// already rejects such a program, so this is defence in depth on the path where being wrong is
/// invisible — a binary whose host effect silently never happens.
#[test]
fn resource_parameters_are_refused_at_emission() {
    let err = emit_provider::emit_extern_declarations(&program_with(vec![call_named(
        "p",
        vec![AbiParam::HandleConsumed {
            resource_type: "file".to_string(),
        }],
    )]))
    .expect_err("a resource parameter must not emit");
    let text = format!("{err:?}");
    assert!(
        text.contains("file"),
        "the resource type must be named: {text}"
    );
}

// ------------------------------------------------------ end-to-end generation --

/// The real `stark-time` shape, generated through the full backend: one `ScalarOut(U64)`, declared
/// and called. This is the artefact C7.8.2e will link against a real provider.
#[test]
fn stark_time_declaration_is_generated() {
    let src = externs(vec![
        call_named(
            "stark_time_monotonic_now_ns",
            vec![AbiParam::ScalarOut(ScalarTy::U64)],
        ),
        call_named(
            "stark_time_unix_now",
            vec![AbiParam::ScalarOut(ScalarTy::I64)],
        ),
    ]);

    assert!(
        src.contains("fn stark_time_monotonic_now_ns(a0: *mut u64)"),
        "{src}"
    );
    assert!(
        src.contains("fn stark_time_unix_now(a0: *mut i64)"),
        "{src}"
    );
    // The provider and capability are recorded in a comment, so a reader of generated source can
    // tell which provider a symbol came from without consulting the metadata.
    assert!(src.contains("stark-std-time"), "{src}");
    assert!(src.contains("clock"), "{src}");
}

/// No `catch_unwind` anywhere in generated provider code. The generated workspace builds with
/// `panic = "abort"` in both profiles, so a provider panic aborts rather than unwinding; wrapping
/// it would misclassify a provider defect as recoverable (Packet 1 §1.1) and split panic semantics
/// within one workspace.
#[test]
fn generated_provider_code_never_catches_unwind() {
    let src = externs(vec![call_named(
        "stark_time_monotonic_now_ns",
        vec![AbiParam::ScalarOut(ScalarTy::U64)],
    )]);
    assert!(!src.contains("catch_unwind"), "{src}");
}

// ------------------------------------------ call-site emission (invariant 6) --

fn info() -> SourceInfo {
    SourceInfo {
        file: mir::FileId(0),
        span: Span { lo: 0, hi: 0 },
        origin: mir::Origin::UserCode,
    }
}

fn place(index: u32) -> Place {
    Place {
        local: mir::LocalId(index),
        projection: Vec::new(),
    }
}

/// An entry body that calls `stark_time_monotonic_now_ns` with its declared `&mut u64` out-slot and
/// a `UInt32` status destination.
fn entry_calling_provider() -> MirBody {
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
            LocalDecl {
                ty: MirTy::UInt64,
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: true,
                    inner: Box::new(MirTy::UInt64),
                },
                kind: LocalKind::Temp,
            },
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
                            place(1),
                            Rvalue::Use(Operand::Const(Constant::Int(0, MirTy::UInt64))),
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(2),
                            Rvalue::RefOf {
                                mutable: true,
                                place: place(1),
                            },
                        ),
                        info(),
                    ),
                ],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(0)),
                        args: vec![Operand::Move(place(2))],
                        dest: place(3),
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

fn generate_call_site() -> String {
    let mut program = program_with(vec![call_named(
        "stark_time_monotonic_now_ns",
        vec![AbiParam::ScalarOut(ScalarTy::U64)],
    )]);
    program.bodies = vec![entry_calling_provider()];

    let versions = build_versions(
        "0.0.0-test".to_string(),
        "test-triple".to_string(),
        starkc::backend::generated_rust::Profile::Debug,
    );
    emit_program::emit(&program, &versions, &TargetLayout::default())
        .expect("provider program must emit")
        .main_rs
}

/// **Invariant 6.** Every provider borrow is materialised from a *named* binding, and the pointer
/// is taken from that binding rather than from a temporary.
///
/// The shape this rules out — `stark_provider_fn(make_bytes(v).as_ptr(), ...)` — compiles, links,
/// and is undefined behaviour at runtime, so no later stage would catch it.
#[test]
fn borrows_are_materialised_from_named_bindings() {
    let src = generate_call_site();

    assert!(
        src.contains("let __prov_a0 ="),
        "the argument must be bound to a named local before the call:\n{src}"
    );
    let binding = src.find("let __prov_a0 =").expect("binding");
    let call = src.find("stark_time_monotonic_now_ns(").expect("call site");
    // The declaration also mentions the symbol; find the CALL, which follows the binding.
    let call = src[binding..]
        .find("unsafe {")
        .map(|o| binding + o)
        .unwrap_or(call);
    assert!(
        binding < call,
        "the named binding must precede the call:\n{src}"
    );
    assert!(
        src.contains("__prov_a0 as *mut u64"),
        "the pointer must be taken from the named binding:\n{src}"
    );
}

/// The `unsafe` block wraps **only** the FFI call, so a reviewer sees exactly what is unchecked.
#[test]
fn unsafe_wraps_only_the_call() {
    let src = generate_call_site();
    assert!(
        src.contains("unsafe { stark_time_monotonic_now_ns("),
        "unsafe must wrap the call itself:\n{src}"
    );
    assert!(!src.contains("catch_unwind"), "{src}");
}

/// The raw `ProviderStatus.code` reaches the MIR destination and is **not** interpreted here.
/// Success/declared-error/contract-violation dispatch is C7.8.2d-3's job; doing it in the emitter
/// would put channel policy outside the binding plan built to hold it.
#[test]
fn the_raw_status_code_reaches_the_destination() {
    let src = generate_call_site();
    assert!(
        src.contains("let __prov_status: u32 ="),
        "the status must be bound as a raw u32:\n{src}"
    );
    assert!(
        src.contains("}.code;"),
        "the code field must be read:\n{src}"
    );
    // No dispatch yet: nothing in d-2 branches on the status.
    assert!(
        !src.contains("__prov_status =="),
        "d-2 must not interpret the status:\n{src}"
    );
}

/// The program declares the symbol it calls — declaration and call site agree.
#[test]
fn the_called_symbol_is_declared_in_the_same_program() {
    let src = generate_call_site();
    assert!(
        src.contains("fn stark_time_monotonic_now_ns(a0: *mut u64)"),
        "{src}"
    );
    assert!(
        src.matches("stark_time_monotonic_now_ns").count() >= 2,
        "expected a declaration and a call:\n{src}"
    );
}
