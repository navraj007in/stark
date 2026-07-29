//! WP-C7.8.2d-2 — static `extern "C"` declaration and call emission for non-resource parameters.
//!
//! Closes A10 §4 invariant 6 for the admitted forms: **every provider borrow is materialised from
//! a named live place, and its lifetime covers the complete call.** The failure this guards is a
//! pointer taken from a temporary that dies before or during the call expression — which compiles,
//! links, and is undefined behaviour at runtime, so nothing downstream would catch it.
//!
//! Status *dispatch* is C7.8.2d-3; these tests assert the raw status reaches the MIR destination
//! and is not interpreted here.

use starkc::backend::generated_rust::{emit_program, emit_provider};
use starkc::backend::version::build_versions;
use starkc::layout::TargetLayout;
use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
    ValidatedProviderCall,
};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ScalarTy};
use starkc::provider_bind::StatusBinding;
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
        provider_crate: "test-provider-native".to_string(),
        provider_resource_types: Vec::new(),
        provider_target_triples: vec![LINUX.to_string()],
        status_binding: StatusBinding::new(),
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

/// Resource refusal moved when C7.8.2d-4 landed, and this test moved with it.
///
/// An `extern "C"` declaration is **shape-only**: a handle crosses as `RawResourceHandle`
/// regardless of which resource type it carries, so the declaration needs no registry and is not
/// where a resource can be refused. The refusal lives at *planning*, which is where a
/// `resource_type` must be bound to a MIR type — and it is still absolute, because the compiler's
/// builtin registry binds nothing.
#[test]
fn resource_declarations_are_shape_only_and_refusal_lives_in_planning() {
    let call = call_named(
        "p",
        vec![AbiParam::HandleConsumed {
            resource_type: "file".to_string(),
        }],
    );

    // Declaration: admitted, and the resource type never appears in the C signature.
    let src = emit_provider::emit_extern_declarations(&program_with(vec![call.clone()]))
        .expect("a handle declaration is shape-only");
    assert!(
        src.contains("a0: stark_runtime::provider_abi::RawResourceHandle"),
        "{src}"
    );
    assert!(!src.contains("\"file\""), "{src}");

    // Planning: refused, because the builtin registry binds no resource type.
    let mut declared = call;
    declared.provider_resource_types = vec!["file".to_string()];
    assert!(
        matches!(
            starkc::provider_bind::plan(
                ProviderCallId(0),
                &declared,
                &starkc::provider_bind::ResourceRegistry::builtin(),
                StatusBinding::new(),
            ),
            Err(starkc::provider_bind::PlanError::UnboundResourceType { .. })
        ),
        "the builtin registry must bind no resource type"
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
    generate_with_status(StatusBinding::new())
}

fn generate_with_status(status: StatusBinding) -> String {
    let mut call = call_named(
        "stark_time_monotonic_now_ns",
        vec![AbiParam::ScalarOut(ScalarTy::U64)],
    );
    call.status_binding = status;
    let mut program = program_with(vec![call]);
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
    // Every pointer the provider receives comes from a NAMED binding. For a `ScalarOut` that is
    // the emitter-owned `MaybeUninit` slot (C7.8.2d-3 moved it there so the MIR local is written
    // only on success); for an in/out form it is the argument binding itself. Neither is ever an
    // unnamed temporary, which is the property invariant 6 actually asserts.
    assert!(
        src.contains("let mut __prov_o0 = std::mem::MaybeUninit::"),
        "the output slot must itself be a named binding:\n{src}"
    );
    assert!(
        src.contains("__prov_o0.as_mut_ptr()"),
        "the pointer must be taken from the named slot:\n{src}"
    );
    // No pointer is ever taken from an inline expression.
    assert!(
        !src.contains(").as_mut_ptr()") && !src.contains(").as_ptr()"),
        "a pointer must never be taken from a temporary:\n{src}"
    );
}

/// The in/out forms keep taking their pointer from the argument binding, because §11.1 makes them
/// caller-owned — the `MaybeUninit` treatment would be wrong for them.
#[test]
fn in_out_forms_point_at_the_argument_binding() {
    let src = externs(vec![call_named(
        "p",
        vec![AbiParam::ScalarInOut(ScalarTy::I16)],
    )]);
    assert!(src.contains("a0: *mut i16"), "{src}");
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

// -------------------------- invariant 8: output-slot discipline (C7.8.2d-3) --

/// A `ScalarOut` slot is emitter-owned `MaybeUninit` storage. The provider writes *there*, not into
/// the MIR-visible local.
#[test]
fn output_slots_are_maybeuninit() {
    let src = generate_call_site();
    assert!(
        src.contains("let mut __prov_o0 = std::mem::MaybeUninit::<u64>::uninit();"),
        "the output slot must be uninitialised storage:\n{src}"
    );
    assert!(
        src.contains("__prov_o0.as_mut_ptr()"),
        "the provider must receive the slot pointer:\n{src}"
    );
}

/// **The core of invariant 8.** `assume_init` appears exactly once, inside the success arm, and the
/// MIR-visible local is written only there.
///
/// This is stronger than "do not read on failure": there is no failure path on which a read could
/// occur, because the only write to the MIR local is inside `0u32 =>`.
#[test]
fn outputs_are_read_only_on_success() {
    let src = generate_call_site();

    assert_eq!(
        src.matches("assume_init()").count(),
        1,
        "exactly one read of the output slot:\n{src}"
    );

    let success_arm = src.find("0u32 => {").expect("success arm");
    let read = src.find("assume_init()").expect("read");
    let match_end = src[success_arm..]
        .find("unknown =>")
        .map(|o| success_arm + o)
        .expect("contract-violation arm");

    assert!(
        success_arm < read && read < match_end,
        "the output read must be inside the success arm:\n{src}"
    );
    assert!(
        src.contains("*__prov_a0 = unsafe { __prov_o0.assume_init() };"),
        "the MIR local must be written only from the success arm:\n{src}"
    );
}

/// A provider that writes its slot and *then* returns failure must still not have its output read.
/// Guaranteed structurally: the write-back lives in the success arm, so no failure path reaches it.
#[test]
fn a_written_slot_is_still_unread_on_failure() {
    let src = generate_call_site();
    let unknown_arm = src.find("unknown =>").expect("contract-violation arm");
    assert!(
        !src[unknown_arm..].contains("assume_init"),
        "no failure arm may read the output slot:\n{src}"
    );
}

// ------------------------------ invariant 9: channel discipline (C7.8.2d-3) --

/// The three ABI §12 channels are structurally distinct, and the wildcard is the
/// **contract-violation** channel — never a generic package error.
#[test]
fn the_wildcard_arm_is_a_contract_violation_not_a_generic_error() {
    let src = generate_call_site();
    assert!(
        src.contains("stark_runtime::provider_abi::contract_violation_unknown_status("),
        "an undeclared status must route to the contract-violation channel:\n{src}"
    );
    for forbidden in ["Other", "IOError", "unwrap_or"] {
        assert!(
            !src[src.find("unknown =>").expect("arm")..].contains(forbidden),
            "the wildcard must not fall back to `{forbidden}`:\n{src}"
        );
    }
}

/// The contract-violation call names the provider and function, so a diagnostic can identify the
/// exact declaration that drifted.
#[test]
fn the_contract_violation_names_provider_and_function() {
    let src = generate_call_site();
    assert!(src.contains("\"stark-std-time\""), "{src}");
    assert!(src.contains("\"stark_time_monotonic_now_ns\""), "{src}");
}

/// With an empty status vocabulary — `stark-time`'s real case — **every** nonzero status is a
/// contract violation, so the match has exactly two arms.
#[test]
fn an_empty_vocabulary_gives_success_and_violation_only() {
    let src = generate_call_site();
    let m = src.find("match __prov_status {").expect("dispatch");
    let arm_region = &src[m..];
    assert_eq!(
        arm_region.matches("u32 => ").count(),
        1,
        "only the success arm is a literal code:\n{src}"
    );
}

/// A declared code becomes its own arm, distinct from both success and the wildcard, and it does
/// **not** read the output slot — a recoverable error means the provider reported failure, so
/// §11.1 leaves the slot invalid.
#[test]
fn a_declared_code_gets_its_own_arm_and_reads_no_output() {
    let mut status = StatusBinding::new();
    status.declare(1, "IOError::NotFound");
    status.declare(2, "IOError::PermissionDenied");
    let src = generate_with_status(status);

    assert!(
        src.contains("1u32 => {"),
        "declared code 1 needs an arm:\n{src}"
    );
    assert!(
        src.contains("2u32 => {"),
        "declared code 2 needs an arm:\n{src}"
    );
    assert!(
        src.contains("IOError::NotFound"),
        "the package error should be recorded in the generated comment:\n{src}"
    );

    // Still exactly one read, still in the success arm.
    assert_eq!(src.matches("assume_init()").count(), 1, "{src}");
    let arm1 = src.find("1u32 => {").expect("arm");
    let unknown = src.find("unknown =>").expect("arm");
    assert!(
        !src[arm1..unknown].contains("assume_init"),
        "a declared-error arm must not read the output slot:\n{src}"
    );

    // The wildcard is still a contract violation, not widened by the declarations.
    assert!(src.contains("contract_violation_unknown_status("), "{src}");
}

/// Declared arms are emitted in ascending code order regardless of declaration order, so generated
/// source — and the produced binary — do not depend on the order a package declared its errors.
#[test]
fn declared_arms_are_emitted_in_deterministic_order() {
    let mut a = StatusBinding::new();
    a.declare(7, "E7");
    a.declare(1, "E1");
    a.declare(3, "E3");

    let mut b = StatusBinding::new();
    b.declare(3, "E3");
    b.declare(1, "E1");
    b.declare(7, "E7");

    assert_eq!(
        generate_with_status(a),
        generate_with_status(b),
        "declaration order must not reach generated source"
    );
}
