//! WP-C7.8.2c — negative fixtures for A10 §4's verifier invariants.
//!
//! One case per rule the verifier can enforce today, each built by taking a **valid** provider call
//! and breaking exactly one thing. Building them by mutation rather than from scratch is
//! deliberate: a fixture assembled independently can fail for an unrelated reason and still look
//! like it proved the rule.
//!
//! Invariants 6–9 (borrow validity, consumed-resource invalidation, output-slot discipline,
//! channel discipline) are not covered here and cannot be: they constrain the *binding layer's*
//! generated control flow — reads on failure paths, handle liveness after a consuming call — and
//! no lowering produces that structure yet. They land with C7.8.2d/e. Their absence is a scope
//! boundary, not an oversight.

use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
    ValidatedProviderCall,
};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ScalarTy};
use starkc::source::SourceFile;
use std::sync::Arc;

/// AS1b-ii: a hand-built MIR program still needs a real registered source for its spans.
fn test_source() -> starkc::source::RegisteredSource {
    let mut registry = starkc::source::SourceRegistry::default();
    registry.intern(std::sync::Arc::new(starkc::source::SourceFile::new(
        "test.stark",
        "",
    )))
}

const LINUX: &str = "x86_64-unknown-linux-gnu";

// ---------------------------------------------------------------- fixtures --

/// A valid `stark_time_monotonic_now_ns` call record: one `ScalarOut(U64)` slot, resolved for a
/// target the provider declares.
fn valid_call() -> ValidatedProviderCall {
    ValidatedProviderCall {
        // CD-360: predates cross-provider transfer; consumes nothing foreign.
        foreign_resources: Vec::new(),
        provider: ProviderIdentity {
            name: "stark-std-time".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "clock".to_string(),
        function: FunctionDecl {
            name: "stark_time_monotonic_now_ns".to_string(),
            capability: "clock".to_string(),
            params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
            is_close_for: None,
            may_block: false,
        },
        target_triple: LINUX.to_string(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        provider_crate: "test-provider-native".to_string(),
        provider_resource_types: Vec::new(),
        provider_target_triples: vec![LINUX.to_string(), "aarch64-apple-darwin".to_string()],
    }
}

fn info() -> SourceInfo {
    SourceInfo {
        file: mir::FileId(0),
        span: test_source().synthetic_span(),
        origin: mir::Origin::UserCode,
    }
}

fn place(index: u32) -> Place {
    Place {
        local: mir::LocalId(index),
        projection: Vec::new(),
    }
}

/// Body shape matching the ABI signature. `slot_ty` and `dest_ty` are parameterised so a fixture
/// can break the argument or destination type without rebuilding the body.
fn body_with(slot_ty: MirTy, dest_ty: MirTy, args: Vec<Operand>) -> MirBody {
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
                ty: slot_ty.clone(),
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: true,
                    inner: Box::new(slot_ty.clone()),
                },
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: dest_ty,
                kind: LocalKind::Temp,
            },
        ],
        blocks: vec![
            BasicBlock {
                statements: vec![
                    (
                        Statement::Assign(
                            place(1),
                            Rvalue::Use(Operand::Const(Constant::Int(0, slot_ty))),
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
                        args,
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

fn program(call: ValidatedProviderCall, body: MirBody) -> MirProgram {
    MirProgram {
        entry_source: test_source().id(),
        files: vec![Arc::new(SourceFile::new("a10.stark", ""))],
        bodies: vec![body],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: vec![call],
        resource_bindings: Vec::new(),
        provider_closes: Vec::new(),
    }
}

/// The unbroken program: every fixture below is this, with exactly one thing changed.
fn valid_program() -> MirProgram {
    program(
        valid_call(),
        body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
    )
}

fn expect_code(program: &MirProgram, code: &str) {
    match mir::verify::verify_program(program) {
        Ok(_) => panic!("expected {code}, got a clean pass"),
        Err(errors) => assert!(
            errors.iter().any(|e| e.code == code),
            "expected {code}, got: {errors:#?}"
        ),
    }
}

/// The control. If this ever fails, every negative below is proving nothing.
#[test]
fn the_baseline_program_verifies() {
    if let Err(errors) = mir::verify::verify_program(&valid_program()) {
        panic!("baseline must verify: {errors:#?}");
    }
}

// -------------------------------------------------------------- invariants --

/// Invariant 2 — the provider must declare the target the call was resolved for. The record
/// carries the declared list precisely so this is a check rather than an assumption.
#[test]
fn target_not_declared_by_the_provider_is_rejected() {
    let mut call = valid_call();
    call.target_triple = "riscv64gc-unknown-linux-gnu".to_string();
    expect_code(
        &program(
            call,
            body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
        ),
        "MIR-0021",
    );
}

/// ABI §16 check 1, re-asserted in MIR: an artifact must not carry a call validated against a
/// different ABI revision.
#[test]
fn a_foreign_abi_version_is_rejected() {
    let mut call = valid_call();
    call.provider.abi_version = "0.2".to_string();
    expect_code(
        &program(
            call,
            body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
        ),
        "MIR-0025",
    );
}

/// Invariant 3 — a function reachable from the right provider can still belong to a different
/// capability, and that must not silently widen what the capability admits.
#[test]
fn function_from_another_capability_is_rejected() {
    let mut call = valid_call();
    call.function.capability = "filesystem".to_string();
    expect_code(
        &program(
            call,
            body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
        ),
        "MIR-0022",
    );
}

/// Invariant 4 — the symbol is emitted verbatim, so it must *be* a legal C identifier in the
/// record. Re-checked in MIR rather than trusted from resolution, because emission reads this
/// record, not the resolver's transient state.
#[test]
fn an_invalid_symbol_in_the_record_is_rejected() {
    for bad in ["", "has space", "9lives", "has-hyphen"] {
        let mut call = valid_call();
        call.function.name = bad.to_string();
        expect_code(
            &program(
                call,
                body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
            ),
            "MIR-0023",
        );
    }
}

/// Invariant 5, boundary — a resource-typed parameter has no MIR type yet, so it is **refused**
/// rather than guessed at. Guessing would invent a type for a resource whose identity the compiler
/// does not know, leaving ABI §11.1's `resource_type` validation nothing to check against.
#[test]
fn a_resource_typed_parameter_is_refused_until_c784() {
    for param in [
        AbiParam::HandleBorrowed {
            resource_type: "File".to_string(),
        },
        AbiParam::HandleConsumed {
            resource_type: "File".to_string(),
        },
        AbiParam::HandleOut {
            resource_type: "File".to_string(),
        },
    ] {
        let mut call = valid_call();
        call.function.params = vec![param];
        expect_code(
            &program(
                call,
                body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
            ),
            "MIR-0024",
        );
    }
}

// ------------------------------------------------- invariant 5: ABI shape --

/// Deriving the signature from the declaration *is* invariant 5: too few arguments for the
/// declared parameters is caught by the shared arity check.
#[test]
fn missing_the_declared_out_slot_argument_is_rejected() {
    expect_code(
        &program(
            valid_call(),
            body_with(MirTy::UInt64, MirTy::UInt32, Vec::new()),
        ),
        "MIR-0005",
    );
}

/// A `ScalarOut(U64)` slot is `&mut UInt64` (ABI §6.1). Passing `&mut UInt32` is a shape mismatch,
/// not a widening.
#[test]
fn a_wrongly_typed_out_slot_is_rejected() {
    expect_code(
        &program(
            valid_call(),
            body_with(MirTy::UInt32, MirTy::UInt32, vec![Operand::Move(place(2))]),
        ),
        "MIR-0005",
    );
}

/// The destination receives `ProviderStatus.code` — `UInt32`, never a STARK `Result`. Converting a
/// status into a `Result::Err` is the binding layer's job; doing it at this layer would collapse
/// the three failure channels at exactly the point that must keep them apart.
#[test]
fn a_destination_that_is_not_the_status_code_is_rejected() {
    expect_code(
        &program(
            valid_call(),
            body_with(MirTy::UInt64, MirTy::Bool, vec![Operand::Move(place(2))]),
        ),
        "MIR-0005",
    );
}

/// Every failure is reported, not just the first — a record broken three ways should produce three
/// findings rather than three verification runs.
#[test]
fn independent_invariant_failures_are_all_reported() {
    let mut call = valid_call();
    call.target_triple = "riscv64gc-unknown-linux-gnu".to_string();
    call.provider.abi_version = "0.2".to_string();
    call.function.capability = "filesystem".to_string();

    let errors = match mir::verify::verify_program(&program(
        call,
        body_with(MirTy::UInt64, MirTy::UInt32, vec![Operand::Move(place(2))]),
    )) {
        Err(e) => e,
        Ok(_) => panic!("expected rejection"),
    };

    for code in ["MIR-0021", "MIR-0025", "MIR-0022"] {
        assert!(
            errors.iter().any(|e| e.code == code),
            "expected {code} among {errors:#?}"
        );
    }
}
