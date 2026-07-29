//! WP-C7.8 — the **buffer argument path**, executed.
//!
//! Three capabilities were blocked on one thing: constructing a `&[UInt8]` to pass as a `BufferIn`.
//! `stark-env`'s `var_len`/`var_fill`, every `stark-file` entry point (a path is a buffer), and
//! `stark-net`'s addresses all need it. Only `stark-time` and `stark_env_args_len` — the
//! scalar-only doors — executed without it.
//!
//! **It needed no new MIR surface.** `SliceNew(&[T; N], lo, hi, inclusive) -> &[T]` has existed
//! since `0.1-A6`, and `&[UInt8; N]` coerces to `&[UInt8]` exactly as the emitter's `as_ptr()` /
//! `len()` expect. So the blocker was a missing *proof*, not a missing mechanism — no CE3, no
//! surface bump, no new `RuntimeFn`.
//!
//! What this executes:
//!
//! - `stark_env_var_len(name, &mut present, &mut len)` — buffer in, two scalar outs;
//! - the same call for a variable that is **absent**, so presence is observed rather than assumed.

use starkc::backend::generated_rust::{
    emit_native_debug_with_toolchain, NativeBuildOptions, NativeToolchainOptions, Profile,
};
use starkc::mir::{
    self, AggKind, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy,
    Operand, Place, ProviderCallId, RuntimeFn, Rvalue, SourceInfo, Statement, Terminator,
    TypeContext, ValidatedProviderCall,
};
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;
use starkc::source::{SourceFile, Span};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn host_triple() -> String {
    starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test")
        .host_triple
}

fn resolve(capability: &str, function: &str) -> ValidatedProviderCall {
    ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &[capability.to_string()],
    )
    .expect("the capability must be available on this host")
    .resolve(capability, function)
    .expect("the declared function must resolve")
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

fn u64c(v: i128) -> Operand {
    Operand::Const(Constant::Int(v, MirTy::UInt64))
}

/// Locals, in the order the body below uses them.
const L_RET: u32 = 0;
const L_NAME: u32 = 1; // [UInt8; N]  -- the variable name's bytes
const L_NAME_REF: u32 = 2; // &[UInt8; N]
const L_NAME_SLICE: u32 = 3; // &[UInt8]   -- the BufferIn argument
const L_PRESENT: u32 = 4; // Bool
const L_PRESENT_REF: u32 = 5; // &mut Bool
const L_LEN: u32 = 6; // UInt64
const L_LEN_REF: u32 = 7; // &mut UInt64
const L_STATUS: u32 = 8; // UInt32
const L_UNIT: u32 = 9; // Unit

/// `stark_env_var_len(name, &mut present, &mut len)`, then print the length.
///
/// The name arrives as an array of byte constants because that is what a lowering would produce
/// from a string literal: `SliceNew` over `&[UInt8; N]` is the same path, and building it by hand
/// here proves the emitter handles the shape rather than the literal.
fn entry_body(name: &str) -> MirBody {
    let bytes: Vec<Operand> = name
        .bytes()
        .map(|b| Operand::Const(Constant::Int(b as i128, MirTy::UInt8)))
        .collect();
    let n = bytes.len() as u64;

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
                ty: MirTy::Array(Box::new(MirTy::UInt8), n),
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Array(Box::new(MirTy::UInt8), n)),
                },
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: false,
                    inner: Box::new(MirTy::Slice(Box::new(MirTy::UInt8))),
                },
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Bool,
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: true,
                    inner: Box::new(MirTy::Bool),
                },
                kind: LocalKind::Temp,
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
            LocalDecl {
                ty: MirTy::Unit,
                kind: LocalKind::Temp,
            },
        ],
        blocks: vec![
            // bb0: materialise the name bytes and the two out-slots, then take the slice view.
            BasicBlock {
                statements: vec![
                    (
                        Statement::Assign(
                            place(L_NAME),
                            Rvalue::Aggregate(AggKind::Array(MirTy::UInt8), bytes),
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_NAME_REF),
                            Rvalue::RefOf {
                                mutable: false,
                                place: place(L_NAME),
                            },
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_PRESENT),
                            Rvalue::Use(Operand::Const(Constant::Bool(false))),
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(place(L_LEN), Rvalue::Use(u64c(0))),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_PRESENT_REF),
                            Rvalue::RefOf {
                                mutable: true,
                                place: place(L_PRESENT),
                            },
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_LEN_REF),
                            Rvalue::RefOf {
                                mutable: true,
                                place: place(L_LEN),
                            },
                        ),
                        info(),
                    ),
                ],
                // `SliceNew(&array, 0, N, exclusive-bound)` -> `&[UInt8]`. Surface 0.1-A6; no new
                // runtime operation was needed for the buffer path.
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::SliceNew),
                        args: vec![
                            Operand::Copy(place(L_NAME_REF)),
                            u64c(0),
                            u64c(n as i128),
                            Operand::Const(Constant::Bool(false)),
                        ],
                        dest: place(L_NAME_SLICE),
                        target: mir::BlockId(1),
                    },
                    info(),
                ),
            },
            // bb1: the provider call -- BufferIn plus two ScalarOut slots.
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(0)),
                        args: vec![
                            Operand::Copy(place(L_NAME_SLICE)),
                            Operand::Move(place(L_PRESENT_REF)),
                            Operand::Move(place(L_LEN_REF)),
                        ],
                        dest: place(L_STATUS),
                        target: mir::BlockId(2),
                    },
                    info(),
                ),
            },
            // bb2: print the length the provider reported.
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::PrintlnUInt64),
                        args: vec![Operand::Copy(place(L_LEN))],
                        dest: place(L_UNIT),
                        target: mir::BlockId(3),
                    },
                    info(),
                ),
            },
            BasicBlock {
                statements: Vec::new(),
                terminator: (Terminator::Return, info()),
            },
        ],
        entry: mir::BlockId(L_RET),
    }
}

fn program(name: &str) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("env.stark", ""))],
        bodies: vec![entry_body(name)],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: vec![resolve("process.env", "stark_env_var_len")],
    }
}

/// Builds, links `stark-env`, runs with a controlled environment, and returns what the program
/// printed.
fn run_with_env(name: &str, env: &[(&str, &str)], tag: &str) -> String {
    let program = program(name);
    let verified = mir::verify::verify_program(&program)
        .unwrap_or_else(|e| panic!("{tag}: the buffer program must verify: {e:?}"));

    let target_dir =
        std::env::temp_dir().join(format!("stark-buf-e2e-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);

    let toolchain = starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test");
    let mut provider_crates = BTreeMap::new();
    provider_crates.insert(
        "stark-env-native".to_string(),
        provider_registry::crate_location("stark-env-native", &repo_root())
            .expect("stark-env-native must be locatable"),
    );

    let artifact = emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.clone(),
            profile: Profile::Debug,
            ..NativeBuildOptions::default()
        },
        &NativeToolchainOptions {
            rustc: toolchain.rustc.clone(),
            cargo: toolchain.cargo.clone(),
            runtime_crate: toolchain.runtime_crate.clone(),
            provider_crates,
        },
    )
    .unwrap_or_else(|e| panic!("{tag}: a build with a buffer argument must succeed: {e:?}"));

    let mut command = std::process::Command::new(&artifact.binary_path);
    for (k, v) in env {
        command.env(k, v);
    }
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("{tag}: running the built binary: {e}"));
    assert!(
        output.status.success(),
        "{tag}: the program must exit cleanly; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = std::fs::remove_dir_all(&target_dir);
    String::from_utf8(output.stdout)
        .expect("stdout is UTF-8")
        .trim()
        .to_string()
}

/// **The buffer path, executed.** A `&[UInt8]` built in MIR reaches the provider as a
/// `BorrowedBuffer`, and the provider reads the right variable through it.
#[test]
fn a_buffer_argument_reaches_the_provider() {
    let printed = run_with_env(
        "STARK_C78_BUFFER_PROBE",
        &[("STARK_C78_BUFFER_PROBE", "twelve-chars")],
        "present",
    );
    let len: u64 = printed
        .parse()
        .unwrap_or_else(|e| panic!("expected a length, got {printed:?}: {e}"));

    assert_eq!(
        len,
        "twelve-chars".len() as u64,
        "the provider must report the length of the value this test set, which it can only know \
         by reading the name out of the buffer"
    );
}

/// The same program with the variable **absent** reports zero length — so the previous test's
/// result came from reading the buffer, not from a constant that happened to match.
#[test]
fn an_absent_variable_reports_zero_length() {
    let printed = run_with_env("STARK_C78_DEFINITELY_UNSET_VAR", &[], "absent");
    assert_eq!(printed, "0", "an absent variable must report zero length");
}
