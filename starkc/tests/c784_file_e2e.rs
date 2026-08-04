//! WP-C7.8.4 — `stark-file` end to end: a real file created, written, completed and closed.
//!
//! This is the first execution of a **resource-carrying** provider call. Everything before it moved
//! scalars and buffers; this moves an owning handle across the ABI boundary and back, and exercises
//! the three ownership forms in one program:
//!
//! - `stark_file_create` produces a handle (`HandleOut`, validated on success only);
//! - `stark_file_write` and `stark_file_complete` **borrow** it, so the file survives the call;
//! - `stark_file_close` **consumes** it (`HandleConsumed`, `is_close_for: Some("file")`).
//!
//! It is also the first execution of Packet 3's close semantics: `complete` is the recoverable
//! completion operation, `close` is the ABI close, and they are separate calls because ABI §13.1
//! leaves a close no way to report anything but a status.

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
use std::sync::Arc;

const CONTENT: &str = "stark-c784-e2e";

#[path = "support/paths.rs"]
mod paths;
use paths::repo_provider_root;

fn host_triple() -> String {
    starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test")
        .host_triple
}

fn filesystem() -> ProviderSet {
    ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["filesystem".to_string()],
    )
    .expect("filesystem must be available on this host")
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

fn bytes_of(s: &str) -> Vec<Operand> {
    s.bytes()
        .map(|b| Operand::Const(Constant::Int(b as i128, MirTy::UInt8)))
        .collect()
}

// Locals.
const L_RET: u32 = 0;
const L_PATH: u32 = 1;
const L_PATH_REF: u32 = 2;
const L_PATH_SLICE: u32 = 3;
const L_FILE: u32 = 4;
const L_DATA: u32 = 5;
const L_DATA_REF: u32 = 6;
const L_DATA_SLICE: u32 = 7;
const L_WRITTEN: u32 = 8;
const L_WRITTEN_REF: u32 = 9;
const L_FILE_BORROW: u32 = 10; // &File -- the HandleBorrowed argument
const L_STATUS: u32 = 11;
const L_UNIT: u32 = 12;

/// `create(path, &mut f); write(&f, data, &mut n); complete(&f); close(f); println(n);`
fn entry_body(path: &str) -> MirBody {
    let path_bytes = bytes_of(path);
    let path_len = path_bytes.len() as u64;
    let data_bytes = bytes_of(CONTENT);
    let data_len = data_bytes.len() as u64;

    let file_ty = MirTy::Core(starkc::hir::CoreType::File, Vec::new());

    let arr = |n: u64| MirTy::Array(Box::new(MirTy::UInt8), n);
    let shared = |t: MirTy| MirTy::Ref {
        mutable: false,
        inner: Box::new(t),
    };
    let exclusive = |t: MirTy| MirTy::Ref {
        mutable: true,
        inner: Box::new(t),
    };
    let slice = || shared(MirTy::Slice(Box::new(MirTy::UInt8)));

    let temp = |ty: MirTy| LocalDecl {
        ty,
        kind: LocalKind::Temp,
    };

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
            temp(arr(path_len)),
            temp(shared(arr(path_len))),
            temp(slice()),
            temp(file_ty.clone()),
            temp(arr(data_len)),
            temp(shared(arr(data_len))),
            temp(slice()),
            temp(MirTy::UInt64),
            temp(exclusive(MirTy::UInt64)),
            temp(shared(file_ty)),
            temp(MirTy::UInt32),
            temp(MirTy::Unit),
        ],
        blocks: vec![
            // bb0: materialise both byte arrays and their references.
            BasicBlock {
                statements: vec![
                    (
                        Statement::Assign(
                            place(L_PATH),
                            Rvalue::Aggregate(AggKind::Array(MirTy::UInt8), path_bytes),
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_PATH_REF),
                            Rvalue::RefOf {
                                mutable: false,
                                place: place(L_PATH),
                            },
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_DATA),
                            Rvalue::Aggregate(AggKind::Array(MirTy::UInt8), data_bytes),
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_DATA_REF),
                            Rvalue::RefOf {
                                mutable: false,
                                place: place(L_DATA),
                            },
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(place(L_WRITTEN), Rvalue::Use(u64c(0))),
                        info(),
                    ),
                ],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::SliceNew),
                        args: vec![
                            Operand::Copy(place(L_PATH_REF)),
                            u64c(0),
                            u64c(path_len as i128),
                            Operand::Const(Constant::Bool(false)),
                        ],
                        dest: place(L_PATH_SLICE),
                        target: mir::BlockId(1),
                    },
                    info(),
                ),
            },
            // bb1: the data slice.
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::SliceNew),
                        args: vec![
                            Operand::Copy(place(L_DATA_REF)),
                            u64c(0),
                            u64c(data_len as i128),
                            Operand::Const(Constant::Bool(false)),
                        ],
                        dest: place(L_DATA_SLICE),
                        target: mir::BlockId(2),
                    },
                    info(),
                ),
            },
            // bb2: create -- the handle is written into the destination on success only. The
            // argument is the PLACE, not a reference to it: a slot becomes live by being written.
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(0)),
                        args: vec![
                            Operand::Copy(place(L_PATH_SLICE)),
                            Operand::Move(place(L_FILE)),
                        ],
                        dest: place(L_STATUS),
                        target: mir::BlockId(3),
                    },
                    info(),
                ),
            },
            // bb3: write -- BORROWS the file, so it survives for the calls below.
            BasicBlock {
                statements: vec![
                    (
                        Statement::Assign(
                            place(L_FILE_BORROW),
                            Rvalue::RefOf {
                                mutable: false,
                                place: place(L_FILE),
                            },
                        ),
                        info(),
                    ),
                    (
                        Statement::Assign(
                            place(L_WRITTEN_REF),
                            Rvalue::RefOf {
                                mutable: true,
                                place: place(L_WRITTEN),
                            },
                        ),
                        info(),
                    ),
                ],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(1)),
                        args: vec![
                            Operand::Copy(place(L_FILE_BORROW)),
                            Operand::Copy(place(L_DATA_SLICE)),
                            Operand::Move(place(L_WRITTEN_REF)),
                        ],
                        dest: place(L_STATUS),
                        target: mir::BlockId(4),
                    },
                    info(),
                ),
            },
            // bb4: complete -- Packet 3's recoverable completion, before the consuming close.
            BasicBlock {
                statements: vec![(
                    Statement::Assign(
                        place(L_FILE_BORROW),
                        Rvalue::RefOf {
                            mutable: false,
                            place: place(L_FILE),
                        },
                    ),
                    info(),
                )],
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(2)),
                        args: vec![Operand::Copy(place(L_FILE_BORROW))],
                        dest: place(L_STATUS),
                        target: mir::BlockId(5),
                    },
                    info(),
                ),
            },
            // bb5: close -- CONSUMES the file. Ownership transfers at call entry.
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(3)),
                        args: vec![Operand::Move(place(L_FILE))],
                        dest: place(L_STATUS),
                        target: mir::BlockId(6),
                    },
                    info(),
                ),
            },
            // bb6: report the byte count the provider accepted.
            BasicBlock {
                statements: Vec::new(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(RuntimeFn::PrintlnUInt64),
                        args: vec![Operand::Copy(place(L_WRITTEN))],
                        dest: place(L_UNIT),
                        target: mir::BlockId(7),
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

fn resolve(function: &str) -> ValidatedProviderCall {
    filesystem()
        .resolve("filesystem", function)
        .unwrap_or_else(|e| panic!("{function} must resolve: {e:#?}"))
}

fn program(path: &str) -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("file.stark", ""))],
        bodies: vec![entry_body(path)],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        resource_bindings: Vec::new(),
        provider_closes: Vec::new(),
        // Order matters: the body addresses these by id.
        provider_calls: vec![
            resolve("stark_file_create"),
            resolve("stark_file_write"),
            resolve("stark_file_complete"),
            resolve("stark_file_close"),
        ],
    }
}

/// **A resource crosses the ABI boundary and back.** The program creates a file, writes to it
/// through a borrowed handle, completes it, closes it by consuming the handle — and the file is on
/// disk afterwards with the right contents.
#[test]
fn stark_file_creates_writes_and_closes_a_real_file() {
    let dir = std::env::temp_dir().join(format!("stark-c784-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let target = dir.join("written.txt");
    let _ = std::fs::remove_file(&target);

    let program = program(&target.to_string_lossy());
    let verified = mir::verify::verify_program(&program)
        .unwrap_or_else(|e| panic!("the file program must verify: {e:?}"));

    let build_dir = dir.join("build");
    let toolchain = starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test");
    let mut provider_crates = BTreeMap::new();
    provider_crates.insert(
        "stark-file-native".to_string(),
        provider_registry::built_in_crate_location("stark-file-native", &repo_provider_root())
            .expect("stark-file-native must be locatable"),
    );

    let artifact = emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: build_dir,
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
    .unwrap_or_else(|e| panic!("a build with a resource-carrying provider must succeed: {e:?}"));

    let output = std::process::Command::new(&artifact.binary_path)
        .output()
        .unwrap_or_else(|e| panic!("running the built binary: {e}"));
    assert!(
        output.status.success(),
        "the program must exit cleanly; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let printed = String::from_utf8(output.stdout).expect("stdout is UTF-8");
    let written: u64 = printed
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("expected a byte count, got {printed:?}: {e}"));
    assert_eq!(
        written,
        CONTENT.len() as u64,
        "the provider must report the bytes it accepted"
    );

    // The observable effect: a real file, with the bytes the program wrote. Nothing about the
    // generated source could have faked this.
    let on_disk = std::fs::read_to_string(&target)
        .unwrap_or_else(|e| panic!("the program must have created {}: {e}", target.display()));
    assert_eq!(on_disk, CONTENT);

    let _ = std::fs::remove_dir_all(&dir);
}
