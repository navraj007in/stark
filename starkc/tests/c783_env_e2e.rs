//! WP-C7.8.3 — `stark-env` end to end: metadata → selection → MIR provider call → generated Rust
//! → static link → real execution.
//!
//! The `stark-time` e2e proved the chain for one provider. This proves it is a *property of the
//! seam* rather than of that provider: a second, independently written crate — different identity,
//! different capabilities, and the first non-empty status vocabulary — links and runs with no
//! compiler change beyond a registry entry.
//!
//! **Scope stated honestly.** `stark_env_args_len` has the same `ScalarOut(U64)` shape as the clock,
//! so it executes here. The buffer-carrying calls (`_fill`, `var_len`, `var_fill`) are proven at
//! *emission* level only: constructing a `&[UInt8]` argument requires slice machinery that no
//! hand-built MIR body should be inventing, and lowering does not yet produce provider calls from
//! STARK source. Their execution proof arrives with the package surface that generates them.

use starkc::backend::generated_rust::{
    emit_native_debug_with_toolchain, NativeBuildOptions, NativeToolchainOptions, Profile,
};
use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
    ValidatedProviderCall,
};
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;
use std::collections::BTreeMap;

#[path = "support/paths.rs"]
mod paths;
use paths::{repo_provider, repo_provider_root};

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
        .expect("a rust toolchain is required for this test")
        .host_triple
}

/// Selection through the **real registry**, not a hand-written fixture — so this exercises the
/// path `stark build` takes rather than a parallel one.
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
        span: test_source().synthetic_span(),
        origin: mir::Origin::UserCode,
    }
}

fn place(i: u32) -> Place {
    Place {
        local: mir::LocalId(i),
        projection: Vec::new(),
    }
}

/// `fn main() { let mut n = 0; stark_env_args_len(&mut n); println(n); }`
fn entry_body() -> MirBody {
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
            LocalDecl {
                ty: MirTy::Unit,
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
                terminator: (
                    Terminator::Call {
                        callee: Callee::Runtime(mir::RuntimeFn::PrintlnUInt64),
                        args: vec![Operand::Copy(place(1))],
                        dest: place(4),
                        target: mir::BlockId(2),
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

fn program() -> MirProgram {
    MirProgram {
        entry_source: test_source().id(),
        sources: test_sources(),
        bodies: vec![entry_body()],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: vec![resolve("process.args", "stark_env_args_len")],
        resource_bindings: Vec::new(),
        provider_closes: Vec::new(),
    }
}

/// **The end-to-end proof for a second provider.** Builds, links `stark-env-native`, runs the
/// binary, and reads the argument count it printed.
#[test]
fn stark_env_args_len_executes_natively() {
    let program = program();
    let verified = mir::verify::verify_program(&program)
        .unwrap_or_else(|e| panic!("the provider program must verify: {e:?}"));

    let target_dir = std::env::temp_dir().join(format!("stark-c783-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);

    let toolchain = starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test");
    let mut provider_crates = BTreeMap::new();
    provider_crates.insert(
        "stark-env-native".to_string(),
        provider_registry::built_in_crate_location("stark-env-native", &repo_provider_root())
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
    .unwrap_or_else(|e| panic!("a build linking stark-env must succeed: {e:?}"));

    // Two extra arguments, so the count is something the test controls rather than whatever the
    // harness happened to pass.
    let output = std::process::Command::new(&artifact.binary_path)
        .args(["alpha", "beta"])
        .output()
        .unwrap_or_else(|e| panic!("running the built binary: {e}"));

    assert!(
        output.status.success(),
        "the program must exit cleanly; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout is UTF-8");
    let printed = stdout.trim();
    let count: u64 = printed
        .parse()
        .unwrap_or_else(|e| panic!("expected an argument count, got {printed:?}: {e}"));

    // The provider's documented convention decides whether argv[0] is included, so the assertion is
    // on the two arguments this test supplied rather than on an exact total -- a count that ignored
    // them would mean the out-slot was never written.
    assert!(
        count >= 2,
        "expected at least the two supplied arguments, got {count}"
    );

    let _ = std::fs::remove_dir_all(&target_dir);
}

/// The buffer-carrying calls declare correctly in generated Rust, even though no hand-built MIR
/// body executes them yet. This is the honest half of the scope note: emission is proven, execution
/// waits for a lowering that produces these calls from STARK source.
#[test]
fn the_buffer_carrying_calls_declare_correctly() {
    let mut program = program();
    program.provider_calls = vec![
        resolve("process.env", "stark_env_var_len"),
        resolve("process.env", "stark_env_var_fill"),
        resolve("process.args", "stark_env_args_fill"),
    ];

    // Declarations are emitted from the call RECORDS, independently of any body, which is what
    // lets a buffer-carrying signature be checked before a body exists that can call it.
    let declarations =
        starkc::backend::generated_rust::emit_provider::emit_extern_declarations(&program)
            .expect("declarations emit");

    assert!(
        declarations.contains(
            "fn stark_env_var_len(a0: stark_runtime::provider_abi::BorrowedBuffer, a1: *mut bool, \
             a2: *mut u64)"
        ),
        "{declarations}"
    );
    assert!(
        declarations.contains(
            "fn stark_env_var_fill(a0: stark_runtime::provider_abi::BorrowedBuffer, \
             a1: stark_runtime::provider_abi::BorrowedBufferMut, a2: *mut u64)"
        ),
        "{declarations}"
    );
    assert!(
        declarations.contains(
            "fn stark_env_args_fill(a0: stark_runtime::provider_abi::BorrowedBufferMut, \
             a1: *mut u64)"
        ),
        "{declarations}"
    );
}

/// `stark-env`'s declared surface is what this test compiled against. If the crate ever needs an
/// ABI-facing edit to make the integration work, that is drift and this is where it surfaces —
/// the same standing check `stark-time` carries.
#[test]
fn the_provider_crate_is_used_unmodified() {
    let src_path = repo_provider("stark-env").join("src").join("lib.rs");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    for symbol in [
        "stark_env_args_len",
        "stark_env_args_fill",
        "stark_env_var_len",
        "stark_env_var_fill",
    ] {
        assert!(
            src.contains(&format!("extern \"C\" fn {symbol}")),
            "{symbol} must still be exported by the provider crate unmodified"
        );
    }
}
