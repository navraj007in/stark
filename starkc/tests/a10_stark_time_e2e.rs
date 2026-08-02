//! WP-C7.8.2e — `stark-time` end to end: metadata → selection → MIR → generated Rust → **link** →
//! **execute**.
//!
//! Every earlier A10 test asserts emitted *source text*. This one compiles it. That distinction is
//! the point: a generated snippet can read correctly, be asserted correctly, and still fail to
//! compile or misbehave at the ABI boundary — and nothing text-level can catch that.
//!
//! The ten points this closes, from the C7.8.2e checklist:
//!
//! 1. provider metadata validates
//! 2. provider selection resolves
//! 3. MIR uses `Callee::Provider`
//! 4. generated Rust declares the exact `extern "C"` symbol
//! 5. provider is linked statically
//! 6. output slot begins uninitialised
//! 7. output is read only on status zero
//! 8. declared errors map correctly (vacuous here — `stark-time` declares none, which is itself
//!    the meaningful case: every nonzero status is a contract violation)
//! 9. unknown status becomes a contract violation
//! 10. provider source has no semantic or ABI-facing edit
//!
//! Point 10 is the one that constrains everything else: `stark-time/native/` is used **exactly as
//! it sits in the tree**. If this test needed to change that crate, the integration model would
//! have drifted from ABI v0.1, and the drift would be the finding.

use starkc::backend::generated_rust::{
    emit_native_debug_with_toolchain, NativeBuildOptions, NativeToolchainOptions, Profile,
};
use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity, ProviderMetadata, ScalarTy};
use starkc::provider_resolve::{DeclaredProvider, ProviderSet};
use starkc::source::{SourceFile, Span};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Seeded into the clock's output slot so "was it written?" is testable without depending on clock
/// resolution. `u64::MAX` nanoseconds is ~584 years of uptime, so no reading can collide with it.
const SENTINEL: u64 = u64::MAX;

/// The `stark-time` provider crate, exactly where it sits in the repository.
fn stark_time_crate() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .join("stark-time")
        .join("native")
}

/// `stark-time`'s metadata as its own crate declares it (point 1/10).
///
/// Mirrored rather than imported because `starkc` cannot depend on a provider crate — the
/// dependency runs the other way. The `provider_metadata_validates_against_abi_v0_1` test inside
/// that crate is what keeps this mirror honest.
fn stark_time_provider() -> DeclaredProvider {
    DeclaredProvider {
        metadata: ProviderMetadata {
            // CD-360: predates cross-provider transfer; consumes nothing foreign.
            foreign_resources: Vec::new(),
            identity: ProviderIdentity {
                name: "stark-std-time".to_string(),
                semver: (0, 1, 0),
                abi_version: "0.1".to_string(),
            },
            target_triples: vec![
                "aarch64-apple-darwin".to_string(),
                "x86_64-apple-darwin".to_string(),
                "x86_64-unknown-linux-gnu".to_string(),
                "x86_64-pc-windows-msvc".to_string(),
            ],
            capabilities: vec!["clock".to_string()],
            resource_types: vec![],
            functions: vec![
                FunctionDecl {
                    name: "stark_time_monotonic_now_ns".to_string(),
                    capability: "clock".to_string(),
                    params: vec![AbiParam::ScalarOut(ScalarTy::U64)],
                    is_close_for: None,
                    may_block: false,
                },
                FunctionDecl {
                    name: "stark_time_unix_now".to_string(),
                    capability: "clock".to_string(),
                    // TWO out-slots: seconds and nanoseconds, matching the provider crate and
                    // `provider_registry`. This mirror said one slot until now. It did not fail,
                    // because this test never calls `unix_now` through it -- which is precisely how
                    // a wrong mirror survives: nothing checks a declaration nobody calls. CD-219 is
                    // the same defect where it *was* called, and it aborted at the boundary.
                    params: vec![
                        AbiParam::ScalarOut(ScalarTy::I64),
                        AbiParam::ScalarOut(ScalarTy::U32),
                    ],
                    is_close_for: None,
                    may_block: false,
                },
            ],
        },
        crate_name: "stark-time-native".to_string(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        origin: "stark-time/native/Cargo.toml".to_string(),
    }
}

fn host_triple() -> String {
    starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test")
        .host_triple
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

/// `fn main() { let mut ns = 0; provider(&mut ns); println(ns); }`, hand-built in MIR.
///
/// Hand-built because no STARK surface syntax reaches a provider yet — that is the package work of
/// C7.8.3 onward. What matters here is that the MIR is the same shape lowering will produce.
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
                        // Initialised to a SENTINEL, not zero. `0` is a legitimate reading -- the
                        // provider's origin is lazily set on the first call, so the first call's
                        // elapsed time is the gap between two adjacent instructions, and on Windows
                        // `Instant` is coarse enough for both to land in one tick. Seeding the slot
                        // with a value the clock cannot produce is what makes "the slot was written"
                        // testable independently of clock resolution.
                        Statement::Assign(
                            place(1),
                            Rvalue::Use(Operand::Const(Constant::Int(
                                SENTINEL as i128,
                                MirTy::UInt64,
                            ))),
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
            // Observing the value is what makes this an execution test rather than a link test.
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

/// Points 1 and 2: the metadata validates and selection resolves, for the host target.
fn resolve_monotonic() -> starkc::mir::ValidatedProviderCall {
    let set = ProviderSet::select(
        vec![stark_time_provider()],
        &host_triple(),
        &["clock".to_string()],
    )
    .expect("stark-time must select for the host target");

    set.resolve("clock", "stark_time_monotonic_now_ns")
        .expect("the declared clock function must resolve")
}

fn program() -> MirProgram {
    MirProgram {
        files: vec![Arc::new(SourceFile::new("clock.stark", ""))],
        bodies: vec![entry_body()],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: vec![resolve_monotonic()],
        resource_bindings: Vec::new(),
        provider_closes: Vec::new(),
    }
}

/// **The end-to-end proof.** Builds, links `stark-time-native`, runs the binary, and reads the
/// clock value it printed.
#[test]
fn stark_time_monotonic_clock_executes_natively() {
    // Point 3: the program really does use `Callee::Provider`, not a runtime intrinsic.
    let program = program();
    let body = &program.bodies[0];
    assert!(
        matches!(
            &body.blocks[0].terminator.0,
            Terminator::Call {
                callee: Callee::Provider(_),
                ..
            }
        ),
        "the clock call must be a provider call, not a RuntimeFn"
    );

    // Verified MIR is a precondition for emission, and it exercises invariants 1-5 on a real
    // record rather than a fixture.
    let verified = mir::verify::verify_program(&program)
        .unwrap_or_else(|e| panic!("provider program must verify: {e:?}"));

    let target_dir = std::env::temp_dir().join(format!("stark-a10-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);

    let mut provider_crates = BTreeMap::new();
    provider_crates.insert("stark-time-native".to_string(), stark_time_crate());

    let toolchain = starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test");

    // Point 5: the provider is linked statically, as an ordinary path dependency of the generated
    // crate. No dynamic loading, no dlopen -- Packet 1's option B.
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
    .unwrap_or_else(|e| panic!("native build with a provider must succeed: {e:?}"));

    assert!(
        artifact.binary_path.is_file(),
        "no binary at {}",
        artifact.binary_path.display()
    );

    // Execution. This is the line no text-level assertion could stand in for.
    let output = std::process::Command::new(&artifact.binary_path)
        .output()
        .unwrap_or_else(|e| panic!("running the built binary: {e}"));

    assert!(
        output.status.success(),
        "the program must exit cleanly; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout is UTF-8");
    let printed = stdout.trim();
    let nanos: u64 = printed
        .parse()
        .unwrap_or_else(|e| panic!("expected a monotonic nanosecond count, got {printed:?}: {e}"));

    // Points 6 and 7, observed rather than asserted on text: the success arm actually copied the
    // provider's value out of the `MaybeUninit` slot.
    //
    // Tested against the SENTINEL rather than against zero. `nanos > 0` was the original assertion
    // and it was wrong -- not merely fragile: zero is a value this provider genuinely returns, since
    // its origin is initialised on the first call, so the first reading measures the gap between two
    // adjacent instructions. It passed on Linux and macOS because their `Instant` resolves that gap,
    // and failed on Windows CI when both calls landed in the same tick. A sentinel the clock cannot
    // produce tests the write-back itself, which is the actual property.
    assert_ne!(
        nanos, SENTINEL,
        "the output slot still holds its sentinel: the provider's write-back never reached it"
    );

    // WP-C6.4c §10.7's property, checked against the REAL lock rather than a string builder.
    //
    // Its unit test was removed when the hand-authored lock was: that lock assumed a two-package
    // path-only graph, and it broke every native build the moment `stark-runtime` itself gained a
    // dependency. Cargo resolves the lock now, so the property has to be asserted on Cargo's
    // output -- which is stronger, because it is the file the build actually used.
    let lock = find_generated_lock(&target_dir).expect("the generated crate must have a lock");
    let lock_text = std::fs::read_to_string(&lock).expect("reading the generated lock");
    assert!(
        lock_text.contains("name = \"stark-time-native\""),
        "the provider must appear in the lock:\n{lock_text}"
    );
    // A path-only graph has no registry source and no checksum -- which is what makes the
    // `--offline` build provably network-free rather than warm-cache-dependent.
    assert!(!lock_text.contains("source = "), "{lock_text}");
    assert!(!lock_text.contains("checksum = "), "{lock_text}");

    let _ = std::fs::remove_dir_all(&target_dir);
}

/// The generated crate's `Cargo.lock`, wherever under the build root it landed.
fn find_generated_lock(root: &std::path::Path) -> Option<PathBuf> {
    walk(root)
        .into_iter()
        .find(|entry| entry.file_name().is_some_and(|n| n == "Cargo.lock"))
}

fn walk(root: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out
}

/// Point 4, checked on the artefact's own generated source: the declared symbol appears verbatim.
#[test]
fn the_generated_source_declares_the_exact_symbol() {
    let program = program();
    let verified = mir::verify::verify_program(&program).expect("verifies");
    let versions = starkc::backend::version::build_versions(
        "0.0.0-test".to_string(),
        host_triple(),
        Profile::Debug,
    );
    let src = starkc::backend::generated_rust::emit_program::emit(
        verified.program(),
        &versions,
        &starkc::layout::TargetLayout::default(),
    )
    .expect("emits")
    .main_rs;

    assert!(
        src.contains("fn stark_time_monotonic_now_ns(a0: *mut u64)"),
        "{src}"
    );
    // Point 6: the slot starts uninitialised.
    assert!(src.contains("MaybeUninit::<u64>::uninit()"), "{src}");
    // Point 7: read only on success.
    assert!(src.contains("0u32 => {"), "{src}");
    // Point 9: an undeclared status is a contract violation -- and `stark-time` declares none, so
    // EVERY nonzero status takes that path (point 8's meaningful form).
    assert!(src.contains("contract_violation_unknown_status("), "{src}");
}

/// Point 10, as a standing check rather than a promise: the provider crate's declared surface is
/// what this test compiled against. If `stark-time/native` ever needs an ABI-facing edit to make
/// the integration work, that is drift from ABI v0.1 and this assertion is where it surfaces.
#[test]
fn the_provider_crate_is_used_unmodified() {
    let src_path = stark_time_crate().join("src").join("lib.rs");
    let src = std::fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", src_path.display()));

    for symbol in ["stark_time_monotonic_now_ns", "stark_time_unix_now"] {
        assert!(
            src.contains(&format!("pub unsafe extern \"C\" fn {symbol}")),
            "{symbol} must still be exported by the provider crate unmodified"
        );
    }
    assert_eq!(
        src.matches("#[no_mangle]").count(),
        2,
        "the provider must export exactly its two declared symbols"
    );
}
