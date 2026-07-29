//! WP-C7.8.5 close-out — `stark-time`'s recorded blocker, discharged against its exact wording.
//!
//! `stark-time/BLOCKERS.md` names one blocker and names it precisely:
//!
//! > Real `extern "C"` provider linkage/invocation: STARK generated code calling
//! > `stark_time_monotonic_now_ns` / `stark_time_unix_now` through Native Provider ABI v0.1 and
//! > observing their `ProviderStatus`/output-slot results.
//!
//! **Both** functions, so both are executed here. The earlier `a10_stark_time_e2e` ran only the
//! monotonic clock, which would have discharged half a blocker while reading as a whole one.
//!
//! What this does **not** discharge is recorded in the same file's §24.1: `Instant::now`,
//! `UnixTimestamp::now` and `Instant::elapsed` are STARK-level package APIs, and no package surface
//! generates provider calls yet. The seam is proven; the package API is not.

use starkc::backend::generated_rust::{
    emit_native_debug_with_toolchain, NativeBuildOptions, NativeToolchainOptions, Profile,
};
use starkc::mir::{
    self, BasicBlock, Callee, Constant, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand,
    Place, ProviderCallId, RuntimeFn, Rvalue, SourceInfo, Statement, Terminator, TypeContext,
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

/// `fn main() { let mut a = 0; [let mut b = 0;] f(&mut a[, &mut b]); println(a); }`
///
/// `extra` carries the second out-slot `stark_time_unix_now` declares (nanoseconds), so one body
/// serves both clock functions and the arity comes from the declaration rather than a guess.
fn entry_body(scalar: MirTy, printer: RuntimeFn, extra: Option<MirTy>) -> MirBody {
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
                ty: scalar.clone(),
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: true,
                    inner: Box::new(scalar.clone()),
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
            LocalDecl {
                ty: extra.clone().unwrap_or(MirTy::Unit),
                kind: LocalKind::Temp,
            },
            LocalDecl {
                ty: MirTy::Ref {
                    mutable: true,
                    inner: Box::new(extra.clone().unwrap_or(MirTy::Unit)),
                },
                kind: LocalKind::Temp,
            },
        ],
        blocks: vec![
            BasicBlock {
                statements: vec![
                    (
                        Statement::Assign(
                            place(1),
                            Rvalue::Use(Operand::Const(Constant::Int(0, scalar))),
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
                ]
                .into_iter()
                .chain(extra.iter().flat_map(|t| {
                    [
                        (
                            Statement::Assign(
                                place(5),
                                Rvalue::Use(Operand::Const(Constant::Int(0, t.clone()))),
                            ),
                            info(),
                        ),
                        (
                            Statement::Assign(
                                place(6),
                                Rvalue::RefOf {
                                    mutable: true,
                                    place: place(5),
                                },
                            ),
                            info(),
                        ),
                    ]
                }))
                .collect(),
                terminator: (
                    Terminator::Call {
                        callee: Callee::Provider(ProviderCallId(0)),
                        args: if extra.is_some() {
                            vec![Operand::Move(place(2)), Operand::Move(place(6))]
                        } else {
                            vec![Operand::Move(place(2))]
                        },
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
                        callee: Callee::Runtime(printer),
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

/// Builds and runs a one-call clock program, returning what it printed.
fn run_clock(function: &str, scalar: MirTy, printer: RuntimeFn, extra: Option<MirTy>) -> String {
    let call = ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["clock".to_string()],
    )
    .expect("clock must be available on this host")
    .resolve("clock", function)
    .unwrap_or_else(|e| panic!("{function} must resolve: {e:#?}"));

    let program = MirProgram {
        files: vec![Arc::new(SourceFile::new("clock.stark", ""))],
        bodies: vec![entry_body(scalar, printer, extra)],
        types: TypeContext::default(),
        mir_version: mir::MIR_VERSION.to_string(),
        runtime_surface: mir::MIR_RUNTIME_SURFACE.to_string(),
        provider_calls: vec![call],
        resource_bindings: Vec::new(),
    };
    let verified = mir::verify::verify_program(&program)
        .unwrap_or_else(|e| panic!("{function}: must verify: {e:?}"));

    let target_dir =
        std::env::temp_dir().join(format!("stark-c785-{}-{function}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);

    let toolchain = starkc::native_toolchain::discover(None).expect("toolchain");
    let mut provider_crates = BTreeMap::new();
    provider_crates.insert(
        "stark-time-native".to_string(),
        provider_registry::crate_location("stark-time-native", &repo_root())
            .expect("stark-time-native must be locatable"),
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
    .unwrap_or_else(|e| panic!("{function}: build must succeed: {e:?}"));

    let output = std::process::Command::new(&artifact.binary_path)
        .output()
        .unwrap_or_else(|e| panic!("{function}: running the binary: {e}"));
    assert!(
        output.status.success(),
        "{function}: must exit cleanly; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = std::fs::remove_dir_all(&target_dir);
    String::from_utf8(output.stdout)
        .expect("stdout is UTF-8")
        .trim()
        .to_string()
}

/// The blocker's first named function, observing its output slot.
#[test]
fn monotonic_now_executes_and_reports_a_reading() {
    let printed = run_clock(
        "stark_time_monotonic_now_ns",
        MirTy::UInt64,
        RuntimeFn::PrintlnUInt64,
        None,
    );
    let ns: u64 = printed
        .parse()
        .unwrap_or_else(|e| panic!("expected nanoseconds, got {printed:?}: {e}"));
    assert!(
        ns > 0,
        "a monotonic reading of 0 means the slot was never written"
    );
}

/// **The half the earlier e2e did not cover.** The blocker names `stark_time_unix_now` too, and a
/// discharge that ran only the monotonic clock would have read as complete while covering one of
/// two symbols.
#[test]
fn unix_now_executes_and_reports_a_plausible_wall_clock() {
    let printed = run_clock(
        "stark_time_unix_now",
        MirTy::Int64,
        RuntimeFn::PrintlnInt64,
        Some(MirTy::UInt32),
    );
    let secs: i64 = printed
        .parse()
        .unwrap_or_else(|e| panic!("expected a unix timestamp, got {printed:?}: {e}"));

    // Bounded rather than exact: the value must be a real clock reading, not a zero slot or a
    // sentinel. 1.7e9 is 2023; 4e9 is 2096.
    assert!(
        (1_700_000_000..4_000_000_000).contains(&secs),
        "unix timestamp {secs} is outside any plausible range, so the slot was not written by a \
         real clock"
    );
}

/// The blocker's own §24.1 records what could not be claimed. Two of its items are now false, and
/// this asserts the ones that made them false are real rather than assumed.
#[test]
fn the_recorded_blocker_is_discharged_at_its_own_terms() {
    // "Real linkage/invocation is explicitly deferred" -- no longer: both symbols are declared by
    // the registry and reachable through selection.
    let set = ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["clock".to_string()],
    )
    .expect("clock selects");

    for function in ["stark_time_monotonic_now_ns", "stark_time_unix_now"] {
        let call = set.resolve("clock", function).expect("resolves");
        assert_eq!(call.provider_crate, "stark-time-native");
        assert_eq!(call.symbol(), function, "the symbol is emitted verbatim");
    }

    // "the native crate is a standalone, unlinked Rust library" -- no longer: it has a location the
    // build resolves, and the two e2e tests above link and run it.
    assert!(
        provider_registry::crate_location("stark-time-native", &repo_root())
            .expect("locatable")
            .join("Cargo.toml")
            .is_file()
    );
}
