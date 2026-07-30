//! C7.8 resource-lifecycle evidence.
//!
//! These tests pin the current executable boundary for package host resources. The build driver
//! still refuses source-level resource APIs before lowering, while the lower-level close arena and
//! generated-Rust close emission can be exercised directly. That split is deliberate evidence:
//! when the driver refusal is removed, the source-level lifecycle cases should move onto the normal
//! `stark build` path instead of being inferred from unit tests.

use starkc::backend::generated_rust::emit_program;
use starkc::backend::version::build_versions;
use starkc::hir::{ItemId, ItemKind};
use starkc::mir::provider_lower::ProviderLowering;
use starkc::mir::{
    self, BasicBlock, Callee, LocalDecl, LocalKind, MirBody, MirProgram, MirTy, Operand, Place,
    SourceInfo, Terminator, TypeContext, ValidatedProviderCall,
};
use starkc::native_build::{build_current_package, BuildCommandOptions};
use starkc::provider_abi::{AbiParam, FunctionDecl, ProviderIdentity};
use starkc::provider_derive::derive;
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;
use starkc::provider_synth::synthesize_with_resources;
use starkc::source::{SourceFile, Span};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn fixture_root(name: &str) -> PathBuf {
    repo_root()
        .join("target")
        .join("c788-resource-lifecycle")
        .join(format!("{name}-{}", std::process::id()))
}

fn write_package(root: &Path, manifest: &str, source: &str) {
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create package src dir");
    std::fs::write(root.join("starkpkg.json"), manifest).expect("write manifest");
    std::fs::write(src.join("main.stark"), source).expect("write source");
}

fn host_triple() -> String {
    starkc::native_toolchain::discover(None)
        .expect("a Rust toolchain is required for provider selection")
        .host_triple
}

fn source_info() -> SourceInfo {
    SourceInfo {
        file: mir::FileId(0),
        span: Span { lo: 0, hi: 0 },
        origin: mir::Origin::UserCode,
    }
}

fn tcp_stream_ty() -> MirTy {
    MirTy::host_resource(
        mir::HostResourceNominal::Item(ItemId(7)),
        "stark-std-net",
        "tcp_stream",
    )
}

fn tcp_call(name: &str, params: Vec<AbiParam>) -> ValidatedProviderCall {
    ValidatedProviderCall {
        provider: ProviderIdentity {
            name: "stark-std-net".to_string(),
            semver: (0, 1, 0),
            abi_version: "0.1".to_string(),
        },
        capability: "tcp".to_string(),
        function: FunctionDecl {
            name: name.to_string(),
            capability: "tcp".to_string(),
            params,
            is_close_for: (name == "stark_tcp_stream_close").then(|| "tcp_stream".to_string()),
            may_block: false,
        },
        target_triple: host_triple(),
        status_binding: starkc::provider_bind::StatusBinding::new(),
        provider_crate: "stark-net-native".to_string(),
        provider_resource_types: vec!["tcp_stream".to_string()],
        provider_target_triples: vec![host_triple()],
    }
}

#[test]
fn the_build_driver_no_longer_refuses_a_resource_bearing_provider_api() {
    let root = fixture_root("driver-refusal");
    let _ = std::fs::remove_dir_all(&root);
    write_package(
        &root,
        r#"{
  "name": "c788_lifecycle_refusal",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["tcp"],
  "provider_api": {
    "errors": { "tcp": "RawNetError" },
    "resources": {
      "TcpStream": { "capability": "tcp", "resource": "tcp_stream" }
    },
    "functions": {
      "connect_raw": {
        "capability": "tcp",
        "symbol": "stark_tcp_stream_connect"
      }
    }
  }
}"#,
        "fn main() { }\n",
    );

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    );

    // **Inverted (CD-250), and this is the SECOND of two tests that pinned the refusal.** CD-248
    // lifted it and updated the one in `c788_starkc_build.rs`, missing this one -- the same failure
    // as every other red in this stretch: changing behaviour without auditing what pinned the old
    // behaviour.
    //
    // The refusal existed because a resource obtained through a build could never be released. The
    // close arena and the Drop-terminator close both exist now (CD-237/CD-239/CD-240), and the
    // driver selects a close for every bound resource (CD-248).
    //
    // Narrow on purpose: whatever else this build does, it must not fail BECAUSE the signature
    // carries a resource. Executing the close is the lifecycle e2e's job, not a diagnostic's.
    let rendered = match &result {
        Ok(_) => String::new(),
        Err(error) => format!("{error:?}"),
    };
    assert!(
        !rendered.contains("resource-bearing provider signature"),
        "the categorical resource refusal is gone; got:\n{rendered}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn build_driver_selects_closes_for_bound_resource_nominals() {
    let root = fixture_root("driver-close-wiring");
    let _ = std::fs::remove_dir_all(&root);
    write_package(
        &root,
        r#"{
  "name": "c788_lifecycle_close_wiring",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["tcp"],
  "provider_api": {
    "errors": { "tcp": "RawNetError" },
    "resources": {
      "TcpStream": { "capability": "tcp", "resource": "tcp_stream" }
    },
    "functions": {}
  }
}"#,
        "fn main() { println(7); }\n",
    );

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            emit_rust: true,
            ..BuildCommandOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("resource nominal close wiring must build: {error:?}"));
    let output = std::process::Command::new(&result.artifact_path)
        .output()
        .unwrap_or_else(|error| panic!("run built binary: {error}"));
    assert!(
        output.status.success(),
        "built program failed; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), "7");

    let generated =
        std::fs::read_to_string(result.generated_rust.expect("generated rust retained"))
            .expect("read generated Rust");
    assert!(
        generated.contains("stark_tcp_stream_close"),
        "the driver must call ProviderLowering::select_closes so the selected close enters the \
         provider arena:\n{generated}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn lowering_carries_a_manually_selected_close_arena_into_mir() {
    let set = ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["tcp".into()],
    )
    .expect("tcp provider selects for host");
    let connect = set
        .resolve("tcp", "stark_tcp_stream_connect")
        .expect("connect resolves");
    let connect_sig = derive(
        "connect_raw",
        "tcp",
        &connect.function,
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
        &BTreeMap::from([("tcp".to_string(), "RawNetError".to_string())]),
    )
    .expect("connect signature derives");
    let layer = synthesize_with_resources(
        &[connect_sig],
        &BTreeMap::from([("tcp".to_string(), connect.status_binding.clone())]),
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
    )
    .expect("resource nominal and free connect binding synthesize");
    let source = format!("{}\nfn main() {{ }}\n", layer.source);
    let file = Arc::new(SourceFile::new("resource_main.stark", source));
    let (ast, parse_diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(parse_diags.is_empty(), "{parse_diags:#?}");
    let (hir, resolve_diags) = starkc::resolve::resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "{resolve_diags:#?}");
    let checked = starkc::typecheck::analyze(&hir, file.clone());
    assert!(
        checked
            .diagnostics
            .iter()
            .all(|d| d.severity != starkc::diag::Severity::Error),
        "{:#?}",
        checked.diagnostics
    );

    // The synthesized nominal really is in HIR as a zero-variant enum (CD-234). Lowering resolves
    // it by name; this only asserts it is there to resolve.
    assert!(
        hir.items
            .iter()
            .any(|item| matches!(item.kind, ItemKind::Enum { .. })),
        "the synthesized TcpStream enum nominal must be in HIR"
    );

    let mut providers = ProviderLowering::build_with_errors(
        &layer.bindings,
        &layer.error_variants,
        &BTreeMap::from([("connect_raw".to_string(), "RawNetError".to_string())]),
        |cap, symbol| set.resolve(cap, symbol).map_err(|e| format!("{e:?}")),
    )
    .expect("provider lowering builds");
    // CD-248: the nominal NAME is what the manifest supplies and what `select_closes` iterates;
    // lowering resolves it to an item id once `ProgramMeta` can read item names. Setting
    // `resource_items` directly (as this did) left `resource_nominal_names` empty, so selection
    // would iterate nothing and silently choose no close.
    providers
        .resource_nominal_names
        .insert("tcp_stream".to_string(), "TcpStream".to_string());
    providers
        .select_closes(|resource| {
            set.providers()[0]
                .metadata
                .functions
                .iter()
                .find(|f| f.is_close_for.as_deref() == Some(resource))
                .cloned()
                .map(|function| ValidatedProviderCall {
                    provider: set.providers()[0].metadata.identity.clone(),
                    capability: function.capability.clone(),
                    function,
                    target_triple: set.target().to_string(),
                    status_binding: set.providers()[0].status_binding.clone(),
                    provider_crate: set.providers()[0].crate_name.clone(),
                    provider_resource_types: set.providers()[0].metadata.resource_types.clone(),
                    provider_target_triples: set.providers()[0].metadata.target_triples.clone(),
                })
                .ok_or_else(|| format!("no close for {resource}"))
        })
        .expect("close selection succeeds");
    assert_eq!(
        providers.pending_closes.len(),
        1,
        "the driver records resource -> close id; lowering completes it once the nominal resolves"
    );

    let program =
        starkc::mir::lower::lower_program_with_providers(&hir, &checked.tables, file, &providers)
            .unwrap_or_else(|e| panic!("lowering with injected close arena succeeds: {}", e.what));
    assert_eq!(program.provider_closes.len(), 1);
    assert_eq!(program.provider_calls.len(), 2);
    assert!(
        program
            .types
            .host_resource_closes
            .contains_key(&program.provider_closes[0].resource),
        "lowering must make the close reachable through TypeContext for DropPlan"
    );
}

#[test]
fn host_resource_drop_emission_calls_the_selected_provider_close() {
    let resource = tcp_stream_ty();
    let close = tcp_call(
        "stark_tcp_stream_close",
        vec![AbiParam::HandleConsumed {
            resource_type: "tcp_stream".to_string(),
        }],
    );
    let mut types = TypeContext::default();
    types
        .host_resource_closes
        .insert(resource.clone(), mir::ProviderCallId(0));
    let program = MirProgram {
        files: vec![Arc::new(SourceFile::new("drop_resource.stark", ""))],
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
                    ty: resource.clone(),
                    kind: LocalKind::User("stream".to_string()),
                },
            ],
            blocks: vec![
                BasicBlock {
                    statements: Vec::new(),
                    terminator: (
                        Terminator::Drop {
                            place: Place::local(mir::LocalId(1)),
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
        // A11: the program carries its OWN resource bindings, so planning and verification need no
        // external registry lookup (`program_resource_registry`). Without this the planner cannot
        // resolve `tcp_stream` and reports `UnboundResourceType`.
        resource_bindings: vec![(
            "tcp_stream".to_string(),
            mir::HostResourceNominal::Item(ItemId(7)),
        )],
        provider_calls: vec![close],
        // The validated binding, not just the type-context entry: MIR-0034 requires every emitted
        // close to be one the five obligations actually checked, so a program carrying only the
        // type-context half is one the verifier would reject.
        provider_closes: vec![mir::ValidatedProviderClose {
            resource: resource.clone(),
            close: mir::ProviderCallId(0),
        }],
    };

    let emitted = emit_program::emit(
        &program,
        &build_versions(
            "rustc-test".to_string(),
            host_triple(),
            starkc::backend::generated_rust::Profile::Debug,
        ),
        &starkc::layout::TargetLayout::default(),
    )
    .expect("a host-resource Drop must emit its selected close")
    .main_rs;

    // **Upgraded from an `expect_err` (CD-240).** This test previously pinned the eager-default
    // boundary and carried a tripwire saying to upgrade it once `HostResource` stopped falling
    // through `is_copy`'s wildcard `Copy` arm. That is exactly what changed, so the boundary moved
    // and the assertion follows it.
    assert!(
        emitted.contains("stark_tcp_stream_close(__v.take_raw())"),
        "the Drop must call the SELECTED close, consuming the handle:\n{emitted}"
    );
    assert!(
        emitted.contains(".drop_with(|__v| unsafe"),
        "the close must go through slot liveness, so a dead slot closes nothing:\n{emitted}"
    );
    assert!(
        !TypeContext::default().is_copy(&resource),
        "a host resource must never be Copy: Copy is the licence to duplicate a handle, and two \
         owners of one resource close it twice"
    );
    assert!(
        emitted.contains("ValueSlot::dead()"),
        "the resource local must be declared dead, never default-materialised:\n{emitted}"
    );
}

#[test]
fn handle_out_emission_writes_the_slot_only_on_success() {
    let call = tcp_call(
        "stark_tcp_stream_connect",
        vec![
            AbiParam::BufferIn,
            AbiParam::HandleOut {
                resource_type: "tcp_stream".to_string(),
            },
        ],
    );
    let mut types = TypeContext::default();
    types
        .host_resource_closes
        .insert(tcp_stream_ty(), mir::ProviderCallId(1));
    let program = MirProgram {
        files: vec![Arc::new(SourceFile::new("handle_out.stark", ""))],
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
                    ty: MirTy::Ref {
                        mutable: false,
                        inner: Box::new(MirTy::Core(
                            starkc::hir::CoreType::Vec,
                            vec![MirTy::UInt8],
                        )),
                    },
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    ty: tcp_stream_ty(),
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
                                Operand::Copy(Place::local(mir::LocalId(1))),
                                Operand::Move(Place::local(mir::LocalId(2))),
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
        // A11: the program carries its OWN resource bindings, so planning and verification need no
        // external registry lookup (`program_resource_registry`). Without this the planner cannot
        // resolve `tcp_stream` and reports `UnboundResourceType`.
        resource_bindings: vec![(
            "tcp_stream".to_string(),
            mir::HostResourceNominal::Item(ItemId(7)),
        )],
        provider_calls: vec![
            call,
            tcp_call(
                "stark_tcp_stream_close",
                vec![AbiParam::HandleConsumed {
                    resource_type: "tcp_stream".to_string(),
                }],
            ),
        ],
        provider_closes: Vec::new(),
    };

    let emitted = emit_program::emit(
        &program,
        &build_versions(
            "rustc-test".to_string(),
            host_triple(),
            starkc::backend::generated_rust::Profile::Debug,
        ),
        &starkc::layout::TargetLayout::default(),
    )
    .expect("a HandleOut resource destination must emit")
    .main_rs;

    // **Upgraded from an `expect_err` (CD-240)**, per this test's own tripwire.
    //
    // The destination is declared DEAD and written only under the status-zero arm. That ordering is
    // the whole guarantee: a failed call leaves the slot dead, so the later implicit `Drop` finds
    // nothing to close and the program cannot close a handle the provider never issued.
    assert!(
        emitted.contains("ValueSlot::dead()"),
        "the handle destination must begin dead:\n{emitted}"
    );
    let zero_arm = emitted
        .find("0u32 =>")
        .expect("the status dispatch must have a success arm");
    let write = emitted
        .find(".write(")
        .expect("the destination must be written somewhere");
    assert!(
        write > zero_arm,
        "the handle must be written INSIDE the success arm, not before the status is known:\n{emitted}"
    );
    assert!(
        !TypeContext::default().is_copy(&tcp_stream_ty()),
        "a host resource must never be Copy"
    );
}
