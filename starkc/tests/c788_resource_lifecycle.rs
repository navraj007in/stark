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
fn build_driver_still_refuses_resource_lifecycle_at_the_source_boundary() {
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

    let error = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    )
    .expect_err("resource-bearing provider APIs must not pass the driver until lifecycle lands");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("resource-bearing provider signature"),
        "{rendered}"
    );
    assert!(
        rendered.contains("close arena") && rendered.contains("Drop-terminator close"),
        "the diagnostic must name the exact current lifecycle blockers: {rendered}"
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

    let stream_item = hir
        .items
        .iter()
        .enumerate()
        .find_map(|(idx, item)| {
            matches!(item.kind, ItemKind::Enum { .. }).then_some(ItemId(idx as u32))
        })
        .expect("the synthesized TcpStream enum nominal must be in HIR");

    let mut providers = ProviderLowering::build_with_errors(
        &layer.bindings,
        &layer.error_variants,
        &BTreeMap::from([("connect_raw".to_string(), "RawNetError".to_string())]),
        |cap, symbol| set.resolve(cap, symbol).map_err(|e| format!("{e:?}")),
    )
    .expect("provider lowering builds");
    providers
        .resource_items
        .insert("tcp_stream".to_string(), stream_item);
    let closes = providers
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
    assert_eq!(closes.len(), 1);

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
fn host_resource_drop_emission_is_currently_blocked_by_eager_local_materialisation() {
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
        resource_bindings: Vec::new(),
        provider_calls: vec![close],
        // The validated binding, not just the type-context entry: MIR-0034 requires every emitted
        // close to be one the five obligations actually checked, so a program carrying only the
        // type-context half is one the verifier would reject.
        provider_closes: vec![mir::ValidatedProviderClose {
            resource: resource.clone(),
            close: mir::ProviderCallId(0),
        }],
    };

    let error = emit_program::emit(
        &program,
        &build_versions(
            "rustc-test".to_string(),
            host_triple(),
            starkc::backend::generated_rust::Profile::Debug,
        ),
        &starkc::layout::TargetLayout::default(),
    )
    .err()
    .expect("host-resource locals currently reach default materialisation before Drop emits");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("has no default value") && rendered.contains("forged handle"),
        "{rendered}"
    );
    assert!(
        TypeContext::default().is_copy(&resource),
        "current failing point changed: HostResource no longer falls through the wildcard Copy arm; \
         this test should be upgraded to assert close emission"
    );
}

#[test]
fn handle_out_emission_is_currently_blocked_by_eager_local_materialisation() {
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
        resource_bindings: Vec::new(),
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

    let error = emit_program::emit(
        &program,
        &build_versions(
            "rustc-test".to_string(),
            host_triple(),
            starkc::backend::generated_rust::Profile::Debug,
        ),
        &starkc::layout::TargetLayout::default(),
    )
    .err()
    .expect("HandleOut resource locals currently reach default materialisation first");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("has no default value") && rendered.contains("successful HandleOut"),
        "{rendered}"
    );
    assert!(
        TypeContext::default().is_copy(&tcp_stream_ty()),
        "current failing point changed: HostResource no longer falls through the wildcard Copy arm; \
         this test should be upgraded to assert success-only HandleOut writeback"
    );
}
