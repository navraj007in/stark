//! WP-C7.8.8 step 8 — **the monotonic clock, reached from ordinary STARK source.**
//!
//! Every provider that has executed so far executed from **hand-built MIR**
//! (`a10_stark_time_e2e`, `c783_env_e2e`, `c784_file_e2e`, `c78_buffer_e2e`). Those proved the ABI,
//! the emission and the ownership rules — and they proved nothing at all about whether a STARK
//! program can reach a provider, because `lower_program` hard-coded `provider_calls: Vec::new()`.
//! That gap, not TCP, was the critical path (CD-220).
//!
//! This test writes a `.stark` program, compiles it through the ordinary front end, lowers it,
//! links `stark-time-native`, runs the binary and reads the number it printed. Nothing in the path
//! is hand-built: the function being called is synthesized from the manifest binding, and the call
//! becomes `Callee::Provider` in lowering like any other call becomes `Callee::Instance`.
//!
//! **The bar this has to clear:** a test that asserted MIR contained `Callee::Provider` would pass
//! on a program that does not link, and one that asserted it links would pass on a clock stuck at
//! zero. So the assertion is on the value the binary printed.

use starkc::mir::provider_lower::ProviderLowering;
use starkc::mir::{self, Callee, Terminator};
use starkc::provider_derive::derive;
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;
use starkc::provider_synth::synthesize;
use starkc::source::SourceFile;
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

/// The manifest binding, as a package would write it: **identity only** — a capability and a
/// symbol. No parameter types, no return shape, no ownership modes. Everything about the signature
/// is derived from validated provider metadata (CD-224).
fn binding() -> (String, String, String) {
    (
        "time::monotonic_now_ns".to_string(),
        "clock".to_string(),
        "stark_time_monotonic_now_ns".to_string(),
    )
}

/// Derives the signature and synthesizes the raw layer — steps 2 and 3, used as a library rather
/// than re-stated, so this test cannot pass against a synthesis its own suite would reject.
fn raw_layer() -> (String, BTreeMap<String, (String, String)>) {
    let (item_path, capability, symbol) = binding();

    let decl = provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == "stark-std-time")
        .expect("stark-time is a first-party provider")
        .metadata
        .functions
        .into_iter()
        .find(|f| f.name == symbol)
        .expect("the clock function is declared");

    let errors = BTreeMap::from([(capability.clone(), "RawTimeError".to_string())]);
    let sig = derive(&item_path, &capability, &decl, &BTreeMap::new(), &errors)
        .expect("the clock signature must derive");

    let layer = synthesize(&[sig]).expect("the clock layer must synthesize");
    (layer.source, layer.bindings)
}

/// The whole program: the generated raw layer, the package's raw error type, and an application
/// `main` that calls the bound function with **ordinary syntax** and prints the reading.
fn program_source(raw: &str) -> String {
    format!(
        "enum RawTimeError {{ Failed }}\n\
         {raw}\n\
         fn main() {{\n\
         \x20   match monotonic_now_ns() {{\n\
         \x20       Ok(ns) => {{ println(ns); }}\n\
         \x20       Err(_e) => {{ panic(\"the clock provider failed\") }}\n\
         \x20   }}\n\
         }}\n"
    )
}

/// **The proof.** Source → front end → lowering → generated Rust → link → execute.
#[test]
fn the_monotonic_clock_is_reachable_from_stark_source() {
    let (raw, bindings) = raw_layer();
    let source = program_source(&raw);

    // ---- ordinary front end. No provider vocabulary reaches it. ----
    let file = Arc::new(SourceFile::new("clock_main.stark", source.clone()));
    let (ast, parse_diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(
        parse_diags.is_empty(),
        "the program must parse:\n{source}\n{parse_diags:#?}"
    );
    let (hir, resolve_diags) = starkc::resolve::resolve(&ast, file.clone());
    assert!(
        resolve_diags.is_empty(),
        "the program must resolve:\n{source}\n{resolve_diags:#?}"
    );
    let checked = starkc::typecheck::analyze(&hir, file.clone());
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| d.message.clone())
        .collect();
    assert!(
        errors.is_empty(),
        "the program must typecheck:\n{source}\n{errors:#?}"
    );

    // ---- provider selection, BEFORE lowering (A10 §3). ----
    let (_, capability, _) = binding();
    let set = ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &[capability],
    )
    .expect("stark-time must select for the host target");

    let providers = ProviderLowering::build(&bindings, |cap, symbol| {
        set.resolve(cap, symbol).map_err(|e| format!("{e:?}"))
    })
    .expect("the binding must resolve to a validated call");
    assert_eq!(providers.arena.len(), 1);

    // ---- lowering. This is the step that did not exist. ----
    let program =
        starkc::mir::lower::lower_program_with_providers(&hir, &checked.tables, file, &providers)
            .unwrap_or_else(|e| panic!("lowering must succeed: {} at {:?}", e.what, e.span));

    assert_eq!(
        program.provider_calls.len(),
        1,
        "the resolved arena must reach the program"
    );

    // The call really is a provider call, emitted by lowering from source — not a runtime
    // intrinsic, and not hand-placed.
    let lowered_a_provider_call = program.bodies.iter().any(|b| {
        b.blocks.iter().any(|blk| {
            matches!(
                &blk.terminator.0,
                Terminator::Call {
                    callee: Callee::Provider(_),
                    ..
                }
            )
        })
    });
    assert!(
        lowered_a_provider_call,
        "lowering must have produced Callee::Provider from source"
    );

    // **The placeholder body must never reach MIR.** A synthesized binding's body is
    // `panic("provider binding not lowered")` — it exists only because Core v1 has no bodyless-`fn`
    // grammar. Lowering emits the provider call from the binding and never discovers the callee, so
    // the body is not lowered at all. If it ever were, a call reaching it would abort at runtime
    // instead of reading the clock, and the failure would look like a provider fault.
    let lowered_bodies: Vec<&str> = program
        .bodies
        .iter()
        .map(|b| b.instance.symbol.as_str())
        .collect();
    assert!(
        !lowered_bodies
            .iter()
            .any(|s| s.contains("monotonic_now_ns")),
        "the synthesized placeholder body must not be lowered; got {lowered_bodies:?}"
    );

    // ---- verification, emission, link, execute. ----
    let verified = mir::verify::verify_program(&program)
        .unwrap_or_else(|e| panic!("the lowered program must verify: {e:?}"));

    let target_dir = std::env::temp_dir().join(format!("stark-c788-src-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&target_dir);

    let mut provider_crates = BTreeMap::new();
    provider_crates.insert(
        "stark-time-native".to_string(),
        provider_registry::crate_location("stark-time-native", &repo_root())
            .expect("the provider crate must be locatable"),
    );

    let toolchain = starkc::native_toolchain::discover(None)
        .expect("a rust toolchain is required for this test");

    let artifact = starkc::backend::generated_rust::emit_native_debug_with_toolchain(
        &verified,
        &starkc::backend::generated_rust::NativeBuildOptions {
            target_dir: target_dir.clone(),
            profile: starkc::backend::generated_rust::Profile::Debug,
            ..Default::default()
        },
        &starkc::backend::generated_rust::NativeToolchainOptions {
            rustc: toolchain.rustc.clone(),
            cargo: toolchain.cargo.clone(),
            runtime_crate: toolchain.runtime_crate.clone(),
            provider_crates,
        },
    )
    .unwrap_or_else(|e| panic!("the native build must succeed: {e:?}"));

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
        .unwrap_or_else(|e| panic!("expected a nanosecond count, got {printed:?}: {e}"));

    // A clock stuck at zero would satisfy "it compiled, linked and ran". This is the assertion that
    // says the provider produced a reading and the success arm copied it out of the output slot.
    assert!(
        nanos > 0,
        "the monotonic clock returned {nanos}: the output slot was never written back"
    );

    let _ = std::fs::remove_dir_all(&target_dir);
}

/// A program that binds no provider gets an empty arena and no behaviour change — the property that
/// keeps this work off every other program's path.
#[test]
fn a_program_binding_no_provider_is_unaffected() {
    let file = Arc::new(SourceFile::new(
        "plain.stark",
        "fn main() { println(1); }".to_string(),
    ));
    let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(pd.is_empty(), "{pd:?}");
    let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{rd:?}");
    let checked = starkc::typecheck::analyze(&hir, file.clone());

    let program = starkc::mir::lower::lower_program(&hir, &checked.tables, file)
        .unwrap_or_else(|e| panic!("lowering: {}", e.what));

    assert!(program.provider_calls.is_empty());
    assert!(!program.bodies.iter().any(|b| b.blocks.iter().any(|blk| {
        matches!(
            &blk.terminator.0,
            Terminator::Call {
                callee: Callee::Provider(_),
                ..
            }
        )
    })));
}

/// The binding table drives lowering by **name**, so an item that is not bound stays an ordinary
/// call. Without this, adding a provider to a program could silently capture a user function that
/// happened to share a name.
#[test]
fn an_unbound_function_is_not_captured() {
    let bindings = BTreeMap::from([(
        "time::monotonic_now_ns".to_string(),
        (
            "clock".to_string(),
            "stark_time_monotonic_now_ns".to_string(),
        ),
    )]);
    let set = ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["clock".to_string()],
    )
    .expect("selects");
    let providers = ProviderLowering::build(&bindings, |c, s| {
        set.resolve(c, s).map_err(|e| format!("{e:?}"))
    })
    .expect("resolves");

    assert!(providers.call_for("monotonic_now_ns").is_some());
    assert!(providers.call_for("something_else").is_none());
    // Keyed on the leaf, since synthesis emits free functions.
    assert!(providers.call_for("time::monotonic_now_ns").is_none());
}
