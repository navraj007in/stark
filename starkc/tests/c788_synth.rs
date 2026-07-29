//! WP-C7.8.8 step 3 — the synthesized raw layer, and whether it is real STARK.
//!
//! The design claimed synthesized items would be **ordinary** — that name resolution, type checking
//! and visibility would need no special case (§3). This tests that literally: the generated text is
//! parsed and typechecked by the ordinary front end, not inspected as a string.
//!
//! A test that only asserted "the source contains `fn now_ns`" would pass on text that does not
//! compile, which is the class of defect the whole C7.8 e2e sequence kept finding.

use starkc::provider_abi::ScalarTy;
use starkc::provider_derive::{derive, DerivedSignature, DerivedTy};
use starkc::provider_registry;
use starkc::provider_synth::{synthesize, RESOURCE_SYNTHESIS_LIMIT};
use starkc::source::SourceFile;
use std::collections::BTreeMap;
use std::sync::Arc;

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn decl(provider: &str, function: &str) -> starkc::provider_abi::FunctionDecl {
    provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == provider)
        .expect("registered")
        .metadata
        .functions
        .into_iter()
        .find(|f| f.name == function)
        .expect("declared")
}

fn time_signatures() -> Vec<DerivedSignature> {
    ["stark_time_monotonic_now_ns", "stark_time_unix_now"]
        .iter()
        .map(|f| {
            derive(
                &format!("time::{}", f.trim_start_matches("stark_time_")),
                "clock",
                &decl("stark-std-time", f),
                &map(&[]),
                &map(&[("clock", "RawTimeError")]),
            )
            .expect("derives")
        })
        .collect()
}

/// Compiles the raw layer alongside a minimal error type and entry point, through the **ordinary**
/// front end, and returns any errors.
fn compile(raw_layer: &str) -> Vec<String> {
    let program = format!("enum RawTimeError {{ Failed }}\n{raw_layer}\nfn main() {{ }}\n");
    let file = Arc::new(SourceFile::new("synth.stark", program));

    let (ast, parse_diags) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    if !parse_diags.is_empty() {
        return parse_diags
            .iter()
            .map(|d| format!("parse: {d:?}"))
            .collect();
    }
    let (hir, resolve_diags) = starkc::resolve::resolve(&ast, file.clone());
    if !resolve_diags.is_empty() {
        return resolve_diags
            .iter()
            .map(|d| format!("resolve: {d:?}"))
            .collect();
    }
    starkc::typecheck::analyze(&hir, file)
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("typecheck: {}", d.message))
        .collect()
}

// ------------------------------------------------------------------- the claim --

/// **The step's whole claim.** The generated raw layer parses, resolves and typechecks as ordinary
/// STARK — no special case anywhere in the front end.
#[test]
fn the_synthesized_layer_is_ordinary_stark() {
    let layer = synthesize(&time_signatures()).expect("synthesizes");
    let errors = compile(&layer.source);
    assert!(
        errors.is_empty(),
        "the generated raw layer must compile:\n{}\n\nerrors: {errors:#?}",
        layer.source
    );
}

/// The derived shapes survive into source text: one out-slot is a plain result, two are a tuple.
/// This is CD-219's case reaching actual STARK.
#[test]
fn derived_shapes_reach_the_generated_source() {
    let layer = synthesize(&time_signatures()).expect("synthesizes");
    assert!(
        layer
            .source
            .contains("fn monotonic_now_ns() -> Result<UInt64, RawTimeError>"),
        "{}",
        layer.source
    );
    assert!(
        layer
            .source
            .contains("fn unix_now() -> Result<(Int64, UInt32), RawTimeError>"),
        "{}",
        layer.source
    );
}

/// The binding table carries capability and symbol for lowering, keyed by item path. It is a side
/// table because HIR is built by the ordinary parser and carries no provider vocabulary — the
/// binding is carried, not consulted, until step 6.
#[test]
fn the_binding_table_carries_capability_and_symbol() {
    let layer = synthesize(&time_signatures()).expect("synthesizes");
    assert_eq!(
        layer.bindings.get("time::monotonic_now_ns"),
        Some(&(
            "clock".to_string(),
            "stark_time_monotonic_now_ns".to_string()
        ))
    );
    assert_eq!(
        layer.bindings.get("time::unix_now"),
        Some(&("clock".to_string(), "stark_time_unix_now".to_string()))
    );
}

/// A package binding nothing synthesizes nothing — no source, no table, no trace.
#[test]
fn a_package_binding_nothing_synthesizes_nothing() {
    let layer = synthesize(&[]).expect("synthesizes");
    assert!(layer.source.is_empty());
    assert!(layer.bindings.is_empty());
}

/// Buffers and in/out slots reach source in their derived forms, so §11.1's caller-owned split is
/// visible in a signature a programmer could read.
#[test]
fn buffer_and_slot_forms_reach_the_generated_source() {
    let sig = derive(
        "env::var_fill",
        "process.env",
        &decl("stark-std-env", "stark_env_var_fill"),
        &map(&[]),
        &map(&[("process.env", "RawEnvError")]),
    )
    .expect("derives");
    let layer = synthesize(&[sig]).expect("synthesizes");

    assert!(
        layer
            .source
            .contains("fn var_fill(a0: &[UInt8], a1: &mut [UInt8]) -> Result<UInt64, RawEnvError>"),
        "{}",
        layer.source
    );

    // And the form must be real STARK, not merely the right text. `&[UInt8]` is the shape the whole
    // buffer capability rests on; if Core v1 does not admit it, that is a finding for step 3, not a
    // detail for whoever first compiles `stark-env`.
    let program = layer.source.replace("RawEnvError", "RawTimeError");
    let errors = compile(&program);
    assert!(
        errors.is_empty(),
        "the buffer raw layer must compile:\n{program}\n\nerrors: {errors:#?}"
    );
}

// -------------------------------------------------- the limit, stated not hidden --

/// **Resource nominals cannot be synthesized by this mechanism, and the attempt is refused.**
///
/// Every source form that declares a nominal is constructible — `struct S;` and `struct S {}` both
/// admit a value at a use site — and a host resource must not be. Emitting one anyway would let a
/// program forge a handle no provider produced, and `from_raw_checked` would not catch it: the
/// `resource_type` would be whatever the forger wrote.
#[test]
fn a_signature_touching_a_resource_is_refused() {
    let file_open = derive(
        "file::open_raw",
        "filesystem",
        &decl("stark-std-file", "stark_file_open"),
        &map(&[("file", "File")]),
        &map(&[("filesystem", "RawIoError")]),
    )
    .expect("derives");

    let e = synthesize(&[file_open]).expect_err("a resource result must be refused");
    assert!(e.contains(RESOURCE_SYNTHESIS_LIMIT), "{e}");
    assert!(e.contains("file::open_raw"), "{e}");
}

/// A receiver is refused for the same reason: associated placement needs the nominal to exist, and
/// emitting it as a free function would silently change the call shape a programmer writes.
#[test]
fn a_signature_with_a_receiver_is_refused() {
    let read = derive(
        "File::read_raw",
        "filesystem",
        &decl("stark-std-file", "stark_file_read"),
        &map(&[("file", "File")]),
        &map(&[("filesystem", "RawIoError")]),
    )
    .expect("derives");

    let e = synthesize(&[read]).expect_err("a receiver must be refused");
    assert!(e.contains(RESOURCE_SYNTHESIS_LIMIT), "{e}");
}

/// Scalar-only signatures — the whole `clock` capability — are unaffected by the limit, which is
/// why time is the first capability to prove the source path (CD-225's implementation order).
#[test]
fn scalar_only_capabilities_are_unaffected() {
    for sig in time_signatures() {
        assert!(
            !sig.params
                .iter()
                .chain(sig.results.iter())
                .any(|t| matches!(
                    t,
                    DerivedTy::SharedResource { .. } | DerivedTy::OwnedResource { .. }
                )),
            "clock must carry no resource"
        );
        assert!(sig.receiver.is_none());
    }
    assert!(synthesize(&time_signatures()).is_ok());
}

/// Generation is deterministic: the same signatures produce byte-identical source, so nothing about
/// iteration order reaches generated code or the build key.
#[test]
fn generation_is_deterministic() {
    let a = synthesize(&time_signatures()).expect("synthesizes");
    let b = synthesize(&time_signatures()).expect("synthesizes");
    assert_eq!(a, b);
    let _ = ScalarTy::U64;
}
