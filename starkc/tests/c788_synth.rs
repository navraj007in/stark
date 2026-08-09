//! WP-C7.8.8 step 3 — the synthesized raw layer, and whether it is real STARK.
//!
//! The design claimed synthesized items would be **ordinary** — that name resolution, type checking
//! and visibility would need no special case (§3). This tests that literally: the generated text is
//! parsed and typechecked by the ordinary front end, not inspected as a string.
//!
//! A test that only asserted "the source contains `fn now_ns`" would pass on text that does not
//! compile, which is the class of defect the whole C7.8 e2e sequence kept finding.

use starkc::provider_abi::ScalarTy;
use starkc::provider_bind::StatusBinding;
use starkc::provider_derive::{derive, DerivedSignature, DerivedTy};
use starkc::provider_registry;
use starkc::provider_synth::synthesize;
use starkc::source::SourceFile;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A vocabulary for each capability the test's signatures use. `codes` empty means "no recoverable
/// status", which generates an uninhabited raw error type.
fn vocab(entries: &[(&str, &[(u32, &str)])]) -> BTreeMap<String, StatusBinding> {
    entries
        .iter()
        .map(|(cap, codes)| {
            let mut b = StatusBinding::new();
            for (code, name) in codes.iter() {
                b.declare(*code, *name);
            }
            (cap.to_string(), b)
        })
        .collect()
}

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

/// Compiles the raw layer plus an entry point, through the **ordinary** front end, and returns any
/// errors.
///
/// The layer declares its own raw error type now (§7.2 — the compiler owns it, since the manifest
/// carries no code→variant table), so nothing is prepended here. Prepending one would be a duplicate
/// definition and would also hide whether synthesis emitted the type at all.
fn compile(raw_layer: &str) -> Vec<String> {
    let program = format!("{raw_layer}\nfn main() {{ }}\n");
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
    starkc::typecheck::analyze(&hir)
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
    let layer = synthesize(&time_signatures(), &vocab(&[("clock", &[])])).expect("synthesizes");
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
    let layer = synthesize(&time_signatures(), &vocab(&[("clock", &[])])).expect("synthesizes");
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
    let layer = synthesize(&time_signatures(), &vocab(&[("clock", &[])])).expect("synthesizes");
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
    let layer = synthesize(&[], &BTreeMap::new()).expect("synthesizes");
    assert!(layer.source.is_empty());
    assert!(layer.bindings.is_empty());
}

/// Buffers and in/out slots reach source in their derived forms, so §11.1's caller-owned split is
/// visible in a signature a programmer could read.
#[test]
fn buffer_and_slot_forms_reach_the_generated_source() {
    let sig = derive(
        "env::var_fill",
        "environment-read",
        &decl("stark-std-env", "stark_env_var_fill"),
        &map(&[]),
        &map(&[("environment-read", "RawEnvError")]),
    )
    .expect("derives");
    let layer = synthesize(&[sig], &vocab(&[("environment-read", &[])])).expect("synthesizes");

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
    let errors = compile(&layer.source);
    assert!(
        errors.is_empty(),
        "the buffer raw layer must compile:\n{}\n\nerrors: {errors:#?}",
        layer.source
    );
}

// -------------------------------------------------- the limit, stated not hidden --

/// **Resource nominals cannot be synthesized by this mechanism, and the attempt is refused.**
///
/// Every source form that declares a nominal is constructible — `struct S;` and `struct S {}` both
/// admit a value at a use site — and a host resource must not be. Emitting one anyway would let a
/// program forge a handle no provider produced, and `from_raw_checked` would not catch it: the
/// `resource_type` would be whatever the forger wrote.
/// **Rewritten for CD-234/CD-235.** This asserted that ANY resource-bearing signature was refused,
/// which was true while `RESOURCE_SYNTHESIS_LIMIT` had no answer. CD-234 gave it one — a zero-variant
/// enum — so the refusal narrowed to a resource the package does not BIND, which is the case that
/// would otherwise generate source naming a type that does not exist.
#[test]
fn a_signature_touching_an_unbound_resource_is_refused() {
    let file_open = derive(
        "file::open_raw",
        "filesystem-read",
        &decl("stark-std-file", "stark_file_open"),
        &map(&[("file", "File")]),
        &map(&[("filesystem-read", "RawIoError")]),
    )
    .expect("derives");

    // No resource nominals bound, so `File` names nothing.
    let e = synthesize(&[file_open], &vocab(&[("filesystem-read", &[])]))
        .expect_err("a resource the package does not bind must be refused");
    assert!(e.contains("File"), "{e}");
    // HC9 widened the rule: a nominal is admissible if the package BINDS it or declares it FOREIGN.
    // The refusal must name both routes, or an author who declared neither is told to do half of
    // what would fix it.
    assert!(e.contains("neither binds nor declares"), "{e}");
    assert!(e.contains("foreign_resources"), "{e}");
    assert!(e.contains("file::open_raw"), "{e}");
}

/// A receiver is still refused, but for a narrower reason than before: associated placement needs an
/// `impl` block on the nominal, and emitting the item as a free function instead would silently
/// change the call shape a programmer writes. The resource TYPE is no longer the problem (CD-234).
#[test]
fn a_signature_with_a_receiver_is_refused() {
    let read = derive(
        "File::read_raw",
        "filesystem-read",
        &decl("stark-std-file", "stark_file_read"),
        &map(&[("file", "File")]),
        &map(&[("filesystem-read", "RawIoError")]),
    )
    .expect("derives");

    let e = synthesize(&[read], &vocab(&[("filesystem-read", &[])]))
        .expect_err("a receiver must be refused");
    assert!(e.contains("associated placement"), "{e}");
    assert!(e.contains("File::read_raw"), "{e}");
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
    assert!(synthesize(&time_signatures(), &vocab(&[("clock", &[])])).is_ok());
}

/// Generation is deterministic: the same signatures produce byte-identical source, so nothing about
/// iteration order reaches generated code or the build key.
#[test]
fn generation_is_deterministic() {
    let a = synthesize(&time_signatures(), &vocab(&[("clock", &[])])).expect("synthesizes");
    let b = synthesize(&time_signatures(), &vocab(&[("clock", &[])])).expect("synthesizes");
    assert_eq!(a, b);
    let _ = ScalarTy::U64;
}

// ------------------------------------------------- raw error types (§7.2) --

/// **An empty status vocabulary generates an UNINHABITED enum**, and that is the point.
///
/// A capability declaring no recoverable status gets a `RawTimeError` with no values, so the `Err`
/// arm of `Result<UInt64, RawTimeError>` cannot be constructed at all. The type system then says
/// exactly what Packet 1 §1.2 says: every nonzero status from that provider is a contract
/// violation, and no package code runs on it.
///
/// The vocabulary here is this test's own, not the registry's: `clock` used to be the real example
/// and stopped being one when `stark-time` gained live linkage and declared `ClockUnavailable` and
/// `OutOfRange`. The empty case remains a legal input to synthesis, so it is still worth pinning —
/// it just no longer has a first-party provider behind it.
#[test]
fn an_empty_vocabulary_generates_an_uninhabited_error_type() {
    let layer = synthesize(&time_signatures(), &vocab(&[("clock", &[])])).expect("synthesizes");
    assert!(
        layer.source.contains("enum RawTimeError { }"),
        "{}",
        layer.source
    );
    assert_eq!(
        layer.error_variants.get("RawTimeError"),
        Some(&BTreeMap::new()),
        "an uninhabited type has no code→variant entries"
    );
    assert!(compile(&layer.source).is_empty(), "{}", layer.source);
}

/// A declared vocabulary becomes one variant per status code, **code-ordered**, and the variant
/// index table matches the generated declaration order — the thing lowering builds `Err` from.
#[test]
fn a_declared_vocabulary_becomes_one_variant_per_code() {
    let sig = derive(
        "env::var_fill",
        "environment-read",
        &decl("stark-std-env", "stark_env_var_fill"),
        &map(&[]),
        &map(&[("environment-read", "RawEnvError")]),
    )
    .expect("derives");

    // Declared out of code order on purpose: the generated order must come from the code, not from
    // the order they were declared in.
    let layer = synthesize(
        &[sig],
        &vocab(&[("environment-read", &[(7, "TooLong"), (3, "NotFound")])]),
    )
    .expect("synthesizes");

    assert!(
        layer
            .source
            .contains("enum RawEnvError { NotFound, TooLong }"),
        "{}",
        layer.source
    );
    assert_eq!(
        layer.error_variants.get("RawEnvError"),
        Some(&BTreeMap::from([(3, 0), (7, 1)])),
        "code 3 is variant 0 and code 7 is variant 1, matching the declaration"
    );
    assert!(compile(&layer.source).is_empty(), "{}", layer.source);
}

/// Two capabilities may share a raw error type while they agree; a **disagreement is refused**.
/// Silently keeping one name would give a single status code two meanings in one enum.
#[test]
fn conflicting_names_for_one_status_code_are_refused() {
    let a = derive(
        "env::var_fill",
        "environment-read",
        &decl("stark-std-env", "stark_env_var_fill"),
        &map(&[]),
        &map(&[("environment-read", "Shared")]),
    )
    .expect("derives");
    let b = derive(
        "time::monotonic_now_ns",
        "clock",
        &decl("stark-std-time", "stark_time_monotonic_now_ns"),
        &map(&[]),
        &map(&[("clock", "Shared")]),
    )
    .expect("derives");

    let e = synthesize(
        &[a, b],
        &vocab(&[
            ("environment-read", &[(3, "NotFound")]),
            ("clock", &[(3, "Skewed")]),
        ]),
    )
    .expect_err("a code with two names must be refused");
    assert!(e.contains("Shared"), "{e}");
    assert!(e.contains('3'), "{e}");
}

/// A capability with no vocabulary entry at all is refused rather than defaulted to empty: "no
/// recoverable statuses" and "nobody told us the vocabulary" are different claims, and defaulting
/// would silently turn the second into the first.
#[test]
fn a_missing_vocabulary_is_refused() {
    let e = synthesize(&time_signatures(), &BTreeMap::new())
        .expect_err("a missing vocabulary must be refused");
    assert!(e.contains("clock"), "{e}");
}

/// Status 0 is success, so declaring it as a **recoverable error** is refused.
///
/// Tolerating it would fail twice over without naming the mistake: `ProviderBindingPlan::classify`
/// tests success first, so the declaration would be silently shadowed, and lowering would emit two
/// `SwitchInt` arms for the same value.
#[test]
fn declaring_status_zero_as_an_error_is_refused() {
    let e = synthesize(
        &time_signatures(),
        &vocab(&[("clock", &[(0, "NotReally")])]),
    )
    .expect_err("status 0 must not be declarable as a recoverable error");
    assert!(e.contains("SUCCESS"), "{e}");
    assert!(e.contains("RawTimeError"), "{e}");
}

/// **The real `stark-env` vocabulary, not a hand-made one.**
///
/// The registry declares its codes as qualified *public* error paths —
/// `"ProcessError::InvalidName"` — because a vocabulary names the package-facing error a status
/// corresponds to. `ProcessError` is the public type package code maps the raw result *to* (§7.2), so
/// the raw variant is the final segment.
///
/// This is the test that matters, because every hand-written vocabulary in this file uses bare names
/// and would never have exposed the qualified form. Emitting it verbatim would produce
/// `enum RawEnvError { ProcessError::InvalidName, … }`, which does not parse.
#[test]
fn the_real_env_vocabulary_generates_a_compilable_error_type() {
    let provider = provider_registry::first_party()
        .into_iter()
        .find(|p| p.metadata.identity.name == "stark-std-env")
        .expect("stark-env is a first-party provider");

    let sig = derive(
        "env::var_fill",
        "environment-read",
        &decl("stark-std-env", "stark_env_var_fill"),
        &map(&[]),
        &map(&[("environment-read", "RawEnvError")]),
    )
    .expect("derives");

    let vocabularies = BTreeMap::from([(
        "environment-read".to_string(),
        provider.status_binding.clone(),
    )]);
    let layer = synthesize(&[sig], &vocabularies).expect("the env layer must synthesize");

    // Final segments only, code-ordered.
    assert!(
        layer.source.contains(
            "enum RawEnvError { InvalidName, InvalidEncoding, BufferTooSmall, Unsupported }"
        ),
        "{}",
        layer.source
    );
    assert_eq!(
        layer.error_variants.get("RawEnvError"),
        Some(&BTreeMap::from([(1, 0), (2, 1), (3, 2), (4, 3)])),
        "each declared code maps to its variant index"
    );

    // And it must be real STARK. This is the assertion a string check could not make.
    let errors = compile(&layer.source);
    assert!(
        errors.is_empty(),
        "the env raw layer must compile:\n{}\n\nerrors: {errors:#?}",
        layer.source
    );
}

/// A vocabulary name whose final segment is not a legal identifier is **refused**, pointing at the
/// vocabulary rather than emitting source that fails to parse somewhere downstream.
#[test]
fn an_unusable_vocabulary_name_is_refused() {
    for bad in ["Process Error", "9Lives", "Err::", "Err::has-dash", ""] {
        let e = synthesize(&time_signatures(), &vocab(&[("clock", &[(1, bad)])]))
            .expect_err("an illegal variant name must be refused");
        assert!(
            e.contains("not a legal STARK variant name"),
            "for {bad:?}: {e}"
        );
    }
}

// ------------------------------------ CD-234: resource nominals as enums --

/// **The nominal is a zero-variant enum, and it compiles.**
///
/// CD-234's form. Opacity is structural: there is no variant to name and no struct-literal syntax, so
/// no expression and no pattern can manufacture a value — and no checker rule has to remember that.
#[test]
fn a_resource_nominal_is_a_zero_variant_enum() {
    let layer = starkc::provider_synth::synthesize_with_resources(
        &[],
        &BTreeMap::new(),
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
        &BTreeMap::new(),
    )
    .expect("synthesizes");

    assert!(
        layer.source.contains("enum TcpStream { }"),
        "{}",
        layer.source
    );
    assert_eq!(
        layer.resource_nominals.get("tcp_stream"),
        Some(&"TcpStream".to_string())
    );
    assert!(compile(&layer.source).is_empty(), "{}", layer.source);
}

/// **Nothing in STARK can construct or match one into existence.** This is the property the whole
/// mechanism rests on, so it is tested against the front end rather than argued from the grammar.
#[test]
fn the_nominal_cannot_be_constructed_or_matched_into_existence() {
    let layer = starkc::provider_synth::synthesize_with_resources(
        &[],
        &BTreeMap::new(),
        &BTreeMap::from([("tcp_stream".to_string(), "TcpStream".to_string())]),
        &BTreeMap::new(),
    )
    .expect("synthesizes");

    for attempt in [
        // a bare path, as a unit-struct-like value
        "fn forge() -> TcpStream { TcpStream }",
        // a struct literal
        "fn forge() -> TcpStream { TcpStream {} }",
        // a variant path that does not exist
        "fn forge() -> TcpStream { TcpStream::V0 }",
        // a call form
        "fn forge() -> TcpStream { TcpStream() }",
    ] {
        let program = format!("{}\n{attempt}\nfn main() {{ }}\n", layer.source);
        let errors = compile(&program);
        assert!(
            !errors.is_empty(),
            "a host-resource nominal must not be constructible, but this compiled:\n{attempt}"
        );
    }
}

/// A signature naming a resource the package does not bind is refused — otherwise the generated
/// source would reference a type that does not exist, surfacing as an unresolved name in code nobody
/// wrote.
#[test]
fn a_signature_naming_an_unbound_nominal_is_refused() {
    let sig = derive(
        "file::open_raw",
        "filesystem-read",
        &decl("stark-std-file", "stark_file_open"),
        &map(&[("file", "File")]),
        &map(&[("filesystem-read", "RawIoError")]),
    )
    .expect("derives");

    let e = starkc::provider_synth::synthesize_with_resources(
        &[sig],
        &vocab(&[("filesystem-read", &[])]),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
    .expect_err("an unbound nominal must be refused");
    assert!(e.contains("File"), "{e}");
    // HC9 widened the rule: a nominal is admissible if the package BINDS it or declares it
    // FOREIGN. The refusal must name both routes, or an author who declared neither is told to
    // do only half of what would fix it.
    assert!(e.contains("neither binds nor declares"), "{e}");
    assert!(e.contains("foreign_resources"), "{e}");
}

/// With the nominal bound, the same signature synthesizes and compiles — so the refusal above is
/// about the binding, not about resources being unsupported.
#[test]
fn a_bound_nominal_lets_a_resource_signature_synthesize() {
    let sig = derive(
        "file::open_raw",
        "filesystem-read",
        &decl("stark-std-file", "stark_file_open"),
        &map(&[("file", "File")]),
        &map(&[("filesystem-read", "RawIoError")]),
    )
    .expect("derives");

    let layer = starkc::provider_synth::synthesize_with_resources(
        &[sig],
        &vocab(&[("filesystem-read", &[])]),
        &BTreeMap::from([("file".to_string(), "File".to_string())]),
        &BTreeMap::new(),
    )
    .expect("synthesizes");

    assert!(layer.source.contains("enum File { }"), "{}", layer.source);
    assert!(compile(&layer.source).is_empty(), "{}", layer.source);
}
