//! **HC9 — a package may NAME a resource another package owns.**
//!
//! CD-360 ruled what a cross-provider transfer means and proved it lowers. It left one thing
//! unreachable from STARK: a package binding such a function has a derived signature whose first
//! parameter is a `TcpStream`, and `TcpStream` is `stark-net`'s nominal, not the binding package's.
//! Derivation failed with `UnboundResourceInSignature` — a transfer that was declarable in a
//! provider manifest and not in a package one.
//!
//! `provider_api.foreign_resources` closes that. The rules it must enforce are all here, because
//! each of the alternatives is a plausible design that produces a subtly broken program:
//!
//! | if the compiler instead | the program would |
//! | --- | --- |
//! | let the package bind `tcp_stream` as an ordinary resource | get a SECOND `enum TcpStream {}`, a distinct `ItemId`, and a handle it could not pass anywhere |
//! | infer the owner from the dependency graph | resolve a typo (`tcp_strem`) to nothing, far from its cause |
//! | allow a package alias it does not depend on | render a path into a package absent from the graph |
//!
//! The end-to-end proof that this works is `stark-tls`, which the package gate builds and runs.
//! What is pinned here is the RULES, which a passing gate does not exercise.

use starkc::package::Package;
use starkc::provider_abi::ScalarTy;
use starkc::provider_derive::{derive, DeriveError, DerivedTy};
use starkc::provider_synth::synthesize_with_resources;
use std::collections::BTreeMap;

mod common;

/// Writes a package manifest and parses it, returning the error text on refusal.
///
/// A real `../stark-net` sibling is created alongside it. `Package::from_manifest` checks that a
/// declared dependency path exists, so without one every case here fails on the dependency rather
/// than on the rule it is testing — and the foreign-resource check is specifically about whether an
/// alias IS a declared dependency, which cannot be tested if no dependency can be declared.
fn parse_manifest(body: &str) -> Result<Package, String> {
    let root = tempdir();
    let dir = root.join("pkg");
    std::fs::create_dir_all(dir.join("src")).expect("src must be creatable");
    std::fs::write(dir.join("starkpkg.json"), body).expect("manifest must be writable");
    std::fs::write(dir.join("src").join("lib.stark"), "").expect("entry must be writable");

    let net = root.join("stark-net");
    std::fs::create_dir_all(net.join("src")).expect("dependency src must be creatable");
    std::fs::write(
        net.join("starkpkg.json"),
        r#"{ "name": "stark-net", "version": "0.1.0", "entry": "src/lib.stark", "dependencies": {} }"#,
    )
    .expect("dependency manifest must be writable");
    std::fs::write(net.join("src").join("lib.stark"), "").expect("dependency entry");

    Package::from_manifest(&dir.join("starkpkg.json"))
}

/// A directory unique to this call.
///
/// **An atomic counter, not a timestamp.** `cargo test` runs these in parallel within one process,
/// and a nanosecond clock is not guaranteed to advance between two threads reading it — two tests
/// took the SAME directory, clobbered each other's manifests, and failed with a diagnostic about a
/// missing `package` key that neither manifest was missing. They passed in isolation, which is the
/// tell. A counter cannot collide.
fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let base = std::env::temp_dir().join(format!(
        "hc9-foreign-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    // Removed first: a leftover from a previous run in the same process id would otherwise supply
    // a stale manifest to a test that thinks it wrote its own.
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).expect("temp dir must be creatable");
    base
}

const DEPS: &str =
    r#""stark_net": { "package": "stark-net", "path": "../stark-net", "version": "0.1.0" }"#;

fn manifest_with(foreign: &str) -> String {
    format!(
        r#"{{
  "name": "stark-tls",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "capabilities": ["tls"],
  "dependencies": {{ {DEPS} }},
  "provider_api": {{
    "errors": {{ "tls": "RawTlsError" }},
    "resources": {{ "TlsStream": {{ "capability": "tls", "resource": "tls_stream" }} }},
    {foreign}
    "functions": {{
      "tls_stream_connect_raw": {{ "capability": "tls", "symbol": "stark_tls_stream_connect" }}
    }}
  }}
}}"#
    )
}

const VALID_FOREIGN: &str =
    r#""foreign_resources": { "tcp_stream": { "package": "stark_net", "nominal": "TcpStream" } },"#;

// -------------------------------------------------------------------------------------------
// Parsing
// -------------------------------------------------------------------------------------------

#[test]
fn a_foreign_resource_resolves_to_a_qualified_path_at_the_owning_package() {
    let package = parse_manifest(&manifest_with(VALID_FOREIGN)).expect("the manifest must parse");
    let foreign = &package.provider_api.foreign_resources;
    assert_eq!(foreign.len(), 1);
    assert_eq!(foreign[0].resource, "tcp_stream");
    assert_eq!(foreign[0].package, "stark_net");
    assert_eq!(foreign[0].nominal, "TcpStream");
    // The QUALIFIED form is what a derived signature renders. Unqualified would resolve to
    // whatever the binding package happens to have in scope -- or to nothing.
    assert_eq!(foreign[0].qualified_nominal(), "stark_net::TcpStream");
}

/// The alias must be one the package actually depends on. A path into a package absent from the
/// graph surfaces as an unresolved name inside generated source nobody wrote, which is the worst
/// place for a manifest typo to appear.
#[test]
fn a_foreign_resource_in_a_package_that_is_not_a_dependency_is_refused() {
    let error = parse_manifest(&manifest_with(
        r#""foreign_resources": { "tcp_stream": { "package": "stark_sockets", "nominal": "TcpStream" } },"#,
    ))
    .expect_err("an undeclared dependency must be refused");
    assert!(error.contains("stark_sockets"), "{error}");
    assert!(error.contains("not a dependency"), "{error}");
}

/// **The rule that prevents a second type with the same name.** Owning a resource synthesizes a
/// nominal; borrowing one references the owner's. Doing both would put two `ItemId`s behind one
/// resource name, and a handle produced through one could not be passed to the other.
#[test]
fn a_resource_declared_both_owned_and_foreign_is_refused() {
    let body = format!(
        r#"{{
  "name": "stark-tls",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "capabilities": ["tls"],
  "dependencies": {{ {DEPS} }},
  "provider_api": {{
    "errors": {{ "tls": "RawTlsError" }},
    "resources": {{
      "TlsStream": {{ "capability": "tls", "resource": "tls_stream" }},
      "MyTcpStream": {{ "capability": "tls", "resource": "tcp_stream" }}
    }},
    {VALID_FOREIGN}
    "functions": {{
      "tls_stream_connect_raw": {{ "capability": "tls", "symbol": "stark_tls_stream_connect" }}
    }}
  }}
}}"#
    );
    let error = parse_manifest(&body).expect_err("owned AND foreign must be refused");
    assert!(error.contains("tcp_stream"), "{error}");
    assert!(error.contains("both owned and foreign"), "{error}");
}

/// A Core resource is compiler-owned (CD-224). No package owns one, so no package can lend one.
#[test]
fn a_core_resource_cannot_be_borrowed_as_foreign() {
    let error = parse_manifest(&manifest_with(
        r#""foreign_resources": { "file": { "package": "stark_net", "nominal": "File" } },"#,
    ))
    .expect_err("a Core resource must be refused");
    assert!(error.contains("Core resource"), "{error}");
}

#[test]
fn a_foreign_entry_missing_its_package_or_nominal_is_refused_by_name() {
    for (body, missing) in [
        (
            r#""foreign_resources": { "tcp_stream": { "nominal": "TcpStream" } },"#,
            "package",
        ),
        (
            r#""foreign_resources": { "tcp_stream": { "package": "stark_net" } },"#,
            "nominal",
        ),
    ] {
        let error = parse_manifest(&manifest_with(body)).expect_err("an incomplete entry");
        assert!(
            error.contains(missing) && error.contains("tcp_stream"),
            "the diagnostic must name the missing key and the entry: {error}"
        );
    }
}

// -------------------------------------------------------------------------------------------
// Derivation and synthesis
// -------------------------------------------------------------------------------------------

fn connect_decl() -> starkc::provider_abi::FunctionDecl {
    use starkc::provider_abi::{AbiParam, FunctionDecl};
    FunctionDecl {
        name: "stark_tls_stream_connect".to_string(),
        capability: "tls".to_string(),
        params: vec![
            AbiParam::HandleConsumed {
                resource_type: "tcp_stream".to_string(),
            },
            AbiParam::BufferIn,
            AbiParam::ScalarIn(ScalarTy::U64),
            AbiParam::HandleOut {
                resource_type: "tls_stream".to_string(),
            },
        ],
        is_close_for: None,
        may_block: true,
    }
}

fn errors() -> BTreeMap<String, String> {
    BTreeMap::from([("tls".to_string(), "RawTlsError".to_string())])
}

fn owned() -> BTreeMap<String, String> {
    BTreeMap::from([("tls_stream".to_string(), "TlsStream".to_string())])
}

fn foreign() -> BTreeMap<String, String> {
    BTreeMap::from([("tcp_stream".to_string(), "stark_net::TcpStream".to_string())])
}

/// The consumed handle derives as an OWNED parameter typed at the owner's qualified nominal, and
/// the produced handle at this package's own.
#[test]
fn a_transfer_derives_with_the_owners_nominal_for_the_consumed_handle() {
    let mut both = owned();
    both.extend(foreign());
    let sig = derive(
        "tls_stream_connect_raw",
        "tls",
        &connect_decl(),
        &both,
        &errors(),
    )
    .expect("a transfer must derive");

    assert_eq!(
        sig.params[0],
        DerivedTy::OwnedResource {
            nominal: "stark_net::TcpStream".to_string()
        },
        "the consumed handle is by value at the OWNER's nominal: ownership genuinely transfers"
    );
    assert_eq!(
        sig.results[0],
        DerivedTy::OwnedResource {
            nominal: "TlsStream".to_string()
        }
    );
    assert!(
        sig.receiver.is_none(),
        "a free binding takes no receiver, so the transfer is not mistaken for a method on the \
         consumed type"
    );
}

/// Without the declaration, derivation still fails as it always did. The new mechanism widens what
/// can be expressed; it does not weaken the check.
#[test]
fn a_transfer_without_the_declaration_is_still_refused() {
    let error = derive(
        "tls_stream_connect_raw",
        "tls",
        &connect_decl(),
        &owned(),
        &errors(),
    )
    .expect_err("an undeclared foreign resource must still fail");
    assert!(matches!(
        error,
        DeriveError::UnboundResourceInSignature { ref resource, .. } if resource == "tcp_stream"
    ));
}

/// **The load-bearing separation.** Synthesis generates a nominal for what the package OWNS and
/// nothing for what it borrows. A generated `enum TcpStream {}` here would be a distinct type from
/// the net package's, spelled the same — the failure mode with no visible symptom until a handle
/// cannot be passed.
#[test]
fn synthesis_generates_the_owned_nominal_and_never_the_foreign_one() {
    let mut both = owned();
    both.extend(foreign());
    let sig = derive(
        "tls_stream_connect_raw",
        "tls",
        &connect_decl(),
        &both,
        &errors(),
    )
    .unwrap();

    let vocabularies = BTreeMap::from([(
        "tls".to_string(),
        starkc::provider_bind::StatusBinding::new(),
    )]);
    let layer = synthesize_with_resources(&[sig], &vocabularies, &owned(), &foreign())
        .expect("a transfer binding must synthesize");

    assert!(
        layer.source.contains("pub enum TlsStream { }"),
        "the owned nominal must be generated:\n{}",
        layer.source
    );
    assert!(
        !layer.source.contains("enum TcpStream"),
        "a foreign nominal must NOT be generated -- it already exists in the owning package, and a \
         second one is a second type:\n{}",
        layer.source
    );
    assert!(
        layer.source.contains("a0: stark_net::TcpStream"),
        "the parameter must be a QUALIFIED path at the owning package:\n{}",
        layer.source
    );
    assert!(
        !layer.resource_nominals.contains_key("tcp_stream"),
        "a foreign resource must not enter this package's nominal table: that table is what \
         resolves to an ItemId, and the owner's entry is the only correct one"
    );
}

/// Synthesis still refuses a nominal that is neither owned nor declared foreign, and the refusal
/// names BOTH routes — an author told only half of the fix does half of it.
#[test]
fn synthesis_refuses_a_nominal_that_is_neither_owned_nor_foreign() {
    let mut both = owned();
    both.extend(foreign());
    let sig = derive(
        "tls_stream_connect_raw",
        "tls",
        &connect_decl(),
        &both,
        &errors(),
    )
    .unwrap();

    let vocabularies = BTreeMap::from([(
        "tls".to_string(),
        starkc::provider_bind::StatusBinding::new(),
    )]);
    let error = synthesize_with_resources(&[sig], &vocabularies, &owned(), &BTreeMap::new())
        .expect_err("an undeclared nominal must be refused");
    assert!(error.contains("stark_net::TcpStream"), "{error}");
    assert!(error.contains("neither binds nor declares"), "{error}");
    assert!(error.contains("foreign_resources"), "{error}");
}

// -------------------------------------------------------------------------------------------
// The shipped provider set
// -------------------------------------------------------------------------------------------

/// The TLS provider is a built-in, and its manifest must be the only place its details live.
#[test]
fn the_tls_provider_ships_and_declares_its_consumption() {
    let provider = starkc::provider_registry::first_party()
        .into_iter()
        .find(|p| p.crate_name == "stark-tls-native")
        .expect("stark-tls-native must be in the built-in provider set");

    assert_eq!(provider.metadata.identity.name, "stark-std-tls");
    assert_eq!(provider.metadata.capabilities, vec!["tls"]);
    assert_eq!(provider.metadata.resource_types, vec!["tls_stream"]);
    assert_eq!(provider.crate_path, "stark-tls/native");

    let foreign = &provider.metadata.foreign_resources;
    assert_eq!(foreign.len(), 1);
    assert_eq!(foreign[0].provider, "stark-std-net");
    assert_eq!(foreign[0].resource, "tcp_stream");

    assert_eq!(
        starkc::provider_abi::validate(&provider.metadata),
        Ok(()),
        "the shipped TLS manifest must satisfy the ABI validator"
    );
}

/// `built_in_crate_location` is a lookup OVER the manifests (CD-364), so a new provider needs no
/// compiler-source change beyond its manifest. Asserted for the one provider added since.
#[test]
fn the_tls_provider_crate_is_located_from_its_manifest() {
    let root = std::path::Path::new("/repo");
    assert_eq!(
        starkc::provider_registry::built_in_crate_location("stark-tls-native", root),
        Some(root.join("stark-tls/native"))
    );
    assert!(starkc::provider_registry::known_capabilities()
        .iter()
        .any(|c| c == "tls"));
}
