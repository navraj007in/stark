//! **P0.2: a provider declared by MANIFEST is indistinguishable from a hardcoded one.**
//!
//! > A provider can be supplied outside the compiler repository without modifying compiler source,
//! > while preserving ABI validation, reproducibility, target qualification, and explicit trust
//! > policy.
//!
//! # The property under test
//!
//! Discovery changes; nothing else does. A manifest is a different **source** for
//! `ProviderMetadata`, never a different **standard** for it — so the load path must produce a
//! `DeclaredProvider` that `provider_abi::validate` and `ProviderSet::select` treat exactly as they
//! treat a first-party entry, including refusing it for exactly the same reasons.
//!
//! That is why the loader deliberately does **not** validate: if it screened its input, a
//! manifest-declared provider could pass a different bar than a hardcoded one, and the two
//! mechanisms would drift. Loading and validation stay separate, and the tests below check both
//! halves of that — a good manifest reaching `Ok(())`, and a bad one reaching the SAME violation a
//! hardcoded provider would.
//!
//! # Why this matters beyond databases
//!
//! Before this, every native capability required a change to `provider_registry.rs`. That made
//! providers compiler-integrated extensions rather than an ecosystem mechanism: provider versioning
//! welded to compiler releases, no external publication, and implicit trust because we wrote
//! everything that existed.
//!
//! It also made an ABI field addition a breaking change in every provider repository — which is not
//! hypothetical: adding CD-360's `foreign_resources` broke all five provider crates, exactly as
//! `DeclaredProvider`'s own doc had predicted. A manifest-declared provider has no
//! `ProviderMetadata` struct literal at all.

use starkc::provider_abi::{validate, AbiParam, AbiViolation};
use starkc::provider_manifest::{parse_provider_manifest, ManifestError};

/// A complete, well-formed provider manifest — the shape an external package would ship.
const POSTGRES: &str = r#"{
  "name": "stark-postgres-native",
  "version": "0.3.1",
  "provider": {
    "abi": "0.1",
    "identity": "stark-db-postgres",
    "crate": "native",
    "capabilities": ["stark.db.postgres"],
    "targets": ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"],
    "resources": [{ "name": "pg_connection" }],
    "status": { "1": "ConnectionRefused", "2": "AuthFailed" },
    "functions": [
      {
        "symbol": "stark_pg_connect",
        "capability": "stark.db.postgres",
        "may_block": true,
        "params": [
          { "form": "buffer_in" },
          { "form": "handle_out", "resource": "pg_connection" }
        ]
      },
      {
        "symbol": "stark_pg_query",
        "capability": "stark.db.postgres",
        "may_block": true,
        "params": [
          { "form": "handle_borrowed", "resource": "pg_connection" },
          { "form": "buffer_in" },
          { "form": "scalar_out", "type": "u64" }
        ]
      },
      {
        "symbol": "stark_pg_connection_close",
        "capability": "stark.db.postgres",
        "close_for": "pg_connection",
        "params": [{ "form": "handle_consumed", "resource": "pg_connection" }]
      }
    ]
  }
}"#;

fn load(text: &str) -> starkc::provider_resolve::DeclaredProvider {
    parse_provider_manifest(text, "test-manifest.json").expect("manifest must parse")
}

/// **The exit-criterion test.** A provider defined entirely by manifest — no compiler source
/// change — validates under the unmodified ABI validator.
#[test]
fn a_manifest_declared_provider_validates() {
    let provider = load(POSTGRES);
    assert_eq!(
        validate(&provider.metadata),
        Ok(()),
        "a manifest-declared provider must satisfy the SAME validator as a hardcoded one"
    );
}

/// Every field survives the round trip with the meaning the manifest gave it.
#[test]
fn the_manifest_maps_onto_provider_metadata_faithfully() {
    let provider = load(POSTGRES);
    let metadata = &provider.metadata;

    // Identity is separate from the package name on purpose: a package may ship under a
    // distribution name while declaring a stable ABI identity, and resource identity is structural
    // over that identity (CD-360).
    assert_eq!(metadata.identity.name, "stark-db-postgres");
    assert_eq!(provider.crate_name, "native");
    assert_eq!(metadata.identity.semver, (0, 3, 1));
    assert_eq!(metadata.identity.abi_version, "0.1");

    assert_eq!(metadata.capabilities, vec!["stark.db.postgres"]);
    assert_eq!(metadata.target_triples.len(), 2);
    assert_eq!(metadata.resource_types, vec!["pg_connection"]);
    assert_eq!(metadata.functions.len(), 3);

    let connect = &metadata.functions[0];
    assert!(connect.may_block, "`may_block` must survive the round trip");
    assert!(matches!(
        connect.params.as_slice(),
        [AbiParam::BufferIn, AbiParam::HandleOut { .. }]
    ));

    let close = &metadata.functions[2];
    assert_eq!(close.is_close_for.as_deref(), Some("pg_connection"));
    assert!(
        !close.may_block,
        "`may_block` must default to false when absent, not to true"
    );

    // The status vocabulary is a PACKAGE concern the ABI deliberately does not carry, so it rides
    // on `DeclaredProvider` rather than on `ProviderMetadata`.
    let codes: Vec<(u32, String)> = provider
        .status_binding
        .declared_codes()
        .map(|(c, n)| (*c, n.clone()))
        .collect();
    assert_eq!(
        codes,
        vec![
            (1, "ConnectionRefused".to_string()),
            (2, "AuthFailed".to_string())
        ]
    );

    assert_eq!(
        provider.origin, "test-manifest.json",
        "origin must carry provenance for diagnostics"
    );
}

/// An absent `identity` defaults to the package name, and an absent `crate` likewise — the common
/// case for a provider that ships as one package.
#[test]
fn identity_and_crate_default_to_the_package_name() {
    let provider = load(
        r#"{
          "name": "stark-widget-native",
          "version": "1.0.0",
          "provider": {
            "abi": "0.1",
            "capabilities": ["widget"],
            "targets": ["x86_64-unknown-linux-gnu"],
            "functions": [
              { "symbol": "stark_widget_ping", "capability": "widget", "params": [] }
            ]
          }
        }"#,
    );
    assert_eq!(provider.metadata.identity.name, "stark-widget-native");
    assert_eq!(provider.crate_name, "stark-widget-native");
    // An empty status vocabulary is MEANINGFUL — every nonzero status is a contract violation — so
    // an absent key must not be an error.
    assert_eq!(provider.status_binding.declared_codes().count(), 0);
}

/// **CD-360 travels through the manifest.** `consumes` is how an external provider declares a
/// cross-provider transfer, and it must reach `foreign_resources` unchanged.
#[test]
fn a_consuming_transfer_is_declarable_in_a_manifest() {
    let provider = load(
        r#"{
          "name": "stark-tls-native",
          "version": "0.1.0",
          "provider": {
            "abi": "0.1",
            "capabilities": ["tls"],
            "targets": ["x86_64-unknown-linux-gnu"],
            "resources": [{ "name": "tls_stream" }],
            "consumes": [{ "provider": "stark-std-net", "resource": "tcp_stream" }],
            "functions": [
              {
                "symbol": "stark_tls_client_connect",
                "capability": "tls",
                "may_block": true,
                "params": [
                  { "form": "handle_consumed", "resource": "tcp_stream" },
                  { "form": "buffer_in" },
                  { "form": "handle_out", "resource": "tls_stream" }
                ]
              },
              {
                "symbol": "stark_tls_stream_close",
                "capability": "tls",
                "close_for": "tls_stream",
                "params": [{ "form": "handle_consumed", "resource": "tls_stream" }]
              }
            ]
          }
        }"#,
    );
    assert_eq!(provider.metadata.foreign_resources.len(), 1);
    assert_eq!(provider.metadata.foreign_resources[0].provider, "stark-std-net");
    assert_eq!(provider.metadata.foreign_resources[0].resource, "tcp_stream");
    assert_eq!(
        validate(&provider.metadata),
        Ok(()),
        "the HC9 shape must be declarable entirely by manifest"
    );
}

// ------------------------------------------------------------------ same bar --

/// **The load path must not become a second, weaker gate.** A manifest that parses but declares an
/// invalid ABI reaches the SAME violation a hardcoded provider would — the loader screens nothing.
#[test]
fn a_parseable_manifest_with_an_invalid_abi_is_refused_by_the_validator() {
    let provider = load(
        r#"{
          "name": "stark-leaky-native",
          "version": "0.1.0",
          "provider": {
            "abi": "0.1",
            "capabilities": ["leak"],
            "targets": ["x86_64-unknown-linux-gnu"],
            "resources": [{ "name": "leaky_handle" }],
            "functions": [
              {
                "symbol": "stark_leaky_open",
                "capability": "leak",
                "params": [{ "form": "handle_out", "resource": "leaky_handle" }]
              }
            ]
          }
        }"#,
    );
    let violations = validate(&provider.metadata).expect_err("a closeless resource must be refused");
    assert!(
        violations.iter().any(|v| matches!(
            v,
            AbiViolation::ResourceTypeMissingClose { resource_type } if resource_type == "leaky_handle"
        )),
        "a manifest-declared resource with no close must be refused exactly as a hardcoded one is: \
         {violations:?}"
    );
}

// -------------------------------------------------------------- diagnostics --

/// A manifest is written by someone outside this repository, so a diagnostic that does not name the
/// offending key is unusable to them. Each of these names its field.
#[test]
fn a_missing_required_field_names_the_field() {
    let error = parse_provider_manifest(
        r#"{ "name": "x", "version": "1.0.0", "provider": { "abi": "0.1" } }"#,
        "m.json",
    )
    .expect_err("a manifest without capabilities must be refused");
    assert!(
        matches!(&error, ManifestError::MissingField { path } if path == "provider.capabilities"),
        "expected the missing field to be named, got: {error}"
    );
}

/// An unknown parameter form is refused rather than skipped. Skipping would silently produce a
/// DIFFERENT ABI than the author wrote, and the mismatch would surface as a link error or worse.
#[test]
fn an_unknown_parameter_form_is_refused_and_names_the_function() {
    let error = parse_provider_manifest(
        r#"{
          "name": "x", "version": "1.0.0",
          "provider": {
            "abi": "0.1", "capabilities": ["c"], "targets": ["t"],
            "functions": [
              { "symbol": "stark_x", "capability": "c", "params": [{ "form": "handle_maybe" }] }
            ]
          }
        }"#,
        "m.json",
    )
    .expect_err("an unknown parameter form must be refused");
    assert!(
        matches!(&error, ManifestError::UnknownParamForm { function, form }
                 if function == "stark_x" && form == "handle_maybe"),
        "expected the form and function named, got: {error}"
    );
}

/// A handle form with no `resource` is refused: a handle must say which resource type it carries,
/// or §13's wrong-resource-type rule has nothing to check against.
#[test]
fn a_handle_without_a_resource_is_refused() {
    let error = parse_provider_manifest(
        r#"{
          "name": "x", "version": "1.0.0",
          "provider": {
            "abi": "0.1", "capabilities": ["c"], "targets": ["t"],
            "functions": [
              { "symbol": "stark_x", "capability": "c", "params": [{ "form": "handle_out" }] }
            ]
          }
        }"#,
        "m.json",
    )
    .expect_err("a handle without a resource must be refused");
    assert!(
        matches!(&error, ManifestError::HandleWithoutResource { function, .. } if function == "stark_x"),
        "expected a handle-without-resource error, got: {error}"
    );
}

/// Malformed JSON fails as malformed, not as a missing field — the reader needs to know the file is
/// broken, not hunt for a key that was never reachable.
#[test]
fn malformed_json_is_reported_as_malformed() {
    let error = parse_provider_manifest("{ not json", "m.json")
        .expect_err("malformed JSON must be refused");
    assert!(
        matches!(error, ManifestError::Malformed(_)),
        "expected a malformed-JSON error, got: {error}"
    );
}

// ---------------------------------------------------- one mechanism, not two --

/// **The P0.2 structural claim.** `first_party()` is no longer a parallel authority: the built-in
/// providers are declared in manifests and parsed by the same loader an external provider uses.
///
/// "First party" is now a TRUST and DEFAULT classification, not an implementation path. This is the
/// permanent guard that replaced the one-shot migration-equivalence test — every built-in manifest
/// must parse AND satisfy the unmodified validator, so a malformed built-in fails here rather than
/// as "capability unsupplied" somewhere far from the cause.
#[test]
fn every_built_in_provider_loads_and_validates_through_the_manifest_path() {
    let providers = starkc::provider_registry::first_party();
    assert_eq!(
        providers.len(),
        5,
        "the built-in set must still supply clock, env, file, net and random"
    );
    for provider in &providers {
        assert_eq!(
            validate(&provider.metadata),
            Ok(()),
            "built-in provider `{}` must satisfy the same validator as an external one",
            provider.crate_name
        );
        assert!(
            provider.origin.starts_with("built-in:"),
            "a built-in provider's origin must say so, since origin is what a diagnostic quotes: \
             got `{}`",
            provider.origin
        );
        assert!(
            !provider.metadata.capabilities.is_empty(),
            "`{}` supplies no capability",
            provider.crate_name
        );
    }
}

/// The built-in set still supplies exactly the capabilities the compiler's own packages require. A
/// migration that quietly dropped one would otherwise surface as an unrelated resolution failure.
#[test]
fn the_built_in_set_supplies_the_expected_capabilities() {
    let mut capabilities: Vec<String> = starkc::provider_registry::first_party()
        .iter()
        .flat_map(|p| p.metadata.capabilities.clone())
        .collect();
    capabilities.sort();
    assert_eq!(
        capabilities,
        vec![
            "clock".to_string(),
            "dns".to_string(),
            "filesystem".to_string(),
            "process.args".to_string(),
            "process.env".to_string(),
            "random".to_string(),
            "tcp".to_string(),
        ]
    );
}
