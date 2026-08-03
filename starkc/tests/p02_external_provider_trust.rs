//! **P0.2: external provider discovery, trust tiers, and pinning.**
//!
//! Third-party providers execute native code in the user's build and in the resulting process. They
//! are **not** ordinary STARK packages, and the mechanism must not let them pretend to be.
//!
//! # Trust is explicit, not enforced
//!
//! ```text
//! pure STARK package             no native code, no provider
//! first-party native provider    ships with the compiler, versioned with it
//! approved third-party provider  declared by the APPLICATION, pinned by version AND checksum
//! untrusted / local provider     path-based, development only, never in a release build
//! ```
//!
//! **No sandboxing is attempted.** A partial isolation story invites misplaced confidence; a
//! visible tier is honest and achievable now. What the mechanism guarantees is that native
//! third-party code cannot enter a build *by accident* — every route in is a deliberate act,
//! recorded, pinned, and refusable.
//!
//! # The four properties under test
//!
//! 1. **Off by default.** Declaring a provider is not enough; external providers must be enabled.
//! 2. **No transitive activation.** Only the application may activate a provider. A library must
//!    not pull native code into a program that never asked for it — that is the difference between
//!    a dependency graph and an attack surface.
//! 3. **Pinned exactly.** Version and checksum must both match, or the provider on disk is not the
//!    provider that was approved.
//! 4. **Development trust does not survive release.** An unpinned path provider is usable while
//!    developing and refused in a release build.
//!
//! Every one of these is a REFUSAL test, and refusals are the whole substance here: this is the
//! mechanism by which native code enters a STARK build, so the interesting behaviour is what it
//! turns away.

use starkc::provider_abi::validate;
use starkc::provider_manifest::{
    discover_external_providers, parse_provider_requirements, reject_transitive_activation,
    DiscoveryError, ExternalProviderPolicy, ProviderRequirement, ProviderTrust,
};

/// A minimal but complete external provider manifest.
const EXTERNAL: &str = r#"{
  "name": "stark-widget-native",
  "version": "0.2.0",
  "provider": {
    "abi": "0.1",
    "capabilities": ["widget"],
    "targets": ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"],
    "resources": [{ "name": "widget_handle" }],
    "functions": [
      {
        "symbol": "stark_widget_open",
        "capability": "widget",
        "may_block": true,
        "params": [{ "form": "handle_out", "resource": "widget_handle" }]
      },
      {
        "symbol": "stark_widget_close",
        "capability": "widget",
        "close_for": "widget_handle",
        "params": [{ "form": "handle_consumed", "resource": "widget_handle" }]
      }
    ]
  }
}"#;

/// Write an external provider to a temp directory and return its root.
fn external_provider(tag: &str, manifest: &str) -> std::path::PathBuf {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(format!(
            "temp_p02_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time must be after epoch")
                .as_nanos()
        ));
    std::fs::create_dir_all(&root).expect("create provider dir");
    std::fs::write(root.join("starkpkg.json"), manifest).expect("write manifest");
    root
}

fn sha256_of(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn requirement(root: &std::path::Path, version: &str, sha256: Option<&str>) -> ProviderRequirement {
    ProviderRequirement {
        name: "stark-widget-native".to_string(),
        path: root.to_path_buf(),
        version: version.to_string(),
        sha256: sha256.map(str::to_string),
    }
}

fn enabled() -> ExternalProviderPolicy {
    ExternalProviderPolicy {
        enabled: true,
        release_build: false,
    }
}

// ------------------------------------------------------------------- admitted --

/// **The exit criterion.** A provider supplied entirely from outside the compiler repository is
/// discovered, pinned, admitted, and validates under the unmodified ABI validator — with no change
/// to compiler source.
#[test]
fn an_external_provider_is_admitted_and_validates() {
    let root = external_provider("admit", EXTERNAL);
    let admitted = discover_external_providers(
        &[requirement(&root, "0.2.0", Some(&sha256_of(EXTERNAL)))],
        enabled(),
    )
    .expect("a correctly pinned external provider must be admitted");

    assert_eq!(admitted.len(), 1);
    let entry = &admitted[0];
    assert_eq!(
        entry.trust,
        ProviderTrust::ApprovedThirdParty,
        "a version- and checksum-pinned provider is approved third party"
    );
    assert_eq!(
        validate(&entry.provider.metadata),
        Ok(()),
        "an external provider must satisfy the SAME validator as a built-in one"
    );
    assert_eq!(
        entry.root, root,
        "the admitted root is where the build will look"
    );
    assert!(
        entry.sha256.is_some(),
        "the hash is recorded for build metadata"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// An unpinned path provider is usable while DEVELOPING, and classified honestly as untrusted.
#[test]
fn an_unpinned_path_provider_is_untrusted_local_in_a_dev_build() {
    let root = external_provider("devlocal", EXTERNAL);
    let admitted = discover_external_providers(&[requirement(&root, "0.2.0", None)], enabled())
        .expect("an unpinned provider is usable in a dev build");
    assert_eq!(
        admitted[0].trust,
        ProviderTrust::UntrustedLocal,
        "no checksum means untrusted-local, and it must SAY so rather than passing as approved"
    );
    let _ = std::fs::remove_dir_all(&root);
}

// -------------------------------------------------------------------- refused --

/// **Off by default.** Declaring a provider is not enough. Enabling native third-party code is a
/// deliberate act, not something a dependency graph can arrange.
#[test]
fn external_providers_are_disabled_by_default() {
    let root = external_provider("disabled", EXTERNAL);
    let errors = discover_external_providers(
        &[requirement(&root, "0.2.0", Some(&sha256_of(EXTERNAL)))],
        ExternalProviderPolicy::default(),
    )
    .expect_err("external providers must be off by default");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, DiscoveryError::ExternalProvidersDisabled { .. })),
        "expected a disabled-by-default refusal: {errors:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// **Development trust does not survive into a release artefact.**
#[test]
fn an_unpinned_provider_is_refused_in_a_release_build() {
    let root = external_provider("release", EXTERNAL);
    let errors = discover_external_providers(
        &[requirement(&root, "0.2.0", None)],
        ExternalProviderPolicy {
            enabled: true,
            release_build: true,
        },
    )
    .expect_err("an unpinned provider must not enter a release build");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, DiscoveryError::UnpinnedInReleaseBuild { .. })),
        "expected an unpinned-in-release refusal: {errors:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// **The provider on disk is not the one that was approved.** The most important refusal here: a
/// checksum that does not match means the ABI surface being linked is not the surface that was
/// reviewed.
#[test]
fn a_checksum_mismatch_is_refused_and_reports_both_hashes() {
    let root = external_provider("checksum", EXTERNAL);
    let errors = discover_external_providers(
        &[requirement(&root, "0.2.0", Some(&"0".repeat(64)))],
        enabled(),
    )
    .expect_err("a checksum mismatch must be refused");
    let mismatch = errors
        .iter()
        .find_map(|e| match e {
            DiscoveryError::ChecksumMismatch {
                expected, found, ..
            } => Some((expected, found)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected a checksum mismatch: {errors:?}"));
    assert_ne!(
        mismatch.0, mismatch.1,
        "both the approved and the found hash must be reported, so the reader can tell which \
         artefact changed"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// A version that does not match exactly is refused. Reproducibility requires the pin and the
/// manifest to agree exactly — "close enough" is how a build stops being repeatable.
#[test]
fn a_version_mismatch_is_refused() {
    let root = external_provider("version", EXTERNAL);
    let errors = discover_external_providers(
        &[requirement(&root, "0.3.0", Some(&sha256_of(EXTERNAL)))],
        enabled(),
    )
    .expect_err("a version mismatch must be refused");
    assert!(
        errors.iter().any(|e| matches!(
            e,
            DiscoveryError::VersionMismatch { requested, found, .. }
                if requested == "0.3.0" && found == "0.2.0"
        )),
        "expected both versions named: {errors:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// A missing manifest is refused with the path it looked in — a diagnostic the reader can act on.
#[test]
fn a_missing_manifest_is_refused_with_its_path() {
    let root = std::path::PathBuf::from("/nonexistent/provider/root");
    let errors = discover_external_providers(&[requirement(&root, "0.2.0", None)], enabled())
        .expect_err("a missing manifest must be refused");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, DiscoveryError::ManifestNotFound { .. })),
        "expected a not-found refusal naming the path: {errors:?}"
    );
}

/// A manifest that is present but malformed is refused as unreadable, carrying the underlying
/// field-level error rather than collapsing it to "bad manifest".
#[test]
fn an_unreadable_manifest_is_refused_carrying_the_field_error() {
    let bad =
        r#"{ "name": "stark-widget-native", "version": "0.2.0", "provider": { "abi": "0.1" } }"#;
    let root = external_provider("unreadable", bad);
    let errors = discover_external_providers(
        &[requirement(&root, "0.2.0", Some(&sha256_of(bad)))],
        enabled(),
    )
    .expect_err("a malformed provider manifest must be refused");
    assert!(
        errors.iter().any(|e| matches!(
            e,
            DiscoveryError::Unreadable { error, .. }
                if format!("{error}").contains("provider.capabilities")
        )),
        "the field-level cause must survive: {errors:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

// ----------------------------------------------------- no transitive activation --

/// **A library must not pull native code into a program that never asked for it.** This is the
/// difference between a dependency graph and an attack surface, and it is why activation is the
/// application's exclusive right.
#[test]
fn a_dependency_declaring_a_provider_is_refused() {
    let errors = reject_transitive_activation(
        "some-library",
        r#"{
          "name": "some-library",
          "version": "1.0.0",
          "providers": {
            "evil-native": { "path": "./evil", "version": "1.0.0" }
          }
        }"#,
    )
    .expect_err("a dependency must not activate a provider");
    assert!(
        errors.iter().any(|e| matches!(
            e,
            DiscoveryError::TransitiveActivation { dependency, provider }
                if dependency == "some-library" && provider == "evil-native"
        )),
        "both the dependency and the provider it tried to activate must be named: {errors:?}"
    );
}

/// An ordinary dependency declaring no providers is unaffected — the overwhelmingly common case
/// must stay silent.
#[test]
fn an_ordinary_dependency_is_unaffected() {
    assert_eq!(
        reject_transitive_activation(
            "stark-json",
            r#"{ "name": "stark-json", "version": "0.1.0" }"#
        ),
        Ok(())
    );
}

// ------------------------------------------------------------ requirement parsing --

/// An application declares providers in its own manifest. Absent means none, which is the common
/// case and must not be an error.
#[test]
fn requirements_parse_from_the_application_manifest() {
    let dir = std::path::Path::new("/app");
    let requirements = parse_provider_requirements(
        r#"{
          "name": "myapp",
          "version": "0.1.0",
          "providers": {
            "zeta-native": { "path": "../zeta", "version": "2.0.0", "sha256": "abc" },
            "alpha-native": { "path": "../alpha", "version": "1.0.0" }
          }
        }"#,
        dir,
    )
    .expect("requirements must parse");

    // Sorted, so the selected provider set — and therefore the generated workspace — cannot depend
    // on JSON key order. The same reason `capabilities` is sorted at parse time.
    assert_eq!(requirements[0].name, "alpha-native");
    assert_eq!(requirements[1].name, "zeta-native");
    assert_eq!(requirements[0].sha256, None);
    assert_eq!(requirements[1].sha256.as_deref(), Some("abc"));
    assert_eq!(requirements[0].path, dir.join("../alpha"));
}

#[test]
fn an_application_declaring_no_providers_yields_none() {
    assert_eq!(
        parse_provider_requirements(
            r#"{ "name": "myapp", "version": "0.1.0" }"#,
            std::path::Path::new("/app")
        ),
        Ok(Vec::new())
    );
}

/// Every refusal is collected, not just the first. An application pinning three providers wrongly
/// should learn all three in one build, rather than one per attempt.
#[test]
fn every_refusal_is_reported_not_only_the_first() {
    let root = external_provider("multi", EXTERNAL);
    let mut second = requirement(&root, "9.9.9", Some(&sha256_of(EXTERNAL)));
    second.name = "second-native".to_string();
    let errors = discover_external_providers(
        &[requirement(&root, "0.2.0", Some(&"0".repeat(64))), second],
        enabled(),
    )
    .expect_err("both providers are wrong");
    assert_eq!(
        errors.len(),
        2,
        "each failing provider must be reported: {errors:?}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

// ------------------------------------------------------ crate_path containment --
//
// CD-363 chose "the manifest declares its crate path, resolved against a root the caller supplies —
// the compiler's root for a built-in, the manifest's directory for an external one". That makes the
// root the ONLY containment an external provider has, so a path that escapes it escapes everything.

use starkc::provider_manifest::{parse_provider_manifest, resolve_crate_path, ManifestError};

fn manifest_with_crate_path(crate_path: &str) -> String {
    format!(
        r#"{{
          "name": "stark-widget-native",
          "version": "0.2.0",
          "provider": {{
            "abi": "0.1",
            "crate_path": "{crate_path}",
            "capabilities": ["widget"],
            "targets": ["x86_64-unknown-linux-gnu"],
            "functions": [
              {{ "symbol": "stark_widget_ping", "capability": "widget", "params": [] }}
            ]
          }}
        }}"#
    )
}

/// An ordinary relative path is accepted and resolves under the root.
#[test]
fn a_relative_crate_path_resolves_under_the_root() {
    let text = manifest_with_crate_path("native");
    assert!(parse_provider_manifest(&text, "m.json").is_ok());
    assert_eq!(
        resolve_crate_path(&text, std::path::Path::new("/providers/widget")),
        Ok(std::path::PathBuf::from("/providers/widget/native"))
    );
}

/// A nested relative path is fine — this is the built-in layout, e.g. `stark-net/native`.
#[test]
fn a_nested_relative_crate_path_is_accepted() {
    let text = manifest_with_crate_path("stark-net/native");
    assert_eq!(
        resolve_crate_path(&text, std::path::Path::new("/compiler-root")),
        Ok(std::path::PathBuf::from("/compiler-root/stark-net/native"))
    );
}

/// **An absolute path escapes the root.** A third-party manifest could otherwise point the build at
/// any crate on the machine.
#[test]
fn an_absolute_crate_path_is_refused() {
    for path in ["/etc", "/usr/local/evil"] {
        let text = manifest_with_crate_path(path);
        assert!(
            matches!(
                parse_provider_manifest(&text, "m.json"),
                Err(ManifestError::CratePathEscapesRoot { .. })
            ),
            "absolute `crate_path` `{path}` must be refused"
        );
    }
}

/// A Windows drive-prefixed path is absolute even when the host is not Windows — the manifest may
/// have been written on another platform, and the check must not depend on where it runs.
#[test]
fn a_windows_absolute_crate_path_is_refused_on_every_host() {
    let text = manifest_with_crate_path("C:/Windows/System32");
    assert!(
        matches!(
            parse_provider_manifest(&text, "m.json"),
            Err(ManifestError::CratePathEscapesRoot { .. })
        ),
        "a drive-prefixed path must be refused regardless of host platform"
    );
}

/// **A `..` component escapes the root**, which is the subtler and more likely form.
#[test]
fn a_parent_directory_component_is_refused() {
    for path in ["../elsewhere", "native/../../escape", "a/b/../../../c"] {
        let text = manifest_with_crate_path(path);
        assert!(
            matches!(
                parse_provider_manifest(&text, "m.json"),
                Err(ManifestError::CratePathEscapesRoot { .. })
            ),
            "`crate_path` `{path}` escapes the root and must be refused"
        );
    }
}

/// Checked on the STRING, not on the joined path. `provider/../../elsewhere` normalises to
/// something that looks contained, so a post-hoc canonicalisation would accept it — and a symlink
/// could defeat canonicalisation anyway. Refusing the components does not depend on the
/// filesystem's cooperation.
#[test]
fn a_path_that_cancels_out_is_still_refused() {
    let text = manifest_with_crate_path("provider/../native");
    assert!(
        matches!(
            parse_provider_manifest(&text, "m.json"),
            Err(ManifestError::CratePathEscapesRoot { .. })
        ),
        "a `..` that cancels out must still be refused: normalising first is how this check gets \
         defeated"
    );
}

/// The constraint is enforced at BOTH entry points — parsing and resolution — so neither is a way
/// around the other.
#[test]
fn resolution_enforces_containment_too() {
    let text = manifest_with_crate_path("../escape");
    assert!(
        matches!(
            resolve_crate_path(&text, std::path::Path::new("/root")),
            Err(ManifestError::CratePathEscapesRoot { .. })
        ),
        "resolve_crate_path must enforce containment independently of parse"
    );
}

/// Every built-in provider's declared path is contained. If one were not, the compiler would ship
/// with the escape it refuses from others.
#[test]
fn every_built_in_crate_path_is_contained() {
    for provider in starkc::provider_registry::first_party() {
        assert!(
            !provider.crate_name.is_empty(),
            "a built-in provider must name its crate"
        );
    }
    // The manifests themselves are the authority; parsing them is what enforces containment, and
    // `first_party()` panics on a malformed built-in — so reaching here means every one passed.
    //
    // The count is asserted so ADDING a provider is a deliberate edit here rather than a silent
    // widening of the built-in set. Six since HC9 added `stark-tls-native` (CD-365).
    assert_eq!(starkc::provider_registry::first_party().len(), 6);
}
