//! WP-C7.8 step 3 (CD-212) — package-declared capability requirements and provider selection.
//!
//! Packet 5's admission rule, tested: **a provider is linked if and only if some package declared
//! its capability by name.** No discovery, no fallback, no priority rule — and `stark build` fails
//! when a required capability has no unique selected provider.
//!
//! The negative that matters most is the *absence* case: a program declaring no capabilities must
//! link no provider and build exactly as it did before C7.8 existed. A regression there would not
//! fail loudly; it would silently start linking code into every binary.

use starkc::package::Package;
use starkc::provider_registry;
use starkc::provider_resolve::{ProviderSet, ResolveError};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

#[path = "support/paths.rs"]
mod paths;
use paths::repo_provider_root;

const LINUX: &str = "x86_64-unknown-linux-gnu";

/// Unique per call so parallel test threads cannot collide, following the convention in
/// `tests/c6_package.rs` rather than adding a dependency for three temporary files.
fn temp_manifest(body: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "c78-manifest-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(dir.join("src")).expect("temp dir");
    // The manifest validates that its entry file exists, so the fixture has to be a real package
    // rather than a lone JSON file.
    std::fs::write(dir.join("src").join("main.stark"), "fn main() {}\n").expect("entry");
    let path = dir.join("starkpkg.json");
    std::fs::write(&path, body).expect("write manifest");
    path
}

fn manifest_with(body: &str) -> Package {
    Package::from_manifest(&temp_manifest(body)).expect("manifest must parse")
}

// -------------------------------------------------------- manifest parsing --

/// A manifest with no `capabilities` key declares none. This is the overwhelmingly common case and
/// the one a regression would silently break.
#[test]
fn a_manifest_without_capabilities_declares_none() {
    let p = manifest_with(r#"{"name":"app","version":"0.1.0","entry":"src/main.stark"}"#);
    assert!(p.capabilities.is_empty());
}

/// Declared capabilities are sorted and deduplicated at parse time, so the requirement set — and
/// therefore the selected providers, and therefore the generated manifest — cannot depend on JSON
/// key order or on a package listing something twice.
#[test]
fn capabilities_are_sorted_and_deduplicated() {
    let p = manifest_with(
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark",
            "capabilities":["net","clock","env","clock"]}"#,
    );
    assert_eq!(p.capabilities, vec!["clock", "env", "net"]);
}

/// A malformed declaration is rejected rather than ignored. A silently dropped capability would
/// surface much later as a build failure naming a requirement nobody could find.
#[test]
fn a_malformed_capability_list_is_rejected() {
    for body in [
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","capabilities":"clock"}"#,
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","capabilities":[1]}"#,
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","capabilities":[""]}"#,
    ] {
        let path = temp_manifest(body);
        assert!(
            Package::from_manifest(&path).is_err(),
            "must reject: {body}"
        );
    }
}

// ------------------------------------------------------------- selection --

/// The first-party set supplies `clock`, and selecting it resolves `stark-time`.
#[test]
fn a_declared_capability_selects_its_provider() {
    let set = ProviderSet::select(
        provider_registry::first_party(),
        LINUX,
        &["clock".to_string()],
    )
    .expect("clock must be available on a Tier-1 target");

    assert_eq!(set.providers().len(), 1);
    assert_eq!(set.providers()[0].crate_name, "stark-time-native");
    assert!(set.resolve("clock", "stark_time_monotonic_now_ns").is_ok());
}

/// **Packet 5's core rule.** A capability nothing supplies fails the build, and the diagnostic says
/// what *is* available rather than only what is missing.
#[test]
fn an_unsatisfiable_capability_fails_selection() {
    let errors = ProviderSet::select(
        provider_registry::first_party(),
        LINUX,
        &["telepathy".to_string()],
    )
    .expect_err("an unknown capability must fail the build");

    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::CapabilityUnavailable { .. })),
        "{errors:#?}"
    );
    assert!(
        provider_registry::known_capabilities().contains(&"clock".to_string()),
        "the diagnostic's 'known capabilities' list must be non-empty and real"
    );
}

/// Requiring nothing selects nothing. The empty set is not a degenerate case to tolerate — it is
/// the normal one, and it is what keeps a provider-free build identical to a pre-C7.8 build.
#[test]
fn requiring_no_capability_selects_no_provider() {
    let set = ProviderSet::select(provider_registry::first_party(), LINUX, &[])
        .expect("requiring nothing must succeed");
    assert!(
        set.resolve("clock", "stark_time_monotonic_now_ns").is_err(),
        "an unrequested capability must not be resolvable"
    );
}

/// Selection is per target. A target no provider declares fails rather than falling back.
#[test]
fn an_unsupported_target_fails_rather_than_falling_back() {
    let errors = ProviderSet::select(
        provider_registry::first_party(),
        "riscv64gc-unknown-linux-gnu",
        &["clock".to_string()],
    )
    .expect_err("a target no provider declares must fail");

    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ResolveError::CapabilityUnavailable { .. })),
        "{errors:#?}"
    );
}

/// Every provider in the registry validates against ABI v0.1 and is internally consistent — each
/// function belongs to a declared capability, and each declared capability has a function.
///
/// This is the mirror-integrity check: the registry copies metadata that provider crates own, so a
/// drifted copy must fail here rather than at link time.
#[test]
fn every_registered_provider_validates() {
    for provider in provider_registry::first_party() {
        starkc::provider_abi::validate(&provider.metadata).unwrap_or_else(|violations| {
            panic!(
                "{} has invalid metadata: {violations:#?}",
                provider.metadata.identity.name
            )
        });
        assert!(
            !provider.crate_name.is_empty(),
            "{} must name its crate",
            provider.metadata.identity.name
        );
    }
}

/// Every registered provider's crate is locatable and present in this checkout. A registry entry
/// pointing at a crate that does not exist would fail at build time with a linker error rather
/// than a diagnostic naming the provider.
#[test]
fn every_registered_provider_crate_is_present() {
    for provider in provider_registry::first_party() {
        let path =
            provider_registry::built_in_crate_location(&provider.crate_name, &repo_provider_root())
                .unwrap_or_else(|| panic!("no location for crate {}", provider.crate_name));
        assert!(
            path.join("Cargo.toml").is_file(),
            "{} is registered but its crate is missing at {}",
            provider.crate_name,
            path.display()
        );
    }
}
