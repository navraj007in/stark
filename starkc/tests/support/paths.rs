//! Repository-path helpers.
//!
//! These exist because the first-party packages moved from the repository root into `packages/`,
//! and the move found fourteen call sites that had spelled the old location as a string literal. A
//! path that appears once can be corrected once; a path that appears fourteen times gets corrected
//! thirteen times and then fails in CI on the fourteenth.
//!
//! Use [`repo_package`] and [`repo_provider`] rather than joining `"packages"` by hand — the point
//! is that the directory name lives here and nowhere else.
//!
//! `#![allow(dead_code)]` because each including test binary uses a different subset: a test that
//! needs only `repo_provider_root` should not have to reference the others to satisfy `-D warnings`.

#![allow(dead_code)]

use std::path::PathBuf;

/// The repository root — the parent of `starkc/`.
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("starkc/ must have a parent")
        .to_path_buf()
}

/// A first-party STARK package directory, by package name (`repo_package("stark-io")`).
pub fn repo_package(name: &str) -> PathBuf {
    repo_root().join("packages").join(name)
}

/// A package's native provider crate — `packages/<name>/native`.
///
/// The depth matters beyond tidiness: each provider crate reaches the ABI through
/// `../../../starkc/stark-provider-abi`, which is correct only at this level.
pub fn repo_provider(name: &str) -> PathBuf {
    repo_package(name).join("native")
}

/// The root that `provider_registry::built_in_crate_location` resolves `crate_path` against.
///
/// **Not the repository root.** A provider manifest declares `crate_path: "stark-time/native"`,
/// relative to the directory holding the packages — so this is `<repo>/packages`, and passing
/// `repo_root()` instead yields `<repo>/stark-time/native`, a path that stopped existing when the
/// packages moved. The compiler finds this same directory by walking up from the package until it
/// sees `stark-time/native/Cargo.toml`.
pub fn repo_provider_root() -> PathBuf {
    repo_root().join("packages")
}
