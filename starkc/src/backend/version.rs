//! §9.2 version-identity record, computed by the backend at generation time and embedded in
//! the generated crate's `main`, which calls `stark_runtime::version::check` against it. This
//! crate depends on `stark-runtime` for exactly that one shared type -- a generated binary
//! links its own copy of the runtime crate; `starkc` only needs the type to embed a value.

use crate::mir::{MIR_RUNTIME_SURFACE, MIR_VERSION};
pub use stark_runtime::version::{BuildVersions, RUNTIME_VERSION};

/// Bumped whenever the generated-Rust backend's emission contract changes in a way a linked
/// runtime or a hand-inspecting reader could observe. Independent of `MIR_VERSION`/
/// `RUNTIME_VERSION` (§9.2 -- these version axes move independently by design).
pub const BACKEND_VERSION: &str = "0.1";

pub fn compiler_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn build_versions(rustc_version: String, target_triple: String) -> BuildVersions {
    BuildVersions {
        compiler_version: compiler_version().to_string(),
        mir_version: MIR_VERSION.to_string(),
        mir_runtime_surface: MIR_RUNTIME_SURFACE.to_string(),
        runtime_version: RUNTIME_VERSION.to_string(),
        backend_version: BACKEND_VERSION.to_string(),
        rustc_version,
        target_triple,
        // WP-C5.1b only ever builds the debug profile (§12.1: release stays C7).
        profile: "debug".to_string(),
    }
}
