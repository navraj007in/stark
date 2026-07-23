//! §5.1's module boundary. Not every file carries real logic yet -- §5.1 is explicit that this
//! is "a responsibility map, not a requirement to create every file immediately" -- but the
//! shape is established now so later work packages extend files instead of restructuring them.

pub mod build;
pub mod emit_bodies;
pub mod emit_places;
pub mod emit_program;
pub mod emit_projections;
pub mod emit_runtime;
pub mod emit_types;
pub mod linkage;
pub mod mangle;
pub mod source_map;

use crate::mir::verify::VerifiedMirProgram;
use std::path::PathBuf;

/// §5.2: never a STARK user-source error. `Unsupported` means "not yet lowered by this backend
/// increment," classified per §17's defect table by the caller, not by this type.
#[derive(Clone, Debug)]
pub enum BackendDiagnostic {
    Unsupported(String),
    /// WP-C5.5: the generated crate's Cargo process failed (or reported success without the
    /// promised artifact). This is structured process evidence for the CLI, never a STARK
    /// source diagnostic.
    BuildFailed(Box<BackendBuildFailure>),
    Io(String),
}

#[derive(Clone, Debug)]
pub struct BackendBuildFailure {
    pub summary: String,
    pub stdout: String,
    pub stderr: String,
    pub build_dir: PathBuf,
    pub command: Vec<String>,
    pub status: Option<i32>,
}

/// Explicit external inputs to generated-crate construction. The production CLI supplies these
/// from `native_toolchain`; keeping them separate from semantic build options lets older direct
/// backend callers retain the compatibility entry point below.
#[derive(Clone, Debug)]
pub struct NativeToolchainOptions {
    pub rustc: PathBuf,
    pub cargo: PathBuf,
    pub runtime_crate: PathBuf,
}

impl NativeToolchainOptions {
    fn development() -> Self {
        Self {
            rustc: PathBuf::from("rustc"),
            cargo: PathBuf::from("cargo"),
            runtime_crate: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stark-runtime"),
        }
    }
}

pub struct NativeBuildOptions {
    /// §11: the STARK target directory root generated crates are written under
    /// (`<target_dir>/debug/<build-key>/`). WP-C5.1b's tests pass a scratch directory; a
    /// stable default (and the `target/stark/debug/<binary-name>` user-facing path from §12.3)
    /// is WP-C5.5a's job.
    pub target_dir: PathBuf,
    /// WP-C5.3e (CD-067): the named layout contract this build answers `size_of`/`align_of` from.
    /// Resolved through `layout::contract_for`, which REJECTS a name this compiler has no contract
    /// for rather than falling back to a default -- the answer is observable and target-specific
    /// (LAYOUT-ABI-001), so a silent default would let a build report values for a target it was
    /// not asked about. Its identity is part of the build key.
    pub target_contract: String,
}

impl Default for NativeBuildOptions {
    fn default() -> Self {
        Self {
            target_dir: PathBuf::from("target/stark"),
            target_contract: "stark-64-v1".to_string(),
        }
    }
}

pub struct NativeArtifact {
    /// The compiled binary's actual on-disk path (currently inside the generated crate's own
    /// `target/debug/`, not yet the stable `stark build` output path -- §12.3 is WP-C5.5a).
    pub binary_path: PathBuf,
    pub build_dir: PathBuf,
}

/// §5's entry point. The verified-program precondition is encoded in the parameter type --
/// `VerifiedMirProgram` is constructible only via `mir::verify::verify_program` -- rather than
/// re-checked here, per the review correction §5 records ("no backend bypasses MIR validation"
/// as an API property).
pub fn emit_native_debug(
    program: &VerifiedMirProgram<'_>,
    options: &NativeBuildOptions,
) -> Result<NativeArtifact, BackendDiagnostic> {
    emit_native_debug_with_toolchain(program, options, &NativeToolchainOptions::development())
}

/// WP-C5.5 production entry point. Every external command and runtime path is an explicit,
/// preflighted input rather than a source-checkout or PATH assumption hidden in the backend.
pub fn emit_native_debug_with_toolchain(
    program: &VerifiedMirProgram<'_>,
    options: &NativeBuildOptions,
    toolchain: &NativeToolchainOptions,
) -> Result<NativeArtifact, BackendDiagnostic> {
    build::build_and_link(program.program(), options, toolchain)
}
