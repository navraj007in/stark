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
pub mod mangle;
pub mod source_map;

use crate::mir::verify::VerifiedMirProgram;
use std::path::PathBuf;

/// §5.2: never a STARK user-source error. `Unsupported` means "not yet lowered by this backend
/// increment," classified per §17's defect table by the caller, not by this type.
#[derive(Clone, Debug)]
pub enum BackendDiagnostic {
    Unsupported(String),
    /// §12.5: the generated crate's own `cargo build` failed. Carries raw cargo/rustc output
    /// for now; §12.5's STARK-facing classification (backend defect vs. toolchain vs.
    /// environmental) is WP-C5.5b's deliverable.
    BuildFailed(String),
    Io(String),
}

pub struct NativeBuildOptions {
    /// §11: the STARK target directory root generated crates are written under
    /// (`<target_dir>/debug/<build-key>/`). WP-C5.1b's tests pass a scratch directory; a
    /// stable default (and the `target/stark/debug/<binary-name>` user-facing path from §12.3)
    /// is WP-C5.5a's job.
    pub target_dir: PathBuf,
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
    build::build_and_link(program.program(), options)
}
