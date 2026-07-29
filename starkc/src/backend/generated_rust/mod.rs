//! §5.1's module boundary. Not every file carries real logic yet -- §5.1 is explicit that this
//! is "a responsibility map, not a requirement to create every file immediately" -- but the
//! shape is established now so later work packages extend files instead of restructuring them.

pub mod build;
pub mod emit_bodies;
pub mod emit_places;
pub mod emit_program;
pub mod emit_projections;
pub mod emit_provider;
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
    /// WP-C6.4a: target preflight refused the build. A separate variant from `Unsupported`
    /// because the two mean opposite things to a user -- `Unsupported` is "this backend increment
    /// does not lower that construct yet", while this is "this compiler does not build for that
    /// machine". It carries the classification rather than a formatted string so the CLI, the
    /// tests, and the qualification harness can distinguish an unsupported target from a missing
    /// toolchain (§8.3) without matching on prose.
    TargetRejected(crate::target::TargetError),
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
    /// A10 (C7.8.2e): Cargo package name → crate location, for every provider the build may link.
    ///
    /// Locations live **here**, not in MIR: a provider's path is a property of the machine doing
    /// the build, while the name is a property of the program. Keeping them apart is what lets a
    /// verified MIR artefact stay relocation-stable while still naming its providers.
    pub provider_crates: std::collections::BTreeMap<String, PathBuf>,
}

impl NativeToolchainOptions {
    fn development() -> Self {
        Self {
            rustc: PathBuf::from("rustc"),
            cargo: PathBuf::from("cargo"),
            runtime_crate: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stark-runtime"),
            provider_crates: std::collections::BTreeMap::new(),
        }
    }
}

/// WP-C7.1. The build profile — a STARK-level concept that happens to map onto a Cargo profile.
///
/// It is NOT a passthrough of Cargo's, because the mapping is where STARK's semantics could be lost:
/// Cargo's release defaults would give `panic = "unwind"`, and `generated_cargo_toml` must override
/// that rather than inherit it. Every setting the profile implies is written explicitly (§6.6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Profile {
    #[default]
    Debug,
    Release,
}

impl Profile {
    /// The directory component and the identity string in `BuildVersions` — one function so the
    /// artefact path and the build key can never disagree about which profile produced a build.
    pub fn as_str(self) -> &'static str {
        match self {
            Profile::Debug => "debug",
            Profile::Release => "release",
        }
    }

    pub fn is_release(self) -> bool {
        matches!(self, Profile::Release)
    }
}

pub struct NativeBuildOptions {
    /// §11: the STARK target root. Generated crates live under
    /// `<target_dir>/debug/<build-key>/`; the C5.5 CLI also installs the stable package-named
    /// executable under `<target_dir>/debug/`.
    pub target_dir: PathBuf,
    /// WP-C5.3e (CD-067): the named layout contract this build answers `size_of`/`align_of` from.
    /// Resolved through `layout::contract_for`, which REJECTS a name this compiler has no contract
    /// for rather than falling back to a default -- the answer is observable and target-specific
    /// (LAYOUT-ABI-001), so a silent default would let a build report values for a target it was
    /// not asked about. Its identity is part of the build key.
    pub target_contract: String,
    /// WP-C7.1: which profile to build. Part of the artefact path AND the build key, so a debug and
    /// a release build of the same source can coexist and can never be mistaken for one another.
    pub profile: Profile,
    /// WP-C7.1: the requested target TRIPLE, when cross-compiling. `None` means the host.
    ///
    /// Distinct from `target_contract`, which is a LAYOUT contract (`stark-64-v1`) answering
    /// `size_of`/`align_of`. Conflating them would let a cross-build silently answer layout
    /// questions for the host.
    pub target_triple: Option<String>,
}

impl Default for NativeBuildOptions {
    fn default() -> Self {
        Self {
            target_dir: PathBuf::from("target/stark"),
            target_contract: "stark-64-v1".to_string(),
            profile: Profile::Debug,
            target_triple: None,
        }
    }
}

pub struct NativeArtifact {
    /// The backend-local compiled binary inside the generated crate. The C5.5 build driver copies
    /// it to the stable package-named output before optionally deleting `build_dir`.
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
