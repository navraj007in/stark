//! §9.2: every generated crate records STARK compiler version, MIR version, MIR runtime-surface
//! version, native runtime ABI version, backend version, rustc version, target triple, and
//! debug profile. A mismatch must fail before user code executes.
//!
//! This runtime crate owns the "native runtime ABI version" half of that record and the
//! generated crate's `main` calls [`check`] with the versions the *compiler build* stamped in,
//! comparing them against what *this compiled runtime* actually is. The other fields (compiler/
//! MIR/backend/rustc/target/profile) are recorded for diagnostics but are not this crate's
//! authority to validate -- only `RUNTIME_VERSION` is.

/// Bumped whenever this crate's public runtime surface changes in a way a generated caller
/// could observe. Independent of `starkc::mir::MIR_VERSION`/`MIR_RUNTIME_SURFACE` (§9.2).
pub const RUNTIME_VERSION: &str = "0.1";

/// The full version-identity record a generated crate embeds (§9.2). Every field is recorded
/// for diagnostics; `check` validates only `runtime_version` against this build's
/// [`RUNTIME_VERSION`], since the other fields are the compiler's/backend's own authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildVersions {
    pub compiler_version: String,
    pub mir_version: String,
    pub mir_runtime_surface: String,
    pub runtime_version: String,
    pub backend_version: String,
    pub rustc_version: String,
    pub target_triple: String,
    pub profile: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionMismatch {
    pub expected_runtime_version: &'static str,
    pub actual_runtime_version: String,
}

/// Fails before any user code runs (§9.2) when the runtime this binary linked does not match
/// the runtime version the compiler build recorded at generation time.
pub fn check(recorded: &BuildVersions) -> Result<(), VersionMismatch> {
    if recorded.runtime_version != RUNTIME_VERSION {
        return Err(VersionMismatch {
            expected_runtime_version: RUNTIME_VERSION,
            actual_runtime_version: recorded.runtime_version.clone(),
        });
    }
    Ok(())
}
