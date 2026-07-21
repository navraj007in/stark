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

/// Which side is "expected" and which is "actual" is fixed by the generated crate's point of
/// view, and the two must not be swapped: the version the compiler RECORDED at generation time
/// is what the binary expects, and the version of the runtime it actually LINKED is what it got.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VersionMismatch {
    /// From `BuildVersions::runtime_version` -- stamped in when the crate was generated.
    pub expected_runtime_version: String,
    /// This linked runtime's own [`RUNTIME_VERSION`].
    pub actual_runtime_version: &'static str,
}

/// Fails before any user code runs (§9.2) when the runtime this binary linked does not match
/// the runtime version the compiler build recorded at generation time.
pub fn check(recorded: &BuildVersions) -> Result<(), VersionMismatch> {
    if recorded.runtime_version != RUNTIME_VERSION {
        return Err(VersionMismatch {
            expected_runtime_version: recorded.runtime_version.clone(),
            actual_runtime_version: RUNTIME_VERSION,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn versions_recording(runtime_version: &str) -> BuildVersions {
        BuildVersions {
            compiler_version: "0.1.0".to_string(),
            mir_version: "0.1".to_string(),
            mir_runtime_surface: "0.1".to_string(),
            runtime_version: runtime_version.to_string(),
            backend_version: "0.1".to_string(),
            rustc_version: "irrelevant".to_string(),
            target_triple: "irrelevant".to_string(),
            profile: "debug".to_string(),
        }
    }

    #[test]
    fn matching_runtime_version_passes() {
        assert_eq!(check(&versions_recording(RUNTIME_VERSION)), Ok(()));
    }

    /// Pins the field-to-side assignment, not merely that a mismatch is detected -- the generated
    /// crate prints these as "generated for runtime {expected}, linked against {actual}", so
    /// swapping them produces a message that names the wrong version on each side.
    #[test]
    fn mismatch_reports_the_recorded_version_as_expected_and_this_runtime_as_actual() {
        let mismatch = check(&versions_recording("0.0-ancient")).unwrap_err();
        assert_eq!(mismatch.expected_runtime_version, "0.0-ancient");
        assert_eq!(mismatch.actual_runtime_version, RUNTIME_VERSION);
    }
}
