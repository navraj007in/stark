//! Library-owned orchestration for `stark build`.

use crate::analysis::{analyze_project, ProjectInput};
use crate::backend::generated_rust::{
    emit_native_debug_with_toolchain, BackendDiagnostic, NativeBuildOptions,
    NativeToolchainOptions, Profile,
};
use crate::mir::{lower::lower_program, verify::verify_program};
use crate::native_toolchain::{self, ToolchainError, ToolchainInfo};
use crate::options::LanguageOptions;
use crate::package::{find_package_root, PackageGraph};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildCommandOptions {
    pub locked: bool,
    pub offline: bool,
    pub keep_generated: bool,
    pub emit_rust: bool,
    pub verbose: bool,
    /// WP-C7.1 `--release`.
    pub release: bool,
    /// WP-C7.1 `--target <triple>`. `None` builds for the host.
    pub target: Option<String>,
    /// WP-C7.3 `--no-build-cache`: delete the generated crate after the build, as every build did
    /// before the cache existed. This is the qualification path — a run that must not benefit from,
    /// or be influenced by, anything a previous build left behind.
    pub no_build_cache: bool,
    /// WP-C7.4 `--no-mir-opt`: lower to MIR and hand it to the backend exactly as lowered.
    ///
    /// The baseline optimisations are on by default because they are required to be observationally
    /// transparent — if one is not, that is a defect to fix, not a reason to keep the pass off. The
    /// flag exists so a divergence can be BISECTED against unoptimised MIR, which Gate C7 makes the
    /// higher authority, without rebuilding the compiler.
    pub no_mir_opt: bool,
}

impl BuildCommandOptions {
    pub fn profile(&self) -> Profile {
        if self.release {
            Profile::Release
        } else {
            Profile::Debug
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuildCommandResult {
    pub package_name: String,
    /// WP-C7.1: the profile this build actually used. Reported rather than re-derived: the CLI
    /// printed a hard-coded "[debug]" for a `--release` build until this field existed, so the
    /// message and the artefact path disagreed about the same build.
    pub profile: Profile,
    pub package_root: PathBuf,
    pub artifact_path: PathBuf,
    pub generated_dir: Option<PathBuf>,
    pub generated_rust: Option<PathBuf>,
    pub backend_artifact: Option<PathBuf>,
    pub mir_bodies: usize,
    pub toolchain: ToolchainInfo,
    /// WP-C7.3: what the post-build LRU sweep removed, or `None` when the cache was disabled.
    pub cache_eviction: Option<crate::build_cache::EvictionReport>,
    /// WP-C7.4: what the MIR optimiser changed, or `None` when `--no-mir-opt` was passed.
    pub mir_opt: Option<crate::mir::opt::OptStats>,
}

#[derive(Clone, Debug)]
pub enum BuildCommandError {
    Package(String),
    /// WP-C7.1: a target this compiler does not name. Distinct from a target it names but whose
    /// toolchain is missing locally, which `target::preflight` reports separately.
    Target(String),
    Analysis {
        rendered: String,
        package_name: String,
    },
    Lowering(String),
    MirVerification(String),
    Toolchain(ToolchainError),
    UnsupportedNative(String),
    /// WP-C6.4a: target preflight refused the build, before any crate was generated. Kept
    /// distinct from `UnsupportedNative` (an unlowered construct) and from `BackendBuild` (a
    /// rustc/linker failure) because §8.4 requires an unsupported target to be rejected on its
    /// own terms rather than surfacing as a downstream tool failure.
    TargetRejected(crate::target::TargetError),
    BackendBuild(Box<NativeBackendBuildError>),
    ArtifactMissing(PathBuf),
    ArtifactInstall {
        from: PathBuf,
        to: PathBuf,
        detail: String,
    },
    Io {
        action: String,
        path: Option<PathBuf>,
        detail: String,
    },
}

#[derive(Clone, Debug)]
pub struct NativeBackendBuildError {
    pub failure: crate::backend::generated_rust::BackendBuildFailure,
    pub toolchain: ToolchainInfo,
}

pub fn build_current_package(
    current_dir: &Path,
    options: &BuildCommandOptions,
) -> Result<BuildCommandResult, BuildCommandError> {
    // WP-C7.1 (§3.3): the target is validated BEFORE any expensive work — before the package graph
    // is loaded, before analysis, before a crate is emitted. A bad triple must cost nothing.
    if let Some(requested) = &options.target {
        if crate::target::classify(requested).is_none() {
            return Err(BuildCommandError::Target(format!(
                "unsupported target `{requested}`\n  STARK names these targets: {}",
                crate::target::known_targets()
                    .iter()
                    .map(|t| format!("{} ({})", t.triple, t.tier))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    let manifest = find_package_root(current_dir).map_err(BuildCommandError::Package)?;
    let package_root = manifest
        .parent()
        .ok_or_else(|| {
            BuildCommandError::Package("package manifest has no parent directory".into())
        })?
        .to_path_buf();
    let graph = PackageGraph::load_from_root_with_modes(&manifest, options.locked, options.offline)
        .map_err(BuildCommandError::Package)?;
    let package_name = graph.root_package_name.clone();
    validate_binary_name(&package_name).map_err(BuildCommandError::Package)?;
    let analysis = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);
    if analysis.has_errors() {
        return Err(BuildCommandError::Analysis {
            rendered: analysis
                .diagnostic_batch(&HashMap::new())
                .render(&analysis.source_map),
            package_name,
        });
    }
    let hir = analysis.hir.as_ref().ok_or_else(|| {
        BuildCommandError::Lowering("successful analysis did not produce HIR".into())
    })?;
    let tables = analysis.type_tables.as_ref().ok_or_else(|| {
        BuildCommandError::Lowering("successful analysis did not produce type tables".into())
    })?;
    let mut mir = lower_program(hir, tables, analysis.root_file.clone())
        .map_err(|error| BuildCommandError::Lowering(error.what))?;
    let mir_bodies = mir.bodies.len();
    // WP-C7.4. Optimise BEFORE verifying, deliberately: the verifier then checks the program that
    // is actually compiled and executed, rather than a form the backend never sees. An optimiser
    // that produced ill-formed MIR would otherwise be caught only by whatever the backend happened
    // to notice downstream.
    let mir_opt = (!options.no_mir_opt).then(|| crate::mir::opt::optimise(&mut mir));
    let verified = verify_program(&mir).map_err(|errors| {
        BuildCommandError::MirVerification(
            errors
                .into_iter()
                .map(|error| {
                    format!(
                        "{} {} bb{}: {}",
                        error.code, error.symbol, error.block, error.message
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
    })?;

    // Source diagnostics deliberately precede all external tool probes.
    let toolchain = native_toolchain::discover(std::env::current_exe().ok().as_deref())
        .map_err(BuildCommandError::Toolchain)?;
    // WP-C7.1 (§3.4): the layout is parameterised by TARGET and PROFILE, so a debug build, a
    // release build and a cross-target build of one package cannot overwrite each other. The old
    // layout was `target/stark/debug/` with both components fixed.
    let profile = options.profile();
    let target_root = package_root.join("target/stark");
    let final_dir = match &options.target {
        Some(triple) => target_root.join(triple).join(profile.as_str()),
        None => target_root.join(profile.as_str()),
    };
    // WP-C7.3: the cache root is where the backend puts content-addressed crate directories —
    // `target/stark/<profile>/`. Per-profile by construction, so a debug entry can never be reused
    // for a release build; TARGET separation comes from the build key, which carries the triple.
    let cache_root = target_root.join(profile.as_str());
    let artifact = emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: target_root.clone(),
            profile,
            target_triple: options.target.clone(),
            ..NativeBuildOptions::default()
        },
        &NativeToolchainOptions {
            rustc: toolchain.rustc.clone(),
            cargo: toolchain.cargo.clone(),
            runtime_crate: toolchain.runtime_crate.clone(),
        },
    )
    .map_err(|error| map_backend_error(error, &toolchain))?;
    if !artifact.binary_path.is_file() {
        return Err(BuildCommandError::ArtifactMissing(artifact.binary_path));
    }
    let final_path = final_dir.join(binary_filename(&package_name, &artifact.binary_path));
    install_artifact(&artifact.binary_path, &final_path)?;

    let generated_rust_path = artifact.build_dir.join("src/main.rs");
    if options.emit_rust && !generated_rust_path.is_file() {
        return Err(BuildCommandError::ArtifactMissing(generated_rust_path));
    }
    // WP-C7.3. The generated crate is RETAINED by default — it is content-addressed, so keeping it
    // makes the next build of the same source a cache hit rather than a rebuild. Before this, it was
    // deleted immediately and every rebuild paid the full cost including recompiling the runtime.
    //
    // `--keep-generated`/`--emit-rust` still mean something distinct from "cached": they PIN the
    // entry, so eviction never removes something the user explicitly asked to keep. An ordinary
    // cached entry is evictable; a requested one is not.
    let pinned = options.keep_generated || options.emit_rust;
    let generated_dir = pinned.then(|| artifact.build_dir.clone());
    let generated_rust = options.emit_rust.then_some(generated_rust_path);
    let backend_artifact = pinned.then(|| artifact.binary_path.clone());
    let mut cache_eviction = None;
    if options.no_build_cache {
        std::fs::remove_dir_all(&artifact.build_dir).map_err(|error| BuildCommandError::Io {
            action: "removing generated crate".into(),
            path: Some(artifact.build_dir.clone()),
            detail: error.to_string(),
        })?;
    } else {
        crate::build_cache::touch(&artifact.build_dir, pinned);
        // Sweep AFTER a successful build, never before: an eviction that ran first could remove the
        // very entry this build was about to reuse.
        cache_eviction = Some(crate::build_cache::evict(
            &cache_root,
            Some(&artifact.build_dir),
            crate::build_cache::DEFAULT_MAX_BYTES,
            crate::build_cache::DEFAULT_MAX_AGE,
        ));
    }
    Ok(BuildCommandResult {
        package_name,
        profile,
        package_root,
        artifact_path: final_path,
        generated_dir,
        generated_rust,
        backend_artifact,
        mir_bodies,
        toolchain,
        cache_eviction,
        mir_opt,
    })
}

fn map_backend_error(error: BackendDiagnostic, toolchain: &ToolchainInfo) -> BuildCommandError {
    match error {
        BackendDiagnostic::Unsupported(message) => BuildCommandError::UnsupportedNative(message),
        BackendDiagnostic::TargetRejected(error) => BuildCommandError::TargetRejected(error),
        BackendDiagnostic::BuildFailed(failure) => {
            BuildCommandError::BackendBuild(Box::new(NativeBackendBuildError {
                failure: *failure,
                toolchain: toolchain.clone(),
            }))
        }
        BackendDiagnostic::Io(detail) => BuildCommandError::Io {
            action: "running the native backend".to_string(),
            path: None,
            detail,
        },
    }
}

pub fn validate_binary_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || matches!(name, "." | "..")
        || name.contains('/')
        || name.contains('\\')
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        return Err(format!(
            "package name '{name}' is not a safe executable name"
        ));
    }
    Ok(())
}

fn binary_filename(package: &str, backend: &Path) -> String {
    backend
        .extension()
        .and_then(|value| value.to_str())
        .map(|suffix| format!("{package}.{suffix}"))
        .unwrap_or_else(|| package.to_string())
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn install_artifact(source: &Path, destination: &Path) -> Result<(), BuildCommandError> {
    std::fs::create_dir_all(destination.parent().expect("final artifact has parent")).map_err(
        |error| BuildCommandError::ArtifactInstall {
            from: source.to_path_buf(),
            to: destination.to_path_buf(),
            detail: error.to_string(),
        },
    )?;
    let temp = destination.with_file_name(format!(
        "{}.tmp-{}-{}",
        destination.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if temp.exists() {
        std::fs::remove_file(&temp).map_err(|error| BuildCommandError::ArtifactInstall {
            from: source.to_path_buf(),
            to: temp.clone(),
            detail: error.to_string(),
        })?;
    }
    std::fs::copy(source, &temp).map_err(|error| BuildCommandError::ArtifactInstall {
        from: source.to_path_buf(),
        to: destination.to_path_buf(),
        detail: error.to_string(),
    })?;
    if !temp.is_file() {
        return Err(BuildCommandError::ArtifactMissing(temp));
    }
    replace_artifact(&temp, destination, source)?;
    if !destination.is_file() {
        return Err(BuildCommandError::ArtifactMissing(
            destination.to_path_buf(),
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn replace_artifact(
    temp: &Path,
    destination: &Path,
    source: &Path,
) -> Result<(), BuildCommandError> {
    std::fs::rename(temp, destination).map_err(|error| BuildCommandError::ArtifactInstall {
        from: source.to_path_buf(),
        to: destination.to_path_buf(),
        detail: error.to_string(),
    })
}

#[cfg(windows)]
fn replace_artifact(
    temp: &Path,
    destination: &Path,
    source: &Path,
) -> Result<(), BuildCommandError> {
    // Windows rename does not replace an existing destination. Preserve the old executable
    // until the new one is ready, and roll back if the second half of the swap fails.
    let backup = destination.with_file_name(format!(
        "{}.old-{}-{}",
        destination.file_name().unwrap().to_string_lossy(),
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let had_destination = destination.is_file();
    if had_destination {
        std::fs::rename(destination, &backup).map_err(|error| {
            BuildCommandError::ArtifactInstall {
                from: source.to_path_buf(),
                to: destination.to_path_buf(),
                detail: format!("preserving previous artifact: {error}"),
            }
        })?;
    }
    match std::fs::rename(temp, destination) {
        Ok(()) => {
            if had_destination {
                std::fs::remove_file(&backup).map_err(|error| {
                    BuildCommandError::ArtifactInstall {
                        from: source.to_path_buf(),
                        to: destination.to_path_buf(),
                        detail: format!("removing previous artifact backup: {error}"),
                    }
                })?;
            }
            Ok(())
        }
        Err(error) => {
            if had_destination {
                let _ = std::fs::rename(&backup, destination);
            }
            Err(BuildCommandError::ArtifactInstall {
                from: source.to_path_buf(),
                to: destination.to_path_buf(),
                detail: error.to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_safe_binary_names() {
        for valid in ["app", "my-app", "my_app", "app.v1"] {
            assert!(validate_binary_name(valid).is_ok());
        }
        for invalid in ["", ".", "..", "../app", "a/b", "a\\b", "bad name"] {
            assert!(validate_binary_name(invalid).is_err());
        }
    }

    #[test]
    fn preserves_backend_executable_suffix() {
        assert_eq!(binary_filename("demo", Path::new("stark_program")), "demo");
        assert_eq!(
            binary_filename("demo", Path::new("stark_program.exe")),
            "demo.exe"
        );
    }
}
