//! Library-owned orchestration for `stark build`.

use crate::analysis::{analyze_project, ProjectInput};
use crate::backend::generated_rust::{
    emit_native_debug_with_toolchain, BackendDiagnostic, NativeBuildOptions, NativeToolchainOptions,
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
}

#[derive(Clone, Debug)]
pub struct BuildCommandResult {
    pub package_name: String,
    pub package_root: PathBuf,
    pub artifact_path: PathBuf,
    pub generated_dir: Option<PathBuf>,
    pub generated_rust: Option<PathBuf>,
    pub backend_artifact: Option<PathBuf>,
    pub mir_bodies: usize,
    pub toolchain: ToolchainInfo,
}

#[derive(Clone, Debug)]
pub enum BuildCommandError {
    Package(String),
    Analysis {
        rendered: String,
        package_name: String,
    },
    Lowering(String),
    MirVerification(String),
    Toolchain(ToolchainError),
    UnsupportedNative(String),
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
    let mir = lower_program(hir, tables, analysis.root_file.clone())
        .map_err(|error| BuildCommandError::Lowering(error.what))?;
    let mir_bodies = mir.bodies.len();
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
    let target_root = package_root.join("target/stark");
    let final_dir = target_root.join("debug");
    let artifact = emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: target_root.clone(),
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
    let keep = options.keep_generated || options.emit_rust;
    let generated_dir = keep.then(|| artifact.build_dir.clone());
    let generated_rust = options.emit_rust.then_some(generated_rust_path);
    let backend_artifact = keep.then(|| artifact.binary_path.clone());
    if !keep {
        std::fs::remove_dir_all(&artifact.build_dir).map_err(|error| BuildCommandError::Io {
            action: "removing generated crate".into(),
            path: Some(artifact.build_dir.clone()),
            detail: error.to_string(),
        })?;
    }
    Ok(BuildCommandResult {
        package_name,
        package_root,
        artifact_path: final_path,
        generated_dir,
        generated_rust,
        backend_artifact,
        mir_bodies,
        toolchain,
    })
}

fn map_backend_error(error: BackendDiagnostic, toolchain: &ToolchainInfo) -> BuildCommandError {
    match error {
        BackendDiagnostic::Unsupported(message) => BuildCommandError::UnsupportedNative(message),
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
    std::fs::rename(&temp, destination).map_err(|error| BuildCommandError::ArtifactInstall {
        from: source.to_path_buf(),
        to: destination.to_path_buf(),
        detail: error.to_string(),
    })?;
    if !destination.is_file() {
        return Err(BuildCommandError::ArtifactMissing(
            destination.to_path_buf(),
        ));
    }
    Ok(())
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
