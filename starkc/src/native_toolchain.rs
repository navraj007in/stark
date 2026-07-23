//! Rust toolchain and installed STARK runtime discovery for native debug builds.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const MINIMUM_RUSTC_VERSION: &str = "1.85.0";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolchainInfo {
    pub rustc: PathBuf,
    pub cargo: PathBuf,
    pub rustc_release: String,
    pub cargo_release: String,
    pub host_triple: String,
    pub sysroot: Option<PathBuf>,
    pub runtime_crate: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolchainError {
    Missing {
        tool: &'static str,
        attempted: PathBuf,
        detail: String,
    },
    InvalidVersion {
        tool: &'static str,
        output: String,
    },
    TooOld {
        found: String,
        required: &'static str,
    },
    RuntimeMissing {
        attempted: Vec<PathBuf>,
    },
}

pub fn discover(current_exe: Option<&Path>) -> Result<ToolchainInfo, ToolchainError> {
    let rustc = command_path("STARK_RUSTC", "RUSTC", "rustc");
    let cargo = command_path("STARK_CARGO", "CARGO", "cargo");
    let rustc_verbose = run_probe(&rustc, &["-vV"], "rustc")?;
    let rustc_release = rustc_verbose
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .ok_or_else(|| ToolchainError::InvalidVersion {
            tool: "rustc",
            output: rustc_verbose.clone(),
        })?
        .to_string();
    let host_triple = rustc_verbose
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| ToolchainError::InvalidVersion {
            tool: "rustc",
            output: rustc_verbose.clone(),
        })?
        .to_string();
    if parse_version(&rustc_release).ok_or_else(|| ToolchainError::InvalidVersion {
        tool: "rustc",
        output: rustc_release.clone(),
    })? < parse_version(MINIMUM_RUSTC_VERSION).expect("constant is valid")
    {
        return Err(ToolchainError::TooOld {
            found: rustc_release,
            required: MINIMUM_RUSTC_VERSION,
        });
    }

    let cargo_output = run_probe(&cargo, &["--version"], "cargo")?;
    let cargo_release = cargo_output
        .split_whitespace()
        .nth(1)
        .filter(|value| parse_version(value).is_some())
        .ok_or_else(|| ToolchainError::InvalidVersion {
            tool: "cargo",
            output: cargo_output.clone(),
        })?
        .to_string();
    let sysroot = run_probe(&rustc, &["--print", "sysroot"], "rustc")
        .ok()
        .map(|value| PathBuf::from(value.trim()));
    let runtime_crate = discover_runtime(current_exe)?;
    Ok(ToolchainInfo {
        rustc,
        cargo,
        rustc_release,
        cargo_release,
        host_triple,
        sysroot,
        runtime_crate,
    })
}

fn command_path(stark_var: &str, conventional_var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(stark_var)
        .or_else(|| std::env::var_os(conventional_var))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(fallback))
}

fn run_probe(path: &Path, args: &[&str], tool: &'static str) -> Result<String, ToolchainError> {
    let output =
        Command::new(path)
            .args(args)
            .output()
            .map_err(|error| ToolchainError::Missing {
                tool,
                attempted: path.to_path_buf(),
                detail: error.to_string(),
            })?;
    if !output.status.success() {
        return Err(ToolchainError::Missing {
            tool,
            attempted: path.to_path_buf(),
            detail: format!(
                "command exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub fn parse_version(value: &str) -> Option<(u64, u64, u64)> {
    let numeric = value
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .next()?;
    let mut parts = numeric.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    Some((major, minor, patch))
}

pub fn discover_runtime(current_exe: Option<&Path>) -> Result<PathBuf, ToolchainError> {
    if let Some(override_path) = std::env::var_os("STARK_RUNTIME_DIR") {
        let path = PathBuf::from(override_path);
        return validate_runtime(&path).ok_or(ToolchainError::RuntimeMissing {
            attempted: vec![path],
        });
    }
    let mut attempted = Vec::new();
    if let Some(exe) = current_exe {
        if let Some(bin_dir) = exe.parent() {
            let candidate = bin_dir.join("../lib/stark/stark-runtime");
            attempted.push(candidate.clone());
            if let Some(path) = validate_runtime(&candidate) {
                return Ok(path);
            }
        }
    }
    let candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stark-runtime");
    attempted.push(candidate.clone());
    validate_runtime(&candidate).ok_or(ToolchainError::RuntimeMissing { attempted })
}

fn validate_runtime(path: &Path) -> Option<PathBuf> {
    (path.is_dir() && path.join("Cargo.toml").is_file() && path.join("src/lib.rs").is_file())
        .then(|| path.canonicalize().unwrap_or_else(|_| path.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn parses_supported_rust_versions() {
        assert_eq!(parse_version("1.85.0"), Some((1, 85, 0)));
        assert_eq!(parse_version("1.93.0"), Some((1, 93, 0)));
        assert_eq!(parse_version("1.93.0-nightly"), Some((1, 93, 0)));
        assert_eq!(parse_version("not-a-version"), None);
    }

    #[test]
    fn discovers_runtime_from_installed_toolchain_layout() {
        let root = std::env::temp_dir().join(format!(
            "stark_installed_runtime_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let runtime = root.join("lib/stark/stark-runtime");
        std::fs::create_dir_all(runtime.join("src")).unwrap();
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(
            runtime.join("Cargo.toml"),
            "[package]\nname='stark-runtime'\n",
        )
        .unwrap();
        std::fs::write(runtime.join("src/lib.rs"), "").unwrap();

        let found = discover_runtime(Some(&root.join("bin/stark"))).unwrap();
        assert_eq!(found, runtime.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(root);
    }
}
