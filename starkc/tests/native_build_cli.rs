use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Package {
    root: PathBuf,
}

impl Package {
    fn new(name: &str, source: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "stark_c5_5_{}_{}_{}",
            name,
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("starkpkg.json"),
            format!(r#"{{"name":"{name}","version":"0.1.0","entry":"src/main.stark"}}"#),
        )
        .unwrap();
        std::fs::write(root.join("src/main.stark"), source).unwrap();
        Self { root }
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_stark"))
            .args(args)
            .current_dir(&self.root)
            .output()
            .unwrap()
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}
fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let destination = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &destination);
        } else {
            std::fs::copy(entry.path(), destination).unwrap();
        }
    }
}

const SCALAR: &str = r#"
fn add(a: Int32, b: Int32) -> Int32 { a + b }
fn main() { assert_eq(add(20, 22), 42); }
"#;

#[test]
fn build_places_and_runs_stable_artifact_then_replaces_it() {
    let package = Package::new("build-app", SCALAR);
    let first = package.run(&["build", "--offline"]);
    assert!(first.status.success(), "{}", stderr(&first));
    let final_path = package
        .root
        .join("target/stark/debug")
        .join(format!("build-app{}", std::env::consts::EXE_SUFFIX));
    assert!(final_path.is_file());
    let first_stdout = stdout(&first);
    let reported_path = first_stdout
        .lines()
        .find_map(|line| line.strip_prefix("Built build-app [debug] -> "))
        .expect("build output must report the stable artifact path");
    assert_eq!(
        Path::new(reported_path).canonicalize().unwrap(),
        final_path.canonicalize().unwrap()
    );
    assert!(Command::new(&final_path).status().unwrap().success());
    let generated: Vec<_> = std::fs::read_dir(package.root.join("target/stark/debug"))
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .collect();
    assert!(
        generated.is_empty(),
        "default build leaked generated crates"
    );

    let before = std::fs::metadata(&final_path).unwrap().modified().unwrap();
    let second = package.run(&["build", "--offline"]);
    assert!(second.status.success(), "{}", stderr(&second));
    assert!(std::fs::metadata(&final_path).unwrap().modified().unwrap() >= before);
}

#[test]
fn verbose_default_build_reports_only_the_retained_final_artifact() {
    let package = Package::new("verbose-cleanup", SCALAR);
    let output = package.run(&["build", "--verbose"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        !text.contains("[stark build] backend binary:"),
        "verbose output advertised a backend artifact deleted during cleanup:\n{text}"
    );
    assert!(!text.contains("[stark build] generated crate:"));

    let final_path = text
        .lines()
        .find_map(|line| line.strip_prefix("[stark build] final artifact: "))
        .expect("verbose output must report the stable final artifact");
    assert!(Path::new(final_path).is_file());
    assert!(
        std::fs::read_dir(package.root.join("target/stark/debug"))
            .unwrap()
            .flatten()
            .all(|entry| !entry.path().is_dir()),
        "default verbose build leaked a generated crate"
    );
}

#[test]
fn retention_and_emit_rust_report_existing_paths() {
    let package = Package::new("retained", SCALAR);
    let output = package.run(&["build", "--emit-rust", "--verbose"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    let rust_line = text
        .lines()
        .find(|line| line.starts_with("Generated Rust -> "))
        .unwrap();
    let rust_path = Path::new(rust_line.trim_start_matches("Generated Rust -> "));
    assert!(rust_path.is_file());
    assert!(text.contains("Generated crate -> "));
    assert!(text.contains("[stark build] MIR verification: complete"));
}

#[cfg(unix)]
#[test]
fn aggregate_program_builds_offline_with_empty_cargo_home() {
    use std::os::unix::fs::PermissionsExt;

    let package = Package::new(
        "aggregate",
        r#"
struct Pair { left: Int32, right: Int32 }
fn main() { let pair = Pair { left: 19, right: 23 }; assert_eq(pair.left + pair.right, 42); }
"#,
    );
    let cargo_home = package.root.join("empty-cargo-home");
    std::fs::create_dir(&cargo_home).unwrap();
    let runtime = package.root.join("installed-runtime");
    copy_dir(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stark-runtime"),
        &runtime,
    );
    let cargo_log = package.root.join("cargo-invocations.txt");
    let cargo_wrapper = package.root.join("stark-cargo-wrapper");
    std::fs::write(
        &cargo_wrapper,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nexec cargo \"$@\"\n",
            cargo_log.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&cargo_wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_stark"))
        .args(["build", "--offline", "--keep-generated"])
        .env("CARGO_HOME", &cargo_home)
        .env("STARK_RUNTIME_DIR", &runtime)
        .env("STARK_CARGO", &cargo_wrapper)
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let invocations = std::fs::read_to_string(cargo_log).unwrap();
    assert!(invocations.lines().any(|line| line == "--version"));
    assert!(invocations
        .lines()
        .any(|line| line.starts_with("build --offline --manifest-path")));
    let output_text = stdout(&output);
    let generated = output_text
        .lines()
        .find_map(|line| line.strip_prefix("Generated crate -> "))
        .unwrap();
    let manifest = std::fs::read_to_string(Path::new(generated).join("Cargo.toml")).unwrap();
    assert!(manifest.contains(runtime.canonicalize().unwrap().to_string_lossy().as_ref()));
}

#[test]
fn frozen_three_package_workspace_builds_through_cli_after_relocation() {
    let root = std::env::temp_dir().join(format!(
        "stark_c5_5_workspace_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    copy_dir(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/c5-native-workspace"),
        &root,
    );
    let output = Command::new(env!("CARGO_BIN_EXE_stark"))
        .args(["build", "--locked", "--offline", "--emit-rust"])
        .current_dir(root.join("app"))
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    let binary = root
        .join("app/target/stark/debug")
        .join(format!("app{}", std::env::consts::EXE_SUFFIX));
    assert!(binary.is_file());
    assert!(Command::new(&binary).status().unwrap().success());
    assert!(stdout(&output).contains("Built app [debug] -> "));
    assert!(stdout(&output).contains("Generated Rust -> "));
    let _ = std::fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn backend_failure_reports_and_retains_exact_generated_directory() {
    use std::os::unix::fs::PermissionsExt;

    let package = Package::new("backend-failure", SCALAR);
    let cargo_wrapper = package.root.join("failing-cargo");
    std::fs::write(
        &cargo_wrapper,
        "#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then exec cargo \"$@\"; fi\necho intentional-backend-failure >&2\nexit 23\n",
    )
    .unwrap();
    std::fs::set_permissions(&cargo_wrapper, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_stark"))
        .args(["build", "--verbose"])
        .env("STARK_RUSTC", "rustc")
        .env("STARK_CARGO", &cargo_wrapper)
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = stderr(&output);
    assert!(error.contains("generated crate retained at "), "{error}");
    assert!(error.contains("rustc: "), "{error}");
    assert!(error.contains("cargo: "), "{error}");
    assert!(
        error.contains(&format!(
            "command: RUSTC=rustc {} build --offline --manifest-path ",
            cargo_wrapper.display()
        )),
        "{error}"
    );
    assert!(error.contains("exit status: 23"), "{error}");
    assert!(error.contains("intentional-backend-failure"), "{error}");
    let build_dir = error
        .lines()
        .find_map(|line| line.strip_prefix("note: generated crate retained at "))
        .unwrap();
    assert!(Path::new(build_dir).join("src/main.rs").is_file());
}

#[test]
fn invalid_build_arguments_exit_two() {
    let package = Package::new("usage", SCALAR);
    for args in [["build", "--release"], ["build", "somewhere"]] {
        let output = package.run(&args);
        assert_eq!(output.status.code(), Some(2));
    }
    assert!(package.run(&["build", "--help"]).status.success());
}

#[test]
fn source_errors_precede_toolchain_probes_and_old_artifact_is_not_claimed() {
    let package = Package::new("source-error", "fn main() { let value: bool = 42; }");
    let old = package
        .root
        .join("target/stark/debug")
        .join(format!("source-error{}", std::env::consts::EXE_SUFFIX));
    std::fs::create_dir_all(old.parent().unwrap()).unwrap();
    std::fs::write(&old, "old artifact").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_stark"))
        .arg("build")
        .env("STARK_RUSTC", package.root.join("missing-rustc"))
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = stderr(&output);
    assert!(
        error.contains("type") || error.contains("mismatch"),
        "{error}"
    );
    assert!(!error.contains("toolchain component"), "{error}");
    assert!(!stdout(&output).contains("Built "));
    assert_eq!(std::fs::read_to_string(old).unwrap(), "old artifact");
}

#[test]
fn missing_tool_and_runtime_have_stark_facing_diagnostics() {
    let package = Package::new("preflight", SCALAR);
    let missing_tool = Command::new(env!("CARGO_BIN_EXE_stark"))
        .arg("build")
        .env("STARK_RUSTC", package.root.join("missing-rustc"))
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(stderr(&missing_tool).contains("toolchain component 'rustc' not found"));

    let missing_cargo = Command::new(env!("CARGO_BIN_EXE_stark"))
        .arg("build")
        .env("STARK_CARGO", package.root.join("missing-cargo"))
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(stderr(&missing_cargo).contains("toolchain component 'cargo' not found"));

    let missing_runtime = Command::new(env!("CARGO_BIN_EXE_stark"))
        .arg("build")
        .env("STARK_RUNTIME_DIR", package.root.join("missing-runtime"))
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(stderr(&missing_runtime).contains("native runtime installation is missing"));
}
