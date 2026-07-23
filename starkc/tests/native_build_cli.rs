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

const SCALAR: &str = r#"
fn add(a: i32, b: i32) -> i32 { a + b }
fn main() { assert_eq(add(20, 22), 42); }
"#;

#[test]
fn build_places_and_runs_stable_artifact_then_replaces_it() {
    let package = Package::new("build-app", SCALAR);
    let first = package.run(&["build", "--offline"]);
    assert!(first.status.success(), "{}", stderr(&first));
    let final_path = package.root.join("target/stark/debug/build-app");
    assert!(final_path.is_file());
    assert!(stdout(&first).contains(&format!(
        "Built build-app [debug] -> {}",
        final_path.display()
    )));
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

#[test]
fn aggregate_program_builds_offline_with_empty_cargo_home() {
    let package = Package::new(
        "aggregate",
        r#"
struct Pair { left: i32, right: i32 }
fn main() { let pair = Pair { left: 19, right: 23 }; assert_eq(pair.left + pair.right, 42); }
"#,
    );
    let cargo_home = package.root.join("empty-cargo-home");
    std::fs::create_dir(&cargo_home).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_stark"))
        .args(["build", "--offline", "--keep-generated"])
        .env("CARGO_HOME", &cargo_home)
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
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
    let old = package.root.join("target/stark/debug/source-error");
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

    let missing_runtime = Command::new(env!("CARGO_BIN_EXE_stark"))
        .arg("build")
        .env("STARK_RUNTIME_DIR", package.root.join("missing-runtime"))
        .current_dir(&package.root)
        .output()
        .unwrap();
    assert!(stderr(&missing_runtime).contains("native runtime installation is missing"));
}
