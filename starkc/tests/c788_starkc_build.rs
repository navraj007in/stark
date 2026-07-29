//! WP-C7.8.8 — provider API synthesis through the normal `starkc build` path.
//!
//! Earlier source-level proofs drove the compiler library directly. These tests go through
//! `native_build::build_current_package`, so they exercise manifest parsing, provider selection,
//! synthesis overlays, `lower_program_with_providers`, generated Rust, linking, and execution as the
//! command does.

use starkc::native_build::{build_current_package, BuildCommandOptions};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn fixture_root(name: &str) -> PathBuf {
    repo_root()
        .join("target")
        .join("c788-starkc-build")
        .join(format!("{name}-{}", std::process::id()))
}

fn write_package(root: &std::path::Path, manifest: &str, source: &str) {
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create package src dir");
    std::fs::write(root.join("starkpkg.json"), manifest).expect("write manifest");
    std::fs::write(src.join("main.stark"), source).expect("write source");
}

#[test]
fn args_len_executes_from_source_through_build_command() {
    let root = fixture_root("args");
    let _ = std::fs::remove_dir_all(&root);
    write_package(
        &root,
        r#"{
  "name": "c788_args",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["process.args"],
  "provider_api": {
    "errors": { "process.args": "RawArgsError" },
    "functions": {
      "args_len": {
        "capability": "process.args",
        "symbol": "stark_env_args_len"
      }
    }
  }
}"#,
        r#"fn main() {
    let mut count: UInt64 = 0u64;
    match args_len() {
        Ok(n) => { count = n; }
        Err(_e) => { panic("args provider failed"); }
    }
    println(count);
}"#,
    );

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("starkc build path must succeed: {error:?}"));

    let output = std::process::Command::new(&result.artifact_path)
        .args(["alpha", "beta"])
        .output()
        .unwrap_or_else(|error| panic!("run built binary: {error}"));
    assert!(
        output.status.success(),
        "built program failed; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    let count: u64 = stdout
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("expected an argv count, got {stdout:?}: {error}"));
    assert!(
        count >= 2,
        "the provider must observe the controlled source-level argv, got {count}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn resource_bearing_provider_api_is_reported_as_out_of_scope_for_this_slice() {
    let root = fixture_root("resource");
    let _ = std::fs::remove_dir_all(&root);
    write_package(
        &root,
        r#"{
  "name": "c788_resource",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["filesystem"],
  "provider_api": {
    "errors": { "filesystem": "RawIoError" },
    "functions": {
      "open_raw": {
        "capability": "filesystem",
        "symbol": "stark_file_open"
      }
    }
  }
}"#,
        "fn main() { }\n",
    );

    let error = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    )
    .expect_err("resource-bearing provider API must not be partially lowered in this slice");
    let rendered = format!("{error:?}");
    assert!(
        rendered.contains("resource-bearing provider signature"),
        "{rendered}"
    );

    let _ = std::fs::remove_dir_all(&root);
}
