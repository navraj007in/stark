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
fn env_var_success_and_recoverable_invalid_name_execute_through_build_command() {
    let root = fixture_root("env");
    let _ = std::fs::remove_dir_all(&root);
    write_package(
        &root,
        r#"{
  "name": "c788_env",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["process.env"],
  "provider_api": {
    "errors": { "process.env": "RawEnvError" },
    "functions": {
      "var_len": {
        "capability": "process.env",
        "symbol": "stark_env_var_len"
      },
      "var_fill": {
        "capability": "process.env",
        "symbol": "stark_env_var_fill"
      }
    }
  }
}"#,
        r#"fn main() {
    let name = "STARK_C788_ENV_PROBE".bytes();
    match var_len(name) {
        Ok(result) => {
            println(result);
        }
        Err(_e) => {
            panic("env var_len failed for a valid name");
        }
    }

    let mut out: [UInt8; 5] = [0u8; 5];
    let out_slice = &mut out[0..5];
    match var_fill(name, out_slice) {
        Ok(n) => {
            println(n);
            println(out[0u64]);
            println(out[1u64]);
            println(out[2u64]);
            println(out[3u64]);
            println(out[4u64]);
        }
        Err(_e) => {
            panic("env var_fill failed for a valid name");
        }
    }

    let invalid = "".bytes();
    match var_len(invalid) {
        Ok(_result) => {
            panic("empty environment variable names must be recoverable errors");
        }
        Err(RawEnvError::InvalidName) => {
            println(1);
        }
        Err(_e) => {
            panic("empty environment variable name produced the wrong recoverable error");
        }
    }
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
        .env("STARK_C788_ENV_PROBE", "Codex")
        .output()
        .unwrap_or_else(|error| panic!("run built binary: {error}"));
    assert!(
        output.status.success(),
        "built program failed; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(
        stdout.trim(),
        "(true, 5)\n5\n67\n111\n100\n101\n120\n1",
        "the built binary must print the observed env value bytes and the InvalidName Err arm"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_resource_bearing_provider_api_is_no_longer_refused_categorically() {
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

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    );

    // **Inverted deliberately (CD-248).** This asserted the driver refuses any resource-bearing
    // provider API. That refusal existed because a resource obtained through a build could never be
    // released -- no close arena, no Drop-terminator close. Both now exist (CD-237/CD-239/CD-240),
    // and the driver selects a close for every bound resource, so the categorical refusal is gone.
    //
    // The assertion is deliberately narrow: whatever else this build does, it must not fail
    // BECAUSE the signature carries a resource. A stronger claim belongs in the lifecycle e2e,
    // which executes the close rather than inspecting a diagnostic.
    let rendered = match &result {
        Ok(_) => String::new(),
        Err(error) => format!("{error:?}"),
    };
    assert!(
        !rendered.contains("resource-bearing provider signature"),
        "the categorical resource refusal must be gone; got:\n{rendered}"
    );

    let _ = std::fs::remove_dir_all(&root);
}
