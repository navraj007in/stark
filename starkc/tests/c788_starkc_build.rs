//! WP-C7.8.8 — provider API synthesis through the normal `starkc build` path.
//!
//! Earlier source-level proofs drove the compiler library directly. These tests go through
//! `native_build::build_current_package`, so they exercise manifest parsing, provider selection,
//! synthesis overlays, `lower_program_with_providers`, generated Rust, linking, and execution as the
//! command does.

use starkc::native_build::{build_current_package, BuildCommandOptions};
use std::io::Read;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

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

fn write_package_with_entry(root: &std::path::Path, manifest: &str, entry: &str, source: &str) {
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create package src dir");
    std::fs::write(root.join("starkpkg.json"), manifest).expect("write manifest");
    std::fs::write(root.join(entry), source).expect("write source");
}

fn tcp_manifest(name: &str) -> String {
    format!(
        r#"{{
  "name": "{name}",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["tcp"],
  "provider_api": {{
    "errors": {{ "tcp": "RawNetworkError" }},
    "resources": {{
      "TcpListener": {{ "capability": "tcp", "resource": "tcp_listener" }},
      "TcpStream": {{ "capability": "tcp", "resource": "tcp_stream" }}
    }},
    "functions": {{
      "tcp_listener_bind": {{
        "capability": "tcp",
        "symbol": "stark_tcp_listener_bind"
      }},
      "tcp_listener_accept": {{
        "capability": "tcp",
        "symbol": "stark_tcp_listener_accept"
      }},
      "tcp_stream_connect": {{
        "capability": "tcp",
        "symbol": "stark_tcp_stream_connect"
      }},
      "tcp_stream_read": {{
        "capability": "tcp",
        "symbol": "stark_tcp_stream_read"
      }},
      "tcp_stream_write": {{
        "capability": "tcp",
        "symbol": "stark_tcp_stream_write"
      }}
    }}
  }}
}}"#
    )
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
fn tcp_bind_accept_connect_and_echo_execute_from_source_through_build_command() {
    let probe = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
    let address = probe.local_addr().expect("read loopback address");
    drop(probe);
    let address = address.to_string();

    let server_root = fixture_root("tcp-server");
    let client_root = fixture_root("tcp-client");
    let _ = std::fs::remove_dir_all(&server_root);
    let _ = std::fs::remove_dir_all(&client_root);

    let server_source = format!(
        r#"fn main() {{
    let address = "{address}".bytes();
    let listener: TcpListener;
    match tcp_listener_bind(address) {{
        Ok(value) => {{ listener = value; }}
        Err(_e) => {{ panic("tcp bind failed"); return; }}
    }}
    let stream: TcpStream;
    match tcp_listener_accept(&listener) {{
        Ok(value) => {{ stream = value; }}
        Err(_e) => {{ panic("tcp accept failed"); return; }}
    }}
    let mut buffer: [UInt8; 7] = [0u8; 7];
    {{
        let out = &mut buffer[0..7];
        match tcp_stream_read(&stream, out) {{
            Ok(n) => {{ assert_eq(n, 7u64); }}
            Err(_e) => {{ panic("tcp read failed"); }}
        }}
    }}
    let echo = &buffer[0..7];
    match tcp_stream_write(&stream, echo) {{
        Ok(n) => {{ assert_eq(n, 7u64); }}
        Err(_e) => {{ panic("tcp write failed"); }}
    }}
    println("server-done");
}}"#
    );
    write_package(
        &server_root,
        &tcp_manifest("c788_tcp_server"),
        server_source.as_str(),
    );
    let client_source = format!(
        r#"fn main() {{
    let address = "{address}".bytes();
    let stream: TcpStream;
    match tcp_stream_connect(address) {{
        Ok(value) => {{ stream = value; }}
        Err(_e) => {{ panic("tcp connect failed"); return; }}
    }}
    let payload = "stark!!".bytes();
    match tcp_stream_write(&stream, payload) {{
        Ok(n) => {{ assert_eq(n, 7u64); }}
        Err(_e) => {{ panic("tcp write failed"); }}
    }}
    let mut buffer: [UInt8; 7] = [0u8; 7];
    {{
        let out = &mut buffer[0..7];
        match tcp_stream_read(&stream, out) {{
            Ok(n) => {{ assert_eq(n, 7u64); }}
            Err(_e) => {{ panic("tcp read failed"); }}
        }}
    }}
    println(buffer[0u64]);
    println(buffer[1u64]);
    println(buffer[2u64]);
    println(buffer[3u64]);
    println(buffer[4u64]);
    println(buffer[5u64]);
    println(buffer[6u64]);
}}"#
    );
    write_package(
        &client_root,
        &tcp_manifest("c788_tcp_client"),
        client_source.as_str(),
    );

    let options = BuildCommandOptions {
        no_build_cache: true,
        ..BuildCommandOptions::default()
    };
    let server = build_current_package(&server_root, &options)
        .unwrap_or_else(|error| panic!("server build must succeed: {error:?}"));
    let client = build_current_package(&client_root, &options)
        .unwrap_or_else(|error| panic!("client build must succeed: {error:?}"));

    let mut server_child = std::process::Command::new(&server.artifact_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("spawn source-built TCP server: {error}"));

    let deadline = Instant::now() + Duration::from_secs(5);
    let client_output = loop {
        let output = std::process::Command::new(&client.artifact_path)
            .output()
            .unwrap_or_else(|error| panic!("run source-built TCP client: {error}"));
        if output.status.success() {
            break output;
        }
        assert!(
            Instant::now() < deadline,
            "client never connected; last stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::thread::sleep(Duration::from_millis(25));
    };

    let server_status = loop {
        if let Some(status) = server_child
            .try_wait()
            .unwrap_or_else(|error| panic!("poll source-built TCP server: {error}"))
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = server_child.kill();
            let _ = server_child.wait();
            panic!("source-built TCP server did not exit after client completed");
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let mut server_stdout = String::new();
    if let Some(mut stdout) = server_child.stdout.take() {
        stdout
            .read_to_string(&mut server_stdout)
            .expect("server stdout utf8-ish");
    }
    let mut server_stderr = String::new();
    if let Some(mut stderr) = server_child.stderr.take() {
        stderr
            .read_to_string(&mut server_stderr)
            .expect("server stderr utf8-ish");
    }
    assert!(
        server_status.success(),
        "source-built TCP server failed; stdout:\n{server_stdout}\nstderr:\n{server_stderr}"
    );

    let client_stdout = String::from_utf8(client_output.stdout).expect("client stdout utf8");
    assert_eq!(
        client_stdout.trim(),
        "115\n116\n97\n114\n107\n33\n33",
        "client must print the echoed bytes for `stark!!`"
    );
    assert_eq!(server_stdout.trim(), "server-done");

    let _ = std::fs::remove_dir_all(&server_root);
    let _ = std::fs::remove_dir_all(&client_root);
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

/// **WP-IO.1's end-to-end, running on its own resource identity.** Real STARK source opening,
/// writing, reading and closing a file through the `stark-io` package and the first-party
/// filesystem provider.
///
/// This was `#[ignore]`d for one commit, and the reason is worth keeping. The slice originally bound
/// `NativeFile` to the resource `file` — which is Core-owned, so CD-224 rejects it — and was made to
/// run by deleting that guard plus two verifier guards. The result was the state SELECT-C exists to
/// refuse: `file` reachable as a `HostResource` for some rules while Core `File` kept legacy
/// direct-close semantics. One resource name, two MIR representations, two destruction paths.
///
/// The fix was not to migrate Core `File` (a three-engine change, still open) but to notice the
/// package never needed Core's identity in the first place. `stark-io` now binds **`io_file`**: its
/// own resource type, absent from the builtin registry, wholly on the `HostResource` path. Every
/// guard is intact and unexempted — the handle is owned, moved and closed exactly once from a `Drop`
/// terminator, exactly as `tcp_stream`'s is.
#[test]
fn io_minimal_executes_from_source_through_stark_io_package() {
    let root = fixture_root("io-minimal");
    let io_dir = root.join("io-data");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&io_dir).expect("create io data dir");
    let vendored_io = root.join("vendor").join("stark-io");
    std::fs::create_dir_all(vendored_io.join("src")).expect("create vendored stark-io");
    std::fs::copy(
        repo_root().join("stark-io").join("starkpkg.json"),
        vendored_io.join("starkpkg.json"),
    )
    .expect("copy stark-io manifest");
    std::fs::copy(
        repo_root().join("stark-io").join("src").join("lib.stark"),
        vendored_io.join("src").join("lib.stark"),
    )
    .expect("copy stark-io source");
    let file_path = io_dir.join("sample.txt");
    let file_path = file_path
        .to_str()
        .expect("fixture path must be valid utf8")
        .replace('\\', "\\\\");
    let missing_path = io_dir
        .join("missing.txt")
        .to_str()
        .expect("fixture path must be valid utf8")
        .replace('\\', "\\\\");

    write_package_with_entry(
        &root,
        r#"{
  "name": "c788_io_minimal",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["filesystem"],
  "dependencies": {
    "stark_io": {
      "package": "stark-io",
      "path": "vendor/stark-io"
    }
  }
}"#,
        "src/main.stark",
        &format!(
            r#"use stark_io::NativeFile;
use stark_io::FileError;
use stark_io::file_close;
use stark_io::file_read;
use stark_io::open_file;
use stark_io::read_text;
use stark_io::write_text;

fn main() {{
    let path = "{file_path}";
    match write_text(path, "hello stark") {{
        Ok(_) => {{ }}
        Err(_) => {{ panic("write_text failed"); }}
    }}

    match read_text(path, 64u64) {{
        Ok(text) => {{
            if text.as_str() != "hello stark" {{
                panic("read_text mismatch");
            }}
        }}
        Err(_) => {{ panic("read_text failed"); }}
    }}

    let file: NativeFile;
    match open_file(path) {{
        Ok(value) => {{ file = value; }}
        Err(_) => {{ panic("open failed"); return; }}
    }}
    let mut buffer: [UInt8; 5] = [0u8; 5];
    {{
        let out = &mut buffer[0u64..5u64];
        match file_read(&file, out) {{
            Ok(count) => {{
                if count != 5u64 {{
                    panic("read count mismatch");
                }}
            }}
            Err(_) => {{ panic("read failed"); }}
        }}
    }}
    if buffer[0u64] != 104u8 || buffer[1u64] != 101u8 || buffer[2u64] != 108u8 || buffer[3u64] != 108u8 || buffer[4u64] != 111u8 {{
        panic("read bytes mismatch");
    }}
    match file_close(file) {{
        Ok(_) => {{ }}
        Err(_) => {{ panic("close failed"); }}
    }}

    match open_file("{missing_path}") {{
        Ok(_file) => {{ panic("missing file opened"); }}
        Err(FileError::NotFound) => {{ println("io-minimal-ok"); }}
        Err(_) => {{ panic("missing file wrong error"); }}
    }}
}}"#
        ),
    );

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("stark-io source package build must succeed: {error:?}"));

    let output = std::process::Command::new(&result.artifact_path)
        .output()
        .unwrap_or_else(|error| panic!("run stark-io minimal binary: {error}"));
    assert!(
        output.status.success(),
        "built program failed; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(stdout.trim(), "io-minimal-ok");

    let _ = std::fs::remove_dir_all(&root);
}
