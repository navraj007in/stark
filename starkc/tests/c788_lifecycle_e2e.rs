//! WP-C7.8.8 — **the host-resource lifecycle, observed by execution.**
//!
//! CD-234 required a set of lifecycle guarantees. Three are already proven statically:
//! `c788_synth` shows the synthesized nominal cannot be constructed or matched into existence, and
//! `a11_host_resource` shows MIR refuses to manufacture or default-initialise one and that generated
//! Rust carries no placeholder handle.
//!
//! The rest are runtime properties, and until now they were argued structurally and unit-tested but
//! never **observed**. This file observes them.
//!
//! **How a violation is detected.** `stark_tcp_stream_close` calls `abort_contract()` when the handle
//! is not in the provider's live table — so closing a resource that was never opened, or closing one
//! twice, aborts the process. Every test here therefore asserts the built binary *exits cleanly and
//! prints its marker*: a spurious or duplicated close cannot pass silently, because the provider
//! itself refuses it.
//!
//! That inversion is deliberate. Asserting on a diagnostic or on emitted text would prove what the
//! compiler *intended*; asserting on a clean exit under a provider that aborts on misuse proves what
//! actually happened.
//!
//! **The one case with no vehicle**, stated rather than faked: "a consuming close prevents a later
//! implicit close" cannot be written here. `stark-net`'s only `HandleConsumed` functions are its two
//! closes, a package may not bind a close (design §2), and `MIR-0033` rejects a direct call to one.
//! MIR owns the only close path, so "explicit close then implicit close" is unreachable from source
//! by construction — which `a11_host_resource`'s rule-4 tests pin directly.

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
        .join("c788-lifecycle")
        .join(format!("{name}-{}", std::process::id()))
}

fn manifest(name: &str) -> String {
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
      "tcp_listener_bind": {{ "capability": "tcp", "symbol": "stark_tcp_listener_bind" }},
      "tcp_listener_accept": {{ "capability": "tcp", "symbol": "stark_tcp_listener_accept" }},
      "tcp_stream_connect": {{ "capability": "tcp", "symbol": "stark_tcp_stream_connect" }}
    }}
  }}
}}"#
    )
}

/// Builds and runs a program, returning `(stdout, generated Rust)`.
///
/// Panics if it does not exit cleanly — which is how a spurious or duplicated close is caught: the
/// provider aborts. The generated Rust is returned because a clean exit proves only that the close
/// did not happen *twice*; showing it happens *at all* needs the emitted code.
fn build_and_run(case: &str, source: &str) -> (String, String) {
    build_and_run_with(case, source, || {})
}

/// `build_and_run`, with `before_run` invoked **after the build and before execution**.
///
/// The accept case needs a peer dialling the server, and the build takes seconds — a peer spawned
/// before the build gives up long before the server binds, and the server then blocks in `accept`
/// forever. Sequencing it here makes the race impossible rather than unlikely.
fn build_and_run_with(case: &str, source: &str, before_run: impl FnOnce()) -> (String, String) {
    let root = fixture_root(case);
    let _ = std::fs::remove_dir_all(&root);
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create package dir");
    std::fs::write(
        root.join("starkpkg.json"),
        manifest(&format!("c788_{case}")),
    )
    .expect("write manifest");
    std::fs::write(src.join("main.stark"), source).expect("write source");

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            // `keep_generated` so the emitted Rust survives for inspection; `no_build_cache` is
            // deliberately OFF here, since deleting the crate would take the generated source with
            // it and leave nothing to check the close against.
            keep_generated: true,
            emit_rust: true,
            ..BuildCommandOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{case}: build must succeed: {e:?}"));

    before_run();
    let output = std::process::Command::new(&result.artifact_path)
        .output()
        .unwrap_or_else(|e| panic!("{case}: running the binary: {e}"));

    assert!(
        output.status.success(),
        "{case}: the program must exit cleanly. A non-zero exit here is most likely the provider \
         aborting on a close it did not expect -- which is exactly the violation under test.\n\
         stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let generated = result
        .generated_rust
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default();
    let stdout = String::from_utf8(output.stdout).expect("stdout is UTF-8");
    let _ = std::fs::remove_dir_all(&root);
    (stdout, generated)
}

/// An address nothing is listening on, so `connect` fails and its `HandleOut` destination is never
/// written. Port 1 is privileged and unused; a connection there fails promptly.
const DEAD_ADDRESS: &str = "127.0.0.1:1";

// ------------------------------------------------------------- the cases --

/// **A declared-but-never-initialised resource does not close.**
///
/// The local is declared and never assigned, so its slot stays dead. If drop elaboration closed it
/// anyway, the provider would receive a handle it never issued and abort. The clean exit is the
/// proof.
#[test]
fn a_never_initialised_resource_does_not_close() {
    let (out, _) = build_and_run(
        "never_init",
        "fn main() {\n\
         \x20   let _s: TcpStream;\n\
         \x20   println(\"reached-end\");\n\
         }\n",
    );
    assert_eq!(out.trim(), "reached-end");
}

/// **A failed `HandleOut` does not close.**
///
/// `connect` to a dead address returns `Err`, so the destination slot is never written and stays
/// dead. CD-234 requires the slot to begin dead and only success to make it live; if the failure
/// path left it live, scope exit would close a handle the provider never issued and abort.
#[test]
fn a_failed_handle_out_does_not_close() {
    let (out, _) = build_and_run(
        "failed_out",
        &format!(
            "fn main() {{\n\
             \x20   let address = \"{DEAD_ADDRESS}\".bytes();\n\
             \x20   match tcp_stream_connect(address) {{\n\
             \x20       Ok(_s) => {{ println(\"unexpected-connect\"); }}\n\
             \x20       Err(_e) => {{ println(\"connect-failed\"); }}\n\
             \x20   }}\n\
             }}\n"
        ),
    );
    assert_eq!(
        out.trim(),
        "connect-failed",
        "the connection must fail for this test to be meaningful"
    );
}

/// **A successful `HandleOut` closes exactly once.**
///
/// One connection to a live listener, dropped at scope exit. A second close would find the handle
/// gone from the provider's table and abort, so a clean exit proves *at most* one close — and the
/// close must have happened, since the drop path is unconditional for a live slot.
#[test]
fn a_successful_handle_out_closes_exactly_once() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a local listener");
    let address = listener.local_addr().expect("listener address").to_string();
    let accepting = std::thread::spawn(move || {
        // Hold the accepted connection open until the client has exited.
        let _ = listener.accept();
        std::thread::sleep(std::time::Duration::from_millis(200));
    });

    let (out, generated) = build_and_run(
        "close_once",
        &format!(
            "fn main() {{\n\
             \x20   let address = \"{address}\".bytes();\n\
             \x20   match tcp_stream_connect(address) {{\n\
             \x20       Ok(_s) => {{ println(\"connected\"); }}\n\
             \x20       Err(_e) => {{ panic(\"connect failed\"); }}\n\
             \x20   }}\n\
             }}\n"
        ),
    );
    assert_eq!(out.trim(), "connected");

    // **The other half of "exactly once".** A clean exit proves the close did not happen TWICE --
    // the provider aborts on the second. That it happens AT ALL is shown here: the generated Rust
    // closes the handle through the slot's liveness guard, taking the handle rather than borrowing
    // it, which is what makes a second close impossible rather than merely unlikely.
    assert!(
        generated.contains("stark_tcp_stream_close(__v.as_raw())"),
        "the emitted code must close the stream, consuming the handle:\n{generated}"
    );
    assert!(
        generated.contains(".drop_with(|__v| unsafe"),
        "and it must go through slot liveness, so a dead slot closes nothing"
    );
    let _ = accepting.join();
}

/// **Move then drop closes only the destination.**
///
/// `let b = a;` moves the handle. The source is dead afterwards — which CD-251 made true by amending
/// `OWN-COPY-001`, since a vacuously-`Copy` zero-variant enum would have *duplicated* it here instead.
/// If both locals closed, the second would abort.
#[test]
fn move_then_drop_closes_only_the_destination() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a local listener");
    let address = listener.local_addr().expect("listener address").to_string();
    let accepting = std::thread::spawn(move || {
        let _ = listener.accept();
        std::thread::sleep(std::time::Duration::from_millis(200));
    });

    let (out, _) = build_and_run(
        "move_once",
        &format!(
            "fn main() {{\n\
             \x20   let address = \"{address}\".bytes();\n\
             \x20   match tcp_stream_connect(address) {{\n\
             \x20       Ok(first) => {{\n\
             \x20           let _second = first;\n\
             \x20           println(\"moved\");\n\
             \x20       }}\n\
             \x20       Err(_e) => {{ panic(\"connect failed\"); }}\n\
             \x20   }}\n\
             }}\n"
        ),
    );
    assert_eq!(out.trim(), "moved");
    let _ = accepting.join();
}

// ---------------------------------- the remaining CD-234 lifecycle cases --

/// **Accept/release: two resources, closed independently.**
///
/// A listener and the stream it accepts are *different* resources with *different* closes —
/// `stark_tcp_listener_close` and `stark_tcp_stream_close` have identical shapes and differ only in
/// the resource they name, which is the pairing `MIR-0030` exists to enforce. If either close were
/// selected for the wrong resource, the provider's `validate(handle, RESOURCE_TYPE)` would reject the
/// handle and abort.
#[test]
fn accept_and_release_close_two_resources_independently() {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("probe a free port");
    let address = probe.local_addr().expect("addr").to_string();
    drop(probe);

    let connecting_address = address.clone();
    let mut connector = None;

    let (out, generated) = build_and_run_with(
        "accept_release",
        &format!(
            "fn main() {{\n\
             \x20   let address = \"{address}\".bytes();\n\
             \x20   let listener: TcpListener;\n\
             \x20   match tcp_listener_bind(address) {{\n\
             \x20       Ok(l) => {{ listener = l; }}\n\
             \x20       Err(_e) => {{ panic(\"bind failed\"); }}\n\
             \x20   }}\n\
             \x20   match tcp_listener_accept(&listener) {{\n\
             \x20       Ok(_stream) => {{ println(\"accepted\"); }}\n\
             \x20       Err(_e) => {{ panic(\"accept failed\"); }}\n\
             \x20   }}\n\
             }}\n"
        ),
        || {
            // Spawned only now: the binary exists and is about to run, so the peer's retry window
            // covers the server's bind rather than the compiler's build.
            connector = Some(std::thread::spawn(move || {
                for _ in 0..100 {
                    if let Ok(s) = std::net::TcpStream::connect(&connecting_address) {
                        std::thread::sleep(std::time::Duration::from_millis(150));
                        drop(s);
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }));
        },
    );
    assert_eq!(out.trim(), "accepted");
    if let Some(handle) = connector {
        let _ = handle.join();
    }

    // Each resource is closed by ITS OWN provider function. A single close serving both would be the
    // paired-resource confusion A11 §5 obligation 4 names, and no shape check would catch it.
    assert!(
        generated.contains("stark_tcp_listener_close(__v.as_raw())"),
        "the listener must close through its own close:\n{generated}"
    );
    assert!(
        generated.contains("stark_tcp_stream_close(__v.as_raw())"),
        "the accepted stream must close through its own close:\n{generated}"
    );
}

/// **An early return with a live resource still closes it.**
///
/// The resource is live when `return` executes, so the return path must run the `Drop` — a leak here
/// would be invisible at runtime (the provider is never told), which is why this asserts on the
/// generated code as well as on a clean exit.
#[test]
fn an_early_return_with_a_live_resource_closes_it() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let accepting = std::thread::spawn(move || {
        let _ = listener.accept();
        std::thread::sleep(std::time::Duration::from_millis(200));
    });

    let (out, generated) = build_and_run(
        "early_return",
        &format!(
            "fn hold(a: &[UInt8]) {{\n\
             \x20   match tcp_stream_connect(a) {{\n\
             \x20       Ok(_s) => {{ println(\"held\"); return; }}\n\
             \x20       Err(_e) => {{ panic(\"connect failed\"); }}\n\
             \x20   }}\n\
             }}\n\
             fn main() {{\n\
             \x20   let address = \"{address}\".bytes();\n\
             \x20   hold(address);\n\
             \x20   println(\"done\");\n\
             }}\n"
        ),
    );
    assert_eq!(out.trim().lines().collect::<Vec<_>>(), vec!["held", "done"]);
    assert!(
        generated.contains("stark_tcp_stream_close(__v.as_raw())"),
        "the early-return path must still close the live resource:\n{generated}"
    );
    let _ = accepting.join();
}

/// **A resource moved through a call transfers the close obligation to the callee.**
///
/// The caller's local is dead after the move, so exactly one close runs — in the callee's scope. If
/// the move left the caller's slot live, both would close and the second would abort.
#[test]
fn a_resource_moved_through_a_call_closes_once_in_the_callee() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("addr").to_string();
    let accepting = std::thread::spawn(move || {
        let _ = listener.accept();
        std::thread::sleep(std::time::Duration::from_millis(200));
    });

    let (out, _) = build_and_run(
        "move_through_call",
        &format!(
            "fn consume(s: TcpStream) {{\n\
             \x20   println(\"consumed\");\n\
             }}\n\
             fn main() {{\n\
             \x20   let address = \"{address}\".bytes();\n\
             \x20   match tcp_stream_connect(address) {{\n\
             \x20       Ok(s) => {{ consume(s); println(\"after\"); }}\n\
             \x20       Err(_e) => {{ panic(\"connect failed\"); }}\n\
             \x20   }}\n\
             }}\n"
        ),
    );
    assert_eq!(
        out.trim().lines().collect::<Vec<_>>(),
        vec!["consumed", "after"],
        "the callee runs, then the caller continues with a dead local"
    );
    let _ = accepting.join();
}
