//! WP-C7.8.8 step 1 — `provider_api` parsing and validation.
//!
//! The manifest binding names a **callable surface**, never a signature. CD-224's standing
//! invariant: validated provider metadata is the one authoritative signature, and a binding carries
//! only identity, so there is no second copy to drift. CD-219 is the evidence — a mirrored
//! `unix_now` declared one out-slot where the provider declared two, and metadata validation could
//! not see it because the wrong mirror was internally consistent.
//!
//! This file covers what is checkable **without** provider metadata. Symbol existence, resource
//! existence, close-binding rejection and signature derivability need a selected provider and are
//! validated at selection, not here.

use starkc::package::Package;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

fn temp_manifest(body: &str) -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "c788-manifest-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(dir.join("src")).expect("temp dir");
    std::fs::write(dir.join("src").join("main.stark"), "fn main() {}\n").expect("entry");
    let path = dir.join("starkpkg.json");
    std::fs::write(&path, body).expect("write manifest");
    path
}

fn parse(body: &str) -> Result<Package, String> {
    Package::from_manifest(&temp_manifest(body))
}

fn ok(body: &str) -> Package {
    parse(body).unwrap_or_else(|e| panic!("manifest must parse: {e}"))
}

fn err(body: &str) -> String {
    parse(body).expect_err("manifest must be rejected")
}

const HEAD: &str = r#""name":"p","version":"0.1.0","entry":"src/main.stark""#;

// ------------------------------------------------------------- the common case --

/// A package binding nothing has an empty `provider_api`, and the field is absent from almost every
/// manifest. A regression here would not fail loudly — it would start attaching bindings to
/// packages that declared none.
#[test]
fn a_package_without_provider_api_binds_nothing() {
    let p = ok(&format!("{{{HEAD}}}"));
    assert!(p.provider_api.functions.is_empty());
    assert!(p.provider_api.resources.is_empty());
    assert!(p.provider_api.errors.is_empty());
}

// ------------------------------------------------------------------- time (§9.1) --

/// The worked `std-time` declaration: two functions, one capability, one raw error type. Note what
/// is **absent** — no parameter types, no ownership modes, no return shape. Those are derived from
/// validated metadata.
#[test]
fn the_time_declaration_parses_and_carries_only_identity() {
    let p = ok(&format!(
        r#"{{{HEAD},"capabilities":["clock"],
            "provider_api":{{
              "errors":{{"clock":"RawTimeError"}},
              "functions":{{
                "Instant::now_ns":{{"capability":"clock","symbol":"stark_time_monotonic_now_ns"}},
                "SystemTime::unix_now":{{"capability":"clock","symbol":"stark_time_unix_now"}}
              }}}}}}"#
    ));

    assert_eq!(p.provider_api.functions.len(), 2);
    let f = &p.provider_api.functions[0];
    assert_eq!(f.item_path, "Instant::now_ns");
    assert_eq!(f.capability, "clock");
    assert_eq!(f.symbol, "stark_time_monotonic_now_ns");
    assert_eq!(
        p.provider_api.errors,
        vec![("clock".to_string(), "RawTimeError".to_string())]
    );
    assert!(p.provider_api.resources.is_empty());
}

/// Bindings are sorted, so manifest key order reaches neither the build key nor generated code —
/// the property CD-213 gave capabilities and CD-205 gave the status vocabulary.
#[test]
fn bindings_are_sorted_so_key_order_cannot_escape() {
    let forward = ok(&format!(
        r#"{{{HEAD},"capabilities":["clock"],
            "provider_api":{{"errors":{{"clock":"E"}},"functions":{{
              "Zeta::z":{{"capability":"clock","symbol":"stark_time_unix_now"}},
              "Alpha::a":{{"capability":"clock","symbol":"stark_time_monotonic_now_ns"}}
            }}}}}}"#
    ));
    let reverse = ok(&format!(
        r#"{{{HEAD},"capabilities":["clock"],
            "provider_api":{{"errors":{{"clock":"E"}},"functions":{{
              "Alpha::a":{{"capability":"clock","symbol":"stark_time_monotonic_now_ns"}},
              "Zeta::z":{{"capability":"clock","symbol":"stark_time_unix_now"}}
            }}}}}}"#
    ));
    assert_eq!(forward.provider_api, reverse.provider_api);
    assert_eq!(forward.provider_api.functions[0].item_path, "Alpha::a");
}

// ---------------------------------------------------------------- TCP (§9.4/9.5) --

/// The package nominal path: two resources, five functions, both closes absent.
#[test]
fn the_tcp_declaration_binds_nominals_to_provider_resources() {
    let p = ok(&format!(
        r#"{{{HEAD},"capabilities":["network-client","network-listen"],
            "provider_api":{{
              "errors":{{"network-client":"RawNetError","network-listen":"RawNetError"}},
              "resources":{{
                "TcpListener":{{"capability":"network-listen","resource":"tcp_listener"}},
                "TcpStream":{{"capability":"network-client","resource":"tcp_stream"}}
              }},
              "functions":{{
                "TcpListener::bind_raw":{{"capability":"network-listen","symbol":"stark_tcp_listener_bind"}},
                "TcpStream::connect_raw":{{"capability":"network-client","symbol":"stark_tcp_stream_connect"}}
              }}}}}}"#
    ));

    assert_eq!(p.provider_api.resources.len(), 2);
    assert_eq!(p.provider_api.resources[0].nominal, "TcpListener");
    assert_eq!(p.provider_api.resources[0].resource, "tcp_listener");
    assert_eq!(p.provider_api.resources[1].nominal, "TcpStream");
}

// ------------------------------------------------------------- negative cases --

/// §13.6 — a binding is a *use* of a capability, so Packet 5's admission rule applies. Without this
/// a package could reach a provider it never required.
#[test]
fn a_binding_for_an_undeclared_capability_is_rejected() {
    let message = err(&format!(
        r#"{{{HEAD},"capabilities":["clock"],
            "provider_api":{{"errors":{{"clock":"E"}},"functions":{{
              "T::c":{{"capability":"network-client","symbol":"stark_tcp_stream_connect"}}
            }}}}}}"#
    ));
    assert!(message.contains("network-client"), "{message}");
    assert!(message.contains("capabilities"), "{message}");
}

/// **CD-224: Core resources are compiler-owned.** A package declaring `file` would be claiming
/// authority over a Core type — exactly what the two-mechanism ruling forbids — so it is rejected
/// rather than silently shadowing the built-in.
///
/// **WP-IO.1 inverted this test** to `a_package_may_declare_the_file_resource_for_stark_io`, so that
/// `stark-io` could bind `NativeFile` to the Core-owned resource `file`. Restored, because the
/// binding it enabled is the hybrid SELECT-C exists to prevent: a package nominal on the
/// `HostResource` path carrying Core `File`'s resource identity, while `File` keeps its legacy
/// direct-close semantics. One resource name, two MIR representations, two destruction paths.
///
/// The IO slice's need is real and is not answered by weakening this. It is answered by migrating
/// `file` off the legacy path WHOLLY (Route B), which `partially_migrated_core` already permits and
/// this guard does not obstruct. See `stark-io/BLOCKERS.md`.
#[test]
fn a_package_may_not_declare_a_core_resource() {
    let message = err(&format!(
        r#"{{{HEAD},"capabilities":["filesystem-read"],
            "provider_api":{{"resources":{{
              "MyFile":{{"capability":"filesystem-read","resource":"file"}}
            }}}}}}"#
    ));
    assert!(message.contains("file"), "{message}");
    assert!(message.contains("Core"), "{message}");
}

/// §13.3 — **rejected, not warned.** Two nominals on one resource would be distinct STARK types
/// that are identical at the boundary: one would satisfy the other dynamically while failing
/// statically, and each would record its own close for one resource, breaking exactly-once.
#[test]
fn two_nominals_on_one_resource_are_rejected() {
    let message = err(&format!(
        r#"{{{HEAD},"capabilities":["network-client"],
            "provider_api":{{"resources":{{
              "TcpStream":{{"capability":"network-client","resource":"tcp_stream"}},
              "Socket":{{"capability":"network-client","resource":"tcp_stream"}}
            }}}}}}"#
    ));
    assert!(message.contains("tcp_stream"), "{message}");
    assert!(
        message.contains("Socket") && message.contains("TcpStream"),
        "{message}"
    );
}

/// A bound function needs a raw error type: the derived signature is `Result<_, E>`, and there is
/// no `E` without one.
#[test]
fn a_bound_function_without_an_error_type_is_rejected() {
    let message = err(&format!(
        r#"{{{HEAD},"capabilities":["clock"],
            "provider_api":{{"functions":{{
              "Instant::now_ns":{{"capability":"clock","symbol":"stark_time_monotonic_now_ns"}}
            }}}}}}"#
    ));
    assert!(message.contains("errors"), "{message}");
}

/// Malformed shapes are rejected rather than ignored — a silently dropped binding would surface
/// much later as a missing item nobody could trace to a manifest.
#[test]
fn malformed_bindings_are_rejected() {
    for body in [
        // provider_api not an object
        format!(r#"{{{HEAD},"capabilities":["clock"],"provider_api":[]}}"#),
        // functions not an object
        format!(r#"{{{HEAD},"capabilities":["clock"],"provider_api":{{"functions":[]}}}}"#),
        // binding not an object
        format!(
            r#"{{{HEAD},"capabilities":["clock"],"provider_api":{{"functions":{{"a":"b"}}}}}}"#
        ),
        // missing symbol
        format!(
            r#"{{{HEAD},"capabilities":["clock"],
                "provider_api":{{"functions":{{"a":{{"capability":"clock"}}}}}}}}"#
        ),
        // empty symbol
        format!(
            r#"{{{HEAD},"capabilities":["clock"],
                "provider_api":{{"functions":{{"a":{{"capability":"clock","symbol":""}}}}}}}}"#
        ),
        // error type not a string
        format!(r#"{{{HEAD},"capabilities":["clock"],"provider_api":{{"errors":{{"clock":1}}}}}}"#),
    ] {
        assert!(parse(&body).is_err(), "must reject: {body}");
    }
}

/// The binding carries **no** signature information, and nothing in the manifest schema admits any.
/// This is CD-224's invariant asserted against the parser rather than trusted from the design.
#[test]
fn a_binding_carries_no_signature_information() {
    let p = ok(&format!(
        r#"{{{HEAD},"capabilities":["clock"],
            "provider_api":{{"errors":{{"clock":"E"}},"functions":{{
              "Instant::now_ns":{{"capability":"clock","symbol":"stark_time_monotonic_now_ns"}}
            }}}}}}"#
    ));
    let f = &p.provider_api.functions[0];

    // The struct has exactly three fields, all identity. If a signature field is ever added, this
    // stops compiling -- which is the intent.
    let starkc::package::ProviderFunctionBinding {
        item_path,
        capability,
        symbol,
    } = f;
    assert_eq!(item_path, "Instant::now_ns");
    assert_eq!(capability, "clock");
    assert_eq!(symbol, "stark_time_monotonic_now_ns");
}
