//! WP-C6.4 — the Tier-1 platform matrix's permanent test suite.
//!
//! `WP-C6-ENTRY.md` §§32–37 asks C6.4 to prove that the already-admitted native runtime behaves
//! the same on both Tier-1 targets, that target preflight rejects what it should before linking,
//! and that no hidden host assumption survives in the qualified path.
//!
//! # What this file can and cannot prove
//!
//! A test binary runs on ONE platform. Nothing here can establish cross-platform agreement by
//! itself — that comes from running this same suite on both Tier-1 runners at the same commit and
//! comparing the two evidence records (`scripts/run-c64-qualification.py`). What this file does is
//! make each platform's observations **exact and comparable**: byte-level stdout, trap category and
//! provenance, recorded target metadata, and a deterministic build key. §35's rule — "no real
//! platform run means no platform claim" — is why the assertions are written against exact bytes
//! rather than "contains" checks that two platforms could satisfy differently.
//!
//! Tests are named `target_preflight_*`, `portability_*`, `platform_*` and `determinism_*` so the
//! qualification harness can run one group (`cargo test --test c64_platform_matrix
//! target_preflight`) as §8.6 specifies.

use starkc::backend::generated_rust::{
    emit_native_debug_with_toolchain, NativeArtifact, NativeBuildOptions, NativeToolchainOptions,
};
use starkc::diag::Severity;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::target::{self, TargetError, Tier};
use starkc::typecheck;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

static NEXT: AtomicU64 = AtomicU64::new(0);

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A unique temporary root. `std::env::temp_dir` rather than `/tmp` (§9.8), and PID + counter
/// rather than PID alone so parallel tests in this binary cannot collide on a shared path.
fn temp_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "stark_c64_{tag}_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&root);
    root
}

/// Copy the runtime crate to `dest` — `Cargo.toml` and `src/*.rs`, nothing else — so a build
/// pointed at it cannot succeed because of anything else in the checkout.
fn install_runtime_into(dest: &Path) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stark-runtime");
    std::fs::create_dir_all(dest.join("src")).expect("create install dir");
    std::fs::copy(source.join("Cargo.toml"), dest.join("Cargo.toml")).expect("copy Cargo.toml");
    for entry in std::fs::read_dir(source.join("src")).expect("read runtime src") {
        let entry = entry.expect("runtime src entry");
        if entry.path().extension().is_some_and(|e| e == "rs") {
            std::fs::copy(entry.path(), dest.join("src").join(entry.file_name()))
                .expect("copy runtime module");
        }
    }
}

fn build_native(source: &str, name: &str, target_dir: &Path, runtime: &Path) -> NativeArtifact {
    let file = Arc::new(SourceFile::new(name.to_string(), source.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    assert!(errs.is_empty(), "typecheck: {errs:?}");
    let program =
        lower_program(&hir, &checked.tables, file).unwrap_or_else(|e| panic!("lower: {}", e.what));
    let verified = verify_program(&program).unwrap_or_else(|e| panic!("verify: {e:?}"));
    emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: target_dir.to_path_buf(),
            target_contract: "stark-64-v1".to_string(),
        },
        &NativeToolchainOptions {
            rustc: PathBuf::from("rustc"),
            cargo: PathBuf::from("cargo"),
            runtime_crate: runtime.to_path_buf(),
        },
    )
    .unwrap_or_else(|e| panic!("native build failed: {e:?}"))
}

/// The exact JSON string value of `field` in a generated `build.json`. A narrow scan, deliberately:
/// the manifest is written by this repository in a fixed shape, and adding a JSON dependency to
/// read six fields from a file we also wrote is not a trade worth making.
fn manifest_field(manifest: &str, field: &str) -> String {
    let needle = format!("\"{field}\":");
    let rest = manifest
        .split(&needle)
        .nth(1)
        .unwrap_or_else(|| panic!("no `{field}` in build manifest:\n{manifest}"))
        .trim_start();
    if let Some(quoted) = rest.strip_prefix('"') {
        quoted[..quoted.find('"').expect("closing quote")].to_string()
    } else {
        rest.split([',', '\n', '}'])
            .next()
            .unwrap()
            .trim()
            .to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// Target preflight (§33, §8.5's inventory)
// ---------------------------------------------------------------------------------------------

/// A probe that reports every target unavailable, so §8.5(7)'s missing-toolchain branch is
/// reachable without uninstalling anything on the machine running the test.
struct NoToolchain;
impl target::TargetAvailability for NoToolchain {
    fn is_available(&self, _host: &str, _target: &str) -> Result<(), String> {
        Err("rust-std for this target is not installed".to_string())
    }
}

struct AnyToolchain;
impl target::TargetAvailability for AnyToolchain {
    fn is_available(&self, _host: &str, _target: &str) -> Result<(), String> {
        Ok(())
    }
}

#[test]
fn target_preflight_accepts_both_tier1_targets_with_the_declared_contract_and_suffix() {
    for triple in target::tier1_triples() {
        let spec = target::classify(triple).expect("tier-1 target is named");
        assert_eq!(spec.tier, Tier::One, "{triple}");
        assert_eq!(spec.layout_contract, "stark-64-v1", "{triple}");
        assert_eq!(spec.executable_suffix, "", "{triple}");
        assert_eq!(spec.pointer_width, 64, "{triple}");
    }
}

#[test]
fn target_preflight_classifies_windows_tier2_and_intel_mac_tier3() {
    let windows = target::classify("x86_64-pc-windows-msvc").expect("named");
    assert_eq!(windows.tier, Tier::Two);
    assert_eq!(windows.executable_suffix, ".exe");
    assert_eq!(
        target::classify("x86_64-apple-darwin").unwrap().tier,
        Tier::Three
    );
}

#[test]
fn target_preflight_rejects_unknown_targets_of_either_width() {
    for unknown in [
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
        "i686-unknown-linux-gnu",
        "wasm32-unknown-unknown",
    ] {
        assert!(target::classify(unknown).is_none(), "{unknown}");
        assert!(
            matches!(
                target::select(unknown, None, &AnyToolchain),
                Err(TargetError::UnsupportedByStark { .. })
            ),
            "{unknown} must be rejected, not classified"
        );
    }
}

/// §8.1: the diagnostic has to name the targets a C6 claim depends on, or a user has no way to
/// know what to build for.
#[test]
fn target_preflight_diagnostic_names_the_supported_tier1_targets() {
    let message = target::select("mips64-unknown-linux-gnuabi64", None, &AnyToolchain)
        .unwrap_err()
        .to_string();
    for triple in target::tier1_triples() {
        assert!(
            message.contains(triple),
            "diagnostic must name {triple}: {message}"
        );
    }
}

/// §8.5(14). The two failures need different remedies — retarget versus install — so they must be
/// different classes, not two spellings of one.
#[test]
fn target_preflight_separates_an_unsupported_target_from_a_missing_toolchain() {
    let missing = target::select("aarch64-apple-darwin", None, &NoToolchain).unwrap_err();
    let unsupported =
        target::select("armv7-unknown-linux-gnueabihf", None, &AnyToolchain).unwrap_err();
    assert!(matches!(
        missing,
        TargetError::SupportedButToolchainMissing { .. }
    ));
    assert!(matches!(
        unsupported,
        TargetError::UnsupportedByStark { .. }
    ));
    assert_ne!(missing.to_string(), unsupported.to_string());
}

/// The precondition for every other row in this file: qualification evidence gathered on a host
/// STARK does not name would describe a target no claim covers.
#[test]
fn target_preflight_this_host_is_a_named_target() {
    let host = host_triple().expect("rustc reports a host triple");
    let spec = target::classify(&host)
        .unwrap_or_else(|| panic!("this host `{host}` is not a target STARK names"));
    eprintln!(
        "host {host} is {} (contract {})",
        spec.tier, spec.layout_contract
    );
}

fn host_triple() -> Option<String> {
    let out = std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(str::to_string)
}

// ---------------------------------------------------------------------------------------------
// Portability (§34)
// ---------------------------------------------------------------------------------------------

/// §9.7: a workspace path with SPACES and a runtime install prefix with SPACES, together, through
/// a real build. This is the end-to-end check on manifest escaping (F6) — the runtime path is
/// interpolated into the generated `Cargo.toml`, so a quoting defect fails here as a Cargo parse
/// error rather than in a unit test's expectations.
#[test]
fn portability_builds_and_runs_under_paths_containing_spaces() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = temp_root("spaces");
    let runtime = root.join("install prefix/lib/stark/stark-runtime");
    install_runtime_into(&runtime);
    let artifact = build_native(
        "fn main() { println(42); }\n",
        "spaced path.stark",
        &root.join("build dir/target"),
        &runtime,
    );
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run binary built under a spaced path");
    assert!(
        run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.stdout, b"42\n");
    let _ = std::fs::remove_dir_all(&root);
}

/// §9.7: the same, with non-ASCII path components. `Debug`-based quoting survived this one (Rust
/// keeps printable Unicode literal), which is exactly why it needed a test rather than an
/// assumption.
#[test]
fn portability_builds_and_runs_under_paths_containing_unicode() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = temp_root("unicode");
    let runtime = root.join("préfixe/lib/stark/stark-runtime");
    install_runtime_into(&runtime);
    let artifact = build_native(
        "fn main() { println(7); }\n",
        "naïve.stark",
        &root.join("répertoire/target"),
        &runtime,
    );
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run binary built under a Unicode path");
    assert!(
        run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.stdout, b"7\n");
    let _ = std::fs::remove_dir_all(&root);
}

/// §8.5(11): the manifest records host and selected target as separate fields, plus the tier and
/// pointer width the evidence record needs. Equal today — the assertion is that both are *present
/// and named*, because a record that cannot distinguish them will silently report the host as the
/// target the day they differ.
#[test]
fn portability_build_manifest_records_host_and_selected_target_separately() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = temp_root("manifest");
    let runtime = root.join("runtime");
    install_runtime_into(&runtime);
    let artifact = build_native(
        "fn main() { println(1); }\n",
        "manifest.stark",
        &root.join("target"),
        &runtime,
    );
    let manifest = std::fs::read_to_string(artifact.build_dir.join("build.json"))
        .expect("generated build.json");

    let host = manifest_field(&manifest, "host_triple");
    let selected = manifest_field(&manifest, "target_triple");
    assert!(!host.is_empty() && !selected.is_empty(), "{manifest}");
    assert_eq!(host, selected, "only host builds are admitted today");
    assert_eq!(host, host_triple().expect("rustc host"));

    let spec = target::classify(&selected).expect("selected target is named");
    assert_eq!(
        manifest_field(&manifest, "target_tier"),
        spec.tier.to_string()
    );
    assert_eq!(
        manifest_field(&manifest, "target_pointer_width"),
        spec.pointer_width.to_string()
    );
    assert_eq!(
        manifest_field(&manifest, "target_contract"),
        spec.layout_contract,
        "the recorded layout contract must be the one the target declares"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// §10.7: the generated crate carries a lock, and it is a path-only graph — no registry source and
/// no checksum, which is what makes `--locked --offline` provable rather than cache-dependent.
#[test]
fn portability_generated_crate_is_locked_and_network_free() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = temp_root("locked");
    let runtime = root.join("runtime");
    install_runtime_into(&runtime);
    let artifact = build_native(
        "fn main() { println(1); }\n",
        "locked.stark",
        &root.join("target"),
        &runtime,
    );
    let lock = std::fs::read_to_string(artifact.build_dir.join("Cargo.lock"))
        .expect("generated crate carries a Cargo.lock");
    assert!(lock.contains("name = \"stark-runtime\""), "{lock}");
    assert!(
        !lock.contains("source = "),
        "a path-only graph has no registry source:\n{lock}"
    );
    assert!(!lock.contains("checksum = "), "{lock}");
    let _ = std::fs::remove_dir_all(&root);
}

/// §10.6(5). The source-checkout fallback is compiled into the binary, so an installed-runtime
/// test can pass for the wrong reason unless the fallback can be switched off. This asserts the
/// switch works: with it set and no installed layout present, discovery FAILS rather than
/// silently reaching the checkout.
#[test]
fn portability_installed_runtime_requirement_refuses_the_checkout_fallback() {
    let empty = temp_root("nofallback");
    std::fs::create_dir_all(empty.join("bin")).expect("create fake install prefix");
    let fake_exe = empty.join("bin").join("stark");

    // Without the switch, discovery falls back to the checkout and succeeds.
    let permissive = starkc::native_toolchain::discover_runtime(Some(&fake_exe));
    assert!(
        permissive.is_ok(),
        "the checkout fallback is the default behaviour"
    );

    // Set for this process only; no other test in this binary reads it.
    std::env::set_var(starkc::native_toolchain::REQUIRE_INSTALLED_RUNTIME_VAR, "1");
    let strict = starkc::native_toolchain::discover_runtime(Some(&fake_exe));
    std::env::remove_var(starkc::native_toolchain::REQUIRE_INSTALLED_RUNTIME_VAR);

    match strict {
        Err(starkc::native_toolchain::ToolchainError::RuntimeMissing { attempted }) => {
            assert!(
                attempted
                    .iter()
                    .all(|p| !p.ends_with("starkc/stark-runtime")),
                "the checkout must not even be attempted under the switch: {attempted:?}"
            );
        }
        other => panic!("expected RuntimeMissing under the switch, got {other:?}"),
    }
    let _ = std::fs::remove_dir_all(&empty);
}

// ---------------------------------------------------------------------------------------------
// Platform-neutral execution (§16)
// ---------------------------------------------------------------------------------------------

/// §9.5: STARK's output contract is bytes. `\n` is `\n` on every platform — no host line-ending
/// convention, no text-mode translation — and Unicode reaches stdout as UTF-8. Asserted as exact
/// bytes so a platform that translated line endings would fail rather than be normalised away.
#[test]
fn platform_stdout_is_exact_bytes_including_unicode_and_line_termination() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = temp_root("stdout");
    let runtime = root.join("runtime");
    install_runtime_into(&runtime);
    let artifact = build_native(
        "fn main() {\n  println(\"héllo wörld\");\n  print(\"no newline\");\n}\n",
        "bytes.stark",
        &root.join("target"),
        &runtime,
    );
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run the output binary");
    assert!(
        run.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        run.stdout,
        "héllo wörld\nno newline".as_bytes(),
        "stdout must be these exact bytes on every platform"
    );
    assert!(
        !run.stdout.contains(&b'\r'),
        "no platform may translate STARK's newline into CRLF"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// §16: trap category, source provenance, and exit status, from a real native binary. These are
/// the observations a Tier-1 comparison compares, so they are asserted exactly: category text,
/// `file:line:column`, exit 101, and the pre-trap stdout prefix (CD-120 Contract B).
#[test]
fn platform_trap_reports_category_provenance_and_exit_status() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let root = temp_root("trap");
    let runtime = root.join("runtime");
    install_runtime_into(&runtime);
    let artifact = build_native(
        "fn main() {\n  print(\"before\");\n  let v: Vec<Int32> = Vec::new();\n  println(v[3]);\n}\n",
        "trapsite.stark",
        &root.join("target"),
        &runtime,
    );
    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run the trapping binary");
    assert_eq!(
        run.status.code(),
        Some(101),
        "traps exit 101 on every platform"
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("error: runtime trap: index out of bounds"),
        "trap category: {stderr}"
    );
    assert!(
        stderr.contains("--> trapsite.stark:4:11"),
        "trap provenance must be the USER's `v[3]`, not a runtime location: {stderr}"
    );
    assert_eq!(
        run.stdout, b"before",
        "output written before the trap must be flushed (CD-120 Contract B)"
    );
    let _ = std::fs::remove_dir_all(&root);
}

// ---------------------------------------------------------------------------------------------
// Determinism (§10.8)
// ---------------------------------------------------------------------------------------------

/// §10.8: the same input built twice into two clean output directories produces the same build
/// key, the same generated source, and the same recorded metadata. Not a reproducible-binary
/// claim — that is C7's (§10.8's closing line) — the subject here is the compiler's own output.
#[test]
fn determinism_two_clean_builds_agree_on_key_source_and_metadata() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source =
        "fn main() {\n  let mut v: Vec<Int32> = Vec::new();\n  v.push(5);\n  println(v);\n}\n";
    let mut observations = Vec::new();
    for run in 0..2 {
        let root = temp_root(&format!("determinism{run}"));
        let runtime = root.join("runtime");
        install_runtime_into(&runtime);
        let artifact = build_native(source, "det.stark", &root.join("target"), &runtime);
        let manifest = std::fs::read_to_string(artifact.build_dir.join("build.json")).unwrap();
        let generated = std::fs::read_to_string(artifact.build_dir.join("src/main.rs")).unwrap();
        let output = std::process::Command::new(&artifact.binary_path)
            .output()
            .expect("run");
        observations.push((
            manifest_field(&manifest, "build_key"),
            generated,
            manifest_field(&manifest, "target_contract"),
            output.stdout,
        ));
        let _ = std::fs::remove_dir_all(&root);
    }
    assert_eq!(observations[0].0, observations[1].0, "build key");
    assert_eq!(observations[0].1, observations[1].1, "generated source");
    assert_eq!(observations[0].2, observations[1].2, "layout contract");
    assert_eq!(observations[0].3, observations[1].3, "program output");
    assert_eq!(observations[0].3, b"[5]\n");

    // The qualification harness runs this test twice, in two separate processes, and compares
    // these two lines. That is the real §10.8 rerun: the loop above proves the compiler is
    // deterministic within one process, and only a second invocation can show it stayed
    // deterministic across them.
    let mut source_hash = DefaultHasher::new();
    observations[0].1.hash(&mut source_hash);
    println!(
        "C64-DETERMINISM key={} source={:016x}",
        observations[0].0,
        source_hash.finish()
    );
}
