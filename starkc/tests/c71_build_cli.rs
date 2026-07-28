//! WP-C7.1 §3.5 — the `stark build` CLI surface: profiles, targets, and their refusals.
//!
//! These exercise the ARGUMENT layer and the paths it produces. Semantic equivalence between the
//! profiles is `c71_profile_agreement.rs`; this is the part that decides whether a build happens at
//! all, and where its output lands.
//!
//! §3.3 requires an unknown target to be rejected "before expensive compilation" and an unsupported
//! target to be distinguishable from a missing local toolchain. Both are asserted here by their
//! diagnostics, because the difference matters to a user: one means "STARK will never build this",
//! the other means "install something".

use std::path::{Path, PathBuf};
use std::process::Command;

fn stark_binary() -> PathBuf {
    // The test binary lives in `target/debug/deps/`; `stark` is two levels up.
    let mut path = std::env::current_exe().expect("test binary path");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join(if cfg!(windows) { "stark.exe" } else { "stark" })
}

fn workload() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/c7-workloads/w01_minimal")
}

/// A scratch copy, so a test never mutates the frozen workload or races another test's `target/`.
fn scratch(tag: &str) -> PathBuf {
    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("scratch dir");
        for entry in std::fs::read_dir(from).expect("read") {
            let entry = entry.expect("entry");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("kind").is_dir() {
                if entry.file_name() == "target" {
                    continue;
                }
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("copy");
            }
        }
    }
    let dir = std::env::temp_dir().join(format!(
        "stark_c71_{tag}_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(&workload(), &dir);
    dir
}

fn run(dir: &Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(stark_binary())
        .args(args)
        .current_dir(dir)
        .output()
        .expect("running stark");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn skip_without_binary() -> bool {
    if stark_binary().is_file() {
        return false;
    }
    eprintln!("SKIP: `stark` is not built in this target directory.");
    true
}

#[test]
fn debug_is_the_default_and_lands_under_debug() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("default");
    let (ok, stdout, stderr) = run(&dir, &["build"]);
    assert!(ok, "default build failed: {stderr}");
    assert!(
        stdout.contains("[debug]"),
        "expected a debug label: {stdout}"
    );
    assert!(
        dir.join("target/stark/debug").is_dir(),
        "no debug output dir"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The two profiles must not collide — §3.4's whole purpose. Both artefacts exist afterwards.
#[test]
fn release_and_debug_coexist_in_separate_paths() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("both");
    assert!(run(&dir, &["build"]).0, "debug build failed");
    let (ok, stdout, stderr) = run(&dir, &["build", "--release"]);
    assert!(ok, "release build failed: {stderr}");
    assert!(
        stdout.contains("[release]"),
        "a release build must not report itself as debug: {stdout}"
    );
    let debug = dir.join("target/stark/debug/w01_minimal");
    let release = dir.join("target/stark/release/w01_minimal");
    assert!(debug.is_file(), "debug artefact missing");
    assert!(release.is_file(), "release artefact missing");
    assert_ne!(debug, release, "the two profiles must not share a path");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The generated manifest must OVERRIDE Cargo's release defaults, not inherit them. `panic` is the
/// setting that would break DROP-ABORT-001 if it were left as Cargo's `"unwind"`.
#[test]
fn the_generated_release_profile_overrides_cargo_defaults() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("manifest");
    let (ok, _, stderr) = run(&dir, &["build", "--release", "--keep-generated"]);
    assert!(ok, "release build failed: {stderr}");
    let manifest = walk_for(&dir.join("target/stark/release"), "Cargo.toml")
        .expect("generated Cargo.toml was kept");
    let text = std::fs::read_to_string(manifest).expect("read manifest");
    let release = text
        .split("[profile.release]")
        .nth(1)
        .expect("the generated manifest must declare [profile.release]");
    for setting in [
        "panic = \"abort\"",
        "opt-level = 3",
        "overflow-checks = true",
        "debug-assertions = false",
    ] {
        assert!(
            release.contains(setting),
            "[profile.release] must state `{setting}` explicitly rather than inherit it:\n{release}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn walk_for(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = walk_for(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    None
}

/// An unknown triple is refused with the names STARK does know — and refused CHEAPLY, before the
/// package graph is loaded or anything is compiled.
#[test]
fn an_unknown_target_is_rejected_with_the_supported_list() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("badtarget");
    let (ok, _, stderr) = run(&dir, &["build", "--target", "sparc-unknown-none"]);
    assert!(!ok, "an unknown target must fail the build");
    assert!(
        stderr.contains("unsupported target `sparc-unknown-none`"),
        "the diagnostic must name the rejected target: {stderr}"
    );
    assert!(
        stderr.contains("aarch64-apple-darwin") && stderr.contains("tier-1"),
        "the diagnostic must list the targets STARK does name, with tiers: {stderr}"
    );
    assert!(
        !dir.join("target/stark").exists(),
        "nothing may be compiled before the target is validated"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// §3.3: "do not silently compile for the host when a target was requested". A supported but
/// non-host target is refused with its OWN reason, distinct from an unknown triple.
#[test]
fn a_supported_but_non_host_target_is_refused_not_silently_retargeted() {
    if skip_without_binary() {
        return;
    }
    let host_is_mac = cfg!(target_os = "macos");
    let other = if host_is_mac {
        "x86_64-unknown-linux-gnu"
    } else {
        "aarch64-apple-darwin"
    };
    let dir = scratch("crosstarget");
    let (ok, stdout, stderr) = run(&dir, &["build", "--target", other]);
    assert!(!ok, "a cross-target build must not silently succeed");
    assert!(
        stderr.contains(other),
        "the refusal must name the requested target: {stderr}"
    );
    assert!(
        !stdout.contains("Built"),
        "a cross-target request must never produce a HOST binary: {stdout}"
    );
    // Distinct from the unknown-target message: this one says the target IS supported.
    assert!(
        stderr.contains("supported target"),
        "a supported-but-unavailable target must be distinguishable from an unknown one: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn target_without_a_value_is_a_usage_error() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("notriple");
    let (ok, _, stderr) = run(&dir, &["build", "--target"]);
    assert!(!ok, "`--target` with no triple must fail");
    assert!(
        stderr.contains("requires a target triple"),
        "the diagnostic must say what is missing: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--target=<triple>` is accepted as one argument, and refused for the same reason as the spaced
/// form — so the two spellings cannot diverge.
#[test]
fn the_equals_form_of_target_is_accepted() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("equals");
    let (ok, _, stderr) = run(&dir, &["build", "--target=sparc-unknown-none"]);
    assert!(!ok);
    assert!(
        stderr.contains("unsupported target `sparc-unknown-none`"),
        "`--target=` must behave as `--target `: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Rebuilding after a source change must produce the changed program, in release as in debug.
#[test]
fn a_release_build_after_a_source_change_reflects_it() {
    if skip_without_binary() {
        return;
    }
    let dir = scratch("rebuild");
    assert!(run(&dir, &["build", "--release"]).0, "first build failed");
    let main = dir.join("src/main.stark");
    std::fs::write(&main, "fn main() {\n    print(\"changed\");\n}\n").expect("rewrite");
    assert!(run(&dir, &["build", "--release"]).0, "rebuild failed");
    let out = Command::new(dir.join("target/stark/release/w01_minimal"))
        .output()
        .expect("run rebuilt binary");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "changed",
        "the rebuilt release binary must reflect the edited source"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
