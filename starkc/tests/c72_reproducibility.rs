//! WP-C7.2 — what reproduces, per artefact and per profile, measured rather than asserted.
//!
//! §4.1 forbids one global "reproducible" label, and the measurements are why: the same build
//! produces artefacts in three different classes at once, and the linked executable's class depends
//! on the PROFILE.
//!
//! | artefact | class |
//! | --- | --- |
//! | generated Rust | `BYTE-REPRODUCIBLE` |
//! | generated `Cargo.toml` | `SEMANTICALLY-REPRODUCIBLE` — byte-identical per machine, embeds the compiler's own runtime path |
//! | `stark.lock` | `BYTE-REPRODUCIBLE` |
//! | executable, release | `BYTE-REPRODUCIBLE` |
//! | executable, debug | `NOT-YET-REPRODUCIBLE` |
//!
//! **Why debug is not, and why remapping did not fix it.** A debug binary embeds paths from two
//! separate mechanisms. `--remap-path-prefix` covers the ones rustc records from source spans, and
//! WP-C7.2 added that — it removes 31 strings including every reference to the compiler's own
//! installation directory. The rest are recorded by the LINKER: macOS writes object-file paths into
//! the debug map so `dsymutil` can find them later, and no rustc flag reaches those. Release does
//! not carry them because it does not carry that debug information.
//!
//! **The remapping is therefore not what makes release reproducible.** Measured directly by
//! reverting it: release was already byte-identical across paths, and embeds zero remapped markers.
//! It is kept because a debug binary should not name the machine that built it, not because it
//! delivers reproducibility. Saying otherwise would credit a fix for an outcome it did not cause.

use std::path::{Path, PathBuf};
use std::process::Command;

fn stark_binary() -> PathBuf {
    let mut path = std::env::current_exe().expect("test binary");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.join(if cfg!(windows) { "stark.exe" } else { "stark" })
}

fn workload(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/c7-workloads")
        .join(name)
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("mkdir");
    for entry in std::fs::read_dir(from).expect("read") {
        let entry = entry.expect("entry");
        if entry.file_name() == "target" {
            continue;
        }
        let dest = to.join(entry.file_name());
        if entry.file_type().expect("kind").is_dir() {
            copy_tree(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), &dest).expect("copy");
        }
    }
}

/// Build one workload twice, from roots whose absolute paths differ in BOTH name and length —
/// length matters because a path difference that changes a binary's size is the easiest kind to
/// detect and the easiest to accidentally avoid by using equal-length names.
fn build_twice(workload_name: &str, args: &[&str]) -> Option<(PathBuf, Vec<u8>, PathBuf, Vec<u8>)> {
    if !stark_binary().is_file() {
        eprintln!("SKIP: `stark` is not built in this target directory.");
        return None;
    }
    let profile = if args.contains(&"--release") {
        "release"
    } else {
        "debug"
    };
    let mut built = Vec::new();
    for prefix in ["c72_a_", "c72_bbbbbbbbbbbb_much_longer_"] {
        let root = std::env::temp_dir().join(format!(
            "{prefix}{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        copy_tree(&workload(workload_name), &root);
        // A multi-package workload keeps its root package in `app/`; `stark build` must run where
        // the manifest is.
        let package_dir = if root.join("app/starkpkg.json").is_file() {
            root.join("app")
        } else {
            root.clone()
        };
        let out = Command::new(stark_binary())
            .args(args)
            .current_dir(&package_dir)
            .output()
            .expect("stark build");
        assert!(
            out.status.success(),
            "{workload_name} [{profile}] build failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let dir = package_dir.join("target/stark").join(profile);
        let binary = std::fs::read_dir(&dir)
            .expect("output dir")
            .flatten()
            .map(|e| e.path())
            // On Windows the executable is `<name>.exe`, so "has no extension" would find
            // nothing. Matching the platform's suffix keeps one predicate honest on all three.
            .find(|p| {
                p.is_file()
                    && if cfg!(windows) {
                        p.extension().is_some_and(|e| e == "exe")
                    } else {
                        p.extension().is_none()
                    }
            })
            .expect("an executable");
        built.push((root, std::fs::read(&binary).expect("read binary")));
    }
    let (rb, b) = built.pop().unwrap();
    let (ra, a) = built.pop().unwrap();
    Some((ra, a, rb, b))
}

/// **The C7.2 headline.** A release executable is byte-identical across two different absolute
/// build paths.
#[test]
fn release_executables_are_byte_reproducible_across_build_paths() {
    for name in ["w01_minimal", "w02_arith_control", "w06_multi_package"] {
        let Some((ra, a, rb, b)) = build_twice(name, &["build", "--release"]) else {
            return;
        };
        assert_eq!(
            a.len(),
            b.len(),
            "{name}: release binaries differ in SIZE across build paths, which usually means a \
             path length leaked into the artefact"
        );
        assert!(
            a == b,
            "{name}: release binaries differ across build paths despite equal size"
        );
        let _ = std::fs::remove_dir_all(&ra);
        let _ = std::fs::remove_dir_all(&rb);
    }
}

/// A release executable must not name where it was built, nor where the compiler lives.
#[test]
fn release_executables_embed_no_build_or_runtime_path() {
    let Some((ra, a, rb, _b)) = build_twice("w02_arith_control", &["build", "--release"]) else {
        return;
    };
    // The runtime needle is a PATH fragment, not the bare crate name: `stark-runtime` also appears
    // as a crate IDENTIFIER in symbol names, which is legitimate and is not a path leak.
    for needle in [
        ra.to_string_lossy().to_string(),
        "/stark-runtime/src".to_string(),
    ] {
        let bytes = needle.as_bytes();
        assert!(
            !a.windows(bytes.len()).any(|w| w == bytes),
            "a release binary must not embed `{needle}`"
        );
    }
    let _ = std::fs::remove_dir_all(&ra);
    let _ = std::fs::remove_dir_all(&rb);
}

/// The generated SOURCE reproduces in both profiles — this is the artefact STARK fully controls,
/// and the one a backend nondeterminism (symbol numbering, map iteration, instantiation order)
/// would break first.
#[test]
fn generated_rust_is_byte_reproducible_in_both_profiles() {
    if !stark_binary().is_file() {
        eprintln!("SKIP: `stark` is not built in this target directory.");
        return;
    }
    for profile_args in [
        vec!["build", "--emit-rust"],
        vec!["build", "--release", "--emit-rust"],
    ] {
        let mut sources = Vec::new();
        for prefix in ["c72src_a_", "c72src_bbbbbbbbb_longer_"] {
            let root = std::env::temp_dir().join(format!(
                "{prefix}{}_{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&root);
            copy_tree(&workload("w03_generic_trait"), &root);
            let out = Command::new(stark_binary())
                .args(&profile_args)
                .current_dir(&root)
                .output()
                .expect("stark build");
            assert!(
                out.status.success(),
                "build failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let generated =
                find_file(&root.join("target/stark"), "main.rs").expect("generated main.rs");
            sources.push(std::fs::read(generated).expect("read generated"));
            let _ = std::fs::remove_dir_all(&root);
        }
        assert_eq!(
            sources[0], sources[1],
            "generated Rust must be byte-identical across build paths for {profile_args:?}"
        );
    }
}

fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    None
}

/// Debug is recorded as NOT-YET-REPRODUCIBLE, and that is pinned rather than left as prose: if a
/// future toolchain or flag makes debug reproduce, this fails and the classification gets updated
/// deliberately instead of the document quietly going stale.
#[test]
fn debug_executables_are_not_yet_reproducible_and_that_is_recorded() {
    let Some((ra, a, rb, b)) = build_twice("w01_minimal", &["build"]) else {
        return;
    };
    assert_ne!(
        a, b,
        "debug executables now reproduce across build paths — that is an IMPROVEMENT, so update \
         the WP-C7.2 classification from NOT-YET-REPRODUCIBLE and change this test to require it"
    );
    let _ = std::fs::remove_dir_all(&ra);
    let _ = std::fs::remove_dir_all(&rb);
}

/// **Regression, WP-C7.4 (CD-190).** Remapping must survive a build directory containing a SPACE.
///
/// The first implementation passed the flags through `RUSTFLAGS`, which Cargo splits on spaces. A
/// path like `/tmp/build dir/...` therefore tore one `--remap-path-prefix=FROM=TO` into two
/// arguments and rustc rejected the fragment — so every build under such a path failed outright,
/// not merely un-remapped. `CARGO_ENCODED_RUSTFLAGS` separates on `\x1f` instead.
///
/// This asserts both halves, because either alone would let the defect back in: the build must
/// SUCCEED (the flags reached rustc intact) and the release binary must NOT embed the path (the
/// flags were actually applied, rather than dropped in a way that happens to build fine).
#[test]
fn remapping_survives_a_build_path_containing_spaces() {
    if !stark_binary().is_file() {
        eprintln!("SKIP: `stark` is not built in this target directory.");
        return;
    }
    let root = std::env::temp_dir().join(format!(
        "c72 spaced root {}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    copy_tree(&workload("w01_minimal"), &root);

    let out = Command::new(stark_binary())
        .args(["build", "--release"])
        .current_dir(&root)
        .output()
        .expect("stark build");
    assert!(
        out.status.success(),
        "a build under a path containing spaces failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let binary = std::fs::read_dir(root.join("target/stark/release"))
        .expect("output dir")
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.is_file()
                && if cfg!(windows) {
                    p.extension().is_some_and(|e| e == "exe")
                } else {
                    p.extension().is_none()
                }
        })
        .expect("an executable");
    let bytes = std::fs::read(&binary).expect("read binary");
    let needle_owned = root.to_string_lossy().to_string();
    let needle = needle_owned.as_bytes();
    assert!(
        !bytes.windows(needle.len()).any(|w| w == needle),
        "the release binary embeds its spaced build path, so remapping did not apply"
    );
    let _ = std::fs::remove_dir_all(&root);
}
