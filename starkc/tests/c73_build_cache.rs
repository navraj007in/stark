//! WP-C7.3 — the bounded build cache.
//!
//! **The claim this suite defends, stated exactly:** the implementation reuses complete
//! content-addressed generated crates and Cargo artefacts. It is a bounded build cache, **not
//! fine-grained incremental compilation**. Nothing here understands functions, packages or
//! interfaces; a one-character edit changes the key and produces a cold build. These tests are
//! written to hold that line — several assert the *absence* of incrementality, so a future change
//! that quietly added interface-level reuse would have to update them deliberately.
//!
//! The correctness property that matters most is at the bottom: cached, uncached and
//! clean-build outputs must be identical. A cache that changed a program's behaviour would be worse
//! than no cache at any speed.

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

fn skip() -> bool {
    if stark_binary().is_file() {
        return false;
    }
    eprintln!("SKIP: `stark` is not built in this target directory.");
    true
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

struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "stark_c73_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        copy_tree(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/c7-workloads/w01_minimal"),
            &root,
        );
        Sandbox { root }
    }

    fn build(&self, args: &[&str]) -> String {
        let out = Command::new(stark_binary())
            .arg("build")
            .args(args)
            .current_dir(&self.root)
            .output()
            .expect("stark build");
        assert!(
            out.status.success(),
            "build {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    fn write_source(&self, text: &str) {
        std::fs::write(self.root.join("src/main.stark"), text).expect("write source");
    }

    fn entries(&self, profile: &str) -> Vec<PathBuf> {
        let dir = self.root.join("target/stark").join(profile);
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        found.sort();
        found
    }

    fn run_binary(&self, profile: &str) -> String {
        let dir = self.root.join("target/stark").join(profile);
        let binary = std::fs::read_dir(&dir)
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
        let out = Command::new(binary).output().expect("run");
        String::from_utf8_lossy(&out.stdout).into_owned()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A rebuild with no source change reuses the entry rather than creating a second one.
#[test]
fn a_no_change_rebuild_reuses_the_same_entry() {
    if skip() {
        return;
    }
    let s = Sandbox::new("nochange");
    s.build(&[]);
    let first = s.entries("debug");
    assert_eq!(first.len(), 1, "one entry after the first build");
    s.build(&[]);
    assert_eq!(
        s.entries("debug"),
        first,
        "a no-change rebuild must reuse the entry, not add one"
    );
}

/// An edit changes the key. This asserts the ABSENCE of incrementality on purpose: a bounded cache
/// keys on whole content, so any edit is a miss.
#[test]
fn a_source_edit_produces_a_new_entry() {
    if skip() {
        return;
    }
    let s = Sandbox::new("edit");
    s.build(&[]);
    let before = s.entries("debug");
    s.write_source("fn main() {\n    print(\"changed\");\n}\n");
    s.build(&[]);
    let after = s.entries("debug");
    assert_eq!(after.len(), before.len() + 1, "an edit must add an entry");
    assert_eq!(s.run_binary("debug"), "changed");
}

/// Returning to an earlier source version HITS — the defining property of a content-addressed
/// cache, and what distinguishes it from "keep the last build".
#[test]
fn returning_to_an_earlier_source_version_reuses_its_entry() {
    if skip() {
        return;
    }
    let s = Sandbox::new("revisit");
    let original = std::fs::read_to_string(s.root.join("src/main.stark")).expect("read");
    s.build(&[]);
    let first = s.entries("debug");
    for n in 1..=3 {
        s.write_source(&format!("fn main() {{\n    print(\"v{n}\");\n}}\n"));
        s.build(&[]);
    }
    assert_eq!(s.entries("debug").len(), 4, "four distinct versions built");
    s.write_source(&original);
    s.build(&[]);
    assert_eq!(
        s.entries("debug").len(),
        4,
        "returning to the first version must REUSE its entry, not create a fifth"
    );
    assert!(s.entries("debug").contains(&first[0]));
}

/// Debug and release entries live under different roots and cannot be confused for one another.
#[test]
fn profiles_do_not_share_cache_entries() {
    if skip() {
        return;
    }
    let s = Sandbox::new("profiles");
    s.build(&[]);
    s.build(&["--release"]);
    assert_eq!(s.entries("debug").len(), 1);
    assert_eq!(s.entries("release").len(), 1);
    assert_ne!(
        s.entries("debug")[0].file_name(),
        s.entries("release")[0].file_name(),
        "the two profiles must not produce the same build key"
    );
}

/// `--no-build-cache` leaves nothing behind — the qualification path.
#[test]
fn no_build_cache_leaves_no_entry() {
    if skip() {
        return;
    }
    let s = Sandbox::new("nocache");
    s.build(&["--no-build-cache"]);
    assert!(
        s.entries("debug").is_empty(),
        "a cache-disabled build must leave no entry"
    );
    assert_eq!(
        s.run_binary("debug"),
        "ok",
        "the artefact is still produced"
    );
}

/// A corrupt metadata file must make the entry evictable, never break the build. This is the
/// failure §5.5 cares about: a bad cache entry must degrade, not poison.
#[test]
fn a_corrupt_metadata_file_does_not_break_the_build() {
    if skip() {
        return;
    }
    let s = Sandbox::new("corrupt");
    s.build(&[]);
    let entry = s.entries("debug").remove(0);
    std::fs::write(entry.join(".stark-cache-entry"), "\0\0not metadata\0\0").expect("corrupt it");
    s.build(&[]);
    assert_eq!(s.run_binary("debug"), "ok");
}

/// An interrupted metadata write leaves a stray temporary. It must be ignored, not read as an
/// entry — writes are atomic via rename precisely so a partial file is never observable as one.
#[test]
fn an_interrupted_metadata_write_is_ignored() {
    if skip() {
        return;
    }
    let s = Sandbox::new("interrupted");
    s.build(&[]);
    let entry = s.entries("debug").remove(0);
    std::fs::write(entry.join(".stark-cache-entry.tmp99999"), "last_used=1\n").expect("stray");
    s.build(&[]);
    assert_eq!(s.run_binary("debug"), "ok");
    assert_eq!(
        s.entries("debug").len(),
        1,
        "a stray temporary is not an entry"
    );
}

/// Two builds at once must both succeed. The sweep lock is advisory and skipping is correct — a
/// build must never fail because the cache could not be tidied.
#[test]
fn concurrent_builds_both_succeed() {
    if skip() {
        return;
    }
    let a = Sandbox::new("concurrent_a");
    let b = Sandbox::new("concurrent_b");
    let handles: Vec<_> = [a.root.clone(), b.root.clone()]
        .into_iter()
        .map(|root| {
            std::thread::spawn(move || {
                Command::new(stark_binary())
                    .arg("build")
                    .current_dir(&root)
                    .output()
                    .expect("stark build")
                    .status
                    .success()
            })
        })
        .collect();
    for handle in handles {
        assert!(handle.join().expect("thread"), "a concurrent build failed");
    }
}

/// A held sweep lock must not fail a build — it skips the sweep and proceeds.
#[test]
fn a_held_sweep_lock_does_not_fail_the_build() {
    if skip() {
        return;
    }
    let s = Sandbox::new("locked");
    s.build(&[]);
    let lock = s.root.join("target/stark/debug/.stark-cache-lock");
    std::fs::write(&lock, "held by a test").expect("take the lock");
    s.write_source("fn main() {\n    print(\"locked\");\n}\n");
    s.build(&[]);
    assert_eq!(s.run_binary("debug"), "locked");
    let _ = std::fs::remove_file(&lock);
}

/// **The property that outranks every performance claim.** Cached, uncached and clean-build
/// outputs are identical.
#[test]
fn cached_uncached_and_clean_builds_agree() {
    if skip() {
        return;
    }
    let source = "fn main() {\n    let mut total: Int32 = 0;\n    let mut i: Int32 = 0;\n\
                  while i < 20 { total = total + i * 3; i = i + 1; }\n    print(total);\n}\n";
    let mut outputs = Vec::new();

    let cached = Sandbox::new("agree_cached");
    cached.write_source(source);
    cached.build(&[]);
    cached.build(&[]); // second build is the cache hit
    outputs.push(cached.run_binary("debug"));

    let uncached = Sandbox::new("agree_uncached");
    uncached.write_source(source);
    uncached.build(&["--no-build-cache"]);
    outputs.push(uncached.run_binary("debug"));

    let clean = Sandbox::new("agree_clean");
    clean.write_source(source);
    clean.build(&[]);
    outputs.push(clean.run_binary("debug"));

    assert_eq!(
        outputs[0], outputs[1],
        "a cached build must observe identically to an uncached one"
    );
    assert_eq!(
        outputs[1], outputs[2],
        "an uncached build must observe identically to a clean one"
    );
    assert_eq!(outputs[0], "570");
}
