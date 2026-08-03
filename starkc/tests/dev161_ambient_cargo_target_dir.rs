//! **DEV-161: an ambient `CARGO_TARGET_DIR` must not be able to relocate the generated binary.**
//!
//! Cargo's default output is `<manifest dir>/target`, which is where the native backend looks for
//! the executable it just built. `CARGO_TARGET_DIR` in the environment silently overrides that, and
//! the child `cargo` inherits it. The build then SUCCEEDS, writes the executable somewhere else,
//! and the backend reports:
//!
//! ```text
//! Cargo succeeded but the expected binary is missing at <...>/target/debug/stark_program
//! ```
//!
//! — a diagnostic naming neither the cause nor the variable.
//!
//! # Why this is not a corner case
//!
//! `CARGO_TARGET_DIR` is a common global setting: a shared build cache across projects. Any
//! developer with it exported could not `stark build` **at all**, and the message would send them
//! looking at their program.
//!
//! # How it was found, which is the uncomfortable part
//!
//! It broke two of this repository's own tests — `mir_statement_consumers` and
//! `c788_resource_lifecycle` — and I twice reported them as pre-existing environmental failures
//! unrelated to my changes. The second time I "confirmed" it by stashing every change and
//! re-running. That control was worthless: the stashed run had the same variable exported, so it
//! reproduced the same failure and appeared to exonerate the tree. **Controlling for the code while
//! holding the environment fixed proves nothing about the environment.** An external review pushed
//! back on the dismissal, which is what prompted actually looking.
//!
//! The fix passes `--target-dir` explicitly, so the path the build writes and the path the backend
//! reads come from one value that no environment can separate.

use std::path::PathBuf;

mod common;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("starkc has a parent")
        .to_path_buf()
}

/// The build command must name its own target directory, whatever the environment says.
///
/// Asserted on the ARGV rather than by running a build under a mutated environment:
/// `std::env::set_var` is process-wide and these tests run in parallel, so a test that exported
/// `CARGO_TARGET_DIR` would corrupt whatever else was building at that moment. The argv is the
/// mechanism, and it is what a regression would remove.
#[test]
fn the_generated_build_pins_its_own_target_directory() {
    let source =
        std::fs::read_to_string(repo_root().join("starkc/src/backend/generated_rust/build.rs"))
            .expect("the backend build driver must be readable");

    assert!(
        source.contains("--target-dir"),
        "the generated crate's cargo invocation must pass `--target-dir` explicitly. Without it an \
         ambient CARGO_TARGET_DIR relocates the executable and the build fails with \"Cargo \
         succeeded but the expected binary is missing\" (DEV-161)"
    );

    // The read path must come from the SAME value. A `--target-dir` that pointed somewhere the
    // binary lookup did not follow would move the bug rather than fix it.
    assert!(
        source.contains("let target_dir = crate_dir.join(\"target\");"),
        "the target directory must be computed once"
    );
    assert!(
        source.contains("let mut binary_dir = target_dir;"),
        "the binary lookup must reuse the same `target_dir` value that was passed to cargo, so the \
         write path and the read path cannot diverge"
    );
}

/// The two suites the ambient variable broke. Named here so the connection is discoverable from
/// either end: if these start failing again with a "binary is missing" summary, this is the file
/// that explains why.
#[test]
fn the_affected_suites_are_recorded() {
    for suite in [
        "starkc/tests/mir_statement_consumers.rs",
        "starkc/tests/c788_resource_lifecycle.rs",
    ] {
        assert!(
            repo_root().join(suite).is_file(),
            "{suite} is the suite DEV-161 broke; if it moved, update this note rather than \
             deleting it"
        );
    }
}
