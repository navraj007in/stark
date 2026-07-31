//! `stark-random` provider registration and native execution.

use starkc::provider_abi::AbiParam;
use starkc::provider_registry;
use starkc::provider_resolve::ProviderSet;

fn host_triple() -> String {
    "x86_64-unknown-linux-gnu".to_string()
}

fn random() -> ProviderSet {
    ProviderSet::select(
        provider_registry::first_party(),
        &host_triple(),
        &["random".to_string()],
    )
    .expect("random provider selects for host")
}

#[test]
fn random_capability_resolves_to_first_party_provider() {
    let set = random();
    assert_eq!(set.providers().len(), 1);
    assert_eq!(set.providers()[0].crate_name, "stark-random-native");

    let call = set
        .resolve("random", "stark_random_secure_fill")
        .expect("secure fill resolves");
    assert_eq!(call.function.params, vec![AbiParam::BufferInOut]);
    assert_eq!(call.function.may_block, false);
    assert!(call.function.is_close_for.is_none());
}

#[test]
fn random_status_vocabulary_is_bounded() {
    let call = random()
        .resolve("random", "stark_random_secure_fill")
        .expect("secure fill resolves");

    let declared: Vec<(u32, String)> = call
        .status_binding
        .declared_codes()
        .map(|(code, name)| (*code, name.clone()))
        .collect();
    assert_eq!(
        declared,
        vec![
            (1, "RandomError::Unavailable".to_string()),
            (2, "RandomError::LimitExceeded".to_string()),
            (3, "RandomError::Other".to_string()),
        ],
        "secure random admits only unavailable, limit exceeded, and other"
    );
}

#[test]
fn random_provider_crate_is_locatable() {
    let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf();
    let location = provider_registry::crate_location("stark-random-native", &repo)
        .expect("random provider has a registry location");
    assert!(location.join("Cargo.toml").exists());
}

// ------------------------------------------------------------------ execution --
//
// The three tests above are declaration-level: they prove the capability RESOLVES, that its status
// vocabulary is bounded, and that the crate is locatable. None of them runs a line of STARK.
//
// That matters here more than usual. `stark-random/src/tests.stark` does test behaviour —
// reproducibility, zero-seed normalisation, the over-limit refusal — and it CANNOT RUN: `stark
// build` reports "program without a main function" for a library package, the same library-only
// build-mode gap recorded in `stark-io/BLOCKERS.md`. So `secure_bytes`, `next_u64` and `fill_bytes`
// were reachable by no executing test on any path.
//
// This closes that the way `io_minimal_executes_from_source_through_stark_io_package` does: vendor
// the package into a consumer, build it natively, run the binary, and assert on OBSERVED VALUES.

use starkc::native_build::{build_current_package, BuildCommandOptions};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("repo root")
        .to_path_buf()
}

/// **Secure and deterministic randomness, executed.**
///
/// Every assertion is on a value the program observed, not on the absence of an error:
///
/// - two generators on the SAME seed agree for four draws — reproducibility, which is the entire
///   contract of the deterministic API;
/// - two generators on DIFFERENT seeds disagree on the first draw — without this, a `next_u64` that
///   ignored its seed would satisfy the reproducibility check perfectly;
/// - seed `0` does not yield `0` — xorshift64 has a fixed point at zero and stays there forever, so
///   the normalisation in `deterministic` is load-bearing rather than cosmetic;
/// - `secure_bytes(32)` returns exactly 32 bytes, and they are not all zero — a provider that
///   silently filled nothing would otherwise pass on length alone;
/// - `secure_bytes(4097)` is refused as `LimitExceeded` by the PACKAGE, before the provider is
///   called. The native side enforces the same bound independently; this asserts the near one.
#[test]
fn random_secure_and_deterministic_execute_from_source() {
    let root = repo_root()
        .join("target")
        .join("c789-random")
        .join(format!("exec-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let vendored = root.join("vendor").join("stark-random");
    std::fs::create_dir_all(vendored.join("src")).expect("create vendored stark-random");
    std::fs::copy(
        repo_root().join("stark-random").join("starkpkg.json"),
        vendored.join("starkpkg.json"),
    )
    .expect("copy stark-random manifest");
    std::fs::copy(
        repo_root()
            .join("stark-random")
            .join("src")
            .join("lib.stark"),
        vendored.join("src").join("lib.stark"),
    )
    .expect("copy stark-random source");
    // `lib.stark` declares `mod tests;`, so the module file is part of the package and must be
    // vendored with it. Omitting it is E0208, not a missing test — the package does not compile.
    std::fs::copy(
        repo_root()
            .join("stark-random")
            .join("src")
            .join("tests.stark"),
        vendored.join("src").join("tests.stark"),
    )
    .expect("copy stark-random tests module");

    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create consumer src");
    std::fs::write(
        root.join("starkpkg.json"),
        r#"{
  "name": "c789_random_exec",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "capabilities": ["random"],
  "dependencies": {
    "stark_random": {
      "package": "stark-random",
      "path": "vendor/stark-random"
    }
  }
}"#,
    )
    .expect("write consumer manifest");
    std::fs::write(
        src.join("main.stark"),
        r#"use stark_random::RandomError;
use stark_random::deterministic;
use stark_random::secure_bytes;

fn main() {
    let mut left = deterministic(42u64);
    let mut right = deterministic(42u64);
    let mut i: Int32 = 0;
    while i < 4 {
        if left.next_u64() != right.next_u64() { panic("seeded sequences diverged"); }
        i = i + 1;
    }

    let mut a = deterministic(1u64);
    let mut b = deterministic(2u64);
    if a.next_u64() == b.next_u64() { panic("distinct seeds produced the same draw"); }

    let mut z = deterministic(0u64);
    if z.next_u64() == 0u64 { panic("zero seed was not normalized"); }

    match secure_bytes(32u64) {
        Ok(bytes) => {
            if bytes.len() != 32u64 { panic("secure_bytes returned the wrong length"); }
            let mut nonzero = false;
            let mut j: UInt64 = 0u64;
            while j < bytes.len() {
                if bytes[j] != 0u8 { nonzero = true; }
                j = j + 1u64;
            }
            if !nonzero { panic("secure_bytes returned all zeroes"); }
        }
        Err(_) => { panic("secure_bytes failed"); }
    }

    match secure_bytes(4097u64) {
        Ok(_) => { panic("the package limit was not enforced"); }
        Err(RandomError::LimitExceeded) => { }
        Err(_) => { panic("over-limit produced the wrong error"); }
    }

    println("random-ok");
}
"#,
    )
    .expect("write consumer source");

    let result = build_current_package(
        &root,
        &BuildCommandOptions {
            no_build_cache: true,
            ..BuildCommandOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("stark-random consumer must build: {error:?}"));

    let output = std::process::Command::new(&result.artifact_path)
        .output()
        .unwrap_or_else(|error| panic!("run the stark-random binary: {error}"));
    assert!(
        output.status.success(),
        "built program failed; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert_eq!(stdout.trim(), "random-ok");

    let _ = std::fs::remove_dir_all(&root);
}
