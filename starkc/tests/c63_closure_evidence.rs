//! WP-C6.3 CLOSURE EVIDENCE — installed-runtime layout, offline build, and version identity.
//!
//! CD-116 recorded this as a requirement that must land before C6.3 closes: the generated-code tests
//! prove the RUNTIME'S BEHAVIOUR, but nothing proved that a generated crate builds against an
//! **installed** runtime (rather than this source checkout), that the build needs **no network**, or
//! that a runtime version mismatch is **detected rather than silently linked**. C6.3 added a great
//! deal of runtime surface since — `format`, `vec`, `string`, `map` — so the gap widened with it.
//!
//! What makes the install test meaningful: `NativeToolchainOptions::runtime_crate` is a PATH, and
//! the development default points at `starkc/stark-runtime`. Every other native test therefore
//! builds against the working tree. Here the runtime is COPIED to a temp directory first and the
//! build is pointed at the copy, so a program that only compiles because of something in the
//! checkout — a path assumption, an uncommitted file, a stale `target/` artefact — fails here.

use starkc::backend::generated_rust::{
    emit_native_debug_with_toolchain, NativeBuildOptions, NativeToolchainOptions,
};
use starkc::diag::Severity;
use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Copy the runtime crate into `dest`, simulating an INSTALLED runtime: `Cargo.toml` plus `src/`,
/// and nothing else — no `target/`, no `.git`, no stray files the checkout happens to carry.
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

/// WP-C6.3 closure: a program exercising the surface C6.3 ADDED — primitive and composite
/// formatting, `String`, `Vec`, iteration, and `HashMap` — builds against an INSTALLED runtime, with
/// the build run `--offline`, and produces the expected output.
///
/// The offline half needs no separate test: `build_and_link` passes `--offline` unconditionally
/// (§11.3 — `stark-runtime` is dependency-free, so no build ever needs the network). A regression
/// that introduced a runtime dependency would fail HERE, because the copied crate has no vendored
/// registry and no network to fetch from.
#[test]
fn generated_crate_builds_against_an_installed_runtime_offline() {
    if !rustc_available() {
        eprintln!("SKIP: no rustc in this environment.");
        return;
    }
    let source = "fn main() {\n\
         let mut v: Vec<String> = Vec::new();\n\
         v.push(String::from(\"a\"));\n\
         v.push(String::from(\"b\"));\n\
         println(v);\n\
         let mut m: HashMap<Int32, Int32> = HashMap::new();\n\
         m.insert(1, 10);\n\
         m.insert(2, 20);\n\
         println(m.len());\n\
         let mut total: Int32 = 0;\n\
         let mut n: Vec<Int32> = Vec::new();\n\
         n.push(3);\n\
         n.push(4);\n\
         for x in n.iter() { total = total + *x; }\n\
         println(total);\n\
         println((1, true, 2.5));\n\
     }\n";
    let file = Arc::new(SourceFile::new(
        "c63_closure.stark".to_string(),
        source.to_string(),
    ));
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

    let root = std::env::temp_dir().join(format!("stark_c63_install_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let installed = root.join("installed-runtime");
    install_runtime_into(&installed);

    let toolchain = NativeToolchainOptions {
        rustc: PathBuf::from("rustc"),
        cargo: PathBuf::from("cargo"),
        runtime_crate: installed.clone(),
    };
    let artifact = emit_native_debug_with_toolchain(
        &verified,
        &NativeBuildOptions {
            target_dir: root.join("build"),
            target_contract: "stark-64-v1".to_string(),
            ..NativeBuildOptions::default()
        },
        &toolchain,
    )
    .unwrap_or_else(|e| panic!("build against the installed runtime failed: {e:?}"));

    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("run the installed-runtime binary");
    assert!(
        run.status.success(),
        "installed-runtime binary must exit 0; stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "[a, b]\n2\n7\n(1, true, 2.5)\n",
        "installed-runtime binary output"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// WP-C6.3 closure: the runtime VERSION identity is checked, not merely recorded. A binary whose
/// recorded runtime version differs from the runtime it linked must be rejected before any user code
/// runs (§9.2) — otherwise a stale installed runtime would be linked silently, which is exactly the
/// failure an installed (rather than in-tree) runtime makes possible.
#[test]
fn a_runtime_version_mismatch_is_detected() {
    let mut recorded = stark_runtime::version::BuildVersions {
        compiler_version: "0.1.0".to_string(),
        mir_version: "0.1".to_string(),
        mir_runtime_surface: "0.1-A8".to_string(),
        runtime_version: stark_runtime::version::RUNTIME_VERSION.to_string(),
        backend_version: "0.1".to_string(),
        rustc_version: "rustc-x".to_string(),
        target_triple: "aarch64-apple-darwin".to_string(),
        profile: "debug".to_string(),
    };
    // The matching case must PASS, or the mismatch assertion below would hold vacuously.
    assert!(stark_runtime::version::check(&recorded).is_ok());

    recorded.runtime_version = format!("{}-stale", stark_runtime::version::RUNTIME_VERSION);
    let mismatch = stark_runtime::version::check(&recorded)
        .expect_err("a differing runtime version must be rejected");
    assert_eq!(
        mismatch.actual_runtime_version,
        stark_runtime::version::RUNTIME_VERSION,
        "the mismatch must report the runtime actually linked"
    );
    assert!(
        mismatch.expected_runtime_version.ends_with("-stale"),
        "the mismatch must report the version the crate was generated for"
    );
}
