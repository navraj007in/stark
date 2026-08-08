//! §11/§11.1/§11.2/§12: generated-crate topology, the build-manifest schema, and driving
//! `cargo build` on the generated crate. The production `stark build` path supplies its resolved
//! rustc, Cargo, and runtime paths explicitly; direct backend tests use the compatibility entry
//! point in the parent module.

use super::{
    emit_program, BackendBuildFailure, BackendDiagnostic, NativeArtifact, NativeBuildOptions,
    NativeToolchainOptions,
};
use crate::backend::version::{self, BuildVersions};
use crate::mir::MirProgram;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

pub fn build_and_link(
    program: &MirProgram,
    options: &NativeBuildOptions,
    toolchain: &NativeToolchainOptions,
) -> Result<NativeArtifact, BackendDiagnostic> {
    let rustc_verbose = query_rustc_verbose(&toolchain.rustc)?;
    let rustc_version = parse_rustc_field(&rustc_verbose, "release: ")
        .ok_or_else(|| BackendDiagnostic::Io("could not parse `release:` from rustc -vV".into()))?
        .to_string();

    // WP-C6.4a (§8.4): target preflight, from the rustc probe alone -- before any source is
    // emitted, before the generated crate exists, and before Cargo or the linker runs. An
    // unsupported target must not be discovered by a later rustc error.
    let selection = preflight_from_rustc_verbose(&rustc_verbose, options.target_triple.as_deref())?;
    let versions = version::build_versions(
        rustc_version,
        selection.selected_triple().to_string(),
        options.profile,
    );

    // WP-C5.3e (CD-067): resolve the requested named contract BEFORE emitting. An unknown name
    // is rejected here rather than defaulted, because a layout answer is observable and
    // target-specific -- a silent fallback would report values for a target nobody asked about.
    let layout = crate::layout::contract_for(&options.target_contract)
        .map_err(|e| BackendDiagnostic::Unsupported(e.0))?;
    // WP-C6.4a: and it must be the contract the SELECTED TARGET declares. Resolving a valid
    // contract name is not the same as resolving the right one -- without this, a build could
    // answer `size_of` from one target's contract while recording another target's triple.
    if layout.identity.target_contract != selection.selected.layout_contract {
        return Err(BackendDiagnostic::TargetRejected(
            crate::target::TargetError::LayoutContractMismatch {
                target: selection.selected_triple(),
                declared: selection.selected.layout_contract,
                requested: layout.identity.target_contract.clone(),
            },
        ));
    }
    let source = emit_program::emit(program, &versions, &layout)?;
    let build_key = compute_build_key(program, &versions, &layout);
    let crate_dir = options
        .target_dir
        .join(options.profile.as_str())
        .join(&build_key);
    let expected_manifest = build_manifest_json(&versions, &build_key, &layout, &selection);
    reject_stale_artifact_version(&crate_dir, &expected_manifest)?;
    let src_dir = crate_dir.join("src");
    std::fs::create_dir_all(&src_dir)
        .map_err(|e| BackendDiagnostic::Io(format!("creating {}: {e}", src_dir.display())))?;

    // A10: the provider crates this program needs, resolved to locations before anything is
    // written -- an unlocatable provider must fail here, not as a linker error later.
    let providers = required_provider_crates(program, toolchain)?;
    let mut provider_versions = std::collections::BTreeMap::new();
    for (name, path) in &providers {
        provider_versions.insert(name.clone(), read_crate_version(path)?);
    }
    write_file(
        &crate_dir.join("Cargo.toml"),
        &generated_cargo_toml(&toolchain.runtime_crate, &providers),
    )?;
    // WP-C6.4c (§10.7): the lock the `--locked` below is checked against. `stark-runtime` is
    // dependency-free, so the whole graph is two path-only packages and the lock is fully
    // determined by the generated crate itself -- no registry, no versions to resolve, no
    // network. Writing it is what makes `--locked` a real assertion rather than a flag Cargo
    // would satisfy by generating whatever it liked.
    // WP-C6.4c hand-authored this lock on a stated premise: "stark-runtime is dependency-free, so
    // the whole graph is two path-only packages and the lock is fully determined by the generated
    // crate itself". That premise is false in both directions now. Providers bring their own path
    // dependencies, AND `stark-runtime` itself gained one (`stark-provider-abi`) -- which broke
    // EVERY native build, provider-free ones included, because a two-package lock no longer
    // matched a three-package graph and `--locked` correctly refused it.
    //
    // A hand-authored lock therefore has to track the runtime's own dependency graph forever, and
    // it silently breaks every build the moment that graph changes. It broke within days of being
    // written. Cargo resolves the lock instead -- ONCE, offline, so §11.3's no-network property is
    // untouched -- and the build below still passes `--locked`. What is given up is AUTHORSHIP of
    // the lock; the assertion that survives is the one that mattered: the build must not alter the
    // lock it was given.
    generate_lockfile_offline(toolchain, &crate_dir.join("Cargo.toml"))?;
    write_file(&src_dir.join("main.rs"), &source.main_rs)?;
    write_file(&crate_dir.join("build.json"), &expected_manifest)?;

    // §11.3 offline rule: `stark-runtime` is dependency-free, so `--offline` never needs a
    // registry index and proves no accidental network dependency crept in. `--locked` (WP-C6.4c)
    // adds the other half: Cargo must accept the lock as written rather than update it, so a
    // dependency appearing in the generated graph fails the build instead of being resolved
    // silently.
    //
    // **`--target-dir` is PASSED EXPLICITLY, and that is not redundant.** Cargo's default is
    // `<manifest dir>/target`, which is where the binary is looked for below — but an ambient
    // `CARGO_TARGET_DIR` in the environment silently overrides it, and the child process inherits
    // it. The build then succeeds, writes the executable somewhere else entirely, and this function
    // reports "Cargo succeeded but the expected binary is missing" — a diagnostic that names
    // neither the cause nor the variable.
    //
    // `CARGO_TARGET_DIR` is a common global setting (a shared build cache across projects), so this
    // is not a corner case: any such user could not `stark build` at all. Found because it broke
    // two of this repository's own tests, which were twice misdiagnosed as environmental
    // pre-existing failures — the control run had the same variable set.
    //
    // An explicit flag rather than clearing the variable: the path the build WRITES and the path
    // this function READS are then derived from the same value, and nothing about the caller's
    // environment can separate them.
    let manifest_path = crate_dir.join("Cargo.toml");
    let target_dir = crate_dir.join("target");
    let mut cargo_args = vec![
        OsString::from("build"),
        OsString::from("--locked"),
        OsString::from("--offline"),
        OsString::from("--manifest-path"),
        manifest_path.into_os_string(),
        OsString::from("--target-dir"),
        target_dir.clone().into_os_string(),
    ];
    if options.profile.is_release() {
        cargo_args.push(OsString::from("--release"));
    }
    if let Some(triple) = &options.target_triple {
        // Passed as SEPARATE argv entries, never concatenated into one string (§12): a target name
        // reaching a shell would be an injection surface, and `Command::args` never involves one.
        cargo_args.push(OsString::from("--target"));
        cargo_args.push(OsString::from(triple));
    }
    let command: Vec<String> = std::iter::once(format!("RUSTC={}", toolchain.rustc.display()))
        .chain(std::iter::once(toolchain.cargo.display().to_string()))
        .chain(
            cargo_args
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned()),
        )
        .collect();
    // WP-C7.2: path remapping, so the linked executable does not embed where it was built.
    //
    // C7.0 measured the problem precisely: two clean builds of one source from different absolute
    // paths produced DIFFERENT binaries, and each embedded 40 strings naming its own build
    // directory plus 22 naming the runtime crate's source. Both are rustc putting real paths into
    // debug info and `panic!` locations. Rust's own std is already remapped to `/rustc/<hash>/`,
    // which is the same remedy applied upstream.
    //
    // Two prefixes are remapped, and the second matters as much as the first: the build directory
    // varies per BUILD, and the runtime source path varies per INSTALLATION, so leaving it would
    // keep binaries machine-dependent even after the build dir was handled.
    //
    // This deliberately does not remap to the empty string. `/stark/crate` and `/stark/runtime` keep
    // a diagnostic readable — a panic location still says which file and line — while removing the
    // part that is nobody's business and is not reproducible.
    // Each prefix is remapped in BOTH its literal and its canonicalised form. On macOS `/var` is a
    // symlink to `/private/var`, so a build under `/var/folders/...` has rustc recording
    // `/private/var/folders/...` — and a remap written with the literal path silently matches
    // nothing. That is exactly what happened here: the first attempt removed the source-span paths
    // and left every linker artefact untouched, because those carry the resolved form.
    //
    // The flags go through `CARGO_ENCODED_RUSTFLAGS`, not `RUSTFLAGS`, and that is a correctness
    // requirement rather than a preference. `RUSTFLAGS` is SPACE-SEPARATED, so a build directory
    // containing a space — `/tmp/build dir/...` — splits one `--remap-path-prefix=FROM=TO` into two
    // arguments, and rustc rejects the fragment with "must contain '=' between FROM and TO". The
    // build then fails for every project under such a path, which is how this was found: the
    // spaces-in-paths portability test broke on all three platforms the moment remapping landed.
    // `CARGO_ENCODED_RUSTFLAGS` separates on `\x1f`, a byte no path contains, so each flag survives
    // as one argument whatever the path holds.
    let mut rustflags = std::ffi::OsString::new();
    let remap = |path: &Path, label: &str, flags: &mut std::ffi::OsString| {
        let mut forms = vec![path.to_path_buf()];
        if let Ok(canonical) = path.canonicalize() {
            if canonical != path {
                forms.push(canonical);
            }
        }
        for form in forms {
            if !flags.is_empty() {
                flags.push("\u{1f}");
            }
            flags.push("--remap-path-prefix=");
            flags.push(form.as_os_str());
            flags.push("=");
            flags.push(label);
        }
    };
    remap(&crate_dir, "/stark/crate", &mut rustflags);
    remap(&toolchain.runtime_crate, "/stark/runtime", &mut rustflags);

    let output = Command::new(&toolchain.cargo)
        .args(&cargo_args)
        .env("RUSTC", &toolchain.rustc)
        .env("CARGO_ENCODED_RUSTFLAGS", &rustflags)
        // Cargo ignores `RUSTFLAGS` whenever the encoded form is set, so removing it changes
        // nothing about this build — it is removed so the child environment has exactly one place
        // rustflags come from, and an inherited `RUSTFLAGS` cannot look like it is in effect.
        .env_remove("RUSTFLAGS")
        .output()
        .map_err(|e| {
            BackendDiagnostic::BuildFailed(Box::new(BackendBuildFailure {
                summary: "could not start Cargo for the generated crate".to_string(),
                stdout: String::new(),
                stderr: e.to_string(),
                build_dir: crate_dir.clone(),
                command: command.clone(),
                status: None,
            }))
        })?;
    if !output.status.success() {
        return Err(BackendDiagnostic::BuildFailed(Box::new(
            BackendBuildFailure {
                summary: "generated-crate build failed".to_string(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                build_dir: crate_dir.clone(),
                command,
                status: output.status.code(),
            },
        )));
    }

    // WP-C6.4a: the suffix comes from the SELECTED TARGET, not from `std::env::consts::EXE_SUFFIX`
    // (the compiler's own host). Identical today, since preflight admits only host builds; the
    // point is that the value is now derived from the thing it describes.
    // Cargo puts a `--release` build under `target/release/`, and a `--target`ed build under
    // `target/<triple>/<profile>/`. Reading the wrong one would find a STALE binary from an earlier
    // profile rather than failing, so the path is derived from the same options the command was.
    // The SAME value passed as `--target-dir` above, so the write path and the read path cannot
    // diverge however the environment is configured.
    let mut binary_dir = target_dir;
    if let Some(triple) = &options.target_triple {
        binary_dir = binary_dir.join(triple);
    }
    let binary_path = binary_dir
        .join(options.profile.as_str())
        .join(generated_binary_filename(
            selection.selected.executable_suffix,
        ));
    if !binary_path.exists() {
        return Err(BackendDiagnostic::BuildFailed(Box::new(
            BackendBuildFailure {
                summary: format!(
                    "Cargo succeeded but the expected binary is missing at {}",
                    binary_path.display()
                ),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                build_dir: crate_dir.clone(),
                command,
                status: output.status.code(),
            },
        )));
    }

    Ok(NativeArtifact {
        binary_path,
        build_dir: crate_dir,
    })
}

const BIN_NAME: &str = "stark_program";

fn generated_binary_filename(executable_suffix: &str) -> String {
    format!("{BIN_NAME}{executable_suffix}")
}

fn reject_stale_artifact_version(
    crate_dir: &Path,
    expected_manifest: &str,
) -> Result<(), BackendDiagnostic> {
    let manifest = crate_dir.join("build.json");
    let Ok(existing) = std::fs::read_to_string(&manifest) else {
        return Ok(());
    };
    if existing == expected_manifest {
        return Ok(());
    }
    std::fs::remove_dir_all(crate_dir).map_err(|error| {
        BackendDiagnostic::Io(format!(
            "rejecting stale generated artifact at {}: {error}",
            crate_dir.display()
        ))
    })
}

fn query_rustc_verbose(rustc: &Path) -> Result<String, BackendDiagnostic> {
    let output = Command::new(rustc)
        .arg("-vV")
        .output()
        .map_err(|e| BackendDiagnostic::Io(format!("invoking `{} -vV`: {e}", rustc.display())))?;
    if !output.status.success() {
        return Err(BackendDiagnostic::Io(
            "`rustc -vV` did not succeed".to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn parse_rustc_field<'a>(verbose: &'a str, field: &str) -> Option<&'a str> {
    verbose.lines().find_map(|line| line.strip_prefix(field))
}

/// WP-C6.4a (§8.4). Split out from [`build_and_link`] so the rejection can be tested from a
/// synthetic `rustc -vV` transcript alone: that proves the refusal happens at the probe, with no
/// crate emitted and no Cargo invoked, without requiring an unsupported machine to run the test on.
fn preflight_from_rustc_verbose(
    rustc_verbose: &str,
    requested_triple: Option<&str>,
) -> Result<crate::target::TargetSelection, BackendDiagnostic> {
    let host_triple = parse_rustc_field(rustc_verbose, "host: ")
        .ok_or_else(|| BackendDiagnostic::Io("could not parse `host:` from rustc -vV".into()))?;
    // WP-C7.1: the REQUESTED triple is passed through rather than hard-coded `None`.
    //
    // This is what makes `--target` honest. `preflight` refuses a non-native selection with
    // `HostOrTargetMetadataMismatch`, so a cross-target request is REJECTED with its own reason
    // instead of silently producing a host binary — which §3.3 forbids outright. An unknown triple
    // is rejected earlier still, by `select`, as `UnsupportedByStark`.
    crate::target::preflight(
        host_triple,
        requested_triple,
        &crate::target::HostOnlyAvailability,
    )
    .map_err(BackendDiagnostic::TargetRejected)
}

/// A10 (C7.8.2e): the distinct provider crates this program must link, name → location.
///
/// Names come from the program's validated call records; locations come from the build options.
/// A named crate with no configured location is a hard error rather than a silently unlinked
/// provider — the failure mode that would produce is a binary that fails to link with an
/// unresolved symbol, reported by the linker rather than by the compiler that knew better.
fn required_provider_crates(
    program: &MirProgram,
    toolchain: &NativeToolchainOptions,
) -> Result<std::collections::BTreeMap<String, std::path::PathBuf>, BackendDiagnostic> {
    let mut out = std::collections::BTreeMap::new();
    for call in &program.provider_calls {
        if out.contains_key(&call.provider_crate) {
            continue;
        }
        let path = toolchain
            .provider_crates
            .get(&call.provider_crate)
            .ok_or_else(|| {
                BackendDiagnostic::Unsupported(format!(
                    "provider `{}` needs crate `{}`, which this build has no location for",
                    call.provider.name, call.provider_crate
                ))
            })?;
        out.insert(call.provider_crate.clone(), path.clone());
    }
    Ok(out)
}

fn generated_cargo_toml(
    runtime_path: &Path,
    providers: &std::collections::BTreeMap<String, std::path::PathBuf>,
) -> String {
    format!(
        "# GENERATED by the STARK native backend (WP-C5.1b). Do not edit.\n\
         [package]\n\
         name = \"stark-generated\"\n\
         version = \"0.0.0\"\n\
         edition = \"2021\"\n\
         publish = false\n\
         \n\
         # Cuts inheritance from any ancestor workspace (here, starkc's) -- a generated crate\n\
         # is its own workspace root, never a member of the compiler's own workspace.\n\
         [workspace]\n\
         \n\
         [[bin]]\n\
         name = \"{BIN_NAME}\"\n\
         path = \"src/main.rs\"\n\
         \n\
         [dependencies]\n\
         stark-runtime = {{ path = {} }}\n\
         {}\
         \n\
         [profile.dev]\n\
         panic = \"abort\"\n\
         \n\
         # WP-C7.1 (§6.6): every release setting is written EXPLICITLY rather than inherited.\n\
         #\n\
         # `panic` is the one that matters. Cargo's release default is \"unwind\", and unwinding\n\
         # runs destructors -- which DROP-ABORT-001 forbids after a trap. STARK's own traps exit\n\
         # through `process::exit` and so are safe either way, and the C7.0 panic-site audit found\n\
         # no user-reachable Rust panic in the runtime today. This is set anyway, as\n\
         # defence-in-depth: the guarantee should rest on the build, not only on that audit\n\
         # staying true as the runtime grows.\n\
         #\n\
         # `overflow-checks` is recorded rather than relied upon. STARK arithmetic lowers to\n\
         # explicit `checked_*` calls, so trapping does not depend on this -- but leaving it\n\
         # unstated would invite the reader to assume it does.\n\
         [profile.release]\n\
         panic = \"abort\"\n\
         opt-level = 3\n\
         overflow-checks = true\n\
         debug-assertions = false\n\
         lto = false\n\
         codegen-units = 16\n\
         strip = false\n",
        toml_basic_string(runtime_path),
        // One `name = { path = "…" }` line per provider crate, in name order so the manifest is
        // byte-identical across builds of the same program.
        providers
            .iter()
            .map(|(name, path)| {
                format!("{name} = {{ path = {} }}\n", toml_basic_string(path))
            })
            .collect::<String>(),
    )
}

/// A TOML **basic string** for a filesystem path (§9.10).
///
/// This used to be `{:?}` on the `Path` -- Rust's `Debug`, used as if it were TOML quoting. The
/// two agree on the cases that occur constantly (a backslash becomes `\\`, a quote becomes `\"`,
/// so Windows paths happen to come out right) and disagree exactly where a hand-rolled escape
/// always disagrees: `Debug` renders a control character as `\u{7}` and a non-UTF-8 byte as
/// `\xNN`, and TOML accepts neither spelling. Escaping to TOML's own rules removes the pun.
///
/// Lossy conversion is deliberate and safe here: a path that is not valid UTF-8 cannot be written
/// into a TOML document at all, and the replacement character produces a path that fails loudly at
/// Cargo rather than a document that fails to parse.
fn toml_basic_string(path: &Path) -> String {
    let mut out = String::with_capacity(path.as_os_str().len() + 2);
    out.push('"');
    for c in path.to_string_lossy().chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            // TOML requires the remaining control characters as `\uXXXX` -- four hex digits, not
            // Rust's `\u{...}` form.
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Has Cargo resolve the generated crate's lock, offline.
///
/// `--offline` keeps §11.3's no-network property; the resolution itself is Cargo's, because the
/// dependency graph is Cargo's to know. See the call site for why the backend stopped authoring
/// this file.
fn generate_lockfile_offline(
    toolchain: &NativeToolchainOptions,
    manifest_path: &Path,
) -> Result<(), BackendDiagnostic> {
    let command = vec![
        toolchain.cargo.display().to_string(),
        "generate-lockfile".to_string(),
        "--offline".to_string(),
        "--manifest-path".to_string(),
        manifest_path.display().to_string(),
    ];
    let crate_dir = manifest_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();

    // Failures here are reported in the SAME shape as a build failure -- summary, streams, retained
    // directory and the exact command. Cargo runs twice now (resolve, then build), and a caller
    // debugging a broken toolchain should not get a materially poorer diagnostic depending on
    // which invocation failed first.
    let output = Command::new(&toolchain.cargo)
        .arg("generate-lockfile")
        .arg("--offline")
        .arg("--manifest-path")
        .arg(manifest_path)
        .output()
        .map_err(|e| {
            BackendDiagnostic::BuildFailed(Box::new(BackendBuildFailure {
                summary: "could not start Cargo to resolve the generated crate's lock".to_string(),
                stdout: String::new(),
                stderr: e.to_string(),
                build_dir: crate_dir.clone(),
                command: command.clone(),
                status: None,
            }))
        })?;
    if !output.status.success() {
        return Err(BackendDiagnostic::BuildFailed(Box::new(
            BackendBuildFailure {
                summary: "generated-crate lock resolution failed".to_string(),
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                build_dir: crate_dir,
                command,
                status: output.status.code(),
            },
        )));
    }
    Ok(())
}

/// The `version = "…"` of a path dependency's `[package]` table. Same narrow scan as the runtime's,
/// for the same reason: these are manifests this repository writes.
fn read_crate_version(crate_dir: &Path) -> Result<String, BackendDiagnostic> {
    let manifest_path = crate_dir.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| BackendDiagnostic::Io(format!("reading {}: {e}", manifest_path.display())))?;
    text.lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("version"))
        .and_then(|rest| rest.trim_start().strip_prefix('='))
        .map(|rest| rest.trim().trim_matches('"').to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            BackendDiagnostic::Io(format!(
                "no `version = \"…\"` in {}; cannot write a lock for the generated crate",
                manifest_path.display()
            ))
        })
}

/// §11.1: source-content + version + target + profile hash, for build isolation/diagnostics
/// (explicitly not a security boundary, and not the incremental cache §11 says C5 doesn't need).
///
/// **DEV-095 (WP-C5.3 opening condition, CD-052/CD-053).** This used to hash `program.dump()`
/// alone, and `dump()` serializes only the version header and the bodies. The MIR contract is
/// explicit that the **nominal type context and the destructor map are in-memory parts of the
/// compilation unit that the textual dump does not serialize** — so two programs with identical
/// dumps but different struct fields, different `Drop` impls, or different `Copy` classification
/// hashed to the SAME key, and the second build could silently reuse the first's generated crate.
///
/// That could not bite while the backend admitted only primitives (no aggregates, no `Drop`), and
/// it was recorded rather than fixed at the time. It is fixed **before** WP-C5.3 makes it
/// reachable, which is what "opening condition" means: the key covers every semantic input that
/// can affect generated code, not merely the ones the current backend happens to read.
fn compute_build_key(
    program: &MirProgram,
    versions: &BuildVersions,
    layout: &crate::layout::TargetLayout,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(build_key_input(program, versions, layout).as_bytes());
    let digest = hasher.finalize();
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}

/// The canonical, deterministic encoding of everything [`compute_build_key`] hashes.
///
/// Separated from the hashing so it can be inspected and diffed directly in tests — a test that
/// asserts "these two keys differ" says nothing about *which* input made them differ, but a test
/// that diffs this string does. Determinism comes from the data structures themselves: the type
/// context is `BTreeMap`/`BTreeSet` (sorted iteration) and `program.bodies` is sorted by canonical
/// symbol for exactly this reason.
///
/// Sections, and why each is a generated-code input:
///
/// - **versions** — compiler/MIR/runtime-surface/runtime/backend/rustc/target/profile. All eight
///   are embedded in or shape the generated crate (§9.2's version-identity record).
/// - **entry** — the entry symbol becomes Rust's literal `fn main()`.
/// - **sources** — file *names* reach generated code verbatim (trap sites resolve `file:line:col`
///   at compile time, `emit_bodies::resolve_source_location`), and §11.1 requires source-content
///   hashes outright.
/// - **types** — struct fields and enum variants determine layout and projection typing;
///   `drop_impls` determines which destructor a `Drop` terminator dispatches to; `copy_types`
///   determines whether a move is a copy. None of these appear in `dump()`. This is DEV-095.
/// - **bodies** — `dump()`, which is already the contract's deterministic body serialization.
fn build_key_input(
    program: &MirProgram,
    versions: &BuildVersions,
    layout: &crate::layout::TargetLayout,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    // A format tag, so a future change to this encoding is visibly a different scheme rather than
    // silently colliding with keys computed under the old one.
    let _ = writeln!(out, "=== stark build key v2 (DEV-095) ===");

    let _ = writeln!(out, "[versions]");
    let _ = writeln!(out, "compiler={}", versions.compiler_version);
    let _ = writeln!(out, "mir={}", versions.mir_version);
    let _ = writeln!(out, "mir-runtime-surface={}", versions.mir_runtime_surface);
    let _ = writeln!(out, "runtime={}", versions.runtime_version);
    let _ = writeln!(out, "backend={}", versions.backend_version);
    let _ = writeln!(out, "rustc={}", versions.rustc_version);
    let _ = writeln!(out, "target={}", versions.target_triple);
    let _ = writeln!(out, "profile={}", versions.profile);

    // WP-C5.3e (CD-067): the layout contract's IDENTITY, not its values. Two builds that answer
    // `size_of` differently must not share a cache entry, and the identity is what a build report
    // can be held to. The contract version and the compiler's revision of it move independently:
    // the first changes observable answers, the second does not.
    let _ = writeln!(out, "[layout]");
    let _ = writeln!(out, "target-contract={}", layout.identity.target_contract);
    let _ = writeln!(
        out,
        "layout-contract-version={}",
        layout.identity.layout_contract_version
    );
    let _ = writeln!(
        out,
        "compiler-layout-revision={}",
        layout.identity.compiler_layout_revision
    );

    let _ = writeln!(out, "[entry]");
    let _ = writeln!(out, "{}", super::mangle::ENTRY_SYMBOL);

    // Package graph identity (§11.1's own list) has no separate representation at this scope.
    // C5.4 linkage merges the verified package bodies into one compilation unit; the source table
    // below plus the canonical bodies serialized later carry the inputs that affect generated
    // code.
    let _ = writeln!(out, "[sources]");
    for file in program.sources.iter() {
        let mut content = Sha256::new();
        content.update(file.src.as_bytes());
        let digest = content.finalize();
        let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        // AS1b-iii: the index is the registry's `SourceId`, which is what every span names. It was
        // the MIR-local `FileId`. Both are dense and load-ordered, so the table's shape is
        // unchanged; what it indexes is now the identity the rest of the compiler uses.
        let _ = writeln!(out, "{} {} {hex}", file.id().as_u32(), file.name);
    }

    let types = &program.types;
    let _ = writeln!(out, "[types.struct_fields]");
    for ((item, args), fields) in &types.struct_fields {
        let _ = writeln!(
            out,
            "{}: {}",
            nominal_key(*item, args),
            join_tys(fields.iter())
        );
    }
    let _ = writeln!(out, "[types.enum_variants]");
    for ((item, args), variants) in &types.enum_variants {
        let payloads: Vec<String> = variants
            .iter()
            .map(|payload| format!("[{}]", join_tys(payload.iter())))
            .collect();
        let _ = writeln!(out, "{}: {}", nominal_key(*item, args), payloads.join(", "));
    }
    let _ = writeln!(out, "[types.drop_impls]");
    for ((item, args), symbol) in &types.drop_impls {
        let _ = writeln!(out, "{}: {symbol}", nominal_key(*item, args));
    }
    let _ = writeln!(out, "[types.copy_types]");
    for (item, args) in &types.copy_types {
        let _ = writeln!(out, "{}", nominal_key(*item, args));
    }

    let _ = writeln!(out, "[bodies]");
    out.push_str(&program.dump());
    out
}

fn nominal_key(item: u32, args: &[crate::mir::MirTy]) -> String {
    format!("{item}[{}]", join_tys(args.iter()))
}

fn join_tys<'a>(tys: impl Iterator<Item = &'a crate::mir::MirTy>) -> String {
    tys.map(crate::mir::dump_ty).collect::<Vec<_>>().join(", ")
}

/// WP-C5.3e (CD-067): the report carries the layout contract's IDENTITY, so a build's observable
/// `size_of`/`align_of` answers can always be attributed to a named contract at a stated version.
/// WP-C6.4a adds `host_triple`, `target_tier` and `target_pointer_width` beside the existing
/// `target_triple`, which now names the SELECTED target. Host and selected target are separate
/// fields even while they are always equal, because §33 asks the record to identify them
/// separately -- a manifest that cannot tell them apart is one that will report the host as the
/// target the day they differ, and say nothing about having done so.
///
/// The version record embedded in the binary itself (`stark_runtime::version::BuildVersions`) is
/// deliberately NOT extended: it is the runtime crate's shared type, its surface is separately
/// versioned (§9.2), and a binary that can only ever be a host build has nothing to disambiguate.
/// The compiler-side manifest is where target metadata belongs while cross-compilation is C7's.
fn build_manifest_json(
    versions: &BuildVersions,
    build_key: &str,
    layout: &crate::layout::TargetLayout,
    selection: &crate::target::TargetSelection,
) -> String {
    format!(
        "{{\n  \"build_key\": {},\n  \"compiler_version\": {},\n  \"mir_version\": {},\n  \
         \"mir_runtime_surface\": {},\n  \"runtime_version\": {},\n  \"backend_version\": {},\n  \
         \"rustc_version\": {},\n  \"host_triple\": {},\n  \"target_triple\": {},\n  \
         \"target_tier\": {},\n  \"target_pointer_width\": {},\n  \"profile\": {},\n  \
         \"target_contract\": {},\n  \"layout_contract_version\": {},\n  \
         \"compiler_layout_revision\": {}\n}}\n",
        json_str(build_key),
        json_str(&versions.compiler_version),
        json_str(&versions.mir_version),
        json_str(&versions.mir_runtime_surface),
        json_str(&versions.runtime_version),
        json_str(&versions.backend_version),
        json_str(&versions.rustc_version),
        json_str(&selection.host_triple),
        json_str(&versions.target_triple),
        json_str(&selection.selected.tier.to_string()),
        selection.selected.pointer_width,
        json_str(&versions.profile),
        json_str(&layout.identity.target_contract),
        layout.identity.layout_contract_version,
        layout.identity.compiler_layout_revision,
    )
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

fn write_file(path: &Path, contents: &str) -> Result<(), BackendDiagnostic> {
    std::fs::write(path, contents)
        .map_err(|e| BackendDiagnostic::Io(format!("writing {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    //! DEV-095's cache-invalidation coverage: **every semantic input that can affect generated
    //! code must change the build key.** The defect this fixes was not that some input hashed
    //! wrongly — it was that four inputs were not hashed at all, and nothing would have noticed
    //! until a stale generated crate was silently reused.
    //!
    //! Each test mutates exactly ONE input of an otherwise identical program, so a failure names
    //! the input that stopped being covered rather than reporting "the key changed" or not.

    use super::*;
    use crate::layout::Layout;
    use crate::mir::{MirProgram, MirTy};
    use crate::parser::{parse, ParseMode};
    use crate::resolve::resolve;
    use crate::source::SourceFile;
    use crate::typecheck;
    use std::sync::Arc;

    /// A synthetic `rustc -vV` transcript. The preflight tests below read one of these and never
    /// touch a real toolchain — which is the point: §8.4 says an unsupported target must be
    /// rejected before Cargo and the linker, and the only way to *prove* that is to show the
    /// refusal happening with nothing but the probe's output in hand.
    fn rustc_verbose_for(host: &str) -> String {
        format!(
            "rustc 1.93.0 (254b59607 2026-01-19)\nbinary: rustc\ncommit-hash: 254b59607\n\
             commit-date: 2026-01-19\nhost: {host}\nrelease: 1.93.0\nLLVM version: 21.1.8\n"
        )
    }

    #[test]
    fn preflight_accepts_a_tier1_host_and_reports_it_as_the_selected_target() {
        for host in crate::target::tier1_triples() {
            let selection = preflight_from_rustc_verbose(&rustc_verbose_for(host), None).unwrap();
            assert_eq!(selection.host_triple, host);
            assert_eq!(selection.selected_triple(), host);
            assert_eq!(selection.selected.tier, crate::target::Tier::One);
        }
    }

    /// §8.4/§8.5(5). The rejection is produced from the rustc probe alone — no crate directory,
    /// no `Cargo.toml`, no Cargo process — so an unsupported target can never be discovered by a
    /// later rustc or linker error.
    #[test]
    fn preflight_rejects_an_unsupported_host_before_anything_is_generated() {
        let error =
            preflight_from_rustc_verbose(&rustc_verbose_for("riscv64gc-unknown-linux-gnu"), None)
                .unwrap_err();
        match error {
            BackendDiagnostic::TargetRejected(crate::target::TargetError::UnsupportedByStark {
                ref requested,
                ..
            }) => assert_eq!(requested, "riscv64gc-unknown-linux-gnu"),
            other => panic!("expected TargetRejected/UnsupportedByStark, got {other:?}"),
        }
        // And it is not the "backend cannot lower this" class, which means something else.
        assert!(!matches!(
            preflight_from_rustc_verbose(&rustc_verbose_for("riscv64gc-unknown-linux-gnu"), None),
            Err(BackendDiagnostic::Unsupported(_))
        ));
    }

    #[test]
    fn a_rustc_transcript_without_a_host_field_is_an_io_error_not_a_target_rejection() {
        assert!(matches!(
            preflight_from_rustc_verbose("release: 1.93.0\n", None),
            Err(BackendDiagnostic::Io(_))
        ));
    }

    /// §9.10. The old `{:?}` spelling passed the first three of these and failed the fourth.
    #[test]
    fn manifest_paths_are_escaped_to_toml_rules_not_rust_debug_rules() {
        assert_eq!(
            toml_basic_string(Path::new("/plain/path")),
            "\"/plain/path\""
        );
        assert_eq!(
            toml_basic_string(Path::new("/with space/rt")),
            "\"/with space/rt\""
        );
        assert_eq!(
            toml_basic_string(Path::new(r"C:\Users\runner\stark-runtime")),
            r#""C:\\Users\\runner\\stark-runtime""#
        );
        // Unicode is left literal: TOML basic strings take it verbatim, and escaping it would
        // change a path that was already correct.
        assert_eq!(
            toml_basic_string(Path::new("/tmp/naïve/ünïcode")),
            "\"/tmp/naïve/ünïcode\""
        );
        // The case Rust's `Debug` renders as `\u{7}`, which TOML does not accept.
        assert_eq!(
            toml_basic_string(Path::new("/tmp/a\u{7}b")),
            "\"/tmp/a\\u0007b\""
        );
        assert_eq!(toml_basic_string(Path::new("/tmp/a\tb")), "\"/tmp/a\\tb\"");
        assert_eq!(
            toml_basic_string(Path::new("/tmp/say \"hi\"")),
            "\"/tmp/say \\\"hi\\\"\""
        );
    }

    /// A generated manifest carrying an adversarial path must still be a document Cargo can read.
    /// Checked structurally rather than by parsing TOML (the compiler has no TOML dependency, by
    /// design): the escaped path occupies exactly one line, and its quotes are balanced.
    #[test]
    fn a_generated_manifest_with_an_adversarial_runtime_path_stays_one_well_formed_line() {
        for path in [
            "/tmp/with space/stark-runtime",
            r"C:\Users\stark runner\lib\stark\stark-runtime",
            "/tmp/naïve path/stark-runtime",
        ] {
            // A10: a provider dependency line is built the same way and is exposed to the same
            // adversarial paths, so it is checked here rather than in a parallel test that could
            // drift.
            let mut providers = std::collections::BTreeMap::new();
            providers.insert(
                "stark-time-native".to_string(),
                std::path::PathBuf::from(path),
            );
            let manifest = generated_cargo_toml(Path::new(path), &providers);
            for prefix in ["stark-runtime = ", "stark-time-native = "] {
                let dep_line = manifest
                    .lines()
                    .find(|l| l.starts_with(prefix))
                    .unwrap_or_else(|| panic!("no dependency line for {prefix} in\n{manifest}"));
                assert!(dep_line.ends_with('}'), "{dep_line}");
                let quotes = dep_line.matches('"').count() - dep_line.matches("\\\"").count();
                assert_eq!(quotes, 2, "unbalanced quoting in {dep_line}");
            }
        }
    }

    #[test]
    fn a_crate_version_is_read_from_the_crate_being_linked() {
        // Generalised from the runtime-only reader when A10 made provider crates linkable: the
        // same narrow scan now serves any path dependency, so it is tested against the runtime it
        // was written for AND a provider crate.
        let runtime = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("stark-runtime");
        assert_eq!(read_crate_version(&runtime).unwrap(), "0.1.0");
        let missing = read_crate_version(Path::new("/nonexistent/stark-runtime"));
        assert!(matches!(missing, Err(BackendDiagnostic::Io(_))));
    }

    /// One labelled, single-input mutation. Named aliases rather than inline `Box<dyn Fn>`
    /// signatures so the tests below read as what they are: a list of "change exactly this, then
    /// assert the key moved".
    type ProgramMutation = (&'static str, Box<dyn Fn(&mut MirProgram)>);
    type VersionMutation = (&'static str, Box<dyn Fn(&mut BuildVersions)>);

    fn versions() -> BuildVersions {
        version::build_versions(
            "1.99.0".to_string(),
            "aarch64-apple-darwin".to_string(),
            crate::backend::generated_rust::Profile::Debug,
        )
    }

    fn program(name: &str, source: &str) -> MirProgram {
        let file = Arc::new(SourceFile::new(name, source.to_string()));
        let (ast, pd) = parse(&file, ParseMode::Program);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (hir, rd) = resolve(&ast, file.clone());
        assert!(rd.is_empty(), "resolve: {rd:?}");
        let checked = typecheck::analyze(&hir);
        match crate::mir::lower::lower_program(
            &hir,
            &checked.tables,
            hir.source_named(&file.name).expect("registered"),
        ) {
            Ok(program) => program,
            Err(e) => panic!("must lower: {} @ {:?}", e.what, e.span),
        }
    }

    fn trivial() -> MirProgram {
        program("key.stark", "fn main() { let a: Int32 = 1; }")
    }

    fn key(p: &MirProgram) -> String {
        compute_build_key(p, &versions(), &crate::layout::TargetLayout::default())
    }

    /// Baseline: the key is a pure function of its inputs. Without this, every "the key changed"
    /// assertion below could be satisfied by a key that simply changes every time.
    #[test]
    fn identical_programs_produce_identical_keys() {
        assert_eq!(key(&trivial()), key(&trivial()));
        // And repeated computation over one program is stable (BTreeMap/BTreeSet iteration and
        // the sorted body order are what guarantee this).
        let p = trivial();
        assert_eq!(key(&p), key(&p));
    }

    #[test]
    fn generated_binary_filename_is_platform_aware() {
        assert_eq!(generated_binary_filename(""), "stark_program");
        assert_eq!(generated_binary_filename(".exe"), "stark_program.exe");
    }

    #[test]
    fn a_different_body_produces_a_different_key() {
        let a = trivial();
        let b = program("key.stark", "fn main() { let a: Int32 = 2; }");
        assert_ne!(key(&a), key(&b));
    }

    /// **THE DEV-095 REGRESSION.** Two programs whose `dump()` output is byte-identical but whose
    /// nominal type context differs must not share a key. Under the old `hash(dump())` key they
    /// did — and `dump()` equality is asserted here explicitly, so this test is meaningless the
    /// day it stops being the actual condition.
    #[test]
    fn type_context_changes_are_invisible_to_dump_but_must_change_the_key() {
        let base = trivial();

        for (label, mutate) in mutations() {
            let mut mutated = base.clone();
            mutate(&mut mutated);
            assert_eq!(
                base.dump(),
                mutated.dump(),
                "{label}: precondition — this mutation must be invisible to dump(), \
                 otherwise the test proves nothing about the type context"
            );
            assert_ne!(
                key(&base),
                key(&mutated),
                "{label}: build key did not change — a stale generated crate would be reused"
            );
        }
    }

    /// One mutation per `TypeContext` field, so a field that stops being hashed is named by the
    /// failure rather than hidden behind a single omnibus assertion.
    fn mutations() -> Vec<ProgramMutation> {
        vec![
            (
                "struct_fields: a new nominal",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .struct_fields
                        .insert((7, vec![]), vec![MirTy::Int32]);
                }),
            ),
            (
                "struct_fields: same nominal, different field type",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .struct_fields
                        .insert((7, vec![]), vec![MirTy::Int64]);
                }),
            ),
            (
                "struct_fields: same nominal, different type arguments",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .struct_fields
                        .insert((7, vec![MirTy::Bool]), vec![MirTy::Int32]);
                }),
            ),
            (
                "enum_variants: a new nominal",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .enum_variants
                        .insert((8, vec![]), vec![vec![], vec![MirTy::Int32]]);
                }),
            ),
            (
                "enum_variants: same variants, different order",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .enum_variants
                        .insert((8, vec![]), vec![vec![MirTy::Int32], vec![]]);
                }),
            ),
            (
                "drop_impls: a nominal gains a destructor",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .drop_impls
                        .insert((7, vec![]), "Foo::drop@[]".to_string());
                }),
            ),
            (
                "drop_impls: the destructor instance changes",
                Box::new(|p: &mut MirProgram| {
                    p.types
                        .drop_impls
                        .insert((7, vec![]), "Bar::drop@[]".to_string());
                }),
            ),
            (
                "copy_types: a nominal becomes Copy",
                Box::new(|p: &mut MirProgram| {
                    p.types.copy_types.insert((7, vec![]));
                }),
            ),
        ]
    }

    /// File NAMES reach generated code verbatim — a trap site emits `file:line:column` resolved
    /// at compile time — so two builds of the same MIR from differently-named files must not
    /// share a crate.
    #[test]
    fn a_different_source_file_name_produces_a_different_key() {
        let a = trivial();
        let b = program("other.stark", "fn main() { let a: Int32 = 1; }");
        assert_ne!(key(&a), key(&b));
    }

    /// §11.1 requires source-content hashes. A content change that happens not to move any span
    /// (here, an appended comment) leaves `dump()` identical — and must still change the key.
    #[test]
    fn a_source_content_change_invisible_to_dump_produces_a_different_key() {
        let a = trivial();
        let b = program(
            "key.stark",
            "fn main() { let a: Int32 = 1; }\n// trailing\n",
        );
        assert_eq!(
            a.dump(),
            b.dump(),
            "precondition: the appended comment must not move any span"
        );
        assert_ne!(key(&a), key(&b));
    }

    /// All eight version axes, each moved independently. A version that stops being hashed is
    /// named by its own failure.
    #[test]
    fn every_version_axis_changes_the_key() {
        let p = trivial();
        let base = versions();
        let baseline = compute_build_key(&p, &base, &crate::layout::TargetLayout::default());

        let axes: Vec<VersionMutation> = vec![
            (
                "compiler",
                Box::new(|v: &mut BuildVersions| v.compiler_version = "9.9.9".into()),
            ),
            (
                "mir",
                Box::new(|v: &mut BuildVersions| v.mir_version = "9.9".into()),
            ),
            (
                "mir_runtime_surface",
                Box::new(|v: &mut BuildVersions| v.mir_runtime_surface = "9.9-Z9".into()),
            ),
            (
                "runtime",
                Box::new(|v: &mut BuildVersions| v.runtime_version = "9.9".into()),
            ),
            (
                "backend",
                Box::new(|v: &mut BuildVersions| v.backend_version = "9.9".into()),
            ),
            (
                "rustc",
                Box::new(|v: &mut BuildVersions| v.rustc_version = "9.9.9".into()),
            ),
            (
                "target_triple",
                Box::new(|v: &mut BuildVersions| v.target_triple = "wasm32-unknown-unknown".into()),
            ),
            (
                "profile",
                Box::new(|v: &mut BuildVersions| v.profile = "release".into()),
            ),
        ];

        for (label, mutate) in axes {
            let mut v = base.clone();
            mutate(&mut v);
            assert_ne!(
                baseline,
                compute_build_key(&p, &v, &crate::layout::TargetLayout::default()),
                "{label}: version axis is not in the build key"
            );
        }
    }

    #[test]
    fn the_mir_shape_revision_is_part_of_the_build_key_input() {
        let input = build_key_input(
            &trivial(),
            &versions(),
            &crate::layout::TargetLayout::default(),
        );
        // Pinned against the constant, not a literal. A12 bumped this to 0.3 and the previous
        // hard-coded `mir=0.2` failed — which proved the axis is wired up, and then had to be
        // hand-edited to say so again. Reading the constant keeps the property under test (the
        // revision reaches the key) without re-breaking on every future bump.
        let expected = format!("mir={}", crate::mir::MIR_VERSION);
        assert!(
            input.contains(&expected),
            "the MIR shape revision ({expected}) must be visible in cache-key input:\n{input}"
        );
    }

    #[test]
    fn stale_artifact_manifest_is_rejected_before_reuse() {
        let root =
            std::env::temp_dir().join(format!("stark-stale-artifact-{}", std::process::id()));
        let crate_dir = root.join("debug").join("abc");
        std::fs::create_dir_all(crate_dir.join("src")).expect("create stale crate");
        std::fs::write(crate_dir.join("src").join("main.rs"), "stale").expect("write stale source");
        std::fs::write(
            crate_dir.join("build.json"),
            "{\n  \"build_key\": \"abc\",\n  \"mir_version\": \"0.1\"\n}\n",
        )
        .expect("write stale manifest");

        reject_stale_artifact_version(
            &crate_dir,
            "{\n  \"build_key\": \"abc\",\n  \"mir_version\": \"0.2\"\n}\n",
        )
        .expect("stale crate must be removable");
        assert!(
            !crate_dir.exists(),
            "stale generated crate must be removed before the backend can reuse it"
        );

        let _ = std::fs::remove_dir_all(root);
    }

    /// WP-C5.3e (CD-067): two builds whose layout contract identity differs answer `size_of`
    /// differently, so they must not share a cache entry. The contract VERSION and the compiler's
    /// revision of it move independently and both count.
    #[test]
    fn the_build_key_changes_with_the_layout_contract_identity() {
        let p = trivial();
        let v = versions();
        let base = crate::layout::TargetLayout::default();
        let baseline = compute_build_key(&p, &v, &base);

        let mut renamed = base.clone();
        renamed.identity.target_contract = "stark-32-v1".to_string();
        assert_ne!(
            baseline,
            compute_build_key(&p, &v, &renamed),
            "the target contract name is not in the build key"
        );

        let mut revised = base.clone();
        revised.identity.layout_contract_version = 2;
        assert_ne!(
            baseline,
            compute_build_key(&p, &v, &revised),
            "the layout contract version is not in the build key"
        );

        let mut reimplemented = base.clone();
        reimplemented.identity.compiler_layout_revision = 2;
        assert_ne!(
            baseline,
            compute_build_key(&p, &v, &reimplemented),
            "the compiler layout revision is not in the build key"
        );

        // The VALUES are deliberately not hashed -- the identity is what a build is accountable
        // to, and hashing values as well would make the key change without the identity changing,
        // which is precisely the drift the identity exists to make visible.
        let mut silently_changed = base.clone();
        silently_changed.int32 = Layout::new(8, 8);
        assert_eq!(
            baseline,
            compute_build_key(&p, &v, &silently_changed),
            "changing a value without bumping the identity must be visible as a STALE key, not \
             hidden behind a new one"
        );
    }

    /// The encoding is what tests can diff; this pins that it actually carries every section, so
    /// a section deleted from `build_key_input` fails here with a name rather than silently
    /// weakening every other test in this module.
    #[test]
    fn the_key_input_carries_every_documented_section() {
        let input = build_key_input(
            &trivial(),
            &versions(),
            &crate::layout::TargetLayout::default(),
        );
        for section in [
            "[versions]",
            "[layout]",
            "[entry]",
            "[sources]",
            "[types.struct_fields]",
            "[types.enum_variants]",
            "[types.drop_impls]",
            "[types.copy_types]",
            "[bodies]",
        ] {
            assert!(
                input.contains(section),
                "build key input is missing {section}:\n{input}"
            );
        }
    }
}
