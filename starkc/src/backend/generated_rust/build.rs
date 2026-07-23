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
    let target_triple = parse_rustc_field(&rustc_verbose, "host: ")
        .ok_or_else(|| BackendDiagnostic::Io("could not parse `host:` from rustc -vV".into()))?
        .to_string();
    let versions = version::build_versions(rustc_version, target_triple);

    // WP-C5.3e (CD-067): resolve the requested named contract BEFORE emitting. An unknown name
    // is rejected here rather than defaulted, because a layout answer is observable and
    // target-specific -- a silent fallback would report values for a target nobody asked about.
    let layout = crate::layout::contract_for(&options.target_contract)
        .map_err(|e| BackendDiagnostic::Unsupported(e.0))?;
    let source = emit_program::emit(program, &versions, &layout)?;
    let build_key = compute_build_key(program, &versions, &layout);
    let crate_dir = options.target_dir.join("debug").join(&build_key);
    let src_dir = crate_dir.join("src");
    std::fs::create_dir_all(&src_dir)
        .map_err(|e| BackendDiagnostic::Io(format!("creating {}: {e}", src_dir.display())))?;

    write_file(
        &crate_dir.join("Cargo.toml"),
        &generated_cargo_toml(&toolchain.runtime_crate),
    )?;
    write_file(&src_dir.join("main.rs"), &source.main_rs)?;
    write_file(
        &crate_dir.join("build.json"),
        &build_manifest_json(&versions, &build_key, &layout),
    )?;

    // §11.3 offline rule: `stark-runtime` is dependency-free, so `--offline` never needs a
    // registry index and proves no accidental network dependency crept in.
    let manifest_path = crate_dir.join("Cargo.toml");
    let cargo_args = vec![
        OsString::from("build"),
        OsString::from("--offline"),
        OsString::from("--manifest-path"),
        manifest_path.into_os_string(),
    ];
    let command: Vec<String> = std::iter::once(format!("RUSTC={}", toolchain.rustc.display()))
        .chain(std::iter::once(toolchain.cargo.display().to_string()))
        .chain(
            cargo_args
                .iter()
                .map(|arg| arg.to_string_lossy().into_owned()),
        )
        .collect();
    let output = Command::new(&toolchain.cargo)
        .args(&cargo_args)
        .env("RUSTC", &toolchain.rustc)
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

    let binary_path = crate_dir.join("target").join("debug").join(BIN_NAME);
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

fn generated_cargo_toml(runtime_path: &Path) -> String {
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
         stark-runtime = {{ path = {:?} }}\n\
         \n\
         [profile.dev]\n\
         panic = \"abort\"\n",
        runtime_path,
    )
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
    for (i, file) in program.files.iter().enumerate() {
        let mut content = Sha256::new();
        content.update(file.src.as_bytes());
        let digest = content.finalize();
        let hex: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        let _ = writeln!(out, "{i} {} {hex}", file.name);
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
fn build_manifest_json(
    versions: &BuildVersions,
    build_key: &str,
    layout: &crate::layout::TargetLayout,
) -> String {
    format!(
        "{{\n  \"build_key\": {},\n  \"compiler_version\": {},\n  \"mir_version\": {},\n  \
         \"mir_runtime_surface\": {},\n  \"runtime_version\": {},\n  \"backend_version\": {},\n  \
         \"rustc_version\": {},\n  \"target_triple\": {},\n  \"profile\": {},\n  \
         \"target_contract\": {},\n  \"layout_contract_version\": {},\n  \
         \"compiler_layout_revision\": {}\n}}\n",
        json_str(build_key),
        json_str(&versions.compiler_version),
        json_str(&versions.mir_version),
        json_str(&versions.mir_runtime_surface),
        json_str(&versions.runtime_version),
        json_str(&versions.backend_version),
        json_str(&versions.rustc_version),
        json_str(&versions.target_triple),
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

    /// One labelled, single-input mutation. Named aliases rather than inline `Box<dyn Fn>`
    /// signatures so the tests below read as what they are: a list of "change exactly this, then
    /// assert the key moved".
    type ProgramMutation = (&'static str, Box<dyn Fn(&mut MirProgram)>);
    type VersionMutation = (&'static str, Box<dyn Fn(&mut BuildVersions)>);

    fn versions() -> BuildVersions {
        version::build_versions("1.99.0".to_string(), "aarch64-apple-darwin".to_string())
    }

    fn program(name: &str, source: &str) -> MirProgram {
        let file = Arc::new(SourceFile::new(name, source.to_string()));
        let (ast, pd) = parse(&file, ParseMode::Program);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (hir, rd) = resolve(&ast, file.clone());
        assert!(rd.is_empty(), "resolve: {rd:?}");
        let checked = typecheck::analyze(&hir, file.clone());
        match crate::mir::lower::lower_program(&hir, &checked.tables, file) {
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
