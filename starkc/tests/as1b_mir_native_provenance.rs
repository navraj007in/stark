//! AS1b-ii-e / iii — the invariant at the MIR and native boundaries.
//!
//! `as1b_span_provenance.rs` covers the compile-time and HIR-runtime paths: a span belonging to a
//! dependency resolves against that dependency. These are the two boundaries below it — the MIR
//! engine's trap location, and the location a *generated native binary* prints when it aborts,
//! which is baked in at compile time and is the last thing a user sees.
//!
//! **What changed under these tests.** When ii-e wrote them, `SourceInfo` named a source twice:
//! a MIR-local `FileId` into `MirProgram::files`, and the `SourceId` on the span. Everything
//! downstream read the `FileId`, and nothing made the two agree, so a fourth test swept every
//! `SourceInfo` in a lowered program asserting they did — a compensating control for a
//! representable disagreement.
//!
//! AS1b-iii removed the duplication: `SourceInfo` is `{ span, origin }` and `MirProgram` carries
//! the source registry. **The sweep is deleted, not weakened** — it existed because two identities
//! could disagree, and there is now one. What replaces it is V-SRC-1, which the verifier checks on
//! every `SourceInfo` of every program: the span names a source the program can resolve. A
//! verifier proving a claim the lowerer cannot satisfy by construction is worth more than a test
//! comparing lowering against itself.
//!
//! These four remain because they are behavioural: they pin the answer a user is given, which no
//! amount of internal agreement establishes.

use starkc::backend::generated_rust::{emit_native_debug, NativeBuildOptions};
use starkc::mir::{MirProgram, Origin, SourceInfo, Terminator};
use starkc::options::LanguageOptions;
use starkc::package::PackageGraph;
use starkc::session::CompilerSession;
use starkc::source::SourceTable;

/// 8 lines of padding, so the dependency's interesting line sits past the end of the 3-line app.
/// A span resolved against the wrong file would have to clamp — CD-306's exact shape.
const PAD: &str = "// 1\n// 2\n// 3\n// 4\n// 5\n// 6\n// 7\n// 8\n";

fn unique_base(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let base = std::env::temp_dir().join(format!("as1be_{tag}_{}_{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// `app` depending on a sibling `lib`.
fn stage(base: &std::path::Path, lib_body: &str, app_body: &str) -> std::path::PathBuf {
    let lib = base.join("lib");
    std::fs::create_dir_all(lib.join("src")).unwrap();
    std::fs::write(
        lib.join("starkpkg.json"),
        r#"{"name":"lib","version":"0.1.0","entry":"src/lib.stark","dependencies":{}}"#,
    )
    .unwrap();
    std::fs::write(lib.join("src").join("lib.stark"), lib_body).unwrap();

    let app = base.join("app");
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::write(
        app.join("starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","dependencies":{"lib":{"package":"lib","path":"../lib","version":"0.1.0"}}}"#,
    )
    .unwrap();
    std::fs::write(app.join("src").join("main.stark"), app_body).unwrap();
    app
}

fn session(app: &std::path::Path) -> CompilerSession {
    let graph =
        PackageGraph::load_from_root_with_modes(&app.join("starkpkg.json"), false, true).unwrap();
    CompilerSession::for_package(graph, LanguageOptions::CORE)
}

/// Every `SourceInfo` in `program`, with the body symbol and a tag for the failure message.
///
/// There are three places one hides: on a statement, on a terminator, and **inside a terminator's
/// `TrapInfo`** — and the third is the one users actually see, because it is the location baked
/// into the abort call. A sweep that walked only the first two would report agreement while
/// checking nothing that reaches a trap message.
fn every_source_info(program: &MirProgram) -> Vec<(String, SourceInfo)> {
    let mut out = Vec::new();
    for body in &program.bodies {
        let symbol = body.instance.symbol.clone();
        for block in &body.blocks {
            for (_statement, source) in &block.statements {
                out.push((format!("{symbol} (statement)"), *source));
            }
            out.push((format!("{symbol} (terminator)"), block.terminator.1));
            match &block.terminator.0 {
                Terminator::Checked { trap, .. } => {
                    out.push((format!("{symbol} (checked trap)"), trap.source))
                }
                Terminator::Trap { info, .. } => {
                    out.push((format!("{symbol} (trap)"), info.source))
                }
                _ => {}
            }
        }
    }
    out
}

/// The `TrapInfo` a terminator carries, if it is a trap site.
fn trap_source(terminator: &Terminator) -> Option<SourceInfo> {
    match terminator {
        Terminator::Checked { trap, .. } => Some(trap.source),
        Terminator::Trap { info, .. } => Some(info.source),
        _ => None,
    }
}

/// Every `SourceInfo` in a real cross-package program resolves — the V-SRC-1 claim, checked from
/// outside the verifier and against a program the verifier has not been asked about.
///
/// This is what is left of ii-e's agreement sweep. That version compared MIR's `FileId` against the
/// span's `SourceId` and required them to name the same file; AS1b-iii deleted the `FileId`, so the
/// comparison has no second term. The remaining question — does the one identity resolve — is worth
/// asking here because lowering, not the verifier, is what could get it wrong.
#[test]
fn every_mir_source_info_resolves_in_the_programs_own_registry() {
    let base = unique_base("resolve");
    let app = stage(
        &base,
        &format!("{PAD}pub fn boom(a: Int32) -> Int32 {{\n    a / 0\n}}\npub fn fine(a: Int32) -> Int32 {{\n    a + 1\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::fine(7);\n}\n",
    );

    let program = session(&app)
        .check()
        .unwrap_or_else(|failure| panic!("the fixture must compile:\n{}", failure.render()));
    let mir = program
        .lower_mir()
        .unwrap_or_else(|error| panic!("the fixture must lower: {}", error.what));
    let sources: &SourceTable = &mir.sources;

    let infos = every_source_info(&mir);
    let mut named = std::collections::BTreeSet::new();
    for (tag, info) in &infos {
        let source = sources.get(info.span.source).unwrap_or_else(|| {
            panic!(
                "{tag}: span names source {:?}, which the program's registry cannot resolve",
                info.span.source
            )
        });
        named.insert(source.name.clone());
    }

    // Non-vacuity, stated as facts about the fixture rather than a site count: the sweep really
    // did cross a package boundary, and really did see a trap site — the case that matters.
    assert!(
        named.len() >= 2,
        "the fixture must lower code from more than one file, got {named:?}"
    );
    assert!(
        named.iter().any(|name| name.starts_with("lib/")),
        "the dependency's own file must appear among the lowered bodies: {named:?}"
    );
    assert!(
        infos.iter().any(|(tag, _)| tag.contains("trap")),
        "the sweep must include at least one trap site: {:?}",
        infos.iter().map(|(tag, _)| tag).collect::<Vec<_>>()
    );

    // The program's registry IS the compilation's registry — not a copy that could drift.
    assert_eq!(
        mir.sources.len(),
        program.sources().len(),
        "MIR must carry the compilation's own sources"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// The MIR engine's own trap location, for a fault inside a dependency.
///
/// The line must be the dependency's own — past the end of the 3-line app, so a span resolved
/// against the app would have to clamp to something ≤ 3.
#[test]
fn a_mir_trap_inside_a_dependency_reports_that_dependency() {
    let base = unique_base("mirtrap");
    let app = stage(
        &base,
        &format!("{PAD}pub fn boom(a: Int32) -> Int32 {{\n    a / 0\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::boom(7);\n}\n",
    );

    let program = session(&app)
        .check()
        .unwrap_or_else(|failure| panic!("the fixture must compile:\n{}", failure.render()));
    let mir = program
        .lower_mir()
        .unwrap_or_else(|error| panic!("the fixture must lower: {}", error.what));
    let verified = starkc::mir::verify::verify_program(&mir)
        .unwrap_or_else(|errors| panic!("lowered MIR must verify: {errors:#?}"));

    let failure = starkc::mir::interp::run_program(verified)
        .err()
        .expect("dividing by zero must trap");
    let starkc::mir::interp::MirRunError::Trap { source, .. } = failure.error else {
        panic!("expected a trap, got {:?}", failure.error);
    };
    assert!(
        matches!(source.origin, Origin::UserCode),
        "the trap must come from user code, got {:?}",
        source.origin
    );

    let by_span = mir
        .sources
        .get(source.span.source)
        .expect("the trap's span must name a registered source");
    assert_eq!(
        by_span.name, "lib/src/lib.stark",
        "the trap belongs to the dependency, not the root"
    );
    let (line, _column) = by_span.line_col(source.span.lo);
    assert_eq!(
        line, 10,
        "the dependency's own line number, not one clamped to the 3-line root"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// The native boundary: the location a generated binary prints when it aborts.
///
/// This is the end of the chain — `resolve_source_location` bakes `(file, line, column)` into the
/// abort call at COMPILE time, so if the invariant is lost anywhere between lowering and emission,
/// it is lost in a string the user reads at runtime with no way to check it.
#[test]
fn a_native_trap_inside_a_dependency_reports_that_dependency() {
    let base = unique_base("nativetrap");
    let app = stage(
        &base,
        &format!("{PAD}pub fn boom(a: Int32) -> Int32 {{\n    a / 0\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::boom(7);\n}\n",
    );

    let program = session(&app)
        .check()
        .unwrap_or_else(|failure| panic!("the fixture must compile:\n{}", failure.render()));
    let mir = program
        .lower_mir()
        .unwrap_or_else(|error| panic!("the fixture must lower: {}", error.what));
    let verified = starkc::mir::verify::verify_program(&mir)
        .unwrap_or_else(|errors| panic!("lowered MIR must verify: {errors:#?}"));

    let artifact = emit_native_debug(
        &verified,
        &NativeBuildOptions {
            target_dir: base.join("out"),
            target_contract: "stark-64-v1".to_string(),
            ..NativeBuildOptions::default()
        },
    )
    .expect("the fixture must emit and build natively");

    let run = std::process::Command::new(&artifact.binary_path)
        .output()
        .expect("running the generated binary failed");
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "dividing by zero must abort; stdout: {}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        stderr.contains("lib/src/lib.stark:10"),
        "the native abort must name the dependency's own file and line, got:\n{stderr}"
    );
    assert!(
        !stderr.contains("app/src/main.stark"),
        "the native abort must not attribute the dependency's fault to the root:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&base);
}

/// A terminator is where a trap is raised, so its `SourceInfo` is the one users see. Kept separate
/// from the whole-program sweep so a regression names the narrower fact.
#[test]
fn every_trap_site_names_a_resolvable_source() {
    let base = unique_base("terminators");
    let app = stage(
        &base,
        &format!("{PAD}pub fn boom(a: Int32, b: Int32) -> Int32 {{\n    a / b\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::boom(7, 1);\n}\n",
    );

    let program = session(&app)
        .check()
        .unwrap_or_else(|failure| panic!("the fixture must compile:\n{}", failure.render()));
    let mir = program
        .lower_mir()
        .unwrap_or_else(|error| panic!("the fixture must lower: {}", error.what));

    let mut checked = 0usize;
    for body in &mir.bodies {
        for block in &body.blocks {
            let Some(info) = trap_source(&block.terminator.0) else {
                continue;
            };
            checked += 1;
            let by_span = mir.sources.get(info.span.source).unwrap_or_else(|| {
                panic!(
                    "trap site in {}: span names an unregistered source",
                    body.instance.symbol
                )
            });
            // The location a user is shown, resolved the way the backend resolves it.
            let (line, _column) = by_span.line_col(info.span.lo);
            assert!(
                by_span.name != "app/src/main.stark" || line <= 3,
                "a location in the 3-line app cannot be line {line} — a clamped read"
            );
        }
    }
    assert!(
        checked > 0,
        "the fixture must produce at least one trap site, or this proves nothing"
    );

    let _ = std::fs::remove_dir_all(&base);
}
