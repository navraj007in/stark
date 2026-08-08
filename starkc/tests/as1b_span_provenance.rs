//! AS1b acceptance criterion 3 — a span belonging to a dependency resolves against **that
//! dependency**, in the compile-time and runtime paths.
//!
//! `WP-SPAN-SOURCEID.md` §6 warns that the `SourceId` change is easy to make *compile* while
//! threading a plausible-but-wrong id at some sites, reproducing DEV-122 with better types. So the
//! criterion is a behavioural test on a real dependency, not a type-level argument — and it is
//! committed **before** the refactor, so it is a regression guard rather than a post-hoc claim.
//!
//! These pass today. That is the point: DEV-122's two recorded instances (CD-302, CD-306) are
//! fixed, and this pins them fixed while the span representation changes underneath.
//!
//! What this does **not** prove: CD-309's guard only suppresses spans whose offsets fall out of
//! range for the file they are measured against. A wrong-source span whose offsets are
//! coincidentally in range still renders a confident wrong location, and cannot be produced from
//! ordinary source input. Eliminating that is what AS1b is for; no test here can demonstrate it.

use starkc::options::LanguageOptions;
use starkc::package::PackageGraph;
use starkc::session::CompilerSession;

/// `app` depending on a sibling `lib`. `lib` is deliberately PADDED so its interesting line number
/// lies past the end of `app` — a span resolved against the wrong file would have to clamp, which
/// is exactly the shape CD-306 observed (line 31 of a 21-line file).
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

fn unique_base(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let base = std::env::temp_dir().join(format!("as1b_{tag}_{}_{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn session(app: &std::path::Path) -> CompilerSession {
    let graph =
        PackageGraph::load_from_root_with_modes(&app.join("starkpkg.json"), false, true).unwrap();
    CompilerSession::for_package(graph, LanguageOptions::CORE)
}

/// 8 lines of padding, so the item of interest sits at line 9-10 — past the end of every `app`
/// fixture here, which are 3 lines.
const PAD: &str = "// 1\n// 2\n// 3\n// 4\n// 5\n// 6\n// 7\n// 8\n";

#[test]
fn a_runtime_trap_inside_a_dependency_reports_that_dependency() {
    let base = unique_base("trap");
    let app = stage(
        &base,
        &format!("{PAD}pub fn boom(a: Int32) -> Int32 {{\n    a / 0\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::boom(7);\n}\n",
    );

    let program = session(&app).check().unwrap_or_else(|f| {
        panic!("the fixture must compile:\n{}", f.render());
    });
    let error = program
        .execute_hir()
        .expect_err("dividing by zero must trap");

    // AS1b-ii-d: resolved through the program's own registry, from the span alone. There is no
    // file carried on the error to fall back to — if the span named the wrong source, this fails.
    let file = program
        .sources()
        .get(error.span.source)
        .expect("a trap's span must name a registered source");
    assert_eq!(
        file.name, "lib/src/lib.stark",
        "the trap belongs to the dependency, not the root"
    );

    // The line must be the dependency's own — 10, which is past the end of `app`'s 3 lines. A span
    // resolved against the root would clamp to the root's end and report something ≤ 3.
    let (line, _col) = file.line_col(error.span.lo);
    assert_eq!(
        line, 10,
        "the dependency's own line number, not a clamped one"
    );

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_trap_inside_a_cross_package_generic_reports_the_generic_s_own_file() {
    // Generics instantiated across a package boundary are where provenance has drifted before
    // (DEV-101 was a cross-package generic provenance defect), so this is the harder case.
    let base = unique_base("generic");
    let app = stage(
        &base,
        &format!("{PAD}pub fn div<T: Num>(a: T, b: T) -> T {{\n    a / b\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::div(7, 0);\n}\n",
    );

    let program = session(&app).check().unwrap_or_else(|f| {
        panic!("the fixture must compile:\n{}", f.render());
    });
    let error = program
        .execute_hir()
        .expect_err("dividing by zero must trap");

    let file = program
        .sources()
        .get(error.span.source)
        .expect("a trap's span must name a registered source");
    assert_eq!(
        file.name, "lib/src/lib.stark",
        "the trap belongs to the generic's defining package, not the instantiating one"
    );
    let (line, _) = file.line_col(error.span.lo);
    assert_eq!(line, 10);

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn a_compile_error_inside_a_dependency_reports_that_dependency() {
    let base = unique_base("diag");
    let app = stage(
        &base,
        &format!("{PAD}pub fn seven() -> Int32 {{\n    undefined_name()\n}}\n"),
        "fn main() {\n    let v: Int32 = lib::seven();\n}\n",
    );

    let failure = session(&app)
        .check()
        .err()
        .expect("an undefined name in the dependency must fail the check");

    let errors: Vec<_> = failure
        .diagnostics()
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert_eq!(errors.len(), 1, "exactly one error: {errors:?}");

    let file = failure
        .sources()
        .get(errors[0].span.source)
        .expect("a diagnostic's span must name a registered source");
    assert_eq!(file.name, "lib/src/lib.stark");
    let (line, _) = file.line_col(errors[0].span.lo);
    assert_eq!(
        line, 10,
        "the dependency's own line, past the end of the 3-line root"
    );

    // The rendered text must agree with the structured fields — this is the surface a human reads,
    // and CD-306 was visible there before it was visible anywhere else.
    let rendered = failure.render();
    assert!(
        rendered.contains("lib/src/lib.stark:10"),
        "rendered diagnostic names the dependency and line: {rendered}"
    );

    let _ = std::fs::remove_dir_all(&base);
}
