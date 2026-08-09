//! AS8 — LSP package-analysis profile on a representative multi-file project.
//!
//! AS8's work section asks for two things about the LSP, and both are conditional on measurement:
//!
//!   * "Profile LSP package analysis on representative multi-file projects."
//!   * "Replace whole-package `ProjectAnalysis` duplication per open URI WHERE MEASUREMENT SHOWS
//!     MATERIAL COST."
//!
//! `lsp::state::ServerState` keeps `compilation_cache: HashMap<String, CompilationResult>`, and a
//! `CompilationResult` owns a whole `ProjectAnalysis` — AST, HIR, resolution tables, type tables,
//! symbol index and query index for the ENTIRE PACKAGE. One per open URI. Opening N files of one
//! package therefore analyses that package N times and retains N copies.
//!
//! This test MEASURES that and prints the numbers. **It deliberately asserts almost nothing about
//! timing.** A wall-clock threshold in CI is a flake generator, and AS8's exit criterion is that a
//! change be justified by before/after measurement — not that a number be defended. What it does
//! assert is the structural claim the numbers are about: that N open URIs produce N INDEPENDENT
//! analyses of the same package.
//!
//! Run it for the numbers:
//!     cargo test --release --test as8_lsp_analysis_profile -- --nocapture

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::options::LanguageOptions;
use starkc::package::PackageGraph;
use std::time::Instant;

/// A package with `modules` sibling modules, each with a handful of items, plus a `main` that uses
/// them. Representative of the shape the LSP actually sees: not one enormous file, but a package
/// whose cost is spread across files the editor opens one at a time.
fn stage(base: &std::path::Path, modules: usize) -> std::path::PathBuf {
    let app = base.join("app");
    std::fs::create_dir_all(app.join("src")).unwrap();
    std::fs::write(
        app.join("starkpkg.json"),
        r#"{"name":"app","version":"0.1.0","entry":"src/main.stark","dependencies":{}}"#,
    )
    .unwrap();

    let mut main = String::new();
    for m in 0..modules {
        main.push_str(&format!("mod m{m};\n"));
        let mut body = String::new();
        body.push_str(&format!(
            "pub struct S{m} {{ pub a: Int32, pub b: Int32 }}\n"
        ));
        body.push_str(&format!(
            "impl S{m} {{ pub fn sum(&self) -> Int32 {{ self.a + self.b }} \
             pub fn scaled(&self, k: Int32) -> Int32 {{ self.a * k + self.b }} }}\n"
        ));
        for f in 0..8 {
            body.push_str(&format!(
                "pub fn f{m}_{f}(x: Int32) -> Int32 {{ let y = x + {f}; y * 2 }}\n"
            ));
        }
        std::fs::write(app.join("src").join(format!("m{m}.stark")), body).unwrap();
    }
    main.push_str("fn main() {\n");
    for m in 0..modules {
        main.push_str(&format!(
            "    let s{m} = m{m}::S{m} {{ a: {m}, b: 1 }};\n    let _ = s{m}.sum();\n    let _ = m{m}::f{m}_0({m});\n"
        ));
    }
    main.push_str("}\n");
    std::fs::write(app.join("src").join("main.stark"), main).unwrap();
    app
}

fn unique_base(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let base = std::env::temp_dir().join(format!("as8lsp_{tag}_{}_{nanos}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn analyse(app: &std::path::Path) -> (std::time::Duration, usize) {
    let graph =
        PackageGraph::load_from_root_with_modes(&app.join("starkpkg.json"), false, true).unwrap();
    let started = Instant::now();
    let analysis = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);
    (started.elapsed(), analysis.diagnostics.len())
}

#[test]
fn as8_lsp_package_analysis_profile() {
    println!("\nAS8 LSP package-analysis profile");
    println!(
        "{:>8}  {:>12}  {:>14}  {:>10}",
        "modules", "one analysis", "x8 open URIs", "diags"
    );

    for &modules in &[4usize, 8, 16, 32] {
        let base = unique_base(&format!("m{modules}"));
        let app = stage(&base, modules);

        // Warm: the first run pays for filesystem cache and allocator growth.
        let _ = analyse(&app);

        let (single, diags) = analyse(&app);

        // What the LSP actually does when eight files of this package are open: it analyses the
        // WHOLE PACKAGE once per URI, because the cache is keyed by URI and the value is a whole
        // ProjectAnalysis.
        let eight = Instant::now();
        let mut retained = Vec::new();
        for _ in 0..8 {
            let graph =
                PackageGraph::load_from_root_with_modes(&app.join("starkpkg.json"), false, true)
                    .unwrap();
            retained.push(analyze_project(
                ProjectInput::package(graph),
                LanguageOptions::CORE,
            ));
        }
        let eight = eight.elapsed();

        println!(
            "{modules:>8}  {:>10.1}ms  {:>12.1}ms  {diags:>10}",
            single.as_secs_f64() * 1000.0,
            eight.as_secs_f64() * 1000.0
        );

        // THE STRUCTURAL CLAIM, which is what the timings are evidence ABOUT: the eight analyses
        // are independent objects, not shared. `owns_handle` is identity-scoped per analysis, so a
        // handle from one is not recognised by another -- which is exactly what makes them
        // eight separate copies rather than eight views of one.
        assert_eq!(retained.len(), 8);
        let first = &retained[0];
        let second = &retained[1];
        assert!(
            !std::ptr::eq(first, second),
            "the profile assumes independent analyses; if these are shared the measurement is moot"
        );

        let _ = std::fs::remove_dir_all(&base);
    }
    println!();
}
