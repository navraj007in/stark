//! C10-B reproduction tool for DEV-214 -- unbounded AST depth from a left-associative operator
//! chain, which aborts the process with a stack overflow.
//!
//! An EXAMPLE rather than a test on purpose: the failure is a stack overflow, which aborts the
//! whole process (SIGABRT). Running it inside the test binary would take every other test in that
//! binary down with it, so `c10b_robustness.rs` pins the SAFE boundary and points here for the
//! failing side.
//!
//! ```text
//! cargo run --manifest-path starkc/Cargo.toml --example c10b_repro -- 245   # completes
//! cargo run --manifest-path starkc/Cargo.toml --example c10b_repro -- 250   # aborts
//! ```
use starkc::analysis::{analyze_project, ProjectInput};
use starkc::options::LanguageOptions;
use starkc::source::SourceFile;
use std::sync::Arc;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(250);
    // `1 + 1 + 1 + ...`, n operands. Parses iteratively (16 precedence levels, one loop each), so
    // the parser's MAX_DEPTH guard never fires -- but the AST it builds is n deep.
    let src = format!("fn main() {{ let x = {}; }}", vec!["1"; n].join(" + "));
    let file = Arc::new(SourceFile::new("dev214.stark", src));
    let analysis = analyze_project(ProjectInput::program(file), LanguageOptions::CORE);
    println!(
        "completed at n={n}: {} diagnostics",
        analysis.diagnostics.len()
    );
}
