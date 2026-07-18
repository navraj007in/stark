//! `stark doc` — documentation generator (WP8.5).
//!
//! Extracts `///` doc comments from public items (`extract.rs`), renders
//! them as a minimal Markdown subset (`markdown.rs`) with STARK-lexer-based
//! syntax highlighting for code blocks (`highlight.rs`), and emits a
//! static HTML site plus a client-side search index (`html.rs`,
//! `search.rs`).

pub mod extract;
pub mod highlight;
pub mod html;
pub mod markdown;
pub mod search;

use extract::DocItem;

/// Write the HTML site and search index for an already-extracted (and,
/// for multi-file packages, already-merged) item list. Does not validate
/// doc examples — see [`validate_examples`], which needs the *source
/// file* an example's item came from (an example commonly calls the very
/// function it documents, per the plan's own
/// `/// assert_eq(add(2, 3), 5);` on `fn add`, so it must be compiled
/// with that file's other definitions in scope, not in isolation).
pub fn generate_from_items(
    items: &[DocItem],
    package_name: &str,
    output_dir: &std::path::Path,
) -> std::io::Result<usize> {
    html::write_site(items, package_name, output_dir)?;
    let index = search::build_index(items, package_name);
    search::write_index(&index, output_dir)?;
    Ok(count_items(items))
}

fn count_items(items: &[DocItem]) -> usize {
    items.iter().map(|i| 1 + count_items(&i.members)).sum()
}

/// Compile-check every example in `examples` (all doc comments found in
/// one source file) with `file_source` — the file they were extracted
/// from — prepended, so an example that calls the function/type it
/// documents (the common case) resolves correctly instead of failing on
/// an "undefined" error for a name that's only undefined in isolation.
/// Returns `(owning item name, error message)` for each example that
/// still fails to compile.
pub fn validate_examples(
    examples: &[extract::DocExample],
    file_source: &str,
) -> Vec<(String, String)> {
    examples
        .iter()
        .filter_map(|example| {
            validate_example_in_context(file_source, &example.code)
                .err()
                .map(|message| (example.owner.clone(), message))
        })
        .collect()
}

/// Parse+resolve+typecheck+**run** `file_source` with `example_code`
/// appended as the body of a synthetic `fn __doc_example__()`, matching
/// `cargo test --doc`-style doctest semantics rather than a compile-only
/// check: the plan's own reference example (`assert_eq(add(2, 3), 5);`)
/// is a *runtime* assertion, and `assert_eq(add(2, 2), 999)` typechecks
/// fine — it only fails when actually executed. `file_source` is assumed
/// to already be a clean, error-free program (the caller only extracts
/// examples from files that parsed successfully), so any error here
/// originates in the appended example.
fn validate_example_in_context(file_source: &str, example_code: &str) -> Result<(), String> {
    use crate::analysis::{analyze_project, ProjectInput};
    use crate::diag::Severity;
    use crate::hir::ItemKind;
    use crate::options::LanguageOptions;
    use crate::source::SourceFile as Src;
    use std::sync::Arc;

    let combined = format!("{file_source}\nfn __doc_example__() {{\n{example_code}\n}}\n");
    let file = Arc::new(Src::new("doc-example.stark", combined));
    let options = LanguageOptions::CORE;
    let analysis = analyze_project(ProjectInput::program(file.clone()), options);
    if let Some(d) = analysis
        .diagnostics
        .iter()
        .find(|d| d.severity == Severity::Error)
    {
        return Err(format!("analysis error: {}", d.message));
    }
    let hir = analysis.hir.as_ref().expect("successful analysis has HIR");
    let tables = analysis
        .type_tables
        .as_ref()
        .expect("successful analysis has type tables");

    let example_item = hir.items.iter().enumerate().find_map(|(index, item)| {
        let ItemKind::Fn(def) = &item.kind else {
            return None;
        };
        let name = &file.src[def.sig.name.lo as usize..def.sig.name.hi as usize];
        (name == "__doc_example__").then_some(crate::hir::ItemId(index as u32))
    });
    let Some(example_item) = example_item else {
        return Err(
            "internal error: synthetic example function not found after resolve".to_string(),
        );
    };

    crate::interp::run_item(hir, file, tables, example_item)
        .map(|_| ())
        .map_err(|e| format!("runtime error: {}", e.message))
}
