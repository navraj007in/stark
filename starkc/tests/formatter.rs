//! WP8.2 formatter test suite: golden-file cases plus a corpus sweep over
//! every `.stark` fixture in the repo that parses cleanly as a standalone
//! `Program` (each file formatted independently of its package — see
//! `formatter::format_file`'s doc comment). Per file: format must succeed,
//! must be idempotent (`format(format(x)) == format(x)`), and the
//! formatted output must re-parse with the same top-level item count as
//! the original (a structural-preservation sanity check cheaper than full
//! AST equality).

use starkc::ast::{Ast, ItemKind, Root, UseTree};
use starkc::options::LanguageOptions;
use starkc::parser::{parse_with_options_into, ParseMode};
use starkc::source::SourceFile;

fn use_leaf_count(tree: &UseTree) -> usize {
    match tree {
        UseTree::Path { .. } | UseTree::Glob { .. } | UseTree::SelfImport { .. } => 1,
        UseTree::Group { items, .. } => items.iter().map(use_leaf_count).sum(),
    }
}

/// Parses `src` exactly as `formatter::format_file` does (this file's own
/// top-level items only, no recursion into `mod name;` submodule files —
/// `parse_with_options`'s `Program` mode uses `parse_project`, which *does*
/// walk submodules on disk, the wrong comparison here since a fixture's
/// `mod foo;` submodule need not exist as a sibling file), then returns a
/// structural item count: each non-`use` item counts once, and each `use`
/// item counts its flattened leaf paths, so intentionally flattening
/// `use a::{b, c};` into two `use` statements doesn't register as a
/// structural change.
fn item_count(src: &str, options: LanguageOptions) -> Option<usize> {
    let file = SourceFile::new("t.stark", src.to_string());
    let mut ast = Ast::default();
    let (root, diags) = parse_with_options_into(&file, ParseMode::Program, options, &mut ast);
    if diags
        .iter()
        .any(|d| d.severity == starkc::diag::Severity::Error)
    {
        return None;
    }
    match root {
        Root::Program(items) => Some(
            items
                .iter()
                .map(|&id| match &ast.item(id).kind {
                    ItemKind::Use(tree) => use_leaf_count(tree),
                    _ => 1,
                })
                .sum(),
        ),
        _ => None,
    }
}

fn collect_stark_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            // Skip pre-pivot archive content and build output; not Core v1.
            if matches!(name, "archive" | "target" | "node_modules" | ".git") {
                continue;
            }
            collect_stark_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("stark") {
            out.push(path);
        }
    }
}

/// Every `.stark` fixture under the repo root, excluding the pre-pivot
/// archive tree (`STARKLANG/compiler/`, `**/archive/`) which targets a
/// different, non-Core-v1 grammar per `CLAUDE.md`.
fn repo_stark_files() -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("starkc/ has a parent directory")
        .to_path_buf();
    let mut files = Vec::new();
    collect_stark_files(&root, &mut files);
    files
        .into_iter()
        .filter(|p| !p.components().any(|c| c.as_os_str() == "compiler"))
        .collect()
}

#[test]
fn corpus_sweep_format_is_idempotent_and_structure_preserving() {
    let files = repo_stark_files();
    assert!(
        files.len() > 50,
        "expected to find a substantial .stark corpus, found {}",
        files.len()
    );

    let mut formatted = 0usize;
    let mut skipped_parse_error = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        let file = SourceFile::new(path.display().to_string(), src.clone());

        let options = starkc::options::options_from_extension_flags(&["tensor".to_string()])
            .expect("tensor is a known extension");
        let once = match starkc::formatter::format_file(&file, options) {
            Ok(s) => s,
            Err(_) => {
                // Not every fixture in the repo is a standalone Core v1
                // Program parsed alone (some are intentionally
                // semantic-error fixtures, snippets, or extension-gated);
                // the formatter correctly refuses those. Only a clean
                // parse is in scope for this sweep.
                skipped_parse_error += 1;
                continue;
            }
        };
        formatted += 1;

        let Some(original_items) = item_count(&src, options) else {
            failures.push(format!("{}: original failed to parse??", path.display()));
            continue;
        };
        let Some(reformatted_items) = item_count(&once, options) else {
            failures.push(format!(
                "{}: formatted output does not re-parse",
                path.display()
            ));
            continue;
        };
        if original_items != reformatted_items {
            failures.push(format!(
                "{}: item count changed ({} -> {})",
                path.display(),
                original_items,
                reformatted_items
            ));
            continue;
        }

        let file2 = SourceFile::new(path.display().to_string(), once.clone());
        match starkc::formatter::format_file(&file2, options) {
            Ok(twice) if twice == once => {}
            Ok(twice) => failures.push(format!(
                "{}: not idempotent\n--- pass 1 ---\n{}\n--- pass 2 ---\n{}",
                path.display(),
                once,
                twice
            )),
            Err(diags) => failures.push(format!(
                "{}: formatted output failed to re-format: {:?}",
                path.display(),
                diags
            )),
        }
    }

    eprintln!(
        "formatter corpus sweep: {formatted} formatted, {skipped_parse_error} skipped (parse error), {} failed",
        failures.len()
    );

    assert!(
        formatted > 20,
        "expected a meaningful number of fixtures to parse cleanly, got {formatted}"
    );
    assert!(
        failures.is_empty(),
        "{} formatter corpus failures:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

// ------------------------------------------------------------- golden cases --

fn fmt(src: &str) -> String {
    let file = SourceFile::new("t.stark", src.to_string());
    starkc::formatter::format_file(&file, LanguageOptions::CORE).expect("should format")
}

#[test]
fn golden_struct_and_impl() {
    let src =
        "struct Point{x:Int32,y:Int32}\nimpl Point{fn new(x:Int32,y:Int32)->Point{Point{x,y}}}\n";
    let out = fmt(src);
    assert_eq!(
        out,
        "struct Point { x: Int32, y: Int32 }\n\
         impl Point {\n\
         \x20   fn new(x: Int32, y: Int32) -> Point {\n\
         \x20       Point { x, y }\n\
         \x20   }\n\
         }\n"
    );
}

#[test]
fn golden_enum_and_match() {
    let src = "enum Color{Red,Green,Blue}\nfn name(c:Color)->str{match c{Color::Red=>\"red\",Color::Green=>\"green\",Color::Blue=>\"blue\"}}\n";
    let out = fmt(src);
    assert!(out.starts_with("enum Color {\n    Red,\n    Green,\n    Blue,\n}\n"));
    assert!(out.contains("match c {"));
    assert!(out.contains("Color::Red => \"red\","));
}

#[test]
fn golden_long_call_breaks_to_multiline() {
    let src = "fn f() { some_function_with_a_rather_long_name(argument_number_one, argument_number_two, argument_number_three, argument_number_four); }\n";
    let out = fmt(src);
    assert!(out.contains("(\n"), "expected a broken call, got: {out}");
    assert!(out.contains("        argument_number_one,\n"), "got: {out}");
}

#[test]
fn golden_generic_fn_and_where_like_bounds() {
    let src = "fn max<T:Ord>(a:T,b:T)->T{if a>b{a}else{b}}\n";
    let out = fmt(src);
    assert_eq!(
        out,
        "fn max<T: Ord>(a: T, b: T) -> T {\n    if a > b {\n        a\n    } else {\n        b\n    }\n}\n"
    );
}

#[test]
fn idempotent_across_full_corpus_pass_matches_single_file_checks() {
    // Re-affirm at the golden-case granularity too, since the corpus sweep
    // only runs on repo fixtures.
    for src in [
        "struct P { x: Int32, y: Int32 }\n",
        "enum E { A, B(Int32), C { x: Int32 } }\n",
        "trait T { fn f(&self) -> Int32; }\n",
        "fn f<T: Ord>(x: T) -> T { x }\n",
        "use a::b::{c, d as e};\n",
    ] {
        let once = fmt(src);
        let twice = fmt(&once);
        assert_eq!(once, twice, "not idempotent for: {src}");
    }
}
