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

/// DEV-156 — a doc comment on a struct FIELD stays on that field.
///
/// The formatter measured a flat rendering of the field list into a scratch buffer, so its item
/// printer was forbidden from consuming comments; field docs therefore survived only as unconsumed
/// trivia and were flushed AFTER the struct. A documented struct came back as a one-line struct
/// followed by its own orphaned field docs — the formatter silently detaching documentation from
/// what it documents.
#[test]
fn dev156_field_doc_comments_stay_on_their_fields() {
    let src = "\
/// A point.
struct Point {
    /// The x coordinate.
    x: Int32,
    /// The y coordinate.
    y: Int32,
}
";
    let formatted = fmt(src);
    assert_eq!(
        formatted, src,
        "a struct whose fields are documented is already canonical"
    );
}

/// DEV-156 — the ordering is the assertion, not merely the presence of the text.
///
/// The defect PRESERVED every comment; it moved them. A test that only checked the comments still
/// existed somewhere in the output would have passed against the broken formatter, so this pins
/// that each doc precedes its own field and both are inside the braces.
#[test]
fn dev156_field_docs_precede_their_own_field() {
    let formatted =
        fmt("struct S {\n    /// first\n    a: Int32,\n    /// second\n    b: Int32,\n}\n");
    let first = formatted.find("/// first").expect("first doc survives");
    let a = formatted.find("a: Int32").expect("field a survives");
    let second = formatted.find("/// second").expect("second doc survives");
    let b = formatted.find("b: Int32").expect("field b survives");
    let close = formatted.rfind('}').expect("the struct closes");
    assert!(
        first < a && a < second && second < b,
        "each doc must precede its own field, got:\n{formatted}"
    );
    assert!(
        b < close,
        "the fields and their docs must be INSIDE the braces, got:\n{formatted}"
    );
}

/// DEV-156 — an undocumented struct keeps the flat form.
///
/// The repair skips flat measurement only when a comment is present. Without this case, forcing
/// every struct broken would also satisfy the assertions above.
#[test]
fn dev156_an_undocumented_struct_is_still_flat() {
    assert_eq!(
        fmt("struct Small { x: Int32, y: Int32 }\n"),
        "struct Small { x: Int32, y: Int32 }\n"
    );
    assert_eq!(fmt("struct Empty {}\n"), "struct Empty {}\n");
}

/// DEV-156 — a comment after the last field belongs inside the braces, and formatting is stable.
#[test]
fn dev156_trailing_interior_comment_and_idempotence() {
    let src = "\
struct S {
    /// documented
    pub id: Int64,

    /// also documented
    label: String,
    // a note before the brace
}
";
    let once = fmt(src);
    assert!(
        once.contains("// a note before the brace"),
        "the trailing interior comment survives, got:\n{once}"
    );
    let note = once.find("// a note").expect("note survives");
    let close = once.rfind('}').expect("the struct closes");
    assert!(note < close, "it stays inside the braces, got:\n{once}");
    // The blank line is preserved because the field that follows it is DOCUMENTED: the rule runs
    // per emitted comment. A blank line before an UNDOCUMENTED field is still dropped, which is
    // `delimited_list`'s long-standing behaviour and not something DEV-156 changed in either
    // direction. Asserted as it actually is, rather than as it ideally would be.
    assert!(
        once.contains("\n\n    /// also documented"),
        "the blank line before a documented field is preserved, got:\n{once}"
    );
    assert_eq!(fmt(&once), once, "formatting is idempotent");
}
