//! WP8.5 documentation generator test suite: extraction, markdown
//! rendering, syntax highlighting, HTML/search-index generation, and doc
//! example validation, exercised end to end.

use starkc::doc_gen::extract::{extract, ItemDocKind};
use starkc::doc_gen::{highlight, markdown, validate_examples};
use starkc::lexer::tokenize_with_comments;
use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;

fn extract_from(source: &str) -> Vec<starkc::doc_gen::extract::DocItem> {
    let file = SourceFile::new("t.stark", source.to_string());
    let (ast, diags) = parse(&file, ParseMode::Program);
    assert!(diags.is_empty(), "parse failed: {diags:?}");
    let (_, comments, _) = tokenize_with_comments(&file);
    extract(&ast, &file, &comments)
}

#[test]
fn extracts_doc_comment_and_signature_for_a_public_fn() {
    let items = extract_from(
        "/// Add two numbers.\n\
         ///\n\
         /// # Returns\n\
         /// The sum.\n\
         pub fn add(a: Int32, b: Int32) -> Int32 { a + b }\n",
    );
    assert_eq!(items.len(), 1);
    let item = &items[0];
    assert_eq!(item.name, "add");
    assert_eq!(item.kind, ItemDocKind::Fn);
    assert_eq!(item.signature, "pub fn add(a: Int32, b: Int32) -> Int32");
    assert!(item.doc.contains("Add two numbers."));
    assert!(item.doc.contains("# Returns"));
}

#[test]
fn private_items_are_not_extracted() {
    let items = extract_from("fn helper() -> Int32 { 0 }\n");
    assert!(items.is_empty());
}

#[test]
fn undocumented_public_items_are_still_extracted_with_empty_doc() {
    let items = extract_from("pub fn f() {}\n");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].doc, "");
}

#[test]
fn struct_signature_excludes_field_list() {
    let items = extract_from(
        "/// A point.\n\
         pub struct Point {\n    pub x: Int32,\n    pub y: Int32,\n}\n",
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].signature, "pub struct Point");
    let names: Vec<&str> = items[0].members.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, vec!["x", "y"]);
    assert_eq!(items[0].members[0].kind, ItemDocKind::Field);
}

#[test]
fn private_struct_fields_are_not_extracted() {
    let items = extract_from("pub struct Point {\n    pub x: Int32,\n    y: Int32,\n}\n");
    assert_eq!(items[0].members.len(), 1);
    assert_eq!(items[0].members[0].name, "x");
}

#[test]
fn enum_variant_signatures_include_tuple_types() {
    let items = extract_from("pub enum Shape {\n    Circle(Int32),\n    Point,\n}\n");
    assert_eq!(items[0].members[0].signature, "Circle(Int32)");
    assert_eq!(items[0].members[1].signature, "Point");
}

#[test]
fn impl_methods_merge_into_their_struct() {
    let items = extract_from(
        "pub struct Point { pub x: Int32 }\n\
         impl Point {\n    /// Origin point.\n    pub fn origin() -> Point { Point { x: 0 } }\n\
         fn private_helper() {}\n}\n",
    );
    assert_eq!(items.len(), 1, "impl block should not become its own page");
    let point = &items[0];
    let methods: Vec<&str> = point
        .members
        .iter()
        .filter(|m| m.kind == ItemDocKind::Method)
        .map(|m| m.name.as_str())
        .collect();
    assert_eq!(
        methods,
        vec!["origin"],
        "private_helper must not be documented"
    );
    let origin = point.members.iter().find(|m| m.name == "origin").unwrap();
    assert_eq!(origin.signature, "pub fn origin() -> Point");
    assert!(origin.doc.contains("Origin point."));
}

#[test]
fn nested_pub_mod_sets_module_path() {
    let items = extract_from("pub mod shapes {\n    pub struct Triangle {}\n}\n");
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].module_path, "shapes");
}

#[test]
fn private_mod_contents_are_not_extracted() {
    let items = extract_from("mod hidden {\n    pub fn f() {}\n}\n");
    assert!(items.is_empty());
}

#[test]
fn doc_comment_must_be_immediately_adjacent_not_separated_by_blank_line() {
    let items = extract_from("/// Orphaned comment.\n\npub fn f() {}\n");
    assert_eq!(
        items[0].doc, "",
        "a blank line should break the doc-comment run"
    );
}

#[test]
fn non_doc_line_comments_are_not_picked_up_as_docs() {
    let items = extract_from("// not a doc comment\npub fn f() {}\n");
    assert_eq!(items[0].doc, "");
}

#[test]
fn markdown_renders_headings_lists_and_inline_code() {
    let html = markdown::render("# Title\n\nSome `code` here.\n\n* one\n* two\n");
    assert!(html.contains("<h1>Title</h1>"));
    assert!(html.contains("<code>code</code>"));
    assert!(html.contains("<li>one</li>"));
    assert!(html.contains("<li>two</li>"));
}

#[test]
fn markdown_fenced_stark_block_is_syntax_highlighted() {
    let html = markdown::render("```stark\nfn f() {}\n```\n");
    assert!(
        html.contains("tok-kw"),
        "expected highlighted keyword span, got: {html}"
    );
}

#[test]
fn markdown_fenced_non_stark_block_is_escaped_not_highlighted() {
    let html = markdown::render("```text\n<script>\n```\n");
    assert!(!html.contains("tok-kw"));
    assert!(html.contains("&lt;script&gt;"));
}

#[test]
fn highlight_escapes_html_special_characters() {
    let html = highlight::highlight("let x = 1 < 2;");
    assert!(html.contains("&lt;"));
    assert!(!html.contains("1 < 2"));
}

#[test]
fn highlight_classifies_keywords_types_and_strings() {
    let html = highlight::highlight("pub fn f(x: Int32) -> String { \"hi\" }");
    assert!(html.contains("tok-kw"));
    assert!(html.contains("tok-type"));
    assert!(html.contains("tok-str"));
}

#[test]
fn validate_examples_passes_when_example_calls_documented_function() {
    let source = "pub fn add(a: Int32, b: Int32) -> Int32 { a + b }\n";
    let items = extract_from(
        "/// ```stark\n/// assert_eq(add(2, 3), 5);\n/// ```\n\
         pub fn add(a: Int32, b: Int32) -> Int32 { a + b }\n",
    );
    let examples = starkc::doc_gen::extract::collect_examples(&items);
    assert_eq!(examples.len(), 1);
    let failures = validate_examples(&examples, source);
    assert!(failures.is_empty(), "unexpected failures: {failures:?}");
}

#[test]
fn validate_examples_catches_a_runtime_assertion_failure() {
    // Compiles fine (assert_eq is generically typed); must fail because
    // 2 + 2 != 999 at *runtime* — not just a compile-time check.
    let source = "pub fn add(a: Int32, b: Int32) -> Int32 { a + b }\n";
    let items = extract_from(
        "/// ```stark\n/// assert_eq(add(2, 2), 999);\n/// ```\n\
         pub fn add(a: Int32, b: Int32) -> Int32 { a + b }\n",
    );
    let examples = starkc::doc_gen::extract::collect_examples(&items);
    let failures = validate_examples(&examples, source);
    assert_eq!(failures.len(), 1);
    assert!(
        failures[0].1.contains("runtime error"),
        "got: {:?}",
        failures[0]
    );
}

#[test]
fn validate_examples_catches_an_undefined_reference() {
    let source = "pub fn f() {}\n";
    let items = extract_from(
        "/// ```stark\n/// this_function_does_not_exist();\n/// ```\n\
         pub fn f() {}\n",
    );
    let examples = starkc::doc_gen::extract::collect_examples(&items);
    let failures = validate_examples(&examples, source);
    assert_eq!(failures.len(), 1);
}

#[test]
fn generate_from_items_writes_a_valid_site_to_disk() {
    let dir = std::env::temp_dir().join(format!("stark-doc-gen-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let items = extract_from(
        "/// Add two numbers.\npub fn add(a: Int32, b: Int32) -> Int32 { a + b }\n\
         pub struct Point { pub x: Int32 }\n",
    );
    let count = starkc::doc_gen::generate_from_items(&items, "mypkg", &dir)
        .expect("site generation should succeed");
    // `add` (no members) + `Point` + `Point::x` = 3; count_items counts
    // every item and member recursively, not just page-level items.
    assert_eq!(count, 3);

    assert!(dir.join("index.html").exists());
    assert!(dir.join("search.html").exists());
    assert!(dir.join("search.json").exists());
    assert!(dir.join("style.css").exists());
    assert!(dir.join("add/index.html").exists());
    assert!(dir.join("Point/index.html").exists());

    let index_html = std::fs::read_to_string(dir.join("index.html")).unwrap();
    assert!(index_html.starts_with("<!doctype html>"));
    assert!(index_html.contains("add/index.html"));
    assert!(index_html.contains("Point/index.html"));

    let search_json = std::fs::read_to_string(dir.join("search.json")).unwrap();
    assert!(search_json.contains("\"name\":\"add\""));
    assert!(search_json.contains("\"name\":\"Point\""));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn generated_page_links_are_relative_and_depth_correct_for_nested_modules() {
    let dir =
        std::env::temp_dir().join(format!("stark-doc-gen-test-nested-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let items = extract_from("pub mod shapes {\n    pub struct Triangle {}\n}\n");
    starkc::doc_gen::generate_from_items(&items, "mypkg", &dir).unwrap();

    let page = std::fs::read_to_string(dir.join("shapes/Triangle/index.html")).unwrap();
    assert!(page.contains("href=\"../../style.css\""), "got: {page}");
    assert!(page.contains("href=\"../../index.html\""));

    let _ = std::fs::remove_dir_all(&dir);
}
