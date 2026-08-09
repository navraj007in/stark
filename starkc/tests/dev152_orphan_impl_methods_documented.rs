//! **DEV-152: an `impl` whose type has no page-level item had its methods silently dropped.**
//!
//! `extract` collects `impl` members separately, then attaches them to the type's own doc item:
//!
//! ```text
//! for (type_name, methods) in impls {
//!     if let Some(owner) = items.iter_mut().find(|i| i.name == type_name && ..) {
//!         owner.members.extend(methods);
//!     }
//! }
//! ```
//!
//! With no `else`. If the type had no page-level item in this package, **every method vanished,
//! with no diagnostic** — not skipped with a warning, not rendered on a stub page, just gone.
//!
//! # Why this is not a rare corner
//!
//! A provider-bound resource nominal is SYNTHESIZED, not written (CD-234), so there is no
//! `pub struct TcpStream` in `stark-net`'s source for `impl TcpStream { .. }` to attach to. All
//! seven of its public methods — `connect`, `read`, `write`, `write_all`, `set_read_timeout`,
//! `set_write_timeout`, `shutdown_write` — were absent from the generated documentation.
//!
//! It compounded with DEV-151. Those same two timeout setters could not be BUILT at a call site,
//! and the reason nobody noticed is that nobody called them — and one reason nobody called them is
//! that the documentation did not say they existed. An undocumented API and an uncalled API are
//! the same failure viewed from two sides.
//!
//! It also blocked CD-355's surface gate, which asks "is every public callable called?" using
//! `stark doc` as the authority on what is public. Built on the old extractor, that gate would have
//! reported `stark-net` as fully covered while seven methods had never been called by anything.

use starkc::doc_gen::extract::{extract, ItemDocKind};
use starkc::lexer::tokenize_with_comments;
use starkc::parser::{parse, ParseMode};
use starkc::source::SourceFile;

fn extract_from(source: &str) -> Vec<starkc::doc_gen::extract::DocItem> {
    let file = SourceFile::new("t.stark", source.to_string());
    let (ast, diags) = parse(&file, ParseMode::Program);
    assert!(diags.is_empty(), "parse failed: {diags:?}");
    let doc_source = ast
        .sources
        .id_for_name(&file.name)
        .expect("the parse registered this file");
    let (_, comments, _) = tokenize_with_comments(&file, doc_source);
    extract(&ast, &file, &comments)
}

fn member_names(items: &[starkc::doc_gen::extract::DocItem], type_name: &str) -> Vec<String> {
    items
        .iter()
        .find(|item| item.name == type_name)
        .map(|item| item.members.iter().map(|m| m.name.clone()).collect())
        .unwrap_or_default()
}

/// **The reproducer.** An `impl` on a type this file does not declare — the shape a synthesized
/// resource nominal produces. Before the repair this yielded NO items at all.
#[test]
fn methods_survive_when_their_type_has_no_declaration() {
    let items = extract_from(
        "impl TcpStream {\n\
         \x20   /// Read into a caller buffer.\n\
         \x20   pub fn read(&mut self) -> Int32 { 0 }\n\
         \x20   /// Bound how long a read may block.\n\
         \x20   pub fn set_read_timeout(&mut self, seconds: UInt64) -> Int32 { 0 }\n\
         }\n",
    );

    let page = items
        .iter()
        .find(|item| item.name == "TcpStream")
        .expect("an impl with no declared type must still produce a page, not vanish");
    assert!(
        page.kind.is_page_level(),
        "the synthesized page must be page-level or the search index will skip it, which is \
         exactly how the methods disappeared"
    );
    assert_eq!(
        member_names(&items, "TcpStream"),
        vec!["read".to_string(), "set_read_timeout".to_string()]
    );
    assert_eq!(page.members[0].kind, ItemDocKind::Method);
    assert!(
        page.members[1].doc.contains("Bound how long"),
        "a method's own doc comment must survive the rehoming"
    );
}

/// The ordinary case is unchanged: a declared type still owns its methods directly, and no
/// duplicate page is synthesized alongside it.
#[test]
fn a_declared_type_still_owns_its_methods_and_gains_no_duplicate() {
    let items = extract_from(
        "/// A holder.\n\
         pub struct Holder { pub v: Int32 }\n\
         impl Holder {\n\
         \x20   pub fn get(&self) -> Int32 { self.v }\n\
         }\n",
    );
    assert_eq!(
        items.iter().filter(|item| item.name == "Holder").count(),
        1,
        "the repair must not synthesize a second page beside a declared type"
    );
    let holder = items.iter().find(|item| item.name == "Holder").unwrap();
    assert_eq!(holder.kind, ItemDocKind::Struct);
    assert!(holder.doc.contains("A holder."), "the real doc must win");
    assert!(member_names(&items, "Holder").contains(&"get".to_string()));
}

/// Several `impl` blocks on the same undeclared type each contribute a page-or-members. Whichever
/// way they merge, no method may be lost — losing one silently is the defect.
#[test]
fn multiple_impls_on_an_undeclared_type_lose_nothing() {
    let items = extract_from(
        "impl Stream {\n\
         \x20   pub fn a(&self) -> Int32 { 0 }\n\
         }\n\
         impl Stream {\n\
         \x20   pub fn b(&self) -> Int32 { 0 }\n\
         }\n",
    );
    let all: Vec<String> = items
        .iter()
        .filter(|item| item.name == "Stream")
        .flat_map(|item| item.members.iter().map(|m| m.name.clone()))
        .collect();
    assert!(
        all.contains(&"a".to_string()),
        "method `a` was lost: {all:?}"
    );
    assert!(
        all.contains(&"b".to_string()),
        "method `b` was lost: {all:?}"
    );
}

/// A private method stays out of the documentation. The repair rehomes methods; it must not
/// promote anything the extractor would otherwise have excluded.
#[test]
fn private_methods_are_still_excluded() {
    let items = extract_from(
        "impl Hidden {\n\
         \x20   fn secret(&self) -> Int32 { 0 }\n\
         \x20   pub fn shown(&self) -> Int32 { 0 }\n\
         }\n",
    );
    let names = member_names(&items, "Hidden");
    assert!(
        !names.contains(&"secret".to_string()),
        "a private method must not become public by being rehomed: {names:?}"
    );
    assert!(names.contains(&"shown".to_string()));
}
