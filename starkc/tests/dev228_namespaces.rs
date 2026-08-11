//! **DEV-228: the resolver carries the namespaces NAME-RESOLVE-001 specifies.**
//!
//! `04-Semantic-Analysis.md` NAME-RESOLVE-001: *"Core has distinct module, type, value, and
//! associated-item namespaces… The same spelling may coexist in different namespaces, but two
//! declarations in one namespace and scope are duplicates."*
//!
//! `ModuleData` held **one** `HashMap<String, Res>` for all of them, so `struct Pair` alongside
//! `fn Pair()` was rejected as a duplicate although the rule permits it. That was the visible
//! symptom. The corrosive part was that every lookup then needed a PRECEDENCE over names that
//! were never meant to compete: DEV-223 and DEV-225 were both repaired by ordering one lookup
//! ahead of another, and a third such exception was the trajectory this replaces.
//!
//! Three module-level maps now exist — `modules`, `types`, `values` — and the associated-item
//! namespace was already separate, answered from `item_details` by `qualified_associated_name`.
//! Reads are directed by the position that asks: a type annotation searches types, an expression
//! searches values, a qualifier or an import searches any, per MOD-USE-001.
//!
//! # Why these negative controls
//!
//! Splitting a map and directing reads can lose names as easily as it can find the right ones. The
//! controls pin that a duplicate WITHIN one namespace is still a duplicate — the half of
//! NAME-RESOLVE-001 that must not be weakened — and that ordinary single-namespace programs, module
//! paths and imports all still resolve.

mod support;

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn errors(src: &str, tag: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file);
    let mut out: Vec<String> = rd
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect();
    let checked = typecheck::analyze(&hir);
    out.extend(
        checked
            .diagnostics
            .iter()
            .filter(|d| d.severity == starkc::diag::Severity::Error)
            .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message)),
    );
    out
}

fn run(src: &str, tag: &str) -> String {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    let errs: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errs.is_empty(), "{tag}: check: {errs:?}");
    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(outcome.result.is_ok(), "{tag}: run: {:?}", outcome.result);
    outcome.output
}

/// The reproducer the deviation was filed on.
#[test]
fn a_type_and_a_value_may_share_a_spelling() {
    let out = run(
        "\
struct Pair { a: Int64 }
fn Pair() -> Int64 { 5i64 }
fn main() {
    let p = Pair { a: 1i64 };
    println(p.a);
    println(Pair());
}
",
        "dev228_type_and_value",
    );
    assert!(
        out.contains('1') && out.contains('5'),
        "the type position must find the struct and the expression position the function: {out:?}"
    );
}

#[test]
fn a_module_and_a_value_may_share_a_spelling() {
    let out = run(
        "\
mod thing { pub fn f() -> Int64 { 1i64 } }
fn thing() -> Int64 { 2i64 }
fn main() { println(thing()); println(thing::f()); }
",
        "dev228_module_and_value",
    );
    assert!(out.contains('2') && out.contains('1'), "{out:?}");
}

#[test]
fn a_type_annotation_reaches_the_type_when_a_value_shares_its_name() {
    assert_eq!(
        errors(
            "\
struct Config { n: Int64 }
fn Config() -> Int64 { 7i64 }
fn take(c: Config) -> Int64 { c.n }
fn main() { println(take(Config { n: 3i64 })); }
",
            "dev228_annotation"
        ),
        Vec::<String>::new(),
        "`c: Config` is a type position and must not find the function"
    );
}

// -- controls: the half of the rule that must NOT be weakened ------------------------------------

#[test]
fn two_values_sharing_a_spelling_are_still_duplicates() {
    let errs = errors(
        "\
fn dup() -> Int64 { 1i64 }
fn dup() -> Int64 { 2i64 }
fn main() { println(dup()); }
",
        "dev228_control_two_values",
    );
    assert!(
        errs.iter().any(|e| e.contains("E0204")),
        "a duplicate WITHIN one namespace is still a duplicate: {errs:?}"
    );
}

#[test]
fn two_types_sharing_a_spelling_are_still_duplicates() {
    let errs = errors(
        "\
struct Dup { a: Int64 }
enum Dup { One }
fn main() { println(1i64); }
",
        "dev228_control_two_types",
    );
    assert!(
        errs.iter().any(|e| e.contains("E0204")),
        "a struct and an enum are both in the type namespace: {errs:?}"
    );
}

#[test]
fn ordinary_programs_with_no_collision_are_unaffected() {
    let out = run(
        "\
mod inner { pub enum Hue { Blue, Teal } pub fn pick() -> Hue { Hue::Teal } }
use inner::Hue;
struct Wrap { h: Hue }
fn main() {
    let w = Wrap { h: inner::pick() };
    match w.h { Hue::Blue => println(\"blue\"), Hue::Teal => println(\"teal\") }
}
",
        "dev228_control_ordinary",
    );
    assert!(
        out.contains("teal"),
        "module paths, imports, types and values with no collision must be unchanged: {out:?}"
    );
}
