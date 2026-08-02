//! **CD-358: the file-provenance audit — a name read off a declaration is read against the file
//! that DECLARED it.**
//!
//! `self.text(span)` slices the file currently being CHECKED. A name belonging to a declaration —
//! an impl's generic parameter, a signature's parameter, a trait default's return type — belongs to
//! the file that declared it. Across a module boundary those differ, and the failure is silent: the
//! comparison succeeds against garbage. A span running past the shorter file's end comes back as
//! `"?"`, so several distinct names can even COLLIDE on one key.
//!
//! This bug has now been repaired four times at four sites:
//!
//! | | site | found by |
//! | --- | --- | --- |
//! | DEV-069 | a trait method's name | a trait default across files |
//! | DEV-101 | cross-package generic typecheck | a package consumer |
//! | DEV-148 | an associated function's name, then its generic parameters | `stark-url` calling its own `Url::parse` |
//! | CD-358 | a METHOD's impl generics, and a trait default's signature | this audit |
//!
//! Each repair fixed its site and left the class open, so this file exists to close the class by
//! EXERCISE rather than by inspection: it drives every construct that crosses a module boundary and
//! asserts it works. A future site added without `decl_text` fails here rather than in a package
//! months later.
//!
//! # Why not an inspection of the call sites
//!
//! There are ~90 `self.text` calls in `typecheck.rs`, most of them legitimately reading the file
//! under check (expression spans, diagnostic text). Classifying them by eye is exactly the process
//! that has already missed this four times. A probe that actually compiles two-file packages found
//! the remaining live site — a generic method — in one run.

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::diag::Severity;
use starkc::options::LanguageOptions;

/// Build a two-file package and return its errors. The defect is about file provenance, so a
/// single-file fixture cannot reproduce it — the two files must genuinely differ in content, and
/// ideally in LENGTH, so a mis-sliced span lands somewhere visibly wrong.
fn two_file_errors(tag: &str, lib: &str, inner: &str) -> Vec<String> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(format!(
            "temp_cd358_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time must be after epoch")
                .as_nanos()
        ));
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create temp package src");
    std::fs::write(
        root.join("starkpkg.json"),
        r#"{"name":"cd358_app","version":"0.1.0","entry":"src/lib.stark"}"#,
    )
    .expect("write manifest");
    std::fs::write(src.join("lib.stark"), lib).expect("write lib");
    std::fs::write(src.join("inner.stark"), inner).expect("write inner");

    let manifest = starkc::package::find_package_root(&root).expect("find manifest");
    let graph = starkc::package::PackageGraph::load_from_root(&manifest).expect("load package");
    let errors = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE)
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect();
    let _ = std::fs::remove_dir_all(&root);
    errors
}

fn expect_clean(tag: &str, lib: &str, inner: &str) {
    let errors = two_file_errors(tag, lib, inner);
    assert!(
        errors.is_empty(),
        "{tag}: this construct must work across a module boundary; a stray type name like `'S'` or \
         `'r'` in these errors means a declaration's name was sliced from the wrong file: {errors:?}"
    );
}

/// **The live site this audit found.** A generic method on a generic impl. `Wrap<T>::get(&self)`
/// returned `&S` — `T` sliced from the caller's file — so nothing could unify against it.
#[test]
fn a_generic_method_on_a_generic_impl_resolves() {
    expect_clean(
        "genericmethod",
        "pub struct Wrap<T> { pub inner: T }\n\
         \n\
         impl<T> Wrap<T> {\n\
         \x20   pub fn get(&self) -> &T {\n\
         \x20       &self.inner\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Wrap;\n\
         fn use_it() -> Bool {\n\
         \x20   let w = Wrap { inner: 11 };\n\
         \x20   *w.get() == 11\n\
         }\n",
    );
}

/// A method declaring its OWN generic parameter, on top of the impl's.
#[test]
fn a_method_level_generic_parameter_resolves() {
    expect_clean(
        "methodgeneric",
        "pub struct Wrap<T> { pub inner: T }\n\
         \n\
         impl<T> Wrap<T> {\n\
         \x20   pub fn pair<U>(&self, other: U) -> U {\n\
         \x20       other\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Wrap;\n\
         fn use_it() -> Bool {\n\
         \x20   let w = Wrap { inner: 3 };\n\
         \x20   w.pair(9) == 9\n\
         }\n",
    );
}

/// A trait DEFAULT method body, whose signature is declared in the trait's file — a third file's
/// worth of provenance in play at once (trait, impl, caller).
#[test]
fn a_trait_default_method_resolves() {
    expect_clean(
        "traitdefault",
        "pub struct Point { pub x: Int32, pub y: Int32 }\n\
         \n\
         pub trait Area {\n\
         \x20   fn area(&self) -> Int32;\n\
         \x20   fn describe(&self) -> Int32 {\n\
         \x20       self.area() + 1\n\
         \x20   }\n\
         }\n\
         \n\
         impl Area for Point {\n\
         \x20   fn area(&self) -> Int32 {\n\
         \x20       self.x * self.y\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Area;\n\
         use super::Point;\n\
         fn use_it() -> Bool {\n\
         \x20   let p = Point { x: 2, y: 3 };\n\
         \x20   p.area() == 6 && p.describe() == 7\n\
         }\n",
    );
}

/// An associated TYPE resolved through a trait impl — the open question DEV-148 left behind.
#[test]
fn an_associated_type_resolves() {
    expect_clean(
        "assoctype",
        "pub struct Wrap { pub inner: Int32 }\n\
         \n\
         pub trait Produce {\n\
         \x20   type Out;\n\
         \x20   fn produce(&self) -> Self::Out;\n\
         }\n\
         \n\
         impl Produce for Wrap {\n\
         \x20   type Out = Int32;\n\
         \x20   fn produce(&self) -> Int32 {\n\
         \x20       self.inner\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Produce;\n\
         use super::Wrap;\n\
         fn use_it() -> Bool {\n\
         \x20   let w = Wrap { inner: 5 };\n\
         \x20   w.produce() == 5\n\
         }\n",
    );
}

/// A generic free function with a TRAIT BOUND, calling the bound's method.
#[test]
fn a_bounded_generic_function_resolves() {
    expect_clean(
        "boundedfn",
        "pub struct Small { pub v: Int32 }\n\
         \n\
         pub trait Bounded {\n\
         \x20   fn value(&self) -> Int32;\n\
         }\n\
         \n\
         impl Bounded for Small {\n\
         \x20   fn value(&self) -> Int32 {\n\
         \x20       self.v\n\
         \x20   }\n\
         }\n\
         \n\
         pub fn largest<T: Bounded>(a: &T, b: &T) -> Int32 {\n\
         \x20   let av = a.value();\n\
         \x20   let bv = b.value();\n\
         \x20   if av > bv { av } else { bv }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::largest;\n\
         use super::Small;\n\
         fn use_it() -> Bool {\n\
         \x20   let a = Small { v: 2 };\n\
         \x20   let b = Small { v: 8 };\n\
         \x20   largest(&a, &b) == 8\n\
         }\n",
    );
}

/// Struct fields, enum variants of all three shapes, and matching on them. These read names off
/// declarations too, and a mis-slice here would misname a field or variant.
#[test]
fn fields_and_enum_variants_resolve() {
    expect_clean(
        "fieldsvariants",
        "pub struct Point { pub x: Int32, pub y: Int32 }\n\
         \n\
         pub enum Shape { Dot, Line(Int32), Named { size: Int32 } }\n\
         \n\
         mod inner;\n",
        "use super::Point;\n\
         use super::Shape;\n\
         fn use_it() -> Bool {\n\
         \x20   let p = Point { x: 4, y: 5 };\n\
         \x20   let named = Shape::Named { size: 9 };\n\
         \x20   let sized = match named {\n\
         \x20       Shape::Named { size } => size,\n\
         \x20       Shape::Line(v) => v,\n\
         \x20       Shape::Dot => 0,\n\
         \x20   };\n\
         \x20   p.x == 4 && p.y == 5 && sized == 9\n\
         }\n",
    );
}

/// **Two generic parameters, deliberately.** `item_text` returns `"?"` for an out-of-range span, so
/// two mis-sliced names could collide on one key and substitute each other's types — a wrong
/// program rather than a rejected one. Distinct parameters instantiated at distinct types pin that
/// this cannot happen.
#[test]
fn two_generic_parameters_do_not_collide() {
    expect_clean(
        "twoparams",
        "pub struct Both<A, B> { pub a: A, pub b: B }\n\
         \n\
         impl<A, B> Both<A, B> {\n\
         \x20   pub fn first(&self) -> &A {\n\
         \x20       &self.a\n\
         \x20   }\n\
         \x20   pub fn second(&self) -> &B {\n\
         \x20       &self.b\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Both;\n\
         fn use_it() -> Bool {\n\
         \x20   let v = Both { a: 1, b: true };\n\
         \x20   *v.first() == 1 && *v.second()\n\
         }\n",
    );
}

/// The associated-function path DEV-148 fixed, kept here so the whole class lives in one file.
#[test]
fn an_associated_function_still_resolves() {
    expect_clean(
        "assocfn",
        "pub struct Wrap { pub v: Int32 }\n\
         \n\
         impl Wrap {\n\
         \x20   pub fn make(v: Int32) -> Wrap {\n\
         \x20       Wrap { v: v }\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Wrap;\n\
         fn use_it() -> Bool {\n\
         \x20   let w = Wrap::make(2);\n\
         \x20   w.v == 2\n\
         }\n",
    );
}
