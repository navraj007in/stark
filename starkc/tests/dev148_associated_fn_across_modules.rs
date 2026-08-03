//! **DEV-148: an associated function was unresolvable across any module boundary, because its name
//! was sliced out of the wrong file.**
//!
//! ```stark
//! // src/lib.stark
//! pub struct Wrap { pub v: Int32 }
//! impl Wrap { pub fn make(v: Int32) -> Wrap { Wrap { v: v } } }
//! mod tests;
//!
//! // src/tests.stark
//! use super::Wrap;
//! let b = Wrap::make(2);          // E0200 associated function 'make' not found
//! let c = super::Wrap::make(2);   // and the fully qualified path failed too
//! ```
//!
//! # What it actually was
//!
//! Not visibility, not coherence, not path resolution — the path reached `Res::AssociatedFn`
//! correctly. `typecheck`'s lookup then compared member names using `self.text(span)`, which slices
//! **the file currently being checked**. A member's name span belongs to the file that declared the
//! `impl`. Across a module boundary those differ, so it compared garbage. Instrumented, the two
//! members of `impl Wrap` read back as:
//!
//! ```text
//! member name_text="rap:"  has_receiver=false
//! member name_text="?"     has_receiver=true
//! ```
//!
//! `"rap:"` is `make`'s offsets applied to the other file; `"?"` is `item_text`'s out-of-range
//! fallback, for a span running past the shorter file's end. No candidate ever matched.
//!
//! **METHODS were unaffected**, because method lookup selects on the receiver's TYPE rather than by
//! slicing a name. That asymmetry is what made this look like a language rule about associated
//! functions — it was filed as "cross-package associated functions are unresolvable" — rather than
//! a text bug. It is neither about packages nor about associated functions per se; it is about file
//! provenance, and `stark-url` could not call its own `Url::parse` from its own test submodule.
//!
//! # Second site: generic parameter names
//!
//! Fixing the name comparison exposed the same defect one layer down. A GENERIC associated function
//! resolved, then produced a stray parameter type:
//!
//! ```text
//! error: [E0500] type 'r' does not satisfy operator trait 'Eq'
//! ```
//!
//! `'r'` is `T` sliced from the wrong file. The substitution map's keys and the `Ty::Param`s they
//! substitute into must be read from the SAME file or substitution silently fails to fire, so
//! `foreign_sig_item` now carries the declaring item across the whole signature conversion.
//!
//! # The precedent this should have followed
//!
//! DEV-069 fixed exactly this for trait methods ("the trait's method names belong to the TRAIT's
//! declaring file") and `build_assoc_projections` converts "against the impl's own file". The rule
//! was already known and written down twice; this site simply missed it. Worth stating plainly:
//! `self.text` is only correct for spans from the file under check, and every lookup that reads a
//! name off a foreign declaration needs `item_text`.

use starkc::analysis::{analyze_project, ProjectInput};
use starkc::diag::Severity;
use starkc::options::LanguageOptions;

/// A two-file package: `lib.stark` declaring the impl, `tests.stark` calling across the boundary.
/// The defect is about file PROVENANCE, so a single-file fixture cannot reproduce it — that is the
/// reason this suite builds a real package graph instead of analysing a string.
fn two_file_package(tag: &str, lib: &str, submodule: &str) -> Vec<String> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(format!(
            "temp_dev148_{tag}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time must be after epoch")
                .as_nanos()
        ));
    let src = root.join("src");
    std::fs::create_dir_all(&src).expect("create temp package src");
    std::fs::write(
        root.join("starkpkg.json"),
        r#"{"name":"dev148_app","version":"0.1.0","entry":"src/lib.stark"}"#,
    )
    .expect("write manifest");
    std::fs::write(src.join("lib.stark"), lib).expect("write lib");
    std::fs::write(src.join("inner.stark"), submodule).expect("write submodule");

    let manifest = starkc::package::find_package_root(&root).expect("find manifest");
    let graph = starkc::package::PackageGraph::load_from_root(&manifest).expect("load package");
    let analysis = analyze_project(ProjectInput::package(graph), LanguageOptions::CORE);
    let messages = analysis
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect();
    let _ = std::fs::remove_dir_all(&root);
    messages
}

const PLAIN_LIB: &str = "\
pub struct Wrap { pub v: Int32 }

impl Wrap {
    pub fn make(v: Int32) -> Wrap {
        Wrap { v: v }
    }
    pub fn get(&self) -> Int32 {
        self.v
    }
}

mod inner;
";

/// **The reproducer.** An associated function called from a submodule of its own package.
#[test]
fn an_associated_function_resolves_from_a_submodule() {
    let errors = two_file_package(
        "plain",
        PLAIN_LIB,
        "use super::Wrap;\n\
         fn use_it() -> Int32 {\n\
         \x20   let w = Wrap::make(2);\n\
         \x20   w.get()\n\
         }\n",
    );
    assert!(
        errors.is_empty(),
        "an associated function must resolve across a module boundary: {errors:?}"
    );
}

/// The fully qualified form, which failed identically and proves the defect was not in the `use`.
#[test]
fn a_fully_qualified_associated_function_resolves() {
    let errors = two_file_package(
        "qualified",
        PLAIN_LIB,
        "fn use_it() -> Int32 {\n\
         \x20   let w = super::Wrap::make(2);\n\
         \x20   w.get()\n\
         }\n",
    );
    assert!(errors.is_empty(), "{errors:?}");
}

/// **The second site.** A GENERIC associated function: substitution must actually fire, or the
/// caller is handed a stray `Ty::Param` named from the wrong file and the error surfaces somewhere
/// unrelated — `type 'r' does not satisfy operator trait 'Eq'`.
#[test]
fn a_generic_associated_function_substitutes_across_a_boundary() {
    let errors = two_file_package(
        "generic",
        "pub struct Pair<T> { pub a: T, pub b: T }\n\
         \n\
         impl<T> Pair<T> {\n\
         \x20   pub fn first(a: T, b: T) -> Pair<T> {\n\
         \x20       Pair { a: a, b: b }\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Pair;\n\
         fn use_it() -> Bool {\n\
         \x20   let p = Pair::first(5, 6);\n\
         \x20   p.a != 5\n\
         }\n",
    );
    assert!(
        errors.is_empty(),
        "a generic associated function must instantiate to the caller's argument types; a stray \
         `Ty::Param` here means the substitution map was keyed from the wrong file: {errors:?}"
    );
}

/// Two parameters, because `item_text` returns `\"?\"` for an out-of-range span — so several
/// mis-sliced names could COLLIDE on one key and substitute each other's types. Distinct
/// parameters with distinct instantiations pin that this cannot happen.
#[test]
fn two_generic_parameters_do_not_collide() {
    let errors = two_file_package(
        "twoparams",
        "pub struct Both<A, B> { pub a: A, pub b: B }\n\
         \n\
         impl<A, B> Both<A, B> {\n\
         \x20   pub fn make(a: A, b: B) -> Both<A, B> {\n\
         \x20       Both { a: a, b: b }\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Both;\n\
         fn use_it() -> Bool {\n\
         \x20   let v = Both::make(1, true);\n\
         \x20   v.a == 1 && v.b\n\
         }\n",
    );
    assert!(
        errors.is_empty(),
        "distinct parameters must keep distinct types across the boundary: {errors:?}"
    );
}

/// **The control that matters most.** Methods always worked; the repair must not have changed how
/// they resolve, or it traded one defect for another in a path with far more coverage.
#[test]
fn methods_still_resolve_across_a_boundary() {
    let errors = two_file_package(
        "methods",
        PLAIN_LIB,
        "use super::Wrap;\n\
         fn use_it(w: &Wrap) -> Int32 {\n\
         \x20   w.get()\n\
         }\n",
    );
    assert!(errors.is_empty(), "{errors:?}");
}

/// **The negative control.** Visibility is enforced by a separate check, and making the name
/// comparison work must not have made a PRIVATE associated function callable from another module.
#[test]
fn a_private_associated_function_is_still_refused() {
    let errors = two_file_package(
        "private",
        "pub struct Wrap { pub v: Int32 }\n\
         \n\
         impl Wrap {\n\
         \x20   fn secret(v: Int32) -> Wrap {\n\
         \x20       Wrap { v: v }\n\
         \x20   }\n\
         }\n\
         \n\
         mod inner;\n",
        "use super::Wrap;\n\
         fn use_it() -> Int32 {\n\
         \x20   let w = Wrap::secret(2);\n\
         \x20   w.v\n\
         }\n",
    );
    assert!(
        !errors.is_empty(),
        "a private associated function must stay inaccessible from another module — the repair \
         fixed a name comparison, not a visibility rule"
    );
}

/// A name that genuinely does not exist must still be refused, and with the right diagnostic. If
/// the comparison had been loosened rather than corrected, this would start passing.
#[test]
fn a_missing_associated_function_is_still_refused() {
    let errors = two_file_package(
        "missing",
        PLAIN_LIB,
        "use super::Wrap;\n\
         fn use_it() -> Int32 {\n\
         \x20   let w = Wrap::nonexistent(2);\n\
         \x20   w.get()\n\
         }\n",
    );
    assert!(
        errors.iter().any(|e| e.contains("nonexistent")
            || e.contains("associated function")
            || e.starts_with("E0200")),
        "expected a not-found diagnostic naming the missing function, got: {errors:?}"
    );
}
