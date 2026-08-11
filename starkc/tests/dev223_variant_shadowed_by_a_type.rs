//! **DEV-223: a qualified enum's own variants outrank unrelated enclosing-module items.**
//!
//! `enum Attr { Flag, Policy(Policy) }` alongside `enum Policy { A, B }` broke in two places at
//! once, because `resolve_path_relative`'s subsequent-segment loop consulted the enclosing
//! module's items BEFORE the variants of the item being qualified. `Attr` is not a submodule, so
//! `current_mod` stayed the enclosing module, `Policy` was found there as the TYPE, and
//! `Attr::Policy` never reached the variant lookup below.
//!
//! - **Pattern position:** the arm carried a non-`Res::Variant` resolution, exhaustiveness did not
//!   count it, and a genuinely exhaustive match was rejected as `E0303 non-exhaustive`.
//! - **Expression position:** `Attr::Policy(Policy::A)` — an ordinary constructor — passed
//!   `stark check` and then failed at RUNTIME with `item is not callable`. This face is why the
//!   deviation was re-filed: it is not the fail-safe over-rejection it first looked like.
//!
//! # Why these negative controls
//!
//! The repair moves one lookup ahead of another, so the risk is **stealing names that belong to
//! the module**. The fix is deliberately narrow — it answers only for a name that IS a variant of
//! the enum being qualified — and the controls below pin the boundary: a non-variant name must
//! still reach the module lookup and then the associated-function fallback, module paths and
//! imports must be unaffected, and an enum whose variants do not collide must behave exactly as
//! before.

mod support;

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Both stages, because DEV-223's two faces failed at different ones: the pattern face reached
/// `typecheck` and was rejected as non-exhaustive, the expression face passed both and failed at
/// runtime.
fn check_errors(src: &str, tag: &str) -> Vec<String> {
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

const COLLIDING: &str = "\
enum Policy { A, B }
enum Attr { Flag, Policy(Policy) }
fn render(attr: &Attr) -> String {
    match *attr {
        Attr::Flag => String::from(\"flag\"),
        Attr::Policy(p) => match p {
            Policy::A => String::from(\"a\"),
            Policy::B => String::from(\"b\"),
        },
    }
}
fn build() -> Attr { Attr::Policy(Policy::A) }
fn main() {
    println(render(&build()).as_str());
    println(render(&Attr::Flag).as_str());
}
";

#[test]
fn a_variant_named_after_a_type_is_matchable_and_exhaustive() {
    assert_eq!(
        check_errors(COLLIDING, "dev223_pattern_face"),
        Vec::<String>::new(),
        "the match covers both variants; a module-level type sharing a variant name must not \
         make it non-exhaustive"
    );
}

/// The face that made this wrong-code rather than merely a spurious rejection — and it has to be
/// EXECUTED to be caught. Before the repair this program type-checked clean and died at runtime
/// with `item is not callable`, so a test that stopped at `typecheck` passed against the defect.
#[test]
fn the_constructor_expression_face_resolves_to_the_variant() {
    let source = "\
enum Policy { A, B }
enum Attr { Flag, Policy(Policy) }
fn main() {
    let a = Attr::Policy(Policy::B);
    match a {
        Attr::Policy(_p) => println(\"policy\"),
        Attr::Flag => println(\"flag\"),
    }
}
";
    let file = Arc::new(SourceFile::new("dev223_expression_face.stark", source));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "the program must type-check: {errors:?}");

    let outcome = interp::run_capturing(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    );
    assert!(
        outcome.result.is_ok(),
        "`Attr::Policy(..)` must construct Attr's variant, not trap as 'item is not callable': \
         {:?}",
        outcome.result
    );
    assert!(
        outcome.output.contains("policy"),
        "the constructed value must match Attr's Policy arm: {:?}",
        outcome.output
    );
}

/// Task requirement 6, stated directly.
#[test]
fn a_module_level_type_does_not_shadow_a_variant_of_the_qualified_enum() {
    let src = "\
struct Flag { on: Bool }
enum Signal { Flag, Off }
fn text(s: &Signal) -> String {
    match *s {
        Signal::Flag => String::from(\"flag\"),
        Signal::Off => String::from(\"off\"),
    }
}
fn main() {
    let f = Flag { on: true };
    if f.on { println(text(&Signal::Flag).as_str()); }
}
";
    assert_eq!(
        check_errors(src, "dev223_struct_shadow"),
        Vec::<String>::new(),
        "a struct named `Flag` must not shadow `Signal::Flag`"
    );
}

// -- the boundary: names that are NOT variants must keep their old resolution -------------------

/// The reorder answers only for actual variants. A non-variant name must still fall through to the
/// module lookup and then to the associated-function fallback, which is what DEV-222's repair
/// relies on being intact.
#[test]
fn a_non_variant_name_still_falls_through_to_the_associated_function() {
    let src = "\
enum Colour { Red, Green }
impl Colour {
    pub fn count() -> Int64 { 2i64 }
}
fn main() { println(Colour::count()); }
";
    assert_eq!(
        check_errors(src, "dev223_assoc_fallthrough"),
        Vec::<String>::new(),
        "`Colour::count` is not a variant and must still resolve as an associated function"
    );
}

#[test]
fn a_non_colliding_enum_behaves_exactly_as_before() {
    let src = "\
enum Colour { Red, Green }
fn name(c: &Colour) -> String {
    match *c {
        Colour::Red => String::from(\"red\"),
        Colour::Green => String::from(\"green\"),
    }
}
fn main() { println(name(&Colour::Green).as_str()); }
";
    assert_eq!(
        check_errors(src, "dev223_no_collision"),
        Vec::<String>::new(),
        "an enum with no name collision must be unaffected by the reorder"
    );
}

/// `use Enum::{A, B}` and `use Enum::*` route through the same resolver, so the reorder reaches
/// them. Importing a variant is the behaviour that should win here, and must keep working.
#[test]
fn enum_variant_imports_still_resolve() {
    let src = "\
mod inner {
    pub enum Colour { Red, Green }
}
use inner::Colour;
use inner::Colour::Red;
fn main() {
    let c = Red;
    match c {
        Colour::Red => println(\"red\"),
        Colour::Green => println(\"green\"),
    }
}
";
    assert_eq!(
        check_errors(src, "dev223_variant_import"),
        Vec::<String>::new(),
        "enum variant imports must not regress"
    );
}
