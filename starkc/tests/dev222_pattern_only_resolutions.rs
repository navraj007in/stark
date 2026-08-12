//! **DEV-222: a pattern may not name a resolution that is not a pattern.**
//!
//! `match c { Colour::Blu => .., _ => .. }` — where `Colour` has no variant `Blu` — type-checked
//! clean and silently never matched, falling through to the wildcard at runtime with no
//! diagnostic. Remove the wildcard and it was caught, but by `E0303 non-exhaustive`, which points
//! at the `match` rather than at the typo; the obvious response to that is to add a wildcard,
//! which converts a caught bug into a silent one.
//!
//! # Where the defect was NOT
//!
//! Not in `resolve_path`. For a name that is not a variant of the qualified enum or struct, the
//! resolver answers `Res::AssociatedFn`, and in expression position that answer is **correct and
//! load-bearing** — `Duration::from_seconds`, `Instant::now` and `Line::new` all reach their
//! definitions through it. The defect was that `lower_pattern`'s three branches asked only
//! "is it `Res::Err`" before accepting the resolution, when a pattern may name far less than an
//! expression may.
//!
//! # Why these negative controls
//!
//! The risk in this repair is **over-rejection**: a guard that is too eager would refuse ordinary
//! patterns. So the accepting side is pinned as hard as the rejecting side — valid unit variants,
//! valid tuple variants, a struct pattern and a `None`/`Some` builtin must all still resolve, and
//! associated-function calls in expression position must be untouched. This is the same class as
//! DEV-053/054, where a pattern path that did not resolve to a variant silently became an
//! unconditional wildcard.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use std::sync::Arc;

/// Resolution diagnostics only. DEV-222 is a resolution-stage rejection, so a program that gets
/// past this stage has already lost the property under test.
fn resolve_errors(src: &str, tag: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (_hir, rd) = resolve(&ast, file);
    rd.iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| {
            format!(
                "{} {}",
                d.code.as_deref().unwrap_or("-"),
                d.message.replace('\n', " ")
            )
        })
        .collect()
}

#[test]
fn a_misspelled_enum_variant_is_rejected_even_with_a_wildcard_present() {
    let src = "\
enum Colour { Red, Green }
fn describe(c: &Colour) -> String {
    match *c {
        Colour::Blu => String::from(\"blue\"),
        Colour::Red => String::from(\"red\"),
        _other => String::from(\"wildcard\"),
    }
}
fn main() { println(describe(&Colour::Green).as_str()); }
";
    let errors = resolve_errors(src, "dev222_enum_typo");
    assert!(
        errors.iter().any(|e| e.contains("Colour::Blu")),
        "a misspelled variant must be rejected, and the diagnostic must name the typo \
         rather than the match: {errors:?}"
    );
}

#[test]
fn a_struct_cannot_pretend_to_have_variants() {
    let src = "\
struct Thing { value: Int64 }
fn main() {
    let t = Thing { value: 7i64 };
    let r = &t;
    match *r {
        Thing::Missing(n) => println(n),
        _other => println(\"wildcard\"),
    }
}
";
    let errors = resolve_errors(src, "dev222_struct_variant");
    assert!(
        errors.iter().any(|e| e.contains("Thing::Missing")),
        "a struct has no variants, so a variant pattern on one must be rejected: {errors:?}"
    );
}

/// F from the task list, stated as its own case because it is the whole point: the wildcard is
/// exactly what made the original defect silent.
#[test]
fn a_wildcard_does_not_launder_an_invalid_qualified_variant() {
    let with_wildcard = "\
enum Colour { Red, Green }
fn f(c: &Colour) -> Int64 { match *c { Colour::Blu => 1i64, _other => 2i64 } }
fn main() { println(f(&Colour::Red)); }
";
    assert!(
        !resolve_errors(with_wildcard, "dev222_wildcard").is_empty(),
        "a wildcard arm must not make an invalid qualified variant acceptable"
    );
}

// -- the accepting side: nothing below may regress ---------------------------------------------

#[test]
fn valid_unit_and_tuple_variants_still_resolve() {
    let src = "\
enum Colour { Red, Green }
enum Shape { Dot, Line(Int64) }
fn name(c: &Colour) -> String {
    match *c {
        Colour::Red => String::from(\"red\"),
        Colour::Green => String::from(\"green\"),
    }
}
fn size(s: &Shape) -> Int64 {
    match *s {
        Shape::Dot => 0i64,
        Shape::Line(n) => n,
    }
}
fn main() {
    println(name(&Colour::Red).as_str());
    println(size(&Shape::Line(3i64)));
}
";
    assert_eq!(
        resolve_errors(src, "dev222_valid_variants"),
        Vec::<String>::new(),
        "valid unit and tuple variant patterns must keep resolving"
    );
}

/// The regression this repair could most plausibly have caused. `Res::AssociatedFn` is the
/// resolver's correct answer in expression position for a qualified name that is not a variant,
/// and over sixty call sites across `packages/` depend on it.
#[test]
fn associated_functions_in_expression_position_are_untouched() {
    let src = "\
struct Line { n: Int64 }
impl Line {
    pub fn new(n: Int64) -> Line { Line { n: n } }
    pub fn value(&self) -> Int64 { self.n }
}
struct Duration { secs: Int64 }
impl Duration {
    pub fn from_seconds(s: Int64) -> Duration { Duration { secs: s } }
}
fn main() {
    let l = Line::new(41i64);
    let d = Duration::from_seconds(2i64);
    println(l.value() + d.secs);
}
";
    assert_eq!(
        resolve_errors(src, "dev222_assoc_fn"),
        Vec::<String>::new(),
        "an inherent associated function must still resolve in expression position"
    );
}

#[test]
fn builtin_option_patterns_are_still_patterns() {
    let src = "\
fn unwrap_or(o: Option<Int64>, d: Int64) -> Int64 {
    match o {
        Some(v) => v,
        None => d,
    }
}
fn main() { println(unwrap_or(Some(7i64), 0i64)); }
";
    assert_eq!(
        resolve_errors(src, "dev222_builtin_patterns"),
        Vec::<String>::new(),
        "`Some`/`None` resolve to builtins and must remain legal patterns"
    );
}

/// **WP-ARCH-CLOSE AC4 — the control this file was missing.**
///
/// `resolution_is_pattern_legal`'s `Res::Item` arm admits a struct or a constant and rejects
/// everything else — a function, a trait, a module item. **Nothing tested that arm.** AC4's
/// mutation `AC4-MUT-PAT-002` replaced it with `Res::Item(_) => true` and the whole suite stayed
/// green, so the arm was unguarded: a regression there would restore DEV-227's defect, where a
/// pattern naming a non-constructor compiles, reports nothing, and silently never matches.
///
/// The shape has to be a QUALIFIED path. A bare identifier — even a capitalised one — binds a fresh
/// variable in pattern position and never reaches the authority at all, which is why the first
/// three probes AC4 tried proved nothing and had to be discarded.
#[test]
fn a_qualified_path_naming_a_module_function_is_not_a_pattern() {
    let source = r#"
mod m { pub fn f() -> Int32 { 1 } }
fn main() {
    let n: Int32 = 1;
    match n { m::f => { println(1); } _ => { println(2); } }
}
"#;
    let errors = resolve_errors(source, "dev222_module_fn_pattern");
    assert!(
        !errors.is_empty(),
        "`m::f` is a function, not a constructor or a constant, so it may not appear in pattern \
         position. Accepting it produces a pattern that never matches and reports nothing — \
         DEV-227's defect. Got no diagnostics at all."
    );
}
