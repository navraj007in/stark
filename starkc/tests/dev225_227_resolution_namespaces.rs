//! **DEV-225/226/227: the resolution namespaces NAME-RESOLVE-001 actually specifies.**
//!
//! Found by auditing outward from DEV-222/223 rather than by writing code, on the theory that a
//! defect which lets an illegal resolution through in one context probably does so in others. It
//! did.
//!
//! - **DEV-225** — `04-Semantic-Analysis.md` NAME-RESOLVE-001: *"Associated names are searched
//!   only after resolving their qualifying type or trait."* The subsequent-segment loop searched
//!   the enclosing module first, so a module-level `new` beat `impl Foo { fn new() }`. DEV-223 was
//!   the enum-variant face of this; the rule covers structs and traits too.
//! - **DEV-226** — every `Res::Builtin` was accepted as a pattern, so `Vec::new(x)` — a function —
//!   was a tuple-variant pattern that silently never matched.
//! - **DEV-227** — every `Res::Item` was accepted as a by-value pattern, so a bare `helper` naming
//!   a FUNCTION became a value pattern that could never equal anything.
//!   `02-Syntax-Grammar.md` SYN-PATTERN-001 says a bare identifier matches by value only for "a
//!   unit enum variant or a constant in scope" and "otherwise introduces a new binding" — so the
//!   repair here is to BIND, not to reject.
//!
//! # Why these negative controls
//!
//! Every fix in this family narrows something, so each carries the risk of narrowing too far. The
//! controls pin what must keep working: a constant is still a by-value pattern, bare `None` still
//! matches by value (DEV-053's repair), module paths and `super`/`crate` still resolve, and
//! associated functions still resolve across module boundaries.

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

// -- DEV-225: the qualifier's associated namespace outranks the enclosing module ----------------

#[test]
fn a_struct_associated_function_outranks_a_module_level_name() {
    let out = run(
        "\
struct Foo { a: Int64 }
impl Foo { pub fn new() -> Foo { Foo { a: 1i64 } } }
fn new() -> Int64 { 99i64 }
fn main() { let f = Foo::new(); println(f.a); }
",
        "dev225_struct_assoc",
    );
    assert!(
        out.contains('1'),
        "`Foo::new` must be Foo's associated function, not the module-level `new`: {out:?}"
    );
}

#[test]
fn an_enum_variant_outranks_a_module_level_type() {
    let out = run(
        "\
enum Policy { A, B }
enum Attr { Flag, Policy(Policy) }
fn main() {
    match Attr::Policy(Policy::B) {
        Attr::Policy(_p) => println(\"variant\"),
        Attr::Flag => println(\"flag\"),
    }
}
",
        "dev225_enum_variant",
    );
    assert!(out.contains("variant"), "{out:?}");
}

#[test]
fn a_trait_member_outranks_a_module_level_name() {
    assert_eq!(
        errors(
            "\
trait Describe { fn describe(&self) -> String; }
fn describe() -> Int64 { 0i64 }
struct Point { x: Int64 }
impl Describe for Point {
    fn describe(&self) -> String { String::from(\"point\") }
}
fn main() { let p = Point { x: 1i64 }; println(p.describe().as_str()); }
",
            "dev225_trait_member"
        ),
        Vec::<String>::new(),
        "a module-level `describe` must not disturb the trait's member"
    );
}

// -- DEV-226: only builtin CONSTRUCTORS are patterns --------------------------------------------

#[test]
fn a_builtin_function_is_not_a_tuple_pattern() {
    let errs = errors(
        "\
fn main() {
    let v: Option<Int64> = Some(3i64);
    match v { Vec::new(x) => println(x), _other => println(\"fell through\") }
}
",
        "dev226_builtin_fn_pattern",
    );
    assert!(
        errs.iter().any(|e| e.contains("Vec::new")),
        "`Vec::new` is a function, not a pattern constructor: {errs:?}"
    );
}

#[test]
fn builtin_constructors_are_still_patterns() {
    let out = run(
        "\
fn main() {
    let o: Option<Int64> = None;
    match o { Some(v) => println(v), None => println(\"none\") }
}
",
        "dev226_builtin_ctor",
    );
    assert!(
        out.contains("none"),
        "bare `None` must still match by value -- DEV-053's repair: {out:?}"
    );
}

// -- DEV-227: a bare identifier is a value pattern only for a variant or a constant -------------

#[test]
fn a_bare_function_name_introduces_a_binding() {
    let out = run(
        "\
fn helper() -> Int64 { 1i64 }
fn main() { let n = 3i64; match n { helper => println(helper), } }
",
        "dev227_fn_name_binds",
    );
    assert!(
        out.contains('3'),
        "SYN-PATTERN-001: a name that is not a unit variant or a constant introduces a new \
         binding, so this must print the scrutinee: {out:?}"
    );
}

#[test]
fn a_constant_is_still_a_by_value_pattern() {
    let out = run(
        "\
const LIMIT: Int64 = 3i64;
fn main() { let n = 3i64; match n { LIMIT => println(\"limit\"), _other => println(\"other\") } }
",
        "dev227_const_pattern",
    );
    assert!(
        out.contains("limit"),
        "a constant must keep matching by value: {out:?}"
    );
}

// -- controls for the precedence reorder --------------------------------------------------------

#[test]
fn module_paths_and_super_still_resolve() {
    assert_eq!(
        errors(
            "\
mod inner {
    pub struct Wrap { pub n: Int64 }
    impl Wrap { pub fn make() -> Wrap { Wrap { n: 2i64 } } }
    pub mod deeper {
        pub fn reach() -> Int64 { super::Wrap::make().n }
    }
}
fn main() { println(inner::deeper::reach()); }
",
            "dev225_module_paths"
        ),
        Vec::<String>::new(),
        "`super::Wrap::make` steers through the MODULE namespace and must be unaffected"
    );
}
