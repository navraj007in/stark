//! **DEV-229: a builtin spelling does not pre-empt a declared name.**
//!
//! `resolve_path_relative` opened with a `match self.path_to_string(path).as_str()` over
//! twenty-six spellings — `String::from`, `Vec::new`, `Ordering::Less`, `IOError::Other` and the
//! rest — returning `Res::Builtin` **before any namespace was consulted**. A declared name could
//! therefore never win.
//!
//! The symptom was unactionable on its face:
//!
//! ```text
//! enum Ordering { Less, Equal, Greater }
//! // error: type mismatch: expected 'Ordering', found 'Ordering'
//! ```
//!
//! Two types, one spelling, one program: the scrutinee resolved to the user's enum through the
//! type namespace, the match arms to the builtin through the table. Nothing in that message
//! suggests the fix is "do not name your enum Ordering".
//!
//! The table is now consulted only when ordinary resolution finds nothing. NAME-RESOLVE-001 gives
//! a builtin SPELLING no precedence over a declared name.
//!
//! # Why these negative controls carry the weight
//!
//! This repair changes which of two candidates wins for twenty-six spellings, so the risk is not
//! that the user's declaration fails to win — it is that the BUILTIN stops resolving when nothing
//! shadows it. An intermediate version of this repair placed the fallback at the end of the
//! segment walk, where it was unreachable: that walk returns `Res::Err` the moment a segment names
//! nothing declared, which is exactly the state a builtin spelling is in. The reproducer passed
//! and `String::from` silently stopped resolving. Only a control caught it.

mod support;

use starkc::interp;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

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

/// The reproducer that turned DEV-229 from suspected into confirmed.
#[test]
fn a_user_enum_beats_the_builtin_of_the_same_spelling() {
    let out = run(
        "\
enum Ordering { Less, Equal, Greater }
fn tag(o: Ordering) -> Int64 {
    match o { Ordering::Less => 111i64, Ordering::Equal => 222i64, Ordering::Greater => 333i64 }
}
fn main() { println(tag(Ordering::Less)); }
",
        "dev229_user_ordering",
    );
    assert!(
        out.contains("111"),
        "the user's enum must win, and its arms must cover its own type: {out:?}"
    );
}

#[test]
fn a_user_type_may_take_a_builtin_qualifier_spelling() {
    let out = run(
        "\
struct Random { seed: Int64 }
impl Random { pub fn new(seed: Int64) -> Random { Random { seed: seed } } }
fn main() { println(Random::new(9i64).seed); }
",
        "dev229_user_random",
    );
    assert!(out.contains('9'), "{out:?}");
}

// -- controls: the builtins must still resolve when nothing shadows them -------------------------

/// The control the intermediate version of this repair failed. `String::from` reaches the table
/// only through the fallback, and the fallback only works because it sits ABOVE the segment walk
/// rather than at the end of it.
#[test]
fn the_builtins_still_resolve_when_nothing_shadows_them() {
    let out = run(
        "\
fn main() {
    let s = String::from(\"hi\");
    let mut v: Vec<Int64> = Vec::new();
    v.push(1i64);
    println(s.as_str());
    println(v.len());
    match 1i64.cmp(&2i64) { Ordering::Less => println(\"less\"), _ => println(\"other\") }
}
",
        "dev229_control_builtins",
    );
    assert!(out.contains("hi"), "String::from must resolve: {out:?}");
    assert!(out.contains('1'), "Vec::new must resolve: {out:?}");
    assert!(out.contains("less"), "Ordering::Less must resolve: {out:?}");
}

#[test]
fn the_wider_builtin_surface_still_resolves() {
    let out = run(
        "\
fn main() {
    let mut m: HashMap<Int64, Int64> = HashMap::new();
    m.insert(1i64, 2i64);
    let mut hs: HashSet<Int64> = HashSet::new();
    hs.insert(3i64);
    let b = Box::new(7i64);
    println(m.len());
    println(hs.len());
    println(Box::into_inner(b));
    match Char::from_u32(65u32) { Some(c) => println(c), None => println(\"none\") }
    println(String::with_capacity(4u64).len());
}
",
        "dev229_control_wider",
    );
    for expect in ["1", "7", "A", "0"] {
        assert!(
            out.contains(expect),
            "expected {expect:?} in the builtin surface output: {out:?}"
        );
    }
}
