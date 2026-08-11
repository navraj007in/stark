//! **DEV-234: a `Copy` bound is answered by the same classifier the rest of the compiler uses.**
//!
//! It was two gaps, and repairing either alone leaves the bound useless.
//!
//! 1. `typecheck/traits.rs`'s `satisfies_bound_identity` had cases for `Num`, `Eq`, `Ord`,
//!    `Display`, `Clone`, `Default` and `Hash` — and none for `Copy` — so every concrete type fell
//!    through to `false`. `Int64` did not satisfy `Copy`, although `is_copy_type` says it is
//!    copyable (which is why `let a = b;` copies it) and `03-Type-System.md` NUM-FLOAT-TRAIT-001
//!    says the float primitives implement it.
//! 2. `borrowck::is_copy_type` answers `false` for every `Ty::Param`. Correct for a parameter that
//!    declares nothing, wrong for one that declares `Copy` — so with only the first gap repaired,
//!    `fn take<T: Copy>(r: &T) -> T { *r }` was satisfiable at the call site and rejected inside
//!    its own body. The bound was not an escape hatch from the move rule, which is the only thing
//!    it exists to be.
//!
//! # Why this matters beyond the bound
//!
//! DEV-232 enforces the move rule for a field read through a reference. Its first attempt was
//! reverted because the code it forbids had **no legal spelling**: `impl<T: Copy> Iterator for
//! Repeat<T>` could not be instantiated at a primitive. This repair is what let DEV-232 re-land
//! without deleting a shape of generic code from the language, and both generic-iterator tests in
//! the corpus now carry that bound.
//!
//! # Why these negative controls
//!
//! Both halves widen what is accepted, so the risk is accepting too much. A non-`Copy` argument
//! must still be refused at the call site, and an UNBOUNDED parameter must still be refused in the
//! body — otherwise the second half would have re-opened DEV-232 while claiming to unblock it.

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

/// Both halves at once: the bound is satisfied by primitives at the call site, AND the body is
/// allowed to move out of the reference because the parameter declares it.
#[test]
fn a_primitive_satisfies_a_copy_bound_and_the_body_may_rely_on_it() {
    let out = run(
        "\
fn take<T: Copy>(r: &T) -> T { *r }
fn main() {
    println(take(&5i64));
    println(take(&true));
    println(take(&2.5f64));
}
",
        "dev234_primitives",
    );
    assert!(out.contains('5'), "{out:?}");
    assert!(out.contains("true"), "{out:?}");
}

#[test]
fn a_user_struct_with_impl_copy_satisfies_the_bound() {
    let out = run(
        "\
struct P { x: Int64 }
impl Copy for P {}
fn take<T: Copy>(r: &T) -> T { *r }
fn main() { let p = P { x: 9i64 }; println(take(&p).x); }
",
        "dev234_user_copy",
    );
    assert!(out.contains('9'), "{out:?}");
}

/// The shape DEV-232 forbids, made legal by the bound. This is the whole point of the repair: the
/// move rule can be enforced because this spelling exists.
#[test]
fn a_generic_field_move_is_legal_under_a_copy_bound() {
    assert_eq!(
        errors(
            "\
struct Holder<T> { item: T }
fn get<T: Copy>(h: &Holder<T>) -> T { h.item }
fn main() { let h = Holder { item: 4i64 }; println(get(&h)); }
",
            "dev234_field_under_bound"
        ),
        Vec::<String>::new(),
        "DEV-232's rejection must have a legal spelling, and this is it"
    );
}

// -- controls -----------------------------------------------------------------------------------

#[test]
fn a_non_copy_argument_is_still_refused_at_the_call_site() {
    let errs = errors(
        "\
fn take<T: Copy>(r: &T) -> T { *r }
fn main() { let s = String::from(\"x\"); println(take(&s).as_str()); }
",
        "dev234_control_noncopy",
    );
    assert!(
        errs.iter().any(|e| e.contains("E0500")),
        "String is not Copy and the bound must still say so: {errs:?}"
    );
}

/// The control that keeps the second half honest. If a declared bound were not required — if any
/// `Ty::Param` were treated as `Copy` — this would compile, and DEV-232 would be re-opened by the
/// very change that claims to unblock it.
#[test]
fn an_unbounded_parameter_is_still_refused_in_the_body() {
    let errs = errors(
        "\
fn take<T>(r: &T) -> T { *r }
fn main() { println(take(&5i64)); }
",
        "dev234_control_unbounded",
    );
    assert!(
        errs.iter().any(|e| e.contains("E0100")),
        "an unbounded T is not Copy, so the body may not move out of the reference: {errs:?}"
    );
}

#[test]
fn an_unbounded_parameter_is_still_refused_for_a_field_too() {
    let errs = errors(
        "\
struct Holder<T> { item: T }
fn get<T>(h: &Holder<T>) -> T { h.item }
fn main() { let h = Holder { item: 4i64 }; println(get(&h)); }
",
        "dev234_control_unbounded_field",
    );
    assert!(
        !errs.is_empty(),
        "DEV-232's rule must still apply when the bound is absent: {errs:?}"
    );
}
