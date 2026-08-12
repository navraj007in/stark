//! **AC4-F3's repair: the bound-satisfaction arms that no test executed.**
//!
//! `typecheck::traits::satisfies_bound_identity` decides nine bound names across five semantic arms.
//! An arm census during WP-ARCH-CLOSE AC4 found the whole test suite reaching it **26 times and
//! exercising two arms** — `Ty::Primitive` and `Ty::Struct`/`Ty::Enum` — for `Eq`, `Hash` and one
//! `Ord`. Three arms executed **zero** times:
//!
//! ```text
//! Ty::Ref     whether `&T` forwards Eq/Ord/Clone/Hash/Display to `T`
//! Ty::Core    Clone/Display/Hash/Eq/Ord/Default over Core types, AND the eight-member
//!             Iterator membership list
//! Ty::Param   a bound discharged by the ENCLOSING function's declared bounds -- DEV-067(a),
//!             whose absence once failed simple recursion with E0500
//! ```
//!
//! **Arm-level mutations of all three SURVIVED, and that said nothing**: they were unreachable, not
//! undetected. A survival is not evidence of a missing control until reachability is demonstrated —
//! the binding rule AC4 added to the shared-fate register.
//!
//! These cases exist to make the arms execute. Each was confirmed by probe to reach the arm it
//! names before being written here, because a test that merely *looks* like it exercises an arm is
//! how the gap arose in the first place.

mod support;

use starkc::diag::Severity;
use std::sync::Arc;

fn errors(name: &str, src: &str) -> Vec<String> {
    let file = Arc::new(starkc::source::SourceFile::new(name, src.to_string()));
    let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    starkc::typecheck::analyze(&hir)
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| format!("{} {}", d.code.as_deref().unwrap_or("-"), d.message))
        .collect()
}

/// **`Ty::Ref` forwarding, `Display`.** Reaches the arm because `T` is INSTANTIATED with a
/// reference — `show(&n)` makes the bound check ask whether `&Int32` satisfies `Display`, which is
/// the forwarding question. Writing `fn show<T: Display>(v: &T)` does NOT reach it: the body
/// dereferences and the check sees `T`.
#[test]
fn a_reference_forwards_display_to_its_referent() {
    assert!(
        errors(
            "arm_ref_display.stark",
            r#"
fn show<T: Display>(v: T) { println(v); }
fn main() { let n: Int32 = 1; show(&n); }
"#,
        )
        .is_empty(),
        "`&Int32` must satisfy `Display` by forwarding to `Int32`. This is the Ty::Ref arm, and \
         removing `Display` from its forwarding list is the mutation this test exists to kill."
    );
}

/// The same arm for `Eq`, because the forwarding list names five traits and one of them being
/// dropped is the realistic defect.
#[test]
fn a_reference_forwards_eq_to_its_referent() {
    assert!(
        errors(
            "arm_ref_eq.stark",
            r#"
fn same<T: Eq>(a: T, b: T) -> Bool { a == b }
fn main() { let x: Int32 = 1; let y: Int32 = 1; println(same(&x, &y)); }
"#,
        )
        .is_empty(),
        "`&Int32` must satisfy `Eq` by forwarding"
    );
}

/// **`Ty::Core`, `Clone`.** A Core type satisfies `Clone` when all its arguments do — the arm
/// recurses, and nothing executed it.
#[test]
fn a_core_type_satisfies_clone_through_its_arguments() {
    assert!(
        errors(
            "arm_core_clone.stark",
            r#"
fn dup<T: Clone>(v: T) -> T { v.clone() }
fn main() {
    let v: Vec<Int32> = Vec::new();
    let w = dup(v);
    println(w.len());
}
"#,
        )
        .is_empty(),
        "`Vec<Int32>` must satisfy `Clone` because `Int32` does. This is the Ty::Core arm."
    );
}

/// **`Ty::Param` discharge — DEV-067(a).**
///
/// A generic function calling another generic function with a bounded parameter. The callee's
/// obligation is discharged by the CALLER's own declared bound (TYPE-GENERIC-001). This arm did not
/// exist once, and its absence failed exactly this shape — including simple recursion — with
/// `E0500 "type 'T' does not satisfy trait bound 'Ord'"` even though `T: Ord` was written right
/// there.
#[test]
fn an_enclosing_bound_discharges_a_callees_obligation() {
    assert!(
        errors(
            "arm_param_discharge.stark",
            r#"
fn inner<U: Ord>(a: U, b: U) -> Bool { a < b }
fn outer<T: Ord>(a: T, b: T) -> Bool { inner(a, b) }
fn main() { println(outer(1i32, 2i32)); }
"#,
        )
        .is_empty(),
        "`T: Ord` on the caller must discharge `U: Ord` on the callee (DEV-067(a), \
         TYPE-GENERIC-001). This is the Ty::Param arm."
    );
}

/// The negative half of the same arm: a parameter with **no** bound must NOT discharge the
/// obligation. Without this, an arm that answered `true` unconditionally would pass every test
/// above.
#[test]
fn a_parameter_without_the_bound_does_not_discharge_it() {
    let errs = errors(
        "arm_param_missing.stark",
        r#"
fn inner<U: Ord>(a: U, b: U) -> Bool { a < b }
fn outer<T>(a: T, b: T) -> Bool { inner(a, b) }
fn main() { println(outer(1i32, 2i32)); }
"#,
    );
    assert!(
        !errs.is_empty(),
        "`T` carries no `Ord` bound, so it cannot discharge the callee's. An arm that answered \
         `true` unconditionally would satisfy every positive case above and only this one catches \
         it."
    );
}

/// **The `Ty::Primitive` matrix, at the one token that separates `Ord` from `Eq`.**
///
/// DEV-075's matrix: `Ord` is `Eq`'s set MINUS `Bool`. `Char` is ordered; `Bool` is not. The arm
/// executed once in the whole suite before this, and not with `Bool` — so admitting `Bool` as `Ord`,
/// a one-token edit directly below the `Eq` arm, changed nothing observable.
#[test]
fn bool_satisfies_eq_but_not_ord() {
    assert!(
        errors(
            "arm_prim_bool_eq.stark",
            r#"
fn same<T: Eq>(a: T, b: T) -> Bool { a == b }
fn main() { println(same(true, true)); }
"#,
        )
        .is_empty(),
        "DEV-075: `Bool` IS in the `Eq` set"
    );

    let errs = errors(
        "arm_prim_bool_ord.stark",
        r#"
fn less<T: Ord>(a: T, b: T) -> Bool { a < b }
fn main() { println(less(true, false)); }
"#,
    );
    assert!(
        !errs.is_empty(),
        "DEV-075: `Ord` is `Eq`'s set MINUS `Bool`. Accepting `Bool` here is one token's \
         difference from the `Eq` arm above it, and nothing else in the suite catches it."
    );
}

/// `Char` is the other side of that same token: ordered, where `Bool` is not. Without it, an arm
/// that excluded both would pass the test above.
#[test]
fn char_is_ordered() {
    assert!(
        errors(
            "arm_prim_char_ord.stark",
            r#"
fn less<T: Ord>(a: T, b: T) -> Bool { a < b }
fn main() { println(less('a', 'b')); }
"#,
        )
        .is_empty(),
        "DEV-075: `Char` IS ordered. An arm excluding both Bool and Char would satisfy the \
         previous test while being wrong."
    );
}

/// **The `Ty::Core` Iterator membership list**, a closed set of eight core iterator types that was
/// executed zero times. `VecIter` is its most-used member.
///
/// Uses `it.next()` rather than a `for` loop deliberately: `for _x in it` where `it: I` with
/// `I: Iterator` is refused with `E0001 "for-loop requires an iterable value, found 'I'"` — a real
/// language limitation, found while writing this. The bound check is what this test is for, so it
/// exercises the bound through the method the language does support.
#[test]
fn a_vec_cursor_satisfies_iterator() {
    assert!(
        errors(
            "arm_core_iterator.stark",
            r#"
fn advance<I: Iterator>(mut it: I) -> Bool {
    let _first = it.next();
    true
}
fn main() {
    let mut v: Vec<Int64> = Vec::new();
    v.push(1i64);
    println(advance(v.iter()));
}
"#,
        )
        .is_empty(),
        "`VecIter` must satisfy `Iterator`. The membership list is closed and dropping its \
         most-used member is the realistic defect."
    );
}
