//! **AS4 property 3 — "does this nominal have a user destructor?" has one authority.**
//!
//! The answer is `Res::CoreTrait(CoreTrait::Drop)` — **resolved identity**, never the trait's
//! spelling. CD-379 settled that rule for `Display`: a user trait merely *named* `Display` does not
//! satisfy the bound. DEV-210 is the same defect, found in the borrow checker, which asked whether
//! the written trait name `.ends_with("Drop")`.
//!
//! The consequence was a **reduction in valid language capability**: a legal partial move out of a
//! field was refused with E0100 on any type implementing a user trait whose name happened to end in
//! those four letters. Nothing in the suite noticed, because no test had ever declared such a trait.
//!
//! The repair was not to fix the string comparison. `copy_eligible_types` had already computed
//! exactly this set, by identity, and kept it private — so the borrow checker had written a second,
//! weaker answer to a question the checker was already answering correctly. Publishing
//! `nominals_with_destructor` removes the incentive to write a third.

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn errors(source: &str) -> Vec<String> {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, parse_diags) = parse(&file, ParseMode::Program);
    assert!(parse_diags.is_empty(), "parse: {parse_diags:?}");
    let (hir, resolve_diags) = resolve(&ast, file.clone());
    assert!(resolve_diags.is_empty(), "resolve: {resolve_diags:?}");
    typecheck::analyze(&hir)
        .diagnostics
        .into_iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .map(|d| format!("{}: {}", d.code.clone().unwrap_or_default(), d.message))
        .collect()
}

const MOVE_A_FIELD: &str =
    " fn main() { let s = S { a: String::from(\"x\") }; let t = s.a; println(t); }";

/// **The control.** A real destructor must still refuse the partial move — otherwise the test
/// below would pass against a borrow checker that had simply stopped checking.
#[test]
fn a_real_destructor_still_refuses_a_partial_move() {
    let errs = errors(&format!(
        "struct S {{ a: String }} impl Drop for S {{ fn drop(&mut self) {{}} }}{MOVE_A_FIELD}"
    ));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("E0100"), "{}", errs[0]);
}

/// No destructor: the partial move is legal, and is accepted.
#[test]
fn a_type_without_a_destructor_permits_the_partial_move() {
    assert!(
        errors(&format!("struct S {{ a: String }}{MOVE_A_FIELD}")).is_empty(),
        "a type with no `Drop` impl must permit a partial move out of a field"
    );
}

/// **DEV-210.** A user trait whose name merely ends with `Drop` is not `Drop`.
#[test]
fn a_user_trait_named_like_drop_is_not_a_destructor() {
    let errs = errors(&format!(
        "struct S {{ a: String }} trait MyDrop {{ fn go(&self); }} \
         impl MyDrop for S {{ fn go(&self) {{}} }}{MOVE_A_FIELD}"
    ));
    assert!(
        errs.is_empty(),
        "`impl MyDrop for S` gives `S` no destructor, so the partial move is legal. \
         Identifying `Drop` by spelling refused valid Core: {errs:?}"
    );
}

/// The same rule from the other direction: a trait whose name merely *contains* `Drop` is also not
/// `Drop`, so the defect cannot be half-fixed by tightening the string test.
#[test]
fn a_user_trait_containing_drop_in_its_name_is_not_a_destructor() {
    let errs = errors(&format!(
        "struct S {{ a: String }} trait DropLike {{ fn go(&self); }} \
         impl DropLike for S {{ fn go(&self) {{}} }}{MOVE_A_FIELD}"
    ));
    assert!(errs.is_empty(), "{errs:?}");
}

/// **The authority itself**, asserted directly rather than only through a diagnostic: the set
/// contains the nominal with a real `Drop` impl and nothing else.
#[test]
fn the_published_set_is_keyed_by_identity() {
    let source = "struct Real { a: Int32 } \
                  struct Fake { a: Int32 } \
                  impl Drop for Real { fn drop(&mut self) {} } \
                  trait MyDrop { fn go(&self); } \
                  impl MyDrop for Fake { fn go(&self) {} } \
                  fn main() {}";
    let file = Arc::new(SourceFile::new("test.stark", source));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let set = typecheck::nominals_with_destructor(&hir);
    assert_eq!(
        set.len(),
        1,
        "exactly one nominal declares a destructor; got {set:?}"
    );
}

/// Enums are nominals too, and the borrow checker's local lookup handles both — so the authority
/// must cover both or the consolidation is only half done.
#[test]
fn an_enum_destructor_is_recognised() {
    let errs = errors(
        "enum E { A(String), B } impl Drop for E { fn drop(&mut self) {} } \
         fn main() { let e = E::A(String::from(\"x\")); \
         match e { E::A(s) => println(s), E::B => println(\"b\") } }",
    );
    assert!(
        !errs.is_empty(),
        "moving a payload out of a `Drop` enum must still be refused"
    );
}
