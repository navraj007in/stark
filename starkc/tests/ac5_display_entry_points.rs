//! **DEV-236 — the `Display` obligation is enforced where the generic is written.**
//!
//! `Display` has two entry points, and the repo names them as two — `typecheck/body.rs` calls
//! interpolation *"the SECOND `Display` entry point"* (AS3 Boundary 4). WP-ARCH-CLOSE AC5 found
//! them applying different policies to one obligation, and CD-401 ruled under CE1 that the
//! definition-time policy is the correct one:
//!
//! > `PRINT-DISPLAY-001` — the print family are *"implementation-provided generic functions …
//! > **not** syntax hooks"*, constrained `T: Display`.
//! > `TYPE-METHOD-003` — each bound resolves to one trait identity, *"and not a spelling"*.
//!
//! These tests were written first as CHARACTERIZATION — asserting the defective behaviour so it
//! could not drift while the owner decided — and say so in their own commit. They are now
//! conformance assertions. Both halves are kept, because the pair is the evidence: interpolation
//! was always right, `println` was not, and the repair had to fix one without weakening the other.
//!
//! **What the repair is NOT.** There is no `if callee == println` anywhere. `type_is_displayable`
//! answered `Ty::Param(_) => true` on a stated assumption — *"discharged by the caller's own
//! bound"* — that was never true, because `builtin_type` types the parameter as a bare inference
//! variable and no obligation was ever attached. It now asks `param_declares_bound`, the authority
//! that already existed, with `Res::CoreTrait(CoreTrait::Display)` as the required identity.

mod support;

use starkc::diag::Severity;
use std::sync::Arc;

fn front_end_errors(name: &str, src: &str) -> Vec<String> {
    let file = Arc::new(starkc::source::SourceFile::new(name, src.to_string()));
    let (ast, pd) = starkc::parser::parse(&file, starkc::parser::ParseMode::Program);
    assert!(pd.is_empty(), "parse: {pd:?}");
    let (hir, rd) = starkc::resolve::resolve(&ast, file.clone());
    assert!(rd.is_empty(), "resolve: {rd:?}");
    starkc::typecheck::analyze(&hir)
        .diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Error)
        .map(|d| d.code.clone().unwrap_or_else(|| "-".to_string()))
        .collect()
}

/// Interpolation refuses a generic with no `Display` bound, at the definition. **Unchanged by the
/// repair, and that is half the point** — the ruling fixed `println` without weakening this.
#[test]
fn interpolation_enforces_the_obligation_where_it_is_written() {
    assert_eq!(
        front_end_errors(
            "ac5_interp.stark",
            r#"
fn show<T>(x: T) { println(f"{x}"); }
fn main() { show(1i32); }
"#,
        ),
        vec!["E0306"],
        "interpolation must refuse a parameter with no bound providing Display, at the definition"
    );
}

/// **DEV-236: `println` now refuses the same shape.** Previously accepted, with the obligation
/// deferred to monomorphisation where it surfaced as a MIR error.
#[test]
fn println_enforces_the_obligation_at_the_definition() {
    assert_eq!(
        front_end_errors(
            "ac5_println_nobound.stark",
            r#"
fn show<T>(x: T) { println(x); }
fn main() { show(1i32); }
"#,
        ),
        vec!["E0500"],
        "PRINT-DISPLAY-001: `println` is an ordinary generic function constrained `T: Display`, so \
         a parameter with no such bound is refused where the generic is WRITTEN"
    );
}

/// An unrelated bound does not discharge it either — the obligation is `Display`, not "some bound".
#[test]
fn an_unrelated_bound_does_not_discharge_the_obligation() {
    assert_eq!(
        front_end_errors(
            "ac5_println_clone.stark",
            r#"
fn show<T: Clone>(x: T) { println(x); }
fn main() { show(1i32); }
"#,
        ),
        vec!["E0500"],
        "`T: Clone` supplies no `Display`"
    );
}

/// **The identity rule, which is why this is not a spelling comparison.**
///
/// A user trait spelled `Display` satisfies the written bound — correctly; it is the trait the user
/// declared and resolution finds it. It does **not** satisfy the print obligation, because
/// `TYPE-METHOD-003` says a bound resolves to a trait IDENTITY *"and not a spelling"*.
///
/// Before the repair this program was accepted and then refused by MIR with
/// `Display::fmt not found for printed type` — an accepted-but-unbuildable program, the E0105 class.
#[test]
fn a_user_trait_spelled_display_does_not_satisfy_the_obligation() {
    assert_eq!(
        front_end_errors(
            "ac5_user_display.stark",
            r#"
trait Display { fn unrelated(&self) -> Int32; }
struct P { a: Int32 }
impl Display for P { fn unrelated(&self) -> Int32 { 7 } }
fn show<T: Display>(x: T) { println(x); }
fn main() { show(P { a: 1 }); }
"#,
        ),
        vec!["E0500"],
        "identity, not spelling: the user's `Display` is a different trait"
    );
}

/// **The over-refusal control, and it caught a real defect during the repair.**
///
/// A bound that IS Core `Display` must still be accepted. The first version of this repair rejected
/// it: `display_checks` is drained in Pass 3, and answering `Ty::Param` from declared bounds made
/// that obligation **scope-sensitive** while it still carried no scope — so the query ran with no
/// generics visible and refused a bound plainly written.
///
/// `DeferredDisplayPlan`'s own doc comment had already stated the rule that was broken: *"a deferred
/// obligation may read resolved types freely, but any scope-sensitive question it asks is a question
/// about a scope that no longer exists. Capture the scope with the obligation."*
#[test]
fn a_core_display_bound_is_accepted() {
    assert!(
        front_end_errors(
            "ac5_core_display.stark",
            r#"
fn show<T: Display>(x: T) { println(x); }
fn main() { show(1i32); }
"#,
        )
        .is_empty(),
        "a written `T: Display` must discharge the obligation -- refusing it is the over-refusal \
         this control exists to catch"
    );
}

/// Concrete types are untouched: the repair changes the GENERIC path only.
#[test]
fn printing_concrete_types_is_unchanged() {
    for (label, src) in [
        ("Int32", "fn main() { println(1i32); }"),
        ("String", "fn main() { println(String::from(\"x\")); }"),
    ] {
        assert!(
            front_end_errors("ac5_concrete.stark", src).is_empty(),
            "{label}: printing a concrete Display type must be unaffected"
        );
    }
}
