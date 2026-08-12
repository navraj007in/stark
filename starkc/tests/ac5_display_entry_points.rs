//! **WP-ARCH-CLOSE AC5 finding: `println` and interpolation enforce the `Display` obligation
//! differently on a generic parameter.**
//!
//! `Display` has two entry points, and the repo already names them as two — `typecheck/body.rs`
//! calls interpolation *"the SECOND `Display` entry point"* (AS3 Boundary 4). What AC5 measured is
//! that the two apply **different policies** to the same rule:
//!
//! ```text
//! f"{x}"      on `fn show<T>(..)`            REFUSED at the definition, E0306
//! println(x)  on `fn show<T>(..)`            ACCEPTED; the obligation is deferred to instantiation
//! ```
//!
//! Deferring to instantiation is a defensible policy. Enforcing at the definition is a defensible
//! policy. **Applying one to each entry point is neither**, and it has a visible consequence: when
//! the deferred obligation is not met, the program is refused by MIR lowering rather than by a
//! front-end diagnostic — the accepted-but-unbuildable shape this repo tracks as the E0105 class
//! and audits in `layer_audit.rs`.
//!
//! These tests CHARACTERIZE the current behaviour. They assert what is, not what should be, so the
//! divergence cannot change silently while the owner decides which policy is right.

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

/// Interpolation refuses a generic with no `Display` bound, at the definition.
#[test]
fn interpolation_enforces_the_obligation_where_it_is_written() {
    let codes = front_end_errors(
        "ac5_interp.stark",
        r#"
fn show<T>(x: T) { println(f"{x}"); }
fn main() { show(1i32); }
"#,
    );
    assert_eq!(
        codes,
        vec!["E0306"],
        "interpolation must refuse a parameter with no bound providing Display, at the definition"
    );
}

/// `println` does not — with no bound at all, the definition is accepted.
#[test]
fn println_defers_the_obligation_to_the_instantiation() {
    let codes = front_end_errors(
        "ac5_println.stark",
        r#"
fn show<T>(x: T) { println(x); }
fn main() { show(1i32); }
"#,
    );
    assert!(
        codes.is_empty(),
        "CHARACTERIZATION, not an endorsement: `println` on an unbounded parameter is accepted at \
         the definition and the obligation is discharged at the instantiation. Interpolation \
         refuses the same shape. If this changes, the divergence has been resolved — say which \
         policy won. Got {codes:?}"
    );
}

/// **The consequence: an accepted program MIR cannot build.**
///
/// A user trait spelled `Display` satisfies the written bound (correctly — it is the trait the
/// user declared and resolution finds it), and `println` asks nothing further at the definition.
/// The instantiation then has no core `Display`, and the refusal arrives from MIR lowering rather
/// than as a front-end diagnostic.
#[test]
fn the_deferred_obligation_fails_at_lowering_not_at_the_front_end() {
    let source = r#"
trait Display { fn unrelated(&self) -> Int32; }
struct P { a: Int32 }
impl Display for P { fn unrelated(&self) -> Int32 { 7 } }
fn show<T: Display>(x: T) { println(x); }
fn main() { show(P { a: 1 }); }
"#;
    assert!(
        front_end_errors("ac5_layer.stark", source).is_empty(),
        "the front end accepts this program"
    );

    let front = support::differential::front_end("ac5_layer.stark", source);
    let refusal = starkc::mir::lower::lower_program(&front.hir, &front.tables, front.file.clone())
        .err()
        .map(|e| e.what)
        .expect("MIR must refuse what the front end accepted -- that is the finding");
    assert!(
        refusal.contains("Display::fmt not found"),
        "the refusal must be MIR's own Display lookup failing: {refusal}"
    );
}
