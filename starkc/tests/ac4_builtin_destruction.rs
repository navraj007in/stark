//! **AC4-F4's control: built-in destruction, observed in the MIR rather than in memory.**
//!
//! The finding: disabling `MirTy::String => true` in `mir::drop_rule::requires_drop_glue_with` — so
//! the most-owned type in the language needs no destruction — passed `three_engine_differential`
//! (129), `native_c6_1_ownership` (24) and `as4_destructor_authority` (6). The only killers were two
//! borrow-origin tests written the previous day that pin local numbering. Detection was incidental.
//!
//! **The cause is structural and the comparator documents it.** A drop-observing case emits a
//! reserved frame *from its own `Drop` impl*; a `String` has no user `Drop` impl, so its destruction
//! cannot emit a frame. **The drop log is incapable of observing built-in destruction** — for every
//! built-in owning type, not just `String`.
//!
//! # Why this observes MIR and not memory, stated plainly
//!
//! The honest control for a leak is a leak detector, and the two candidates do not fit:
//!
//! ```text
//! the Miri lane      runs with `-Zmiri-ignore-leaks`, so it is not a leak detector today. The
//!                    flag is needed because three `should_panic` tests hold heap values when the
//!                    panic aborts -- removing it would fail them for the reason they exist
//! a native run       observing an allocation leak in a GENERATED binary needs a leak-checking
//!                    harness around `stark build` output, which is a work packet, not a test
//! ```
//!
//! So this asserts the **decision** rather than its consequence: if `requires_drop_glue_with` says a
//! `String` owns nothing, lowering emits no `Drop` for it, and that absence is visible in the MIR.
//! **This is a lowering-structure control, not a leak observation, and it should not be cited as
//! one.** It falsifies the mutation, which is what AC4 requires; it does not prove no leak exists.

use starkc::mir::{self, Terminator};
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn lower(name: &str, src: &str) -> mir::MirProgram {
    let file = Arc::new(SourceFile::new(name, src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{name}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{name}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
    let errors: Vec<_> = checked
        .diagnostics
        .iter()
        .filter(|d| d.severity == starkc::diag::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{name}: typecheck: {errors:?}");
    let registered = hir.source_named(&file.name).expect("registered");
    mir::lower::lower_program(&hir, &checked.tables, registered)
        .unwrap_or_else(|e| panic!("{name}: lowering: {}", e.what))
}

/// How many `Drop` terminators the program contains, over locals of the given type.
fn drops_of(program: &mir::MirProgram, want: &mir::MirTy) -> usize {
    let mut n = 0;
    for body in &program.bodies {
        for block in &body.blocks {
            if let Terminator::Drop { place, .. } = &block.terminator.0 {
                if &body.locals[place.local.0 as usize].ty == want {
                    n += 1;
                }
            }
        }
    }
    n
}

/// **A `String` going out of scope must be destroyed.**
///
/// If `requires_drop_glue_with` stops classifying `MirTy::String` as owning, drop elaboration emits
/// no `Drop` for it and this count falls to zero — the leak, made visible one layer before it
/// happens.
#[test]
fn a_string_local_is_destroyed() {
    let program = lower(
        "f4_string.stark",
        r#"
fn main() {
    let s = String::from("owned");
    println(s.len());
}
"#,
    );
    assert!(
        drops_of(&program, &mir::MirTy::String) > 0,
        "a `String` local owns a heap buffer and must be dropped. Zero `Drop` terminators means \
         lowering believes it owns nothing, which is a leak in every engine and observable by \
         none of their drop logs."
    );
}

/// The same for a `String` inside a composite, which reaches the struct-field recursion arm rather
/// than the `MirTy::String` arm directly.
#[test]
fn a_string_inside_a_struct_is_destroyed() {
    let program = lower(
        "f4_struct.stark",
        r#"
struct Holder { name: String, n: Int32 }
fn main() {
    let h = Holder { name: String::from("owned"), n: 1 };
    println(h.n);
}
"#,
    );
    let struct_drops: usize = program
        .bodies
        .iter()
        .flat_map(|b| b.blocks.iter().map(move |bl| (b, bl)))
        .filter(|(b, bl)| match &bl.terminator.0 {
            Terminator::Drop { place, .. } => {
                matches!(
                    b.locals[place.local.0 as usize].ty,
                    mir::MirTy::Struct(_, _)
                )
            }
            _ => false,
        })
        .count();
    assert!(
        struct_drops > 0,
        "a struct owning a `String` field must be dropped; the field recursion arm decides this"
    );
}

/// **The negative control.** A type owning nothing must NOT acquire a drop, or the assertions above
/// would pass on a rule that answered `true` for everything.
#[test]
fn a_scalar_local_is_not_dropped() {
    let program = lower(
        "f4_scalar.stark",
        r#"
fn main() {
    let n: Int32 = 7;
    println(n);
}
"#,
    );
    assert_eq!(
        drops_of(&program, &mir::MirTy::Int32),
        0,
        "an `Int32` owns nothing and must not be dropped. Without this, a rule answering `true` \
         for every type would satisfy both positive cases above."
    );
}
