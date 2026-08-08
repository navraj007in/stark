//! **AS4 — the reference rule: three predicates, one live disagreement, measured end to end.**
//!
//! `AS0-RB0-PREDICATE-INVENTORY.md` §2 measured the three implementations across 33 samples and
//! found they agree on **every** `MirTy` variant except `FnPtr`:
//!
//! ```text
//! FnPtr(ret &T)     lower::ty_carries_ref = false   emit::ty_carries_reference = true   emit::ty_contains_ref = false
//! FnPtr(param &T)   false                            true                                false
//! ```
//!
//! RB0's Q2 forbade resolving that by fiat and asked the prior question: **do they ask the same
//! thing?** This file answers it the way the drop rule was answered — by reachability and by
//! consumers, not by reading.
//!
//! # Reachability: unlike `Core(File)`, this row is live
//!
//! A `FnPtr` carrying `&T` is constructible and reaches MIR, both taking and returning a reference:
//!
//! ```stark
//! let g: fn(&Int32) -> Int32 = takes;
//! let h: fn(&Int32) -> &Int32 = gives;
//! ```
//!
//! So the disagreement cannot be dismissed as an unreachable legacy shape.
//!
//! # Consumers: three different questions
//!
//! | Predicate | Consumer | The question it actually asks |
//! | --- | --- | --- |
//! | `lower::ty_carries_ref` | Display of a droppable composite | *would emitting this require a lifetime parameter the backend cannot generate (E0106)?* |
//! | `emit::ty_carries_reference` | local initialisation (`emit_bodies`, `emit_places`) | *is there a reference here with no default value to fabricate?* |
//! | `emit::ty_contains_ref` | `derives_for` | *does a structural `PartialEq` derive apply?* |
//!
//! `FnPtr` separates them because a Rust `fn(&T)` is **higher-ranked** — `for<'a> fn(&'a T)` — so it
//! needs no lifetime parameter (the first question: no), while it is still a value with no
//! fabricable default (the second question: yes).
//!
//! **So these are near neighbours like `may_need_drop`, not duplicates like the two precise drop
//! rules were.** The tests below establish that the current answers produce correct programs at
//! every consumer, which is the evidence AS4 needs before naming them apart rather than merging
//! them.

mod support;

use support::differential::agree_completing_with_stdout;

/// A `FnPtr` **taking** a reference: exercises `emit_places`/`emit_bodies` initialisation, where
/// `ty_carries_reference` answers `true`.
#[test]
fn a_function_value_taking_a_reference_runs_in_every_engine() {
    agree_completing_with_stdout(
        "as4_ref_fnptr_param",
        "fn takes(r: &Int32) -> Int32 { *r + 1 }\n\
         fn main() {\n\
         \x20   let g: fn(&Int32) -> Int32 = takes;\n\
         \x20   let n: Int32 = 7;\n\
         \x20   println(g(&n));\n}\n",
        "8\n",
    );
}

/// A `FnPtr` **returning** a reference — the sample RB0 recorded first, and the one where a
/// lifetime would be needed if the type were not higher-ranked.
#[test]
fn a_function_value_returning_a_reference_runs_in_every_engine() {
    agree_completing_with_stdout(
        "as4_ref_fnptr_ret",
        "fn gives(r: &Int32) -> &Int32 { r }\n\
         fn main() {\n\
         \x20   let g: fn(&Int32) -> &Int32 = gives;\n\
         \x20   let n: Int32 = 7;\n\
         \x20   println(*g(&n));\n}\n",
        "7\n",
    );
}

/// The control: a `FnPtr` with no reference anywhere. All three predicates answer `false`, so a
/// failure here would mean the disagreement is not what the other two cases are testing.
#[test]
fn a_function_value_without_references_runs_in_every_engine() {
    agree_completing_with_stdout(
        "as4_ref_fnptr_plain",
        "fn plain(x: Int32) -> Int32 { x + 1 }\n\
         fn main() {\n\
         \x20   let g: fn(Int32) -> Int32 = plain;\n\
         \x20   println(g(1));\n}\n",
        "2\n",
    );
}

/// A function value reassigned before use, so the local genuinely needs the initialisation
/// treatment `emit::ty_carries_reference` selects rather than being written once at its
/// declaration.
#[test]
fn a_reassigned_reference_carrying_function_value_runs_in_every_engine() {
    agree_completing_with_stdout(
        "as4_ref_fnptr_reassigned",
        "fn one(r: &Int32) -> Int32 { *r + 1 }\n\
         fn two(r: &Int32) -> Int32 { *r + 2 }\n\
         fn main() {\n\
         \x20   let n: Int32 = 10;\n\
         \x20   let mut g: fn(&Int32) -> Int32 = one;\n\
         \x20   println(g(&n));\n\
         \x20   g = two;\n\
         \x20   println(g(&n));\n}\n",
        "11\n12\n",
    );
}

/// **The reachability limit, recorded rather than left as an absence.** A `FnPtr` nested inside a
/// composite cannot be *called* through the composite — `indirect callee expression (C4.5)` — so
/// the disagreement cannot be exercised at that depth today.
#[test]
fn a_function_value_inside_a_composite_cannot_be_called_through_it() {
    use starkc::mir::lower::lower_program;
    use starkc::options::LanguageOptions;
    use starkc::session::CompilerSession;
    use starkc::source::SourceFile;
    use std::sync::Arc;

    let source = "fn takes(r: &Int32) -> Int32 { *r }\n\
                  fn main() {\n\
                  \x20   let t: (fn(&Int32) -> Int32, Int32) = (takes, 5);\n\
                  \x20   let n: Int32 = 7;\n\
                  \x20   println(t.0(&n));\n}\n";
    let file = Arc::new(SourceFile::new("test.stark", source));
    let checked = CompilerSession::for_source(file, LanguageOptions::CORE)
        .check()
        .unwrap_or_else(|f| panic!("the checker accepts it:\n{}", f.render()));
    match lower_program(checked.hir(), checked.tables(), checked.root_source()) {
        Err(error) => assert!(
            error.what.contains("indirect callee expression"),
            "expected the indirect-callee limit, got: {}",
            error.what
        ),
        Ok(_) => panic!(
            "a composite-nested function value now lowers. The reference-rule disagreement becomes \
             exercisable one level deeper — re-measure the three predicates at that depth before \
             assuming the current answers still produce correct code."
        ),
    }
}
