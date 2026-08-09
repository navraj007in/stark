//! **AS4 / DEV-195 — `Vec<CharsIter>::clear()` is accepted. The ruling, and its regression.**
//!
//! This file was written as a *refusal* characterization: the checker accepted
//! `Vec<CharsIter>::clear()`, the oracle ran it, lowering emitted the fast `VecClear`, and the
//! verifier alone rejected it with MIR-0016. It is now the *acceptance* regression, because the
//! owner ruled:
//!
//! ```text
//! DEV-195 ruling
//!
//! Core(CharsIter) requires_drop_glue = false.
//!
//! A CharsIter is a borrowed cursor. Destroying it has no STARK-visible destruction
//! action and releases no owned language or provider resource.
//!
//! Therefore Vec<CharsIter>::clear() may use VecClear.
//!
//! The verifier's blanket MirTy::Core(..) => true classification is not authoritative
//! for CharsIter and must not reject that lowering.
//! ```
//!
//! The evidence is semantic, not merely "one side accepts more": `CharsIter` is a borrowed `&str`
//! cursor yielding `Char` by value, and the native runtime is a wrapper around
//! `std::str::Chars<'a>` — the backend emits it as intrinsically borrow-carrying. It owns nothing
//! that destruction could release.
//!
//! **`File` is deliberately NOT covered by this ruling.** It is the other reachable row of the same
//! disagreement and has the opposite ownership: legacy Core `File` is an owning
//! `OwnedResourceHandle` released through the MIR/provider close path, with
//! `drop_plan::plan_for(Core(File))` currently `Noop`. The verifier's `File => true` may therefore
//! be an accidental safety barrier, and it stays. See `AS4-DROP-RULE-MEASUREMENT.md` §8.
//!
//! **Why the fast path is the whole mechanism.** Lowering emits `VecClear` only when it believes
//! the element needs no drop glue; a droppable element takes a different path that drops each
//! element. `Vec<String>::clear()` and `Vec<Vec<Int32>>::clear()` emit no `VecClear` at all. So
//! MIR-0016 guards that fast path, and the disagreement had lowering on one side and the verifier
//! on the other.

mod support;

use starkc::mir::lower::lower_program;
use starkc::mir::verify::verify_program;
use starkc::options::LanguageOptions;
use starkc::session::CompilerSession;
use starkc::source::SourceFile;
use std::sync::Arc;

enum Outcome {
    CheckerRejected,
    LoweringRefused(String),
    VerifierRejected(String),
    Ok { emitted_vec_clear: bool },
}

fn pipeline(source: &str) -> Outcome {
    let file = Arc::new(SourceFile::new("test.stark", source));
    let Ok(checked) = CompilerSession::for_source(file, LanguageOptions::CORE).check() else {
        return Outcome::CheckerRejected;
    };
    let mir = match lower_program(checked.hir(), checked.tables(), checked.root_source()) {
        Ok(mir) => mir,
        Err(e) => return Outcome::LoweringRefused(e.what),
    };
    let emitted_vec_clear = format!("{mir:?}").contains("VecClear");
    match verify_program(&mir) {
        Ok(_) => Outcome::Ok { emitted_vec_clear },
        Err(errors) => Outcome::VerifierRejected(format!("{:?}", errors[0])),
    }
}

const CLEAR_CHARS_ITER: &str = "fn main() {\n\
                                \x20   let s: String = String::from(\"ab\");\n\
                                \x20   let mut v: Vec<CharsIter> = Vec::new();\n\
                                \x20   v.push(s.chars());\n\
                                \x20   v.clear();\n\
                                \x20   println(v.len());\n}\n";

#[test]
fn dev195_vec_of_chars_iter_can_be_cleared_through_every_stage() {
    // The front end and the reference engine agree the program is valid.
    let file = Arc::new(SourceFile::new("test.stark", CLEAR_CHARS_ITER));
    let checked = CompilerSession::for_source(file, LanguageOptions::CORE)
        .check()
        .unwrap_or_else(|f| panic!("the checker must accept it:\n{}", f.render()));
    assert_eq!(
        checked
            .execute_hir()
            .expect("the oracle must run it")
            .output,
        "0\n",
        "cleared, so the length is zero"
    );

    // And every later stage accepts it too. Before the ruling this asserted the opposite.
    match pipeline(CLEAR_CHARS_ITER) {
        Outcome::Ok { emitted_vec_clear } => assert!(
            emitted_vec_clear,
            "a CharsIter element needs no glue, so lowering must take the fast VecClear path"
        ),
        Outcome::VerifierRejected(error) => panic!(
            "DEV-195 REGRESSED: the verifier refuses `Vec<CharsIter>::clear()` again ({error}).\n\
             The ruling is that lowering is right — a borrowed cursor requires no drop glue. If \
             `verify::requires_drop_glue` has been changed back to a blanket `Core(..) => true`, \
             that reverses a decision without a decision record."
        ),
        Outcome::CheckerRejected => panic!("the checker rejected it; the fixture has drifted"),
        Outcome::LoweringRefused(what) => panic!("lowering refused it: {what}"),
    }
}

/// **`File` is the other reachable row, and it is NOT ruled on.** Pinned at the level this test
/// crate can see — the observable behaviour — so a future consolidation cannot quietly extend
/// DEV-195's answer to it. The two variants have opposite ownership: `CharsIter`'s classification
/// was hygiene, `File`'s is safety-critical.
///
/// `Core(File)` is refused by `mir_ty` outright, so no ordinary program reaches it; it is produced
/// only by provider binding (`ResourceBinding::LegacyCore`) in a capability-declared build. That is
/// where `Vec<File>` must be characterized, and until it is, the verifier's `File => true` stays.
#[test]
fn core_file_is_not_reachable_through_ordinary_lowering() {
    match pipeline(
        "fn main() {\n\
         \x20   match File::create(\"/tmp/as4_probe_unused.txt\") {\n\
         \x20       Ok(f) => { println(1); }\n\
         \x20       Err(e) => { println(0); }\n\
         \x20   }\n}\n",
    ) {
        Outcome::LoweringRefused(what) => assert!(
            what.contains("Core(File"),
            "the refusal must be about the File type itself, got: {what}"
        ),
        Outcome::Ok { .. } => panic!(
            "`Core(File)` now lowers through the ordinary path. That makes `Vec<File>::clear()` \
             reachable without a capability-declared build, and the drop classification of `File` \
             must be characterized end to end BEFORE this is allowed — see AS4 §8."
        ),
        Outcome::VerifierRejected(e) => {
            panic!("expected a lowering refusal, got a verifier one: {e}")
        }
        Outcome::CheckerRejected => panic!("the checker must still accept `File::create`"),
    }
}

#[test]
fn a_droppable_element_takes_a_different_path_and_is_unaffected() {
    // The mechanism: lowering emits the fast `VecClear` ONLY when it believes the element needs no
    // glue. These two are droppable, emit no `VecClear`, and never reach MIR-0016 — which is why
    // the disagreement bites on `CharsIter` (no glue per lowering) and not on `String`.
    for (label, source) in [
        (
            "Vec<String>",
            "fn main() {\n    let mut v: Vec<String> = Vec::new();\n\
             \x20   v.push(String::from(\"a\"));\n    v.clear();\n    println(v.len());\n}\n",
        ),
        (
            "Vec<Vec<Int32>>",
            "fn main() {\n    let mut v: Vec<Vec<Int32>> = Vec::new();\n\
             \x20   v.push(Vec::new());\n    v.clear();\n    println(v.len());\n}\n",
        ),
    ] {
        match pipeline(source) {
            Outcome::Ok { emitted_vec_clear } => assert!(
                !emitted_vec_clear,
                "{label}: a droppable element must NOT take the fast VecClear path"
            ),
            other => panic!(
                "{label}: must pass the whole pipeline, got {}",
                match other {
                    Outcome::VerifierRejected(e) => format!("verifier rejection {e}"),
                    Outcome::LoweringRefused(w) => format!("lowering refusal {w}"),
                    _ => "checker rejection".to_string(),
                }
            ),
        }
    }
}

#[test]
fn a_non_droppable_element_takes_the_fast_path_and_passes() {
    // The control that proves MIR-0016 is not simply rejecting every `clear()`.
    match pipeline(
        "fn main() {\n    let mut v: Vec<Int32> = Vec::new();\n\
         \x20   v.push(1);\n    v.clear();\n    println(v.len());\n}\n",
    ) {
        Outcome::Ok { emitted_vec_clear } => {
            assert!(emitted_vec_clear, "an Int32 element takes the fast path")
        }
        _ => panic!("`Vec<Int32>::clear()` must pass the whole pipeline"),
    }
}

/// **DEV-196, answered: `Vec<File>` is unreachable, so `File`'s classification guards nothing.**
///
/// Measured three ways in a **capability-declared** package build (`stark build`, `filesystem`
/// declared), not just through `starkc run`:
///
/// ```text
/// let mut v: Vec<File> = Vec::new(); ... v.clear();   refused: type Core(File, []) (C4.5)
/// match File::create(..) { Ok(f) => { let g: File = f; ... } }  refused: same
/// match File::create(..) { Ok(f) => { println(1); } }           refused: same
/// no File at all                                                 BUILT
/// ```
///
/// `mir_ty` refuses `Core(File)` outright, so the `Ok(f)` binding alone is enough — `File::create`'s
/// `Result<File, IOError>` payload is unlowerable. No source program can construct a `Vec<File>`,
/// let alone clear one.
///
/// Where `Core(File)` IS used — the WP-C7.8.4 provider path — the MIR is built by hand and the
/// handle is closed **explicitly** (`stark_file_close`, `HandleConsumed`), never through drop
/// planning. So `drop_plan::plan_for(Core(File)) = Noop` is consistent with actual use rather than
/// a hole, and the verifier's `File => true` guards a path nothing reaches.
///
/// **Consequence for AS4:** the feared resource-lifecycle defect is not live, and the equivalence
/// `fast_clear_safe(T) == !requires_drop_glue(T)` is not currently *tested* by `File` either — so
/// the fourth predicate (`is_trivially_discardable`) is not needed yet. `File => true` stays
/// because it costs nothing and its real resolution is the HostResource migration.
#[test]
fn dev196_a_vec_of_core_file_cannot_be_lowered_at_all() {
    match pipeline(
        "fn main() {\n\
         \x20   let mut v: Vec<File> = Vec::new();\n\
         \x20   match File::create(\"/tmp/dev196_unit.txt\") {\n\
         \x20       Ok(f) => { v.push(f); }\n\
         \x20       Err(e) => { println(0); }\n\
         \x20   }\n\
         \x20   v.clear();\n\
         \x20   println(v.len());\n}\n",
    ) {
        Outcome::LoweringRefused(what) => assert!(
            what.contains("Core(File"),
            "the refusal must be the File type itself, got: {what}"
        ),
        Outcome::Ok { .. } => panic!(
            "`Vec<File>::clear()` now lowers. DEV-196's safety question becomes LIVE: lowering \
             says `Core(File)` needs no glue, `drop_plan::plan_for` is Noop, and only the \
             verifier's `File => true` stands between this and discarded handles. Characterize \
             the destruction path before allowing it."
        ),
        Outcome::VerifierRejected(e) => panic!(
            "expected a LOWERING refusal, got a verifier one: {e}. If `Core(File)` now lowers and \
             only the verifier refuses, DEV-196 is live — see above."
        ),
        Outcome::CheckerRejected => panic!("the checker must still accept the program"),
    }
}
