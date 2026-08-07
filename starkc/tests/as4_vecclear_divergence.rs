//! **AS4 — the reachable consequence of the two precise drop rules disagreeing.**
//!
//! `mir::lower::as4_drop_predicate_inventory` measures that `lower::ty_requires_drop_glue` and
//! `verify::requires_drop_glue` disagree on 14 `MirTy::Core` variants, always `lower=false
//! verify=true`, and that only `CharsIter` and `File` are reachable shapes. This file establishes
//! what that costs on a real program, because "two predicates disagree" and "a valid program is
//! refused" are different claims and only the second matters to a user.
//!
//! ```text
//! let mut v: Vec<CharsIter> = Vec::new();
//! v.push(s.chars());
//! v.clear();
//! ```
//!
//! | Stage | Verdict |
//! | --- | --- |
//! | checker | accepts |
//! | HIR interpreter | runs it, prints `0` |
//! | MIR lowering | lowers it, emitting the fast `VecClear` |
//! | MIR verifier | **rejects — MIR-0016** |
//!
//! **Why the fast path is the whole mechanism.** Lowering emits `VecClear` only when it believes
//! the element needs no drop glue; a droppable element takes a different path that drops each
//! element. `Vec<String>::clear()` and `Vec<Vec<Int32>>::clear()` emit no `VecClear` at all and
//! pass. So MIR-0016 is the guard on that fast path, and the disagreement puts lowering on one side
//! of the guard and the verifier on the other.
//!
//! **This is a characterization, not a repair.** Making the two agree changes which programs the
//! compiler accepts, so it is a behavioural correction owing its own decision record (AS4 work item
//! 5). Pinned so the current behaviour cannot change silently while that decision is pending —
//! and so that when it is taken, this test is the thing that flips.

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
fn the_checker_and_interpreter_accept_what_the_verifier_refuses() {
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

    // And the verifier refuses it.
    match pipeline(CLEAR_CHARS_ITER) {
        Outcome::VerifierRejected(error) => {
            assert!(
                error.contains("MIR-0016") && error.contains("CharsIter"),
                "the refusal must be MIR-0016 on the element type, got: {error}"
            );
        }
        Outcome::Ok { .. } => panic!(
            "the verifier now ACCEPTS `Vec<CharsIter>::clear()`. If that was a deliberate \
             correction, it needed a decision record and this test should have been the thing \
             that flipped — update it in the same change (AS4 work item 5)."
        ),
        Outcome::CheckerRejected => panic!("the checker rejected it; the fixture has drifted"),
        Outcome::LoweringRefused(what) => panic!("lowering refused it: {what}"),
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
