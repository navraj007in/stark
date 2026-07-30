//! **A12 — the storage-end shape matrix (`DEFECT-C788-LOOP-TEMP`).**
//!
//! Every case here is a place whose storage is emptied **piecewise** and then reused across a loop
//! back edge. Before A12 each one either aborted at runtime with
//!
//! ```text
//! generated-code invariant violated: write to a live slot
//! ```
//!
//! or failed to build. The amendment cites this matrix; this file is it, so the claim is checkable
//! from a clone rather than resting on a session's scratch file.
//!
//! # What each case asserts, and why it is not "does it run"
//!
//! Each case requires the **MIR interpreter and the native backend to agree** on stdout and exit
//! status. Running is not enough: the destructor in `Res` prints, so the output encodes how many
//! times each value was destroyed and in what order. A storage end that silently abandoned a live
//! value would still *run* — it would print one line fewer, and only a comparison catches that.
//!
//! # The one that mattered most
//!
//! [`question_mark_in_a_loop`] is the case a sixteen-shape `match` matrix did not contain and a real
//! package did. `?` builds its own scrutinee temporary instead of going through `lower_match`, so
//! the storage end added for match arms never covered it. It was found by requalifying
//! `stark-json`, whose parser is `?` in loops throughout — the package built, and its native binary
//! aborted on the first parse.

mod support;

use starkc::mir::lower::lower_program;
use support::differential::{
    canonical_form, first_difference, front_end, run_mir, run_native, rustc_available,
};

/// `Res` prints its id when destroyed, so the output is a destruction log, not just a result.
const PRELUDE: &str = r#"
struct Res { id: Int32 }

impl Drop for Res {
    fn drop(&mut self) {
        print(self.id);
    }
}

enum Maybe { Some(Res), None }
enum Mixed { Payload(Res), Plain(Int32), Empty }
enum Pair { Both(Res, Res), Neither }

fn make(i: Int32) -> Maybe { Maybe::Some(Res { id: i }) }
fn empty() -> Maybe { Maybe::None }
fn plain(i: Int32) -> Mixed { Mixed::Plain(i) }
fn nothing() -> Mixed { Mixed::Empty }
fn both(i: Int32) -> Pair { Pair::Both(Res { id: i }, Res { id: i }) }
fn opt(i: Int32) -> Option<Res> { Some(Res { id: i }) }
fn res(i: Int32) -> Result<Res, Int32> { Ok(Res { id: i }) }
fn pair(i: Int32) -> (Res, Int32) { (Res { id: i }, i) }
fn eat(r: Res) -> Int32 { r.id }
"#;

/// Lower `body`, then require the MIR interpreter and the native backend to observe identically.
fn agree(tag: &str, body: &str) {
    let source = format!("{PRELUDE}\n{body}\n");
    let front = front_end(tag, &source);
    let program = lower_program(&front.hir, &front.tables, front.file.clone())
        .unwrap_or_else(|e| panic!("{tag}: lowering failed: {}", e.what));
    let mir = run_mir(tag, &program);
    if !rustc_available() {
        eprintln!("SKIP native half of {tag}: no rustc");
        return;
    }
    let native = run_native(tag, tag, &program);
    if let Some(field) = first_difference(&mir, &native) {
        panic!(
            "{tag}: MIR and native disagree on `{field}`.\n--- mir ---\n{}\n--- native ---\n{}",
            canonical_form(&mir),
            canonical_form(&native)
        );
    }
}

/// Wrap `inner` in `while i < 3 { .. }` — the back edge is what turns a stale storage state from
/// invisible into fatal.
fn in_a_loop(inner: &str) -> String {
    format!("fn main() {{\n    let mut i: Int32 = 0;\n    while i < 3 {{\n{inner}\n        i = i + 1;\n    }}\n}}")
}

// ------------------------------------------------- the shape the defect was found through --

#[test]
fn a_match_arm_that_moves_a_non_copy_payload_out() {
    agree(
        "payload_arm",
        &in_a_loop(
            "        match make(i) {\n            Maybe::Some(r) => { print(r.id); }\n            Maybe::None => { }\n        }",
        ),
    );
}

/// The opposite storage state: the arm moves *nothing*, so the slot stays whole rather than becoming
/// partial. It needs the other storage end, and a single unconditional form got this wrong.
#[test]
fn a_match_arm_that_moves_nothing_out() {
    agree(
        "unit_arm",
        &in_a_loop(
            "        match empty() {\n            Maybe::Some(r) => { print(r.id); }\n            Maybe::None => { print(9); }\n        }",
        ),
    );
}

/// A `Copy` payload on a droppable enum — whole storage again, for a different reason.
#[test]
fn a_copy_payload_arm_of_a_droppable_enum() {
    agree(
        "copy_arm",
        &in_a_loop(
            "        match plain(i) {\n            Mixed::Payload(r) => { print(r.id); }\n            Mixed::Plain(n) => { print(n); }\n            Mixed::Empty => { }\n        }",
        ),
    );
}

#[test]
fn an_empty_variant_of_a_droppable_enum() {
    agree(
        "empty_variant",
        &in_a_loop(
            "        match nothing() {\n            Mixed::Payload(r) => { print(r.id); }\n            Mixed::Plain(n) => { print(n); }\n            Mixed::Empty => { print(7); }\n        }",
        ),
    );
}

/// Multi-field payload: the C6.1c decomposition temporary is a **second** compiler-generated
/// temporary on the same reassignment path. Missing it kept this case failing after the
/// single-field one was fixed.
#[test]
fn a_multi_field_payload_uses_a_decomposition_temporary() {
    agree(
        "multi_field",
        &in_a_loop(
            "        match both(i) {\n            Pair::Both(a, b) => { print(a.id); print(b.id); }\n            Pair::Neither => { }\n        }",
        ),
    );
}

#[test]
fn option_of_a_droppable_payload() {
    agree(
        "option",
        &in_a_loop(
            "        match opt(i) {\n            Some(r) => { print(r.id); }\n            None => { }\n        }",
        ),
    );
}

#[test]
fn result_of_a_droppable_payload() {
    agree(
        "result",
        &in_a_loop(
            "        match res(i) {\n            Ok(r) => { print(r.id); }\n            Err(e) => { print(e); }\n        }",
        ),
    );
}

#[test]
fn a_nested_match_inside_an_arm() {
    agree(
        "nested",
        &in_a_loop(
            "        match opt(i) {\n            Some(r) => {\n                match res(r.id) {\n                    Ok(inner) => { print(inner.id); }\n                    Err(e) => { print(e); }\n                }\n            }\n            None => { }\n        }",
        ),
    );
}

#[test]
fn a_wildcard_arm() {
    agree(
        "wildcard",
        &in_a_loop(
            "        match make(i) {\n            Maybe::None => { print(5); }\n            _ => { print(6); }\n        }",
        ),
    );
}

#[test]
fn a_catch_all_binding_arm() {
    agree(
        "binding_arm",
        &in_a_loop(
            "        match make(i) {\n            Maybe::None => { print(5); }\n            other => { print(4); }\n        }",
        ),
    );
}

/// The payload leaves the arm entirely, so the destructor must run in the callee — once per
/// iteration, not twice and not never.
#[test]
fn a_payload_moved_out_through_a_call() {
    agree(
        "moved_through_call",
        &in_a_loop(
            "        match make(i) {\n            Maybe::Some(r) => { print(eat(r)); }\n            Maybe::None => { }\n        }",
        ),
    );
}

/// **A user local, with no `match` in the program at all.** The case that corrected the recorded
/// scope: `DEFECT-C788-LOOP-TEMP` was written up as affecting compiler temporaries, and an ordinary
/// binding with one field moved out fails identically.
#[test]
fn a_user_local_with_one_field_moved_out() {
    agree(
        "user_local",
        &in_a_loop(
            "        let t: (Res, Int32) = pair(i);\n        let a: Res = t.0;\n        print(a.id);",
        ),
    );
}

#[test]
fn two_scrutinee_temporaries_in_one_body() {
    agree(
        "two_temps",
        &in_a_loop(
            "        match make(i) {\n            Maybe::Some(r) => { print(r.id); }\n            Maybe::None => { }\n        }\n        match make(i) {\n            Maybe::Some(r) => { print(r.id); }\n            Maybe::None => { }\n        }",
        ),
    );
}

#[test]
fn continue_out_of_the_loop_body() {
    agree(
        "continue",
        &in_a_loop(
            "        match make(i) {\n            Maybe::Some(r) => { print(r.id); }\n            Maybe::None => { }\n        }\n        if i == 1 { i = i + 1; continue; }",
        ),
    );
}

#[test]
fn break_out_of_an_arm() {
    agree(
        "break",
        &in_a_loop(
            "        match make(i) {\n            Maybe::Some(r) => { if r.id == 1 { break; } print(r.id); }\n            Maybe::None => { }\n        }",
        ),
    );
}

#[test]
fn a_match_used_as_an_expression() {
    agree(
        "match_expr",
        &in_a_loop(
            "        let v: Int32 = match make(i) {\n            Maybe::Some(r) => { r.id }\n            Maybe::None => { 0 }\n        };\n        print(v);",
        ),
    );
}

// --------------------------------------------------------- the case a match matrix missed --

/// **`?` inside a loop.** `lower_try` builds its own scrutinee temporary, so it was not covered by
/// the storage end added for match arms, and the `Ok` path *continues executing* — so the next
/// iteration wrote over a partially moved slot.
///
/// Found by requalifying `stark-json` rather than by extending this matrix, which is the honest
/// provenance: sixteen deliberately chosen `match` shapes did not include it, and one real package
/// hit it on its first parse.
#[test]
fn question_mark_in_a_loop() {
    agree(
        "question_mark",
        "fn get(i: Int32) -> Result<Res, Int32> { Ok(Res { id: i }) }\n\
         fn run() -> Result<Int32, Int32> {\n\
        \x20   let mut i: Int32 = 0;\n\
        \x20   let mut total: Int32 = 0;\n\
        \x20   while i < 3 {\n\
        \x20       let r: Res = get(i)?;\n\
        \x20       total = total + r.id;\n\
        \x20       i = i + 1;\n\
        \x20   }\n\
        \x20   Ok(total)\n\
         }\n\
         fn main() {\n\
        \x20   match run() {\n\
        \x20       Ok(v) => { print(v); }\n\
        \x20       Err(e) => { print(e); }\n\
        \x20   }\n\
         }",
    );
}

/// `?` in a loop on the **propagating** path — the error is taken, so the scrutinee's `Err` payload
/// is what moves out. The `Ok` and `Err` paths empty the temporary differently and each needs its
/// own storage end.
#[test]
fn question_mark_in_a_loop_that_propagates() {
    agree(
        "question_mark_err",
        "fn get(i: Int32) -> Result<Res, Int32> { if i == 2 { Err(9) } else { Ok(Res { id: i }) } }\n\
         fn run() -> Result<Int32, Int32> {\n\
        \x20   let mut i: Int32 = 0;\n\
        \x20   let mut total: Int32 = 0;\n\
        \x20   while i < 5 {\n\
        \x20       let r: Res = get(i)?;\n\
        \x20       total = total + r.id;\n\
        \x20       i = i + 1;\n\
        \x20   }\n\
        \x20   Ok(total)\n\
         }\n\
         fn main() {\n\
        \x20   match run() {\n\
        \x20       Ok(v) => { print(v); }\n\
        \x20       Err(e) => { print(e); }\n\
        \x20   }\n\
         }",
    );
}

/// `?` on an `Option`, whose propagating path carries no payload at all — the third storage state
/// the `?` desugar can leave behind.
#[test]
fn question_mark_on_an_option_in_a_loop() {
    agree(
        "question_mark_option",
        "fn get(i: Int32) -> Option<Res> { if i == 2 { None } else { Some(Res { id: i }) } }\n\
         fn run() -> Option<Int32> {\n\
        \x20   let mut i: Int32 = 0;\n\
        \x20   let mut total: Int32 = 0;\n\
        \x20   while i < 5 {\n\
        \x20       let r: Res = get(i)?;\n\
        \x20       total = total + r.id;\n\
        \x20       i = i + 1;\n\
        \x20   }\n\
        \x20   Some(total)\n\
         }\n\
         fn main() {\n\
        \x20   match run() {\n\
        \x20       Some(v) => { print(v); }\n\
        \x20       None => { print(0); }\n\
        \x20   }\n\
         }",
    );
}
