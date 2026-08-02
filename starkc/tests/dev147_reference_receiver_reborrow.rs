//! **DEV-147: a receiver that is already a reference is REBORROWED, not moved.**
//!
//! `borrow_{vec,string,map,set}_receiver` all took the same shortcut — when the receiver was
//! already a reference, pass it straight through:
//!
//! ```text
//! if layers > 0 { return self.lower_expr_to_operand(base); }
//! ```
//!
//! `&mut T` is not `Copy`, so "pass through" lowers to a **`Move` of the caller's reference**.
//! Harmless once; fatal in a loop, because the back-edge then sees the parameter possibly-moved:
//!
//! ```text
//! fn push_all(out: &mut Vec<UInt8>, ..) { while i < n { out.push(..); .. } }
//!   -> MIR-0007 push_all@[] bb6: move from possibly-moved place _1[]
//! ```
//!
//! The checker accepted it and the HIR oracle executed it CORRECTLY, so this was
//! accepted-but-unbuildable — the DEV-132/133/146 class, fourth mechanism. It blocked "append into
//! a caller's buffer in a loop", which is the shape of every serializer, encoder and formatter.
//!
//! # The repair
//!
//! `reborrow_reference_receiver` builds `&mut *base`, which is exactly what the `layers == 0` path
//! already does one deref further down. Written once and called from all four sites, for the
//! DEV-128/DEV-130 reason.
//!
//! It is deliberately narrow: a SHARED reference passes through unchanged, because `&T` is `Copy`
//! and reading it moves nothing; and a non-place base passes through, because there is no caller
//! reference to preserve.
//!
//! # Why the negative controls are the important half
//!
//! A reborrow that fired too eagerly would let a caller keep using a reference it had genuinely
//! given away, or hand out exclusive access it never held. The aliasing rules must be exactly as
//! strict as before.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Front end + lowering + MIR verification: `Ok` only when the program would BUILD. The defect was
/// invisible to the checker alone, so a checker-only harness could not have caught it.
fn build(src: &str, tag: &str) -> Result<(), String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    if let Some(first) = checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        return Err(format!(
            "CHECK {} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        ));
    }
    let program =
        lower_program(&hir, &checked.tables, file).map_err(|e| format!("LOWER: {}", e.what))?;
    starkc::mir::verify::verify_program(&program).map_err(|errors| {
        format!(
            "VERIFY {}",
            errors
                .iter()
                .map(|e| format!("{} {}", e.code, e.message))
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    Ok(())
}

fn expect_builds(src: &str, tag: &str) {
    if let Err(why) = build(src, tag) {
        panic!("{tag}: expected a clean build, got: {why}");
    }
}

fn expect_refused(src: &str, tag: &str) -> String {
    match build(src, tag) {
        Ok(()) => panic!("{tag}: expected a refusal, but the program built"),
        Err(why) => why,
    }
}

// -------------------------------------------------------------------- must build --

/// The CD-350 reproducer, found by `stark-http-serialize`.
#[test]
fn a_mutable_vec_receiver_survives_a_loop() {
    expect_builds(
        "fn push_all(out: &mut Vec<UInt8>, text: &str) {\n\
         \x20   let bytes = text.bytes();\n\
         \x20   let n = bytes.len();\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < n {\n\
         \x20       out.push(bytes[i]);\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   push_all(&mut v, \"ab\");\n\
         \x20   println(v.len());\n\
         }\n",
        "vecloop",
    );
}

/// `String`, whose receiver borrower had its own copy of the shortcut with a different comment.
#[test]
fn a_mutable_string_receiver_survives_a_loop() {
    expect_builds(
        "fn fill(out: &mut String, n: Int32) {\n\
         \x20   let mut i = 0;\n\
         \x20   while i < n {\n\
         \x20       out.push('x');\n\
         \x20       i = i + 1;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut s = String::new();\n\
         \x20   fill(&mut s, 3);\n\
         \x20   println(s.len());\n\
         }\n",
        "stringloop",
    );
}

#[test]
fn a_mutable_map_receiver_survives_a_loop() {
    expect_builds(
        "fn fill(m: &mut HashMap<Int32, Int32>, n: Int32) {\n\
         \x20   let mut i = 0;\n\
         \x20   while i < n {\n\
         \x20       m.insert(i, i);\n\
         \x20       i = i + 1;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut m: HashMap<Int32, Int32> = HashMap::new();\n\
         \x20   fill(&mut m, 3);\n\
         \x20   println(m.len());\n\
         }\n",
        "maploop",
    );
}

#[test]
fn a_mutable_set_receiver_survives_a_loop() {
    expect_builds(
        "fn fill(s: &mut HashSet<Int32>, n: Int32) {\n\
         \x20   let mut i = 0;\n\
         \x20   while i < n {\n\
         \x20       s.insert(i);\n\
         \x20       i = i + 1;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut s: HashSet<Int32> = HashSet::new();\n\
         \x20   fill(&mut s, 3);\n\
         \x20   println(s.len());\n\
         }\n",
        "setloop",
    );
}

/// A reborrow forwarded through two frames: the inner call reborrows what the outer one already
/// reborrowed. If the repair only worked one level deep, this is where it would show.
#[test]
fn a_reborrow_forwards_through_two_frames() {
    expect_builds(
        "fn inner(out: &mut Vec<UInt8>, b: UInt8) { out.push(b); }\n\
         fn outer(out: &mut Vec<UInt8>, n: Int32) {\n\
         \x20   let mut i = 0;\n\
         \x20   while i < n {\n\
         \x20       inner(out, 65u8);\n\
         \x20       i = i + 1;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   outer(&mut v, 4);\n\
         \x20   println(v.len());\n\
         }\n",
        "twoframes",
    );
}

/// Repeated calls passing the same `&mut` — legal precisely BECAUSE each is a reborrow. Without
/// reborrowing, the first call would move `r` and the second would be a use-after-move.
///
/// The borrow is scoped so it ENDS before the owner is read: `let`-bound borrows are lexical
/// (03-Type-System.md), so reading `v` while `r` is still live is a genuine E0101 and has nothing
/// to do with this repair.
#[test]
fn the_same_mutable_reference_is_passed_repeatedly() {
    expect_builds(
        "fn take(v: &mut Vec<Int32>) { v.push(1); }\n\
         fn main() {\n\
         \x20   let mut v: Vec<Int32> = Vec::new();\n\
         \x20   {\n\
         \x20       let r = &mut v;\n\
         \x20       take(r);\n\
         \x20       take(r);\n\
         \x20   }\n\
         \x20   println(v.len());\n\
         }\n",
        "repeatedpass",
    );
}

/// A SHARED receiver is untouched by the repair: `&T` is `Copy`, so it never moved and the
/// pass-through path stays correct.
#[test]
fn a_shared_receiver_is_unaffected() {
    expect_builds(
        "fn total(v: &Vec<Int32>) -> Int32 {\n\
         \x20   let n = v.len();\n\
         \x20   let mut t = 0;\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < n {\n\
         \x20       t = t + v[i];\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   t\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<Int32> = Vec::new();\n\
         \x20   v.push(1);\n\
         \x20   println(total(&v));\n\
         }\n",
        "sharedreceiver",
    );
}

// ------------------------------------------------------------------ must refuse --

/// **The aliasing control.** The owner may not be used while an exclusive borrow of it is live.
/// A reborrow must not weaken this: if it did, the repair would have bought loop support by
/// giving up exclusivity.
#[test]
fn the_owner_is_still_refused_while_a_mutable_borrow_lives() {
    let why = expect_refused(
        "fn main() {\n\
         \x20   let mut v: Vec<Int32> = Vec::new();\n\
         \x20   let r = &mut v;\n\
         \x20   v.push(1);\n\
         \x20   r.push(2);\n\
         }\n",
        "ownerwhileborrowed",
    );
    assert!(
        why.contains("E0101"),
        "expected a borrow-check refusal, got: {why}"
    );
}

/// Two live exclusive borrows of one owner remain refused.
#[test]
fn two_live_mutable_borrows_are_still_refused() {
    expect_refused(
        "fn main() {\n\
         \x20   let mut v: Vec<Int32> = Vec::new();\n\
         \x20   let a = &mut v;\n\
         \x20   let b = &mut v;\n\
         \x20   a.push(1);\n\
         \x20   b.push(2);\n\
         }\n",
        "twoborrows",
    );
}

/// A shared borrow still cannot satisfy a `&mut` parameter. The repair must not invent the
/// capability — this is the same direction-of-weakening question DEV-146 settled.
#[test]
fn a_shared_borrow_still_cannot_become_exclusive() {
    expect_refused(
        "fn take(v: &mut Vec<Int32>) { v.push(1); }\n\
         fn main() {\n\
         \x20   let mut v: Vec<Int32> = Vec::new();\n\
         \x20   let r = &v;\n\
         \x20   take(r);\n\
         }\n",
        "sharednotexclusive",
    );
}

/// An owned value moved twice is still refused — the repair touches references, not values.
#[test]
fn an_owned_value_moved_twice_is_still_refused() {
    let why = expect_refused(
        "fn take(v: Vec<Int32>) -> UInt64 { v.len() }\n\
         fn main() {\n\
         \x20   let v: Vec<Int32> = Vec::new();\n\
         \x20   println(take(v));\n\
         \x20   println(take(v));\n\
         }\n",
        "ownedtwice",
    );
    assert!(
        why.contains("E0100"),
        "expected a use-after-move refusal, got: {why}"
    );
}
