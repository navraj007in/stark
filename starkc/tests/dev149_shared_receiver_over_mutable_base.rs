//! **DEV-149: a `&self` method on a `&mut` receiver is reborrowed-and-weakened, not passed through.**
//!
//! DEV-147 taught `borrow_{vec,string,map,set}_receiver` to reborrow a reference receiver instead
//! of moving it, then narrowed the repair to the case where the METHOD wants `&mut`. That was the
//! wrong axis, and it left the other half of the same defect in place. Eight lines reproduce it:
//!
//! ```text
//! fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }
//!   -> MIR-0005 bb0: expected Ref { mutable: false, .. }, found Ref { mutable: true, .. }
//!   -> MIR-0007 bb4: move from possibly-moved place _1[]
//! ```
//!
//! Two failures at once, from one omission:
//!
//! * **MIR-0005** — `len` takes `&self`, and the `&mut` was handed over unweakened. This is the
//!   DEV-133/DEV-146 coercion family again, at a site that never called `weaken_ref_to`.
//! * **MIR-0007** — `&mut T` is not `Copy`, so reading it to pass it MOVED the caller's reference,
//!   which the next loop iteration then refuses. This is DEV-147's own failure, unfixed for shared
//!   receivers.
//!
//! Accepted-but-unbuildable, the DEV-132/133/146/147 class: the checker accepted it and the HIR
//! oracle returned the right answer, so only a native build could see it. `stark-http-parser`'s
//! `drop_front` — read the length of a caller's buffer, then rewrite it — was the package that
//! surfaced it, and "measure then modify a caller's buffer" is not an exotic shape.
//!
//! # The repair
//!
//! One reborrow serves both failures, because `&*base` **is** the weakening: from a `&mut` base it
//! produces the shared reference the callee wants without consuming the caller's. So the gate moves
//! from the receiver's mutability to the BASE's — is there a non-`Copy` reference at risk here —
//! and the reference built takes the RECEIVER's mutability, which is what the callee asked for.
//!
//! # Why the negative controls are the important half
//!
//! Weakening runs in ONE direction. If this had also let a `&T` base satisfy a `&mut` receiver, the
//! repair would have handed out exclusive access from a shared borrow — a real aliasing hole, and
//! far worse than the build failure it fixed. That direction is pinned below.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Front end + lowering + MIR verification: `Ok` only when the program would BUILD. A checker-only
/// harness could not have caught this defect, because the checker was never wrong about it.
fn build(src: &str, tag: &str) -> Result<(), String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir);
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
    let program = lower_program(
        &hir,
        &checked.tables,
        hir.source_named(&file.name).expect("registered"),
    )
    .map_err(|e| format!("LOWER: {}", e.what))?;
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

/// **The reproducer**, reduced from `stark-http-parser::drop_front` to eight lines.
#[test]
fn a_shared_method_on_a_mutable_vec_base() {
    expect_builds(
        "fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   v.push(1u8);\n\
         \x20   println(count(&mut v));\n\
         }\n",
        "vecleninfn",
    );
}

/// The shape that actually surfaced it: MEASURE the caller's buffer, then MODIFY it. Both
/// mutabilities of receiver over one `&mut` base, which is why narrowing by receiver was wrong.
#[test]
fn measuring_then_modifying_one_mutable_base() {
    expect_builds(
        "fn drop_front(pending: &mut Vec<UInt8>, count: UInt64) {\n\
         \x20   let n = pending.len();\n\
         \x20   let mut kept: Vec<UInt8> = Vec::new();\n\
         \x20   let mut i = count;\n\
         \x20   while i < n {\n\
         \x20       kept.push(pending[i]);\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   pending.clear();\n\
         \x20   let m = kept.len();\n\
         \x20   let mut k = 0u64;\n\
         \x20   while k < m {\n\
         \x20       pending.push(kept[k]);\n\
         \x20       k = k + 1u64;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   v.push(1u8);\n\
         \x20   v.push(2u8);\n\
         \x20   drop_front(&mut v, 1u64);\n\
         \x20   println(v.len());\n\
         }\n",
        "dropfront",
    );
}

/// In a LOOP, which is where the MIR-0007 half bites: the back-edge sees the parameter
/// possibly-moved and refuses the second iteration.
#[test]
fn a_shared_method_on_a_mutable_base_inside_a_loop() {
    expect_builds(
        "fn total(v: &mut Vec<UInt64>) -> UInt64 {\n\
         \x20   let mut sum = 0u64;\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < v.len() {\n\
         \x20       sum = sum + v[i];\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         \x20   sum\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt64> = Vec::new();\n\
         \x20   v.push(7u64);\n\
         \x20   println(total(&mut v));\n\
         }\n",
        "loopedlen",
    );
}

/// `String`, the second of the four receiver sites.
#[test]
fn a_shared_method_on_a_mutable_string_base() {
    expect_builds(
        "fn describe(s: &mut String) -> UInt64 {\n\
         \x20   let n = s.len();\n\
         \x20   s.push('!');\n\
         \x20   n\n\
         }\n\
         fn main() {\n\
         \x20   let mut s = String::from(\"hi\");\n\
         \x20   println(describe(&mut s));\n\
         }\n",
        "stringlen",
    );
}

/// `HashMap`, the third.
#[test]
fn a_shared_method_on_a_mutable_map_base() {
    expect_builds(
        "fn grow(m: &mut HashMap<Int32, Int32>) -> UInt64 {\n\
         \x20   let before = m.len();\n\
         \x20   m.insert(1, 2);\n\
         \x20   before\n\
         }\n\
         fn main() {\n\
         \x20   let mut m: HashMap<Int32, Int32> = HashMap::new();\n\
         \x20   println(grow(&mut m));\n\
         }\n",
        "maplen",
    );
}

/// `HashSet`, the fourth. All four sites route through the one helper, so all four are pinned.
#[test]
fn a_shared_method_on_a_mutable_set_base() {
    expect_builds(
        "fn grow(s: &mut HashSet<Int32>) -> UInt64 {\n\
         \x20   let before = s.len();\n\
         \x20   s.insert(1);\n\
         \x20   before\n\
         }\n\
         fn main() {\n\
         \x20   let mut s: HashSet<Int32> = HashSet::new();\n\
         \x20   println(grow(&mut s));\n\
         }\n",
        "setlen",
    );
}

/// Repeated shared reads of one `&mut` base. Each must reborrow independently; if any consumed the
/// base, the next would fail.
#[test]
fn repeated_shared_reads_of_one_mutable_base() {
    expect_builds(
        "fn thrice(v: &mut Vec<UInt8>) -> UInt64 {\n\
         \x20   v.len() + v.len() + v.len()\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   v.push(1u8);\n\
         \x20   println(thrice(&mut v));\n\
         }\n",
        "thricelen",
    );
}

/// DEV-147's own case must still build: this repair generalised that one rather than replacing it.
#[test]
fn the_dev147_mutable_receiver_case_still_builds() {
    expect_builds(
        "fn push_all(out: &mut Vec<UInt8>, n: UInt64) {\n\
         \x20   let mut i = 0u64;\n\
         \x20   while i < n {\n\
         \x20       out.push(65u8);\n\
         \x20       i = i + 1u64;\n\
         \x20   }\n\
         }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   push_all(&mut v, 3u64);\n\
         \x20   println(v.len());\n\
         }\n",
        "dev147still",
    );
}

/// A SHARED base under a shared receiver is untouched by the repair — `&T` is `Copy`, so there was
/// never anything to fix, and the gate on `base_mut` must leave it alone.
#[test]
fn a_shared_method_on_a_shared_base_is_unchanged() {
    expect_builds(
        "fn count(v: &Vec<UInt8>) -> UInt64 { v.len() }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   v.push(1u8);\n\
         \x20   println(count(&v));\n\
         }\n",
        "sharedbase",
    );
}

/// A `&mut` base handed to an ordinary function expecting `&` — the DEV-133 path, which already
/// worked. Pinned so the two coercion routes cannot diverge.
#[test]
fn a_mutable_base_still_weakens_at_an_ordinary_call() {
    expect_builds(
        "fn count(v: &Vec<UInt8>) -> UInt64 { v.len() }\n\
         fn forward(v: &mut Vec<UInt8>) -> UInt64 { count(v) }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   println(forward(&mut v));\n\
         }\n",
        "ordinarycall",
    );
}

// ------------------------------------------------------------------ must refuse --

/// **The negative control, and the important half.** Weakening runs in ONE direction: a shared base
/// must never satisfy a `&mut` receiver. If the repair had allowed it, it would have manufactured
/// exclusive access out of a shared borrow — an aliasing hole far worse than the build failure it
/// was fixing.
#[test]
fn a_mutable_method_on_a_shared_base_is_refused() {
    let why = expect_refused(
        "fn grow(v: &Vec<UInt8>) { v.push(1u8); }\n\
         fn main() {\n\
         \x20   let v: Vec<UInt8> = Vec::new();\n\
         \x20   grow(&v);\n\
         }\n",
        "sharedbasemutrecv",
    );
    assert!(
        why.starts_with("CHECK"),
        "a `&mut` method on a shared base must be refused by the CHECKER, not survive to \
         lowering, got: {why}"
    );
}

/// Exclusivity is still enforced around the reborrow: a second borrow while one is live is refused.
/// The reborrow must not have widened what a caller may alias.
#[test]
fn two_live_mutable_borrows_are_still_refused() {
    let why = expect_refused(
        "fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   let first = &mut v;\n\
         \x20   let second = &mut v;\n\
         \x20   println(count(first) + count(second));\n\
         }\n",
        "twoborrows",
    );
    assert!(
        why.starts_with("CHECK"),
        "overlapping `&mut` borrows must still be refused by the checker, got: {why}"
    );
}

/// A shared borrow taken while a `&mut` is live is still refused. The reborrow produces a shared
/// reference, and that must not become a way to read around an exclusive borrow.
#[test]
fn a_shared_borrow_under_a_live_mutable_borrow_is_still_refused() {
    let why = expect_refused(
        "fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }\n\
         fn main() {\n\
         \x20   let mut v: Vec<UInt8> = Vec::new();\n\
         \x20   let exclusive = &mut v;\n\
         \x20   println(v.len());\n\
         \x20   println(count(exclusive));\n\
         }\n",
        "readaround",
    );
    assert!(
        why.starts_with("CHECK"),
        "reading around a live `&mut` must still be refused by the checker, got: {why}"
    );
}
