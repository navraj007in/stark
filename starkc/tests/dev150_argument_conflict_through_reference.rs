//! **DEV-150: the argument read-conflict rule does not fire through a reference base — OPEN, and
//! deliberately not resolved here.**
//!
//! `f(&mut x, x.field)` borrows `x` exclusively and then reads it in the same argument list. The
//! checker refuses that for a LOCAL, correctly:
//!
//! ```text
//! bump(&mut h, h.limit);
//!              ^^^^^^^ read conflict: variable is currently mutably borrowed
//! ```
//!
//! It ACCEPTS the identical shape when the base is a `&mut` parameter:
//!
//! ```text
//! fn forward(h: &mut Holder) { bump(h, h.limit); }   // accepted
//! ```
//!
//! The HIR oracle then executes it and produces the right answer, and the native backend emits Rust
//! that rustc refuses:
//!
//! ```text
//! error[E0503]: cannot use `_1.f0.f1` because it was mutably borrowed
//!   _188 = (&mut (*_1));
//!   _187.write(..take_line..(_188, (*_1).f0.f1, _189));
//! ```
//!
//! So it is accepted-but-unbuildable in FORM — the DEV-132/133/146/147/149 class — but unlike those
//! five it is **not** a mechanical omission with one right repair. It is a semantics question, and
//! this file pins the inconsistency without pretending to settle it.
//!
//! # The two candidate rulings
//!
//! **(A) The checker is right to accept; the backend must sequence.** Under this reading the
//! conflict is an artifact of evaluation order: read the field into a temporary BEFORE forming the
//! `&mut`, and nothing aliases. This is close to Rust's two-phase borrows, which exist precisely so
//! `v.push(v.len())` works. Cost: it makes the reference-base case legal while the local case stays
//! refused, so the two must be unified — meaning the LOCAL case has to start being accepted too,
//! which is a real widening of the borrow rule, not a backend change.
//!
//! **(B) The checker is wrong to accept; the rule must fire through a reference base.** Uniform,
//! conservative, and matches 03-Type-System.md as written — one `&mut` XOR many `&`, with no
//! exception for reference bases. Cost: `f(buf, buf.len())` stops compiling, and that is a common
//! shape. Every caller must hoist the read into a `let` first.
//!
//! Both are defensible and they disagree about what the language IS, so this is escalated rather
//! than resolved in a repair commit. What is NOT in question is that the current state — accepted
//! here, refused one indirection away, unbuildable natively — is wrong under either ruling.
//!
//! # What this file asserts today
//!
//! Only the facts, so that whichever ruling lands, the test that contradicts it fails loudly and
//! this file must be revisited rather than quietly outliving the decision:
//!
//! * the LOCAL case is refused (true under both rulings — (A) would change it, and this test is
//!   then the one that must be updated deliberately);
//! * the REFERENCE-BASE case is accepted by the checker and passes MIR verification;
//! * the two disagree, which is the defect itself.

mod support;

use starkc::mir::lower::lower_program;
use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

/// Front end only: `Ok` when the checker accepts.
fn check(src: &str, tag: &str) -> Result<(), String> {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    match checked
        .diagnostics
        .iter()
        .find(|d| d.severity == starkc::diag::Severity::Error)
    {
        Some(first) => Err(format!(
            "{} {}",
            first.code.as_deref().unwrap_or("-"),
            first.message
        )),
        None => Ok(()),
    }
}

/// Front end + lowering + MIR verification.
fn build(src: &str, tag: &str) -> Result<(), String> {
    check(src, tag)?;
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, _) = parse(&file, ParseMode::Program);
    let (hir, _) = resolve(&ast, file.clone());
    let checked = typecheck::analyze(&hir, file.clone());
    let program =
        lower_program(&hir, &checked.tables, file).map_err(|e| format!("LOWER: {}", e.what))?;
    starkc::mir::verify::verify_program(&program)
        .map_err(|errors| format!("VERIFY {}", errors.len()))?;
    Ok(())
}

const LOCAL_BASE: &str = "struct Holder { limit: UInt64, seen: UInt64 }\n\
                          fn bump(h: &mut Holder, by: UInt64) { h.seen = h.seen + by; }\n\
                          fn main() {\n\
                          \x20   let mut h = Holder { limit: 3u64, seen: 0u64 };\n\
                          \x20   bump(&mut h, h.limit);\n\
                          \x20   println(h.seen);\n\
                          }\n";

const REFERENCE_BASE: &str = "struct Holder { limit: UInt64, seen: UInt64 }\n\
                              fn bump(h: &mut Holder, by: UInt64) { h.seen = h.seen + by; }\n\
                              fn forward(h: &mut Holder) { bump(h, h.limit); }\n\
                              fn main() {\n\
                              \x20   let mut h = Holder { limit: 3u64, seen: 0u64 };\n\
                              \x20   forward(&mut h);\n\
                              \x20   println(h.seen);\n\
                              }\n";

/// The rule fires for a local base. Under ruling (B) this stays; under ruling (A) this test is the
/// one that must be deliberately changed, which is the point of asserting it.
#[test]
fn a_read_conflict_is_refused_for_a_local_base() {
    let why = check(LOCAL_BASE, "localbase").expect_err(
        "`bump(&mut h, h.limit)` on a local must be refused — if this now builds, ruling (A) has \
         landed and DEV-150 must be closed rather than left open",
    );
    assert!(
        why.contains("read conflict") || why.starts_with("E0101"),
        "expected a read-conflict diagnostic, got: {why}"
    );
}

/// The same shape one indirection away is accepted, and reaches MIR intact. This is the defect.
#[test]
fn the_same_conflict_is_accepted_through_a_reference_base() {
    assert_eq!(
        build(REFERENCE_BASE, "referencebase"),
        Ok(()),
        "DEV-150 records that this is ACCEPTED. If it now fails, ruling (B) has landed: the rule \
         fires through a reference base, DEV-150 closes, and this test must be inverted."
    );
}

/// The inconsistency itself, stated as one assertion so it cannot be read as two unrelated facts.
/// This is what makes the current state wrong under EITHER ruling: one of these two answers has to
/// change, and today the language gives both.
#[test]
fn the_two_bases_disagree_which_is_the_defect() {
    let local = check(LOCAL_BASE, "disagree_local");
    let reference = check(REFERENCE_BASE, "disagree_reference");
    assert!(
        local.is_err() && reference.is_ok(),
        "DEV-150 is the disagreement between these two: local={local:?}, reference={reference:?}. \
         If they now AGREE, the defect is resolved and this file must be rewritten around the \
         ruling that resolved it — not deleted, because which way they agree is the record."
    );
}

/// The hoisted form is what a caller must write today, under either ruling. `stark-http-parser`'s
/// `take_line` call sites were rewritten into exactly this shape.
#[test]
fn the_hoisted_form_builds() {
    assert_eq!(
        build(
            "struct Holder { limit: UInt64, seen: UInt64 }\n\
             fn bump(h: &mut Holder, by: UInt64) { h.seen = h.seen + by; }\n\
             fn forward(h: &mut Holder) {\n\
             \x20   let limit = h.limit;\n\
             \x20   bump(h, limit);\n\
             }\n\
             fn main() {\n\
             \x20   let mut h = Holder { limit: 3u64, seen: 0u64 };\n\
             \x20   forward(&mut h);\n\
             \x20   println(h.seen);\n\
             }\n",
            "hoisted",
        ),
        Ok(())
    );
}
