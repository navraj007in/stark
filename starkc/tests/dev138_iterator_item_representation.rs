//! **DEV-138: an iterator's item is represented by its declared type, not by its producer.**
//!
//! `for word in text.split(" ")` yields `&str` (`06-Standard-Library.md`: `SplitIter` /
//! `String::split` / `&str`). The HIR interpreter yielded `Value::String` — an OWNED value, which
//! `value_is_copy` reports as non-Copy — so the first use consumed it and a second use in the same
//! iteration trapped "use of unavailable value". The checker and MIR both saw a `Copy` shared
//! reference; only the HIR engine disagreed.
//!
//! # This is DEV-121, not a new class
//!
//! DEV-121's governing rule, verbatim: Copy/move behaviour — and the runtime representation that
//! carries it — is determined exclusively by the normalized semantic type, never by the expression
//! that produced the value. Its original instance was `String::bytes()` returning `Value::Vec` for
//! a declared `&[UInt8]`. This is the same defect one producer over. The classification matrix that
//! established the fold:
//!
//! ```text
//! declared item type   &str
//! HIR runtime value    Value::String   (owned)          <- the defect
//! value_is_copy        Value::Str true, Value::String false
//! front end            accepts (sees a Copy shared reference)
//! MIR / native         VACUOUS - both refuse SplitIter outright (C4.5)
//! ```
//!
//! The MIR and native rows are vacuous rather than confirming, and are recorded that way: those
//! engines do not implement `SplitIter`, so they could not have disagreed. WP §9.3's "treat as
//! distinct" criteria require MIR to emit `Move` for a Copy shared-reference item and all engines
//! to consume it; neither holds.
//!
//! # Why it was producer-specific, and why that matters
//!
//! Six shapes were probed. `&Vec<String>`, `&Vec<Int32>`, `chars()`, and a plain `&str` outside a
//! loop were all already correct. Only `split` was wrong — and `trim`/`substring`, which have the
//! same declared return type, already yielded `Value::Str`. So the repair makes `split` consistent
//! with its siblings rather than introducing a new rule.
//!
//! # Residual exposure this pins
//!
//! INV-VALUE-REP-001 checks at every `let` that a binding declared `&str`/`&[T]` does not hold
//! owned storage. A FOR-LOOP BINDING is not a `let`, which is why the invariant did not catch
//! this. That gap is recorded against DEV-121; the tests below are the interim guard.

mod support;

use starkc::parser::{parse, ParseMode};
use starkc::resolve::resolve;
use starkc::source::SourceFile;
use starkc::typecheck;
use std::sync::Arc;

fn run(src: &str, tag: &str) -> String {
    let file = Arc::new(SourceFile::new(format!("{tag}.stark"), src.to_string()));
    let (ast, pd) = parse(&file, ParseMode::Program);
    assert!(pd.is_empty(), "{tag}: parse: {pd:?}");
    let (hir, rd) = resolve(&ast, file.clone());
    assert!(rd.is_empty(), "{tag}: resolve: {rd:?}");
    let checked = typecheck::analyze(&hir, file.clone());
    assert!(
        !checked
            .diagnostics
            .iter()
            .any(|d| d.severity == starkc::diag::Severity::Error),
        "{tag}: check: {:?}",
        checked.diagnostics
    );
    match starkc::interp::run(
        &hir,
        hir.source_named(&file.name).expect("registered"),
        &checked.tables,
    ) {
        Ok(execution) => execution.output,
        Err(error) => panic!("{tag}: runtime error: {}", error.message),
    }
}

/// The CD-334 reproducer: the item converted to an owned `String` twice in one iteration.
#[test]
fn a_split_item_survives_two_conversions() {
    let out = run(
        "fn main() {\n\
         \x20   for word in \"alpha beta\".split(\" \") {\n\
         \x20       let first = String::from(word);\n\
         \x20       let second = String::from(word);\n\
         \x20       println(first.as_str());\n\
         \x20       println(second.as_str());\n\
         \x20   }\n\
         }\n",
        "twice",
    );
    assert_eq!(out.trim(), "alpha\nalpha\nbeta\nbeta");
}

/// The item passed to a helper twice — the same question at a call boundary.
#[test]
fn a_split_item_survives_two_helper_calls() {
    let out = run(
        "fn width(text: &str) -> UInt64 { text.len() }\n\
         fn main() {\n\
         \x20   for word in \"aa bbb\".split(\" \") {\n\
         \x20       println(width(word));\n\
         \x20       println(width(word));\n\
         \x20   }\n\
         }\n",
        "helper",
    );
    assert_eq!(out.trim(), "2\n2\n3\n3");
}

/// Several distinct uses in one iteration, mixing methods and calls.
#[test]
fn a_split_item_survives_many_mixed_uses() {
    let out = run(
        "fn width(text: &str) -> UInt64 { text.len() }\n\
         fn main() {\n\
         \x20   for word in \"one\".split(\" \") {\n\
         \x20       println(word.len());\n\
         \x20       println(width(word));\n\
         \x20       println(word.to_uppercase());\n\
         \x20       println(String::from(word).as_str());\n\
         \x20       println(word.starts_with(\"o\"));\n\
         \x20   }\n\
         }\n",
        "mixed",
    );
    assert_eq!(out.trim(), "3\n3\nONE\none\ntrue");
}

/// Splitting on a multi-part input, so the cursor advances between reuses.
#[test]
fn every_item_of_a_multi_part_split_is_reusable() {
    let out = run(
        "fn main() {\n\
         \x20   let mut count = 0;\n\
         \x20   for field in \"a,b,c,d\".split(\",\") {\n\
         \x20       let copy = String::from(field);\n\
         \x20       count = count + 1;\n\
         \x20       println(copy.as_str());\n\
         \x20       println(field);\n\
         \x20   }\n\
         \x20   println(count);\n\
         }\n",
        "multipart",
    );
    assert_eq!(out.trim(), "a\na\nb\nb\nc\nc\nd\nd\n4");
}

// -------------------------------------------------------- producers already correct --

/// `&Vec<String>` yields `&String`. Already correct before this repair; pinned so the fix cannot
/// regress the producers it was NOT about.
#[test]
fn a_vec_borrow_item_is_reusable() {
    let out = run(
        "fn main() {\n\
         \x20   let mut values: Vec<String> = Vec::new();\n\
         \x20   values.push(String::from(\"x\"));\n\
         \x20   for value in &values {\n\
         \x20       println(value.len());\n\
         \x20       println(value.as_str());\n\
         \x20   }\n\
         }\n",
        "vecborrow",
    );
    assert_eq!(out.trim(), "1\nx");
}

/// `chars()` yields `Char`, a Copy scalar.
#[test]
fn a_chars_item_is_reusable() {
    let out = run(
        "fn main() {\n\
         \x20   for ch in \"ab\".chars() {\n\
         \x20       println(ch);\n\
         \x20       println(ch);\n\
         \x20   }\n\
         }\n",
        "chars",
    );
    assert_eq!(out.trim(), "a\na\nb\nb");
}

/// A `&str` that never went through an iterator, used twice — the control that shows the defect
/// was in the producer and not in `&str` handling generally.
#[test]
fn a_plain_str_binding_is_reusable() {
    let out = run(
        "fn main() {\n\
         \x20   let text = \"hello\";\n\
         \x20   let a = String::from(text);\n\
         \x20   let b = String::from(text);\n\
         \x20   println(a.as_str());\n\
         \x20   println(b.as_str());\n\
         }\n",
        "plainstr",
    );
    assert_eq!(out.trim(), "hello\nhello");
}

/// `trim` and `substring` return `&str` too, and already yielded the borrowed representation.
/// Pinned alongside `split` so the three stay consistent.
#[test]
fn trim_and_substring_items_are_reusable() {
    let out = run(
        "fn main() {\n\
         \x20   let padded = String::from(\"  hi  \");\n\
         \x20   let trimmed = padded.trim();\n\
         \x20   println(String::from(trimmed).as_str());\n\
         \x20   println(String::from(trimmed).as_str());\n\
         \x20   let part = padded.substring(2u64, 4u64);\n\
         \x20   println(String::from(part).as_str());\n\
         \x20   println(String::from(part).as_str());\n\
         }\n",
        "trimsub",
    );
    assert_eq!(out.trim(), "hi\nhi\nhi\nhi");
}

// ---------------------------------------------------- the item is still only a VIEW --

/// **The item must not become independently owned storage.** Making it Copy is correct; making it
/// an owned `String` that outlives its source would be a different defect. The split source here
/// is a temporary, and the item is consumed within the iteration, which is the only lifetime the
/// language grants it.
#[test]
fn a_split_item_still_reads_as_its_source_content() {
    let out = run(
        "fn main() {\n\
         \x20   let source = String::from(\"alpha:beta\");\n\
         \x20   for part in source.split(\":\") {\n\
         \x20       println(part);\n\
         \x20   }\n\
         \x20   println(source.as_str());\n\
         }\n",
        "stillview",
    );
    assert_eq!(
        out.trim(),
        "alpha\nbeta\nalpha:beta",
        "iterating must not disturb the source string"
    );
}

/// String-literal pattern matching against a split item still compares by CONTENT. DEV-129 covers
/// that rule; this pins that changing the item's representation did not break it.
#[test]
fn a_split_item_matches_string_literal_patterns_by_content() {
    let out = run(
        "fn main() {\n\
         \x20   for word in \"alpha beta\".split(\" \") {\n\
         \x20       let label = match word {\n\
         \x20           \"alpha\" => \"first\",\n\
         \x20           \"beta\" => \"second\",\n\
         \x20           _ => \"other\",\n\
         \x20       };\n\
         \x20       println(label);\n\
         \x20   }\n\
         }\n",
        "literalpattern",
    );
    assert_eq!(out.trim(), "first\nsecond");
}
