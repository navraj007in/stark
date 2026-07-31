//! `chars()` over Unicode, across every engine — the §4.7 regression matrix.
//!
//! **The defect.** The HIR oracle's character iterator held a SCALAR index and compared it against
//! `s.len()`, which is a BYTE count. For any string containing a multi-byte scalar the two
//! disagree — `"Stark語"` is six scalars in eight bytes — so the loop ran past the end and
//! `nth(6).unwrap()` panicked the host process. `starkc check` passed; `starkc run` killed the
//! interpreter.
//!
//! **Why nothing caught it.** Three reasons, each worth stating because each is a gap in a
//! different control:
//!
//! - ASCII hides it exactly. Byte length equals scalar count for ASCII, so every existing
//!   `chars()` case was in the one regime where the two indices agree.
//! - It was an **oracle-only** divergence. The MIR interpreter bounded its cursor with `nth()`
//!   returning `None`, and the native runtime wraps `std::str::Chars`; both were already correct.
//! - A panicking host produces **no observation to compare**, so the differential comparator
//!   could not see a disagreement — the process died before it reported anything.
//!
//! Every case below therefore pins the exact scalar sequence or count independently, and runs
//! through HIR, MIR, native-debug and native-release.

mod support;

use support::differential::agree_completing_with_stdout;

/// Counts the scalars in `text`, printing the count — the shape the original defect crashed on.
fn count_case(tag: &str, literal: &str, expected: usize) {
    agree_completing_with_stdout(
        tag,
        &format!(
            "fn main() {{ let s: String = String::from(\"{literal}\"); \
             let mut n: Int32 = 0; \
             for c in s.as_str().chars() {{ n = n + 1; }} \
             println(n); }}"
        ),
        &format!("{expected}\n"),
    );
}

/// Prints each scalar on its own line — proves the VALUES and their ORDER, not just the count. A
/// cursor that advanced by the wrong width would produce replacement characters or split a scalar.
fn sequence_case(tag: &str, literal: &str, expected: &str) {
    agree_completing_with_stdout(
        tag,
        &format!(
            "fn main() {{ let s: String = String::from(\"{literal}\"); \
             for c in s.as_str().chars() {{ println(c); }} }}"
        ),
        expected,
    );
}

// ------------------------------------------------------------------- the width matrix --

#[test]
fn an_empty_string_yields_nothing() {
    count_case("chars_empty", "", 0);
}

#[test]
fn ascii_only() {
    count_case("chars_ascii", "Stark", 5);
    sequence_case("chars_ascii_seq", "abc", "a\nb\nc\n");
}

/// Two bytes. The first width at which a scalar index and a byte index diverge.
#[test]
fn a_two_byte_scalar() {
    count_case("chars_two_byte", "é", 1);
    sequence_case("chars_two_byte_seq", "é", "é\n");
}

/// Three bytes — the exact character from the original reproduction.
#[test]
fn a_three_byte_scalar() {
    count_case("chars_three_byte", "語", 1);
    sequence_case("chars_three_byte_seq", "語", "語\n");
}

/// Four bytes, the widest UTF-8 encoding.
#[test]
fn a_four_byte_scalar() {
    count_case("chars_four_byte", "😀", 1);
    sequence_case("chars_four_byte_seq", "😀", "😀\n");
}

/// The reproduction itself: ASCII followed by a three-byte scalar. Six scalars, eight bytes — the
/// program that panicked the host.
#[test]
fn mixed_ascii_and_unicode() {
    count_case("chars_mixed", "Stark語", 6);
    sequence_case("chars_mixed_seq", "Stark語", "S\nt\na\nr\nk\n語\n");
}

/// Consecutive multi-byte scalars of differing widths, so a fixed-stride cursor cannot pass by
/// accident.
#[test]
fn consecutive_unicode_of_mixed_widths() {
    count_case("chars_consecutive", "é語😀é", 4);
    sequence_case("chars_consecutive_seq", "é語😀é", "é\n語\n😀\né\n");
}

// ------------------------------------------------------------------ source and repetition --

/// A `str` literal's `chars()`, without an owned `String` in between.
#[test]
fn a_literal_iterates() {
    agree_completing_with_stdout(
        "chars_literal",
        "fn main() { let mut n: Int32 = 0; for c in \"a語b\".chars() { n = n + 1; } println(n); }",
        "3\n",
    );
}

/// **Repeated iteration over the same source is deterministic**, and the source survives it. A
/// cursor stored on the string rather than on the iterator would give a different answer the
/// second time.
#[test]
fn repeated_iteration_is_deterministic() {
    agree_completing_with_stdout(
        "chars_repeat",
        "fn main() { let s: String = String::from(\"a語b\"); \
         let mut first: Int32 = 0; for c in s.as_str().chars() { first = first + 1; } \
         let mut second: Int32 = 0; for c in s.as_str().chars() { second = second + 1; } \
         println(first); println(second); println(s.len()); }",
        "3\n3\n5\n",
    );
}

/// A `String` MUTATED before iteration — the original reproduction's exact shape, where the
/// multi-byte scalar arrives via `push` rather than in the literal.
#[test]
fn a_mutated_string_then_iterated() {
    agree_completing_with_stdout(
        "chars_mutated",
        "fn main() { let mut s: String = String::from(\"Stark\"); s.push('語'); \
         let mut n: Int32 = 0; for c in s.as_str().chars() { n = n + 1; } \
         println(n); println(s.len()); }",
        "6\n8\n",
    );
}

/// Byte length and scalar count are different questions and must give different answers. This is
/// the invariant the defect violated, stated directly: had `len()` and the scalar count been
/// required to agree, the old implementation would have looked correct.
#[test]
fn byte_length_and_scalar_count_differ_and_both_are_right() {
    agree_completing_with_stdout(
        "chars_len_vs_count",
        "fn main() { let s: String = String::from(\"é語😀\"); \
         let mut n: Int32 = 0; for c in s.as_str().chars() { n = n + 1; } \
         println(s.len()); println(n); }",
        "9\n3\n",
    );
}
