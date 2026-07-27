//! WP-C6.3a — native String and str runtime.
//!
//! The first slice of the Core native runtime (WP-C6.3 §23/§24). `stark_runtime::string` defines
//! the STARK semantics for `String`/`str` (byte `len`, UTF-8, lexicographic ordering — matching
//! 06-Standard-Library), and the generated backend emits every `String`/`str` `RuntimeFn` and the
//! str-output path as calls into it. `String` is Rust `String` (owning, non-`Copy`, slot-backed so
//! MIR controls destruction); `str` is `&str`; a `str` literal is a Rust `&'static str` literal.
//!
//! Each case runs through all three engines (HIR interpreter, MIR interpreter, native binary) and,
//! where it prints, the native stdout bytes are checked against the expectation.
//!
//! Deferred boundary (native only; HIR+MIR pass): a STORED interior `&str` that borrows an OWNED
//! `String` and is held across a block — including `String`'s own `==`/`<` (which lowers through
//! `String::as_str`) and an explicit `let v = s.as_str();` used after a branch. The stored borrow
//! overlaps the `String`'s slot-drop across the generated block-dispatch `loop { match __bb }`
//! back-edges (E0502) — the SAME dispatch-loop borrow-linearisation problem as WP-C6.1g-c, not a
//! String-specific defect. `str`-value comparison (literals, `&str` params) works natively.

mod support;

/// Delegates to the shared comparator (R-02), keeping this suite's independent stdout pin.
fn agree_out(tag: &str, src: &str, expect_out: &str) {
    support::differential::agree_completing_with_stdout(tag, src, expect_out);
}

/// No output expected -- and now that is CHECKED rather than assumed.
fn agree(tag: &str, src: &str) {
    agree_out(tag, src, "");
}

/// HIR + MIR + native all exit 0; the native stdout must equal `expect_out`. In-program `assert_eq`
/// carries value checks; `expect_out` carries the output-byte check.
/// No output expected.
#[test]
fn from_and_len() {
    agree(
        "from_len",
        "fn main() { let s = String::from(\"hello\"); assert_eq(s.len(), 5); }",
    );
}

#[test]
fn new_and_is_empty() {
    agree(
        "new_empty",
        "fn main() { let s = String::new(); assert_eq(s.is_empty(), true); }",
    );
}

#[test]
fn push_str_grows() {
    agree(
        "push_str",
        "fn main() { let mut s = String::from(\"ab\"); s.push_str(\"cd\"); assert_eq(s.len(), 4); }",
    );
}

#[test]
fn clear_empties() {
    agree(
        "clear",
        "fn main() { let mut s = String::from(\"ab\"); s.clear(); assert_eq(s.is_empty(), true); }",
    );
}

#[test]
fn contains_substring() {
    agree(
        "contains",
        "fn main() { let s = String::from(\"hello\"); assert_eq(s.contains(\"ell\"), true); }",
    );
}

#[test]
fn clone_copies_value() {
    agree(
        "clone",
        "fn main() { let a = String::from(\"hi\"); let b = a.clone(); assert_eq(b.len(), 2); }",
    );
}

#[test]
fn str_len_of_literal() {
    agree("str_len", "fn main() { assert_eq(\"abc\".len(), 3); }");
}

#[test]
fn str_to_string() {
    agree(
        "to_string",
        "fn main() { let s = \"xy\".to_string(); assert_eq(s.len(), 2); }",
    );
}

#[test]
fn str_bytes_index_and_len() {
    agree(
        "str_bytes",
        "fn main() { let b = \"Az\".bytes(); assert_eq(b.len(), 2); \
         assert_eq(b[0u64], 65u8); assert_eq(b[1u64], 122u8); }",
    );
}

#[test]
fn string_bytes_index_and_len() {
    agree(
        "string_bytes",
        "fn main() { let s = String::from(\"Hi\"); let b = s.bytes(); assert_eq(b.len(), 2); \
         assert_eq(b[0u64], 72u8); assert_eq(b[1u64], 105u8); }",
    );
}

#[test]
fn str_bytes_are_utf8_bytes() {
    agree(
        "str_bytes_utf8",
        "fn main() { let b = \"é\".bytes(); assert_eq(b.len(), 2); \
         assert_eq(b[0u64], 195u8); assert_eq(b[1u64], 169u8); }",
    );
}

#[test]
fn string_returned_across_function() {
    agree(
        "ret_string",
        "fn mk() -> String { String::from(\"made\") }\n\
         fn main() { let s = mk(); assert_eq(s.len(), 4); }",
    );
}

#[test]
fn println_str_literal() {
    agree_out(
        "println_lit",
        "fn main() { println(\"greetings\"); }",
        "greetings\n",
    );
}

#[test]
fn print_str_no_newline() {
    agree_out(
        "print_lit",
        "fn main() { print(\"ab\"); print(\"cd\"); }",
        "abcd",
    );
}

#[test]
fn str_literal_equality() {
    agree(
        "lit_eq",
        "fn main() { let r = if \"a\" == \"a\" { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

#[test]
fn str_literal_ordering() {
    agree(
        "lit_ord",
        "fn main() { let r = if \"a\" < \"b\" { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

// ---- Char operations (Char is a Copy scalar) ----

#[test]
fn println_char_value() {
    agree_out("println_char", "fn main() { println('A'); }", "A\n");
}

#[test]
fn print_char_unicode() {
    // A multi-byte scalar exercises UTF-8 encoding on the output path.
    agree_out("unicode_char", "fn main() { print('\u{3bb}'); }", "\u{3bb}");
}

#[test]
fn push_char_grows() {
    agree(
        "push_char",
        "fn main() { let mut s = String::from(\"ab\"); s.push('c'); assert_eq(s.len(), 3); }",
    );
}

/// `String::pop` returns `Option<Char>` — the runtime `Option` is wrapped into the program's
/// generated Option enum (the bridge every collection accessor reuses).
#[test]
fn pop_char_some() {
    agree(
        "pop_some",
        "fn main() { let mut s = String::from(\"aX\"); let c = s.pop(); assert_eq(c.unwrap_or('?'), 'X'); }",
    );
}

#[test]
fn pop_char_none_on_empty() {
    agree(
        "pop_none",
        "fn main() { let mut s = String::new(); let c = s.pop(); assert_eq(c.is_some(), false); }",
    );
}

// ---- Formerly deferred to WP-C6.1g-c; native since CD-112 (dispatch-loop linearisation). ----

/// `String` `==` lowers through `String::as_str`, producing a stored `&str` that borrows the owned
/// `String` across the branch. Under CD-112's linearised (labelled-block) emission rustc sees the
/// borrow with its real once-through lifetime, so this now builds and runs natively.
#[test]
fn owned_string_equality() {
    agree(
        "string_eq",
        "fn main() { let a = String::from(\"x\"); let b = String::from(\"x\"); \
         let r = if a == b { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

/// `String` `<` — ordered comparison through `String::as_str` then `StrCmp`.
#[test]
fn owned_string_ordering() {
    agree(
        "string_lt",
        "fn main() { let a = String::from(\"a\"); let b = String::from(\"b\"); \
         let r = if a < b { 1 } else { 0 }; assert_eq(r, 1); }",
    );
}

/// An explicit stored interior `&str` used after being bound.
#[test]
fn stored_as_str() {
    agree(
        "as_str_store",
        "fn main() { let s = String::from(\"hello\"); let v = s.as_str(); assert_eq(v.len(), 5); }",
    );
}
