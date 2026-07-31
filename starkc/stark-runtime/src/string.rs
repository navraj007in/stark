//! WP-C6.3a — STARK Core `String` and `str` operations.
//!
//! The generated backend calls these functions instead of Rust's inherent `String`/`str` methods,
//! so the observable semantics are defined HERE, versioned with the runtime, and cannot silently
//! drift with the host `std`. STARK's spec (06-Standard-Library) for these operations matches
//! Rust's UTF-8 `String` today — `len` is the BYTE length (`UInt64`), ordering is lexicographic by
//! bytes — so each entry is a thin, explicit wrapper that pins that choice in one reviewed place.
//!
//! Representation: a STARK `String` is a Rust `String`; a STARK `str` behind a reference is a Rust
//! `&str`. A `String` is not `Copy`, so a local lives in a [`crate::slot::ValueSlot`] and MIR
//! controls WHEN the buffer is released (the slot's drop invokes Rust's `String` drop, which frees
//! it) — Rust never auto-drops a value MIR is responsible for destroying.
//!
//! Receivers arrive as `&String` (a `&self` method) or `&str`; taking `&str` lets both forms call
//! one function via deref coercion.

/// `String::new()` — a fresh empty owned string.
pub fn new() -> String {
    String::new()
}

/// `String::from(s: &str)` — an owned copy of a string slice.
pub fn from_str(s: &str) -> String {
    s.to_string()
}

/// `str::to_string(&self)` — an owned copy of a string slice.
pub fn to_string(s: &str) -> String {
    s.to_string()
}

/// `String::clone(&self)` — an owned copy (observable value equals the source).
pub fn clone_string(s: &str) -> String {
    s.to_string()
}

/// `String::len`/`str::len` — the BYTE length (06: `UInt64`).
pub fn len(s: &str) -> u64 {
    s.len() as u64
}

/// `String::is_empty`/`str::is_empty`.
pub fn is_empty(s: &str) -> bool {
    s.is_empty()
}

/// `String::contains(&self, pattern: &str)` — substring containment.
pub fn contains(s: &str, pattern: &str) -> bool {
    s.contains(pattern)
}

/// `String::as_str(&self)` — a borrowed view of the whole string. Takes `&str` (a `&String`
/// receiver deref-coerces) and returns it with the same lifetime.
pub fn as_str(s: &str) -> &str {
    s
}

/// `str::substring(start, end)` — byte-indexed slice, trapping on invalid UTF-8 boundaries.
pub fn substring(s: &str, start: u64, end: u64) -> &str {
    let start = start as usize;
    let end = end as usize;
    s.get(start..end)
        .expect("String::substring range is not on valid UTF-8 boundaries")
}

/// `String::push_str(&mut self, s: &str)` — append in place.
pub fn push_str(s: &mut String, suffix: &str) {
    s.push_str(suffix);
}

/// `String::clear(&mut self)` — truncate to empty (capacity unobservable).
pub fn clear(s: &mut String) {
    s.clear();
}

/// `String::push(&mut self, c: Char)` — append one Unicode scalar.
pub fn push_char(s: &mut String, c: char) {
    s.push(c);
}

/// `String::pop(&mut self) -> Option<Char>` — remove and return the last scalar, if any. The
/// backend wraps the Rust `Option<char>` into the program's generated `Option<Char>` enum.
pub fn pop_char(s: &mut String) -> Option<char> {
    s.pop()
}

/// `Char`-output: submit a char's UTF-8 bytes to stdout, without / with a trailing newline.
pub fn print_char(c: char) {
    let mut buf = [0u8; 4];
    crate::output::stdout_bytes(c.encode_utf8(&mut buf).as_bytes());
}

pub fn println_char(c: char) {
    let mut buf = [0u8; 4];
    crate::output::stdout_line(c.encode_utf8(&mut buf).as_bytes());
}

/// WP-C7.9 Packet D: the same, to the stderr sink.
pub fn eprint_char(c: char) {
    let mut buf = [0u8; 4];
    crate::output::stderr_bytes(c.encode_utf8(&mut buf).as_bytes());
}

pub fn eprintln_char(c: char) {
    let mut buf = [0u8; 4];
    crate::output::stderr_line(c.encode_utf8(&mut buf).as_bytes());
}

/// `Char::from_u32(value)` — validated Unicode scalar construction.
pub fn char_from_u32(value: u32) -> Option<char> {
    char::from_u32(value)
}

/// `==` on `String`/`str` (V-STR-2 routes through here, never a structural comparison).
pub fn eq(a: &str, b: &str) -> bool {
    a == b
}

/// `str`/`String` ordering: −1 / 0 / +1 (lexicographic by bytes), as `Int64`. Ordered comparison
/// operators derive from this against zero.
pub fn cmp(a: &str, b: &str) -> i64 {
    match a.cmp(b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// WP-C6.3c (0.1-A5): `str::chars`/`String::chars` iteration. The cursor borrows the source string
/// for as long as it lives (STARK's borrow checker forbids mutating it meanwhile). `Char` is `Copy`,
/// so elements are yielded BY VALUE and nothing is lent out of the cursor.
pub struct CharsIter<'a> {
    inner: std::str::Chars<'a>,
}

/// `s.chars()` (`RuntimeFn::CharsIterNew`).
pub fn chars_new(s: &str) -> CharsIter<'_> {
    CharsIter { inner: s.chars() }
}

/// `RuntimeFn::CharsIterNext` — the next Unicode scalar value, or `None` once exhausted. Iteration
/// is over CHARACTERS, not bytes, matching the interpreter's `Display`/`chars` semantics.
pub fn chars_next(it: &mut CharsIter<'_>) -> Option<char> {
    it.inner.next()
}
