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

/// `String::push_str(&mut self, s: &str)` — append in place.
pub fn push_str(s: &mut String, suffix: &str) {
    s.push_str(suffix);
}

/// `String::clear(&mut self)` — truncate to empty (capacity unobservable).
pub fn clear(s: &mut String) {
    s.clear();
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
