//! **DEV-162: reading one field of partially-moved storage.**
//!
//! Sibling of DEV-158, and the same root cause: whole-value machinery over place-granular
//! semantics. Once a field is moved out, the storage is `Partial` — and a read of an UNTOUCHED
//! sibling was emitted as `&slot.get().f1`, where `get` requires a complete value:
//!
//! ```text
//! _7.reinit(stark_proj::stark_move_23struct_230_23f0(&mut _1));
//! _13 = (&_1.get().f1);   // aborts: the slot is PARTIAL
//! ```
//!
//! `copy_field` already covered the `Copy` case by value (WP-C6.1b). This is the rest: a non-`Copy`
//! field, borrowed. `ValueSlot::field_ref` reads through a raw projection, so it never materialises
//! a reference to the surrounding value and never asserts that value is valid.
//!
//! The emitted form is `(*stark_proj::stark_ref_…(&_1))` rather than the bare call, because callers
//! in `Borrow` mode prepend their own `&` and need a Rust place expression.
//!
//! # The part that was missed first
//!
//! `Rvalue::RefOf` carries a PLACE, not an operand, so `rvalue_operands` returns nothing for it and
//! the collector never generated the helper the emitter had already named. That surfaces as `E0425`
//! inside the generated crate — a name error in code nobody wrote — rather than as any diagnostic
//! the compiler produces. Collector and emitter must agree, and nothing but a build proves it.

mod support;

/// Both halves of the read path, in one program: the `Copy` case that already worked, and the
/// non-`Copy` case that did not. Kept together so a regression in either is visible against the
/// other.
#[test]
fn a_sibling_field_is_readable_after_a_partial_move() {
    let source = r#"
struct Two { a: String, b: String, n: UInt32 }

fn main() {
    let mut t = Two { a: String::from("aaa"), b: String::from("bbb"), n: 7u32 };
    let moved = t.a;
    if moved.as_str() != "aaa" {
        panic("the moved field is wrong");
    }
    // A NON-Copy sibling, borrowed. This is DEV-162.
    if t.b.as_str() != "bbb" {
        panic("the non-Copy sibling is wrong");
    }
    // A Copy sibling, read by value. This already worked (WP-C6.1b) and must keep working.
    if t.n != 7u32 {
        panic("the Copy sibling is wrong");
    }
    println("DEV162_OK");
}
"#;
    // Three engines, COMPARED rather than each asserted to exit 0 — the distinction
    // `agree_completing` exists to make.
    let done = support::differential::agree_completing("dev162_partial_field_read", source);
    assert_eq!(
        String::from_utf8_lossy(&done.stdout_bytes),
        "DEV162_OK\n",
        "every engine must reach the end with the same output"
    );
}
