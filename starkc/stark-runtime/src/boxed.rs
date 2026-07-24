//! WP-C6.3b — STARK Core `Box<T>` operations.
//!
//! A STARK `Box<T>` is a Rust `Box<T>`: an owning heap allocation, not `Copy`, slot-backed so MIR
//! controls its destruction (the box's `Drop` drops the contained `T` exactly once and frees the
//! allocation). Core v1 has NO `Deref` trait, so there is no `*box`; extraction is `into_inner`
//! only, which moves the value out and releases the box.

/// `Box::new(x)` — move `x` onto the heap.
pub fn new<T>(x: T) -> Box<T> {
    Box::new(x)
}

/// `Box::into_inner(b)` — move the contained value out, consuming the box. The `Box<T>` parameter
/// is the STARK ABI type (this mirrors a MIR `BoxIntoInner`), not an incidental local, so the
/// `boxed_local` lint does not apply.
#[allow(clippy::boxed_local)]
pub fn into_inner<T>(b: Box<T>) -> T {
    *b
}
