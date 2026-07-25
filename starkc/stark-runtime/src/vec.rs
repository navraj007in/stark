//! WP-C6.3b — STARK Core `Vec<T>` operations.
//!
//! The generated backend calls these instead of Rust's inherent `Vec` methods, pinning the STARK
//! semantics (06-Standard-Library) in one reviewed place. A STARK `Vec<T>` is a Rust `Vec<T>`; it
//! is owning and not `Copy`, so a local lives in a [`crate::slot::ValueSlot`] and MIR controls when
//! it (and its remaining elements) are destroyed.
//!
//! Only the value operations that neither trap nor hand out an interior reference live here.
//! Index/replace/remove trap on out-of-bounds and need a source-located trap emitted at the MIR
//! call site; `get`/`get_mut`, by-reference iteration, and slice views hand out interior borrows —
//! both land in later slices.

/// `Vec::new()` — an empty vector.
pub fn new<T>() -> Vec<T> {
    Vec::new()
}

/// `Vec::with_capacity(n)` — an empty vector (capacity is unobservable in STARK; `len` is 0).
pub fn with_capacity<T>(n: u64) -> Vec<T> {
    Vec::with_capacity(n as usize)
}

/// `Vec::push(&mut self, item: T)` — append, taking ownership of `item`.
pub fn push<T>(v: &mut Vec<T>, item: T) {
    v.push(item);
}

/// `Vec::pop(&mut self) -> Option<T>` — remove and return the last element. The backend wraps the
/// Rust `Option<T>` into the program's generated Option enum.
pub fn pop<T>(v: &mut Vec<T>) -> Option<T> {
    v.pop()
}

/// `Vec::len(&self) -> UInt64`.
pub fn len<T>(v: &[T]) -> u64 {
    v.len() as u64
}

/// `Vec::is_empty(&self) -> Bool`.
pub fn is_empty<T>(v: &[T]) -> bool {
    v.is_empty()
}

/// WP-C6.3c (0.1-A2): by-reference `Vec` iteration. The cursor BORROWS its source for as long as it
/// lives — STARK's borrow checker forbids mutating the `Vec` meanwhile, which is what makes holding
/// the slice sound — and carries the position.
pub struct VecIter<'a, T> {
    slice: &'a [T],
    index: usize,
}

/// `v.iter()` (`RuntimeFn::VecIterNew`). Takes `&[T]`, which a `&Vec<T>` receiver coerces to.
pub fn iter_new<T>(v: &[T]) -> VecIter<'_, T> {
    VecIter { slice: v, index: 0 }
}

/// `RuntimeFn::VecIterNext` — `Option<&T>`, or `None` once exhausted.
///
/// The yielded reference borrows the SOURCE (`'a`), not the `&mut` borrow of the cursor: that is
/// what lets the loop variable outlive the `next` call, which the `for` desugaring requires (it
/// binds the element and then runs the body, by which time the cursor borrow is over).
pub fn iter_next<'a, T>(it: &mut VecIter<'a, T>) -> Option<&'a T> {
    let item = it.slice.get(it.index)?;
    it.index += 1;
    Some(item)
}

/// `v[i]` — the by-COPY checked indexed read (`RuntimeFn::VecIndexGet`, V-COPY-1). STARK's `v[i]`
/// TRAPS `IndexOutOfBounds` on an out-of-range index (it is a checked operation), so this reports
/// the trap through the trap ABI (exit 101, correct category) rather than Rust's own index panic.
///
/// DEV-107: the reported source location is this runtime call, NOT the user's `v[i]` span. Unlike
/// `Terminator::Checked` (arrays/slices), the `RuntimeFn` call ABI carries no per-call `SourceInfo`
/// to bake in, so precise provenance awaits the native Vec-trapping-ops WP (which threads a location
/// into the call, or lowers `v[i]` through a `Checked` proof). The Vec Display loop guarantees
/// `i < len`, so this trap path is dead there.
pub fn index_get<T: Copy>(v: &[T], i: u64) -> T {
    if i as usize >= v.len() {
        crate::trap::abort(
            crate::trap::TrapCategory::IndexOutOfBounds,
            "<vec index>",
            0,
            0,
        );
    }
    v[i as usize]
}

/// `Vec::clear(&mut self)` — drop every element, length becomes 0.
pub fn clear<T>(v: &mut Vec<T>) {
    v.clear();
}
