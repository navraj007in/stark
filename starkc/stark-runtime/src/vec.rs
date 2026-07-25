//! WP-C6.3b — STARK Core `Vec<T>` operations.
//!
//! The generated backend calls these instead of Rust's inherent `Vec` methods, pinning the STARK
//! semantics (06-Standard-Library) in one reviewed place. A STARK `Vec<T>` is a Rust `Vec<T>`; it
//! is owning and not `Copy`, so a local lives in a [`crate::slot::ValueSlot`] and MIR controls when
//! it (and its remaining elements) are destroyed.
//!
//! This module now covers the whole surface: the value operations, the TRAPPING ones (`v[i]`,
//! `remove`, slice construction — each taking the user's source location so the trap reports where
//! the STARK expression is, not where the runtime is), and the ones handing out an INTERIOR borrow
//! (`get`/`get_mut`, by-reference iteration, slice views).

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
/// The source location is the USER's `v[i]` expression: the backend resolves the call terminator's
/// own `SourceInfo` at compile time and bakes it in, exactly as `Terminator::Checked` does for array
/// and arithmetic traps (DEV-107, closed — the earlier `"<vec index>"` placeholder is gone).
pub fn index_get<T: Copy>(v: &[T], i: u64, file: &str, line: u32, column: u32) -> T {
    if i as usize >= v.len() {
        crate::trap::abort(
            crate::trap::TrapCategory::IndexOutOfBounds,
            file,
            line,
            column,
        );
    }
    v[i as usize]
}

/// `v.remove(i)` — removes and RETURNS the element at `i`, shifting the rest left. Traps
/// `IndexOutOfBounds` at the user's location when `i` is past the end.
pub fn remove<T>(v: &mut Vec<T>, i: u64, file: &str, line: u32, column: u32) -> T {
    if i as usize >= v.len() {
        crate::trap::abort(
            crate::trap::TrapCategory::IndexOutOfBounds,
            file,
            line,
            column,
        );
    }
    v.remove(i as usize)
}

/// `v.get(i)` / `v.get_mut(i)` (0.1-A4) — CHECKED interior access that never traps: `None` when out
/// of range, distinct from the trapping `v[i]`. The reference borrows into the live `Vec`.
pub fn get_ref<T>(v: &[T], i: u64) -> Option<&T> {
    v.get(i as usize)
}

/// Takes `&mut [T]` rather than `&mut Vec<T>`: checked access never changes the length, and a
/// `&mut Vec<T>` receiver coerces. (`remove` DOES change the length, so it keeps the `Vec`.)
pub fn get_mut_ref<T>(v: &mut [T], i: u64) -> Option<&mut T> {
    v.get_mut(i as usize)
}

/// `Vec::clear(&mut self)` — drop every element, length becomes 0.
pub fn clear<T>(v: &mut Vec<T>) {
    v.clear();
}

/// WP-C6.3b (0.1-A6): a slice VIEW over an array, `Vec`, or another slice. `&[T; N]` and `&Vec<T>`
/// both coerce to `&[T]`, so one entry point serves every base.
///
/// 06-Standard-Library makes a bad window a TRAP, not a clamp: a NEGATIVE bound, an inverted one
/// (`lo > hi`), or one past the end traps `IndexOutOfBounds` at the user's location. Bounds arrive
/// SIGNED (a STARK range is `Int`-typed, so `&a[-1..2]` is expressible and must trap rather than
/// wrap), which is why these take `i64` rather than `u64`. `inclusive` distinguishes `a..b` from
/// `a..=b`: the inclusive form's upper bound is `hi + 1`.
pub fn slice_new<'a, T>(
    base: &'a [T],
    lo: i64,
    hi: i64,
    inclusive: bool,
    file: &str,
    line: u32,
    column: u32,
) -> &'a [T] {
    let (lo, end) = slice_bounds(base.len(), lo, hi, inclusive, file, line, column);
    &base[lo..end]
}

/// The `&mut [T]` view (0.1-A8): writes through it reach the base object (REF-SLICE-001).
pub fn slice_new_mut<'a, T>(
    base: &'a mut [T],
    lo: i64,
    hi: i64,
    inclusive: bool,
    file: &str,
    line: u32,
    column: u32,
) -> &'a mut [T] {
    let (lo, end) = slice_bounds(base.len(), lo, hi, inclusive, file, line, column);
    &mut base[lo..end]
}

/// The shared bound check, so the shared and exclusive views cannot drift apart.
fn slice_bounds(
    len: usize,
    lo: i64,
    hi: i64,
    inclusive: bool,
    file: &str,
    line: u32,
    column: u32,
) -> (usize, usize) {
    let trap = || {
        crate::trap::abort(
            crate::trap::TrapCategory::IndexOutOfBounds,
            file,
            line,
            column,
        )
    };
    if lo < 0 || hi < 0 {
        trap();
    }
    let end = match (hi as u64).checked_add(u64::from(inclusive)) {
        Some(end) => end,
        None => trap(),
    };
    let (lo, end) = (lo as u64, end);
    if lo > end || end > len as u64 {
        trap();
    }
    (lo as usize, end as usize)
}

/// `SliceLen` / `SliceIsEmpty` — the view's own length, not the base's.
pub fn slice_len<T>(s: &[T]) -> u64 {
    s.len() as u64
}

pub fn slice_is_empty<T>(s: &[T]) -> bool {
    s.is_empty()
}
