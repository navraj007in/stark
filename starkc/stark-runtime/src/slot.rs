//! WP-C5.3d-0 — controlled manual storage for non-`Copy` MIR locals (`WP-C5-ENTRY.md` §7.2, as
//! decided by CD-058).
//!
//! # Why generated code cannot just use Rust values
//!
//! The backend lowers MIR's basic-block graph to `loop { match __bb { .. } }`, so every block is
//! one iteration of one Rust loop. Rust's borrow checker is conservative across iterations: it
//! cannot see that MIR's control flow never revisits a moved-from local, and rejects a move that
//! verified MIR proves is sound ("value moved here, in previous iteration of loop"). Ordinary
//! Rust ownership therefore cannot express MIR liveness, and §7.1 forbids the alternative —
//! letting Rust add a second, implicit destruction schedule.
//!
//! [`ValueSlot`] is the answer: **verified MIR owns liveness, and this type merely obeys it.**
//!
//! # Why `MaybeUninit<ManuallyDrop<T>>` and not something simpler
//!
//! Two simpler shapes were considered and rejected (CD-058):
//!
//! - **`Option<T>`** — introduces Rust-owned destruction. Overwriting or dropping the `Option`
//!   runs `T`'s destructor on Rust's schedule, which is exactly the second schedule §7.1
//!   forbids.
//! - **`Option<ManuallyDrop<T>>`** — fixes destruction but not *representation*. It is adequate
//!   only for whole-value liveness: once a field or constant-index element has been moved out,
//!   the remaining bytes **no longer necessarily form a valid complete `T`**, and holding them as
//!   a `T` is undefined behaviour regardless of whether anything ever destroys it. Only
//!   `MaybeUninit` may legally hold a partially moved value.
//!
//! # The three states, and why two were not enough
//!
//! An earlier version of this module tracked only `live: bool` and let `move_sub` take `&mut T`,
//! move one field out, and leave the slot live. **That was unsound**, and the owner review caught
//! it: after a non-`Copy` field is moved out, the storage no longer holds a valid `T`, so
//! *every* whole-value operation becomes undefined behaviour — `get`/`get_mut` construct a
//! reference to an invalid `T` even if the caller only reads a different, still-valid field;
//! `take` reads moved-out bytes; and `drop_value` destroys the moved-out field a second time.
//! The module's own test asserted `slot.get().1` after moving `.0`, so the bug was written into
//! its evidence.
//!
//! Hence [`SlotState`]:
//!
//! | State | Meaning | Permitted |
//! |---|---|---|
//! | `Dead` | holds nothing | `write` |
//! | `Whole` | holds a valid complete `T` | `get`, `get_mut`, `take`, `drop_value`, `drop_with`, and the partial operations |
//! | `Partial` | at least one drop unit has been moved out | **only** raw-pointer field operations, and `finish_partial` |
//!
//! Once a slot is `Partial`, no operation that would materialise a `T` or a reference to one is
//! permitted. Partial access uses raw-pointer projection exclusively: a field pointer is computed
//! with `addr_of_mut!`, which never dereferences and never asserts that the surrounding value is
//! valid.
//!
//! # The safety contract
//!
//! Every operation here is a **safe function**, because generated MIR bodies must contain no
//! `unsafe` blocks of their own (§7.8): all unsafe operations route through this reviewed module
//! and, for per-type field projections, through the backend's generated helper module. Their
//! preconditions are guaranteed by verified MIR, and each one is *checked* rather than assumed —
//! a violation reaches [`slot_violation`], which names the invariant.
//!
//! Note what is deliberately absent: **`ValueSlot` implements no `Drop`.** A slot that generated
//! code never empties leaks, and that is the intended trade — leaking is what MIR verification
//! exists to exclude, and a Rust destructor here would make the exactly-once obligation
//! unfalsifiable by silently covering for a lowering bug.

use core::mem::{ManuallyDrop, MaybeUninit};

/// A generated-code invariant was violated: a compiler defect, not a language outcome.
///
/// This panics, and the generated crate's `panic = "abort"` profile turns that into an abort
/// without unwinding, as §7.7 requires. It stays distinguishable from a STARK trap — which exits
/// 101 through `trap::abort` and is a defined outcome a correct program can reach — because
/// reaching here means the backend emitted code contradicting verified MIR. Panicking rather than
/// exiting also makes the invariants testable: `#[should_panic]` can assert that an illegal
/// sequence is refused, which an immediate `process::exit` would make impossible to observe.
pub fn slot_violation(what: &str) -> ! {
    panic!("generated-code invariant violated: {what} (STARK compiler defect, not a program fault)")
}

/// What a slot currently holds. See the module docs for why `Partial` must be distinct from
/// `Whole` rather than folded into "live".
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotState {
    /// Holds nothing.
    Dead,
    /// Holds a valid, complete `T`.
    Whole,
    /// At least one drop unit has been moved out. The bytes no longer form a valid `T`.
    Partial,
}

/// Controlled storage for one non-`Copy` MIR local.
///
/// Whole-place liveness lives here. Which *individual* drop units remain live in the `Partial`
/// state stays where MIR already puts it — in the drop-flag locals MIR's own drop elaboration
/// produces — so this type does not invent a second, parallel notion of per-field liveness. What
/// it does track is the one thing MIR's flags cannot express to Rust: whether the storage may
/// still be treated as a `T` at all.
pub struct ValueSlot<T> {
    storage: MaybeUninit<ManuallyDrop<T>>,
    state: SlotState,
}

impl<T> ValueSlot<T> {
    /// A slot holding nothing. Every non-`Copy` local starts here, which is what lets generated
    /// code stop fabricating a default value it then immediately overwrites.
    pub const fn dead() -> Self {
        ValueSlot {
            storage: MaybeUninit::uninit(),
            state: SlotState::Dead,
        }
    }

    pub fn state(&self) -> SlotState {
        self.state
    }

    pub fn is_live(&self) -> bool {
        !matches!(self.state, SlotState::Dead)
    }

    pub fn is_whole(&self) -> bool {
        matches!(self.state, SlotState::Whole)
    }

    /// Guard for every operation that materialises a `T` or a reference to one.
    fn require_whole(&self, what: &str) {
        match self.state {
            SlotState::Whole => {}
            SlotState::Dead => slot_violation(what),
            SlotState::Partial => slot_violation(&format!(
                "{what}: the slot is PARTIAL — a drop unit has been moved out, so the storage no \
                 longer holds a valid complete value and may only be accessed field by field"
            )),
        }
    }

    /// Destination write: initialise the slot with `value`.
    ///
    /// The slot must be `Dead`. Writing over a live slot would abandon the previous value without
    /// destroying it — a leak MIR never asks for, since MIR emits an explicit `Drop` before any
    /// reassignment of a live place.
    pub fn write(&mut self, value: T) {
        if self.state != SlotState::Dead {
            slot_violation(
                "write to a live slot (MIR must Drop or move out before reassigning a live place)",
            );
        }
        self.storage = MaybeUninit::new(ManuallyDrop::new(value));
        self.state = SlotState::Whole;
    }

    /// Shared place access. Requires `Whole`.
    pub fn get(&self) -> &T {
        self.require_whole("read of a dead slot");
        // SAFETY: `Whole` is set only by `write`, which initialises `storage`, and is cleared by
        // every operation that de-initialises it or invalidates the representation.
        unsafe { &*self.storage.as_ptr() }
    }

    /// Mutable place access. Requires `Whole`.
    pub fn get_mut(&mut self) -> &mut T {
        self.require_whole("mutable access to a dead slot");
        // SAFETY: as `get`.
        unsafe { &mut *self.storage.as_mut_ptr() }
    }

    /// Whole-value move out. Requires `Whole`; the slot becomes `Dead` **before** the value is
    /// handed over, so no path can observe a slot that is simultaneously live and moved-from.
    pub fn take(&mut self) -> T {
        self.require_whole("take from a dead slot (a moved-from local was moved again)");
        self.state = SlotState::Dead;
        // SAFETY: was `Whole`, so `storage` held a valid complete `T`; now marked dead, so this
        // read cannot be repeated and nothing else will read the bytes.
        let held = unsafe { self.storage.as_ptr().read() };
        ManuallyDrop::into_inner(held)
    }

    /// Explicit destruction running Rust's own `Drop for T`. Requires `Whole`.
    ///
    /// Generated nominal types implement no Rust `Drop` (§6.3), so for them this destroys
    /// nothing; it exists for the types that do carry Rust drop glue.
    pub fn drop_value(&mut self) {
        self.require_whole("drop of a dead slot (a value was dropped twice)");
        self.state = SlotState::Dead;
        // SAFETY: was `Whole`; marked dead first, so this can neither repeat nor race a later
        // read.
        let mut held = unsafe { self.storage.as_ptr().read() };
        unsafe { ManuallyDrop::drop(&mut held) };
    }

    /// Explicit destruction with **MIR-directed glue**: mark the unit dead, then run `glue` on
    /// the still-readable storage. Requires `Whole`.
    ///
    /// This is the operation a MIR `Drop` terminator lowers to. It is separate from
    /// [`ValueSlot::drop_value`] because the two destroy through different authorities:
    /// `drop_value` runs Rust's `Drop for T`, while this runs the destructor MIR names — the
    /// concrete user `Drop` instance from `TypeContext::drop_impls`, followed by field glue in
    /// MIR's order.
    ///
    /// Dead **before** the glue runs (§7.5): if the glue itself traps, the abort path sees a dead
    /// slot and cannot re-enter the value, which is what makes exactly-once hold even when a
    /// destructor fails.
    pub fn drop_with(&mut self, glue: impl FnOnce(&mut T)) {
        self.require_whole("drop of a dead slot (a value was dropped twice)");
        self.state = SlotState::Dead;
        // SAFETY: was `Whole`. Marked dead first, so this cannot re-enter, and nothing reads the
        // bytes afterwards.
        let held: &mut ManuallyDrop<T> = unsafe { &mut *self.storage.as_mut_ptr() };
        glue(&mut *held);
    }

    // ------------------------------------------------------------- partial access --
    //
    // Everything below operates through RAW POINTERS and never constructs a `T`, a `&T`, or a
    // `&mut T`. That is what makes them legal once the storage is partially moved: a field
    // pointer computed with `addr_of_mut!` does not dereference, and does not assert that the
    // surrounding value is valid.
    //
    // `project` is a per-type, per-field function the BACKEND generates (one per field actually
    // used) so that emitted MIR bodies contain no `unsafe` of their own — the projection's
    // `unsafe` lives in the generated helper module, and the call site here is safe.

    fn base_ptr(&mut self) -> *mut T {
        // `ManuallyDrop<T>` is `#[repr(transparent)]`, so this cast is layout-exact.
        self.storage.as_mut_ptr() as *mut T
    }

    fn require_accessible(&self, what: &str) {
        if self.state == SlotState::Dead {
            slot_violation(what);
        }
    }

    /// Typed sub-place move: move ONE field or constant-index element out.
    ///
    /// The slot becomes `Partial` — not dead, because the other drop units are still live and MIR
    /// tracks them individually (collapsing that into whole-local liveness is what §7.6 forbids);
    /// and not `Whole`, because the storage no longer holds a valid complete `T`.
    pub fn move_field<F>(&mut self, project: fn(*mut T) -> *mut F) -> F {
        self.require_accessible("sub-place move out of a dead slot");
        self.state = SlotState::Partial;
        let field = project(self.base_ptr());
        // SAFETY: `project` computes a pointer to one field of storage that was initialised by
        // `write`. The caller's MIR drop flags record that this unit is now dead, so it is never
        // read again; and the slot is `Partial`, so no whole-value operation can observe the gap.
        unsafe { field.read() }
    }

    /// Read a `Copy` field out of a possibly-partial value, without materialising the whole `T`.
    pub fn copy_field<F: Copy>(&mut self, project: fn(*mut T) -> *mut F) -> F {
        self.require_accessible("field read from a dead slot");
        let field = project(self.base_ptr());
        // SAFETY: as `move_field`, except that `F: Copy` means the read leaves the field's own
        // drop obligation untouched -- a `Copy` type has none.
        unsafe { field.read() }
    }

    /// Destroy ONE drop unit of a possibly-partial value, running `glue` on a pointer to it.
    ///
    /// The slot becomes `Partial`: after destroying a field the value is no longer complete, for
    /// exactly the same reason moving one out makes it incomplete.
    pub fn drop_field_with<F>(&mut self, project: fn(*mut T) -> *mut F, glue: impl FnOnce(*mut F)) {
        self.require_accessible("field drop on a dead slot");
        self.state = SlotState::Partial;
        let field = project(self.base_ptr());
        glue(field);
    }

    /// Mark a `Partial` slot dead once MIR has accounted for every remaining drop unit.
    ///
    /// Deliberately not automatic: this type does not track which units remain, so only the
    /// generated code that followed MIR's drop flags can know the value is fully accounted for.
    pub fn finish_partial(&mut self) {
        match self.state {
            SlotState::Partial | SlotState::Dead => self.state = SlotState::Dead,
            SlotState::Whole => slot_violation(
                "finish_partial on a WHOLE slot: the value is still complete and needs a real \
                 drop or move, not a liveness reset",
            ),
        }
    }
}

impl<T> Default for ValueSlot<T> {
    fn default() -> Self {
        Self::dead()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// A value whose destruction is observable, so "exactly once" and "in this order" can be
    /// asserted rather than assumed. This is the shape C5.3d-1's fixture will use.
    struct Observed<'a> {
        log: &'a RefCell<Vec<&'static str>>,
        name: &'static str,
    }

    impl Drop for Observed<'_> {
        fn drop(&mut self) {
            self.log.borrow_mut().push(self.name);
        }
    }

    /// The projection a backend would generate for field `.0` of a `(String, i32)`.
    ///
    /// A safe `fn` whose body is `unsafe`: `addr_of_mut!` computes a field address WITHOUT
    /// dereferencing, so it is defined even when the surrounding value is partially moved — which
    /// is precisely why partial access must go through this shape rather than through `&mut T`.
    fn project_0(p: *mut (String, i32)) -> *mut String {
        unsafe { core::ptr::addr_of_mut!((*p).0) }
    }

    fn project_1(p: *mut (String, i32)) -> *mut i32 {
        unsafe { core::ptr::addr_of_mut!((*p).1) }
    }

    #[test]
    fn a_slot_starts_dead_and_becomes_live_on_write() {
        let mut slot: ValueSlot<i32> = ValueSlot::dead();
        assert_eq!(slot.state(), SlotState::Dead);
        slot.write(7);
        assert_eq!(slot.state(), SlotState::Whole);
        assert_eq!(*slot.get(), 7);
    }

    #[test]
    fn take_yields_the_value_and_marks_the_slot_dead() {
        let mut slot: ValueSlot<String> = ValueSlot::dead();
        slot.write("owned".to_string());
        assert_eq!(slot.take(), "owned");
        assert_eq!(
            slot.state(),
            SlotState::Dead,
            "take must mark the slot dead"
        );
    }

    /// The property the whole module exists for: **Rust never destroys the contents.**
    #[test]
    fn dropping_a_live_slot_does_not_run_the_value_destructor() {
        let log = RefCell::new(Vec::new());
        {
            let mut slot: ValueSlot<Observed> = ValueSlot::dead();
            slot.write(Observed {
                log: &log,
                name: "leaked",
            });
        }
        assert!(
            log.borrow().is_empty(),
            "Rust scope exit must NOT destroy a slot's contents; got {:?}",
            log.borrow()
        );
    }

    #[test]
    fn explicit_drop_runs_the_destructor_exactly_once_and_marks_the_slot_dead() {
        let log = RefCell::new(Vec::new());
        let mut slot: ValueSlot<Observed> = ValueSlot::dead();
        slot.write(Observed {
            log: &log,
            name: "dropped",
        });
        slot.drop_value();
        assert_eq!(*log.borrow(), vec!["dropped"]);
        assert_eq!(slot.state(), SlotState::Dead);
    }

    #[test]
    fn a_moved_out_value_is_destroyed_by_its_new_owner_only() {
        let log = RefCell::new(Vec::new());
        let mut slot: ValueSlot<Observed> = ValueSlot::dead();
        slot.write(Observed {
            log: &log,
            name: "moved",
        });
        let taken = slot.take();
        assert!(log.borrow().is_empty(), "take must not destroy anything");
        drop(taken);
        assert_eq!(
            *log.borrow(),
            vec!["moved"],
            "the new owner destroys it once"
        );
    }

    #[test]
    fn drop_with_runs_mir_directed_glue_exactly_once() {
        let log = RefCell::new(Vec::new());
        let mut slot: ValueSlot<i32> = ValueSlot::dead();
        slot.write(5);
        slot.drop_with(|v| {
            log.borrow_mut().push("glue");
            assert_eq!(*v, 5, "glue must see the value it is destroying");
        });
        assert_eq!(*log.borrow(), vec!["glue"]);
        assert_eq!(slot.state(), SlotState::Dead);
    }

    #[test]
    fn a_slot_can_be_rewritten_after_being_emptied() {
        let mut slot: ValueSlot<String> = ValueSlot::dead();
        slot.write("one".to_string());
        assert_eq!(slot.take(), "one");
        slot.write("two".to_string());
        assert_eq!(slot.take(), "two");
    }

    // ------------------------------------------------- partial-move soundness --
    //
    // These are the cases the owner review identified as unsound in the previous two-state
    // design. Each must be REFUSED rather than reaching undefined behaviour, and each is written
    // to be meaningful under Miri: the whole point is that no `&T`/`&mut T`/`T` is ever
    // materialised over partially moved storage.

    #[test]
    fn a_sub_place_move_makes_the_slot_partial_not_whole() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("first".to_string(), 9));
        let first: String = slot.move_field(project_0);
        assert_eq!(first, "first");
        assert_eq!(
            slot.state(),
            SlotState::Partial,
            "the other drop units are still live, but the value is no longer complete"
        );
        // The untouched Copy element is still readable -- through a RAW projection, never
        // through `&T`.
        assert_eq!(slot.copy_field(project_1), 9);
        slot.finish_partial();
        assert_eq!(slot.state(), SlotState::Dead);
    }

    #[test]
    #[should_panic(expected = "PARTIAL")]
    fn move_field_then_get_is_refused() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("gone".to_string(), 1));
        let _ = slot.move_field(project_0);
        let _ = slot.get();
    }

    #[test]
    #[should_panic(expected = "PARTIAL")]
    fn move_field_then_get_mut_is_refused() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("gone".to_string(), 1));
        let _ = slot.move_field(project_0);
        let _ = slot.get_mut();
    }

    #[test]
    #[should_panic(expected = "PARTIAL")]
    fn move_field_then_take_is_refused() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("gone".to_string(), 1));
        let _ = slot.move_field(project_0);
        let _ = slot.take();
    }

    #[test]
    #[should_panic(expected = "PARTIAL")]
    fn move_field_then_drop_value_is_refused() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("gone".to_string(), 1));
        let _ = slot.move_field(project_0);
        slot.drop_value();
    }

    #[test]
    #[should_panic(expected = "PARTIAL")]
    fn move_field_then_drop_with_is_refused() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("gone".to_string(), 1));
        let _ = slot.move_field(project_0);
        slot.drop_with(|_| {});
    }

    /// Moving a SECOND, different field is legal — it is the whole-value operations that are
    /// refused, not further field-precise ones. MIR's drop flags are what prevent moving the
    /// *same* unit twice; this type does not duplicate that bookkeeping.
    #[test]
    fn a_second_distinct_field_move_is_permitted() {
        let mut slot: ValueSlot<(String, String)> = ValueSlot::dead();
        slot.write(("a".to_string(), "b".to_string()));
        fn p0(p: *mut (String, String)) -> *mut String {
            unsafe { core::ptr::addr_of_mut!((*p).0) }
        }
        fn p1(p: *mut (String, String)) -> *mut String {
            unsafe { core::ptr::addr_of_mut!((*p).1) }
        }
        assert_eq!(slot.move_field(p0), "a");
        assert_eq!(slot.move_field(p1), "b");
        assert_eq!(slot.state(), SlotState::Partial);
        slot.finish_partial();
    }

    /// Field-precise destruction of a partial value, with the destructor observed.
    #[test]
    fn a_field_can_be_dropped_precisely_while_the_slot_is_partial() {
        let log = RefCell::new(Vec::new());
        let mut slot: ValueSlot<(Observed, i32)> = ValueSlot::dead();
        slot.write((
            Observed {
                log: &log,
                name: "field",
            },
            4,
        ));
        fn p0<'a>(p: *mut (Observed<'a>, i32)) -> *mut Observed<'a> {
            unsafe { core::ptr::addr_of_mut!((*p).0) }
        }
        slot.drop_field_with(p0, |field| {
            // SAFETY-equivalent for the test: the field is live and this is its only destruction.
            unsafe { core::ptr::drop_in_place(field) };
        });
        assert_eq!(*log.borrow(), vec!["field"]);
        assert_eq!(slot.state(), SlotState::Partial);
        slot.finish_partial();
    }

    #[test]
    #[should_panic(expected = "WHOLE")]
    fn finish_partial_on_a_whole_slot_is_refused() {
        let mut slot: ValueSlot<String> = ValueSlot::dead();
        slot.write("still here".to_string());
        slot.finish_partial();
    }
}
