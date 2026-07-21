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
//! An Option-shaped slot may later be admitted as an optimisation for locals MIR dataflow proves
//! have no partial-move paths. That is a narrowing of this general case, not a replacement.
//!
//! # The safety contract
//!
//! Every operation here is a **safe function**, because generated MIR bodies must contain no
//! `unsafe` blocks of their own (§7.8): all unsafe operations route through this reviewed module.
//! Their preconditions are guaranteed by verified MIR, and each one is *checked* rather than
//! assumed — a violation aborts through [`slot_violation`] with a message naming the invariant,
//! which is a compiler defect, never a language-level outcome.
//!
//! Note what is deliberately absent: **`ValueSlot` implements no `Drop`.** A slot that generated
//! code never empties leaks, and that is the intended trade — leaking is what MIR verification
//! exists to exclude, and a Rust destructor here would make the exactly-once obligation
//! unfalsifiable by silently covering for a lowering bug.

use core::mem::{ManuallyDrop, MaybeUninit};

/// A generated-code invariant was violated: a compiler defect, not a language outcome.
///
/// Exit code **102**, deliberately distinct from a STARK trap's 101 — a trap is a defined
/// language outcome that a correct program can reach, whereas reaching here means the backend
/// emitted code contradicting verified MIR. Conflating the two exit codes would let a compiler
/// bug masquerade as a program's own trap.
pub fn slot_violation(what: &str) -> ! {
    eprintln!("error: generated-code invariant violated: {what}");
    eprintln!("  this is a STARK compiler defect, not a fault in the STARK program");
    std::process::exit(102);
}

/// Controlled storage for one non-`Copy` MIR local.
///
/// Whole-place liveness lives here. Sub-place (drop-unit) liveness stays where MIR already puts
/// it — in the drop-flag locals MIR's own drop elaboration produces — so this type does not
/// invent a second, parallel notion of which fields are live.
pub struct ValueSlot<T> {
    storage: MaybeUninit<ManuallyDrop<T>>,
    live: bool,
}

impl<T> ValueSlot<T> {
    /// A slot holding nothing. Every non-`Copy` local starts here, which is what lets generated
    /// code stop fabricating a default value it then immediately overwrites.
    pub const fn dead() -> Self {
        ValueSlot {
            storage: MaybeUninit::uninit(),
            live: false,
        }
    }

    pub fn is_live(&self) -> bool {
        self.live
    }

    /// Destination write: initialise the slot with `value`.
    ///
    /// The slot must be dead. Writing over a live slot would abandon the previous value without
    /// destroying it — a leak MIR never asks for, since MIR emits an explicit `Drop` before any
    /// reassignment of a live place.
    pub fn write(&mut self, value: T) {
        if self.live {
            slot_violation("write to a live slot (MIR must Drop before reassigning a live place)");
        }
        self.storage = MaybeUninit::new(ManuallyDrop::new(value));
        self.live = true;
    }

    /// Shared place access.
    pub fn get(&self) -> &T {
        if !self.live {
            slot_violation("read of a dead slot");
        }
        // SAFETY: `live` is set only by `write`, which initialises `storage`, and cleared by
        // every operation that de-initialises it (`take`, `drop_value`). So `live` implies
        // initialised.
        unsafe { &*self.storage.as_ptr() }
    }

    /// Mutable place access.
    pub fn get_mut(&mut self) -> &mut T {
        if !self.live {
            slot_violation("mutable access to a dead slot");
        }
        // SAFETY: as `get`.
        unsafe { &mut *self.storage.as_mut_ptr() }
    }

    /// Whole-value move out. The slot becomes dead **before** the value is handed over, so no
    /// path can observe a slot that is simultaneously live and moved-from.
    pub fn take(&mut self) -> T {
        if !self.live {
            slot_violation("take from a dead slot (a moved-from local was moved again)");
        }
        self.live = false;
        // SAFETY: `live` was true, so `storage` was initialised; it is now marked dead, so this
        // read cannot be repeated and nothing else will read the bytes.
        let held = unsafe { self.storage.as_ptr().read() };
        ManuallyDrop::into_inner(held)
    }

    /// Typed sub-place move: move ONE field or constant-index element out, leaving the rest of
    /// the value in place.
    ///
    /// The slot stays live, because the *other* drop units are still live and MIR tracks them
    /// individually — collapsing this into whole-local liveness is exactly what §7.6 forbids.
    /// After this call the storage holds a partially moved value, which is why the storage type
    /// must be `MaybeUninit`: those bytes no longer form a valid complete `T`.
    ///
    /// `project` names the sub-place. Generated code passes a closure such as `|v| &mut v.f0`;
    /// it is a closure in the *generated Rust*, not a STARK closure, so it implies nothing about
    /// Core v1's feature set.
    pub fn move_sub<F>(&mut self, project: impl FnOnce(&mut T) -> &mut F) -> F {
        if !self.live {
            slot_violation("sub-place move out of a dead slot");
        }
        // SAFETY: `live` implies initialised. `project` yields a pointer to one sub-place of that
        // value; reading it moves exactly that sub-place out. The caller's MIR drop flags record
        // that this unit is now dead, so nothing reads it again, and the surrounding value is
        // never treated as a whole `T` afterwards.
        let value = self.get_mut();
        let field: *mut F = project(value);
        unsafe { field.read() }
    }

    /// Explicit drop-unit destruction: run `T`'s destructor exactly once.
    ///
    /// The slot is marked dead **before** the destructor runs (§7.5's ordering requirement). If
    /// the destructor itself traps, the abort path sees a dead slot and cannot re-enter this
    /// value — which is what makes "exactly once" hold even when the drop glue fails.
    pub fn drop_value(&mut self) {
        if !self.live {
            slot_violation("drop of a dead slot (a value was dropped twice)");
        }
        self.live = false;
        // SAFETY: `live` was true, so `storage` was initialised; marked dead first, so this can
        // neither repeat nor race with a later read.
        let mut held = unsafe { self.storage.as_ptr().read() };
        // `ManuallyDrop::drop` is the only place in the generated program where a `T` destructor
        // runs, and it runs because MIR's `Drop` terminator said so.
        unsafe { ManuallyDrop::drop(&mut held) };
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

    #[test]
    fn a_slot_starts_dead_and_becomes_live_on_write() {
        let mut slot: ValueSlot<i32> = ValueSlot::dead();
        assert!(!slot.is_live());
        slot.write(7);
        assert!(slot.is_live());
        assert_eq!(*slot.get(), 7);
    }

    #[test]
    fn take_yields_the_value_and_marks_the_slot_dead() {
        let mut slot: ValueSlot<String> = ValueSlot::dead();
        slot.write("owned".to_string());
        assert_eq!(slot.take(), "owned");
        assert!(!slot.is_live(), "take must mark the slot dead");
    }

    /// The property the whole module exists for: **Rust never destroys the contents.** A slot
    /// left live at scope exit leaks, and that is intended — leaking is what MIR verification
    /// excludes, whereas a Rust destructor here would silently cover for a lowering bug and make
    /// the exactly-once obligation unfalsifiable.
    #[test]
    fn dropping_a_live_slot_does_not_run_the_value_destructor() {
        let log = RefCell::new(Vec::new());
        {
            let mut slot: ValueSlot<Observed> = ValueSlot::dead();
            slot.write(Observed {
                log: &log,
                name: "leaked",
            });
            // `slot` goes out of scope here, still live.
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
        assert!(!slot.is_live(), "drop must mark the slot dead");
    }

    /// A moved-out value's destructor belongs to whoever received it, and must NOT also run at
    /// the source — §7.3's "transfer without running Drop at the source".
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

    /// A sub-place move leaves the slot LIVE, because MIR tracks the remaining drop units
    /// individually. Collapsing this into whole-local liveness is what §7.6 forbids.
    #[test]
    fn a_sub_place_move_leaves_the_slot_live() {
        let mut slot: ValueSlot<(String, i32)> = ValueSlot::dead();
        slot.write(("first".to_string(), 9));
        let first: String = slot.move_sub(|v| &mut v.0);
        assert_eq!(first, "first");
        assert!(
            slot.is_live(),
            "the other drop units are still live; whole-local liveness would lose that"
        );
        assert_eq!(slot.get().1, 9, "the untouched element is still readable");
    }

    #[test]
    fn a_slot_can_be_rewritten_after_being_emptied() {
        let mut slot: ValueSlot<String> = ValueSlot::dead();
        slot.write("one".to_string());
        assert_eq!(slot.take(), "one");
        slot.write("two".to_string());
        assert_eq!(slot.take(), "two");
    }
}
