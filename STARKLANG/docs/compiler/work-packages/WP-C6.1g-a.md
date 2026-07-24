# WP-C6.1g-a — Borrow-Carrying Nominal Lifetime Emission

**Track:** A (Claude)
**Status:** OPEN (owner disposition CD-097 item 1, 2026-07-24)
**Base:** `main` @ CD-096
**Blocks:** **Gate C6 closure.** Does *not* block WP-C6.1f closure.

---

## 1. Why

`Option<&T>`, user generic nominals instantiated with borrow-carrying arguments, and their
**storage, passage and return** are **normative Core v1** — OWN-CARRY-001 makes borrow provenance
structural through generic arguments and enum payloads. The current deterministic pre-rustc refusal
is approved **only as a temporary C6.1f deviation**.

C6.1f already landed lifetime parameters on generated nominals, so most instances work:
construction, `None`, matching, nesting, and embedding in a tuple all build and run. Two shapes
remain refused (`C6-REFERENCE-MATRIX.md` §13.2):

1. a **slot-backed** (non-`Copy`) borrow-carrying nominal — a user struct/enum at a reference;
2. a function **returning** a borrow-carrying nominal.

Both fail as `E0502` in the generated crate: the `ValueSlot`'s destruction and moves need `&mut`
while the reference it stores still borrows its referent's slot immutably.

## 2. Approach — as directed

**The initial implementation approach is generated lifetime-parameter threading.**

**No `ValueSlot` change and no CE4 runtime-layout change is authorised without a probe
demonstrating that it is necessary.** If threading alone cannot express these shapes, produce the
probe first — a concrete program, the generated code, and the exact rustc diagnostic showing why a
lifetime-only solution fails — and escalate with it rather than reaching for a representation
change.

Recorded from C6.1f so it is not re-derived: removing the slot was tried and is **not** an escape.
The slot also carries **move** liveness, so without it the mover fails instead
(`move out of the non-slot place`). Any proposal must account for both roles.

## 3. Scope

- Slot-backed borrow-carrying nominals: locals, storage, passage.
- Returning a borrow-carrying nominal.
- The lifetime relationship between a nominal's parameter and the function signature's own
  lifetimes (this meets WP-C6.1g-b at the return boundary — coordinate, do not duplicate).
- Remove the corresponding refusals in `emit_types::refuse_borrow_carrying_nominals` as each shape
  lands; keep the pre-rustc boundary intact for whatever remains.

## 4. Exit criteria

- Both refused shapes build and run natively with three-engine agreement.
- The pre-rustc refusal is removed only for shapes that actually work; anything still unsupported is
  refused deterministically with a named limitation, never left to rustc.
- The C6.1f negative corpus (`c61f_reference_boundary.rs`) passes unaltered — no NLL expansion.
- Full workspace suite, `fmt --check`, strict `clippy` clean.
