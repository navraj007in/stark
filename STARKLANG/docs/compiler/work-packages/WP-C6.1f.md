# WP-C6.1f — General Reference Storage, Reborrowing, and Provenance

**Track:** A (Claude) — semantic integration lead
**Status:** OPEN (authorised by the C6.2b F3 owner ruling, 2026-07-23)
**Base:** `main` @ CD-087
**Gate:** C6
**Escalation boundaries:** any MIR or verifier **contract** change stops for **CE3**; any runtime
representation or ABI change stops for **CE4**.

---

## 1. Why this package exists

C5 explicitly deferred *"General references and full borrow/lifetime representation"* to C6 (C5 exit
report, "Listed deviations and future ownership"). The C6 entry plan then distributed that
deferral's **consumers** across two packages — C6.2b needs shared and nested-reference receivers
(WP-C6-ENTRY §18), C6.3b needs slice provenance, `Box` borrow/deref, mutable views and returned
references (WP-C6-ENTRY §25) — **without assigning the shared prerequisite itself to any package.**

C6.2b's §18 probe surfaced the consequence as finding **F3**: `let r = &p; r.get()` is refused by
the backend as outside the C5 *ephemeral reference lane*, though the front end and the HIR oracle
both accept it.

**This is a scope correction, not a defect in the completed ownership work.** C6.1a–e closed
correctly; C6.1f is opened because the deferral was never assigned, not because C6.1's Drop and
ownership results were wrong.

### Why it is not part of C6.2b

Method resolution merely *exposed* the gap. Per WP-C6-ENTRY §18, "the backend consumes the
front-end-selected callee. It does not redo lookup" — C6.2b's job ends at callee selection, whereas
the actual problem is **reference storage, liveness, provenance, MIR verification and native
emission**. Absorbing F3 into C6.2b would put reference representation inside a method-resolution
package.

### Why Track A owns it

The work intersects ownership-liveness, MIR lowering and verification, `ValueSlot` conventions, and
backend place emission — all Track A areas. Track C is expressly prohibited from changing
ownership-liveness. Track B retains method-selection behaviour built *on top of* the reference
contract this package establishes (notably F4's repeated auto-deref selection).

### What must NOT be done

The current backend validator deliberately admits only compiler temporaries within a single basic
block, and expressly rejects user bindings, cross-block flow, aggregate storage, and returned
references. **Removing a check so that `let r = &p;` passes is an unsafe patch, not an
implementation of F3.** The validator's restrictions are load-bearing until a reference
representation exists that makes them unnecessary.

---

## 2. Scope

At minimum, all ten:

| # | Item |
|---|---|
| 1 | References stored in user locals |
| 2 | Reference flow across basic blocks |
| 3 | Shared and mutable reference parameters |
| 4 | Nested references and repeated dereference |
| 5 | Reborrowing and mutable exclusivity |
| 6 | Reference returns and provenance validation |
| 7 | Owner move/drop while borrowed |
| 8 | Array-, struct-, `Box`-, `Vec`- and slice-derived references required by Core v1 |
| 9 | HIR/MIR/native agreement, plus negative verifier tests |
| 10 | **No NLL expansion** — Core v1's lexical borrow duration is unchanged |

Item 10 is a hard constraint, not a default. Core v1 borrows bound with `let` are lexically scoped
to end-of-block (03-Type-System "References and Lifetimes"), and C6.2b confirmed the front end
enforces exactly that: `let r = &mut p; r.bump(); p.get()` is **correctly** rejected (E0101). This
package must not make that program compile. A conflict that only NLL would resolve is evidence of a
test written against Rust intuition, not a gap to close.

Also normative and in scope: a returned reference must derive from a reference parameter and takes
the *shortest* input lifetime; struct/enum declarations cannot write reference field types, though
generic instantiation (`Option<&T>`) is allowed and produces a borrow-carrying value.

---

## 3. Non-goals

- No NLL, no lifetime annotations, no `Rc`/`RefCell`, no raw pointers, no `unsafe`, no general
  `Deref` trait (WP-C6-ENTRY §25: "Do not add a general Deref trait unless Core v1 defines it").
- No method-selection changes — F4's repeated auto-deref *selection* is Track B's, after this
  package lands the representation it needs.
- No relaxation of the lexical borrow rule (§2 item 10).

---

## 4. Method

The C6.1 discipline applies, including the correction recorded at C6.1b and confirmed at C6.2a:
**probe by native execution, not by emit success or a green suite.** A shape that compiles is not a
shape that runs, and a mechanism covered by tests is not the same as the surface that uses it.

1. **C6.1f-a — matrix.** Probe all ten scope items end-to-end
   (`parse → resolve → typecheck → HIR-run → lower → verify → emit → native-run`) and classify each
   as working / front-end-refused / verifier-refused / backend-refused / **accepted-but-wrong**.
   Record in a new `C6-REFERENCE-MATRIX.md`. Do not design before the matrix exists.
2. **C6.1f-b…** — implementation sub-packages driven by what the matrix finds, each with native
   regression tests and three-engine agreement.
3. **Negative tests are mandatory**, not optional: every rejection the validator currently makes
   must either survive with a test pinning it, or be replaced by a *stronger* check — never simply
   deleted.

---

## 5. Escalation

| Trigger | Class |
|---|---|
| New MIR projection/rvalue/terminator, or any change to `mir_version` or a verifier rule | **CE3** |
| Any change to runtime representation, `ValueSlot` layout, ABI, or the target-layout contract | **CE4** |
| Discovering that Core v1's lexical borrow rule cannot express a required Core program | **CE1/CE2** (spec question — do not resolve in-package) |

`C6-SHARED-CONTRACTS.md` is frozen; a needed change to it is a stop-and-escalate, not an edit.

---

## 6. Exit criteria

- All ten §2 items classified, and each either implemented with native evidence or recorded as a
  deviation with an owner-approved disposition.
- `let r = &p; r.get()` and the §18 shared/nested-reference receiver rows build and run natively,
  unblocking C6.2b's remaining matrix rows.
- The C6.3b prerequisites (slice provenance, `Box` borrow/deref, mutable views, returned
  references) are satisfied or explicitly scoped out with owner approval.
- E0101 and the other lexical-borrow rejections still fire, with tests pinning them.
- Three-engine agreement across the new corpus; `fmt --check` and strict workspace `clippy` clean.
