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

---

## 5. Probe (2026-07-24) — lifetime threading done; the residual needs a `ValueSlot` decision

Per §2, no `ValueSlot`/CE4 change is attempted without a probe demonstrating necessity. Here it is.

### 5.1 Lifetime threading is complete and sufficient for most shapes

Generated nominals carry lifetime parameters (`Name<'a>` in the declaration, `Name<'_>` at use
sites — CD-096). The generated types are correct: e.g. `struct H_…<'a> { f0: &'a P }` and
`let _3: ValueSlot<H_…<'_>> = ValueSlot::dead();`. This is what already makes `Option<&T>`,
`Option<Option<&T>>`, `Option<&T>`-in-a-tuple, and every tuple/array of references build and run.

### 5.2 The residual two shapes fail for a reason lifetimes cannot fix

Bounding probe (four programs, native rustc):

| Shape | Borrow-carrier | Slot-backed? | Result |
|---|---|---|---|
| `Option<&Int32>` local | `Option<&_>` | **no** (Copy) | ✅ runs |
| `Option<&P>` local (P non-Copy) | `Option<&_>` | **no** (Copy) | ✅ runs |
| `H<&Int32>` (Copy referent) | `struct H<T>` | **yes** (Move) | ❌ `E0506` |
| `H<&P>` (non-Copy referent) | `struct H<T>` | **yes** (Move) | ❌ `E0502` |

The failure tracks **exactly one variable: is the borrow-carrier slot-backed.** `Option<&T>` is
`Copy`, so it is never slot-backed and always works. A user `struct H<T>` is `Move` even when its
only field is a reference (CD-031: only an `impl Copy` makes a struct `Copy`), so `H<&P>` is
slot-backed and always fails. The referent's own copy-ness is irrelevant — `H<&Int32>` fails too.

### 5.3 Exact mechanism

Every body is one `loop { match __bb { … } }`. For `let h: H<&P> = H { r: &p }`:

```rust
0 => {
    _1.reinit(_2.take());   // &mut _1   (the referent slot)
    _4 = _1.get();          // &_1       (immutable borrow of the referent)
    _5.reinit(H { f0: _4 }); // _4 flows into a ValueSlot that persists across the loop
    _3.reinit(_5.take());    // … and into _3
    ...
```

rustc reports `E0502` on `_1.reinit` with *"immutable borrow later used here"*. In straight-line
order there is no conflict (`reinit` precedes the borrow). The conflict is the **loop back-edge**:
rustc cannot see block 0 runs once, so it assumes `_1.reinit` (a `&mut`) could re-execute while the
borrow held in `_3` — a `ValueSlot` **live for the entire loop** — is still outstanding. This is the
borrow-analog of the `E0381` definite-assignment wall b3 hit; there the dispatch loop defeated
rustc's initialization analysis, here it defeats its borrow analysis.

### 5.4 Why lifetime threading cannot resolve it, and why the slot cannot simply be removed

- The lifetimes are already correct; the problem is a **value that outlives the loop while holding a
  borrow**, not a mis-spelled lifetime. No arrangement of `'a`/`'_` changes that a `ValueSlot`
  local is live across every block.
- Removing the slot is **not** available: it also carries **move** liveness. `_5.take()` moves the
  nominal slot-to-slot, and a prior attempt to skip the slot for no-drop borrow-carriers produced
  `move out of the non-slot place`. The slot is load-bearing for moves independently of drops.

### 5.5 Escalation

This is the `ValueSlot`-versus-Rust-borrow-region tension the C6.1f-a matrix named as the central
design question, now isolated to non-Copy borrow-carrying nominals and demonstrated to be
lifetime-irreducible. Resolving it needs one of:

1. a **slot representation** that carries move/drop liveness without pinning a stored borrow across
   the dispatch loop (runtime-shape change — **CE4-shaped**); or
2. classifying an `impl Copy`-less, all-reference-field nominal as **Copy** so it is never
   slot-backed (a front-end/`TypeContext` semantics change — touches CD-031's Move-by-default rule,
   **CE3-shaped**, and cross-cuts Track B); or
3. a **generated control-flow** structure rustc can linearize per block (a whole-backend change far
   beyond this package).

Per §2 this stops here for an owner decision on the approach rather than proceeding into a
representation change. The current deterministic pre-rustc refusal (CD-096) remains in force and
sound in the meantime.

---

## 6. Follow-on: referent-storage stabilization (owner direction, 2026-07-24)

Structural Copy is implemented and correct across spec + type checker + move checker + MIR +
HIR interpreter + backend derive (the four engines share one predicate, `copy_eligible_types`).
It is **held, not landed**, because it regresses one already-working shape:

```stark
fn wrap(r: &P) -> Option<&P> { Some(r) }
fn main() { let p = P { v: 3 }; let o = wrap(&p); assert_eq(o.unwrap().get(), 3); }
```

fails native build with `E0506` once `struct P { v: Int32 }` becomes structurally `Copy`.

**Diagnosis.** A `Copy` referent lowers to a plain, loop-reassigned local (`_1 = _2` in block 0).
The returned borrow is held across the dispatch loop, and `Option::unwrap`'s panic-branch
discriminant match adds enough blocks that rustc's back-edge reasoning flags `_1`'s own block-0
assignment as conflicting. `H<&P>` returned passes only by having fewer blocks — luck, not a
semantic difference. **This worked before structural Copy, when `P` was slot-backed.**

**The fix is NOT to back out structural Copy** (it is the right semantic move) and **NOT a targeted
refusal** (too shape-sensitive — it depends on caller lowering, `unwrap` expansion, and rustc
back-edge behaviour; it would be a brittle "known bad pattern" rule, not a language rule). The bug
is in **native storage for borrowed Copy referents.** Copy semantics do not require plain-reassigned
backend storage: a value can be surface `Copy` while the backend gives it stable storage when it is
borrowed. b3 already proved a `ValueSlot`-backed referent survives a cross-loop borrow.

**Fix:** an address-taken Copy local whose borrow can escape across generated dispatch-loop blocks
(through a call return, enum wrapper, `unwrap`/`match`, or a later block) is lowered into **stable
(slot) storage** instead of being reassigned inside the loop. The intricate part is the read path:
a Copy value read out of a slot must **deref-copy** (`*slot.get()`), never `take()` — a `take`
would empty the slot and break Copy's defining multi-read.

**Acceptance bar before the structural-Copy commit lands (all in one change):**

Native green:
- `wrap(&p).unwrap().get()` where `P` is structurally Copy
- direct `Option<&P>` return
- `H<&P>` return; local `H<&P>` across blocks
- plain `&P` return
- `Int32` / `Point` Copy reuse

Negative (still Move):
- `String` field; `Box`/`Vec` field; `&mut` field; a `Drop`-bearing nominal; a mixed
  Copy/non-Copy nominal.

Plus: OWN-COPY-001 regenerated into `STARK-Core-v1.md`; positive + negative spec fixtures; broad
three-engine regression green.
