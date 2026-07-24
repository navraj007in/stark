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

## 6. Landing boundary and corrected diagnosis (owner ruling, 2026-07-24)

Structural Copy (OWN-COPY-001, amended) is implemented and **landed** with one predicate shared by
the type checker, move checker, MIR, HIR interpreter, and native backend. The earlier
"referent-storage stabilization" section is **withdrawn as factually obsolete** — the following is
the corrected finding.

### 6.1 The `Option<&P>` return is NOT a structural-Copy regression

`fn wrap(r: &P) -> Option<&P> { Some(r) }` then `wrap(&p).unwrap().get()` fails native build. But it
fails **identically when `P` is Move** (a `Drop` field → E0502), so it is a **general
borrow-through-return limitation**, not a Copy issue. `unwrap`'s panic-branch match extends the
returned borrow across enough dispatch-loop blocks to collide with the referent's block-0
assignment. Verified boundary:

| Shape | Result |
|---|---|
| inline `let o: Option<&P> = Some(&p); o.unwrap()` | works |
| `wrap(&p) -> H<&P>` then field access | works (accidental backend-shape success) |
| `wrap(&p) -> Option<&P>` then `unwrap` | fails — for Copy **and** Move |

Referent-storage stabilization (slot-backing the Copy referent) was tried and **does not fix it** —
it only changes E0506→E0502, because the conflict is the borrow living across many blocks vs. any
referent mutation, independent of slot-vs-plain storage.

### 6.2 The landing boundary (`emit_types::refuse_borrow_carrying_nominals`)

- **Copy** borrow-carrying nominals in **locals and across blocks** — admitted (structural Copy makes
  them non-slot-backed; they flow through the CD-095 aggregate path).
- **Move** borrow-carrying nominal locals — refused pre-rustc (still slot-backed; the slot pins the
  borrow across the loop).
- **Any function whose return type is a borrow-carrying nominal** — refused pre-rustc, regardless of
  Copy. `wrap -> H<&P>` is treated as accidental success, not supported semantics, and refused
  uniformly. This is a clean **type-based** refusal, not the shape-sensitive one that was rejected.
- **Plain reference returns** (`fn f(r: &P) -> &P`) — supported (`MirTy::Ref`, not a nominal).

### 6.3 The original "uniform returns green" acceptance bar is REVISED

Uniform borrow-carrier returns are **not** achievable in this package: they require solving the
general borrow-through-return-across-many-blocks problem, which fails for Move too and is rooted in
the block-dispatch `loop { match __bb }` defeating rustc's single-assignment view. That is
**`WP-C6.1g-c` (General borrow-through-return / dispatch-loop linearisation)** — an independent
backend/control-flow package. Until it lands, "return a borrow-carrying nominal" is a recorded
deviation and uniform borrow-carrier returns must not be claimed.

### 6.4 Evidence at landing

- `cargo test --lib`: 441/441.
- `native_c5_3_aggregates_enums`: 21/21 (Move stand-ins switched to Drop-bearing types).
- `native_c61f_nominals`: Copy borrow-carrier local/xblock works; Move local and any borrow-carrier
  return refused pre-rustc.
- `c61f_structural_copy`: positive (primitive-field, nested, generic-at-Copy, borrow-carrying,
  all-Copy enum) + negative (`String`, `Vec`, `Box`, `&mut`, `Drop`, mixed) fixtures.
- A DEV-072-class divergence was found and fixed by the fixtures: `borrowck::is_copy_type` ignored
  type arguments (`H<&mut P>` read Copy there, Move in the checker); it now recurses arguments.
