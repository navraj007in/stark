# C6-REFERENCE-MATRIX — WP-C6.1f-a

**Track:** A (Claude)
**Status:** C6.1f-a COMPLETE. **C6.1f-b1 COMPLETE (CD-090)**; **b2 COMPLETE incl. generic callees (CD-092/CD-098)**; **b3 stored references COMPLETE (CD-093)** — §10; **returning a reference COMPLETE (CD-094)** — §11; **aggregates: tuples/arrays (CD-095) — §12; borrow-carrying nominals mostly land (CD-096) — §13.**
**Base:** `main` @ CD-088
**Method:** every case driven end-to-end
(`parse → resolve → typecheck → HIR-run → lower → verify → emit → native-run`), 51 cases across the
ten `WP-C6.1f.md` §2 scope items. Per the C6.1b method correction, reconfirmed at C6.2a: **probe by
native execution, never by emit success or a green suite.**

Classification: **RUN** (all three engines complete) · **BACKEND** (MIR verifies *and the MIR
interpreter runs it correctly*, only generated-Rust emission refuses) · **VERIFY** (MIR verifier
refuses; front end accepted) · **FRONT-END** (parse/resolve/typecheck refuses) · **LOWER** ·
**CORRECT-REJECT** (refusal is conformant and must be preserved).

---

## 1. Four structural findings

These matter more than any individual row, and they narrow the package substantially.

### 1.1 There is no silent miscompilation anywhere in the reference surface

In **every** case that reached two or more engines, HIR and MIR agreed. No case produced different
answers across engines, and no case was **accepted-but-wrong**. Every gap in this matrix is a
**refusal**, not a wrong result. C6.1f is therefore a capability package, not a soundness repair —
which is the opposite of the C6.2b F1 finding it is sequenced behind.

### 1.2 MIR already represents and executes references-in-locals correctly

All fifteen **BACKEND** rows verify under the MIR verifier *and run to a correct answer under the
MIR interpreter* (`hir=0 mir=0`). Storing a reference in a user local, flowing it across basic
blocks, returning it, and taking one to a field or array element are all already expressed in MIR
and already executed correctly by the reference engine.

**The gap is generated-Rust emission, not reference representation.** That materially changes the
package's shape from "design a reference representation" to "emit the representation MIR already
has". It does not make it small — see §4 — but it removes the largest unknown.

### 1.3 The lane boundary is "freshly-taken borrow", not "reference"

Reference **parameters** work natively today, including storing one in a user local
(`fn f(r: &P) { let q = r; q.get() }` → RUN). What the lane refuses is *materialising a new borrow*
(`Rvalue::RefOf`) into anything but a same-block compiler temporary.

So the backend already emits, passes, stores and calls through reference **values**. What it cannot
do is create one that outlives a statement. This is a much sharper boundary than "general references
are unsupported".

### 1.4 Two independent mechanisms are missing, and they are not reference *storage*

The nine **VERIFY** rows are not about storage at all. They are two absent conversions:

- **Reborrow `&mut T` → `&T`** — `let m = &mut p; m.get()` (a `&self` method on a `&mut` receiver),
  `f(m)` where `f` takes `&P`. MIR passes `&mut P` where `&P` is required. *(§8 later split this by
  position: the receiver half was a real gap and is fixed; the argument half is a front-end
  over-acceptance.)*
- **Array → slice unsizing** — `f(&a)` where `a: [Int32; 3]` and `f` takes `&[Int32]`. *(TYPE-COERCE-003;
  moved to C6.3b by CD-091, since slice-parameter representability is its prerequisite.)*

Plus a third at the front end: **`&mut` parameters are moved, not reborrowed**
(`fn g(m: &mut P) { f(m); f(m); }` → E0100 "use of moved value"). The same missing concept surfaces
as **two different failures in two different phases** — E0100 at typecheck when passed to a
function, and "move from possibly-moved place" at MIR verify when used as a method receiver.

Reborrowing is §2 item 5 and is genuinely independent of items 1/2/6. It could be fixed first and on
its own.

---

## 2. The matrix

Rows are as observed at **C6.1f-a**. Where a later sub-package changed a row, the change is marked
inline as **→ (b1)** and explained in §8; §1 records the C6.1f-a state and is not rewritten.

### Item 1 — references stored in user locals

| Case | Result | Detail |
|---|---|---|
| `let r = &p; r.get()` | **BACKEND** | lane: borrow into a non-`Temp` local |
| `let r = &p; r.v` | **BACKEND** | same |
| `let r = &x` (primitive) | **BACKEND** | same |
| `let a = &p; let b = &p;` (two shared) | **BACKEND** | same |
| `let r = &mut p; r.bump(); r.get()` | **VERIFY** → **BACKEND (b1)** | reborrow fixed by b1; now only the lane blocks it |

### Item 2 — reference flow across basic blocks

| Case | Result | Detail |
|---|---|---|
| ref used inside `if` | **BACKEND** | lane: created block ≠ use block |
| ref used inside `while` | **BACKEND** | same |
| `let r = if c { &p } else { &q };` | **FRONT-END** | E0103 "cannot return reference to local stack variable" — **misleading: nothing is returned**. Also fires when both branches borrow the *same* owner. Over-rejection candidate |

### Item 3 — shared and mutable reference parameters ✅

| Case | Result |
|---|---|
| `fn f(r: &P)` | **RUN** |
| `fn f(r: &mut P)` | **RUN** |
| `fn f(r: &P) { let q = r; q.get() }` | **RUN** — a param ref stored in a user local |
| two reference params | **RUN** |
| shared param used twice | **RUN** |

**Item 3 is already complete.** This is the §1.3 boundary in evidence.

### Item 4 — nested references and repeated dereference

| Case | Result | Detail |
|---|---|---|
| `let r = &p; let rr = &r; rr.get()` | **VERIFY** → **BACKEND (b1)** | repeated auto-deref now lowers and verifies (§8.3); only the lane blocks it |
| `fn f(r: &&P)` | **FRONT-END (parse)** | "expected a type, found `&&`" — **`&&T` is unspellable**; the lexer's `&&` token is never split in type position |
| `**rr` | **FRONT-END (parse)** | "expected an expression, found `**`" — same class, `**` in expression position |

### Item 5 — reborrowing and mutable exclusivity

| Case | Result | Detail |
|---|---|---|
| `fn g(m: &mut P) -> Int32 { f(m) }`, `f(r: &P)` | **VERIFY** | **argument** position — the `&mut T -> &T` reference coercion is normative here; an implementation gap. **Revised b2**, §8.4 |
| `fn g(m: &mut P) { f(m); f(m); }` | **FRONT-END** | E0100 — must **re-borrow rather than move** the `&mut` at the expected-type boundary. **Revised b2**, §8.4 |
| `fn f(m: &mut P) { m.bump(); m.bump(); }` | **VERIFY** → **✅ native (b1)** | receiver position; each re-borrow is a temporary borrow |
| `fn f(m: &mut P) { m.bump(); m.get() }` | **VERIFY** → **✅ native (b1)** | receiver position |

### Item 6 — reference returns and provenance

| Case | Result | Detail |
|---|---|---|
| `fn f(r: &P) -> &P { r }` | **BACKEND** → **✅ native (ret)** | §11 |
| `fn f(r: &P) -> &Int32 { &r.v }` | **BACKEND** → **✅ native (ret)** | §11 |
| `fn get(&self) -> &Int32 { &self.v }` | **BACKEND** → **✅ native (ret)** | §11 |

### Item 7 — owner move/drop while borrowed

| Case | Result | Detail |
|---|---|---|
| move owner while borrowed | **CORRECT-REJECT** | E0101 ✅ |
| move a `Drop` owner while borrowed | **CORRECT-REJECT** | E0101 ✅ |
| borrow ends, *then* move | **BACKEND** | lane — the accepted program is blocked only by emission |

### Item 8 — references derived from array / struct / `Box` / `Vec` / slice

| Case | Result | Detail |
|---|---|---|
| `&a[1]` (array element) | **BACKEND** | lane |
| `&p.v` / `&o.i` (struct field, nested) | **BACKEND** | lane |
| `&v[0]` (Vec element) | **BACKEND** | lane |
| `f(&a)` array→`&[Int32]` | **VERIFY** | TYPE-COERCE-003 — normative; **moved to C6.3b** with slice representability (§8.4) |
| `f(&mut a)` array→`&mut [Int32]` | **VERIFY** | same — **C6.3b** |
| `s[0]` on a `&[Int32]` param | **VERIFY** | slice-parameter representability — **C6.3b** |
| borrow a `Drop`-bearing owner (param) | **RUN** ✅ |
| borrow a `Drop`-bearing owner (local) | **BACKEND** | lane |
| `*b`, `(*b).v`, `b.get()` on `Box<T>` | **CORRECT-REJECT** | E0001 / E0304 — Core v1 has **no `Deref` trait** and TYPE-METHOD-002 peels only `&`/`&mut`; `Box` is defined by `new`/`into_inner` alone. Rejecting these is conformant (owner disposition CD-097 item 4) |
| `Box::into_inner`, `v[0]` read, `&str` param | **EMIT (representability)** | `Box`/`Vec`/`str` are not C5-representable — **C6.3, Track C** |
| `Vec::as_slice` | **LOWER** | "a later C4.5e sub-slice" — **C6.3, Track C** |

### Item 9 — HIR/MIR/native agreement

No divergence found (§1.1). Negative verifier tests: §3.

### Item 10 — no NLL expansion ✅

| Case | Result |
|---|---|
| use owner while a `&mut` borrow is live | **CORRECT-REJECT** E0101 ✅ |

---

## 3. The negative corpus — refusals that must survive C6.1f

Six conformant rejections. These pin `WP-C6.1f.md` §2 item 10 and are the likeliest casualties of an
implementation that widens the lane carelessly. **Now locked by permanent tests**
(`starkc/tests/c61f_reference_boundary.rs`), added at C6.1f-a *before* any implementation, because
that is when they are most at risk.

| # | Program | Code |
|---|---|---|
| 1 | two `&mut` borrows of one owner | E0101 |
| 2 | `&` borrow while a `&mut` is live | E0101 |
| 3 | return a reference to a local | E0103 |
| 4 | move an owner while borrowed | E0101 |
| 5 | move a `Drop`-bearing owner while borrowed | E0101 |
| 6 | use owner while a `&mut` borrow is live (the no-NLL case) | E0101 |

Case 6 is the load-bearing one: **Rust's NLL accepts it and Core v1 does not.** Any implementation
that lets generated Rust decide borrow validity will silently start accepting it.

---

## 4. What the lane validator actually enforces

`emit_bodies.rs::validate_ephemeral_references` (CD-062) is a whole-body pre-emission gate with
**five** distinct checks. C6.1f must *replace* these with something stronger, never delete them:

1. a `RefOf` written through a projection;
2. a `RefOf` landing in a non-`Temp` (user) local;
3. a reference value flowing into an aggregate or a non-temporary place;
4. a reference temporary created in one block and used in another;
5. a body whose return type is a reference.

Checks 2–5 map one-to-one onto scope items 1, 6, and 2. Deleting any of them makes `let r = &p;`
"pass" while leaving generated Rust to decide correctness — which §3 case 6 shows is *not* equivalent
to Core v1 semantics.

---

## 5. The central design question for C6.1f-b (stated, not answered)

The package forbids designing before the matrix exists, so this records the question the matrix
raises rather than choosing an answer.

Generated code binds a **non-`Copy` local as a `ValueSlot<T>`**, and a **reference local as a bare
uninitialised Rust reference** (`let mut _3: &P;`), deliberately using rustc's definite-assignment as
a second check on the lane. Consequently:

- a user borrow of a non-`Copy` owner must borrow **through the slot**, while the slot's own
  drop-flag machinery takes `&mut` on that same slot; and
- once a reference outlives a block, **Rust's** borrow checker adjudicates the generated program.

Both halves of that are the crux. Note the direction of the mismatch is favourable for user-level
aliasing — STARK's lexical borrows are *stricter* than NLL, so STARK-accepted programs are broadly a
subset — but §3 case 6 proves the two are **not** ordered in the way an implementer might assume, and
the conflict that actually bites is between user borrows and the slot machinery's own accesses, not
between user borrows themselves.

---

## 6. Explicitly not C6.1f

Track C (C6.3) owns representability: `Box`, `Vec`, `String`/`str` are not C5-representable, and
`Vec::as_slice` is unlowered. `Box` **deref** is NOT a gap — Core v1 defines no `Deref` trait, so rejecting `*b` is conformant (CD-097 item 4). C6.1f
should assume these arrive from C6.3 and must not implement them; the reference machinery it builds
is what C6.3b then consumes for slice provenance and `Box` borrow/deref.

---

## 7. Proposed sub-package split — **requires owner approval before implementation**

Derived from the matrix, ordered by independence. C6.1f-b1 is separable and unblocks nine rows
without touching the lane at all.

| Sub-package | Content | Rows unblocked |
|---|---|---|
| **C6.1f-b1** | Reborrowing: `&mut T` → `&T` at receiver and argument position, and `&mut` params reborrowed rather than moved (§1.4) | 9 **VERIFY** rows |
| **C6.1f-b2** | Array → slice unsizing at argument position | 3 of those rows (overlaps b1's set) |
| **C6.1f-b3** | The lane replacement: references in user locals, across blocks, and returned, with provenance validation replacing checks 2–5 (§4) | 15 **BACKEND** rows |
| **C6.1f-b4** | Nested references: split `&&`/`**` in type and expression position (parser), then repeated auto-deref lowering. Selection half stays Track B per the F4 ruling | 3 rows |
| **C6.1f-b5** | Front-end over-rejections: the E0103 `if`-branch message and behaviour | 2 rows |

Open question for the owner: whether **b1/b2 land first** as independent conformance fixes — they
need no lane change and no CE3 — or whether the whole package waits on the b3 lane design.


---

## 8. C6.1f-b1 CLOSED, and why b2 cannot proceed as scoped (CD-090)

### 8.1 The probe that changed the scope — **AMENDED (CD-091)**

Before implementing, the explicit forms were probed. **Explicit re-borrow syntax already works
end-to-end natively**: `f(&*m)` and `f(&mut *m); f(&mut *m);` both run with all three engines at 0.

> **CORRECTION (owner ruling, CD-091).** The original §8.1 read TYPE-METHOD-002's "no
> argument-position ... user coercion exists" as excluding **all** argument-position conversion.
> **That was wrong, and the error was mine: I cited TYPE-METHOD-002 without checking the coercion
> rules it defers to.**
>
> A function parameter is an **expected-type boundary**, and the type system applies the closed set
> of built-in coercions at expected-type boundaries. 03-Type-System "Reference Coercions" gives
> `&mut T -> &T` normatively, and **TYPE-COERCE-003** gives `&[T; N] -> &[T]`,
> `&mut [T; N] -> &mut [T]`, and mutable-weakened-to-shared. TYPE-METHOD-002 prohibits
> argument-position **auto-borrow**, **auto-dereference**, and **user-defined** coercion — not the
> fixed built-in set.
>
> Therefore **the checker is correct to accept these argument forms, and their later
> verifier/backend refusal is an implementation gap, not front-end over-acceptance.** Rejecting them
> would have contradicted frozen Core v1 coercion rules. TYPE-METHOD-002 has been clarified
> editorially to prevent recurrence (a clarification of existing frozen semantics, not an
> amendment).

What survives from the original analysis is the **position split**, which is still real and is what
b1 acted on:

- **Receiver position** — TYPE-METHOD-002 *requires* auto-deref and auto-borrow. A real lowering
  gap. **This is b1, and it is closed.**
- **Argument position** — no auto-borrow/auto-deref, **but the built-in coercions do apply**. These
  rows are an implementation gap in the verifier/backend. **This is the revised b2** (§8.4).

### 8.2 What b1 changed

Lowering passed an already-reference receiver through as a value. That was wrong twice over: it never
adjusted `&mut T` to `&T`, and it **moved** the reference (`&mut T` is not `Copy`). Receivers are now
dereferenced via `lower_place_autoderef` and **re-borrowed at the method's required mutability**.

Each re-borrow is a *temporary* borrow ending with its statement (03 rule 4), so no borrow duration
changed and Core v1's lexical rule is untouched — the negative corpus still passes unaltered.

| Row | Before | After |
|---|---|---|
| `&mut` receiver → `&self` method | VERIFY | ✅ native |
| `&mut` receiver used twice (`m.bump(); m.bump();`) | VERIFY (V-MOVE-1) | ✅ native |
| `&mut` receiver mixing `&mut self` and `&self` | VERIFY | ✅ native |

### 8.3 Free gain: F4's representation half is done

Because `lower_place_autoderef` peels **every** layer, **repeated auto-deref now lowers and verifies**
— the F4 nested-receiver rows moved from **VERIFY** to **BACKEND (lane)**. They are pinned by
`c61f_b1_nested_reference_receiver_now_lowers_and_verifies`, which deliberately stops at "lowers,
verifies and runs under the MIR interpreter" so b3 does not have to rediscover the gain.

Per the F4 ruling this covers the **MIR/reference-representation** half only. The **parser** half
(`&&T` and `**x` are unspellable — the lexer never splits those tokens) is untouched, and repeated
auto-deref *selection* remains Track B's.

### 8.4 b2 REVISED — expected-type reference weakening (owner ruling, CD-091)

b2 is **retained and narrowed**, not dropped:

> **C6.1f-b2 — Expected-type reference weakening.** Track A implements `&mut T -> &T` at
> expected-type boundaries, at minimum: ordinary function arguments; fully qualified trait-call
> arguments; annotated local initialisation; assignment; return expressions; and aggregate fields
> where applicable. It must **re-borrow rather than move** the `&mut`, preserving the lexical borrow
> rules b1 already proved. This does **not** depend on slice representation and does not wait for
> C6.3.

**Array-to-slice coercion moves to C6.3b.** TYPE-COERCE-003 native execution becomes part of C6.3b
slice representability, covering `n(&a)`, `n(&mut a)` and the already-explicit `n(&a[0..3])`
together. The evidence for the move: `n(&a[0..3])` — no coercion involved — is refused at emission
with *"param 0 is not C5-representable"*, so **slice parameter representation is the prerequisite**.
That prerequisite does **not** justify rejecting `n(&a)` once representation exists; C6.3 already
owns slices, shared/mutable views, range slicing and returned-reference provenance.

**Checker behaviour is fixed by the ruling:** the checker must **not** reject either normative
coercion merely because native support is incomplete. Until C6.3b lands, native build may issue a
deterministic unsupported-profile diagnostic for slice parameters, but `check` must continue to
accept valid Core source. **C6 cannot close while either normative coercion remains unsupported.**


---

## 9. C6.1f-b2 — expected-type reference weakening (CD-092)

Two defects had to be fixed **together**; either alone leaves the boundary unusable.

| Layer | Defect | Fix |
|---|---|---|
| `borrowck.rs` | a `&mut` argument was **consumed**, so `f(m); f(m);` was E0100 | argument-position `&mut T` **re-borrows** (`check_place_available`, no move mark) |
| `mir/lower.rs` | the conversion was never emitted, so MIR verification rejected the call | `weaken_ref_to` re-borrows at the **expected mutability** |

`weaken_ref_to` also covers the **same-mutability** case: passing `&mut T` where `&mut T` is
expected must re-borrow too, or the reference is moved and a second use fails V-MOVE-1 — the
MIR-level twin of the borrowck E0100. Both halves were needed for `f(m); f(m);`.

Each re-borrow is a *temporary* borrow ending with its statement (03 rule 4), so **no borrow
duration changed**: the §3 negative corpus passes unaltered, no-NLL case included.

### Boundary status

| Boundary | Status |
|---|---|
| ordinary function arguments | ✅ **native** |
| fully qualified trait-call arguments | ✅ **native** |
| annotated local initialisation (`let r: &P = m;`) | ✅ weakening emitted — **reaches the lane (b3)** |
| assignment (`r = m;`) | ✅ weakening emitted — **reaches the lane (b3)** |
| return expressions (both `return m;` and a tail `m`) | ✅ weakening emitted — **reaches the lane (b3)** |
| **aggregate fields** (`H<&P> { r: m }`) | ❌ **NOT DONE** — see below |

Three boundaries now fail only at the ephemeral-reference lane, which is b3's job, not b2's: the
weakening itself is correct and the MIR interpreter runs them.

### The one gap, and why it was not guessed at

Aggregate fields need the **expected field types of a generic nominal instantiation** — for
`H<&P> { r: m }`, the declared field type is the parameter `T` and the instantiation supplies
`&P`. `TypeContext::struct_fields` is populated at program level, not inside `FnLowerer`, and no
nominal-generic substitution helper exists in lowering (`impl_generic_subst` covers impl heads,
not struct instantiations).

Substituting incorrectly here would produce a **silent miscompile** rather than a refusal, which is
the one failure mode this whole package has so far been free of (§1.1). The boundary is therefore
left explicitly unimplemented and reported, rather than approximated. It needs a small
nominal-generic substitution helper first.


---

## 10. C6.1f-b3 — stored references (CD-093)

### 10.1 The design question had the wrong answer

§5 flagged the crux as `ValueSlot` versus Rust's borrow checker: a user borrow of a non-`Copy` owner
must go through the slot, whose drop-flag machinery takes `&mut` on that same slot.

**Probing with the lane disabled showed that is not the blocker.** A same-block borrow bound to a
user local already built and ran — **including for a `Drop`-bearing owner**. What failed was:

```text
error[E0381]: used binding `_3` isn't initialized
```

**rustc's definite-assignment analysis, not its borrow checker.** A reference local is declared
uninitialised before the generated block-dispatch `loop { match … }`, assigned inside one arm and
read in another; rustc cannot follow that. No borrow error appeared in any case.

This is the third time in C6 that probing overturned an assumption held before measuring, and the
second where the *predicted* hard part was not the real one.

### 10.2 The fix

A reference bound to a **user** local is declared `Option<&T> = None` — definitely initialised at
its declaration. MIR's own liveness rules still decide whether a read is legal; `unwrap` names a
state MIR has already proven unreachable, the same posture as `slot_violation`.

**Compiler temporaries keep the bare `&T` form.** They are same-block by construction, so rustc's
definite-assignment check still guards them exactly as the lane intended, and every previously
working reference path is byte-identical. The change is confined to the shapes that were refused.

Two details that were not obvious:

- **`Option<&mut T>` is not `Copy`.** Access must re-borrow out of the `Option`
  (`(*_n.as_mut().unwrap())`) rather than move out of it; moving would make a second use fail.
- **Borrowing needs a place expression.** Read mode may substitute a raw-projection **copy** helper
  for a `Copy` field, and `&<copy>` would reference a temporary rather than the field — a silently
  wrong reference, not a compile error. A distinct `PlaceMode::Borrow` keeps the place form.

### 10.3 Lane checks after b3

| # | Check | Status |
|---|---|---|
| 1 | `RefOf` through a projection | **kept** |
| 2 | `RefOf` into a non-`Temp` local | **relaxed** — user bindings admitted; all other kinds refused |
| 3 | reference into an aggregate or non-temporary | **relaxed** for user bindings; **aggregates still refused** |
| 4 | reference temporary used in another block | **kept for temporaries**; user bindings may cross blocks |
| 5 | body returning a reference | **kept** |

Checks were narrowed, never deleted — §4's requirement. The `c61f_reference_boundary.rs` negative
corpus passes unaltered, no-NLL case included.

`native_c5_3_aggregates_enums.rs`'s lane test carried the instruction *"if it is now legitimately
supported, move it to a positive test"*; its **store** case did exactly that, and its **ret** case
stays refused.

### 10.4 Rows now closed, and what is left

Twelve shapes build and run natively: references in user locals (struct, primitive, two
simultaneous shared borrows), across `if` and `while`, `&mut` in a user local, borrows of struct
fields / nested fields / array elements, a `Drop`-bearing owner, borrow-then-move, and the b2
annotated-local weakening that was waiting on the lane.

Still open in C6.1f:

| Item | Where |
|---|---|
| aggregate-field weakening — open only because borrow-carrying nominals are (`WP-C6.1g-a`) | C6.1g-a |
| `&&T` / `**x` unspellable (parser) | b4 |
| the E0103 `if`-branch message | b5 |


---

## 11. Returning a reference (CD-094)

The last of the five ephemeral-lane checks with real semantics behind it. Provenance —
OWN-RETURN-001 rules 2/3, that a returned reference derives only from a reference *parameter* — is
already enforced by the **front end** (E0103), so the backend does not re-check it; it emits, and
the lane's blanket "a reference may never be returned" (check 5) is removed. What made the emission
compile was two mechanisms, each found by probing rather than predicted:

### 11.1 Definite assignment (the E0381 wall again)

A reference that is a `Call` destination, or an `if`/`match` join result, is written in one basic
block and read in another; the generated block-dispatch `loop { match … }` hides that from rustc,
which rejects it E0381 — **exactly the b3 wall, in the caller and in join blocks rather than in a
`let`.** b3's fix generalised cleanly: a reference **temporary that spans more than one block** is
`Option<&T>`-backed, subsuming the two concrete triggers (call-dest, if-join) into the property that
actually matters. Parameters (initialised at entry) and same-block ephemeral temporaries stay bare,
so every previously working path is unchanged.

Two supporting pieces:
- **Return-position access moves out of the `Option`** (`unwrap()`), never re-borrows — a re-borrow
  would borrow from the dying return-slot local and dangle.
- **Projecting through a returned reference** (`f(&p).field`, `f(&p).method()`) materialises the
  call result into a temp and projects through it — the same non-place fallback the `RefOf` and
  receiver paths already used.

### 11.2 Lifetimes (OWN-RETURN-001's shortest-input rule)

`fn pick(a: &T, b: &T) -> &T` needs an explicit lifetime once there are **two or more** reference
parameters (E0106); with zero or one, Rust's own elision suffices (which is why a `&self` accessor
needed nothing). A **single shared `'a`** on every reference parameter and the return encodes the
*shortest of all inputs* (03 rule 3) — the intersection of the input regions.

**Conservative, and reported:** for `pick(a, b) -> a` STARK's shortest is `a`'s lifetime alone (the
return provably derives from `a` only), but the shared `'a` also ties it to `b`. Sound — it never
accepts a program STARK rejects — but it can reject a valid one whose return derives from a
longer-lived subset. Precise per-path provenance (each parameter its own lifetime, the return tied
only to those it derives from) is a later refinement.

### 11.3 Still refused

Returning a reference to a **local** stays E0103 (front end). A reference stored in an **aggregate**
stays a backend refusal (lane check 3) — that is b3/aggregate continuation work. The
`native_c5_3_aggregates_enums.rs` lane test's `ret` case followed its own "move it to a positive
test" instruction (→ `native_c61f_ret_refs.rs`); its remaining case is now the aggregate one.

### 11.4 Lane checks after returning-a-reference

| # | Check | Status |
|---|---|---|
| 1 | `RefOf` through a projection | **kept** |
| 2 | `RefOf` into a non-`Temp` local | admits user bindings **and the return slot**; else refused |
| 3 | reference into an aggregate or disallowed place | **aggregates still refused**; return slot admitted |
| 4 | reference temporary used in another block | temporaries that span blocks are now `Option`-backed rather than refused |
| 5 | body returning a reference | **removed** — provenance is the front end's (E0103) |


---

## 12. Aggregates carrying borrows (CD-095)

OWN-CARRY-001 makes borrow provenance **structural** — it flows through tuples, generic arguments
and enum payloads — so a tuple or array of references is ordinary Core v1, not an escape hatch.
Declared reference *fields* stay forbidden (03 rule 1, front-end E0001), and that is pinned.

### 12.1 The property is "carries a borrow", not "is a reference"

Relaxing the lane to admit aggregates only moved the failure: a **`Copy` aggregate of references**
(`(&T, &T)`, `[&T; N]`) is not slot-backed, so it was declared through `default_value_expr` — which
cannot fabricate a reference, one level down for exactly the reason it cannot fabricate one
directly. Generalising b3's rule from *is a reference* to *carries a reference*
(`emit_types::ty_carries_reference`) fixed the whole class at once: such locals defer initialisation
like a bare reference — `Option<T> = None` when they cross basic blocks, bare-uninitialised when
same-block. A **non-`Copy`** borrow-carrying aggregate is already slot-backed (`ValueSlot::dead()`
needs no default) and is untouched.

Supported and running natively: tuple of two references; tuple of struct references; mixed tuple
(only one element borrowing); array of references; nested borrow-carrying tuple; a borrow-carrying
tuple crossing basic blocks; a tuple of references to **`Drop`-bearing** values.

### 12.2 Borrow-carrying nominals are refused — deliberately, and before rustc

`Option<&T>`, and a user generic instantiated at a reference (`H<&P>`), are **not** supported. A
generated Rust struct/enum has no lifetime parameters, so a reference in a field cannot be spelled
and rustc reports `E0106: missing lifetime specifier` *in the generated crate*.

Letting that happen would break this backend's defining property — an unsupported program must be
refused on **our** side of the boundary, as a named STARK limitation, never as a compiler error in
code the user never wrote. So `emit_types::refuse_borrow_carrying_nominals` refuses it
deterministically pre-rustc, naming the missing capability. This is the case
`native_c5_3_aggregates_enums.rs` now pins (its third rotation: `store` → b3, `ret` → the return
step, `ref_in_tuple` → here, each following that test's own "if it is now legitimately supported,
move it to a positive test" instruction).

**Why tuples work and nominals do not** is the whole distinction: a tuple is a *structural* Rust
type whose lifetimes rustc infers; a generated nominal is a *declared* type that would need explicit
lifetime parameters.

### 12.3 What lifting the nominal restriction needs

Lifetime parameters threaded through generated type declarations (`enum core_x<'a> { … }`) and
**every** use site — field types, local declarations, function signatures, drop glue, variant
construction, and match patterns. It interacts with the shared-`'a` signature machinery §11.2 added
for reference returns. That is a self-contained next step, not a small edit.


---

## 13. Borrow-carrying nominals (CD-096)

A generated nominal is a **declared** Rust type, so unlike a tuple it cannot borrow implicitly: a
reference in a field needs a lifetime parameter or rustc reports `E0106`. Generated nominals now
carry one.

### 13.1 Two spellings, not one

`Name<'a>` in the type's own declaration; `Name<'_>` at every use site. They are **not**
interchangeable — `'_` is illegal in a field type, which has no enclosing binder to infer from,
while a named `'a` at a use site would demand every use site bind one. `emit_types::LifetimePosition`
makes the distinction explicit, and `emit_ty_at` threads it through nested types, so a nominal
inside a nominal's declaration resolves to `'a` while the same type in a local resolves to `'_`.

Only instances that actually carry a borrow gain the parameter, so every existing generated type is
byte-identical.

**Working natively:** `Some(&x)` and `None` at `Option<&T>`; matching on `Option<&P>` and using the
bound reference; `Option<Option<&T>>`; `Option<&T>` inside a tuple; and — checked deliberately — a
plain `Option<Int32>`, to confirm non-borrowing nominals are untouched.

### 13.2 The C6.1f-a design question, finally located

§5 predicted the crux would be `ValueSlot` versus Rust's borrow checker. b3 showed it was *not* the
blocker for plain references (that was definite assignment), and aggregates showed it was not the
blocker for tuples (they are not slot-backed). **It is real here, and only here.**

Two shapes remain refused, both `E0502` in the generated crate:

| Shape | Why |
|---|---|
| **slot-backed** (non-`Copy`) borrow-carrying nominal — a user struct/enum at a reference | Its `ValueSlot`'s destruction and moves need `&mut` on the slot, while the reference it stores still borrows its referent's slot immutably. Rust treats those as overlapping across the local's whole lexical region, even though MIR drops the borrower first. |
| a function **returning** a borrow-carrying nominal | The elided output lifetime keeps the borrow live across the referent's own slot destruction. |

**Removing the slot is not an escape.** It was tried: a `ValueSlot` also carries **move** liveness,
so without it the mover fails instead (`move out of the non-slot place`). The slot is load-bearing
for two independent reasons and only one of them is destruction.

Both shapes are refused **before rustc**, with messages naming the shape and what does work — the
same boundary discipline as everywhere else in this package.

### 13.3 What resolving it would need

Reconciling MIR's slot-based liveness with Rust's borrow regions: either a slot representation whose
destruction does not require `&mut` over the borrow's region, or emitting borrow-carrying nominals
without slots while recovering move liveness some other way. That is a representation change, so it
is **CE4-shaped** and should not be attempted inside this package without a ruling.
