# C6-REFERENCE-MATRIX — WP-C6.1f-a

**Track:** A (Claude)
**Status:** C6.1f-a COMPLETE — classification only, no source change
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
  `f(m)` where `f` takes `&P`. MIR passes `&mut P` where `&P` is required.
- **Array → slice unsizing** — `f(&a)` where `a: [Int32; 3]` and `f` takes `&[Int32]`.

Plus a third at the front end: **`&mut` parameters are moved, not reborrowed**
(`fn g(m: &mut P) { f(m); f(m); }` → E0100 "use of moved value"). The same missing concept surfaces
as **two different failures in two different phases** — E0100 at typecheck when passed to a
function, and "move from possibly-moved place" at MIR verify when used as a method receiver.

Reborrowing is §2 item 5 and is genuinely independent of items 1/2/6. It could be fixed first and on
its own.

---

## 2. The matrix

### Item 1 — references stored in user locals

| Case | Result | Detail |
|---|---|---|
| `let r = &p; r.get()` | **BACKEND** | lane: borrow into a non-`Temp` local |
| `let r = &p; r.v` | **BACKEND** | same |
| `let r = &x` (primitive) | **BACKEND** | same |
| `let a = &p; let b = &p;` (two shared) | **BACKEND** | same |
| `let r = &mut p; r.bump(); r.get()` | **VERIFY** | §1.4 reborrow `&mut`→`&` |

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
| `let r = &p; let rr = &r; rr.get()` | **VERIFY** | `&&P` where `&P` expected — repeated auto-deref not lowered (TYPE-METHOD-002 makes it normative) |
| `fn f(r: &&P)` | **FRONT-END (parse)** | "expected a type, found `&&`" — **`&&T` is unspellable**; the lexer's `&&` token is never split in type position |
| `**rr` | **FRONT-END (parse)** | "expected an expression, found `**`" — same class, `**` in expression position |

### Item 5 — reborrowing and mutable exclusivity

| Case | Result | Detail |
|---|---|---|
| `fn g(m: &mut P) -> Int32 { f(m) }`, `f(r: &P)` | **VERIFY** | reborrow `&mut`→`&` |
| `fn g(m: &mut P) { f(m); f(m); }` | **FRONT-END** | E0100 "use of moved value 'm'" — `&mut` params move instead of reborrowing |
| `fn f(m: &mut P) { m.bump(); m.bump(); }` | **VERIFY** | "move from possibly-moved place" — the same missing concept, different phase |
| `fn f(m: &mut P) { m.bump(); m.get() }` | **VERIFY** | reborrow `&mut`→`&` |

### Item 6 — reference returns and provenance

| Case | Result | Detail |
|---|---|---|
| `fn f(r: &P) -> &P { r }` | **BACKEND** | lane: "returning a reference is outside the lane" (a distinct check) |
| `fn f(r: &P) -> &Int32 { &r.v }` | **BACKEND** | lane |
| `fn get(&self) -> &Int32 { &self.v }` | **BACKEND** | lane |

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
| `f(&a)` array→`&[Int32]` | **VERIFY** | §1.4 unsizing |
| `f(&mut a)` array→`&mut [Int32]` | **VERIFY** | same |
| `s[0]` on a `&[Int32]` param | **VERIFY** | same |
| borrow a `Drop`-bearing owner (param) | **RUN** ✅ |
| borrow a `Drop`-bearing owner (local) | **BACKEND** | lane |
| `*b`, `(*b).v`, `b.get()` on `Box<T>` | **FRONT-END** | E0001 / E0304 — **`Box` deref is unimplemented in the front end** |
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
`Vec::as_slice` is unlowered. `Box` **deref** is additionally a front-end gap (E0001/E0304). C6.1f
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
