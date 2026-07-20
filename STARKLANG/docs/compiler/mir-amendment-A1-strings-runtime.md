# MIR v0.1 Amendment A1 — String Constants and the String/str/Vec/Slice Runtime Surface

Status: **APPROVED under CE3 as an additive MIR v0.1 amendment, runtime surface `0.1-A1`**
(owner decision, 2026-07-19), conditional on rev. 3's four narrow contract-precision
corrections, which this revision incorporates (see §11 revision log). Rev. 1's central design
was approved in principle; rev. 2 resolved eight required corrections; rev. 3 resolves the
final four. Implementation of the C4.5e main body may begin against this revision.
**Current runtime surface after subsequent dated enumerations (§11): `0.1-A8`** (rev. 5
activated Vec iteration as `0.1-A2`; rev. 6 activated the HashMap group and Char ops as
`0.1-A3`; rev. 8 activated checked interior Vec access as `0.1-A4`; rev. 9 activated string
chars iteration as `0.1-A5`; rev. 10 activated shared slice views as `0.1-A6`, completing the
A4 `core-min` MIR runtime surface; rev. 11 activated the `Box<T>` group as `0.1-A7`; rev. 12 activated the EXCLUSIVE slice view as
`0.1-A8`).

Scope class: **narrow additive amendment to MIR v0.1** (`mir.md`, APPROVED CD-028, amended
CD-029). It adds one `Constant` form, one optional `Terminator::Trap` field, **one** additive
in-memory `TypeContext` field (`copy_types`), **two `MirProgram` surface-identifier fields**
(`mir_version`, `runtime_surface`), and a versioned `RuntimeFn` appendix table. It changes no
existing construct's syntax, typing, or semantics. Rationale for staying v0.1 is §8.

Machine-visible surface identifier (owner points 6 + 3): the principal MIR version stays
`MIR_VERSION = "0.1"`. A1 additionally defines

```text
MIR_RUNTIME_SURFACE = "0.1-A1"
```

a machine-visible constant naming the runtime-surface revision, **and stamps both identifiers
onto every `MirProgram`** as fields `mir_version` and `runtime_surface` (§6). A consumer
(verifier or backend) MUST reject a program whose `runtime_surface` it does not support
**before consuming any body** — the constant and dump header alone are not authoritative; the
per-program fields are. Bumping `runtime_surface` (`0.1-A1` → `0.1-A2` …) is how a later dated
enumeration activates reserved ops (§5, owner point 8) without touching `MIR_VERSION`.

---

## 1. `Constant::Str` — what it denotes

```text
Constant ::= Int(i128, MirTy) | Float(f64, MirTy) | Bool | Unit | FnPtr(Instance)
           | Str(String)                                            -- NEW (A1)
```

`Constant::Str(bytes)` denotes **a shared reference to an immutable, program-lifetime UTF-8
string object** — a value of MIR type `Ref { mutable: false, inner: Str }` (`&str`). It is the
lowering of a source string literal, carrying the **decoded** content (escapes already resolved
by the front end), never the source spelling.

It is *not* an owned `String`, and *not* a literal-pool index:

- **Not owned:** per the front end's frozen typing (`Lit::Str → &str`), literals are borrows of
  static string objects. Owned `String` construction is always explicit through the runtime
  surface (`StringFromStr`, §5), mirroring source-level `String::from("...")`.
- **Not a pool reference:** the in-memory constant carries its bytes inline. A program-level
  literal pool would be a new top-level section of the compilation unit — a larger shape change
  than v0.1 needs. Backends may pool/intern **at codegen**, because identity is unobservable
  (§2).

## 2. Ownership, lifetime, identity, encoding

- **Ownership/Drop:** the literal value is a shared `&str` reference; it is `Copy` and never
  dropped. The referenced string object has program lifetime; no drop obligation exists for it.
  (Owned `String` values, by contrast, **are** drop-elaborated — §5a.)
- **Identity:** *unobservable.* Core v1 has no address-of or pointer-equality on `&str`; `==` on
  `str`/`String` is content equality through the runtime surface (§5). Implementations may
  intern, deduplicate, or duplicate literal objects freely. (Precedent: TYPE-FN-001 makes
  function-value identity unobservable.)
- **Encoding:** the carried bytes MUST be valid UTF-8 (TEXT-UTF8-001), enforced by construction
  in-memory and re-checked at any future dump-ingestion boundary (V-STR-1, §6). Byte
  offsets/lengths follow TEXT-INDEX-001 (unsigned UTF-8 byte offsets) throughout the surface.

## 3. MIR interpreter representation

`MirValue` gains two variants:

- `Str(Rc<str>)` — a `&str` value. `Rc` is an unobservable implementation convenience (no
  identity ops exist). Evaluating `Constant::Str` produces one.
- `String(String)` — an owned `String` value. Non-`Copy`; moves under the existing verifier move
  discipline; **drop-elaborated** (§5a).

**`&str` is always self-contained `Str(Rc<str>)` in the interpreter**, including the result of
the interior-reference op `StringAsStr` (§5b) — never a frame path. This is sound and is the key
simplification: an interior `&str` view is read-only and borrowck guarantees the source `String`
cannot mutate for the view's lifetime, so a byte snapshot is observationally identical to a live
borrow (str identity is unobservable). The interpreter therefore needs **no** new interior-
reference machinery for strings, and A1 gains no dependency on the deferred C4.5f frame-
generation work. Backends use true zero-copy borrows (§4); the difference is unobservable.

`&String` arguments arrive as ordinary frame references (existing `Ref` machinery). Runtime str
ops (`StrLen`/`StrEq`/`StrCmp`/…) accept a `Str(Rc<str>)` operand; when the source is a `String`,
lowering inserts `StringAsStr` first, so str ops always see a `Str` value.

## 4. Backend lowering

- **Generated Rust (selected backend, CD-026):** `Constant::Str` → a Rust `&'static str` literal
  (deterministically re-escaped); `MirTy::String` → `std::string::String`;
  `MirTy::Core(Vec,[T])` → `Vec<T'>`. Each §5 `RuntimeFn` maps 1:1 onto the corresponding Rust
  method or a trivial wrapper (`StrCmp` → `Ord::cmp(...)` normalized to −1/0/+1). Trap-capable
  ops emit the same check-then-abort pattern used for `Checked` terminators. **Element/owned
  Drop (owner point 4):** a `Drop` terminator on a `String`/`Vec`-typed place is realized by a
  backend drop-glue routine. For `Vec<T>` with droppable `T`, that routine destroys elements in
  **reverse index order** (§5a) — it does **not** inherit Rust's front-to-back `Vec<T'>::drop`.
  Because the backend performs this explicit STARK-ordered element destruction, it **MUST
  suppress any subsequent automatic (Rust) element drop** for the same buffer (e.g. represent
  the buffer as `Vec<ManuallyDrop<T'>>` or a raw allocation, or give `T'` a no-op Rust `Drop`):
  a Vec element must be destroyed exactly once, through STARK glue. `String` glue reclaims the
  buffer with no element destructors.
- **Future direct backend (kept-open C7 migration):** literal bytes → read-only data section;
  `&str` → fat pointer (ptr, len); `String`/`Vec` → runtime structs (ptr, len, cap) whose
  allocation/free/drop-glue functions belong to the C5.1 runtime ABI. Signatures here are stated
  in MIR types only; no generated-Rust representation is presumed (charter §1.6 rule 9).

## 5. RuntimeFn additions — versioned appendix table (surface `0.1-A1`)

The base contract (§7) requires the runtime surface to be *closed, enumerated, and versioned*;
this table is that enumeration for the String/str/Vec/slice groups. An unknown variant fails
loudly at any backend (V-RT-1). Signatures use MIR types; `T` is the schematic element type
resolved per §6 (owner point 4).

### 5a. Owned-value Drop through runtime glue (owner point 1)

**`String` and `Vec<T>` ALWAYS require runtime drop glue; whether that glue executes user
destructors is conditional on `T`** (owner point 4). `ty_needs_drop` returns true for
`MirTy::String` and for `MirTy::Core(Vec, [T])` **unconditionally** — both own a heap buffer
that must be reclaimed, so both always get a `DropFlag` and a `Drop` terminator, whether or not
any element is itself droppable. In C4.5d's drop-unit model they are **leaf units** (a type
carrying runtime drop glue, analogous to a type with its own `Drop` impl); a `String`/`Vec`
local, or a struct field of such a type, is drop-elaborated like any droppable value. Buffer
reclamation is unobservable, but it is **never omitted and never delegated implicitly to the
backend**: the `Drop { place }` terminator is emitted by lowering and realized as glue by every
consumer.

Drop glue semantics:
- `String`: reclaim the buffer. No elements, no user code, no output; in the interpreter,
  dropping a `String` value simply consumes it. (Always glue; never any destructor execution.)
- `Vec<T>`: **if `T` is droppable, destroy each element through STARK drop glue in reverse index
  order** (`len-1, …, 0`) — running any user element destructors in that order — **then** reclaim
  the buffer; **if `T` is non-droppable, reclaim the buffer only** (no per-element work). Reverse
  index order is the observable ordering fixed by A1 to match the frozen reference oracle
  (`interp.rs::drop_value`, which drops `Vec`/`Array`/`Tuple` elements via `.iter().rev()`),
  stated normatively so backends reproduce it (see §4's suppress-automatic-drop rule) rather than
  inherit Rust's front-to-back order.

This whole-value element destruction happens **only** at a `Drop` terminator — the sanctioned
honesty boundary for running user code. **No `RuntimeFn` call ever runs a user element
destructor.** In particular (owner point 1), `v.clear()` on a droppable `T` is **not** lowered to
an opaque `VecClear` runtime call, which would hide per-element destructors inside a `Call`
terminator where the CFG and verifier cannot see them; it is lowered to an **explicit
pop-and-drop loop** — `loop { match VecPop(&mut v) { Some(x) => Drop(x), None => break } }` —
whose `Drop` terminators are visible drop points (this also yields reverse index order, since
`VecPop` removes from the end). The `VecClear` `RuntimeFn` (§5c) remains available **only for
non-droppable `T`** (buffer reset with nothing to destroy), verifier-enforced. `VecReplace` and
`VecRemove` (§5c) return the displaced element to the caller, who drops it at a visible `Drop`
terminator — likewise never hiding a destructor in a runtime op.

### 5b. Interior-reference operations (owner point 3)

`StringAsStr : (&String) -> &str` is an **interior-reference** operation: at the MIR *type* level
it returns a `&str` borrowing the receiver `String`'s buffer.
- **Borrow/lifetime trust boundary:** upstream borrowck guarantees the `String` outlives the
  `&str` and is not mutated (no overlapping `&mut`) for the view's lifetime. A1 adds no new
  borrow rule; this is the same trust boundary as any `&self`-derived returned reference already
  relied on since C4.5b-2.
- **Backend representation:** a true zero-copy `&str` into the `String`'s buffer.
- **MIR-interpreter representation:** a read-only byte **snapshot** `Str(Rc<str>)` (§3),
  observationally identical because the view is read-only and str identity is unobservable. This
  is *not weaker* than existing reference handling — it sidesteps interior references entirely
  rather than relying on the (deferred) frame-generation defense.

`StringSubstring` is an interior-reference op returning a `&str` over a **byte sub-range**, whose
owner/view relationship and byte-boundary trap semantics (what happens on a non-char-boundary or
out-of-range `[start,end)`) require full definition. **Deferred** to the reserved group (§5d)
until those semantics are specified; not lowered in C4.5e.

### 5c. Lowered in C4.5e

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `StringNew` | `() -> String` | — | `String::new` |
| `StringWithCapacity` | `(UInt64) -> String` | — | capacity hint; unobservable |
| `StringFromStr` | `(&str) -> String` | — | `String::from(lit)` — the only literal→owned path |
| `StringLen` | `(&String) -> UInt64` | — | UTF-8 **byte** length (TEXT-INDEX-001) |
| `StringIsEmpty` | `(&String) -> Bool` | — | |
| `StringPushChar` | `(&mut String, Char) -> Unit` | — | |
| `StringPushStr` | `(&mut String, &str) -> Unit` | — | |
| `StringPopChar` | `(&mut String) -> Option<Char>` | — | logical-enum result (CD-028) |
| `StringClear` | `(&mut String) -> Unit` | — | no elements; buffer reset only |
| `StringAsStr` | `(&String) -> &str` | — | interior reference (§5b) |
| `StringClone` | `(&String) -> String` | — | `Clone` dispatch target for `String` |
| `StringContains` | `(&String, &str) -> Bool` | — | |
| `StrLen` | `(&str) -> UInt64` | — | |
| `StrIsEmpty` | `(&str) -> Bool` | — | |
| `StrToString` | `(&str) -> String` | — | `str::to_string` |
| `StrEq` | `(&str, &str) -> Bool` | — | content equality; serves `==`/`!=` on `str` AND `String` (via `StringAsStr`) |
| `StrCmp` | `(&str, &str) -> Int64` | — | −1/0/+1; lowering derives all ordered operators; **no `Ordering` runtime value introduced** |
| `PrintlnStr` / `PrintStr` | `(&str) -> Unit` | — | joins the §7 print family |
| `PrintlnChar` / `PrintChar` | `(Char) -> Unit` | — | completes the print family for `Char` |
| `VecNew` | `() -> Vec<T>` | — | `T` from **destination** (§6) |
| `VecWithCapacity` | `(UInt64) -> Vec<T>` | — | `T` from destination |
| `VecPush` | `(&mut Vec<T>, T) -> Unit` | — | |
| `VecPop` | `(&mut Vec<T>) -> Option<T>` | — | |
| `VecLen` | `(&Vec<T>) -> UInt64` | — | |
| `VecIsEmpty` | `(&Vec<T>) -> Bool` | — | |
| `VecIndexGet` | `(&Vec<T>, UInt64) -> T` | IndexOutOfBounds | **requires `T: Copy`** (V-COPY-1, §6, owner point 5). `v[i]` copying read / auto-borrow receiver-read step. Runtime-checked (mutable length; the §6 proof discipline deliberately excludes `Vec` in v0.1) |
| `VecReplace` | `(&mut Vec<T>, UInt64, T) -> T` | IndexOutOfBounds | **replaces `VecIndexSet`** (owner point 2). `v[i] = x` lowers to `old = VecReplace(&mut v, i, x); «drop old if T droppable»`, giving the caller the replaced value so install-then-destroy (CD-012 order) is representable for droppable `T` |
| `VecRemove` | `(&mut Vec<T>, UInt64) -> T` | IndexOutOfBounds | returns the removed element (caller owns/drops it) |
| `VecClear` | `(&mut Vec<T>) -> Unit` | — | **non-droppable `T` only** (V-COPY-1-adjacent, §6); buffer reset, no destructors. For droppable `T`, `clear()` lowers to an explicit pop-and-drop loop (§5a), not this op |
| ~~`VecIterNew`~~ | — | — | **MOVED to C4.5f / a future `0.1-A2` bump (CD-032).** See the §5e correction: STARK's `.iter()` is by-reference (`&T`), not the by-value `Option<T>` A1 originally specced here, so *all* Vec iteration is the reserved interior-reference form. Not part of surface `0.1-A1`. |
| ~~`VecIterNext`~~ | — | — | **MOVED to C4.5f / `0.1-A2` (CD-032).** The by-value `Option<T>` form had no STARK source trigger (`for x in v` does not exist; `for x in v.iter()` binds `&T`). |

### 5d. Reserved — named, not lowered in C4.5e (owner point 8)

Each reserved op is activated later by the **same appendix mechanism**, but activation is **not
silent**: it requires a dated, reviewed enumeration entry that bumps `MIR_RUNTIME_SURFACE`
(`0.1-A1` → `0.1-A2` …) and records the added op(s), exactly as A1 records this set. Reserved:
`StringSubstring` (§5b), `StringBytes`, `StringIntoBytes`, `StringFind`, `StringReplace`,
`StringSplit`, `StringTrim`, `StringToLower`, `StringToUpper`, `StringStartsWith`,
`StringEndsWith`, `StringCharsIter`/`CharsIterNext`, `VecGetRef`/`VecGetMutRef` (interior
references into runtime values — need the C4.5f frame-generation work first), `VecInsert`,
`VecAppend`, `SliceFromVec` (`(&Vec<T>) -> &[T]`), `SliceLen` (`(&[T]) -> UInt64`), and
by-reference/by-value-move `Vec` iteration. Slice **indexing** needs no runtime op: `&[T]` behind
an unchanged reference already indexes through the existing `CheckIndex` proof discipline (base
§6), which A1 does not alter.

### 5e. `Vec` iteration — CORRECTED and MOVED to C4.5f (CD-032)

**Correction (CD-032, owner decision 2026-07-19, found in WP-C4.5e-2).** A1 rev. 1–3 specced a
**by-value** `VecIterNext : (&mut Core(VecIter,[T])) -> Option<T>` as "the `for x in v` desugar."
That form has **no STARK source trigger**: STARK has no by-value `for x in v`; the only iteration
form is `for x in v.iter()`, and `Vec::iter()` binds the loop variable as `&T`
(`03-Type-System`/stdlib: `iter(&self) -> VecIter<T>`, yielding `&T`). So **all** Vec iteration in
STARK is **by-reference**, which is precisely an interior reference into a runtime container — the
work A1 §5d already reserved and tied to the C4.5f frame-generation hardening.

**Resolution:** Vec iteration (`VecIterNew`/`VecIterNext`, in a by-reference `Option<&T>` form) is
**removed from surface `0.1-A1`** and **folded into C4.5f**, activated by a future dated
`0.1-A2` surface bump alongside the interior-reference (`VecGetRef`) work. Surface `0.1-A1` as
implemented in C4.5e never contained iteration ops. The design notes below (interior-borrow
cursor, snapshot-for-Copy interpreter representation, non-Copy iterator, drop-elaborated cleanup)
**carry forward to C4.5f** as the starting point for the by-reference `Option<&T>` design; they
are retained here for reference, not as `0.1-A1` contract.

`VecIterNew : (&Vec<T>) -> Core(VecIter, [T])` is an **interior-borrow** operation: the iterator
borrows the source `Vec` for the duration of the loop.
- **Borrow/lifetime trust boundary:** borrowck guarantees the `Vec` outlives the iterator and is
  not mutated (no overlapping `&mut`) while it is live — the same boundary as `StringAsStr`
  (§5b), no new rule.
- **Backend representation:** a true borrowed cursor (a `&Vec` plus an index).
- **MIR-interpreter representation:** an **immutable snapshot** of the elements. This is sound
  because C4.5e restricts `VecIterNext` to `T: Copy` (V-COPY-1) — so snapshot copies are
  observationally identical to borrowed reads — and borrowck forbids mutation of the `Vec` while
  the iterator lives. As with `StringAsStr`, this keeps A1 free of interior-reference /
  frame-generation dependencies.
- **Identity / mutability:** the iterator value is **non-Copy** (it carries advancing cursor
  state); it moves under the ordinary verifier discipline.
- **Cleanup:** the iterator is a runtime value and, like `String`/`Vec`, **always requires
  runtime drop glue** (§5a): the `for`-loop desugar drop-elaborates it, and its `Drop` terminator
  releases the borrow/cursor. The iterator owns no elements (it borrows, never owns — see rev. 7,
  which replaced the original `T: Copy` snapshot with a borrowed cursor), so its glue runs **no**
  element destructors — cursor release only, unobservable.

## 6. Verifier rules

- **Surface-version gate (owner point 3):** before verifying any body, `verify_program` checks
  `program.mir_version` and `program.runtime_surface` against the versions this build supports; an
  unsupported surface is rejected (a program-level error, distinct from the per-body MIR-xxxx
  codes) and **no body is consumed**. Backends perform the same check before codegen. The
  `MIR_VERSION`/`MIR_RUNTIME_SURFACE` constants are what a producer stamps; the per-program fields
  are what every consumer authoritatively reads.
- **Signature table:** every §5c op gets a `runtime_sig` entry (existing V-RT-1 / MIR-0012).
- **`VecClear` element-type restriction (owner point 1):** `VecClear`'s resolved `T` must be
  **non-droppable** (MIR-0016); a droppable `T` at `VecClear` is invalid MIR, because `clear()` on
  droppable `T` must lower to the explicit pop-and-drop loop (§5a), never this op — this keeps
  every user element destructor at a visible `Drop` terminator.
- **Schematic `T` resolution (owner point 4), MIR-0012 on any inconsistency:**
  - *Vec constructors* (`VecNew`, `VecWithCapacity`): `T` = the destination place's element type
    (`MirTy::Core(Vec,[T])`); there is no Vec operand to read.
  - *Vec methods* (`VecPush`/`VecPop`/`VecLen`/`VecIsEmpty`/`VecIndexGet`/`VecReplace`/
    `VecRemove`/`VecClear`): `T` = the element type of the **first operand**, which
    must be `&Vec`/`&mut Vec`; every other operand and the destination is checked against it.
  - *Iterator ops* (`VecIterNew`/`VecIterNext`): **C4.5f carry-forward (CD-032), not
    `0.1-A1`.** When iteration lands: constructor `T` from the first `&Vec` operand, `VecIterNext`
    `T` from the `Core(VecIter,[T])` operand.
- **V-COPY-1 (new, code MIR-0016, owner point 5):** `VecIndexGet` (and, when iteration lands in
  C4.5f, `VecIterNext`) require their resolved element `T` to be `Copy`. Verifier-enforced via a new additive in-memory
  `TypeContext::copy_types` set (which nominal instances carry an `impl Copy`; populated during
  lowering exactly like `drop_impls` under CD-029). `is_copy` over `MirTy` reads it; a non-Copy
  `T` at these ops is MIR-0016. (Primitives/refs/all-Copy aggregates resolve without the table.)
- **V-STR-1 (new, code MIR-0015):** `Constant::Str` content must be valid UTF-8 (by construction
  in-memory; checked at any dump-ingestion boundary).
- **V-STR-2 (new, code MIR-0015):** `String`, and `Str` behind any reference depth, never appear
  as `BinOp`/`UnOp` operands — string equality/ordering route through `StrEq`/`StrCmp` only.
- **Typing:** `Constant::Str` types as `Ref{ mutable:false, inner:Str }` in `operand_ty`; a `Str`
  where `String` is expected is the existing MIR-0004 mismatch. `Trap.message`, when present,
  must type as `&str` (MIR-0015 otherwise).
- **`TypeContext` addition — exactly ONE new field (owner point 4):** A1 adds
  `copy_types` (additive, in-memory, not dump-serialized — CD-029 class), the set of nominal
  instances carrying an `impl Copy`, populated during lowering like the pre-existing `drop_impls`.
  `drop_impls` is **not** an A1 addition (it is C4.5d's) and is unchanged; the runtime-glue drop of
  `String`/`Vec`/`VecIter` (§5a) is recognized **structurally** by the drop machinery
  (`ty_needs_drop` on `MirTy::String`/`MirTy::Core(Vec|VecIter,…)`), **not** via any table entry.
  (Rev. 2 miscounted this as "two new fields"; the correct count is one — `copy_types`. The two
  other new shape positions A1 introduces are `MirProgram` fields, not `TypeContext` fields.)
- Existing rules untouched; `MirTy::Str` stays legal only behind `Ref` (V-TY-3); `String` stays
  a sized first-class type.

### 6a. `Trap.message` participates in all analyses (owner point 7)

The new `Trap.message: Option<Operand>` is an **ordinary operand everywhere**, not merely
type-checked. Every terminator-operand walk includes it:
- **Move dataflow (V-MOVE-1):** a `Move` message operand consumes; a possibly-moved message
  operand is MIR-0007. (In practice the message is `&str`/`Copy`, but the analysis treats it
  uniformly.)
- **Reference / place analysis:** the operand's place is typed and V-REF-1-checked like any read.
- **Proof-token scan (V-IDX-2):** the message operand is scanned; a stray `IndexProof` there is
  MIR-0010.
- **Drop-flag discipline (V-DROP-2):** a `DropFlag` local used as the message operand is
  MIR-0009.
- **`SourceInfo`/provenance:** the message evaluates before the abort; the trap outcome carries
  category + provenance (CD-029) **and** the resolved message string, compared by the C4.5e-0
  differential.

## 7. Textual dump syntax

- Header gains the surface identifier: `// STARK MIR v0.1 (runtime-surface 0.1-A1)`. **Every
  post-A1 dump emits the surface identifier** (owner point 3) — it renders from the program's
  `runtime_surface` field, not conditionally on whether A1 constructs happen to appear, so the
  header is always authoritative about which surface produced the dump. Body lines with no A1
  construct still render identically to pre-A1 below the header.
- `Constant::Str` renders `const "…"` with Rust `escape_default` content (deterministic,
  injective, line-safe); the `&str` type is implied. Example: `_3 = const "hi\n"`.
- A `Trap` message renders appended: `trap Panic msg(copy _2)`; message-less traps render exactly
  as today (`trap Panic`).
- New `RuntimeFn`s render through the existing `runtime:{name}` form; no grammar change.

## 8. Why this stays MIR v0.1

Every change is additive and reachable only from programs that previously failed lowering with
clean `Unsupported` (string literals, String/Vec ops, `panic` with a message):

1. No existing construct changes meaning, typing, or dump rendering; a pre-A1 body renders and
   executes identically post-A1. The one textual delta is the header's surface line, which every
   post-A1 dump now carries (§7).
2. `RuntimeFn` growth is the §7-anticipated extension mechanism (closed, enumerated, versioned via
   `MIR_RUNTIME_SURFACE`; unknown variants fail loudly — V-RT-1).
3. `Constant::Str`, `Trap.message`, the one new `TypeContext` field (`copy_types`), and the two
   new `MirProgram` fields (`mir_version`, `runtime_surface`) are new positions, not
   reinterpretations; pre-A1 consumers reject them loudly (non-exhaustive match / missing-field)
   rather than misreading them, and the surface-version gate (§6) makes a version mismatch an
   explicit rejection rather than a silent misparse.
4. Precedent: CD-029 amended the compilation-unit shape (TypeContext) additively within v0.1
   under exactly this rule; the `MirProgram` field additions are the same class.

The `MIR_VERSION` bump trigger remains real *shape reinterpretation* (changing `Checked`,
altering Option/Result representation, introducing a literal-pool section). Runtime-surface growth
bumps `MIR_RUNTIME_SURFACE` only.

## 9. Test plan

**Positive (lowering + dump):** string literal → `const "…"` with `&str` type; header shows
`runtime-surface 0.1-A1`; `String::from`/`as_str`/`push_str`/`len` chain → table ops;
deterministic dump; escape-heavy literal (`"a\"b\\c\n\u{1F600}"`) round-trips dump escaping;
`Vec` of a droppable struct emits reverse-index-order element `Drop` glue.

**Negative (verifier):** `Str` assigned to a `String` local (MIR-0004); `BinOp::Eq` on two
`String` locals (MIR-0015/V-STR-2); `VecPush` element operand disagreeing with the vector element
(MIR-0012 schematic); `VecPush` first operand not `&mut Vec` (MIR-0012); `VecIndexGet`
on a non-Copy element type (MIR-0016/V-COPY-1 — **note: `VecIterNew`/`VecIterNext` no longer
carry this constraint since rev. 7 made Vec iteration a borrowed cursor**); `VecClear` on a
**droppable** element
type (MIR-0016, §6); `Trap.message` operand typed `Int32` (MIR-0015); a `DropFlag` local used as a
`Trap.message` operand (MIR-0009, exercising §6a); a program stamped with an unsupported
`runtime_surface` rejected before any body is verified (§6 surface-version gate).

**Differential (oracle vs MIR, full C4.5e-0 comparator — output, trap category, provenance,
pre-trap prefix, trap message):**
- the two frozen `ownership_drop__*` corpus cases (String-labelled Drop instrumentation) — the
  first String-dependent corpus cases to go differential-green;
- `collection_iter__01` (Vec push/index/iterate);
- string equality/ordering (`==`, `!=`, `<` on `str` and `String` via `StrCmp`);
- `panic("message")` after partial output — category `Panic`, matching message, matching pre-trap
  stdout;
- `v[i]` out-of-bounds → `IndexOutOfBounds` with the call site's provenance;
- **`Vec` of droppable elements dropped at scope exit** — element destructors fire in reverse
  index order, matching the oracle (validates §5a end to end);
- `v[i] = x` overwrite on droppable `T` — the replaced element's destructor runs
  (install-then-destroy via `VecReplace`), order matching the oracle;
- `v.clear()` on droppable `T` — every element destructor runs (via the pop-and-drop loop, §5a),
  order matching the oracle;
- `String` moves (into struct, into call); Drop struct with a `String` field — user dtor ordering
  unchanged, String contributes no observable output but its glue is present.

## 10. Explicitly out of A1

`Ordering` as a runtime value and user-nominal `Eq`/`Ord` impl dispatch (own short design note
when C4.5e reaches it); `HashMap`/`HashSet` ops (**`HashMap` since activated by the rev. 6
`0.1-A3` dated enumeration** — keys-only iteration, no `values()`/`remove()`; `HashSet` remains
reserved); `CharsIter`/`SplitIter` lowering; `StringSubstring`
and other interior views into runtime containers (§5d, after C4.5f frame generations); file/provider
I/O (C5.1 ABI); any literal-pool/dump-section mechanism.

## 11. Revision log

**Rev. 12 — surface `0.1-A8` activation (2026-07-20, WP-C4.7-8.6, owner-decided; per CD-032's
dated-enumeration rule).** Activates the **exclusive** slice view, completing the slicing surface
rev. 10 opened.

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `SliceNewMut` | `(&mut (Array<T,N> \| Vec<T> \| [T]), I, I, Bool) -> &mut [T]` | **IndexOutOfBounds** | same bounds validation as `SliceNew`; requires an EXCLUSIVE receiver borrow and yields `&mut [T]` |

Rev. 10 reserved this deliberately, and the owner decided (2026-07-20) it is required for C4
exit: **REF-SLICE-001** states that "writes through an exclusive slice reference update the
original object", 03-Type-System gives `&mut expr[r]` the type `&mut [T]` (§107) and lists
`&mut [T; N] -> &mut [T]` among the permitted coercions (§547). Deferring it would have exited
C4 with a gap in a rule the abstract machine states outright.

**Write-through.** A `Slice { start, len }` window followed by an `Index(i)` composes to the
absolute element `start + i` — the interpreter's WRITE path now performs the same composition its
READ path already did, which is precisely what makes a write through the view reach the base
object. A bare window with no following index is not a writable place (it denotes the sub-view as
a value) and is rejected loudly.

**Verifier.** `SliceNewMut` requires an exclusive receiver reference (MIR-0012 otherwise) — a
shared base cannot produce a writable view — and yields `&mut [T]`. `SliceLen`/`SliceIsEmpty`
accept a receiver of either mutability, since they only read.

**Receiver reads are copies, not moves.** `len`/`is_empty` read through the reference, so lowering
passes the receiver by `Copy` (the MIR-level equivalent of a shared reborrow). Moving it would
consume an exclusive view and make a second use of the same `&mut [T]` local fail V-MOVE-1, which
the language does not say: `s.len(); s[0]` is legal. The matching front-end defect is **DEV-082**.

`MIR_RUNTIME_SURFACE = "0.1-A8"`.

**Rev. 11 — surface `0.1-A7` activation (2026-07-20, WP-C4.7-6.1, owner-decided; per CD-032's
dated-enumeration rule).** Activates `Box<T>` construction and extraction. 06-Standard-Library
lists `Box<T>` in `core-min` and gives it exactly `new` and `into_inner`; both reached the front
end and the HIR oracle but had no MIR lowering at all.

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `BoxNew` | `(T) -> Box<T>` | — | moves `T` into a fresh allocation; consumes its argument exactly once |
| `BoxIntoInner` | `(Box<T>) -> T` | — | consumes the box and transfers the contained value out **without dropping it** (ownership moves to the caller), releasing the allocation |

`Box<T>` is `MirTy::Core(Box, [T])` — **no new `MirTy`**. It is an **opaque owning** runtime
type, deliberately NOT lowered transparently as `T`: a transparent box would make a recursive
type such as `struct Node { next: Option<Box<Node>> }` infinitely sized. Both ops are schematic
in `T` (`box_runtime_sig`), so a mismatched destination is an ordinary MIR-0005.

**No public box-drop operation.** Ordinary destruction goes through the existing `Drop`
terminator's structural glue: dropping a `Box<T>` drops the contained `T` **exactly once** and
then releases the allocation (unobservable). A box consumed by `into_inner` no longer holds the
value, so nothing is dropped twice. Interpreter representation: a one-element aggregate — Core v1
makes addresses unobservable (LAYOUT-QUERY-001), so the reference interpreter models only what a
program can observe, namely that the box *owns* its value. Allocation failure remains a
classified host/resource failure, not a language trap; the reference interpreter cannot fail to
allocate and so raises none.

**Core v1 has no `Deref` trait**, so `*box` is not a construct: `into_inner` is the only
extraction. The rejection of `*Box::new(5)` is specification-conformant and is pinned by a
negative front-end test. This corrects the WP-C4.6 gate audit, which listed "`Box` deref" as a
`core-min` hole — it is not; the real gap was the construction/extraction pair recorded here.

Two defects were found and fixed while implementing this, both pre-existing:
- **Drop-instance discovery never descended into `Core` container type arguments**, so a
  `Box<Tag>`'s `Drop` terminator fired and silently found no destructor registered. The walk now
  descends into every `Core` container's arguments.
- **The walk needed a cycle guard** once `Box` made types recursive: `Node -> Option<Box<Node>>
  -> Box<Node> -> Node` overflowed the stack. Guarded by visited-type set.
- **DEV-077** (oracle): `Box::into_inner` operated on a CLONE of the receiver, so the original
  box kept the value and dropped it again at end of scope — an observable double drop with a
  `Drop` payload, and a divergence from MIR. It now consumes the real place.

`MIR_RUNTIME_SURFACE = "0.1-A7"`. Evidence: `box_new_and_into_inner_agree`,
`box_drop_timing_agrees` (exact destructor interleaving, both engines), `box_recursive_type_agrees`
(a finite value of a recursive type) — differential; `rejects_box_into_inner_on_non_box`,
`rejects_box_new_with_mismatched_dest` — verifier; `box_deref_is_rejected` — front-end negative.

**Rev. 10 — surface `0.1-A6` activation (2026-07-20, WP-C4.6 A4 slicing, per CD-032's
dated-enumeration rule; completes the A4 core-min surface).** Activates shared slice views:

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `SliceNew` | `(&(Array<T,N> \| Vec<T> \| [T]), I, I, Bool) -> &[T]` | **IndexOutOfBounds** | `I` is the range's integer element type (both bounds match); the Bool is the inclusive flag. Traps on a negative, inverted, or out-of-range bound (06-Standard-Library behavioral requirement), with the call site's provenance. Re-slicing a `&[T]` **composes** windows — a slice path never stacks two views. |
| `SliceLen` | `(&[T]) -> UInt64` | — | view length |
| `SliceIsEmpty` | `(&[T]) -> Bool` | — | |

`&[T]` = `Ref { mutable: false, inner: Slice(T) }`. Interpreter representation: a slice value
is a `Ref` whose path ends in a `Slice { start, len }` window step over the live Array/Vec
referent (frame-generation guarded); a window followed by `Index(i)` composes to the absolute
element `start + i`, and `s[i]` uses the ordinary `CheckIndex` proof discipline **checked
against the VIEW length**. Slices are shared-only in this revision (`&mut base[range]` stays
reserved); no writes route through a view. The oracle's three slice-bound error messages were
aligned to the "out of bounds" family (the spec groups all bound failures as one trap; the
fragment comparator requires it). `MIR_RUNTIME_SURFACE = "0.1-A6"`. **This completes the A4 `core-min` MIR *runtime surface*** —
front-end `core-min` items then attributed to it (`Box` deref, primitive `cmp`) are separate and
were owned by WP-C4.7-6, which found `Box` deref to be **spec-conformant to reject** (no `Deref`
trait in Core v1) and added primitive `cmp` (6.2). The real Box gap — `new`/`into_inner` — is
rev. 11 above.
The message alignment described above is numbered **DEV-074** (closed at creation; see
`KNOWN-DEVIATIONS.md`).

**Rev. 9 — surface `0.1-A5` activation (2026-07-20, WP-C4.6 A4-2d, per CD-032's
dated-enumeration rule).** Activates string chars iteration:

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `CharsIterNew` | `(&str) -> Core(CharsIter, [])` | — | borrowed snapshot over the string's chars (`Char` is `Copy`, so the snapshot matches the oracle's borrowed `CharsIter`); a `String` receiver is converted to `&str` first |
| `CharsIterNext` | `(&mut Core(CharsIter,[])) -> Option<Char>` | — | yields each `Char` in source order, by value |

Interpreter representation: the iterator is `Aggregate([Str-snapshot, cursor])` in a frame
local; `Next` reads the cursor-th char and advances. `MIR_RUNTIME_SURFACE = "0.1-A5"`; the
surface gate (§6) and dump header (§7) carry it. `str::chars`/`String::chars` and
`for c in s.chars()` lower against these ops. Array/Vec slicing remains the last core-min
surface piece still to activate.

**Rev. 8 — surface `0.1-A4` activation (2026-07-20, WP-C4.6 A4-2b, per CD-032's
dated-enumeration rule; the A4 runtime-surface amendment CD-033 pre-authorized).** Activates
checked interior Vec access:

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `VecGetRef` | `(&Vec<T>, UInt64) -> Option<&T>` | — | **never traps** — returns `None` on out-of-bounds (distinct from the trapping `VecIndexGet`/`v[i]`); interior borrow into the live Vec; no `T: Copy` requirement (yields a reference, not a value) |
| `VecGetMutRef` | `(&mut Vec<T>, UInt64) -> Option<&mut T>` | — | as above, mutable interior borrow |

`T` resolves from the first `&Vec` operand (as for the other Vec methods). Interpreter
representation: index the live Vec through the receiver reference and hand out `Some(&v[i])`
(interior `&T`/`&mut T`, protected by the C4.5f-1 frame generations) or `None`.
`MIR_RUNTIME_SURFACE = "0.1-A4"`; the surface gate (§6) and dump header (§7) carry it. Programs
using neither op are unchanged. Remaining A4 core-min surface still to activate under this or a
following revision: `str::chars` iteration and array/Vec slicing.

**Rev. 7 — Vec iteration made a true borrowed cursor (2026-07-19, WP-C4.6 A6, CD-033).** No
surface change (stays `0.1-A3`), no new ops. `VecIterNew`/`VecIterNext` are re-specified from the
rev. 5 *snapshot* representation (an immutable copy of the Vec in a frame local, which forced
`T: Copy`) to a **true borrowed cursor** identical to the HashMap `KeysIter`: the iterator value
is `[vec-ref, cursor]`, and `VecIterNext` indexes the *live* Vec through the reference to hand
out an interior `&T`, protected by the C4.5f-1 frame generations. Consequence: the `T: Copy`
restriction (V-COPY-1/MIR-0016) on `VecIterNew`/`VecIterNext` is **dropped** — iterating a
`Vec<String>` (or any non-Copy element) now lowers. The signatures are unchanged
(`VecIterNew : (&Vec<T>) -> Core(VecIter,[T])`, `VecIterNext : (&mut Core(VecIter,[T])) ->
Option<&T>`); only the interpreter representation and the Copy gate change. The borrow checker
already forbids mutating the source Vec while the iterator is live, so the live-indexing read is
sound. `VecIndexGet` keeps its `T: Copy` requirement (it returns `T` by value, not `&T`).

**Rev. 6 — surface `0.1-A3` activation (2026-07-19, WP-C4.5f-3a/3b, per CD-032's
dated-enumeration rule).** Activates the `HashMap<K, V>` group and the Char printing/String-Char
ops:

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `HashMapNew` | `() -> Core(HashMap, [K, V])` | — | schematic `(K, V)` from destination; deterministic insertion order (CD-009) |
| `HashMapInsert` | `(&mut HashMap<K,V>, K, V) -> Option<V>` | — | returns the displaced old value — the honesty rule's visible form: the runtime never runs a user destructor, the caller drops the returned `Option<V>` at a visible `Drop` |
| `HashMapGet` | `(&HashMap<K,V>, &K) -> Option<&V>` | — | interior reference into the map local, frame-generation-guarded (C4.5f-1) |
| `HashMapLen` | `(&HashMap<K,V>) -> UInt64` | — | |
| `HashMapIsEmpty` | `(&HashMap<K,V>) -> Bool` | — | |
| `HashMapContainsKey` | `(&HashMap<K,V>, &K) -> Bool` | — | |
| `HashMapKeysIterNew` | `(&HashMap<K,V>) -> Core(KeysIter, [K])` | — | borrowed cursor over the map (NOT a snapshot — the map stays borrowed for the iteration) |
| `HashMapKeysIterNext` | `(&mut Core(KeysIter,[K])) -> Option<&K>` | — | yields interior `&K` in insertion order; schematic `K` resolved from the `Core(KeysIter,[K])` operand |
| `PrintlnChar` / `PrintChar` | `(Char) -> Unit` | — | `Char` is `Constant::Int` with `MirTy::Char` (Unicode scalar value) |
| `StringPushChar` | `(&mut String, Char) -> Unit` | — | |
| `StringPopChar` | `(&mut String) -> Option<Char>` | — | |

Restrictions: user-`Drop` `K`/`V` are excluded from the HashMap surface (lowering refuses;
element destruction paths through `HashMap` internals would otherwise hide destructors,
violating §5a's honesty rule). `values()`, `remove()`, and `HashSet` remain reserved (§5d/§10).
Schematic `(K, V)`: constructors from destination, methods from the first `&HashMap` operand,
`HashMapKeysIterNext` from the `Core(KeysIter,[K])` operand.
`MIR_RUNTIME_SURFACE = "0.1-A3"`; the surface gate (§6) and dump header (§7) carry it.
Programs using none of these ops are unchanged.

**Rev. 5 — surface `0.1-A2` activation (2026-07-19, WP-C4.5f-2, per CD-032's dated-enumeration
rule).** Activates by-reference Vec iteration, the §5e carry-forward design:

| RuntimeFn | Signature (MIR types) | Traps | Notes |
|---|---|---|---|
| `VecIterNew` | `(&Vec<T>) -> Core(VecIter, [T])` | — | interior-borrow (§5e); **requires `T: Copy`** (V-COPY-1/MIR-0016); iterator value is non-Copy, drop-elaborated (no-op glue — no droppable elements by the Copy bound) |
| `VecIterNext` | `(&mut Core(VecIter,[T])) -> Option<&T>` | — | **by-reference** (the CD-032 correction of rev. 1–3's by-value form); yields an interior `&T`; requires `T: Copy` |

Schematic `T`: `VecIterNew` from the first `&Vec` operand; `VecIterNext` from the
`Core(VecIter,[T])` operand (as §6 already stated for the carry-forward). Interpreter
representation exactly per §5e: the iterator is a snapshot aggregate `[Vec, cursor]` in a frame
local; `VecIterNext` hands out interior references into that local, protected by the C4.5f-1
frame-generation guard once the iterator dies (the guard landed first, deliberately).
`MIR_RUNTIME_SURFACE = "0.1-A2"`; the surface gate (§6) and dump header (§7) carry it. Programs
without iteration are unchanged. `StringSubstring`/`VecGetRef` and by-value/non-Copy iteration
remain reserved (§5d).

**Rev. 4 — CD-032 (owner decision 2026-07-19, post-C4.5e-2).** Vec iteration corrected and
moved out of surface `0.1-A1`. A1's by-value `VecIterNext -> Option<T>` had no STARK source
trigger (STARK iteration is `for x in v.iter()`, binding `&T`), so all Vec iteration is
by-reference — an interior reference into a runtime container. Iteration (a by-reference
`Option<&T>` `VecIterNew`/`VecIterNext`) is **folded into C4.5f** and activated by a future
`0.1-A2` surface bump alongside the interior-reference (`VecGetRef`) work; `0.1-A1` as
implemented never contained iteration ops. §5c iteration rows struck; §5e reframed as the
carry-forward design for C4.5f; the corrected surface remains `0.1-A1` (no bump — the removed
ops were never implemented). Does not touch strings (e-1) or the Vec data surface (e-2).

**Rev. 3 — CE3-APPROVED.** Applied the four final owner corrections to rev. 2:
1. `VecClear` no longer hides element destructors: droppable `T` lowers `clear()` to an explicit
   pop-and-drop loop with visible `Drop` terminators; the `VecClear` `RuntimeFn` is restricted to
   non-droppable `T` and verifier-enforced (§5a, §5c, §6). The governing principle is stated
   generally — no `RuntimeFn` ever runs a user element destructor; those run only at `Drop`
   terminators.
2. `VecIterNew` specified as an interior-borrow op with stated borrow/lifetime, backend (borrowed
   cursor), and interpreter (immutable snapshot, sound because iteration is `T: Copy`)
   representations; the iterator value is non-Copy and always drop-elaborated with unobservable
   no-element cleanup (§5e).
3. `mir_version` and `runtime_surface` stamped on every `MirProgram`; the verifier and backends
   reject an unsupported surface before consuming any body (surface-version gate, §6); every
   post-A1 dump emits the surface identifier from the field, unconditionally (§7).
4. Clarified that `String`/`Vec<T>` **always** require runtime drop glue while element-destructor
   execution is conditional on `T` (§5a); the generated-Rust backend doing explicit reverse-order
   element destruction must suppress Rust's automatic element drop to avoid double destruction
   (§4); corrected the new-`TypeContext`-field count from "two" to **one** (`copy_types`; the two
   other new shape positions are `MirProgram` fields) (§6).

**Rev. 2** — applied the eight owner-required corrections to rev. 1:
1. `String`/`Vec` now participate in Drop elaboration through runtime glue (§5a); `Vec<T>`
   element drop is explicit **reverse index order**, matched to the frozen oracle; cleanup is
   never delegated implicitly to the backend.
2. `VecIndexSet(...)->Unit` replaced by `VecReplace(...)->T` so install-then-destroy is
   representable; `VecClear`/`VecReplace`/`VecRemove` route element destruction through §5a glue
   (§5c).
3. `StringAsStr`/`StringSubstring` treated explicitly as interior references with stated
   borrow/lifetime, backend, and interpreter representations (§5b); `StringAsStr` retained as a
   read-only snapshot in the interpreter; `StringSubstring` deferred (§5d).
4. Schematic `T` resolution defined per op class — constructors from destination, methods from
   the first Vec operand, iterators from `Core(VecIter,[T])` (§6).
5. Copy restriction on `VecIndexGet`/`VecIterNext` made normative and verifier-enforced
   (V-COPY-1 / MIR-0016, `TypeContext::copy_types`) (§6).
6. Machine-visible `MIR_RUNTIME_SURFACE = "0.1-A1"` added, in the dump header and checkable
   (§ preamble, §7).
7. `Trap.message` specified to participate in move, reference, proof-token, drop-flag, and
   provenance analyses, not only type checking (§6a).
8. Reserved-op activation required to use the same appendix mechanism **and** a dated reviewed
   enumeration that bumps `MIR_RUNTIME_SURFACE` (§5d).

**Rev. 1** — initial draft; central direction approved in principle.
