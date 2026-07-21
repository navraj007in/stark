# WP-C5.3 — Aggregates, Enums, and Error Values

Gate: C5 (Native Core Backend MVP). Scope from `COMPILER-ROADMAP.md` WP-C5.3, detailed by
`WP-C5-ENTRY.md` §14 (C5.3a-d). Opened 2026-07-21 by owner directive, after DEV-095 (build-key
completeness) discharged its blocking entry condition under CD-055.

**Entry condition satisfied before opening.** CD-053 part 4 made DEV-095 a hard entry condition:
no aggregate or Drop-bearing native generation until the build key covers every semantic input
affecting generated code. Discharged by CD-055 — the key now covers the nominal type context and
the Drop map, with mutation-verified cache-invalidation tests. That mattered immediately: this
package is the first to put `struct_fields` and `copy_types` entries into generated code, which
under the old `hash(dump())` key would have been invisible to build identity.

## C5.3a — Tuples, arrays, and structs

### Status: CLOSED 2026-07-21 (CD-056).

Delivered per `WP-C5-ENTRY.md` §14 ("construction, field projection, copying/moving, and layout
queries"):

- **§6.2 type mapping extended** — `emit_types::emit_ty` now covers `Tuple`, `Array<T, N>`, and
  `Struct(item, args)`. §6.2 offers "generated concrete tuple **or** named internal aggregate;
  choose one canonical form" for tuples: a **Rust tuple** is chosen. It needs no generated
  definition, no name to keep deterministic, and no reachability walk to decide which shapes to
  emit. The cost is paid in `emit_places` (below).
- **Nominal definitions (§6.3)** — `emit_types::emit_nominal_definitions` emits exactly one Rust
  `struct` per entry in the type context's `struct_fields`, in `BTreeMap` order, with positional
  field names `f0..fn`. Positional because MIR carries fields by index and has already discarded
  the source names; inventing them back would be the backend semantic reconstruction §5.2
  forbids.
- **`mangle::type_name_for_nominal`** — reuses `sanitize_symbol`'s injective encoding over a
  `ty#<item>@[<args>]` key. Type names cannot collide with function names because `#` is not a
  legal STARK identifier character, so no `key_symbol` output can encode to the `ty#` prefix.
  That is an argument about the source language, recorded in the doc comment so it is re-checked
  if identifiers ever admit `#`.
- **A type environment for projections** — `emit_places::TyEnv`. MIR's `Projection::Field(i)` is
  ONE variant covering both struct fields and tuple elements, but generated Rust needs `.f0` for
  the first and `.0` for the second. Choosing requires the type of the place being projected,
  which means walking the projection chain from the local's declared type through the nominal
  type context. `TyEnv::place_ty` is that walk, and it is also what let `operand_mir_ty` stop
  refusing projected operands (a `SwitchInt` on a struct field or an array element now works).
- **Aggregate construction** — `Rvalue::Aggregate` for `Struct`/`Tuple`/`Array`. The struct case
  reads its type ARGUMENTS from the destination place: `AggKind::Struct(item)` names the item but
  not the instance, and only the destination knows the arguments. Not inference — verified MIR
  guarantees the assignment is well-typed, so the destination's type *is* the aggregate's type.
- **Indexing** — `Projection::ConstIndex(n)` → `[n]` (statically verifier-checked, no bounds
  check of its own); `CheckedOp::CheckIndex` → an `i128`-width range check that defines the proof
  value, and `Projection::Index(proof)` → `[proof as usize]`. Compared at `i128` for the same
  reason the arithmetic ops are: the index operand may be any signed width, and a negative index
  must trap rather than wrap into a huge `usize`.
- **`LocalKind::IndexProof` admitted** — an ordinary integer local in generated Rust. Its opacity
  is a MIR-level property the verifier enforces, not something the backend re-creates.

### Evidence

Seven new three-engine cases in `three_engine_differential.rs` (struct construction/projection
including two-level nesting and field order; tuple construction/projection including nested
tuples and a Copy tuple read after rebinding; array construction, constant indexing, dynamic
indexing in a loop, computed indices, `Bool` and nested-array element types; both ends of the
index range trapping; and the last-valid-index negative control that keeps a bounds check which
rejected everything from passing). Four native-only cases in `native_c5_3a_aggregates.rs` for
what a three-engine comparator structurally cannot cover: generated-source shape, and programs
one engine must reject.

### Two findings the harness produced immediately

**DEV-097 — the HIR oracle blamed two different columns for two ends of one bounds check.
FIXED.** An out-of-range index trapped at `node.span` (the whole index expression) while a
negative index trapped at the index OPERAND's span, so the oracle disagreed with both other
engines on one of the two. Neither the frozen corpus nor any inline case had ever indexed with a
negative value. Fixed in `interp.rs` by using `node.span` for both, which is also what MIR and
the native backend report.

**The layout-query exit condition cannot be satisfied as literally stated — owner decision
required (CD-056).** §14's C5.3 exit lists "target layout queries" among the dimensions requiring
three-engine agreement. Both interpreters answer **8 for every type** (`mir::interp::
reference_layout`, whose own doc comment says a real per-type algorithm is the backend's job),
while the native engine answers its **actual Rust target layout** (`size_of::<Int32>()` is 4).
`assert_eq(size_of::<Int32>(), 4)` therefore traps in both interpreters and succeeds natively.
This is not a backend defect — LAYOUT-ABI-001 makes layout target-dependent *by design* — but it
means "three-engine agreement on layout queries" needs a definition. Until the owner supplies
one, the harness asserts only that layout queries RUN in all three engines and agree on
completion-vs-trap, plus relations that hold under both answers (a query is deterministic within
an engine; alignment does not exceed size for these primitives). The value question is open and
recorded, not quietly dropped.

### The scope boundary, made explicit

A **non-`Copy` value moved out of a local initialised in an earlier block** is refused as a
backend `Unsupported`, naming WP-C5.3d — not emitted and left for rustc to reject.

The backend lowers MIR's block graph to `loop { match __bb { .. } }`, so every block is one
iteration of one Rust loop, and Rust's borrow checker is conservative across iterations: it
cannot see that MIR's control flow never revisits a moved-from local, and reports "value moved
here, in previous iteration of loop" even though verified MIR proves exactly that. Moving within
a single block is fine — which is why ordinary aggregate construction (`_2 = aggregate ..;
_1 = move _2;`) works today and is covered by its own test.

This was found the way it should be: a three-engine case that passed a struct by value produced a
`BuildFailed` carrying a rustc borrow-check error. A scope limit surfacing as a rustc error is a
defect in the diagnostic, so the guard now names the limit and the package that lifts it. The
real fix is §7.2's controlled storage for non-`Copy` locals, which is WP-C5.3d's deliverable and
needs the storage decision recorded in CD-056.

### One flagged reading, CE4-shaped (CD-056)

§6.3 says "do not derive `Clone`, `Copy`, `Eq`, `Ord`, or `Hash` as a shortcut for STARK
semantics". §7.4 says a MIR copy is emitted only for a MIR-`Copy` type and the backend must not
broaden that set. A STARK struct with an `impl Copy` needs *some* mechanism for `Operand::Copy`
to read it twice.

The reading taken: deriving `Clone`/`Copy` on exactly the instances MIR classifies `Copy` is not
a shortcut — MIR decides, the derive follows, and the set is neither broadened nor narrowed. No
other trait is derived; `Eq`/`Ord`/`Hash` stay forbidden because those *would* substitute Rust
behaviour for STARK's. `emit_types::mir_ty_is_copy` mirrors `mir::lower::is_copy` exactly rather
than asking Rust anything.

If the owner reads §6.3 as forbidding this outright, the alternative is a generated copy helper
per nominal, and the change is confined to `emit_types::derives_for` plus one test.

## C5.3b — Enums and discriminants

### Status: CLOSED 2026-07-21 (CD-057).

Delivered per `WP-C5-ENTRY.md` §14 ("user enums, payload access, discriminant switching, and
verifier-assumed variant correctness"):

- **User enums map to generated Rust enums** with uniformly **tuple** variants (`V0()`,
  `V1(i32)`, `V2(i32, i32)`), positional like struct fields and for the same reason. Empty
  payloads stay tuple variants: `V0()` is legal Rust, and the uniformity removes a special case
  from construction, from patterns (`V0(..)` matches it), and from the discriminant match — a
  unit variant would need different syntax in all three.
- **`AggKind::EnumVariant`** construction, taking its type ARGUMENTS from the destination for the
  same reason struct aggregates do: the kind names the enum and the variant, not the instance.
- **`Projection::VariantField`** — the one projection that cannot be a Rust suffix.

### The structural problem, and the shape it forces

Rust has no way to project into an enum variant's field outside a `match`. Every other MIR
projection appends to a place expression (`.f0`, `[2]`); this one has to *wrap* what came before:

```rust
(match &_5 { Ty::V1(__payload) => *__payload,
             _ => unreachable!("V-DISC-1: variant-field projection without a discriminant test") })
```

Two consequences, both deliberate:

- **The `_` arm is provably dead**, because V-DISC-1 makes a variant-field projection legal only
  after a discriminant test. It gets the same `unreachable!()` treatment the verifier-proved
  dead-block path already has, naming the rule — not a fabricated value that would paper over a
  lowering bug.
- **The result is an expression, not a place**, so it cannot be spliced as an assignment
  destination. `emit_dest_place` refuses it explicitly. This is a guard rather than a
  limitation: MIR lowering emits `VariantField` only through `read_place` and pattern tests, and
  STARK source has no syntax for assigning into a payload, so no path reaches it.

`Rvalue::Discriminant` takes the same shape for the same reason — an enum with payloads has no
integer `as` conversion. Every variant is listed with **no catch-all arm**, so adding a variant
cannot silently fall through to a wrong index. Its arms are typed by the destination local, not
by a fixed width; a hardcoded `i128` made the generated crate fail to compile against MIR's
`Int64` discriminant local, which the first native probe caught.

### Evidence

Four new three-engine cases (all three payload arities constructed and matched; payload field
ORDER, using a non-commutative operation so a wrongly-bound two-field payload cannot pass;
discriminant selection across four variants driven by a loop, with distinct per-variant values so
any mis-selected arm changes the sum; and a trap raised from a value that came out of a payload).
Three new native-only cases for the generated shape: one definition per instance with uniform
tuple variants, a discriminant match naming every variant, and the `unreachable!()` arm citing
V-DISC-1.

### The limitation, stated precisely (CD-058)

**C5.3b supports Copy payload reads. Non-Copy payload movement remains blocked on the
controlled-storage foundation and is not claimed complete merely by the current `VariantField`
expression.**

### What C5.3b makes newly urgent

The cross-block non-`Copy` move boundary from C5.3a **bites far harder for enums than for
structs**. Conditionally constructing a value and then matching it — the ordinary way enums are
used — puts construction in one basic block and the match in another, which is exactly the shape
the block-dispatch loop cannot express for a non-`Copy` value. The discriminant-selection test
above needs `impl Copy for Colour {}` to cross that boundary at all.

This is the strongest argument yet for settling **CD-056 decision 3 (the non-`Copy` storage
strategy)** before C5.3c: `Option`/`Result` payloads are frequently non-`Copy`, and `?` is
inherently cross-block.

## C5.3d-0 — Non-Copy storage and movement foundation

### Status: IN PROGRESS (opened 2026-07-21 under CD-058).

A **bounded prerequisite inserted before C5.3c** by owner directive. Its purpose is to unblock
C5.3c; **it does not close C5.3d**, whose observable destruction fixture and final
exactly-once/order/no-Drop-after-trap proof become C5.3d-1.

### The approved representation (CD-058, decision 3)

```text
ValueSlot<T> {
    storage: MaybeUninit<ManuallyDrop<T>>,
    whole-place live state,
    typed drop-unit live state where MIR distinguishes sub-places
}
```

Ordinary `Option<T>` is rejected: it introduces Rust-owned destruction, which §7.1 forbids.
`Option<ManuallyDrop<T>>` is rejected as the *general* representation for a subtler reason worth
keeping visible — it is adequate only for whole-value liveness, and **once a field or
constant-index element has been moved, the remaining bytes no longer necessarily form a valid
complete `T`**. Only `MaybeUninit` may legally hold that partially moved state. An Option-shaped
slot may later be admitted as an optimisation for locals MIR dataflow proves have no partial-move
paths.

### Progress

**Increment 1 (2026-07-21): the helper module.** `stark-runtime/src/slot.rs` — `ValueSlot<T>`
over `MaybeUninit<ManuallyDrop<T>>` with whole-place liveness, and six operations: `dead`
(initialisation), `get`/`get_mut` (shared and mutable place access), `take` (whole-value move),
`move_sub` (typed sub-place move), `write` (destination write), `drop_value` (explicit drop-unit
destruction). Deliverable 1 complete; deliverables 3 and 4 implemented at the module level;
deliverable 7 verified for the two mutations reachable without generated code.

Design points worth keeping visible:

- **Every operation is a SAFE function.** Deliverable 2 requires no ad hoc `unsafe` in emitted
  bodies, which means the helper API cannot be `unsafe fn` — otherwise generated code would need
  `unsafe {}` blocks of its own. Preconditions are guaranteed by verified MIR and *checked* here,
  not assumed.
- **`ValueSlot` implements no `Drop`.** A slot generated code never empties leaks, and that is
  the intended trade: leaking is what MIR verification excludes, whereas a Rust destructor here
  would silently cover for a lowering bug and make exactly-once unfalsifiable.
- **Liveness transitions happen BEFORE the operation they guard.** `take` and `drop_value` mark
  the slot dead first, so no path can observe a slot that is simultaneously live and moved-from —
  and if drop glue itself traps, the abort path sees a dead slot and cannot re-enter the value.
  That ordering is what makes exactly-once hold even when a destructor fails.
- **`move_sub` leaves the slot LIVE**, because the other drop units still are. Collapsing that
  into whole-local liveness is exactly what §7.6 forbids. This is also the case that rules out
  `Option<ManuallyDrop<T>>`: after a sub-place move the storage no longer holds a valid complete
  `T`.
- **Violations exit 102, not 101.** A STARK trap is a defined language outcome a correct program
  can reach; a slot violation means the backend emitted code contradicting verified MIR.
  Conflating the exit codes would let a compiler defect masquerade as the program's own trap.

Mutation-verified (deliverable 7, the two mutations reachable without generated code): omitting
the dead transition after a move exits **102** (the invariant check fires when the slot is later
rewritten); adding a Rust `Drop` impl fails the observable-destruction fixture with
`Rust scope exit must NOT destroy a slot's contents; got ["leaked"]`. Both reverted.

**Increment 2 (2026-07-21): soundness correction + backend integration.**

An owner review found the first `ValueSlot` **unsound for partial moves**, and it was right.
`move_sub` took `&mut T`, moved a field out, and left the slot "live" — after which `get`,
`get_mut`, `take` and `drop_value` all remained callable over storage that no longer held a valid
`T`. The module's own test asserted `slot.get().1` after moving `.0`, so **the bug was written
into its evidence**.

Corrected to a three-state machine — `Dead` / `Whole` / `Partial`:

- `get`, `get_mut`, `take`, `drop_value` and `drop_with` require `Whole`;
- the first partial move takes `Whole` → `Partial`;
- partial access is **raw-pointer only** (`move_field`, `copy_field`, `drop_field_with`), never
  constructing a `T`, `&T` or `&mut T`;
- `finish_partial` is the explicit `Partial` → `Dead` transition, deliberately not automatic,
  because only generated code following MIR's drop flags knows every unit is accounted for.

Projections are per-type, per-field `fn(*mut T) -> *mut F` helpers the backend generates, using
`addr_of_mut!` — which computes a field address **without dereferencing**, and is therefore
defined even over partially moved storage. That is what keeps `unsafe` out of emitted MIR bodies
while still allowing partial access.

**Miri evidence.** All 16 slot tests pass under Miri with zero UB
(`MIRIFLAGS=-Zmiri-ignore-leaks cargo +nightly miri test slot::` — leaks are ignored deliberately,
since an unemptied slot leaking is the module's documented contract). Mutation-verified:
restoring the old permissive `drop_value` guard makes Miri report **`Undefined Behavior:
constructing invalid value of type &mut [u8]: encountered a dangling reference (use-after-free)`**
— a real double-destruction, not a theoretical one.

One honest qualification: Miri did **not** independently flag `move_field` → `get` for a
`(String, i32)`. A moved-out `String`'s bytes stay bit-valid, so no validity check fires. That
sequence is still refused by the state machine and still violates the abstract machine's rules —
but the evidence for that specific case is the state machine, not Miri.

Backend integration landed on the corrected API:

- non-`Copy` locals are `ValueSlot<T>` starting **dead**, so generated code no longer fabricates a
  default value it immediately overwrites;
- reads go through `get()`, whole-local moves through `take()`, assignments through `write()`;
- non-`Copy` **parameters** arrive as `__pN` and are moved into their slot in the prologue —
  binding them as `_N` would collide with the slot the body declares;
- `Terminator::Drop` lowers to `drop_with`, running the destructor `TypeContext::drop_impls`
  names, then fields in **reverse declaration order** — mirroring `mir::interp::drop_in_place`,
  which is the semantic authority;
- **the C5.3a cross-block move guard is gone**: what it refused now compiles and runs;
- sub-place moves are refused *before rustc* pending the generated projection helpers, because
  routing them through the whole-value path would be silently unsound rather than merely
  unsupported.

Movement shapes now working (CD-058 deliverable 5): whole-local cross-block move (1);
conditional construction then discriminant read on a non-`Copy` enum (2); non-`Copy` direct-call
argument and return (4); whole-value reassignment after an explicit move (5). Shape 3 works for
`Copy` payloads; non-`Copy` payload extraction awaits the projection helpers.

**Remaining for C5.3d-0**: the generated per-type projection helper module (unblocking sub-place
moves and shape 3 for non-`Copy` payloads), partial-move discipline across the frozen fixtures,
and the three mutation cases that need generated code. Superseded text below:

**Previously remaining**: backend integration — emitting slots for non-`Copy` locals
(deliverable 2 in generated code), the five initial movement shapes (deliverable 5), partial-move
discipline with a backend diagnostic before rustc (deliverable 6), and the three mutation cases
that need generated code (whole-local collapse of a field/index drop unit; Drop after a trap;
second take through emitted paths).

### Required deliverables

1. **A small fixed helper module** covering: slot initialisation; shared and mutable place
   access; whole-value take; typed sub-place move; destination write; explicit drop-unit
   destruction.
2. **No ad hoc unsafe blocks in emitted MIR bodies.** Every unsafe operation routes through those
   reviewed helpers — which means the helper API must be *safe to call*, with preconditions
   guaranteed by verified MIR and checked in generated debug builds.
3. **Move semantics**: source must be live; bytes/value transfer without running source Drop; the
   exact source drop unit becomes dead; destination becomes live; moving an already-dead unit is
   rejected by generated debug checks or made unreachable through verified invariants.
4. **Drop semantics**: operate only on a live unit; mark the unit dead **before** invoking user or
   structural Drop glue; execute exactly once; preserve MIR field and constant-index order; never
   run after an aborting trap; never rely on Rust scope exit.
5. **Initial supported movement shapes**: cross-block whole-local move; conditional construction
   followed by discriminant read; consuming match of a non-`Copy` single-payload enum; non-`Copy`
   direct-call argument and return; whole-value reassignment after the previous value was
   explicitly moved or dropped.
6. **Partial moves**: implement the typed paths the frozen C5 fixtures require; reject other
   partial-move forms with a STARK backend diagnostic **before rustc**; never silently collapse
   field, variant-field or constant-index liveness into whole-local state.
7. **Mutation-tested evidence** — each of these must FAIL: omitting the dead transition after a
   move; allowing a second take; allowing an automatic Rust `Drop`; changing a field/index drop
   unit to whole-local; running Drop after a trap.

## C5.3c — Option, Result, matches, and `?`

Not started, and **not next**: CD-058 inserts C5.3d-0 first. `EnumRef::CoreOption`/`CoreResult`/
`CoreOrdering` are deliberately excluded from C5.3b rather than half-supported — they arrive with
match/`?` lowering, on the slot abstraction, since `Option`/`Result` payloads are frequently
non-`Copy` and `?` is inherently cross-block.

## C5.3d-1 — Bounded Drop proof

Not started. Follows C5.3c. Contains the dedicated observable destruction fixture and the final
exactly-once, ordering, and no-Drop-after-trap proof. The storage decision it was previously
blocked on is resolved (CD-058) and its foundation is C5.3d-0's deliverable.

## C5.3 exit

Not reached. §14 requires three-engine agreement for aggregate values (C5.3a: done), payload
variants (C5.3b: done), match paths (C5.3b: done for user enums), Option/Result, `?`, target
layout queries (**defined by CD-058: exact values under one injectable target-layout manifest;
the current relations-only case is a placeholder, not evidence**), and the dedicated C5 Drop fixture.
