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

Not started.

## C5.3c — Option, Result, matches, and `?`

Not started.

## C5.3d — Bounded Drop proof

Not started. Blocked on the non-`Copy` storage decision (CD-056), which the C5.3a scope boundary
above already makes visible.

## C5.3 exit

Not reached. §14 requires three-engine agreement for aggregate values (C5.3a: done), payload
variants, match paths, Option/Result, `?`, target layout queries (**definition open — CD-056**),
and the dedicated C5 Drop fixture.
