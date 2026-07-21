# WP-C5.2 — Scalar Native Lowering

Gate: C5 (Native Core Backend MVP). Scope from `COMPILER-ROADMAP.md` WP-C5.2, detailed by
`WP-C5-ENTRY.md` §14 (C5.2a-e). Builds directly on WP-C5.1's approved representation contract
(§6-10, CD-042/046) and the backend/runtime skeleton (CD-044) — no new CE4-shaped decisions are
expected in C5.2 itself; it implements what C5.1 already specified.

## C5.2a — Primitive values and constants

### Status: CLOSED 2026-07-21 (CD-047).

Delivered: `starkc/src/backend/generated_rust/emit_types.rs` gained `emit_constant(&Constant) ->
Result<String, BackendDiagnostic>`, covering every primitive `Constant` variant per the C5.1a
`MirTy` matrix's IN set:

- `Bool`/`Unit` — direct token emission.
- `Int(i128, MirTy)` — value + Rust integer-literal suffix (`i8`/`u32`/etc., reusing `emit_ty`'s
  own strings, which are already valid Rust literal suffixes).
- `Int(codepoint, MirTy::Char)` — `mir::lower`'s own encoding for a `Char` constant (f-3b: "a
  Char literal is its Unicode scalar codepoint, typed Char"); reconstructed via
  `char::from_u32(codepoint).unwrap()` since Rust has no `char` literal suffix. The `.unwrap()`
  cannot fail for a program that reached verified MIR — a failure would mean a compiler defect
  upstream, not a reachable condition.
- `Float(f64, MirTy)` — Rust's `Debug` formatting for `f64` (not `Display`) is used because it
  always includes a decimal point/exponent, guaranteeing the result parses back as a float
  literal once suffixed, and is already the shortest round-tripping string; `NaN`/`Infinity`/
  `-Infinity` have no Rust literal syntax and become `f64::NAN`/`f64::INFINITY`/
  `f64::NEG_INFINITY` expressions instead. A `Float32` constant casts an already-typed `f64`
  expression rather than implementing a separate f32 round-trip formatter.
- `FnPtr`/`Str` — `Unsupported`, deferred to WP-C5.2d/C5.4c (function values) and wherever String/
  output support first lands, respectively; not silently guessed.

**Real bug caught during bring-up, fixed before commit:** the first version appended a `f64`
suffix unconditionally, producing `f64::NANf64` for the NaN case (invalid Rust) — caught by the
test harness described below, not discovered later. Fixed by having the NaN/Infinity branches
return an already-fully-typed expression that the caller does not re-suffix.

**Test approach:** unit tests in `emit_types.rs` round-trip every emitted expression through a
real `rustc --edition 2021 --crate-type lib` parse-and-typecheck (skipped, not failed, when no
Rust toolchain is available — the project's existing `rustc_available()` convention from
`tests/spike_genrust.rs`/`tests/native_c5_1b_skeleton.rs`), rather than asserting on the string
shape alone. 5/5 pass, including negative zero, NaN, and both infinities.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, full workspace suite green (0 failures), `cargo test --test
exec_snapshots` green (4/4).

## C5.2b — Locals, places, assignments, copies, and moves

**Not started.** Implement the C5 place subset and the approved storage strategy (`WP-C5-ENTRY.md`
§7.2-§7.4) — this is where `emit_bodies.rs`'s current C5.1b-only trivial-body restriction (single
entry block, `Nop`/return-slot-`Unit` statements only) gets replaced with real place-based local
declarations, assignments, and Copy semantics for `Copy`-classified scalar types. Non-`Copy`
storage (`MaybeUninit<ManuallyDrop<T>>`) is deferred until a non-`Copy` local is actually needed
by an admitted C5.2/5.3 construct.

## C5.2c — Operations and control flow

**Not started.** Pure rvalues, checked terminators (arithmetic/cast traps), branches, switches,
loops, returns.

## C5.2d — Direct functions and calls

**Not started.** Concrete instances, parameters, return destinations, call continuations — this
is also where `emit_program.rs`'s current C5.1b-only single-body-program restriction lifts.

## C5.2e — Trap path

**Not started.** Native trap records, source IDs, abort, no-unwind configuration —
`stark-runtime::trap`'s category-vocabulary-only placeholder gets real content here.

## C5.2 exit

Not yet reached. Requires three-engine (HIR/MIR/native) agreement for: scalar arithmetic;
branches; loops; direct calls; successful checked operations; each admitted trap category.
