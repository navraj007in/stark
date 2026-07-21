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

### Status: CLOSED 2026-07-21 (CD-048).

Delivered per `WP-C5-ENTRY.md` §7.2-§7.4:

- **Real place emission** — `starkc/src/backend/generated_rust/emit_places.rs`:
  `emit_place(&Place)` supports bare locals only (`place.projection` must be empty); any
  projection (`Field`/`VariantField`/`Deref`/`Index`/`ConstIndex`) is `Unsupported`, deferred to
  WP-C5.2c/C5.3 once aggregates/references exist. Local naming reuses MIR's own dump convention
  (`_0`, `_1`, ...) verbatim — already a valid Rust identifier, and its leading underscore
  incidentally also suppresses Rust's `unused_variables` lint on its own.
- **General local declarations** — `emit_bodies::emit_body` (renamed from C5.1b's
  `emit_trivial_unit_body`, which it fully subsumes and replaces, not wraps) declares every body
  local as `let mut _N: T;`, uninitialised. Uniformly `mut` regardless of whether a local is ever
  reassigned, rather than tracking first-vs-later-assignment (which would need a pre-pass once
  branches exist in C5.2c) — harmless because the generated file's `#![allow(unused)]`
  (`emit_program.rs`) already suppresses the resulting `unused_mut` lint. Leaving locals
  genuinely uninitialised (no fabricated default value) means a lowering bug that reads a local
  before any assignment reaches it is caught by rustc's own definite-assignment analysis as a
  compile error in the generated crate — a safety property inherited for free at this scope, not
  implemented by this WP.
- **Assignments** — `Statement::Assign(place, Rvalue::Use(operand))` lowers to `{dest} =
  {value};`; any other `Rvalue` (arithmetic, aggregates, discriminants, layout queries) is
  `Unsupported`, deferred to WP-C5.2c.
- **Copies and moves** — both `Operand::Copy`/`Operand::Move` of a place emit the same bare place
  reference. Sound at this scope specifically because `emit_types::emit_ty` only ever admits
  primitive `MirTy`s, and every primitive is `Copy` by construction (CLAUDE.md: "Copy requires
  all-Copy fields") — a `Copy`-classified value's `Move` is value-identical to its `Copy`. Real
  non-`Copy` move/liveness tracking (`WP-C5-ENTRY.md` §7.2's `MaybeUninit<ManuallyDrop<T>>`
  strategy) is deferred to whichever WP first admits a non-`Copy` `MirTy` (WP-C5.3+).
- **Entry-point Unit-return check relocated** — `emit_program.rs` now checks
  `entry.ret == MirTy::Unit` itself (Rust's `fn main()` must return `()`), rather than
  `emit_body` enforcing it internally; `emit_body` is reusable for an arbitrary-return-type
  ordinary function once WP-C5.2d lifts the single-body-program restriction.

**Test approach:** `starkc/tests/native_c5_2b_locals.rs` — two new end-to-end native
compile-and-run proofs (real `let` locals of `Int32`/`Bool`/`Char`/`Float64`/`UInt8` types, a
copy of one local into another, and separate `Float32`/`Float64` locals), plus
`native_c5_1b_skeleton.rs`'s existing empty-`main` proof re-run unchanged as a regression check
that the more general `emit_body` still handles the C5.1b shape correctly (it does — the old
trivial-body special case is now just one instance of the general local-declare/assign/return
path, not a separately maintained code path).

**One STARK-level, not backend-level, snag caught while writing the test:** `let y: Float32 =
2.5;` fails STARK's own typecheck (`E0001`, `Float64` literal does not coerce to `Float32`) — not
a bug, just a reminder that Core v1 float literals default `Float64` and need an explicit `f32`
suffix; fixed in the test source, not the compiler.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, scoped tests green (`backend::` 16/16, `native_c5_2b_locals`
2/2, `native_c5_1b_skeleton` regression 1/1), `cargo test --test exec_snapshots` green (4/4). Per
the new test-run-frequency policy (owner feedback, WP-C5.2a close-out), the full
`cargo test --workspace --all-targets --all-features` run was not re-run for this WP — it was
last confirmed green at WP-C5.2a's close and this WP's changes are additive/narrowly scoped to
`backend::generated_rust`.

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
