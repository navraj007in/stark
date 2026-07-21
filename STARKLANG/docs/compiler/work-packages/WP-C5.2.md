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

### Status: CLOSED 2026-07-21 (CD-049).

Delivered per `WP-C5-ENTRY.md` §14:

- **Arbitrary control flow via block-index dispatch** — `emit_bodies.rs` was restructured from
  C5.2b's single-block-plus-dead-trailer shape to `let mut __bb: u32 = <entry>; loop { match __bb
  { 0 => { ... }, 1 => { ... }, ... } }`, one match arm per real MIR block. Rust has no `goto`;
  this is the standard technique for emitting an arbitrary basic-block graph (including cycles,
  for loops) without first recovering structured `if`/`while` shapes — the same approach rustc's
  own backends use at the LLVM-IR level. `Terminator::Goto`/`SwitchInt` both reduce to `__bb =
  target; continue;`, so branches and loops need no special-casing beyond emitting the right
  target index; `Terminator::Return` becomes `break _0;` (the loop's value becomes the function's
  tail expression); `Terminator::Unreachable` becomes Rust's `unreachable!()` (aborts, since the
  generated crate's `panic = "abort"` profile applies).
- **Pure rvalues** — `UnOp::Not`/`FloatNeg`; `BinOp` exhaustively matched (no fallback arm, so a
  new `MirBinOp` variant would fail to compile rather than silently reach `Unsupported`):
  `Eq`/`Ne`/`Lt`/`Le`/`Gt`/`Ge` → `==`/`!=`/`<`/`<=`/`>`/`>=`, `FloatAdd`/`Sub`/`Mul` →
  `+`/`-`/`*`, `BitAnd`/`Or`/`Xor` → `&`/`|`/`^`; `LayoutQuery` → real target-dependent
  `core::mem::size_of::<T>()`/`align_of::<T>()` against the canonical generated type (§8.2 — no
  longer the C4 interpreter's placeholder `(8, 8)`).
- **Checked terminators** — `Add`/`Sub`/`Mul`/`Div`/`Rem`/`Neg`/`Pow`/`Shl`/`Shr`/`FloatDiv`/
  `FloatRem`/`Cast`, matching `mir::interp::eval_checked` (the semantic oracle) exactly: every
  integer op widens both operands to `i128`, computes with Rust's native `checked_*`, then
  range-filters the result against the DESTINATION type before narrowing back — not native
  narrow-width `checked_*` directly. Provably equivalent to native narrow-width checked
  arithmetic for `Add`/`Sub`/`Mul`/`Div`/`Rem`/`Neg`/`Pow` (the true mathematical result either
  fits the narrow range or it doesn't, and `i128` can never itself overflow at these widths), but
  NOT optional for `Shl`: Rust's native `checked_shl` only validates the shift count, silently
  dropping overflowed bits within the narrow type, whereas STARK traps `IntegerOverflow` when a
  left shift's true result does not fit — the widen-then-filter approach is used uniformly rather
  than optimising the operators where it isn't strictly required, to stay a provable match
  against the oracle rather than a per-operator "should be equivalent" argument. Trap categories
  are read directly from the terminator's own `TrapInfo` (already assigned correctly per-operator
  by lowering) rather than re-derived, with the one documented exception `mir::interp` itself
  makes: a `Shl`/`Shr` bad shift count overrides the terminator's default category with
  `InvalidShift`.
- **Minimal trap abort** — `stark_runtime::trap::abort_minimal(category)` reports the category on
  stderr and exits nonzero. Explicitly NOT the final trap ABI (§13.1's source-map/span lookup and
  §13.2's canonical format are WP-C5.2e's deliverable) — necessary now because "an overflow adds
  and silently continues" would violate STARK's always-trap semantics, and C5.2e's own scope is
  specifically the *diagnostic richness* of the abort, not whether one exists at all.
- **`Cast`** — Int↔Int (range-checked against the destination), Int→Float and Float→Float
  (always succeed, native Rust `as` matches interp's rounding), Float→Int (NaN/range-checked via
  `.trunc()`, matching interp's exact condition).

**Real bug caught during bring-up, fixed before commit — not cosmetic, a genuine soundness gap:**
the first version kept WP-C5.2b's "declare every local uninitialised, let rustc's
definite-assignment analysis catch a lowering bug" strategy. That only worked for a single
straight-line block. Once a body has more than one block, each `match __bb { N => {...} } `arm is
an independent branch of one ordinary Rust match from rustc's point of view — it has no notion
that arm 1 is only reachable after arm 0 already ran and assigned a local, because that fact
lives in the *data flowing through* `__bb`, which rustc does not track across `continue`. Every
one of the first real multi-block test programs (arithmetic, division-by-zero, the `while` loop)
failed to compile with `E0381 used binding isn't initialized` the first time this was tried
against real generated code, not a hypothetical review comment. Fixed by default-initialising
every local (`emit_types::default_value_expr`) — the standard fix for CFG-to-match-dispatch
codegen, trading away the "lowering bug catches itself" property C5.2b's record claimed (revised
here rather than left stale) in exchange for correctness across arbitrary control flow; MIR's own
verifier (V-MOVE-1) remains responsible for catching genuine lowering bugs.

**Test approach:** `starkc/tests/native_c5_2c_operations.rs` — five new end-to-end native
compile-and-run proofs: full checked-arithmetic/comparison coverage (add/sub/mul/div/rem/shift/
bitwise/float, all succeeding), an `Int32` overflow that must trap (nonzero exit), a
division-by-zero that must trap, an `if`/`else`, and a `while` loop counting to 5 — plus the
existing C5.1b/C5.2b proofs re-run unchanged as regressions on the restructured emitter.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, scoped tests green (`backend::` 16/16, new test 5/5,
`native_c5_2b_locals` 2/2, `native_c5_1b_skeleton` 1/1), `cargo test --test exec_snapshots` green
(4/4). Full workspace suite not re-run this WP per the test-run-frequency policy (last green at
WP-C5.2a; this WP's changes are additive and narrowly scoped to `backend::generated_rust` +
`stark-runtime::trap`).

## C5.2d — Direct functions and calls

### Status: CLOSED 2026-07-21 (CD-050).

Delivered per `WP-C5-ENTRY.md` §14:

- **Multi-function programs** — `emit_program.rs`'s single-body restriction (present since
  WP-C5.1b) is lifted. Every body in `program.bodies` is emitted as its own Rust item; the entry
  instance is still special-cased as Rust's literal `fn main()` with the version-check prologue
  (a Rust requirement, not a STARK one), every other body goes through the new
  `emit_bodies::emit_function`. `lower_program`'s own doc comment already guarantees the body set
  is self-contained and transitively-reachable ("the entry `main` plus every transitively-called
  supported function"), so no separate reachability/linking logic was needed here.
- **Real parameters** — `emit_bodies::emit_param_list` maps each `body.params[j]` to the local
  whose `LocalKind` is `Param(j)` (a local's `LocalId`/position and its parameter INDEX are not
  the same number and must not be assumed to line up positionally) and emits it as a `mut`
  Rust function parameter under that local's own `_N` name, so the body's existing statement
  emission needs no special-casing to read a parameter — it is just another local. Parameters are
  `mut` for the same reason every other local is: MIR does not distinguish a reassignable
  parameter from any other local, so nothing here tries to prove one is never reassigned.
  `emit_block_body`'s local-declaration loop now explicitly `continue`s past `Param`-kinded
  locals rather than declaring them a second time.
- **Direct calls** — `Terminator::Call` with `Callee::Instance` lowers to an ordinary Rust call
  expression, `{name}({args})`, using the same `mangle::function_name_for_symbol` naming
  authority for both defining a function and calling it (the entry symbol always maps to `main`,
  every other symbol to its sanitized form) — one source of truth, not two conventions that could
  drift. `Callee::FnValue` (indirect) stays deferred to WP-C5.4c; `Callee::Runtime` stays deferred
  to wherever the first `RuntimeFn` group lands.

**No bug this time** — unlike C5.2b (the entry Unit-return check needed relocating) and C5.2c
(the uninitialised-locals soundness gap), the parameter-shadowing hazard this WP's own design
raised (declaring a `Param`-kinded local a second time inside the block body would silently
shadow the real argument with a fabricated default, since `emit_block_body`'s default-init loop
runs over every local by position) was caught in review before writing the test, not by the test
failing — the loop explicitly `continue`s past `Param`-kinded locals rather than defaulting them.
Both new end-to-end tests passed on the first run.

**Test approach:** `starkc/tests/native_c5_2d_calls.rs` — a two-parameter `add` called from
`main`, and a richer case (a three-parameter `clamp` helper feeding an `if`, plus a second
`Float64`-parameter/`Bool`-returning helper) proving parameter-index-to-local mapping and
multi-function symbol naming both hold beyond the single-call trivial case — plus the C5.1b/
C5.2b/C5.2c proofs re-run unchanged as regressions.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, scoped tests green (`backend::` 18/18, new test 2/2, prior
regressions 8/8), `cargo test --test exec_snapshots` green (4/4). Full workspace suite not
re-run this WP per the test-run-frequency policy.

## C5.2e — Trap path

### Status: CLOSED 2026-07-21 (CD-051).

Delivered per `WP-C5-ENTRY.md` §13.1/§13.2:

- **Real source location, resolved at compile time** — every checked-operation trap site now
  carries the STARK source file path, 1-based line, and 1-based column of the trapping
  terminator. Deliberately resolved at CODEGEN time (`SourceFile::line_col` against
  `MirProgram::files`, both already available to the backend) and baked into the generated call
  site as literals, rather than building §13.1's compact-span-ID-plus-runtime-lookup-table —
  documented as a legitimate simpler MVP alternative (the ID/table design exists to deduplicate
  span data for large programs with many trap sites; at MVP program sizes, inlined literals are
  simpler and exactly as correct), not an oversight, revisit-able if generated-binary size ever
  makes deduplication worth it.
- **The real trap ABI** — `stark_runtime::trap::abort(category, file, line, column) -> !`
  replaces C5.2c's `abort_minimal` placeholder outright (not wrapped): reports the category's
  message and `file:line:column` on stderr, then exits with code **101** — matching `stark
  run`'s own already-established convention exactly (`starkc/src/bin/stark.rs`:
  `ExitCode::from(if error.is_trap { 101 } else { 1 })`), reused rather than invented. Category
  messages (e.g. "integer overflow", "division by zero") are NOT claimed to match the HIR
  interpreter's own ad hoc per-call-site strings byte-for-byte — no such canonical table exists
  there to match, and the differential comparator (§15.1) checks category plus source file/line,
  not stderr text.
- **One naming authority for both trap sites** — `emit_abort_call` is the single place that
  assembles a `stark_runtime::trap::abort(...)` call, used both for a terminator's own default
  category and for the `Shl`/`Shr` `InvalidShift` override, so the two trap sites within one
  checked operation can never independently drift in how they resolve or format a location.
- **Retrofit:** C5.2c's own two trap tests (`integer_overflow_traps_natively`,
  `division_by_zero_traps_natively`) were tightened from a loose `assert_ne!(status, Some(0))` to
  the exact `assert_eq!(status, Some(101))` now that the precise contract exists — the kind of
  strengthening a later WP is expected to do to an earlier WP's own tests once it settles a
  question the earlier WP had deliberately left loose.

**Test approach:** `starkc/tests/native_c5_2e_traps.rs` — four new tests: an overflow trap
asserting BOTH the category message and an EXACT `file:line` match (source deliberately written
with no leading blank line so the trapping statement's line number is unambiguous, not a loose
"some plausible number" check), plus division-by-zero, invalid-shift, and cast-failure traps each
asserting category message and exit code.

**Validation:** `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean, scoped tests green (`backend::` 18/18, new test 4/4, all
prior native regressions including the two retrofitted C5.2c tests), `cargo test --test
exec_snapshots` green (4/4).

## C5.2 exit

> **SUPERSEDED 2026-07-21 by "C5.2 closure" below (CD-053).** The owner resolved the open
> decision this section ends on: build the harness now as the C5.2 closure addendum, do **not**
> defer it to WP-C5.6. It is built, the exit condition is met, and **WP-C5.2 is CLOSED**. The
> text from here to the end of the "Review-response pass" section is the historical pre-closure
> record and is left unedited.

**Not yet reached — and this WP is not claiming otherwise.** §14's stated exit condition is
explicit: *"Require three-engine agreement for: scalar arithmetic; branches; loops; direct
calls; successful checked operations; each admitted trap category."* Every `native_c5_2*.rs` test
written across C5.2a-e compiles a program and asserts on the NATIVE engine's own output in
isolation (exit code, stdout, stderr content) — none of them actually run the SAME source through
the HIR interpreter and the MIR interpreter and automatically diff all three, the way
`mir_differential.rs` already does for the HIR-vs-MIR pair. The individual constructs have been
reasoned through carefully against `mir::interp` as the semantic oracle (cited throughout this
document and the state-file CD entries) and tested for internally-consistent native correctness,
but that is not the same evidentiary bar as an automated three-engine comparator, and this record
says so rather than quietly treating "native looks right" as satisfying "three engines agree."
**Building that harness (`WP-C5-ENTRY.md` §15.1's `source case -> HIR -> MIR -> native ->
comparator` pipeline, `native_scalar.rs`/`native_traps.rs`/etc. per §15.2) is the next real
decision point** — whether it lands as a C5.2-closing addendum or is deliberately deferred to
WP-C5.6 (which already co-owns cross-backend snapshot replay per the WP-C4.4/CD-018 carry-forward)
is an open question for the owner, not resolved unilaterally here.

## Review-response pass (CD-052, 2026-07-21)

An external review of head `37828a07` raised seven findings against C5.1/C5.2. **All seven were
verified as real against the code — no false positives.** Four are fixed here, one is recorded as
a WP-C5.3 opening condition, and two are escalated to the ABI's owner rather than resolved. Full
record: `COMPILER-STATE.md` CD-052 (DEV-091 … DEV-096). The two escalated findings are written up
in `STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md` (PROPOSED — the Native
Provider ABI v0.1 is owner-approved under CD-046, so neither it nor `provider_abi.rs` is changed
without a CE4 decision).

Two things about this pass are worth stating plainly, because both cut against the C5.2a-e record
above:

**1. `Terminator::Trap` was still `Unsupported` when C5.2e was recorded as closed (CD-051).** The
trap ABI landed for `Terminator::Checked`'s *conditional* traps — the ones checked operations
raise — but the *unconditional* `Trap` terminator that `mir::lower` emits for `panic`/`assert`/
`assert_eq`/`assert_ne` still returned a backend `Unsupported` diagnostic. That is squarely
C5.2e's own deliverable, and CD-051's "real trap ABI delivered" claim was therefore broader than
what had been built. It is now implemented for the message-less form (every compiler-generated
trap, and all three `assert*` builtins, which `mir::lower` emits with `message: None`); the form
carrying a user `&str` message still needs string values and remains WP-C5.3.

**2. Every C5.2a-d success-path test observed only an exit code, never a computed value.** The
arithmetic test computed sums, products, shifts and comparisons; the call tests computed
`add(2, 3)` and `clamp(15, 0, 10)`. None of the results was checked. A backend that returned zero
from every function would have passed nearly all of them — so those tests proved the programs
*compiled and ran*, not that they *computed*. This was not a gap the C5.2a-e records acknowledged.

Both are now closed by the same mechanism: with `Terminator::Trap` supported, `assert_eq`/`assert`
inside the STARK program itself is the observation channel that native `println` cannot yet be
(WP-C5.3), and a failed assertion reaches the C5.2e trap ABI as an exit-101 `assertion failed`.
`native_c5_2c_operations.rs::a_false_assertion_traps_natively` is the **negative control** that
makes the rest of the suite meaningful: without it, "exit 0" stays ambiguous between "every
assertion held" and "assertions were compiled away to nothing".

### The semantic defect, and why the existing evidence missed it

The most serious finding (DEV-091) was that float→integer casts *accepted out-of-range values at
64-bit widths* in both the MIR interpreter and the native backend: both compared the truncated
value against `max as f64`, which rounds **up** at those widths (`u64::MAX as f64` is 2^64), so
exactly 2^64 passed the range guard and the saturating `as` then clamped it to `u64::MAX` instead
of trapping.

The HIR oracle was already correct — it truncates to `i128` and range-checks in exact integer
arithmetic — so this was a genuine two-engine divergence from the oracle, and
`mir_differential.rs` would have caught it on the day it was written. **It survived because no
case in the frozen corpus or the inline set had ever exercised a 64-bit cast boundary.** The gap
was in the corpus, not the comparator. Ten boundary cases now exist — seven differential (2^64,
greatest f64 below 2^64, 2^63, greatest below 2^63, -2^63 inclusive, below -2^63, and truncation
ordering) and three native.

Writing those tests immediately surfaced an eighth defect the review had not named (DEV-096): the
oracle reported *every* out-of-range cast, at every width, as `IntegerOverflow` rather than
`CastFailure`, because both cast arms in `interp.rs` routed through `check_integer_range` and
inherited its hardcoded arithmetic-overflow message. The new tests failed on trap **category**,
not on the bound. That is the pattern worth noting: the boundary tests were written to pin one
defect and found a second, unrelated one in a different engine, at widths the first defect never
touched.

### Standing on the C5.2 exit condition

Unchanged by this pass. The three-engine comparator §14 requires still does not exist, and the
new native tests are still native-only assertions — stronger ones, since they now observe values
rather than exit codes, but not automated three-engine agreement. The DEV-091 boundary cases *are*
covered by the real HIR-vs-MIR comparator (`mir_differential.rs`), with the native engine pinned
separately against the same expected values; that is two automated engines plus a manually
corresponding third, not the §15.1 pipeline. **WP-C5.2 remains not closed**, and the open decision
recorded above — build the harness now, or defer to WP-C5.6 — is still the owner's.

---

## C5.2 closure — the three-engine differential harness (CD-053, 2026-07-21)

### Status: WP-C5.2 CLOSED 2026-07-21 (CD-053).

The owner directed that the harness be built now, as this package's closure addendum, rather than
deferred to WP-C5.6: §14's exit condition is C5.2's own, and deferring it would have left the
package open while its closure evidence moved into an unrelated later one. Delivered:
`starkc/tests/three_engine_differential.rs`, 20 tests, all green.

### What the harness is

It implements **§15.1's three-engine pipeline**, and compares traps in **normalised** form for
C5.2. **One** source string per case, run through **all three** engines — the HIR interpreter
(the semantic oracle, charter §1.6 rule 6), the MIR pipeline (lower → verify → execute), and the
native binary (lower → verify → emit → cargo build → run the executable) — with each result
**normalised into one common `Outcome`** and all three required equal.

Stating the one deviation from §15.1's comparison list up front rather than in a footnote: **raw
stderr byte equality is not compared, because the HIR oracle has no canonical stderr format to
compare against.** Its trap text is a set of ad hoc per-call-site message strings, which
`stark_runtime::trap`'s own doc comment already records it does not attempt to match byte for
byte. What the harness compares instead is what those bytes *mean* — trap category plus exact
file/line/column, parsed back out of the native binary's stderr. §15.1's remaining dimensions are
covered or explicitly N/A; see the dimension table below.

```rust
enum Outcome {
    Completed { stdout: String, exit: i32 },
    Trapped { category, file, line, column, stdout_before },
}
```

The normalisation is the substance, because the three engines report failure in three unrelated
vocabularies: the oracle raises prose plus a byte span, MIR raises a `TrapCategory` plus a
`SourceInfo`, and the native binary writes a line of stderr text and exits with a process code.
Projecting all three onto one type is what makes "agreement" a mechanical check rather than a
human reading two test files side by side. Concretely, each case compares:

| Dimension | How each engine supplies it |
|---|---|
| normal completion vs. trap | `Ok`/`Err` (HIR, MIR); exit code 101 vs. other (native) |
| exit status | `Execution.status` (HIR, MIR); process exit code (native) |
| trap category | prose → category via `oracle_category` (HIR); `TrapCategory` (MIR); stderr header matched against `stark_runtime::trap::TrapCategory::message()` (native) |
| trap file/line/column | `SourceFile::line_col(span.lo)` (HIR, MIR); parsed from the `-->` line (native) |
| observable output | `Execution.output` / pre-trap partial output (HIR, MIR); captured stdout (native) |

Three deliberate properties:

- **The oracle's prose is normalised explicitly, never fuzzily.** `oracle_category` maps known
  messages to categories and **panics** on anything unrecognised. A silent fallback would let a
  wrong-category trap normalise into whatever the other two engines happened to say — the exact
  failure the comparator exists to catch. Categories outside the C5.2-admitted surface
  (`IndexOutOfBounds`, `UnwrapNone`/`UnwrapErr`, message-carrying `Panic` — all WP-C5.3) are
  listed as explicit "not admitted yet" panics rather than guessed at.
- **The native category table is the runtime's own, not a copy.** `stark_runtime::trap::
  TrapCategory::message()` was made `pub` for this (the only production change in the addendum,
  five lines of doc plus a visibility keyword); a second table in a test file would have drifted
  the first time a category's wording changed.
- **Trap location is compared exactly, and stated independently.** `agree_trapping` takes the
  expected line as a parameter, so a case whose three engines agree on the *wrong* line still
  fails. Agreement alone is not correctness.

### The observable-output dimension, handled honestly

Native `println` is `Unsupported` until WP-C5.3 (no string values yet), so the C5.2-admitted
source surface produces no observable output at all, and value observation runs through in-program
`assert`/`assert_eq` — which reach the C5.2e trap ABI in all three engines when they fail.

That could have been a quietly excluded comparison dimension. Instead the harness **enforces** it:
`NATIVE_STDOUT_SUPPORTED: bool = false` gates a precondition asserting every case's oracle run
produced no output, so full `Outcome` equality across three engines is *total* rather than
skipping a field. When native output lands, flipping that constant drops the precondition and the
same equality check starts comparing real stdout bytes on all three sides — no other change.

### Coverage against §14's exit condition

| §14 requires three-engine agreement for | Case(s) |
|---|---|
| scalar arithmetic | `scalar_arithmetic_agrees` — `+ - * / %`, unary minus, precedence and parens, all six comparisons, `<< >> & \| ^`, negative-operand division/remainder, `Int8`/`Int64`/`UInt32` widths, `Float64` arithmetic |
| branches | `branches_both_directions_agree` — both directions of two `if`/`else`s, an `else if` chain taking the middle and the final arm, nested `if`, `if` with no `else`, `if` as an expression in both directions, `&&`/`\|\|`/`!` |
| loops | `zero_iteration_loop_agrees` (body never runs, two shapes), `multi_iteration_loop_agrees` (accumulate, `continue`, `break`, nested loops) |
| direct calls | `direct_calls_agree` — multi-function program, argument order via a non-commutative callee, a no-argument function, a `Unit`-returning function, nested calls as arguments, recursion, a call inside a loop |
| successful checked operations | `successful_checked_operations_agree` — arithmetic that lands exactly on `Int32::MAX`/`MIN`, shift counts at width-1, in-range casts at the exact boundary of the narrower type, widening casts, int↔float |
| each admitted trap category | `IntegerOverflow`, `DivideByZero` (twice: `/` and `%`), `InvalidShift`, `CastFailure`, `AssertFailure` (twice: `assert_eq` and bare `assert`) |

Review regressions from CD-052, re-pinned as three-engine agreement rather than per-engine
assertions:

- **DEV-091** (float→int casts accepted out-of-range values at 64-bit widths): four cases —
  in-range boundary conversions, exactly 2^64 → `UInt64`, exactly 2^63 → `Int64`, and the first
  f64 below `Int64::MIN`. Both sides of every bound.
- **DEV-096** (the oracle reported every out-of-range cast as `IntegerOverflow`):
  `out_of_range_cast_is_a_cast_failure_not_an_overflow` — a case only a three-engine comparison of
  *categories* can hold, since all three engines exit 101 either way.
- **DEV-092** (symbol mangling was not injective): `colliding_symbol_names_stay_distinct_functions`
  — the source-level consequence, not just the encoding. `mod m { pub fn f() }` and a top-level
  `fn m_3a_3af()` collided into one Rust identifier under the previous encoding; both are called
  and both return values are observed.
- **The negative control**: `a_false_assertion_traps_in_all_three_engines`. Every completing case
  observes values through assertions, so "all three completed with exit 0" is evidence *only*
  because a false assertion demonstrably fails the run in all three engines. Without this case the
  whole harness would pass against three engines that ignored assertions.

Plus `the_comparator_rejects_disagreeing_outcomes`, a guard on the comparator itself: it asserts
that outcomes differing only in trap column, or only in category, are unequal — so a future
weakening of the normalisation into something coarser fails a test instead of silently passing
everything.

### The harness was mutation-tested before being trusted

A comparator that passes proves nothing until it has been shown to fail. Two mutations were
injected into the native backend, run, and reverted:

1. `checked_add` → `checked_sub` in `emit_bodies.rs`'s checked-arithmetic emission. Result:
   `scalar_arithmetic_agrees` failed with `MIR/NATIVE DISAGREEMENT`, MIR `Completed` vs. native
   `Trapped { AssertFailure, line 4 }`. The value dimension is live.
2. `line as u32` → `line as u32 + 1` in `resolve_source_location`. Result:
   `integer_overflow_trap_agrees` failed with the same category and file but line 4 vs. line 5.
   The location dimension is live, independently of the category dimension.

Both mutations were reverted and the suite re-verified green. `git diff` confirms no backend
change survives from either.

### Against §15.1's comparison list, dimension by dimension

§15.1 enumerates seven things the native differential harness must compare. Stating where each
one stands, rather than only the ones that are covered:

| §15.1 dimension | Status at C5.2 scope |
|---|---|
| stdout bytes | Compared as part of `Outcome` equality. The admitted surface produces none, and the harness **enforces** that (see above) rather than skipping the field |
| stderr bytes | **Not compared as bytes** — the HIR oracle has no canonical stderr format to compare against (ad hoc per-call-site strings, which `stark_runtime::trap`'s own doc records it does not attempt to match). Compared as *meaning* instead: the trap header and `-->` location are parsed back into category + file/line/column. On the non-trap path, stderr is asserted **empty** for the oracle and the native binary |
| exit code | Compared (`Completed.exit`; trap ⇒ 101) |
| trap category | Compared |
| trap source file and line | Compared, plus **column**, which §15.1 does not require |
| observable Drop events | **N/A at C5.2** — no C5.2-admitted type has a destructor (every admitted `MirTy` is a primitive, hence `Copy`). This is WP-C5.3d's dimension and needs a channel that does not exist yet |
| build success/failure classification | Every case must build; a `BackendDiagnostic` fails the case with the diagnostic attached. Classifying *expected* build failures is WP-C5.5b's (§12.5), which owns the classification vocabulary |

§15.2 asks for distinct suites whose categories stay visible, naming `native_scalar.rs`/
`native_traps.rs`/etc. and allowing different names. This lands as one comparator file with the
categories visible as case groups (arithmetic/branches/loops/calls/checked-operations = scalar;
every case expecting a trap = traps), because at C5.2 scope splitting the same comparator across
two files would
duplicate it rather than separate anything. The remaining §15.2 suites (aggregates, layout, drop,
function values, packages, build CLI, snapshot replay) belong to the packages that introduce what
they test.

### What this closure does NOT claim

- Only the C5.2-admitted surface is covered. Aggregates, enums, `Option`/`Result`, matches, `?`,
  strings, and Drop-bearing types are WP-C5.3 and are absent by construction, not by oversight.
- Native stdout is not compared against real bytes, because it cannot be produced yet. The
  harness enforces the precondition that makes its absence total rather than pretending
  otherwise (see above).
- Per-engine tests (`native_c5_2*.rs`, `mir_differential.rs`) remain and remain useful, but they
  are **supplementary**. The §14 exit condition is satisfied by this harness alone; the owner's
  direction is explicit that engine-specific tests do not satisfy it.
- The `exec_snapshots` frozen corpus is not yet routed through the native engine. Cross-backend
  snapshot replay stays WP-C5.6's (per the WP-C4.4/CD-018 carry-forward); what moved out of
  C5.6 is the three-engine comparator, not the corpus.

**One cross-document divergence, stated rather than left for a reviewer to find.**
`COMPILER-ROADMAP.md`'s own WP-C5.2 bullet list includes "tuples and simple structs", which this
closure does **not** deliver. That is not a gap in the closure: the owner-approved entry plan
(`WP-C5-ENTRY.md` §14, CD-042) decomposes C5.2 into C5.2a-e with no aggregate step, and moves
aggregates into WP-C5.3, which it correspondingly re-titles "Aggregates, enums, and error values"
against the roadmap's "Enums, matches, and error values". The approved decomposition governs; the
roadmap's earlier sketch has not been amended to match. Closing C5.2 against §14 rather than
against the roadmap bullet is deliberate and is recorded here so the discrepancy is visible.

### Validation

`cargo fmt --all -- --check` clean; `cargo clippy --workspace --all-targets --all-features --
-D warnings` clean; `three_engine_differential` 20/20; `mir_differential` and all five
`native_c5_*` suites green; `cargo test --workspace` green — **884 passed / 0 failed / 2 ignored
across 52 test binaries**. The pass's only production change is a visibility widening
(`TrapCategory::message()` → `pub`).

(The figure first recorded here, 818 across 40 binaries, was an undercount of the same green run:
the background capture lost its first 24 lines to buffering, dropping 12 suites from the tally.
Corrected after a complete re-capture disagreed on the suite count.)
