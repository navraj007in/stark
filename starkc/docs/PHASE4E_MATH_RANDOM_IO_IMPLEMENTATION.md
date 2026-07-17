# Phase 4E — Math, Random, and I/O Implementation

**Status:** Partially complete (`File` struct deferred — see below)
**Roadmap:** `stark-spec-parity-roadmap.md` Phase 4E
**Spec:** `STARKLANG/docs/spec/06-Standard-Library.md` (Math/IO modules)

## Overview

Closes most of the remaining `core-min` -> full Core v1 stdlib gap:

- Math constants (`PI`, `E`) and functions (`abs`, `min`/`max`/`clamp`,
  `pow`, `log`, `log10`, `exp`, trig functions, rounding functions).
- `Random` — a simple linear congruential generator, per the spec's own
  "(simple linear congruential generator)" annotation.
- `IOError` — a 5-variant error enum (`NotFound`, `PermissionDenied`,
  `AlreadyExists`, `InvalidInput`, `Other(String)`).
- `eprint`/`eprintln` (stderr).
- `read_file`/`write_file` upgraded to return `Result<String, IOError>`
  (was `Result<String, String>`), matching the spec.

All implemented as new interpreter `Builtin`s and a new `Value::Random`/
`Value::IOError` runtime representation — the same architecture every
prior Phase 4 stdlib addition (String/Vec/HashMap/HashSet/Iterator) uses;
no changes to the grammar or AST were needed for any of this.

## Two real gaps found while implementing

### 1. `math::min`/`math::max` collide with the `tensor` extension's bare `min`/`max`

The `tensor` extension already claims bare `min`/`max` as free-function
names (`Builtin::TensorMin`/`TensorMax`, element-wise tensor min/max) —
confirmed no test fixture anywhere in the repo actually exercises bare
`min`/`max`, and a related gating inconsistency exists (`resolve_path_relative`
correctly gates tensor builtins behind `options.tensor()`, but
`resolve_unqualified` — used for simple bare-identifier expressions —
does not, so a bare `min(3, 5)` call in a Core-only program currently
resolves to `Builtin::TensorMin` unconditionally rather than "unknown
function"). Not fixed here (out of scope, touches shared ambiguous-name
resolution code with no dedicated test coverage); Math's `min`/`max`/
`abs`/`clamp` sidestep it entirely — `abs`/`clamp` don't collide and are
bare; `min`/`max` are qualified-only (`math::min`/`math::max`, also
`std::math::min`/`std::math::max`), mirroring the existing
`std::fs::read_file` qualified-path pattern in `resolve_path_relative`.

### 2. `File` can't get first-class runtime representation without restructuring `Value`

The interpreter's `Value` enum (`interp.rs`) derives `Clone` + `PartialEq`
uniformly across ~30 variants (used everywhere — assignment, `assert_eq`,
`HashMap`/`HashSet` keys via a hand-written `Ord`). A real OS file handle
(`std::fs::File`) implements neither trait, so `File` can't just become
`Value::File(std::fs::File)` — it would need either a fallible/panicking
manual `Clone` (files can't always be duplicated) or restructuring
`Value`'s equality to special-case one variant as always-unequal/
identity-compared, which touches a much larger surface than the rest of
this phase for one type. Deferred; `read_file`/`write_file` (whole-file
operations, no persistent handle needed) cover the common cases the
spec's `File::open` + `read_to_string` / `File::create` + `write_str`
combinations exist for. `IOError` — the *other* half of `File`'s spec
surface — has no such problem (it's plain data, no OS resource) and is
fully implemented, using the same "special-cased `Value` variant" pattern
`Option`/`Result` already use (`IOError::NotFound` etc. resolve directly
to `Builtin` constructors, not through the generic `Value::Enum` path,
since there's no real HIR item behind them).

## Architecture notes

- **PI/E as bare constants**: resolved via `Res::Builtin` returning
  `Ty::Primitive(Float64)` directly from `builtin_type` (not wrapped in
  `Ty::Fn`) — the same pattern `None` already uses to be a valid bare
  value, not just a call target. Evaluated in `eval_path` (the Path
  expression evaluator), not `call_builtin` (which only ever sees actual
  calls).
- **`Random`**: `Ty::Core(CoreType::Random, [])`, `Value::Random(u64)`.
  Its three methods (`next_int`/`next_float`/`range`, all `&mut self` per
  spec) go through the exact same "mutating core method" dispatch path
  `Vec::push`/`HashMap::insert` use (`core_method_signature` in
  `typecheck.rs`, the `mutating` name list + big `match target` in
  `interp.rs::call_core_method`) — just a new `Value::Random(seed) =>`
  arm, no new dispatch machinery. LCG uses Knuth/MMIX's 64-bit multiplier
  and increment (`6364136223846793005`, `1442695040888963407`); any
  full-period 64-bit LCG constants satisfy the spec's "simple" LCG
  requirement.
- **`IOError`**: `Ty::Core(CoreType::IOError, [])`, `Value::IOError(IOErrorKind)`
  where `IOErrorKind` is a small internal Rust enum. Pattern matching
  (`match err { IOError::NotFound => ..., IOError::Other(msg) => ... }`)
  is special-cased in `interp.rs::match_pattern`, mirroring the existing
  `(Res::Builtin(Builtin::Some), Value::Option(Some(v)))`-style arms for
  `Option`/`Result`. `std::io::Error -> IOErrorKind` mapping
  (`IOErrorKind::from_io_error`) covers the four specific kinds via
  `std::io::ErrorKind`, falling back to `Other(message)`.

## Testing

`starkc/tests/phase4e_math_random_io.rs` — 13 end-to-end tests (parse ->
resolve -> typecheck -> execute), covering: math constants, `abs` (Int and
Float), qualified `math::min`/`math::max`, `clamp`, the transcendental/
rounding function set, `Random` determinism-per-seed and value-range
correctness (`next_float` in `[0,1)`, `range` bounds respected across 20
draws), `eprintln` not polluting captured stdout, `read_file` on a missing
path returning `IOError::NotFound`, a real `write_file`->`read_file`
round trip against a temp file, and `IOError::Other` + all four unit
variants constructing and matching correctly.

362 tests pass across the whole workspace (up from 349 after WP8.3); zero
new clippy warnings; `cargo fmt --check` clean.

## Next steps

`File` struct + methods is the one clearly-scoped remaining piece of
Phase 4E, blocked on deciding how `Value` should represent non-Clone/
non-PartialEq runtime resources — worth a dedicated design pass rather
than folding into a "close the stdlib gap" phase. The "real `std` module
tree instead of hardcoded resolver names" exit criterion is a standing
gap across all of Phase 4 (4A-4E alike), not new to this phase.
