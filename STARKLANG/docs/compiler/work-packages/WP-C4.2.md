# WP-C4.2 — Typed HIR to MIR Lowering: Scalar Core

Gate: C4. Scope from `COMPILER-ROADMAP.md` WP-C4.2, implemented against the APPROVED MIR v0.1
contract (`mir.md`, CD-028).

## Scope (roadmap)

Lower: literals and locals; unary/binary operations; blocks and assignments; functions and
calls; `if`, loops, break/continue, and return; tuples, arrays, structs, and basic enums;
pattern matching without advanced drop elaboration. Every MIR instruction retains a source span
or documented synthetic origin.

## Deliverables (done, 2026-07-19)

- `starkc/src/mir/mod.rs` — the full MIR v0.1 data model per the approved contract (all
  CD-028-revised shapes: `Drop` terminator, logical `Option`/`Result` enums via `EnumRef`,
  `IndexProof` local kind, `Checked` with one normal successor, closed `RuntimeFn` surface,
  interned `FileId` + `SourceInfo` on every statement/terminator) and the deterministic
  versioned textual dump.
- `starkc/src/mir/lower.rs` — the scalar-core lowering: monomorphised-only instance discovery
  (deterministic, deduplicated, from `main`); checked integer arithmetic/negation and float
  div/rem as `Checked` terminators; float add/sub/mul and comparisons as total rvalues;
  short-circuit `&&`/`||` as control flow; compound assignment; `if` (statement and value
  forms), `while`, `loop`, `for`-over-range (desugared with labeled synthetic provenance),
  `break`/`continue`/`return`; direct calls to non-generic instances; **function values and
  indirect calls** (`FnPtr` constants + `FnValue` callees, CD-021 items 16/17); tuples, arrays,
  structs (written-order evaluation, declaration-order aggregation), user enums incl. unit
  variants and struct-variant literals; `Option`/`Result` construction as logical-enum
  aggregates (`Some`/`None`/`Ok`/`Err`) and matching via `Discriminant` + `SwitchInt` with
  `VariantField` payload binding; `print`/`println` via the closed runtime surface with
  uniform checked widening casts.
- `starkc/tests/mir_lowering.rs` — 6 tests: frozen-corpus scalar cases lower with contract
  structural invariants (sealed single-terminator blocks, in-bounds targets, valid `FileId` on
  every statement/terminator); dump determinism + version header; a reviewable golden
  mini-dump pinning the approved shapes; fn-value/indirect-call lowering; Option-as-logical-
  enum lowering (aggregate + discriminant, no runtime call); clean `Unsupported` reporting for
  out-of-subset constructs (generics, strings, methods) naming C4.5.

## Boundaries honored

- **Scalar-core drop restriction:** any type with a `Drop` impl is `Unsupported` here — drop
  elaboration is C4.5; consequently no `Drop` terminators are emitted yet (the model carries
  them).
- Everything out of subset (generics/monomorphisation, methods/trait dispatch, strings/Vec,
  `?`, casts-from-source, indexing/`CheckIndex` use, `panic`, references in MIR) returns a
  clean `LowerError::Unsupported` naming C4.5 — nothing is silently mislowered.
- Evaluation order (CD-007/CD-010) is preserved structurally: left-to-right operands/arguments/
  fields, RHS-before-LHS-place, condition/scrutinee before branches.

## Corpus coverage (lowering-level)

Lowers today: `expr_stmt__01`, `expr_stmt__03`, `primitive__01`, `primitive__02`,
`struct_enum_trait__02` (enum + struct-variant literals + Float64 + match). The remaining
corpus cases need C4.5 constructs and are covered by the unsupported-reporting test.

## Done when (roadmap) — status

- [x] The listed scalar constructs lower.
- [x] Every MIR instruction retains a source span or documented synthetic origin (asserted
      per-statement/terminator in tests; synthetic kinds: ForLoopDesugar, ShortCircuit,
      MatchDesugar, ReturnSlot).
- [x] Full workspace green (611/0/2), fmt + clippy clean.

## Next

WP-C4.3 — MIR verifier (V-CFG/V-TY/V-MOVE/V-DISC/V-DROP/V-IDX/V-FN/V-SRC/V-RT from the
contract §10, `MIR-xxxx` internal diagnostics, safe failure). Then WP-C4.4 (MIR interpreter,
differential vs the HIR oracle over the frozen corpus).
