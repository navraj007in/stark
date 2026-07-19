# WP-C4.3 — MIR Verifier

Gate: C4. Scope from `COMPILER-ROADMAP.md` WP-C4.3, implemented against the APPROVED MIR v0.1
contract (`mir.md` §10, CD-028).

## Scope (roadmap)

Validation for: well-formed block graph; terminator presence; valid local/place references;
type consistency; move-before-use invariants at MIR level; valid discriminant operations;
drop-flag consistency; no unsupported instruction reaching a backend silently; source/
provenance availability. Invalid MIR produces a compiler-internal diagnostic and fails safely —
never undefined backend behaviour.

## Deliverables (done, 2026-07-19)

- `starkc/src/mir/verify.rs` — `verify_program(&MirProgram) -> Result<(), Vec<MirError>>`
  implementing all §10 obligations with the first allocation of the `MIR-xxxx` internal
  diagnostic namespace (charter §5.1):

```text
MIR-0001 block target out of bounds            (V-CFG-1/2)
MIR-0002 local/place reference out of bounds   (V-CFG-2)
MIR-0003 projection type mismatch              (V-CFG-2, step-by-step typing)
MIR-0004 assignment/operand type mismatch      (V-TY-1)
MIR-0005 call/checked signature mismatch       (V-TY-1)
MIR-0006 bare unsized type outside Ref         (V-TY-3)
MIR-0007 use of a possibly-moved place         (V-MOVE-1)
MIR-0008 discriminant/variant misuse           (V-DISC-1)
MIR-0009 drop on non-droppable / drop-flag     (V-DROP-1/2)
MIR-0010 index-proof discipline violation      (V-IDX-1/2)
MIR-0011 arithmetic/comparison on FnPtr        (V-FN-1, TYPE-FN-001)
MIR-0012 (reserved: runtime-set violation — structurally impossible while RuntimeFn is a
          closed Rust enum; the code is reserved for a serialized-MIR future)
MIR-0013 SourceInfo without a valid FileId     (V-SRC-1)
```

  (V-TY-2, monomorphised-only, is structural: `MirTy` has no `Param`/`Infer` variants.)
- Typing walks places projection-by-projection through the lowering-populated `TypeContext`
  (struct fields, user-enum variant payloads; `Option`/`Result` payloads derived from type
  args). Bidirectional aggregate checking (expected-type driven, so `None` checks against
  `Option<T>` without inference).
- **V-MOVE-1 dataflow**: conservative whole-local, any-path (union-join) forward analysis to a
  fixpoint; projected moves conservatively move the whole local; whole-local assignment
  reinitializes. Documented as a refinement point — it can reject over-clever legal MIR, never
  accept a moved-from read.
- **Safe failure hardened by test**: the negative suite caught the move-dataflow walking a
  broken CFG edge and panicking — exactly the unsafe failure the contract forbids; fixed by
  skipping already-reported broken edges. The verifier now reports-and-continues on every
  invalid input in the suite.

## Tests (`starkc/tests/mir_verify.rs`, 14)

- **Positive (the load-bearing one):** every WP-C4.2-lowerable program — 5 frozen-corpus cases
  plus fn-value/Option/struct inline programs — passes verification. Lowering and verifier are
  two independent readings of the contract; their agreement is the faithfulness evidence.
- **Negative:** 13 hand-crafted invalid bodies, one per obligation family, each asserting its
  specific `MIR-xxxx` code: bad target, bad local, type mismatch, runtime signature mismatch,
  use-after-move, discriminant-of-non-enum, drop-flag written non-constant, drop of
  undroppable, `Index` without a proof token (the CE3-revised rule), `Eq` on fn values
  (TYPE-FN-001 at MIR level), invalid FileId, bare unsized local, enum payload arity mismatch.

## Done when (roadmap) — status

- [x] All §10 obligations implemented with `MIR-xxxx` codes.
- [x] Invalid MIR fails safely (panic found by test, fixed) — never undefined behaviour.
- [x] Workspace 625/0/2, fmt + clippy clean.

## Next

WP-C4.4 — MIR interpreter: execute verified MIR and differentially compare against the HIR
interpreter (the semantic oracle) over the frozen corpus (`corpus_version = 1.0.0`), per the
contract's observable comparator.
