# WP-C4.1 — MIR Design Review

Gate: C4 (MIR Contract and Verified Lowering). Scope from `COMPILER-ROADMAP.md` WP-C4.1.
Escalation: **CE3 — MIR design and verifier contract are foundational architecture; owner
review required before implementation begins.**

## Scope

Define `STARKLANG/docs/compiler/mir.md` covering functions/basic blocks, typed locals and
temporaries, places and projections, constants, explicit control-flow terminators, calls and
returns, aggregates and discriminants, moves and copies, borrows as required by lowering, drop
flags and drop operations, panic/trap abort paths, source spans and provenance, monomorphised
versus generic representation, validation invariants, and the textual dump format with
versioning. Per CD-021: function-value constants and indirect-call representation. Exclusion
(roadmap): do not design an optimisation IR and executable VM bytecode simultaneously.

## Deliverable

`STARKLANG/docs/compiler/mir.md` — **STARK MIR v0.1, status PROPOSED**, drafted 2026-07-19.
Design highlights:

- **No unwinding anywhere** — traps abort, so the IR has zero cleanup edges (the structural
  simplification the frozen semantics buy).
- **Every trapping operation is a terminator** (`Checked`/`Trap` with category + source);
  the rvalue set is total. A backend cannot skip a trap check without breaking the CFG.
- **Monomorphised-only verified MIR**; instances with deterministic injective symbol naming
  (identity unobservable per TYPE-FN-001, so naming is a codegen concern only). DEV-064's
  rejection lands upstream in typecheck.
- **Drop is a statement** (no failure edge exists to model); drop flags are ordinary `Bool`
  locals + ordinary branching — no special conditional-drop instruction.
- **Indirect calls** via `FnPtr(Instance)` constants and `FnValue(operand)` callees (CD-021);
  verifier forbids arithmetic/comparison on fn values (TYPE-FN-001).
- **Closed, versioned `RuntimeFn` surface** — unsupported runtime calls fail loudly at codegen.
- **Mandatory per-statement provenance with explicit `FileId`** (the DEV-006 lesson) and
  labeled synthetic origins.
- 13 verifier obligations (V-CFG/V-TY/V-MOVE/V-DISC/V-DROP/V-IDX/V-FN/V-SRC/V-RT) mapped to
  WP-C4.3; invalid MIR = `MIR-xxxx` internal diagnostic, safe failure.

## Owner review (CE3) — outcome (2026-07-19, CD-028)

Verdict: **APPROVE WITH REQUIRED CHANGES** — three required changes, all applied:

1. Drop as a statement → **REVISED: `Drop { place, target }` terminator**, no unwind edge. The
   statement form violated the contract's own totality invariant (destructors are user code).
2. All trapping ops as terminators → **APPROVED**, with the one-normal-successor /
   implicit-abort / no-recovery-successor refinement made explicit.
3. Monomorphised-only MIR → **APPROVED for v0.1**, with three qualifications recorded:
   mangling reproducible but not a stable external ABI; named resource limit for instantiation
   explosion; deduplicated instance discovery.
4. Opaque `Option`/`Result` → **REVISED: logical MIR enums** (`EnumRef::CoreOption`/
   `CoreResult`) sharing the user-enum aggregate/discriminant/match machinery; physical layout
   stays a C5.1/ABI concern; combinators may remain runtime calls.
5. `CheckIndex`→`Index` split → **APPROVED with revision: opaque index-proof tokens**
   (`IndexProof` locals binding base+index+length) replace ordinary integer locals; `Vec`
   indexing stays on runtime operations in v0.1 (mutable length).

## Done when

- [x] `mir.md` drafted with all roadmap-required coverage, marked PROPOSED.
- [x] CE3 owner review completed; each §12 question confirmed or amended
      (**approve-with-required-changes, all three revisions applied**).
- [x] Contract status flipped to APPROVED (CD-028); WP-C4.2 opens against it.

**WP-C4.1 CLOSED 2026-07-19. Next: WP-C4.2 (typed HIR → MIR lowering, scalar core).**
