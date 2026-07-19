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

## Owner review (CE3) — five flagged judgment calls (mir.md §12)

1. Drop as a statement (divergence from Rust-MIR shape; justified by abort-no-unwind).
2. All trapping ops as terminators (explicitness vs block count).
3. Monomorphised-only MIR + whole-program compilation assumption.
4. `Option`/`Result` as opaque Core runtime types in v0.1 (not ordinary MIR enums).
5. Split `CheckIndex`-dominates-`Index` discipline (enables later bounds-check elimination).

## Done when

- [x] `mir.md` drafted with all roadmap-required coverage, marked PROPOSED.
- [ ] CE3 owner review completed; each §12 question confirmed or amended.
- [ ] Contract status flipped to APPROVED (with a CD entry); WP-C4.2 opens against it.
