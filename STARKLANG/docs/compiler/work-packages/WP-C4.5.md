# WP-C4.5 — Complete Core Lowering (increment plan)

Gate: C4. Scope from `COMPILER-ROADMAP.md` WP-C4.5. Split into bounded increments per charter
§2.2 (its acceptance surface is far too large for one change); order adopted from the external
review of the C4.1–C4.4 foundation (CD-029), which also calibrated honest maturity:
architecture ~90%, implementation breadth ~35–45%, validation infrastructure ~70% — **most of
C4's engineering effort lives in this WP.**

Every increment is differential-first: each construct lands with lowering + verifier coverage +
HIR/MIR differential agreement before the next begins.

## Increments

1. **C4.5-contract-cleanup — DONE 2026-07-19 (CD-029).** TypeContext formally amended into the
   MIR program shape (mir.md §2, still v0.1, additive); trap outcomes carry full `TrapInfo`
   (category + `SourceInfo`) with the differential comparing provenance (user-origin spans must
   equal the oracle's; synthetic origins compare classification); `VerifiedMirProgram` wrapper —
   `run_program` (and eventually the generated-Rust backend) can only consume proof-of-
   verification, making "no backend bypasses MIR validation" an API property; canonical_float
   spec-derived golden/property tests (the shared formatter is invisible to the differential —
   these are the compensating control).
2. **C4.5a (methods/dispatch) — DONE 2026-07-19** (landed before this ordering was adopted;
   stands): methods, associated fns, trait dispatch with defaults, `FnKey` instances, Self
   substitution; interim by-value reference model documented; `&mut self` deferred to the
   references increment.
3. **C4.5b — indexing and references: DONE 2026-07-19** in two parts. b-1: `CheckIndex` proof
   tokens for arrays (same-base verifier binding; OOB differential w/ provenance; DEV-065
   oracle-message fix). b-2: real reference places — frame-stack MIR interpreter with
   references as (frame, local, concrete-path), `RefOf`/`Deref` lowering with method
   auto-deref, real `&self`/`&mut self` receivers (interim by-value model removed), verified
   cross-frame mutation; DEV-066 (borrowck moved a reference on deref-read, rejecting
   `*r = *r + 1`) found by the differential and fixed. **Slices deferred to C4.5e** where
   their consumers (String/Vec views) live — nothing in the current workload reaches them.
4. **C4.5c — generics and full static dispatch:** real `Instance.type_args` monomorphisation
   with deterministic deduplication and the named resource limit; generic fns/types; operator
   dispatch on generic parameters; DEV-064's rejection in typecheck.
5. **C4.5d — ownership and Drop:** field-sensitive initialization, partial moves, drop flags,
   `Drop` terminator elaboration (reverse declaration order, exactly once), verifier's
   field-precise move refinement.
6. **C4.5e — runtime values:** String/str, Vec/slices, Option/Result combinators, panic/assert
   paths, the widened `RuntimeFn` surface.
7. **C4.5f — multi-package lowering:** package-stable canonical symbols
   (`⟨package⟩::⟨module⟩::…`), cross-package instances and linkage.

## Differential-independence caveat (recorded per CD-029)

"Zero semantic differences" between HIR and MIR means: **no difference in lowering and MIR
execution for the tested subset, with some runtime algorithms intentionally shared** (float
formatting via `interp::canonical_float`). Shared algorithms are covered by their own
spec-derived tests (`tests/canonical_float.rs`), not by the differential.

## Exit

WP-C4.5 completes when every normative Core construct required by C5 has verified MIR lowering
and the full frozen corpus (`corpus_version` current) runs equivalently through both
interpreters — feeding directly into the C4.6 gate exit.
