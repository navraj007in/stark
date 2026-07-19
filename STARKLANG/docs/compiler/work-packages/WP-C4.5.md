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
4. **C4.5c — generics and full static dispatch: DONE 2026-07-19.** Real
   `Instance.type_args` monomorphisation: the checker records every generic-fn use's ordered
   instantiation (`TypeTables::generic_insts`, grounded; undetermined ⇒ E0004 — DEV-064
   closed, TYPE-FN-002/TYPE-GENERIC-001), and lowering consumes it to build concrete
   `FnKey::Top(item, type_args)` instances with injective `name@[args]` symbols,
   worklist-deduplicated, capped by the named resource limit `LIMIT-MIR-MONO-INSTANCES`
   (=512; polymorphic recursion trips it deterministically, negatively tested). Generic
   bodies lower once per instantiation via `param_subst` (`Ty::Param` → concrete), giving
   operator dispatch on generic parameters per instantiation (checked-int vs IEEE-float);
   trait-bound method dispatch on `T` receivers resolves statically after substitution.
   Generic structs/enums instantiate (`MirTy` args populated; `TypeContext` keys became
   `(item, type_args)` per contract §2's nominal-instance language, entries registered by a
   reachability fixpoint over body locals). Guard: comparisons on user nominal types are
   clean-Unsupported naming C4.5e (they dispatch through user `Eq`/`Ord` impls, which the
   structural `BinOp` would silently diverge from). Differential: 6 new tests (primitives ×
   operators, generic-calling-generic, generic recursion, bound dispatch, generic nominals +
   match, monomorphised fn value — CD-021 item 21). Found & recorded DEV-067 (pre-existing
   checker over-rejection: bounded params at intra-generic call sites, `&T` bounded-method
   receivers), owner: later C4.5 increment. Methods/assoc fns *on generic nominals* and
   generic impl/trait methods are clean-Unsupported, owned by the increment that adds
   impl-level substitution. Workspace 658/0/2 (baseline re-measured: 646 at C4.5b-2, the
   recorded 640 was stale).
5. **C4.5d — ownership and Drop: DONE 2026-07-19.** Full drop elaboration over the current
   subset, oracle-timing-confirmed before implementation (probe programs pinned scope order,
   overwrite, discard, and early-exit timing empirically). Design: every droppable local
   decomposes into **drop units** — outermost sub-places that stop static decomposition (own
   `Drop` impl, enum, array) reached through dtor-less structs/tuples — each guarded by its
   own `DropFlag`; moving a place clears exactly the units it covers, which is what makes
   partial moves representable. Emission: scope exits (reverse declaration order), early
   exits (`return` all scopes, `break`/`continue` to the loop's depth) after the
   value-out move, assignment overwrite as install-then-destroy (CD-012: guarded save-old →
   store → guarded drop-old → flags true), immediate drops for discarded values and the
   `drop(x)` builtin; params/by-value receivers register flags-true at entry. Dtor instances
   (`Loud::Drop::drop@[]` — `Res::CoreTrait` trait names now render in symbols) are
   discovered into the worklist and registered in `TypeContext::drop_impls` for glue
   dispatch. MIR interp: recursive glue on `Drop` terminators — own dtor via `&mut` ref
   (mutations visible to the field destruction that follows, like the oracle), then
   fields/payload reverse order, enums by runtime discriminant; whole-local drops poison the
   slot. Verifier: V-MOVE-1 refined **field-precise** ((local, pure-Field-path) entries,
   prefix-related conflict, subtree reinit; `Drop` of possibly-moved is legal — flag-guarded
   conditional drops are that state by design), V-DROP-2 read half added (flags read only by
   `SwitchInt` `Copy`). Traps abort without drops (differentially tested). Boundary, clean
   Unsupported: match on an owned Drop-bearing scrutinee (needs the oracle's `drop_unbound`
   partial-drop — C4.5e territory with the runtime values it needs), `Drop` impls on generic
   nominals (needs generic impls). 5 differential + 2 lowering + 3 verifier tests; no new
   oracle defects (first increment where the differential matched the oracle on first run).
   Workspace 668/0/2.
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
