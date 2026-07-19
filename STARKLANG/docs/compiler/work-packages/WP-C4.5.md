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
6. **C4.5e — runtime values (implements Amendment A1, CD-031):** String/str, Vec/slices,
   Option/Result combinators, panic/assert paths, the widened `RuntimeFn` surface. Sequenced
   into sub-slices.
   - **C4.5e-1 — strings + panic/assert: DONE 2026-07-19.** A1 shape foundation
     (`MIR_RUNTIME_SURFACE`, `MirProgram.mir_version`/`runtime_surface`, `Constant::Str`,
     `Trap.message`, `TypeContext.copy_types`, String/str `RuntimeFn` group, dump header +
     `const "…"` escaping). Lowering: string literals → `Constant::Str`; `String::new`/`from`;
     String/str method dispatch (`as_str`/`len`/`is_empty`/`push_str`/`clear`/`clone`/
     `contains`, str `len`/`is_empty`/`to_string`); `&str`/`String` printing; String/str
     comparison → `StrEq`/`StrCmp` (V-STR-2 keeps structural BinOp off strings); `panic(msg)`
     → `Trap{Panic, Some(msg)}`, `assert(cond)` → conditional `Trap{AssertFailure}`; String is
     a droppable **leaf unit** (buffer reclaim, unobservable); user `as` casts (were never
     lowered — needed by `s.len() as Int32`). Verifier: surface gate (MIR-0017), string
     `runtime_sig`, V-STR-1/2 (MIR-0015), Trap.message threaded through move/proof/drop-flag/
     typing analyses. Interp: `MirValue::Str`/`String`, string runtime ops (`&mut String`
     mutated in place through the reference; `as_str` returns a read-only snapshot per §5b),
     `Trap.message` resolved into `MirRunError::Trap.message` and compared by the differential
     (panic messages compared exactly, compiler traps by category fragment). **The two frozen
     `ownership_drop__*` corpus cases are now differential-green** — the first String-dependent
     corpus cases to pass. 6 differential + 3 verifier + 1 dump test. Deferred to later e
     sub-slices: Char + Char-dependent String ops (`push`/`pop`, Print(ln)Char), `assert_eq`/
     `assert_ne` formatting. Workspace 684/0/2.
   - **C4.5e-2 — Vec data surface + Vec drop: DONE 2026-07-19.** `RuntimeFn` Vec group
     (`VecNew`/`WithCapacity`/`Push`/`Pop`/`Len`/`IsEmpty`/`IndexGet`/`Replace`/`Remove`/
     `Clear`), `MirValue::Vec`. Lowering: `Vec::new`/`with_capacity`; Vec method dispatch
     (`push`/`pop`/`remove`/`clear`/`len`/`is_empty`); `v[i]` read → `VecIndexGet` (Copy T);
     `v[i] = x` → `VecReplace` + drop-old (install-then-destroy, CD-012); `clear()` on a
     droppable element → explicit pop-and-drop loop (A1 §5a — no `RuntimeFn` runs a user
     destructor); `Vec<T>` is a droppable leaf unit. Verifier: schematic-T `runtime_sig`
     (constructor from dest element, methods from first `&Vec` operand), V-COPY-1 (MIR-0016):
     `VecIndexGet` requires Copy T, `VecClear` requires non-droppable T; `copy_types` populated;
     precise `mir_needs_drop` mirrors lowering's `ty_needs_drop`. Interp: Vec ops mutate `&mut
     Vec` in place through the reference; index/replace/remove trap `IndexOutOfBounds` with the
     **call site's** provenance; **Vec drop drops elements in reverse index order** (matched to
     the frozen oracle) then reclaims. 4 differential + 2 verifier tests. **Deferred — flagged
     for owner (A1 gap):** `.iter()` iteration. STARK's `.iter()` binds `value: &T`
     (by-reference), which A1 lowered as by-value `VecIterNext -> Option<T>` and separately
     reserved by-reference iteration to a C4.5f-dependent sub-slice. `collection_iter__01`'s
     `for value in values.iter()` therefore stays clean-Unsupported; its push/index/len half
     lowers. Resolving whether/how A1 adds by-reference Vec iteration (interior references into
     runtime containers) needs an owner-reviewed surface bump (`0.1-A2`). Workspace 691/0/2.
   - **C4.5e-3 — `?` operator + Option/Result methods: DONE 2026-07-19.** `ExprKind::Try`
     lowering: operand into a temp (not scope-registered — both switch arms consume it, so no
     drop owed), `Discriminant` + `SwitchInt`; Ok/Some payload becomes the expression value,
     None/Err propagates as an early return of the enclosing fn's own Option/Result (drops live
     scopes first). Option/Result methods `is_some`/`is_none`/`is_ok`/`is_err` (discriminant
     compare) and `unwrap` (`SwitchInt`; wrong variant → `Trap{UnwrapNone|UnwrapErr}`). User
     `as` cast fix carried from e-1. **`option_result__01` corpus case now differential-green**;
     4 new differential tests (`?` on Option+Result, chained propagation, unwrap-None trap).
     **`option_result__02` stays blocked** on the match-drop increment below (its `?` lowers).
     Deferred with e-1: Char + Char String ops, `assert_eq`/`assert_ne`. Workspace 695/0/2.
   - **Match-drop increment — match on owned Drop-bearing scrutinees: DONE 2026-07-19.** Oracle
     drop timing pinned empirically first: the matched arm consumes the scrutinee, and every
     payload — bound, unbound (`_`), or catch-all-bound — drops at **arm end** (after the arm
     body). Lowering rewrite (`lower_enum_match`): each arm is a drop scope; every payload
     field is moved out of the (always-materialized temp) scrutinee — bound fields into
     registered binding locals, unbound droppable fields into registered temps, an
     unmentioned struct field likewise, a catch-all binding/`_` handling the whole value — so
     the scrutinee shell is fully consumed (no double-drop) and everything drops at arm-scope
     exit. A binding moved by the arm body clears its flag, so only the callee's drop fires
     (tested). The blanket C4.5d restriction is removed. **`option_result__02` corpus case now
     differential-green** — the last runtime-values corpus case reachable without interior
     references. 4 new differential tests (bound/Wild/catch-all payload drop, move-no-double-
     drop). Workspace 698/0/2.
   Also owns (accumulated boundaries): user-nominal
   `Eq`/`Ord` operator dispatch (needs `Ordering` runtime values), projected-move
   take-and-poison in the MIR interp (CD-030 deferral), and the CE3-shaped mir.md §5/§7
   amendment for string-literal/String value representation, which must land before lowering
   starts.
   - **C4.5e-0 — pre-runtime-values hardening: DONE 2026-07-19 (CD-030,** disposition of the
     external C4.5c-head review). IndexProof definite-initialization dataflow: must-analysis
     (intersection joins, unique-definition rule) so every `Index(proof)` is definitely
     preceded by its `CheckIndex` on all paths — the prior global name→base map accepted
     one-branch checks; 4 adversarial negatives. V-REF-1 (MIR-0014): writes crossing a
     `Deref` require that layer `&mut` (write-path place typing; Call dests and Drop places
     included). Pre-trap stdout is now observable and compared: oracle
     `run_with_partial_output`, MIR `MirFailure { error, output }`, differential trap arm
     asserts prefix equality (drop-output-before-trap regression). DEV-068 found via the
     review's warning and fixed: user `impl Copy` structs were always-Move, which the
     field-precise verifier rejected as use-after-move on valid programs. Deferred per
     CD-030: frame-generation identities (owner C4.5f), projected-move take-and-poison
     (owner C4.5e proper). Workspace 675/0/2.
7. **C4.5f — multi-package lowering + interior references + Vec/String iteration:**
   package-stable canonical symbols (`⟨package⟩::⟨module⟩::…`), cross-package instances and
   linkage. Also owns:
   - **C4.5f-1 — frame generations + projected-move poison: DONE 2026-07-19** (both CD-030
     deferrals). `Frame.generation` (monotonic, never reused) + `MirValue::Ref` carries the
     pointee's generation; every deref and every `&String`/`&mut String`/`&Vec`/`&mut Vec`
     runtime-op helper validates (slot, generation) — a stale reference to a reused frame slot
     now fails loudly instead of silently aliasing (adversarial hand-built MIR test: verifier
     passes it by design, the interpreter rejects with the generation mismatch). Projected
     `Move`s now TAKE, leaving a `MirValue::Moved` poison; any later read of the hole is a
     loud internal error (previously projected moves cloned, relying on the verifier alone);
     the full suite staying green with the poison live confirms nothing in the tested subset
     re-reads a moved place. Workspace 699/0/2.
   - (CD-030 deferral — resolved by f-1 above) frame-generation identities in the MIR
     interpreter, before cross-package call graphs grow the frame traffic.
   - (CD-032) **Vec/collection iteration**, folded here from A1. STARK's `.iter()` is
     by-reference (`value: &T`), i.e. an interior reference into a runtime container — the same
     machinery as the reserved `VecGetRef`/`StringSubstring` views. Design carry-forward is in
     the amendment §5e (interior-borrow cursor; snapshot-for-Copy interpreter representation;
     non-Copy drop-elaborated iterator). Activated by a dated `0.1-A2` runtime-surface bump
     (`VecIterNew`/`VecIterNext` in a by-reference `Option<&T>` form). Unblocks
     `collection_iter__01`'s `for value in values.iter()` (its push/index/len half already
     lowers under e-2).

## Differential-independence caveat (recorded per CD-029)

"Zero semantic differences" between HIR and MIR means: **no difference in lowering and MIR
execution for the tested subset, with some runtime algorithms intentionally shared** (float
formatting via `interp::canonical_float`). Shared algorithms are covered by their own
spec-derived tests (`tests/canonical_float.rs`), not by the differential.

## Exit

WP-C4.5 completes when every normative Core construct required by C5 has verified MIR lowering
and the full frozen corpus (`corpus_version` current) runs equivalently through both
interpreters — feeding directly into the C4.6 gate exit.
