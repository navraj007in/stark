# WP-C4.7 — C4 Correction + Re-audit Closure Package (executor plan)

Gate: C4 (stays OPEN until this WP completes and the owner approves the fresh exit report).
Origin: the WP-C4.6 Class-A campaign completed all seven classes (see `WP-C4.6.md`), but an
external review plus self-audit identified corrections required before an honest exit. This
document is the **complete execution plan**, written so any session can pick up any increment
without prior context. Read §0 and §1 before touching code.

---

## 0. Rules of engagement (do not skip)

1. **One increment per go-ahead.** The owner approves work increment-by-increment. Do ONE
   numbered increment (C4.7-1, C4.7-2, …), commit it, push, verify CI, report, STOP — unless
   the owner explicitly says to continue or "finish all".
2. **Validation gate for every increment** (all from `starkc/`, NOT the repo root — `cargo fmt`
   fails at the root):
   ```bash
   cd starkc
   cargo fmt
   cargo test --workspace          # must be zero failures
   cargo clippy --workspace --all-targets                      # 0 warnings
   rustup run 1.97.0 cargo clippy --workspace --all-targets    # 0 warnings (CI parity)
   ```
3. **Commit per increment**, push, then verify CI:
   ```bash
   gh run list --limit 1 --json status,conclusion --jq '.[0]'
   ```
   Poll until `completed`/`success`. Commit messages end with the Co-Authored-By line the
   session harness specifies.
4. **Docs move with code, same commit**: update `COMPILER-STATE.md` (Position header + a
   dated record), this file's increment status, `KNOWN-DEVIATIONS.md` (entry + tail count +
   the "Currently open" enumeration — there are TWO lists; update BOTH), and any amendment
   doc touched. Then run the cross-reference sweep: grep every identifier you changed across
   `COMPILER-STATE.md`, `WP-C4.6.md`, `WP-C4.7.md`, `KNOWN-DEVIATIONS.md`,
   `STARKLANG/docs/compiler/mir.md`, and the amendment docs; fix every stale count/claim in
   the same commit.
5. **Escalation boundary (CE rule).** These require OWNER approval before implementation —
   draft the decision document, present it, STOP:
   - Any new `MirTy`/`Rvalue`/`Terminator`/`Constant`/`CheckedOp`/`TrapCategory`/`EnumRef`
     variant or any other MIR **shape** change (the C4-open policy in `mir.md` requires
     per-amendment CE3 approval + recording in `mir.md`). Increment C4.7-3 hits this.
   - Any change to the `MIR_RUNTIME_SURFACE` version or new `RuntimeFn` (dated enumeration
     revision in `mir-amendment-A1-strings-runtime.md` §11 — follow the rev. 5–10 format).
   - Any change to normative spec files under `STARKLANG/docs/spec/`.
   - Any behavior change to the HIR oracle beyond a bug-fix that a DEV entry documents.
6. **The oracle is the semantics authority.** Before implementing any behavior with
   observable ordering (drops, evaluation order), pin the oracle's behavior EMPIRICALLY first
   with probe programs, then implement MIR to match. Never guess.
7. **Never edit `STARKLANG/docs/spec/STARK-Core-v1.md`** (generated). Edit the numbered
   source docs and regenerate with `python3 STARKLANG/tools/build-core-spec.py` — only if an
   increment requires spec edits (C4.7-3 may, with owner approval).

## 1. Repo execution knowledge (learned the hard way — read before coding)

### Probe harnesses (both committed)
- `starkc/examples/c46_probe.rs` — full pipeline on a file:
  `cargo run --example c46_probe -- path.stark` prints one of `PARSE-ERR` / `RESOLVE-ERR` /
  `TYPECHECK-ERR` (front-end scope) / `LOWER-UNSUPPORTED` (MIR gap) / `VERIFY-ERR` /
  `MIR-RUN-ERR` / `OK: ran, stdout=…`.
- `starkc/examples/oracle_run.rs` — HIR oracle only: `ORACLE-OK`/`ORACLE-ERR`.
- Use both on every new construct: a case must be **typecheck-clean AND oracle-supported**
  before its MIR gap counts as MIR work. Front-end failures are front-end increments.

### The differential harness (`starkc/tests/mir_differential.rs`)
- `differential(name, source)` runs oracle vs lower→verify→MIR-run and compares: stdout,
  exit status, trap category (via `oracle_fragment` message-fragment mapping at the top of
  the file — extend it if you add a `TrapCategory`), **exact user-origin trap provenance
  spans**, and pre-trap stdout prefix.
- Provenance gotcha: the oracle attributes traps to the INNER expression span (e.g.
  `a[1..9]`, not `&a[1..9]`). If a provenance assert fails ±1 byte, your lowering used the
  wrong expr's span.
- Trap-message rule: compiler traps carry `message: None` and compare by category fragment;
  only `panic(msg)` compares messages exactly.

### Front-end quirks to ROUTE AROUND in test programs (do not fight them)
- ~~Integer literals don't coerce to `UInt64` params~~ — **fixed by C4.7-6.3 (DEV-078)**: an
  unsuffixed literal adopts an expected integer type, so `v.get(0)` and `i = i + 1` are correct.
  A suffixed literal and a typed value still do not convert.
- ~~All-three-variant `Ordering` matches are wrongly non-exhaustive (DEV-071)~~ — **fixed by
  C4.7-7**; write all three variants, no wildcard needed.
- ~~Generic impls don't satisfy operator/iterable bounds (DEV-073)~~ — **fixed by C4.7-5**;
  `a == b` on `W<Int32>` with `impl<T> Eq for W<T>` and `for x in it` on a generic iterator
  struct both work now.
- ~~Binding a NON-Copy payload through `&self` is front-end-accepted-but-MIR-rejected
  (DEV-072)~~ — **fixed by C4.7-5**: both engines now reject it (E0101). Wildcards and `Copy`
  payload bindings under a by-reference scrutinee remain legal, as they always were.

### Lowering machinery map (`starkc/src/mir/lower.rs`, ~7k lines)
- `FnKey::{Top, ImplFn, TraitDefault}` all carry concrete type args; bodies lower once per
  key via a worklist (`discovered_callees`); `key_symbol` names instances
  (`Stack::push_item@[Int32]`). `param_subst: HashMap<String, MirTy>` is the active generic
  substitution; `impl_generic_subst(impl_item, type_args)` derives it for impls by aligning
  the impl's WRITTEN self-type args (bare param names only) with the instantiation.
- Block emission: `self.emit(stmt, info)` appends to the current block;
  `self.terminate(term, info, next)` SEALS the current block and makes `next` current. Every
  allocated block must be sealed or lowering panics "every allocated block must be sealed".
  The `lower_arm_body_scoped` pattern allocates a dead block after a `Goto` then
  `self.blocks.pop()`s it.
- Places: `place_or_temp(expr, ty, span)` materializes non-place expressions (call results)
  — use it for any receiver/borrow of an arbitrary expression. `read_place` picks
  `Operand::Copy` vs `Move` by `is_copy`.
- Match machinery: `lower_match` decides `MatchMode::{Consuming, ByRef}`
  (`scrutinee_reads_through_ref`); FLAT enum arms go to the drop-elaborated
  `lower_enum_match` (C4.5d semantics); scalars to `lower_int_match`; everything else to the
  recursive `lower_general_match` (`emit_pattern_test` + `bind_pattern`), which currently
  REQUIRES a drop-free scrutinee in Consuming mode.
- Drop elaboration: `register_droppable_local`, `ty_needs_drop`, `collect_drop_units`,
  `discover_drop_impls` (dtor instances keyed `(item, args)` in
  `TypeContext::drop_impls`). Runtime types (String/Vec/HashMap/iterators) are leaf units.
- Runtime surface: `RuntimeFn` in `mod.rs`; verifier signatures in `verify.rs` —
  fixed table `runtime_sig` + schematic `vec_runtime_sig`/`map_runtime_sig`/
  `slice_runtime_sig`; interp dispatch `run_runtime` → `run_{string,vec,map,slice}_runtime`.
  Adding a RuntimeFn touches ALL FOUR places + `is_*_runtime(_fn)` guards + a surface bump.
- Surface bumps change the dump header: update the two golden strings in
  `tests/mir_lowering.rs` (`// STARK MIR v0.1 (runtime-surface 0.1-AX)` and the
  `dump.contains(...)` assert).

### Editing discipline
- Prefer exact-string Edit calls. For python patch scripts: assert `count(old) == 1` per
  replacement and write the file only at the end (a failed assert then leaves the file
  untouched). `cargo fmt` reflows aggressively — re-read actual text after fmt before
  further patching.

---

## 2. Increments

Status legend: each increment gets `_pending_` → `DONE <date>` edited into this section.

### C4.7-1 — Documentation/evidence reconciliation — DONE 2026-07-20 (partially, see below)

DONE in the planning commit: `COMPILER-STATE.md` stale A3-era position block neutralized
(the "A4/A2/A1 remain / DEV-070 open / Blocked:" paragraph); `KNOWN-DEVIATIONS.md`
"Currently open" enumeration refreshed to post-Class-A reality with C4.7 owners.

REMAINING (finish as the first coding increment):
1. **Record A5's shape additions in `mir.md`** (policy: every additive shape amendment must
   be recorded in the contract). Add, next to the existing A1/A2 amendment notes, an "A3
   shape amendment (WP-C4.6 A5, CD-033)" paragraph covering: `MirBinOp::BitAnd/BitOr/BitXor`
   (pure, integer-only, result-typed as operands), `CheckedOp::Pow` (NUM-INT-ARITH-001
   nonnegative exponent, checked intermediate multiplies), `CheckedOp::Shl/Shr` semantics now
   ACTIVE (NUM-SHIFT-001 count bound), and `TrapCategory::InvalidShift` (distinct from
   IntegerOverflow; category-override via the interpreter's `CheckedOutcome`). Present the
   paragraph to the owner in the increment report for post-hoc CE3 ratification (CD-033
   approved the A5 *class*; the per-amendment recording was missed).
2. **DEV-074**: number the oracle slice-bound message alignment (three messages folded into
   the "out of bounds" family during A4-2e, documented only in amendment A1 rev. 10). Ledger
   entry: what/why (spec groups all slice-bound failures as one trap; the differential
   fragment comparator requires it), CLOSED-at-creation status, pointer to rev. 10. Bump the
   tail count 71 → 72 and add to BOTH open/closed enumerations appropriately.
3. **Tighten the A4 wording** in `WP-C4.6.md`: where it says the A4 surface "completes the
   core-min runtime surface", add "(the MIR runtime surface; front-end core-min holes —
   `Box` deref, primitive `cmp` — are C4.7-6)".
Validation: doc-only; run the §0.4 sweep; no test changes. Single commit.

### C4.7-2 — Evidence symmetry: verifier negatives + unsupported fixtures  _pending_

Goal: every Class-A class has hand-built verifier negatives, and every recorded residual is
pinned by a clean-Unsupported fixture (per CD-033's evidence rule).

**A. Verifier negatives** — add to `starkc/tests/mir_verify.rs`, modeled on
`rejects_invalid_core_ordering_variant` / `rejects_slice_new_with_bad_dest_type` (hand-built
`MirBody` via the file's `body`/`block`/`local`/`expect_code` helpers):
1. `rejects_bitwise_binop_on_floats` — `Rvalue::BinOp(MirBinOp::BitAnd, Float64, Float64)`
   → expect `MIR-0004` ("bitwise BinOp on").
2. `rejects_pow_on_non_integer_dest` — `CheckedOp::Pow` with a Float64 dest → `MIR-0004`.
3. `rejects_vec_get_ref_with_wrong_dest` — `RuntimeFn::VecGetRef` on `&Vec<Int32>` with dest
   typed `Option<&Bool>` → `MIR-0005` (arg/sig mismatch path in `vec_runtime_sig`).
4. `rejects_chars_iter_next_on_non_iterator` — `CharsIterNext` with an `Int32` operand →
   `MIR-0005` (fixed-table sig).
5. `rejects_switch_on_float` — `SwitchInt` scrutinee `Float64` → `MIR-0004` (guards the A2
   Char-scrutinee widening from over-widening).
6. `rejects_impl_symbol_mismatch` is NOT needed (symbols aren't verifier-checked); instead:
   `rejects_call_arity_against_instance` already exists — verify; if not, skip with a note.

**B. Unsupported fixtures** — extend `unsupported_constructs_report_cleanly` in
`starkc/tests/mir_lowering.rs` (each entry: name, source, expected message NEEDLE — keep
needles loose, one distinctive word):
1. Droppable scrutinee + nested pattern: `match Some((String::from("x"), 1)) { Some((s, n)) => … }`
   → needle `"A2 residual"`.
2. Method-own generics: `impl Holder { fn map<U>(…) }` … careful: needs a call site; use a
   non-generic nominal with `fn conv<U: …>`-shape? If the front end rejects independently,
   drop this fixture and note it. Probe FIRST with c46_probe.
3. Non-bare impl self args: `impl<T> Wrap for Holder<Vec<T>>`-shape — probe first; front end
   may reject; if it reaches lowering → needle `"non-parameter self argument"`.
4. Droppable Iterator Item: Iterator impl with `next() -> Option<String>` + `for` →
   needle `"droppable Item"`.
5. Mutable slice view: `&mut a[0..2]` → needle `"mutable slice"`.
6. `unwrap_or` on droppable payload: `Option<String>::unwrap_or(String::new())` →
   needle `"droppable payload"`.
For every fixture: run c46_probe FIRST; if the front end rejects the program, the fixture
can't live in mir_lowering — record which in the increment report instead of forcing it.
Validation gate + docs + commit.

### C4.7-3 — Type-preserving layout queries (`size_of`/`align_of`)  _pending_  ⚠ CE3

Problem: 06-Standard-Library classifies them as **"target-layout queries"** ("C2.9 completes
the target results"), but A4-1 lowered both to `Const 8` with the queried type ERASED, and
the HIR oracle also returns 8 for every type (`src/interp.rs`, `Builtin::SizeOf | AlignOf`).
The differential passes only because both share the placeholder. A C5 backend cannot answer a
target-layout query from a MIR that discarded `T`.

Steps:
1. **Research first**: find what C2.9 actually decided about target results — search
   `COMPILER-STATE.md` and `STARKLANG/docs/spec/` for `C2.9` + `size_of`. Report findings.
2. **Draft a CE3 amendment** (new file `STARKLANG/docs/compiler/mir-amendment-A4-layout.md` —
   **renumbered from A3**, because C4.7-1 recorded the WP-C4.6 A5 arithmetic work as MIR
   amendment A3; modeled on `mir-amendment-A2-ordering.md`): recommended design —
   `Rvalue::LayoutQuery { kind: SizeOf | AlignOf, ty: MirTy }`, typed `UInt64`, evaluated by
   consumers via a layout service; the C4 reference interpreter's service returns the frozen
   reference-target answer (8 today — behavior unchanged, representation fixed); verifier
   checks dest ty `UInt64`; dump as `layout_size_of(<ty>)` / `layout_align_of(<ty>)`.
   Include: why a RuntimeFn is NOT suitable (erases T unless made schematic in a way the
   surface tables don't support), backend note (C5.1 substitutes its real layout), and the
   §7 dump grammar addition. **STOP and present for owner approval before any code.**
3. After approval: implement in `mod.rs` (Rvalue + dump), `lower.rs` (the
   `Res::Builtin(SizeOf|AlignOf)` arm in `lower_call` — currently emits `Const 8`; keep the
   `expr_mir_ty`-independent `UInt64` dest), `verify.rs` (Rvalue typing arm), `interp.rs`
   (`eval_rvalue` arm returning the reference answer via one `fn reference_layout(ty) -> (u64, u64)`
   so C5 has a single override point). Oracle: LEAVE returning 8 (same value) — no oracle
   change needed. Differential test: `size_of_align_of_agree` already exists; add a dump
   golden showing the type is preserved.
Validation gate + docs (mir.md amendment note per policy) + commit.

### C4.7-4 — DEV-069: front-end multi-file span discipline  _pending_  (largest; C5 prerequisite)

Problem (full detail in the DEV-069 ledger entry): the type checker and HIR oracle resolve
`Span`s against the ENTRY file only. Cross-file methods mis-resolve, cross-file literals
mis-parse, cross-file field reads fail, and `TypeChecker::text` can panic out-of-bounds. MIR
lowering is already multi-file-clean via `ProgramMeta` (each item knows its declaring file;
`meta.item_text(item, span)` reads in the owning file) — use it as the model.

Approach (front end):
1. Reproduce first: `starkc/tests/mir_differential.rs::multi_file_module_program_agrees_with_qualified_symbols`
   is pinned to the front-end-safe subset; write throwaway two-file probes (tempdir, like
   that test does) for each failure shape: cross-file method call, cross-file literal,
   cross-file field read.
2. The resolver already tracks which file each item came from (ProgramMeta::build derives
   it — study `lower.rs` `ProgramMeta` + `hir.rs` `synthetic_spans`). Thread the same
   item→file mapping through `typecheck.rs`, `borrowck.rs`, and `interp.rs`: every
   `self.text(span)` / literal-parse / field-name read must resolve the span against the
   file OWNING the item being read, not `self.file`. Grep for `.text(` and `src[` uses in
   those three files; classify each read by "whose span is this" before changing anything.
3. Expect this to be wide but mechanical. Do it in TWO commits: (a) typecheck+borrowck,
   (b) oracle — each with the probes re-run.
4. Then WIDEN the multi-file differential test past the safe subset (cross-file struct +
   Drop impl + methods + literals — the original pre-narrowing content is described in the
   test's NOTE comment) and mark DEV-069 CLOSED (ledger + both enumerations + count).
Validation gate + docs + commit(s).

### C4.7-5 — DEV-072 + DEV-073 (front-end typecheck/borrowck)  _pending_

**DEV-072** (borrowck): reject moving a non-Copy value out of a shared borrow via match
bindings. Repro in the ledger entry. Where: `borrowck.rs` — pattern-binding handling must
treat a binding of a non-Copy payload whose scrutinee place crosses a shared `Deref` as a
move-out-of-borrow error (there is existing move-out-of-borrow machinery for direct
expressions; extend it to match bindings). After the fix: the MIR guard message
("binding a non-Copy payload through a shared reference") becomes unreachable-by-checked-
programs — KEEP the guard (defense in depth) but update its comment. Add front-end negative
tests (existing borrowck test file conventions) + keep
`match_deref_self_noncopy_wildcard_agree` green (wildcards must STAY legal).

**DEV-073** (typecheck): make generic impls satisfy operator-trait/iterable bounds.
Where: `typecheck.rs` `require_operator_bound` and the for-loop iterable check — both
currently match impls only for exact non-generic self types. Extend the impl search to unify
the impl's written self type (bare-param generic head) with the concrete nominal
instantiation (the method-resolution path already does this — find and reuse its matching,
don't reinvent). After the fix, ADD the two differential tests that were removed for this
reason: user Eq on `W<Int32>` via `impl<T> Eq for W<T>` (see WP-C4.6 A1 section), and a
generic user-iterator for-loop (`Repeat<T>`). MIR side is already instantiation-ready — no
lowering changes expected; if lowering breaks, that's a real finding, report it.
Close both DEVs in the ledger. Validation gate + docs + commit.

### C4.7-6 — Front-end `core-min` completions  _pending_

1. **`Box<T>` deref**: `*Box::new(5)` fails E0001 "cannot dereference non-reference". Spec:
   06 defines `Box::new`/`into_inner` (implementation-provided Core type). Scope: typecheck
   deref of `Ty::Core(Box, [T])` → `T`; oracle deref/auto-deref of the Box value; MIR: probe
   what's needed (likely `Ty::Core(Box,_)` → a MirTy representation + deref semantics —
   if a new MirTy/runtime-op is needed, that's a CE3/surface stop per §0.5; report before
   implementing the MIR half).
2. **Primitive `cmp`/`Ordering` surface**: `3.cmp(&5)` fails E0304. Add `cmp` to the
   checker's primitive-method surface returning `Core(Ordering)`; oracle: evaluate via the
   existing `Value::Ordering`; MIR: lower primitive `.cmp` to the existing comparison +
   `CoreOrdering` construction (no new ops — a small lowering arm; pattern after
   `lower_user_ord`'s discriminant machinery in reverse).
3. **`Vec::get` literal typing**: `v.get(0)` fails (UInt64 vs Int32 literal). Root cause:
   integer-literal defaulting against an expected `UInt64` param — find where method-arg
   unification runs vs literal defaulting in `typecheck.rs`; the fix should make an
   un-suffixed integer literal adopt the expected primitive integer type (this likely also
   fixes `i + 1` against UInt64 and array-index ergonomics — run the full suite to see what
   it unlocks; remove now-unneeded `as UInt64` casts from tests ONLY if the owner agrees the
   semantics change is spec-clean — check 03's literal-inference rules first; if 03 forbids
   it, record as spec-conformant behavior instead and close the item as "not a bug").
Each with front-end tests + differential where MIR is touched. Validation gate + docs + commit.

### C4.7-7 — DEV-067 + DEV-071  _pending_

**DEV-071** (exhaustiveness): register the prelude `Ordering` variant set
(`Less/Equal/Greater`) with the checker's usefulness/exhaustiveness algorithm so an
all-three-variant match is exhaustive. Then update
`ordering_value_round_trips_through_match_agree` to use three explicit arms (drop the
wildcard workaround) and remove the §1 quirk note here and in the harness comments.

**DEV-067** (bounded generics): bounds lost at intra-generic call sites (E0500) and behind
`&T` receivers (E0302). Repro: ledger entry + probe
`fn call_speak<T: Speak>(t: &T) -> Int32 { t.speak() }`. Where: `typecheck.rs` — when the
receiver type resolves to `Ty::Param(name)` (or `&Param`), method lookup must consult the
param's declared bounds (trait list) instead of failing; operator bounds similarly at
intra-generic call sites. After the fix, add differential coverage: a generic fn calling a
bound method through `&T`, instantiated twice. Close both DEVs. Validation gate + docs + commit.

### C4.7-8 — Remaining normative MIR residuals  _pending_  (split into 8a/8b as needed)

Order within the increment (each independently commit-able):
1. **Droppable Option/Result `unwrap_or`/combinators**: the discarded branch's value needs a
   drop. For `unwrap_or`: in the Some/Ok arm the DEFAULT is unused → move it to a
   drop-registered temp in that arm; in the None/Err arm the default is consumed. Follow the
   arm-scoped pattern of `lower_enum_match` (scopes.push / register / emit_scope_drops).
   Same treatment for `map`/`and_then`/`map_err` payload+fn cases (the fn value is Copy;
   only payloads matter). Differential tests with `Drop`-impl payloads printing from dtors —
   PIN THE ORACLE'S DROP TIMING FIRST (§0.6).
2. **Droppable Iterator Item**: per-iteration scope around the loop-var binding in
   `lower_for_over_user_iter` (scopes.push before the bind, register the binding local
   flags-true, emit_scope_drops at the latch AND at break/exit paths — study how
   `lower_arm_body_scoped` + `LoopTargets.scope_depth` interact; `break` already drops
   scopes deeper than the loop's `scope_depth`, so pushing the scope AFTER `loops.push`
   gives correct break behavior). Oracle-pin first.
3. **Droppable scrutinee + nested patterns**: generalize consumption in
   `lower_general_match`. Design: in Consuming mode over a droppable scrutinee, after tests
   pass, recursively move EVERY droppable leaf of the scrutinee that the pattern reaches:
   bound leaves → registered binding locals (flags true); tested-but-unbound droppable
   sub-places (wildcards, unmentioned struct fields, enum shells whose payload was extracted)
   → registered temps, mirroring `consume_variant_payload`'s flat rules recursively. The
   scrutinee temp itself is never auto-dropped (it was fully decomposed). Pin oracle timing
   for a two-level case FIRST (`Some((TagA, TagB))` with printing dtors, matched by
   `Some((a, _))` — which dtor fires when?). This is the hardest piece; budget a session.
4. **Method-own generic parameters** (`fn map<U>`): **C4.7-2 finding — this is front-end-first.**
   `impl Holder { fn first<U>(&self, a: U, b: U) -> U }` called as `h.first(7, 9)` fails
   typecheck with E0001 "expected 'U', found 'Int32'": method-own params are not substituted at
   the call site at all, so nothing reaches lowering. Fix the checker first (per-call-site
   instantiation recording for METHODS), then the MIR half below. Original MIR plan: needs
   checker-recorded per-call-site
   instantiations for METHODS (the C4.5c `generic_insts` machinery records top-level fn
   uses; extend recording to method calls, keyed by the method-call expr), then a
   `FnKey::ImplFn` extension carrying method-level args alongside impl-level args — this is
   a FnKey shape change: check whether `mir.md` describes FnKey (it's lowering-internal;
   if not contract-visible, no CE3 needed — verify, state your conclusion in the report).
5. **Non-bare impl heads** (`impl<T> Holder<Vec<T>>`): **C4.7-2 finding — also front-end-first.**
   `impl<T> Wrap for Holder<Vec<T>>` + `h.wrapped()` on `Holder<Vec<Int32>>` fails typecheck with
   E0302 "method 'wrapped' not found", so method resolution does not structurally unify non-bare
   impl heads either — despite handling bare-param generic impls (DEV-073's entry notes the
   method path works for those). Fix resolution first, then the MIR half: extend
   `impl_generic_subst` to unify
   the written self args STRUCTURALLY against the instantiation (reuse `bind_written_ty`,
   which already walks written-HIR-vs-MirTy) instead of requiring bare params.
6. **Mutable slice views** (`&mut base[range]`): decide with the owner whether required for
   C4 exit (REF-SLICE-001 allows views; 06's behavioral requirements mention slicing
   generally). If yes: `SliceNewMut` runtime op (surface bump, dated revision), `&mut [T]`
   dest, `write_resolved` through a Slice window (currently rejected — lift for mutable
   views only), borrowck implications probed first. If no: record the deferral in the exit
   report with the owner's dated decision.

### C4.7-9 — Fresh audit + the real C4 exit report  _pending_

1. Re-run the unsupported-site sweep exactly as WP-C4.6's audit did: enumerate every
   `unsupported(` in `lower.rs`, partition defensive-vs-construct, probe every construct
   candidate with c46_probe + oracle_run, classify against 02/03/06 + core-min.
2. Verify the 17-case frozen corpus still passes (`entire_frozen_corpus_agrees`) plus the
   full differential suite.
3. Write the exit report as a new final section of `WP-C4.6.md`: exit conditions 1–3
   re-evaluated, every remaining rejection classified (defensive / spec-conformant /
   deferred-with-owner-decision), deviation ledger state, and the recommendation.
4. Present to the owner for the C4 closure decision. Do NOT close the gate yourself.
5. Owner-table items to include: post-hoc CE3 ratification of the A5 shape additions
   (C4.7-1), the three A4 surface bumps executed under one pre-authorization (revs 8–10),
   whether to add frozen-corpus cases for Class-A constructs (corpus_version bump is
   governance-controlled), and the mutable-slice / literal-typing decisions from C4.7-6/8.

---

## 3. Increment status tracker

- C4.7-1: **DONE 2026-07-20** — doc half (state-file + ledger reconciliation, this plan) plus the
  coding-session remainder: `mir.md` **A3 shape amendment** recorded (bitwise/Pow/Shl/Shr/
  `InvalidShift`, presented for post-hoc CE3 ratification), **DEV-074** numbered (closed at
  creation; count 71 → 72; both enumerations), A4 wording tightened to "MIR runtime surface" in
  `WP-C4.6.md` and A1 rev. 10. **Naming note:** the A5 work is recorded as MIR amendment **A3**
  per this plan's §2 C4.7-1 wording, so C4.7-3's layout amendment is renumbered **A4**
  (`mir-amendment-A4-layout.md`) to avoid two A3s.
- C4.7-2: **DONE 2026-07-20** — 6 verifier negatives (`rejects_bitwise_binop_on_floats`,
  `rejects_pow_on_non_integer_dest`, `rejects_vec_get_ref_with_wrong_dest`,
  `rejects_chars_iter_next_on_non_iterator`, `rejects_runtime_call_arity_mismatch`,
  `rejects_switch_on_float`; the plan's item A.6 `rejects_call_arity_against_instance` did NOT
  exist, so the arity path is pinned by the runtime-arity test instead) + 4 clean-Unsupported
  fixtures, each probed typecheck-clean AND oracle-supported first. **Finding:** the plan's
  fixtures B.2 (method-own generics) and B.3 (non-bare impl heads) are **front-end-blocked**
  (E0001 / E0302 — they never reach lowering), which reclassifies C4.7-8.4 and C4.7-8.5 as
  front-end-first work. Workspace 752/0/2.
- C4.7-3: **DONE 2026-07-20** — `mir-amendment-A4-layout.md` drafted and **APPROVED by the owner
  under CE3 as drafted (CD-036)**; implemented the same session. `Rvalue::LayoutQuery { kind, ty }`
  (pure, dest `UInt64`) replaces the type-erasing `Const 8`; one `reference_layout(ty)` service in
  the MIR interpreter returns the frozen `(8, 8)`, so behavior is unchanged and the HIR oracle was
  not touched — `size_of_align_of_agree` stays green **unmodified**, which is the proof. Research
  finding: CD-015/C2.9 approved only that `size_of`/`align_of` are the sole layout observations
  and that Core promises no ABI; it fixed no per-type numbers, and LAYOUT-ABI-001 makes them
  target-/version-dependent — so real numbers are C5.1's, not C4's. 4 new tests; workspace 756/0/2.
- C4.7-4: **DONE 2026-07-20 — DEV-069 CLOSED.** Root cause was one class, not four bugs: all
  three modules read every span against a single "current file", which is right for the item
  being CHECKED and wrong for every item being LOOKED UP. Fix: `item_text(item, span)` in
  typecheck/borrowck/interp resolves against the declaring file (`hir.item_files`, the same map
  MIR's `ProgramMeta` uses), applied to every cross-item scan; plus a per-body file swap in the
  oracle, which never swapped at all — `Callable` carries its declaring file and all THREE
  body-execution funnels (`call_callable`, `call_user_method`, `drop_value`) save/restore around
  the body including error paths. `text()` is now non-panicking in all three as a backstop.
  Landed as ONE commit rather than the planned two: the tests exercise both halves end-to-end,
  so a typecheck-only commit would have left them red. 3 new tests + the multi-file differential
  WIDENED off its safe subset (cross-file struct/methods/literal/field/Drop, exact output
  pinned). Workspace 759/0/2.
- C4.7-5: **DONE 2026-07-20 — DEV-072 + DEV-073 CLOSED.** DEV-073's real root cause was
  `type_from_hir_without_diagnostics` dropping generic args (invisible for non-generic nominals);
  new `impl_self_ty_with_args` + `match_impl_type` in BOTH the operator-bound and iterable checks,
  with `Item` substitution for the latter. **MIR unchanged, as the plan predicted.** DEV-072:
  borrowck mirrors MIR's `scrutinee_reads_through_ref` and rejects non-Copy bindings under it
  (E0101), recursing through nested/shorthand patterns; wildcards and Copy bindings stay legal
  (pinned); MIR guard kept as defense in depth with an updated message. 2 differential + 2
  front-end tests; workspace 763/0/2.
- C4.7-6: **COMPLETE 2026-07-20 (6.1 + 6.2 + 6.3).** 6.3 (DEV-078): unsuffixed integer literals
  adopt an expected integer type, implemented as general inference (integer-kinded variables +
  a real step-5 defaulting pass), with range-checking on binding and the suffix/typed-value/kind
  negatives all still failing. Workspace 778/0/2. Earlier: 6.1 landed per the
  owner's option (a): `Box<T>` as an opaque owning runtime type, `BoxNew`/`BoxIntoInner`, surface
  `0.1-A7` (A1 rev. 11), no new `MirTy`, drop via the existing `Drop` glue. The audit's "`Box`
  deref" entry is corrected — `*box` is spec-conformant to reject and is now pinned negatively.
  Three pre-existing defects surfaced: drop discovery never descended into `Core` type args; that
  walk had no cycle guard (recursive `Box` types overflowed the stack); and **DEV-077**, an oracle
  double-drop in `into_inner`, closed here. Workspace 775/0/2. Earlier: 6.2 primitive `Ord::cmp` across
  all three engines, no new MIR shape or surface (constructs `CoreOrdering` from the comparisons
  `<`/`==` already lower). **Both other items contradict the plan's framing:** 6.1 — `*Box::new(5)`
  failing is **spec-conformant** (no `Deref` trait in Core v1; TYPE-METHOD-002 auto-deref removes
  only `&`/`&mut`; 06 gives `Box` just `new`/`into_inner`), and the real gap is `Box::new`/
  `into_inner` being MIR-unsupported, which is a §0.5 shape/surface decision; 6.3 — 03 REQUIRES
  the literal to adopt an expected `UInt64` (expected types flow inward from function parameters;
  defaulting applies only to *unconstrained* literals), so it is a real over-rejection, not
  spec-conformant. **DEV-075 opened** (pre-existing): `Bool`/`Char` ordered comparison is accepted
  by the checker but fails in both engines for `Bool` and DIVERGES for `Char` (MIR succeeds,
  oracle rejects). Workspace 765/0/2.
- C4.7-7: **DONE 2026-07-20 — DEV-067 + DEV-071 CLOSED**, which closes every front-end deviation
  the C4 track owned. DEV-071: `Ordering` is `Ty::Core` with `Res::Builtin` variants (like
  `Option`/`Result`) and had no explicit exhaustiveness arm, so it hit WP-C1.5's require-a-wildcard
  default; now tracks all three variants, two-variant still E0303. DEV-067 was TWO causes: (b) the
  bounded-parameter lookup tested the unpeeled receiver (TYPE-METHOD-002 requires the peel, and the
  path below already computed one — it just ran after); (a) `satisfies_bound` had no `Ty::Param`
  arm, AND bounds are verified in a deferred pass, so each obligation now carries the generic
  environment it arose in. 4 differential/front-end tests + the `_`-workaround dropped from
  `ordering_value_round_trips_through_match_agree`. Workspace 769/0/2.
- **DEV-075 increment: DONE 2026-07-20** (owner specification decision; sequenced after C4.7-7,
  which had already landed). `Char` is ordered by Unicode scalar value in BOTH engines (the oracle
  was aligned to MIR, which was already right) and joins the primitive `cmp` surface; `Bool` is
  not `Ord`, so its ordered operators and `Bool::cmp` are now compile-time errors. New normative
  **`PRIM-TRAIT-001`** primitive trait/operator matrix in 06 (+ 03 cross-reference), compiled spec
  regenerated, fixture corpus re-extracted. Pinned the float row's operator/trait split, which
  briefly broke `1.5 < 2.5` during implementation.
- **C4.7-8.1a: DONE 2026-07-20 — DEV-076 CLOSED** (oracle half of 8.1). `unwrap_or` no longer
  double-drops the payload or leaks the discarded default; the discarded value is destroyed **at
  the call**, not at scope exit — that is the timing the MIR half must match. The MIR half stays a
  clean `Unsupported`: moving a payload out of a drop-tracked local via `VariantField` hits the
  C4.5d guard and needs `lower_enum_match`'s drop-flag machinery.
- **C4.7-8.1: COMPLETE 2026-07-20** (MIR half). Droppable `unwrap_or` lowers; the discarded value
  is destroyed AT THE CALL, and a `Result`'s displaced `Err` payload with it. The C4.5d
  drop-tracked-local guard was cleared by reusing `lower_match`'s discipline — materialize the
  receiver into a temp, whose move clears the source's flags. Non-droppable lowering unchanged
  byte-for-byte. 3 differential tests; the stale Unsupported fixture removed.
- **C4.7-8.2: DONE 2026-07-20.** Droppable `Iterator` `Item` lowers with a per-iteration drop
  scope; each value dies at the end of its own iteration, and `break`/`continue` destroy the
  current one — the latter for free, by capturing the loop's `scope_depth` before pushing the
  per-iteration scope. 2 differential tests (3 programs); stale fixture removed.
- **C4.7-8.3a: DONE 2026-07-20 — DEV-079 + DEV-080.** Found while oracle-pinning for 8.3, both in
  the FLAT path A2/C4.5d had signed off: the verifier rejected every enum variant with 2+
  droppable payload fields (lowering accepted it — an internal inconsistency), and fixing that
  exposed an arm-end drop-ORDER divergence it had been masking. 2 differential tests (4 programs).
- C4.7-8: _pending_ — **8.3b (nested patterns) is the last MIR residual**; 8.6 is an owner decision (oracle `unwrap_or` double-drop); 8.4/8.5
  reclassified front-end-first by C4.7-2; 8.6 (mutable slices) is an owner decision.
- C4.7-9: _pending_ (last)

Recommended execution order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Increments 1–2 are
low-risk warm-ups; 3 requires an owner stop; 4 is the largest front-end change; 8.3 is the
hardest MIR change. 5/6/7 are independent of each other and of 8 — reorder if the owner asks.
