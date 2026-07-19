# STARK Compiler STATE
Updated: 2026-07-19 after WP-C4.5f-3 — **WP-C4.5 EXIT SATISFIED** (surface 0.1-A3)

## Position
Gate: C4  Next: WP-C4.6 (gate exit review). **WP-C4.5 "Complete Core Lowering" is COMPLETE:
all 17 frozen corpus cases run differentially green through both interpreters**
(`entire_frozen_corpus_agrees` — output, trap category, provenance, message, pre-trap stdout).
Runtime surface `0.1-A3` (A1 rev. 6). Open non-blocking boundaries carried forward: DEV-067,
DEV-069 (front-end multi-file spans), user-nominal `Eq`/`Ord` dispatch, generic impl/trait
methods, reserved ops (`values()`/`remove()`/`HashSet`/`StringSubstring`/`VecGetRef`),
assert-message fidelity.
Blocked: none
**WP-C4.5f-3 done 2026-07-19, closing WP-C4.5** — three sub-slices in one increment:
- **f-3a HashMap surface (`0.1-A3`, amendment rev. 6):** `RuntimeFn` HashMap group
  (New/Insert/Get/Len/IsEmpty/ContainsKey/KeysIterNew/KeysIterNext); insertion-ordered
  (CD-009) `MirValue::Vec` of `[k,v]` aggregates; `insert` returns the displaced `Option<V>`
  (honesty rule §5a — caller drops it at a visible Drop; user-`Drop` K/V refused); `get` →
  interior `Option<&V>`; `keys()` a true borrowed cursor reusing the f-2 for-desugar;
  schematic-(K,V) `map_runtime_sig`. **`collection_iter__02` differential-green.**
- **f-3b Char + assert_eq/ne (rev. 6):** `MirTy::Char` (`Constant::Int` Unicode scalar),
  `PrintlnChar`/`PrintChar`, `StringPushChar`/`StringPopChar`; `assert_eq`/`assert_ne` →
  scalar `BinOp::Eq` or `StrEq`/`StrCmp` into conditional `Trap{AssertFailure}` (message
  fidelity deferred with the e-1 boundary).
- **f-3c multi-file lowering:** `ProgramMeta` interns all source files (FileId(0)=entry),
  maps items to declaring file + module path; all cross-item name reads go against the owning
  item's file; `synthetic_spans` for generated wrappers; **module-qualified canonical symbols**
  (`helper::add_self@[]`) — package-stable linkage identity for C5. **Found DEV-069 (open,
  front-end WP):** checker + HIR oracle read cross-file spans against the entry file
  (cross-file methods/literals/field reads break); the differential test pins the
  front-end-safe subset; MIR side is multi-file-clean.
- **Exit-sweep fixes:** MIR-interp call args were bound positionally over locals `1..n`,
  clobbering interleaved drop-flag locals for callees with droppable params (bit
  `largest::<String>` in `struct_enum_trait__03`) — now bound by declared `Param(i)` kind
  with arity checks; non-place method receivers/`&expr` (call results) materialize via
  `place_or_temp`. 6 new differential tests + `entire_frozen_corpus_agrees` (all 17).
  Workspace 707/0; fmt+clippy clean 1.93/1.97.
**WP-C4.5f-2 done 2026-07-19** (by-reference Vec iteration, surface `0.1-A2` per CD-032's
dated-enumeration rule, amendment rev. 5): `VecIterNew`/`VecIterNext -> Option<&T>` (`T: Copy`,
V-COPY-1/MIR-0016); interpreter iterator = snapshot aggregate `[Vec, cursor]` in a frame local
handing out interior `&T` refs — protected by f-1's frame generations (built first,
deliberately); `for value in v.iter()` desugar; Index-on-Vec projection arms;
`MIR_RUNTIME_SURFACE = "0.1-A2"`. **`collection_iter__01` corpus case differential-green.**
Workspace 701/0/2; fmt+clippy clean 1.93/1.97.
**WP-C4.5f-1 done 2026-07-19** (both CD-030 deferrals): `Frame.generation` (monotonic) +
`MirValue::Ref` carries the pointee's generation; every deref and runtime-op ref helper
validates (slot, generation) — stale references to reused frame slots fail loudly (adversarial
hand-built MIR test: verifies by design, interpreter rejects). Projected `Move`s now TAKE with
a `MirValue::Moved` poison; any read of the hole is a loud internal error; full suite green
with the poison live confirms the tested subset never re-reads a moved place. Workspace
699/0/2; fmt+clippy clean 1.93/1.97.
**Match-drop increment done 2026-07-19** (match on owned Drop-bearing scrutinees): oracle drop
timing pinned empirically (matched arm consumes the scrutinee; bound, unbound `_`, and
catch-all payloads all drop at **arm end**). `lower_enum_match` rewritten — each arm a drop
scope; every payload field moved out of the materialized-temp scrutinee (bound → registered
binding local; unbound droppable → registered temp; catch-all → whole value), so the shell is
fully consumed (no double-drop) and everything drops at arm-scope exit; a body-moved binding
clears its flag so only the callee drops. Blanket C4.5d restriction removed. **`option_result__02`
corpus case now differential-green.** 4 new differential tests. Workspace 698/0/2; fmt+clippy
clean 1.93/1.97.
**WP-C4.5e-3 done 2026-07-19** (`?` + Option/Result methods): `ExprKind::Try` lowering
(operand in a temp consumed by both switch arms; Ok/Some payload = expr value, None/Err
early-returns the enclosing fn's Option/Result after dropping live scopes);
`is_some`/`is_none`/`is_ok`/`is_err` + `unwrap` (SwitchInt; wrong variant →
`Trap{UnwrapNone|UnwrapErr}`). `option_result__01` corpus case differential-green.
**A1 iteration gap RESOLVED — CD-032 (owner, 2026-07-19):** Vec iteration folds into C4.5f.
STARK's `.iter()` binds `value: &T` (by-reference = an interior reference into a runtime
container); A1's by-value `VecIterNext -> Option<T>` had no STARK trigger and is struck.
Iteration (by-reference `Option<&T>`) activates via a future `0.1-A2` surface bump alongside
the interior-reference/frame-generation work in C4.5f. `collection_iter__01`'s iteration half
stays Unsupported until then.
**WP-C4.5e-2 done 2026-07-19** (Vec data surface, A1/CD-031): `RuntimeFn` Vec group +
`MirValue::Vec`; `Vec::new`/`with_capacity`, method dispatch (push/pop/remove/clear/len/
is_empty), `v[i]` read → `VecIndexGet` (Copy T), `v[i]=x` → `VecReplace`+drop-old, `clear()`
on droppable T → pop-and-drop loop (§5a — destructors only at visible Drop terminators),
`Vec<T>` a droppable leaf unit dropping elements **reverse index order** (matched to oracle);
verifier schematic-T `runtime_sig` + V-COPY-1 (MIR-0016, `copy_types` populated,
`mir_needs_drop` precise); interp Vec ops (in-place `&mut Vec` mutation, call-site trap
provenance). 4 differential + 2 verifier tests. Workspace 691/0/2; fmt+clippy clean 1.93/1.97.
**WP-C4.5e-1 done 2026-07-19** (strings, implementing Amendment A1/CD-031): A1 shape
foundation landed (`MIR_RUNTIME_SURFACE`, `MirProgram.mir_version`/`runtime_surface`,
`Constant::Str`, `Trap.message`, `TypeContext.copy_types`, String/str `RuntimeFn` group, dump
header + `const "…"`). String literals, `String::new`/`from`, String/str method dispatch,
`&str`/`String` print, String/str comparison via `StrEq`/`StrCmp` (V-STR-2), `panic(msg)`/
`assert(cond)` traps, String as a droppable leaf unit, and user `as` casts (were unlowered)
all lower; verifier surface gate (MIR-0017) + V-STR-1/2 (MIR-0015) + Trap.message threaded
through every operand analysis; MIR interp gained `Str`/`String` values, in-place `&mut String`
mutation, snapshot `as_str`, and trap-message comparison. **The two frozen `ownership_drop__*`
corpus cases are differential-green** (first String-dependent corpus cases). Deferred to later
e sub-slices: Char + Char String ops, `assert_eq`/`assert_ne`. Workspace 684/0/2; fmt+clippy
clean on 1.93 and 1.97.
**WP-C4.5e-0 done 2026-07-19** (pre-runtime-values hardening, CD-030 review disposition):
IndexProof definite-initialization dataflow (must-analysis + unique-definition rule; 4
adversarial negatives incl. the review's one-branch example); V-REF-1/MIR-0014
write-through-shared-reference rejection (write-path place typing); pre-trap stdout now
observable and compared by the differential (`run_with_partial_output` + `MirFailure`;
drop-output-before-trap regression test); DEV-068 fixed (user `impl Copy` structs were
always-Move → field-precise verifier rejected valid double-use programs). Deferred with
owners per CD-030: frame generations (C4.5f), projected-move take-and-poison (C4.5e proper).
Workspace 675/0/2.
**WP-C4.5d done 2026-07-19** (ownership and Drop): droppable locals decompose into per-unit
`DropFlag`-guarded drops (units = outermost dtor-bearing/enum/array sub-places through
dtor-less structs/tuples — partial moves clear exactly the covered units); emission at scope
exits (reverse decl order), early exits, assignment overwrite (install-then-destroy per
CD-012), discards, and the `drop(x)` builtin; dtor instances discovered + registered in
`TypeContext::drop_impls`; MIR-interp recursive glue (own dtor through `&mut` ref, then
fields/payload reverse, enums by runtime discriminant); verifier V-MOVE-1 refined
field-precise with Drop-of-possibly-moved legal by design, V-DROP-2 read half added. Oracle
drop timing pinned empirically before implementation; the differential then matched on first
run (no new oracle defects — first increment where that happened). Boundaries (clean
Unsupported): match on owned Drop-bearing scrutinee (C4.5e, needs drop_unbound), Drop impls
on generic nominals (needs generic impls). Workspace 668/0/2.
C4.1-C4.4 done; WP-C4.5 split into increments (WP-C4.5.md). Done so far: C4.5a
(methods/assoc-fns/trait dispatch incl. defaults; corpus __01 differential-green),
C4.5-contract-cleanup (CD-029: trap provenance through outcomes + differential span
comparison; VerifiedMirProgram wrapper — run_program consumes proof-of-verification only;
TypeContext amended into mir.md §2, still v0.1; canonical_float spec tests as the
compensating control for the intentionally-shared formatter), C4.5b (indexing via CheckIndex
proof tokens + real reference places; DEV-065/066 oracle fixes), and **C4.5c 2026-07-19**
(external framing per CD-030: *top-level generic monomorphisation and static bound dispatch*
— generic methods/impls stay later-increment work: checker-recorded instantiations in
`TypeTables::generic_insts` with E0004 undetermined-rejection — DEV-064 closed; monomorphised
`FnKey::Top(item, type_args)` instances, injective `name@[args]` symbols, named
`LIMIT-MIR-MONO-INSTANCES`=512 limit negatively tested on polymorphic recursion; generic
nominal instantiations registered per `(item, args)` in TypeContext; operator + trait-bound
method dispatch per instantiation; comparisons on user nominals clean-Unsupported until
C4.5e's Eq/Ord impl dispatch; DEV-067 recorded — pre-existing checker over-rejection of
bounded params at intra-generic call sites and `&T` receivers, owner: later C4.5 increment;
6 new differential + 3 lowering + 3 typecheck tests). Same session: fixed the CI break — a
`collapsible_match` lint new in CI's clippy 1.97 (verify.rs; local was 1.93, 1.97 installed
side-by-side and both fmt+clippy verified clean at CI parity), failing every run since the
WP-C4.3 push. Differential status: no difference in lowering and MIR execution for the tested
subset, with some runtime algorithms intentionally shared and separately spec-tested.
Workspace 658/0/2 (C4.5b-2 baseline re-measured 646; the previously recorded 640 was stale).
WP-C4.3 done 2026-07-19: `src/mir/verify.rs` implements all 13 contract §10 obligations with
the MIR-xxxx internal namespace (first allocation, see Diagnostic codes); every lowered program
verifies clean; 13 hand-crafted invalid bodies each rejected with their specific code; one
unsafe-failure bug (panic on broken CFG edge in the move dataflow) caught by the negative suite
and fixed. Workspace 625/0/2.
WP-C4.2 done 2026-07-19: `starkc/src/mir/` implements the approved MIR v0.1 model (all CD-028
shapes) + scalar-core lowering + deterministic dump; 5 frozen-corpus cases lower; fn-values,
Option/Result-as-logical-enums, checked-terminator arithmetic all verified by tests (6 new,
workspace 611/0/2). Out-of-subset constructs report clean Unsupported naming C4.5.
MIR v0.1 contract APPROVED under CE3 (CD-028, approve-with-required-changes — Drop terminator,
Option/Result as logical enums, index-proof tokens; all applied). `mir.md` is the binding
implementation contract; changes to its shape need a new CE3 review + version bump.
Gate C3 complete 2026-07-19: WP-C3.1 (workload freeze + framework), WP-C3.2 (generated-Rust spike
4/17→8/17 with breadth), WP-C3.3 (direct Cranelift spike 3/17), WP-C3 breadth run, and **WP-C3.4
backend selection = `SELECT-GENERATED`** (owner CE5 decision, CD-026): generated Rust as the
initial production backend behind verified MIR, backend-neutral MIR keeping direct-Cranelift open
as a C7 migration. Decision analysis:
`starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`. Native backend selection
status: SELECTED. Next: Gate C4 (MIR contract + verified lowering) — WP-C4.1 defines the MIR
under CE3; the generated-Rust emitter will consume that verified MIR, not typed HIR.
Mandatory compiler path: Core=CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS (C2
closed)  Backend=SELECTED (generated Rust/C, CD-026)  MIR=open (Gate C4 next, WP-C4.1/CE3)
Native=blocked (behind C4, mandatory per CD-004)
Optional tracks: ArtifactInfra=blocked (no second artifact impl yet)  TensorExpansion=blocked (no approved workload, Conditional Track T)

## Repository baseline
- Last completed transition: WP-C2.13 (Gate C2 exit and Core v1 semantic freeze). Verdict
  **CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS** — all 24 high-cost open
  questions (CORE-Q-001..024) approved, 166-row completeness inventory has zero
  absent/contradictory/unclassified rows (6 remain `pending-owner-approval` governance
  bookkeeping only, behavior already implemented/tested), 33 deviations closed this gate
  (seventeen WP-C2.2 runtime-semantics defects, six WP-C2.11 items, DEV-036, seven
  post-WP-C2.11 correction-pass items, DEV-053/054), 8 remained open and non-soundness-relevant
  at gate close (current open set after the post-Gate-C2 correction brief: DEV-005/010/011/012,
  DEV-017 partial, DEV-060 — see the open index below).
  Full report: `starkc/docs/compiler/C2-exit-report.md`. C3-entry is the active transition
  before WP-C3.1.
- Transition base commit: `c268d7c` (`Add systems ecosystem roadmap`), after the post-Gate-C2
  correction-brief commit that resolved DEV-051, DEV-052, and DEV-055 and opened DEV-060.
- Amendment base commit: `60b49e2` (`CD-021 function-value native validation...`) — the head
  this state revision was written against. (Field renamed from "Current committed head" under
  CD-022: a commit cannot record its own SHA, so that framing was permanently one behind;
  the live head is always `git log`, never this file.) Commit only on explicit user request.
- Rust toolchain: `starkc/rust-toolchain.toml` pins `channel = "stable"` (no version number, tracks
  stable) with `rustfmt`/`clippy` components. Active environment measured: `cargo 1.93.0
  (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607 2026-01-19)`. `starkc/Cargo.toml` declares
  `rust-version = "1.85"` (crate MSRV). The Gate-5 *generated deployment host* (not `starkc`
  itself) separately requires Rust 1.88 due to the `ort` crate's MSRV
  (`starkc/docs/gate5-backend-decision.md:107-110`) — this does not raise `starkc`'s MSRV.
- Latest verified code baseline: `cargo test --workspace --all-targets --all-features`
  (starkc/, post-CD-025, 2026-07-19):
  **597 passed, 0 failed, 2 ignored** (594 → 596 from DEV-060's fix: one new typecheck
  regression test, one new interp execution test, one existing test rewritten in place; 596 →
  597 from CD-025's `corpus_lock_matches_frozen_snapshot` integrity test)
  across **4 unittest binaries** (`src/lib.rs`,
  `src/main.rs`, `src/bin/stark.rs`, `src/bin/starkide.rs`) **+ 32 integration-test files**
  (`find starkc/tests -maxdepth 1 -type f -name '*.rs' | wc -l`,
  re-counted against the
  post-WP-C2.7 tree — the
  "3 unittest binaries + 31/32 files" figure quoted in several prior session records below was
  never actually verified against `ls`/`cargo test`'s own "Running ..." lines and had drifted;
  not chasing down exactly which prior WP's arithmetic first went wrong, since that would need
  checking out old commits for no real benefit — this line is now the corrected, directly-counted
  baseline going forward). Up from 383/0/2 at Gate C0 close (file count at that point not
  re-verified for the same reason). WP-C1.1 added `span_integrity.rs` + 12 tests, WP-C1.2 added
  15 more across `resolve.rs`'s inline tests and `gate2_package.rs`, WP-C1.3 added 8 more across
  `typecheck.rs`'s and `interp.rs`'s inline test modules, WP-C1.4 added 11 more across
  `gate2_valid.rs` and `gate3_execution.rs`, WP-C1.5 added 21 more to `gate2_valid.rs`, WP-C1.6
  added `conformance_report.rs` (new file) + 4 tests.
  Both ignored tests are
  intentionally opt-in (a checksum-pinned live ONNX artifact test in `tests/gate4_onnx.rs`, and
  a live-ORT-download inference test in `tests/gate5_codegen.rs`). Full per-file breakdown
  recorded in `starkc/docs/dev/compiler-map.md` (WP-C0.1; not re-regenerated for the WP-C1.1/
  C1.2/C1.3 deltas — see that file's own scope note).
  Latest recorded validation also has `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and conformance
  validation/reporting clean.
- Core spec revision: `STARKLANG/docs/spec/` files 00-07 plus
  `CORE-V1-ABSTRACT-MACHINE.md` and `CORE-V1-FUTURE-BOUNDARIES.md`, normative per
  `CLAUDE.md`. Spec fixture corpus:
  `STARKLANG/tests/spec-fixtures/manifest.toml`, 113 entries (parse-pass 65,
  semantic-error 16, notation 27, lex-pass 4, parse-fail 1). WP-C2.7 removed 28 stale,
  duplicative memory-model examples and now contains 13 abstract-machine adversarial examples
  after its correction pass. WP-C2.8 appended five static-semantics review fixtures without
  renumbering existing examples.
- Tensor spec revision: `STARKLANG/docs/extensions/Tensor-Model-Types.md` (extension `tensor`
  v0.1), `AI-Extensions.md` (non-normative sketches).
- Conformance DB: `STARKLANG/conformance/core-v1-coverage.toml`, 59 `[[rule]]` entries.
  **Integrity-audited under WP-C0.3 (2026-07-17)**: no duplicate rule IDs, no references to
  nonexistent spec chapters (both now mechanically checked, see `starkc/scripts/
  check-conformance.py`). Post-correction counts: 53 implemented, 6 partial, 0 missing.
  Pre-correction counts (53 implemented, 2 partial, 4 missing) were **stale**, not accurate — see
  DEV-002. `starkc/scripts/check-conformance.py` now also warns (non-fatal) on `missing` entries
  that still carry a `source`/`tests` field and on likely-semantic-rejection rules with zero
  recorded tests, as a heuristic staleness signal for future audits. Known representational gap:
  the schema's single `tests` array does not distinguish positive from negative test evidence, so
  Charter rule 15 ("positive and negative evidence travel together") cannot be mechanically
  verified from this database alone for every rule. **WP-C1.6** (closed 2026-07-18) addressed
  this with a richer schema (`positive_tests`/`negative_tests`, function-level `path::function`
  citations) and populated it for 20 of 59 rules with real evidence; the remaining 39 still rely
  on the single aggregate `tests` citation and are reported as "unclassified" by the new
  `generate-conformance-report.py`, not silently treated as verified — see DEV-017.
  **Coverage percentages remain provisional**: "implemented" status
  for any individual rule is not re-verified at Core v1 rule-completeness depth until WP-C1.x; see
  governing rule in `COMPILER-CHARTER.md` §1.5 rule 14 and the explicit no-percentage-trust
  statement this state file and the WP-C0.5 exit report both carry.
  WP-C2.6 adds `STARKLANG/conformance/core-v1-rule-id-map.toml`, a mechanically validated
  transition from every one of those 59 broad IDs to the stable granular inventory IDs. It does
  not inherit broad implementation status; C2.11 must classify evidence and status per granular
  rule.

## Current compiler pipeline
- Source -> lexer (`lexer.rs`) -> parser (`parser.rs`) -> AST (`ast.rs`) -> resolve (`resolve.rs`)
  -> HIR (`hir.rs`) -> type/flow/borrow check (`typecheck.rs`, `flow.rs`, `borrowck.rs`) ->
  interpreter (`interp.rs`).
- Extension front end: `extensions/tensor/` (dim algebra, tensor/model types), gated by
  `options.rs` (`LanguageOptions`/`ExtensionSet`).
- Artifact path: `onnx/` (bounded ONNX signature import/verify, no graph execution) ->
  `deploy/` (Gate-5 lowering to a generated Rust host calling ONNX Runtime via the `ort` crate).
- Additional entry points (three separate binaries, non-overlapping command sets — see
  `starkc/docs/dev/compiler-map.md` for full detail):
  - `starkc` (`main.rs`): `check`, `run`, `parse`, `lex`, `lsp`, `import`, `verify`, `deploy`.
  - `stark` (`bin/stark.rs`): `check`, `build`, `run`, `test`, `fmt`, `doc`.
  - `starkide` (`bin/starkide.rs`): interactive terminal IDE, no CLI subcommands.
  - `lsp/` module backs `starkc lsp`; `formatter/` backs `stark fmt`; `doc_gen/` backs
    `stark doc`; `test_runner/` backs `stark test`.
- **Known duplication requiring WP-C0.1 tracing**: `starkc` and `stark` each implement their own
  `check`/`run`, and neither binary exposes the full command surface — a caller needing
  `deploy`/`verify`/`import`/`lsp` together with `build`/`test`/`fmt`/`doc` must invoke both
  binaries. Whether these two `check`/`run` implementations share one pipeline or have drifted is
  unverified; resolve in WP-C0.1 (this is exactly the "shared vs. duplicated entry points"
  question that WP is scoped to answer, and directly bears on Charter rule 18 — cross-tool
  convergence).

## Decision log — append-only
- CD-001 [WP-C0.0] Adopted the "C0-C10" gate numbering from
  `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet.md` as a **new, independent**
  sequence, not a renumbering of the repo's pre-existing (non-prefixed) Gate 1-7 track. The two
  numbering systems now coexist; `COMPILER-ROADMAP.md` carries a note at its top explaining the
  relationship. Rationale: the brief's own gate definitions (front end conformance closure,
  reference execution contract, compiled-language decision spike, MIR, native backend, language
  services, extension isolation, release qualification) do not map one-to-one onto the old
  gates, which were scoped around a single tensor/ONNX vertical-slice demonstrator rather than
  general Core conformance. Renumbering the old track retroactively would rewrite closed
  historical evidence, which Charter §1.5 rule 2 and WP-C0.2 ("do not rewrite historical gate
  evidence to match later implementation") forbid.
- CD-002 [WP-C0.0] Recorded that the strategic question Gate C3 (Compiled-Language Decision
  Spike) exists to answer has **already been examined once**, under the old gate track, and
  closed with a non-GO outcome:
  - `starkc/docs/gate6-memo.md`: Decision **REVISE** (owner-confirmed 2026-07-16) — comparator
    evidence was 5/5 vs 2/5 defects caught pre-inference against Python/ORT baseline, and parity
    (5/5 vs 5/5) against "the strongest typed-Rust host" comparator; recommendation was to
    re-scope the demonstrator, not GO or STOP outright.
  - `starkc/docs/gate7-decision.md`: Decision **RETAIN AS RESEARCH LANGUAGE** (owner-confirmed
    2026-07-16), tensor-track technical verdict POSITIVE, tensor productisation verdict DEFER,
    language thesis UNRESOLVED. Explicitly authorizes only a `stark verify` external-validation
    track as next work and states "No LSP work or language expansion is authorized" (superseded
    for LSP specifically by the subsequent WP8.1-8.5 work, all committed after gate7-decision.md
    per `git log`; that expansion was evidently owner-authorized outside this decision doc's
    text, but the state file flags the textual contradiction for WP-C0.2 to reconcile formally).
  - Disposition: Gate C3 must treat gate6-memo.md/gate7-decision.md as **directly relevant prior
    evidence about interpreter-vs-native tradeoffs**, not reopen the question from zero. This is
    scoped as a C3-entry consideration, not a C0 decision — C0 does not skip ahead of C1/C2. Set
    `Conditional tracks: Native=deferred` above to reflect that the most recent owner decision on
    a related (ONNX-vertical) native-deployment question was non-GO; C3 will need fresh evidence
    for the *general* Core compilation question, which the old gates never tested (old Gate 5's
    "native" path is code generation to a *generated Rust host*, not general Core-to-native
    compilation — it has no bearing on scalar/loop/struct/enum native lowering that C3-C7 would
    need to evaluate).
- CD-003 [WP-C0.0] Confirmed two stale root-adjacent status documents exist and require
  correction under WP-C0.2 (not fixed in this WP — C0.0 is bootstrap-only, per its own "Done
  when" — but recorded now so the fix isn't lost):
  - `CLAUDE.md:110-113,137` states "Gates 1-3 are closed... next: Gate 4" — contradicted by
    `starkc/docs/gate4-exit.md` through `gate7-decision.md`, all closed, and by the root
    `README.md`'s own delivery-gates table which correctly lists all seven gates as
    Complete/Decision-recorded.
  - `starkc/README.md:4` states "Gate 4 (tensor front end and ONNX signatures) is complete" with
    no mention of Gates 5-7, and its module "Layout" table omits `deploy/`, `lsp/`, `formatter/`,
    `doc_gen/`, `test_runner/` — five of the crate's fifteen `pub mod`s are undocumented there.
  - `STARKLANG/docs/PLAN.md:5` says "The roadmap defines what evidence advances the project
    (Gates 1-6)" and has no Gate 7 section, while `STARKLANG/docs/ROADMAP.md` has a full,
    evidence-cited Gate 7 section matching `gate7-decision.md` exactly. PLAN.md was last
    substantively updated for Gates 1-5.
  - By contrast, root `README.md` is internally consistent with all seven gate exit/decision
    docs and is the most reliable of the pre-existing status documents.
- CD-004 [2026-07-17, outside any single WP — a mid-session governance update triggered by a new
  source document] The user provided a revised master brief,
  `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet(1).md` (title: "... (Native Compiler
  Required)"), which supersedes the original `STARK-Compiler-Build-Brief-Revised-Sonnet.md` this
  track was bootstrapped from (WP-C0.0). **This is a real, deliberate scope change, not a
  clarification**: the original brief framed Gate C3 as an open, evidence-based question — GO,
  REVISE, DEFER, or STOP on whether STARK needs a general native Core compiler at all, explicitly
  naming DEFER/STOP as valid, non-failure outcomes. The revised brief removes that question
  entirely: general native Core compilation is now a **mandatory** completion requirement (new
  §1.2 "Guaranteed compiler completion state" in `COMPILER-CHARTER.md`), Gate C3 is renamed
  "Native Compiler Architecture and Backend Selection Spike" and now only selects *how* (backend
  strategy: SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), never *whether*. An
  interpreter-only release is explicitly "not an allowed C3 completion outcome," and Gates
  C4-C7 change from *conditional* on a GO decision to *mandatory* after C3 selects an
  architecture. Diff confirmed Gates C0-C2 and C6/C8/C9 are textually unchanged; the change is
  scoped to §1 (framing/rules), the `COMPILER-STATE.md` template in §2.4, Gate C3's outcome
  vocabulary, Gate C4/C5's conditionality headers, Gate C10's release-statement requirements,
  §4's dependency map (native path folded into the single mandatory path, no more separate
  "native compiler path" branch), §5.3's gate-decision vocabulary (adds `BLOCKED`), §7's session
  budget (single ~57-86 session mandatory-path figure, replacing the old bifurcated
  "interpreter-only 31-48 / full-native 58-88" framing), and §8's strategic-outcome list.
  Regenerated `COMPILER-CHARTER.md` and `COMPILER-ROADMAP.md` in full from the new brief text
  (same extraction method as WP-C0.0) rather than hand-patching, to guarantee fidelity; updated
  this file's Position-line schema (`Mandatory compiler path: Core=/MIR=/Native=` +
  `Optional tracks: ArtifactInfra=/TensorExpansion=`, replacing the old `Conditional tracks:`
  line) and renamed the `## Backend decision` section to `## Native backend selection` with the
  new status vocabulary (`not evaluated | SPIKING | SELECTED | REVISE | BLOCKED` + a `Selected
  strategy` field, replacing `GO | REVISE | DEFER | STOP`). CD-002's own text is **not** rewritten
  (append-only) but is now superseded in one specific respect: its framing that "Gate C3 will
  need fresh evidence for the general Core compilation question" remains true, but its implicit
  suggestion that a DEFER/STOP-style outcome remains available for general native compilation no
  longer holds — see the correction notes added inline in `COMPILER-CHARTER.md` §1.5 and
  `COMPILER-ROADMAP.md`'s header relationship note, both of which point back to this entry.
  Gates C0-C2 work already completed (this entire session, through WP-C1.2) required **no
  rework** — none of it touched native-compilation framing. Both brief files are left on disk
  as-is (the original for historical reference, the "(1)" revision as the new live source); this
  is a content decision, not a file-management one, and neither file was deleted or renamed.
- CD-006 [2026-07-18, WP-C1.5] Resolved a spec-internal tension in `03-Type-System.md`'s Numeric
  Semantics section, found during the WP-C1.5 audit and flagged to the user rather than resolved
  unilaterally (CE2-shaped): the section states both "Division or modulo by zero is a runtime
  error and MUST trap" and, in an adjacent bullet, "Floating-point operations follow IEEE-754
  semantics (NaN, +/-Inf)" — the current implementation traps on `0.0 / 0.0` (a literal reading
  of the first bullet), which is in tension with the second bullet's implied NaN/Inf behavior for
  floats specifically. **User decision: keep trapping (current behavior); no code change.** The
  "MUST trap" rule applies uniformly across all numeric types including floats; the IEEE-754
  bullet is read narrowly (governing ordinary float arithmetic results — e.g. overflow producing
  `+Inf`, not division by zero specifically, which STARK treats as an error condition like any
  other div-by-zero). No spec or code edits made under this decision; recorded so the question is
  not re-litigated in a future WP. `interp.rs`'s Float `BinOp::Div`/`Rem` arms are unchanged.
- CD-007 [2026-07-18, WP-C2.1] Settled a spec-silent gap found while writing
  `STARKLANG/docs/compiler/reference-execution.md` §1: the spec addressed almost no
  subexpression evaluation order (binary operands, call arguments, method receiver-vs-arguments,
  aggregate-literal fields, assignment lhs-vs-rhs, index base-vs-index). Flagged to the user
  rather than resolved unilaterally (CE1/CE2-shaped, per WP-C2.1's own scope-control answer).
  **User decision: adopt the interpreter's observed left-to-right order as normative.** Added a
  new "Evaluation Order (Core v1)" subsection to `03-Type-System.md` (after "Operators and
  Traits," before "Copy and Drop") stating: strict left-to-right evaluation for binary operands
  (non-short-circuit), call arguments, struct/tuple/array literal fields, and index base-before-
  index; short-circuit semantics for `&&`/`||` (already spec-derivable, now stated explicitly);
  condition/scrutinee-before-branches for `if`/`match` (also already spec-derivable); receiver-
  before-arguments for method calls; and right-hand-side-before-left-hand-side-place for
  assignment (explicitly flagged as the most surprising rule, since many C-family languages
  evaluate the LHS place first). `STARK-Core-v1.md`/`.html`/`.pdf` regenerated in the same change.
  No interpreter code changes needed — `interp.rs` already implements exactly this order
  throughout (confirmed during WP-C2.1's own drafting); this decision closes the spec-vs-
  implementation gap from the spec side, not the code side.
- CD-008 [2026-07-18, WP-C2.1] Settled a second spec-silent gap found in the same document, §10.3:
  `HashMap`/`HashSet` iteration order was unaddressed by any normative spec text, while the only
  related prose (`06-Standard-Library.md`'s non-normative "Performance Notes" — "HashMap<T> uses
  open addressing with Robin Hood hashing") implied unordered iteration, in tension with the
  interpreter's actual `BTreeMap`/`BTreeSet`-backed fully-sorted-deterministic behavior. Flagged
  to the user rather than resolved unilaterally (CE1/CE2-shaped). **User decision: adopt
  sorted-deterministic (ascending key order) as normative.** Added a new "Iteration Order (Core
  v1)" subsection to `06-Standard-Library.md` immediately after the `HashSet<T>` API block,
  stating `HashMap::keys`/`values`/`iter`, `HashSet::iter`, and `for`-loops over either MUST visit
  entries in ascending key order per the key type's `Ord` impl, regardless of internal storage
  strategy. Reworded the "Performance Notes" line to remove the implication of unordered
  iteration (now frames storage strategy as implementation-defined but explicitly subordinate to
  the iteration-order requirement — an open-addressing implementation would need to sort at
  iteration time to conform). `STARK-Core-v1.md`/`.html`/`.pdf` regenerated in the same change
  (shared with CD-007). No interpreter code changes needed — `interp.rs`'s `BTreeMap`/`BTreeSet`
  representation already satisfies this rule exactly.
  **Correction (CD-009, same day, external review):** CD-008 as originally written is broken —
  `HashMap<K, V>`/`HashSet<T>` only bound `K`/`T: Hash + Eq` (confirmed:
  `06-Standard-Library.md` lines 271, 293), never `Ord`, so "ascending key order per the key
  type's `Ord` impl" can require an implementation that isn't guaranteed to exist. It is also
  inaccurate to describe the interpreter as already satisfying this rule: `interp.rs`'s
  `BTreeMap`/`BTreeSet` sort by `Value`'s own internal structural `Ord` (a Rust-level total order
  over the runtime representation), not by dispatching to the STARK key type's own `Ord`
  implementation (which, per DEV-027 found in this same WP, cannot even be written today). CD-008
  is left as-is above (append-only — a record of what was decided, even though wrong), superseded
  by CD-009.
- CD-009 [2026-07-18, WP-C2.1 correction pass, external review] Corrects CD-008. **User decision:
  `HashMap`/`HashSet` iterate in first-insertion order**, not sorted-by-key order — no `Ord` bound
  needed (matches the actual `Hash + Eq` bound), still fully deterministic. Reworded
  `06-Standard-Library.md`'s "Iteration Order (Core v1)" subsection accordingly (insert appends to
  iteration order; re-inserting an existing key keeps its position; remove-then-reinsert moves it
  to the end) and reworded "Performance Notes" to match. `STARK-Core-v1.md`/`.html`/`.pdf`
  regenerated. **This is now a real, confirmed WP-C2.2 deviation, not a no-op**: `interp.rs`'s
  `BTreeMap`/`BTreeSet` representation does not track insertion order at all (it sorts by
  structural `Value::Ord`), so it does not satisfy the corrected rule — recorded as DEV-032.
- CD-010 [2026-07-18, WP-C2.1 correction pass, external review] Refines CD-007. **User decision:
  keep "the method receiver evaluates before any argument" as normative** (matching user-defined
  method dispatch and common OOP convention), rather than changing the rule to match a narrower
  implementation detail. However, re-reading `interp.rs::call_core_method` (the dispatch path for
  builtin/stdlib-type methods — `Vec`, `String`, `HashMap`, etc., as opposed to user-defined
  nominal types) during the same review found it evaluates argument expressions *before*
  resolving the receiver — the exact opposite of `call_method`/`call_user_method`'s order for
  user-defined types. CD-007's original claim "no interpreter changes are needed... `interp.rs`
  already implements exactly this order throughout" is therefore **incorrect** for this one path;
  left as-is above (append-only), corrected here. Recorded as a new WP-C2.2 deviation, DEV-033 —
  `call_core_method` needs to resolve the receiver before evaluating arguments, to match the now-
  confirmed-normative rule and `call_method`'s own behavior for user-defined types.
- CD-011 [2026-07-18, WP-C2.1 correction pass, external review] DEV-029 (struct/enum field drop
  order is alphabetical-by-field-name, not declaration order) was recorded as a confirmed
  deviation, but `05-Memory-Model.md`'s Drop Order section only ever demonstrated reverse-
  declaration-order for sibling `let` bindings — it never actually stated a rule for a struct's
  own field-internal drop order; DEV-029's framing called reverse-declaration-order "the only
  coherent extension" (an inference, not a citation). Flagged to the user rather than left as an
  inferred deviation (CE1/CE2-shaped). **User decision: amend the spec to state it explicitly.**
  Added two sentences plus a short example to `05-Memory-Model.md`'s Drop Order section extending
  the existing rule to struct/enum-variant fields (reverse declaration order). `STARK-Core-v1.md`/
  `.html`/`.pdf` regenerated (this addition included a new `stark` code block, requiring a spec-
  fixture re-triage: `05-Memory-Model__22.stark` through `__27.stark` renumbered to `__23`
  through `__28`, new `__22.stark` triaged `parse-pass`/`program`; verdict census updated to 68/
  122; `extract-spec-examples.sh` confirms the manifest is back in sync). DEV-029 is now a
  confirmed, spec-backed deviation rather than an inferred one — its ledger entry updated to cite
  the new normative text instead of describing the rule as inferred.
- CD-012 [2026-07-18, WP-C2.7] Approved CORE-Q-006 and the normative Core v1 abstract machine.
  Runtime authority moves from scattered operational prose to
  `CORE-V1-ABSTRACT-MACHINE.md`. Evaluation is exactly once; assignment evaluates RHS before
  destination, installs the new value before destroying the old; normal early transfers clean
  exited scopes; language traps abort without unwinding, including during destination resolution
  and partial aggregate construction. Reference identity is abstract and survives legal
  ownership/call transfers; returned receiver-derived references designate caller objects and
  range slices are live views. CORE-Q-020 is approved only for runtime ownership/destruction of
  existing Core patterns, and CORE-Q-017 only for the language-trap boundary; C2.8/C2.9 retain
  their remaining portions. This decision defines semantics but deliberately defers compiler/
  interpreter alignment and adversarial rule evidence to C2.11.
- CD-013 [2026-07-18, WP-C2.7 correction] Corrected CD-012's CORE-Q-006 approval scope.
  CORE-Q-006 is approved for runtime abstract-machine semantics only; static place legality,
  borrow coexistence/regions, temporary-reference escape, and returned-reference legality remain
  pending under C2.8. This supersedes only CD-012's phrase "Approved CORE-Q-006", not its
  runtime decisions or its C2.11 implementation-alignment deferral.
- CD-014 [2026-07-18, WP-C2.8] Approved the Core v1 static-semantics freeze. Type aliases are
  transparent; values are finitely sized with only `str`/`[T]` unsized behind references;
  inference is deterministic and function-local; trait selection is source-order-independent
  with no specialization; borrows have conservative lexical regions and no temporary
  extension; patterns use deterministic exhaustiveness/usefulness analysis; and constants use
  a closed side-effect-free evaluator. Standard-library hooks are recognized by canonical item
  identity only. CORE-Q-002/003/004/005A/006/007/015/020 are approved. CORE-Q-005 is partially
  approved because C2.9 still supplies canonical package/version identity. Numeric results,
  float trait participation, layout-query results, and resource-limit classification likewise
  remain C2.9 inputs. Compiler/interpreter alignment and granular evidence remain C2.11 work.
- CD-015 [2026-07-18, WP-C2.9] Approved the numeric, target, text, process, package, and
  standard-library contract freeze. Integers are fixed-width and checked; primitive floats use
  reproducible IEEE operations but do not implement `Eq`/`Ord`/`Hash`; text is valid UTF-8 with
  byte offsets and Unicode 15.1 casing. Package identity is relocation-stable and lock-backed,
  with one selected version per source/name/major line. Only `size_of`/`align_of` expose
  target layout and Core promises no ABI. Four no-argument `main` signatures have deterministic
  status/stream mappings. `core-min` is mandatory and `std-full` is optional but indivisible.
  Resource, compiler-limit, API-error, language-trap, and host/process failures are distinct.
  CORE-Q-005, Q008–Q014, Q017–Q019, Q021, Q023, and Q024 are approved; alignment remains C2.11.
- CD-016 [2026-07-18, WP-C2.10] Approved CORE-Q-016 and the Core v1 future-extension
  boundary. Core execution is safe and single-threaded; capturing closures, explicit lifetime
  syntax/reference fields, trait objects, concurrency, macros, unsafe, and general FFI remain
  outside Core. Future callables must preserve ownership/capture/Drop semantics. Host access is
  limited to metadata-bound approved native providers with explicit identity, integrity, ABI,
  target, provenance, capability, and verification. Extensions require explicit stable
  identity/version enablement and cannot change Core-only behavior. No future feature is
  implemented by this decision; C2.11 owns exclusion/isolation enforcement evidence.
- CD-017 [2026-07-18, C2.8/C2.9 correction] Clarified nine pre-C2.11 freeze points.
  Generic fields may instantiate with references and recursively propagate borrow provenance;
  constant patterns never invoke user `Eq`; positive bounds never prove unifying impl heads
  disjoint. Canonical package names are distinct from identifier-valid aliases, each alias
  selects exactly one major line, and all packages remain library-importable while executable
  mode selects the root `main`. Floating `**` is rejected. Standard hash values use canonical
  FNV-1a encodings and primitive Display bytes are exact. `std-full` freezes availability and
  explicitly stated behavior only; unstated method edge cases are not conformance claims.
- CD-018 [2026-07-18, roadmap amendment before WP-C3.1] Adopted the post-C2 roadmap correction
  brief without replacing the core C3-C7 sequence. Inserted mandatory `C3-ENTRY — Native
  Readiness and Carry-Forward Closure` before WP-C3.1; made pending-owner-approval rows,
  DEV-051/052/055 ownership, WP-C2.12 generated-corpus/cross-backend transfer, versioned corpus
  freeze, and native-path CI baseline explicit. Strengthened C3.1's frozen workload with
  generics, trait dispatch, default trait sibling calls, references/slices, Drop-bearing trait
  dispatch, opaque host resources, and provider-boundary file I/O. Added Native Provider ABI
  v0.1 to C5.1, removed C5.4's "where supported" generic-call escape hatch, introduced platform
  tiers, added real systems workloads to C7 measurement, and created
  `STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md` with S0-S7 plus the post-C6 P1 Native Systems
  Baseline checkpoint. This is a sequencing and evidence-governance amendment; it does not
  reopen C2 or change Core v1 semantics.
- CD-019 [2026-07-19, C3-entry follow-up amendment] Tightened the post-C2 roadmap amendment
  before WP-C3.1. DEV-060 is now owned by C3-ENTRY and must be disposed before the workload
  freeze. P1 now gates C7.5/C7.7 closure and is required for Native Systems Preview and
  STARK v1 General-Purpose Stable claims, while Core v1 Compiler Stable may describe compiler
  maturity without claiming systems-platform maturity. The C3 provider/resource experiment is
  explicitly disposable and non-normative; C5.1 remains the first stable Native Provider ABI.
  Systems S6 is split into joint concurrency tracks for language proposal, compiler
  implementation, runtime/provider work, and ecosystem validation. `COMPILER-STATE.md`'s
  load-bearing header now points at `c268d7c`, the 594/0/2 verified code baseline, and the
  remaining C3-entry blockers.
- CD-020 [2026-07-19, C3-entry governance-repair pass — no semantic or compiler change]
  Repaired the governance surface before C3-ENTRY closure work begins. (a) Created
  `work-packages/WP-C3-ENTRY.md` — the transition's executable WP: named exit artifact
  (`starkc/docs/compiler/C3-entry-exit.md`), mechanical corpus-freeze definition
  (`corpus.lock`, SHA-256 per file, version-bump rules), per-blocker owners, "Done when";
  roadmap C3-ENTRY section now points at it. (b) Amended WP-C4.4/C5.6/C6.5 in
  `COMPILER-ROADMAP.md` to carry their transferred WP-C2.12 generated-corpus/cross-backend
  obligations in the receiving WP text (previously stated only in the C3-ENTRY bullet list,
  invisible to the charter's minimal session-input packet). (c) CI baseline delta:
  `.github/workflows/ci.yml` commands widened to the C3-ENTRY forms, added spec-regeneration
  check (new `--check` mode in `STARKLANG/tools/build-core-spec.py`, Markdown-only since
  pandoc/weasyprint output is not byte-reproducible) and a named execution-snapshot step;
  local fmt + exec_snapshots verified green, full CI run pending. (d) Accuracy corrections:
  `KNOWN-DEVIATIONS.md` tail summary (claimed DEV-009/022/023/024 open; all four resolved by
  WP-C2.11 per their own entries — stale paragraph from C2.6 time), state header current-head
  (`c268d7c` → `9e85396`) and spec-fixture census (112/parse-pass-64 → directly re-counted
  113/parse-pass-65; evidence-inventory "121-fixture" figure also corrected), charter §1.5/§2.4
  "roadmap §5.3" dangling references (vocabulary lives in charter §5.3), charter §2.1 step 10
  commit policy (owner convention: commit only on explicit request), WP-C6.4 tier label ("Core
  v1 Stable" → "Core v1 Compiler Stable" matching the C10 release class), and a new
  "Relationship to the compiler roadmap's P1 checkpoint" section in
  `STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md` (CD-018 described P1 as living there but the
  file never mentioned it; S5 is now explicitly identified as the P1-completing stage).
  (e) Compressed this file from 3,145 to ~700 lines per charter §2.4: deviation seed sections,
  C0/C1-era file inventory, completed follow-ups, and session records through Post-Gate-C2
  Issue 5 moved **verbatim** to `STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md`;
  decision log, conformance summary, gate exit summaries, open-deviation index, and the
  Issues 6-8 session record retained inline. Charter/roadmap edits under this entry are
  governance/bookkeeping repairs, not meaning changes to the extracted brief.
- CD-021 [2026-07-19, owner-approved roadmap amendment — function-value native validation,
  P1 trap report, release deviation sweep] Origin: an external-review debate established that
  non-capturing `fn(...) -> ...` function types are **existing frozen Core v1 capability**
  (`03-Type-System.md:198-200,999`; stdlib contract `06-Standard-Library.md:243-244,260-262,
  663-666`; `interp.rs:260` `Value::Function(ItemId)`), not a future closure feature — so the
  native path must validate them explicitly rather than leave them implicit. Three changes,
  same style/class as CD-018's workload strengthening: (a) WP-C3.1's frozen workload gains
  items 16-21 (typed function-value local; indirect invocation; `Option::map`/`Result::map`
  with a function value; function value in a struct field; cross-package function reference;
  monomorphised-generic function value with an explicit record-the-boundary fallback) — any
  item failing against the current implementation becomes a DEV entry before backend
  selection, deliberately; C4 gains explicit indirect-call ownership (WP-C4.1 MIR
  function-value constants/indirect-call representation, WP-C4.3 indirect-call signature
  verification, WP-C4.5 function-value lowering with provenance); WP-C5.1's runtime ABI list
  gains function-value/code-pointer representation, indirect calling convention,
  cross-package function-symbol identity, and function values in aggregates. (b) P1's exit
  list (roadmap §4.2) and S5's requirements (`SYSTEMS-ROADMAP.md`) gain a documented
  trap-abort operational report — deliberately trap one handler, record the effect on
  in-flight connections/resources/buffered output/process state; evidence input for any
  future fault-isolation proposal, explicitly no semantic change. (c) WP-C10.7 gains a
  release-blocking deviation sweep: every open deviation needs an owning gate/WP or a
  recorded accepted-indefinitely disposition. Related but not enacted here: the planned
  paper-only "Callable ABI and Future Closure Compatibility Spike" memo (existing-capability
  section + future-closure-compatibility section, outcomes GO/REVISE-ABI/
  DEFER-ESCAPING-BORROWS/ANNOTATIONS-LIKELY/NO-CURRENT-DESIGN) remains a separate proposal to
  be drafted before WP-C5.1; it is a recommendation, not yet approved work.
- CD-022 [2026-07-19, owner-approved follow-up amendment — external review of CD-020/CD-021
  commits] Three changes. (a) **Release-class coherence repair, preserving CD-019.** External
  review correctly found two superimposed models: C7.7 requires P1 (CD-019), Core v1 Compiler
  Stable requires C7, so its "must not claim systems-platform maturity unless P1 is complete"
  conditional was vacuous and General-Purpose Stable's "+P1" added no evidence. Resolution
  keeps CD-019's C7 gating (its motive — no toy-workload performance report — stands) and
  recasts the two stable classes as differing in **claim scope, not evidence**: Compiler
  Stable necessarily carries P1 evidence but asserts compiler maturity only; General-Purpose
  Stable adds no evidence gate and is the class permitted to assert systems-platform
  maturity. The reviewer's alternative (decouple C7 from P1) was considered and rejected as a
  CD-019 reversal. (b) **Function-value property validation.** WP-C3.1 gains workload items
  22 (repeated indirect invocation through one local — spec-guaranteed by function values
  being `Copy`, `03-Type-System.md` §Copy and Drop; DEV-060 is this bug class for default
  trait methods) and 23 (`Copy` aggregate with a function-value field, copied, both copies
  invoked), plus a pre-backend-selection requirement to settle the two genuinely open
  properties — `Eq`/`Ord`/`Hash` participation and monomorphised-generic function-value
  identity — from the frozen spec or by CE1/CE2 escalation, never by MIR/ABI accident. The
  reviewer's broader open-question list (Copy? repeated calls? Drop?) was narrowed: those are
  already frozen by the spec's Copy rule. (c) **State-header field rename**: "Current
  committed head" → "Amendment base commit" (self-referential staleness by construction).
  Outstanding from the same review, not part of this entry: a demonstrated green CI run
  (requires pushing to origin; no run exists yet).
- CD-023 [2026-07-19, owner-approved] Approved all six `pending-owner-approval` completeness
  rows (`LEX-COMMENT-001`, `LEX-ERROR-001`, `STD-OPTION-001`, `STD-RESULT-001`, `STD-ITER-001`,
  `STD-VEC-001`) as-is — the behavior each row describes has been implemented and exercised
  throughout Gate C2; the gap was governance bookkeeping only (C2 exit report). All six flipped
  to `settled` in `CORE-V1-COMPLETENESS.md` (`LEX-ERROR-001` keeps its DEV-017 note — an
  evidence-citation-precision gap, not a behavior question). C2-exit-report.md gained a dated
  post-gate update note per the same convention as the DEV-051/052/055 correction, rather than
  rewriting historical gate-close evidence. This closes the first of C3-ENTRY's four blockers;
  DEV-060, the corpus freeze, and the green CI run remain open.
- CD-024 [2026-07-19, owner-approved disposition: fix now] Closed DEV-060 (repeated call to an
  un-overridden trait default method wrongly flagged as a move). Root cause: `borrowck.rs`'s
  `method_receiver` — consulted by the `Call` handler to decide whether a method receiver is
  moved, borrowed, or mutably borrowed — only ever searched `ImplItem::Fn` overrides, with no
  equivalent to `typecheck.rs::resolve_method`'s `default_fallback` (WP-C1.3/DEV-013). A call to
  an un-overridden trait default method therefore returned `None` from `method_receiver`, and
  the `Call` handler's `None => self.check_expr(*base)` arm ran instead of the `Some(Receiver::
  ..)` arms — `check_expr`'s `Path` arm unconditionally consumes (moves) any `Local`/`SelfValue`
  place, regardless of the method's real receiver kind. Fixed by adding the matching
  trait-default-body fallback to `method_receiver` itself, mirroring `typecheck.rs`'s search but
  returning the method's declared `sig.receiver`. Verified both the `&self` case (original
  repro) and a new `&mut self` companion case (the `RefMut` arm wasn't exercised by the original
  repro alone — two sequential calls must register two non-conflicting borrows, not a move), and
  that the original repro now executes with correct output twice, not just "no diagnostic".
  Full workspace suite: 596 passed / 0 failed / 2 ignored (up from 594 — one new typecheck test,
  one new interp execution test, one existing test rewritten in place from
  documenting-the-defect to asserting success). `cargo fmt --all -- --check` and `cargo clippy
  --workspace --all-targets --all-features -- -D warnings` both clean. Full writeup:
  `KNOWN-DEVIATIONS.md`'s DEV-060 entry. This closes the second of C3-ENTRY's four blockers; the
  corpus freeze (now unblocked — WP-C3-ENTRY.md's procedure required this fix to land first) and
  the green CI run remain open.
- CD-025 [2026-07-19] Froze the WP-C2.12 execution-snapshot corpus and closed C3-ENTRY. Blocker
  3 (corpus freeze): `starkc/tests/exec_snapshots/corpus.lock` created at `corpus_version =
  1.0.0`, base commit `3d12f45`, SHA-256 per corpus file (48 files: 31 `.stark` + 17 `.snap`
  incl. `metamorphic/`); lock digest
  `8cda2df5e26aa35dfc8eb222f1e073eb4ea2336297e91ecc4e62b8fbd27dc0dc`. New integrity test
  `corpus_lock_matches_frozen_snapshot` (exec_snapshots.rs) enforces hash-match + no-missing +
  no-unlisted, negatively verified (tampering one `.snap` fails it with the expected message;
  restore passes). Freeze taken after DEV-060's fix per WP-C3-ENTRY.md procedure. Blocker 4 (CI):
  green on `origin/main` @ `3d12f45`, owner-confirmed. With blockers 1 (CD-023) and 2 (CD-024)
  already closed, **C3-ENTRY is closed** — exit artifact `starkc/docs/compiler/C3-entry-exit.md`
  written, Position line flipped to `Gate: C3  Next: WP-C3.1  Blocked: none`. Any future corpus
  change must bump `corpus_version` with a dated note here; a bare `UPDATE_SNAPSHOTS=1`
  regeneration is a freeze violation the integrity test catches. No semantic or Core behavior
  change.

- CD-026 [2026-07-19, WP-C3.4, owner CE5 decision] **Backend selection: `SELECT-GENERATED`.**
  Generated Rust is the initial production backend behind verified MIR; the MIR contract is to be
  designed backend-neutrally so `SELECT-DIRECT` (Cranelift) remains a live C7-gated migration
  (charter §1.6 rule 9). Basis: WP-C3.2 (generated-Rust) reached 8/17 frozen-corpus breadth
  cheaply with zero mismatches and trap parity, the shortest/lowest-risk path to correct broad
  native compilation (charter §1.6 rule 7); WP-C3.3 (direct Cranelift) is correct and self-
  contained (no rustc dep) but owns monomorphization/layout/drop/runtime up front — the better
  *eventual* backend if the self-contained-compiler goal becomes primary, which is a C7 judgment.
  Neither `REVISE` (missing data — exe size/startup, MIR-level comparison — is inherent to
  sequencing, needs C4-C7, not a bounded pre-C4 follow-up) nor `BLOCKED` (both paths demonstrated
  correct native execution). Accepted trade: `stark build` permanently requires a rustc toolchain
  and is slower; acceptable for STARK-as-research-language, re-evaluated at C7. Full three-way
  analysis + the required architecture commitments (MIR boundary, runtime/ABI, targets, debug
  mapping, unsupported-MVP closure, why-direct-rejected-as-initial):
  `starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`. Gate C3 closes; next is
  Gate C4 (MIR contract, CE3). This decision selects a backend strategy only — it does not build
  MIR, define the MIR contract, or fix the runtime ABI (those are C4/CE3 and C5.1/CE4).
- CD-027 [2026-07-19, owner-approved: two CE freezes + a correction-pass authorization] Settled
  the two CD-022 carry-forward function-value properties and repaired the fn-value feature
  cluster found by executing CD-021 workload items 16-22 against the interpreter for the first
  time. **(a) CE1 — TYPE-FN-001** (new normative rule, `03-Type-System.md` §Function Types):
  function values are `Copy`/`Clone`, never `Drop`, and do **not** implement `Eq`/`Ord`/`Hash`
  in Core v1 (float-precedent); consequence: function-value identity is unobservable, so the
  monomorphised-generic-identity question collapses to deterministic symbol naming (C6.2), not
  language semantics. **(b) CE2 — TYPE-FN-002** (same section): a generic fn coerces to a
  concrete fn type only when the expected type fully determines every generic argument;
  semantics = instantiate at the coercion site. Combined spec regenerated; no new code blocks so
  no fixture re-triage; two granular rows (TYPE-FN-001/002) added to `CORE-V1-COMPLETENESS.md`
  (166 → 168 rows — the fn-value questions were a genuine inventory gap). **(c) Pre-C4.1
  correction pass (authorized fix-now):** DEV-061 (indirect calls through fn-value locals/params
  never executed — missing `Res::Local|SelfValue` arm in interp call dispatch; the machinery
  existed one arm below), DEV-062 (fn values not `Copy` in borrowck/typecheck — `Ty::Fn`
  explicitly misclassified against the spec's Copy list), DEV-063 (`Option::map`/`and_then`,
  `Result::map`/`map_err`/`and_then` absent from the method table despite the normative §Option/
  §Result APIs) — all three FIXED with 5 new regression tests; the semantic oracle can now
  execute workload items 16-22. One new narrow deviation found and deliberately not fixed in
  this pass: DEV-064 (undetermined-generic fn coercion accepted; TYPE-FN-002 requires rejection;
  owner C4.5). Note: these settlements landed after CD-026's backend selection but before any
  MIR/ABI work — the selection is unaffected (identity-unobservability removes the one property
  that could have differentiated the candidates' ABIs).

- CD-029 [2026-07-19, review-directed correction pass before C4.5 breadth] Four corrections
  from the external review of the C4.1-C4.4 foundation, applied before they could embed across
  complete-Core lowering. (a) **Trap provenance**: `MirRunError::Trap` was discarding
  `SourceInfo` — a right-category trap at the wrong location would have passed the C4.4
  differential; outcomes now carry full `TrapInfo`, mir.md §6 amended to make provenance part
  of the observable trap outcome, and the differential compares user-origin trap spans exactly
  against the oracle (synthetic origins compare classification). Both existing trap tests pass
  with exact span equality. (b) **TypeContext contract treatment**: formally amended into
  mir.md §2 as part of the in-memory MIR compilation unit (additive, not dump-serialized, MIR
  stays v0.1) — resolving the governance debt the WP-C4.3 record flagged. (c) **Verified-MIR
  wrapper**: `verify_program` returns `VerifiedMirProgram<'_>`; `run_program` (and eventually
  the generated-Rust backend) consumes only that — "no backend bypasses MIR validation" is now
  an API property. (d) **Differential-independence caveat**: the shared `canonical_float`
  formatter is structurally invisible to the HIR/MIR differential; claim qualified everywhere
  going forward ("no difference in lowering and MIR execution for the tested subset, with some
  runtime algorithms intentionally shared") and compensated by new spec-derived golden +
  round-trip property tests (`tests/canonical_float.rs`, incl. NaN/±inf/-0.0/notation
  boundaries at exponent 15↔16 and -4↔-5/subnormals/max-min finite). Also adopted the review's
  C4.5 increment ordering + honest maturity calibration (architecture ~90%, implementation
  breadth ~35-45%, validation ~70%) into WP-C4.5.md.

- CD-030 [2026-07-19, owner-approved disposition of the external C4.5c-head review] The review
  (written against `82211f6`, before WP-C4.5d landed) found three validation holes plus two
  warnings. Disposition: **fold the load-bearing items into C4.5e as its entry step
  (WP-C4.5e-0)** — (1) IndexProof definite-initialization dataflow (the global name→base map
  alone accepted MIR whose check ran on only one branch; slices in C4.5e build directly on the
  proof discipline), (3) V-REF-1 write-through-shared-reference rejection (MIR-0014), (4)
  partial-output-before-trap comparison in the differential (C4.5e's panic/assert paths are
  exactly where it matters; both engines now expose pre-failure stdout —
  `interp::run_with_partial_output`, `MirFailure`), plus the review-warned user-`impl Copy`
  misclassification, confirmed real (valid Copy-struct programs failed MIR verification as
  use-after-move) and fixed as **DEV-068**. **Deferred with owners** (defense-in-depth only,
  no observable-behavior risk in the current subset): frame-generation identities in the MIR
  interpreter (owner: C4.5f, before cross-package call graphs grow frames) and
  projected-move take-and-poison (owner: C4.5e proper, alongside the runtime values that make
  aggregates bigger; the unit-flag design makes the current clone-not-take unobservable, and
  the stale interp comment claiming whole-local verifier conservatism was corrected). Review's
  wording caution accepted: C4.5c externally = "top-level generic monomorphisation and static
  bound dispatch" (generic *methods*/impls and user-nominal Eq/Ord operator lowering remain
  later-increment work). The review's C4.5d checklist was already fully implemented by the
  WP-C4.5d commit it had not seen, except the two deferred items above.
- CD-031 [2026-07-19, CE3 — owner-approved MIR v0.1 Amendment A1] Approved
  `STARKLANG/docs/compiler/mir-amendment-A1-strings-runtime.md` (rev. 3) as a **narrow additive
  amendment to MIR v0.1**, runtime surface `0.1-A1` — the contract prerequisite the C4.5e-main
  body needs before lowering strings/collections. Additions, all additive (no existing construct
  reinterpreted): `Constant::Str(String)` = a decoded immutable UTF-8 literal typed `&str`
  (owned `String` only via runtime `StringFromStr`; literal identity unobservable);
  `Terminator::Trap { message: Option<Operand> }` for `panic`/`assert` messages (participates in
  every operand analysis, not just typing); `String`/`Vec`/`VecIter` become drop-elaborated
  runtime values (**always** buffer-reclaim glue; element-destructor execution conditional on
  `T`; `Vec<T>` element drop in **reverse index order**, matched empirically to the frozen oracle
  `interp.rs::drop_value`); a versioned `RuntimeFn` appendix (30 ops lowered in C4.5e + a reserved
  group activated later only by a dated enumeration bumping the surface id); one new in-memory
  `TypeContext` field (`copy_types`) and two new `MirProgram` fields (`mir_version`/
  `runtime_surface`, consumer-checked before any body); new verifier codes MIR-0015 (V-STR-1/2,
  Trap.message typing), MIR-0016 (V-COPY-1: `VecIndexGet`/`VecIterNext` require `T: Copy`;
  `VecClear` requires non-droppable `T`). Two owner-mandated honesty rules: no `RuntimeFn` ever
  runs a user element destructor (those run only at visible `Drop` terminators — `clear()` on
  droppable `T` lowers to a pop-and-drop loop; `v[i]=x` uses `VecReplace(...)->T` so the caller
  drops the old value); and a backend doing explicit reverse-order element destruction must
  suppress any automatic (Rust) element drop. Three rev cycles (rev. 1 direction approved; rev. 2
  eight corrections; rev. 3 four final corrections) recorded in the doc's §11. `mir.md` §5/§7
  carry pointers to the amendment; `MIR_VERSION` stays `0.1`. This decision approves the contract
  only — no code is written by it; the C4.5e main body implements it next.

- CD-032 [2026-07-19, owner decision — A1 iteration correction, folded into C4.5f] The
  WP-C4.5e-2 implementation surfaced that Amendment A1's by-value `VecIterNext -> Option<T>`
  ("the `for x in v` desugar") has **no STARK source trigger**: STARK has no by-value
  `for x in v`; the only iteration form is `for x in v.iter()`, and `Vec::iter()` binds the
  loop variable as `&T` (stdlib `iter(&self) -> VecIter<T>`). So all Vec/collection iteration
  in STARK is **by-reference** — an interior reference into a runtime container, which is the
  work A1 §5d already reserved and tied to C4.5f's frame-generation hardening. **Owner
  decision: fold iteration into C4.5f.** A1's by-value iteration ops are struck from surface
  `0.1-A1` (they were never added to the `RuntimeFn` enum, so `0.1-A1` as implemented is
  unchanged — no bump); by-reference iteration (`VecIterNew`/`VecIterNext` yielding
  `Option<&T>`) is a C4.5f deliverable activated by a future dated `0.1-A2` surface bump,
  alongside `VecGetRef`/`StringSubstring` interior views and the frame-generation identities.
  Amendment doc updated (rev. 4): §5c iteration rows struck, §5e reframed as the C4.5f
  carry-forward design, rev-4 log added. No code change; strings (e-1) and the Vec data
  surface (e-2) are untouched. `collection_iter__01`'s `for value in values.iter()` stays
  clean-Unsupported until C4.5f; its push/index/len half lowers under e-2.

## Conformance summary
- Lexical: WP-C1.1 requalification complete (2026-07-17). Strengthened: all 15 reserved words
  now tested by name (was 3), reserved-word rejection confirmed in non-expression positions,
  nested-comment depth tested to 4 levels (was 2) with a matching unterminated-at-depth negative
  case. Found and closed one real bug in the process (DEV-014). Found and recorded, but did not
  fix, a real gap outside this rule's own scope (DEV-015, literal overflow never checked).
- Syntax: WP-C1.1 requalification complete. Strengthened: `>>`/`>>=`/`>=` generic-closing-token
  splitting (added the previously-untested `GtEq`→`Eq` split arm and a bare-shift-expression
  contrast case), multi-file `mod` layout (added missing-file, duplicate-declaration, and
  circular-reference cases — the missing-file case is DEV-014's regression test), depth-limit
  boundary behavior (added exact-latch and false-positive-floor assertions, `starkc/tests/
  robustness.rs`), diagnostic determinism across repeated parses of identical input, and AST
  span-containment (new `starkc/tests/span_integrity.rs`, DEV-018 — first-ever programmatic
  span-invariant check in the codebase, covering `Expr`/`Block` nodes across the full parseable
  fixture corpus).
- Types: WP-C1.3 requalification complete (2026-07-17). The equality/trait-dispatch closure the
  roadmap flags is now **fully resolved** (DEV-008 closed — real `Eq::eq` dispatch implemented,
  plus a companion fix so `Ty::Core` container types satisfy Eq/Ord bounds at all). STD-004
  (standard traits) exhaustiveness audit closed (DEV-013) with 2 real bugs found and fixed:
  `.clone()` was entirely non-functional on every compiler-builtin type (String/Vec/Option/
  Result/HashMap/HashSet/Range/IOError), and trait default method bodies were never used as a
  fallback when unoverridden — both now fixed with regression tests. `Error`/`Hash`/`Display`/
  `Clone` as generic *bounds* were already correctly recognized throughout (the DEV-013 seed's
  worry about `Error` support was checking the wrong function). Two new deviations found and
  recorded but deliberately not fixed to keep scope bounded: DEV-023 (`Display`/`Hash` share
  Clone's old "missing as a callable method on builtins" bug, not yet fixed) and DEV-024 (`From`
  trait `Type::from(value)` associated-function calls fail to resolve, root cause not yet
  isolated). Local inference boundaries, generic substitution, associated types, orphan/overlap,
  and conflicting-impl diagnostics were spot-checked against existing tests
  (`gate5_semantic_gaps.rs`, `typecheck.rs`'s own test module) and found adequately covered —
  not subjected to the same exhaustive research-agent audit as WP-C1.1/C1.2 given the WP's time
  budget was consumed by the two substantial bug-fix cycles above; a future pass could still
  deepen this if warranted.
- Semantics: old Gate 2/3 coverage; pending WP-C1.3-C1.5.
- Memory: old Gate 2 M2.4 (ownership/borrows); pending WP-C1.4 full positive/negative corpus
  construction — not yet confirmed to exist at that depth.
- Modules/packages compiler surface: old Gate 2/Phase 1-3 (multi-file modules, `starkpkg.json`
  manifests, dependency resolution/locking per `git log` Phase 1-3 commits). `PKG-004`/`PKG-005`/
  `PKG-006` were incorrectly `missing` in the coverage database — corrected to `partial` under
  WP-C0.3 with real source/test citations; see DEV-002. WP-C1.2 requalification complete
  (2026-07-17): name resolution, module/visibility rules, imports, and re-exports strengthened
  across the full 10-item roadmap matrix; 3 real bugs found and fixed (DEV-004, DEV-006 resolve
  half, DEV-007); 1 new significant finding recorded but not fixed (DEV-019, E-code collisions);
  cross-package coherence checking (SEM-007) and cross-package diagnostic file attribution both
  went from "unverified" to "confirmed working" with real two-package-workspace tests (DEV-021).
  STARK's visibility model confirmed stricter than Rust's (private = exact defining module only,
  no descendant inheritance) — see the dedicated "Design fact pinned down by WP-C1.2" note below.
- Tensor extension: old Gate 4 (`gate4-exit.md`, closed 2026-07-15, "no known deviations")
  covers syntax/resolution/static checking + bounded ONNX metadata decode. Old Gate 7
  (`gate7-decision.md`) added symbolic/computed dimensions and value-range semantics with a
  13/13 defect-detection result. Both predate the new C-numbering; WP-C1.x does not re-audit
  extension code (Core-only scope), but WP-C9.1/C9.2 will need this as input later.

## Known deviations — open index
Canonical ledger (full structured entries, all 66 numbered deviations):
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`. The per-deviation narrative that used to live in
this file (seed list + WP-C1.1/C1.2/C1.3 addition sections) is archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020); the ledger remains the
single source of truth.

Open as of 2026-07-19 (post-WP-C4.5f-3):
- DEV-005 — `starkc` vs `stark` check/run warning-gating drift. Open, unowned since Gate C1.
- DEV-010 — LSP hover/definition/references are protocol stubs. Owner: WP-C8.2/C8.3.
- DEV-011 — doc comments are lexer trivia, not AST/HIR metadata. Unscheduled; needs a scoped
  proposal.
- DEV-012 — VS Code extension UI never interactively verified. Owner: WP-C8.7.
- DEV-017 — 39 of 59 legacy coverage rules still lack function-level positive/negative evidence
  classification (tooling exists; classification unscheduled).
- DEV-067 — bounded generic parameters lose their bounds at intra-generic call sites (E0500)
  and behind `&T` receivers (E0302); over-rejection only, pre-existing, surfaced by WP-C4.5c's
  differential tests. Owner: a later C4.x increment (with generic method monomorphisation).
- DEV-069 — front end + HIR interpreter are not multi-file-span-clean: cross-file spans are
  read against the entry file, so cross-file methods mis-resolve, cross-file literals
  mis-parse, and cross-file field reads fail; found by WP-C4.5f-3c's multi-file work. The MIR
  lowering itself is multi-file-clean (`ProgramMeta`). Owner: a future front-end WP.
- Informational, not owed a fix: DEV-SEED-008 (two hand-rolled JSON parsers), DEV-SEED-014
  (no attribute syntax — deliberate scope fact).

Closed 2026-07-19: DEV-060 (CD-024); DEV-061/062/063 — the function-value cluster — in the
CD-027 pre-C4.1 correction pass; DEV-064 (undetermined-generic rejection, WP-C4.5c, E0004);
DEV-065/066 (C4.5b oracle fixes). See `KNOWN-DEVIATIONS.md`.

## Design fact pinned down by WP-C1.2 (not a deviation, recorded so it isn't re-discovered)
STARK's visibility model is **stricter than Rust's**: per `07-Modules-and-Packages.md` §Visibility
("items are private to their defining module by default"), a private item is visible **only**
within its exact defining module — there is no Rust-style "visible to the defining module and
all its descendants." Confirmed by the pre-existing `module_paths_imports_and_visibility_are_
enforced` test (root cannot access a private item of its own direct child module) and by three
new WP-C1.2 tests (`super_and_crate_navigate_correctly_from_a_nested_module`,
`private_item_is_not_visible_from_a_descendant_module`,
`pub_use_single_level_reexport_is_visible_from_outside`) — the first drafts of the latter two
tests were written assuming Rust-style descendant-inherits-privacy semantics and failed against
the real implementation, which is what surfaced this. Any future WP writing STARK test fixtures
involving nested modules and private items should assume this stricter model.


## Architecture decisions
- AD-001 [pre-existing, old Gate 5] Native artifact-deployment backend is **ONNX Runtime via the
  `ort` crate**, pinned `=2.0.0-rc.12`, statically linked, CPU execution provider only
  (`starkc/docs/gate5-backend-decision.md:11`). IREE/Cranelift/TVM explicitly considered and
  deferred at that time. This is a decision about the *tensor artifact deployment* backend, not
  a decision about general Core native compilation — the two must not be conflated (see CD-002).
- AD-002 [pre-existing] ONNX decoding uses a hand-written protobuf reader with zero new runtime
  dependencies beyond `sha2` (for checksum verification); `ort`, `tract-onnx`, and `onnx-pb`
  crates were evaluated and rejected (`starkc/docs/gate4-design.md:158-169`). `starkc`'s own
  `Cargo.toml` has exactly one dependency, `sha2`, and forbids `unsafe_code` at the lint level.
- AD-003 [pre-existing] Both CLI binaries (`starkc`, `stark`) hand-roll argument parsing against
  a `USAGE` const rather than using `clap` or another CLI-parsing crate (confirmed: no `clap`
  entry anywhere in `Cargo.toml`/`Cargo.lock`).

## Native backend selection
- Status: **SELECTED** (WP-C3.4, owner CE5 decision, 2026-07-19).
- Selected strategy: **generated Rust/C** — generated Rust as the initial production backend
  behind verified MIR, with a **backend-neutral MIR contract that keeps `SELECT-DIRECT`
  (Cranelift) open as a C7-gated migration** (charter §1.6 rule 9, no lock-in). Decision +
  full three-way analysis: `starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`;
  recorded as CD-026.
- Architecture commitments (roadmap WP-C3.4): emitter consumes **verified MIR** (not typed HIR);
  small STARK runtime library (print/panic/trap glue); Rust owns MVP value layout + calling
  convention; Native Provider ABI (C5.1) as `extern "C"` provider calls from generated Rust;
  Tier-1 targets first (linux-x64, macos-arm64) via rustc; debug/trap file:line via a STARK-span
  → generated-Rust-line → rustc-debug-info table; unsupported-MVP closure (floats/`?`/tuple
  patterns/traits/Drop/refs/Vec/HashMap/fn-values) tracked into C4.5/C5/C6.
- **Accepted trade (recorded):** `stark build` requires a full `rustc` toolchain as a permanent
  build dependency, and builds are slower than the direct backend. Acceptable for STARK-as-
  research-language; **re-evaluate the backend choice at C7** if the self-contained-compiler /
  systems-platform goal becomes primary (same evidence-gated pattern as the LLVM decision).
- Workload: 23-item frozen set (`NATIVE-CORE-ARCHITECTURE.md` §5), items 1-10 mapped to the
  frozen `exec_snapshots` corpus v1.0.0 (semantic oracle), items 11-23 specified reference
  programs. Two properties (fn-value Eq/Ord/Hash participation, monomorphised-generic fn-value
  identity) must be settled from the frozen spec or by CE1/CE2 before selection (CD-022).
- Spike evidence so far:
  - **WP-C3.2 generated-Rust (done):** 4/17 frozen corpus cases lower and match the interpreter
    exactly (arithmetic/precedence, loops/for/break/continue, multi-width ints, Int8-overflow
    trap→abort parity); 0 semantic mismatches on supported cases; 13/17 cleanly reported
    unsupported; mean rustc 87 ms/case. Liabilities unresolved (not falsified): rustc
    build-dependency weight, compile-time scaling, exe size, debug-info trap mapping, and the
    unsupported breadth (aggregates/generics/traits/refs/Drop/fn-values). Report:
    `starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md`; artifact `tests/spike_genrust.rs`
    (isolated, disposable).
  - **WP-C3.3 direct Cranelift (done):** 3/17 frozen corpus cases lower and match the interpreter
    exactly (arithmetic, loops/for/break/continue, Int8-overflow trap→abort parity); 0 semantic
    mismatches; 14/17 unsupported (same families as C3.2 plus unsigned ints — spike is
    signed-only, hence 3 vs C3.2's 4). Produces a real standalone native executable (Cranelift
    object + `cc` link). Codegen ~2 ms/case (phase-only), link ~47 ms/case; **defensible
    end-to-end ~49 ms vs rustc ~87 ms ≈ 1.8× on this tiny workload — explicitly NOT a general
    performance multiple** (charter caution; see the report's timing caveat — the raw 2-vs-87
    codegen ratio is not like-for-like). No rustc build dependency. Finding: Cranelift 0.133 needs
    rustc ≥1.94 (>1.93 here) → pinned 0.110, an MSRV-churn maintenance cost. Higher glue than
    generated-Rust (we own CFG/SSA/overflow/Drop/layout); weaker out-of-box debug-info; but the
    bigger beneficiary of the mandatory MIR (MIR ≈ Cranelift's own block/terminator model).
    Report: `starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md`; artifact
    `tests/spike_cranelift.rs` + dev-only Cranelift deps (isolated, disposable).
- **Breadth run (2026-07-19):** generated-Rust extended to structs/impl-methods/generics/
  Option/Result/match/String → **8/17** frozen corpus cases (all matching), via ~250 lines of
  mechanical text emission (rustc absorbs monomorphization/layout/ABI/Drop). Cranelift breadth
  **measured at the struct boundary, not fully implemented** — struct-by-value needs stack-slot
  layout + field offsets + sret ABI; enums need tagged-union layout; generics need a
  monomorphization engine; String/Vec need a runtime — each a subsystem the direct backend owns.
  Cranelift stays 3/17. **Key WP-C3.4 caveat: most of that direct-backend breadth cost is
  mandatory MIR work anyway (Gate C4), so the HIR-level comparison overstates the direct
  backend's long-run cost.** Full head-to-head:
  `starkc/docs/compiler/spikes/WP-C3-breadth-comparison.md`. (Implementing Cranelift
  struct-by-value is a bounded ~150-200-line follow-up if an exact struct head-to-head number is
  wanted.)
- Both spikes done; the tradeoff is symmetric and matches the §4 hypothesis: generated-Rust =
  low glue + free cross-platform/debug-info + broad correctness cheaply + heavy rustc dep; direct
  = fast builds + no rustc + ABI control + biggest MIR beneficiary, but owns monomorphization/
  layout/drop/runtime. Neither falsified nor cleared; WP-C3.4 selects (CE5, owner).
- Evidence: see CD-002 for the closest existing evidence (old Gate 6/7 tensor/ONNX-deployment
  track) — informative precedent for methodology, not a substitute (CD-004).

## Diagnostic codes allocated or changed
- **MIR-0001..MIR-0013** [WP-C4.3, 2026-07-19] First allocation of the `MIR-xxxx`
  compiler-internal namespace (charter §5.1): 0001 target OOB, 0002 local OOB, 0003 projection
  type, 0004 assignment/operand type, 0005 call/checked signature, 0006 bare unsized, 0007
  possibly-moved use, 0008 discriminant/variant misuse, 0009 drop/drop-flag, 0010 index-proof
  discipline, 0011 FnPtr arithmetic/comparison, 0012 reserved (runtime-set violation —
  structurally impossible while RuntimeFn is a closed enum; reserved for serialized MIR), 0013
  invalid FileId in SourceInfo. These are internal invariant failures (lowering bugs), never
  user-source diagnostics. Full map: `src/mir/verify.rs` header + WP-C4.3.md.
- **E0008** [WP-C1.5] Integer literal out of range for its type (suffixed literal exceeds its
  suffix's representable range, or an unsuffixed literal exceeds `Int64`). See DEV-015.
- **E0009** [WP-C1.5] Array repeat count (`[value; count]`) is not a compile-time constant
  expression.
  Both registered in `04-Semantic-Analysis.md`'s normative Error Categories table
  (`STARK-Core-v1.md` regenerated in the same change). No codes allocated or changed by any other
  WP under this governance framework yet. Existing (pre-governance-framework) normative
  `E####`/`W####` codes are inventoried as part of WP-C0.1 (`starkc/src/diag.rs`), not duplicated
  here.

## Evidence inventory
- `starkc/docs/gate1-exit.md` through `gate7-decision.md` — old-numbering gate evidence, see CD-001/CD-002.
- `STARKLANG/tests/spec-fixtures/manifest.toml` — 113-entry spec-fixture corpus (directly
  re-counted 2026-07-19; the "121-fixture" figure this line carried from the C0 audit had
  drifted), verdict census in
  Repository baseline above.
- `cargo test --workspace --all-targets --all-features` output (2026-07-17 audit run) — 383
  passed / 0 failed / 2 ignored, full per-suite breakdown to be carried into
  `starkc/docs/dev/compiler-map.md` (WP-C0.1).
- `STARKLANG/conformance/core-v1-coverage.toml` — 59 rules, 53 implemented / 6 partial / 0
  missing, **integrity-audited under WP-C0.3** (duplicate-ID check, spec-chapter-validity check,
  4 stale `missing` entries corrected with cited evidence). `python3 starkc/scripts/
  check-conformance.py` output (2026-07-17, post-correction): 0 errors, 0 warnings.


## File inventory for current gate
C3-ENTRY (active transition): `STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md` (transition
work package, created 2026-07-19 under CD-020), `.github/workflows/ci.yml` (baseline widened
under CD-020), `STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (new archive).
Closed-gate file inventories (C0/C1): archived verbatim in the state-archive file; per-gate
evidence in the C0/C1/C2 exit reports.

## Follow-ups
- [ ] WP-C0.2 carry-forward (governance-process question, unresolved): gate7-decision.md's "No
      LSP work or language expansion is authorized" text was apparently overridden for WP8.1-8.5,
      but no explicit owner override record exists. Owner should either backfill a decision
      record or confirm WP8.x was tooling, not "language expansion" in Gate 7's sense.
- [ ] DEV-005: pick one warning-gating policy for `starkc check`/`run` vs `stark` — still
      unowned; candidate for C3-ENTRY or a small pre-C3 correction.
- [ ] WP-C8.2/C8.3: implement real LSP hover/definition/references (DEV-010).
- [ ] WP-C8.7: interactive VS Code Extension Development Host validation (DEV-012).
- [ ] WP-C1.1 follow-up (not blocking): underscore-placement rules for binary/octal literals
      untested; no max-value-per-suffix positive test for the 8 int / 2 float suffixes.
- [ ] DEV-017 remainder: classify the 39 unclassified legacy coverage rules (unscheduled).
- [x] DEV-060: dispose before C3 workload freeze (C3-ENTRY blocker). **Closed 2026-07-19,
      CD-024 — fixed in `borrowck.rs::method_receiver`.**
Completed follow-ups through Gate C2 are archived verbatim in the state-archive file.

## Gate exit summaries
- C0: **PASS** (2026-07-17). Bootstrap, current-state audit, and authority repair complete. Full
  report: `starkc/docs/compiler/C0-exit-report.md`. Four stale documents corrected (`CLAUDE.md`,
  root `README.md`, `starkc/README.md`, `STARKLANG/docs/PLAN.md`); conformance database
  integrity-audited with 4 staleness errors fixed (DEV-002, closed); 10 confirmed deviations
  recorded with full structured detail in `starkc/docs/conformance/KNOWN-DEVIATIONS.md`; module-
  by-module compiler map produced (`starkc/docs/dev/compiler-map.md`). Explicit non-claim: no
  conformance percentage from this gate is trusted for Core v1/tensor v0.1 conformance purposes
  — see exit report's "No conformance percentage is trusted" section. Next: Gate C1.
- C1: **CORE-FRONTEND-CONFORMING-WITH-LISTED-DEVIATIONS** (2026-07-17/18). Full report:
  `starkc/docs/compiler/C1-exit-report.md`. Six requalification WPs closed (lexical/syntax, name
  resolution/modules/visibility, types/generics/traits, ownership/borrowing/drop checking,
  control flow/patterns/constants/numerics, conformance evidence generator); 12 of 23 deviations
  closed, 2 partially closed, 9 open and non-soundness-relevant. This entry backfilled during
  WP-C2.13's consistency sweep — not recorded here at the time of C1's own close. Next: Gate C2.
- C2: **CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS** (2026-07-18). Full report:
  `starkc/docs/compiler/C2-exit-report.md`. Reference-execution contract, abstract machine, and
  future-boundaries specifications written from scratch; all 24 high-cost open questions
  approved; 166-row completeness inventory has zero absent/contradictory/unclassified rows (6
  pending-owner-approval governance-only); 33 deviations closed this gate (the largest body of
  runtime-semantics fixes in the compiler track's history, including DEV-053/054 — a bare `None`
  pattern silently matching any value with wrong runtime output, the most severe finding to
  date), 8 remained open and non-soundness-relevant at gate close (see the open index above
  for the current set). WP-C2.12's differential corpus is
  representative, not exhaustive — explicitly disclosed, not disqualifying (cross-backend replay
  is blocked behind Gate C3 by the roadmap's own dependency order). Next: Gate C3, WP-C3.1.

---

## Session records
Records for WP-C0.0 through the Post-Gate-C2 correction brief Issues 1-5 (2026-07-17 through
2026-07-18) are archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020). Gate-level evidence
remains in the C0/C1/C2 exit reports. Records below start at the most recent still-live
transition context.

### Post-Gate-C2 correction brief — Issues 6-8 (DEV-051, DEV-052, DEV-055) — 2026-07-19
DONE: user said "fix them" (Issues 6-8, previously deferred). Reproduced and closed all three
with real fixes, one working method throughout: reproduce on current head, isolate root cause by
reading the relevant resolver/checker code rather than guessing, fix, add regression tests, run
full verification, update docs.
- **DEV-055** (fixed first, most precisely diagnosed already): `use Color::*;`/`use
  Color::{Red, Blue};` silently expanded to nothing when the prefix names an enum rather than a
  real module. Root cause: `resolve_use_tree`'s `Glob`/`Group` arms (and their `_relative`
  counterparts) only ever consulted `submodule_map` (real modules); an enum's variants are
  resolved dynamically through `item_details`, never pre-populated into a module's `items` map.
  Fixed by adding `enum_variant_items`/`resolve_enum_variant_group_item`, wired into both arms in
  both functions. 5 new regression tests (`resolve.rs` x2, `interp.rs` x3), including one
  confirming a variant deliberately left out of a group import correctly stays undefined (rules
  out an overly-broad "import everything" fix).
- **DEV-051**: a trait default method body calling a sibling trait method through `self`
  (`self.name()` inside `fn greeting(&self) { self.name() }`) failed to type-check with `E0302`.
  Root cause: `resolve_method` already had a mechanism for an abstract `Ty::Param` receiver with
  no concrete `impl` to match (a bounded *generic function* type parameter), but it was scoped
  only to that case, never to `self` inside a trait's own default-method body (`current_self_ty
  == Ty::Param("Self")`, checked once, generically, at the trait declaration site). A first
  attempt placed the new check in the same spot as the existing one and it still failed, since
  `self`'s type at that point is `&Self` (a `Ty::Ref`), not bare `Ty::Param("Self")` — moved it
  to after the reference-deref loop, unlike the by-value generic-parameter case. Added
  `current_trait_id` (set alongside `current_self_ty` for trait default bodies) plus two shared
  helpers (`find_trait_method_sig`/`check_trait_member_call`) refactored out of the
  previously-inlined generic-parameter logic. 4 new regression tests, including a
  default-calling-another-default case and a wrong-arg-count case (confirms the fix doesn't
  silently swallow a genuine arity mismatch). **Side finding, NOT fixed** (confirmed pre-existing
  via `git stash`, not introduced by this fix): DEV-060 — calling the same un-overridden default
  method twice on one receiver wrongly raises `E0100 use of moved value` on the second call; two
  calls to an *overridden* trait method or an ordinary inherent method are both unaffected.
  Recorded as a new open deviation with its own regression tests documenting the current
  (defective) behavior and its exact scope, rather than silently worked around.
- **DEV-052**: `Eq::eq(&a, &b)` (fully-qualified call syntax) failed to resolve
  (`E0200 undefined variable 'Eq::eq'`) while the same syntax worked for a user-declared trait.
  Root cause: `resolve_path_relative`'s multi-segment loop only continued past a first segment
  resolving to `Res::Item` (a real trait declaration item, member indexed against
  `ItemDefDetail::Trait`); a `CoreTrait` (`Eq`, `Ord`, ...) has no such declaration item at all.
  Fixed by adding `Res::CoreTraitMember(CoreTrait, Span)`, resolved via a new
  `core_trait_method_name` table (one fixed callable method name per `CoreTrait`: `Eq`→"eq",
  `Ord`→"cmp", `Hash`→"hash", `Clone`→"clone", `Display`→"fmt", `Default`→"default"). Typecheck
  (`check_qualified_core_trait_call`) finds the matching impl's own method signature directly
  (no shared trait declaration to instantiate from, unlike the user-trait case), matching impls
  by trait-ref source text against a new `core_trait_source_name` table (mirroring
  `ty_satisfies_operator_bound`'s existing approach). The interpreter side needed no new
  impl-scanning logic at all: `call_qualified_core_trait` reuses the *exact* `find_method(...,
  Some(Res::CoreTrait(_)))` lookup the `==`/`<` operator sugar already calls for these traits — a
  qualified call is just an explicit spelling of the same dispatch. 4 new regression tests
  (`Eq` and `Ord`, an unimplemented-trait rejection, and a guard confirming the pre-existing
  user-trait qualified-call path is unaffected).
FILES: `starkc/src/resolve.rs` (DEV-055's `enum_variant_items`/`resolve_enum_variant_group_item`;
DEV-052's `core_trait_method_name` table and path-resolution wiring; both regression tests),
`starkc/src/typecheck.rs` (DEV-051's `current_trait_id` field and `find_trait_method_sig`/
`check_trait_member_call` helpers; DEV-052's `check_qualified_core_trait_call`/
`core_trait_source_name`; all three fixes' regression tests plus DEV-060's documentation test),
`starkc/src/interp.rs` (DEV-055/DEV-051 end-to-end regression tests; DEV-052's
`call_qualified_core_trait`; DEV-060's two scope-confirming companion tests),
`starkc/src/hir.rs` (new `Res::CoreTraitMember` variant), `starkc/src/analysis/query.rs`
(exhaustiveness update for the new `Res` variant), `starkc/docs/conformance/
KNOWN-DEVIATIONS.md` (DEV-051/052/055 marked resolved with full root-cause writeups; new
DEV-060 opened; count line updated to 58), this file.
RULES: none — three runtime/type-check-semantics corrections against already-normative rules
(trait default-method dispatch and fully-qualified trait-call syntax per `03-Type-System.md`;
glob-import name resolution per `07-Modules-and-Packages.md`); no conformance-database rule
citation or normative specification text changed.
DECISIONS: none new as CD/AD records. All three are spec-consistent corrections under Charter
§2.2 Sonnet-level autonomy — each makes a previously-rejected legal program accepted and correct,
none weakens an existing check or changes accepted behavior in a way that admits an unsound
program.
EVIDENCE: MANUAL + REG — every fix's original bug and every new regression scenario was run
against the actual compiler (not inferred from code reading alone); DEV-060's pre-existing,
unrelated-to-DEV-051 status was independently confirmed via `git stash` against the pre-fix head
before being recorded, not assumed. `cargo test --workspace --all-targets --all-features`:
**594 passed / 0 failed / 2 ignored** (up from 578/0/2 pre-this-pass, exactly the 16 new tests
across the three fixes — see each fix's own count above — zero regressions elsewhere). `cargo fmt --all -- --check` clean. `cargo clippy --workspace --all-targets
--all-features -- -D warnings` clean. `python3 scripts/check-conformance.py` re-run clean
(89.8%/53-of-59, unchanged -- none of these three fixes touch the conformance evidence database).
NEXT: no further work authorized this pass. DEV-060 (new, open) and DEV-009/DEV-022/DEV-023/
DEV-024 (long-open, C2.8/C2.9-owned) are the remaining known deviations without a fix.

### C3-entry governance-repair pass (CD-020) — 2026-07-19
DONE: full scope of CD-020 (see decision log): WP-C3-ENTRY.md created and wired into the
roadmap's C3-ENTRY section; WP-C4.4/C5.6/C6.5 amended to carry transferred WP-C2.12
obligations; CI widened to the C3-ENTRY baseline command forms plus new spec-regeneration
(`build-core-spec.py --check`) and named execution-snapshot steps; KNOWN-DEVIATIONS.md tail
summary corrected (DEV-009/022/023/024 were resolved by WP-C2.11, not open — the preceding
Issues 6-8 session record's own NEXT line repeats that stale claim and is corrected by this
note, left in place per append-only convention); state header head/fixture-census corrected
(`9e85396`, 113 entries/parse-pass 65); charter §5.3 dangling refs, commit-policy step, and
WP-C6.4 tier label fixed; SYSTEMS-ROADMAP.md gained the P1-relationship section; this file
compressed 3,145 → ~700 lines with all removed material verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md`; C2-exit-report open-deviation
table given a dated post-gate update note.
FILES: COMPILER-STATE.md, STARKLANG/docs/compiler/COMPILER-CHARTER.md,
STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md (new),
STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md (new),
STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
starkc/docs/compiler/C2-exit-report.md, STARKLANG/tools/build-core-spec.py,
.github/workflows/ci.yml.
RULES: none — no normative rule, compiler, or interpreter change; governance surface only.
DECISIONS: CD-020.
EVIDENCE: `python3 STARKLANG/tools/build-core-spec.py --check` clean twice (deterministic);
`cargo fmt --all -- --check` clean; `cargo test --test exec_snapshots` 3 passed / 0 failed;
line-count arithmetic for the compression verified (588 kept + 2,557 archived = 3,145
original). Full `cargo test --workspace` not re-run this pass (no code changed); full CI run
of the updated workflow pending — tracked as the remaining CI blocker item in WP-C3-ENTRY.md.
FOLLOW-UP: owner decisions per WP-C3-ENTRY.md blockers 1-2 (six completeness rows, DEV-060);
corpus freeze after DEV-060 disposition; one demonstrated green CI run.
NEXT: WP-C3-ENTRY blocker closure; then C3-entry exit artifact; then WP-C3.1.

### CD-021 roadmap amendment — 2026-07-19
DONE: applied the owner-approved CD-021 amendment (see decision log): WP-C3.1 workload items
16-21 (existing function-value capability), C4.1/C4.3/C4.5 indirect-call ownership, C5.1
function-value ABI items, P1/S5 trap-abort operational report, WP-C10.7 release-blocking
deviation sweep.
FILES: STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md, COMPILER-STATE.md.
RULES: none — no normative Core rule, compiler, or interpreter change; the workload items
reference already-frozen `fn(...)` semantics.
DECISIONS: CD-021 (owner-approved this session).
EVIDENCE: spec/implementation citations verified by direct grep before recording
(03-Type-System.md:198-200,999; 06-Standard-Library.md:243-244,260-262,663-666;
interp.rs:260). No count/enumeration references to the C3.1 workload existed to go stale
("at least:" phrasing confirmed).
FOLLOW-UP: draft the "Callable ABI and Future Closure Compatibility Spike" proposal before
WP-C5.1 (recommended during C3 spike work); WP-C3-ENTRY blockers unchanged and still open.
NEXT: WP-C3-ENTRY blocker closure (six completeness rows, DEV-060, corpus freeze, green CI
run); then C3-entry exit artifact; then WP-C3.1 with the 21-item workload [23 after CD-022].

### CD-022 follow-up amendment — 2026-07-19
DONE: applied the owner-approved CD-022 (see decision log): release-class claim-scope repair
(Compiler Stable vs General-Purpose Stable, CD-019 preserved), WP-C3.1 workload items 22-23
plus the pre-backend-selection Eq/Hash/monomorphised-identity resolution requirement,
state-header field renamed to "Amendment base commit".
FILES: STARKLANG/docs/compiler/COMPILER-ROADMAP.md, COMPILER-STATE.md.
RULES: none — no normative Core rule, compiler, or interpreter change. The two open
function-value properties are flagged for settlement, not settled here.
DECISIONS: CD-022 (owner-approved this session).
EVIDENCE: spec citation verified by direct read before recording (03-Type-System.md:748-749 —
function values are Copy); release-class contradiction verified against the roadmap text
(C7.7 P1 gate vs the vacuous conditional). Workload numbering re-verified contiguous 1-23.
FOLLOW-UP: push to origin and record one green run of the updated CI workflow (last
C3-entry CI blocker item); callable-ABI/closure-compatibility spike proposal still pending,
pre-C5.1.
NEXT: WP-C3-ENTRY blocker closure (six completeness rows, DEV-060, corpus freeze, green CI);
then C3-entry exit artifact; then WP-C3.1 with the 23-item workload.

### C3-ENTRY blockers 1-2 closure — 2026-07-19 (CD-023/CD-024)
DONE: applied both owner-approved decisions from this session. CD-023: six
`pending-owner-approval` completeness rows approved as-is, flipped to `settled` in
`CORE-V1-COMPLETENESS.md`, C2-exit-report.md given a dated post-gate note, WP-C3-ENTRY.md
blocker 1 marked closed. CD-024: DEV-060 root-caused and fixed in `borrowck.rs::method_receiver`
(missing trait-default-body fallback, mirroring typecheck.rs's own `default_fallback`); two new
regression tests plus one rewritten; KNOWN-DEVIATIONS.md, WP-C3-ENTRY.md blocker 2, and the
open-deviation index all updated to reflect closure.
FILES: starkc/src/borrowck.rs (fix), starkc/src/typecheck.rs (rewrote
`repeated_call_to_unoverridden_default_trait_method_is_wrongly_flagged_as_move` to
`_is_no_longer_flagged_as_move`; added `repeated_call_to_unoverridden_mut_default_trait_
method_is_no_longer_flagged_as_move`), starkc/src/interp.rs (added
`repeated_call_to_unoverridden_default_trait_method_executes_correctly`),
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md,
starkc/docs/compiler/C2-exit-report.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md, COMPILER-STATE.md.
RULES: none — no normative Core rule change; this closes a compiler defect where legal,
spec-conforming code was wrongly rejected (availability bug, not a soundness/acceptance bug).
DECISIONS: CD-023, CD-024 (both owner-approved this session).
EVIDENCE: `cargo build` clean; full `cargo test --workspace --all-targets --all-features`
596 passed / 0 failed / 2 ignored (up from 594); `cargo fmt --all -- --check` clean; `cargo
clippy --workspace --all-targets --all-features -- -D warnings` clean; `python3
starkc/scripts/check-conformance.py` re-run, unchanged (89.8%/53-of-59 — DEV-060 was a
runtime/borrowck defect, not a conformance-database entry). Root cause independently isolated
by direct code reading (borrowck.rs's `method_receiver` vs typecheck.rs's `resolve_method`),
not assumed from the ledger's prior "needs its own investigation" note.
FOLLOW-UP: corpus freeze is now unblocked (WP-C3-ENTRY.md required DEV-060 resolved first,
since a fix could legitimately change corpus output) — next actionable step; green CI run still
needs a push to origin.
NEXT: freeze the versioned execution corpus per WP-C3-ENTRY.md's procedure; then push and
obtain a green CI run; then write starkc/docs/compiler/C3-entry-exit.md; then WP-C3.1.

### C3-ENTRY blockers 3-4 closure + gate close — 2026-07-19 (CD-025)
DONE: froze the execution-snapshot corpus and closed the C3-ENTRY transition. corpus.lock
(v1.0.0, 48 files, base 3d12f45) + integrity test `corpus_lock_matches_frozen_snapshot`
(negatively verified). CI green on origin/main @ 3d12f45 (owner-confirmed). Wrote exit artifact
C3-entry-exit.md; flipped Position to Gate C3 / WP-C3.1 / Blocked: none; checked off all
WP-C3-ENTRY Done-when items. Gate C3 is open.
FILES: starkc/tests/exec_snapshots/corpus.lock (new), starkc/tests/exec_snapshots.rs (new
integrity test), starkc/docs/compiler/C3-entry-exit.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md, COMPILER-STATE.md.
RULES: none — freeze/governance only, no Core behavior change.
DECISIONS: CD-025.
EVIDENCE: `cargo test --test exec_snapshots` 4 passed (incl. integrity test); tamper-then-
restore negative check confirms the integrity test fails on drift; `cargo fmt --all -- --check`
and `cargo clippy --test exec_snapshots --all-features -- -D warnings` clean; full workspace
596/0/2 from CD-024 unchanged (corpus freeze adds one test → next full run will read 597/0/2).
FOLLOW-UP: none blocking. Optional pre-C5.1: draft the "Callable ABI and Future Closure
Compatibility Spike" proposal during C3 spike work (CD-021).
NEXT: WP-C3.1 — freeze the 23-item representative workload, define the measurement set, write
STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md. Gate C3 selects backend
architecture (SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), never interpreter-only.

### WP-C3.1 — Architecture hypothesis and workload freeze — 2026-07-19
DONE: authored the Gate C3 setup deliverables. Wrote
`STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md` (new proposals/ dir): the four
separated questions, pipeline context, the frozen-Core decisions that favor native lowering
(trap-abort-no-unwind, no trait objects, non-capturing fn values, borrow-check-before-codegen,
deterministic order), the architecture hypothesis (Candidate A generated Rust/C vs Candidate B
direct Cranelift; leading hypothesis SELECT-GENERATED with explicit falsifiers), the frozen
23-item workload mapped to corpus v1.0.0 (items 1-10) + specified reference programs (11-23),
the risk register (both candidates, per hard construct), the 13-dimension measurement framework,
and the WP-C3.4 decision preview. Created `work-packages/WP-C3.1.md`. Set Native-backend-
selection status to SPIKING; flipped Position Next to WP-C3.2/C3.3.
FILES: STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3.1.md (new), COMPILER-STATE.md.
RULES: none — non-normative proposal; no Core semantics, compiler, or interpreter change. States
a hypothesis, freezes a workload, defines measurements; selects nothing.
DECISIONS: none at CE level. Leading hypothesis (SELECT-GENERATED) is explicitly flagged as
falsifiable orientation for the spikes, not a decision — CE5 backend selection remains the
owner's at WP-C3.4. Flagged per the CE-escalation convention.
EVIDENCE: all 15 corpus-case references + the workspace-relocation test name + the two
metamorphic pair names verified to resolve against the real tree (no dangling pointers).
Interpreter support for the harder workload items confirmed by direct source read: function
values (`Value::Function`, interp.rs:2168 indirect call), file I/O (`Value::File` +
`read_to_string`/`write`, DEV-009 resolved), references/slices. No build/test run needed — no
code changed.
FOLLOW-UP: recommended (not approved) — draft the "Callable ABI and Future Closure Compatibility
Spike" memo during C3 spike work, before WP-C5.1 freezes the ABI (CD-021). The two open fn-value
properties (Eq/Ord/Hash participation, monomorphised-generic identity) must be settled before
WP-C3.4 selection (CD-022).
NEXT: WP-C3.2 (generated Rust/C spike) and WP-C3.3 (direct Cranelift spike) — each implements
the reachable workload subset and reports every measurement dimension + unsupported constructs;
then WP-C3.4 selects under CE5.

### WP-C3.2 — Generated-Rust backend spike — 2026-07-19
DONE: built and ran the generated-Rust backend spike (Candidate A). Isolated HIR→Rust lowerer +
compile/run/diff harness in `starkc/tests/spike_genrust.rs` (charter §2.2 — NOT wired into
`stark build`, adds nothing to the library surface, disposable). Lowers a supported subset
(integer primitives i8..u64 + Bool, trap-checked arithmetic, comparisons/logic, let/mut/assign,
if/while/loop/for/break/continue, block-tail values, non-generic fns + calls, print/println)
from typed HIR to Rust, compiles with rustc, runs, compares stdout+exit-status to the interpreter
oracle over the frozen exec_snapshots corpus v1.0.0. Wrote the spike report
`starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md` (new spikes/ dir) with every WP-C3.2
measurement record + the NATIVE-CORE-ARCHITECTURE.md §7 dimension mapping. Created WP-C3.2.md.
RESULT: 4/17 corpus cases lowered and matched exactly (arithmetic/precedence,
loops/for/break/continue, multi-width ints, Int8-overflow trap→abort parity); 0 semantic
mismatches on supported cases; 13/17 cleanly reported unsupported with reasons; mean rustc
compile 87 ms/case. Candidate liabilities (rustc dep weight, compile-time scaling, exe size,
debug-info trap mapping, unsupported breadth) neither falsified nor cleared — that needs the
C3.3 spike + a breadth run.
FILES: starkc/tests/spike_genrust.rs (new), starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md
(new), STARKLANG/docs/compiler/work-packages/WP-C3.2.md (new), COMPILER-STATE.md.
RULES: none — spike/evidence only. The spike does NOT bypass front-end checks (it consumes
already-validated typed HIR) and does NOT select a backend (WP-C3.4/CE5). No Core semantics,
compiler, or interpreter change.
DECISIONS: none at CE level. Native-backend-selection status stays SPIKING.
EVIDENCE: `cargo test --test spike_genrust` 2 passed; full workspace
`cargo test --workspace --all-targets --all-features` 599 passed / 0 failed / 2 ignored (597 +
the 2 spike tests); `cargo fmt --all -- --check` and `cargo clippy --test spike_genrust
--all-features -- -D warnings` clean. Coverage table reproduced via `-- --nocapture`.
FOLLOW-UP: WP-C3.3 direct-Cranelift spike must run before any candidate comparison; dimensions
3/5/11/12/13 (exe size, runtime perf, monomorphisation, trait dispatch, ref/slice/Drop ABI) need
a breadth run on both candidates. The two open fn-value properties (CD-022) still pending
pre-C3.4.
NEXT: WP-C3.3 — direct Cranelift spike over the same frozen workload with the same measurement
record; then WP-C3.4 selects under CE5.

### WP-C3.3 — Direct (Cranelift) backend spike — 2026-07-19
DONE: built and ran the direct Cranelift backend spike (Candidate B). Isolated HIR→Cranelift-IR
lowerer + object-emission + `cc`-link + run/diff harness in `starkc/tests/spike_cranelift.rs`
(charter §2.2 — NOT wired into `stark build`, disposable). Same frozen workload subset as C3.2.
Produces a real standalone native executable. Added Cranelift dev-dependencies (pinned 0.110 for
rustc-1.93 compat, with a necessity note in Cargo.toml; dev-only, not the shipped surface).
Object emission (not JIT) → no `unsafe` (crate forbids it). Wrote report
`starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md` with the head-to-head table vs C3.2 and
an explicit timing caveat. Created WP-C3.3.md. Native-backend-selection section updated with both
spikes' results.
RESULT: 3/17 corpus cases matched the interpreter exactly (arithmetic, loops/for/break/continue,
Int8-overflow trap→abort parity); 0 semantic mismatches; 14/17 unsupported (same families as C3.2
plus unsigned ints). Timing: Cranelift codegen ~2 ms/case (phase-only, from built IR, no
parse/typecheck/link), `cc` link ~47 ms/case; end-to-end ~49 ms vs rustc ~87 ms ≈ 1.8× on 3
trivial programs — flagged as NOT a general multiple (charter caution). No rustc build dep.
MSRV-churn finding (0.133→rustc 1.94). Higher glue than generated-Rust; weaker debug-info;
biggest MIR beneficiary.
FILES: starkc/tests/spike_cranelift.rs (new), starkc/docs/compiler/spikes/
WP-C3.3-direct-cranelift.md (new), STARKLANG/docs/compiler/work-packages/WP-C3.3.md (new),
starkc/Cargo.toml (dev-deps), COMPILER-STATE.md.
RULES: none — spike/evidence only, no front-end bypass, no backend selection (WP-C3.4/CE5), no
Core/compiler/interpreter change. Cranelift is a dev-dependency only (charter §1.10 note in
Cargo.toml).
DECISIONS: none at CE level. Native-backend-selection stays SPIKING.
EVIDENCE: `cargo test --test spike_cranelift` 1 passed; full workspace 600 passed / 0 failed / 2
ignored (599 + the cranelift spike); `cargo fmt --all -- --check` + `cargo clippy --test
spike_cranelift --all-features -- -D warnings` clean. Coverage + timings via `-- --nocapture`.
FOLLOW-UP: WP-C3.4 needs a breadth run (aggregates/generics/traits/refs/Drop/fn-values) on both
candidates and exe-size/startup/runtime measurement before a confident selection; the two open
fn-value properties (CD-022) still pending pre-selection.
NEXT: WP-C3.4 — backend and runtime architecture selection under CE5 (owner decision):
SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED.

### WP-C3 breadth run (both spikes) — 2026-07-19
DONE: extended the generated-Rust spike across aggregate/generic breadth (structs, impl/methods,
struct literals, field/method access, generics + trait bounds, Option/Result, match + pattern
lowering, String/&str) → 8/17 frozen corpus cases, all matching the interpreter (was 4/17). ~250
lines of mechanical text emission; rustc absorbs monomorphization/layout/ABI/Drop. Cranelift
breadth measured at the struct boundary rather than fully implemented (would need stack-slot
layout + sret ABI for structs, tagged-union layout for enums, a monomorphization engine for
generics, a runtime for String/Vec — each a subsystem), grounded in the built ~600-line Cranelift
lowerer; Cranelift stays 3/17. Wrote WP-C3-breadth-comparison.md (the head-to-head + the caveat
that most direct-backend breadth cost is mandatory MIR work anyway, so the HIR-level comparison
overstates it). Updated WP-C3.2 and WP-C3.3 reports.
FILES: starkc/tests/spike_genrust.rs (breadth extension + updated unsupported-cases test),
starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md, WP-C3.3-direct-cranelift.md,
WP-C3-breadth-comparison.md (new), COMPILER-STATE.md.
RULES: none — spike/evidence only; no front-end bypass, no backend selection, no Core/compiler/
interpreter change.
DECISIONS: none at CE level. Native-backend-selection stays SPIKING.
EVIDENCE: `cargo test --test spike_genrust` 2 passed (match-interpreter now 8/17 + updated
unsupported-cleanly test); full workspace `cargo test --workspace --all-targets --all-features`
600 passed / 0 failed / 2 ignored; `cargo fmt --all -- --check` + `cargo clippy --test
spike_genrust --all-features -- -D warnings` clean.
FOLLOW-UP: optional exact Cranelift struct head-to-head (~150-200-line sret impl); exe-size/
startup/runtime still unmeasured for both; the fair comparison is at the MIR level (Gate C4), not
HIR. The two open fn-value properties (CD-022) still pending pre-C3.4.
NEXT: WP-C3.4 — backend and runtime architecture selection under CE5 (owner): SELECT-GENERATED /
SELECT-DIRECT / REVISE / BLOCKED.

### WP-C3.4 — Backend selection (owner CE5 decision) — 2026-07-19
DONE: drafted the three-way backend-selection analysis
(`starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`) consolidating the WP-C3.1
framework + WP-C3.2/C3.3 spikes + breadth run, with a reasoned recommendation and the required
architecture commitments; presented the decision to the owner (CE5). **Owner selected
`SELECT-GENERATED`** — generated Rust as the initial production backend behind verified MIR,
backend-neutral MIR keeping direct-Cranelift open as a C7 migration. Recorded as CD-026;
Native-backend-selection section → SELECTED / generated Rust/C; created WP-C3.4.md; Position line
advanced to Gate C4 / WP-C4.1. Gate C3 is complete.
FILES: starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md (new),
STARKLANG/docs/compiler/work-packages/WP-C3.4.md (new), COMPILER-STATE.md.
RULES: none — a strategy selection only; does not build MIR, define the MIR contract (C4/CE3), or
fix the runtime ABI (C5.1/CE4). No Core/compiler/interpreter change.
DECISIONS: CD-026 (owner CE5). Native-backend-selection = SELECTED.
EVIDENCE: decision presented and recorded; the supporting spike evidence (WP-C3.2/C3.3/breadth
reports) is unchanged and already committed. No new code; workspace baseline unchanged (600/0/2).
FOLLOW-UP: the disposable spikes (`tests/spike_genrust.rs`, `tests/spike_cranelift.rs`, Cranelift
dev-deps) are retained for now as C3 evidence; remove/rewrite them when the real MIR-consuming
generated-Rust backend lands (they are not production architecture, charter §2.2). The two open
fn-value properties (CD-022) must be settled during C4/C5. Optional: exe-size/startup measurement
and the Cranelift struct head-to-head remain available if C7 re-evaluation needs them.
NEXT: Gate C4 — WP-C4.1 (MIR design review, CE3): define the backend-neutral verified MIR contract
(`STARKLANG/docs/compiler/mir.md`) that the generated-Rust emitter consumes.

### Pre-C4.1 fn-value settlement and correction pass (CD-027) — 2026-07-19
DONE: settled both CD-022 carry-forward properties (TYPE-FN-001 non-participation in
Eq/Ord/Hash → identity unobservable; TYPE-FN-002 generic coercion = instantiate-at-coercion,
both owner-approved) as normative rules in 03-Type-System.md §Function Types; regenerated the
combined spec (fixtures unchanged — prose-only rules); added TYPE-FN-001/002 rows to the
completeness inventory (166 → 168). Discovered by first-ever execution of workload items 16-22
that the whole fn-value feature was a compile-time façade: recorded DEV-061/062/063, got owner
fix-now authorization, fixed all three (interp dispatch arm; Ty::Fn Copy classification in
borrowck+typecheck; Option/Result combinator signatures + consuming interp interception), and
recorded-but-deferred DEV-064 (undetermined-generic coercion, owner C4.5).
FILES: STARKLANG/docs/spec/03-Type-System.md (+ regenerated STARK-Core-v1.md/.html/.pdf),
STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md, starkc/src/interp.rs,
starkc/src/typecheck.rs, starkc/src/borrowck.rs, starkc/docs/conformance/KNOWN-DEVIATIONS.md,
COMPILER-STATE.md.
RULES: TYPE-FN-001, TYPE-FN-002 (new normative, owner-approved CE1/CE2 under CD-027).
DECISIONS: CD-027.
EVIDENCE: full workspace `cargo test --workspace --all-targets --all-features` **605 passed / 0
failed / 2 ignored** (600 → 605: 3 new interp tests, 2 new typecheck tests); `cargo fmt` +
`cargo clippy --workspace --all-targets --all-features -- -D warnings` clean;
`build-core-spec.py --check` in sync; fixture extraction in sync; check-conformance.py
unchanged (89.8%). All empirical claims verified by running the compiler on real programs
before recording (E0500 rejection, T1/T2/T3 failures pre-fix and outputs post-fix, combinator
outputs incl. pass-through sides, undetermined-generic acceptance).
FOLLOW-UP: DEV-064 owned by C4.5. Workload items 16-22 now have a working oracle; item 23
(Copy aggregate with fn-value field) untested — exercise during C4. The spike reports' "fn
values unsupported" rows are unaffected (spikes are frozen evidence).
NEXT: WP-C4.1 — MIR design review (CE3): draft the backend-neutral verified MIR contract
(STARKLANG/docs/compiler/mir.md) for owner review; the generated-Rust emitter consumes it.

### WP-C4.1 — MIR contract drafted (CE3 review pending) — 2026-07-19
DONE: drafted STARK MIR v0.1 (`STARKLANG/docs/compiler/mir.md`, status PROPOSED) covering every
roadmap-required element: monomorphised-only instances with deterministic injective symbol
naming; closed first-order MirTy set (no Param/Infer); return-place body model; places/
projections with CheckIndex-dominates-Index discipline; total (never-trapping) rvalue set with
every trapping operation as a Checked/Trap terminator carrying category + SourceInfo; no
unwinding/cleanup edges anywhere (abort semantics); Drop as a *statement* with ordinary Bool
drop-flag locals; direct/indirect/runtime callees (FnPtr constants per CD-021/CD-027, closed
versioned RuntimeFn surface); mandatory per-statement provenance with explicit FileId (DEV-006
lesson) and labeled synthetic origins; 13 verifier obligations mapped to WP-C4.3 with MIR-xxxx
safe-failure diagnostics; deterministic versioned textual dump. Five judgment calls flagged for
CE3 in §12. Created WP-C4.1.md.
FILES: STARKLANG/docs/compiler/mir.md (new, PROPOSED),
STARKLANG/docs/compiler/work-packages/WP-C4.1.md (new), COMPILER-STATE.md.
RULES: none — non-normative implementation contract, explicitly subordinate to
CORE-V1-ABSTRACT-MACHINE.md; binding only after CE3 approval.
DECISIONS: none yet — CE3 review is the owner's; WP-C4.2 does not open against an unapproved
contract.
EVIDENCE: design-only; no code changed; workspace baseline 605/0/2 unchanged.
FOLLOW-UP: on approval, record a CD entry flipping mir.md to APPROVED and open WP-C4.2 (scalar
HIR→MIR lowering). DEV-064 fix must land in typecheck before instance collection can rely on
full determination (C4.5 at latest).
NEXT: CE3 owner review of mir.md §12's five questions; then WP-C4.2.

### WP-C4.1 CE3 review outcome (CD-028) — 2026-07-19
DONE: owner CE3 review of the MIR v0.1 contract returned **APPROVE WITH REQUIRED CHANGES**;
all three required changes applied and the contract flipped to APPROVED. (1) Drop moved from
Statement to Terminator (`Drop { place, target }`, no unwind edge) — the review correctly
caught that the statement form violated the contract's own totality invariant, since
destructors are user code that may trap/diverge/mutate; the totality invariant is now stated
in full ("statements/rvalues never trap, never call user code, never diverge") and actually
holds. (2) Option/Result changed from opaque Core runtime types to **logical MIR enums**
(`EnumRef::CoreOption`/`CoreResult`, same aggregate/discriminant/match machinery as user
enums; physical layout stays C5.1/ABI; combinators may remain runtime calls) — the opaque form
had let the current interpreter's representation shape the IR. (3) CheckIndex/Index kept split
but the ordinary integer index local replaced with **opaque IndexProof tokens** binding
base+index+length, consumed only by Index projections on the same base (V-IDX-1/2); Vec
indexing stays on runtime ops in v0.1 (mutable length). Approved unchanged: trapping-ops-as-
terminators (with the one-normal-successor/implicit-abort refinement made explicit) and
monomorphised-only MIR (with three qualifications: mangling not a stable external ABI; named
resource limit; deduplicated discovery). Owner decision wordings recorded verbatim in mir.md
§12.
FILES: STARKLANG/docs/compiler/mir.md (APPROVED), STARKLANG/docs/compiler/work-packages/
WP-C4.1.md (closed), COMPILER-STATE.md.
RULES: none — implementation contract, subordinate to CORE-V1-ABSTRACT-MACHINE.md.
DECISIONS: CD-028 (owner CE3).
EVIDENCE: design review only; no code changed; workspace baseline 605/0/2 unchanged.
FOLLOW-UP: none blocking. DEV-064 (undetermined-generic coercion rejection) still owned by
C4.5; the CD-021 callable-ABI memo still recommended pre-C5.1.
NEXT: WP-C4.2 — typed HIR → MIR lowering, scalar core (literals/locals, unary/binary ops,
blocks/assignments, functions/calls, if/loops/break/continue/return, tuples/arrays/structs/
basic enums, pattern matching without advanced drop elaboration), with every MIR instruction
carrying real or labeled-synthetic SourceInfo.

### WP-C4.2 — Typed HIR → MIR lowering, scalar core — 2026-07-19
DONE: implemented the MIR v0.1 data model (`starkc/src/mir/mod.rs`) exactly per the approved
contract — Drop as terminator, logical Option/Result enums (EnumRef::CoreOption/CoreResult),
IndexProof local kind, Checked with one normal successor + TrapInfo, closed RuntimeFn surface,
interned FileId + SourceInfo on every statement/terminator, versioned deterministic dump — and
the scalar-core lowering (`src/mir/lower.rs`): monomorphised-only deterministic deduplicated
instance discovery from main; trapping ops as Checked terminators (int arith/neg, float
div/rem) with float add/sub/mul + comparisons as total rvalues; short-circuit &&/|| as CFG;
if/while/loop/for-range (labeled synthetic provenance)/break/continue/return; direct calls;
FnPtr constants + FnValue indirect calls (CD-021 items 16/17); tuples/arrays/structs
(written-order eval, decl-order aggregation); user enums incl. unit variants + struct-variant
literals; Option/Result construction as logical-enum aggregates and matching via
Discriminant+SwitchInt with VariantField binding; println/print via runtime surface with
uniform checked widening casts. Scalar-core drop restriction: Drop-impl types are Unsupported
(C4.5 owns elaboration). New `pub mod mir` in lib.rs.
FILES: starkc/src/mir/mod.rs (new), starkc/src/mir/lower.rs (new), starkc/src/lib.rs,
starkc/tests/mir_lowering.rs (new, 6 tests), STARKLANG/docs/compiler/work-packages/WP-C4.2.md
(new), COMPILER-STATE.md.
RULES: none — implementation of the approved contract; no Core semantics change; front-end
checks not bypassed (lowering consumes fully-checked typed HIR + TypeTables).
DECISIONS: none at CE level.
EVIDENCE: `cargo test --test mir_lowering` 6/6 (corpus scalar cases expr_stmt__01/__03,
primitive__01/__02, struct_enum_trait__02 lower with structural invariants — sealed
single-terminator blocks, in-bounds targets, valid FileId everywhere; dump deterministic +
versioned; golden mini-dump pinning Checked-Add/Cast/runtime-call/return-place shapes;
fn-value + indirect-call lowering incl. instance discovery of the target; Option lowers as
aggregate+discriminant with no runtime call; generics/strings/methods report clean Unsupported
naming C4.5). Full workspace 611 passed / 0 failed / 2 ignored (605 → 611). fmt + clippy
-D warnings clean.
FOLLOW-UP: golden documents that unsuffixed int literals infer Int32 and println's Int64
runtime signature forces an explicit (infallible, still Checked) widening cast — revisit cast
uniformity only via a contract version bump. Bool matches without a default arm and bitwise
int ops are recorded Unsupported (contract's non-trapping BinOp set lacks int bitwise ops —
flag for the C4.5-era contract addendum + version note).
NEXT: WP-C4.3 — MIR verifier (contract §10's 13 obligations, MIR-xxxx diagnostics, safe
failure); then WP-C4.4 MIR interpreter differential vs the HIR oracle.

### WP-C4.3 — MIR verifier — 2026-07-19
DONE: implemented `starkc/src/mir/verify.rs` — all 13 contract §10 obligations over MirProgram:
CFG/local/projection well-formedness with step-by-step place typing through a new
lowering-populated TypeContext (struct fields + user-enum variant payloads added to MirProgram
as an additive companion table; Option/Result payloads derived from type args); bidirectional
aggregate checking; call/checked/runtime signature checking; V-MOVE-1 as a conservative
whole-local any-path union-join fixpoint dataflow; drop-flag and index-proof (CE3 tokens)
discipline; TYPE-FN-001 enforcement at MIR level (no arithmetic/comparison on FnPtr); V-SRC-1
FileId validity. First MIR-xxxx namespace allocation recorded in the Diagnostic-codes section.
Safe-failure hardening: the negative test suite caught the move dataflow PANICKING on a broken
CFG edge (exactly the unsafe failure the contract forbids) — fixed to skip already-reported
edges; report-and-continue everywhere.
FILES: starkc/src/mir/verify.rs (new), starkc/src/mir/mod.rs (TypeContext + MirProgram.types),
starkc/src/mir/lower.rs (type-context population + hir_field_ty), starkc/tests/mir_verify.rs
(new, 14 tests), STARKLANG/docs/compiler/work-packages/WP-C4.3.md (new), COMPILER-STATE.md.
RULES: none — verifier implements the approved contract; no Core semantics change.
DECISIONS: none at CE level. MIR-0012 reserved rather than allocated (runtime-set violation is
structurally impossible while RuntimeFn is a closed Rust enum; becomes real with serialized
MIR).
EVIDENCE: `cargo test --test mir_verify` 14/14 — positive: all 5 lowerable corpus cases + 3
inline programs (fn-values, Option, structs) verify clean (lowering and verifier as two
independent contract readings agreeing); negative: 13 hand-crafted invalid bodies each
rejected with the specific MIR-xxxx code. Full workspace 625 passed / 0 failed / 2 ignored
(611 → 625: 14 verifier tests). fmt + clippy -D warnings clean.
FOLLOW-UP: V-MOVE-1 whole-local granularity documented as a refinement point (can reject
over-clever legal MIR, never accepts moved-from reads); field-precise tracking when C4.5's
partial moves need it. TypeContext addition noted as additive (no dump/shape change, no
version bump) — fold into the contract text at the next version bump.
NEXT: WP-C4.4 — MIR interpreter + differential harness vs the HIR oracle over corpus v1.0.0.

### WP-C4.4 — MIR interpreter + HIR/MIR differential — 2026-07-19
DONE: implemented `starkc/src/mir/interp.rs` (executes verified MIR: option-slot locals with
loud use-after-move detection via taking Moves; projection reads/writes; Checked terminators
with per-width trap semantics incl. MIN/-1 and CD-006 float div/rem-by-zero; checked numeric
casts; SwitchInt with the lowering's u128 key wrap; direct/indirect/runtime calls; 50M-step
fuel guard; TrapCategory outcomes distinct from internal errors) and the Gate C4 comparator
`tests/mir_differential.rs`: 7 tests running lower→verify→execute vs the HIR oracle — 5
lowerable frozen-corpus cases (byte-equal stdout+status; primitive__02 traps agree), fn-values
(CD-021 items 16/17/22 through MIR), Option/Result logical enums end-to-end, structs/tuples,
div-zero trap, mid-output trap, recursion+loops. `interp::canonical_float` exposed pub so the
MIR runtime formats floats with the oracle's own algorithm (single source, no drift).
RESULT: **zero semantic differences between HIR and MIR execution** across the supported
workload. One comparator-map bug caught by the harness itself (oracle "division by zero" vs
map's "divide by zero") — a harness fix, not an engine disagreement.
FILES: starkc/src/mir/interp.rs (new), starkc/src/mir/mod.rs (module reg),
starkc/src/interp.rs (canonical_float made pub with doc), starkc/tests/mir_differential.rs
(new, 7 tests), STARKLANG/docs/compiler/work-packages/WP-C4.4.md (new), COMPILER-STATE.md.
RULES: none — differential infrastructure; no Core semantics change. The MIR interpreter is
explicitly not a user-facing VM (charter §1.6 rule 11).
DECISIONS: none at CE level.
EVIDENCE: `cargo test --test mir_differential` 7/7; full workspace 632 passed / 0 failed /
2 ignored (625 → 632); fmt + clippy -D warnings clean. The C4.4 comparator condition — HIR
interpreter output/failure == MIR interpreter output/failure — holds for every workload the
scalar-core lowering supports.
FOLLOW-UP: the differential net must widen with every C4.5 construct as it lands (the roadmap's
"generated corpus" + full-corpus replay obligations, carried per CD-018/CD-020).
NEXT: WP-C4.5 — complete Core lowering (generics/monomorphisation, trait dispatch, patterns,
CheckIndex/indexing, strings/Vec/runtime surface, ownership/drop elaboration with real Drop
terminators, panic paths, multi-package linkage), differential-first.

### C4.5a + CD-029 correction pass — 2026-07-19
DONE: (1) WP-C4.5 split per charter §2.2 with the review-adopted increment order (WP-C4.5.md).
(2) C4.5a landed: FnKey instance identity (Top/ImplFn/TraitDefault-per-implementing-type),
method + associated-fn call lowering (receiver-before-arguments), trait dispatch with
inherent > trait-impl > default precedence, Self substitution; interim by-value reference
model documented in code (&self receivers Copy-passed; &mut self cleanly Unsupported until
C4.5b/d); corpus struct_enum_trait__01 now differential-green; 2 new differential tests
(methods/assoc fns incl. repeated &self + consuming self; trait default-vs-override).
(3) CD-029 corrections applied (see decision log): trap provenance end-to-end with exact-span
differential comparison; VerifiedMirProgram wrapper; TypeContext formalized in mir.md §2;
canonical_float spec tests (6, incl. boundary/subnormal/round-trip property).
FILES: starkc/src/mir/{lower,interp,verify}.rs, starkc/tests/{mir_differential,mir_lowering,
mir_verify,canonical_float}.rs (last new), STARKLANG/docs/compiler/mir.md (CD-029 amendments),
STARKLANG/docs/compiler/work-packages/WP-C4.5.md (new), COMPILER-STATE.md.
RULES: none — implementation + contract bookkeeping under the approved MIR contract.
DECISIONS: CD-029.
EVIDENCE: differential 9/9 with provenance comparison live (user-origin trap spans equal the
oracle's exactly in both trap tests); canonical_float 6/6; full workspace 640 passed / 0
failed / 2 ignored; fmt + clippy clean. Differential claim now stated in qualified form.
FOLLOW-UP: generated-Rust backend must consume VerifiedMirProgram when it lands (C5).
NEXT: WP-C4.5b — indexing and references (CheckIndex proof tokens, arrays/slices, real
reference places replacing the interim by-value model, &mut self).

### C4.5b-1 — array indexing with CheckIndex proof tokens — 2026-07-19
DONE: first real exercise of the CE3 proof-token discipline end to end. Lowering: `base[index]`
(reads, writes, loop-indexed access) emits `Checked { CheckIndex, args: [Copy(base_place),
index] }` defining an IndexProof local consumed by `Index(proof)` projections; base evaluated
before index (CD-007); non-place bases materialize a temp; Vec indexing stays runtime-surface,
slices deferred to C4.5b-2. Verifier: NEW same-base binding pass (`verify_index_bindings`) —
every Index(proof)'s place prefix must equal the base its CheckIndex bound (proof→base map;
place prefix equality; the exact rule CD-028's revision demanded beyond dominance), plus
CheckIndex arg typing (base must be Copy(place) of indexable type, index integer). Interp:
CheckIndex evaluates bounds and defines the proof as the checked index; place reads/writes
resolve proofs (writes pre-resolve before the mutable walk). ORACLE FIX (DEV-065, found by the
differential's category↔message mapping need): array OOB reported "use of moved or invalid
field" — now projection-kind-aware "index out of bounds"; diagnostics-only.
FILES: starkc/src/mir/{lower,verify,interp}.rs, starkc/src/mir/mod.rs (PartialEq on
Place/Projection), starkc/src/interp.rs (DEV-065), starkc/tests/{mir_differential,mir_verify}.rs,
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-065 closed; count 63), COMPILER-STATE.md.
RULES: none — implements the approved contract; DEV-065 is diagnostics-only (no
accepted/rejected or trap-behaviour change).
DECISIONS: none at CE level.
EVIDENCE: differential 11/11 (new: array reads/writes/loop-sum agree; OOB trap agrees in
category AND exact source span through the fixed oracle message); verifier 15/15 (new negative:
proof bound to base _1 used to index _2 → MIR-0010). Full workspace 643 passed / 0 failed / 2
ignored; fmt + clippy clean.
FOLLOW-UP: C4.5b-2 (references/slices/&mut self) needs the MIR-interp frame restructure
(cross-frame reference places) — the interim by-value reference model stays until then.
NEXT: WP-C4.5b-2, then C4.5c generics per WP-C4.5.md's increment order.

### C4.5b-2 — real references and the frame-stack MIR interpreter — 2026-07-19
DONE: the interim by-value reference model is gone. MIR interpreter restructured onto an
explicit frame stack; a reference value is a resolved (frame, local, concrete-projection-path);
`Deref` re-anchors place resolution; index proofs resolve in the evaluating frame before any
re-anchor; dangling-frame access is a loud Internal error (defense behind borrowck). Lowering:
`Ty::Ref` converts to real `MirTy::Ref` (peel removed); `&expr`/`&mut expr` lower to `RefOf`
(borrow of a place, never a value read); `*r` reads/writes via `Deref` projections; field
access and method dispatch auto-deref through reference-typed bases; `&self`/`&mut self`
receivers are real Ref-typed params (borrowed at call sites, forwarded when the receiver is
already a reference). The &mut-self Unsupported is gone — a &mut self method now genuinely
mutates the CALLER's local across the frame boundary (differential-verified). ORACLE FIX
(DEV-066, the differential's second front-end find after DEV-065): borrowck consumed a
reference on every deref-read (&mut T non-Copy → "use" became a move), rejecting the canonical
`*r = *r + 1`; both deref paths now availability-check without consuming; the
move-out-of-non-Copy-pointee rejection is unchanged.
FILES: starkc/src/mir/interp.rs (frame restructure, rewritten), starkc/src/mir/lower.rs,
starkc/src/borrowck.rs (DEV-066), starkc/tests/{mir_differential,mir_lowering}.rs,
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-066; count 64),
STARKLANG/docs/compiler/work-packages/WP-C4.5.md (b marked done; slices explicitly deferred to
C4.5e where their consumers live), COMPILER-STATE.md.
RULES: none — implements the approved contract's reference/Deref semantics; DEV-066 restores
spec-legal programs (rejection-of-legal fix, no new acceptance beyond the spec).
DECISIONS: none at CE level.
EVIDENCE: differential 14/14 — all prior tests pass unchanged under the REAL reference model,
plus 3 new: `&mut self` mutating the caller's local (read back both via method and direct
field), `&`/`&mut` arguments with cross-frame writes and derefs, `&mut` to a struct FIELD
(sibling field untouched). mir_lowering negative case swapped (mut-self now supported; `?`
takes its place). Full workspace 646 passed / 0 failed / 2 ignored; fmt + clippy clean.
FOLLOW-UP: none blocking. C4.5b complete.
NEXT: WP-C4.5c — generics and full static dispatch (real Instance.type_args monomorphisation,
deterministic dedup, named resource limit, operator dispatch on generic params, DEV-064's
typecheck rejection).
