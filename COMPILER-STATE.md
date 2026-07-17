# STARK Compiler STATE
Updated: 2026-07-18 after CD-006 (float division-by-zero trap semantics, WP-C1.5)

## Position
Gate: C1  Next: WP-C1.6  Blocked: none
Mandatory compiler path: Core=open (C1 in progress)  MIR=blocked (behind C1/C2/C3)  Native=blocked (behind C1/C2/C3)
Optional tracks: ArtifactInfra=blocked (no second artifact impl yet)  TensorExpansion=blocked (no approved workload, Conditional Track T)

## Repository baseline
- Head: 80ef50415b4a67c6b9f10122d923b0472bf4550c (`implement WP-C1.3 and WP-C1.4: ...`). WP-C1.5's
  own changes are uncommitted as of this session record; commit only on explicit user request,
  per standing workflow.
- Rust toolchain: `starkc/rust-toolchain.toml` pins `channel = "stable"` (no version number, tracks
  stable) with `rustfmt`/`clippy` components. Active environment measured: `cargo 1.93.0
  (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607 2026-01-19)`. `starkc/Cargo.toml` declares
  `rust-version = "1.85"` (crate MSRV). The Gate-5 *generated deployment host* (not `starkc`
  itself) separately requires Rust 1.88 due to the `ort` crate's MSRV
  (`starkc/docs/gate5-backend-decision.md:107-110`) — this does not raise `starkc`'s MSRV.
- Test count / suites: `cargo test --workspace --all-targets --all-features` (starkc/):
  **450 passed, 0 failed, 2 ignored** across 3 unittest binaries + 31 integration-test files
  (up from 383/0/2 and 30 files at Gate C0 close; WP-C1.1 added `span_integrity.rs` + 12 tests,
  WP-C1.2 added 15 more across `resolve.rs`'s inline tests and `gate2_package.rs`, WP-C1.3 added
  8 more across `typecheck.rs`'s and `interp.rs`'s inline test modules, WP-C1.4 added 11 more
  across `gate2_valid.rs` and `gate3_execution.rs`, WP-C1.5 added 21 more to `gate2_valid.rs`).
  Both ignored tests are
  intentionally opt-in (a checksum-pinned live ONNX artifact test in `tests/gate4_onnx.rs`, and
  a live-ORT-download inference test in `tests/gate5_codegen.rs`). Full per-file breakdown
  recorded in `starkc/docs/dev/compiler-map.md` (WP-C0.1; not re-regenerated for the WP-C1.1/
  C1.2/C1.3 deltas — see that file's own scope note).
- Core spec revision: `STARKLANG/docs/spec/` files 00-07, normative per `CLAUDE.md`. Spec
  fixture corpus: `STARKLANG/tests/spec-fixtures/manifest.toml`, 121 entries (parse-pass 67,
  semantic-error 18, notation 30, lex-pass 4, parse-fail 2).
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
  verified from this database alone — flagged for WP-C1.6 (conformance evidence generator) to
  address with a richer schema. **Coverage percentages remain provisional**: "implemented" status
  for any individual rule is not re-verified at Core v1 rule-completeness depth until WP-C1.x; see
  governing rule in `COMPILER-CHARTER.md` §1.5 rule 14 and the explicit no-percentage-trust
  statement this state file and the WP-C0.5 exit report both carry.

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

## Known deviations — open until closed explicitly
(Seed list only — WP-C0.4 owns the full ledger with normative expectation / current behaviour /
impact / workaround / disposition / owning gate for each. Do not treat this seed list as
complete or as substituting for WP-C0.4.)
- DEV-008 [CLOSED, WP-C1.3, 2026-07-17] Equality was pure structural equality on the
  interpreter's internal `Value` enum — no dispatch through a user's `Eq` trait implementation
  at runtime. **Resolved via option 1** (Charter §1.6 rule 3/WP-C1.3 text): implemented normative
  dispatch, not a spec correction — the spec unambiguously specifies `Eq` as a normal,
  user-implementable trait (`03-Type-System.md:389-406`: `trait Eq { fn eq(&self, other: &Self)
  -> Bool; }` with a worked example `impl Eq for Point` containing real per-field comparison
  logic; confirmed identically in `06-Standard-Library.md:107-109`) and states the operator
  desugaring rule explicitly (`03-Type-System.md:516-523`: `==`/`!=` desugar to `Eq::eq`/negation
  except for primitive types, which keep built-in intrinsic comparison). `typecheck.rs`'s
  `require_operator_bound` (pre-fix) already REQUIRED a real `impl Eq for T` to exist for any
  struct/enum `==` to type-check (`Ty::Struct(..)|Ty::Enum(..)` arm searches `hir.items` for a
  matching impl) — so the type checker's own contract already assumed real dispatch; only the
  interpreter was out of sync.
  - **Fix 1 (interp.rs):** `eval_binary` now looks up a resolved `Eq` impl via the existing
    `find_method`/`call_user_method` machinery (already used for ordinary method calls) before
    falling back to structural comparison; only struct/enum values are looked up (primitives and
    `Ty::Core` container types have no user-overridable Eq per "operator overloading for
    user-defined types... is a future extension," `03-Type-System.md:529-530`, so structural
    comparison remains exactly correct for them). Verified with a deliberately non-structural
    custom `eq()` (always returns `true` regardless of field values) — proves real dispatch, not
    coincidental agreement with structural equality.
  - **Fix 2 (typecheck.rs, found while investigating):** `Ty::Core` (Option/Result/Vec/Box) had
    **no arm at all** in `require_operator_bound` — confirmed empirically that `Option<Int32> ==
    Option<Int32>` was unconditionally rejected with E0500 before this fix, despite `Int32`
    obviously satisfying `Eq`. Added a recursive `ty_satisfies_operator_bound` helper checking
    container type arguments; verified both the positive case and that it still correctly
    rejects `Option<NoEqStruct>`.
  - Regression tests: `interp.rs::custom_eq_impl_is_dispatched_not_structural`,
    `::custom_eq_impl_is_dispatched_for_ne_too`, `::option_and_vec_equality_are_structural`;
    `typecheck.rs::option_result_vec_box_satisfy_eq_when_their_type_args_do`,
    `::option_of_non_eq_type_is_rejected`.
- DEV-002 [CLOSED, WP-C0.3, 2026-07-17] `core-v1-coverage.toml` listed `PKG-004`/`PKG-005`/
  `PKG-006` and `STD-004` as `missing` despite working, tested implementations (Phase 1-4A
  commits `7ac8873`/`22db94f`/`a76e32d`/`f8c64b5`). Corrected in `STARKLANG/conformance/
  core-v1-coverage.toml` to `status = "partial"` with real `source`/`tests` citations and an
  inline dated correction note on each entry (not promoted to `"implemented"` — that judgment is
  deferred to WP-C1.6's rule-level conformance generator, since these are broad rules that
  likely need splitting before "implemented" can be asserted with full confidence per Charter
  rule 14). Root cause: the coverage database was seeded once at Phase 0 (2026-07-16, per
  `stark-spec-parity-roadmap.md`) and never updated as Phases 1-4A subsequently closed exactly
  this scope — a "no status drift" (Charter rule 13) violation in the pre-existing process, now
  fixed. The "fourth unidentified missing rule" from WP-C0.0's provisional note is `STD-004`
  (standard traits) — also corrected. `STD-003`/`STD-005` source/tests citations were also
  corrected for accuracy (status unchanged — see `starkc/scripts/check-conformance.py` output
  and `STARKLANG/conformance/core-v1-coverage.toml` inline notes for full detail).
- DEV-009 [CONFIRMED, WP-C0.2, via `starkc/docs/PHASE4E_MATH_RANDOM_IO_IMPLEMENTATION.md:45-58,
  109`] `File` has no first-class runtime representation. `std::fs::File` doesn't implement
  `Copy`/`Clone`/`PartialEq`, so it can't fit as `Value::File(std::fs::File)` without restructuring
  the `Value` enum's move/copy assumptions. Deferred; `IOError` (the other Phase 4E half) shipped.
  Explicitly the one clearly-scoped remaining piece of Phase 4E per that doc. — owner: future
  interpreter-value-model WP (not yet scheduled in C0-C2; revisit at WP-C2.2 interpreter semantic
  repair if in scope, else a dedicated follow-up).
- DEV-004 [CONFIRMED, WP-C0.1] `resolve.rs` tensor-builtin gating bug. `resolve_unqualified`
  (`resolve.rs:1854-1876`, used only for `self`-resolution and struct-literal shorthand fields)
  calls `resolve_builtin` with no `options.tensor()` gate, unlike the correctly-gated
  `resolve_path_relative` (`resolve.rs:682-685`). Bare `min`/`max` unconditionally resolve to
  `Builtin::TensorMin`/`TensorMax`. Concrete trigger: in Core-only mode, a struct-literal
  shorthand field named exactly `min`/`max` with no local of that name in scope silently
  resolves to the tensor builtin instead of failing "undefined variable". Introduced pre-tensor
  (commit `86afc2e`), never retrofitted when the tensor `min`/`max` overload was added (Gate 4);
  the Phase 4E commit (`e58e948`) discovered and explicitly deferred it in its commit message.
  Full detail: `starkc/docs/dev/compiler-map.md` §6. **[CLOSED, WP-C1.2, 2026-07-17]** — same
  `options.tensor()` gate added to `resolve_unqualified`. Regression test:
  `resolve.rs::bare_min_max_shorthand_field_is_gated_by_tensor_extension`.
- DEV-005 [CONFIRMED, WP-C0.1] `starkc` vs `stark` `check`/`run` gating drift. Both binaries call
  the same shared library functions (`parse_with_options`/`parse_package_graph`,
  `resolve::resolve_with_options`, `typecheck::analyze_with_options`, `interp::run`) — no
  reimplemented pipeline logic — but `starkc check` gates progression on `severity != Error`
  (warnings pass) while `starkc run` gates on `diagnostics.is_empty()` (any warning blocks
  execution entirely). A program with zero errors but one parse/resolve warning: `starkc check`
  reports OK/exit 0, `starkc run` on the same file refuses to execute. Separately, `stark` has no
  `--extension` flag at all (hardcoded `LanguageOptions::CORE`) while `starkc` supports
  `--extension tensor` — by design, but means the two binaries compile genuinely different units
  (bare file vs. package graph) under different default options, not interchangeable front ends.
  Full detail: `starkc/docs/dev/compiler-map.md` §2. — owner: WP-C1.x triage, WP-C0.4 ledger
  entry.
- DEV-006 [RESOLVED — resolve half fixed WP-C1.2, borrowck/flow half fixed WP-C1.4, 2026-07-17]
  Multi-file diagnostic provenance loss. `Span` carries no file
  identity (`source.rs:10-13`) and there is no `FileId`/`SourceId` type anywhere in the crate.
  Parse and typecheck stages correctly back-fill per-item file identity onto diagnostics; resolve
  (20 diagnostic sites, zero `.with_file()` calls), flow analysis (`flow.rs:21-24`, file param
  named `_file`, structurally unused), and borrow checking (`borrowck.rs`, single `self.file` for
  the whole crate, no per-item lookup) do not. For any multi-file `stark` package where an
  error/warning originates outside the entry file, resolve/flow/borrowck diagnostics render
  against the wrong filename and wrong line:col (silently — `SourceFile::line_col` clamps rather
  than panicking). Full detail: `starkc/docs/dev/compiler-map.md` §4. **Resolve half fixed
  WP-C1.2**: added `push_diag`/`current_file_arc()` helpers (resolve.rs), mirroring typecheck.rs's
  own if-none backfill pattern; all 20 `self.diags.push` call sites converted to `self.push_diag`.
  Verified with a same-package regression test AND a new cross-package test
  (`gate2_package.rs::test_cross_package_diagnostic_reports_dependency_file_not_root_file`,
  confirming the trickier case where the file identity must come from a dependency package, not
  just a different file in the same package). **Flow/borrowck half fixed WP-C1.4**: same
  `push_diag`/per-item `self.file` swap pattern (via `hir.item_files`) applied to both
  `borrowck.rs` and `flow.rs`; `flow::check`'s previously-unused `_file` parameter is now a real,
  used field. Verified with two new regression tests
  (`gate2_valid.rs::test_borrowck_diagnostic_in_nonroot_file_reports_correct_file`,
  `test_flow_diagnostic_in_nonroot_file_reports_correct_file`). DEV-006 fully resolved.
- DEV-007 [CONFIRMED, WP-C0.1] `resolve.rs` glob-import (`use mod::*`) nondeterminism. Glob
  expansion copies from an unsorted `HashMap<String, Res>` at two call sites (`resolve.rs:
  475-479`, `:536-540`); which of two colliding glob-imported names is treated as "first"
  (silently wins) vs. "second" (flagged `E0204`) depends on Rust's randomized per-process
  `HashMap` seed. Diagnostics are never sorted before rendering. Affects reproducibility of
  `starkc check` output for programs with colliding glob re-exports. Full detail:
  `starkc/docs/dev/compiler-map.md` §3. **[CLOSED, WP-C1.2, 2026-07-17]** — both call sites now
  sort `items_to_copy` by name before iterating. Regression test:
  `resolve.rs::glob_import_collision_diagnostics_are_deterministic` (25 repeated resolves,
  asserts identical diagnostics every time).
- DEV-010 [CONFIRMED, WP-C0.2, via `starkc/docs/PHASE8_GRAMMAR_GAPS.md:14-37`] LSP
  hover/definition/references are protocol stubs, not real features. The endpoints exist and
  respond correctly per JSON-RPC (`textDocument/hover`, `textDocument/definition`,
  `textDocument/references` all wired into `handle_request`, compiled `TypeTables` cached per
  open document), but the handlers don't use that data: hover returns a raw `line:character`
  string instead of the inferred type/signature; definition/references return `null`/`[]`
  unconditionally. Root cause: no span→node lookup exists (walking HIR/AST for the innermost
  node at a byte offset is real work, not wiring). This is the exact situation Charter WP-C8.2
  warns against ("a cursor-coordinate string is a stub, not hover support") — already true
  today, not a hypothetical future risk. — owner: WP-C8.2/C8.3.
- DEV-011 [CONFIRMED, WP-C0.2, via `starkc/docs/PHASE8_GRAMMAR_GAPS.md:41-67,210-217`] Doc
  comments (`///`) have no AST/HIR storage — they're lexer trivia (`Comment`/`CommentKind`,
  `lexer::tokenize_with_comments`) re-associated with item spans by source position at
  formatter/doc-gen time, not first-class attached metadata. Nothing downstream of parsing
  (resolve, typecheck, interpreter) can see them. Matches the WP-C0.4 seed list item verbatim
  ("doc comments existing as re-associated trivia rather than AST/HIR metadata") — confirmed.
  — owner: not scheduled; would need a scoped proposal (adds an AST/HIR field, not just a
  tooling change).
- DEV-012 [CONFIRMED, WP-C0.2, via `starkc/docs/PHASE8_GRAMMAR_GAPS.md:194-204`] VS Code
  extension UI behavior (status bar, command palette, format-on-save actually firing, hover
  popups) has never been interactively verified — no `code` CLI / Extension Development Host
  available in the implementing environment. Verified so far: TypeScript correctness
  (tsc/ESLint/esbuild) and raw LSP JSON-RPC exchange only. Matches Charter WP-C8.7 exactly
  ("protocol tests alone do not prove UI behaviour"). — owner: WP-C8.7.
- DEV-013 [CLOSED, WP-C1.3, 2026-07-17] `STD-004` (standard traits: Clone/Hash/Default/Display/
  Error/Iterator) exhaustiveness audit. Findings, precise per sub-question:
  - **`Error` trait bound checking: confirmed working, not broken.** `resolve.rs:2079` classifies
    `Error` as `CoreTrait::Error`; `typecheck.rs`'s `satisfies_bound` (the generic trait-bound
    checker used for `fn f<T: Error>(...)`, distinct from `require_operator_bound`) handles any
    struct/enum trait name generically via a real "does `impl <TraitName> for <Type>` exist in
    HIR" search — it was never hardcoded to a fixed string list the way the narrower operator-
    desugaring bound checker is. Verified empirically end-to-end: `fn describe<E: Error>(e: E) ->
    String { e.message() }` called with a real `impl Error for MyError` both type-checks and
    executes correctly. The original DEV-013 seed's "not seen in the bound-name list" observation
    was checking the wrong function (`require_operator_bound`, which only handles the 3 operator
    traits Eq/Ord/Num, not general trait bounds).
  - **`Clone`/`Hash`/`Display` as bounds: confirmed working** (same generic mechanism).
  - **`Clone` as a *callable method* on compiler-builtin types: confirmed BROKEN, now fixed.**
    `s.clone()` on a `String`, `Vec`, `Option`, `Result`, `HashMap`, `HashSet`, `Range`, or
    `IOError` value failed with E0303 "method call on non-struct/enum type" — `Clone` satisfied
    as a *bound* (recognized), but `.clone()` had no method-signature entry in
    `core_method_signature` (typecheck.rs) or dispatch case in `call_core_method` (interp.rs) for
    any builtin type; only struct/enum values with a hand-written `impl Clone for T` worked,
    since those go through ordinary impl-block method resolution. **Fixed**: added a generic
    `.clone()` case in both (interp.rs's is a single dispatch point reusing `Value`'s existing
    derived Rust `Clone`, since none of these types have user-overridable Clone bodies per
    "operator overloading for user-defined types... is a future extension"). Scoped to
    genuinely value-like CoreTypes; iterator/cursor CoreTypes and `Random` deliberately excluded
    (cloning stream/cursor state is unspecified, would be new semantics). Regression tests:
    `interp.rs::clone_works_for_builtin_core_types`.
  - **Default trait method bodies: confirmed BROKEN, now fixed** (found while testing the trait
    family broadly, not originally part of DEV-013's own question, but squarely inside WP-C1.3's
    own checklist — "default methods" is explicitly named). A trait method declared with a real
    body (`03-Type-System.md` trait defaults; HIR already carries it as `TraitItem::Method {
    body: Some(_), .. }`) was never used as a fallback when an implementing type didn't override
    it — confirmed empirically as E0302 "method not found" at typecheck time (interp.rs's
    `find_method` had the same gap, one layer down). **Fixed in both stages**: typecheck.rs gained
    a `default_fallback` search (after ordinary impl-override candidate collection finds
    nothing, before concluding "not found") that looks up the implemented trait's own default
    body and type-checks the call against its signature; interp.rs's `find_method` gained the
    analogous fallback for execution. Verified both that an un-overridden default runs AND that
    an overriding impl still correctly takes precedence. Regression tests:
    `interp.rs::default_trait_method_runs_when_not_overridden`,
    `::overriding_impl_takes_precedence_over_trait_default`.
  - **Whether users can hand-write `impl <StdTrait> for T` vs. only auto-derive: confirmed —
    yes, hand-written impls are the normal, spec-shown mechanism** (see DEV-008's closure; there
    is no separate "auto-derive-only" mode for any of these traits in Core v1 as currently
    specified or implemented).
  - **New, NOT fixed, deviations found while closing this one** — see DEV-023 and DEV-024 below.
- DEV-SEED-014 No attribute syntax (`#[test]`, `#[ignore]`, etc.) exists in Core v1 — confirmed
  deliberate scope restriction, not a bug (`PHASE8_GRAMMAR_GAPS.md:83-113`): no `#` handling in
  the lexer, no attribute AST node, nothing in `01-Lexical-Grammar.md`/`02-Syntax-Grammar.md`.
  `stark test` uses a naming convention (`fn test_*()`) instead, per an explicit user-approved
  workaround. Not a deviation from the spec (the spec doesn't have attributes either) — recorded
  here only because it's a recurring source of plan/reality mismatch (both the Phase 8 production
  tooling plan and, separately, this compiler brief's own not-Core list do not mention attributes
  either way). No fix owed; informational. — owner: none (not a deviation, a scope fact).
- DEV-SEED-008 Two independent hand-rolled JSON parser/value implementations exist:
  `lsp/protocol.rs` (`JsonValue`/`parse_json`) and `package.rs` (`JsonValue`/`parse_json`,
  `:5-58`) do not share code. Not a correctness bug — a simplification candidate, not scoped to
  any active WP. Noted for future consideration only; do not act on it under an unrelated WP
  (Charter: "avoid broad refactors not required by the active WP"). — owner: none (parking lot).

## Known deviations (WP-C1.1 additions)
- DEV-014 [CLOSED, WP-C1.1, 2026-07-17] `load_submodules_recursive` (`parser.rs`) suppressed its
  "file not found for module" diagnostic whenever `std::env::args().any(|arg| arg.contains(
  "test") || arg.contains("conformance"))` — meaning **every real `stark test` invocation**
  (the subcommand name itself contains "test") silently swallowed missing-module-file errors in
  real user packages, not just during the compiler's own test suite. Discovered while writing a
  WP-C1.1 checklist-item-8 test (couldn't write a positive "missing file is reported" test at
  all — the bypass fired unconditionally under `cargo test`). Fixed by removing the `env::args()`
  clause; kept the three filename-based conditions (`test.stark`, contains "spec-fixtures",
  contains "STARKLANG"), which are legitimately needed for one fixture
  (`07-Modules-and-Packages__01.stark`, notation, `mod math;` with no backing file). That fixture
  only worked before because `conformance.rs` never actually matched those filename conditions
  (it uses bare filenames, not full paths) — the env::args() clause was doing ALL the real work
  for it. Also fixed `conformance.rs`'s `parse_fixture`/`check_fixture` to use full fixture paths
  as the `SourceFile` name (so the filename-based conditions now genuinely match, independent of
  argv). Full regression test: `starkc/tests/gate2_valid.rs::
  test_missing_module_file_is_reported_not_silently_accepted`. Verified: full suite green twice,
  `cargo fmt`/targeted clippy clean on changed files.
- DEV-015 [RESOLVED, WP-C1.5] No pipeline stage checked a suffixed integer literal's magnitude
  against its suffix's representable range (`let x: UInt8 = 300u8;` compiled clean), and
  unsuffixed literals never got the spec's "Int32 if it fits, else Int64" treatment
  (03-Type-System.md:28) -- `let x = 99999999999;` silently typed as a broken Int32. Fixed in
  `check_expr`'s `Lit::Int` arm (`typecheck.rs`): suffixed literals are checked against their
  suffix's exact range (new `literal::int_suffix_range_contains`, code **E0008**); unsuffixed
  literals are promoted to Int64 if they don't fit Int32, and rejected outright if they don't fit
  Int64 either. A defense-in-depth suffix-range re-check was also added to `interp.rs::eval_lit`.
  Design question settled (user-approved 2026-07-18): typecheck/const-eval time, not the lexer --
  an unsuffixed literal's fit-check needs its inferred target type, which the lexer never has.
  Both checks share a new `src/literal.rs` module; building it also fixed a second,
  previously-unknown bug found in the process -- `pat_subsumes` compared literal *patterns* by
  shape only (no value), so `match x { 1 => .., 2 => .. }` spuriously flagged the second arm as
  unreachable. Full detail in the `### WP-C1.5` session record.
- DEV-016 [RESOLVED, WP-C1.4] `cargo clippy --all-targets -- -D warnings` failed with 22
  pre-existing warnings, none in files touched by WP-C1.1 (confirmed via `grep -v typecheck.rs`
  isolation — hits spanned `typecheck.rs`, `interp.rs`, `lsp/protocol.rs`, `lsp/server.rs`). CI
  had been red since the 2026-07-17 03:29 push for this exact reason (`cargo clippy --all-targets
  -- -D warnings` in `.github/workflows/ci.yml`'s `fmt, clippy, test` job), continuously through
  several unrelated feature commits and both governance-bootstrap commits. Fixed as a standalone
  cleanup during WP-C1.4 at the user's explicit request (mechanical, zero-behavior-change):
  13x `args.get(0)` → `args.first()` in `typecheck.rs`; 2x explicit-closure-clone → `.cloned()`
  (`interp.rs`, `lsp/server.rs`); 2x manual `if let Some` in a `for` loop → `.into_iter().flatten()`
  (`interp.rs`); 3x `*inner = Box::new(x)` → `**inner = x` (avoids a needless allocation,
  `interp.rs`); `JsonValue`'s inherent `to_string` → `impl std::fmt::Display` (`lsp/protocol.rs`,
  no call-site changes needed since the blanket `ToString` impl covers `Display`); one
  `.and_then(|x| Some(y))` → `.map(|x| y)` (`lsp/protocol.rs`). Verified: `cargo clippy
  --all-targets -- -D warnings` clean, `cargo fmt --check` clean, full workspace test suite green
  twice consecutively (unchanged pass count from before the cleanup).
- DEV-017 [CONFIRMED, WP-C1.1] `core-v1-coverage.toml`'s `tests` field cites files, not
  functions, and for several rules (before this WP) cited only the aggregate
  `starkc/tests/conformance.rs` even when that file contributed zero actual coverage (e.g.
  LEX-013 — no reserved-word fixture exists in the manifest at all; real coverage was always in
  `lexer.rs`'s/`parser.rs`'s own inline unit tests, uncited). Partially corrected this WP for the
  specific rules touched (LEX-003/004/005/010/013, SYN-001/013), but the same imprecision likely
  affects other rules not touched this WP. Schema cannot cite individual test functions (would
  break `check-conformance.py`'s path-existence check). — owner: WP-C1.6 (conformance evidence
  generator) — this is exactly the rule-level-precision problem that WP exists to solve.
- DEV-018 [PARTIALLY CLOSED, WP-C1.1] No AST span-integrity invariant checking existed anywhere
  (source or tests) before this WP. Added `starkc/tests/span_integrity.rs`: checks
  child-within-parent span containment for every `Expr`/`Block` node kind with directly-named
  children, across the full parseable fixture corpus (89 fixtures × 2 parse modes) plus targeted
  generic-closing-split cases. Found and correctly excluded 2 false positives (notation fixtures
  containing non-parseable grammar snippets, e.g. `&T // comment`, which produce genuinely
  inconsistent recovery-path spans when force-parsed — not a real bug, matches
  `conformance.rs`'s own notation-skip convention). **Deliberately not built**: `Type`/`Pat`/
  `Item` node containment, and a fully generic/exhaustive AST visitor (vs. this WP's
  hand-enumerated `ExprKind`/`BlockNode` match) — a generic position-indexed walker is properly
  WP-C2.4's job ("innermost syntax/HIR node at byte position" is a named C2.4 deliverable, and
  DEV-010's LSP hover/definition stubs need exactly this). — owner: WP-C2.4 for the general
  walker; WP-C1.1's Expr/Block check stands as the interim positive evidence.

## Known deviations (WP-C1.2 additions)
- DEV-019 [CONFIRMED, WP-C1.2] Diagnostic-code collisions between resolve.rs and the normative
  E-code table (`04-Semantic-Analysis.md`) / other modules' correct usage of the same codes.
  Spec table: E0202="Undefined type", E0203="Ambiguous name", E0401="Use of possibly-
  uninitialized variable". Confirmed collisions: (1) `resolve.rs` uses **E0401** for "unresolved
  import" (4 sites, `resolve.rs:394-434`) — colliding with `flow.rs:365`'s own, spec-correct use
  of E0401 for actual use-of-uninitialized-variable. (2) `resolve.rs` uses **E0203** for "no
  parent module for 'super'" (`:689`) and "item '{name}' is private" (`:748`) — neither is
  "ambiguous name"; colliding with `typecheck.rs:4759`'s spec-correct use of E0203 for "ambiguous
  trait method call". (3) `parser.rs` uses **E0202** for "file not found for module" /
  "conflicting module files" (module-loading errors, `parser.rs:113,234,249,263,298`) — not
  "undefined type"; colliding with `resolve.rs`'s own correct E0202 usage for actual undefined-
  type/enum-variant/struct-variant errors (`resolve.rs:881,1224,1242`). Net effect: three
  genuinely distinct rules currently share E0203 (2 wrong + 1 correct), two share E0401 (1 wrong
  + 1 correct), and module-loading errors share E0202 with unrelated undefined-type errors.
  (E0210, used twice in resolve.rs for "requires extension `tensor`", is NOT a deviation — it's
  an extension-scoped code the Core spec's table has no reason to enumerate.)
  **WP-C1.5 additions to this same class**, found while touching match-arm code for the
  exhaustiveness fix below: (4) `typecheck.rs`'s "unreachable match arm" warning uses **E0500**
  (`:3844` as of this WP) — spec table: E0500="Trait not implemented" (an *error*, not a
  warning); colliding with 15 other, spec-correct E0500 "trait not implemented" error sites in
  the same file. (5) `typecheck.rs`'s "method call on non-struct/enum type" error uses **E0303**
  (`:4955` as of WP-C1.3, now shifted by WP-C1.5's edits) — spec table: E0303="Non-exhaustive
  match"; colliding with the (now WP-C1.5-strengthened) spec-correct E0303 exhaustiveness sites.
  Not fixed here, same reasoning as (1)-(3): reassigning codes is a public diagnostic-contract
  change deserving its own bounded spec-bug-protocol change, not a drive-by inside a
  test/soundness-strengthening WP.
- **User impact:** a tool consuming diagnostic codes programmatically (IDE quick-fixes, CI
  triage, `starkc check --message-format json` consumers) cannot distinguish "no parent for
  super" from "ambiguous trait method call" by code alone, or "unresolved import" from "used a
  possibly-uninitialized variable," or "module file not found" from "undefined type."
- **Security/soundness impact:** none — diagnostics still render correct messages; this is a
  machine-readable-contract integrity issue, not a correctness issue for compiled programs.
- **Workaround:** consumers must currently match on message text, not code, for these specific
  cases.
- **Proposed disposition:** genuine spec-bug-protocol candidate (Charter §1.5 rule 2) — needs
  new normative E02xx codes allocated for: unresolved import, no-parent-for-super, private-item-
  access, module-file-not-found, conflicting-module-files; plus (from WP-C1.5) a new W0xxx code
  for "unreachable match arm" (it's a warning, not an error, so E0500 was always the wrong
  category regardless of collision) and a new E00xx code for "method call on non-struct/enum
  type". Not fixed here: reassigning codes is
  a public diagnostic-contract change touching tests across multiple files
  (`starkc/tests/diag_format.rs` and others assert on some of these exact strings/codes), and
  deserves the full spec-fix-plus-executable-evidence treatment in one bounded change, not a
  unilateral edit inside a test-strengthening WP.
- **Owning gate:** WP-C1.6 (conformance evidence generator) is the natural place to catch this
  class of issue systematically going forward; the actual reallocation is its own bounded
  spec-bug-protocol change, owner TBD.

- DEV-020 [CONFIRMED DESIGN BEHAVIOR, WP-C1.2, not a defect] `pub use` of a private item leaks
  it. `name_is_visible_from` (`resolve.rs:822-833`) treats `reexport_vis` as authoritative over
  the original item's own `vis` once a `pub use` populates it. Pinned down by
  `resolve.rs::pub_use_of_a_private_item_leaks_it` since this had zero prior test coverage and
  would be easy to accidentally change later without realizing it's a deliberate behavior, not
  an oversight. Not a bug; recorded so a future change to this behavior goes through CE1/CE2
  escalation rather than being treated as a routine test update.
- DEV-021 [VERIFIED CORRECT, WP-C1.2, not a defect] Cross-package coherence checking (SEM-007,
  orphan rule) was flagged by WP-C1.2's research as unverified — `typecheck.rs`'s `find_package_
  root` re-derives package membership from a pure filesystem walk-up per diagnostic-relevant
  item's file path, entirely independent of the `PackageGraph` object, and every existing
  coherence test used a bare in-memory `"test.stark"` with no real `starkpkg.json` on disk,
  making it impossible to tell whether cross-package detection actually worked or whether every
  impl was just being treated as same-package (`None`). `gate2_package.rs::
  test_cross_package_coherence_orphan_rule_with_real_packages` (new, WP-C1.2) builds a real
  two-package workspace with a genuine orphan-rule violation (impl of a dependency's trait for
  the dependency's own struct, written in the importing package) and confirms E0500 fires
  correctly. **Cross-package coherence checking does work end-to-end** — this closes the
  uncertainty with positive evidence rather than leaving it as an open risk.
- DEV-022 [SPEC-SILENT GAP, WP-C1.2] Private-item leakage through public signatures (a `pub fn`
  exposing a private type in its parameter/return signature, or a `pub struct` with a private
  field type reachable from outside) has **no implementation anywhere** (confirmed: zero hits in
  resolve.rs or typecheck.rs for any check resembling this). Unlike DEV-004/005/006/007, this is
  not a spec-vs-implementation mismatch — the spec itself (`07-Modules-and-Packages.md`,
  `04-Semantic-Analysis.md`) is silent on this question; no normative rule requires it. Not
  authorized for implementation under this WP regardless: Charter §1.5 rule 4 ("no new Core
  syntax or semantics inside an implementation WP") applies — adding this check would create a
  new rejection rule for programs that currently compile cleanly, which needs a language-design
  proposal, not a test-strengthening WP. — owner: needs a proposal/decision on whether STARK
  wants this check at all before any WP can implement it; not scheduled.

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

## Known deviations (WP-C1.3 additions)
- DEV-023 [CONFIRMED, WP-C1.3, not fixed] `Display`/`.fmt()` and `Hash`/`.hash()` are missing as
  *callable methods* on compiler-builtin types — the same bug class as DEV-013's Clone finding
  (now fixed for Clone), confirmed present for these two but deliberately not fixed in this WP to
  keep scope bounded after two substantial fixes already landed. Empirically confirmed:
  `String::from("hi").fmt()` and `"hi".hash()`-style calls both fail with E0303 "method call on
  non-struct/enum type 'String'", exactly mirroring the pre-fix Clone symptom. `Display`/`Hash`
  as *bounds* are already correctly recognized (same `satisfies_bound`/`Ty::Core` recursive-arg
  mechanism as Clone/Eq/Ord). Likely fix shape, by analogy with the Clone fix: `.fmt()` could
  reuse the interpreter's existing `impl fmt::Display for Value` (already used by `print`/
  `println` for these exact types) as a generic dispatch point; `.hash()` would need a new
  UInt64-producing hash function reused from wherever `HashMap`/`HashSet` key hashing is
  currently implemented internally (unverified whether that internal hash is itself exposed in a
  reusable form — needs its own investigation, unlike Clone/Display which had directly reusable
  machinery already in hand). — owner: unscheduled; candidate for a focused follow-up WP given
  the fix pattern is now well-understood from the Clone precedent.
- DEV-024 [CONFIRMED, WP-C1.3, not fixed] `From` trait associated-function calls fail to
  resolve. Empirically confirmed: a real `impl From<Celsius> for Fahrenheit { fn from(c:
  Celsius) -> Fahrenheit {...} }` followed by `Fahrenheit::from(c)` fails with E0200 "associated
  function 'from' not found" despite the impl existing. Unlike DEV-008/013's method-call path
  (`receiver.method()`), this is an *associated/static* function call (`Type::function()`,
  called without an existing receiver value) — likely a different resolution path
  (`find_associated_fn` in interp.rs, glimpsed at interp.rs:1643 during DEV-013's investigation,
  and its typecheck.rs counterpart) that may have the same "doesn't search trait impls, only
  inherent impls" gap as the method-call path had for defaults, or may be specific to `From`'s
  generic trait parameter (`From<Celsius>`) confusing the self-type match. Root cause not yet
  isolated — this needs its own investigation before a fix, not assumed to be the identical
  pattern to DEV-013's fixes. `Into`/`TryFrom` (the same CoreTrait family, `resolve.rs:2080-2082`)
  were not independently tested but likely share the same gap given they're conventionally
  implemented in terms of `From`. — owner: unscheduled; needs root-cause investigation first.
- DEV-025 [RESOLVED, WP-C1.5] `pat_subsumes` (typecheck.rs) compared literal *patterns* by shape
  only (`Lit` carries no value for Int/Float/Str, only base/suffix/raw tags), so any two
  same-kind literal patterns were treated as equal regardless of actual value. Confirmed
  empirically: `match x: Int32 { 1 => .., 2 => .. }` and `match x: &str { "a" => .., "b" => .. }`
  both spuriously flagged the second, genuinely-distinct arm as unreachable — fired on
  essentially every real-world literal match with 2+ arms. Found while building `src/literal.rs`
  for DEV-015. Fixed: `pat_subsumes` now parses both literals' actual values via
  `literal::eval_lit_value` and compares those instead of the shape-only `Lit` tag.

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
- Status: not evaluated (Gate C3 has not opened; C0-C2 are prerequisite per the mandatory
  compiler completion path in `COMPILER-ROADMAP.md` §4.1). Per CD-004, this is no longer a
  whether-question — native compilation is mandatory (`COMPILER-CHARTER.md` §1.2); C3 will
  select *which* backend architecture (`SELECT-GENERATED`/`SELECT-DIRECT`), not decide whether
  one is built.
- Selected strategy: none yet.
- Evidence: see CD-002 for the closest existing evidence (old Gate 6/7 tensor/ONNX-deployment
  track), which bears on a narrower tensor-deployment question, not general Core native
  compilation — informative precedent for C3's methodology, not a substitute for it. See CD-004
  for why that old evidence can no longer be read as license to skip general native compilation.

## Diagnostic codes allocated or changed
- None allocated yet under this governance framework. Existing normative `E####`/`W####` codes
  are inventoried as part of WP-C0.1 (`starkc/src/diag.rs`) and WP-C1.6 (conformance evidence
  generator), not duplicated here until that inventory exists.

## Evidence inventory
- `starkc/docs/gate1-exit.md` through `gate7-decision.md` — old-numbering gate evidence, see CD-001/CD-002.
- `STARKLANG/tests/spec-fixtures/manifest.toml` — 121-fixture spec corpus, verdict census in
  Repository baseline above.
- `cargo test --workspace --all-targets --all-features` output (2026-07-17 audit run) — 383
  passed / 0 failed / 2 ignored, full per-suite breakdown to be carried into
  `starkc/docs/dev/compiler-map.md` (WP-C0.1).
- `STARKLANG/conformance/core-v1-coverage.toml` — 59 rules, 53 implemented / 6 partial / 0
  missing, **integrity-audited under WP-C0.3** (duplicate-ID check, spec-chapter-validity check,
  4 stale `missing` entries corrected with cited evidence). `python3 starkc/scripts/
  check-conformance.py` output (2026-07-17, post-correction): 0 errors, 0 warnings.

## File inventory for current gate
- `STARKLANG/docs/compiler/COMPILER-CHARTER.md` — created WP-C0.0; fully regenerated 2026-07-17
  under CD-004 from the revised ("Native Compiler Required") brief.
- `STARKLANG/docs/compiler/COMPILER-ROADMAP.md` — created WP-C0.0; fully regenerated 2026-07-17
  under CD-004.
- `COMPILER-STATE.md` (this file) — created WP-C0.0; Position-line schema and Native backend
  selection section updated 2026-07-17 under CD-004.
- `STARKLANG/docs/compiler/work-packages/` — created WP-C0.0.
- `starkc/docs/dev/compiler-map.md` — created WP-C0.1. Module table (21 modules + 3 binaries),
  shared/duplicated entry points, global state/fs/process/nondeterminism audit, provenance flow,
  test-file-to-subsystem map, stub audit, archived-code confirmation.
- `CLAUDE.md`, `README.md`, `starkc/README.md`, `STARKLANG/docs/PLAN.md` — corrected under
  WP-C0.2 (stale status claims; see that WP's session record below for exact changes).
- `STARKLANG/conformance/core-v1-coverage.toml`, `starkc/scripts/check-conformance.py` —
  corrected/extended under WP-C0.3 (4 stale entries fixed, checker gained duplicate-ID and
  spec-chapter-validity checks plus staleness/negative-test-coverage warnings).
- `starkc/docs/conformance/KNOWN-DEVIATIONS.md` — created WP-C0.4. Structured ledger for
  DEV-004 through DEV-013 (normative expectation / current behaviour / user impact /
  security-soundness impact / workaround / proposed disposition / owning gate each), plus two
  informational (not-owned) entries.
- `starkc/docs/compiler/C0-exit-report.md` — created WP-C0.5. Gate C0 exit report: PASS
  decision, head/test/fixture counts, authoritative document list, subsystem status matrix,
  deviation ledger summary, explicit no-conformance-percentage-trusted statement, next WP.
- `STARKLANG/docs/compiler/work-packages/WP-C1.1.md` — created WP-C1.1. Active WP scope,
  scope-control answers, input packet.
- `starkc/src/parser.rs` — DEV-014 fix (removed unsound `env::args()` bypass); +5 new inline
  unit tests (reserved words full coverage, non-expression-position reserved-word rejection,
  `GtEq`→`Eq` generic split, bare-shift-vs-generic contrast).
- `starkc/src/lexer.rs` — +2 new inline unit tests (all 15 reserved words, 4-level nested
  comments + unterminated-at-depth negative case).
- `starkc/tests/conformance.rs` — fixed `parse_fixture`/`check_fixture` to use full fixture
  paths (needed for DEV-014's fix to not regress the one fixture that legitimately needs the
  filename-based module-load bypass).
- `starkc/tests/gate2_valid.rs` — +3 new tests (missing-module-file = DEV-014 regression test,
  duplicate-mod-declaration, circular-mod-reference/termination).
- `starkc/tests/robustness.rs` — +2 new tests (depth-limit latch/floor boundary assertions,
  diagnostic determinism across repeated parses).
- `starkc/tests/span_integrity.rs` — new file. Expr/Block span-containment checker across the
  full parseable fixture corpus + targeted generic-split cases (DEV-018).
- `STARKLANG/conformance/core-v1-coverage.toml` — LEX-003/004/005/010/013, SYN-001/013 test
  citations and scope-clarifying comments updated to reflect WP-C1.1's new evidence and DEV-015/
  DEV-017's findings.
- `STARKLANG/docs/compiler/work-packages/WP-C1.2.md` — created WP-C1.2. Active WP scope,
  inherited-findings section (DEV-004/006/007), scope-control answers, input packet.
- `starkc/src/resolve.rs` — DEV-004 fix (tensor-builtin gate on `resolve_unqualified`), DEV-006
  resolve-half fix (`push_diag`/`current_file_arc()` helpers, all 20 call sites converted),
  DEV-007 fix (sort glob-import items before iterating); +12 new inline unit tests covering
  lexical/module scope shadowing, `super`/`crate` navigation (including the previously-
  unexercised E0203 "no parent for super" path), `pub use` single/multi-level re-exports,
  pub-use-of-private-item, and explicit-use/declared-item collisions.
- `starkc/tests/gate2_package.rs` — +3 new tests: undeclared-dependency-import rejection,
  cross-package diagnostic file attribution, cross-package coherence with a real two-package
  workspace (the latter two were previously unverified end-to-end; both now confirmed working).
- `STARKLANG/conformance/core-v1-coverage.toml` — SEM-001, SEM-007, PKG-002, PKG-003 test
  citations and scope-clarifying comments updated.
- `STARKLANG/docs/compiler/work-packages/WP-C1.3.md` — created WP-C1.3. Active WP scope,
  inherited-findings section (DEV-008/013), scope-control answers, input packet.
- `starkc/src/typecheck.rs` — DEV-008 companion fix (`ty_satisfies_operator_bound`, recursive
  `Ty::Core` Eq/Ord bound checking); DEV-013 fixes (`.clone()` method-signature entry for builtin
  types in `core_method_signature`; `default_fallback` trait-default-method search in the
  method-call-checking function); +4 new inline unit tests.
- `starkc/src/interp.rs` — DEV-008 fix (`eval_binary` dispatches to a resolved `Eq::eq` impl for
  struct/enum values before falling back to structural comparison; signature changed to
  `&mut self` plus a new `lhs_expr: ExprId` parameter); DEV-013 fixes (generic `.clone()`
  dispatch point in `call_core_method`; `find_method` gained a trait-default-method fallback);
  +6 new inline unit tests.

## Follow-ups
- [ ] Housekeeping: this file is at 927 lines, over the Charter's ~700-line budget (§2.4:
      "compressing closed gate detail into summaries"). C0's detailed session records (WP-C0.0
      through WP-C0.5) are now fully duplicated in `starkc/docs/compiler/C0-exit-report.md` and
      could be compressed to short summaries here without losing information. Do this as its own
      deliberate pass, not folded into an unrelated WP, so nothing is accidentally dropped.
- [x] WP-C0.1: produce `starkc/docs/dev/compiler-map.md` — done, see File inventory below.
- [x] WP-C0.2: fixed `CLAUDE.md` (opening-summary/status contradiction, Gates-1-3/Gate-4-next
      claim), `starkc/README.md` (stale headline + missing 5 modules — compiler-map.md §1 is now
      the authoritative module table, README points to it), `STARKLANG/docs/PLAN.md` (added
      post-Gate-5 status note + changelog entry, did not fabricate WP-level detail for Gates 6-7),
      root `README.md` (fixed a genuine prose-vs-table contradiction on Gate 5 status, completed
      the truncated further-reading list to include gate5/6/7 docs and the new governance files).
- [x] WP-C0.2: confirmed `stark-spec-parity-roadmap.md` (Phases 0-9) and all `starkc/docs/
      WP8_*_IMPLEMENTATION.md`/`PHASE4E_MATH_RANDOM_IO_IMPLEMENTATION.md` files are **current and
      evidenced** — no edits needed. `starkc/docs/PHASE8_GRAMMAR_GAPS.md` in particular is an
      existing high-quality deviation log that directly confirmed DEV-008 (structural equality),
      DEV-009 (File/OS handle), DEV-010 (LSP stubs), DEV-011 (doc comments as trivia), DEV-012
      (VS Code UI unverified) — see Known deviations below. This file should be a primary input
      to WP-C0.4's ledger, not superseded by it.
- [ ] WP-C0.2 (unresolved governance-process question, carried forward — not blocking C0 exit):
      gate7-decision.md's "No LSP work or language expansion is authorized" text was apparently
      overridden for WP8.1-8.5, but no record of an explicit owner override was found anywhere in
      this repo. Recommend the project owner either (a) confirm and backfill a decision record,
      or (b) confirm no override was needed because WP8.x was scoped as tooling, not "language
      expansion" in the sense Gate 7 meant. Cannot be resolved by audit alone.
- [x] WP-C0.3: reconciled DEV-002 (was DEV-SEED-002) — PKG-004/005/006 and STD-004 corrected
      from stale `missing` to evidence-cited `partial`; checker upgraded with duplicate-ID and
      spec-chapter-validity checks plus non-fatal staleness/negative-test-coverage warnings.
- [x] WP-C0.4: built `starkc/docs/conformance/KNOWN-DEVIATIONS.md` — DEV-004 through DEV-013
      (10 entries, full structured format) plus DEV-SEED-008/014 as informational/not-owned.
- [x] WP-C1.3: resolved DEV-013 (STD-004 exhaustiveness) — closed with 2 real bugs found and
      fixed (Clone-on-builtins, default trait methods) plus 2 new deviations recorded
      (DEV-023, DEV-024). See session record below.
- [ ] WP-C1.6: extend the coverage-database schema to distinguish positive/negative test
      evidence per rule, so Charter rule 15 can be mechanically checked (currently only a
      heuristic warning in `check-conformance.py`, see WP-C0.3 note).
- [x] WP-C1.2: fixed DEV-004 (resolve.rs tensor-builtin gating) and DEV-007 (glob-import
      nondeterminism) — see session record below.
- [x] WP-C1.2: fixed DEV-006's resolve.rs half (multi-file diagnostic provenance). Borrowck/flow
      half closed in WP-C1.4 — see session record below. DEV-006 fully resolved.
- [x] WP-C1.3: closed DEV-008 (structural-vs-trait equality) by implementing normative dispatch
      (option 1 — the spec unambiguously specifies real Eq dispatch, not a spec defect).
- [ ] WP-C1.x (triage which WP owns CLI-behavior consistency): resolve DEV-005 (starkc
      check/run gating disagreement on warnings).
- [ ] Unscheduled: DEV-023 (Display/.fmt() and Hash/.hash() missing on builtin types — same bug
      class as DEV-013's Clone finding, fix pattern well-understood from that precedent).
- [ ] Unscheduled, needs root-cause investigation first: DEV-024 (From trait associated-function
      calls fail to resolve — Type::from(value) form doesn't work despite a real impl existing).
- [ ] WP-C8.2/C8.3: implement real LSP hover/definition/references (DEV-010).
- [ ] WP-C8.7: interactive VS Code Extension Development Host validation (DEV-012).
- [x] WP-C1.1: lexical and syntax requalification — closed 2026-07-17. See session record below.
- [x] WP-C1.5: fixed DEV-015 (suffixed-literal overflow never checked) — see session record
      below.
- [x] WP-C1.4: DEV-016 (repo-wide clippy debt, 22 pre-existing warnings) — resolved, see session
      record below.
- [ ] WP-C1.6: DEV-017 (coverage-database test citations lack function-level precision).
- [ ] WP-C2.4: DEV-018 (build the full generic AST position-indexed walker; WP-C1.1's
      Expr/Block-only span_integrity.rs check is interim evidence, extend rather than redo).
- [ ] WP-C1.1 follow-up (not blocking, noted in span_integrity.rs's own scope comment):
      underscore-placement rules for binary/octal literals still untested (decimal/hex only);
      no max-value-per-suffix-type positive test exists for any of the 8 int / 2 float suffixes.
- [x] WP-C1.2: name resolution, modules, and visibility — closed 2026-07-17. See session record
      below.
- [ ] WP-C1.6 (or a dedicated spec-bug-protocol change): DEV-019 — reallocate distinct E02xx
      codes for unresolved-import, no-parent-for-super, private-item-access, module-file-not-
      found, and conflicting-module-files (currently colliding with unrelated correct uses of
      E0401/E0203/E0202 in flow.rs/typecheck.rs/resolve.rs itself); plus (WP-C1.5 additions)
      a new W0xxx code for "unreachable match arm" (mislabeled E0500) and a new E00xx code for
      "method call on non-struct/enum type" (mislabeled E0303).
- [ ] Unscheduled, needs a proposal first: DEV-022 (private-item-leakage-through-public-
      signature checking doesn't exist; spec is silent on whether it should).

## Gate exit summaries
- C0: **PASS** (2026-07-17). Bootstrap, current-state audit, and authority repair complete. Full
  report: `starkc/docs/compiler/C0-exit-report.md`. Four stale documents corrected (`CLAUDE.md`,
  root `README.md`, `starkc/README.md`, `STARKLANG/docs/PLAN.md`); conformance database
  integrity-audited with 4 staleness errors fixed (DEV-002, closed); 10 confirmed deviations
  recorded with full structured detail in `starkc/docs/conformance/KNOWN-DEVIATIONS.md`; module-
  by-module compiler map produced (`starkc/docs/dev/compiler-map.md`). Explicit non-claim: no
  conformance percentage from this gate is trusted for Core v1/tensor v0.1 conformance purposes
  — see exit report's "No conformance percentage is trusted" section. Next: Gate C1.

---

### WP-C0.0 — 2026-07-17
DONE: Bootstrapped compiler governance. Split the master brief into
`STARKLANG/docs/compiler/COMPILER-CHARTER.md` (rules/boundaries/escalations/not-yet list) and
`COMPILER-ROADMAP.md` (gates/WPs/dependencies), created this state file, and created
`STARKLANG/docs/compiler/work-packages/`. Audited repo ground truth (head, toolchain, test
counts, existing gate docs, conformance DB shape) to seed the state file with evidence rather
than assumptions.
FILES: STARKLANG/docs/compiler/COMPILER-CHARTER.md, STARKLANG/docs/compiler/COMPILER-ROADMAP.md,
COMPILER-STATE.md, STARKLANG/docs/compiler/work-packages/ (created, empty)
RULES: none (no normative rule changed — governance/documentation only)
DECISIONS: CD-001, CD-002, CD-003
EVIDENCE: MANUAL — read starkc/docs/gate1-exit.md through gate7-decision.md,
STARKLANG/docs/ROADMAP.md, STARKLANG/docs/PLAN.md, root README.md, starkc/README.md, CLAUDE.md;
ran `cd starkc && cargo test --workspace --all-targets --all-features` (383 passed/0 failed/2
ignored); ran `cargo --version && rustc --version`; inspected
STARKLANG/conformance/core-v1-coverage.toml and STARKLANG/tests/spec-fixtures/manifest.toml
structure via grep/wc.
FOLLOW-UP: see "Follow-ups" section above (5 items seeded for WP-C0.1 through WP-C0.4).
NEXT: WP-C0.1

---

### WP-C0.1 — 2026-07-17
DONE: Produced `starkc/docs/dev/compiler-map.md`: full module table (purpose/input/output/entry
points) for all 21 `pub mod`s and 3 binaries; traced `starkc`/`stark` `check`/`run` call
sequences line-by-line and found they share underlying library functions but have drifted in
gating policy (DEV-005); audited global state/process-execution/HashMap-vs-BTreeMap usage and
confirmed a real glob-import nondeterminism in `resolve.rs` (DEV-007); traced `Span`/file-identity
provenance through all 5 pipeline stages and found parse+typecheck are correct but
resolve+flow+borrowck structurally lose multi-file provenance (DEV-006); confirmed the
previously-seeded `resolve.rs` tensor-builtin gating bug is real and still unfixed with exact
trigger conditions (DEV-004, promoted from DEV-SEED-004); mapped all 27 test files to the
subsystems they exercise; confirmed zero marker-tagged stubs (`todo!`/`unimplemented!`/`TODO`)
exist in `starkc/src`; confirmed archived-code paths and corrected a path error in the WP's own
premise (`web-docs/` is at repo root, not `STARKLANG/web-docs/`).
FILES: starkc/docs/dev/compiler-map.md (created); COMPILER-STATE.md (updated: DEV-004 through
DEV-007 promoted from seeds to confirmed deviations with full detail, DEV-SEED-008 added,
Follow-ups updated, File inventory updated)
RULES: none changed (audit/documentation only, no compiler behavior touched)
DECISIONS: none new (DEV-004 through DEV-007 are deviation records, not CD/AD decisions)
EVIDENCE: MANUAL — direct source reads and grep-based verification across lexer.rs, parser.rs,
ast.rs, resolve.rs, hir.rs, typecheck.rs, flow.rs, borrowck.rs, interp.rs, onnx/*, deploy/*,
formatter/*, doc_gen/*, lsp/*, test_runner/*, package.rs, options.rs, extensions/*, source.rs,
diag.rs, ast_dump.rs, main.rs, bin/stark.rs, bin/starkide.rs; `git blame`/`git log` on
resolve.rs to confirm DEV-004's age and unfixed status.
FOLLOW-UP: DEV-004/005/006/007 all require fixes owned by future WP-C1.2/WP-C1.4 (not this WP —
C0 is audit-only, no compiler behavior changes are authorized here per Charter §2.2). DEV-SEED-008
(duplicate JSON parsers) parked, no owner. See "Follow-ups" section above for the complete,
current list.
NEXT: WP-C0.2

---

### WP-C0.2 — 2026-07-17
DONE: Audited and reconciled documentation/status claims across `CLAUDE.md`, root `README.md`,
`starkc/README.md`, `STARKLANG/docs/PLAN.md`, `stark-spec-parity-roadmap.md`, and every
`starkc/docs/*IMPLEMENTATION*.md`/`PHASE*.md` file. Fixed four genuinely stale/contradictory
documents (`CLAUDE.md`, `starkc/README.md`, `STARKLANG/docs/PLAN.md`, root `README.md`) without
rewriting any historical gate-exit evidence — only current-status summaries were corrected, each
with a pointer to this state file or to `compiler-map.md` as the up-to-date source. Confirmed
`stark-spec-parity-roadmap.md` and all `WP8_*_IMPLEMENTATION.md`/`PHASE4E_*.md` docs are already
accurate (no edits needed). Discovered that `starkc/docs/PHASE8_GRAMMAR_GAPS.md` is a pre-existing,
high-quality deviation log and used it to confirm five more deviations (DEV-008 through DEV-012),
including closing out DEV-SEED-001 (equality/trait-dispatch — confirmed real: `==`/`!=` is pure
structural equality on `Value`, no `Eq` trait dispatch at runtime) and DEV-SEED-003 (File/OS
handle — confirmed real and still deferred).
FILES: CLAUDE.md (opening summary + Implementation Status + footer rewritten); README.md (fixed
Gate-5 prose/table contradiction, completed truncated further-reading list); starkc/README.md
(headline, Use section — added `deploy`/`lsp`/`stark` subcommands, Layout table — added 5 missing
modules and gate5-7 rows, pointer to compiler-map.md); STARKLANG/docs/PLAN.md (added Post-Gate-5
status note under §5, v0.6 changelog entry); COMPILER-STATE.md (DEV-008 through DEV-012 added,
DEV-SEED-001/003 promoted to confirmed DEV-008/DEV-009, Follow-ups updated)
RULES: none changed (documentation only, no compiler behavior touched)
DECISIONS: none new (deviation records, not CD/AD decisions)
EVIDENCE: MANUAL — read all six documents in full or via targeted grep; cross-checked dates/claims
against `git log` and against the WP-C0.1 compiler-map.md findings; read
`starkc/docs/PHASE8_GRAMMAR_GAPS.md` and `PHASE4E_MATH_RANDOM_IO_IMPLEMENTATION.md` in full.
FOLLOW-UP: one governance-process question left genuinely unresolved (gate7-decision.md's "no
language expansion authorized" vs. the subsequently-shipped WP8.x work — cannot be resolved by
audit, needs an owner decision). All other WP-C0.2 "Done when" criteria met: no active document
claims a closed gate is "next," no README advertises an LSP stub as implemented (DEV-010 now
explicitly documented instead), status summaries agree with this file.
NEXT: WP-C0.3

---

### WP-C0.3 — 2026-07-17
DONE: Full integrity audit of `STARKLANG/conformance/core-v1-coverage.toml` (59 rules across 7
chapters: LEX 14, SYN 13, TYP 8, SEM 7, MEM 6, STD 5, PKG 6). Confirmed no duplicate rule IDs and
no references to nonexistent spec chapters. Found and fixed a real, significant staleness bug:
`STD-004`, `PKG-004`, `PKG-005`, `PKG-006` were marked `missing` despite working, dedicated-test-
covered implementations (verified via direct source reads of `typecheck.rs`, `parser.rs`,
`package.rs` plus `starkc/tests/gate4a_prelude_traits.rs`, `gate2_package.rs`,
`gate3_package_resolution.rs`) — root cause was that the database was seeded once at Phase 0
(2026-07-16) and never updated as Phases 1-4A subsequently closed exactly that scope. Corrected
all four to `partial` (not `implemented` — deferring the stronger claim to WP-C1.6's rule-level
audit) with real source/tests citations and inline dated correction notes. Also corrected
`STD-003`/`STD-005`'s source/tests citations to the actual dedicated test files (status
unchanged for these two — already accurate). Upgraded `starkc/scripts/check-conformance.py`:
added duplicate-rule-ID detection (hard fail), spec-chapter-existence validation derived from
the real `STARKLANG/docs/spec/` directory rather than a hardcoded list (hard fail), a warning for
`missing` entries that still carry `source`/`tests` fields (staleness signal), and a
heuristic warning for likely semantic-rejection rules with zero recorded tests. Documented one
schema limitation the checker cannot fix without a rule-format change: the single `tests` array
per rule doesn't distinguish positive from negative test evidence, so Charter rule 15 can't be
mechanically verified yet.
FILES: STARKLANG/conformance/core-v1-coverage.toml (4 status corrections + 2 citation
corrections, each with inline dated notes); starkc/scripts/check-conformance.py (duplicate-ID
check, chapter-validity check, two new warning classes); COMPILER-STATE.md (DEV-SEED-002
promoted to closed DEV-002, new DEV-013, Conformance summary and Repository baseline sections
updated with corrected counts, Follow-ups updated)
RULES: none changed (documentation/metadata correction only — no compiler source or behavior
touched; the coverage database is a status record, not normative behavior)
DECISIONS: none new (DEV-002 close-out and DEV-013 are deviation records, not CD/AD decisions)
EVIDENCE: MANUAL — read core-v1-coverage.toml in full (471 lines); ran `python3 starkc/scripts/
check-conformance.py` before and after corrections (before: 89.8% overall, 4 chapters showing
"Missing" > 0 for STD/PKG; after: same 89.8% implemented-only figure — corrections moved
missing→partial, not missing→implemented — but Missing column now correctly reads 0 for both
affected chapters); grepped `typecheck.rs`, `parser.rs`, `package.rs` for the specific
trait-bound-name/mod-loading/manifest-loading logic cited in each correction note; ran
`grep -c "#\[test\]"` against `gate4a_prelude_traits.rs` (12), `gate2_package.rs` (6),
`gate3_package_resolution.rs` (3) to confirm real test coverage, not just source existence.
FOLLOW-UP: DEV-013 (STD-004 exhaustiveness, esp. Error trait) owned by WP-C1.3. Schema
enhancement for positive/negative test distinction owned by WP-C1.6. See "Follow-ups" section
above for the complete, current list.
NEXT: WP-C0.4

---

### WP-C0.4 — 2026-07-17
DONE: Consolidated every confirmed deviation from WP-C0.1 through WP-C0.3 (DEV-004 through
DEV-013) into `starkc/docs/conformance/KNOWN-DEVIATIONS.md`, using the exact structured format
the brief requires: normative expectation, current behaviour, user impact, security/soundness
impact, workaround, proposed disposition, owning future gate — for every entry. Added two
informational (not-owned) entries for items investigated but judged not to be conformance gaps
(duplicate JSON parsers, absence of attribute syntax). No new deviations were discovered in this
WP specifically — its job was consolidation and format compliance, not new investigation; all
underlying findings originate from WP-C0.1 (compiler-map.md), WP-C0.2 (PHASE8_GRAMMAR_GAPS.md
cross-reference), and WP-C0.3 (coverage database audit).
FILES: starkc/docs/conformance/KNOWN-DEVIATIONS.md (created); COMPILER-STATE.md (Follow-ups
updated, File inventory updated)
RULES: none changed (documentation only)
DECISIONS: none new
EVIDENCE: MANUAL — cross-checked every DEV-NNN entry in KNOWN-DEVIATIONS.md against its source
citation in COMPILER-STATE.md/compiler-map.md/PHASE8_GRAMMAR_GAPS.md/core-v1-coverage.toml to
confirm no detail was lost or altered in consolidation.
FOLLOW-UP: none new — all 10 deviation entries already have an owning future WP recorded (see
Follow-ups above); this ledger will need regeneration/updating whenever a listed deviation is
closed, per Charter §2.4 ("deviations are never deleted without a closing note and evidence
link").
NEXT: WP-C0.5

---

### WP-C0.5 — 2026-07-17
DONE: Wrote `starkc/docs/compiler/C0-exit-report.md` and closed Gate C0 with a **PASS**
decision. Re-ran `cargo test --workspace --all-targets --all-features` one final time (383
passed / 0 failed / 2 ignored, unchanged — no Rust source was touched across WP-C0.0-C0.4) as
fresh evidence for the report. Report covers: current head/toolchain, test/fixture/coverage
counts, the authoritative document list in source-of-truth order, a subsystem status matrix
distinguishing old-numbering (Gates 1-7) status from new-numbering (C-gate) status per
subsystem, a summary table of all 10 open deviations plus DEV-002's closure, and an explicit
"no conformance percentage is trusted" section per Charter rule 14 and this gate's own closing
requirement. Named the exact next WP (WP-C1.1) per the mandatory correctness path.
FILES: starkc/docs/compiler/C0-exit-report.md (created); COMPILER-STATE.md (Position moved to
Gate C1/WP-C1.1, Gate exit summaries section filled in with the C0 PASS entry, File inventory
updated)
RULES: none changed (documentation only)
DECISIONS: none new (this is the gate-close event itself, not a new CD/AD/DEV record)
EVIDENCE: MANUAL — re-ran `cargo test --workspace --all-targets --all-features` and `python3
starkc/scripts/check-conformance.py` as final-state verification before writing the report;
cross-checked every table/count in the exit report against COMPILER-STATE.md's own sections to
ensure internal consistency (per the standing multi-document consistency sweep obligation).
FOLLOW-UP: none new for C0 itself — it is closed. All deviation-level follow-ups (DEV-004 through
DEV-013's owning WPs) and doc-level follow-ups (the one unresolved gate7-decision.md-vs-WP8.x
governance question) carry forward into Gate C1 unchanged; see "Follow-ups" section above.
NEXT: WP-C1.1 (Gate C1 — Core v1 Conformance Closure opens)

---

### WP-C1.1 — 2026-07-17
DONE: Lexical and syntax requalification. Researched current test coverage against all 10
roadmap checklist items with file:line citations (delegated to a research agent, synthesized
personally). Found and fixed one real, production-impacting bug (DEV-014: `parser.rs`'s
module-loading error suppression triggered on any process whose argv contained "test" —
including every real `stark test` invocation — silently swallowing missing-module-file errors;
also had to fix `conformance.rs` to use full fixture paths so the legitimate filename-based
bypass for one notation fixture kept working after removing the unsound clause). Added 12 new
tests across 5 files closing or narrowing 6 of the 10 checklist items: reserved-token coverage
(3/15→15/15 words, +non-expression-position case), nested-comment depth (2→4 levels + negative
case), `>>`/`>>=` generic-closing-split edge cases (`GtEq`→`Eq` arm, bare-shift contrast),
multi-file module edge cases (missing file, duplicate declaration, circular reference/
termination), recursion-depth-limit boundary assertions (was: crash-only test with zero
assertions on message/count/false-positive floor), diagnostic determinism across repeated
parses, and — the largest single addition — a new `starkc/tests/span_integrity.rs` performing
the first-ever programmatic AST span-containment check in the codebase (Expr/Block node kinds,
full parseable fixture corpus, found and correctly excluded 2 false positives from
deliberately-non-parseable notation fixtures). Two items (parser-recovery progress guarantees,
extension-syntax gating) were found already well-covered and left as-is. Two gaps were found,
documented, and explicitly deferred rather than fixed in-scope: DEV-015 (literal-overflow
checking doesn't exist anywhere in the pipeline — a design-scope question, not a test gap) and
DEV-018's full generalization (a complete generic AST walker belongs to WP-C2.4, which needs the
same mechanism for LSP position lookups — building it twice would waste effort). Also surfaced
two lower-priority findings in passing: DEV-016 (repo-wide pre-existing clippy debt, unrelated
to files this WP touched) and DEV-017 (coverage-database test citations lack function-level
granularity — exactly WP-C1.6's job to fix systematically). Updated `core-v1-coverage.toml`
citations/notes for the 7 LEX/SYN rules most directly touched.
FILES: starkc/src/parser.rs, starkc/src/lexer.rs, starkc/tests/conformance.rs,
starkc/tests/gate2_valid.rs, starkc/tests/robustness.rs, starkc/tests/span_integrity.rs (new),
STARKLANG/conformance/core-v1-coverage.toml, STARKLANG/docs/compiler/work-packages/WP-C1.1.md
(new), COMPILER-STATE.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md
RULES: LEX-003, LEX-004, LEX-005, LEX-010, LEX-013, SYN-001, SYN-013 — citations/status notes
updated (no status value changed; all remained `implemented`, evidence strengthened)
DECISIONS: none new (DEV-014 through DEV-018 are deviation records; DEV-014 is the only one
that involved an actual code fix, authorized under Charter §2.2 Sonnet-level autonomy as a
spec-consistent parser fix that does not change the normative accepted/rejected program set in
the direction of weakening — if anything it makes rejection *more* correct)
EVIDENCE: MANUAL + REG — full research agent report (10-item checklist, cited); `cargo test
--workspace --all-targets --all-features` run 3+ times across the session, final state 395
passed/0 failed/2 ignored (up from 383/0/2); `cargo fmt --check` clean; `cargo clippy
--all-targets -- -D warnings` confirmed clean on all WP-C1.1-touched files specifically (22
pre-existing failures elsewhere, recorded as DEV-016, not introduced by this WP); `python3
starkc/scripts/check-conformance.py` clean (0 errors, 0 warnings) after coverage.toml edits.
Every new test was run individually to confirm it passes for the right reason (not just
suite-green), and the DEV-014 fix was verified against the full suite twice before and after to
confirm no regression.
FOLLOW-UP: DEV-015 (owner: WP-C1.3/C1.5, needs triage), DEV-016 (unscheduled), DEV-017 (owner:
WP-C1.6), DEV-018 general case (owner: WP-C2.4). Narrower residual gaps noted in
span_integrity.rs's and the individual test functions' own doc comments: binary/octal
underscore-placement rules still untested, no max-value-per-suffix-type positive test exists,
Type/Pat/Item span containment unchecked. See "Follow-ups" section above for the complete list.
NEXT: WP-C1.2 (name resolution, modules, and visibility)

---

### WP-C1.2 — 2026-07-17
DONE: Name resolution, modules, and visibility requalification. Fixed all three deviations
inherited from WP-C0.1/C1.1 as this WP's explicit starting scope: DEV-004 (tensor-builtin gate
added to `resolve_unqualified`), DEV-007 (glob-import items sorted before iterating), and
DEV-006's resolve half (new `push_diag`/`current_file_arc()` helpers mirroring typecheck.rs's
backfill pattern, all 20 diagnostic-push sites converted). Each fix has a dedicated regression
test, verified individually before the full-suite sweep. Then researched the full 10-item
roadmap checklist (delegated to a research agent, synthesized personally) and closed the gaps
found: lexical-vs-module-scope shadowing, `self`/`super`/`crate` path navigation (found the
E0203 "no parent module for super" diagnostic had *zero* test evidence of any kind despite being
a real code path — now tested at both the root-module-error case and the nested-module-success
case), `pub use` single- and multi-level re-exports (found real, dedicated `reexport_vis`
machinery with zero test coverage — the single largest gap in the whole matrix; added 3 tests
including pinning down that `pub use` of a private item deliberately leaks it), explicit-use
import collisions, and three package-level cases (undeclared dependency import, cross-package
diagnostic file attribution, cross-package coherence) via three new `gate2_package.rs` tests
using real multi-package workspaces on disk rather than in-memory single files. The coherence
and file-attribution tests were framed as "report whatever the real behavior is" rather than
assuming success, given the research flagged both as genuinely unverified — both came back
positive (DEV-021: cross-package coherence checking does work correctly end-to-end). Along the
way, discovered a significant new finding while writing tests: three diagnostic codes
(`E0401`, `E0203`, `E0202`) are each reused by resolve.rs/parser.rs for meanings that don't match
the normative E-code table in `04-Semantic-Analysis.md`, colliding with other modules'
*correct* uses of those same codes (DEV-019) — recorded with full detail but not fixed, since
reallocating codes is a public-contract change spanning multiple files' test assertions and
deserves its own bounded spec-bug-protocol treatment. Also discovered, while writing the
super/crate navigation tests, that STARK's visibility model is genuinely stricter than Rust's
(private = visible only in the exact defining module, no descendant inheritance) — my first
draft of two tests assumed Rust-style semantics and failed against the real implementation,
which is exactly what surfaced this; recorded as a pinned-down design fact so future WPs don't
rediscover it the same way. Investigated but did not implement private-item-leakage-through-
public-signature checking (checklist item 7): confirmed it doesn't exist anywhere in the
pipeline, but also confirmed the spec is silent on whether it should exist — recorded as
DEV-022, a feature-needing-a-proposal, not a bug, per Charter rule 4 (no new semantics inside an
implementation WP).
FILES: starkc/src/resolve.rs (3 bug fixes + 15 new inline tests), starkc/tests/gate2_package.rs
(+3 new tests), STARKLANG/conformance/core-v1-coverage.toml (SEM-001/007, PKG-002/003 citations
updated), STARKLANG/docs/compiler/work-packages/WP-C1.2.md (new), COMPILER-STATE.md,
starkc/docs/conformance/KNOWN-DEVIATIONS.md
RULES: SEM-001, SEM-007, PKG-002, PKG-003 — citations/notes updated (status unchanged,
`implemented`, evidence strengthened; SEM-007 specifically went from "aggregate fixture citation
only" to "real cross-package test, positively verified")
DECISIONS: none new (DEV-019 through DEV-022 are deviation/confirmation records; the three bug
fixes are spec-consistent parser/resolver corrections under Charter §2.2 Sonnet-level autonomy,
not new CD/AD decisions — none of them change the normative accepted/rejected program set in a
weakening direction)
EVIDENCE: MANUAL + REG — full research agent report (10-item checklist, cited); every new test
run individually before the full-suite sweep; `cargo test --workspace --all-targets
--all-features` run 3+ times across the session, final state 410 passed/0 failed/2 ignored (up
from 395/0/2); `cargo fmt --check` clean; `cargo clippy --all-targets -- -D warnings` clean on
all WP-C1.2-touched files; `python3 starkc/scripts/check-conformance.py` clean (0 errors, 0
warnings) after coverage.toml edits. The three inherited-bug fixes were each verified with a
dedicated regression test that fails against the pre-fix code (confirmed by construction, since
each test was written to reproduce the exact failure mode WP-C0.1/C1.1 documented).
FOLLOW-UP: DEV-019 (owner: WP-C1.6 to catch systematically; reallocation itself unowned),
DEV-022 (unscheduled, needs a proposal). DEV-006's flow/borrowck half remains open for WP-C1.4.
See "Follow-ups" section above for the complete, current list.
NEXT: WP-C1.3 (types, generics, traits, and operator semantics)

---

### WP-C1.3 — 2026-07-17
DONE: Types, generics, traits, and operator semantics requalification. Closed both inherited
findings with real fixes, not just tests: **DEV-008** (equality dispatch) — confirmed via direct
spec reading (`03-Type-System.md:389-406,516-531`) that `Eq` is a normal, user-implementable
trait with `==`/`!=` normatively desugaring to `Eq::eq`, settling the "spec vs. compiler" question
in favor of implementing real dispatch (not a spec defect). Implemented it in `interp.rs`
(`eval_binary` now dispatches to a resolved `Eq` impl for struct/enum values via the existing
`find_method`/`call_user_method` machinery, verified with a deliberately non-structural custom
`eq()` to prove real dispatch) and found+fixed a companion gap while investigating
(`typecheck.rs`'s `require_operator_bound` had no `Ty::Core` arm at all, so `Option<Int32> ==
Option<Int32>` was unconditionally rejected — confirmed empirically before fixing). **DEV-013**
(STD-004 exhaustiveness) — refuted the original "Error trait may be unsupported" worry (it was
checking the wrong function; the real bound-checker handles arbitrary trait names generically)
but discovered two significant, previously-unknown, real bugs while testing the trait family
broadly: `.clone()` was completely non-functional on every compiler-builtin type (String, Vec,
Option, Result, HashMap, HashSet, Range, IOError — confirmed via direct empirical testing, not
inferred), and trait default method bodies were never used as a fallback when an implementing
type didn't override them (a core, spec-documented Core v1 feature, explicitly named in this
WP's own checklist, completely broken). Fixed both, in both typecheck.rs and interp.rs
(two-stage fixes, since a call must both type-check and execute correctly). While testing this
trait family, found two more real bugs and made a deliberate scope-discipline decision to record
rather than fix them, having already delivered four substantial fixes: DEV-023 (`Display`/`.fmt()`
and `Hash`/`.hash()` share Clone's exact "missing as callable method on builtins" bug, not yet
applied to them) and DEV-024 (`From` trait `Type::from(value)` associated-function calls fail to
resolve — root cause not yet isolated, likely a different code path than the method-call fixes
above). Spot-checked (not exhaustively audited) the remaining checklist items — local inference,
generic substitution, associated types, orphan/overlap, conflicting-impl diagnostics — against
existing test coverage and found it adequate; did not run the full research-agent audit process
used for WP-C1.1/C1.2 given the WP's effort budget was consumed by the two deep bug-fix cycles.
FILES: starkc/src/typecheck.rs (2 fixes: `ty_satisfies_operator_bound` extraction + `Ty::Core`
arm; `.clone()` core-method-signature entry; `default_fallback` trait-default search; +4 tests),
starkc/src/interp.rs (2 fixes: `eval_binary` Eq dispatch + signature change; generic `.clone()`
dispatch; `find_method` trait-default fallback; +6 tests), STARKLANG/conformance/
core-v1-coverage.toml (unchanged this WP — no rule-status citations needed updating),
STARKLANG/docs/compiler/work-packages/WP-C1.3.md (new), COMPILER-STATE.md,
starkc/docs/conformance/KNOWN-DEVIATIONS.md
RULES: none formally re-cited in coverage.toml this WP (the fixes are runtime/type-checker
behavior corrections underlying SEM-002 "Trait Resolution" and TYP-006 "Trait Bounds," not new
rule-level evidence entries — a future WP-C1.6 pass should retroactively cite these tests)
DECISIONS: none new as CD/AD records; DEV-008 and DEV-013 closures plus DEV-023/DEV-024 are
deviation records. The DEV-008 fix is explicitly authorized as "option 1" under WP-C1.3's own
text ("implementing the normative dispatch semantics consistently in checking and execution");
not an escalation-requiring change since it doesn't alter the accepted/rejected program set,
only makes already-type-checked programs execute consistently with what the type checker already
required.
EVIDENCE: MANUAL + REG — direct spec reading for the Eq dispatch question (not assumption);
every fix verified empirically via scratch `.stark` programs BEFORE writing permanent tests, to
confirm the exact before/after behavior change (e.g. `Option<Int32> == Option<Int32>` rejected
before, accepted after; custom non-structural `eq()` ignored before, dispatched after; `.clone()`
"method call on non-struct/enum type" before, working after; default trait method "not found"
before, running after). `cargo test --workspace --all-targets --all-features` run 4+ times across
the session, final state 418 passed/0 failed/2 ignored (up from 410/0/2). `cargo fmt --check`
clean. `cargo clippy --all-targets -- -D warnings` clean on all WP-C1.3-touched files (confirmed
via line-range diff that all remaining warnings are pre-existing DEV-016 debt, not introduced
here). `check-conformance.py` unaffected (no coverage.toml changes this WP).
FOLLOW-UP: DEV-023 (owner: unscheduled, fix pattern well-understood), DEV-024 (owner:
unscheduled, needs root-cause investigation first — do not assume it matches DEV-013's fix
pattern). A future WP-C1.6 pass should add rule-level test citations to coverage.toml for the
Eq/Clone/default-method fixes landed here. The lighter-weight checklist spot-check (vs. C1.1/
C1.2's exhaustive research-agent audit) means local inference/generic substitution/associated
types/orphan-overlap/conflicting-impls have NOT been exhaustively re-verified — recorded as a
known gap in this WP's own coverage, not silently claimed as thorough.
NEXT: WP-C1.4 (ownership, borrowing, lifetimes, and drop checking)

### WP-C1.4 — 2026-07-17
DONE: Ownership, borrowing, lifetimes, and drop checking. Closed the inherited finding
(**DEV-006**'s borrowck/flow half — multi-file diagnostic provenance) with the same
`push_diag`/`item_files`-swap pattern used for resolve.rs in WP-C1.2; DEV-006 is now fully
resolved. Research (dispatched to a research agent given the user's standing preference to route
research-heavy sub-tasks to a stronger model where useful) surfaced three candidate soundness
gaps against the checklist; flagged all three to the user as CE2-shaped before resolving, per the
user's 2026-07-17 standing preference (`stark-ce-escalation-flagging` memory) — the user answered
"I'll fix them myself now," authorizing full fixes with my own engineering judgment on design.
Investigation and fixes, in order:
- **Deref-move of a non-Copy value (`*r` for `r: &T`)** — confirmed `place_of`/`consume_place`
  had no `Deref` case at all, so `*r` type-checked and silently deep-cloned `T` out of borrowed
  storage at runtime (`interp.rs`'s `UnOp::Deref` uses `clone_place_value`, not move semantics);
  for a `Drop`-implementing `T` this double-runs the destructor with no explicit `.clone()` ever
  written. Fixed via a new `check_owned_value` helper (narrower than `check_expr`'s general
  traversal, which is also reached from genuinely-sound read-only contexts — comparisons,
  reborrowing `&*r`, field/index access through a deref — that must NOT be treated as moves).
  Initially wired into the three confirmed-dangerous "owning" positions (let-init, return,
  block-tail); while documenting the fix, recognized the same pattern applied to call arguments
  and aggregate construction (tuple/array elements, struct-literal fields) — confirmed
  empirically that `take(*r)` and `(*r, 1)` for a `Drop` type both compiled and double-dropped —
  and closed those too, verifying along the way that free-function call arguments need no
  callee-signature awareness (STARK has no argument-position auto-ref/deref-coercion, so `*r`
  only type-checks against a callee parameter when the parameter is already the pointee type by
  value).
- **Iterator/collection borrow exclusivity** — `let it = v.iter();` must keep a shared borrow of
  `v` alive as long as `it` is (a live aliasing view), per 03-Type-System.md's borrow-carrying-
  types rule. Two independent bugs combined to defeat this, both confirmed via before/after
  empirical testing (the dangerous program compiled and crashed at runtime pre-fix): (1)
  `type_carries_borrow` had no case for iterator CoreTypes (`VecIter`, `CharsIter`, etc. — their
  only generic argument is the *element* type, not a reference, so the generic
  recurse-into-args-looking-for-`Ty::Ref` rule never matched them), so the borrow registered
  while evaluating `v.iter()` was truncated at the end of the `let` statement instead of
  persisting; (2) even with the borrow correctly kept alive, `consume_place`'s conflict check
  (`check_read_borrow_conflict`) only rejected moves against a *mutable* active borrow — correct
  for reads (concurrent shared reads are sound) but wrong for a move, which invalidates storage a
  shared borrow still aliases regardless of mutability. Fixed both: a new iterator-CoreType arm
  in `type_carries_borrow`, and a new any-active-borrow check in `consume_place` gated to the
  non-Copy (move) path only.
- **Shortest-input-lifetime rule** (the `fn longest(x: &str, y: &str) -> &str` worked example) —
  flagged by the research agent as the most architecturally significant gap, on the theory that
  no code traces a call's reference-typed arguments to its borrow-carrying return type. Empirical
  verification found this was a false alarm: `check_expr`'s handling of `&expr`/ref-returning-
  method arguments already registers a borrow on the argument's local unconditionally, wherever
  that argument expression appears, and `check_stmt`'s `Let` branch already keeps every borrow
  registered while evaluating the init expression alive past the `let` whenever the bound type
  carries a borrow (`longest`'s `&str` return type does). The combination already implements rule
  3's "regardless of which branch was taken" conservative semantics — confirmed against the
  spec's own literal worked example — with zero code changes needed. Recorded as a verified-sound
  regression test, not a fix.
Also fixed, at the user's explicit request after a GitHub Actions failure notification mid-WP:
**DEV-016** (repo-wide clippy debt, 22 pre-existing warnings unrelated to this WP's own files,
had kept CI red since the 2026-07-17 03:29 push across several unrelated commits) — see DEV-016's
entry above for the itemized mechanical fixes; now closed.
FILES: starkc/src/borrowck.rs (push_diag/item_files-swap; `check_owned_value` new method, wired
into 6 call sites; `type_carries_borrow` iterator-CoreType arm; `consume_place` any-active-borrow
move check), starkc/src/flow.rs (push_diag/item_files-swap; `file` field now used, no longer
`_file`), starkc/src/interp.rs (DEV-016: 2 clippy fixes, no behavior change),
starkc/src/lsp/protocol.rs (DEV-016: 2 clippy fixes, `JsonValue::to_string` → `impl Display`, no
call-site changes needed), starkc/src/lsp/server.rs (DEV-016: 1 clippy fix), starkc/src/
typecheck.rs (DEV-016: 13x `args.get(0)` → `args.first()`), starkc/tests/gate2_valid.rs (+10
tests: 2 DEV-006 provenance regressions, 8 soundness-fix regressions), starkc/tests/
gate3_execution.rs (+1 test: abort-without-drop semantics, item 16 of the WP checklist — found
correctly implemented, previously untested), STARKLANG/docs/compiler/work-packages/WP-C1.4.md
(new), COMPILER-STATE.md, starkc/docs/conformance/KNOWN-DEVIATIONS.md.
RULES: none formally re-cited in coverage.toml this WP (a future WP-C1.6 pass should retroactively
cite the new borrowck/drop tests against the relevant OWN-xxx/memory-model rule IDs).
DECISIONS: none new as CD/AD records. All three soundness fixes tighten existing, already-
approved checks to close confirmed holes in their own coverage (not new semantic decisions, and
not weakenings requiring Charter §2.2 escalation) — consistent with the CE-escalation-watch note
in WP-C1.4.md and the user's explicit "I'll fix them myself now" authorization.
EVIDENCE: MANUAL + REG — every fix (and the one refuted false alarm) verified empirically via
scratch `.stark`/`.rs` programs BEFORE writing permanent tests, confirming the exact before/after
diagnostic or runtime behavior (e.g. deref-move: silently accepted and double-cloned before,
E0100 after; iterator move: compiled and crashed with "use of unavailable value" before, E0101
after; shortest-lifetime: E0101 already fired with zero code changes). `cargo test --workspace
--all-targets --all-features` run twice consecutively at the end of the WP, final state 429
passed/0 failed/2 ignored (up from 418/0/2 at WP-C1.3's close), deterministic across both runs.
`cargo fmt --check` clean. `cargo clippy --all-targets -- -D warnings` fully clean repo-wide
(DEV-016 closed this WP, not just avoided in touched files).
FOLLOW-UP: item 13 (partial-move-of-Drop-type) and item 16 (abort-without-drop) from the WP
checklist were found correctly implemented but previously untested; both now have permanent
regression tests, no further action needed. The lighter-weight per-item checklist coverage (1, 4,
8, 9, 11, 12, 14a, 15a/15b) from the original WP-C1.4.md scope was not exhaustively re-audited
this WP given the effort budget was consumed by the three deep soundness-fix cycles plus the
CI-driven DEV-016 detour — recorded as a known gap in this WP's own coverage, matching WP-C1.3's
same disclosure pattern, not silently claimed as thorough.
NEXT: WP-C1.5 (control flow, patterns, constants, and numeric semantics)

### WP-C1.5 — 2026-07-18
DONE: Control flow, patterns, constants, and numeric semantics. Research (dispatched to a
research agent) audited the WP's 13-item checklist plus the inherited DEV-015 finding against
`flow.rs`/`typecheck.rs`/`interp.rs`, verifying every claim empirically via scratch `.stark`
programs rather than from code reading alone. Found six items needing real fixes and one item
(build-mode invariance of traps) confirmed already sound. Three findings were CE1/CE2-shaped
(genuine semantic/spec-interpretation decisions, not just bugs) and were flagged to the user
before resolving, per the standing `stark-ce-escalation-flagging` preference; the user answered
all three (2026-07-18):
- **DEV-015** (inherited): fix at typecheck/const-eval time plus a defensive runtime re-check —
  **approved and implemented**. See DEV-015's own ledger entry above for the fix detail. Building
  its literal-parsing logic surfaced a companion, previously-unknown bug (`pat_subsumes` compared
  literal patterns by shape only, not value — `match x { 1 => .., 2 => .. }` spuriously flagged
  the second arm unreachable); fixed in the same pass via a new shared `src/literal.rs` module
  (also refactored `interp.rs::eval_lit` to use it instead of its own private copy — pure
  refactor, verified zero behavior change before building on top of it).
- **Match exhaustiveness gap**: compile-time exhaustiveness was previously implemented only for
  `Enum` and `Bool` scrutinees — **approved and implemented** ("implement real compile-time
  exhaustiveness"). Extended the existing arm-tracking loop (`typecheck.rs`'s `ExprKind::Match`
  arm) to also recognize `Option`/`Result` coverage (their `Some`/`None`/`Ok`/`Err` patterns
  resolve via `Res::Builtin`, not `Res::Variant`, so the pre-existing `matched_variants` tracking
  never covered them — `match opt { Some(v) => .. }` missing `None` compiled clean before this).
  For every other scrutinee type, added a general rule: an explicit wildcard/binding arm is now
  required to be considered exhaustive (a real usefulness/coverage algorithm for arbitrary types
  is out of this WP's scope; this is sound — never accepts a genuinely non-exhaustive match —
  though it can over-reject some in-practice-exhaustive matches, matching this codebase's
  existing "reject some safe programs is intentional" philosophy). Added a recursive
  `is_irrefutable` helper so fully-binding compound patterns (`(a, b)`, `[a, b, c]`) still count
  as exhaustive without a redundant trailing wildcard, and exempted `Ty::Struct` scrutinees
  entirely (a struct has exactly one shape, so a single covering arm is exhaustive by
  construction) — both needed to avoid regressing `tests/gate2_valid.rs`'s existing fixture
  corpus and one `interp.rs` inline test. Caught one real regression during verification: fixing
  the `?`-operator bug below (see next item) broke `tests/gate2-valid/21_try_operator.stark`,
  which declared its own `enum Option<T> { Some(T), None }` shadowing the prelude type — flagged
  to the user as a genuine ambiguity (does `?`/exhaustiveness apply to shadowed lookalike types?)
  rather than resolved unilaterally; user chose "prelude types only, update the fixture" — the
  fixture's shadowing enum declaration was removed so it now exercises the real prelude `Option`.
- **Float division/modulo by zero**: `03-Type-System.md`'s Numeric Semantics section states both
  "MUST trap" and, adjacently, IEEE-754 "NaN, +/-Inf" semantics — a real spec-internal tension,
  not just a code bug. **User decision: keep trapping, no code change** — see CD-006 above.
Three more items were fixed directly (mechanical/tightening, not semantic-policy decisions,
consistent with the CE-escalation-watch's "diagnostics/soundness-tightening fix directly, no
escalation needed" carve-out):
- Array-repeat-expression count (`[value; count]`) previously computed its length by parsing the
  *raw source text* of `count` as a bare unsuffixed decimal — a suffixed literal (`5u32`), an
  underscore-grouped literal (`1_0`), or a `const` item reference all silently computed length 0,
  falsely rejecting every subsequent valid index with E0007. Added a minimal `const_eval_u64`
  (literal, or a reference to a `const` item — not a full constant-folding pass, out of scope)
  and a new diagnostic (**E0009**) for count expressions that aren't constant at all, replacing
  the previous silent wrong-answer fallback.
- The `?` operator's Result/Option identity check was a substring search over an enum's *entire
  declaration source text* for "Result"/"Option" — confirmed exploitable: an unrelated user enum
  with a variant literally named `ResultVariant` satisfied it, letting `?` type-check against a
  function that doesn't return `Result`/`Option` at all. `Option`/`Result` always resolve to
  `Ty::Core(CoreType::Option|Result, _)`, never `Ty::Enum` (confirmed via `hir::CoreType`), so the
  `Ty::Enum` special-case was dead weight for real Option/Result and existed only to create the
  bug; removed both `Ty::Enum` match arms entirely (2 call sites) rather than trying to make the
  substring search more precise.
FILES: starkc/src/literal.rs (new -- shared literal-value parsing), starkc/src/lib.rs (register
`literal` module), starkc/src/interp.rs (refactored `eval_lit` onto the shared module; DEV-015
defensive runtime check), starkc/src/typecheck.rs (`pat_subsumes` real literal-value comparison;
DEV-015 magnitude checks in `Lit::Int`'s `check_expr` arm; `const_eval_u64` + array-repeat-count
fix; `?`-operator `Ty::Enum` arm removal x2; match-exhaustiveness Option/Result tracking +
general wildcard-required rule + `is_irrefutable` helper), starkc/tests/gate2_valid.rs (+21
tests), starkc/tests/gate2-valid/21_try_operator.stark (removed a shadowing `enum Option<T>`
declaration so the fixture exercises the real prelude type),
STARKLANG/docs/compiler/work-packages/WP-C1.5.md (new), COMPILER-STATE.md,
starkc/docs/conformance/KNOWN-DEVIATIONS.md.
RULES: none formally re-cited in coverage.toml this WP (a future WP-C1.6 pass should
retroactively cite the new control-flow/pattern/numeric tests against the relevant rule IDs).
DECISIONS: CD-006 (float division-by-zero trap semantics, keep current behavior, no code change).
No new CE-escalation deferrals remain open from this WP -- all three flagged decisions were
resolved by the user within this session (two approved-and-implemented, one settled by explicit
non-action).
EVIDENCE: MANUAL + REG -- every fix (and the settled spec-tension question) verified empirically
via scratch `.stark` programs before writing permanent tests, confirming exact before/after
behavior (e.g. `300u8` accepted then E0008-rejected; `[0; 5u32]` false-E0007 then correctly
length-5; the substring-exploiting `?` case accepted then E0006-rejected; `match opt { Some(v) =>
.. }` accepted then E0303-rejected). One real regression caught and fixed during verification
(the `21_try_operator.stark` fixture, see above) -- confirms the empirical-verification-before-
permanent-tests discipline is pulling its weight, not just ceremony. `cargo test --workspace
--all-targets --all-features` run twice consecutively at the end of the WP, final state 450
passed/0 failed/2 ignored (up from 429/0/2 at WP-C1.4's close), deterministic across both runs.
`cargo fmt --check` clean. `cargo clippy --all-targets -- -D warnings` fully clean repo-wide.
FOLLOW-UP: DEV-019 gained two more collision instances (E0500 misused for "unreachable match
arm," which is a warning not the spec's "trait not implemented" error; E0303 misused for "method
call on non-struct/enum type," not the spec's "non-exhaustive match") -- folded into DEV-019's
existing ledger entry rather than fixed here, same reasoning as its original three (public
diagnostic-contract change, deserves its own bounded spec-bug-protocol change). The general
"requires wildcard" exhaustiveness rule is intentionally imprecise for compound types beyond the
irrefutable-pattern check (e.g. it cannot prove `match x: Bool8Tuple { (true,true)=>.., ...all 4
combinations... }` is actually exhaustive without a wildcard) -- backstopped by the interpreter's
pre-existing "non-exhaustive match reached" runtime trap, and consistent with this WP's own
scope-control answer that some over-rejection is an acceptable, intentional tradeoff. A real
usefulness/coverage algorithm remains a candidate for a future WP if the over-rejection proves
practically annoying. Also fixed a pre-existing, unrelated documentation staleness bug found
during the consistency sweep: this file's header cited "CD-005" for the mandatory-native-compiler
governance update, but the actual decision log entry was filed as CD-004 (off-by-one, predates
this WP) -- corrected.
NEXT: WP-C1.6 (conformance evidence generator)
