# STARK Compiler STATE
Updated: 2026-07-19 after CD-022 follow-up amendment

## Position
Gate: C3-ENTRY  Next: complete native-readiness transition before WP-C3.1
Blocked: six completeness-row approvals; DEV-060 disposition; versioned execution-corpus freeze; demonstrated green CI run pending (baseline steps added 2026-07-19, not yet run)
Mandatory compiler path: Core=CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS (C2
closed, see starkc/docs/compiler/C2-exit-report.md)  MIR=blocked (behind C3)  Native=blocked
(behind C3, mandatory per CD-004 — C3 selects how, not whether)
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
  (starkc/, against the post-DEV-051/052/055 correction baseline recorded below):
  **594 passed, 0 failed, 2 ignored** across **4 unittest binaries** (`src/lib.rs`,
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
Canonical ledger (full structured entries, all 58 numbered deviations):
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`. The per-deviation narrative that used to live in
this file (seed list + WP-C1.1/C1.2/C1.3 addition sections) is archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020); the ledger remains the
single source of truth.

Open as of 2026-07-19:
- DEV-005 — `starkc` vs `stark` check/run warning-gating drift. Open, unowned since Gate C1.
- DEV-010 — LSP hover/definition/references are protocol stubs. Owner: WP-C8.2/C8.3.
- DEV-011 — doc comments are lexer trivia, not AST/HIR metadata. Unscheduled; needs a scoped
  proposal.
- DEV-012 — VS Code extension UI never interactively verified. Owner: WP-C8.7.
- DEV-017 — 39 of 59 legacy coverage rules still lack function-level positive/negative evidence
  classification (tooling exists; classification unscheduled).
- DEV-060 — repeated call to an un-overridden trait default method wrongly flagged as a move.
  Owner: C3-ENTRY; disposition required before the C3 workload freeze.
- Informational, not owed a fix: DEV-SEED-008 (two hand-rolled JSON parsers), DEV-SEED-014
  (no attribute syntax — deliberate scope fact).

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
- [ ] DEV-060: dispose before C3 workload freeze (C3-ENTRY blocker).
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
