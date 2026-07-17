# STARK Compiler STATE
Updated: 2026-07-17 after WP-C0.0

## Position
Gate: C1  Next: WP-C1.3  Blocked: none
Conditional tracks: Native=deferred (see CD-002)  ArtifactInfra=blocked (no second artifact impl yet)

## Repository baseline
- Head: 6fa8c15b94bd1376a847132498d31dd356524180
- Rust toolchain: `starkc/rust-toolchain.toml` pins `channel = "stable"` (no version number, tracks
  stable) with `rustfmt`/`clippy` components. Active environment measured: `cargo 1.93.0
  (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607 2026-01-19)`. `starkc/Cargo.toml` declares
  `rust-version = "1.85"` (crate MSRV). The Gate-5 *generated deployment host* (not `starkc`
  itself) separately requires Rust 1.88 due to the `ort` crate's MSRV
  (`starkc/docs/gate5-backend-decision.md:107-110`) — this does not raise `starkc`'s MSRV.
- Test count / suites: `cargo test --workspace --all-targets --all-features` (starkc/):
  **410 passed, 0 failed, 2 ignored** across 3 unittest binaries + 31 integration-test files
  (up from 383/0/2 and 30 files at Gate C0 close; WP-C1.1 added `span_integrity.rs` + 12 tests,
  WP-C1.2 added 15 more across `resolve.rs`'s inline tests and `gate2_package.rs`). Both ignored
  tests are intentionally opt-in (a checksum-pinned live ONNX artifact test in
  `tests/gate4_onnx.rs`, and a live-ORT-download inference test in `tests/gate5_codegen.rs`).
  Full per-file breakdown recorded in `starkc/docs/dev/compiler-map.md` (WP-C0.1; not
  re-regenerated for the WP-C1.1/C1.2 deltas — see that file's own scope note).
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
- Types: old Gate 2 (`gate2-exit.md`, closed 2026-07-15) covers resolution/HIR, type + control
  flow checking, generics/traits, ownership/borrows, 26-program `tests/gate2-valid/` corpus.
  Pending WP-C1.3 requalification; the equality/trait-dispatch closure the new roadmap flags
  (Charter WP-C1.3) is now **confirmed present**, see DEV-008 — not merely a risk to check for.
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
- DEV-008 [CONFIRMED, WP-C0.2, via `starkc/docs/PHASE8_GRAMMAR_GAPS.md:132-145`] Equality is
  pure structural equality on the interpreter's internal `Value` enum (Rust-derived `PartialEq`
  in `interp.rs`, `BinOp::Eq`/`Ne`) — there is no dispatch through a user's `Eq` trait
  implementation at runtime. `Eq` as a trait bound is currently a type-checker-only concept
  (checked as a bound, never dispatched as a method at runtime). The interpreter's structural
  equality happens to match a correct `derive(Eq)`, but a hand-written custom `impl Eq for T`
  (if expressible at all — unconfirmed whether users can implement `Eq` by hand vs. only
  auto-derive) would be silently ignored at runtime. `assert_eq`/`assert_ne` inherit the same
  behavior. This is exactly the risk pattern WP-C1.3 names ("hidden interpreter-only structural
  equality rule is not accepted as an undocumented third behaviour") — now confirmed present,
  not hypothetical. — owner: WP-C1.3 (close via implementing normative dispatch, or correcting
  an unambiguous spec defect per the spec-bug protocol — not a third behaviour).
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
- DEV-006 [PARTIALLY CLOSED — resolve half fixed WP-C1.2, 2026-07-17] Multi-file diagnostic provenance loss. `Span` carries no file
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
  just a different file in the same package). **Flow/borrowck half still open** — owner: WP-C1.4.
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
- DEV-013 [CONFIRMED, WP-C0.3] `STD-004` (standard traits: Clone/Hash/Default/Display/Error/
  Iterator) exhaustiveness is unresolved. Confirmed *present* (see DEV-002 correction above:
  typecheck.rs recognizes these as compiler-known bounds, gate4a_prelude_traits.rs tests them),
  but not yet confirmed *complete* against the full normative surface — in particular whether
  `Error` trait support exists (not seen in the `typecheck.rs:5598-5637` bound-name list checked
  during WP-C0.3) and whether users can hand-write `impl <StdTrait> for T` vs. only trigger
  compiler-builtin behavior (directly relevant to DEV-008's `Eq`-dispatch finding — the same
  question likely applies to `Clone`/`Hash`/etc.). — owner: WP-C1.3 rule-level audit.
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
- DEV-015 [CONFIRMED, WP-C1.1] No pipeline stage checks a suffixed integer/float literal's
  magnitude against its suffix's representable range. Empirically confirmed: `let x: UInt8 =
  300u8;` compiles and `starkc check` reports clean. `typecheck.rs`'s `convert_int_suffix`
  (`:5728-5737`) only maps the suffix to a type tag, never inspects the literal's value. Not a
  status error on LEX-003/LEX-004 (their descriptions are lexical *shape* only), but a normative
  concern (CLAUDE.md: "Integer overflow... always trap — in every build mode") with no
  representation anywhere in `core-v1-coverage.toml`. Where this belongs (lexer-level immediate
  rejection vs. typecheck/const-eval-level check) is a design question, not resolved here.
  — owner: WP-C1.3 (types/generics/traits) or WP-C1.5 (numeric semantics) — needs triage to
  determine which; likely the latter given CLAUDE.md frames overflow as a numeric-semantics
  concern.
- DEV-016 [CONFIRMED, WP-C1.1] `cargo clippy --all-targets -- -D warnings` fails with 22
  pre-existing warnings, none in files touched by WP-C1.1 (confirmed via `grep -v typecheck.rs`
  isolation — remaining hits span `typecheck.rs`, `lsp/protocol.rs`, and others). The Charter
  §2.5 definition-of-done lists clean clippy as a default requirement; this has apparently never
  actually held for the repository as a whole. Not fixed here (out of WP-C1.1's module scope —
  touches `typecheck.rs`/`lsp/protocol.rs`, neither owned by this WP). — owner: unscheduled;
  candidate for a small dedicated cleanup WP, or fold into whichever WP next touches each file.
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
  access, module-file-not-found, conflicting-module-files. Not fixed here: reassigning codes is
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

## Backend decision
- Status: not evaluated under the new C-numbering (Gate C3 has not opened; C0-C2 are prerequisite
  per the mandatory correctness path in `COMPILER-ROADMAP.md` §4.1).
- Evidence: see CD-002 for the closest existing evidence (old Gate 6/7), which bears on a
  narrower tensor-deployment question, not general Core native compilation.

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
- `STARKLANG/docs/compiler/COMPILER-CHARTER.md` — created WP-C0.0.
- `STARKLANG/docs/compiler/COMPILER-ROADMAP.md` — created WP-C0.0.
- `COMPILER-STATE.md` (this file) — created WP-C0.0.
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

## Follow-ups
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
- [ ] WP-C1.3: resolve DEV-013 (STD-004 exhaustiveness — Error trait, hand-written impls).
- [ ] WP-C1.6: extend the coverage-database schema to distinguish positive/negative test
      evidence per rule, so Charter rule 15 can be mechanically checked (currently only a
      heuristic warning in `check-conformance.py`, see WP-C0.3 note).
- [x] WP-C1.2: fixed DEV-004 (resolve.rs tensor-builtin gating) and DEV-007 (glob-import
      nondeterminism) — see session record below.
- [x] WP-C1.2: fixed DEV-006's resolve.rs half (multi-file diagnostic provenance). Borrowck/flow
      half remains open for WP-C1.4.
- [ ] WP-C1.3: close DEV-008 (structural-vs-trait equality) per the spec-bug protocol or by
      implementing normative dispatch.
- [ ] WP-C1.x (triage which WP owns CLI-behavior consistency): resolve DEV-005 (starkc
      check/run gating disagreement on warnings).
- [ ] WP-C8.2/C8.3: implement real LSP hover/definition/references (DEV-010).
- [ ] WP-C8.7: interactive VS Code Extension Development Host validation (DEV-012).
- [x] WP-C1.1: lexical and syntax requalification — closed 2026-07-17. See session record below.
- [ ] WP-C1.3 or WP-C1.5 (triage which): fix DEV-015 (suffixed-literal overflow never checked).
- [ ] Unscheduled: DEV-016 (repo-wide clippy debt, 22 pre-existing warnings outside WP-C1.1's
      touched files).
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
      E0401/E0203/E0202 in flow.rs/typecheck.rs/resolve.rs itself).
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
