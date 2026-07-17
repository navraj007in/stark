# STARK Compiler — Known Deviations and Stub Ledger

WP-C0.4 deliverable. Every entry below was independently verified against current source (head
`6fa8c15b94bd1376a847132498d31dd356524180`, 2026-07-17), not merely copied from a seed list or a
prior session's memory — several seeded suspicions turned out to need correction in the process
(see DEV-002's actual finding vs. its original framing). Deviation IDs (`DEV-NNN`) are shared
with `COMPILER-STATE.md`, which is the append-only decision log; this file is the structured,
query-friendly ledger. Where the two disagree, `COMPILER-STATE.md`'s dated session records are
authoritative (this file may lag by one session).

Each entry states: normative expectation, current behaviour, user impact, security/soundness
impact, workaround, proposed disposition, owning future gate.

---

## DEV-004 — `resolve.rs` tensor-builtin gating bug (bare `min`/`max`)

- **Normative expectation:** Core-only compilation (no `--extension tensor`) must never resolve
  a name to a tensor-extension builtin. Charter §1.5 rule 5: "Core remains extension-neutral."
- **Current behaviour:** `resolve_unqualified` (`resolve.rs:1854-1876`) calls `resolve_builtin`
  with no `options.tensor()` gate, unlike the correctly-gated `resolve_path_relative`
  (`resolve.rs:682-685`). Bare `min`/`max` unconditionally resolve to
  `Builtin::TensorMin`/`TensorMax`. `resolve_unqualified` has exactly two call sites: resolving
  `self` (`resolve.rs:659`) and struct-literal shorthand-field lowering (`resolve.rs:1000`).
- **User impact:** narrow but real. In Core-only mode, a struct-literal shorthand field named
  exactly `min` or `max` with no local/module item of that name in scope silently resolves to
  the tensor builtin instead of correctly failing "undefined variable 'min' (shorthand field)".
  If a local named `min`/`max` genuinely exists, normal scope lookup takes precedence and this
  bug does not fire.
- **Security/soundness impact:** none directly (no memory/type safety violation), but it is an
  extension-isolation leak — Core-only programs can observably depend on tensor-extension
  identity by accident, undermining the isolation guarantee Gate C9 will need to certify.
- **Workaround:** avoid struct-literal shorthand fields literally named `min`/`max` without an
  in-scope local of that name; use the explicit `Point { min: min }` form instead.
- **Proposed disposition:** fix by adding the same `options.tensor()` gate to
  `resolve_unqualified`'s builtin-resolution branch that `resolve_path_relative` already has.
- **Owning gate:** WP-C1.2 (name resolution, modules, visibility).

## DEV-005 — `starkc` vs `stark` CLI gating drift on warnings

- **Normative expectation:** Charter §1.5 rule 18: "cross-tool compiler behaviour must
  converge... rather than subtly different pipelines." One program's accept/reject outcome
  should not depend on which subcommand of the same binary invoked the compiler.
- **Current behaviour:** `starkc check` (`main.rs:551-602`) gates progression on `severity !=
  Error` — warnings pass through. `starkc run` (`main.rs:702-745`) gates on
  `diagnostics.is_empty()` — any diagnostic of any severity blocks. Both call the same
  underlying `parse_with_options`/`resolve::resolve_with_options`/
  `typecheck::analyze_with_options`/`interp::run` — the drift is in caller-side gating policy,
  not duplicated compiler logic.
- **User impact:** a program that produces exactly one parse- or resolve-stage warning (zero
  errors) is reported `starkc check` → OK / exit 0, but `starkc run` on the identical file
  refuses to execute at all, falling to the diagnostic-rendering path and `ExitCode::FAILURE`.
  Confusing and inconsistent from a single binary.
- **Security/soundness impact:** none — this is a availability/usability inconsistency, not a
  correctness or safety gap; if anything `run`'s stricter gate is the safer default.
- **Workaround:** treat `starkc check` passing as necessary but not sufficient for `starkc run`
  to succeed; always test `run` directly rather than inferring its outcome from `check`.
- **Proposed disposition:** pick one gating policy (most likely: both should gate on
  `severity != Error`, matching `check`, since blocking execution on a mere warning is unusually
  strict) and apply it uniformly; requires a CE2-class judgment call (spec-vs-implementation
  ambiguity — the spec doesn't mandate CLI gating policy, so this is implementation-convenience
  territory, likely resolvable without escalation, but flagged since it changes observable CLI
  behavior).
- **Owning gate:** WP-C1.x (triage exact ownership; behavior consistency isn't cleanly owned by
  any single C1 sub-WP as currently scoped).

## DEV-006 — Multi-file diagnostic provenance loss (resolve/flow/borrowck) (RESOLVED in WP-C1.4)

- **Normative expectation:** Charter §1.5 rule 17: "Source identity must survive the pipeline.
  AST/HIR/MIR/query results and diagnostics must retain the correct file, module, package, and
  artifact provenance."
- **Original behaviour:** `Span` carries no file identity at all (`source.rs:10-13`); there is no
  `FileId`/`SourceId` type anywhere in the crate. Parse (`parser.rs:359-363`) and typecheck
  (`typecheck.rs:1916-1919,2065-2068` plus 4 backfill sites) correctly reconstructed per-item
  file identity via a `HashMap<ItemId, Arc<SourceFile>>` side table. Resolve (`resolve.rs`, 20
  diagnostic sites, zero `.with_file()` calls), flow analysis (`flow.rs:21-24`, file parameter
  named `_file` and structurally unused), and borrow checking (`borrowck.rs`, single
  whole-crate `self.file`, no per-item lookup) did not — every diagnostic rendered against
  whichever file the top-level caller happened to pass (always the package's root file).
- **User impact (while open):** for any multi-file `stark` package (the only paths using
  `PackageGraph` — `stark check`/`build`/`run`/`test`), a name-resolution, control-flow, or
  borrow-check error/warning originating in a non-root `mod`-loaded file rendered with the
  **wrong filename** and byte offsets mapped against the **wrong file's** line-start table.
  `SourceFile::line_col` clamps out-of-range offsets (`source.rs:70`) rather than panicking, so
  this failed silently, producing a plausible-looking but incorrect `-->` diagnostic header —
  actively misleading during debugging, not just an omission.
- **Security/soundness impact:** none to compiled-program safety (the underlying check still ran
  correctly; only its *reported location* was wrong), but it was a diagnostics-integrity gap that
  Charter rule 16 ("diagnostics are part of behaviour") treats as a first-class defect, not
  cosmetic.
- **Resolution:** fixed in two stages. **WP-C1.2** fixed the resolve half: added
  `push_diag`/`current_file_arc()` helpers (`resolve.rs`), mirroring typecheck.rs's own
  if-none backfill pattern; all 20 `self.diags.push` call sites converted to `self.push_diag`.
  Verified with a same-package regression test and a cross-package test
  (`gate2_package.rs::test_cross_package_diagnostic_reports_dependency_file_not_root_file`).
  **WP-C1.4** fixed the flow/borrowck half: the same `push_diag` pattern plus a per-item
  `self.file` swap (via `hir.item_files`) applied to both `borrowck.rs` and `flow.rs`;
  `flow::check`'s previously-unused `_file` parameter is now a real, used field. Verified with
  two regression tests
  (`gate2_valid.rs::test_borrowck_diagnostic_in_nonroot_file_reports_correct_file`,
  `test_flow_diagnostic_in_nonroot_file_reports_correct_file`). A real `FileId`/`SourceId` type
  was considered and explicitly not pursued — the ad hoc `Arc<SourceFile>`-threading fix pattern
  proved sufficient across all four pipeline stages, so the larger architectural question never
  became load-bearing.
- **Owning gate:** WP-C1.2 (resolve, closed) and WP-C1.4 (flow/borrowck, closed). DEV-006 fully
  resolved.

## DEV-007 — `resolve.rs` glob-import (`use mod::*`) nondeterminism

- **Normative expectation:** Charter definition-of-done: "no new... nondeterministic iteration...
  introduced in compiler paths" and "generated output is deterministic across two runs."
- **Current behaviour:** glob-import expansion copies from an unsorted `HashMap<String, Res>`
  (`ModuleData::items`, `resolve.rs:45`) at two call sites (`resolve.rs:475-479` absolute-path,
  `:536-540` relative-path). `insert_module_item` (`:571-595`) raises `E0204` ("duplicate
  definition... in the same module scope") on a colliding different `Res`. Because Rust's
  default `HashMap` uses a randomized per-process hash seed, which of two glob-colliding names
  is treated as "first" (silently wins) vs. "second" (flagged `E0204`) varies across runs of the
  same compiler on the same source. Diagnostics are never sorted before rendering (`grep -n
  "sort" diag.rs` → no hits).
- **User impact:** a program using `use mod::*` with a genuine name collision across two glob
  sources may see `starkc check` report `E0204` on one run and silently accept a different
  (arbitrary) resolution on another run of the identical source and compiler binary — a
  reproducibility failure for both CI and local development.
- **Security/soundness impact:** none directly, but nondeterministic accept/reject on identical
  input undermines trust in diagnostic reproducibility, which downstream tooling (this ledger's
  own WP-C1.6 conformance evidence generator, CI gating) depends on being stable.
- **Workaround:** avoid `use mod::*` glob imports where two glob sources could plausibly define
  the same name; use explicit `use mod::{name}` imports instead.
- **Proposed disposition:** sort glob source items by name before iterating (or by `Span`, for
  stable "declaration order" semantics if that's the intended tie-break) at both call sites.
- **Owning gate:** WP-C1.2.

## DEV-008 — Structural equality, not `Eq` trait dispatch, at runtime (CLOSED)

- **Normative expectation:** per the roadmap's own framing (WP-C1.3): equality dispatch must
  follow one consistent, documented semantics. The spec settles which one: `Eq` is a normal,
  user-implementable trait (`03-Type-System.md:389-406`, worked example `impl Eq for Point` with
  real per-field logic; `06-Standard-Library.md:107-109` identically) and `==`/`!=` normatively
  desugar to `Eq::eq`/negation except for primitive types, which keep built-in intrinsic
  comparison (`03-Type-System.md:516-531`).
- **Previous behaviour:** `==`/`!=` were pure structural equality on the interpreter's `Value`
  enum unconditionally — no dispatch through a user's `Eq` implementation, even though
  `typecheck.rs`'s `require_operator_bound` already required a real `impl Eq for T` to exist for
  any struct/enum `==` to type-check in the first place.
- **Resolved 2026-07-17 (WP-C1.3), option 1 — implemented normative dispatch:**
  `eval_binary` (interp.rs) now looks up a resolved `Eq` impl via the existing `find_method`/
  `call_user_method` machinery before falling back to structural comparison; only struct/enum
  values are looked up (primitives and `Ty::Core` containers have no user-overridable Eq per
  Core v1's "operator overloading... is a future extension" rule, so structural comparison
  remains exactly correct for them). Verified with a deliberately non-structural custom `eq()`
  (always returns `true`) to prove real dispatch, not coincidental agreement.
- **Companion fix found while investigating:** `Ty::Core` (Option/Result/Vec/Box) had no arm at
  all in `require_operator_bound` — `Option<Int32> == Option<Int32>` was unconditionally
  rejected with E0500 despite `Int32` obviously satisfying `Eq`. Added a recursive bound-
  satisfaction check over container type arguments.
- **User impact:** none remaining — both the dispatch gap and the Option/Result/Vec rejection
  gap are closed.
- **Regression tests:** `interp.rs::custom_eq_impl_is_dispatched_not_structural`,
  `::custom_eq_impl_is_dispatched_for_ne_too`, `::option_and_vec_equality_are_structural`;
  `typecheck.rs::option_result_vec_box_satisfy_eq_when_their_type_args_do`,
  `::option_of_non_eq_type_is_rejected`.
- **Owning gate:** closed, WP-C1.3.

## DEV-009 — `File` has no first-class runtime representation

- **Normative expectation:** `06-Standard-Library.md`'s IO module specifies `File::open` +
  `read_to_string` / `File::create` + `write_str`.
- **Current behaviour:** `std::fs::File` doesn't implement `Copy`/`Clone`/`PartialEq`, so it
  cannot fit as `Value::File(std::fs::File)` without restructuring the interpreter's `Value`
  enum's move/copy assumptions. Deferred during Phase 4E; `IOError` (the other half of that
  phase's scope) shipped.
- **User impact:** STARK programs cannot currently open/read/write files through the
  interpreter — any spec-described `File` API is unusable.
- **Security/soundness impact:** none (a missing feature, not an unsound one).
- **Workaround:** none within the language; file I/O must happen outside STARK programs for now.
- **Proposed disposition:** restructure the `Value` enum's move/copy handling to accommodate a
  non-`Copy`, non-`Clone` runtime resource type (`File` would be the first of its kind — `Vec`/
  `String`/`HashMap`/`HashSet` are all currently `Clone`-able Rust types wrapped as `Value`
  variants). This is a real interpreter-value-model design question, not a small patch.
- **Owning gate:** not currently scheduled in C0-C2; candidate for WP-C2.2 (interpreter semantic
  repair) if judged in scope, otherwise needs a dedicated follow-up WP.

## DEV-010 — LSP hover/definition/references are protocol stubs

- **Normative expectation:** Charter WP-C8.2: "a cursor-coordinate string is a stub, not hover
  support." Real semantic services are expected to come from resolved compiler identity.
- **Current behaviour:** the JSON-RPC endpoints exist and respond correctly per protocol
  (`textDocument/hover`, `textDocument/definition`, `textDocument/references` all wired into
  `handle_request`; compiled `TypeTables` are cached per open document), but the handlers don't
  use that data: hover returns a raw `line:character` position string instead of the inferred
  type/signature at that position; definition/references return `null`/`[]` unconditionally.
  Root cause: no span→node lookup exists (walking HIR/AST for the innermost node at a byte
  offset is real work, not wiring) — see `starkc/docs/PHASE8_GRAMMAR_GAPS.md` for the original,
  self-disclosed account.
- **User impact:** an editor client (e.g. the VS Code extension) that requests hover or
  go-to-definition receives a technically-valid but useless response — not an error, just no
  real information. Could be mistaken for "hover works, there's just nothing to show here"
  rather than "hover is unimplemented."
- **Security/soundness impact:** none.
- **Workaround:** none at the tooling level; users needing real navigation must read source
  directly.
- **Proposed disposition:** implement a span→node lookup and a `Ty`-to-source-text renderer
  (the formatter's `printer.rs` already has reusable type-printing logic) as scoped by Charter
  WP-C8.2/C8.3.
- **Owning gate:** WP-C8.2 (hover), WP-C8.3 (definition/references).

## DEV-011 — Doc comments are trivia, not AST/HIR metadata

- **Normative expectation:** none explicitly normative (Core v1 spec doesn't mandate a doc-
  comment representation), but Charter rule 17 ("source identity must survive the pipeline")
  and general tooling-correctness expectations imply queryable structure for anything tools
  (doc generator, future LSP hover) need to associate with specific items.
- **Current behaviour:** `///` doc comments are collected as lexer trivia
  (`Comment`/`CommentKind`, `lexer::tokenize_with_comments`) and re-associated with item spans
  by source position at formatter/doc-gen time — not stored as a first-class AST/HIR field.
  Nothing downstream of parsing (resolve, typecheck, interpreter) can see them.
- **User impact:** none today (the formatter and doc generator both already work around this
  successfully via position-based re-association), but it constrains future tooling: any future
  feature needing "which comment documents this specific resolved item" (e.g. LSP hover showing
  doc text, not just inferred type) must either reuse the same fragile position-matching
  approach or wait for this to be fixed properly.
- **Security/soundness impact:** none.
- **Workaround:** the formatter/doc-gen position-matching approach is the workaround, and it is
  already in production use — this is a forward-looking architecture note, not an active user
  complaint.
- **Proposed disposition:** if a future WP needs item-attached doc comments beyond position
  re-matching, add an `attrs`/`doc` field to `ast::ItemNode` (parallel to how attribute syntax
  would need to attach, see DEV-SEED-014) carrying the associated comment text through to HIR.
  Not scoped to any current WP — informational until a concrete need arises.
- **Owning gate:** none scheduled; revisit if WP-C8.2 (hover) or a documentation feature needs
  it.

## DEV-012 — VS Code extension UI never interactively verified

- **Normative expectation:** Charter WP-C8.7: "protocol tests alone do not prove UI behaviour"
  — real editor validation requires an Extension Development Host or packaged-extension session.
- **Current behaviour:** no `code` CLI / Extension Development Host has been available in the
  implementing environment. Verified so far: TypeScript correctness (`tsc`/ESLint/esbuild
  bundling) and raw LSP JSON-RPC exchange (a script bypassing VS Code entirely). Status bar
  rendering, command palette entries, format-on-save actually firing on a real save, and hover
  popups have never been interactively confirmed.
- **User impact:** unknown — the extension may work correctly in a real VS Code session, or may
  not; this has genuinely not been tested at the UI level despite being labeled "Complete" in
  `WP8_4_VSCODE_EXTENSION_IMPLEMENTATION.md` (with the honest caveat "interactive VS Code
  testing not possible in this environment" already present in that doc's own status line).
- **Security/soundness impact:** none directly, but an untested UI surface is a real release-
  readiness gap if the extension is ever distributed to users.
- **Workaround:** none; a VS Code-capable environment is required to close this gap.
- **Proposed disposition:** run a real Extension Development Host session once available,
  covering at minimum: diagnostics on open/edit/save, hover, format-on-save, and the
  `stark.generateDocs`/tensor-mode-toggle commands.
- **Owning gate:** WP-C8.7.

## DEV-013 — `STD-004` (standard traits) exhaustiveness audit (CLOSED, with new findings)

- **Normative expectation:** `06-Standard-Library.md`'s trait surface (Clone, Hash, Default,
  Display, Error, Iterator) should be recognized both as trait *bounds* and as callable
  *methods*, with default method bodies used when not overridden.
- **Findings, resolved 2026-07-17 (WP-C1.3):**
  - `Error` trait bound checking: **confirmed working.** The original "not seen in the bound-
    name list" observation was checking the wrong function — `satisfies_bound` (the general
    trait-bound checker) handles any struct/enum trait name generically via a real impl-existence
    search, unlike the narrower `require_operator_bound` (only Eq/Ord/Num). Verified end-to-end
    with a real `impl Error for MyError` and a generic `fn describe<E: Error>(e: E) -> String`.
  - `Clone`/`Hash`/`Display` as bounds: confirmed working (same mechanism).
  - **`Clone` as a callable method on compiler-builtin types: confirmed BROKEN, now FIXED.**
    `.clone()` on `String`/`Vec`/`Option`/`Result`/`HashMap`/`HashSet`/`Range`/`IOError` failed
    with E0303 "method call on non-struct/enum type" — recognized as a bound, but with no
    method-signature entry or dispatch case anywhere for any builtin type. Fixed with a generic
    dispatch point in both `core_method_signature` (typecheck.rs) and `call_core_method`
    (interp.rs, reusing `Value`'s existing derived Rust `Clone`).
  - **Default trait method bodies: confirmed BROKEN, now FIXED** (found while testing the trait
    family broadly; squarely inside WP-C1.3's own checklist item "default methods"). A trait
    method with a real default body was never used as a fallback when unoverridden — the HIR
    already carried `TraitItem::Method { body: Some(_), .. }`, it was simply never consulted.
    Fixed in both typecheck.rs (a `default_fallback` search before concluding "not found") and
    interp.rs (`find_method` gained the analogous fallback). Verified both that an unoverridden
    default runs and that an overriding impl still takes precedence.
  - Hand-written-impl-vs-builtin-only question: confirmed — hand-written impls are the normal,
    spec-shown mechanism for all these traits (no separate auto-derive-only mode exists).
- **Regression tests:** `interp.rs::clone_works_for_builtin_core_types`,
  `::default_trait_method_runs_when_not_overridden`,
  `::overriding_impl_takes_precedence_over_trait_default`.
- **New deviations found while closing this one, deliberately NOT fixed in this WP (scope
  discipline after two substantial fixes already landed) — see DEV-023 and DEV-024 below.**
- **Owning gate:** closed, WP-C1.3.

## DEV-014 — `parser.rs` test-environment detection suppressed real errors (CLOSED)

- **Normative expectation:** a genuinely missing `mod foo;` backing file must always produce
  E0202, in every real invocation, per `07-Modules-and-Packages.md`'s multi-file layout rules.
- **Previous behaviour:** `load_submodules_recursive` (`parser.rs`) additionally suppressed this
  diagnostic whenever `std::env::args().any(|arg| arg.contains("test") || arg.contains(
  "conformance"))` — since the `stark test` subcommand's own name contains "test", **every real
  invocation of `stark test` against a package with a genuinely missing module file silently
  accepted it instead of reporting the error.**
- **User impact:** severe if untriggered by this fix — a package author running their own test
  suite (the single most common `stark` invocation pattern) would never see this class of error.
- **Security/soundness impact:** none (availability/correctness, not memory/type safety), but a
  real, unconditional, silently-wrong production behavior in the actual `stark test` path.
- **Workaround:** was none; now fixed.
- **Proposed disposition:** done — removed the `env::args()` clause; kept the filename-based
  bypass needed for one legitimate notation fixture; fixed `conformance.rs` to use full fixture
  paths so that bypass now matches for the reason it was originally intended, not by accident.
- **Owning gate:** closed under WP-C1.1 (2026-07-17). Regression test:
  `starkc/tests/gate2_valid.rs::test_missing_module_file_is_reported_not_silently_accepted`.

## DEV-015 — Suffixed literal overflow is never checked

- **Normative expectation:** per CLAUDE.md, "Integer overflow... always trap — in every build
  mode." A literal whose value exceeds its suffix type's range should be rejected.
- **Current behaviour:** confirmed empirically — `let x: UInt8 = 300u8;` compiles and
  `starkc check` reports clean. No stage checks literal magnitude against suffix range;
  `typecheck.rs`'s `convert_int_suffix` only maps the suffix to a type tag.
- **User impact:** a program can declare an integer literal with a suffix that cannot represent
  its value, and the compiler accepts it silently — later runtime behavior for that value is
  presumably whatever bit-truncation Rust's own literal parsing does underneath, which is itself
  unspecified from STARK's perspective.
- **Security/soundness impact:** low-moderate — not a memory-safety issue, but a real type-system
  soundness gap: the declared type's range guarantee doesn't actually hold for literals.
- **Workaround:** none; be careful with literal/suffix combinations near type boundaries.
- **Proposed disposition:** triage where the check belongs (lexer-level immediate rejection,
  since the lexer already knows the suffix at tokenization time, vs. typecheck/const-eval-level)
  and implement it. Not attempted in WP-C1.1 — a real design decision, not a test-strengthening
  task.
- **Owning gate:** WP-C1.3 or WP-C1.5 (needs triage to pick one).

## DEV-016 — Repository-wide clippy debt (RESOLVED in WP-C1.4)

- **Normative expectation:** Charter §2.5 lists `cargo clippy --all-targets -- -D warnings`
  passing as a default definition-of-done requirement.
- **Original behaviour:** 22 clippy errors existed across `typecheck.rs`, `interp.rs`,
  `lsp/protocol.rs`, and `lsp/server.rs`, none touched by WP-C1.1 (confirmed by isolating clippy
  output to files that WP changed: zero hits). CI (`.github/workflows/ci.yml`'s `fmt, clippy,
  test` job) had been red since the 2026-07-17 03:29 push for exactly this reason, across several
  unrelated feature commits and both governance-bootstrap commits.
- **User impact:** none to compiled-program behavior; this was a code-quality/CI-hygiene gap.
- **Security/soundness impact:** none identified.
- **Resolution:** fixed as a standalone cleanup during WP-C1.4 at the user's explicit request.
  All 22 fixes are mechanical and zero-behavior-change: 13x `args.get(0)` → `args.first()`
  (`typecheck.rs`); 2x explicit-closure-clone → `.cloned()` (`interp.rs`, `lsp/server.rs`); 2x
  manual `if let Some` inside a `for` loop → `.into_iter().flatten()` (`interp.rs`); 3x
  `*inner = Box::new(x)` → `**inner = x` (avoids a needless allocation, `interp.rs`);
  `JsonValue`'s inherent `to_string` → `impl std::fmt::Display` (`lsp/protocol.rs` — no call-site
  changes needed, the blanket `ToString` impl covers `Display`); one
  `.and_then(|x| Some(y))` → `.map(|x| y)` (`lsp/protocol.rs`). Verified clean clippy, clean fmt,
  and the full workspace test suite green twice consecutively with an unchanged pass count.
- **Owning gate:** WP-C1.4 (closed).

## DEV-017 — Coverage database test citations lack function-level precision

- **Normative expectation:** Charter rule 14 — conformance claims require executable evidence,
  ideally traceable to the specific test(s), not just "some test exists in this file somewhere."
- **Current behaviour:** `tests` fields cite files only; `check-conformance.py` validates path
  existence, not that the file's tests actually exercise the described rule. Before WP-C1.1,
  several rules (e.g. LEX-013) cited only `starkc/tests/conformance.rs` despite that file
  contributing zero real coverage for them — actual coverage lived, uncited, in `lexer.rs`'s/
  `parser.rs`'s own inline unit test modules.
- **User impact:** none direct; an engineering-process/auditability gap.
- **Security/soundness impact:** none.
- **Workaround:** cross-reference `starkc/docs/dev/compiler-map.md` and this ledger's inline
  `core-v1-coverage.toml` comments for function-level detail until the schema is extended.
- **Proposed disposition:** WP-C1.6's conformance evidence generator is explicitly scoped to
  produce "positive tests / negative tests" at rule granularity — this deviation is exactly the
  problem that WP exists to solve.
- **Owning gate:** WP-C1.6.

## DEV-018 — AST span-integrity checking was entirely absent (partially closed)

- **Normative expectation:** Charter rule 17 — source identity (spans) must survive the
  pipeline; child nodes' spans should be contained within their parent's.
- **Previous behaviour:** no validation helper existed in `ast.rs`, and no test anywhere
  programmatically checked span containment — `starkc/tests/snapshots.rs` renders span positions
  as text for regression comparison against 15 golden fixtures, which would not catch a
  systematically-wrong-but-stable span.
- **Current behaviour:** `starkc/tests/span_integrity.rs` (new, WP-C1.1) checks child-within-
  parent containment for every `Expr`/`Block` node kind with directly-named children, across the
  full parseable fixture corpus. `Type`/`Pat`/`Item` containment and a fully generic/exhaustive
  visitor remain unchecked.
- **User impact:** none identified from the checking that now exists (all fixtures pass); the
  remaining gap (Type/Pat/Item, generic visitor) is a residual verification blind spot.
- **Security/soundness impact:** none identified; primarily relevant to future tooling
  correctness (e.g. LSP position lookups, DEV-010) rather than compiled-program behavior.
- **Workaround:** none needed for the checked subset.
- **Proposed disposition:** build the full generic, position-indexed AST walker as part of
  WP-C2.4 ("position and symbol query infrastructure"), which needs this exact mechanism for
  span→node lookup regardless of testing concerns — building it twice (once minimal here, once
  properly in C2.4) would be wasted effort.
- **Owning gate:** WP-C2.4 for the general case; WP-C1.1's Expr/Block check is the interim
  evidence and does not need to be redone, only extended.

## DEV-019 — Diagnostic-code collisions with the normative E-code table

- **Normative expectation:** `04-Semantic-Analysis.md`'s E-code table is the single source of
  truth for what each code means; Charter rule 16 requires diagnostics (including codes) remain
  part of testable, deterministic behavior.
- **Current behaviour:** three confirmed collisions. `resolve.rs` uses E0401 ("unresolved
  import") which collides with `flow.rs`'s correct use of E0401 ("use of possibly-uninitialized
  variable" per spec). `resolve.rs` uses E0203 for both "no parent module for super" and "item is
  private," neither of which is "ambiguous name" (spec's actual E0203), colliding with
  `typecheck.rs`'s correct E0203 use for "ambiguous trait method call." `parser.rs` uses E0202
  for module-loading errors ("file not found for module," "conflicting module files"), colliding
  with `resolve.rs`'s own correct E0202 use for "undefined type."
- **User impact:** any tool matching on diagnostic code alone (not message text) cannot
  distinguish these semantically distinct errors.
- **Security/soundness impact:** none — messages are still correct; this is a machine-readable-
  contract gap.
- **Workaround:** match on message text for the affected codes until resolved.
- **Proposed disposition:** spec-bug-protocol candidate — allocate distinct normative E02xx
  codes for the module/import-specific errors currently borrowing codes with unrelated meanings.
  Not done here: reassignment is a public contract change touching multiple test files' exact
  assertions, deserving its own bounded, evidence-backed change.
- **Owning gate:** WP-C1.6 to catch systematically; the reallocation itself is a separate,
  unowned spec-bug-protocol change.

## DEV-020 — `pub use` of a private item leaks it (confirmed design, not a defect)

- **Normative expectation:** none explicit; this pins down an implementation behavior that had
  zero prior test coverage despite dedicated, purpose-built code (`reexport_vis` in resolve.rs).
- **Current behaviour:** a `pub use` of a private item makes it visible from outside — the
  re-export's own visibility overrides the original item's privacy.
- **User impact:** none negative — this is standard re-export/facade-pattern behavior, now just
  verified and pinned down rather than assumed.
- **Security/soundness impact:** none.
- **Workaround:** n/a.
- **Proposed disposition:** none needed; recorded so a future change to this behavior is treated
  as a deliberate semantic change requiring CE1/CE2 escalation, not a routine test update.
- **Owning gate:** closed — informational/confirmed, WP-C1.2.

## DEV-021 — Cross-package coherence checking verified working (previously unverified)

- **Normative expectation:** SEM-007 (orphan rule / overlapping impls) should apply correctly
  across package boundaries in a multi-package workspace.
- **Previous state:** every existing coherence test used an in-memory single file with no real
  `starkpkg.json`, under which `typecheck.rs`'s filesystem-walk-up package-root detection
  (`find_package_root`) always returns `None` — making it impossible to tell from existing tests
  whether cross-package detection worked or every impl was silently treated as same-package.
- **Current state:** a new real two-package-workspace test
  (`gate2_package.rs::test_cross_package_coherence_orphan_rule_with_real_packages`) confirms
  E0500 correctly fires for a genuine cross-package orphan-rule violation.
- **User impact:** none negative — positive confirmation.
- **Security/soundness impact:** none; this closes a soundness *question*, not a soundness gap.
- **Workaround:** n/a.
- **Proposed disposition:** none needed.
- **Owning gate:** closed — verified correct, WP-C1.2.

## DEV-022 — Private-item leakage through public signatures: unimplemented, spec-silent

- **Normative expectation:** none — the spec does not require this check either way.
- **Current behaviour:** no stage checks whether a `pub fn`'s signature or a `pub struct`'s
  fields transitively expose a private type. Confirmed absent in both resolve.rs and
  typecheck.rs.
- **User impact:** a public API can silently expose a type that callers outside the module
  cannot actually name, which is a usability rough edge (a "leaky" public API) rather than a
  soundness gap.
- **Security/soundness impact:** none identified — this affects API ergonomics, not memory or
  type safety.
- **Workaround:** none; be conscious of this when designing public APIs across module
  boundaries.
- **Proposed disposition:** requires a language-design decision (does STARK want this check?)
  and, if yes, a normative spec addition before any implementation — Charter rule 4 forbids
  adding a new rejection rule inside an implementation WP. Not a bug to fix, a feature to decide
  on.
- **Owning gate:** unscheduled; needs a proposal.

---

## DEV-023 — `Display`/`.fmt()` and `Hash`/`.hash()` missing as callable methods on builtin types

- **Normative expectation:** `Display`/`Hash` should be callable as methods on any type
  satisfying the bound, including compiler-builtin types (String, Vec, Option, ...), matching
  the pattern `Clone` now follows after DEV-013's fix.
- **Current behaviour:** the same bug class as DEV-013's Clone finding, confirmed present but
  not fixed: `String::from("hi").fmt()` and `"hi".hash()`-style calls fail with E0303 "method
  call on non-struct/enum type 'String'". `Display`/`Hash` as *bounds* are already correctly
  recognized (same mechanism as Clone/Eq/Ord).
- **User impact:** a generic function bound by `T: Display` or `T: Hash` cannot actually call
  `.fmt()`/`.hash()` on a `T` instantiated with a builtin type, even though the bound check
  passes — the same "bound satisfied, method missing" trap DEV-013 found for Clone.
- **Security/soundness impact:** none identified — a missing-method usability gap, not a
  soundness issue.
- **Workaround:** none for builtin types; works normally for struct/enum types with a
  hand-written `impl Display`/`impl Hash`.
- **Proposed disposition:** by analogy with the Clone fix: `.fmt()` could reuse the
  interpreter's existing `impl fmt::Display for Value` (already used by `print`/`println` for
  exactly these types) as a generic dispatch point. `.hash()` needs its own investigation —
  unverified whether the internal hash used for `HashMap`/`HashSet` keys is exposed in a form
  reusable for a user-callable `.hash()` returning `UInt64`.
- **Owning gate:** unscheduled; candidate for a focused follow-up WP given the fix pattern is
  now well-understood from the Clone precedent.

## DEV-024 — `From` trait associated-function calls fail to resolve

- **Normative expectation:** `impl From<A> for B { fn from(a: A) -> B {...} }` followed by
  `B::from(a)` should resolve and execute the impl.
- **Current behaviour:** confirmed empirically broken — a real `impl From<Celsius> for
  Fahrenheit` followed by `Fahrenheit::from(c)` fails to type-check with E0200 "associated
  function 'from' not found" despite the impl existing.
- **User impact:** the `From`/`Into`/`TryFrom` conversion pattern (`resolve.rs:2080-2082`
  classifies all three as `CoreTrait`s) does not work via the conventional `Type::from(value)`
  call form.
- **Security/soundness impact:** none identified — a missing-resolution usability gap.
- **Workaround:** none currently known; would need a manually-named conversion function instead
  of implementing `From`.
- **Proposed disposition:** root cause not yet isolated — unlike DEV-013's method-call findings,
  this is an *associated/static* function call (`Type::function()`, no receiver value), a
  different resolution path (`find_associated_fn` in interp.rs and its typecheck.rs counterpart)
  that may have an analogous "doesn't search trait impls" gap, or may be specific to `From`'s
  generic trait parameter confusing the self-type match. Needs its own investigation before a
  fix is attempted, not assumed to be the same pattern as DEV-013's fixes. `Into`/`TryFrom` not
  independently tested but plausibly share the same gap.
- **Owning gate:** unscheduled; needs root-cause investigation first.

## Informational (not owned deviations)

These were investigated during WP-C0.2/C0.4 and are recorded for completeness, but are not
normative-conformance gaps requiring a fix — they are either deliberate scope decisions or
low-priority simplification candidates outside any active WP.

### DEV-SEED-008 — Duplicate hand-rolled JSON implementations

`lsp/protocol.rs` and `package.rs` each implement their own independent `JsonValue`/`parse_json`
(`lsp/protocol.rs:17-52,100`; `package.rs:5-58,30`). Not a correctness bug. A future
simplification candidate, out of scope for any current WP per Charter guidance ("avoid broad
refactors that are not required by the active WP"). No owner; revisit opportunistically.

### DEV-SEED-014 — No attribute syntax (`#[test]`, `#[ignore]`, ...)

Confirmed deliberate: no `#` handling in the lexer, no attribute AST node, nothing in
`01-Lexical-Grammar.md`/`02-Syntax-Grammar.md`. `stark test` uses a naming convention
(`fn test_*()`) instead — an explicit, user-approved WP8.3 workaround, not a bug. Not a
deviation from the spec (the spec doesn't have attributes either). Recorded for completeness
since it's a recurring source of plan-vs-reality mismatch in planning documents that assumed
attribute syntax existed. No fix owed.

---

## Cross-references

- `COMPILER-STATE.md` — the append-only decision log this ledger is derived from; carries the
  dated session record for when each entry was found/closed and any status changes since this
  file was last regenerated.
- `starkc/docs/dev/compiler-map.md` — source of DEV-004 through DEV-007 (WP-C0.1 audit).
- `starkc/docs/PHASE8_GRAMMAR_GAPS.md` — source of DEV-010 through DEV-012 (pre-existing,
  independently authored deviation log; this ledger consolidates and cross-cites it rather than
  duplicating its narrative).
- `STARKLANG/conformance/core-v1-coverage.toml` — source of DEV-002 (closed) and DEV-013.
- `starkc/tests/span_integrity.rs`, `starkc/tests/gate2_valid.rs` (new WP-C1.1 tests) — source
  of DEV-014 (closed) through DEV-018.
- `starkc/src/resolve.rs` and `starkc/tests/gate2_package.rs` (new WP-C1.2 tests) — source of
  DEV-019 through DEV-022.
- `starkc/src/typecheck.rs` and `starkc/src/interp.rs` (new WP-C1.3 tests) — source of DEV-008
  and DEV-013's closure, plus DEV-023/DEV-024.
- DEV-001, DEV-003 do not appear above: both IDs were retired when their original seed framing
  was superseded by confirmed findings under different numbers (DEV-SEED-001 → DEV-008;
  DEV-SEED-003 → DEV-009) during WP-C0.2, to avoid two IDs describing the same issue.

Current count: 24 numbered deviations total (DEV-002 through DEV-024), of which DEV-002,
DEV-008, DEV-013, DEV-014, DEV-020, and DEV-021 are closed/confirmed-correct (no fix owed); 2
informational not-owned items (DEV-SEED-008, DEV-SEED-014).
