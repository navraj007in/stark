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

## DEV-004 — `resolve.rs` tensor-builtin gating bug (bare `min`/`max`) (RESOLVED in WP-C1.2)

- **Normative expectation:** Core-only compilation (no `--extension tensor`) must never resolve
  a name to a tensor-extension builtin. Charter §1.5 rule 5: "Core remains extension-neutral."
- **Original behaviour:** `resolve_unqualified` (`resolve.rs:1854-1876`) called `resolve_builtin`
  with no `options.tensor()` gate, unlike the correctly-gated `resolve_path_relative`
  (`resolve.rs:682-685`). Bare `min`/`max` unconditionally resolved to
  `Builtin::TensorMin`/`TensorMax`. `resolve_unqualified` has exactly two call sites: resolving
  `self` (`resolve.rs:659`) and struct-literal shorthand-field lowering (`resolve.rs:1000`).
- **User impact (while open):** narrow but real. In Core-only mode, a struct-literal shorthand
  field named exactly `min` or `max` with no local/module item of that name in scope silently
  resolved to the tensor builtin instead of correctly failing "undefined variable 'min'
  (shorthand field)". If a local named `min`/`max` genuinely existed, normal scope lookup took
  precedence and the bug did not fire.
- **Security/soundness impact:** none directly (no memory/type safety violation), but it was an
  extension-isolation leak — Core-only programs could observably depend on tensor-extension
  identity by accident, undermining the isolation guarantee Gate C9 will need to certify.
- **Resolution:** `resolve.rs`'s `resolve_unqualified` (WP-C1.2, 2026-07-17) now applies the same
  `options.tensor()` gate `resolve_path_relative` already had, before falling back to
  `resolve_builtin`. Verified with a regression test:
  `resolve.rs::bare_min_max_shorthand_field_is_gated_by_tensor_extension`.
- **Owning gate:** WP-C1.2 (closed).

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

## DEV-007 — `resolve.rs` glob-import (`use mod::*`) nondeterminism (RESOLVED in WP-C1.2)

- **Normative expectation:** Charter definition-of-done: "no new... nondeterministic iteration...
  introduced in compiler paths" and "generated output is deterministic across two runs."
- **Original behaviour:** glob-import expansion copied from an unsorted `HashMap<String, Res>`
  (`ModuleData::items`, `resolve.rs:45`) at two call sites (absolute-path and relative-path).
  `insert_module_item` raises `E0204` ("duplicate definition... in the same module scope") on a
  colliding different `Res`. Because Rust's default `HashMap` uses a randomized per-process hash
  seed, which of two glob-colliding names was treated as "first" (silently wins) vs. "second"
  (flagged `E0204`) varied across runs of the same compiler on the same source.
- **User impact (while open):** a program using `use mod::*` with a genuine name collision
  across two glob sources could see `starkc check` report `E0204` on one run and silently accept
  a different (arbitrary) resolution on another run of the identical source and compiler binary —
  a reproducibility failure for both CI and local development.
- **Security/soundness impact:** none directly, but nondeterministic accept/reject on identical
  input undermines trust in diagnostic reproducibility, which downstream tooling (the WP-C1.6
  conformance evidence generator, CI gating) depends on being stable.
- **Resolution:** both glob-expansion call sites in `resolve.rs` (WP-C1.2, 2026-07-17) now sort
  the collected items by name before iterating, making collision-winner selection deterministic.
- **Owning gate:** WP-C1.2 (closed).

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

## DEV-009 — `File` has no first-class runtime representation (RESOLVED in WP-C2.11)

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
- **Resolution:** `FileResource` is a first-class, non-`Copy` interpreter value with
  open/create/read/write/close behavior, consuming close, UTF-8 validation, stable `IOError`
  mapping, and destructor-backed best-effort close. Positive and failure evidence is recorded
  under `STD-IO-001`.
- **Owning gate:** WP-C2.11 (closed).

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

## DEV-015 — Suffixed literal overflow is never checked (RESOLVED in WP-C1.5)

- **Normative expectation:** per CLAUDE.md, "Integer overflow... always trap — in every build
  mode." A literal whose value exceeds its suffix type's range should be rejected. Also
  03-Type-System.md:28: "Default integer type is Int32 for literals that fit, Int64 otherwise"
  (unsuffixed literals).
- **Original behaviour:** confirmed empirically — `let x: UInt8 = 300u8;` compiled and
  `starkc check` reported clean; `let x = 99999999999;` (unsuffixed, exceeds Int32) silently
  typed as a broken Int32 instead of promoting to Int64. No stage checked literal magnitude
  against suffix range; `typecheck.rs`'s `convert_int_suffix` only mapped the suffix to a type
  tag.
- **User impact (while open):** a program could declare an integer literal with a suffix that
  cannot represent its value, and the compiler accepted it silently.
- **Security/soundness impact:** low-moderate — not a memory-safety issue, but a real type-system
  soundness gap: the declared type's range guarantee didn't actually hold for literals.
- **Resolution:** design question settled (user-approved 2026-07-18): typecheck/const-eval time,
  not the lexer — an unsuffixed literal's fit-check needs its inferred target type, which the
  lexer never has. Fixed in `typecheck.rs`'s `check_expr` `Lit::Int` arm: suffixed literals
  checked against their suffix's exact range (new **E0008**, via a new
  `literal::int_suffix_range_contains` helper); unsuffixed literals promoted to Int64 if they
  don't fit Int32, rejected (E0008) if they don't fit Int64 either. A defense-in-depth suffix
  re-check was also added to `interp.rs::eval_lit`. Both share a new `src/literal.rs` module
  (also used to fix a second, previously-unknown bug found while building it — see DEV-025).
- **Owning gate:** WP-C1.5 (closed).

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

## DEV-017 — Coverage database test citations lack function-level precision (PARTIALLY CLOSED
in WP-C1.6)

- **Normative expectation:** Charter rule 14 — conformance claims require executable evidence,
  ideally traceable to the specific test(s), not just "some test exists in this file somewhere."
- **Original behaviour:** `tests` fields cited files only; `check-conformance.py` validated path
  existence, not that the file's tests actually exercise the described rule. Before WP-C1.1,
  several rules (e.g. LEX-013) cited only `starkc/tests/conformance.rs` despite that file
  contributing zero real coverage for them — actual coverage lived, uncited, in `lexer.rs`'s/
  `parser.rs`'s own inline unit test modules.
- **User impact:** none direct; an engineering-process/auditability gap.
- **Security/soundness impact:** none.
- **Resolution (partial):** WP-C1.6 built the conformance evidence generator this deviation was
  explicitly assigned to. Schema extended with `positive_tests`/`negative_tests` (bare path or
  `path::function_name`, the latter validated by `check-conformance.py` for both file and
  function existence) and `deviation` (DEV-NNN cross-reference). Of the 59 tracked rules, 20 now
  have real, individually-verified function-level citations (the 19 that already had
  rule-specific test files, plus LEX-006, found to have real dedicated coverage that was never
  cited at all). `starkc/scripts/generate-conformance-report.py` emits the full per-rule report
  (rule id, chapter, status, source, positive/negative tests, deviation, last-verified commit —
  the last computed fresh via `git log` at generation time, never hand-typed) in JSON or
  Markdown, wired into CI (`fixture-conformance` job: validated, generated, posted to the job
  summary, and uploaded as an artifact).
- **Remaining gap:** 39 of 59 rules still cite only the aggregate `starkc/tests/conformance.rs`
  fixture-corpus runner, which mixes positive/negative coverage for every rule at once with no
  per-rule attribution. Genuinely re-deriving that split would mean determining which of ~121
  shared spec fixtures individually prove which rule — confirmed with the user as out of
  WP-C1.6's effort budget, a real scope tradeoff rather than an oversight. The generator reports
  these 39 explicitly as "unclassified," which is itself new, precise signal (previously only a
  vague "some rules" note existed anywhere).
- **Proposed disposition:** the C2.6 granular split map prevents broad legacy status from being
  copied forward. C2.11 re-cites and classifies positive/negative evidence per granular rule.
- **C2.11 update:** `core-v1-c2.11-evidence.toml` now provides mechanically validated,
  function-level positive and negative citations for the high-cost alignment surface. The 59
  broad entries remain historical transition records rather than being falsely promoted into
  granular claims; C2.12 expands differential evidence across the rest of the executable corpus.
- **Owning gate:** WP-C1.6 tooling and the C2.11 high-cost evidence slice are closed; exhaustive
  differential expansion is WP-C2.12.

## DEV-018 — AST span-integrity checking was entirely absent (RESOLVED in WP-C2.11)

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
- **Proposed disposition:** WP-C2.4 supplied compiler-owned position queries, but did not turn
  Type/Pat/Item containment into exhaustive conformance evidence. C2.11 must either add that
  adversarial evidence or narrow the invariant explicitly.
- **Resolution:** the fixture-corpus containment walk now covers Type, Pattern, and Item arenas
  in addition to Expression, Statement, and Block nodes, including nested item/type/body edges.
- **Owning gate:** WP-C2.11 (closed).

## DEV-019 — Diagnostic-code collisions with the normative E-code table (RESOLVED in WP-C2.11)

- **Normative expectation:** `04-Semantic-Analysis.md`'s E-code table is the single source of
  truth for what each code means; Charter rule 16 requires diagnostics (including codes) remain
  part of testable, deterministic behavior.
- **Current behaviour:** five confirmed collisions. Three from WP-C1.2: `resolve.rs` uses E0401
  ("unresolved import") which collides with `flow.rs`'s correct use of E0401 ("use of
  possibly-uninitialized variable" per spec). `resolve.rs` uses E0203 for both "no parent module
  for super" and "item is private," neither of which is "ambiguous name" (spec's actual E0203),
  colliding with `typecheck.rs`'s correct E0203 use for "ambiguous trait method call." `parser.rs`
  uses E0202 for module-loading errors ("file not found for module," "conflicting module files"),
  colliding with `resolve.rs`'s own correct E0202 use for "undefined type." Two more found during
  WP-C1.5, while touching match-arm code for the exhaustiveness fix: `typecheck.rs`'s "unreachable
  match arm" warning uses E0500 — spec table: E0500="Trait not implemented" (an *error*, not a
  warning) — colliding with 15 other, spec-correct E0500 "trait not implemented" error sites in
  the same file. `typecheck.rs`'s "method call on non-struct/enum type" error uses E0303 — spec
  table: E0303="Non-exhaustive match" — colliding with the (WP-C1.5-strengthened) spec-correct
  E0303 exhaustiveness sites.
- **User impact:** any tool matching on diagnostic code alone (not message text) cannot
  distinguish these semantically distinct errors.
- **Security/soundness impact:** none — messages are still correct; this is a machine-readable-
  contract gap.
- **Workaround:** match on message text for the affected codes until resolved.
- **Proposed disposition:** spec-bug-protocol candidate — allocate distinct normative E02xx
  codes for the module/import-specific errors currently borrowing codes with unrelated meanings,
  plus (WP-C1.5 additions) a new W0xxx code for "unreachable match arm" (it's a warning, not an
  error, so E0500 was always the wrong category regardless of the collision) and a new E00xx code
  for "method call on non-struct/enum type." Not done here: reassignment is a public contract
  change touching multiple test files' exact assertions, deserving its own bounded,
  evidence-backed change.
- **Resolution:** module/import/private/public-API failures now use distinct `E0205`–`E0209`;
  executable, constant, alias, and sizedness failures use `E0214`–`E0217`; invalid receiver and
  constant-pattern categories use `E0304`/`E0305`; unreachable arms use warning `W0006`.
  Exact-code regression assertions were updated with the catalogue.
- **Owning gate:** WP-C2.11 (closed).

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

- **Normative expectation:** `TRAIT-COHERENCE-001` and `TRAIT-COHERENCE-002` require the
  orphan and overlap rules to apply across the complete resolved package graph, independent of
  source order. C2.9 supplies the canonical package/version token used by those algorithms.
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
- **Owning gate:** closed as the original orphan-rule verification, WP-C1.2. C2.11 must
  reclassify granular evidence against both C2.8 coherence rules after C2.9 fixes package
  identity; this does not reopen DEV-021 as a known compiler defect.

## DEV-022 — Private-item leakage through public signatures (RESOLVED in WP-C2.11)

- **Normative expectation:** `MOD-REEXPORT-001` requires every transitive item in a public
  signature to be publicly nameable by consumers.
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
- **Proposed disposition:** implement the C2.9-approved public-reachability check with positive
  and negative cross-package evidence.
- **Resolution:** type checking walks every exported signature recursively and reports E0209 for
  unnameable types while accepting types made nameable by a public re-export.
- **Owning gate:** WP-C2.11 (closed).

---

## DEV-023 — builtin `Display`/`Hash` methods (RESOLVED in WP-C2.11)

- **Normative expectation:** `TYPE-METHOD-001` and `TYPE-METHOD-002` require ordinary,
  source-order-independent trait-method selection for any receiver satisfying the bound.
  `STD-HOOK-001` does not classify `Display::fmt` or `Hash::hash` as compiler hooks, so builtin
  types must participate through the same ordinary dispatch contract rather than name-based
  interpreter handling.
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
- **Resolution:** builtin receivers expose callable `.fmt()` and `.hash()` with the frozen
  canonical display bytes and standard FNV-1a encoding; float Hash bounds remain rejected.
- **Owning gate:** WP-C2.11 (closed).

## DEV-024 — `From` trait associated-function calls fail to resolve (RESOLVED in WP-C2.11)

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
- **Resolution:** static associated lookup searches matching trait impls after inherent
  candidates, executes the selected body, and reports E0204 for ambiguous trait candidates.
- **Owning gate:** WP-C2.11 (closed).

## DEV-025 — `pat_subsumes` compared literal patterns by shape only, not value (RESOLVED in
WP-C1.5)

- **Normative expectation:** the "unreachable match arm" lint should only fire when a later arm's
  pattern is genuinely covered by an earlier one.
- **Original behaviour:** `Lit` (the AST/HIR literal-shape tag) carries no value for Int/Float/Str
  — only base/suffix/raw shape info. `pat_subsumes` compared `Lit == Lit` directly, so any two
  same-kind literal patterns were treated as equal regardless of actual value. Confirmed
  empirically: `match x: Int32 { 1 => .., 2 => .. }` and `match x: &str { "a" => .., "b" => .. }`
  both spuriously flagged the second, genuinely-distinct arm as redundant/unreachable. This fired
  on essentially every real-world literal match with 2+ arms.
- **User impact:** false-positive "unreachable match arm" (W-class, currently mislabeled E0500 —
  see DEV-019) warnings on common, correct code.
- **Security/soundness impact:** none — a spurious warning, not an incorrect accept/reject.
- **Resolution:** found while building `src/literal.rs` for DEV-015. `pat_subsumes` now parses
  both literals' actual values via `literal::eval_lit_value` and compares those instead of the
  shape-only `Lit` tag.
- **Owning gate:** WP-C1.5 (closed).

## DEV-026 — Method dispatch priority ignores "inherent shadows trait" (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md` "Method Calls and Auto-Borrowing" (line 493–494):
  inherent methods shadow trait methods of the same name — this must hold unconditionally.
- **Current behaviour:** `interp.rs::find_method` resolves a method call via a linear scan over
  every HIR item in source/declaration order, returning the first matching `impl` block
  (inherent or trait) it finds — there is no separate "check inherent first" pass. Confirmed
  empirically: with a struct `Thing`, a trait `Speak` providing a default `fn say(&self) ->
  String { "trait-default" }`, `impl Speak for Thing {}` (uses the default), and a separate
  inherent `impl Thing { fn say(&self) -> String { "inherent" } }`, calling `t.say()` returns
  whichever impl block appears first in the source file — `"trait-default"` if the trait impl is
  textually first, `"inherent"` if the inherent impl is textually first. Per spec, inherent must
  win unconditionally, regardless of source order.
- **User impact:** a program relying on inherent-method-shadows-trait-default (a normal pattern:
  "use the trait's default unless I override it inherently") gets silently wrong behavior
  whenever the trait impl happens to be declared before the inherent impl in the file — no
  diagnostic, no error, just the wrong method body runs.
- **Security/soundness impact:** none identified — a correctness/predictability gap, not a
  memory-safety or type-safety violation.
- **Workaround:** declare inherent `impl` blocks before any trait `impl` block that could provide
  a same-named default method, for any type where this matters.
- **Proposed disposition:** `find_method` needs a two-pass search (inherent impls first,
  unconditionally, then trait impls) rather than a single source-order scan.
- **Resolution:** `find_method` now performs inherent-first and trait-second lookup, independent
  of declaration order. Regression:
  `interp::tests::inherent_method_shadows_trait_method_regardless_of_declaration_order`.
- **Owning gate:** closed, WP-C2.2.

## DEV-027 — `Ordering` prelude type unresolvable; no runtime `Ord`/`cmp` dispatch for struct/enum (RESOLVED in WP-C2.2)

- **Normative expectation:** `06-Standard-Library.md` line 585 lists `Ordering` as part of the
  normative prelude ("Prelude: primitive types, `Option`, `Result`, `Ordering`, essential
  traits"); lines 76–81 define `enum Ordering { Less, Equal, Greater }`; the `Ord` trait's
  required signature (line 111–113) is `fn cmp(&self, other: &Self) -> Ordering`.
  `03-Type-System.md` line 516–531's operator-desugaring table maps `<`/`<=`/`>`/`>=` to
  `Ord::cmp` compared against `Ordering`.
- **Current behaviour:** two-part finding. (a) `Ordering` does not exist as a resolvable name
  anywhere in the compiler — no `hir::CoreType` entry, no prelude registration. A program
  declaring `impl Ord for Point { fn cmp(&self, other: &Point) -> Ordering { ... } }` and
  returning `Ordering::Less`/`Greater`/`Equal` fails to compile with `[E0202] undefined type
  'Ordering'` plus `[E0200] undefined variable 'Ordering::...'` errors — a conforming `impl Ord`
  per the spec's own trait signature cannot currently be written at all. (b) Independently,
  `interp.rs::eval_binary`'s `<`/`<=`/`>`/`>=` handling has arms only for `(Int, Int)`,
  `(Float, Float)`, and `(String|Str, String|Str)` — no struct/enum arm exists, unlike the
  `Eq`/`eq` dispatch DEV-008 added. `typecheck.rs::ty_satisfies_operator_bound` already accepts
  `Ty::Struct`/`Ty::Enum` for the `"Ord"` bound whenever a matching `impl Ord for T` exists, so if
  (a) were fixed in isolation, a struct/enum `<` comparison would type-check and then crash at
  runtime with `"invalid binary operation"` — the same compile-time/runtime mismatch class `==`/
  `!=` had before DEV-008's fix.
- **User impact:** the entire `Ord`/comparison-operator-overloading feature for user types is
  currently non-functional, both at the "can I even write it" layer and the "does it dispatch at
  runtime" layer.
- **Security/soundness impact:** none identified — a missing-feature gap, not unsound.
- **Workaround:** none; a user type cannot implement `Ord` and use `<`/`<=`/`>`/`>=` today.
- **Proposed disposition:** register `Ordering` as a prelude type (mirroring how `Option`/
  `Result` are registered) with its three unit variants, then add an `eval_binary` struct/enum
  arm mirroring DEV-008's `Eq`/`eq` dispatch fix, calling the resolved `Ord::cmp` and comparing
  the returned `Ordering` against `Less`/`Greater`/`Equal`.
- **Resolution:** `Ordering` is now a resolvable/runtime core type with all three variants, and
  nominal comparison operators dispatch through `Ord::cmp`. Regression:
  `interp::tests::nominal_comparison_dispatches_through_ord_cmp`.
- **Owning gate:** closed, WP-C2.2.

## DEV-028 — `&expr[range]`/`&mut expr[range]` crash at runtime; slice materialization copies instead of viewing (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md` lines 95–98 (Array Types): `expr[r]` for a
  `Range` `r` denotes a place of unsized slice type `[T]`; `&expr[r]` has type `&[T]` and
  `&mut expr[r]` has type `&mut [T]` — both spec-mandated. `05-Memory-Model.md` lines 51–54
  describe `&[T]`/`&mut [T]` as pointer-plus-length views into existing storage (mutation through
  a mutable slice must be observable in the source collection).
- **Current behaviour:** `interp.rs::expr_place`'s `Index` arm unconditionally calls
  `self.expect_int(*index)`, which fails whenever the index is a `Value::Range`. Both
  `let s: &[Int32] = &arr[1..4];` and `let s: &mut [Int32] = &mut arr[1..4];` crash with
  `runtime error: expected integer` at the range-index's span — the spec-mandated place form
  simply does not work. Separately, the one code path that *does* produce a slice value
  (`slice_value`, reached only in a value context, e.g. `let v = arr[1..4];`, never through
  `&`/`&mut`) clones the underlying elements into a new `Value::Array` — a disconnected copy, not
  a view, so even if the place-crash were fixed by routing through the same helper, mutations
  through a resulting `&mut [T]` would silently fail to propagate back to the source collection.
- **User impact:** taking a reference to a range-indexed place (the normative slice-place syntax)
  is completely broken; the only working range-index path produces a copy, silently diverging
  from view semantics if ever connected to the place path without further work.
- **Security/soundness impact:** none identified for the crash (it's a hard runtime error, not
  silent corruption); the copy-vs-view gap is a correctness issue for any future fix, not a
  currently-observable soundness bug (since no working mutable-slice-through-range path exists at
  all yet to observe the divergence).
- **Workaround:** none for `&`/`&mut` of a range-indexed place; use `.iter()`/index-by-scalar
  loops instead.
- **Proposed disposition:** `expr_place`'s `Index` arm needs a `Value::Range` case producing a
  genuine slice-place representation (not a value copy) so both the crash and the copy-vs-view
  gap are fixed together, not just the crash in isolation.
- **Resolution:** runtime slices now carry a base place plus half-open bounds. Scalar projection
  through a slice maps back to the original array/vector, so shared reads and mutable writes are
  genuine views; range bounds and display are preserved. Regression:
  `interp::tests::range_index_references_are_slice_views`.
- **Owning gate:** closed, WP-C2.2.

## DEV-029 — Struct/enum named-field drop order is alphabetical, not declaration order (RESOLVED in WP-C2.2)

- **Normative expectation:** `05-Memory-Model.md` "Drop Order" now states explicitly (added under
  CD-011, in response to an external review that correctly caught this deviation was originally
  recorded against an *inferred* extension of the spec rather than explicit text): fields drop in
  reverse of their declaration order in the `struct`/`enum` item, extending the pre-existing
  reverse-declaration-order rule for sibling `let` bindings.
- **Current behaviour:** `interp.rs::drop_value` drops a `Value::Struct`'s fields via
  `fields.values_mut().rev()` where `fields: BTreeMap<String, Option<Value>>` — i.e.
  reverse-**alphabetical-by-field-name** order, not reverse-declaration order (same for
  `Value::Enum`'s struct-like-variant `named` map). Verified empirically: a struct with fields
  declared `alpha` then `beta`, and the same struct with fields declared `beta` then `alpha`,
  both drop in the identical order (`beta`, then `alpha`) regardless of which was actually
  declared first — conclusively showing the order tracks alphabetical field-name sort, invariant
  to real declaration order. Tuple/array/tuple-enum-variant fields (`Vec`-backed) are unaffected,
  since a `Vec` preserves insertion order; only `BTreeMap`-backed named fields are affected.
- **User impact:** a `Drop` impl relying on field drop order (e.g. dropping a lock-holder field
  after the fields it protects) gets silently wrong, source-order-independent behavior.
- **Security/soundness impact:** none identified directly, but drop-order-dependent resource
  cleanup (the classic case Drop exists for) could misbehave in ways specific to whatever the
  field names happen to alphabetize to, independent of the programmer's actual declared order.
- **Workaround:** none within the language; be aware field drop order follows alphabetical field
  name, not declaration order, until fixed.
- **Proposed disposition:** either (a) switch the struct/enum-named-field runtime representation
  from `BTreeMap` to an order-preserving map (e.g. an insertion-ordered map or a
  `Vec<(String, Option<Value>)>`), or (b) keep `BTreeMap` for lookup but track declaration order
  separately for drop purposes. Option (a) is likely simpler and also fixes the same underlying
  order-loss for any other alphabetical-iteration-dependent behavior not yet found.
- **Resolution:** named aggregate cleanup recovers declaration order from HIR and drops in
  reverse, including unbound fields in partial pattern moves. Regressions:
  `interp::tests::struct_fields_drop_in_reverse_declaration_order`,
  `::enum_variant_named_fields_drop_in_reverse_declaration_order`, and
  `::unbound_struct_pattern_fields_use_reverse_declaration_order`.
- **Owning gate:** closed, WP-C2.2.

## DEV-030 — Pattern-match wildcard/unbound sub-values of an owned scrutinee are never dropped (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md` line 548–550: "every owned value's destructor
  runs exactly once: at end of scope, at explicit `drop`, or when its owner is consumed — never
  twice" (and, by the same soundness logic that motivates drop-flag tracking for partial moves,
  never *zero* times either for a value whose ownership was genuinely consumed).
- **Current behaviour:** matching an owned (by-value) scrutinee and leaving part of it unbound
  (`_`, an unmentioned struct field, a `Wild` sub-pattern) means that portion's `Drop::drop` is
  **never invoked, for the remainder of the program** — not dropped late, not dropped at the
  wrong time, but permanently skipped. Root cause: `match_pattern`'s handling of `Wild`/unmatched
  fields only *reads* the relevant sub-value for pattern-testing purposes; the original scrutinee
  `Value` (built once, moved from its source place) is a plain Rust-level local that goes out of
  Rust scope at the end of the `Match` expression's evaluation without ever being passed to
  `drop_value`. Verified conclusively: `match (Loud("first"), Loud("second")) { (a, _) => {
  println("matched"); } }` followed by `println("after match")` prints `matched`, `first`, `after
  match` — `"second"`'s destructor never runs, at any point, including after the program's `main`
  returns normally (exit code 0).
- **User impact:** any `Drop`-holding resource (e.g. anything analogous to a file handle, lock,
  or connection, modeled as a struct with a `Drop` impl) that ends up in a wildcard/unmentioned
  position of a by-value match silently leaks — no error, no diagnostic, no crash, the resource
  is simply never released by the language's own cleanup mechanism for the rest of the program's
  execution.
- **Security/soundness impact:** the ledger's other open deviations are explicitly non-soundness-
  relevant (see `C1-exit-report.md`'s "Why not plain CONFORMING" section); this one is closer to
  the line — it is not a memory-safety violation in the Rust-host sense (no use-after-free at the
  interpreter's own implementation level, since `Value` is just dropped as an ordinary Rust value
  eventually), but it is a **violation of Core v1's own stated Drop-soundness invariant** at the
  STARK-program level, silently and with no diagnostic. Recorded as high-priority for this reason
  even though it is not (per this WP's own analysis) a host-memory-safety bug.
- **Workaround:** avoid `_`/wildcard/unmentioned-field patterns when matching an owned scrutinee
  whose unbound portion has a `Drop` impl anywhere in its type; bind every part explicitly (even
  to an unused name) so it participates in normal `cleanup_locals` drop tracking instead.
- **Proposed disposition:** `match_pattern` needs to route any sub-value it does *not* bind
  (i.e., every value reachable from the scrutinee that no `Binding` pattern claims) through
  `drop_value` before the match's Rust-level locals go out of scope — most naturally by having
  the match-evaluation code walk the *unclaimed* portion of the scrutinee's value tree after
  pattern testing completes and explicitly drop it, symmetric to how `cleanup_locals` already
  handles ordinary `let` bindings.
- **Resolution:** after the selected arm and its bindings are cleaned up, the interpreter walks
  the consumed scrutinee and drops every unbound subtree exactly once; reference scrutinees are
  excluded because they do not own the referent. Regressions cover tuple wildcards, enum
  payloads, struct fields, declaration order, and borrowed scrutinees.
- **Owning gate:** closed, WP-C2.2.

## DEV-031 — `for` loops only accept `Range`/`Array`/`Vec` directly, not general `Iterator`-typed expressions (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md` "For Loops" (lines 459–469): `for x in expr`
  requires `expr` to have a type implementing `Iterator`, explicitly citing `.iter()` methods on
  slices/collections as a normal way to produce such an expression.
- **Current behaviour:** `interp.rs::eval_expr`'s `ExprKind::For` calls `iter_values`, which only
  accepts `Value::Range` (eagerly materialized) and `Value::Array`/`Value::Vec` (consumed by
  value) — anything else, including the exact `.iter()` case the spec names, errors with "value
  is not iterable." This is also caught at compile time: `typecheck.rs`'s `for`-loop type-checking
  independently recognizes only the same Range/Array/Vec shapes, so `for x in v.iter() { ... }`
  fails to compile with `[E0001] for-loop requires an iterable value, found 'VecIter<Int32>'`
  (both layers agree with each other — this is a real feature gap, not a compile-succeeds/
  runtime-crashes mismatch like DEV-027).
- **User impact:** `HashMap::keys()`, `.iter()`, and any `MapIter`/`FilterIter` combinator chain,
  or any user type implementing `Iterator`, cannot be used directly as a `for`-loop's iterable —
  only manual `.next()` calls in a `while`/`loop` work for those.
- **Security/soundness impact:** none identified — a missing-feature gap.
- **Workaround:** use a `while let Some(x) = it.next() { ... }`-style manual loop instead of
  `for x in it { ... }` for any iterator that isn't a bare `Range`/`Array`/`Vec`.
- **Proposed disposition:** both `typecheck.rs`'s for-loop type check and `interp.rs::iter_values`
  need to accept any `Iterator`-implementing type (the existing `MapIter`/`FilterIter`/etc.
  `Value` variants and their `iterator_step` protocol), not just the three hardcoded shapes.
- **Resolution:** type checking derives the element type from standard iterator core types or a
  nominal `Iterator::Item` implementation, and execution repeatedly invokes the iterator
  protocol. Regression: `interp::tests::for_loop_accepts_standard_and_user_iterators`.
- **Owning gate:** closed, WP-C2.2.

## DEV-032 — `HashMap`/`HashSet` sort by structural `Ord`, not first-insertion order (RESOLVED in WP-C2.2)

- **Normative expectation:** `06-Standard-Library.md` "Iteration Order" (added under CD-009,
  correcting CD-008's originally-broken sorted-by-`K::Ord` rule — see CD-009's decision-log entry
  for why sorted order doesn't work: `K`/`T` are only bound `Hash + Eq`, never `Ord`):
  `HashMap`/`HashSet` iteration MUST follow first-insertion order.
- **Current behaviour:** `interp.rs`'s `Value::HashMap`/`Value::HashSet` are backed by
  `BTreeMap<Value, Option<Value>>`/`BTreeSet<Value>`, sorted by `Value`'s own internal structural
  `Ord` implementation (a Rust-level total order over the runtime representation, unrelated to
  whether the STARK key type itself implements `Ord`). This tracks insertion order only by
  coincidence, when keys happen to be inserted already in ascending structural order.
- **User impact:** a program relying on the spec's first-insertion-order guarantee (e.g. printing
  a `HashMap`'s contents and expecting insertion order) currently observes sorted order instead —
  silently different from the normative rule for any non-monotonic insertion sequence.
- **Security/soundness impact:** none identified — a determinism-model mismatch, not unsound.
- **Workaround:** none within the language; be aware current iteration order is sorted, not
  insertion-order, until fixed.
- **Proposed disposition:** replace the `BTreeMap`/`BTreeSet` representation with an
  insertion-order-preserving structure (e.g. a `Vec<(Value, Option<Value>)>` for small maps, or
  a proper "indexed map" — an index `HashMap<Value, usize>` alongside an insertion-ordered
  `Vec`, matching how most "ordered map" libraries implement this) so `insert`/`remove`/re-`insert`
  match the spec's stated position rules exactly.
- **Resolution:** both collections now use insertion-ordered reference-interpreter
  representations; replacement preserves position and remove/reinsert appends. Equality remains
  order-independent. Regressions:
  `interp::tests::hashmap_iterates_in_first_insertion_order` and
  `::hashset_iterates_in_first_insertion_order`.
- **Owning gate:** closed, WP-C2.2.

## DEV-033 — `call_core_method` evaluates arguments before resolving the receiver (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md` "Evaluation Order" (added under CD-007,
  confirmed as-is under CD-010): "the receiver evaluates before any argument" for method calls.
- **Current behaviour:** `interp.rs::call_method` correctly resolves the receiver before
  evaluating arguments for user-defined (nominal struct/enum) types — but for builtin/stdlib-type
  methods (`Vec`, `String`, `HashMap`, etc.), routed through `call_core_method`, argument
  expressions are evaluated first (`args.iter().map(|arg| self.expect_value(*arg))...`), and the
  receiver place is resolved lazily, per-operation, inside each method-name branch afterward.
- **User impact:** a program whose receiver expression and argument expressions both have
  observable side effects (e.g. `println`) will see a different interleaving depending on whether
  the receiver is a user-defined type or a builtin/stdlib type — an internal inconsistency a
  program author has no way to predict without knowing the implementation's dispatch mechanism.
- **Security/soundness impact:** none identified — an evaluation-order inconsistency, not unsound.
- **Workaround:** none within the language; avoid relying on receiver-vs-argument evaluation
  order for expressions with observable side effects until fixed.
- **Proposed disposition:** `call_core_method` needs to resolve the receiver place before
  evaluating any argument expression, matching `call_method`'s order for user-defined types.
- **Resolution:** core calls resolve and normalize their receiver place once before evaluating
  arguments, then reuse it throughout dispatch. Regression:
  `interp::tests::core_method_receiver_resolves_before_arguments_and_only_once`.
- **Owning gate:** closed, WP-C2.2.

## DEV-034 — By-value method receiver expressions are evaluated twice (RESOLVED in WP-C2.2)

- **Normative expectation:** each subexpression evaluates exactly once (implicit in
  `03-Type-System.md`'s "Evaluation Order," CD-007/CD-010 — an evaluation-order rule presupposes
  each subexpression has one evaluation to be ordered against others, not that it may re-run).
- **Current behaviour:** for a method call `expr.method(args)` where `method` takes `self` by
  value (not `&self`/`&mut self`) and `expr` is **not** a simple place (e.g. it is itself a
  function call, or any other computed, non-lvalue expression): `call_method` first evaluates
  `expr` once via `clone_expr_place` (to determine, from the resulting value's runtime shape,
  which method implementation to dispatch to) — for a non-place expression this stores the result
  in a synthetic temporary place. `call_user_method`'s `hir::Receiver::Value` arm then calls
  `self.expect_value(base)` on the **original** expression a second time, fully re-evaluating it
  from scratch, completely independent of the first evaluation's stored temporary. Confirmed
  empirically:
  ```stark
  struct Counter { n: Int32 }
  impl Counter { fn consume(self) -> Int32 { self.n } }
  fn make_counter() -> Counter { println("making"); Counter { n: 1 } }
  fn main() -> Unit { let r = make_counter().consume(); println(r); }
  ```
  prints `making` **twice** for one logical call to `make_counter()`.
- **User impact:** any observable side effect in a by-value method's receiver expression
  (printing, mutation of a captured value, a further nested call with its own side effects) is
  silently duplicated. This is not an edge case — `expr.consume_style_method()` where `expr` is
  itself a call or computed expression is an entirely ordinary pattern (method chaining, builder
  patterns, etc.).
- **Security/soundness impact:** none identified as memory-unsafe, but a real correctness defect:
  a program's observable behavior (output, external side effects) differs from what a single
  evaluation of the receiver expression would produce, unconditionally for this call shape.
- **Workaround:** bind the receiver to a `let` first (`let tmp = make_counter(); tmp.consume();`)
  to force a single evaluation, rather than chaining directly.
- **Proposed disposition:** `call_user_method`'s `hir::Receiver::Value` arm should reuse the
  already-computed `borrowed_receiver` value (passed in from `call_method`, already the correct,
  single evaluation of the receiver expression) rather than calling `expect_value(base)` again.
- **Resolution:** method dispatch resolves the receiver once into a caller-side place; by-value
  binding consumes that place rather than re-evaluating the source expression. Regression:
  `interp::tests::by_value_receiver_expression_evaluates_exactly_once`.
- **Owning gate:** closed, WP-C2.2.

## DEV-035 — References returned from `&self` methods dangle after the method frame is popped (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md` "References and Lifetimes" (the shortest-input-
  lifetime rule, WP-C1.4) makes returning a reference derived from a reference parameter —
  including `&self` — an entirely ordinary, spec-legal, borrow-checker-approved pattern. A method
  such as `fn value_ref(&self) -> &Int32 { &self.value }` must work.
- **Current behaviour:** inside a user method call, `self` is stored as a value in a newly-pushed
  method call frame (`call_user_method`). A returned `&self.field` evaluates to a `Value::Ref`
  whose `Place` points into that method's own frame index. `call_user_method` then calls
  `cleanup_current_frame` and pops the frame **before** the return value is handed back to the
  caller — so the returned `Value::Ref` now points at a frame slot that either no longer exists or
  has been reused by a different, unrelated frame. The caller's subsequent dereference of the
  returned reference fails with a runtime `"dangling reference"` error. Confirmed empirically:
  ```stark
  struct BoxedValue { value: Int32 }
  impl BoxedValue { fn value_ref(&self) -> &Int32 { &self.value } }
  fn main() -> Unit {
      let b = BoxedValue { value: 42 };
      let r = b.value_ref();
      println(*r);
  }
  ```
  fails with `runtime error: dangling reference` at the `println(*r)` line.
- **User impact:** severe and broad — essentially every idiomatic accessor/getter method that
  returns a reference into `self` (the exact pattern `03-Type-System.md`'s own shortest-input-
  lifetime rule is written to make legal) crashes unconditionally at runtime, despite compiling
  cleanly (the borrow checker correctly accepts the program, since the *static* analysis is sound
  — this is purely a runtime frame-lifecycle bug).
- **Security/soundness impact:** none identified as memory-unsafe at the interpreter's own
  (Rust-host) level — the failure mode is a caught runtime error, not undefined behavior — but it
  is a compile-accepts/runtime-always-crashes gap for a large, common, spec-legal program shape,
  which in practical terms is more disruptive than DEV-030/DEV-034 despite being "just" an error
  rather than silent misbehavior.
- **Workaround:** avoid returning references derived from `&self`/`&mut self`; return an owned
  clone of the value instead, where possible.
- **Proposed disposition:** the returned value needs to be checked/rewritten before the method
  frame is popped — either by detecting a `Value::Ref` pointing into the about-to-be-popped frame
  and rebasing its `Place` to the caller's own view of the receiver (the place `call_method`
  already resolved via `clone_expr_place`/`expr_place` before the call), or by deferring the frame
  pop until after any such rebasing is complete. Needs careful design given the existing
  `&mut self` write-back path (`call_user_method`'s `RefMut` handling) already does something
  structurally similar for the receiver itself, just not yet for values borrowed *from* it.
- **Resolution:** return values are traversed before handoff and every place derived from the
  method-frame `self` slot is rebased onto the caller's resolved receiver place. Regressions cover
  `&self`, `&mut self`, nested method chains, and preserved write-through semantics.
- **Owning gate:** closed, WP-C2.2.

## DEV-036 — Parser's filename-based module-bypass heuristic remains a residual risk for real user projects (RESOLVED in WP-C2.12)

- **Normative expectation:** a genuinely missing `mod foo;` backing file must always produce
  E0202, in every real invocation, per `07-Modules-and-Packages.md`'s multi-file layout rules —
  the same expectation DEV-014 already states.
- **Current behaviour:** `parser.rs::load_submodules_recursive` still suppresses the "file not
  found for module" diagnostic whenever the current file's name is exactly `"test.stark"`, or
  contains the substring `"spec-fixtures"` or `"STARKLANG"` — a narrower heuristic DEV-014's
  WP-C1.1 fix deliberately kept (after removing the far more dangerous, unconditionally-firing
  `env::args()`-based bypass) because one legitimate spec fixture
  (`07-Modules-and-Packages__01.stark`, a `parse-pass` notation example) needs a
  backing-file-optional `mod math;` declaration to extract correctly.
- **User impact:** narrow but real and not previously flagged as a residual concern when DEV-014
  closed. A real user project whose compiled file's path happens to contain the substring
  `"spec-fixtures"` or `"STARKLANG"` (e.g. a directory literally named `STARKLANGClone`, or a
  project nested under a path containing that substring), or whose entry file is named exactly
  `test.stark`, would silently accept a genuinely missing module file instead of reporting E0202
  — the same class of silent failure DEV-014 was about, at lower but non-zero real-world
  likelihood (filename/path collision, not "every invocation of a specific subcommand").
- **Security/soundness impact:** none identified — an availability/correctness gap (a real error
  silently suppressed), not memory/type unsafety.
- **Workaround:** avoid naming a project's entry file exactly `test.stark`, or placing a project
  under a path containing the substrings `"spec-fixtures"` or `"STARKLANG"`.
- **Proposed disposition:** stop keying this off the compiled file's name/path entirely. The one
  legitimate fixture is already identified precisely in `STARKLANG/tests/spec-fixtures/
  manifest.toml`'s own triage data (machine-readable, structured) — route the exemption through
  that data (e.g. a test-harness-only flag passed explicitly for that one fixture) rather than a
  runtime string-match against arbitrary file paths that can collide with real user projects.
- **Owning gate:** WP-C2.12. Found during the WP-C2.1 correction pass (external review); not a
  new bug (present since WP-C1.1's DEV-014 fix), but its residual risk to real user projects was
  not previously flagged or recorded as its own deviation. Scheduled by the WP-C2.2 correction
  pass alongside the differential corpus's multi-file hardening coverage.
- **Resolution:** implemented exactly the proposed disposition. `parser.rs`'s
  `load_submodules_recursive` no longer string-matches the compiled file's name/path at all —
  the `is_conformance` check (and the three conditions it tested) is removed outright. A new
  `allow_missing_modules: bool` parameter (threaded through the function and its recursive
  self-call) controls whether a missing backing file is reported; every existing public entry
  point (`parse`, `parse_with_options`, `parse_project`, `parse_package_graph`) defaults it to
  `false` and is otherwise unchanged. A new, explicitly-named public function,
  `parse_project_allowing_missing_modules`, sets it to `true`; only
  `starkc/tests/conformance.rs` calls it, gated by a small `const ALLOW_MISSING_MODULE_FILES: &[&str]`
  naming the exact fixture by filename (currently just
  `"07-Modules-and-Packages__01.stark"`) — an explicit, harness-side opt-in rather than a runtime
  path heuristic, matching the disposition's "route the exemption through [structured] data"
  request as closely as the manifest's existing flat-TOML schema allows without a schema change.
  **Correction to this entry's own prior text:** `07-Modules-and-Packages__01.stark`'s manifest
  verdict is `parse-pass`/`mode = "program"`, not `notation` as originally written above (checked
  directly against `STARKLANG/tests/spec-fixtures/manifest.toml:512-514` while implementing the
  fix) — it *is* exercised by `conformance.rs`'s `spec_conformance` enforcement loop, which is
  exactly why the exemption was needed there. The diagnostic code is `E0208` ("file not found for
  module"), not `E0202` as this entry's "Normative expectation" line stated; `E0208` is what
  `parser.rs` actually allocates for every branch of this function (missing file, conflicting
  files, unreadable file) — the `E0202` figure was a typo in the original finding, not a second,
  separate collision (DEV-019 already tracks real E-code collisions elsewhere and is unaffected
  by this correction).
- **Regressions:** one pre-existing test incidentally depended on the removed bypass
  (`parser::tests::item_kinds`'s `"mod math;"` syntax-shape check, via its shared `parse_ok`
  helper's bare `SourceFile` happening to be named `"test.stark"`) and was fixed to call
  `parse_project_allowing_missing_modules` directly, matching its actual intent (syntax
  acceptance, not module-resolution semantics). New regressions, each building a real project on
  disk at a path that collides with one of the three removed conditions and asserting E0208 is
  still reported: `gate2_valid.rs::test_missing_module_file_is_reported_even_when_path_contains_spec_fixtures`,
  `::test_missing_module_file_is_reported_even_when_path_contains_starklang`,
  `::test_missing_module_file_is_reported_even_when_entry_file_is_named_test_stark`. Plus two
  direct unit tests in `parser.rs` pinning down both the negative case (ordinary `parse` still
  reports the diagnostic for a bare in-memory `SourceFile` literally named `"test.stark"`) and
  the positive case (the explicit opt-in still suppresses it), independent of the fixture corpus:
  `parser::tests::ordinary_parse_reports_missing_module_even_for_bare_test_stark_name`,
  `::allowing_missing_modules_suppresses_the_diagnostic_when_explicitly_requested`.

## DEV-037 — Runtime field/index projection did not auto-dereference references (RESOLVED in WP-C2.2)

- **Normative expectation:** `03-Type-System.md`'s auto-dereference rules permit field and index
  projection through references (for example `r.field` where `r: &T`).
- **Original behaviour:** the type checker accepted these projections, but `expr_place` appended
  the projection directly to the place storing `Value::Ref`. Runtime projection then attempted
  to find a field/index on the reference wrapper and trapped with "use of moved or invalid
  field." This was discovered while testing nested returned-reference rebasing for DEV-035.
- **Resolution:** every field, tuple-field, and index projection normalizes its base through any
  reference chain before appending the projection. Regression:
  `interp::tests::field_access_through_reference_auto_derefs`.
- **Owning gate:** found and closed in WP-C2.2.

## DEV-038 — Operator and iterator protocols used unqualified method lookup (RESOLVED in WP-C2.2 correction pass)

- **Normative expectation:** `==` invokes `Eq::eq`, ordering operators invoke `Ord::cmp`, and
  `for` advances through `Iterator::next`; an unrelated inherent method with the same name
  cannot replace a named trait protocol.
- **Original behaviour:** all three runtime paths called `find_method(..., None)`. Because
  ordinary method lookup correctly prefers inherent methods, inherent `eq`, `cmp`, or `next`
  methods hijacked the language protocols.
- **Resolution:** protocol paths pass the corresponding `Res::CoreTrait` identity into method
  lookup; ordinary source calls retain inherent-first behavior. Regression:
  `interp::tests::language_protocols_ignore_same_named_inherent_methods`.
- **Owning gate:** found by post-WP-C2.2 external review and closed in the WP-C2.2 correction
  pass.

## DEV-039 — `for` bindings and unconsumed tails skipped observable destruction (RESOLVED in WP-C2.2 correction pass)

- **Normative expectation:** every owned loop binding is destroyed at its iteration boundary,
  including `continue`, `break`, `return`, and `?` exits; values remaining in a consumed
  iterable are also destroyed exactly once.
- **Original behaviour:** each iteration overwrote the same frame slot and cleaned it only
  after the loop, so only the final binding ran its STARK destructor. Breaking or escaping also
  let the host drop the unconsumed iterator tail without STARK destruction.
- **Resolution:** every body result is followed by explicit binding cleanup, remaining direct
  iterable values are destroyed on early exit, and promoted iterator owners are destroyed when
  the loop ends. Regression: `interp::tests::for_loop_drops_each_binding_and_unconsumed_tail`.
- **Owning gate:** found by post-WP-C2.2 external review and closed in the correction pass.

## DEV-040 — Collection discard paths bypassed STARK destructors (RESOLVED in WP-C2.2 correction pass)

- **Normative expectation:** container destruction and ownership-discarding operations run each
  contained value's language-level destructor exactly once.
- **Original behaviour:** `drop_value` had no `HashMap`/`HashSet` recursion; `Vec::clear`,
  map/set clear, removed stored keys/elements, and duplicate/replacement inputs were discarded
  by Rust rather than the STARK `Drop` protocol.
- **Resolution:** container destruction recursively drains owned entries, clear operations
  extract and destroy contents, and replacement/removal paths explicitly destroy consumed keys
  and elements outside active collection borrows. Regressions:
  `collection_discard_paths_run_stark_destructors` and
  `collection_replacement_and_removal_drop_consumed_keys`.
- **Owning gate:** found by post-WP-C2.2 external review and closed in the correction pass.

## DEV-041 — Returned range-slice references targeted a popped temporary (RESOLVED in WP-C2.2 correction pass)

- **Normative expectation:** `&self.values[a..b]` returned from a borrowing method remains a
  live view into the caller-owned receiver.
- **Original behaviour:** range indexing promoted a `Value::Slice` into a method-frame temporary,
  then `&` returned a reference to that temporary. Receiver rebasing correctly ignored the
  non-receiver local, and the caller trapped with "dangling reference."
- **Resolution:** taking a reference to a range-index place returns the slice view itself, whose
  base place is rebased through the receiver. Regression:
  `interp::tests::returned_range_and_vec_as_slice_are_borrowed_views`.
- **Owning gate:** found by post-WP-C2.2 external review and closed in the correction pass.

## DEV-042 — `Vec::as_slice` cloned elements instead of borrowing (RESOLVED in WP-C2.2 correction pass)

- **Normative expectation:** `Vec::as_slice(&self) -> &[T]` returns a view of the original
  vector and does not clone or acquire ownership of its elements.
- **Original behaviour:** the interpreter returned `Value::Array(vector.clone())`, causing
  observable double destruction for `Drop` elements.
- **Resolution:** `as_slice` returns a `Value::Slice` over the already-resolved vector place.
  Regression: `interp::tests::returned_range_and_vec_as_slice_are_borrowed_views`.
- **Owning gate:** found by post-WP-C2.2 external review and closed in the correction pass.

## DEV-043 — Hash collections used structural host equality for keys (RESOLVED in WP-C2.2 correction pass)

- **Normative expectation:** `HashMap<K, V>` and `HashSet<T>` key identity follows the
  language-level `Eq` implementation required by their public bounds.
- **Original behaviour:** insertion-ordered collection lookup used derived Rust equality over
  `Value`, so custom `Eq::eq` implementations were ignored by insert/get/contains/remove and
  duplicate detection.
- **Resolution:** collection lookup invokes trait-qualified `Eq::eq`; map references use a
  stable insertion index rather than repeating structural lookup. Insert, lookup, removal,
  extend, and collect paths share the language relation. Regression:
  `interp::tests::hash_collections_use_language_eq_for_keys`.
- **Owning gate:** found by post-WP-C2.2 external review and closed in the correction pass.

## DEV-044 — Comparison operators moved non-`Copy` operands instead of borrowing them (RESOLVED post-WP-C2.11)

- **Normative expectation:** `03-Type-System.md` "Operators and Traits" desugars `==`/`!=` to
  `Eq::eq(&self, other: &Self)` and `<`/`<=`/`>`/`>=` to `Ord::cmp(&self, other: &Self)` — both
  operands are borrowed, not consumed.
- **Original behaviour:** `eval_path`'s `Res::Local` arm unconditionally calls `take_place` when
  evaluating a bare local as an expression, so `a == b` for two non-`Copy` values (e.g.
  `String`) moved both operands out of storage before `eval_binary`'s `Eq`/`Ord` dispatch ever
  ran. Using either operand afterward failed at runtime with "use of unavailable value" despite
  the comparison never taking ownership.
- **Resolution:** a new `expect_value_borrowed` evaluates comparison operands: place
  expressions (locals, fields, tuple fields, indices, deref targets) are cloned via
  `clone_place_value` instead of moved; non-place expressions (call results, literals) are
  unaffected, since they have no other owner. Regressions:
  `interp::tests::comparison_operands_remain_usable_afterward`,
  `::generic_eq_and_ord_bounds_do_not_move_their_operands`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work,
  independently reproduced, and closed in this correction pass.

## DEV-045 — `?` inside an aggregate initializer did not stop evaluation of later elements (RESOLVED post-WP-C2.11)

- **Normative expectation:** the abstract machine (`CORE-V1-ABSTRACT-MACHINE.md`) requires
  aggregate initializers to evaluate left to right and stop immediately on early transfer,
  destroying already-completed elements in reverse completion order.
- **Original behaviour:** `expect_value` swallowed `Flow::Propagate` into
  `self.pending_propagation` and returned a dummy `Value::Unit`, so the `.map(expect_value)
  .collect()` pattern used to build tuple/array literals and positional enum-variant
  constructors (`Pair::Two(a, b)`) kept evaluating later elements for their side effects even
  after an earlier element had already propagated a `Result`/`Option` early return via `?`.
  Already-completed elements were also never explicitly destroyed on this path.
- **Resolution:** a new `eval_aggregate_elements` helper evaluates elements left to right,
  checks `pending_propagation` after each one, and on early transfer destroys completed
  elements in reverse order via `drop_value` before returning the propagated value — applied to
  tuple literals, array literals, and positional enum-variant construction (`eval_call`'s
  `Res::Variant` arm). Named struct/enum-struct-variant field construction
  (`eval_struct_lit`) received the same stop-and-clean-up-in-reverse treatment inline, since its
  `BTreeMap`-based field accumulation doesn't fit the same shared-Vec helper. A genuine
  Rust-level trap (`RuntimeError`) is unaffected — it still unwinds immediately via `?` with no
  cleanup, matching existing trap-abort semantics. Regressions:
  `interp::tests::early_transfer_inside_a_tuple_stops_later_elements_from_running`,
  `::early_transfer_inside_an_enum_variant_stops_later_elements_from_running`,
  `::early_transfer_inside_a_tuple_drops_completed_elements_in_reverse_order`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work,
  independently reproduced (confirmed via a side-effecting element genuinely running after an
  earlier `?`), and closed in this correction pass. Struct-literal field construction and
  positional enum-variant construction were not in the review's own repro but share the exact
  same root cause and were fixed in the same pass rather than left as a known adjacent gap.

## DEV-046 — Float-to-integer casts rejected any nonzero fractional part instead of truncating (RESOLVED post-WP-C2.11)

- **Normative expectation:** a finite float-to-integer cast truncates toward zero, then traps
  only when the truncated result is unrepresentable in the target width.
- **Original behaviour:** `eval_cast`'s float-to-integer arm rejected any value with
  `value.fract() != 0.0` outright, so `3.9f64 as Int32` trapped instead of producing `3`.
- **Resolution:** the fractional-part check was replaced with `.trunc()` followed by the
  existing `check_integer_range` call against the target width; NaN and infinities still trap
  (`!value.is_finite()`), unchanged. Regressions:
  `interp::tests::float_to_int_cast_truncates_toward_zero_instead_of_trapping_on_fractions`,
  `::float_to_int_cast_still_traps_on_nan_and_infinity`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work,
  independently reproduced, and closed in this correction pass.

## DEV-047 — Signed `MIN % -1` did not trap (RESOLVED post-WP-C2.11; `MIN / -1` was never broken)

- **Normative expectation:** both `MIN / -1` and `MIN % -1` trap for a signed integer type,
  since the mathematical quotient is not representable at the CPU instruction level.
- **Original behaviour:** all integer arithmetic is carried in a wider `i128`, so
  `checked_div`/`checked_rem` never overflow at that width; the post-hoc `check_integer_range`
  call catches `MIN / -1` (its i128 result doesn't fit the declared width) but not `MIN % -1`
  (its mathematical result, `0`, always fits). An external review's initial claim that *both*
  operators were broken was independently checked before fixing: `MIN / -1` was confirmed
  already trapping correctly and needed no change; only `Rem` had the gap.
- **Resolution:** an explicit guard traps `Rem` when `right == -1` and `left` equals the
  declared signed type's minimum value (new `signed_integer_min` helper), scoped to `Rem` only.
  Regressions: `interp::tests::signed_min_rem_negative_one_traps`,
  `::signed_min_div_negative_one_still_traps` (non-regression guard for the already-correct
  `Div` case), `::rem_and_div_by_values_other_than_negative_one_are_unaffected`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work; the
  review's claim was independently verified per-operator (not trusted as stated) before
  scoping the fix, and closed in this correction pass.

## DEV-048 — `Drop::drop(&mut self)` mutated a clone instead of the destructor's real storage (RESOLVED post-WP-C2.11)

- **Normative expectation:** `Drop::drop(&mut self)` may mutate or replace fields (e.g. via
  `replace(&mut self.field, ..)`), and those mutations determine what the surrounding automatic
  field destruction subsequently sees and destroys.
- **Original behaviour:** `drop_value` bound a *clone* of the value as the destructor's `self`
  local. Any mutation inside `drop()` only affected the throwaway clone; the frame was then
  discarded and the function proceeded to recursively destroy the pristine, never-mutated
  original. A destructor that used `replace()` to swap in a new field value and explicitly drop
  the old one caused double destruction of the pre-destructor field state and silently skipped
  destruction of the replacement value entirely.
- **Resolution:** `drop_value` now moves the real value into the destructor's `self` binding
  (mirroring the existing `RefMut`-receiver move/write-back convention already used by ordinary
  method calls in `call_user_method`), reads back whatever `self` holds after the destructor
  body runs, and uses that (possibly mutated) value for the subsequent recursive field
  destruction instead of the stale pre-destructor snapshot. Regressions:
  `interp::tests::drop_mutation_through_mut_self_affects_real_storage`,
  `::drop_without_self_mutation_still_runs_exactly_once`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work,
  independently reproduced (confirmed the pre-destructor field value printed twice and the
  replacement value never printed at all), and closed in this correction pass.

## DEV-049 — `Float32` display used `Float64`'s shortest-round-trip digits (RESOLVED post-WP-C2.11)

- **Normative expectation:** canonical display uses the shortest decimal representation that
  round-trips to the *declared* IEEE type.
- **Original behaviour:** `Value::Float` stores every float as `f64` (Float32 results are
  rounded to `f32` precision by `normalize_numeric` but kept in the same `f64`-carrying
  representation), and `canonical_float`/`.fmt()` always formatted via `f64::to_string()`'s
  shortest-round-trip algorithm regardless of the checked static type, so `println(0.1f32)`
  printed `0.10000000149011612` instead of `0.1`.
- **Resolution:** `canonical_float`'s digit-formatting body was extracted into a shared
  `canonical_float_digits` helper reused by a new `canonical_float32`. `println`/`print`/`panic`
  (via a new `arg_exprs` parameter threaded into `call_builtin`) and `.fmt()` (which already had
  the checked receiver type in scope) now format through `canonical_float32` when the static
  type is `Float32`. **Known residual gap, not fixed here:** a `Float32` value formatted only
  through the generic `Display for Value` impl with no static-type context available (e.g.
  nested inside a struct/collection printed as a whole) still falls back to `canonical_float`'s
  `f64` digits — fixing that would need a type marker on `Value::Float` itself, a larger change
  touching roughly 40 call sites, out of scope for this correction pass. Regressions:
  `interp::tests::float32_println_and_fmt_use_float32_round_trip_digits_not_float64`,
  `::float64_println_and_fmt_are_unaffected_by_the_float32_fix`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work,
  independently reproduced, and closed (for the two directly-printed cases) in this correction
  pass. The nested-value residual gap was closed by DEV-058 in the following correction-brief
  session.

## DEV-050 — Negative `sqrt` trapped instead of returning NaN (RESOLVED post-WP-C2.11)

- **Normative expectation:** the standard-library math contract classifies transcendental
  domain errors (e.g. `sqrt` of a negative number) as producing NaN, not a language trap —
  distinct from the numeric-trap rules governing integer overflow/division and float-to-int
  casts.
- **Original behaviour:** `Builtin::Sqrt` returned a `RuntimeError` ("sqrt domain error") for
  any negative finite input.
- **Resolution:** the domain check was removed; `f64::sqrt` already returns NaN for negative
  finite inputs. Regressions: `interp::tests::negative_sqrt_returns_nan_instead_of_trapping`,
  `::nonnegative_sqrt_is_unaffected`.
- **Owning gate:** found by external review of the committed WP-C2.11 alignment work,
  independently reproduced, and closed in this correction pass. A companion claim in the same
  review — that `main` entrypoint selection incorrectly counts type-namespace items (e.g. a
  `struct main` coexisting with `fn main`) — was independently tested and **refuted**: the
  compiler already rejects this at name resolution (`E0204` duplicate definition) before
  entrypoint selection runs; no corresponding deviation was opened.

## DEV-051 — Trait default methods cannot call another trait method on `self` (RESOLVED)

- **Normative expectation:** a trait default method body may call other methods of the same
  trait through `self`, exactly as an ordinary method body can call sibling methods.
- **Original behaviour:** confirmed with a minimal repro (a `Greet` trait with a required
  `fn name(&self) -> String` and a default `fn greeting(&self) -> String { self.name() }`,
  implemented for a struct that only overrides `name`): calling `self.name()` from inside the
  default `greeting()` body failed to type-check with `E0302 method 'name' not found for type
  '&Self'`. Ordinary (non-default) method-to-method calls on `self` were unaffected; this was
  specific to a default method body calling a sibling trait method.
- **Root cause:** `resolve_method` already had a mechanism for a receiver with no concrete
  `impl` to match against a bounded *generic function* type parameter (`fn f<T: Greet>(x: T) {
  x.greet() }`, whose receiver type is `Ty::Param("T")`) — but never for `self` inside a
  trait's own default-method body, whose receiver type is `Ty::Param("Self")` (set alongside a
  new `current_trait_id` field while checking `hir::ItemKind::Trait`'s default bodies, which are
  type-checked once, generically, at the trait declaration site rather than once per
  implementor). The generic-parameter mechanism also only ever checked the receiver's type
  *before* the reference-deref loop, which is correct for a by-value generic parameter but wrong
  for `self` (always received by reference) — an early attempt at this fix placed the "Self"
  check in the same spot and it still failed, since `resolved_base` at that point is still `&Self`
  (a `Ty::Ref`), not the bare `Ty::Param("Self")` the check was testing for.
- **Resolution:** extracted the existing per-trait-method-signature lookup/arg-check logic (
  previously inlined only for the generic-parameter case) into two small shared helpers,
  `find_trait_method_sig`/`check_trait_member_call`, and added a second call site for the
  `Self`-receiver case, positioned *after* the reference-deref loop (unlike the generic-parameter
  case, positioned before it) since `self` is always received by reference. Regressions:
  `typecheck::tests::trait_default_method_calling_sibling_trait_method_through_self_type_checks`,
  `::trait_default_method_calling_another_default_method_type_checks` (a default method calling
  *another* un-overridden default, not just a required method),
  `::trait_default_method_wrong_arg_count_to_sibling_trait_method_still_errors` (confirms the
  fix doesn't silently swallow a genuine arity mismatch),
  `interp::tests::trait_default_method_calling_sibling_trait_method_through_self_executes` (the
  decisive end-to-end regression: type-checks *and* executes to the expected output, not just a
  diagnostic-count check).
- **User impact:** a common, idiomatic trait-default pattern (a default method built out of
  calls to other trait methods) did not work at all.
- **Security/soundness impact:** none identified — a rejection of legal code (availability), not
  an acceptance of illegal code.
- **Owning gate:** found while building the WP-C2.12 differential corpus (`starkc/tests/
  exec_snapshots/struct_enum_trait__04_trait_default_and_override.stark` was redesigned to avoid
  this construct rather than fixed). Reproduced against the current head before fixing; closed
  this session. See DEV-060 for a separate, narrower defect found while writing this fix's
  regression tests.

## DEV-052 — `Trait::method(...)` qualified-call syntax fails to resolve for compiler CoreTraits (RESOLVED)

- **Normative expectation:** `03-Type-System.md`:670 documents `Trait::method(receiver, ...)` as
  normal fully-qualified call syntax, with no carve-out for compiler-recognized traits
  (`Eq`, `Ord`, `Hash`, `Display`, `Clone`, etc. — `hir::CoreTrait`).
  `resolve.rs`/`eval_call`'s `Res::TraitMember(trait_id, member)` path (confirmed present and
  working via `call_qualified_trait`) demonstrates the mechanism is implemented in general.
- **Original behaviour:** confirmed with a minimal repro: `Describe::describe(&m)` for a
  user-defined `trait Describe` resolved and executed correctly, but `Eq::eq(&a, &b)` for a
  struct with a real `impl Eq for P { fn eq(&self, other: &P) -> Bool { ... } }` failed at
  resolve time with `E0200 undefined variable 'Eq::eq'`. The same qualified-call syntax worked
  for a user-declared trait and failed for a compiler CoreTrait name.
- **Root cause:** confirmed exactly the suspicion recorded when this was first found:
  `resolve_path_relative`'s multi-segment loop only ever continued past a first segment
  resolving to `Res::Item(item_id)` — a real trait *declaration* item, whose member is looked up
  by matching the segment text against `ItemDefDetail::Trait { items }`. A `CoreTrait` has no
  such declaration item at all (it's resolved directly to `Res::CoreTrait(core_trait)` by name,
  with nothing to index a member against), so the loop's final `else` branch returned `Res::Err`
  for any second segment following a `CoreTrait` name, regardless of what it named.
- **Resolution:** added `Res::CoreTraitMember(CoreTrait, Span)` (the method-name segment's span,
  matching the existing `SelfAssoc`/`ParamAssoc` idiom of carrying a `Span` rather than an owned
  `String`, since `Res` derives `Copy`), resolved when the second segment names that trait's one
  fixed callable method (a new `core_trait_method_name` table — `Eq`→"eq", `Ord`→"cmp",
  `Hash`→"hash", `Clone`→"clone", `Display`→"fmt", `Default`→"default"; other `CoreTrait`s have no
  single directly-callable method and stay unresolved, same as an unknown member of a real
  trait). Typecheck's handling (`check_qualified_core_trait_call`) finds the *matching impl's
  own* method signature directly (a `CoreTrait` has no shared declaration signature the way a
  user trait does — each `impl <CoreTrait> for T` writes its method signature itself), matching
  the impl by its trait-ref's source text against a small `core_trait_source_name` table (the
  same approach `ty_satisfies_operator_bound` already uses for these traits). The interpreter's
  dispatch (`call_qualified_core_trait`) is far simpler: it reuses the *exact same*
  `find_method(nominal, method_name, Some(Res::CoreTrait(core_trait)))` lookup the `==`/`<`
  operator sugar already calls for these traits — a qualified call is just an explicit spelling
  of the same dispatch, not a separate mechanism, so no new impl-scanning logic was needed on
  the interpreter side at all. Regressions:
  `interp::tests::qualified_call_to_core_trait_eq_method_resolves_and_executes`,
  `::qualified_call_to_core_trait_ord_method_resolves_and_executes` (confirms the fix isn't
  accidentally specific to `Eq`), `::qualified_call_to_user_declared_trait_is_unaffected_by_the_
  core_trait_fix` (the pre-existing user-trait path is a separate mechanism, untouched),
  `typecheck::tests::qualified_call_to_unimplemented_core_trait_is_rejected` (confirms the fix
  doesn't accidentally accept a genuinely invalid program just because the syntax now resolves).
- **User impact:** narrow — fully-qualified calls to `Eq`/`Ord`/etc. (needed only to disambiguate
  when a type also has an inherent method of the same name) did not work; the operator sugar
  (`==`, `<`, etc.) was unaffected and remains how these traits are normally invoked.
- **Security/soundness impact:** none identified — a rejection of legal code, not an acceptance
  of illegal code.
- **Owning gate:** found while building the WP-C2.12 differential corpus (the "trait-qualified
  calls" metamorphic pair was redesigned around a user-defined trait instead of `Eq`, rather than
  fixed at the time). Reproduced against the current head before fixing; closed this session.

## DEV-053 — A bare `None` (or any other builtin-resolved) pattern never matched by value; it silently acted as an unconditional wildcard (RESOLVED, root cause corrected from the original finding)

- **Normative expectation:** `02-Syntax-Grammar.md`'s `SYN-PATTERN-001` note: "a single
  `IDENTIFIER` pattern that resolves to a unit enum variant or a constant in scope matches by
  value; otherwise it introduces a new binding." `None` (`Res::Builtin(Builtin::None)`) must
  match only `Option::None`.
- **Original finding (superseded by the investigation below):** this entry originally described
  two "tuple-pattern usefulness/exhaustiveness" false positives -- a spurious `W0006`
  "unreachable arm" for `match (opt, n) { (None, x) => x, (Some(a), _) => a }`, and a spurious
  `E0303` "non-exhaustive" for a fully-covered three-variant-enum-times-wildcarded-Int32 tuple
  match -- and flagged the usefulness *algorithm* as the suspected root cause.
- **Actual root cause, found on investigation: not an exhaustiveness-algorithm bug at all.**
  `resolve.rs`'s `lower_pattern` (`ast::PatKind::Binding` arm) disambiguates every bare
  identifier pattern by checking only `self.modules[current_module].items` for
  `Res::Variant`/`Res::Item` -- it never checked `Res::Builtin`, which is how `None` is
  classified (`resolve_builtin("None") == Some(Builtin::None)`, a lookup only ever called from
  *expression*-position resolution, never from pattern lowering). Every bare `None` pattern
  therefore fell through to "fresh local binding" unconditionally, in every position, not just
  when nested in a tuple.
- **Actual severity: silently WRONG program output, confirmed empirically -- not merely a
  spurious diagnostic.** `fn main() { let value: Option<Int32> = Some(5); let r = match value {
  None => 999, Some(a) => a }; println(r); }` printed **`999`**: the `None` arm silently matched
  `Some(5)` with no diagnostic of any kind, because it was never actually checking the variant.
  This reproduces with a completely flat, non-nested, non-tuple scrutinee -- the original
  finding's framing around "tuple-pattern coverage" was itself an artifact of the same root
  cause (a tuple pattern containing a misclassified `None` was wrongly judged *irrefutable* by
  `is_irrefutable`, which treats `Wild`/`Binding` components as always-matching, letting it
  bypass the exhaustiveness check entirely -- which is also why the original "spurious
  unreachable arm" symptom occurred: `(None, x)` really was behaving like `(_, x)`, so the
  redundancy warning against `(Some(a), _)` was internally consistent with the compiler's wrong
  interpretation, not a bug in the redundancy check itself).
- **The "spurious non-exhaustive" half of the original finding is not a bug and is not part of
  this entry.** Re-reading `typecheck.rs`'s exhaustiveness check (`check_expr`'s `Match` arm)
  confirms it is a deliberate, self-documented, sound-by-construction design choice: any
  scrutinee type outside a small set of exactly-enumerable domains (bool/enum/Option/Result)
  requires at least one *individually* irrefutable arm rather than attempting real cross-arm
  tuple-component usefulness tracking, exactly as its own code comment states ("sound -- never
  accepts a genuinely non-exhaustive match... matches this codebase's existing 'reject some safe
  programs is intentional' philosophy"). This is a known, accepted precision limitation, not a
  correctness defect, and needed no fix.
- **Checked for the dangerous direction and did not find it:** before fixing, `match (color, n) {
  (Color::Red, x) => x, (Color::Green, _) => 2 }` (a `Color`/`Int32` tuple genuinely missing the
  `Blue` case) was confirmed still correctly rejected as non-exhaustive, both before and after
  the fix -- the confirmed defect was over-permissive matching of one misclassified pattern
  value, not an under-strict exhaustiveness gap.
- **Resolution:** `lower_pattern`'s `Binding` arm now also checks `resolve_builtin(name)` (gated
  by the tensor extension exactly as `resolve_unqualified` already gates ordinary bare-identifier
  builtin resolution, per DEV-004) before falling back to "fresh binding," producing a real
  `PatKind::Path { res: Res::Builtin(builtin), .. }` value pattern. `typecheck.rs`'s
  `check_pat` gained a matching `Res::Builtin(Builtin::None) => self.resolve(&expected)` arm
  (mirroring the existing `Res::Builtin(Builtin::Some | ..)` handling already present for the
  `TupleVariant` case). Regressions: `resolve::tests::
  repeated_none_in_one_tuple_pattern_does_not_collide_as_duplicate_bindings`;
  `interp::tests::bare_none_pattern_matches_by_value_not_as_a_wildcard`,
  `::nested_none_pattern_in_a_tuple_matches_by_value_not_as_a_wildcard`,
  `::repeated_none_within_one_tuple_pattern_no_longer_collides`,
  `::ordinary_binding_and_payload_patterns_are_unaffected_by_the_none_fix`.
- **Owning gate:** found while building the WP-C2.12 differential corpus; investigated and
  closed as a dedicated follow-up in the same session. See DEV-055 for a related, narrower
  finding surfaced during this investigation (a separate root cause, closed in a later session).

## DEV-054 — A tuple pattern with the same by-value identifier repeated across components was rejected as a duplicate binding (RESOLVED, same root cause and fix as DEV-053)

- **Normative expectation:** `02-Syntax-Grammar.md`'s `SYN-PATTERN-001` note states "a single
  `IDENTIFIER` pattern that resolves to a unit enum variant or a constant in scope matches by
  value; otherwise it introduces a new binding" — a by-value identifier pattern does not
  introduce any binding at all, so it cannot collide with another occurrence of itself.
- **Original behaviour:** `match pair { (None, None) => 0, _ => 1 }` for `pair:
  (Option<Int32>, Option<Int32>)` failed to resolve with `E0204 duplicate definition of variable
  'None' in the same scope` — both `None`s were independently misclassified as introducing a
  fresh local named "None" (DEV-053's exact root cause), so the second collided with the first.
- **Resolution:** identical fix to DEV-053 (`lower_pattern` now recognizes `None` as a
  `Res::Builtin` value pattern, which introduces no binding at all, so there is nothing left to
  collide). Regression: `interp::tests::repeated_none_within_one_tuple_pattern_no_longer_collides`
  (`resolve::tests::repeated_none_in_one_tuple_pattern_does_not_collide_as_duplicate_bindings`
  covers the same case at the resolve stage).
- **Owning gate:** found while building the WP-C2.12 differential corpus; closed by the same fix
  as DEV-053, same session.

## DEV-055 — A bare, glob-imported unit enum variant name does not resolve at all (as an expression or a pattern) (RESOLVED)

- **Normative expectation:** `use Color::*;` should make `Color`'s variants usable unqualified,
  as both values and patterns, the same as any other glob-imported name.
- **Original behaviour:** confirmed with a minimal repro: after `enum Color { Red, Green, Blue }
  use Color::*;`, a bare `Red` used as an *expression* (`let c: Color = Red;`) failed with `E0200
  undefined variable 'Red'`. Used bare in *pattern* position (`match c { Red => 1, Green => 2,
  Blue => 3 }`, with `c` constructed via the qualified `Color::Blue`), all three arms were
  accepted syntactically but exhibited DEV-053's exact wildcard-collapse symptom: the first arm
  matched unconditionally and the other two were flagged unreachable, printing `1` regardless of
  `c`'s real value. **Not fixed by the DEV-053 fix** — confirmed still present afterward — root
  cause was different from DEV-053's.
- **Root cause:** `resolve_use_tree`'s `Glob`/`Group` arms only ever consulted `submodule_map` (a
  map from real-module items to their `ModuleId`) to find the set of names to copy into scope.
  An enum item is never a key in `submodule_map` — its variants are resolved dynamically through
  `item_details`'s `ItemDefDetail::Enum` arm at path-resolution time (see
  `resolve_path_relative`), never pre-populated into a module's `items` map the way a real
  submodule's contents are. So `use Color::*;`/`use Color::{Red, Blue};` silently expanded to
  *nothing* when the prefix was an enum, rather than erroring or working — this is why the
  qualified forms (`Color::Red`, a direct non-glob `use Color::Red;`) were unaffected: both go
  through `resolve_path_relative`'s per-segment `item_details` lookup directly, never through the
  glob/group expansion machinery.
- **Resolution:** added `enum_variant_items(item_id)`, which returns each variant's name paired
  with its `Res::Variant`, if `item_id` names an enum. Wired into both `resolve_use_tree` and
  `resolve_use_tree_relative`'s `Glob` arms (as an `else if` fallback after the existing
  `submodule_map` check) and both functions' `Group` arms (via a new
  `resolve_enum_variant_group_item` helper, since a group's items must be matched individually
  against the enum's variant list rather than bulk-copied). Regressions:
  `resolve.rs::glob_imported_enum_variant_resolves_as_bare_expression`,
  `::group_imported_enum_variants_resolve_selectively` (also confirms a variant deliberately
  left out of a group import correctly stays undefined, ruling out an overly-broad fix that
  imports every variant regardless of what the group actually names),
  `interp.rs::glob_imported_enum_variant_resolves_and_executes_as_bare_expression`,
  `::glob_imported_enum_variant_discriminates_in_pattern_position_not_wildcard_collapsed` (the
  decisive end-to-end regression: `match Color::Blue { Red => 1, Green => 2, Blue => 3 }` now
  prints `3`, not the wildcard-collapsed `1`), `::group_imported_enum_variants_discriminate_in_
  pattern_position`.
- **User impact:** a glob-imported unit enum variant could not be referred to unqualified at all,
  either as a value or in a pattern — silently wrong pattern-match results if attempted, or a
  compile error for the expression form.
- **Security/soundness impact:** the pattern-position half was the same class of silent-wrong-
  output defect as DEV-053 (a pattern that should discriminate on variant identity instead
  matched unconditionally) — real, but narrower in practice, since it required a glob or group
  import specifically (`use Color::Red;`/`Color::Blue` qualified forms were unaffected).
- **Owning gate:** found while investigating DEV-053 (confirmed as a deliberate differential
  test — bare glob-imported names were used as a control case to scope DEV-053's fix precisely,
  and turned out to have their own, separate defect). Reproduced against the current head before
  fixing; closed this session.

## DEV-056 — `?` propagation was swallowed outside aggregate-construction call sites (RESOLVED)

- **Normative expectation:** early transfer via `?` must stop evaluation of every later
  sub-expression in the same construct, unconditionally — not just inside tuple/array/struct/
  enum-variant literals (DEV-045's original scope).
- **Original behaviour:** `expect_value` converts `Flow::Propagate` into `pending_propagation` +
  a dummy `Value::Unit`. Only aggregate-construction call sites checked the flag before
  continuing (DEV-045). Every other sequential-evaluation context kept going: ordinary/
  associated/builtin function calls, method calls (both user-method and core/builtin-type
  dispatch), binary operands, `&&`/`||` right operands, assignment right-hand sides, ranges,
  repeat expressions, `if`/`while` conditions, match scrutinees, and `break` values (`return`
  alone already checked the flag). Confirmed empirically to run real, visible side effects that
  should never have executed — not merely a spurious diagnostic:
  `sink(fail()?, side_effect())` printed `SIDE EFFECT`/`CALLED` before finally reaching `done`.
- **Resolution:** a new `eval_call_arguments` helper (mirroring `eval_aggregate_elements`'s
  stop-and-clean-up-in-reverse contract, but returning plain owned values instead of
  aggregate-storage `Option<Value>` slots) is now used by every call-argument-evaluating site.
  `call_core_method` (a large dispatcher with a single caller, `call_method`) uses the
  "function-boundary adapter" exception: it re-arms `pending_propagation` and returns a dummy
  value on propagation, and its one caller checks the flag immediately, exactly mirroring
  `expect_value`'s own existing convention but with a caller guaranteed to check it. Binary
  operands, `&&`/`||`, assignment, ranges, repeat, `if`/`while` conditions, match scrutinees, and
  `break` each gained an explicit `pending_propagation` check between their sequential
  sub-evaluations. `expect_bool`/`expect_int` were changed to pass a placeholder through
  (instead of reporting a misleading "expected Bool"/"expected integer" trap) when propagation is
  pending, on the condition that every one of their call sites checks `pending_propagation`
  immediately afterward — verified true for all eight call sites as of this fix, including one
  (`expr_place`'s computed-index case) that cannot itself return `Flow::Propagate` (a documented,
  narrower residual: it fails loudly with a dedicated error instead of truly propagating, which
  is a real, separate architectural gap in `expr_place`'s non-`Flow`-aware return type, left open
  rather than attempted as part of this fix).
  Regressions (all in `interp.rs`): `try_in_call_argument_stops_later_arguments_and_callee`,
  `try_in_method_argument_stops_later_arguments_and_method_body`,
  `try_in_binary_operand_stops_rhs_evaluation`,
  `try_in_and_or_right_operand_propagates_not_converted_to_bool`,
  `try_in_range_low_bound_stops_high_bound_evaluation`,
  `try_in_repeat_value_stops_array_construction`,
  `try_in_break_value_propagates_out_of_the_enclosing_function`,
  `try_drops_completed_call_argument_temporaries_in_reverse_order`,
  `try_in_return_expression_still_propagates_without_dummy_unit`.
- **Owning gate:** found in an external correction brief following WP-C2.12, independently
  reproduced against the current head before fixing, closed the same session.

## DEV-057 — Eq/Ord comparison dispatch passed owned clones instead of true borrowed places (RESOLVED)

- **Normative expectation:** `==`/`!=`/`<`/`<=`/`>`/`>=` desugar to `Eq::eq(&self, &other)`/
  `Ord::cmp(&self, &other)` (`03-Type-System.md` "Operators and Traits") — both operands are
  borrowed, never owned by the comparison.
- **Original behaviour:** `eval_binary`'s nominal Eq/Ord dispatch promoted a *clone* of the left
  operand into a temporary place and passed a *clone* of the right operand as an ordinary owned
  method argument. This was wrong in two independent, differently-manifesting ways: (1) the
  receiver's clone silently vanished via ordinary Rust-level drop when `call_user_method`
  extracted it before frame cleanup (correct for a real reference, since dropping a reference
  does nothing — but here the "reference" was actually an owned clone, so its data and any
  `Drop::drop` call were lost entirely, with no STARK-level destructor firing at all); (2) the
  argument's clone was bound as an ordinary owned parameter local, so the callee's own normal
  per-parameter cleanup gave it a *real*, extra `Drop::drop` call, before the comparison's own
  caller-visible side effects had even finished. Confirmed empirically:
  `println(a == b); println("after");` for a `Drop`-bearing `Key` printed `b`'s destructor
  *before* `"after"`, then printed both `a` and `b` again at their real, correct scope-end —
  i.e. `b` was destroyed twice and out of order.
- **Resolution:** a new `resolve_comparison_operand` helper resolves each comparison operand to
  both a value (for the non-dispatching structural-equality fallback, which never needs place
  identity) and, for a place expression, the *real* `Place`. `eval_binary`'s signature now
  threads `(Value, Option<Place>)` per side; the nominal dispatch path passes `Value::Ref(place)`
  for both operands — a genuine borrow of the original storage — instead of a clone, for both
  the receiver and the argument. A non-place operand (a call result with no other owner) still
  needs a fresh temporary to point the reference at; found and fixed a *second*, broader,
  pre-existing bug while implementing this: `promote_to_temp_place` (used at 15+ call sites
  throughout the interpreter — comparison temporaries, for-loop iterator storage, string/Vec/
  HashMap iteration `Value::Ref` wrapping, range-slice views) bypasses `Frame::insert` entirely
  (a raw `.values.insert(...)` call), so its temporary is never recorded in `Frame::order` and is
  silently discarded via ordinary Rust-level deallocation when the frame is popped, with *no*
  `Drop::drop` call ever firing for it — confirmed empirically (a `Drop`-bearing temporary
  comparison operand never printed its destructor at all, not even at program end). Rather than
  changing the shared, widely-used `promote_to_temp_place` (which several existing call sites
  rely on *not* double-owning a value that is also separately owned elsewhere, e.g. iterator
  snapshots — confirmed by a regression when this was tried broadly first), added a new, narrowly
  -scoped `promote_to_owned_temp_place` that does register through `Frame::insert`, used only at
  the two new non-place-operand fallback sites this fix introduces.
  Regressions (all in `interp.rs`): `eq_on_drop_type_does_not_create_or_drop_clones`,
  `ord_on_drop_type_does_not_create_or_drop_clones`,
  `comparison_of_field_and_index_places_borrows_original_storage`,
  `comparison_of_temporary_operands_evaluates_each_once_and_drops_after_call`,
  `shared_receiver_method_observes_original_place_without_owned_clone_cleanup`.
- **Owning gate:** found in the same external correction brief as DEV-056, independently
  reproduced against the current head before fixing, closed the same session.

## DEV-058 — Float32 nested inside a tuple/array/Option/Result/struct still formatted via Float64 digits (RESOLVED)

- **Normative expectation:** canonical display uses the shortest decimal representation that
  round-trips to the *declared* IEEE type, for a `Float32` value in *any* position, not only when
  it is the immediate operand of `println`/`.fmt()`.
- **Original behaviour:** this is exactly the residual gap DEV-049 left open. `Value::Float`
  stored every float as a bare `f64` with no width marker; `println`/`.fmt()` were special-cased
  (via an external static-type lookup at the call site) to detect a checked-Float32 operand and
  format it through `canonical_float32`, but the *generic*, recursive `Display for Value` impl —
  reached whenever a Float32 is nested inside a printed tuple, array, `Option`, `Result`, or
  struct, all of which format their contents through `ToString`/`Display` with no static-type
  context available at that point — always fell back to `canonical_float`'s `f64` digits.
  Confirmed empirically: `println((0.1f32, 7))` printed `(0.10000000149011612, 7)` instead of
  `(0.1, 7)`.
- **Resolution:** added a `FloatWidth { F32, F64 }` tag carried directly on `Value::Float(f64,
  FloatWidth)`, so the runtime value itself knows its declared width independent of any
  external type-table lookup. `Display for Value`'s `Float` arm now matches on the tag directly
  and picks `canonical_float32`/`canonical_float` accordingly — fixing the nested-formatting gap
  for free, since `write_sequence`/`display_slot`/`Option`/`Result`'s `Display` arms all route
  through this same recursive impl. This let two now-redundant external-type-table special cases
  be deleted entirely: `.fmt()`'s `receiver_ty`-based Float32 check in `call_core_method`, and
  `format_runtime_value`'s `ty: Option<&Ty>` parameter (which also let `call_builtin`'s `arg_ty`
  computation and the `arg_exprs` parameter it depended on be removed, since nothing else in
  `call_builtin` used them). Every other `Value::Float` construction site across the interpreter
  (arithmetic, casts, unary negation, literal evaluation, the `Float64`-only math builtins,
  `MathPi`/`MathE`, `Random::next_float`, `default_value_for`) was updated to tag the correct
  width: arithmetic/casts/negation route through the existing `normalize_numeric` helper (which
  already looked up the expression's static type to decide `f32`-rounding, extended here to also
  set the tag from that same lookup); literal evaluation reads the width straight off the
  literal's own suffix (`0.1f32` vs. unsuffixed, which the checker already defaults to
  `Float64`); the transcendental math builtins are `Float64 -> Float64` only by signature (per
  `typecheck.rs`) and always tag `F64`; `math::abs` is generically `T -> T` and preserves the
  input's own tag. Regressions (all in `interp.rs`):
  `float32_nested_in_tuple_uses_float32_round_trip_digits`,
  `float32_nested_in_array_uses_float32_round_trip_digits`,
  `float32_nested_in_option_and_result_use_float32_round_trip_digits`,
  `float32_nested_in_struct_uses_float32_round_trip_digits`,
  `float32_arithmetic_result_nested_in_tuple_uses_float32_round_trip_digits`,
  `float32_cast_to_float64_uses_float64_round_trip_digits_not_float32` (the last proves the
  formatting difference tracks the value's declared width, not an unconditional `f32`-rounding).
  All pre-existing DEV-049 regressions continue to pass unchanged.
- **Owning gate:** correction-brief Issue 3 (post-WP-C2.12), reproduced against the current head
  before fixing, closed the same session.

## DEV-059 — NaN-producing float operations did not canonicalize to the spec's fixed bit pattern (RESOLVED)

- **Normative expectation:** `NUM-FLOAT-OP-001` (`CORE-V1-ABSTRACT-MACHINE.md`): "NaN propagates
  as a quiet NaN; operations that create a NaN produce the canonical quiet NaN with sign zero and
  all payload bits other than the quiet bit zero" — a specific, fixed bit pattern for a given
  width (`0x7ff8_0000_0000_0000` for `Float64`, `0x7fc0_0000` for `Float32`), not merely "some
  NaN." The same rule carves out unary negation: "Negation flips the sign bit, including for zero
  and NaN" — `-NaN` must flip whatever sign bit the operand already had, not force sign zero.
- **Original behaviour:** every NaN-producing primitive operation (`0.0 / 0.0`, `inf - inf`,
  `sqrt` of a negative number, arithmetic on an already-NaN operand, the transcendental math
  builtins for out-of-domain inputs) simply returned whatever bit pattern the host `f64`/`f32`
  arithmetic instruction happened to produce, with no canonicalization step at all. IEEE 754 only
  mandates the exponent field and the quiet bit for a quiet NaN — sign and the remaining payload
  bits are otherwise unconstrained — so two different NaN-producing paths were not guaranteed
  (and were not verified) to produce bit-identical results, violating the "canonical" requirement
  even though every NaN still printed as `NaN` (which is bit-pattern-insensitive and so never
  surfaced the gap through any existing test).
- **Resolution:** added `canonical_nan_bits(width: FloatWidth) -> f64` (the two literal bit
  patterns above, spelled out explicitly rather than relying on `f32::NAN`/`f64::NAN` — which
  happen to already equal them — so the canonicalization is self-documenting at the call site)
  and `canonicalize_nan(value, width)` (returns `canonical_nan_bits(width)` if `value.is_nan()`,
  else `value` unchanged — infinities, signed zero, and every finite value pass through
  untouched). Wired into every primitive-arithmetic and standard-math-builtin call site that can
  produce a float result: `Add`/`Sub`/`Mul`/`Div`/`Rem` (via a new `canonicalize_float_result`
  wrapper around the existing `normalize_numeric` call), `sqrt`, `math::abs`, `math::pow`,
  `atan2`, and the remaining transcendental builtins (`log`/`log10`/`exp`/`sin`/`cos`/`tan`/
  `asin`/`acos`/`atan`/`floor`/`ceil`/`round`/`trunc`). Unary negation deliberately does **not**
  route through canonicalization — Rust's `-x` for floats lowers to a pure sign-bit-flip
  (`fneg`), matching the spec's explicit negation carve-out; canonicalizing there would have
  wrongly forced a negated NaN back to sign zero. Regressions (all in `interp.rs`, using a new
  `eval_function_result` test helper that runs a zero-argument function through the interpreter
  and returns its `Value` directly, since no STARK-level program can observe a float's bit
  pattern -- there is no bit-reinterpretation primitive in Core v1 and `println`'s `NaN` text is
  identical for every bit pattern):
  `division_by_zero_produces_the_canonical_quiet_nan_bit_pattern_for_float64`,
  `::_for_float32`, `sqrt_of_negative_produces_the_canonical_quiet_nan_bit_pattern`,
  `infinity_minus_infinity_produces_the_canonical_quiet_nan_bit_pattern` (a *created* NaN, not a
  propagated one), `arithmetic_on_an_already_nan_operand_produces_the_canonical_quiet_nan_bit_pattern`
  (a *propagated* NaN, required to canonicalize identically to a created one),
  `every_nan_producing_path_yields_the_same_canonical_bits_for_float64` (the brief's required
  cross-operation assertion — four independently-shaped NaN-producing expressions all compared
  bit-for-bit equal), and `negating_a_canonical_nan_flips_its_sign_bit_instead_of_forcing_sign_zero`
  (proves the negation carve-out is honored, not silently canonicalized away).
- **Owning gate:** correction-brief Issue 4 (post-WP-C2.12), reproduced against the current head
  before fixing, closed the same session.

## DEV-060 — Repeated call to an un-overridden trait default method is wrongly flagged as a move

- **Normative expectation:** calling a `&self` method twice on the same receiver never moves it;
  the second call should see the same borrowed value as the first, exactly as two calls to an
  ordinary inherent method or an overridden trait method already do.
- **Current behaviour:** confirmed with a minimal repro: for a `Greet` trait with a required
  `fn name(&self) -> String` and a default `fn greeting(&self) -> String { self.name() }`,
  implemented for a struct that only overrides `name`, calling `p.greeting(); p.greeting();`
  (two calls, same receiver `p`) raises `E0100 use of moved value 'p'` on the *second* call, even
  though `greeting` only takes `&self`. Confirmed narrow: two calls to an *overridden* trait
  method (`p.name(); p.name();`), or two calls to an ordinary inherent method, are both
  unaffected — the defect is specific to a method resolved through the `default_fallback` path
  in `resolve_method` (an un-overridden trait default), not method dispatch in general.
- **User impact:** any un-overridden trait default method can only be called once per receiver
  per scope — a real, fairly common pattern (calling the same default-implemented method twice)
  is rejected outright.
- **Security/soundness impact:** none identified — a rejection of legal code (availability), not
  an acceptance of illegal code.
- **Workaround:** override the method per-impl instead of relying on the trait default, or bind
  the result of the first call instead of calling it again.
- **Owning gate:** found while writing DEV-051's regression tests (confirmed present on the
  pre-DEV-051-fix head too, via `git stash`, so this is a separate, pre-existing defect, not one
  introduced by DEV-051's fix). Root cause not isolated beyond "the `default_fallback` method
  path in `resolve_method`/its interaction with `borrowck.rs`'s move analysis" — needs its own
  investigation. Regressions documenting the current (defective) behavior and its scope:
  `typecheck::tests::repeated_call_to_unoverridden_default_trait_method_is_wrongly_flagged_as_move`,
  `interp::tests::repeated_call_to_overridden_trait_method_is_unaffected_by_dev060`,
  `::repeated_call_to_inherent_method_is_unaffected_by_dev060`. — unscheduled.

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
- `starkc/src/borrowck.rs`/`flow.rs` (WP-C1.4 tests) — source of DEV-006's closure and DEV-016.
- `starkc/src/literal.rs`/`typecheck.rs` (WP-C1.5 tests) — source of DEV-015's closure and
  DEV-025.
- `starkc/scripts/generate-conformance-report.py` (WP-C1.6) — source of DEV-017's partial
  closure.
- DEV-001, DEV-003 do not appear above: both IDs were retired when their original seed framing
  was superseded by confirmed findings under different numbers (DEV-SEED-001 → DEV-008;
  DEV-SEED-003 → DEV-009) during WP-C0.2, to avoid two IDs describing the same issue.

Current count: 58 numbered deviations total (DEV-002 through DEV-060, DEV-001/DEV-003 retired).
DEV-056 (`?` propagation swallowed outside aggregate construction), DEV-057 (Eq/Ord
comparison dispatch passed owned clones instead of borrowed places), DEV-058 (Float32 nested
inside a tuple/array/Option/Result/struct still formatted via Float64 digits -- the residual gap
DEV-049 left open), and DEV-059 (NaN-producing float operations did not canonicalize to the
spec's fixed bit pattern) were found in an external correction brief following WP-C2.12,
independently reproduced against the current head before fixing, and all four **closed** with
real fixes in the same session -- DEV-057's investigation also found and fixed a second, broader,
pre-existing bug (`promote_to_temp_place`'s 15+ call sites never registered their temporary in
`Frame::order`, so its value was silently discarded via ordinary Rust-level deallocation with no
`Drop::drop` call ever firing).
DEV-051, DEV-052, and DEV-055 were found by WP-C2.12 while building the differential execution
corpus, initially left unfixed (corpus-building is not a semantic-repair WP); all three were
independently reproduced against the current head and **closed** with real fixes in a later
correction-brief session (DEV-051: trait default methods couldn't call a sibling trait method
through `self`, fixed in `typecheck.rs`'s `resolve_method`; DEV-052: qualified `Trait::method(...)`
syntax didn't resolve for compiler `CoreTrait`s, fixed via a new `Res::CoreTraitMember` in
`resolve.rs`/`typecheck.rs`/`interp.rs`; DEV-055: glob-imported unit enum variants didn't resolve
at all, fixed in `resolve.rs`). DEV-053 and DEV-054 were also found there, investigated as a
dedicated follow-up in the same original session, found to share one root cause (a bare `None`
pattern never matched by value -- it silently acted as an unconditional wildcard, confirmed to
produce **wrong runtime output**, not merely a spurious diagnostic -- DEV-053's original
"tuple-pattern usefulness/exhaustiveness" framing was itself a downstream artifact of this same
misclassification, not a separate algorithm bug), and **closed** with a real fix in
`resolve.rs`/`typecheck.rs`. DEV-060 (repeated call to an un-overridden trait default method
wrongly flagged as a move) was found while writing DEV-051's regression tests, confirmed
pre-existing and unrelated to that fix (via `git stash`), and remains open, unscheduled.
DEV-026 through DEV-035 are closed by WP-C2.2, along with DEV-037, which was found and repaired
during that work. DEV-038 through DEV-043 were found by the post-WP-C2.2 review and closed in
the correction pass. DEV-044 through DEV-050 were found by an external review of the committed
WP-C2.11 alignment work -- each independently reproduced against the compiler before being
trusted, one review claim (`MIN / -1` also failing to trap) found overstated and corrected to
its actual `Rem`-only scope, one review claim (`main` entrypoint counting type-namespace items)
independently refuted and not opened as a deviation -- and closed in a post-WP-C2.11 correction
pass; DEV-049 recorded one known residual gap left open at the time (Float32 values formatted
only through the generic, static-type-free `Display for Value` path), closed by DEV-058 in a
later correction-brief session. DEV-017 remains partially closed
(tooling built, 39 of 59 rules remain unclassified). DEV-036 is closed (WP-C2.12): the
filename/path-based module-loader bypass is replaced by an explicit, harness-only opt-in named
by exact fixture. DEV-009, DEV-022, DEV-023, and DEV-024 — which WP-C2.6 had assigned C2.8/C2.9
decision ownership and C2.11 implementation ownership — were all **resolved by WP-C2.11**; see
their individual entries. (A prior revision of this paragraph, written at WP-C2.6 time, still
described them as open; corrected 2026-07-19 during the C3-entry governance-repair pass.)
**Currently open (2026-07-19):** DEV-005 (unowned), DEV-010 (WP-C8.2/C8.3), DEV-011
(unscheduled), DEV-012 (WP-C8.7), DEV-017 (partial, unscheduled remainder), DEV-060 (C3-ENTRY,
disposition required before the C3 workload freeze).
2 informational not-owned items remain (DEV-SEED-008, DEV-SEED-014).

### WP-C2.7 abstract-machine rule mapping

The normative abstract machine now gives every runtime deviation a representation-independent
rule: DEV-024/026/027/038/043 → `EXEC-DISPATCH-001`; DEV-028/041/042 →
`REF-SLICE-001`; DEV-029 → `DROP-ORDER-001`; DEV-030 → `PAT-DROP-001`; DEV-031 →
`EXEC-FOR-001`; DEV-032 → `OBS-COMPARE-001` plus the standard-library iteration rule;
DEV-033 → `EXEC-EVAL-001`; DEV-034 → `EXEC-ONCE-001`; DEV-035 → `REF-RETURN-001`;
DEV-037 → `REF-PROJECT-001`; DEV-039 → `DROP-LOOP-001`; and DEV-040 →
`DROP-COLLECTION-001`. Closed entries remain regression evidence; open entries retain their
C2.8–C2.11 disposition.
