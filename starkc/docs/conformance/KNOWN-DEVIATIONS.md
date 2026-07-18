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
- **Owning gate:** WP-C2.9 decides the Core I/O/profile/process contract; WP-C2.11 implements
  and evidences the approved contract. C2.6 inventory row `STD-IO-001`.

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
- **Owning gate:** WP-C1.6 tooling is closed; the remaining evidence gap is WP-C2.11.

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
- **Proposed disposition:** WP-C2.4 supplied compiler-owned position queries, but did not turn
  Type/Pat/Item containment into exhaustive conformance evidence. C2.11 must either add that
  adversarial evidence or narrow the invariant explicitly.
- **Owning gate:** WP-C2.4 query infrastructure is closed; residual verification is WP-C2.11.

## DEV-019 — Diagnostic-code collisions with the normative E-code table

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
- **Owning gate:** WP-C2.11. (WP-C1.6's conformance evidence generator, closed 2026-07-18,
  reports per-rule test evidence but does not cross-reference diagnostic codes against the spec's
  E-code catalog — that would be a distinct, not-yet-built check; this deviation's earlier "WP-C1.6
  to catch systematically" note assumed more overlap with that WP's actual scope than it turned
  out to have.) WP-C2.5 stabilizes the transport without freezing this pre-alpha catalogue;
  WP-C2.11 owns the evidence-complete reallocation.

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
- **Owning gate:** WP-C2.9 decision (`CORE-Q-023`); WP-C2.11 implementation/evidence if approved.

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
- **Owning gate:** WP-C2.8 settles method/hook semantics; WP-C2.11 implements and evidences the
  approved contract.

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
- **Owning gate:** WP-C2.8 settles associated-item/conversion semantics; WP-C2.11 performs the
  root-cause correction and evidence update.

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

## DEV-036 — Parser's filename-based module-bypass heuristic remains a residual risk for real user projects

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

Current count: 41 numbered deviations total (DEV-002 through DEV-043, DEV-001/DEV-003 retired).
DEV-026 through DEV-035 are closed by WP-C2.2, along with DEV-037, which was found and repaired
during that work. DEV-038 through DEV-043 were found by the post-WP-C2.2 review and closed in
the correction pass. DEV-017 remains partially closed (tooling built, 39 of 59 rules remain
unclassified). DEV-036 remains open, now explicitly owned by WP-C2.12 as a parser-loader
hardening regression in the differential corpus. DEV-009, DEV-022, DEV-023, and DEV-024 remain
open with C2.8/C2.9 decision ownership and C2.11 implementation/evidence ownership assigned by
WP-C2.6.
2 informational not-owned items remain (DEV-SEED-008, DEV-SEED-014).
