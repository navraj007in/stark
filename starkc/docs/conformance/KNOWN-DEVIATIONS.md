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

## DEV-006 — Multi-file diagnostic provenance loss (resolve/flow/borrowck)

- **Normative expectation:** Charter §1.5 rule 17: "Source identity must survive the pipeline.
  AST/HIR/MIR/query results and diagnostics must retain the correct file, module, package, and
  artifact provenance."
- **Current behaviour:** `Span` carries no file identity at all (`source.rs:10-13`); there is no
  `FileId`/`SourceId` type anywhere in the crate. Parse (`parser.rs:359-363`) and typecheck
  (`typecheck.rs:1916-1919,2065-2068` plus 4 backfill sites) correctly reconstruct per-item file
  identity via a `HashMap<ItemId, Arc<SourceFile>>` side table. Resolve (`resolve.rs`, 20
  diagnostic sites, zero `.with_file()` calls), flow analysis (`flow.rs:21-24`, file parameter
  named `_file` and structurally unused), and borrow checking (`borrowck.rs`, single
  whole-crate `self.file`, no per-item lookup) do not — they render every diagnostic against
  whichever file the top-level caller happened to pass (always the package's root file).
- **User impact:** for any multi-file `stark` package (the only paths using `PackageGraph` — 
  `stark check`/`build`/`run`/`test`), a name-resolution, control-flow, or borrow-check
  error/warning originating in a non-root `mod`-loaded file renders with the **wrong filename**
  and byte offsets mapped against the **wrong file's** line-start table. `SourceFile::line_col`
  clamps out-of-range offsets (`source.rs:70`) rather than panicking, so this fails silently,
  producing a plausible-looking but incorrect `-->` diagnostic header — actively misleading
  during debugging, not just an omission.
- **Security/soundness impact:** none to compiled-program safety (the underlying check still
  runs correctly; only its *reported location* is wrong), but it is a diagnostics-integrity gap
  that Charter rule 16 ("diagnostics are part of behaviour") treats as a first-class defect, not
  cosmetic.
- **Workaround:** for multi-file packages, if a resolve/flow/borrowck diagnostic's reported file
  looks suspicious, manually check every file in the package rather than trusting the printed
  location.
- **Proposed disposition:** extend resolve's `.with_file()` usage to match typecheck's pattern
  (resolve already has full access to per-module file identity via `ModuleData::file`); extend
  flow/borrowck to accept and use per-item file lookups the way typecheck does, rather than a
  single whole-crate file. Considering a real `FileId`/`SourceId` type is a larger,
  CE3-adjacent-but-not-quite architectural question worth raising if the ad hoc `Arc<SourceFile>`
  threading pattern proves insufficient once this is fixed properly.
- **Owning gate:** WP-C1.2 (resolve) and WP-C1.4 (borrowck); flow is a smaller fix bundled with
  either.

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

## DEV-008 — Structural equality, not `Eq` trait dispatch, at runtime

- **Normative expectation:** per the roadmap's own framing (WP-C1.3): equality dispatch must
  follow one consistent, documented semantics — either the compiler dispatches through a user's
  `Eq` implementation at runtime, or an unambiguous spec defect is corrected via the spec-bug
  protocol. "A hidden interpreter-only structural equality rule is not accepted as an
  undocumented third behaviour."
- **Current behaviour:** `==`/`!=` (`BinOp::Eq`/`Ne` in `interp.rs`) are implemented as pure
  structural equality on the interpreter's internal `Value` enum via Rust's derived `PartialEq`
  — there is no dispatch through a user's `Eq` trait implementation at runtime. `Eq` as a trait
  bound is currently a type-checker-only concept: checked as a bound during type checking, never
  dispatched as a method during execution. `assert_eq`/`assert_ne` (added in Phase 4E/WP8.3)
  inherit the same behavior, since they compare `Value`s the same way `==`/`!=` do.
- **User impact:** the interpreter's structural equality happens to match what a correct
  `derive(Eq)` would produce for ordinary data. A user-written *custom* `impl Eq for T` (if
  expressible at all in Core v1 — unconfirmed whether users can hand-write `Eq` vs. only
  auto-derive it) would be silently ignored at runtime: the program would type-check against
  the custom impl's bound but execute using structural equality regardless of what that impl
  says.
- **Security/soundness impact:** low-to-moderate — this is exactly the kind of "hidden
  interpreter-only rule" the Charter singles out as unacceptable. It's a semantic-consistency
  gap between what the type checker verifies and what the interpreter executes, which is the
  category of bug that erodes trust in "the interpreter is the semantic reference" (Charter
  rule 6).
- **Workaround:** none at the language level; document that `Eq` behaves as if always
  structurally derived, regardless of any custom implementation, until this is resolved.
- **Proposed disposition:** first determine whether Core v1 actually permits hand-written `impl
  Eq for T` (if not — if `Eq` is auto-derive-only — this deviation may collapse to a
  documentation gap rather than a behavior gap). If hand-written impls are permitted, implement
  real trait dispatch in the interpreter; if not, and structural equality is the intended
  semantics for all `Eq` types, document that explicitly in `04-Semantic-Analysis.md` or
  `06-Standard-Library.md` per the spec-bug protocol so it's no longer an undocumented "hidden"
  rule.
- **Owning gate:** WP-C1.3 — explicitly the correct owner per the roadmap's own text.

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

## DEV-013 — `STD-004` (standard traits) exhaustiveness unresolved

- **Normative expectation:** `06-Standard-Library.md`'s trait surface (Clone, Hash, Default,
  Display, Error, Iterator).
- **Current behaviour:** `typecheck.rs:5598-5637` recognizes `Clone`/`Hash`/`Display`/`Default`/
  `Iterator` as compiler-known trait bounds; `gate4a_prelude_traits.rs` (12 tests) exercises
  them. `grep -n '"Error"' typecheck.rs` found **no** matches during WP-C0.3/C0.4 — `Error`
  trait bound recognition is unconfirmed and may be genuinely absent, not merely untested.
  Separately unconfirmed: whether users can hand-write `impl <StdTrait> for T` versus only
  trigger compiler-builtin behavior when the bound is satisfied structurally (the same open
  question DEV-008 raises for `Eq`, likely generalizing to this whole trait family).
- **User impact:** unknown until resolved — could range from "fully covered, this was just an
  undertested corner" to "Error trait support genuinely doesn't exist yet."
- **Security/soundness impact:** none identified yet; contingent on the resolution.
- **Workaround:** none; unresolved.
- **Proposed disposition:** WP-C1.3 rule-level audit — confirm or refute `Error` trait support,
  and resolve the hand-written-impl-vs-builtin-only question for this trait family alongside
  DEV-008.
- **Owning gate:** WP-C1.3.

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

## DEV-016 — Repository-wide clippy debt (pre-existing, not WP-C1.1-introduced)

- **Normative expectation:** Charter §2.5 lists `cargo clippy --all-targets -- -D warnings`
  passing as a default definition-of-done requirement.
- **Current behaviour:** 22 clippy errors exist across `typecheck.rs`, `lsp/protocol.rs`, and
  other files not touched by WP-C1.1 (confirmed by isolating clippy output to files this WP
  changed: zero hits).
- **User impact:** none to compiled-program behavior; this is a code-quality/CI-hygiene gap.
- **Security/soundness impact:** none identified.
- **Workaround:** none needed; doesn't affect correctness.
- **Proposed disposition:** clean up opportunistically when each affected file is next touched
  by an owning WP, or open a small dedicated cleanup WP.
- **Owning gate:** unscheduled.

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
- DEV-001, DEV-003 do not appear above: both IDs were retired when their original seed framing
  was superseded by confirmed findings under different numbers (DEV-SEED-001 → DEV-008;
  DEV-SEED-003 → DEV-009) during WP-C0.2, to avoid two IDs describing the same issue.

Current count: 22 numbered deviations total (DEV-002 through DEV-022), of which DEV-002,
DEV-014, DEV-020, and DEV-021 are closed/confirmed-correct (no fix owed); 2 informational
not-owned items (DEV-SEED-008, DEV-SEED-014).
