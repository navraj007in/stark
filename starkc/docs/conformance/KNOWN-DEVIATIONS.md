# STARK Compiler — Known Deviations and Stub Ledger

> **How to read this file (added 2026-08-09, AS8).** Entries are **append-only**: a deviation
> tracked across several packets gets a NEW heading each time rather than an edited one. **The
> first heading for a deviation is therefore not its status** — the last one is. `DEV-121` opens
> with *"(OPEN; instance fixed CD-305, class open)"* and is CLOSED **3,558 lines later**.
>
> This is the same hazard AS8 fixed at the top of `COMPILER-STATE.md`, and the same fix applies:
> read the index below before reading the body.
>
> `starkc/scripts/as8-reconcile-deviations.py` regenerates the reconciliation and cross-checks
> every entry against the decision record and the test corpus. Its standing findings, 2026-08-09:
> **7 deviations are closed in the record and named by no test**, and **44 are named in no decision
> record or archive at all**. Neither is automatically wrong — an open deviation may keep a
> reproducing test and a closed one may keep its regression test — but both are short lists worth
> reading rather than 189 entries worth.

## Multi-entry deviations — the last heading is the live one

| Deviation | Entries | First | **Live** | Latest heading |
| --- | ---: | ---: | ---: | --- |
| `DEV-121` | 13 | L2813 | **L6371** | DEV-121 — CLOSED (owner ruling, 2026-08-08) |
| `DEV-197` | 4 | L5730 | **L6403** | DEV-197 — CLASS CLOSED (2026-08-08) |
| `DEV-195` | 2 | L5520 | **L5557** |  DEV-195 RULING (owner, CD-387, 2026-08-07) |
| `DEV-196` | 2 | L5595 | **L5620** |  DEV-196 — ANSWERED by measurement (2026-08-07) |
| `DEV-206` | 2 | L6186 | **L6201** | DEV-206 — REVISED: `Display` accepted an unsized slice place and rejected its borrowed view [CLO |
| `DEV-213` | 2 | L46 | **L6566** | DEV-213 — CLOSED (C10-P, 2026-08-09). The cache is invalidated per PACKAGE, not per URI |
| `DEV-222` | 2 | L8532 | **L8800** | DEV-222 — RESOLVED (2026-08-11): pattern lowering asks what a pattern may name |
| `DEV-223` | 3 | L8618 | **L8818** | DEV-223 — RESOLVED (2026-08-11): the qualifier's associated namespace is searched first |

*Derived by `as8-reconcile-deviations.py`; no status is asserted here that the file does not
already state in its own last heading for that deviation.*

---

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

## DEV-213 — the LSP caches one whole-package analysis PER OPEN URI, and invalidates only the edited one (OPEN, found by AS8, 2026-08-09)

**Demonstrated at HEAD, not hypothesised.** `starkc/src/lsp/server.rs`
`as8_editing_one_file_leaves_other_uris_cached_analyses_stale` passes today, which is the defect.

`ServerState::compilation_cache` is keyed by URI and each value owns a whole-package
`ProjectAnalysis` — AST, HIR, resolution tables, type tables, symbol index. Three facts compose
into a wrong answer:

```text
state.rs:90    one full ProjectAnalysis is cached PER OPEN URI
state.rs:73    update_document removes ONLY the edited URI's entry
server.rs:782  handle_workspace_symbol merges symbols from EVERY cached analysis
```

Open `main.stark` and `child.stark` of one package; rename a symbol in `child.stark`. `didChange`
drops `child.stark`'s entry and recompiles it, and `main.stark`'s analysis — which also contains the
whole package, including `child` — is never invalidated. `workspace/symbol` then answers from both,
so the response contains **the new name AND the name that no longer exists**.

**Severity.** Wrong answers to `workspace/symbol`, not a crash and not a compilation defect: `stark
build` and every engine are unaffected because none of them uses this cache. It is an editor-surface
correctness bug, which is the class C8 closed short on (DEV-012).

**Found by AS8's LSP profiling**, which the packet's scope framed as a PERFORMANCE question —
*"replace whole-package ProjectAnalysis duplication per open URI where measurement shows material
cost"*. The measured cost is modest (32-module package: 22 ms for one analysis, 181 ms for eight
open URIs). The duplication's real consequence is not cost, it is **N copies with independent
invalidation**.

```text
modules   one analysis   x8 open URIs   diagnostics
      4          1.4ms         12.2ms             7
      8          1.8ms         15.2ms            15
     16          7.7ms         51.0ms            31
     32         22.0ms        181.3ms            63
```

**Not fixed here.** AS8 is an assurance packet; a cache-ownership change to the LSP is
implementation and takes ordinary checkpoints. The test is written so the repair flips its polarity
rather than deleting it — its own message says so.

**OWNER RULING 2026-08-09 — DEV-213 DOES NOT BLOCK SPRINT 4.** It is a real HEAD defect and
correctly carries a DEV, but it is an LSP/editor correctness defect found by assurance — not a
reason to mutate the frozen compiler architecture again before closeout. It is fixed in the next
bounded LSP correctness packet.

> **Standing qualification until this closes:** any claim that `workspace/symbol` is correct under
> multi-file editing must be stated as qualified. The LSP answers correctly for a single open file
> and for a freshly opened one; it is the combination of several open URIs of one package and an
> edit to any of them that is wrong.

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
  (`typecheck/:1916-1919,2065-2068` plus 4 backfill sites) correctly reconstructed per-item
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
  `push_diag`/`current_file_arc()` helpers (`resolve.rs`), mirroring typecheck/'s own
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
  `typecheck/`'s `require_operator_bound` already required a real `impl Eq for T` to exist for
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
  `typecheck/::option_result_vec_box_satisfy_eq_when_their_type_args_do`,
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

## DEV-012 — VS Code extension UI interactively verified for 3 of 10 features (OPEN, NARROWED)

**Narrowed 2026-08-06 (CD-385).** Gate C8 closed with this deviation open rather than by declaring
it satisfied — the gate's claim is correspondingly narrower. See
`STARKLANG/docs/compiler/GATE-C8-CLOSURE.md` §2.

- **Normative expectation:** Charter WP-C8.7: "protocol tests alone do not prove UI behaviour"
  — real editor validation requires an Extension Development Host or packaged-extension session.
- **Interactively confirmed (2026-07-31**, VS Code 1.130.0, `starklang.stark-language@0.2.0`,
  macOS 26.5.2 arm64, real STARK package): **hover, go-to-definition, find-references.**
- **Still unverified in an editor, and the remaining scope of this deviation:** diagnostics (on
  type and on save), formatting / format-on-save, completion, signature help, rename, document
  symbols, semantic tokens. Each is covered by protocol tests only, which is the distinction this
  deviation exists to draw. Status bar rendering and command palette entries are likewise
  unconfirmed.
- **Environmental note that will bite the next validator:** the extension defaults
  `stark.compiler.path` to `starkc`, and VS Code launched from Finder does not inherit a shell
  `PATH` — a `~/.local/bin` install is invisible to it unless the setting is given an absolute
  path.
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
    dispatch point in both `core_method_signature` (typecheck/) and `call_core_method`
    (interp.rs, reusing `Value`'s existing derived Rust `Clone`).
  - **Default trait method bodies: confirmed BROKEN, now FIXED** (found while testing the trait
    family broadly; squarely inside WP-C1.3's own checklist item "default methods"). A trait
    method with a real default body was never used as a fallback when unoverridden — the HIR
    already carried `TraitItem::Method { body: Some(_), .. }`, it was simply never consulted.
    Fixed in both typecheck/ (a `default_fallback` search before concluding "not found") and
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
  against suffix range; `typecheck/`'s `convert_int_suffix` only mapped the suffix to a type
  tag.
- **User impact (while open):** a program could declare an integer literal with a suffix that
  cannot represent its value, and the compiler accepted it silently.
- **Security/soundness impact:** low-moderate — not a memory-safety issue, but a real type-system
  soundness gap: the declared type's range guarantee didn't actually hold for literals.
- **Resolution:** design question settled (user-approved 2026-07-18): typecheck/const-eval time,
  not the lexer — an unsuffixed literal's fit-check needs its inferred target type, which the
  lexer never has. Fixed in `typecheck/body.rs`'s `check_expr` `Lit::Int` arm: suffixed literals
  checked against their suffix's exact range (new **E0008**, via a new
  `literal::int_suffix_range_contains` helper); unsuffixed literals promoted to Int64 if they
  don't fit Int32, rejected (E0008) if they don't fit Int64 either. A defense-in-depth suffix
  re-check was also added to `interp.rs::eval_lit`. Both share a new `src/literal.rs` module
  (also used to fix a second, previously-unknown bug found while building it — see DEV-025).
- **Owning gate:** WP-C1.5 (closed).

## DEV-016 — Repository-wide clippy debt (RESOLVED in WP-C1.4)

- **Normative expectation:** Charter §2.5 lists `cargo clippy --all-targets -- -D warnings`
  passing as a default definition-of-done requirement.
- **Original behaviour:** 22 clippy errors existed across `typecheck/`, `interp.rs`,
  `lsp/protocol.rs`, and `lsp/server.rs`, none touched by WP-C1.1 (confirmed by isolating clippy
  output to files that WP changed: zero hits). CI (`.github/workflows/ci.yml`'s `fmt, clippy,
  test` job) had been red since the 2026-07-17 03:29 push for exactly this reason, across several
  unrelated feature commits and both governance-bootstrap commits.
- **User impact:** none to compiled-program behavior; this was a code-quality/CI-hygiene gap.
- **Security/soundness impact:** none identified.
- **Resolution:** fixed as a standalone cleanup during WP-C1.4 at the user's explicit request.
  All 22 fixes are mechanical and zero-behavior-change: 13x `args.get(0)` → `args.first()`
  (`typecheck/`); 2x explicit-closure-clone → `.cloned()` (`interp.rs`, `lsp/server.rs`); 2x
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
  colliding with `typecheck/`'s correct E0203 use for "ambiguous trait method call." `parser.rs`
  uses E0202 for module-loading errors ("file not found for module," "conflicting module files"),
  colliding with `resolve.rs`'s own correct E0202 use for "undefined type." Two more found during
  WP-C1.5, while touching match-arm code for the exhaustiveness fix: `typecheck/`'s "unreachable
  match arm" warning uses E0500 — spec table: E0500="Trait not implemented" (an *error*, not a
  warning) — colliding with 15 other, spec-correct E0500 "trait not implemented" error sites in
  the same file. `typecheck/`'s "method call on non-struct/enum type" error uses E0303 — spec
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
  `starkpkg.json`, under which `typecheck/`'s filesystem-walk-up package-root detection
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
  typecheck/.
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
  different resolution path (`find_associated_fn` in interp.rs and its typecheck/ counterpart)
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
  `Eq`/`eq` dispatch DEV-008 added. `typecheck/body.rs::ty_satisfies_operator_bound` already accepts
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
  is not iterable." This is also caught at compile time: `typecheck/`'s `for`-loop type-checking
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
- **Proposed disposition:** both `typecheck/`'s for-loop type check and `interp.rs::iter_values`
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
  this entry.** Re-reading `typecheck/`'s exhaustiveness check (`check_expr`'s `Match` arm)
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
  `PatKind::Path { res: Res::Builtin(builtin), .. }` value pattern. `typecheck/`'s
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
  `typecheck/`) and always tag `F64`; `math::abs` is generically `T -> T` and preserves the
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

## DEV-060 — Repeated call to an un-overridden trait default method is wrongly flagged as a move [CLOSED, C3-ENTRY, 2026-07-19]

- **Normative expectation:** calling a `&self` method twice on the same receiver never moves it;
  the second call should see the same borrowed value as the first, exactly as two calls to an
  ordinary inherent method or an overridden trait method already do.
- **Previous behaviour:** confirmed with a minimal repro: for a `Greet` trait with a required
  `fn name(&self) -> String` and a default `fn greeting(&self) -> String { self.name() }`,
  implemented for a struct that only overrides `name`, calling `p.greeting(); p.greeting();`
  (two calls, same receiver `p`) raised `E0100 use of moved value 'p'` on the *second* call, even
  though `greeting` only takes `&self`. Confirmed narrow: two calls to an *overridden* trait
  method (`p.name(); p.name();`), or two calls to an ordinary inherent method, were both
  unaffected.
- **Root cause (isolated during C3-ENTRY closure):** `borrowck.rs`'s `method_receiver` — which
  the `Call` handler uses to decide whether a method receiver is moved, borrowed, or mutably
  borrowed before executing the borrow-checked body — only ever searched `ImplItem::Fn`
  overrides. It had no equivalent to `typecheck/body.rs::resolve_method`'s `default_fallback`
  (WP-C1.3/DEV-013), the mechanism that lets an un-overridden trait default method type-check at
  all. So for a call to such a method, `method_receiver` returned `None`, and the `Call`
  handler's `None => self.check_expr(*base)` arm ran instead of the `Some(Receiver::Ref/RefMut/
  Value)` arms — `check_expr`'s `Path` arm unconditionally consumes (moves) any `Local`/
  `SelfValue` place, regardless of the method's real receiver kind. The two typecheck-level
  candidate searches (`resolve_method`'s override collection and its `default_fallback`) and the
  borrowck-level search (`method_receiver`) had silently drifted to cover different sets of
  callable methods.
- **User impact (while open):** any un-overridden trait default method could only be called once
  per receiver per scope — a real, fairly common pattern (calling the same default-implemented
  method twice) was rejected outright.
- **Security/soundness impact:** none identified — this was a rejection of legal code
  (availability), not an acceptance of illegal code.
- **Fix:** added the matching trait-default-body fallback directly to `method_receiver`
  (`borrowck.rs`) — mirrors `typecheck/`'s `default_fallback` search (find a trait impl for
  the receiver's type where the trait declares an un-overridden method with a body matching the
  call name) but returns just that method's declared `sig.receiver`, which the existing
  `Some(Receiver::Ref/RefMut/Value)` arms then handle exactly as they already do for overridden
  methods. Verified both the `&self` case (original repro) and a `&mut self` variant (two
  sequential calls to an un-overridden `&mut self` default must register two non-conflicting
  borrows, not a move) — the `RefMut` arm wasn't exercised by the original repro alone. Full
  workspace suite re-run clean: 596 passed / 0 failed / 2 ignored (up from 594; two new tests,
  net of one rewritten in place), `cargo fmt --all -- --check` and `cargo clippy --workspace
  --all-targets --all-features -- -D warnings` both clean.
- **Regression tests:**
  `typecheck::tests::repeated_call_to_unoverridden_default_trait_method_is_no_longer_flagged_as_move`
  (rewritten from the original defect-documenting test to assert success),
  `typecheck::tests::repeated_call_to_unoverridden_mut_default_trait_method_is_no_longer_flagged_as_move`
  (new, `&mut self` variant),
  `interp::tests::repeated_call_to_unoverridden_default_trait_method_executes_correctly` (new,
  end-to-end execution — proves correct *output*, not just absence of a diagnostic),
  `interp::tests::repeated_call_to_overridden_trait_method_is_unaffected_by_dev060`,
  `::repeated_call_to_inherent_method_is_unaffected_by_dev060`. — unscheduled.

## DEV-061 — Indirect calls through function-value locals/parameters are not executable [CLOSED, pre-C4.1 correction pass, 2026-07-19]

- **Normative expectation:** `fn(...) -> ...` types denote non-capturing function values
  (`03-Type-System.md` §Function Types); a named function assigned to a fn-typed local, or
  received as a fn-typed parameter, can be invoked (`f(x)`). The stdlib contract depends on this
  (`Option::map(f: fn(T) -> U)`, `06-Standard-Library.md`).
- **Current behaviour (confirmed empirically, 2026-07-19, while exercising CD-021 workload items
  16-17 for the first time):** the *simplest* indirect call fails at runtime with
  `"expression is not callable"` — `let f: fn(Int32) -> Int32 = double; f(21)` (item 16-17
  shape), and calling through a fn-typed *parameter* (`fn apply(f: fn(Int32) -> Int32, v: Int32)
  { f(v) }`) both fail. The type checker accepts all of it.
- **Root cause (isolated):** `interp.rs`'s call dispatch for a `Path` callee handles
  `Res::Builtin | Item | Variant | TraitMember | CoreTraitMember | AssociatedFn` and sends
  everything else — including `Res::Local`/`Res::SelfValue` — to the
  `"expression is not callable"` error arm. The *general* indirect-call machinery (evaluate
  callee → `Value::Function(item)` → `item_callable` → `call_callable`) already exists in the
  non-`Path` fallback arm directly below; a local-callee path simply never reaches it.
- **User impact:** the entire fn-value feature (workload items 16-23) is a compile-time façade —
  every spec-legal indirect call fails at runtime. Same severity class as DEV-035 was
  (checker-accepts / runtime-always-fails for a whole feature area).
- **Security/soundness impact:** none — availability (legal code fails), not acceptance of
  illegal code.
- **Workaround:** call functions directly by name; none for callback-shaped APIs.
- **Owning gate:** found during Gate C4 entry (pre-WP-C4.1 fn-value property resolution). The
  interpreter is the semantic oracle for C4.4's HIR/MIR differential — workload items 16-23
  have no oracle until this is fixed. **Owner approved fix-now (CD-027); FIXED**: added a
  `Res::Local | Res::SelfValue` arm to `interp.rs`'s Path-callee call dispatch, routing to the
  same evaluate-callee → `Value::Function` → `item_callable`/`call_callable` machinery the
  non-path fallback already used. Verified: single indirect call, call through a fn-typed
  parameter, `f(f(v))`, and generic-fn coercion (TYPE-FN-002) all execute with correct output.
  Regression tests: `interp::tests::indirect_calls_through_fn_value_locals_and_params_execute`,
  `::generic_fn_coerced_to_fn_value_executes`. — closed.

## DEV-062 — Function-typed values are not treated as `Copy` by borrow checking [CLOSED, pre-C4.1 correction pass, 2026-07-19]

- **Normative expectation:** "reference values, **function values**, `Unit`, and `!` are `Copy`"
  (`03-Type-System.md` §Copy and Drop, line 748). Two uses of the same fn-typed local never move.
- **Current behaviour (confirmed empirically):** `f(f(10))` — CD-022 workload item 22's exact
  shape — fails borrowck with `E0100 use of moved value 'f'`; so does any second use of a
  fn-typed local (`f(21); apply(f, 7)`).
- **Root cause (probable, not yet isolated to the line):** borrowck's/typecheck's Copy
  classification has no `Ty::Fn` arm, so fn-typed values default to move semantics.
- **User impact:** each fn-value local is single-use; `f(f(v))` is impossible.
- **Security/soundness impact:** none — rejection of legal code.
- **Workaround:** rebind the function name per use.
- **Owning gate:** same discovery as DEV-061; owner approved fix-now (CD-027); **FIXED**: added
  `Ty::Fn { .. } => true` arms to `borrowck.rs::is_copy_type` and
  `typecheck/traits.rs::is_copy_with_impls` (the latter previously listed `Ty::Fn` explicitly as
  non-Copy, contradicting the spec). Regression test:
  `typecheck::tests::fn_typed_local_is_copy_and_reusable`. — closed.

## DEV-063 — `Option`/`Result` combinators (`map`, `and_then`, …) missing from the method table [CLOSED, pre-C4.1 correction pass, 2026-07-19]

- **Normative expectation:** `06-Standard-Library.md` §Option lists `map<U>(self, f: fn(T) -> U)`
  and `and_then`; §Result lists `map`/`map_err`/`and_then`; `Iterator` lists `map`/`fold`/
  `reduce` taking `fn(...)` values. At minimum required for the `std-full` claim
  (STD-PROFILE-001: "everything in this document").
- **Current behaviour (confirmed empirically):** `v.map(double)` on `Option<Int32>` fails at
  *typecheck* with `E0304 method call on non-struct/enum type 'Option<Int32>'` — the combinator
  has no entry in `core_method_signature` (other Option methods, e.g. `is_some`/`is_none`,
  dispatch fine, so this is per-method absence, not a broken dispatcher; the diagnostic text is
  also misleading).
- **Honest governance note:** STD-OPTION-001 was approved `settled` under CD-023 on the evidence
  cited at the time; `map`/`and_then`'s absence is a newly surfaced implementation gap *within*
  that row's scope — recorded here as a deviation (the ledger's job), not a reopening of the
  row's normative home.
- **User impact:** the fn-value-consuming half of the Option/Result API is unusable (blocks
  workload item 18).
- **Security/soundness impact:** none.
- **Workaround:** hand-written `match`.
- **Owning gate:** owner approved fix-now (CD-027); **FIXED**: `Option::map`/`and_then` and
  `Result::map`/`map_err`/`and_then` added to `typecheck/`'s core-method signatures (fresh
  inference variable for `U`/`F`, unified through the declared `fn(...)` parameter — the same
  pattern the iterator `.map` signature already used) and to `interp.rs::call_core_method` as a
  consuming pre-match interception (take_place the receiver, call the fn value re-entrantly with
  no receiver borrow outstanding; gated on the receiver being Option/Result so lazy iterator
  `.map` is untouched). Verified incl. all pass-through sides (`None`, `Err` for `map`, `Ok` for
  `map_err`). Regression test:
  `interp::tests::option_result_combinators_execute_with_fn_values`. — closed.

## DEV-064 — Coercion of a generic fn with undetermined parameters is not rejected [CLOSED, C4.5c, 2026-07-19]

- **Normative expectation:** TYPE-FN-002 (`03-Type-System.md` §Function Types, added under
  CD-027): a generic function may coerce to a concrete fn type only when the expected type
  fully determines every generic argument; otherwise the program is rejected at compile time.
  TYPE-GENERIC-001's closing rule ("if any parameter remains unconstrained, the call requires
  explicit arguments") states the same requirement for direct calls.
- **Previous behaviour (confirmed empirically):** `fn count<T>() -> Int32` coerced to
  `fn() -> Int32` — `T` appears nowhere in the signature, so it is undetermined — was accepted
  and ran. Benign in the type-erased interpreter (T never influences execution), but
  ill-defined for a monomorphising backend, which is exactly why TYPE-FN-002 requires
  rejection. The undetermined direct call (`count();`) was equally accepted.
- **Fix (WP-C4.5c):** the checker now records the ordered generic-argument types for every
  use of a generic fn (`TypeTables::generic_insts`, keyed by the referencing path
  expression), grounds them once inference completes, and rejects any instantiation still
  containing an inference variable with **E0004** ("Cannot infer type" — the spec-assigned
  code) — covering both the coercion and direct-call forms uniformly. Tensor-kinded generics
  (`Dim`/`DType`/`Device`) unify through the tensor context and are exempt. The same recorded
  table is what MIR monomorphisation consumes, so instance collection never sees an unnamed
  instantiation (mir.md §2's stated upstream requirement).
- **Regression evidence:** `typecheck/::tests::undetermined_generic_fn_coercion_is_rejected`,
  `::undetermined_generic_call_requires_turbofish` (rejection + turbofish acceptance), and
  `::determined_generic_fn_coercion_publishes_instantiation` (determined coercion stays
  accepted and publishes `[Int32]`). — closed.

## DEV-067 — Bounded generic parameters lose their bounds at intra-generic call sites and behind references [CLOSED, WP-C4.7-7, 2026-07-20]

- **Normative expectation:** inside `fn f<T: Ord>(...)`, the parameter `T` satisfies the
  bound `Ord` — a call `g(x)` to `fn g<U: Ord>(u: U)` with `x: T` must type-check
  (TYPE-GENERIC-001: all bound obligations are discharged by the caller's own bound), and a
  trait-bound method call through a reference receiver (`shape: &T` with `T: Area`,
  `shape.area()`) must resolve exactly as the by-value form does (auto-deref, 03-Type-System
  §Operators and Traits / §Generic Type Inference).
- **Current behaviour (confirmed empirically at `d1c1c25`, pre-C4.5c, so pre-existing):**
  two over-rejections of valid Core programs. (a) A generic fn recursing (or calling any
  other generic fn) with a *bounded* parameter fails `E0500 type 'T' does not satisfy trait
  bound 'Ord'` — the bound check on the callee's instantiation does not consult the enclosing
  body's own parameter bounds. (b) A method call on a `&T` receiver whose `T` carries the
  trait bound fails `E0302 method 'area' not found for type '&T'` — bounded-parameter method
  lookup works by value but does not peel the reference. Both surfaced while writing
  WP-C4.5c's differential tests (the tests use unbounded recursion and by-value receivers as
  workarounds, marked with this ID).
- **User impact:** valid generic code patterns (bounded recursion, `&T` trait dispatch) are
  rejected; workarounds exist (unbounded parameter where possible, by-value receiver).
- **Security/soundness impact:** none — over-rejection only; no invalid program is accepted.
- **Resolution (WP-C4.7-7, 2026-07-20).** Two independent causes, one per symptom.
  **(b) `&T` receivers:** the bounded-parameter method lookup tested the UNPEELED receiver type,
  so it matched `t: T` but never `t: &T`. TYPE-METHOD-002 requires auto-dereference to peel
  leading `&`/`&mut` before receiver matching, and the concrete-type path immediately below it
  already computed exactly such a peeled `receiver_ty` — the peel was simply performed after the
  parameter check instead of before. Moving it above makes both paths obey the same rule.
  **(a) Intra-generic call sites:** `satisfies_bound` had **no `Ty::Param` arm at all** and fell
  through to `_ => false`, so a caller's own `T: Ord` could never discharge a callee's `T: Ord`
  (TYPE-GENERIC-001). Adding the arm was not sufficient on its own: trait-bound obligations are
  collected during body checking and verified in a **deferred pass** that runs after every body,
  by which point `current_fn_generics` belongs to whatever was checked last. Each obligation now
  carries the generic environment it arose in, and the deferred pass restores it before checking.
  The new arm mirrors the `Ty::Param` arm `ty_satisfies_operator_bound` already had, so the two
  bound checks now agree about parameters.
- **Soundness:** over-rejection removed, nothing newly accepted. An obligation is discharged only
  by a bound the enclosing function actually declared — a concrete type without the impl, and an
  UNBOUNDED parameter forwarded into a bounded position, are both still E0500 (pinned by tests).
- **Regression evidence:** `mir_differential.rs::bounded_generic_method_through_reference_agrees`
  (instantiated at two types, so dispatch is exercised, not just the check) and
  `::bounded_generic_call_chain_agrees` (a three-deep chain of bounded generic calls);
  `gate2_valid.rs::unsatisfied_trait_bounds_are_still_rejected` (both negatives).
- **Owning gate:** WP-C4.7-7 (closed).

## DEV-065 — Array index OOB reported "use of moved or invalid field" [CLOSED, C4.5b-1, 2026-07-19]

- **Normative expectation:** out-of-bounds indexing is the language's index-out-of-bounds
  **trap** (`CORE-V1-ABSTRACT-MACHINE`; TrapCategory `IndexOutOfBounds`); its runtime message
  should identify it as such.
- **Previous behaviour (confirmed empirically):** `a[i]` with `i` out of range trapped with
  `"use of moved or invalid field"` — the generic place-projection failure message
  (`interp.rs::place_value`/`place_value_mut` used one message for every failed projection).
  Trap *behaviour* was correct; the *diagnostic* was misleading for the most common trap a
  user can hit, and made the HIR↔MIR trap-category correspondence unmappable.
- **Found:** while building C4.5b-1's MIR indexing differential (the comparator needed a
  category↔message mapping for `IndexOutOfBounds` and the oracle had none).
- **Fix:** projection-kind-aware failure message (`projection_failure_message`): `Index`/
  `MapIndex` projections report `"index out of bounds"`; field projections keep the moved-field
  message. Diagnostics-only change: no accepted/rejected program change, no trap-behaviour
  change; no corpus snapshot referenced the old message.
- **Regression evidence:** `tests/mir_differential.rs::array_out_of_bounds_trap_agrees_with_
  provenance` — both engines trap `IndexOutOfBounds` at the same source span, with the oracle
  message matching the category fragment. — closed.

## DEV-066 — Reading through a reference wrongly moved the reference [CLOSED, C4.5b-2, 2026-07-19]

- **Normative expectation:** dereferencing uses a reference by *read*; it never consumes the
  reference. `*r = *r + 1` for `r: &mut Int32` — the canonical mutation-through-`&mut`
  pattern — is legal Core (03-Type-System §References and Lifetimes).
- **Previous behaviour (confirmed empirically):** any value use of `*r` marked `r` moved
  (borrowck routed the deref operand through the generic consuming path; `&mut T` is
  non-Copy, so the "use" became a move), and the subsequent write through `*r` failed with
  `E0100 use of moved value 'r'`. Every write-after-read through a mutable reference — incl.
  `fn write_it(r: &mut Int32) { *r = *r * 2; }` — was rejected. Method receivers were
  unaffected (separate `method_receiver` path), which is why `&mut self` methods worked while
  free functions taking `&mut T` did not.
- **Found:** by the C4.5b-2 MIR differential — the new reference-argument tests failed at the
  *front end*, not in MIR (the second oracle defect the differential infrastructure has
  surfaced, after DEV-065).
- **Fix:** both deref paths in `borrowck.rs` (`check_expr`'s unary arm and
  `check_owned_value`'s deref branch) now availability-check the reference place without
  consuming it. The existing move-out-of-non-Copy-pointee rejection is unchanged.
- **Regression evidence:** `tests/mir_differential.rs::reference_arguments_and_derefs_agree`
  and `::reference_to_struct_field_agrees` (both engines agree end-to-end on read + write +
  re-read through `&`/`&mut`), plus `::mut_self_receiver_mutates_caller_local`. — closed.

## DEV-068 — User `impl Copy` structs classified as Move in MIR lowering [CLOSED, C4.5e-0, 2026-07-19]

- **Normative expectation:** a nominal type with `impl Copy` is Copy (03-Type-System §Copy
  and Drop); using such a value twice, or reading a field after passing it to a function, is
  legal and must execute.
- **Previous behaviour (confirmed empirically):** `mir/lower.rs::is_copy` returned `false`
  for every user struct/enum regardless of Copy impls, so every use lowered as a `Move` —
  and the C4.5d field-precise move verifier then *rejected* valid programs
  (`MIR-0007 move from possibly-moved place`) that the HIR oracle accepts and runs. Latent
  until the field-precise refinement made the verifier strict enough to notice; surfaced by
  the external C4.5c-head review's warning (CD-030) and confirmed against the tree before
  fixing.
- **Fix:** `is_copy` consults `type_has_copy_impl` (same impl-search pattern as
  `type_has_drop_impl`). Lowering trusts the impl's presence: the front end has already
  enforced the all-Copy-fields and no-`Copy`+`Drop` rules for the impl to exist.
- **Regression evidence:** `tests/mir_differential.rs::user_copy_impl_struct_is_copy_in_mir`
  (Copy struct passed twice by value, field read afterwards; both engines agree). — closed.

## DEV-069 — Front end + HIR interpreter are not multi-file-span-clean [CLOSED, WP-C4.7-4, 2026-07-20]

- **What:** the type checker and the HIR reference interpreter resolve `Span`s against the
  **entry file only**. In a multi-file program (`mod helper;` loading `helper.stark`), any
  name, literal, or field whose span lies in a dependency file is read against the entry
  file's text. Observed failure shapes, in increasing subtlety: (a) `TypeChecker::text`
  panics "byte index N out of bounds" when the dependency file is longer than the entry
  file; (b) cross-file **method** resolution reads garbage method names (e.g. method
  `'\nfn '` not found); (c) cross-file **literals** fail to parse ("invalid literal");
  (d) cross-file **field reads** resolve the wrong field name and report "use of moved or
  invalid field" at runtime. All four were reproduced from one two-file program during
  WP-C4.5f-3c.
- **Scope:** front end + oracle only. The MIR lowering built in f-3c is multi-file-clean:
  `ProgramMeta` interns every source file (`FileId(0)` = entry), maps each item to its
  declaring file and module path, and reads every cross-item name against the owning item's
  file. `resolve.rs`/`hir.rs` carry `synthetic_spans` for generated wrappers so lowering
  never text-reads a synthetic span.
- **Why open:** fixing the front end means threading per-item file identity through
  `typecheck/`, `borrowck.rs`, and `interp.rs` — a front-end WP, out of WP-C4.5's
  MIR scope. Until then the differential multi-file test pins the front-end-safe subset
  (scalar free functions, literal-free dependency bodies, no cross-file methods/fields),
  padded so dependency name spans stay in-bounds.
- **Resolution (WP-C4.7-4, 2026-07-20).** Root cause: `typecheck/`, `borrowck.rs`, and
  `interp.rs` each hold ONE "current file" and read every span against it. For the item being
  checked that is correct — `check_crate` already swapped `self.file` per item — but every
  *lookup* of another item (a nominal's name, an impl's method names, a trait's default method
  names, a `Drop` impl) scans all items in the program and read their spans against the wrong
  file. Two mechanisms fix the whole class:
  1. **`item_text(item, span)`** in all three modules reads a span against the file that
     DECLARES `item` (via the `hir.item_files` map the resolver already populated, which is
     also what MIR's `ProgramMeta` uses). Every cross-item scan now uses it: method resolution,
     trait-default fallback, associated-fn lookup, `Drop` discovery, nominal formatting,
     `item_name`.
  2. **Per-body file swap in the oracle.** The interpreter never swapped file at all. `Callable`
     now carries its declaring file, and all THREE body-execution funnels — `call_callable`,
     `call_user_method`, and the destructor path in `drop_value` — save/restore `self.file`
     around the body, on error paths too. A trait default's body correctly takes the TRAIT's
     file, not the impl's.
  `text()` in all three modules is additionally non-panicking now (`.get(..).unwrap_or("?")`),
  so a residual wrong-file read degrades to a visible `"?"` in a diagnostic rather than
  aborting the compiler. That is a backstop, not the mechanism.
- **Regression evidence:** `tests/multi_file_spans.rs` (new, 3 tests, one per failure shape):
  `cross_file_methods_fields_and_literals_check_and_run` (shapes b/c/d, checked AND executed),
  `a_long_dependency_file_does_not_panic_the_front_end` (shape a — the dependency file is
  deliberately longer than the entry file, which is what turned a wrong-file read into an
  out-of-bounds panic), and `cross_file_trait_impls_and_drop_run_correctly` (cross-file trait
  override, un-overridden trait default, and a cross-file destructor whose ORDER is the
  observable). Plus `tests/mir_differential.rs::multi_file_module_program_agrees_with_qualified_symbols`,
  **widened** from the front-end-safe subset to a cross-file struct with methods, a cross-file
  literal, a cross-file field read, and a cross-file `Drop` impl, with the exact expected
  output pinned (so agreement cannot be vacuous).

## DEV-070 — `match` on a scrutinee behind a shared reference moves it out [CLOSED, WP-C4.6 A2, 2026-07-20]

- **What:** `lower_match` always materializes the scrutinee by value
  (`lower_expr_to_operand(scrutinee)` → an `Operand::Move` for a non-`Copy` enum), then the
  arms consume that temp. When the scrutinee is a place the function does **not own** — a
  `Deref` of a shared reference, e.g. `match *self` inside a `&self` method — this MOVES the
  enum out of the borrowed place and (C4.5f-1) poisons it. A single match still runs (the
  moved-from place is never read again), but a **second** read of the same borrowed value
  traps the MIR interpreter with `read of a moved-out place … (C4.5f-1 poison)`.
- **Repro:** `enum Color { Red, Green, Blue }` with
  `impl Color { fn ord(&self) -> Int32 { match *self { Color::Red => 0, Color::Green => 1,
  Color::Blue => 2 } } }` and `fn main() { let a = Color::Green; println(a.ord());
  println(a.ord()); }` → poison on the second call. Independent of A3 (no `==` involved); A3's
  user-`Eq` dispatch merely made it easy to reach, since a realistic enum `Eq::eq` body matches
  `*self`/`*other`.
- **Scope:** MIR pattern lowering. The fix is the **A2** work (non-consuming / by-reference
  match): when the scrutinee is a borrowed place (or `Copy`), the match must read the
  discriminant and bind payloads **without** moving/dropping the scrutinee. Owned by WP-C4.6 A2.
- **Consequence for A3:** user-`Eq` dispatch itself is complete and correct (proven by
  `user_struct_eq_dispatch_agrees`, whose `eq` body reads `self.id`/`other.id` fields rather
  than matching). Enum `Eq` impls whose body matches `*self` are blocked on this deviation, not
  on A3's dispatch mechanism.
- **CLOSED (WP-C4.6 A2, 2026-07-20), both engines.** Root cause was in BOTH interpreters:
  (a) **oracle** — `Receiver::Ref` bound `self` to a value CLONE, not `Value::Ref(place)`
  (the same bug class the correction brief fixed for `Eq::eq`/`Ord::cmp` dispatch, Issue 2),
  so `*self` failed "cannot dereference non-reference" before any match ran; fixed by binding
  a genuine reference (observationally equivalent otherwise — the referent cannot mutate
  during the call and the old clone was discarded without STARK drop effects). (b) **MIR** —
  `lower_match` gained `MatchMode::ByRef`: a scrutinee read through a shared reference is
  matched IN PLACE (discriminant on the place; `Copy` payloads bound by copy; unbound payloads
  untouched — the referent keeps ownership; no arm-end drops), while owned scrutinees keep the
  C4.5d consuming semantics — per the CE3 rule, consumption depends on the scrutinee, never a
  blanket "all matches borrow". Guards: a user-`Drop` scrutinee type and a non-Copy BOUND
  payload through a reference stay clean-Unsupported (see DEV-072, since CLOSED — the front end
  failed to
  reject that move-out-of-borrow).
- **Regression evidence (the CE3 matrix):** `match_deref_self_twice_fieldless_agree`,
  `match_deref_self_copy_payload_agree`, `match_deref_self_noncopy_wildcard_agree`,
  `match_copy_scrutinee_reusable_agree`, `match_owned_drop_scrutinee_still_consumes_agree`.

## DEV-072 — Binding a non-Copy payload through a shared reference passes borrowck [CLOSED, WP-C4.7-5, 2026-07-20]

- **What:** `match *self { Holder::Val(s) => … }` inside a `&self` method, where the payload
  is non-`Copy` (e.g. `String`), passes the front end — but binding `s` moves the payload out
  of a shared borrow, which the ownership rules forbid (MOVE/OWN rules; a shared borrow never
  transfers ownership). The oracle's legacy clone semantics masked it (the clone was consumed,
  not the referent).
- **Scope:** front-end borrow checking (move-out-of-borrow through match bindings unchecked).
  MIR keeps such programs clean-Unsupported ("binding a non-Copy payload through a shared
  reference"), so nothing mislowers. Owner: front end.
- **Repro:** `enum Holder { Empty, Val(String) }` +
  `impl Holder { fn peek(&self) -> Int32 { match *self { Holder::Val(s) => 1, Holder::Empty => 0 } } }`.
- **Resolution (WP-C4.7-5, 2026-07-20).** `borrowck.rs`'s `match` handling did not inspect
  patterns at all. It now classifies the scrutinee with `scrutinee_reads_through_ref` — a
  deliberate mirror of MIR lowering's function of the same name, so the two engines classify
  by-reference matching identically **by construction** rather than by coincidence, which is
  precisely what this deviation was — and walks each arm's pattern (recursively, including
  nested tuple/array/struct patterns and shorthand struct-field bindings), reporting **E0101**
  for any binding whose type is non-`Copy`. Both shared and mutable derefs count: ownership
  cannot be moved out of either.
  What stays legal is as important as what does not: wildcards, literals, and unit-variant path
  patterns bind nothing, and `Copy` bindings copy rather than move — matching by reference is
  fine, only *taking ownership* is not. A fix that rejected all by-reference matching would have
  broken far more than it repaired, so both positive cases are pinned by tests.
  The MIR guard is **kept as defense in depth** (its message now says so): it is unreachable for
  checked programs, but the charter's rule is that nothing unsupported reaches a backend
  silently, and an unreachable guard costs nothing while a missing one would mislower a move out
  of a borrow.
- **Regression evidence:** `tests/gate2_valid.rs::binding_a_non_copy_payload_through_a_reference_is_rejected`
  (E0101) and `::matching_through_a_reference_without_taking_ownership_is_accepted` (wildcard and
  `Copy`-payload positives); `mir_differential.rs::match_deref_self_noncopy_wildcard_agree`
  continues to pass unchanged.

## DEV-071 — `match` on `Ordering` with all three variants is flagged non-exhaustive [CLOSED, WP-C4.7-7, 2026-07-20]

- **What:** the front-end exhaustiveness checker does not recognize the prelude `Ordering` enum
  as having exactly `{ Less, Equal, Greater }`, so a `match` covering all three explicit
  variants is rejected `E0303` "non-exhaustive pattern match". A wildcard arm (or two explicit
  variants + `_`) is accepted.
- **Repro:** `let o = a.cmp(&b); match o { Ordering::Less => …, Ordering::Equal => …,
  Ordering::Greater => … }` → E0303.
- **Scope:** front-end typecheck exhaustiveness (`Ordering`'s variant set isn't registered for
  the usefulness/exhaustiveness algorithm). Independent of MIR: the MIR `CoreOrdering` match
  path lowers and runs correctly (proven by `ordering_value_round_trips_through_match_agree`,
  which uses an explicit-plus-wildcard match). Owner: front end (adjacent to the A2 pattern
  work but distinct — this is exhaustiveness, not by-reference matching).
- **Consequence for A3-Ord:** none for the dispatch/round-trip mechanism; only cosmetic (users
  had to add a `_` arm to an all-variants `Ordering` match until fixed).
- **Resolution (WP-C4.7-7, 2026-07-20).** The prelude `Ordering` is `Ty::Core(CoreType::Ordering)`
  with `Res::Builtin` variants — structurally like `Option`/`Result`, and invisible to the
  `Ty::Enum`/`matched_variants` machinery for the same reason. `Option`/`Result` had already been
  given explicit arms; `Ordering` had not, so it fell through to the "unknown domain, require a
  wildcard" default that WP-C1.5 introduced. The exhaustiveness check now tracks
  `Ordering::Less`/`Equal`/`Greater` and treats all three as exhaustive.
- **Soundness:** the enumeration is exact — a two-variant `Ordering` match is still E0303 (pinned
  by a test, since an over-generous domain enumeration would silently accept unsound matches).
- **Regression evidence:** `gate2_valid.rs::ordering_match_exhaustiveness_counts_all_three_variants`
  (both directions); `mir_differential.rs::ordering_value_round_trips_through_match_agree` was
  **rewritten to use three explicit arms**, dropping the `_` workaround it had carried for this
  deviation.

## DEV-073 — Checker does not match GENERIC impls for operator/iterable bounds [CLOSED, WP-C4.7-5, 2026-07-20]

- **What:** the front end's bound-satisfaction checks do not recognize a generic impl as
  satisfying a concrete instantiation's bound: (a) `impl<T> Eq for W<T>` does not satisfy
  `W<Int32>: Eq` — `a == b` on `W<Int32>` is rejected E0500 "does not satisfy operator trait
  'Eq'"; (b) `impl<T> Iterator for Repeat<T>` is not recognized by the for-loop iterable check
  — `for x in r` on `Repeat<Int32>` is rejected E0001 "for-loop requires an iterable value".
  Non-generic impls match fine; ordinary METHOD calls on generic nominals also work (the
  method-resolution path does handle generic impls) — only the operator-trait/iterable
  bound-satisfaction paths lack generic-impl matching.
- **Scope:** front-end typecheck (impl matching in `require_operator_bound` / the for-loop
  iterable check). Distinct from DEV-067 (bounds lost at intra-generic call sites): this is
  about CONCRETE instantiations failing to find their generic impl. MIR-side A1 dispatch is
  ready for these the moment the checker admits them (the `find_impl_fn` path is
  instantiation-aware). Owner: front end.
- **Repro:** `struct W<T> { v: T } impl<T> Eq for W<T> { fn eq(&self, o: &W<T>) -> Bool { true } }
  fn main() { let a = W { v: 1 }; if a == W { v: 2 } { println(1); } }` → E0500.
- **Resolution (WP-C4.7-5, 2026-07-20).** The root cause was one level below the two failing
  checks: `type_from_hir_without_diagnostics` **drops generic arguments**
  (`Ty::Struct(item, Vec::new())`). That was invisible while its only consumers compared
  NON-generic nominals — `struct P` converts to `Struct(id, [])` either way — but it meant an
  impl's written `W<T>` converted to `W<>`, whose argument count could never match `W<Int32>`'s,
  so the exact-match test in both checks failed for every generic impl. A new
  `impl_self_ty_with_args(impl_item, ty)` preserves the arguments and keeps type parameters as
  `Ty::Param`, and both checks now unify through **`match_impl_type`** — the same one-way
  unification method resolution already used for this exact question, which is why method calls
  on generic nominals had always worked while operators and `for` loops on the same types did
  not. The iterable half additionally applies the resulting substitution to the associated
  `Item`, so `type Item = T` on `Repeat<Int32>` yields `Int32` rather than a dangling parameter.
- **MIR impact: none.** WP-C4.6 A1 had already made dispatch instantiation-ready; both programs
  lowered and ran correctly the moment the checker admitted them, with no lowering change —
  confirmed by the two differential tests below, which are the ones this deviation had blocked.
- **Regression evidence:** `mir_differential.rs::generic_impl_eq_dispatch_agrees` and
  `::generic_user_iterator_for_loop_agrees`.

## DEV-074 — HIR oracle slice-bound trap messages folded into the "out of bounds" family [CLOSED at creation, WP-C4.6 A4-2e, 2026-07-20; numbered by WP-C4.7-1]

- **What:** during A4-2e (shared slice views) three HIR-interpreter slice-bound error messages
  were rewritten so that every slice-bound failure reports as one message family:
  `"slice start is negative"` → `"slice range out of bounds (negative start)"`,
  `"slice end is negative"` → `"slice range out of bounds (negative end)"`, and
  `"slice range overflow"` → `"slice range out of bounds (inclusive end overflow)"`
  (`starkc/src/interp.rs`, commit `2a53c47`). This is an oracle **behavior** change (observable
  message text) that §0.5's escalation boundary says must be documented by a ledger entry; it
  was recorded only in `mir-amendment-A1-strings-runtime.md` rev. 10 at the time. This entry is
  that record, written retroactively during WP-C4.7-1.
- **Why it is correct, not a regression:** 06-Standard-Library and the abstract machine group
  *all* slice-bound failures as a single trap (`IndexOutOfBounds`); the three prior messages
  implied three distinct failure kinds that the language does not distinguish. The differential
  comparator matches trap categories by message fragment (`oracle_fragment` in
  `tests/mir_differential.rs`), so a single family is required for MIR and the oracle to agree
  on a construct the spec says has one outcome. No trap category, provenance span, or exit
  status changed — only the human-readable text, and only in the direction of the spec.
- **Scope:** HIR interpreter (`interp.rs`) only. MIR side unaffected (compiler traps carry
  `message: None` and compare by category).
- **Status:** CLOSED at creation — the change is the intended behavior and is fully implemented;
  the deviation being recorded is the *governance* gap (an oracle behavior change that went
  unnumbered), not an outstanding code defect. See `mir-amendment-A1-strings-runtime.md` rev. 10.

## DEV-075 — Ordered comparison on `Bool` and `Char` is accepted but unimplemented, and the two engines disagree [CLOSED, WP-C4.7 DEV-075 increment, 2026-07-20]

- **What:** the type checker accepts `<`/`<=`/`>`/`>=` on `Bool` and `Char`
  (`ty_satisfies_operator_bound` admits `Eq`/`Ord` for every non-`Unit` primitive), but neither
  engine implements them consistently:
  - `false < true` — **HIR oracle**: runtime error "invalid binary operation". **MIR**: internal
    error `BinOp Lt on Bool(false), Bool(true)`. Both fail, so this is an accept-then-fail: a
    program the checker approved cannot run in either engine.
  - `'a' < 'b'` — **HIR oracle**: runtime error "invalid binary operation". **MIR**: succeeds,
    printing the correct answer. This is an **engine divergence**: MIR accepts and produces
    output where the oracle refuses, which is exactly the class of defect the differential
    harness exists to catch. It went unnoticed because no test compares an ordered operator on
    `Char`.
- **Scope:** the checker's operator-bound surface vs. both engines' `BinOp` evaluation. Neither
  is obviously the "right" side: the fix is either to implement the comparisons in both engines
  (if 03 intends `Bool`/`Char` to be ordered) or to narrow the checker to reject them (if it does
  not). 03's operator section gives primitives "built-in meaning (Numeric Semantics below)",
  which speaks to numeric types and does not clearly settle `Bool`/`Char` ordering — so this
  needs a spec reading, not just a code fix. `Char` ordering by Unicode scalar value is the
  conventional answer and is already what MIR does.
- **Found by:** WP-C4.7-6.2, while scoping which primitives should get `Ord::cmp`. `cmp` was
  deliberately restricted to integers, `String`, and `str` — the types both engines fully
  support — rather than built on top of this gap. Enabling `cmp` for `Bool`/`Char` belongs in
  the change that closes this deviation.
- **Repro:** `fn main() { if false < true { println(1); } else { println(0); } }` and the same
  with `'a' < 'b'`; run each through `cargo run --example c46_probe` and `--example oracle_run`.
- **Resolution (owner SPECIFICATION decision, 2026-07-20).** The owner split the two types rather
  than treating them as one gap, and directed that the normative documents carry an explicit
  matrix so "similar for other types" could not remain the authority.
  - **`Char` is ordered by Unicode scalar value.** It implements `Eq`, `Ord` and `Hash`; all four
    ordered operators compare scalar values; `Char::cmp` returns the corresponding `Ordering`.
    Explicitly NOT locale-sensitive or linguistic collation. MIR's existing behaviour was
    directionally correct, so the ORACLE was aligned to it (a `(Char, Char)` arm in `eval_binary`)
    and `Char` was added to the primitive `cmp` surface in both the checker and lowering.
  - **`Bool` implements `Eq` and `Hash` but NOT `Ord`.** `<`, `<=`, `>`, `>=` and `Bool::cmp` are
    now compile-time errors; `==`/`!=` remain valid. An ordering could be defined, but Core v1 has
    no use for ordering truth values, and rejecting is clearer than fixing an arbitrary one.
- **Spec change (the first in WP-C4.7):** `PRIM-TRAIT-001` — a normative "Primitive Trait and
  Operator Matrix" in 06-Standard-Library, replacing the illustrative `impl Eq for Int32` plus
  "// ... similar for other types", with a cross-reference from 03-Type-System's operator table.
  The compiled `STARK-Core-v1.md`/`.html`/`.pdf` were regenerated and the spec-fixture corpus
  re-extracted (one fixture changed, manifest in sync).
- **A distinction the matrix had to make explicit:** for primitives, operators have built-in
  meaning and do NOT dispatch through the traits. So `Float64` admits `<` and `==` as IEEE
  operations while implementing neither `Eq` nor `Ord` (IEEE comparison is not an equivalence
  relation or a total order — NaN is unordered and unequal to itself), and therefore cannot
  satisfy a `T: Ord` bound or key a `HashMap`. Conflating the operator gate with the trait gate
  silently broke ordinary float comparison once during implementation; both are now pinned by
  `floats_compare_but_do_not_satisfy_ord_bounds`.
- **Regression evidence:** `mir_differential.rs::char_ordering_agrees` (all four operators plus
  `cmp`, both engines) and `::char_ordering_is_scalar_value_not_collation_agrees` (`'Z' < 'a'`,
  `'0' < 'A'` — comparisons a collation order would get WRONG, so the test distinguishes the
  specified rule from a plausible alternative); `gate2_valid.rs::bool_is_not_ordered` (all four
  operators, `Bool::cmp`, and `==` still accepted) and `::floats_compare_but_do_not_satisfy_ord_bounds`.

## DEV-076 — HIR oracle `Option::unwrap_or` double-drops the payload and leaks the unused default [CLOSED, WP-C4.7-8.1a, 2026-07-20]

- **What:** with a `Drop`-carrying payload, `Option::unwrap_or` in the HIR interpreter runs the
  payload's destructor **twice** and the discarded default's destructor **never**. Both halves
  violate the abstract machine: `EXEC-ONCE-001`/`DROP-ORDER-001` require every value's destructor
  to run exactly once.
- **Repro** (`starkc/examples/oracle_run.rs`):
  ```stark
  struct Tag { id: Int32 }
  impl Drop for Tag { fn drop(&mut self) { println(self.id); } }
  fn main() {
      println(100);
      let o: Option<Tag> = Some(Tag { id: 1 });
      let t = o.unwrap_or(Tag { id: 2 });
      println(999);
  }
  ```
  Observed: `100 999 1 1` — the payload (`id: 1`) is destroyed twice and the unused default
  (`id: 2`) is never destroyed. Expected: `2` once (the default, discarded) and `1` once (the
  bound value, at end of scope). The `None` case is correct (`100 200 2 300 2`), and a plain
  `match` over the same droppable Option is also correct (`100 777 1 999` — exactly one drop),
  so the defect is specific to `unwrap_or`'s value handling, not to droppable payloads generally.
- **Severity:** this is a **soundness** defect (double destruction), not an over/under-rejection.
  It is masked today because MIR refuses `unwrap_or` on a droppable payload as a clean
  `Unsupported` ("droppable payload"), so the differential never compares the two engines on this
  construct and the divergence cannot be observed through the harness.
- **Why it matters beyond the oracle:** WP-C4.7-8.1 is the increment that adds droppable
  `unwrap_or` to MIR, and §0.6 makes the oracle the semantics authority that MIR must match.
  **MIR must not be built to match this behaviour.** The oracle must be fixed first, then MIR
  implemented against the corrected timing. Fixing the oracle is an oracle BEHAVIOUR change, so
  per WP-C4.7 §0.5 it needs this ledger entry — which is what this is — and the corrected drop
  timing must be re-pinned empirically before 8.1's lowering is written.
- **Scope:** `starkc/src/interp.rs`, the `unwrap_or` core-method path.
- **Root cause:** the same one as DEV-077 — `unwrap_or` was handled on the *borrowing* method
  path, which operates on a **clone** of the receiver. Taking the payload emptied the clone while
  the original `Option` kept it and destroyed it again at end of scope. The discarded default
  fared worse: nothing consumed it, so its destructor never ran at all. (Core has no laziness, so
  the default is always *evaluated*, which is precisely why it always owes a destruction.)
- **Resolution (WP-C4.7-8.1a, 2026-07-20).** `unwrap_or` now consumes the receiver from the real
  place (`take_place`) alongside `into_inner`/`close`, and explicitly drops whichever value it
  discards: on `Some`/`Ok` it yields the payload and drops the default; on `None` it yields the
  default; on `Err` it yields the default and drops the displaced error payload.
- **Pinned timing (this is what MIR must match, and it is not the obvious answer):** the
  discarded default is destroyed **at the `unwrap_or` call**, not at end of scope. For
  `let t = Some(Tag{1}).unwrap_or(Tag{2})` the observable order is `2` (default, at the call)
  then `1` (the bound value, at scope exit) — where the defect previously produced `1` twice and
  no `2` at all.
- **Consequence for the MIR half:** still open. `unwrap_or` on a droppable payload remains a clean
  `Unsupported` in lowering, because moving a payload out of a **drop-tracked** local through a
  `VariantField` projection is refused by the C4.5d guard ("move through a non-field projection of
  a drop-tracked local"); the consuming path needs the drop-flag machinery `lower_enum_match`
  uses. That is the remainder of WP-C4.7-8.1, and it can now be written against a correct oracle.

## DEV-077 — HIR oracle `Box::into_inner` double-drops the payload [CLOSED, WP-C4.7-6.1, 2026-07-20]

- **What:** `Box::into_inner` in the HIR interpreter read its receiver through the ordinary
  *borrowing* method path, which operates on a **clone** of the receiver place. `.take()` emptied
  the clone; the original box kept the value and destroyed it again at end of scope. With a
  `Drop`-carrying payload this was an observable **double drop** (violating `EXEC-ONCE-001`), and
  it diverged from MIR, which drops exactly once.
- **Repro:** `struct Tag { id: Int32 } impl Drop for Tag { fn drop(&mut self) { println(self.id); } }`
  with `let b = Box::new(Tag { id: 1 }); let t = b.into_inner();` — the oracle printed `1` twice.
- **Resolution (WP-C4.7-6.1):** `into_inner` **consumes** the box, so it now takes from the real
  place (`take_place`), exactly like the pre-existing `File::close` case immediately below it in
  `call_core_method`. The dead borrowing arm was removed. Found while building the Box MIR
  surface, because the differential could not agree until the oracle was correct.
- **Relationship to DEV-076:** same family (a consuming operation implemented on a cloned
  receiver) but a different operation and a separate fix. DEV-076 (`unwrap_or`) remains OPEN and
  is WP-C4.7-8.1's blocking prerequisite.

## DEV-078 — Unsuffixed integer literals never adopt an expected integer type [CLOSED, WP-C4.7-6.3, 2026-07-20]

- **Normative expectation:** 03-Type-System's inference algorithm states that expected types
  "flow inward from explicit annotations, **function parameters**, return types, assignment
  destinations, aggregate fields, branch/arm unification, and an enclosing call's expected
  result", and that solving step 5 defaults "an **unconstrained** integer literal to `Int32` when
  representable, otherwise `Int64`". A literal in a `UInt64` parameter position is *constrained*,
  so defaulting must not apply to it — it adopts `UInt64`.
- **Previous behaviour:** the checker assigned every unsuffixed integer literal `Int32`/`Int64`
  **at the literal itself**, before any expectation could reach it. Every use of an integer
  literal where a non-`Int32` integer type was expected was rejected `E0001 type mismatch:
  expected 'UInt64', found 'Int32'` — `v.get(0)`, `takes_u64(0)`, `let a: UInt64 = 9`, and a
  `UInt64` struct-field initializer alike. The workaround (`0 as UInt64`) appears throughout the
  test corpus and in WP-C4.7 §1's guidance to test authors.
- **User impact:** an over-rejection affecting ordinary code, and one that trained casts into the
  codebase. It was originally recorded as a "`Vec::get` literal-typing quirk", which understated
  it: nothing about it is specific to `Vec::get`.
- **Resolution (WP-C4.7-6.3, owner-decided).** Implemented as **general expected-type inference**,
  not a special case. An unsuffixed integer literal now takes a fresh *integer-kinded* inference
  variable; ordinary unification carries the expected type into it; and 03's step 5 is a real
  defaulting pass (`default_unconstrained_int_literals`) that runs after every body is checked and
  before the deferred bound checks. Binding such a variable range-checks the value, so
  `takes_u8(300)` is `E0008` at compile time. The variable is integer-KINDED: it unifies only with
  primitive integer types (plus `!` and error-recovery), so an integer literal cannot stand in for
  a `Bool`. It is expected-type propagation, **not a coercion** — 03's step 4 confines coercions
  to explicit coercion sites — so a suffixed literal (`0i32`) and a typed value (`x: Int32`) both
  still fail against a `UInt64` parameter.
- **Places that must settle a literal eagerly** (they branch on a concrete type and have no later
  constraint to wait for): method-call receivers (`3.cmp(&5)`) and cast operands (`5 as UInt8`).
- **Subtlety worth recording:** a literal variable is often unified with *another* variable rather
  than a concrete type (`MyOpt::Some2(7)` unifies it with the enum's element variable). Defaulting
  therefore has to RESOLVE first and default the end of the chain; defaulting only variables
  absent from the substitution left such chains unbound and they surfaced as `type Infer(N)` at
  MIR lowering.
- **Regression evidence:** `gate2_valid.rs::unsuffixed_integer_literals_adopt_the_expected_integer_type`
  (parameter, annotation, struct field, and the TYPE-INFER-001 later-use case) and
  `::integer_literal_typing_negatives_still_fail` (range, suffix, typed value, non-integer kind);
  `mir_differential.rs::expected_typed_integer_literals_agree` (the adopted widths are observable
  at runtime, so both engines must agree). Unnecessary `as UInt64` casts were removed from the
  differential corpus; casts of genuinely typed values were retained.

## DEV-079 — MIR verification rejects any enum variant with two or more droppable payload fields [CLOSED, WP-C4.7-8.3, 2026-07-20]

- **What:** V-MOVE-1's move dataflow keyed places as `(local, pure-Field path)` and collapsed
  **any** non-`Field` projection to the whole local. `VariantField(v, i)` is a non-`Field`
  projection, so moving two different payload fields out of the same enum local looked like two
  moves of the same whole place: the second was reported `MIR-0007 move from possibly-moved place
  _N[]`.
- **Impact:** `match v { Two::Pair(a, b) => … }` where the payload fields need dropping produced
  MIR that **lowering accepted and verification rejected** — an internal inconsistency between
  two components that are supposed to be independent readings of the same contract, and strictly
  worse than a clean `Unsupported`. It applied to every multi-droppable-field variant, with or
  without a wildcard, including `String` payloads; only single-field variants worked.
- **Why it was not caught earlier:** the differential corpus had no case with a variant carrying
  two droppable fields. WP-C4.6 A2 signed off the "general pattern engine" on nested/tuple/array
  scrutinees, and C4.5d signed off match-drop elaboration, but neither exercised this shape. It
  surfaced while pinning oracle behaviour for WP-C4.7-8.3.
- **Resolution (WP-C4.7-8.3):** `moved_key` gives `VariantField(v, i)` **two** path components —
  the variant, then the field — so sibling fields are distinguishable. This cannot collide with a
  struct's `Field` path because a local has exactly one type, so its projections are either
  struct/tuple fields or variant fields, never both. `Deref` and `Index` still collapse to the
  whole local, which is conservative and correct since neither denotes a statically-known
  disjoint sub-place. The verifier's "honest limitations" note was updated to match.

## DEV-080 — Arm-end drop ORDER for a variant payload with unbound fields diverged from the oracle [CLOSED, WP-C4.7-8.3, 2026-07-20]

- **What:** with a variant payload where some fields are bound and others are wildcards, MIR
  destroyed the payload leaves in plain reverse-FIELD order, while the HIR oracle destroys **all
  bound bindings first (in reverse binding order), then the discarded leaves**.
- **Repro:** `enum Two { Pair(Tag, Tag), Empty }` matched by `Two::Pair(a, _)` over
  `(Tag{id:1}, Tag{id:2})`, with a printing destructor: the oracle prints `1, 2`; MIR printed
  `2, 1`.
- **Why it was invisible:** every program that could expose it was blocked by DEV-079 above — it
  could not reach execution to be compared. Fixing the verifier is what made the divergence
  observable, which is a good argument for fixing conservative-rejection bugs rather than living
  with them.
- **Resolution (WP-C4.7-8.3):** `consume_variant_payload` consumes UNBOUND fields first and bound
  fields second. Arm-end drops run in reverse registration order, so registering the discarded
  leaves first makes the bindings drop first — in reverse binding order — and the discarded
  leaves after them, matching the oracle. Verified against three shapes, including a three-field
  `(a, _, c)` whose expected order (`c`, `a`, then the wildcard) distinguishes this rule from
  both plain reverse-field and plain declaration order.

## DEV-081 — Shorthand struct-field bindings in a consuming match never drop (a leak) [CLOSED, WP-C4.7-8.3b, 2026-07-20]

- **What:** `bind_shorthand` — the lowering for a shorthand field binding, `P { a, b }` rather
  than `P { a: a, b: b }` — moved the field's value into the binding local but **never registered
  that local as droppable**, in any mode. The value was moved out of the scrutinee (so the
  scrutinee no longer owns it) and nothing ever destroyed it.
- **Impact:** a **leak**, not a double drop, which is exactly why nothing failed loudly: no
  verifier rule was violated, no assertion tripped, and a program that does not print from its
  destructor looks entirely correct. It affected both the FLAT path (`consume_variant_payload`'s
  struct-variant arm) and, once WP-C4.7-8.3b enabled droppable scrutinees there, the general
  recursive engine.
- **Repro:** `struct Tag { id: Int32 } impl Drop for Tag { fn drop(&mut self) { println(self.id); } }`
  with `struct P { a: Tag, b: Tag }` and `match Some(P { a: Tag{id:1}, b: Tag{id:2} }) { Some(P { a, b }) => … }`
  — the oracle prints `2, 1` at arm end; MIR printed nothing. The enum-variant form
  (`enum E { V { a: Tag, b: Tag } }` matched by `E::V { a, b }`) behaved identically, confirming
  the flat path was affected before 8.3b existed.
- **Resolution (WP-C4.7-8.3b):** `bind_shorthand` registers the binding as droppable with flags
  true in `Consuming` mode, exactly as `bind_field_local` (the named-binding path) already did.
  The two paths differed only in this one respect, which is what made the gap easy to miss.
- **Regression evidence:** `mir_differential.rs::struct_shorthand_bindings_drop_agrees`, covering
  both the struct-nominal and the struct-shaped-enum-variant forms.

## DEV-082 — A method call on a slice receiver consumes it, so a `&mut [T]` local cannot be used twice [CLOSED, WP-C4.7-8.6, 2026-07-20]

- **What:** `borrowck.rs`'s `method_receiver` had no arm for slice or array receivers, so a method
  call on one returned `None` and the caller's fallback **consumed** the receiver.
- **Impact:** for a `&[T]` local this is harmless — shared references are `Copy`, so the "move" is
  a copy — which is exactly why shared slices shipped in A4-2e without anyone noticing. For a
  `&mut [T]` local it is a real move: `let s = &mut a[1..4]; s.len(); s[0]` failed
  `E0100 use of moved value 's'`. The defect was therefore invisible until exclusive slice views
  existed to expose it.
- **Repro:** `fn main() { let mut a = [1,2,3,4,5]; let s = &mut a[1..4]; println(s.len()); println(s.len()); }`
  → E0100. The same program with `&a[1..4]` is accepted.
- **Resolution (WP-C4.7-8.6):** `method_receiver` returns `Receiver::Ref` for slice and array
  receivers. The slice methods in the surface (`len`, `is_empty`) only read, so a shared borrow is
  the correct receiver kind.
- **MIR counterpart:** lowering also passed the receiver by MOVE. It now reads it by `Copy` — the
  MIR-level equivalent of a shared reborrow — so V-MOVE-1 does not treat a read-only method call
  as consuming an exclusive view. Both engines agree on `s.len(); s[0]`.
- **Regression evidence:** `mir_differential.rs::mutable_slice_views_agree` exercises repeated use
  of a `&mut [T]` local alongside the write-through cases.

## DEV-083 — A concrete position in an impl head cannot match an unresolved receiver type argument [OPEN, found WP-C4.7-8.5, 2026-07-20]

- **What:** method resolution matches an impl's written self type against the receiver's type
  ONE-WAY — impl parameters bind, receiver types do not. When the receiver's own type arguments
  are still unresolved inference variables at resolution time, a **concrete** (non-parameter)
  position in the impl head has nothing to compare against and the match fails.
- **Repro:** `struct Pair<A, B> { x: A, y: B }` + `impl<T> Pair<Option<T>, Int32> { fn tag(&self) -> Int32 { self.y } }`
  with `let p = Pair { x: Some(5), y: 42 }; p.tag();` → `E0302 method 'tag' not found for type
  'Pair<Option<_infer_4>, _infer_5>'`. The `Int32` position in the impl head meets `_infer_5`.
- **Scope:** narrow. It needs ALL of: a generic impl, a **concrete** argument position in its
  self type, and a receiver whose corresponding argument is still an inference variable. The
  common non-bare-head forms are unaffected — `impl<T> Holder<Option<T>>` matches
  `Holder<Option<Int32>>` fine (every position is either the nominal itself or a parameter), and
  WP-C4.7-8.5's tests cover two such instantiations dispatching correctly.
- **Workaround:** give the receiver a known type — annotate the local, or obtain it from a
  function with a declared return type. `fn make() -> Pair<Option<Int32>, Int32>` makes the same
  program compile and run.
- **Why it is NOT fixed here:** the fix would require committing inference variables during
  candidate search, which is a known hazard — binding a variable while probing one candidate can
  select the wrong impl and is a semantics change rather than a bug fix. 03-Type-System's
  TYPE-METHOD-001 requires resolution to be independent of declaration order and to yield exactly
  one candidate; a speculative-binding search needs its own design and evidence. Recorded rather
  than rushed.
- **Impact:** over-rejection only; no invalid program is accepted, and both engines reject
  identically (the checker refuses before either interpreter sees it).
- **DISPOSITION (owner decision, 2026-07-20):** *DEV-083 is deferred to a dedicated post-C5-front-end
  work package. The eventual design must use candidate-local inference snapshots and
  declaration-order-independent candidate evaluation. It must not mutate global inference state
  while probing candidates.*
- **Owner-approved as OUTSIDE the mandatory C5 lowering baseline**, because it is a front-end
  inference-completeness issue with a workaround (annotate the receiver), no MIR or backend
  effect, and no engine divergence. This does not waive exit condition 2 — it records an explicit
  scope decision about what that condition requires.
- **Assigned:** `WP-C6.x Method Resolution Completion` (provisional). Must remain visible in the
  deviation ledger and in release/conformance reporting until closed.

## DEV-084 — `print`/`println` accepted any type, including one with no `Display` impl [CLOSED, WP-C4.7-9 audit, 2026-07-20]

- **What:** `print`/`println` typed their argument as a fresh inference variable, so they accepted
  **any** type. 06-Standard-Library states `Display` is not a syntax hook and that user types must
  implement it, so this was an over-acceptance.
- **Three engines, three answers, for a program the spec says is invalid:** the checker accepted
  `println(p)` on a `Display`-less struct; the HIR oracle rendered it in an unspecified debug-ish
  form (`{x: 1}`); MIR refused it outright ("print/println of this type"). None of the three was
  wrong in isolation — the CHECKER was, by admitting it at all.
- **Resolution (WP-C4.7-9 audit):** the argument type is recorded and checked after inference
  settles (deferred to the same pass as the trait-bound checks, so an argument still under
  inference is not judged early). A standard `Display` type, a container of displayable types, or
  a nominal with its own `Display` impl passes; anything else is E0500.
- **Test impact, recorded because it is a behaviour change:** one interpreter test
  (`float32_nested_in_struct_uses_float32_round_trip_digits`) printed a bare struct to observe
  `Float32` digits and depended on the over-acceptance. It now asserts the REJECTION; its original
  subject — a `Float32` nested in an aggregate keeps `f32` round-trip digits — is unchanged and
  already covered by the `Option`/`Result` and tuple siblings exercising the same
  width-selection path.
- **Regression evidence:** `gate2_valid.rs::printing_requires_display` (rejection, plus the
  standard displayable types and containers still printing) and
  `interp.rs::printing_a_struct_without_a_display_impl_is_rejected`.

## DEV-085 — `for` over a fixed-length array was unsupported in MIR only [CLOSED, WP-C4.7-9 audit, 2026-07-20]

- **What:** the checker accepted `for x in a` over an array and the HIR oracle executed it, while
  MIR refused ("for over a non-range, non-Vec iterator"). An internal inconsistency between
  engines, not a language boundary.
- **Resolution:** lowered as a counting loop reading one element per iteration through the
  ordinary `CheckIndex` proof discipline. Elements are read by COPY; a non-`Copy` element type is
  a precise `Unsupported` for the reason recorded in DEV-086.
- **A bug this found in its own implementation:** `continue` initially targeted the loop header
  directly, skipping the counter increment and spinning forever (the MIR interpreter's fuel limit
  caught it). The continue target is now a latch block that increments first. The control-flow
  test that exposed it was written before the fix, not after.
- **Regression evidence:** `mir_differential.rs::for_over_array_agrees` (values and a running
  total; plus `break`/`continue` and a single-element array).

## DEV-086 — Droppable elements in array patterns and by-value array iteration [CLOSED (patterns) / narrowed (iteration), WP-C4.7 post-exit, 2026-07-20]

- **What:** an array element place is reached through `Projection::Index(ProofLocal)`, and the
  only way to mint a proof is a `CheckIndex` terminator — which READS the array to validate the
  bound. Moving one element out therefore poisons the whole local for V-MOVE-1 (`Index` must
  collapse to the whole local in `moved_key`: a dynamic proof names no statically-known
  sub-place), so the next element's `CheckIndex` reads a possibly-moved place.
- **Consequence:** two constructs stay cleanly `Unsupported` — a droppable element in an array
  pattern (`Some([a, _])` over `[Tag; 2]`) and `for` over an array with a non-`Copy` element type.
  Non-droppable array patterns and `Copy`-element iteration are unaffected and lower normally.
- **Why it is NOT fixed here:** the fix is a **constant-index projection form** that carries no
  proof — the MIR contract has none, and adding one is a shape change requiring CE3 approval
  (WP-C4.7 §0.5). The contract already anticipates the direction: §6 notes the proof discipline
  "covers fixed-length `Array` (verifier may validate against the compile-time length)", so a
  constant form is a natural extension rather than a new mechanism — but it is the owner's call.
- **Resolution (CD-038, owner-approved under CE3).** `Projection::ConstIndex(u64)` — a statically
  known array element, valid only on `Array<T, N>`, bounds-checked by the verifier itself, needing
  no `CheckIndex` and no `IndexProof`, invalid on `Vec`/slice. It participates PRECISELY in move
  analysis, so sibling elements move independently. Consuming array patterns over droppable
  elements now lower and agree with the oracle, including drop order.
- **Same decision required typed internal paths**, and they were adopted: move-dataflow paths and
  drop-unit paths are no longer raw `u32` sequences but typed components (field / variant field /
  constant index), so distinct projection kinds cannot compare equal. Fixed-length arrays also
  decompose into PER-ELEMENT drop units — without that, moving one element out and then dropping
  the array would destroy the moved-out element twice.
- **NARROWED, not fully closed — by-value iteration over a non-`Copy` array element.** The loop
  index is a runtime counter, so no `ConstIndex` names the element being consumed and V-MOVE-1 has
  nothing precise to track. Reading by copy instead would be **unsound**: the array would still own
  the element and destroy it again — a double free for a `String` in a real backend. `Copy`-element
  array iteration is unaffected and lowers normally. **This residual was split out into DEV-090**
  (WP-C4.7 close-out §6): it now rejects in the front end (`E0104`) before either engine, and full
  ownership-transferring non-`Copy` array iteration is deferred to a later language-completion
  package. See DEV-090.
- **Regression evidence:** `mir_differential.rs::droppable_array_pattern_agrees` (wildcard, both
  bound, a discriminating three-element case, and a `String` element);
  `mir_verify.rs::rejects_const_index_out_of_bounds`, `::rejects_const_index_on_non_array`,
  `::accepts_const_index_within_bounds`; corpus case
  `collection_iter__03_slice_views_and_array_iteration`.

## DEV-087 — HIR oracle treats a slice reference as non-`Copy`, so passing one consumes it [CLOSED, WP-C4.7-9 corpus, 2026-07-20]

- **What:** the interpreter's `value_is_copy` classified `Value::Slice` as NOT `Copy`. A slice
  value is a shared reference (`&[T]`), and shared references ARE `Copy` (03-Type-System) — the
  neighbouring `Value::Ref` was already treated that way. Passing a slice to a function therefore
  MOVED it out of the caller's binding.
- **Repro:** `fn total(v: &[Int32]) -> Int32 { … } fn main() { let a = [1,2,3,4,5]; let s = &a[1..4]; println(total(s)); println(s[0]); }`
  — the checker accepts it (correctly), MIR runs it, and the oracle failed
  `"use of unavailable value"` on the second use.
- **How it was found:** writing the `collection_iter__03` frozen-corpus case. The differential
  suite's slice tests happened never to pass a slice to a function *and then reuse it*, so the
  divergence had no coverage. It is the fourth defect in this work package that existed only in
  the gap between two engines rather than inside either one.
- **Resolution (WP-C4.7-9):** `Value::Slice` is `Copy`. Exclusive (`&mut [T]`) views are not
  distinguished, for the same reason `Value::Ref` is not: the interpreter's reference values carry
  no mutability, and write permission is a static property the front end and the verifier enforce.
- **Regression evidence:** the `collection_iter__03_slice_views_and_array_iteration` corpus case
  (shared re-slicing, a slice passed to a function and reused afterwards, exclusive views written
  through, and array iteration), which runs in `exec_snapshots` and in
  `entire_frozen_corpus_agrees`.

## DEV-088 — Cross-file `const`: declaration fixed; USE now deterministically rejected [DECL RESOLVED / USE DEFERRED with a deterministic rejection, WP-C4.7 close-out §7, 2026-07-21]

- **What:** `check_constants` evaluated every `const` initializer with the interpreter still
  pointed at the ENTRY file, so a constant declared in a dependency had its literal read from the
  wrong text — `pub const N: Int32 = 31415;` in `helper.stark` failed
  `E0215 constant evaluation failed: invalid literal`. Same per-item file discipline as DEV-069;
  constants were a fourth site that closure missed, because no test or corpus case had a
  cross-file constant.
- **Declaration-time: RESOLVED.** Evaluation now runs against the declaring file, so the spurious
  E0215 during the const pre-pass is gone.
- **USE site (`helper::N` referenced from another file): DEFERRED, with a deterministic
  rejection.** The oracle's use-site evaluation reads the constant's literal against the USE
  file's text (`"invalid literal"` at runtime) UNLESS the const's value was already cached by an
  earlier same-file touch — so the failure was even data-dependent — while MIR does not lower a
  `const` in value position at all ("use of a non-function item as a function"). The owner's §7
  instruction was to reject unsupported cross-file constant use **deterministically, before
  either engine**, rather than let one engine fail during interpretation and the other during
  lowering.
- **Resolution (owner §7, this close-out).** Using a `const` whose declaring file differs from the
  use site's file is now rejected in the type checker (`E0215`, "using a `const` declared in
  another file is not yet supported") — a single deterministic front-end error that forecloses
  the inconsistency. **Same-file `const` use is unaffected**, and a cross-file *function* that
  reads a same-file `const` internally is unaffected (the use is same-file). MIR's separate
  inability to lower ANY `const` in value position is a distinct clean over-rejection, latent (no
  corpus/differential case exercises it) and likewise deferred.
- **Found by:** writing the `multi_file__01` corpus case. Per scope discipline the case was
  reduced to its actual subject (cross-file structs, methods, trait default + override, cross-file
  `Drop`, provenance) and the constant removed. Both engines agree on the reduced case.
- **DISPOSITION (owner §7):** do not implement cross-file constant use during C4. Deferred to the
  same front-end/multi-file completion package as **DEV-083**. Must remain visible in the deviation
  ledger and release/conformance reporting until implemented. No further cross-file-constant
  implementation campaign is required for C4.
- **Regression evidence:** `gate2_valid.rs::cross_file_const_use_is_rejected`.

## DEV-089 — `println` of a user type with a `Display` impl now dispatches to that impl in both engines [CLOSED, WP-C4.7 close-out, 2026-07-21]

- **What (before):** for a user nominal that DOES implement `Display`, the three components
  disagreed — the **checker** accepted (correct: DEV-084 made the check "has a `Display` impl"),
  the **HIR oracle** ran it but printed its own aggregate/debug rendering (`{x: 1}`), **ignoring
  the user's `Display::fmt`**, and **MIR** refused to lower it.
- **Two problems, both resolved.** (1) an engine divergence (oracle ran what MIR rejected);
  (2) an oracle-correctness question — 06-Standard-Library treats `Display` as an ordinary trait,
  so the statically selected `Display::fmt` must execute.
- **Why it only surfaced then:** before DEV-084, `println` accepted ANY type, so a type WITH an
  impl was indistinguishable from the common no-impl case. Narrowing the checker isolated it.
- **Resolution (owner decision, 2026-07-21): user `Display` dispatch in both engines.**
  - **Spec (06-Standard-Library, PRINT-DISPLAY-001):** `print`/`println`/`eprint`/`eprintln` are
    implementation-provided generic functions `fn print<T: Display>(value: T)` (etc.). They
    evaluate the argument once, select the unique coherent `Display` impl by ordinary trait
    resolution, invoke `Display::fmt` once, print exactly the returned `String`'s UTF-8 bytes
    (`println`/`eprintln` then one `0x0A`), and destroy the formatting `String` after its bytes are
    submitted. No fallback debug/structural rendering exists for a type lacking `Display` — such a
    program is rejected by the checker (E0500).
  - **HIR oracle (`interp.rs::display_text`/`finish_display`):** a user nominal's own `Display::fmt`
    is resolved and executed; its returned `String`'s bytes are printed; the by-value argument is
    destroyed AFTER the bytes are submitted (ordinary by-value call ownership). The internal
    aggregate rendering (`format_runtime_value`) is retained for diagnostics but is no longer the
    language-level `Display` for `print`. `eprint`/`eprintln` route through the same path.
  - **MIR (`mir/lower.rs::lower_print_display`):** ordinary visible MIR — a static
    `Callee::Instance` call to the selected `Display::fmt`, then the existing `StringAsStr` +
    `Print(ln)Str` surface, then a visible `Drop` of the formatting `String` and of the by-value
    argument. **No new MIR shape, no new `RuntimeFn`, no runtime-surface bump.** `fmt` stays a
    normal instance call, so user code, traps and provenance remain visible.
- **Repro (now):** `struct P { x: Int32 } impl Display for P { fn fmt(&self) -> String { String::from("P") } }`
  with `println(p)` prints `P` in both engines; the argument's `Drop` (if any) runs after the
  printed bytes and before the next statement.
- **Regression evidence:** `mir_differential.rs::dev089_user_struct_display_agrees`,
  `::dev089_user_enum_display_agrees`, `::dev089_display_called_once_with_side_effect_agrees`,
  `::dev089_dynamically_constructed_string_agrees`,
  `::dev089_generic_function_with_display_bound_agrees`,
  `::dev089_generic_nominal_display_agrees`,
  `::dev089_formatter_result_and_argument_drop_timing_agrees`, `::dev089_trap_inside_fmt_agrees`;
  and the checker's positive/negative coverage in `gate2_valid.rs::printing_requires_display`.

## DEV-090 — By-value iteration over a non-`Copy` array element [FULLY CLOSED — feature IMPLEMENTED, WP-C6.1d, 2026-07-23]

- **Resolution (WP-C6.1d):** now SUPPORTED. Lowering unrolls the fixed-length iteration into `N`
  `ConstIndex(i)` moves into a per-iteration fresh binding, with the array moved once into a
  per-element-drop-tracked owner and correct break/continue/return/`?`/trap cleanup. The front-end
  E0104 rejection is removed; HIR, MIR, and native execution agree (`native_c6_1_ownership.rs`
  `c61d_*`). The text below is the historical record of the earlier deterministic-rejection state.



- **What:** split from DEV-086. `for x in arr` over a fixed-length array whose element type is
  NOT `Copy` binds each element by value, moving it out of a place named by a **runtime** loop
  index. No static constant-index projection (A5's `ConstIndex`) can name the consumed element, and
  reading it by copy instead would be **unsound** — the array still owns the element and would
  destroy it again (a double free for a `String` in a real backend).
- **Before:** the checker accepted it, the **oracle ran it** (cloning each element out), and MIR
  **refused to lower it** (`LOWER-UNSUPPORTED`) — an engine divergence masked by the checker's
  acceptance and MIR being reached only in the differential path.
- **Resolution (owner §6): reject in the front end, deterministically, before either engine.**
  `borrowck.rs` now rejects by-value iteration over a non-`Copy` array element with `E0104` and a
  diagnostic recommending iterating over a borrow. `Copy`-element array iteration is unaffected and
  lowers normally; consuming array PATTERNS over droppable elements (DEV-086's closed portion) are
  unaffected. The MIR guard remains as defence-in-depth but is no longer reachable for this case.
- **DISPOSITION (owner §6):** full ownership-transferring non-`Copy` array iteration (via loop
  unrolling for small `N`, or runtime-indexed drop flags) is an explicitly accepted limitation
  **outside the C5 baseline**, scheduled for a later language-completion package. Must remain
  visible in the deviation ledger until implemented.
- **Regression evidence:** `gate2_valid.rs::rejects_by_value_iteration_over_non_copy_array` /
  `::accepts_by_value_iteration_over_copy_array`, and the `E0104` registry entry in
  `04-Semantic-Analysis.md`.

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
- `starkc/src/typecheck/` and `starkc/src/interp.rs` (new WP-C1.3 tests) — source of DEV-008
  and DEV-013's closure, plus DEV-023/DEV-024.
- `starkc/src/borrowck.rs`/`flow.rs` (WP-C1.4 tests) — source of DEV-006's closure and DEV-016.
- `starkc/src/literal.rs`/`typecheck/` (WP-C1.5 tests) — source of DEV-015's closure and
  DEV-025.
- `starkc/scripts/generate-conformance-report.py` (WP-C1.6) — source of DEV-017's partial
  closure.
- DEV-001, DEV-003 do not appear above: both IDs were retired when their original seed framing
  was superseded by confirmed findings under different numbers (DEV-SEED-001 → DEV-008;
  DEV-SEED-003 → DEV-009) during WP-C0.2, to avoid two IDs describing the same issue.

Count **as of the WP-C4.7 close-out (2026-07-21)**, and not maintained since: 88 numbered
deviations total (DEV-002 through DEV-090, DEV-001/DEV-003 retired). For the current total see
`COMPILER-STATE.md`'s "Known deviations — open index".
DEV-090 (by-value iteration over a non-`Copy` array element) was split from DEV-086's narrowed
remainder during the WP-C4.7 close-out and rejected in the front end (`E0104`), the feature itself
deferred. DEV-089 (user `Display` dispatch through `print`/`println`) was closed the same day by
implementing dispatch in both engines (CD-041), which was the last item gating Gate C4 closure.
DEV-074 (HIR oracle slice-bound messages folded into the "out of bounds" family) was made during
WP-C4.6 A4-2e, recorded then only in the A1 amendment doc, and numbered retroactively by
WP-C4.7-1 as **closed at creation** — the code is correct and shipped; the gap was governance.
DEV-069 (front end + HIR interpreter not multi-file-span-clean: cross-file spans read against
the entry file, breaking cross-file methods/literals/field reads) was found during WP-C4.5f-3c's
multi-file lowering work — the MIR lowering itself was already multi-file-clean via
`ProgramMeta` — and was **closed by WP-C4.7-4** (2026-07-20), removing CD-033's C5 multi-file
prerequisite.
DEV-068 (user `impl Copy` structs always-Move in MIR lowering, rejecting valid programs at
MIR verification) was surfaced by the external C4.5c-head review, confirmed empirically, and
closed the same day in WP-C4.5e-0 (CD-030).
DEV-061/062/063 (the function-value cluster: indirect calls not executable, fn values not Copy in
borrowck, Option/Result combinators missing) were found 2026-07-19 during Gate C4 entry by
executing CD-021 workload items 16-22 against the interpreter for the first time — exactly the
early-surfacing those workload items were frozen to provide — and **closed the same day** in the
owner-approved pre-C4.1 correction pass (CD-027). DEV-064 (undetermined-generic fn coercion not
rejected, a TYPE-FN-002 conformance gap) was found during the same pass and **closed by
WP-C4.5c** (E0004 rejection in the checker). DEV-067 (bounded generic parameters lose their
bounds at intra-generic call sites and behind reference receivers, an over-rejection) was found
while writing WP-C4.5c's differential tests, confirmed pre-existing, and remains open, owned by
a later C4.5 increment.
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
through `self`, fixed in `typecheck/body.rs`'s `resolve_method`; DEV-052: qualified `Trait::method(...)`
syntax didn't resolve for compiler `CoreTrait`s, fixed via a new `Res::CoreTraitMember` in
`resolve.rs`/`typecheck/`/`interp.rs`; DEV-055: glob-imported unit enum variants didn't resolve
at all, fixed in `resolve.rs`). DEV-053 and DEV-054 were also found there, investigated as a
dedicated follow-up in the same original session, found to share one root cause (a bare `None`
pattern never matched by value -- it silently acted as an unconditional wildcard, confirmed to
produce **wrong runtime output**, not merely a spurious diagnostic -- DEV-053's original
"tuple-pattern usefulness/exhaustiveness" framing was itself a downstream artifact of this same
misclassification, not a separate algorithm bug), and **closed** with a real fix in
`resolve.rs`/`typecheck/`. DEV-060 (repeated call to an un-overridden trait default method
wrongly flagged as a move) was found while writing DEV-051's regression tests, confirmed
pre-existing and unrelated to that fix (via `git stash`), and was **closed during C3-ENTRY
closure** (2026-07-19) with a real fix in `borrowck.rs`'s `method_receiver`.
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
**Currently open (2026-07-21, at Gate C4 closure):** DEV-005 (unowned), DEV-010
(WP-C8.2/C8.3), DEV-011 (unscheduled), DEV-012 (WP-C8.7), DEV-017 (partial, unscheduled
remainder), DEV-083 (**owner-deferred** to `WP-C6.x Method Resolution Completion`; explicitly
outside the mandatory C5 lowering baseline), DEV-088 use-site (cross-file `const` USE; declaration
fixed, USE now rejected deterministically in the checker and deferred to the front-end/multi-file
completion package with DEV-083), and DEV-090 (by-value non-`Copy` array iteration; rejected in the
front end and deferred to a later language-completion package). All are over-rejections with both
engines consistent — each rejects at a single deterministic front-end point, none diverges between
engines. **DEV-089 is now CLOSED** (user `Display` dispatch implemented in both engines, CD-041);
**DEV-086 is CLOSED** for patterns and its narrowed remainder is tracked as DEV-090. DEV-087 (oracle treated a slice reference as non-`Copy`) was found while
writing the corpus cases and closed by WP-C4.7-9. DEV-076 was CLOSED by WP-C4.7-8.1a (the oracle half); the MIR half landed in WP-C4.7-8.1. DEV-077 (the same family, in
`Box::into_inner`) was found and CLOSED by WP-C4.7-6.1; DEV-078 (unsuffixed integer literals
never adopting an expected integer type) was closed by WP-C4.7-6.3; DEV-075 (Char/Bool ordering)
was closed by the WP-C4.7 DEV-075 increment under an owner specification decision, which also
added `PRIM-TRAIT-001` to the normative spec. **DEV-079 and DEV-080** were found and closed by
WP-C4.7-8.3 — a verifier false-positive that rejected every multi-droppable-field enum variant,
and the arm-end drop-order divergence it had been masking. **DEV-081** (shorthand struct-field
bindings never dropped — a silent leak in both the flat and general match paths) was found and
closed by WP-C4.7-8.3b. **DEV-082** (a method call on a slice receiver consumed it, so a
`&mut [T]` local could not be used twice — invisible until exclusive views existed) was found and
closed by WP-C4.7-8.6. DEV-070 was closed by
WP-C4.6 A2 in both engines; DEV-074 (WP-C4.7-1) is closed at creation; **DEV-069 was closed by
WP-C4.7-4**, which also removes CD-033's C5 multi-file prerequisite; **DEV-072 and DEV-073 were
closed by WP-C4.7-5** (move-out-of-borrow via match bindings; generic-impl bound matching);
**DEV-067 and DEV-071 were closed by WP-C4.7-7** (bounded-parameter bounds behind references and
at intra-generic call sites; `Ordering` exhaustiveness). DEV-060 closed the same day it was made a C3-ENTRY blocker.
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

---

## DEV-120 — native call-depth exhaustion is a bounded host limitation (OPEN, documented; WP-C7.9 Packet F)

- **Normative expectation:** `LIMIT-RESOURCE-001` — "Allocation, address-space, stack, call-depth,
  file-descriptor, stream, and other host-resource exhaustion are host/process failures unless an
  API returns a specified `Result`. Implementations must prevent host undefined behavior and report
  the classified failure **when the host permits**; exact capacities are implementation/target-
  defined."
- **Current behaviour:** the two INTERPRETERS honour the whole rule. Both check
  `interp::MAX_CALL_DEPTH` before pushing a frame and report a classified host/resource failure
  (`FailureClass::HostResource` / `MirRunError::HostResource`), so a runaway recursion ends as a
  reported outcome with a stable non-trap exit status and never as a process abort. **Native
  execution does not**: a generated binary recurses on the host's own stack, and stack exhaustion
  there terminates the process by signal, before any STARK-level code can observe it.
- **Why it is not fixed here:** reporting it natively means per-call depth instrumentation in every
  generated function — a cost paid by every program to report a condition that is already
  host-defined, and one that still could not cover host stack growth from the runtime or a provider.
  Owner ruling D4 for WP-C7.9: record the boundary rather than instrument the backend.
- **Why this is conformant, not a divergence:** the rule's own qualifier is "when the host permits".
  A signalled stack overflow is the host declining to permit it. The capacities are also explicitly
  implementation-defined, so the interpreters' 512-frame capacity and a native binary's stack-shaped
  capacity are not required to match, and **no claim is made that they do**.
- **User impact:** a program that recurses without a base case is reported cleanly under
  `stark run`, and terminates by signal when built natively. The three-engine claim is unaffected:
  resource exhaustion is not a language outcome and is excluded from engine comparison by
  construction — the comparator refuses to normalise it into one.
- **Evidence:** `starkc/tests/resource_exhaustion.rs` (subprocess cases: below the limit completes,
  above it is classified, mutual recursion is caught, the counter is restored after an error, and
  no test process aborts).
- **Owning gate:** none scheduled. Revisit only if native execution acquires a reason to bound call
  depth for its own sake.

---

## DEV-118 — `HashMap`/`HashSet` key bounds were unenforced (CLOSED by WP-C7.9 Packet I)

- **Normative expectation:** `06-Standard-Library.md` declares `HashMap<K: Hash + Eq, V>` and
  `HashSet<T: Hash + Eq>`. A type used as a key must satisfy both.
- **Behaviour while open:** neither bound was checked, in any engine. A `HashMap<Float64, Int32>`
  type-checked, lowered, ran, and produced answers.
- **Why it survived a three-engine differential:** because it was not a differential defect. All
  three engines shared the omission — the storage scans by `Eq` and never consults a hash, so every
  engine accepted the same invalid programs and agreed on their results. Agreement proves
  consistency, not conformance; this entry is the reason the two are now distinguished in the
  comparator's expectations (WP-C7.9 G.4).
- **The fuse:** the omission becomes a live cross-engine divergence the moment ONE implementation
  starts using the hash — a real hash table in the native runtime, for instance — in programs that
  had compiled cleanly until then.
- **Fix:** the bound is enforced at TYPE INSTANTIATION, through a general mechanism for
  implementation-declared generic bounds (`typecheck/`: `builtin_type_bounds` /
  `check_builtin_type_bounds`), not at `insert`. `HashMap<Float64, Int32>` is therefore ill-typed
  wherever it is written, including in a signature that is never called. Rejection is `E0500`, the
  same code every other unsatisfied bound uses.
- **Evidence:** `starkc/tests/adversarial_hash_bounds.rs` — `Eq` without `Hash`, `Hash` without
  `Eq`, float keys, an uncalled signature, a generic function with insufficient bounds, and a
  nested position, all rejected; primitives, a user nominal with both impls, a generic function
  with both bounds, and the unconstrained VALUE position, all accepted.
- **Owning gate:** was WP-C6.3 (carried at C6 closure); closed here.

## DEV-121 — engine value representation diverges from the normalized type (OPEN; instance fixed CD-305, class open)

- **Normative expectation:** `03-Type-System.md` — a shared reference (`&T`, `&[T]`) is `Copy`.
  Passing one to a function copies it; the source binding stays live. The governing rule this
  entry establishes: **after expression typing, Copy/move behaviour — and the runtime
  representation that carries it — is determined exclusively by the normalized semantic type,
  never by the expression that produced the value.** It binds the checker, MIR lowering, the
  native backend, and **each interpreter's value model** equally.
- **Current behaviour:** `String::bytes()` is declared `&[UInt8]` but the HIR interpreter returned
  `Value::Vec` — an owned, non-`Copy` runtime value. Passing it therefore MOVED it, emptying the
  caller's local, and any later use trapped with "use of unavailable value". The checker accepted
  the program and MIR emitted `copy` for both call operands; the HIR engine alone was wrong.
  `bytes()` shared an implementation arm with `into_bytes()` (`Vec<UInt8>`, genuinely owned), so
  one arm served two types with opposite ownership.
- **Layer, established not assumed:** the emitted MIR for the reproducer is
  `_9 = call use_len@[](copy _4)` / `_11 = call use_len@[](copy _4)` — `copy`, both calls. The
  defect was below MIR, in the interpreter's value model. An earlier framing of this defect as
  "emitted as a consuming call operand" was wrong and is corrected here.
- **Classification:** CORPUS-GAP, not oracle-blind. The HIR engine diverged from MIR and native, so
  differential testing could have caught it given matrix-shaped input; no input exercised the shape.
  The sentinel matrix is therefore a permanent corpus obligation, not one-time DEV evidence.
- **User impact:** a valid program is accepted and then traps at run time, with a message naming
  neither the moved value nor (before CD-306) the right file. Found by `stark-mime`, `stark-query`
  and `stark-form`, whose consumers all failed; `stark-percent` passed only because it indexes its
  view rather than passing it.
- **Security/soundness impact:** no memory unsafety — the interpreter detects the empty slot and
  traps. The soundness cost is to the type system's contract: two runtime representations claimed
  one static type and only one obeyed its ownership rule.
- **Workaround:** none required, and none permitted in package code — copying a view into an owned
  `Vec`, or inlining a helper to avoid a call, hides the defect the packages exist to expose.
- **Instance fixed:** CD-305 — `bytes()` now materialises a temp place and returns
  `Value::Slice`; `into_bytes()` keeps `Value::Vec`. Six-case three-engine regression, including a
  move-semantics control proving owned values still move (E0100).
- **Why the class stays open:** other producers are unaudited. Every intrinsic whose declared
  return is `&T`/`&[T]` must be checked against its runtime representation, and the invariant that
  would have caught this on the first execution (INV-VALUE-REP-001) does not exist yet.
- **Sibling, not dual:** `P1-COMPILER-001` / `DEFECT-C788-LOOP-TEMP` (DEV-123) shares the
  accepted-but-traps SYMPTOM class and nothing else — different engine, different layer, different
  mechanism (piecewise-emptied storage vs. wrong value kind). Recorded so the two are not conflated.
- **Proposed disposition:** WP-COPY-CANON Phases 1–3 — producer audit from the method registry,
  per-engine canonicalization, and INV-VALUE-REP-001.
- **Owning gate:** WP-COPY-CANON.

## DEV-122 — span source-identity gap (OPEN, guarded; instance fixed CD-306)

- **Normative expectation:** a diagnostic or trap location identifies the source it belongs to. A
  span offset is meaningless without the identity of the file it indexes.
- **Current behaviour:** spans carry no source identity of their own. Rendering selects a file by
  convention and indexes it, so a span from one file resolved against another produces a location
  in the wrong file. `SourceFile::line_col` CLAMPS an out-of-range offset to end-of-file, so the
  result is not a visible failure but a well-formed, plausible, wrong location.
- **Observed:** a runtime fault inside `stark-mime` reported at `stark-mime-consumer/src/main.stark:31:1`
  — line 31 of a 21-line file, in the wrong package. Second instance of the class after CD-302,
  where the test runner sliced a dependency's span against the root file and panicked.
- **User impact:** measurable and demonstrated. On DEV-121 the wrong file sent the investigation to
  the wrong shape entirely: the span pointed into the consumer, so the first characterisation
  described the consumer's use of a match binding, and a reproducer built from that description
  passed. The real fault was three call frames away.
- **Security/soundness impact:** none direct. The cost is diagnostic trust — a location that is
  confidently wrong is worse than one that is absent, because a reader cannot tell them apart.
- **Instance fixed:** CD-306 — runtime rendering now uses the file DEV-113-B already stamps on the
  error, plus a backstop that refuses to locate a span past end-of-file.
- **Why it stays open:** the backstop checks only `span.lo > src.len()`. It does not check
  `start <= end`, nor that the resolved column lies within the resolved line, and compile-time and
  runtime rendering remain SEPARATE paths — so a future caller can reintroduce the fault on either.
- **Proposed disposition:** WP-COPY-CANON Phase 4 — one checked `resolve_span` used by both paths,
  never panicking and never falling back to the root source. The platform correction (mandatory
  `SourceId` on every span, resolution total by construction) is filed as a separate future WP; the
  interim guard must not be mistaken for it.
- **Owning gate:** WP-COPY-CANON Phase 4, then the SourceId WP.

## DEV-123 — `P1-COMPILER-001` / `DEFECT-C788-LOOP-TEMP`: repeated enum-result assignment retained a live generated slot (CLOSED by MIR A12, CD-265/CD-269)

- **Registered for findability, not as a new finding.** This defect was governed and discharged
  before this entry existed; it was recorded only inside `WP-C7-P1-REST-REPORT.md`, so the ledger
  could not answer a query about it. `P1-COMPILER-001` is a LOCAL LABEL used by the P1 workload
  report; `DEFECT-C788-LOOP-TEMP` is the same defect.
- **Normative expectation:** storage for a place emptied piecewise must end when its units are
  accounted for.
- **Root cause:** **any** place whose storage is emptied piecewise — not temporaries specifically.
- **Trail:** recorded CD-263; ruled a non-blocking C7 deviation CD-264; fixed by MIR amendment A12
  (`Statement::StorageDead`, MIR `0.2` → `0.3`) at CD-265, approved retrospectively as CE3; a
  surviving `?`-in-a-loop instance — the shape A12's sixteen-shape matrix missed, because
  `lower_try` builds its own scrutinee temporary — found by `stark-json` and fixed at CD-269.
- **Regression:** `starkc/tests/a12_storage_end_shapes.rs`.
- **Relationship to DEV-121:** sibling in SYMPTOM class only (accepted-but-traps). Different engine,
  layer and mechanism; not a shared root cause and not a dual.
- **Owning gate:** closed under C7; full argument in `mir-amendment-A12-storage-end.md`.

## DEV-124 — iterator desugar moves a `Copy` loop variable (CLOSED; found by INV-MOVE-001)

- **Normative expectation:** a `Copy` value is read with `copy`. Moving one empties the source
  place, which is a claim the type contradicts.
- **Current behaviour:** `lower_for_over_iter` binds the loop variable with
  `value: &T = move nxt.v1.0` — its own doc comment says so — moving the payload out of the
  `Option<&T>` that `*IterNext` returned. Four instantiations observed, one lowering site:

  ```
  MIR-0036  move from a place of Copy type Ref { mutable: false, inner: Int32 }   for x in v.iter()
  MIR-0036  move from a place of Copy type Ref { mutable: false, inner: String }  non-Copy element
  MIR-0036  move from a place of Copy type Char                                   s.chars()
  MIR-0036  move from a place of Copy type Int                                    user Iterator impl
  ```
- **How it was found:** INV-MOVE-001 (WP-COPY-CANON Phase 3). It had been in the tree indefinitely
  and no test could see it, because nothing asked whether an operand's move was licensed by its
  type. This is the invariant doing the job it was added for, on its first run.
- **User impact:** none observed. The `Option` temp is reassigned on every iteration, so nothing
  reads the emptied place before it is overwritten — the move is unobservable rather than harmless
  by design. That is exactly the condition under which a latent defect survives: correct by
  accident of scheduling.
- **Security/soundness impact:** none today. The hazard is that the emptied place is only safe
  while no path reads it between the move and the next assignment; any future change to the
  desugar's block structure could introduce one, and nothing would flag it.
- **Why it was reported before being fixed:** WP-COPY-CANON's Phase 3 rule — "report any firings as
  new DEVs; no silent drive-by fixes". The invariant was written, held back, and registered here
  first; the repair landed as its own change.
- **Resolution:** two hand-built `Operand::Move`s in `lower.rs` — the `&T` reference form
  (`lower_for_over_iter`) and the by-value `Item` form — now read through `read_place`, which
  selects the operand from the payload's type.

  **The fix is not "write `copy`", and that distinction is the substance of this entry.** The
  proposed disposition above said `copy`, and it was wrong: a user `Iterator` may yield a non-`Copy`
  `Item`, where `move` is correct and required. Replacing one hardcoded operand with the other
  would have been the same defect facing the other way, and INV-MOVE-001 would not have caught it —
  the invariant only rejects unlicensed moves, so a wrongly-`Copy`ed non-`Copy` payload would have
  passed. What was actually wrong was that the desugar had an *opinion* about the operand at all.
  A third hand-built `Move` in `lower_vec_clear_droppable` was examined and left alone: it runs only
  for a droppable element type, and `Copy + Drop` is forbidden, so its move is always licensed.
- **Consequence for the invariant:** INV-MOVE-001 (MIR-0036) landed in the same change, once no
  program tripped it. Because nothing in the corpus can now reach it, three hand-built MIR fixtures
  in `copy_canon_matrix.rs` keep it honest: a `Move` of a `Copy` place must be rejected, the `Copy`
  form of the same body must verify, and a `Move` of a non-`Copy` place must verify. An invariant
  no test can trip is indistinguishable from `if false`.
- **Second property now enforced:** Copy-ness is decided twice — `LowerCtx::is_copy` picks the
  operand, `TypeContext::is_copy` checks it — over different eligibility sets, and nothing made them
  agree. Drift between them now surfaces as MIR-0036 on a real program instead of as divergence
  between engines. Weaker than unifying the two predicates, which remains worth doing.
- **Owning gate:** WP-COPY-CANON Phase 3.

## DEV-125 — three more hand-built `Move`s on `Copy` places (CLOSED; found by INV-MOVE-001)

- **Normative expectation:** as DEV-124 — the operand follows the place's type.
- **Sites, all on the provider/`Result` path, all found by MIR-0036 on real workloads rather than
  by the unit corpus:

  | Site | Copy type moved | Reached by |
  | --- | --- | --- |
  | `lower.rs` provider status→`Result` binding | `enum#13`, `enum#14` (fieldless provider error enums) | C7 P1 REST workload, C7.8 native |
  | `assign_provider_ok` multi-slot tuple | `(Bool, UInt64)` | `var_len` in the REST workload |
  | `lower_try` — the `?` desugar's `Err` payload | `enum#2` | `c788_lifecycle_e2e::question_mark_propagation_closes_a_live_resource` |

- **Why the unit corpus missed all three:** every one needs a `Result<T, E>` whose `E` is a
  *fieldless* enum, which is what makes it `Copy`. The in-tree tests use `Result` with payload-
  carrying or non-`Copy` errors, so `move` was always licensed there. The provider path produces
  fieldless error enums by construction, so it fires on essentially every provider call.
- **Aggravating detail in `lower_try`:** its `storage_end_after` closure already branches on
  `is_copy` of the very type whose operand was hardcoded `move`, to pick the A12 storage-end
  reason. The distinction was present in the function and the operand ignored it.
- **Resolution:** all three read through `read_place`. Same fix as DEV-124 and, as there, *not*
  "write `copy`" — a non-`Copy` `E` must still move.
- **Process note:** these should have been caught before INV-MOVE-001 landed. The invariant was
  pushed on the strength of the lib suite and four iterator tests; the provider workloads and the
  conformance corpus were left to CI, and both failed. The invariant was right; the local evidence
  was too narrow for a change that constrains every lowering site in the compiler.
- **Owning gate:** WP-COPY-CANON Phase 3.

## DEV-126 — `as_str` returned a detached copy, so a view of it had no owner (CLOSED)

- **Normative expectation:** `as_str` produces a `&str` — a BORROW of the receiver. A value derived
  from it is a view of the receiver's storage and lives as long as that storage.
- **Current behaviour (before this entry):** the HIR interpreter's `as_str` returned
  `Value::Str(string.clone())`, an owned copy with no link to the place it came from. Nothing
  downstream could recover the owner.
- **The symptom, and why it looked like a `bytes()` defect:**

  ```stark
  fn direct(c: &C)     -> &[UInt8] { c.input.bytes() }          // worked
  fn via_as_str(c: &C) -> &[UInt8] { c.input.as_str().bytes() } // "dangling reference"
  ```

  Identical types, identical declared lifetimes, different provenance. CD-305 made `bytes()`
  materialise its bytes into a promoted temp; CD-308 anchored that temp to the RECEIVER's frame so
  the view survives being returned. Correct — but in the chained form the receiver is `as_str`'s
  detached copy, which `expr_place`'s fallback had already promoted into the RUNNING frame. So the
  bytes were anchored to the frame that was about to pop.
- **How it was found:** CI, not the corpus. `stark-json` failed 9/10 on all three platforms with
  "dangling reference"; its hot helper is `cursor.input.as_str().bytes()`. WP-COPY-CANON's matrix
  has both `str::bytes (via as_str)` and an escaping `function returning a reference` producer, but
  never their CROSS — as_str-then-bytes was only exercised locally, and escape was only exercised
  with a direct `bytes()`. The failing cell is the one the matrix does not contain.
- **Resolution:** `as_str` returns `Value::Ref(receiver_place)`. `deref_place`/`deref_value` already
  normalise through it, so a chained call resolves back to the `String`'s own place and `bytes()`
  anchors to the real owner.
- **Consequence:** `s.as_str()` now reaches builtins and core methods as a `Value::Ref`, and
  `string_arg` — a free function with no `&self`, so no way to follow a place — rejected those.
  `flatten_string_refs` derefs a reference argument WHEN ITS REFERENT IS A STRING. The condition is
  the referent's kind, not the callee's name: the pre-existing `remove`/`contains_key`/`contains`
  special case keyed on names and so only ever covered the three that had been reported, while
  every string-taking entry point has the same requirement. It cannot disturb a `&mut Vec`/`&mut
  HashMap` argument, whose referent is not a string.
- **Matrix obligation (open):** the matrix should carry producer×producer chaining, not only
  producer×use-mode. Filed as follow-up work; this entry is the motivating program.
- **Owning gate:** WP-COPY-CANON Phase 2.

## DEV-127 — `borrow_set_receiver` moved a `&HashSet<T>` (CLOSED; found by INV-MOVE-001)

- **Site:** `borrow_set_receiver` returned a hand-built `Operand::Move` of its `&HashSet<T>` temp.
  A shared reference is `Copy`, so the move contradicts the type. Only the shared spellings fired;
  `&mut` is not `Copy`.
- **The tell:** `borrow_map_receiver`, its sibling three lines below, already read through
  `read_place`. One of a matched pair diverged and nothing compared them.
- **Reached by:** the whole DEV-116 HashSet corpus (6 C6 cases), `collection_iteration_order_agrees`,
  `exhausted_set_iter_then_remove`, and the HashSet identity/ordering differentials.
- **Resolution:** `read_place`, matching the sibling.
- **Two test fixtures retyped, and why this is not test weakening:** `mir_verify`'s
  `partial_move_of_one_field_leaves_sibling_readable` and
  `dev117_drop_elaboration_moves_are_exempt_but_user_moves_are_not` hand-build MIR that moves
  `Int32` locals. `Int32` was incidental filler in both — the subjects are V-MOVE-1 field precision
  and MIR-0007/DEV-117's exemption — but under INV-MOVE-001 an `Int32` move is invalid MIR on its
  own account, so both fixtures failed for a reason neither test concerns. The fields/locals are now
  `&mut Int32`: non-`Copy`, no drop glue, and what a partial move actually looks like in lowered
  code. Every assertion is unchanged. (`Constant::Str` was tried first and rejected: it types as
  `&str`, a *shared* reference, hence `Copy` again.) The weakening that was NOT done is exempting
  `Copy` moves in the invariant.
- **Owning gate:** WP-COPY-CANON Phase 3.

## DEV-128 — the `Copy` rule was written twice, and the test guarding that was fictional (CLOSED)

- **Normative expectation:** one rule, one implementation. Copy-ness decides operand selection, drop
  glue, slot backing and duplication licence; two implementations of it can disagree, and every
  consumer is entitled to believe whichever it asked.
- **Current behaviour (before this entry):** `TypeContext::is_copy` and `mir::lower::LowerCtx::is_copy`
  were byte-identical matches differing in exactly one lookup — `copy_eligible_items` versus
  `meta.copy_eligible`, where `lower_program` fills the first FROM the second.
- **Cost, measured rather than asserted:**
  - `HostResource` was corrected **five separate times** across this family, each fix landing in one
    copy of the rule at a time (the trail is in the comments on both functions).
  - CD-240 fixed the `_ => true` wildcard defect in one copy and left the other; the surviving
    wildcard is what made `read_place` emit `Operand::Copy` for a host resource, so a program could
    hold two handles to one resource.
  - DEV-125 and DEV-127 were operand decisions taken against the producer's predicate and rejected
    by the consumer's — INV-MOVE-001 surfaced them as MIR-0036 on real programs.
- **The fictional guard.** `TypeContext::is_copy`'s doc comment named
  `lowered_copy_classification_matches_the_type_context` as the test keeping the two in step. **That
  test does not exist.** The only occurrence of the name anywhere in the tree was the claim itself.
  The same comment's stated rationale for the split was also stale: it says lowering answers the
  nominal case from the HIR via `type_has_copy_impl`, while the code reads a precomputed set.
- **Resolution:** the structural rule lives once, in `mir::mir_ty_is_copy`, with the nominal case
  supplied as a predicate. The two SETS remain separate because they are read at different times;
  only the rule is shared, which is the part that was drifting. Agreement is now structural rather
  than asserted — a stronger guarantee than the missing test would have provided.
- **Second half — `operand_move_inventory`:** INV-MOVE-001 catches a wrong operand only when a
  PROGRAM reaches the site, which is how DEV-124/125/127 surfaced one CI round at a time. The new
  test pins all eleven `Operand::Move` occurrences in `lower.rs` with a stated reason, so a new one
  fails at authoring time. Rows match trimmed source text (stable under line-number churn, not under
  rewording); CRLF is normalised at the read so a Windows checkout does not fail every row.
- **Owning gate:** WP-COPY-CANON Phase 3.

## DEV-129 — string literal patterns compared representation, not content (CLOSED)

- **Normative expectation:** a string literal pattern on a `&str` scrutinee compares by CONTENT
  (StrEq), never structurally. `match s.as_str() { "beta" => .. }` is the canonical form.
- **What broke, and it was mine.** DEV-126 made `as_str` return `Value::Ref(receiver_place)` instead
  of a `Value::Str` clone. Two things then went wrong at once in the HIR interpreter's
  `PatKind::Lit` arm, which compared `eval_lit(..) == *value`:
  1. the scrutinee was a `Value::Ref` and was never dereferenced;
  2. dereferencing it yields `Value::String` (the owning local), while a literal evaluates to
     `Value::Str` — the same text in two wrappers, which `==` on `Value` distinguishes.
  The second was latent all along and hidden by the first: the comparison only ever worked because
  `as_str` happened to hand back the same wrapper the literal used.
- **Symptom:** every arm missed and the match fell through to `_`. `a2_str_pat.stark` printed 0 in
  the oracle and 2 under MIR. **Silently** — no arm is "wrong" to fail, so nothing could report it;
  the three-engine differential is the only thing that could have caught it, and did.
- **Resolution:** the literal arm dereferences the scrutinee, and compares via `string_text`, which
  reads through either wrapper. The same treatment is applied to the const-pattern sub-case, since a
  const pattern is a literal pattern with a name. Deliberately NOT applied to the variant arms: a
  reference-typed enum scrutinee is a type error (PAT-BIND-001, CD-303), and quietly accepting one
  there would re-open it.
- **Wider point:** DEV-126 was a representation change to a value model that several unrelated
  comparisons had been reading structurally. This is the second consequence of it (after
  `flatten_string_refs` for builtin arguments), and the class — "who else compares `Value` by
  variant identity where content is meant?" — is worth an audit rather than waiting for the next
  differential to find one.
- **Owning gate:** WP-COPY-CANON Phase 2.

## DEV-130 — structural equality was written once and omitted three times (CLOSED)

- **Normative expectation:** `&str` and `String` compare by content (`06-Standard-Library.md`).
  `Value` derives `PartialEq`, so `Str("a") != String("a")` — a representation difference the
  language does not have.
- **Found by:** the value-comparison audit DEV-129 called for, run rather than deferred.
- **The finding.** The `Str`/`String` pairing existed inline at exactly ONE site, the `==` operator.
  Three others compared raw:

  | Site | Had the pairing |
  | --- | --- |
  | `==` / `!=` operator | yes, inline |
  | `assert_eq` / `assert_ne` | **no** |
  | `language_equal` (backs `Vec::contains` and friends) | **no** |
  | literal / const patterns | no — closed separately as DEV-129 |

  So `s.as_str() == "beta"` was true while `assert_eq(s.as_str(), "beta")` failed, reporting
  `left: beta` / `right: beta`. Two values that print identically, declared unequal.
- **Resolution:** `values_equal` is the single structural comparison all five sites route through —
  the same correction DEV-128 made for `is_copy`. It recurses into containers deliberately
  (`Some(s.as_str())` against `Some("x")` compares payloads, and a flat rule fails it for the same
  reason). It does NOT follow `Value::Ref`: callers deref first, because following a place needs
  `&self` and because a caller that has not deref'd has a bug this function should not hide.
- **Probed and found clean:** `HashMap::get`/`contains_key`/`remove` and `HashSet::contains` by
  reference, for `String`, `Int32` and user-struct keys with `Eq` + `Hash` impls.
- **Owning gate:** WP-COPY-CANON Phase 2.

## DEV-131 — the string-ref flattening was too broad and broke `take` (CLOSED)

- **What broke:** `take(&mut a)` failed with "take expects mutable reference".
- **Cause, and it was mine.** DEV-126 flattened every reference-to-string argument on the way into
  `call_builtin`. `take` needs the REFERENCE, not the text, and got the text. A blanket rule cannot
  distinguish "reads the string" from "needs the place", because `Value::Ref` does not record which
  the caller meant.
- **Note on how this happened:** DEV-126's entry criticised the pre-existing
  `remove`/`contains_key`/`contains` deref for keying on callee NAMES, then replaced it with a rule
  keyed on referent kind — which is better for the sites that read content and no better for the
  sites that do not. The defect was over-reach, not under-reach.
- **Resolution:** the deref moved to the five sites that call `string_arg`, which demonstrably want
  text. Anything needing a place is untouched by construction rather than by exemption.
- **Found by:** `gate4a_prelude_traits`, a suite not run when DEV-126 landed.
- **Owning gate:** WP-COPY-CANON Phase 2.

## DEV-121 — UPDATE 2 (CD-340): a second producer found, and the invariant's blind spot named

- **New instance: `SplitIter`'s item.** Declared `&str`, represented `Value::String` (owned), so
  its first use consumed it. Registered and repaired as DEV-138; the class remains OPEN. That is
  now TWO producers found by user-facing programs rather than by the invariant — `String::bytes()`
  (CD-305) and `String::split()` (CD-340).
- **The invariant's blind spot, stated so the next instance is found by tooling instead:**
  INV-VALUE-REP-001 checks **`let` bindings**. A `for`-loop binding is not a `let`, so no loop
  item is covered by it at all. Both known instances were reachable through a loop item. Extending
  the invariant to loop bindings — and to call arguments, which are equally uncovered — is the
  work that would close the class rather than another instance. Unowned.

## DEV-121 — UPDATE: narrowed by INV-VALUE-REP-001, class NOT closed

- **What is now enforced.** INV-VALUE-REP-001 checks at every `let` that a binding declared `&[T]`
  or `&str` does not hold owned `Value::Vec`/`Value::String` storage. That is precisely the
  direction DEV-121 broke: `let view = owner.bytes()` had `&[UInt8]` in the type tables and owned
  storage at runtime, so passing it moved it and emptied the caller's binding — on a program the
  checker and MIR both accepted, with correct MIR.
- **A premise in the original entry was wrong.** DEV-121 said the class-closer was blocked because
  the HIR interpreter is "largely untyped at runtime". `Interpreter` already holds
  `tables: &TypeTables`, with both `expr_types` and `local_types`. It has the declared type at every
  `let` and simply never consulted it. The invariant cost far less than the entry implied.
- **Why it is narrow, deliberately.** It asserts one direction of one pairing, not a total
  type→representation mapping — because the oracle's model is not total. `&Int32` may legitimately
  arrive as the bare scalar through auto-deref, and `Value::Str`/`Value::String` both carry text
  where one type is declared (DEV-130 had to make comparison representation-insensitive for exactly
  that reason). A broad rule would fire on correct programs and need exemptions, and an invariant
  with exemptions is advisory. A narrow rule that always means something was the trade taken.
- **Status: NARROWED, not class-closed.** The residual exposure is named rather than implied: `&T`
  for scalar `T`, and the `Str`/`String` duality. Those are the two pairings DEV-129, DEV-130 and
  DEV-131 came out of, so the class is live, not theoretical.
- **Deferred by owner direction** to `WP-VALUE-REP-TOTAL.md`, filed with the ambiguities enumerated
  and the likely finding recorded: the mapping probably cannot be made total without first changing
  the oracle's value model, which is a larger change than the check it enables.
- **Owning gate:** WP-COPY-CANON Phase 2 (narrow); WP-VALUE-REP-TOTAL (remainder).

## DEV-132 — borrowed Vec index projection lowered as a by-value element read (CLOSED)

- **Classification:** borrowed Vec index projection lowered as a by-value element read, incorrectly
  requiring `Copy` for `&v[i].field`.
- **Normative expectation:** the four forms are distinct and must lower distinctly.

  ```text
  v[i].field       value read; may require Copy or move rules
  &v[i].field      shared borrow; must NOT read the element by value
  &mut v[i].field  mutable borrow; separate capability and aliasing question
  v[i].field = x   mutation; must not become admitted through shared borrowing
  ```

- **Current behaviour:** `&v[i].field` on `Vec<NonCopy>` materialises the whole element into a temp
  via `VecIndexGet` — a BY-VALUE read — and then projects the field off it. `VecIndexGet` requires a
  `Copy` element (V-COPY-1), because reading a non-`Copy` element by value would move it out of the
  Vec. So MIR-0016 is CORRECT for the MIR that was emitted; the emission is the defect. Nothing was
  ever going to be moved: a borrow does not need the element by value.

  ```text
  emitted:  VecIndexGet -> element by value -> V-COPY-1 requires Copy   (refused)
  required: VecGetRef   -> shared reference -> Deref place -> field projection -> borrow
  ```

- **Engine divergence:** the checker ACCEPTS it and the HIR oracle EXECUTES it correctly; MIR
  refuses to lower it. An over-refusal, not unsoundness — accepted-but-unbuildable.
- **This is not a MIR feature addition.** `RuntimeFn::VecGetRef` already exists, is described in
  `mir/mod.rs` as "an interior borrow into the live Vec", carries a verified signature
  `(&Vec<T>, u64) -> Option<&T>` with **no Copy requirement**, and is already what `v.get(i)`
  lowers to. It is verified and supported by the native backend. The defect is the failure to
  preserve PLACE CONTEXT through indexing, not a missing primitive.
- **Found by:** extending `qualify-first-party-packages.py` to the five HTTP-substrate packages
  (CD-326). Nothing had ever built them natively. `stark-mime`'s `media_type_parameter` uses
  `&media_type.parameters[i].name`, which is an ordinary valid borrow.
- **No package workaround was introduced.** `v.get(i)` would compile, and rewriting the package to
  use it would conceal a valid source shape behind a compiler defect.
- **Owning gate:** CD-326 (package qualification), repaired under its own CD.

## DEV-133 — array-to-slice unsizing is accepted but not lowered (CLOSED)

- **Classification:** an array-to-slice coercion at a declared `&[T]` binding is accepted by the
  checker and executed by the HIR oracle, but MIR lowering never performs the unsizing, so
  verification rejects the assignment.
- **Minimal reproducer:**

  ```stark
  fn takes(s: &[UInt8]) -> UInt64 { s.len() }

  fn main() {
      let b: UInt8 = 7u8;
      let slice: &[UInt8] = &[b];   // accepted; oracle prints 1
      println(takes(slice));
  }
  ```

  ```text
  MIR-0004 main@[] bb0: assignment: expected Ref { mutable: false, inner: Slice(UInt8) },
                        found Ref { mutable: false, inner: Array(UInt8, 1) }
  ```

- **Engine divergence:** checker ACCEPTS, HIR oracle EXECUTES correctly (prints `1`), MIR refuses.
  Accepted-but-unbuildable — the same CLASS as DEV-132, an entirely different mechanism. DEV-132 was
  a failure to preserve place context through indexing; this is a missing coercion at an assignment
  whose declared type differs from the rvalue's by unsizing alone.
- **Found by:** the ten-package qualification run added under CD-326/CD-328. `stark-form`'s
  `form_encode_string` writes `let slice: &[UInt8] = &[b];` to percent-encode one byte — an ordinary
  valid construct.
- **Not caused by DEV-132's repair.** That change touched only `lower_index_place`'s `Vec` arm; this
  reproducer contains no indexing. Confirmed by reproducing it standalone.
- **Blocking:** `stark-form`'s native build, and therefore the addition of all five HTTP-substrate
  packages to CI qualification as one change. Four of the five build; the ruling on DEV-132 was
  explicit that adding the passing subset while knowingly excluding one would institutionalise an
  avoidable gap, and that reasoning applies here unchanged.
- **No package workaround introduced.** Rewriting `stark-form` to avoid the coercion would conceal a
  valid source shape behind a compiler defect.
- **Resolution (CD-329):** the coercion is emitted in `weaken_ref_to` — already the function that
  coerces an operand to an expected reference type (it does `&mut T` -> `&T`). All six coercion
  sites route through it (`let`, call argument, receiver, return, return-expression, assignment
  RHS), so fixing it there covers every position at once; a new hook would have needed adding at
  each, and whichever was forgotten would have kept the defect. `SliceNew` already accepts an
  `&[T; N]` receiver, so no new `RuntimeFn`, no new `MirTy`, no amendment.
- **Negative controls, because the risk is BROADENING the coercion rather than under-applying it:**
  a mismatched element type must not coerce (that would reinterpret memory, not merely forget a
  length), and a shared array must not become a `&mut` slice (coercion changes shape, never
  capability). Both pinned.
- **Owning gate:** CD-326 (package qualification); repaired under CD-329.

## DEV-134 — `?` neither converts the error type nor requires that a conversion exist [CLOSED, WP-DEV-134-139 Part A, CD-335, 2026-08-02]

- **Normative expectation:** `?` propagates `Err`/`None` early (`04-Semantic-Analysis.md` line 160,
  `CORE-V1-ABSTRACT-MACHINE.md` EXEC-CFLOW-001). The spec does not define a `From` conversion at the
  propagation site, so the only two defensible behaviours are: **reject** a `?` whose error type
  differs from the enclosing function's, or **convert** it and require the `From` impl. The compiler
  does neither.
- **Current behaviour:** the error types are not compared at all. A `Result<T, Low>` propagates
  through a function returning `Result<T, High>` with **no `impl From<Low> for High` anywhere in the
  program**, and the `Low` value is carried out typed as `High`.
- **Minimal reproducer:**

  ```stark
  enum Low { Bad }
  enum High { Other }

  fn low() -> Result<Int32, Low> { Err(Low::Bad) }
  fn viaq() -> Result<Int32, High> { let value = low()?; Ok(value) }

  fn main() {
      match viaq() {
          Ok(value) => println(value),
          Err(error) => match error { High::Other => println("other") },
      }
  }
  ```

  ```text
  starkc check -> OK
  starkc run   -> runtime error: non-exhaustive match reached
  ```

- **Engine behaviour:** front end ACCEPTS; HIR oracle produces a value whose variant tag belongs to
  a different enum, so the inner `match` — which IS exhaustive over `High` — falls through. The
  "non-exhaustive match reached" message is therefore a symptom, not the defect.
- **Security/soundness impact:** **soundness.** This is type confusion, not a diagnostic gap: a
  value of one nominal type is observable at another nominal type. Both enums here are fieldless, so
  the reproducer only mis-tags; with payloads of differing layout the consequence is worse, and that
  variant has not been characterised. Anything downstream that trusts the static type of an `Err`
  payload — a `Display` impl, a field read, a further `match` — is reading the wrong type.
- **Workaround:** use `?` only where the error types are already equal, and convert across error
  types with an explicit `match` plus `From::from`.
- **Proposed disposition:** decide between reject and convert — an owner call, not an
  implementation detail, because "convert" adds a `From` obligation the spec has not scoped and is a
  CE-shaped semantic decision. Rejection is the conservative half and can land first.
- **WIDER THAN FILED, same mechanism.** The repair work found that the CONSTRUCTOR is unrelated
  too: `Option<_>?` inside a function returning `Result<_, _>` — and the reverse — is equally
  accepted, and equally produces a value whose variant tag belongs to a different enum. This is
  not a second defect: the `Try` arm asked "is the return type `?`-capable?" and "is the operand
  `?`-capable?" as two INDEPENDENT questions and never related them, so one missing relation
  produced both symptoms. One mechanism, one repair, no new DEV number (WP-DEV-134-139 §2).
- **Resolution (CD-335):** `check_try_compatibility` in `typecheck/`, recorded during body
  checking and drained after inference settles — the same deferral `display_checks` uses, and for
  the same reason: the operand's error type is routinely an inference variable while the body is
  being checked, so an eager comparison would either reject valid code or force a premature
  binding. The rule is EXACT compatibility: same constructor, and for `Result` an error type
  equal under the compiler's canonical equivalence.
- **The ruling, recorded because it is a language decision and not an implementation detail:**

  ```text
  `?` requires exact error-type compatibility.
  Implicit From-based propagation is not part of this repair.
  ```

  `03-Type-System.md` does not scope a conversion at the propagation site, so applying `From`
  would be new semantics rather than a repair. Rejection is the conservative half. An
  `impl From<Low> for High` being present does NOT license the propagation, and
  `from_impl_present_still_rejected` pins that so the absence of conversion cannot later be
  mistaken for an oversight. Whether Core v1 should gain conversion at `?` remains an OPEN
  language-design question and needs its own proposal; it is not tracked by this entry.
- **Diagnostic:** E0006, the existing `?` code, widened rather than a new code allocated. The
  spec's E0006 line is amended in the same change from "`?` operator in a function that does not
  return `Result` or `Option`" to cover the whole return-type contract. Reusing the code keeps
  the normative table stable and keeps one code per concept; the two conditions are distinguished
  by message, and `non_result_return_reports_once_not_twice` pins that the pre-existing condition
  still reports exactly once rather than twice.
- **Negative controls, because the risk here is OVER-rejection.** A `?` check that is too eager
  breaks every correct propagation in the provider layer and the ten first-party packages, so the
  must-pass set is larger than the must-reject set: different SUCCESS types (legal — `?` relates
  the error position only), identical generic error types, an error type that is the function's
  own type parameter, chained `?`, and `String` as a `Ty::Core` error type.
- **Latent gap found and deliberately NOT repaired:** `types_equal` has no `Ty::Param` arm, so two
  occurrences of the same type parameter compare unequal. Its existing callers are coherence and
  overlap paths where `Ty::Param` is either pre-handled or where a conservative `false` is safe,
  so the gap has no demonstrated symptom there — but it made the first version of this repair
  reject `fn f<E>() -> Result<_, E>` propagating into `fn g<E>() -> Result<_, E>`, caught by this
  entry's own negative control. Rather than widen a shared coherence primitive for a defect with
  no symptom of its own, the structural walk now takes the `Ty::Param` behaviour as a PARAMETER
  (`types_equal_inner`), written once and reached by two entry points. Widening `types_equal`
  itself is a separate question and is unowned; it gets a DEV number if a symptom is found.
- **Evidence:** `starkc/tests/dev134_try_error_type.rs`, 16 cases. Ten first-party packages
  qualify. The external task-shaped suite is 34/34 unchanged.
- **Owning gate:** WP-DEV-134-139 Part A (CD-335).

## DEV-135 — moves of individual struct fields are not tracked [CLOSED, WP-DEV-134-139 Part B, CD-338, 2026-08-02]

- **Normative expectation:** `04-Semantic-Analysis.md` defines partial moves — moving one field of
  a struct leaves the other fields readable and the moved field unusable. A second move of the same
  field must be rejected `E0100`.
- **Current behaviour:** flow analysis tracks moves of a whole binding but not of its fields. The
  same field can be moved out twice; the front end accepts it and the HIR oracle discovers it at run
  time.
- **Minimal reproducer:**

  ```stark
  struct Handle { label: String }
  impl Drop for Handle { fn drop(&mut self) { println(self.label.as_str()); } }
  struct Owner { handle: Handle }

  fn main() {
      let owner = Owner { handle: Handle { label: String::from("only-one") } };
      let first = owner.handle;
      let second = owner.handle;
      println("both bindings exist");
  }
  ```

  ```text
  starkc check -> OK
  starkc run   -> internal compiler error: use of moved or invalid field
  ```

- **Engine behaviour:** front end ACCEPTS; oracle refuses. Note the message class — this surfaces as
  an **internal compiler error**, which is the wrong category for a user-authored program and will
  read as a compiler crash to anyone who hits it.
- **Security/soundness impact:** **soundness, bounded by the oracle's own check.** The oracle
  catches it, so no double-drop is observable there; what is NOT established is whether the native
  backend catches it, and a double-move of a droppable field is a double-free shape. That
  characterisation is owed before the severity can be settled.
- **Workaround:** none needed in practice — the construct is a genuine error; the gap is that it is
  diagnosed late and in the wrong category.
- **INVENTORY (WP-DEV-134-139 §5.3, run 2026-08-02 before choosing a stage-one repair).** The
  question the inventory had to settle is whether "parent poisoning" — marking the whole parent
  unavailable once any non-`Copy` field moves — is an acceptable bounded limitation. **It is not.**
  Sibling-after-partial-move is load-bearing at every layer, and is asserted as REQUIRED behaviour
  by tests that predate this programme:

  | Where | What it pins |
  | --- | --- |
  | `tests/gate2-valid/18_partial_moves.stark` | conformance fixture: `consume(pair.left); consume(pair.right);` |
  | `tests/gate2_valid.rs` `..._without_its_own_drop_impl_is_accepted` | front end must ACCEPT the partial move |
  | `tests/mir_verify.rs` `partial_move_of_one_field_leaves_sibling_readable` | V-MOVE-1 field precision in the verifier |
  | `tests/mir_differential.rs` `conditional_partial_moves_and_loop_scopes_agree` | conditional partial moves + drop flags |
  | `tests/three_engine_differential.rs` | the "partial-move survivor" drop case |
  | `tests/native_c5_3_aggregates_enums.rs` `a_field_move_does_not_kill_its_siblings` | native: "under a whole-local approximation the sibling read would find a dead slot and abort" |
  | `tests/c6-corpus/templates.py` T14 | generated corpus: "partial move and reinitialisation" |

  The native test's own doc comment is an explicit, pre-existing rejection of the whole-local
  approximation at the MIR and native layers. Poisoning would not be a bounded limitation; it
  would contradict the conformance fixture set and four differential suites.
- **Scan method, so the result can be re-derived:** 511 `.stark` sources outside `target/`, plus
  inline STARK sources in `starkc/tests/*.rs` and `starkc/src/*.rs`. Sixteen `let x = y.field;`
  sites appear in first-party packages (`stark-json`, `stark-random`) and ALL sixteen read `Copy`
  scalar fields (`UInt64` offsets, line/column counters, PRNG state) — copies, not moves, so no
  first-party package depends on partial-move behaviour either way. Every genuine partial move in
  the tree is in the compiler's own test corpus, and every one of those requires sibling survival.
- **CONSEQUENCE — the two-stage model collapses to one stage.** WP §5.4 is explicit: "If the
  inventory shows sibling use is load-bearing, do not land poisoning silently. Proceed to
  DEV-135b." So **DEV-135a is NOT the release-gating repair; DEV-135b is.** The release gate in
  WP §15 resolves to its second branch: "DEV-135b is complete because inventory proved parent
  poisoning unacceptable." This is a planned outcome of the inventory, not a scope change.
- **CORRECTION to the disposition recorded above.** That paragraph said "the gap is in the front
  end's `moved_places`, which is keyed on whole locals". **That was wrong**, and it is corrected
  here rather than quietly rewritten because it drove the estimate. `moved_places` is a
  `HashSet<Place>`, `Place` already carries `projections`, and `places_overlap` already does
  prefix matching — the front end was ALREADY field-precise. Moving `pair.left` already left
  `pair.right` live, and moving the parent afterwards was already refused.
- **Actual root cause: field IDENTITY, one enum variant wide.** `place_of` built
  `Projection::Field(name.lo, name.hi)` — the SPAN the field name was written at. Two mentions of
  one field sit at different byte offsets, so `owner.handle` on one line and `owner.handle` on the
  next were two DIFFERENT projections, which `places_overlap` then correctly reported as disjoint.
  Nothing about the move model was missing; the comparison could never succeed.
- **Resolution (CD-338):** `Projection::Field(String)` / `Projection::TupleField(String)`, holding
  the resolved name. Read via `self.text`, which is correct here because every expression reaching
  `place_of` belongs to the item being checked and `self.file` tracks that item (DEV-069). Same
  class as DEV-122: identity taken from a span rather than from what the span denotes.
- **The two-stage model therefore never had to be entered.** WP §5.2 split this into a
  conservative "DEV-135a parent poisoning" gate and a "DEV-135b precision" follow-on, and the
  inventory ruled poisoning out. But the precision the follow-on was meant to BUILD already
  existed, so the repair is neither stage: it is a one-variant identity fix that makes the
  existing precision reachable. **No DEV-135b is filed, and none is owed** — sibling survival,
  nested paths, parent/child ordering, and exactly-once drop are all covered by the tests below,
  which is what DEV-135b's closure criteria asked for.
- **Evidence:** `starkc/tests/dev135_field_move_paths.rs`, 16 cases (6 reject, 10 accept). The
  accepts are the important half here, because the inventory established that sibling survival is
  load-bearing: `moving_one_field_leaves_its_sibling_usable`,
  `moving_a_nested_field_leaves_its_nested_sibling_usable`,
  `sibling_fields_are_each_destroyed_exactly_once` (executes, asserts exactly-once drop), plus
  `partial_move_out_of_a_drop_type_is_still_rejected` to pin that the pre-existing Drop rule is
  untouched. Two cases compose this repair with DEV-136: a field moved on a TERMINATING branch
  does not poison the join, and on a REACHABLE branch it still does.
- **Owning gate:** WP-DEV-134-139 Part B (CD-338).

## DEV-136 — a move on a returning path is treated as unconditional [CLOSED, WP-DEV-134-139 Part D, CD-337, 2026-08-02]

- **Normative expectation:** definite-assignment and move analysis are path-sensitive. A move that
  occurs only on a path that `return`s cannot affect the fall-through path, which that move never
  reaches.
- **Current behaviour:** the move is recorded unconditionally, so every later use of the binding is
  rejected `E0100`.
- **Minimal reproducer:**

  ```stark
  fn build(flag: Bool) -> String {
      let mut out = String::new();
      if flag { return out; }
      out.push('a');
      out
  }
  ```

  ```text
  E0100 use of moved value 'out'   (at `out.push('a')` and at the trailing `out`)
  ```

- **User impact:** high nuisance value, because the shape is idiomatic. "Build a buffer, bail early
  with what you have, otherwise keep filling it" is refused outright. Every early return must
  instead construct a fresh value, which is both slower and misleading to read.
- **Security/soundness impact:** none — a false positive. It rejects valid programs; it does not
  admit invalid ones.
- **Workaround:** return a freshly constructed value from the early-return path rather than the
  accumulator.
- **Layer:** `borrowck.rs`. The `If` arm unioned the then-branch's move set into the post-state
  unconditionally, and the `Match` arm extended the merged set from EVERY arm. Neither asked
  whether the branch reaches the join.
- **Resolution (CD-337):** `block_diverges`/`expr_diverges` decide whether a branch reaches the
  join, and only reaching branches contribute. Divergence is taken from two sources, both already
  authoritative: a `Return`/`Break`/`Continue` statement anywhere in the block's statement
  sequence, and the type checker's own `Ty::Never` for `panic(..)` and any call returning `!`.
  Reusing `Ty::Never` keeps one authority for "does this diverge" rather than re-deriving it from
  syntax. Composite forms recurse: an `if` diverges only when both sides do, a `match` only when
  every arm does.
- **THE DIRECTION OF CONSERVATISM IS THE SAFETY ARGUMENT.** The predicate answers "does this
  definitely NOT reach the join?". A wrong `true` would drop a real move from the join and accept
  a use-after-move — unsound. A wrong `false` merely preserves the old false positive. So every
  arm reports `true` only on evidence and anything unrecognised falls through to `false`;
  `loop` without a reachable `break` is deliberately NOT treated as diverging, because judging it
  needs reachability analysis the checker does not have.
- **Two merge subtleties that are easy to get wrong, both pinned by tests:**
  1. `if` with no `else` and a terminating branch restores the state from BEFORE the `if`, not
     the branch's state — reaching that point proves the branch did not run.
  2. a `match` whose arms ALL diverge would leave the merged set empty, which would silently
     resurrect a value moved BEFORE the `match`. The empty case falls back to the pre-match
     state (`a_move_before_an_all_diverging_match_is_still_rejected`).
- **Negative controls:** a move on a reachable branch, on one of two reachable branches, in a
  reachable `match` arm, and a move placed BEFORE a terminating branch — the last pins that the
  repair excludes a terminating branch's OWN moves, not moves that merely precede one.
- **Drop obligations, not just diagnostics:** `a_droppable_value_survives_a_terminating_branch`
  executes both paths and asserts each `Guard` is destroyed exactly once.
- **Evidence:** `starkc/tests/dev136_terminating_path_moves.rs`, 14 cases (9 accept, 5 reject).
- **Owning gate:** WP-DEV-134-139 Part D (CD-337).

## DEV-137 — a receiver auto-borrow in a `while` condition is live across the loop body [CLOSED, WP-DEV-134-139 Part C, CD-336, 2026-08-02]

- **Normative expectation:** `03-Type-System.md` "References and Lifetimes" — a temporary borrow
  ends with its statement. The auto-borrow a method call takes of its receiver (TYPE-METHOD-002) is
  a temporary; it must not outlive the condition it appears in.
- **Current behaviour:** the borrow is treated as live for the whole loop, so any mutation of the
  receiver inside the body is a conflict.
- **Minimal reproducer:**

  ```stark
  fn main() {
      let mut values: Vec<Int32> = Vec::new();
      values.push(1);
      values.push(2);
      let mut i = 0u64;
      while i < values.len() {
          values[i] = 5;
          i = i + 1u64;
      }
      println(values[0] + values[1]);
  }
  ```

  ```text
  E0101 cannot assign to variable 'values[i]' because it is borrowed
  ```

- **User impact:** this is the single most disruptive of the six in practice. `while i < v.len()` is
  the ordinary way to write an indexed loop, and every in-place algorithm — sorting, filling,
  partitioning — hits it. It also affects `&mut` PARAMETERS, where the same shape appears inside any
  mutating helper.
- **Security/soundness impact:** none — a false positive.
- **Workaround:** hoist the length (`let n = values.len();`) above the loop. **This only works when
  the length does not change**; a queue that grows while being drained must instead track its length
  by hand, which is exactly the bookkeeping the borrow checker is supposed to make unnecessary.
- **LAYER LOCATED (WP-DEV-134-139 §6.3, recorded before repair):** `borrowck.rs`, not MIR, not
  liveness, and not the back-edge. `Borrowck::active_borrows` is a stack, scoped by two mechanisms
  and only two: `check_block` truncates to its entry depth at block end, and `check_stmt` truncates
  after each expression statement. A `while` CONDITION is neither — it is an expression evaluated
  outside any statement of its own, and the `While` arm reads

  ```rust
  hir::ExprKind::While { cond, body } => {
      self.check_expr(*cond);      // pushes the receiver auto-borrow taken by `values.len()`
      self.check_block(*body);     // records borrows_before AFTER that push, so it cannot pop it
  }
  ```

  so the condition's temporaries are still on the stack when the body runs, and `check_block`'s own
  truncate restores to a depth that already includes them. Nothing ever pops them until the
  enclosing statement ends. The `Assign` arm then finds the borrow in `active_borrows` and reports
  E0101.
- **Why this also explains the `&mut` PARAMETER case:** it is the same code path. Nothing about the
  receiver being a parameter matters; what matters is that `len()` was called in the condition.
- **Not the `For` arm.** `for x in &v` must KEEP its iterator borrow alive across the body, so the
  repair must not be generalised to loop headers as a category — only to `while` conditions, whose
  value is consumed by the branch and cannot outlive it.
- **WIDER THAN FILED, same mechanism.** `if` conditions had the identical defect, and it was found
  by this entry's own must-pass case for condition re-evaluation:
  `if values.len() < 5u64 { values.push(1); }` inside a loop body was refused for exactly the same
  reason. A condition is a condition whether it guards a loop or a branch. One mechanism, one
  repair, no new DEV number.
- **Resolution (CD-336):** `Borrowck::check_condition` — snapshot the borrow depth, check the
  condition, truncate back. Used by the `While` and `If` arms. The rule is written ONCE rather
  than inline at both sites, for the DEV-128/DEV-130 reason.
- **Scope boundary, and why it is not "loop and branch headers" as a category.** `match`
  scrutinees and `for` iterators are deliberately NOT routed through `check_condition`.
  PAT-BIND-001 binds a non-`Copy` arm payload BY REFERENCE into the scrutinee, and `for x in &v`
  yields references into the iterated value, so in both cases the borrow must span the body;
  truncating them would hand out references to storage the checker had stopped tracking. Two
  negative controls pin this — `a_match_scrutinee_borrow_still_spans_the_arms` and
  `for_loop_iterator_borrow_still_spans_the_body` — and both would fail if a later change
  generalised the repair to every operand that precedes a block.
- **The other negative control that defines the boundary:** a borrow created BEFORE the loop
  (`let view = &values;`) lives at a shallower stack depth than the snapshot, so the truncate
  cannot reach it and a body mutation through its owner is still refused
  (`borrow_predating_the_loop_stays_live`). This is what makes the repair depth-based rather than
  "clear the borrow set at the loop header", which would have been unsound.
- **Execution evidence, not just acceptance.** `the_indexed_loop_executes_correctly` and
  `a_growing_vector_re_evaluates_its_condition` run through the HIR oracle and assert output. The
  second is the one that proves the hoist-the-length workaround was a SEMANTIC change, not a
  stylistic one: the loop grows the vector it is measuring, so a hoisted bound would stop early.
- **Evidence:** `starkc/tests/dev137_while_condition_borrows.rs`, 16 cases (12 accept, 4 reject).
- **Owning gate:** WP-DEV-134-139 Part C (CD-336).

## DEV-138 — an iterator-yielded `&str` is consumed by its first use [CLOSED as a DEV-121 INSTANCE, WP-DEV-134-139 Part F, CD-340, 2026-08-02]

- **Normative expectation:** `&str` is a shared borrow and is `Copy`-like at use sites; reading it
  does not consume it. `06-Standard-Library.md` gives `SplitIter` an `Item` that is a reference.
- **Current behaviour:** the yielded item is consumed by its first use. A second use of the same
  loop variable in the same iteration fails at run time, and the front end does not catch it.
- **Minimal reproducer:**

  ```stark
  fn main() {
      for word in "alpha beta".split(" ") {
          let first = String::from(word);
          let second = String::from(word);
          println(first.as_str());
          println(second.as_str());
      }
  }
  ```

  ```text
  starkc check -> OK
  starkc run   -> runtime error: use of unavailable value
  ```

- **Engine behaviour:** front end ACCEPTS; oracle refuses on the second use. Same accepted-but-
  unexecutable CLASS as DEV-132/DEV-133, different mechanism — this is a value-representation/
  ownership question about iterator items, so it is plausibly an instance of the still-open
  **DEV-121** class rather than an independent defect. That relationship is asserted as a hypothesis
  here, not established; INV-VALUE-REP-001 is the instrument that would settle it.
- **Security/soundness impact:** **soundness-adjacent.** A shared borrow is being treated as an
  owned value, which is the same category error DEV-121 tracks. No memory unsafety is demonstrated;
  the failure is a refusal, not a corruption.
- **Workaround:** convert the item once (`let key = String::from(word);`) and `clone` from there.
- **CLASSIFICATION RESULT (WP-DEV-134-139 §9.2/§9.3): it IS a DEV-121 instance.** The matrix that
  established it:

  ```text
  declared item type   &str            06-Standard-Library.md: SplitIter / String::split / &str
  HIR runtime value    Value::String   OWNED  <- the defect
  value_is_copy        Value::Str -> true, Value::String -> false
  front end            ACCEPTS (sees a Copy shared reference)
  MIR / native         VACUOUS - both refuse SplitIter outright (C4.5)
  ```

  The MIR and native rows are **vacuous rather than confirming** and are recorded that way: those
  engines do not implement `SplitIter`, so they could not have disagreed. §9.3's "treat as
  distinct" criteria require MIR to emit `Move` for a Copy shared-reference item AND all engines
  to consume it; neither holds. §9.3's fold criteria hold on every testable dimension.
- **Producer-specific, which is the DEV-121 signature.** Six shapes were probed: `&Vec<String>`,
  `&Vec<Int32>`, `chars()`, and a plain `&str` outside a loop were already correct. Only `split`
  was wrong — and `trim`/`substring`, which have the SAME declared return type, already yielded
  `Value::Str`. The repair makes `split` consistent with its siblings rather than adding a rule.
- **Resolution (CD-340):** one line in `interp.rs` — `Value::SplitIter`'s `next` yields
  `Value::Str` rather than `Value::String`. No new `RuntimeFn`, no new `Value` variant, no
  amendment. DEV-121's governing rule verbatim: representation follows the normalized semantic
  type, never the producing expression.
- **RESIDUAL EXPOSURE, recorded against DEV-121 rather than here.** INV-VALUE-REP-001 checks at
  every `let` that a binding declared `&str`/`&[T]` does not hold owned storage. A **for-loop
  binding is not a `let`**, which is exactly why the invariant did not catch this. Extending it to
  loop bindings would have caught this class at the producer rather than at a user program; that
  extension is unowned. The tests below are the interim guard.
- **Evidence:** `starkc/tests/dev138_iterator_item_representation.rs`, 10 cases. Four exercise
  `split` reuse; four pin producers that were ALREADY correct so the fix cannot regress what it
  was not about; two pin that the item is still only a view — the source string is undisturbed by
  iteration, and string-literal pattern matching still compares by content (DEV-129).
- **Owning gate:** WP-DEV-134-139 Part F (CD-340), as a DEV-121 instance. **DEV-121's class stays
  OPEN.**

## DEV-139 — impl-level generic bounds are invisible to operator desugaring [CLOSED, WP-DEV-134-139 Part E, CD-339, 2026-08-02]

- **Normative expectation:** `03-Type-System.md` "Operators and Traits" — `<` on a generic parameter
  desugars to `Ord`. A bound written on the impl head is in scope throughout the impl's method
  bodies; DEV-073 and the WP-C4.7-5 work established exactly this for method resolution.
- **Current behaviour:** `ty_satisfies_operator_bound`'s `Ty::Param` arm consults
  `self.current_fn_generics` — the enclosing FUNCTION's parameters — only. An impl-level bound is
  not in that set, so the operator check fails.
- **Minimal reproducer:**

  ```stark
  struct Pair<T> { a: T, b: T }

  impl<T: Ord> Pair<T> {
      fn larger(&self) -> &T {
          if self.a > self.b { &self.a } else { &self.b }
      }
  }
  ```

  ```text
  E0500 type 'T' does not satisfy operator trait 'Ord'
  ```

- **The contrast that localises it:** the same comparison under the same bound is ACCEPTED as a free
  function — `fn largest<T: Ord>(a: T, b: T) -> T { if a > b { a } else { b } }`. So this is not
  about `Ord` on a parameter; it is about which generic environment the operator check reads.
- **User impact:** any generic container that orders its own elements must move that comparison out
  into a free function. Ordinary API shapes — `Heap<T: Ord>`, `SortedVec<T: Ord>`, a `max` method —
  cannot be written as methods.
- **Security/soundness impact:** none — a false positive.
- **Workaround:** put the comparison in a free generic function with the bound on the function.
- **WIDER THAN FILED: it was TWO lookups, and the second one was deferred.** The title names the
  operator path, but `satisfies_bound` — ordinary trait-bound satisfaction — had the identical gap,
  each keeping its own copy of the parameter lookup and each consulting `current_fn_generics`
  alone. They agreed only by coincidence. Worse, the trait-bound half is DEFERRED: DEV-067(a)
  records the "generic environment this obligation was recorded in" and replays it at drain time,
  and that capture was also `current_fn_generics` alone, so an obligation raised inside
  `impl<T: Ord> Pair<T>` replayed against half its environment and failed even after the operator
  half was fixed. Two of this entry's own tests caught that second half.
- **Resolution (CD-339):** two helpers, each written ONCE.
  `param_declares_bound(param, required)` answers "does this parameter declare this bound?" over
  the combined environment, and both lookups call it. `current_generic_env()` returns that
  combined list for the deferred capture, so the drain needs no second field to restore. Writing
  each once is deliberate — DEV-128 and DEV-130 are both "the rule was written twice and the
  copies drifted", and this was already two copies.
- **Nothing new was brought into scope.** WP-C6.2b-F5 had already installed impl-head generics in
  `current_impl_generics` for method bodies; the bound lookups simply never asked. The repair is a
  read, not a new binding — which is why it cannot change which names are in scope, only which
  declared bounds are found.
- **Negative controls, because WIDENING an environment risks discharging obligations that were
  never declared:** an operator with no bound at all, `Eq` where `Ord` is required, `Ord` where
  `Num` is required, a bound sitting on a DIFFERENT type parameter (pins that the lookup still
  matches on parameter NAME rather than finding any bound in scope), an unbounded method-level
  parameter, and an undischarged callee obligation.
- **Relationship to DEV-083.** Not the same defect and not closed by this. DEV-083 is about
  matching a CONCRETE position in an impl head against an unresolved receiver type argument —
  impl-head *matching*. This was impl-head *bounds being read*. DEV-083 remains OPEN.
- **Evidence:** `starkc/tests/dev139_impl_generic_bounds.rs`, 16 cases (10 accept, 6 reject),
  covering `Ord`/`Eq`/`Num` operators, trait-bound obligations, inherent and trait impls, impl and
  method bounds contributing together, and nested generic nominals.
- **Owning gate:** WP-DEV-134-139 Part E (CD-339).


## DEV-140 — `Vec::` method outside the implemented lowering set (layer defect) [OPEN, registered CD-342, 2026-08-02]

- **Classification:** a REACHABLE lowering refusal — the front end accepts the program, the HIR
  oracle runs it, and MIR lowering refuses. Accepted-but-unbuildable, the E0105 class.
- **Probe:** `L7153` in `starkc/tests/layer_audit.rs`. Reproducer shape: `v.insert(0u64, 2)` after a `push`.
- **Why it is reachable:** MIR lowering implements a subset of `Vec`'s methods; the rest are `unsupported(...)` sites.
- **Not repaired by WP-DEV-134-139.** This entry exists so the finding is REGISTERED rather than
  merely observed: since CD-342 the layer audit is an enforcing gate that fails on any
  UNREGISTERED finding, and equally when a registered one stops reproducing. Six such refusals
  were found and printed by CD-331 and had carried no deviation number since.
- **Disposition:** unscheduled. Two repair shapes exist and the choice is per-site, not global —
  raise the refusal into semantic analysis (as E0105 did) or teach lowering the construct (as
  DEV-132 and DEV-133 did). CD-294 is the precedent for why raising is not always cheap: E0106 was
  reverted because `v[i]` appears in value AND place positions that only later phases distinguish.
- **Owning gate:** unassigned.

## DEV-141 — `HashMap` over a user-`Drop` value type (layer defect) [OPEN, registered CD-342, 2026-08-02]

- **Classification:** a REACHABLE lowering refusal — the front end accepts the program, the HIR
  oracle runs it, and MIR lowering refuses. Accepted-but-unbuildable, the E0105 class.
- **Probe:** `L8093` in `starkc/tests/layer_audit.rs`. Reproducer shape: `HashMap<Int32, D>` where `D` implements `Drop`.
- **Why it is reachable:** Lowering has no drop elaboration for map values whose type carries a destructor.
- **Not repaired by WP-DEV-134-139.** This entry exists so the finding is REGISTERED rather than
  merely observed: since CD-342 the layer audit is an enforcing gate that fails on any
  UNREGISTERED finding, and equally when a registered one stops reproducing. Six such refusals
  were found and printed by CD-331 and had carried no deviation number since.
- **Disposition:** unscheduled. Two repair shapes exist and the choice is per-site, not global —
  raise the refusal into semantic analysis (as E0105 did) or teach lowering the construct (as
  DEV-132 and DEV-133 did). CD-294 is the precedent for why raising is not always cheap: E0106 was
  reverted because `v[i]` appears in value AND place positions that only later phases distinguish.
- **Owning gate:** unassigned.

## DEV-142 — droppable composite carrying a borrowed element (layer defect) [OPEN, registered CD-342, 2026-08-02]

- **Classification:** a REACHABLE lowering refusal — the front end accepts the program, the HIR
  oracle runs it, and MIR lowering refuses. Accepted-but-unbuildable, the E0105 class.
- **Probe:** `L9130` in `starkc/tests/layer_audit.rs`. Reproducer shape: `(String, &str)` printed as a tuple.
- **Why it is reachable:** A composite that mixes an owned droppable and a borrow has no lowering for its drop plan.
- **Not repaired by WP-DEV-134-139.** This entry exists so the finding is REGISTERED rather than
  merely observed: since CD-342 the layer audit is an enforcing gate that fails on any
  UNREGISTERED finding, and equally when a registered one stops reproducing. Six such refusals
  were found and printed by CD-331 and had carried no deviation number since.
- **Disposition:** unscheduled. Two repair shapes exist and the choice is per-site, not global —
  raise the refusal into semantic analysis (as E0105 did) or teach lowering the construct (as
  DEV-132 and DEV-133 did). CD-294 is the precedent for why raising is not always cheap: E0106 was
  reverted because `v[i]` appears in value AND place positions that only later phases distinguish.
- **Owning gate:** unassigned.

## DEV-143 — `assert_eq` on a user-defined type (layer defect) [OPEN, registered CD-342, 2026-08-02]

- **Classification:** a REACHABLE lowering refusal — the front end accepts the program, the HIR
  oracle runs it, and MIR lowering refuses. Accepted-but-unbuildable, the E0105 class.
- **Probe:** `L5346` in `starkc/tests/layer_audit.rs`. Reproducer shape: `assert_eq(x, y)` for a struct with a user `impl Eq`.
- **Why it is reachable:** The assert builtins lower only for the compiler-known comparable types.
- **Not repaired by WP-DEV-134-139.** This entry exists so the finding is REGISTERED rather than
  merely observed: since CD-342 the layer audit is an enforcing gate that fails on any
  UNREGISTERED finding, and equally when a registered one stops reproducing. Six such refusals
  were found and printed by CD-331 and had carried no deviation number since.
- **Disposition:** unscheduled. Two repair shapes exist and the choice is per-site, not global —
  raise the refusal into semantic analysis (as E0105 did) or teach lowering the construct (as
  DEV-132 and DEV-133 did). CD-294 is the precedent for why raising is not always cheap: E0106 was
  reverted because `v[i]` appears in value AND place positions that only later phases distinguish.
- **Owning gate:** unassigned.

## DEV-144 — `for` over a non-range, non-`Vec` iterator (layer defect) [OPEN, registered CD-342, 2026-08-02]

- **Classification:** a REACHABLE lowering refusal — the front end accepts the program, the HIR
  oracle runs it, and MIR lowering refuses. Accepted-but-unbuildable, the E0105 class.
- **Probe:** `L3698` in `starkc/tests/layer_audit.rs`. Reproducer shape: `for` driving an iterator that is neither a range nor a `Vec` cursor.
- **Why it is reachable:** Lowering implements the range and `Vec` cursors; other iterables reach an unsupported site.
- **Not repaired by WP-DEV-134-139.** This entry exists so the finding is REGISTERED rather than
  merely observed: since CD-342 the layer audit is an enforcing gate that fails on any
  UNREGISTERED finding, and equally when a registered one stops reproducing. Six such refusals
  were found and printed by CD-331 and had carried no deviation number since.
- **Disposition:** unscheduled. Two repair shapes exist and the choice is per-site, not global —
  raise the refusal into semantic analysis (as E0105 did) or teach lowering the construct (as
  DEV-132 and DEV-133 did). CD-294 is the precedent for why raising is not always cheap: E0106 was
  reverted because `v[i]` appears in value AND place positions that only later phases distinguish.
- **Owning gate:** unassigned.

## DEV-145 — method on a peeled type outside the implemented slice (layer defect) [OPEN, registered CD-342, 2026-08-02]

- **Classification:** a REACHABLE lowering refusal — the front end accepts the program, the HIR
  oracle runs it, and MIR lowering refuses. Accepted-but-unbuildable, the E0105 class.
- **Probe:** `L6450` in `starkc/tests/layer_audit.rs`. Reproducer shape: a method call whose receiver auto-derefs to a type lowering does not carry.
- **Why it is reachable:** TYPE-METHOD-002 peels references repeatedly; lowering implements a narrower set than the checker accepts.
- **Not repaired by WP-DEV-134-139.** This entry exists so the finding is REGISTERED rather than
  merely observed: since CD-342 the layer audit is an enforcing gate that fails on any
  UNREGISTERED finding, and equally when a registered one stops reproducing. Six such refusals
  were found and printed by CD-331 and had carried no deviation number since.
- **Disposition:** unscheduled. Two repair shapes exist and the choice is per-site, not global —
  raise the refusal into semantic analysis (as E0105 did) or teach lowering the construct (as
  DEV-132 and DEV-133 did). CD-294 is the precedent for why raising is not always cheap: E0106 was
  reverted because `v[i]` appears in value AND place positions that only later phases distinguish.
- **Owning gate:** unassigned.

## DEV-146 — `&mut` to `&` weakening is not applied to a `HostResource` [CLOSED, CD-346, 2026-08-02]

- **Normative expectation:** an exclusive borrow weakens to a shared one at a call site.
  `weaken_ref_to` is the single function that performs this for every coercion position (DEV-133's
  repair routed all six through it), so a `&mut T` argument may satisfy a `&T` parameter.
- **Current behaviour:** the weakening is not applied when `T` is a `HostResource`. The front end
  ACCEPTS the call; MIR verification then rejects it:

  ```text
  MIR-0005 stark_net::write@[] bb0: call argument:
    expected Ref { mutable: false, inner: HostResource(.. resource: "tcp_stream") },
    found    Ref { mutable: true,  inner: HostResource(.. resource: "tcp_stream") }
  ```

- **Minimal shape:** a package wrapper that takes `&mut` over a bound resource and forwards it to
  the derived raw binding, whose `AbiParam::HandleBorrowed` derives a SHARED borrow:

  ```stark
  pub fn write(stream: &mut TcpStream, input: &[UInt8]) -> Result<UInt64, NetworkError> {
      match tcp_stream_write_raw(stream, input) { .. }   // expects &TcpStream
  }
  ```

- **Engine divergence:** checker ACCEPTS, MIR verification refuses. Accepted-but-unbuildable — the
  DEV-132/DEV-133 class, a third mechanism: those were a missing place projection and a missing
  unsize coercion; this is a missing MUTABILITY weakening, and only for host resources.
- **Diagnostic quality is part of this defect.** `stark build` reports only
  `error: internal compiler error: generated MIR failed verification` with no code, no location and
  no detail. The MIR-0005 line above appears only under `--verbose`. A verifier rejection of
  generated MIR is a compiler bug by definition, but the user still needs to know WHERE.
- **User impact:** a package cannot give a resource-mutating operation the `&mut` signature it
  deserves. `stark-net`'s `read`/`write`/`write_all` take `&TcpStream` for this reason alone, which
  understates what they do — recorded in the source at their definition.
- **Security/soundness impact:** none — it refuses a valid program rather than accepting an invalid
  one. The cost is expressiveness and a misleading API shape.
- **Workaround:** take a shared borrow. In force in `stark-net` today.
- **ACTUAL LAYER, which the proposed disposition had wrong.** `weaken_ref_to` was never the
  problem: its mutability arm is type-agnostic and would have handled `HostResource` fine. The
  defect was that **provider calls never reached it**. The `HandleBorrowed` arm of
  `lower_provider_call` pushed its operand with no expected-type coercion at all. DEV-133 routed
  SIX coercion sites through `weaken_ref_to` and its comment warned that "whichever site was
  forgotten would keep this defect" — provider calls were the seventh, forgotten invisibly,
  because no first-party package called a resource function until `stark-net` did.
- **Resolution (CD-346):** the `HandleBorrowed` arm derives the expected type FROM THE OPERAND — if
  what is held is `&mut X`, the borrowed-handle slot wants `&X` — and routes through
  `weaken_ref_to`. Deriving it from the operand rather than rebuilding `HostResourceTy` avoids a
  second copy of `provider_sig`'s mapping that could drift from it.
- **THE RULING (CD-346), which is the part that outlives this repair.** The ABI's derivation and
  the package's declared signature **need not match**:

  ```text
  AbiParam::HandleBorrowed   always derives a SHARED reference   (ABI fact, unchanged)
  package surface            may declare &mut, and the compiler weakens
  ```

  So the surface question is answered by SEMANTICS, not by the ABI:
  - an operation that consumes or produces bytes, or moves a cursor, takes `&mut` — a shared
    borrow would let a caller hold two readers of one stream, making byte-consumption order
    non-local and unreviewable;
  - a purely observational operation stays `&`;
  - neither choice changes what crosses the ABI.

  Settled once here rather than per package, because io v0.2 streams, signals, process handles and
  crypto keys all face it. **Caveat recorded honestly:** this ruling was made from what the ABI
  verifiably does; the CRYPTO0 convergence was not in evidence when it was written and should be
  checked against it before the first crypto package declares a surface.
- **Negative control, because the risk is weakening the WRONG way.** If `&R` could satisfy a
  `&mut R` parameter, the repair would hand out exclusive access from a shared borrow — an
  aliasing hole worse than the defect. `a_shared_borrow_does_not_satisfy_a_mutable_parameter`
  pins that it still refuses.
- **`stark-net`'s `&mut` signatures are restored** with the ruling recorded at their definition.
- **Evidence:** `starkc/tests/dev146_resource_borrow_weakening.rs` (3 cases); `stark-net` builds
  and runs `&mut` through the full provider path; end-to-end native client against a loopback
  listener — `wrote / 5 / closed`, server saw `b'PING\n'`.
- **Owning gate:** CD-346.

## DEV-147 — `&mut Vec<T>` parameter mutated in a loop is accepted but not buildable [CLOSED, CD-352, 2026-08-02]

- **Classification:** accepted-but-unbuildable. The checker accepts, the HIR oracle EXECUTES
  CORRECTLY, and MIR verification refuses. Same class as DEV-132/DEV-133/DEV-146, a fourth
  mechanism.
- **Minimal reproducer:**

  ```stark
  fn push_all(out: &mut Vec<UInt8>, text: &str) {
      let bytes = text.bytes();
      let n = bytes.len();
      let mut i = 0u64;
      while i < n {
          out.push(bytes[i]);
          i = i + 1u64;
      }
  }
  fn main() {
      let mut v: Vec<UInt8> = Vec::new();
      push_all(&mut v, "ab");
      println(v.len());
  }
  ```

  ```text
  stark check  -> OK
  stark run    -> 2          (correct)
  stark build  -> MIR-0007 push_all@[] bb6: move from possibly-moved place _1[]
  ```

- **What it blocks:** "append into a caller's buffer in a loop", which is the fundamental shape of
  every serializer, encoder and formatter. HC6 hit it immediately — `stark-http-serialize`'s
  `push_str_bytes`/`push_header_line` are exactly this, and the package tests PASSED on the oracle
  while the native consumer failed to build.
- **Why the oracle disagrees:** the receiver auto-borrow for `push` is taken from a place reached
  through a `&mut` parameter; MIR's move analysis treats the parameter place as possibly-moved
  across the loop back-edge, while the oracle re-reads the referent each iteration. The two engines
  disagree about whether a borrow through a parameter survives an iteration.
- **Workaround, in force in `stark-http-serialize`:** the helpers take an OWNED `Vec<UInt8>` and
  return it, so the accumulator is a local rather than a borrowed parameter. It costs a move per
  call and reads worse than the `&mut` form.
- **ACTUAL LAYER, and the hypothesis above was wrong.** Not DEV-137 region work, and not the
  verifier's whole-local `Deref` approximation either (that is a documented, deliberate scope note
  and is fine). The defect was in LOWERING: `borrow_{vec,string,map,set}_receiver` each took the
  same shortcut when the receiver was already a reference —

  ```rust
  if layers > 0 { return self.lower_expr_to_operand(base); }
  ```

  — and `&mut T` is not `Copy`, so "pass through" lowers to a **`Move` of the caller's reference**.
  Harmless once; on a loop back-edge the parameter is then possibly-moved.
- **Resolution (CD-352):** `reborrow_reference_receiver` builds `&mut *base`, which is exactly what
  the `layers == 0` path already does one deref further down. Written once and called from all four
  receiver borrowers, for the DEV-128/DEV-130 reason. Deliberately narrow: a SHARED reference passes
  through unchanged (`&T` is `Copy`, nothing moves), and a non-place base passes through (there is
  no caller reference to preserve).
- **Audited, not assumed:** ten sites in `lower.rs` share the `layers > 0` shortcut. Six are not
  receiver borrows — display refs, value refs, index paths — and are untouched.
- **Negative controls, because an over-eager reborrow is worse than the defect:** the owner is still
  refused while an exclusive borrow lives; two live `&mut` to one owner still refused; a shared
  borrow still cannot satisfy a `&mut` parameter; an owned value moved twice still refused.
- **`stark-http-serialize` is restored to the natural `&mut` form**, which is the end-to-end proof —
  that package is what found the defect, and its native consumer now builds.
- **Evidence:** `starkc/tests/dev147_reference_receiver_reborrow.rs`, 11 cases (7 build, 4 refuse);
  MIR/differential/three-engine/lifecycle suites 326 green; 13-package gate exit 0; external suite
  39/39; clippy on CI's 1.97 toolchain, zero diagnostics.
- **Owning gate:** CD-352.

## DEV-148 — an associated function is unresolvable across any MODULE boundary [CLOSED, fixed CD-356, 2026-08-02]

- **Normative expectation:** `07-Modules-and-Packages.md` — a `pub` item of a dependency is
  reachable from a dependent package. Nothing distinguishes an associated function from a free
  function for visibility purposes.
- **Current behaviour:** free functions and METHODS resolve across a module or package boundary;
  ASSOCIATED functions (no receiver) do not.
- **SCOPE CORRECTION (CD-355).** Filed as cross-PACKAGE; it is cross-MODULE, which is strictly
  wider and includes the package case. A submodule of the SAME package cannot call one either:

  ```stark
  // src/lib.stark
  pub struct Wrap { pub v: Int32 }
  impl Wrap { pub fn make(v: Int32) -> Wrap { Wrap { v: v } } }
  mod tests;

  // src/tests.stark
  use super::Wrap;
  let b = Wrap::make(2);          // E0200 associated function 'make' not found
  let c = super::Wrap::make(2);   // E0200 -- the fully qualified path fails too
  ```

  Same FILE resolves. This matters because it means a package cannot even TEST its own associated
  functions: `stark-url`'s `Url::parse`, `stark-mime`'s `MediaType::parse` and `stark-net`'s
  `TcpStream::connect` are unreachable from every test and consumer in the tree, and are recorded
  as `surface_blocked` in the CD-355 gate for exactly that reason.
- **Where the failure is:** NOT the resolver. `super::Wrap::make` reaches `Res::AssociatedFn` and
  then fails in `typecheck/`'s associated-function lookup (the E0200 at the `candidates` empty
  case), which scans impls for one whose `self_ty` path resolves to `Res::Item(nominal)`. Methods
  are unaffected because method lookup goes by the receiver's TYPE, not by path resolution — which
  is precisely why the two behave differently.
- **Minimal reproducer:** two packages, `xapp` depending on `xlib`.

  ```stark
  // xlib
  pub struct Wrap { pub v: Int32 }
  impl Wrap {
      pub fn make(v: Int32) -> Wrap { Wrap { v: v } }   // associated
      pub fn get(&self) -> Int32 { self.v }             // method
  }
  pub fn make_free(v: Int32) -> Wrap { Wrap { v: v } }  // free

  // xapp
  let a = make_free(1);   // OK
  println(a.get());       // OK  -- method resolves
  let b = Wrap::make(2);  // E0200 associated function 'make' not found
  ```

- **User impact: it silently shapes every package API in the tree.** `Type::new()` is the
  idiomatic constructor and is simply unavailable to a consumer, so each package must expose a free
  function instead. Every existing first-party package already does this — `stark-net`'s `ipv4`,
  `socket_address`; `stark-http-core`'s `header`, `new_header_map`; `stark-time`'s own
  `Duration::from_seconds` is the exception and is therefore UNUSABLE from a dependent package.
  The convention was adopted without anyone recording why, which is how a defect becomes a house
  style.
- **Security/soundness impact:** none — it refuses a valid program.
- **Workaround:** export a free constructor alongside any associated one. In force everywhere.
- **ROOT CAUSE (CD-356): the name was sliced out of the wrong file.** Not visibility, not
  coherence, not path resolution — the path reached `Res::AssociatedFn` correctly. `typecheck`'s
  lookup compared member names with `self.text(span)`, which slices THE FILE BEING CHECKED, while a
  member's name span belongs to the file that declared the `impl`. Instrumented, `impl Wrap`'s two
  members read back as:

  ```text
  member name_text="rap:"  has_receiver=false     // `make`'s offsets applied to the other file
  member name_text="?"     has_receiver=true      // a span past the shorter file's end
  ```

  No candidate ever matched. METHODS were unaffected because method lookup selects on the
  receiver's TYPE rather than by slicing a name — which is exactly why this looked like a language
  rule about associated functions rather than a text bug.
- **Second site:** fixing the comparison exposed the same defect one layer down. A GENERIC
  associated function then produced `type 'r' does not satisfy operator trait 'Eq'` — `'r'` being
  `T` sliced from the wrong file. The substitution map's keys and the `Ty::Param`s they substitute
  into must be read from the same file, so `foreign_sig_item` now carries the declaring item across
  the whole signature conversion. `item_text` returning `"?"` for an out-of-range span also means
  several mis-sliced parameters could COLLIDE on one key; a two-parameter test pins that.
- **The precedent this missed:** DEV-069 fixed exactly this for trait methods ("the trait's method
  names belong to the TRAIT's declaring file") and `build_assoc_projections` converts "against the
  impl's own file". The rule was already written down twice. General statement worth keeping:
  `self.text` is correct ONLY for spans from the file under check, and every lookup that reads a
  name off a foreign declaration needs `item_text`.
- **Fix:** `src/typecheck/state.rs` — `item_text` in the member comparison and the three generic-name
  map insertions, plus the `foreign_sig_item` context across the signature conversion. Evidence:
  `tests/dev148_associated_fn_across_modules.rs` (7 tests over a real two-file package graph,
  since a single-file fixture cannot reproduce a provenance bug). Vacuity-checked: reverting the
  repair turns the three cross-boundary positives RED and leaves all four controls green.
- **Still open, adjacent:** whether associated CONSTANTS and associated types have the same gap.
- **Owning gate:** package track, CD-356.

## DEV-149 — a `&self` method on a `&mut` base is neither weakened nor reborrowed [CLOSED, fixed CD-354, 2026-08-02]

- **Normative expectation:** `03-Type-System.md` "References and Lifetimes" — a `&mut T` coerces to
  `&T` at any site expecting a shared borrow, and a reborrow leaves the caller's reference intact.
- **Behaviour before the fix:** `borrow_{vec,string,map,set}_receiver` reborrowed a reference
  receiver only when the METHOD wanted `&mut`. A `&mut` base under a `&self` method was passed
  through unchanged, failing MIR verification twice at once.
- **Minimal reproducer:** eight lines, reduced from `stark-http-parser::drop_front`.

  ```stark
  fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }
  fn main() {
      let mut v: Vec<UInt8> = Vec::new();
      v.push(1u8);
      println(count(&mut v));      // check: OK, run: 1, build: refused
  }
  ```

  ```text
  MIR-0005 bb0: expected Ref { mutable: false, .. }, found Ref { mutable: true, .. }
  MIR-0007 bb4: move from possibly-moved place _1[]
  ```

- **User impact: "measure a caller's buffer, then modify it" did not build.** Accepted by the
  checker, executed correctly by the HIR oracle, refused only by a native build — the
  DEV-132/133/146/147 class, fifth mechanism.
- **Security/soundness impact:** none — it refused a valid program. The repair's own risk (letting
  a `&T` base satisfy a `&mut` receiver) is pinned by three negative controls.
- **Root cause:** DEV-147's repair was narrowed on the wrong axis. The gate belongs on the BASE's
  mutability — is there a non-`Copy` reference at risk — while the reference built takes the
  RECEIVER's mutability. `&*base` from a `&mut` base IS the weakening, so one reborrow fixes both
  the MIR-0005 and the MIR-0007 half.
- **Fix:** `src/mir/lower.rs::reborrow_reference_receiver`. Evidence:
  `tests/dev149_shared_receiver_over_mutable_base.rs` (13 tests: all four receiver sites, the
  loop case, DEV-147's own case, three negative controls).
- **Owning gate:** package track, CD-354.

## DEV-150 — the argument read-conflict rule does not fire through a reference base [CLOSED, ruled and fixed CD-357, 2026-08-02]

- **Normative expectation:** `03-Type-System.md` — one `&mut` XOR many `&`, with no exception for a
  base that is itself a reference.
- **Current behaviour:** `f(&mut x, x.field)` is refused for a LOCAL base and ACCEPTED when the
  base is a `&mut` parameter. The HIR oracle executes the accepted form correctly; the native
  backend emits Rust that rustc refuses with E0503.
- **Minimal reproducer:** 14 lines.

  ```stark
  struct Holder { limit: UInt64, seen: UInt64 }
  fn bump(h: &mut Holder, by: UInt64) { h.seen = h.seen + by; }

  fn main() {
      let mut h = Holder { limit: 3u64, seen: 0u64 };
      bump(&mut h, h.limit);       // E0101 read conflict -- refused, correctly
  }

  fn forward(h: &mut Holder) { bump(h, h.limit); }   // ACCEPTED, runs, does not build
  ```

- **User impact:** `f(buf, buf.len())` inside a function taking `buf: &mut T` builds a program the
  native backend cannot emit. `stark-http-parser`'s four `take_line` call sites hit it.
- **Security/soundness impact:** unresolved, and that is the point. Under ruling (B) below the
  current acceptance is an aliasing hole the checker should have closed; under ruling (A) it is
  benign and the backend is at fault. The two readings disagree about whether a real program is
  sound, which is why this is escalated rather than repaired.
- **NOT A MECHANICAL REPAIR — two candidate rulings, both defensible:**
  - **(A) The checker is right; sequencing is the fix.** Read the field into a temporary BEFORE
    forming the `&mut` and nothing aliases — close to Rust's two-phase borrows, which exist so
    `v.push(v.len())` works. Cost: the LOCAL case must then start being accepted too, so this is a
    widening of the borrow rule, not a backend change.
  - **(B) The checker is wrong; the rule must fire through a reference base.** Uniform,
    conservative, matches the spec as written. Cost: `f(buf, buf.len())` stops compiling and every
    caller hoists the read into a `let`.
- **Workaround, valid under either ruling:** hoist the read. Applied at all four
  `stark-http-parser` sites.
- **RULING (CD-357): (B), uniform rejection; hoisting required.** Now normative as
  **OWN-BORROW-002** in `03-Type-System.md`:

  > A call may not create an exclusive borrow of a place while another argument in the same call
  > reads from or borrows an overlapping place. Such reads must be evaluated into locals before the
  > exclusive borrow is created.

  Uniform in the base — a local, a place reached through `&mut`, a field projection, an index, a
  free function or a method receiver — and independent of argument order. Core v1 therefore does
  NOT define argument evaluation as providing two-phase borrow semantics; that remains reserved,
  and the ruling is reversible if STARK later adopts them deliberately.
- **Why (B) over (A):** blessing the accepted case would have required accepting the local case
  too, which widens the borrow rule into two-phase borrows — evaluation-order machinery and a real
  semantics commitment. (B) keeps ONE backend-neutral rule that every engine satisfies by
  construction.
- **Fix:** `src/borrowck.rs` — `check_argument_overlap` runs as its own pass over the whole
  argument list (a method receiver included) BEFORE the left-to-right walk, because the walk can
  only ever see a conflict where the borrow comes first. `exclusive_borrow_of` treats an explicit
  `&mut place` and a `&mut`-typed place (which reborrows) alike; the second was the invisible one.
  A report-once set keeps one mistake to one diagnostic.
- **Livability:** all 15 first-party packages pass the gate under the rule with zero new
  diagnostics. The only site that ever hit it was `stark-http-parser`'s four `take_line` calls,
  hoisted when the defect was found.
- **Evidence:** `tests/dev150_argument_overlap.rs` (15 tests: negatives varying the base and the
  order, positives for every hoisted and non-overlapping form) and spec fixture
  `03-Type-System__19.stark`, classified `semantic-error` with `errors = "E0101"` — so the spec's
  own example is an executable test of the rule. Supersedes
  `dev150_argument_conflict_through_reference.rs`, which pinned the inconsistency while the ruling
  was open and whose two "the bases disagree" tests went red the moment they agreed.
- **Owning gate:** package track, CD-357.

## DEV-151 — a method on a host-resource receiver did not lower, and written `()` was not `Unit` [CLOSED, fixed CD-354, 2026-08-02]

Two defects, recorded together because the first concealed the second.

### (a) A resource receiver was not treated as a nominal

- **Normative expectation:** a package may declare `impl TcpStream { fn set_read_timeout(&mut self,
  ..) }` — CD-346 rules that a resource operation moving a cursor or consuming bytes takes `&mut
  self` — and a caller may call it.
- **Behaviour before the fix:** `lower_method_call` matched only `Struct`/`Enum` for the receiver
  nominal and refused everything else:

  ```text
  error: native build does not yet support this program: method call on non-nominal receiver
         HostResource(HostResourceTy { nominal: Item(ItemId(381)), provider: "stark-std-net",
         resource: "tcp_stream" }) (C4.5b+)
  ```

- **User impact: CD-346's ruling was unbuildable at every call site.** `stark-net` declared
  `set_read_timeout`/`set_write_timeout` on that ruling and QUALIFIED; nothing had ever called
  them natively, so nobody learned they could not be called. This is CD-345's lesson one level
  down — CD-347 made a package's resource LIFECYCLE executable, and this was a declared surface
  whose CALL SITES were still unexecuted. `stark-http-client` was the first caller.
- **Root cause:** a missing match arm, not a missing capability. `HostResourceTy.nominal` already
  holds the item of the synthesized zero-variant enum (CD-234), which is exactly the item the
  `impl` hangs off.
- **Fix:** `src/mir/lower.rs`, one arm mapping `MirTy::HostResource` to its nominal item. A `Core`
  resource nominal still refuses, per CD-235's sequencing exception.

### (b) `()` written in source lowered to `Tuple([])`

- **Normative expectation:** MIR has one canonical unit type.
- **Behaviour before the fix:** `MirTy::Unit` is used at all 99 synthesized sites and the empty
  tuple is never constructed deliberately, but a written-out `()` in a type annotation reached the
  tuple arm. So `fn f() -> Result<(), E>` declared a return type no constructed value could match:

  ```text
  MIR-0004 stark_net::TcpStream::set_read_timeout@[] bb26: assignment:
    expected Enum(CoreResult, [Tuple([]), ..]), found Enum(CoreResult, [Unit, ..])
  ```

- **User impact:** `Result<(), E>` is an extremely common signature, and it was fine everywhere the
  body was never lowered. It took two unexecuted paths crossing to make it reachable.
- **Fix:** both conversion sites (`mir_ty`, `hir_field_ty`) canonicalise an empty tuple to `Unit`.
- **Evidence for both:** `tests/dev151_resource_method_dispatch.rs` (4 tests, including a structural
  assertion that no lowered signature or local carries `Tuple([])` — which catches a divergence
  that has not yet MET a conflicting value). End-to-end: `stark-http-client-consumer` calls
  `set_read_timeout` on a live socket under the qualification gate's HTTP peer.
- **Owning gate:** package track, CD-354.

## DEV-152 — an `impl` whose type has no page-level item had its methods silently dropped from documentation [CLOSED, fixed CD-355, 2026-08-02]

- **Normative expectation:** `stark doc` documents a package's public items. A `pub fn` in an
  `impl` block is a public item.
- **Behaviour before the fix:** `doc_gen::extract` collected impl members separately and attached
  them to the type's own doc item; when the type had no page-level item **the methods were
  discarded with no diagnostic**.
- **Reproducer:** any `impl T { pub fn .. }` where `T` is not declared in the same package.
- **User impact:** a provider-bound resource nominal is SYNTHESIZED, not written (CD-234), so
  `impl TcpStream` had nothing to attach to. All seven of `stark-net`'s public methods —
  `connect`, `read`, `write`, `write_all`, `set_read_timeout`, `set_write_timeout`,
  `shutdown_write` — were absent from its documentation.
- **Why it compounded:** DEV-151 showed the two timeout setters could not be BUILT at a call site,
  and one reason nobody had called them is that the docs did not say they existed. An undocumented
  API and an uncalled API are the same failure seen from two sides. It also blocked CD-355's
  surface gate, which uses `stark doc` as the authority on what is public: built on the old
  extractor, that gate would have certified `stark-net` as fully covered.
- **Security/soundness impact:** none — it hid a surface rather than mis-compiling one.
- **Fix:** `src/doc_gen/extract.rs` synthesizes a page for the type instead of discarding. Evidence:
  `tests/dev152_orphan_impl_methods_documented.rs` (4 tests, including "a declared type gains no
  duplicate page" and "a private method is still excluded").
- **Owning gate:** package track, CD-355.

## DEV-153 — `hir_field_ty` had no arm for an unsized slice [CLOSED, fixed CD-355, 2026-08-02]

- **Normative expectation:** `&[T]` is a legal parameter type anywhere a parameter is legal.
- **Behaviour before the fix:** a method on a host-resource receiver whose parameter was a slice
  refused to lower, while the identical free function built:

  ```text
  owned.write_all("x".bytes())        -> field type form (C4.5)
  write_all(&mut owned, "x".bytes())  -> builds
  ```

- **Root cause:** `mir_ty` has had a `Ty::Slice` arm all along; `hir_field_ty` did not, because it
  only ever converted struct fields and enum payloads — and Core v1 forbids reference-typed fields,
  so `&[T]` could not reach it.
- **Why it appeared now — and this is the part worth keeping:** DEV-151(a) opened method dispatch on
  a resource receiver, which routed a method's DECLARED parameter types through `hir_field_ty` for
  the first time. **A repair that widens what is reachable will expose whatever the newly reachable
  path never handled.** That is the cost of the DEV-151 class, not an argument against fixing it:
  the alternative is the surface staying unreachable and the gap staying invisible, which is exactly
  how `set_read_timeout` shipped unbuildable.
- **Security/soundness impact:** none — it refused a valid program.
- **Also fixed:** the `_` arm's message named neither the form nor the type, which is why bisecting
  cost as long as it did. It now names the item kind.
- **Fix:** `src/mir/lower.rs::hir_field_ty` gains a `Slice` arm. Evidence:
  `tests/dev153_slice_parameter_in_resource_method.rs` (4 tests: shared slice, mutable slice, two
  slices, plus a non-resource control that was never broken).
- **Owning gate:** package track, CD-355.

## DEV-154 — the read-conflict check is local-granular, so disjoint field projections over-reject [CLOSED, fixed CD-358, 2026-08-03]

- **Normative expectation:** `03-Type-System.md` OWN-BORROW-001 — "Disjoint field projections do
  not overlap."
- **Current behaviour:** `check_read_borrow_conflict` compares only the borrow's LOCAL, ignoring
  projections, so a borrow of one field blocks a read of a sibling field:

  ```stark
  struct P { a: UInt64, b: UInt64 }
  fn f(x: &mut UInt64, y: UInt64) { *x = *x + y; }

  let mut p = P { a: 1u64, b: 2u64 };
  f(&mut p.a, p.b);   // E0101 — but `p.a` and `p.b` are disjoint
  ```

- **PRE-EXISTING, and not introduced by CD-357.** Found because CD-357's own overlap check —
  which IS place-granular and uses `places_overlap` — correctly declined to fire here, leaving the
  older local-granular check as the only reporter. Two checks in the same area disagreeing about
  granularity is how it became visible.
- **User impact:** a valid program is refused, and the workaround (hoist the sibling read) is the
  same shape OWN-BORROW-002 requires — so it is easy to mistake this for the new rule rather than
  a defect. The compiler and the spec disagree, and the spec is right.
- **Security/soundness impact:** none — it refuses a valid program.
- **Fix (CD-358):** the `Borrow` record now carries the borrowed PLACE rather than only its root
  local, and every comparison — borrow creation, assignment, move, method receiver, and the read
  check — goes through `places_overlap`, the helper DEV-135 already made field-precise.
- **Why the negatives are the important half:** this repair makes the checker accept MORE, so each
  refusal test is load-bearing. Identity, parent-over-child, whole-local-over-field, two exclusive
  borrows of one field, assignment to a borrowed place, and move-out-of-borrowed-storage all stay
  refused. The move check is deliberately stricter than the read check — it rejects under ANY live
  borrow, shared included, because moving invalidates storage a live view still points into — and
  making the comparison place-granular did not weaken that.
- **Evidence:** `tests/dev154_place_granular_borrow_conflicts.rs` (10 tests, 4 accepted / 6
  refused). All 15 packages qualify and the external sample suite is 39/39.
- **Owning gate:** package track, CD-358.

## DEV-155 — a method's impl generics and a trait default's signature were read from the wrong file [CLOSED, fixed CD-358, 2026-08-03]

- **Normative expectation:** a generic method resolves the same way regardless of which module
  calls it.
- **Behaviour before the fix:** a generic method on a generic impl, called across a module
  boundary, returned a parameter type named from the CALLER's file:

  ```stark
  // src/lib.stark
  pub struct Wrap<T> { pub inner: T }
  impl<T> Wrap<T> { pub fn get(&self) -> &T { &self.inner } }
  mod inner;

  // src/inner.stark
  let w = Wrap { inner: 11 };
  *w.get() != 11        // E0001 expected 'S', found an integer literal
  ```

  `'S'` is `T`'s offset in `lib.stark` landing on an `S` in `inner.stark`. Nothing could unify
  against it.
- **The same class as DEV-069, DEV-101 and DEV-148**, at a fifth and sixth site: the method
  candidate loop converted the impl's self type and keyed `match_impl_type`'s generic map with
  `self.text`, and the selected signature's parameter and return types were converted with no
  declaring-file context. The trait-default path had the same gap — DEV-069 had fixed that
  default's NAME but not its signature TYPES.
- **Security/soundness impact:** none — it refused valid programs. But note the near miss:
  `item_text` returns `"?"` for an out-of-range span, so two mis-sliced parameter names could
  COLLIDE on one key and substitute each other's types, which would be a WRONG program rather than
  a rejected one. A two-parameter test pins that they cannot.
- **Fix:** `src/typecheck/traits.rs` — a `decl_text` helper that resolves against `foreign_sig_item` when
  a declaring item is in scope, plus that context set across the method candidate loop, the
  selected signature's conversion, and the trait-default path.
- **Evidence:** `tests/cd358_cross_module_provenance.rs` (8 tests over real two-file package
  graphs, covering generic methods, method-level generics, trait defaults, associated types,
  bounded generic functions, fields and enum variants, and two non-colliding parameters).
- **Owning gate:** package track, CD-358.

## DEV-166 — a compiler-known trait bound contributed no callable methods (RESOLVED, DEV-DISPLAY-DISPATCH)

- **Normative expectation:** `03-Type-System.md` TYPE-METHOD-003 and `06-Standard-Library.md`
  STD-TRAIT-002 (both added by this work package, as the canonical statement of a property the
  spec had only implied): a Core trait identity is an ordinary trait for every purpose a program
  can observe, and its declared methods are callable through a generic bound on exactly the terms
  a user-declared trait's are. `STD-HOOK-001` already said `Display` is *not* a syntax hook and
  "uses ordinary name resolution, method selection, trait dispatch".
- **Behaviour before the fix:**

  ```stark
  fn show<T: Display>(x: T) -> String {
      x.fmt()
  }
  ```

  `[E0302] method 'fmt' not found for type 'T'`. The identical shape over a user-declared trait
  compiled and ran. The bound was *checked* — `satisfies_bound` accepted `T: Display` — and then
  contributed nothing to method resolution.
- **Root cause:** `typecheck/body.rs::resolve_method`'s bounded-generic branch resolved each bound by
  searching `hir::ItemKind::Trait` items for a matching name. A compiler-known trait has no
  declaration item, so the search returned `None`, the branch fell through, and the impl scan
  below it could not match a `Ty::Param` receiver either. Method visibility therefore depended on
  whether a trait happened to be compiler-known — two trait models rather than one.
- **DEV-023's relationship to this.** DEV-023 (closed in WP-C2.11) recorded that `Display`/`Hash`
  as *bounds* were "already correctly recognized". That was true of bound CHECKING and false of
  everything downstream of it; the concrete-receiver half it fixed (`"hi".fmt()`) left the generic
  half open, and nothing distinguished the two claims. This entry is the other half.
- **Two further defects the same branch was hiding**, both pre-existing and both fixed here
  because the shape the work package requires cannot work without them:
  - *Ownership.* `borrowck.rs::method_receiver` had no branch for a `Ty::Param` receiver at all,
    so it returned `None` and its caller CONSUMED the receiver. Every `&self` method reached
    through a bound moved its receiver — for user-declared traits too:
    `fn f<T: Named>(x: T) { x.name(); x.name(); }` failed E0100 "use of moved value".
  - *Ambiguity.* The branch returned on the FIRST bound supplying the name, so two bounds
    declaring the same method resolved by written order instead of being reported ambiguous.
- **User impact (while open):** no generic code could format a value. Every `Display`-generic
  library function — the whole reason a `Display` bound exists — was unwritable, which is why
  `packages/` contained no formatting package before this work.
- **Security/soundness impact:** none. It refused valid programs; it never accepted a wrong one.
  The ownership half is the one worth noting: it was also a REFUSAL (a spurious move error), not a
  missed move.
- **Fix:** `typecheck/` collects candidates from both kinds of trait into one list and runs one
  selection over it; a Core trait's signatures come from `core_trait_contract`, the same table
  user `impl` blocks are already checked against, so there is no second signature registry.
  `borrowck.rs` reads the receiver form from the same source. `interp.rs` dispatches a
  generic-parameter receiver through the Core surface when the runtime value has no nominal.
  `mir/lower.rs` lowers `Display::fmt` on a standard-library receiver to new
  `stark_runtime::format` calls that share their renderers with `print`/`println`.
- **Evidence:** `tests/dev_display_dispatch.rs` — 21 tests: three-engine agreement over the
  primitives, user impls, non-`Copy` and affine values, nested forwarding, both bound orders and
  an impl-head bound; missing-bound, wrong-bound, unknown-method, non-`Display`-concrete-type,
  arity and ambiguity rejections; a generated-source check that the native engine uses STARK's
  renderers and not Rust's `Display`/`Debug`/`format!`/`ToString`; and debug-vs-release native
  agreement. Plus `packages/stark-fmt`, whose whole surface is generic over `Display`.
- **Owning gate:** package/application track, CD-378.

## DEV-167 — `Display::fmt` has no method-form `to_string()` counterpart (OPEN, deferred by decision)

- **Normative expectation:** `06-Standard-Library.md` describes `str::to_string`/`String::from`
  but does not promise a `to_string()` method on every `Display` type. This entry exists so the
  absence is recorded rather than rediscovered.
- **Current behaviour:** `x.to_string()` resolves only for `str`/`String` receivers, through the
  ordinary runtime-text surface. A `T: Display` parameter has no `to_string()`.
- **Why it is deferred rather than fixed:** the principled mechanism is
  `impl<T: Display> ToString for T`, and Core v1 has neither blanket implementations nor extension
  traits. The alternative — a resolver branch keyed on the method name `to_string` — would
  reintroduce exactly the two-tier model DEV-166 removed, for an ergonomic gain.
- **Workaround, and it is a real one:** a free function over the ordinary bound.
  `packages/stark-fmt` ships it as `to_string<T: Display>(value: &T) -> String`.
- **Owning gate:** unassigned; blocked on a blanket-implementation decision, which is CE-shaped.

## DEV-168 — a qualified call to a compiler-known trait's method has no MIR lowering (OPEN)

- **Normative expectation:** TYPE-METHOD-001 — "Trait methods can always be called in
  fully-qualified function form — `Display::fmt(&x)`". The spec names this exact call.
- **Current behaviour:** `Display::fmt(&x)` type-checks (DEV-052) and runs under the HIR oracle,
  and MIR lowering refuses it with "callee form (C4.5)". So the shape the spec offers as the
  disambiguation mechanism for an ambiguous trait method cannot be built natively.
- **User impact:** a program whose generic parameter is bounded by two traits declaring the same
  method is correctly reported ambiguous, and the documented way to resolve it runs in only one
  of the three engines. A user trait's qualified call (`OtherFormat::fmt(&x)`) has the same gap.
- **Security/soundness impact:** none — a refusal at lowering, caught before any code is emitted.
- **Discovered by:** DEV-DISPLAY-DISPATCH, while proving that the ambiguity it introduces is
  resolvable. Deliberately not fixed there: adding a qualified-call lowering is a feature, and the
  work package's scope statement excludes expanding into one.
- **Evidence:** `tests/dev_display_dispatch.rs::qualified_calls_disambiguate_the_two_traits`,
  which checks the shape through the front end and the oracle and states this limitation.
- **Owning gate:** unassigned.

## DEV-169 — an explicit `.drop()` call ran the destructor TWICE (RESOLVED, CD-383)

- **Normative expectation:** `03-Type-System.md` "Copy and Drop" — destructors run exactly once
  and there are no explicit `Drop::drop` calls.
- **Current behaviour:** `r.drop()` on a type with `impl Drop for R` type-checks, because
  `impl Drop`'s method is an ordinary `ImplItem::Fn` that the method candidate scan finds like any
  other. Confirmed empirically 2026-08-04.
- **Security/soundness impact:** potentially real — an explicit call followed by the automatic one
  is a double destruction at the source level. Not investigated here; whether the drop flag
  suppresses the second run is unverified.
- **The unverified half, now verified — and it is worse than recorded.** This entry said "whether
  the drop flag suppresses the second run is unverified". It does not. The program above prints:

  ```text
  dropped
  after
  dropped
  ```

  The destructor runs **twice on one value**: once for the explicit call, once at scope end. For a
  resource-bearing type that is a double release. This was an over-acceptance that produced a
  **soundness violation**, not merely a program the language forbids.
- **Fix (CD-383):** the check runs at IMPL-MEMBER SELECTION — when a call resolves into an
  `impl Drop for T` block — rather than on the method's name. An inherent method named `drop` is
  unaffected, which a name-keyed check would have broken. Diagnostic **E0307**, naming the free
  function `drop(value)` as the sanctioned way to destroy early. The free function MOVES its
  argument, so the destructor still runs exactly once.
- **Evidence:** `tests/over_acceptance_audit.rs` — the rejection, an inherent `drop` still callable,
  `drop(value)` destroying exactly once (`released` before `after`, not again at scope end), and
  automatic destruction still running once.
- **Owning gate:** package/application track, CD-383.

## DEV-170 — a generic bound's trait identity was reconstructed from its spelling (RESOLVED, DEV-BOUND-TRAIT-IDENTITY)

- **Normative expectation:** `03-Type-System.md` TYPE-METHOD-003 — a generic parameter's candidates
  come from "the traits named by `T`'s declared bounds", and `TYPE-NOMINAL-001` makes item identity
  a package/module/name triple, not a bare name. Which trait a bound denotes is settled by name
  resolution (04-Semantic-Analysis.md); no later pass re-decides it.
- **Behaviour before the fix.** `typecheck::resolve_bound_trait` and
  `borrowck::bound_method_receiver` each took `text(bound.path.span)` — the bound's SOURCE TEXT —
  and scanned every HIR item for a trait declared with that name. Three failures, all reproduced
  before any code changed:
  1. **A qualified bound matched nothing.** `T: traits::Render` compared `"traits::Render"` against
     the declaration's name `"Render"`. The bound contributed no methods, and `value.render()` was
     rejected with *"method 'render' requires the bound 'T: Render'"* — on a function whose
     signature already wrote exactly that bound.
  2. **An unrelated trait captured the name.** `mod unrelated { pub trait Display { fn other(&self); } }`
     anywhere in the program took over every `T: Display` bound, because a user trait of that
     spelling was found and preferred over the Core trait the resolver had selected. `x.fmt()`
     then failed.
  3. **Declaration order decided ownership.** With two same-named traits, one `&self` and one
     `self`, the borrow checker returned whichever appeared FIRST in HIR item order. The same
     program compiled or failed E0100 depending only on the order its two trait declarations were
     written in.
- **And one level further down.** Even with identity fixed in both front-end passes, execution
  still selected an implementation by method name on the receiver's nominal: a type implementing
  two same-named `Render` traits ran the *same* body for both bounds, in the HIR interpreter and in
  MIR. The type checker was right and every engine below it was wrong in the same way. The native
  linkage preflight caught the underlying cause — `impl left::Render for Item` and
  `impl right::Render for Item` produced the identical canonical symbol `Item::Render::tag@[]`,
  "one symbol, two identities".
- **User impact (while open):** a qualified trait bound was unusable, which is every bound on a
  trait a package exports through a module. A same-named trait anywhere in the program could
  silently redirect an unrelated bound. And ownership was order-dependent.
- **Security/soundness impact:** the first two failures were refusals. The third and fourth were
  **acceptances of the wrong program**: order-dependent move checking, and a call executing a
  different trait's body than the one type checking approved. No memory-safety violation — the
  wrongly selected body is still a well-typed method of the same receiver — but "which code runs"
  differed from "which code was checked", which is the more serious half of this entry.
- **Fix:** `hir::resolved_bound_trait` reads `TraitRef::res` and nothing else, with exhaustive
  matches over `Res` and `ItemKind`; both front-end passes consume it, and `hir::BoundTrait` is now
  shared rather than private to the type checker. The checker records the selected trait per call
  (`TypeTables::bound_trait_calls`); the HIR interpreter passes it as `find_method`'s existing
  trait filter, and MIR lowering passes it to `find_impl_fn`'s new one. Canonical symbols include
  the trait's own module path, so two same-named traits are two symbols. A top-level trait's prefix
  is empty, so every pre-existing symbol is unchanged.
- **Evidence:** `tests/dev_bound_trait_identity.rs` — 15 tests, including the declaration-order
  pair (the same program with its two trait declarations swapped, both of which must compile), the
  cross-module `L`/`R` dispatch case that pins which body ran, receiver identity across `&self`,
  `&mut self` and `self`, and a direct assertion that `resolved_bound_trait` returns the resolver's
  own `Res::Item`.
- **Owning gate:** package/application track, CD-379.

## DEV-171 — an unrelated trait satisfied an OPERATOR bound by spelling (RESOLVED, CD-383)

- **Normative expectation:** `03-Type-System.md` "Operators and Traits" — `==` on a generic
  parameter requires `T: Eq`, meaning the Core `Eq` (06-Standard-Library.md STD-TRAIT-001), not a
  user trait that happens to be spelled that way.
- **Behaviour before the fix:** accepted. Reproduced 2026-08-04:

  ```stark
  mod fake {
      pub trait Eq {
          fn unrelated(&self) -> Int32;
      }
  }

  use fake::Eq;

  fn compare<T: Eq>(a: T, b: T) -> Bool {
      a == b
  }
  ```

  `ty_satisfies_operator_bound`'s generic-parameter branch compares
  `text(bound.path.span) == required` — a string comparison against `"Eq"`. The imported `fake::Eq`
  spells the same, so the operator bound is treated as satisfied. Written *qualified*
  (`T: fake::Eq`) it is correctly rejected, which is the same spelling artefact seen from the other
  side.
- **Security/soundness impact:** this is an **acceptance**, not a refusal — the more serious
  direction. What `a == b` then lowers to for such a `T` was not investigated.
- **Discovered by:** DEV-BOUND-TRAIT-IDENTITY, while confirming no spelling-based bound lookup
  remained. Deliberately not fixed there: this is operator-bound *satisfaction*, not method
  identity, and the repair decides what happens when a user trait shadows a Core trait's name for
  operator purposes — a semantics ruling rather than a mechanical fix.
- **Related:** the same function also serves built-in obligations that have no `TraitRef` at all
  (DEV-118's name-addressable mechanism), so the fix could not simply delete the name comparison.
- **Fix (CD-383):** `param_declares_bound` resolves each bound through `hir::resolved_bound_trait`
  — the identity path CD-379 established — and compares it to the Core trait the operator requires.
  `satisfies_bound_parts` keeps its name-addressable form, because DEV-118's built-in obligations
  genuinely have no `TraitRef`; only the GENERIC-PARAMETER branch changed, which is the one that
  had a written bound to resolve all along.
- **Scope of the repair:** `Eq`, `Ord` and `Num` all go through this branch, so all three are
  covered by construction rather than by enumerating them.
- **This rejects programs that previously compiled.** That is the intent — they were accepted
  against the specification, which requires operators to dispatch to the canonical Core trait — but
  it is a behaviour change and is called one here rather than described as a pure bug fix.
- **Evidence:** `tests/over_acceptance_audit.rs` — the imported and qualified fake `Eq`, fake `Ord`
  and fake `Num`, plus genuine `Eq`/`Ord`/`Num` bounds and a user `impl Eq` still working.
- **Owning gate:** package/application track, CD-383.

## DEV-172 — no signed type can express its own minimum value (OPEN, pre-existing)

- **Normative expectation:** `03-Type-System.md` — `Int8` is "8-bit signed integer (-128 to 127)".
  `-128` is an `Int8`.
- **Current behaviour:** rejected. Confirmed empirically 2026-08-04, with no interpolation involved:

  ```stark
  let a: Int8 = -128;                    // [E0008] integer literal out of range for 'Int8'
  let d: Int64 = -9223372036854775808;   // [E0008] integer literal out of range for 'Int64'
  let u: UInt64 = 18446744073709551615;  // [E0008] out of range for 'Int64'
  ```

  A negative literal is a unary minus applied to a positive literal, and the magnitude is
  range-checked against the target type *before* the negation. `128` does not fit `Int8`, so
  `-128` is refused — and the same argument refuses every signed minimum. `UInt64::MAX` fails for a
  related reason: the literal is classified against the signed default before its `UInt64` context
  is applied.
- **User impact:** the minimum of every signed width, and the maximum of `UInt64`, are
  unwritable. A program needing `Int32::MIN` has no literal for it.
- **Security/soundness impact:** none — a refusal, not an acceptance. But it is a conformance gap
  against a range the specification states explicitly.
- **Discovered by:** WP-FMT-001, while testing that formatting a minimum signed value does not
  overflow while taking its magnitude. The RENDERER handles it correctly — pinned by
  `stark_runtime::fmt_spec::tests::minimum_signed_values_do_not_overflow_their_own_width`, which
  formats `i64::MIN` — but no STARK program can produce the value to hand it.
- **Owning gate:** unassigned. Literal typing, not formatting; a fix belongs with the literal
  range check (DEV-015's area), which must learn that a literal in unary-minus position is checked
  against the negated range.

## DEV-173 — an interpolation field may not contain a nested string literal (RESOLVED, CD-382)

- **Normative expectation:** `01-Lexical-Grammar.md` LEX-FORMAT-002 admits an arbitrary expression
  in a field. A nested string literal is an expression.
- **Current behaviour:** refused with E0218, "an interpolation field may not contain an escape
  sequence". Because the enclosing literal is delimited by `"`, a nested string literal must be
  written `f"{call(\"a\")}"` — so its source carries the OUTER literal's escapes.
- **Why it is refused rather than supported:** every expression node reads its text from its
  `Span`. A field's expression is parsed by lexing the original file over the field's own byte
  range, which keeps spans real; a field containing escapes cannot be lexed that way, because `\"`
  is not valid expression syntax. Parsing a DECODED copy works, but the resulting nodes' spans
  index a scratch buffer no consumer can read, and retagging them to the field's span makes a
  string literal read the field's raw source back — `f"{\"slice\"}"` then renders `"slice"`, with
  quotes, where the program said `slice`. That was observed during implementation, which is why
  this is a refusal: producing the wrong string silently is worse than declining.
- **User impact:** small and with a clean workaround — bind the value first
  (`let s = "slice".to_string(); f"{s}"`). Every other expression form works, including calls,
  indexing, field access, struct literals and qualified paths.
- **Security/soundness impact:** none — a refusal.
- **Resolution (CD-382), in two halves, both required.**
  1. **A length-preserving stand-in.** The field is parsed against a copy of the whole file in
     which each `\"` inside that field becomes ` "` when it opens a nested literal and `" ` when it
     closes one. Every byte offset is unchanged, so the spans the sub-parse produces are already
     real file spans and nothing needs remapping — which matters because spans are embedded
     throughout the AST (paths, segments, names), not only on nodes. **Which side the space lands
     on is load-bearing:** blanking the closing backslash in place puts the space inside the
     literal, and `f"{choose(\"yes\", ..)}"` renders `yes ` with a trailing space. That was
     observed during implementation, not reasoned about afterwards.
  2. **Literals carry their decoded value.** `Ast::str_lits`/`Hir::str_lits` hold every string
     literal's value, interned at parse time from whatever buffer the parser was reading;
     `Lit::Str` names its entry. Spans are now purely diagnostic. Without this the stand-in is not
     enough: a literal would still read its value back from the real file's `\"a\"`.
- **What remains refused, and why that is not the same defect.** An escape OTHER than `\"` in a
  field source belongs to the enclosing literal and *changes* the inner text — `\\` means one
  backslash, `\n` means a newline — so blanking it would silently alter the value. Those fields are
  refused with that reason. The forms the acceptance matrix named all work:
  `f"{choose(\"yes\", \"no\")}"`, `f"{lookup(\"name\")}"`, `f"{parse(\"42\").unwrap()}"`.
- **Recorded as a language rule, not only here.** LEX-FORMAT-004 (01-Lexical-Grammar) states both
  halves normatively: a nested literal's `\"` delimiters are read as delimiters, and a nested
  literal needing a data-bearing escape is rejected with the bind-first workaround shown. A
  restriction that lives only in a defect ledger is one a reader of the grammar never learns about.
- **Do not describe the result as "complete ordinary-expression interpolation."**
  `f"{lookup(\"a\nb\")}"` is a valid ordinary expression and is refused. The accurate claim is
  "complete for the defined Core v1 interpolation surface".
- **Evidence:** `tests/wp_fmt_001_interpolation.rs::a_field_may_contain_a_nested_string_literal`
  (six forms, including a `:` and a `}` inside a nested string, a struct literal, and a format
  specification applied to one) and `::a_field_may_not_contain_an_escape_other_than_a_quote`.
- **Owning gate:** package/application track, CD-382.

## DEV-174 — `eprint`/`eprintln` took `&str` instead of a `Display` value (RESOLVED, CD-381)

- **Normative expectation:** `06-Standard-Library.md` declares `fn eprint<T: Display>(value: T)` and
  `fn eprintln<T: Display>(value: T)`, and PRINT-DISPLAY-001 names all four output functions
  together as "implementation-provided generic functions".
- **Behaviour before the fix:** `eprintln(s)` with `s: String` was rejected —
  `[E0001] type mismatch: expected '&str', found 'String'` — while `println(s)` was accepted.
  `builtin_type` typed the stderr pair with a `&str` parameter and the stdout pair with a fresh
  inference variable. Nothing else differed: the runtime surface has carried the full stderr display
  family (`EprintlnInt64`, `EprintBool`, `EprintlnFloat32`, …) since 0.1-A13, and lowering already
  redirects the display path by channel. **Only the signature lagged.**
- **User impact:** every `Display` type except `&str` was unprintable to stderr. A diagnostic path —
  the one place a program most wants to render a value — was the one place it could not.
- **Security/soundness impact:** none — a refusal.
- **Discovered by:** WP-FMT-001's correction packet, proving `eprintln(f"...")` in its direct form
  rather than through `.as_str()`. The original suite tested `.as_str()` only, which is exactly why
  the gap survived the first pass.
- **Fix:** the stderr pair is typed like the stdout pair, and both pairs now go through the same
  deferred `Display` check, so `eprintln` of a type with no `Display` impl is rejected for the same
  reason `println` is.
- **Evidence:** `tests/wp_fmt_001_interpolation.rs::the_output_family_accepts_an_interpolated_temporary_directly`,
  and `tests/adversarial_stderr.rs::the_eprint_family_accepts_every_display_value` — the three
  shapes WP-C7.9's `the_eprint_family_accepts_only_str_today` had pinned as rejections, now proven
  to render byte for byte on stderr. `::the_eprint_family_still_requires_display` pins that
  widening the signature widened nothing else.
- **A note on how this was found and how it was meant to be found.** WP-C7.9 recorded the
  restriction deliberately, said the lowering already supported every `Display` shape so widening
  would need "only a signature change and cases", and predicted that its pinning test would "fail
  the day that happens, which is the right moment to add them". That is exactly what occurred: the
  test failed in CI on the signature change, and the cases it asked for are the ones now present.
  A recorded limitation with a test that fails when it is lifted is a better artifact than a
  to-do — it is why this repair took one commit rather than a rediscovery.
- **Owning gate:** package/application track, CD-381.

---

## DEV-176 — generic callable bodies execute without their checker-established context (RESOLVED, WP-VALUE-REP-TOTAL A3c-S)

- **Normative expectation:** `03-Type-System.md` §Generics: "Generic parameters are in scope within
  the item body and signatures", and instantiation occurs at use sites with monomorphization and
  dictionary-passing required to be observably equivalent. A generic body must therefore execute
  knowing what its parameters stand for, whatever kind of callable it is.
- **Current behaviour:** the HIR interpreter installs a substitution frame only for **direct calls
  to free functions**. `push_generic_frame` reads its parameter names from `hir::ItemKind::Fn` and
  returns an empty list for every other item kind, and it has exactly one call site — the
  `Res::Item` path in `eval_expr`. A method call never pushes a frame at all. So none of these are
  ever bound: impl-level generics, method-level generics, trait-level generics, or `Self`.
- **Reproducer** (verified 2026-08-05):

  ```stark
  fn free_size<T>() -> UInt64 {
      size_of::<T>()          // works
  }

  struct Wrapper<T> { value: T }

  impl<T> Wrapper<T> {
      fn size(&self) -> UInt64 {
          size_of::<T>()      // fails
      }
  }
  ```

  > `layout query on Param("T") still contains an unsubstituted generic parameter: the active
  > instantiation did not cover it`

- **User impact:** a program the checker accepts fails at run time in the HIR engine, with a message
  describing a compiler-internal condition. Any construct needing the type parameter inside a
  generic method is affected; `size_of::<T>()` is simply the one with an existing observable.
- **Security/soundness impact:** none directly — it is a refusal, not a wrong answer. The
  *classification* is the soundness-adjacent part, and is DEV-176's sibling repair: see below.
- **Misclassification (repaired separately):** the failure was raised through `RuntimeError::new`,
  making it `FailureClass::Trap` — a **language outcome**. The HIR interpreter is the behavioural
  oracle the other three engines are compared against, so an oracle defect presented as a trap is
  something the differential harness can accept as legitimate and then pressure MIR and native into
  reproducing. Only the surviving-`Ty::Param` condition is reclassified; ordinary `layout_of`
  refusals are left alone pending individual classification, because some of them correspond to
  genuinely invalid programs.
- **Measured exposure** (2026-08-05, all 28 first-party packages): **0** generic impls, **0** traits,
  **0** generic methods across 1108 functions; 14 non-generic impls. The construct appears in the
  compiler's own Rust tests (48 generic impls) and in 7 of 116 spec fixtures. Commands:

  ```bash
  grep -rn "^impl<" packages/*/src/*.stark | wc -l
  grep -rn "^pub trait \|^trait " packages/*/src/*.stark | wc -l
  grep -rn "^    fn [a-z_]*<" packages/*/src/*.stark | wc -l
  ```

- **Workaround:** move the parameter-dependent operation into a free generic function and call it
  from the method.
- **Proposed disposition:** repaired by WP-VALUE-REP-TOTAL A3c-S, which replaces
  `TypeTables::generic_insts` with a single provenance-carrying callable-instantiation table and
  installs the checker-selected environment for every source-invoked callable. The interpreter must
  consume that environment, never reconstruct one from names, runtime values or impl scanning — a
  second instantiation algorithm would be a second answer to what a generic call means.
- **Explicitly excluded:** generic `Drop`. `drop_value` receives a `Value` and recovers the nominal
  through `nominal_item`, so `Wrapper<String>` is indistinguishable from `Wrapper<Int32>` at
  destruction. Threading a concrete type through **44** `drop_value` call sites, or retaining type
  arguments in `Value`, is disproportionate to **0** first-party `Drop` impls and 2 generic-`Drop`
  fixtures. A3c-D therefore refuses a generic `Drop` as `InternalInvariant` before running the
  destructor body, rather than guessing or silently skipping it.
- **Fix (A3c-S):** the checker publishes the environment it already selected — `GenericBinder`
  recording each binding's origin, `CallableInstantiation` keyed by CALL EXPRESSION because one
  generic body is legitimately invoked at two types in one program. `push_callable_env` installs it
  for every callable kind and composes it against the caller's active frame.
  `TypeTables::generic_insts` is **deleted**, not supplemented: all four consumers migrated, MIR
  derives its ordered view from `own_arguments()`, and the E0004 undetermined-instantiation check
  moved across but deliberately over the same subset it always covered.
- **Two subtleties the fix cost:** `Self` is published as the impl's self type and so references the
  impl's own parameters, requiring parameters to be concretised BEFORE `Self` is substituted through
  them — a flat loop leaves `Self = Wrapper<Param("T")>`. And DEV-101's provenance rule empties the
  environment *silently*: resolving a callee's parameter names with the caller's `decl_text` rather
  than `item_text` makes every lookup miss, publishing empty bindings instead of failing. Only
  `cross_package_generics` could see it; single-package tests and the three-engine differential were
  green while it was broken.
- **Generic `Drop` is excluded and refused (A3c-D):** destruction retains no type arguments, so a
  generic `Drop` is refused as `InternalInvariant` before its body runs rather than executed with
  unbound parameters. Recorded rather than repaired — 44 `drop_value` call sites against 0
  first-party `Drop` impls. See the A3c-D tests.
- **Evidence:** `tests/dev176_generic_callable_context.rs` (5, including one body answering
  differently at two instantiations), `tests/a3cd_generic_drop.rs` (4),
  `tests/cross_package_generics.rs` (11), `three_engine_differential` (109), lib (523).
- **Owning gate:** compiler track, WP-VALUE-REP-TOTAL A3c.

---

## DEV-177 — generic-parameter shadowing accepted, contrary to NAME-SHADOW-001 (OPEN)

- **Normative expectation:** `04-Semantic-Analysis.md` **NAME-SHADOW-001**: "Generic parameters may
  not duplicate another generic parameter or an item-level `Self`; a nested item introduces fresh
  item scopes."
- **Current behaviour:** the checker accepts a method generic that duplicates its impl's generic,
  and the program runs.
- **Reproducer** (verified 2026-08-05):

  ```stark
  struct Wrapper<T> { value: T }

  impl<T> Wrapper<T> {
      fn choose<T>(self, value: T) -> T {
          value
      }
  }

  fn main() {
      let w = Wrapper { value: 7 };            // impl T = Int32
      let s = w.choose(String::from("text"));  // method T = String
      println(s.as_str());                     // prints: text
  }
  ```

  `stark check` reports **OK** and the program prints `text`. Two distinct types are bound to the
  name `T` in one signature.
- **User impact:** a program the specification forbids is accepted. It currently *runs* only because
  the interpreter never consults the impl binding at all (DEV-176) — the two defects are
  independent, but this one is masked by that one.
- **Security/soundness impact:** no wrong answer today. The exposure is prospective and specific:
  `Ty::Param` identifies a parameter by `String`, so while duplicate names are legal a name-keyed
  substitution environment could bind one concrete type to two different binders. Every tie-break
  available — last-insertion-wins, first-insertion-wins, method-shadows-impl — is a guess at
  semantics the type system does not carry.
- **Why this is a conformance gap and not a design question:** the rule already exists and is
  normative. It does **not** require giving `Ty::Param` declaration identity; that would be
  machinery built to support a construct the language prohibits. Enforcing NAME-SHADOW-001 makes
  `Ty::Param(String)` unambiguous by construction across every set of generic scopes simultaneously
  active in a callable, which is precisely what a name-keyed runtime substitution needs.
- **Enforcement boundary:** reject a duplicate within one generic list, an impl generic duplicated
  by a method generic, a trait generic duplicated by a default-method generic, and a generic named
  `Self` where item-level `Self` is in scope. The distinction is *inherited* scope, not lexical
  nesting: `fn outer<T>() { fn inner<T>() {} }` is legal because a nested item is a fresh item
  scope, and two sibling methods each declaring `U` do not overlap. `check_fn_def` already has
  `current_impl_generics` and `current_fn_generics`, which are the owners normatively in scope.
- **Discovered by:** WP-VALUE-REP-TOTAL's binder-identity probe, run before designing A3c-S's
  substitution representation.
- **Proposed disposition:** enforce the existing rule before A3c-S. Blocks A3c-S's name-keyed
  substitution while it stands.
- **Owning gate:** compiler track, WP-VALUE-REP-TOTAL (prerequisite to A3c-S).

---

## DEV-178 — generic context is not retained for associated-function calls or function values (OPEN)

- **Normative expectation:** as DEV-176 — a generic body executes knowing what its parameters stand
  for, whatever callable kind declares them and however it is invoked.
- **Class:** HIR oracle execution-context omission. Same class as DEV-176; two callable-use paths
  its repair did not cover.
- **Current behaviour:** an accepted generic callable reaches execution with a surviving
  `Ty::Param`. Invisible until A4 validated parameters, because before that the environment was
  consulted only by `size_of::<T>()`.
- **Associated-function cause:** `Type::func()` neither publishes a call-site environment nor
  installs one. The interpreter's `Res::AssociatedFn` branch calls `call_callable` directly while
  ordinary method calls install an environment first — a parallel call funnel that remembered a
  different subset of the steps.
- **Function-value cause:** `Value::Function` retains only an `ItemId`, discarding the instantiation
  selected when the item became a value. The instantiation is fixed at the COERCION, not the call:

  ```stark
  fn type_size<T>() -> UInt64 { size_of::<T>() }
  let f: fn() -> UInt64 = type_size::<Int32>;
  f();
  ```

  The call-site `Ty::Fn` says the result is `UInt64` and cannot tell the body what `T` is.
  Validating indirect calls against the caller-side function type would make a parameter check pass
  while leaving this execution defect in place.
- **Reproducers** (all valid, all rejected by A4's first enforcement):
  `Stack::identity(6)`, `Holder::new(7)`, and `let f: fn(Int32) -> Int32 = identity; f(41)`.
- **Resolution:** publish and install associated-function environments; give the function
  representation a payload carrying its concretised environment, captured at coercion. **No new
  `ValueKind`** — `Ty::Fn` still maps to `ValueKind::Function`, so §6's matrix is unchanged; only
  the payload grows. The environment must be concretised against the active frame BEFORE storage,
  because a function value may outlive the generic frame that created it.
- **How it was found, and how it should have been:** A4's parameter validation asked for the
  environment that nothing else had needed. A3c-S was declared complete on evidence that could not
  reach either path, and A3c-Q's suites did not either. Both paths were named in the work package's
  required-contexts list; the gap was in the evidence, not in the specification.
- **Owning gate:** compiler track, WP-VALUE-REP-TOTAL A3c-S2.

---

## DEV-179 — `MapIter`/`FilterIter` discard a generic callback's instantiation (DORMANT)

- **Status:** **DORMANT — unreachable while E0105 refuses iterator `map`/`filter`.** Not an active
  conformance failure, not an executable oracle divergence, not a DEV-121 blocker, and no
  first-party exposure. It is a **feature-activation prerequisite** and a known implementation
  hazard.
- **Class:** dormant execution-context defect — the same semantic class as DEV-178, reached through
  deferred iterator execution rather than an ordinary indirect call.
- **Cause:** `Value::MapIter` and `Value::FilterIter` retain only the callback's `ItemId`. When the
  iterator steps, it reconstructs a function value with **empty bindings**, so the callback's
  checker-selected environment is gone.
- **Effect if activated:** a generic callback executes without its instantiation. A surviving
  `Ty::Param` either fails as `InternalInvariant` at a validated boundary or — where no boundary is
  enforced — produces incorrect oracle behaviour silently.
- **Reachability gate:** Core v1 rejects the adapters at the front end:

  ```text
  [E0105] iterator method 'map' is not supported by this compiler;
          use a 'for' loop over the iterator instead
  ```

  Verified 2026-08-06. Reachability governs the defect's URGENCY, not whether the implementation
  contains it.
- **Activation condition:** any change permitting `map`/`filter` construction, or otherwise exposing
  these iterator variants directly.
- **Why registered rather than commented:** the implementation looks complete. On the day E0105 is
  lifted it activates silently with an empty environment, and whoever lifts it will be working in
  the front end rather than in `interp.rs`. A comment is local and easy to miss; a ledger entry plus
  a gate test that fails the moment the rejection is removed is what survives. DEV-174 is this
  repo's precedent: a recorded limitation with a test that fails when it is lifted turned a
  rediscovery into a one-commit repair.
- **Resolution:** store a complete `FunctionValue` — or an equivalent captured environment — inside
  `MapIter`/`FilterIter` rather than reconstructing one.
- **Disposition:** **do not repair during A4.** Repair before, or as part of, lifting E0105.
- **Evidence:** `tests/dev179_dormant_iterator_callbacks.rs` — the gate test, which fails the moment
  E0105 stops rejecting these adapters.
- **Owning gate:** compiler track, whichever work package lifts E0105.

---

## DEV-180 — the HIR interpreter flattens `&mut self` into owned receiver storage (OPEN)

- **Class:** runtime representation / receiver-lowering defect. **Independent of DEV-121** — A4 only
  exposed it.
- **Status:** CONFIRMED, reachable from accepted Core v1 programs.
- **Normative expectation:** a value stored under `Ty::Ref { mutable: true, .. }` is a reference.
  WP-VALUE-REP-TOTAL §6.4: a mutable reference must never be flattened to a bare value — it cannot
  write through, and `take(&mut v)` needs the place itself.
- **Current behaviour:** for `hir::Receiver::RefMut`, `call_user_method` removes the owned receiver
  from the caller's place and binds that owned value as `self`:

  ```rust
  hir::Receiver::RefMut => self
      .place_slot_mut(&receiver_place, span)?
      .take()
      .ok_or_else(|| RuntimeError::new("mutable receiver is unavailable", span))?,
  ```

  One arm above, `hir::Receiver::Ref` binds `Value::Ref(receiver_place.clone())` — a genuine
  reference. The asymmetry is deliberate and commented: "(`&mut self` keeps its take/write-back
  model.)"
- **Violation:** the checker types the `self` local `&mut Self`, so a value under
  `Ty::Ref { mutable: true, .. }` has `ValueKind::Struct` rather than `ValueKind::Ref` for the whole
  body.
- **Mechanism:** take from the caller's slot → bind the owned value as `self` → execute → write
  back.
- **Observed exposure:** five receiver tests fail once A4 validates the receiver boundary —
  `mut_reference_returned_from_mut_self_method_writes_through`,
  `receiver_restructure_preserves_mutation_and_move_semantics`,
  `references_write_through_and_core_methods_auto_deref`,
  `language_protocols_ignore_same_named_inherent_methods`,
  `for_loop_accepts_standard_and_user_iterators`.
- **Effect:** the oracle executes a reference-typed local with owned-value semantics, which can
  conceal ownership, aliasing, returned-reference and mutation differences from MIR and native.
- **Two consequences worth checking on their own merits**, independent of DEV-121: the caller's slot
  is EMPTY for the duration of the call, so the receiver can appear moved; and every error path must
  restore it or the value is lost.
- **Candidate repair, NOT approved:** bind `Value::Ref(receiver_place.clone())` as `&self` does,
  with mutability governed by the static `Ty` and the borrow checker. Writes through `*self` and
  `self.field` would then mutate the caller's place directly, returned references would point into
  caller storage rather than a method-frame temporary, and restoration with its failure paths would
  disappear.
- **Explicitly forbidden repairs:** permitting `&mut T → bare T` in `value_matches_ty`; a
  receiver-specific validator exception; or keeping take/write-back while wrapping the taken value
  in a synthetic reference to method-local storage — that would satisfy the shape check while
  leaving returned references pointing at temporary storage.
- **Open before repair:** why DEV-070 excluded `&mut self` when `&self` moved to genuine references;
  whether that limitation still holds; and whether the returned-reference test depends on rebasing
  out of the method frame.
- **Discovered by:** WP-DEV-121 A4 receiver-boundary enforcement.
- **Owning gate:** compiler track, its own repair — sequenced before A4 resumes.

---

## DEV-181 — a borrow taken by an assignment's own right-hand side blocks the assignment (OPEN)

- **Class:** borrow-checker false positive. Same mechanism as DEV-137: a borrow is pushed onto
  `Borrowck::active_borrows` and nothing pops it before the check that consults it.
- **Reproducer** (verified 2026-08-06):

  ```stark
  struct Node { value: Int32, depth: Int32 }

  impl Node {
      fn deeper(&self) -> Node { Node { value: self.value * 2, depth: self.depth + 1 } }
  }

  fn main() {
      let mut n = Node { value: 1, depth: 0 };
      n = n.deeper();
  }
  ```

  ```text
  [E0101] cannot assign to variable 'n' because it is borrowed
  ```

- **Cause:** the `Assign` arm calls `self.check_expr(*rhs)` — which pushes the receiver auto-borrow
  taken by `n.deeper()` — and then runs the write check against `active_borrows` with that borrow
  still on the stack. `check_block` and `check_stmt` are the only things that truncate, and neither
  runs between the two halves of one assignment.
- **User impact:** `x = x.method()` is an everyday idiom — updating a value through a method that
  returns a new one. It is met within the first hour of using the language, and unlike DEV-137 it
  has no hoisting workaround; the statement must be split in two:

  ```stark
  let next = n.deeper();
  n = next;
  ```

- **Security/soundness impact:** none — a false positive, a refusal.
- **Precedent:** DEV-137 (CLOSED, CD-336) fixed the identical mechanism for `while` and `if`
  conditions with `Borrowck::check_condition` — snapshot the depth, check the expression, truncate
  back.
- **Why the repair is NOT a copy of `check_condition`:** the RHS's borrow is sometimes the assigned
  VALUE itself. `n = n.deeper()` produces an owned `Node`, so the temporary's borrow dies with it;
  `r = &v.field` produces a reference whose borrow must survive the assignment. Truncating
  unconditionally would drop a borrow the program still holds. The rule must therefore be gated on
  whether the assigned type is borrow-carrying — the same kind of scope boundary DEV-137 drew when
  it deliberately excluded `match` scrutinees and `for` iterators, whose borrows must span the body.
- **Discovered by:** an ordinary-capability probe — recursion, nested control flow, methods,
  collections — written to check what the language can actually do. Everything else in that program
  worked; this was the only rejection.
- **Owning gate:** compiler track.


---

## DEV-183 — TRAIT-COHERENCE-001's cross-package clause was never enforced (CLOSED, AS1b-iii)

- **Rule:** 03-Type-System, TRAIT-COHERENCE-001: *"Inherent implementations are permitted only for a
  nominal type defined by the current package."*
- **Status:** CLOSED. The check now works; the one first-party violation it found is repaired.
- **Symptom:** none, which is the problem. The compiler accepted every cross-package inherent impl
  in every package build, silently.
- **Cause:** `typecheck::validate_impl_rules` decided "same package" by calling `find_package_root`,
  which walked a file's PATH upward looking for a `starkpkg.json` **on disk, during type checking**.
  After AS1a gave package sources logical `<package>/<path>` names, that path does not exist
  relative to any working directory, so the probe returned `None` for the impl's file *and* for the
  type's file. `None == None` made every type look local, and the rule could not fire.

  It fired before AS1a only by an asymmetry: the root file carried an absolute disk path while every
  other item's file carried a logical name, so the root probe found a manifest and the dependency
  probe found nothing. "Different package" fell out of the difference, not out of a comparison.
- **How it surfaced:** AS1b-ii-d removed the ambient `self.file` the probe read, replacing it with
  the source the impl's span names. That made all three reads consistent, all three answered `None`,
  and `test_cross_package_coherence_orphan_rule_with_real_packages` — a test written specifically to
  pin this behaviour, and passing until then — failed. Replacing the disk probe with `source_package`
  (the leading segment of the logical name) turned the rule on for the first time.
- **What it found:** exactly one violation across the 28 first-party packages.
  `stark-http-client/src/lib.stark:1468` wrote `impl HttpResponse { ... }` for `HttpResponse`, which
  is defined in `stark-http-core`. Three packages failed to check as a result: `stark-http-client`,
  `stark-http-client-consumer` and `stark-get`, all through that one impl.
- **Repair:** the two methods became a locally declared `JsonBody` trait implemented for
  `HttpResponse`. That is what TRAIT-COHERENCE-001 is designed to permit — coherence holds when the
  current package owns *either* the trait or the head type, and `stark-http-client` owns the trait.
  Behaviour and method names are unchanged; call sites still write `response.json()`.
- **User impact:** a package could extend a dependency's type with inherent methods, which the
  language does not permit. Nothing miscompiled — the accepted programs were well-typed — but two
  packages could have added conflicting inherent `json()` methods to the same foreign type with no
  diagnostic, and the ambiguity would have surfaced as a method-resolution failure far from either.
- **Security/soundness impact:** none.
- **Owning gate:** compiler track, AS1b-iii (WP-ARCHITECTURE-STABILIZATION).

---

## DEV-184 — three of the four JSON escapers emitted invalid JSON (CLOSED, AS5-a)

- **Rule:** RFC 8259 §7 — the characters that MUST be escaped inside a JSON string are the quotation
  mark, the reverse solidus, and **the control characters U+0000 through U+001F**.
- **Status:** CLOSED. All three repaired; AS5-c replaces them with one shared authority.
- **Found by:** AS5's opening inventory. `AS0-MANIFEST-STRICTNESS-AUDIT.md` compared the two JSON
  *parsers*; this is the emit side, which that audit did not cover.

| Authority | Escaped | Left raw |
| --- | --- | --- |
| `diag.rs::escape_json` | `"` `\` `\b` `\f` `\n` `\r` `\t`, all C0 as `\u00xx` | — (correct) |
| `lsp/protocol.rs::escape_json_string` | `"` `\` `\n` `\r` `\t` | 29 C0 controls |
| `onnx/verifier.rs::escape_json` | `"` `\` `\n` `\r` `\t` | 29 C0 controls |
| `bin/stark.rs::json_escape` | `"` `\` `\n` | 31 C0 controls, **including TAB** |

- **Demonstrated instance.** `stark doctor --json --root "<path containing a TAB>"` — a legal POSIX
  path — emits a raw U+0009 inside the `install_root` string. A standard parser refuses the
  document: `Invalid control character at: line 3 column 134`. The command advertises
  machine-readable output and produces something no conforming parser accepts.
- **User impact:** any tool consuming `stark doctor --json` fails on such a path, with a parse error
  that points at the compiler's output rather than at the path. On the LSP surface the raw control
  goes onto the wire inside a JSON-RPC message; a lenient client tolerates it and a strict one drops
  the message or the connection.
- **Why it survived:** `GATE-C8-CLOSURE.md` §4 records that C8's protocol validation compared
  **verdicts, not values**. DEV-182 — the LSP parser decoding every escaped non-BMP character to the
  empty string — passed that same evidence. "The LSP protocol suite is green" does not establish
  that what goes on the wire is valid JSON, and this packet's dependencies section says so.
- **Security/soundness impact:** none to the compiler. On the protocol surface, an unescaped control
  character in an attacker-influenced string is a message-framing hazard for a lenient client, which
  is why AS5 exit criterion 5 routes parsing decisions through CE9 review.
- **Repair:** each escaper now escapes every C0 control as `\u00xx`, matching `diag.rs`'s already
  correct implementation. `tests/as5_json_escaping.rs` fails against the previous code — it names
  the exact code points each one leaked — and passes after.
- **Owning gate:** compiler track, AS5-a (WP-ARCHITECTURE-STABILIZATION).

---

## DEV-185 — the JSON layer decoded every number to `f64`, losing the value the input denotes (CLOSED, AS5-b/c)

- **Rule:** RFC 8259 §6 constrains a number's *grammar* and explicitly sets no range or precision
  limit. The governing lesson is DEV-182's, recorded in `GATE-C8-CLOSURE.md` §4: **parsing
  successfully is insufficient; the returned value has to be the value the input denotes.**
- **Status:** CLOSED. `JsonValue::Number` carries a `JsonNumber` preserving the exact lexical value;
  conversion to `i64`/`f64` is an explicit consumer decision that can refuse.
- **Found by:** AS5's review of the shared data model. The two `JsonValue` enums being textually
  identical established that no reconciliation was needed — it did **not** establish that the
  representation was adequate, and it was not.

**Measured on the code before the repair** (`examples/as5_number_probe.rs`):

```text
input      9007199254740993          ← 2^53 + 1, an ordinary 64-bit request id
re-emitted 9007199254740992          ← CHANGED
as_i64     Some(9007199254740992)
1.5 as_i64 Some(1)                   ← truncated, not refused
      01 parses: true                ← leading zero is not JSON
      1. parses: true                ← naked decimal point is not JSON
   1e400 parses: true                ← decodes to infinity
```

- **User impact, in severity order:**
  1. **A JSON-RPC request identifier can change value between arriving and being answered.** `id` is
     correlation identity; a response carrying `…992` answers a request that was never sent. This is
     more serious than DEV-184, because a client cannot detect it — both documents are well-formed.
  2. `as_i64()` performed `n as i64`, so `1.5` became `1`. An integer-typed protocol field accepted
     a fractional number silently instead of refusing it.
  3. `JsonValue::Number(f64)` could hold NaN or an infinity, neither of which has a JSON textual
     form — the type could represent a document that cannot be serialized, with the failure
     surfacing at emit time far from whatever constructed it.
- **Security/soundness impact:** none to the compiler. On the protocol surface, id confusion is a
  correlation hazard rather than a memory-safety one.
- **Repair:** `JsonNumber` keeps the input's exact lexical form. The parser validates the RFC
  grammar — ASCII digits only, no leading zero, no bare `+`, no naked decimal point, and only the
  four JSON whitespace characters around it — and preserves the text. `as_i64` succeeds only for a
  raw integer literal; `as_f64` refuses a value it cannot represent finitely; `from_f64` refuses
  NaN and the infinities, so a non-JSON number is unrepresentable rather than caught late.
- **Deliberately not done:** arbitrary-precision arithmetic. The shared layer's job is to preserve
  what the document said and let each consumer state the numeric type it requires.
- **Owning gate:** compiler track, AS5-b/c (WP-ARCHITECTURE-STABILIZATION).

### Adjacent, NOT repaired here — the LSP request-id surface

The JSON layer is right to make `JsonNumber::as_i64` accept only what was *written* as an integer
literal: `1e3` returning `None` is a deliberate consumer decision, not a limitation. But
`lsp/protocol.rs` converts every id through `as_i64()` and models both halves as `id: i64`, so the
protocol layer accepts only that subset:

| Request id form | Today |
| --- | --- |
| plain JSON integer that fits `i64` | accepted |
| string id (JSON-RPC 2.0 §4, LSP both permit it) | **rejected** |
| other exact spellings of the same number (`1e3`, `1.0`, `+`-free variants) | **rejected** |
| integer outside `i64` | **rejected** |

Not a JSON-authority defect, and not AS5's to fix — covering it would turn a parser consolidation
into an LSP redesign. The eventual shape is almost certainly

```rust
enum RequestId { Number(JsonNumber), String(String) }
```

**echoing the id exactly as received** rather than interpreting it, which is what JSON-RPC requires
of a server. Recorded here so it is not lost; it needs its own packet.

---

## DEV-186 — the LSP transport allocates an unbounded `Content-Length` before parsing (OPEN)

- **Status:** OPEN, registered rather than repaired. Found during AS5's CE9 review of the JSON
  nesting limit; **not AS5's to fix** — see "Why this is not AS5" below.
- **Where:** `src/lsp/server.rs:60-63`.

```rust
if let Some(content_length_str) = headers.get("Content-Length") {
    if let Ok(content_length) = content_length_str.parse::<usize>() {
        let mut content = vec![0u8; content_length];   // ← no bound
        reader.read_exact(&mut content)?;
```

- **Symptom:** a peer that advertises `Content-Length: 9999999999999` causes an allocation of that
  size **before any byte of the body is read and before the JSON parser is reached**. On a 64-bit
  host the request either aborts the process on allocation failure or drives it into swap.
- **Cause:** the framing layer trusts a header field. `usize` parsing bounds the *number*, not the
  *allocation*, and `read_exact` into a pre-sized buffer commits the memory first.
- **Why AS5's `MAX_DEPTH` does not cover it:** the two limits protect different things and belong to
  different authorities.

  | Limit | Protects | Owner |
  | --- | --- | --- |
  | `json::MAX_DEPTH` (128) | the stack, against recursive descent | the shared JSON parser |
  | `Content-Length` cap | total allocation, before parsing | the LSP transport/framing layer |

  A depth limit cannot help here: the parser never runs.
- **Why this is not AS5:** AS5 consolidates JSON *parsing and escaping* authority. Message framing
  is the transport's contract, and widening the packet to cover it would repeat the mistake the
  string-request-id note already refuses — turning a parser consolidation into an LSP redesign.
- **Repair shape, when taken:** a maximum message size checked at the framing layer **before**
  allocating, with a deterministic protocol error for anything larger, and incremental reads rather
  than one pre-sized buffer. It belongs with the request-id work in an LSP hardening packet.
- **Security/soundness impact:** availability only, on a surface that reads from a socket or a pipe.
  No memory-safety consequence — the allocation is safe Rust — and no compiler consequence: nothing
  outside the language server reaches this path.
- **Owning gate:** compiler track; awaiting an LSP hardening packet.

---

## DEV-187 — bound specialisation did not reach generic impls (CLOSED, AS3 Boundary 4)

- **Status:** **CLOSED.** Found by AS3 Boundary 4d's negative control on its first run; closed by
  passing the concrete `Self` from both engines. All four impl×member pairs now resolve through the
  shared specialiser.
- **The compiler defect was real and is fixed.** The *residual* after the fix was a defect in the
  control itself — see "How this was mis-diagnosed twice" below, which is the part worth reading.
- **Not a wrong-answer defect.** Programs still produce correct output; both engines fall back to
  their pre-existing scans, which is what they did before AS3. The defect is that the shared
  authority is bypassed exactly where it is most needed.

`impl<T> Describe for W2<T>` has `self_ty = Struct(W2, [Param("T")])`. Both engines pass the **bare
nominal head** `Struct(W2, [])` to `specialize_bound_callable`, the argument lists differ in length,
`unify_impl_ty_with` refuses, and the specialiser returns `None` — after which each engine silently
falls back:

| Engine | Falls back to |
| --- | --- |
| HIR interpreter | `find_method(nominal, name, trait_filter)` |
| MIR lowering | `find_impl_fn(nominal, name, …, bound_trait)` |

So for generic impls, Boundary 4c and 4d are **not in force**, and `find_method`/`find_impl_fn`
cannot be deleted while that is true.

- **Why the interpreter cannot fix this alone:** `Value::Struct { item, fields }` carries **no type
  arguments** (established by `AS3-DISPLAY-CHARACTERIZATION.md` §2.2). It has no concrete `W2<Int32>`
  to pass. The type is recoverable from the caller's generic frame via `concrete_runtime_ty`, which
  already substitutes through `typecheck::substitute_ty` — so the repair is threading that, not
  changing the runtime representation.
- **MIR does not have the limitation:** it carries the receiver's full `MirTy` with arguments and
  currently discards them at this call. Its repair is smaller.
- **Why this was invisible until now:** the program prints the right answers, so no behavioural test
  could see it. The negative control compares **resolutions**, not output, which is the only reason
  it surfaced — and it surfaced on the first run.
### Refined diagnosis (2026-08-07) — the call-site repair was necessary but NOT sufficient

Both engines now pass the concrete `Self` **including arguments** (`W2<Int32>`, not `W2`), using
`expr_types` plus `concrete_runtime_ty`, which substitutes through the active generic frame via the
shared `substitute_ty`. That change is correct on its own terms and is kept.

**The generic impl still does not resolve.** So the bare-nominal-head account was only half the
cause. **That diagnosis was WRONG and is retracted.** I predicted the index stored a degenerate head; a
probe inside `build_trait_impl_index` shows it stores the correct one:

```text
IDX self_ty=Struct(ItemId(1), [])            generics=[]        // impl Describe for A2
IDX self_ty=Struct(ItemId(3), [Param("T")])  generics=["T"]     // impl<T> Describe for W2<T>
```

So the parametric head is recorded correctly, `convert_hir_type` does resolve `T` to a `Param`, and
the remaining cause is **neither the call sites nor the index shape**. It is somewhere in the
specialiser's candidate loop or in how the test supplies the concrete `Self` — both still unexamined.

Recording the retraction rather than quietly replacing it: the previous entry asserted a cause from
"the repair had no effect", which is evidence that *something else* is wrong, not evidence of *what*.
That inference was unsound and produced a confident wrong answer in a permanent record.

- **Repair:** convert each impl's self type within its own generic scope when building the index,
  then re-check the caller side. Pinned meanwhile by
  `both_engines_resolve_a_bound_call_identically`, which asserts exactly the two non-generic
  resolutions succeed; when the repair lands, that count rises and the test demands this record be
  updated.
- **Owning gate:** compiler track, AS3 Boundary 4 (`WP-CALLABLE-USE-TOTAL.md`).

### How this was mis-diagnosed twice

Worth recording, because the failure was in reasoning rather than in code.

1. **Fix applied** — both engines pass the concrete `Self` with arguments (`5fb8811`). Correct, and
   it *was* the whole compiler-side defect.
2. **Control still showed 2/4.** Concluded the cause was index-side: `convert_hir_type` called
   outside the impl's generic scope. **Wrong** — a probe showed the index stores
   `Struct(W2, [Param("T")])` correctly. Retracted in `d749612`.
3. **Fixed a real latent bug anyway** — the specialiser's member lookup used `?`, abandoning the
   whole search when the first head-matching impl lacked the member, instead of `continue`. Correct
   in itself, but not this cause.
4. **Probed the actual inputs.** The *test* was passing `Struct(W2, [])` — the bare head — because
   its self-type mapping had been reverted when I restored an earlier pinned state. The control was
   lying about its own inputs.

**The lesson is step 2.** "The fix had no effect" is evidence that *something else* is wrong; it is
not evidence of *what*. Both wrong diagnoses came from inferring a cause instead of measuring one,
and the measurement that settled it — printing the values actually passed — was cheaper than either
inference.

A control that misreports its own inputs is worse than no control: it produced two confident wrong
conclusions, one of which reached a permanent record before being retracted.


## DEV-188 — a trait method's own generics were dropped at a bound call site [CLOSED at creation, AS3 Boundary 4, CD-386, 2026-08-07]

- **Rule:** 02-Syntax-Grammar.md permits a trait method to declare its own generic parameters, and
  03-Type-System.md's turbofish rule applies to a method's generics at the call site. Nothing
  restricts either to concrete receivers.
- **Defect:** `check_trait_member_call` converted the declared signature and never read
  `sig.generics`. At a call through a generic parameter's bound, the method's own generics were
  therefore never bound: the turbofish was discarded, no inference variable was created, and the
  argument check compared the caller's types against the type *parameter*.
- **Effect: every trait method mentioning its own generic parameter was uncallable through a
  bound.** Not mis-typed — uncallable. `fn g<T: Conv>(t: T) { t.to::<Int32>(1) }` on
  `fn to<U>(&self, x: U) -> U` reported `type mismatch: expected 'U', found an integer literal`,
  and no call site could satisfy it. The same function serves the `Self`-receiver path, so a trait
  default calling a generic sibling on `self` failed identically.
- **Scope, measured rather than assumed** (`examples/as3_method_args_probe.rs`): the only accepted
  shape was one where the method's generic appears **nowhere in its signature**. `U` in a
  parameter, `U` in the return type, and `U` in both were all rejected.
- **The concrete-receiver path was already correct.** WP-C4.7-8.4 binds these generics, validates
  the turbofish arity, and substitutes. This is the bound path being brought into line with a rule
  the language already had — not a new rule, which is why it closes at creation rather than opening
  a semantic question.
- **Repair:** `check_trait_member_call` binds `sig.generics` from the turbofish when present and
  from fresh inference variables otherwise, validates arity, substitutes into parameters and return
  type, and returns the resolved bindings. `CalleeSelection::Bound::method_args` — carried but
  always empty since Boundary 4 step 2 — is now populated from them. A core trait's contract
  (`ContractTy`) cannot declare method generics, so its empty list is an answer, not a gap.
- **Evidence:** `tests/dev188_bound_method_generics.rs`, 8 tests. Both halves of the repair
  mutation-tested independently: removing the binding fails 6 of 8; removing the arity validation
  fails the other 2.
- **How it was found, and the correction it forces.** AS3-DISPLAY-CHARACTERIZATION.md §5 recorded
  G2 as *accepted* — "method generics through a bound … yes". That measurement was **vacuous**: its
  fixture's `U` appeared nowhere in the signature, the one shape that happened to work. The
  characterization was written to justify *adding a field*, and it stopped measuring as soon as the
  program compiled. Probing the field's actual inputs before populating it is what exposed the
  defect underneath. §5 is corrected in the same change.


## DEV-189 — MIR's bound specialiser passed the bare nominal head [CLOSED at creation, AS3 Boundary 4, 2026-08-07]

- **Defect:** `specialised_bound_key` built `Self` as `Ty::Struct(nominal, Vec::new())`, dropping
  the receiver's type arguments, with a comment asserting "the index matches on the impl head".
  It does not: `impl<T> Describe for W2<T>` is indexed as `Struct(W2, [Param("T")])`, so a head with
  no arguments fails to unify on length and the specialiser returned `None` for **every generic
  impl**. MIR then fell back to `find_impl_fn` while the interpreter used the shared authority.
- **This is DEV-187 on the second engine.** The interpreter was repaired; MIR was not, and the
  control did not notice — see the note on the 4d control below.
- **Repair:** a partial `MirTy -> Ty` bridge (`mir::lower::checker_ty`), because lowering
  substitutes in `MirTy` while the impl index speaks checker `Ty`. Partial on purpose: `FnPtr` and
  `HostResource` return `None`, which means "do not specialise" rather than a fabricated shape.
- **The 4d control did not catch this, and that is its own finding.**
  `both_engines_resolve_a_bound_call_identically` calls `specialize_bound_callable` twice with a
  self type the TEST constructs. It proves the shared authority is deterministic given identical
  inputs — a real property — but it never checks that the two engines *supply* identical inputs,
  which is the divergence its name promises. The census in DEV-190 is what actually found this.

## DEV-190 — `self.m()` inside a trait default body published no callable use [CLOSED at creation, AS3 Boundary 4, 2026-08-07]

- **Defect:** the `Self`-receiver branch of `resolve_method` returned without publishing anything,
  so `self.id()` inside `fn twice(&self) -> Int32 { self.id() * 2 }` had no `CallableUse` and both
  engines fell back to a name scan.
- **Same class as AS3's missing third binding time.** `Self` is a parameter, the trait is known,
  the body is fixed only once an implementor is chosen — a `Bound` selection by the same argument
  that produced the variant.
- **Found by census, not by reading.** The MIR fallback was instrumented and the differential,
  operator, iterator, bound-identity and Display suites run. It fired ~60 times; consuming the
  `Static` selection reduced that to 2, both `self.id()` in a trait default. Publishing them took
  it to 0, which is what licensed deleting the fallback at all.

## DEV-191 — operators on a bounded generic parameter published no callable use [CLOSED at creation, AS3 Boundary 4, 2026-08-07]

- **Defect:** `publish_operator_use` returned early unless the operand was `Ty::Struct`/`Ty::Enum`,
  so `a == b` inside `fn same<T: Eq>(a: T, b: T)` published nothing.
- **Repair:** publish a `Bound` use against the Core trait when the parameter carries that bound.
  The signature comes from the `CoreTraitMethod` contract — the same table a user `impl` is checked
  against — not from `Eq::eq -> Bool` written in by hand.
- **Deliberately not published for `T: Num`:** arithmetic through a `Num` bound is compiler-known
  and primitives-only, so there is no user body for a call site to name. Pinned by
  `arithmetic_on_a_num_bounded_parameter_still_publishes_nothing`.
- **Surfaced only by deleting the fallback.** The `eq` fallback carried a comment stating it was
  "verified unreached ... by mutation". That evidence was real but covered two suites, and neither
  contained an operator on a bounded parameter; `over_acceptance_audit` did. **Unreached in the
  suites you ran is not unreachable** — which is the argument for deleting a dead fallback rather
  than annotating it.

## DEV-192 — `==` through an `Eq` bound silently used structural equality [CLOSED at creation, AS3 Boundary 4, 2026-08-07]

- **Rule:** 03-Type-System.md "Operators and Traits" — `==`/`!=` desugar to `Eq::eq`, `<` and
  friends to `Ord::cmp`. A type with its own impl must run that impl.
- **Defect:** in the HIR oracle both operator paths fell through when the selection was `Bound`,
  and the fall-through for `==` on a struct value is **structural `Value` comparison** (DEV-008).
  So inside `fn same<T: Eq>` a user's `impl Eq` was silently replaced by field-wise equality.
  `<` had no fallback at all and trapped with "invalid binary operation".
- **Measured at HEAD before the repair**, on a `Rec` whose `eq` compares `id` and ignores `tag`:

  ```text
  HIR-at-HEAD "false\nfalse\n"   (correct is "true\nfalse\n")
  ```

  The first answer is **wrong** — not a missing feature, a wrong result from the reference engine.
- **Why it stayed hidden:** every existing fixture had a user `eq` that AGREED with structural
  comparison, so the substitution produced the right answer everywhere it was exercised. A
  differential suite cannot catch two algorithms that coincide on all its inputs. The regression
  tests now use an `eq` that ignores a field and a `cmp` that reverses the order, so the two
  algorithms must disagree.
- **Repair:** `Interpreter::specialised_operator_callable` resolves the `Bound` use through the
  shared specialiser, taking `Self` from the published `self_ty` substituted through the active
  generic frame — the operand VALUES cannot supply it, since a runtime value carries no type
  arguments. Depends on DEV-191 having published the use.
- **Evidence:** `tests/as3_fallback_removal.rs`, 8 tests, all through the full differential harness.

## DEV-193 — a direct call to a known function published `FunctionValue` [CLOSED at creation, AS3 exit criterion 1, 2026-08-07]

- **Defect:** `free(1)`, where `free` names a known `fn` item, fell through the call-checking chain
  into the function-value branch and published `CalleeSelection::FunctionValue` — the selection
  meaning *the body is not knowable here*. It is knowable: the callee path published
  `Direct`/`Static(body)` immediately before.
- **Effect:** `free(1)` and `g(2)` produced **identical records at their call expressions**. A
  consumer reading the call site could not distinguish a direct call to a known body from a call
  through a function value — the exact conflation `CalleeSelection`'s three binding times exist to
  prevent. Nothing consumed it today, so nothing was observably wrong; it was a false statement
  waiting for a consumer.
- **Repair:** suppress the `FunctionValue` publication when the callee resolves to a `fn` item. The
  record for a direct call is the callee path's; a second, weaker one contradicting it is a
  duplicate, not extra information.
- **Found by `tests/as3_callable_use_exactness.rs` on its first run**, which is what that test is
  for: it derives expectations from HIR shape and `expr_types` rather than from the table under
  test, so it can see a record that exists but says the wrong thing.
- **One self-inflicted regression while fixing it**, worth recording: the first version used
  `return *ret` to skip the publication. `return` exits `check_expr` entirely, skipping the
  post-match bookkeeping that records `expr_types` — 2 lib tests and 35 `mir_differential` cases
  failed at once. An early return out of a function whose value is recorded by its *caller-side*
  epilogue is never a local edit.

## DEV-121 — CLASS CLOSED (AS3 work item 6 / exit criterion 5, 2026-08-07) — **WITHDRAWN 2026-08-08, see below**

Exit criterion 5 required *"a class-level evidence statement, not one regression case."* This is it.

### 1. The blind spot is closed

DEV-121 UPDATE 2 named why both instances — `String::bytes()` (CD-305) and `String::split()`'s item
(CD-340) — were found by user-facing packages rather than by tooling: `INV-VALUE-REP-001` checked
**`let` bindings**, and *both were reachable through a `for`-loop item, which is not a `let`*.

The invariant now runs at every position a local receives a value:

| Site | Before | Now |
| --- | --- | --- |
| `let` binding | checked | checked |
| `for`-loop item | **unchecked** — the shape both instances took | checked |
| call parameter | **unchecked** | checked |
| method receiver | **unchecked** | checked |

### 2. The extension is load-bearing, and that is measured rather than asserted

With `String::bytes()` mutated back to its DEV-121 behaviour (returning an owned `Value::Vec`), on a
program where the view reaches a parameter and never binds to a `let`:

```text
invariant wired at parameters   TRAP  "...holds an owned Vec... (DEV-121)"
invariant NOT wired (let-only)  OK    "3\n3\n"      <- defect completely invisible
```

The second row is the finding. Under the old coverage a broken producer yields a program that runs
and prints the right answer — no test and no user would ever see it. That is precisely how both
known instances reached packages.

**A first mutation pass wrongly suggested the extension was inert:** removing the new call sites left
the audit suite green. It did, because the audit exercises *correct* producers, and removing a
detector does not change correct behaviour. The control that means something pairs a **broken
producer** with a binding position the old check could not see. Recorded because "the mutation did
not bite" is a question, not a verdict — the same error made three times earlier in this packet.

### 3. The inventory cannot go stale

`tests/dev121_view_producer_audit.rs::every_view_returning_intrinsic_is_classified` scans
`core_method_signature` and requires **every** method arm mentioning a view type to be classified —
either exercised as a producer or explicitly listed as taking a view only / returning owned storage
deliberately. Adding a new `&[T]`/`&str` intrinsic without a decision fails the test.

The scan deliberately over-approximates (it flags parameter-position mentions too): an extra entry
costs a line in a table, a missed one is an unaudited producer, which is the defect class itself.

Audited producers: `as_str`, `trim`, `bytes`, `as_slice`, `substring` — each exercised bound by
`let`, passed as an argument, and (for `&str` items) as a loop item.

### 4. One language fact recorded, not a gap

`for b in view` where `view: &[UInt8]` is **rejected** — Core v1 does not make a slice iterable. So
there is no loop position for a slice view, and the loop coverage rests on `&str` items from
`split()`. Written down so this file is not later "completed" with a fixture that cannot compile.

### Status

**CLASS CLOSED.** The instances remain fixed (CD-305, CD-340); the detector now covers every binding
position; the inventory is enforced by a test rather than by a date. What remains uncovered is
stated rather than implied: struct fields, indexed slots, and values that never bind to a local at
all. Those need a place-oriented check, which is a different change with its own evidence — not a
silent exemption from this one.

## DEV-194 — a trait DEFAULT body reached by a non-`Static` route ran without its `Self` binding [CLOSED at creation, AS3 Boundary 4, 2026-08-07]

- **One shape, three routes, three separate repairs.** A trait default's body carries
  `Ty::Param("Self")` throughout. Whenever it is reached by a route other than an ordinary `Static`
  method call, something had to supply the `Self` binding — and nothing did:

  | Route | What was missing |
  | --- | --- |
  | bound call — `announce<D: Describe>(item: &D)` → `item.shout()` | the interpreter **discarded** the environment `specialize_bound_callable` returns |
  | bound call, MIR | `specialised_bound_key` used `fn_key_for_body` (impl members only), so a default body produced no `FnKey` |
  | qualified call — `<T as Tr>::m(&x)` | the checker published no selection for a default at all, then no `Self` once it did |

- **Effect:** `self.name()` inside the default failed with *"method 'name' not found at runtime"*,
  or MIR lowering refused the call outright.
- **The fallbacks had been hiding all three.** A name scan finds `name` on the runtime value's
  nominal without needing an environment at all, so a missing `Self` binding was invisible for
  exactly as long as a scan existed to paper over it. Deleting the scans did not cause these
  defects; it revealed them.
- **Two were found by CI, not by the unit suites** — `pkg/07-traits` in the external sample suite,
  and `c62b_fully_qualified_reaches_a_trait_default_body`. That is the argument for both gates: the
  sample suite exercises the interpreter on real programs, and the differential suite exercises
  shapes the samples do not reach (MIR refused a program the interpreter ran).
- **Repairs:**
  - `Interpreter::push_resolved_env` installs the environment the specialiser produced. A `Bound`
    call's environment cannot be published — the body is chosen only once `Self` is concrete.
  - MIR's `specialised_bound_key` uses `key_for_selected_body`, which reaches trait defaults.
  - `check_qualified_trait_call` falls back to `trait_default_member` when the implementor accepts
    the default, publishes the signature from the trait (`trait_member_signature`, with `Self`
    substituted), and publishes `Self` in the environment — which the checker knows here.
- **Honesty note on how the MIR half was found:** not by the probe's output. The probe edit I wrote
  to *observe* the failure also replaced `fn_key_for_body` with `key_for_selected_body`, and the
  symptom moved. I only established the real cause by diffing what I had actually changed.
- **Evidence:** `as3_fallback_removal::dev194_a_trait_default_reached_through_a_bound_gets_its_self_binding`
  — two implementors, one accepting the default and one overriding it, so a resolution that ignored
  `Self` and picked "the first impl declaring the name" prints the same text twice. Plus
  `native_c6_2_generics_traits` 20/20 and the external sample suite 39/39.

## DEV-195 — `Vec<CharsIter>::clear()` was refused by the MIR verifier (CLOSED by owner ruling, CD-387, 2026-08-07)

- **Rule:** a program the checker accepts and the reference engine executes must be compilable. An
  engine refusing it later is a divergence, not a language boundary.
- **Behaviour, measured end to end:**

  | Stage | Verdict |
  | --- | --- |
  | checker | accepts |
  | HIR interpreter | runs it, prints `0` |
  | MIR lowering | lowers it, emitting the fast `VecClear` |
  | MIR verifier | **rejects — MIR-0016** |

- **Cause: the two precise drop rules disagree.** `lower::ty_requires_drop_glue` answers `false` for
  `MirTy::Core(CharsIter)`; `verify::requires_drop_glue` answers `true`, because it opens with
  `MirTy::String | MirTy::Core(..) => true`. Lowering therefore takes the fast `VecClear` path, and
  MIR-0016 is precisely the guard on that path.
- **The mechanism, and why `String` is unaffected:** lowering emits `VecClear` **only** when it
  believes the element needs no glue. `Vec<String>::clear()` and `Vec<Vec<Int32>>::clear()` emit no
  `VecClear` at all and pass. So the disagreement puts lowering on one side of the guard and the
  verifier on the other, for exactly the element types they answer differently about.
- **Scope:** of the 14 measured disagreements, only `CharsIter` and `File` are `MirTy::Core` shapes
  anything constructs. `Vec<CharsIter>` is confirmed constructible. `Vec<File>` is not tested here.
- **User impact:** a valid program cannot be compiled natively. Not unsoundness — the refusal is
  conservative — but a user-visible engine divergence with no diagnostic explaining why the
  interpreter ran what the compiler refused.
- **Why it stays OPEN:** the repair is behavioural. Making the two agree changes the accepted
  program set, so it owes its own decision record under AS4 work item 5, and the decision — which
  side is right — belongs to the owner, not to a refactor. The evidence supports the verifier's
  `Core(..) => true` being an old conservative shortcut, but that reading is recorded, not adopted.
- **Evidence:** `tests/as4_vecclear_divergence.rs`, 3 tests. Pinned as a **characterization**: it
  asserts the current refusal, and is written so that accepting the program in future fails this
  test by name, forcing the decision record to be written rather than the behaviour to drift.
- **Found by:** AS4's lower-vs-verifier matrix, then driving the actual compiler rather than
  reasoning from the matrix. The matrix said "over-rejection is possible"; running it established
  that a real, constructible program is refused today.

### DEV-195 RULING (owner, CD-387, 2026-08-07)

```text
Core(CharsIter) requires_drop_glue = false.

A CharsIter is a borrowed cursor. Destroying it has no STARK-visible destruction
action and releases no owned language or provider resource.

Therefore Vec<CharsIter>::clear() may use VecClear.

The verifier's blanket MirTy::Core(..) => true classification is not authoritative
for CharsIter and must not reject that lowering.
```

**Lowering was right.** The evidence is semantic, not "one side accepts more": `CharsIter` is a
borrowed `&str` cursor yielding `Char` by value, the native runtime is a wrapper around
`std::str::Chars<'a>`, and the backend emits it as intrinsically borrow-carrying. It owns nothing
destruction could release. That also restores `VecClear`'s original contract, where the verifier's
predicate was meant to **mirror** lowering's precise rule.

**Repair:** `verify::requires_drop_glue`'s `MirTy::Core(..) => true` blanket is replaced by an
**exhaustive** per-`CoreType` match — `CharsIter => false`, every other variant unchanged at `true`.
Exhaustive because a new `CoreType` must then be classified rather than inherit a default, which is
the property a producer census cannot provide.

**`File` is deliberately excluded, and this is the important half.** It is the other reachable row
of the same disagreement with the **opposite** ownership: legacy Core `File` is an owning
`OwnedResourceHandle` released through the MIR/provider close path, and
`drop_plan::plan_for(Core(File))` is currently `Noop` — only a true `HostResource` gets
`HostResourceClose`. So `verify`'s `File => true` may be an accidental safety barrier, and removing
it could let the fast `VecClear` discard open handles. **It stays until `Vec<File>` is characterized
end to end.** See DEV-196.

**Evidence:** `tests/as4_vecclear_divergence.rs`, 4 tests — flipped from a refusal characterization
to an acceptance regression, which is the transition it was written for. The lower-vs-verifier
matrix independently dropped from 14 disagreements to 13, and its reachable list from
`[CharsIter, File]` to `[File]`.

## DEV-196 — legacy Core `File` has no drop plan; the feared barrier turns out to guard nothing (NARROWED, not a live defect)

- **The shape:** `lower::ty_requires_drop_glue(Core(File)) = false`, so lowering would take the fast
  `VecClear` path for `Vec<File>`; `drop_plan::plan_for(Core(File))` is `Noop`; only
  `verify::requires_drop_glue(Core(File)) = true` currently prevents that lowering from verifying.
  A predicate disagreement is holding a resource-lifecycle invariant.
- **Reachability, measured:** `mir_ty` **refuses** `Core(File)` outright — a bare `File` binding
  fails to lower with `type Core(File, []) (C4.5)` — so no ordinary program reaches it. It is
  produced only by provider binding (`ResourceBinding::LegacyCore`) in a **capability-declared**
  build. `Vec<File>` must therefore be characterized there, not through `starkc run`.
- **What to measure:** open or create a real `File`, move it into a `Vec<File>`, `clear()`, and
  record checker, HIR, MIR shape, verifier, and — decisively — whether any provider close appears
  in the MIR destruction path. If the fast `VecClear` is emitted, that is a resource-lifecycle
  defect in its own right; **do not weaken MIR-0016 for `File` to make two predicates agree.**
- **The conceptual question it may answer.** `VecClear`'s guard really asks *"can values of `T` be
  discarded by the fast clear without running any language-required destruction?"* For ordinary
  types that equals `!requires_drop_glue(T)`. Legacy Core `File` may be the counterexample: it does
  not participate in ordinary type-driven drop glue, yet discarding it is certainly not
  destruction-free. If the equivalence fails, AS4 has a **fourth** semantic question —
  `is_trivially_discardable` or equivalent — which would explain why `File` keeps resisting
  classification without either existing predicate being wrong.
- **Blocks:** merging the two precise drop authorities (AS4 item 1 for the drop rule).
- **Pinned by:** `as4_vecclear_divergence::core_file_is_not_reachable_through_ordinary_lowering`,
  which fails if `Core(File)` starts lowering through the ordinary path before this is resolved.

### DEV-196 — ANSWERED by measurement (2026-08-07)

The experiment the entry asked for was run: a **capability-declared package** (`filesystem`,
`stark build` — not `starkc run`), a real `File::create`, moved into a `Vec<File>`, then `clear()`.

```text
Vec<File> push + clear                          refused: type Core(File, []) (C4.5)
bare File bound by let                          refused: same
File matched but never bound                    refused: same
no File at all (control)                        BUILT
```

**`Core(File)` is unlowerable from source**, capability or not. `mir_ty` refuses the type, so the
`Ok(f)` binding alone is enough — `File::create`'s `Result<File, IOError>` payload cannot be
lowered. No source program constructs a `Vec<File>`, let alone clears one.

**Where `Core(File)` is used, destruction is explicit.** The WP-C7.8.4 provider path builds its MIR
by hand and closes the handle with `stark_file_close` (`HandleConsumed`), never through drop
planning. So `drop_plan::plan_for(Core(File)) = Noop` is **consistent with how `File` is actually
used**, not a hole — nothing relies on drop glue for it.

**Consequences, and they change AS4's plan:**

1. The feared resource-lifecycle defect is **not live**. The verifier's `File => true` guards a path
   nothing can reach, so it is neither load-bearing nor harmful.
2. The equivalence `fast_clear_safe(T) == !requires_drop_glue(T)` is **not tested by `File` either**,
   because `File` never reaches a `Vec`. So the hypothesised fourth predicate
   (`is_trivially_discardable`) has **no motivating case** and is not introduced.
3. Merging the two precise drop authorities is **no longer blocked by a safety risk**. What remains
   is that `Core(File)`'s classification is untested by any program — an argument for resolving it
   through the HostResource migration, not for keeping two authorities.

**`File => true` stays** in the verifier: it costs nothing, and removing it would be a change with
no evidence behind it in either direction.

**Pinned by** `as4_vecclear_divergence::dev196_a_vec_of_core_file_cannot_be_lowered_at_all`, which
fails loudly — naming the safety question — the day `Core(File)` starts lowering.


## DEV-121 — REOPENED (owner ruling, 2026-08-08)

**The 2026-08-07 "CLASS CLOSED" entry above is premature and is withdrawn.** No new DEV number:
same class, still open.

### The contradiction

That entry declared the class closed while, in its own final paragraph, listing struct fields,
indexed slots and non-binding values as uncovered. It promoted one claim into a larger one:

```text
what was proved     every known view-producing intrinsic is exercised, and the narrow
                    INV-VALUE-REP-001 now runs at four binding positions

what was claimed    the type -> runtime-representation defect class is closed
```

Those are not the same statement, and `WP-VALUE-REP-TOTAL` defines the class by the second.

### What is actually enforced

| | Function | Checks | Production callers |
| --- | --- | --- | ---: |
| the class's named mechanism | `interp::check_value_for_ty` | the total `Ty` → `Value` relation | **0** |
| its only wrapper | `interp::check_local_value` | ditto, per local | **0** — `#[allow(dead_code)]` |
| what runs | `interp::check_value_representation` | INV-VALUE-REP-001 only — a `&[T]`/`&str` binding must not hold owned `Vec`/`String` | 5 |

`check_value_for_ty` is an **executable specification with no production caller**. `37f07ca` records
that A4's enforcement attempt was reverted and the relation would stay unwired until callable-use
totality existed; `WP-VALUE-REP-TOTAL` still says "no boundary is wired". AS3 has now removed that
blocker, so the wiring is unblocked — not done.

### What the narrow work is worth

Kept, and reclassified. The producer audit and the extended `INV-VALUE-REP-001` are
**defence-in-depth evidence**, not the class authority:

```text
total relation           semantic enforcement          (unwired — this is the class)
view-producer inventory  producer-specific adversary   (real, and stays)
```

A producer/verifier relationship, not duplicate authority — provided the narrow rule stops being an
independent *semantic* rule once the total one is wired.

### Closure conditions

1. Wire `check_value_for_ty` as the **one** production relation; do not add a third validator.
2. **Inventory every value boundary first**, exact-set, in AS3's style: parameters, receivers,
   returns, propagation, `let`/`match`/loop bindings, assignment, field writes, element/index
   writes, aggregate fields, and inline values entering builtins and runtime operations.
3. Every applicable boundary calls the total relation, with the expected `Ty` taken from
   **checker-published** types and signatures — never reconstructed from the runtime value.
4. Retire `check_value_representation` as an independent semantic rule: delete it after migrating
   callers, or make it a trivial delegate. Campaign A must not exit with two `Ty`→`Value`
   authorities.
5. Remove `#[allow(dead_code)]` from the total path, so deadness is compiler-visible.
6. Mutation-prove across **several producer classes** — owned/view, reference/pointee, function
   value, aggregate/container — not just `String::bytes()`.
7. An exact boundary-inventory test, so a new HIR value-storage or transfer form forces an explicit
   decision instead of silently bypassing validation.

**The warning that matters:** replacing the current five calls with `check_value_for_ty` and
declaring victory would repeat the same mistake with a broader predicate. *The inventory of where
values cross typed boundaries is as important as the relation itself.*

### Effect

AS3 exit criteria **#3 and #5 are both FAIL** until this closes. `CAMPAIGN-A-EXIT-REPORT.md` §3.1
carries the detail; Campaign A's gate is held on this one item.

## DEV-197 — two dispatch paths ran a callee body with its generic environment missing [CLOSED at creation, A4 boundary wiring, 2026-08-08]

- **Rule:** AS3 exit criterion 2 — *implicit and explicit dispatch install the checker-selected
  generic environment in the HIR oracle*.
- **Defect, two paths:**
  - `Res::AssociatedFn` — `Stack::identity<T>(6)` called `call_callable` with **no**
    `push_callable_env`, unlike the `Res::Item` path beside it. The body ran with `T` unbound.
  - **function values** — `let f: fn(Int32) -> Int32 = identity; f(41)` discarded
    `FunctionValue::bindings`, which DEV-178 had put on the value *precisely because* a function
    value's instantiation cannot be recovered from the call site (`Ty::Fn` cannot say which one
    produced it). The body ran with `T` unbound.
- **Why nothing observed it:** no boundary consulted the callee's declared type, so an unbound `T`
  had nothing to be wrong against. Both bodies were `identity`-shaped and returned their argument
  unchanged, so the missing environment could not alter the answer either.
- **Found by:** wiring the **first** value boundary — `RepBoundary::Return`, checking a returned
  value against `callable_types[body].ret`. Both defects fired on its first run.
- **Repair:** the associated-fn path pushes `push_callable_env(callee, span)` like its neighbour;
  the function-value path pushes `push_function_value_env(&callee.bindings, span)`, a new installer
  for bindings that are already concrete and need no resolution.
- **Evidence:** `--lib` 538, `three_engine_differential` 109, `mir_differential` 132,
  `cross_package_generics`, `dev176_generic_callable_context`, `as3_callable_use_exactness`.
- **Bearing on AS3 criterion 2:** that criterion was recorded PASS in the first exit report. It was
  passing on the paths its tests covered; these two were not covered. The criterion is only sound
  once the boundaries that *consume* the environment exist — which is DEV-121's wiring.

## DEV-197 — UPDATE: a third site, in the class the audit flagged as high-risk

Packet 1's migration found a third dispatch path executing a body with **no** generic environment:
the **`Option`/`Result` combinators** (`map`, `and_then`, `map_err`, `unwrap_or_else`), which take a
`Value::Function` and called the raw executor directly.

Same violated invariant as the other two — *a callee body ran without its checker-selected
environment* — so it extends DEV-197 rather than taking a new number.

**Mutation-proved, unlike the first two.** With the combinators installing `InvocationEnv::Empty`
instead of the function value's captured bindings, `Some(5).map(wrap)` on `fn wrap<T>(x: T) -> T`
fails at the call. The environment is load-bearing there; this is not precautionary wiring.

**Also found: a duplicate installer I had introduced.** `push_function_value_env`, added while
wiring the Return boundary, did the same job as the pre-existing `push_captured_env`. Deleted;
`InvocationEnv::Captured` carries the `FunctionValue` and installs through the original authority.
Adding a second helper for an existing semantic job is exactly what AS4 spent its packets removing.

## DEV-197 — UPDATE: six more sites, found by making the environment a required parameter

Collapsing `call_user_method` into the invocation authority required every method call site to name
its environment. Six of the nine **installed none at all**:

```text
eval_binary            operator dispatch (Eq)      NO ENVIRONMENT
eval_binary            operator dispatch (Ord)     NO ENVIRONMENT
call_qualified_core_trait                          NO ENVIRONMENT
language_equal         container element Eq        NO ENVIRONMENT
next_for_iterator      Iterator::next              NO ENVIRONMENT
display_text/display_deep  Display::fmt            NO ENVIRONMENT
```

Every one executes a user body. All are paths AS3 Boundary 4 wired for **selection** and never for
**environment** — the two were separate concerns and only one had an authority.

Same violated invariant as the first three sites, so this stays DEV-197: *a callee body ran without
its checker-selected environment*. Nine sites now, across three discovery events, each found by
making something mandatory rather than by reading code:

| Found by | Sites |
| --- | --- |
| wiring the `Return` boundary | associated functions, function values |
| routing `call_callable` through the authority | `Option`/`Result` combinators |
| requiring an environment parameter on method dispatch | the six above |

Each is fixed by consuming the environment the checker already published, via `env_for_use` — one
mapping from `GenericEnvironment` to `InvocationEnv`, so no consumer invents its own.

**Why none produced a wrong answer.** These paths dispatch to `Display::fmt`, `Eq::eq`, `Ord::cmp`
and `Iterator::next`, whose bodies rarely mention their own generic parameters. The environment was
missing but unconsulted — the DEV-121 shape again, and the reason nine sites accumulated without a
single failing test.

## DEV-121 — Packet 1 addendum: the destructor receiver was a representation collision, not an exemption

Packet 1 collapsed the destructor's private executor into the invocation authority. That exposed a
disagreement the two executors had been keeping apart:

- `Drop::drop(&mut self)` publishes a receiver of `&mut Self` — that is what `callable_types`
  records, because it records the receiver *as the body binds it*.
- Destruction holds an **owned** value, and the old destructor executor bound that owned value
  directly to `self`.

So the body's `self` was a `Value::Struct` where the published type said `Ty::Ref { mutable: true }`.
Nothing observed it, because no boundary read the receiver — the DEV-121 shape exactly.

Only three repairs exist, and two are wrong:

1. exempt `Drop` from the receiver boundary — a hole in the invariant at the one place destruction
   makes it hardest to reason about;
2. let `&mut T` accept an owned `T` — that is not a narrower rule, it is the *deletion* of the
   distinction DEV-121 exists to enforce, and it would silently re-permit the owned-`Vec`-as-`&[T]`
   defect the class was opened for;
3. **materialize the receiver.**

**The repair.** `ReceiverSource` names where a body's `self` comes from before it is materialized —
`None`, `Place { kind, place }`, or `OwnedForDrop(value)` — and the authority, not the caller, turns
it into a binding:

| Source | Binding | Rule |
| --- | --- | --- |
| `Place { Value, .. }` | `take_place` | DEV-034: consume the resolved place, no re-evaluation |
| `Place { Ref \| RefMut, .. }` | `Value::Ref(place)` | DEV-070: a genuine borrow of the caller's place |
| `OwnedForDrop(v)` | `Value::Ref(backing)` | the owned value is moved into temporary storage in the **caller's** frame, so `self` is a real `&mut Self` |

The backing place is held by the authority and paired structurally with `BodyEpilogue::Destructor`,
which reads the (possibly mutated) value back out of it — `Drop::drop` may replace fields, and the
recursive field destruction that follows must see that. A `Destructor` epilogue reached without an
owned receiver is an `internal` invariant failure, not a fallback: the earlier
`unwrap_or(Value::Unit)` would have let a lost receiver erase the value and skip field destruction
in silence.

**Result:** the receiver boundary below passes for destructors with **no `Drop`-specific
exception**, which is the test of whether the materialization is real or is an exemption in
disguise.

**Evidence:** `--lib` 538, `three_engine_differential` 109, `mir_differential` 132,
`a3cd_generic_drop`, `c788_resource_lifecycle`, `c788_lifecycle_e2e`, `a11_host_resource`,
`as4_property_adversaries`.

## DEV-121 — Packet 2: the Receiver, Parameter and Propagation boundaries

Three more of the eleven `RepBoundary` sites are wired, all in the invocation authority and all
against `callable_types[body]` — the signature the checker published for the body being entered.

**One lookup, three boundaries.** The signature was previously fetched at the return boundary only.
It is now fetched once at the top of `execute_body`, before anything is bound, so a body cannot be
checked on the way out but unchecked on the way in. A missing signature stays an `internal`
invariant failure rather than a skip.

| Boundary | Read against | Note |
| --- | --- | --- |
| `Receiver` | `signature.receiver` | the receiver *as the body binds it*: `Self` / `&Self` / `&mut Self` |
| `Parameter` | `signature.params[i]` | replaces the `local_types` probe, which read the caller-visible local rather than the callee's declared contract |
| `Propagation` | `signature.ret` | a `?` that leaves the body **is** the body's return value (§6.5 requires the error type to match) |

`Propagation` mattered most: it was the one way out of a function that no boundary observed.
`Return` covered explicit returns and block values; `?` bypassed it entirely.

A published parameter count that disagrees with the callable's bound parameters is `internal` —
A3b forms both from the same declaration, so they cannot legitimately differ.

## DEV-198 — the published callee SELECTION was the one table field never grounded [CLOSED at creation, 2026-08-08]

- **Rule:** AS3 exit criterion 2 — dispatch installs the checker-selected generic environment — and
  DEV-121's requirement that a published expected type be *concrete* at a value boundary.
- **Defect:** `analyze` grounds every field of a published `CallableUse` — `environment`,
  `signature.receiver`, `signature.params`, `signature.ret` — and copied `selection` verbatim. A
  `CalleeSelection::Bound` carries three types: `self_ty`, `trait_args`, and the method's own
  `method_args`.
- **What leaked:** `t.to(1)` on `fn to<U>(&self, x: U) -> U`, called through a bound with **no**
  turbofish. `check_trait_member_call` resolves `method_args` before returning, but the integer
  literal that determines `U` is not defaulted until later — so the selection published
  `Infer(TypeVarId(1))`. `specialize_bound_callable` then built an environment binding
  `U -> Infer(1)`, and the boundary compared a runtime `Int` against an inference variable.
- **Found by:** DEV-121's `Return` boundary, then again by `Parameter`. DEV-188's own test asserted
  the *count* of published method arguments (1, correct) and could not see that the *type* was
  unresolved.
- **Repair:** `ground_selection`, an **exhaustive** match over `CalleeSelection` — `Static` and
  `FunctionValue` carry no types and pass through, `Bound` grounds all three. Exhaustive so a
  future variant carrying a type cannot be published ungrounded by omission.

## DEV-199 — an associated-type projection was unresolvable at a runtime value boundary [CLOSED at creation, 2026-08-08]

- **Rule:** DEV-121 — the expected type at a value boundary must be concrete.
- **Defect:** `fn first<T: Holder>(t: T) -> T::Item` publishes a return type of
  `Ty::Param("T::Item")`. The runtime environment binds `T`, not `T::Item`, so `substitute_ty` left
  it alone and `concrete_runtime_ty` reported an unsubstituted parameter — on a program that is
  correct, that MIR and native both execute, and that the checker fully resolved.
- **Why it is a boundary defect and not a language limitation:** once `T` is concrete the projection
  has exactly one answer, and the checker already computed it. `assoc_projections` — keyed by
  (implementing nominal, associated name) — is built in Pass 1 and was simply never published.
- **Repair:** publish `assoc_projections` in `TypeTables`, and give `concrete_runtime_ty` a second
  step: substitute, then discharge projections. The base is looked up in the active generic frame,
  its nominal selects the impl, and the checker's binding replaces the projection. A projection
  whose base is *still* parametric is left alone, so `ty_contains_param` still reports it — an
  unresolvable projection is a missing instantiation, not something to guess at.
- **One authority, not a third one.** The oracle consults the checker's table rather than scanning
  the impl set itself. MIR lowering already keeps its own `ProgramMeta::assoc_projections`; a third
  scan in the interpreter is exactly the duplication AS4 spent its packets removing.
- **Evidence:** `c62c_associated_types` 9/9, including `projection_inferred_from_argument` and
  `projection_used_by_value`, both of which were failing on CI at `1ea5a8b`.

## DEV-200 — `&mut [T]` refused the slice-view representation `&[T]` accepts [CLOSED at creation, 2026-08-08]

- **Rule:** INV-VALUE-REP-001 / §6.4 — a reference type's runtime representation.
- **Defect:** `value_matches_ty` answered `Ty::Ref { mutable: true, .. }` with a single line —
  `kind == ValueKind::Ref` — while `shared_ref_matches` accepts **two** representations for
  `&[T]`: `Value::Slice` (a view) and `Value::Ref`. `Value::Slice(place, lo, hi)` is a view *into a
  place*; writing through it writes to that place, so it is exactly as much a reference as
  `Value::Ref`, and it is what `&mut v[1..3]` produces.
- **Consequence:** the `Parameter` boundary rejected `sentinel__10_slice_mutation_through_view`, a
  correct program in the C6.5 corpus, as an oracle invariant failure.
- **Why this is a repair and not a weakening:** the asymmetry was omission, not rule. The mutable
  arm predates the slice-view representation. The widening is *narrow* — `ValueKind::Slice` is
  admitted only when the pointee is `Ty::Slice(_)` — so `&mut T` for every other `T` still demands a
  genuine `Ref`, and the owned-storage pairing DEV-121 was opened for remains refused.

## AS3 Packet 1 — correction to the record

`1ea5a8b` ("AS3 Packet 1 COMPLETE") was reported here on scoped local evidence and went to CI
**red**: `fmt, clippy, test` failed on all three Tier-1 platforms, and `C6.4 tier-1 qualification`
failed on Linux, with the three defects above (`projection_*`, the bound-method generic, and — once
`Parameter` was wired — the slice view). The defects are real and pre-existed the boundaries that
found them; the reporting was not. The evidence line for a packet is the CI run for its commit, not
the suites chosen to run locally.

## DEV-121 — Packet 3: one funnel for every local binding

`let`, a `match` arm's pattern bindings, and **both** `for` forms each did their own
`frame_mut().insert(local, Some(value))`. Two of the four checked anything.

`bind_typed_local(local, value, span, boundary)` is now the only way a value comes to rest in a
local: `check_local_value` against `local_types[local]`, then the insert. Each caller names its own
`RepBoundary`, because "a value entered a local" is not actionable and which of the four is.

**The finding: the USER-iterator `for` form checked nothing at all.** Two spellings of one loop
boundary — a built-in iterable and `Iterator::next` — and only the built-in one was covered. Same
shape as everything else this campaign has surfaced: the check was a thing a site could remember to
do, so the site that forgot was indistinguishable from a site with nothing to check.

A `let` with no initialiser is handled explicitly rather than skipped by accident. Definite
assignment (§4) guarantees no read precedes the write, so an empty slot is the correct state.

**Packet 7 fell out here.** With all four binding sites on the funnel, the narrow
`check_value_representation` — the `&[T]`/`&str`-only rule — had no callers left, and is deleted.
Its classification test now injects against `check_local_value`, the total relation: same
injection, same `InternalInvariant` assertion, on the rule that is actually load-bearing.

## DEV-121 — Packet 4: one funnel for every write into existing storage

`Assignment`, `FieldWrite` and `ElementWrite` are wired, all in `write_place`, which already had
exactly one caller. **Which write it is follows from the place's last projection**, not from the
caller: no projection is an assignment, a trailing `Field` is a field write, and a trailing `Index`
or `MapIndex` is an element write — a map entry is an element exactly as an indexed slot is, the two
differing in how the position is found rather than in what the write means.

**The earlier "no local to key on" framing was wrong.** The inventory had assumed a field or an
indexed slot could not be checked because neither has a local. Both are named by an EXPRESSION, and
`expr_types[lhs]` is the checker's answer for the target whatever the projection depth. A missing
entry is `internal`: the checker types every expression it accepts, so an absent one means the
tables and the tree disagree.

## DEV-201 — an operator on a GENERIC impl published an empty environment [CLOSED at creation, 2026-08-08]

- **Rule:** AS3 exit criterion 2 — dispatch installs the checker-selected generic environment.
- **Defect:** the operator publication wrote `environment: GenericEnvironment::Static(Vec::new())`
  unconditionally. For `impl Eq for Point` that is correct — there is nothing to bind. For
  `impl<T> Eq for W<T>` it is a body running with `T` unbound.
- **The information was already there and thrown away.** `operator_impl_member` calls
  `match_impl_type` to decide whether the impl applies at all, and discarded the substitution it
  produced. So the publication had no way to say what `T` binds to even though the function that
  chose the impl had just computed it.
- **Found by:** DEV-121's `Receiver` boundary, which read `callable_types[body].receiver` and got
  `&W<Param("T")>` with no `T` in scope — reported as a MISSED TRAP by `mir_differential`
  (`generic_impl_eq_dispatch_agrees`), because the oracle failed where MIR, which monomorphises,
  succeeded.
- **Repair:** `operator_impl_member` returns the substitution; the publication builds its
  environment with `impl_dispatch_bindings` — the **existing** builder, previously named
  `display_impl_bindings`, renamed because it was never Display-specific — and publishes the
  **instantiated** signature per AS3 Boundary 2 §3.4, so a consumer reads `&W<Int32>` rather than
  the declaration's `&W<T>`. `Display` gained the same signature instantiation in the same change;
  it already had the environment right, which is why only the operator path failed.
- **Not a second builder.** Adding `operator_impl_environment` beside `display_impl_bindings` was
  the first attempt and was withdrawn: two constructions of one list is the shape AS4 spent its
  packets removing.

## DEV-121 — Packet 5: the last boundary, and all eleven are wired

`AggregateField` is wired in `eval_struct_lit`, against
`aggregate_field_types[lit][field]` — a new published table holding each field's **declared** type,
instantiated for that literal, recorded by the checker at the same point it unifies the initialisers
against it.

**Why not `expr_types[init]`.** That is the type of the expression that produced the value, so
comparing the value against it would assert nothing — the tautology the inventory's type-source rule
exists to forbid. It also does not exist for a shorthand field: `W { v }` has no initialiser
expression. Keying the published map by field NAME covers shorthand with the same lookup.

Both aggregate forms are covered — struct literals and struct-like enum variants — because both go
through the same `check_field_initializers` call the publication sits beside.

**State of the inventory: 11 of 11 `RepBoundary` variants `Wired`.** The executable pin in
`dev121_boundary_inventory.rs` asserts the exact set, so this cannot drift from the code. Remaining
before DEV-121's class can close: `RepBoundary::ExpressionResult` (Packet 6, the twelfth variant and
the producer-side funnel at `expect_value`), the four-class mutation evidence, and AS3 #2's
requalification.

## DEV-121 — Packet 6: the producer side, and the twelfth boundary

The ruling's closure conditions name a boundary the enum had no word for: **inline values entering
builtins and runtime operations**. A value handed to `call_builtin` or a `RuntimeFn` never binds to
anything, so none of the eleven DESTINATION boundaries could see it. The inventory recorded it as a
gap rather than folding it into `Parameter` — which would have claimed coverage it did not have —
precisely so that closing it would be a visible change and not a redefinition.

`RepBoundary::ExpressionResult` is that word, and `expect_value` enforces it against
`expr_types[expr]`. `expect_value` is the right site for two independent reasons: it is the funnel
every produced value passes through (the census pinned 28 callers against 6 direct `eval_expr`
sites, none of which is a boundary), and it is the one producer path that still carries the
`ExprId`, without which there is no checker-published type to read.

**A propagation is deliberately NOT checked here.** `Flow::Propagate` parks its value and hands back
a placeholder `Unit` that the caller discards; the parked value's type is the enclosing function's
return type, and `RepBoundary::Propagation` reads it against that at the body boundary. Checking it
against the expression that produced it would compare it to the wrong type.

**Defence in depth, not a second authority.** The producer check and all eleven destination checks
consume the same `check_value_for_ty`. What the producer adds is coverage of values that never reach
a destination at all.

**12 of 12 `RepBoundary` variants are `Wired`.** `Class::Unwired` is now unconstructed — that is the
result, not dead code, and it is kept because a new boundary must be classifiable as unwired before
it is wired.

## DEV-121 — class evidence: four producer mutations, four forcing boundaries [2026-08-08]

Exit criterion 5 asks for a **class-level** evidence statement, not one regression case. Twelve
wired boundaries are not that statement on their own: a boundary that never fires is
indistinguishable from a boundary that is not running, and Packet 6 in particular found no defect
while firing on every expression the interpreter evaluates.

**The mutation is applied to a PRODUCER, never to `check_value_for_ty`.** Corrupting the predicate
would only show that the predicate detects an artificial mismatch. Corrupting a producer shows that
a real value, taking a real path, is stopped by the real funnel at the intended boundary.

| Class | Producer mutated | Forcing boundary |
| --- | --- | --- |
| 1 owned/view | `String::as_str` emits the owned `String`; `Vec::as_slice` emits the owned `Vec` | `ExpressionResult` |
| 2 reference | a `&self` receiver binds the pointee **by value** instead of `Value::Ref(place)` | `Receiver` |
| 3 function value | a function item coerces to a non-function value | `ExpressionResult` |
| 4 aggregate | a declared field receives a mis-represented value, injected *after* the producer boundary accepted it | `AggregateField` |

Each test asserts three things, and the middle one is what makes it evidence rather than
decoration:

1. the witness program runs **clean unmutated** — a detection on an already-broken program proves
   nothing;
2. exactly one producer is mutated;
3. the failure is `InternalInvariant` **and names the intended boundary** — a mutation caught by the
   wrong wire is a failure, not a pass.

### Two things the harness itself had to get right

**A thread-local does not reach the interpreter.** The first version armed the mutation in a
`thread_local!` and every mutation silently failed to arm — all five tests reported "NOTHING refused
it", which reads exactly like five inert boundaries. `run` executes the program on a *spawned*
thread with a larger stack (`on_interpreter_stack`), so the interpreter never saw what the test
thread set. A process-global would have armed correctly and been worse: the harness runs tests in
parallel, so an armed mutation would have corrupted whatever unrelated test was executing beside it.
The mutation is therefore a field on the `Interpreter`, scoped to exactly the one execution under
test, reached through a `#[cfg(test)]` `run_with_mutation`. **Nothing here compiles into a shipped
compiler**, so there is no runtime switch that could corrupt a real build.

**Class 2's first witness was wrong, and the relation was right to accept it.** `struct Holder { n:
Int32 }` is `Copy`-eligible, and §6.4 licenses the bare-value form for a `Copy` pointee — copying it
cannot consume, invalidate or destroy the referent, so the two representations are
indistinguishable to any observation the oracle can make. The mutation was not a violation there.
Only a non-`Copy` pointee (`struct Holder { name: String }`) makes the owned form observably wrong,
which is what the class is about. Recorded because the failure looked like an inert boundary and was
a correct acceptance.

### Also fixed here: the missing-metadata escape inside a wired funnel

`bind_typed_local` delegated to a helper that returned `Ok(())` when `local_types[local]` was
absent. So a language-level `let`/`match`/`for` binding whose entry went missing would have been
**skipped, silently, inside a wire the inventory reported as `Wired`** — structurally present,
inert in exactly the case that matters. The funnel now looks the type up itself and treats absence
as `InternalInvariant`; every caller is a language-level binding and the checker types all of them.

The permissive helper had **no remaining callers, production or test**, and is deleted rather than
renamed: a permissive path parked in the file is one a future funnel can pick up by accident.

## DEV-202 — the method-call path installed the callee's environment twice [CLOSED at creation, 2026-08-08]

- **Rule:** AS3 exit criterion 2, and P6 of its requalification — the callee's environment is
  installed by the authority, and is active for the callee's work and nothing else.
- **Defect:** `call_method` chose the environment, **installed it**, and then passed it to
  `call_user_method`, which routes through the invocation authority — which installs it again. Every
  method call therefore pushed the callee's instantiation onto the generic frame stack twice.
- **The redundancy is not the problem; the SCOPE is.** The outer guard was live while the CALLER's
  receiver place was still being resolved and materialized. Caller-side work running under the
  callee's instantiation is the same scope error P6 exists to prevent, running in the other
  direction — and the outer install predates the authority, so nothing had reviewed its extent
  since the extent changed.
- **Why it produced no wrong answer:** the two installations push identical bindings, so a lookup
  during the overlap resolves the same way it would have. It is a defect of the *architecture
  claim*, not of any current output — which is exactly the class AS3 #2 was reopened to find, and
  exactly why the requalification pins the number of installation points rather than asserting a
  table has an entry.
- **Found by:** the AS3 #2 structural pin `the_installer_is_the_single_environment_entry_point`,
  which requires exactly one call to `install_invocation_env` and found two.
- **Repair:** the call site chooses the environment; the authority installs it. That split is what
  the authority was created for.

## DEV-203 — an interpolated field consumed an expression result unchecked [CLOSED at creation, 2026-08-08]

- **Rule:** DEV-121 / `RepBoundary::ExpressionResult` — an inline value entering a runtime operation
  is read against `expr_types[expr]`.
- **Defect:** `f"{expr}"` evaluated its non-place fields with a direct `self.eval_expr(*expr)` and
  handed the value straight to the renderer. It never binds to a local, so no destination boundary
  saw it; it never passed `expect_value`, so the producer boundary did not see it either. **The
  only construct in Core v1 that was invisible to all twelve wires.**
- **Why the census did not catch it:** the direct-`eval_expr` pin asserted `direct <= 8`, and there
  were six. A bound with slack is not a census. It is now an exact count with every site classified
  by name — funnel, checked consumer, or flow-through — so a seventh forces review.
- **Repair:** the `Flow::Value` arm calls `check_expr_value`. The other arms are unchanged: control
  flow leaving an interpolation carries no value that comes to rest there.
- **Mutation-proved in both directions:** with the repair removed,
  `an_interpolated_field_is_a_checked_expression_result` fails — `f"{s.as_str()}"` with the view
  producer emitting owned storage rendered happily before, and is refused now.

## DEV-204 — a missing instantiation silently produced a function value with no bindings [CLOSED at creation, 2026-08-08]

- **Rule:** DEV-178 — a function value carries the instantiation it was created with, because
  `Ty::Fn` records only the signature and cannot say which instantiation produced it.
- **Defect:** `capture_function_value` answered *any* missing `callable_instantiations` entry with
  `FunctionValue { item, bindings: Vec::new() }`. That is DEV-178's defect written as a fallback:
  absence meant both "this function has no generics" and "the publication is missing", and the
  second is unrecoverable downstream by construction.
- **Repair:** the two meanings are separated by information already in hand — whether the item
  *declares* generics. None: an empty binding list is semantically proven. Some: `InternalInvariant`.
- **Behaviour-neutral on the whole suite**, which is the correct outcome and worth stating: no
  reachable program today coerces a generic function without a published instantiation. This closes
  a latent hazard rather than a live bug, and the hazard is the kind that only becomes reachable
  once someone adds a coercion route.
- **Found by:** the final audit's §5 fallback census, searching for
  `FunctionValue { …, bindings: Vec::new() }` outside sites where emptiness is proven — §8 names
  that construction explicitly as a thing to look for.

## DEV-205 — `IOError::Other(msg)` bound a payload the checker never typed [CLOSED at creation, 2026-08-08]

- **Rule:** AS3 — every boundary reads a checker-published type, and the published type is an
  answer.
- **Defect:** the builtin-variant arm of the pattern checker handled `Some`, `Ok` and `Err` and
  nothing else. `IOError::Other(msg)` therefore never had its sub-pattern checked: the binding
  received **no `local_types` entry at all**, and every use of it was published as `Ty::Error`.
- **The program ran and printed the right answer.** The interpreter binds by position and does not
  consult the tables to do it, so nothing observed the gap for as long as nothing read them. This is
  the DEV-121 shape relocated into the checker: metadata that is wrong rather than absent, on a
  program that works.
- **Found by:** Packet 6's `ExpressionResult` boundary, on CI — `expected Error, found String`. It
  was invisible to every local sweep because those sweeps were stopped before reaching
  `phase4e_math_random_io`.
- **Repair:** the arm now maps `(IOErrorOther, Ty::Core(IOError, _))` to the `String` payload the
  constructor's own signature already declares.
- **Forcing control, deliberately general rather than a regression pin:** `audit_published_types`
  asserts that a program the checker accepts with **no diagnostics** publishes no `Ty::Error` in any
  expression or local type, across eight witness families. Any future construct the checker accepts
  without understanding fails there, whether or not it happens to execute correctly.

## DEV-206 — an unsized slice type reaches a value boundary [CLOSED at creation, 2026-08-08]

- **Rule:** INV-VALUE-REP-001 / §6.6 — a type's permitted runtime representations.
- **Defect:** `v[0..2]` is published as `[Int32]` — the *unsized* slice type, not `&[Int32]` — and
  Core v1 lets that expression be used directly (`println(values[0..2])` in the Gate 3 core-min
  example). The relation had no arm for `Ty::Slice` as a standalone value type, so it refused the
  only representation such an expression can have.
- **Repair:** `Ty::Slice(_)` accepts `Value::Slice`, the place-backed view. Same reasoning as
  DEV-200, and it does **not** weaken the pairing DEV-121 exists for: an owned `Value::Vec` behind
  `[T]` stays refused.
- **Recorded outside Campaign A:** whether the checker *should* publish `&[T]` for a range index is
  a language-semantics question. It changes what the expression means, not whether the oracle's
  representation of it is valid, so it is not a Campaign A invariant and is not resolved here.
- **Found by:** Packet 6's `ExpressionResult` boundary, on CI, via `gate3_execution`.

## DEV-206 — REVISED: `Display` accepted an unsized slice place and rejected its borrowed view [CLOSED, owner ruling 2026-08-08]

**The first two diagnoses were wrong, and both are recorded because each would have removed the
symptom by deleting a rule stated on purpose.**

The language model is not in question:

```text
v[0..2]    : [T]     an unsized PLACE expression
&v[0..2]   : &[T]    the runtime-capable slice view
```

- **Withdrawn repair 1** — widen the relation so `Ty::Slice` accepts `Value::Slice`. That conflates
  the unsized pointee type with a runtime view and weakens exactly the distinction DEV-121 protects.
  `unsized_and_non_runtime_types_permit_nothing` states the rule deliberately, alongside the
  identical one for `str`.
- **Withdrawn repair 2** — publish `&[T]` for a range index. The indexing expression *is* a place of
  unsized type; borrowing it is what produces the reference. The change made `&v[0..2]` a double
  reference and broke two lib tests, five differential cases and Gate 3. That breakage is expected,
  not evidence.

**The actual defect was in `Display` eligibility, whose polarity was reversed:**

| Type | Before | After |
| --- | --- | --- |
| `[T]` | Display **accepted** | rejected — unsized, never a value |
| `&[T]` | Display **rejected** | accepted iff `T` is displayable |
| `[T; N]` | accepted | accepted — an array *is* a value |

`[T]` and `[T; N]` shared one arm, which is how the unsized form was blessed. Separating them is
the whole repair. `&mut [T]` is deliberately **not** broadened: DEV-206 is the `[T]`/`&[T]`
contradiction, and nothing in the standard rules currently implies the exclusive form.

The fix is in the canonical eligibility predicate, not in `println` — PRINT-DISPLAY-001 says
printing is ordinary `Display` resolution, not a syntax hook, so interpolation and every other
`Display` consumer inherit the same answer.

**Corpus edit, recorded so AS3 #4's evidence does not look like the corpus silently moved.**
`examples/gate3/05_core_min.stark` line 12: `println(values[0..2])` → `println(&values[0..2])`. The
example encoded an invalid program that was accepted only because of this defect; the new
representation boundary is what exposed it. Output is unchanged (`[40, 2]`).

**Spec clarification** (not a semantic change): PRINT-DISPLAY-001 gains clause 10, stating that a
slice is observed through a reference, that `&[T]` has the standard slice `Display` for a
displayable `T`, and that bare `[T]` does not. Generated spec regenerated.

**Evidence:** `dev206_slice_display` — 7 cases: bare rejected (naming `[Int32]`), borrowed accepted,
bound-then-printed accepted, non-`Display` element still rejected, `Display` element dispatched,
borrowed array slice, and a sized array still printable by value. The fourth is the control that
stops this being "every slice is now printable".

## DEV-207 — a slice view rendered structurally, ignoring the published Display plan [CLOSED at creation, 2026-08-08]

- **Rule:** AS3 Boundary 4 — the engine consumes the checker's published `Display` selection; there
  is no structural fallback (PRINT-DISPLAY-001 clause 9).
- **Defect:** `display_text`'s composite list omitted `Value::Slice`, so a slice fell through to
  `format_runtime_value` — the structural debug form. A `struct X` with its own `Display` printed
  `{n: 1}` instead of running `X::fmt`. The checker had published `DisplayPath([SliceElement])` for
  the position all along; nothing consumed it. `display_deep` had no `Value::Slice` arm either.
- **Why it was unreachable until now:** `&[T]` was refused by `Display` eligibility outright, and
  bare `[T]` was accepted *and* rendered structurally — so the same defect was present and could not
  be observed through a correct program. DEV-206's repair made the position reachable and the gap
  immediately visible.
- **Repair:** a slice is routed into the plan walk before the composite block, not added to it — a
  slice BORROWS its elements, so there is no owned composite to promote and nothing for the caller
  to drop. `display_deep` gains a `Value::Slice` arm that reads the base's elements and renders each
  through `DisplayStep::SliceElement`.
- **Evidence:** `a_slice_of_display_elements_is_accepted` asserts `[x]`, not `[{n: 1}]`.

## Campaign A forcing property — a place-only type may not escape into a value context [2026-08-08]

Owner ruling, 2026-08-08. DEV-121 asks *given a valid runtime type `T`, does `V` represent it*;
DEV-206 asked the question one step upstream — *should `T` have been allowed to reach a value
boundary at all*. Publishing `[T]` for `v[0..2]` is **correct**; letting the bare place escape into
`println(...)` is not.

```text
expr_types[expr] = [T]
        ├── &expr                    legal place -> reference conversion
        ├── assignment / projection  legal place use
        └── println(expr)            value required; [T] has no representation
```

**Derived, never enumerated.** `interp::ty_is_runtime_representable(ty, copy_items)` probes the
canonical relation with every `ValueKind`: "no representation satisfies this type" is the same
semantic decision the oracle makes at every boundary. A second list of runtime-representable types
beside `value_matches_ty` is exactly the duplicate authority this campaign removed.

To make the relation callable by both consumers, `value_matches_ty` and `shared_ref_matches` were
lifted out of `Interpreter` into free functions parameterised by the Copy set — the `&self`
receiver was never carrying anything else. Behaviour-neutral (`--lib` 558, `mir_differential` 132).

**A checker diagnostic was written first and WITHDRAWN.** It could not fire: every value context
already rejects place-only types, through four *different* rules — unification for a user call, the
unsized-local rule for `let`, `Display` eligibility for `print`, and the interpolation check.
Shipping an unreachable diagnostic would be speculative machinery, and the audit's own scope rule
forbids new abstractions without an identified bypass.

So the forcing function is the executable property `audit_value_context_representability`: every
value context is listed, each must **reject the place-only form and accept the reference form**,
and the `str` case pins that the rule is about the unsized *class* rather than slices. What was
missing was never the behaviour — it was anything comparing those four rules against each other or
against the relation. DEV-206 is the proof that mattered: one of the four had the rule backwards.

## DEV-208 — interpolation stripped the reference that makes a slice a value [CLOSED at creation, 2026-08-08]

- **Rule:** PRINT-DISPLAY-001 — printing is ordinary `Display` resolution, so every consumer
  inherits one answer.
- **Defect:** the interpolation check tests the type with references stripped. That is right for
  `fn render<T: Display>(v: &T)` — `Display::fmt` borrows anyway (STD-FORMAT-001), so a reference to
  a displayable type is displayable. It is wrong for `&[T]`: the pointee is **unsized**, the
  reference is not incidental, and stripping it turns the one displayable spelling into the one
  that is not a value at all. After DEV-206 repaired `println`, `f"{&v[0..2]}"` was still rejected.
- **Same defect, second consumer.** Both call `type_is_displayable`, so the predicate was already
  shared; the *stripping* is where they diverged. Repaired by not stripping a reference whose
  pointee is unsized.
- **Found by:** the value-context property's control — the half that requires every context to
  ACCEPT the reference form. Without that direction the property would have been satisfied by
  rejecting slices everywhere, and this defect would have been invisible.

## DEV-209 — a prelude `Option`/`Result` payload was not a place [CLOSED, owner ruling 2026-08-08]

- **Rule:** PAT-BIND-001 — when a scrutinee is read through a reference, a binding to a non-`Copy`
  component receives `&C`, borrowing the component **in place**; the referent is never moved. The
  rule is **uniform** over variant payloads, struct fields and tuple elements.
- **Defect:** the checker published `&String` for `match *r { Some(s) => … }`; the oracle bound an
  owned `String` — moving out of a borrow. The user-enum equivalent was correct, so this was the
  prelude path being the poor relation of the user path, the same shape as DEV-205.
- **Not a limitation to accept, and not the program's fault.** MIR executes the same program and
  prints correctly, so the **oracle was the outlier**. The old comment recorded the narrowing
  deliberately — a `Box<Value>` payload has no `Projection` to name — and that reasoning was sound
  only while nothing compared the two answers.
- **Three resolutions were considered and two rejected by owner ruling:** rejecting at the checker
  would narrow the *language* to fit one engine's value model; a named oracle limitation would make
  a first-party package oracle-ineligible over one missing projection. Neither is proportionate when
  the feature is normative, the checker supports it, MIR supports it, and the repair is local.
- **Repair:** the payload is slot-backed, exactly like every other component.

```text
Some / Ok / Err
 └── payload slot
      ├── Some(Value)   live
      └── None          moved out
```

  `Projection::VariantPayload(n)` names it — deliberately **not** `Index(0)`, because `Index`
  carries bounds-trap classification and an absent payload behind a matched discriminant is an
  invariant violation, not an index trap.

- **One discipline for 84 migrated sites, not 84 opinions.** `require_live_payload` for any
  operation needing a complete `Some`/`Ok`/`Err` — an empty slot there is `InternalInvariant`,
  because the ownership checker is what prevents reading moved storage, and inventing a runtime
  "use of moved value" category would describe a compiler defect as a language outcome.
  `take_payload`/`own_payload` only for operations that genuinely move: `?`, `unwrap`,
  `unwrap_or`, consuming combinators, owned pattern bindings, destruction.
- **A forcing function fired mid-migration:** adding the projection broke `write_place`'s exhaustive
  boundary match, which would not compile until a payload write was classified. It is `FieldWrite`
  — a positionally named component of an aggregate, not an element of a runtime-sized container.
- **Evidence:** `dev209_prelude_payload_place`, 13 cases — borrow semantics for `Some`/`Ok`/`Err`,
  the referent surviving the match, `Copy` payload still by value, an exclusive source still binding
  shared, prelude/user-enum **parity** for both shapes, consumption through
  `unwrap`/`unwrap_or`/`map`/`?`/owned match, `Display` for all four shapes, and lifecycle:
  destroyed exactly once, a moved payload not destroyed twice, a borrowed payload still destroyed by
  its owner.
- **Mutation control:** restoring the old by-value fallback reproduces
  `expected &String, found String` at `MatchBinding`; restoring the repair returns all 13 to green.
- **Application witnesses left unchanged.** `stark-url` is 20/20 and the external sample suite's
  `pkg/05-data-modelling` runs again. Rewriting valid code to avoid a compiler defect would turn
  "an application exposed a missing capability" into "an application learned a workaround".

## DEV-121 — CLOSED (owner ruling, 2026-08-08)

Reopened on 2026-08-07 because the first closure was premature: it proved that every known
view-producing intrinsic was exercised and that a *narrow* rule ran at four binding positions. That
is a regression case, not a class.

The class statement is now this:

| Closure claim | Evidence |
| --- | --- |
| one canonical `Ty` → `Value` relation | `value_matches_ty`, exhaustive with no permissive wildcard; the narrow `check_value_representation` is **deleted**, not merely unused |
| 12 of 12 runtime boundaries wired | `dev121_boundary_inventory`, whose progress pin asserts the exact set |
| an exact-set forcing inventory | `classify` is exhaustive over `RepBoundary`; a new variant does not compile until it is classified |
| no permissive typed-local metadata escape | `bind_typed_local` looks up `local_types` itself; absence is `InternalInvariant` |
| producer-side coverage | `RepBoundary::ExpressionResult` at `expect_value`, with the direct-`eval_expr` census exact and every site named |
| owned/view mutation | `String::as_str` / `Vec::as_slice` emitting owned storage → refused at `ExpressionResult` |
| reference mutation | a `&self` receiver binding the pointee by value → refused at `Receiver` |
| function-value mutation | a function item coercing to a non-function → refused at `ExpressionResult` |
| aggregate mutation | a declared field receiving a mis-represented value → refused at `AggregateField` |
| metadata-removal mutations | deleting any of the five published tables → `InternalInvariant` |

Each mutation modifies a **producer**, never the predicate: corrupting `check_value_for_ty` would
only show that the predicate detects an artificial mismatch. Each requires the witness to run clean
unmutated, and each requires the failure to **name the intended boundary** — a mutation caught by
the wrong wire is a failure, not a pass.

**What the class cost to close, stated because it is the useful part of the record.** Wiring the
boundaries found DEV-197 (nine dispatch sites), DEV-198, DEV-199, DEV-200, DEV-201, DEV-202,
DEV-203, DEV-204, DEV-205, DEV-206, DEV-207, DEV-208 and DEV-209. None changed a visible answer
before it was found. That is the defect class DEV-121 named: metadata and representation
disagreeing on programs that work.

## DEV-197 — CLASS CLOSED (2026-08-08)

*"A callee body ran without its checker-selected generic environment."* Nine sites across three
discovery events, each found by making something mandatory rather than by reading code — wiring the
`Return` boundary, routing `call_callable` through the authority, and making the environment a
required parameter of method dispatch.

Closed on the AS3 #2 requalification: seven dispatch classes (free generic function, generic
associated function, generic inherent method, operator into a generic impl, bound trait dispatch,
function value, nested generic call), each proved by **removing** the environment and requiring the
run to fail — with the mutation asserted to have been *reached*, and every witness answering
`size_of::<T>()` so the instantiation is load-bearing. Structural pins hold the shape: one body
executor, one caller of the raw executor, one environment installer, and an exhaustive
`InvocationEnv` match.

DEV-201 and DEV-202 were found after the class was believed closed, both by the pins rather than by
behaviour. They are recorded under their own numbers because each was a distinct defect, but they
are the same shape, and their discovery is the reason the requalification pins installation points
instead of asserting that a table is populated.

## DEV-210 — the borrow checker identified `Drop` by spelling, not identity [CLOSED at creation, 2026-08-08]

- **Rule:** CD-379 — a core trait is satisfied by RESOLVED IDENTITY. A user trait merely *named*
  like a core one does not satisfy it.
- **Defect:** `borrowck::local_has_drop` scanned the impl set and asked whether the written trait
  name `.ends_with("Drop")`. So `impl MyDrop for S` made `S` "implement `Drop`", and a legal partial
  move out of one of its fields was refused with E0100. **Valid Core rejected because a user trait's
  name ended in four particular letters.**
- **Three answers existed** to "does this nominal have a user destructor": the interpreter's (by
  `Res::CoreTrait(Drop)` — correct), MIR/native's `TypeContext::drop_impls` (correct), and this one.
- **The repair was not to fix the string test.** `copy_eligible_types` already computed exactly this
  set, by identity, and kept it private — so the borrow checker had written a second, weaker answer
  to a question the checker was already answering correctly. `nominals_with_destructor` is now
  published; `copy_eligible_types` consults it rather than repeating the scan.
- **Evidence:** `as4_destructor_authority` — a real destructor still refuses the move (the control),
  no destructor permits it, `MyDrop` and `DropLike` do not count, the published set contains exactly
  the nominal that declares a destructor, and enums are covered.

## DEV-211 — a matched component could move out of a `Drop` nominal [CLOSED at creation, 2026-08-08]

- **Rule:** OWN-PARTIAL-001 — *"Moving a field from a type that implements `Drop` is prohibited,
  because its destructor requires the complete value."*
- **Defect:** `match e { E::A(s) => … }` on an owned `impl Drop` enum was **accepted**, and the
  destructor then never ran: PAT-DROP-001 destroys the *unbound* components, so decomposing left
  nothing to run the type's own `Drop`.
- **Both engines agreed**, so this was a front-end conformance defect rather than an engine
  divergence. The checker had the rule for struct fields (`local_has_drop`, at a projection move)
  and never applied it to a matched component.
- **Repair:** `reject_moves_out_of_drop_scrutinee`, a sibling of the existing
  `reject_moves_out_of_borrow` walk — same prohibition, different reason, so the diagnostics can
  each say what they mean rather than sharing a mode flag.
- **Blast radius measured before implementing:** no first-party package uses `impl Drop`; three
  sample files do, none in this shape.
- **Evidence:** `as4_hostile_combinations` — the move is refused, and a `Copy` payload of the same
  enum still matches, so the rule does not read as "cannot match a `Drop` enum".

## DEV-212 — a `match` skipped a `Drop` nominal's own destructor [CLOSED, 2026-08-08]

- **Rule:** PAT-DROP-001 / OWN-PARTIAL-001 — a value consumed by a match is destroyed exactly once,
  and a type with its own destructor requires the complete value.
- **Defect:** `match e { E::A(n) => … }` on an `impl Drop` enum with a **`Copy`** payload runs the
  arm and **never runs the destructor**. Nothing moves out, so the value is complete; decomposing it
  into components is what skips the nominal's own `Drop`. Present in **both** HIR and MIR.
- **Attempted and withdrawn.** Destroying the value whole in `drop_unbound` caused a DOUBLE drop:
  the guard ran before the `Binding` arm and destroyed components that had already moved into their
  bindings. Reordering it after that check fixed the HIR side cleanly — `--lib` green, destructor
  running. The matching MIR change (`drop_whole_scrutinee_at_arm_end` in place of
  `consume_unbound_leaves`) did **not** take effect, and both halves were withdrawn rather than
  leave the two engines disagreeing.
- **The MIR half was in the wrong function**, and that is the finding worth keeping. It was written
  into `lower_arms_consuming`; an instrumented probe printed **nothing**, which revealed that enum
  matches take their own lowering route — `lower_enum_match` → `consume_variant_payload`. Two
  match-lowering paths exist, and a fix applied to one of them silently does nothing to the other.
  A test would not have found this: the code compiled, ran, and changed no behaviour.
- **Repair, both engines.** HIR: `drop_unbound` destroys a value whose nominal has a destructor
  whole, guarded AFTER the `Binding` check. MIR: `lower_enum_match` binds the pattern and registers
  a whole-scrutinee arm-end drop instead of consuming the payload piecewise.
- **Evidence:** `a_copy_payload_of_a_drop_enum_still_runs_the_destructor` requires HIR and MIR to
  agree on `7\ndtor\n`, alongside the DEV-211 case asserting the move is refused — so the pair
  distinguishes "destructor runs" from "cannot match a `Drop` enum at all".

## DEV-219 — the root application capability envelope was informational [CLOSED, 2026-08-09]

> **RENUMBERED from DEV-214 at integration, 2026-08-09.** This branch and Gate C10 both allocated
> `DEV-214` on the same day: here for the capability envelope, on `develop` for a
> left-associative operator chain that aborts the compiler (found by C10-B, repaired under OD-9).
> C10's landed on the trunk first, so **C10 keeps 214 and this record becomes 219** — the next free
> number after this branch's own `DEV-215..218`, which do not collide and are unchanged.
>
> The deviation itself is untouched: same finding, same closure, same evidence.

Provider manifests already rejected a binding outside their own declaration, but the root package
did not approve the transitive graph: deleting every root capability still built and performed host
I/O. WP-P1.6 now derives every provider function/resource reference conservatively across the graph
and enforces `derived ⊆ root envelope`. The diagnostic names capability, contributing package, and
interface. `gate2_package::root_capability_envelope_is_transitive_actionable_and_allows_spare_authority`
pins the empty-envelope failure, two-hop derivation, success case, and legal spare declaration.

## DEV-215 — shipped capability names diverged from the ratified durable vocabulary [CLOSED, 2026-08-09]

The implementation serialized transport/provider names (`filesystem`, `process.env`, `tcp`, `tls`,
`random`) while the ratified v1 format specifies authority roles. First-party manifests, provider
metadata and tests now use vocabulary v1; filesystem read/write and network client/listen are split,
and `capability_vocabulary: 1` is recorded in authority-bearing manifests and lockfiles. The
normative mapping is `STARKLANG/docs/spec/packages/capabilities.md`.

## DEV-216 — `stark check` did not report generated-Rust surface gaps [CLOSED, 2026-08-09]

`v[i] = value` typechecked and ran in the interpreters but failed only at `stark build` because
`RuntimeFn::VecReplace` is not emitted. The backend now owns an exhaustive `native_support` table;
adding a runtime function cannot compile without classifying it. `stark check` reports known
exclusions as W0106, and `stark check --target-native` rejects them as E0106 with the construct and
work package. VecReplace itself remains scheduled under WP-C6.3b.

## DEV-217 — external path dependencies were refused without a supported acquisition route [CLOSED, 2026-08-09]

The resolver confined paths to the root package's parent, forcing external authors to copy
first-party packages. Canonical external relative/absolute paths are now accepted. `stark.lock`
records `path:<canonical directory>` and a deterministic content hash; the package test proves both.

## DEV-218 — divergence was lost in value typing, definite assignment, and execution [CLOSED, 2026-08-09]

A block ending in `return` was typed as Unit in a value arm; an `if` branch that returned still
contributed an uninitialized path; repeated `let _` declarations collided. Blocks now produce `!`
when their reachable path diverges, flow joins ignore diverging branches, `_` is not published as a
binding, and both HIR/MIR propagation preserve a return from a value-position arm. Three exact
programs agree across HIR, MIR and native in `three_engine_differential`.
## DEV-213 — CLOSED (C10-P, 2026-08-09). The cache is invalidated per PACKAGE, not per URI

**Repaired under Gate C10's C10-P packet (OD-4, CD-395).** The deviation's first heading, above,
stands unedited: this file is append-only and that entry was correct when written.

### What was wrong

`ServerState::compilation_cache` is keyed by URI and each value owns a **whole-package**
`ProjectAnalysis`. Three facts composed into a wrong answer:

```text
one full ProjectAnalysis cached PER OPEN URI
update_document removed ONLY the edited URI's entry
handle_workspace_symbol merges symbols from EVERY cached analysis
```

So a rename in `child.stark` left `main.stark`'s analysis — which describes the *whole package,
including `child`* — carrying the old name, and `workspace/symbol` answered with both.

### The repair

`CompilationResult` gains `package_root: Option<PathBuf>`, stamped at compile time from the
manifest the analysis was built against. `ServerState::invalidate_package_of` drops the URI's entry
and every sibling entry sharing that package root. It is called from `open_document`,
`update_document` **and** `close_document`, because all three change the overlay set that the
package's analysis is computed from — not only the edit path where the defect was demonstrated.

Two deliberate choices, recorded because each is the kind of thing a later reader will question:

- **The package is read from the cache, not the filesystem.** The entry being invalidated already
  recorded which package it analysed, so an edit stays a pure in-memory operation. A URI with no
  cached entry has no siblings to find — the correct answer, not a missed case, because nothing
  stale can exist for a package this server never analysed.
- **`None` never matches `None`.** Single-file analyses carry no package root, and two unrelated
  loose files must not invalidate each other.

`package_root_for_document` is one function with two callers rather than two copies of "which
package is this URI in" — a second copy is exactly the duplicated-authority shape `AS8-DA-*`
catalogues.

### Evidence, including the negative control

`as8_editing_one_file_leaves_other_uris_cached_analyses_stale` is **renamed to
`dev213_editing_one_file_invalidates_every_analysis_of_its_package` and its polarity flipped**,
exactly as AS8's own assertion message instructed. It is not deleted: what it pins is the same fact
either way — what `workspace/symbol` can see after a single-file edit.

**The test was proved capable of failing before its pass was believed** (Gate C10's binding rule,
inherited from AS8). With the sibling sweep disabled and single-URI removal restored, it fails with
the defect's exact signature:

```text
DEV-213: `alpha_symbol` was renamed and must not survive in ANY cached analysis of this
package ... got ["alpha_symbol", "renamed_symbol"]
```

The control was then removed and the restore verified byte-identical before the pass was recorded.

Extended beyond AS8's case: after the sweep, recompiling the sibling must answer with the new name
and only the new name — so the repair is shown to be an *invalidation* rather than a purge that
merely hides the stale entry by emptying the cache.

```text
cargo test --manifest-path starkc/Cargo.toml --lib lsp::     48 passed, 0 failed
cargo test --manifest-path starkc/Cargo.toml --lib          569 passed, 0 failed
cargo clippy --workspace --all-features --all-targets -D warnings   exit 0, zero warnings
cargo fmt --check                                            clean
```

### What this closure does NOT claim

`workspace/symbol` is now correct under the multi-file editing pattern AS8 demonstrated. **It is
not a claim that the LSP is correct**, and C8 is not reopened. `DEV-012` — seven advertised
features with protocol evidence only — is separate and remains open; and `GATE-C8-CLOSURE.md` §4's
standing limit still applies: protocol validation checks verdicts, not values, and DEV-182 survived
it.

**The standing qualification recorded when DEV-213 was filed is DISCHARGED.** Claims about
`workspace/symbol` correctness under multi-file editing no longer need to be stated as qualified,
within the bound above.

## DEV-214 — a left-associative operator chain aborts the compiler with a stack overflow (OPEN, found by C10-B, 2026-08-09)

**Demonstrated at HEAD.** `cargo run --manifest-path starkc/Cargo.toml --example c10b_repro -- 250`
aborts with `fatal runtime error: stack overflow` (SIGABRT). Not a diagnostic, not a panic with a
message — process death.

### What is wrong

The parser **has** a recursion guard: `MAX_DEPTH = 200` in `parser.rs`, reported as *"this code is
nested too deeply to parse"*. It bounds **syntactic nesting** — parentheses, blocks, calls — because
those are what recurse in a recursive-descent parser.

A left-associative operator chain does **not** recurse in the parser. `parser.rs` implements *"the
16-level precedence table literally (one function per level)"*, and each level is a `loop` that
folds operands iteratively. `1 + 1 + 1 + ...` therefore never increments `depth`, and the guard
never fires.

**The AST it builds is nonetheless `n` deep**, and every recursive walk downstream of the parser
descends it:

```text
parse + resolve             survive at n = 2000
+ typecheck                 overflows between n = 240 and n = 500
full analyze_project        the walk that dies first
```

> **The guard bounds the nesting the parser recursed through, not the depth of the tree it
> produced.** Those coincide for parentheses and diverge for operator chains.

### Severity — it is worse than the headline number

The threshold scales with the thread's stack. Measured with `examples/c10b_thread.rs`, macOS-arm64:

```text
8 MiB stack   a process main thread                       n = 240 OK,  n = 250 ABORTS
2 MiB stack   Rust's default for a SPAWNED thread, and
              what `cargo test` gives each test           n =  60 OK,  n =  65 ABORTS
```

~30 KB of stack per AST level. **On a default-stack thread, sixty-five `+` operators kill the
process.** That matters because the LSP analyses on a server thread and the interpreter runs on a
spawned thread — an embedding is on the low number, not the high one.

`1 + 1 + ...` is a stand-in for the shape, not the only instance. Any left-associative chain
qualifies: string concatenation, a long boolean condition, a numeric sum in generated or
machine-written code. Sixty-five terms is not an exotic program.

### Contrast, which is what makes this a GAP rather than an absence

```text
(((((...1...)))))  300 deep   ->  REJECTED, "this code is nested too deeply to parse"
1 + 1 + ... + 1     65 terms  ->  SIGABRT
```

The bounded-failure behaviour the robustness gate asks for already exists. It is simply not
reachable by this input shape.

### Impact

- **Compiler correctness:** none — no wrong answer is produced.
- **Robustness / availability:** a valid, ordinary program aborts the compiler. Under C10-B's gate
  (*no panic, no hang, bounded failure*) this is a **FAIL**, and it is the only one C10-B found.
- **Security (C10-C surface S13, denial of service):** a hostile or merely generated source file
  kills any process that compiles it. For a batch build that is a failed build; for a
  long-running LSP server it is a crash.

### Why C10-B did NOT repair it — and this is the disciplined answer, not reluctance

Every available fix changes something an autonomous session may not change:

1. **Count chain depth against `MAX_DEPTH`.** The effective limit becomes ~200 terms, so
   expressions of 200–245 terms **that compile today would start being rejected**. That is a change
   to the normative accepted/rejected program set — **CE1/CE2**, Charter §2.2, and C10 plan stop
   condition 5.
2. **Convert the recursive walks to an explicit worklist.** Correct, and a structural change to the
   type checker and index builders — far outside a qualification campaign (plan §3.2).
3. **Raise the stack, or run analysis on a large spawned thread.** An architectural decision about
   where the compiler runs, and it does not remove the cliff — it moves it.

**Owner decision required.** The choice between "reject deep chains cleanly" and "support them" is
a language-surface decision, and the first option needs a number that only the owner can set.

### Evidence

```text
starkc/tests/c10b_robustness.rs
  t9_dev214_operator_chain_depth_is_unguarded_and_the_safe_boundary_holds
      pins the SAFE side at 40 terms, below the lowest measured cliff, and pins the CONTRASTING
      300-deep paren nesting as REJECTED so the guard's existence is part of the record
starkc/examples/c10b_repro.rs      the failing side, as an example rather than a test: a stack
                                   overflow aborts the whole test binary
starkc/examples/c10b_thread.rs     the stack-size dependence
```

The failing side is deliberately **not** a test. It cannot be one: SIGABRT takes every other test in
the binary with it.

---

# OD-7 adjudication (owner ruling, 2026-08-09) — the eight unsettled statuses, and six entries this ledger never had

**Authority:** owner ruling OD-7, recorded during C10-A1/C10-B. C10-0 enumerated eight deviations
whose last heading did not settle their status, and six that were OPEN in `COMPILER-STATE.md` and
owned **no heading here at all**. Both are resolved below.

**Nothing historical is rewritten.** Every entry below is a NEW, dated heading; the originals stand
exactly as written, in this append-only file and in `COMPILER-STATE.md`.

## DEV-005 — OPEN, ACCEPTED RELEASE DEVIATION (OD-7, 2026-08-09)

`starkc check` permits warnings where `starkc run` refuses any diagnostic. A **usability/CLI-policy
inconsistency with no safety impact**, and the specification does not mandate the policy.

```text
status       OPEN
owner        a future bounded CLI-consistency packet
C10 repair   NO
C10-Q        PERMITTED as a named tooling deviation
```

**Condition attached by the owner:** *one current-head reproduction is required before C10-Q.* The
entry is old enough that an incidental later change may already have removed it, and a release must
not name a deviation that no longer exists.

## DEV-010 — CLOSED (OD-7, 2026-08-09). Stale ledger status, not a release deviation

The recorded defect was that hover, definition and references were **protocol stubs**. Gate C8
established compiler-derived semantic services, and those three features are precisely the ones the
owner interactively validated (`GATE-C8-CLOSURE.md` §2). The deviation describes a compiler that no
longer exists.

## DEV-011 — ACCEPTED-INDEFINITELY (OD-7, 2026-08-09). Not OPEN, and not "fixed"

Doc comments are trivia rather than AST/HIR metadata. **The entry itself records that no explicit
normative requirement demands otherwise.**

```text
status       ACCEPTED-INDEFINITELY
reason       an implementation/tooling representation choice; no current normative violation
reopen if    a future documented semantic or tooling requirement cannot be satisfied from
             trivia and reassociation
```

Recorded this way deliberately: C10 must not classify a representation preference as a conformance
defect, and must not claim it was fixed.

## DEV-020 — CLOSED (OD-7, 2026-08-09). Confirmed design

`pub use` of a private item exposes it: the visibility of a re-export is the visibility of the
re-export. Already recorded in C1 as confirmed design rather than a defect.

## DEV-021 — CLOSED (OD-7, 2026-08-09). Verified correct

Cross-package coherence checking was **verified working**. The entry records a verification, not a
continuing defect.

## DEV-083 — OPEN, ACCEPTED-DEFERRED (OD-7, 2026-08-09). Real, and it constrains the claim

A concrete position in an impl head cannot match a receiver type argument that is still unresolved.
Native planning carried this forward as a known front-end limitation and required **deterministic
rejection** rather than pretended support.

```text
status          OPEN
classification  supported-surface limitation
C10 repair      NO
release         permitted ONLY if explicitly listed
future owner    a bounded inference / method-resolution packet
```

**It constrains the Core v1 Compiler Stable claim; it does not block C10.**

## DEV-179 — DORMANT (OD-7, 2026-08-09). Not closed, and not counted as live

`MapIter`/`FilterIter` discard a generic callback's instantiation. **Unreachable while iterator
`map`/`filter` remains refused by `E0105`** — a feature-activation prerequisite, not an active
conformance failure.

```text
status       DORMANT
release      does not block the current supported scope
trigger      MUST be resolved before iterator map/filter becomes accepted
```

Not closed, because the hazardous implementation is still there. Removed from the live
current-defect count, because nothing can reach it.

## DEV-196 — CLOSED / RETIRED AS A LIVE DEFECT (OD-7, 2026-08-09)

Measurement showed legacy `Core(File)` is not ordinarily lowerable from source at all — even
binding the returned `File` is rejected before MIR lowering. The real provider path uses explicit
resource-close semantics, not this legacy `Drop` path.

**The reachability regression test is KEPT**, because it guards the premise:

```text
if Core(File) ever becomes ordinarily lowerable
    -> DEV-196's dormant hazard becomes relevant again, and that test is what will say so
```

---

# The six entries this ledger never had (OD-7 backfill, 2026-08-09)

C10-0 finding **F3**: these six were OPEN in `COMPILER-STATE.md` and owned no heading here, so a
mechanical C10-Q check reading this file alone would have missed six genuine deviations.

**Status is carried across unchanged, and the `COMPILER-STATE.md` records remain authoritative for
detail.** This backfill restores the structured ledger; it does not re-adjudicate anything.

## DEV-156 — `stark fmt` evicts member doc comments (OPEN; backfilled OD-7, 2026-08-09)

A doc comment on a struct **field** is relocated after the struct. Recorded in `COMPILER-STATE.md`.

## DEV-157 — the native backend has no representation for `MirTy::Never` (OPEN; backfilled OD-7, 2026-08-09)

`Err(_) => panic(..)` in match-arm **value** position has no native representation.

## DEV-159 — a native build can race its own dependency build (OPEN; backfilled OD-7, 2026-08-09)

Reported by an outside reviewer: a first native build of an HTTP program can race the build of its
own dependencies.

## DEV-160 — place-granular borrows, whole-value projections (OPEN; backfilled OD-7, 2026-08-09)

The borrow checker is place-granular (DEV-154); whole-slot borrows for disjoint projections remain
open. Guarded in CI by the `DEV-160 raw slot primitives under Miri` job.

## DEV-161 — an ambient `CARGO_TARGET_DIR` breaks every native build (OPEN; backfilled OD-7, 2026-08-09)

Cargo's default output is `<manifest dir>/target`, which is where the generated crate expects it. An
exported `CARGO_TARGET_DIR` redirects it and the build fails. **An operational trap for any session
that sets the variable globally** — including a mutation or coverage run.

## DEV-162 — reading through a whole-value accessor (OPEN; backfilled OD-7, 2026-08-09)

A read through a whole-value accessor on partially-moved storage. Sibling to DEV-158 (CLOSED) and
DEV-160 (OPEN).

## DEV-214 — REPAIRED under OD-9, with one criterion that cannot be met at MAX_DEPTH = 200 (2026-08-09)

**Owner ruling OD-9 authorised the bounded repair.** The entry above stands unedited.

### What was done

`ast::max_expr_depth` computes the deepest expression tree **iteratively** — a forward
dynamic-programming pass over the expression arena, iterated to a fixpoint rather than assuming the
parser's child-before-parent allocation order. A recursive measurement would have overflowed on
exactly the input it exists to reject.

`analyze_project` enforces `parser::MAX_DEPTH` — **the same 200, now `pub`; no second limit was
invented** — after parsing and before resolution, emitting `E0209` blamed on the deepest
expression's own span. `each_child_expr` has **no `_ =>` arm**, so adding an `ExprKind` variant is a
compile error rather than a silent hole.

**One thing the first attempt got wrong, recorded because it is the interesting part.**
`query::QueryIndex::build` walks the expression arena recursively and runs **unconditionally** —
errors do not stop it, because a stale-but-present index is what lets an editor answer while a file
is broken. So the guard fired, resolution and type checking were skipped, and the index then walked
the same deep tree and aborted anyway. It is now skipped for over-limit input. `build_source_map`
was checked and never touches `exprs`, so only the one consumer is short-circuited.

### Measured result

```text
depth        8 MiB (process main thread)   2 MiB (spawned thread / LSP / cargo test)
<= 60        accepted                      accepted
61..=200     accepted                      ABORTS      <- residual, see below
> 200        E0209, one diagnostic         E0209, one diagnostic
```

**The unbounded hole is closed.** 201, 250, 1,000 and 10,000 terms all produce exactly one
diagnostic on a 2 MiB stack, where before the repair 65 terms killed the process.

### The criterion that cannot be met, and why it is not a shortfall in the repair

OD-9 required both *"a 200-term chain is still accepted"* and *"no stack overflow even on a 2 MiB
thread"*. **Those cannot both hold at `MAX_DEPTH = 200`**: a 200-deep expression needs roughly 6 MiB
of stack in the downstream walks, and the ruling forbids reducing the limit.

So the residual is the window `61..=200` **on a small stack only** — depths the limit permits that a
2 MiB thread cannot carry. Closing it needs one of three things, each an owner decision rather than
an implementation choice:

```text
a stack-aware effective limit    lower the bound when analysis runs on a small stack
iterative downstream walks       the broad refactoring OD-9 and plan §3.2 both forbid
a documented minimum stack       state the requirement and let embedders meet it
```

**Recorded as OPEN-RESIDUAL rather than as DEV-214 remaining open**: the defect OD-9 named —
unbounded depth reaching recursive passes — is fixed and verified on the small stack.

### Evidence

`starkc/tests/dev214_expression_depth.rs`, 9 tests. Every over-limit case runs on a **2 MiB thread**,
because C10-B established the cliff scales with the stack and a main-thread test would have reported
a threshold four times too generous. A regression surfaces as a failed `join`, not as a SIGABRT that
takes the binary down.

```text
40 terms                     accepted, 2 MiB
200 terms (exactly the limit) accepted, 8 MiB — and the residual test pins why not 2 MiB
201 terms                     E0209, deterministic across runs, 2 MiB
1,000 / 10,000 terms          E0209, no overflow, 2 MiB
300 nested parentheses        still rejected BY THE PARSER — the contrast is preserved
rejection is not a cascade    exactly one diagnostic; no partial semantic analysis
the diagnostic has a real span inside the source, code E0209
wide-but-shallow shapes       2,000-element tuple, 2,000 locals, 1,500 fields — unaffected
```

`cargo test --lib` 569 passed; `c10b_robustness` 12; `c10c_security` 5; `conformance` 3; clippy
`--workspace --all-features --all-targets -D warnings` exit 0; `fmt --check` clean.

## DEV-012 — CLOSED (C10-P, owner verification, 2026-08-09). Ten of ten features interactively validated

**The deviation's earlier headings stand unedited.** This file is append-only and each was correct
when written.

### What C8 left open, and what closes it

`GATE-C8-CLOSURE.md` closed Gate C8 with interactive VS Code validation recorded for **three of ten**
advertised features — hover, go-to-definition, find-references — and narrowed DEV-012 to the
remaining seven: diagnostics, formatting, completion, signature help, rename, document symbols,
semantic tokens. Item 8 of §2a was an explicit owner override, labelled *"deliberately closed
short"*.

**Those seven were exercised by the owner in a real editor on 2026-08-09 and reported verified.**

### Environment

```text
VS Code           1.132.0 (df53daabb18cd157bdb08c7f01c34df936cf12f4)
extension         starklang.stark-language@0.2.0, built from the C10 candidate 37a0a03
compiler          release `stark` / `starkc` from the same candidate, wired via
                  .vscode/settings.json — NOT a PATH binary of unknown provenance
host              macOS 26.5.2, arm64
subject           a real multi-file package: two modules, a struct, an enum, a cross-file symbol
                  (`parse_fleet`), a decoy sharing its prefix (`parse_fleet_name`), and a
                  commented type error to introduce and withdraw
```

**The build was verified to carry the C10 work before the session**, rather than assumed: a
250-term chain produced `[E0209] this expression is nested too deeply to analyse (250 levels; the
limit is 200)`, which only the DEV-214 repair emits.

### Evidence class, stated precisely

**MANUAL** (Charter §5.2). Not automated coverage, and it must never be described as such.

**One feature has a value-level record; six have a verdict.** The distinction is kept per feature
rather than averaged, because it is the difference the DEV-182 lesson turns on.

```text
COMPLETION   VALUE-LEVEL, owner-reported 2026-08-09:
             "Completion offered parse_fleet, from fleet.stark, with detail
              `fn(Int32) -> Int32`: PASS"

             Three independent facts, and each is one the wrong answer would have failed:
               the CANDIDATE      parse_fleet, not a keyword and not the decoy parse_fleet_name
               the PROVENANCE     fleet.stark — a DIFFERENT module, so cross-module completion
                                  resolved rather than same-file text matching
               the DETAIL         `fn(Int32) -> Int32` — the real resolved signature, so the
                                  detail came from compiler analysis and not from a label

diagnostics, formatting, signature help, rename, document symbols, semantic tokens
             VERDICT — reported verified by the owner, no transcript captured
```

**That distinction is recorded because `GATE-C8-CLOSURE.md` §4 is explicit about why it matters:**

> DEV-182 — the LSP JSON parser silently decoded every escaped non-BMP character to the empty
> string — **passed** protocol validation, because both the parse and the response succeeded and
> only the *value* was wrong.

A verdict-shaped record is exactly what that defect survived. **The owner is the only party who can
produce this evidence and is the authority on their own session**, so DEV-012 closes; but the
release claim should describe it as *interactively validated by the owner in the recorded
environment*, which is what it is, rather than implying a captured per-feature value transcript.

**The completion observation is what the other six would look like if captured**, and it is kept
verbatim as the template — `"offered X, from Y, with detail Z"` names a candidate, a provenance and
a resolved signature, any one of which a stub or a text-matcher would get wrong. A re-validation
should produce six more of these.

### What this closure claims

```text
CLAIMS      all ten advertised language-service features have been exercised interactively in a
            real editor against a compiler built from the C10 candidate
CLAIMS      the C10-G gate's DEV-012 arm is satisfied: the language-services claim need not be
            narrowed on account of missing interactive validation
NOT CLAIMS  that the extension's full UI surface is exercised — ten features, not everything
NOT CLAIMS  automated protection. Nothing here runs in CI; a regression in any of the seven would
            be caught only by another manual session
NOT CLAIMS  a value-level transcript. See above
```

**C10-G status after this and C10-P's DEV-213 closure: both arms satisfied.** The Core v1 Compiler
Stable language-services claim may be stated without the DEV-012 or DEV-213 qualifications.

## DEV-005 — CLOSED (2026-08-09). It does not reproduce; AS2 removed it

**OD-7 attached a condition to this entry: one current-head reproduction before C10-Q, "because the
entry is old enough that a later change may already have removed it, and a release must not name a
deviation that no longer exists."** The condition was right, and this is the result.

### What it claimed

`starkc check` gated on `severity != Error` while `starkc run` gated on `diagnostics.is_empty()`, so
a program with one warning and zero errors was `OK` under `check` and refused outright by `run`.

### Measured at `29ce610` — it does not reproduce

Warning case, `unreachable code` after a `return`:

```text
$ starkc check warn.stark
Warning: [W0005] unreachable code  --> warn.stark:4:5
warn.stark: OK
check EXIT=0

$ starkc run warn.stark
Warning: [W0005] unreachable code  --> warn.stark:4:5
1
run EXIT=0          <- the program RAN, and printed
```

**Both gate on errors. Both report the warning. `run` executes.** The divergence is gone.

### The negative control, because "both exit 0" could equally mean `run` stopped gating at all

```text
$ starkc check err.stark    (let x: Int32 = "not an int")   EXIT=1
$ starkc run   err.stark                                    EXIT=1
```

An error still refuses both. So `run`'s gate is intact and merely no longer fires on warnings —
which is the fix, not a regression. Output is byte-identical across repeated runs.

### Why it went away

Not a targeted repair. **`AS2` — "the ONE pipeline" — removed the hand-assembled pipeline that
`cmd_run` used to carry**, and `main.rs`'s own comment records both the change and the reasoning:

> *"This command used to assemble parse → resolve → typecheck itself and gate each phase on
> `diagnostics.is_empty()` rather than on errors — equivalent today, because only typecheck emits
> warnings … but a warning added to parse or resolve would silently have become fatal here. **The
> session gates on errors.**"*

So the deviation was fixed as a side effect of removing a duplicated pipeline — which is exactly the
class of defect Charter §1.5 rule 18 predicts when tools diverge, and exactly what consolidating them
was expected to cure.

### Disposition

```text
status       CLOSED — does not reproduce at the candidate
supersedes   OD-7's "OPEN, ACCEPTED RELEASE DEVIATION"
consequence  C10-Q must NOT name DEV-005 as a release deviation. Naming a deviation that no longer
             exists is its own kind of false claim
```

**Population A drops from 24 live-OPEN to 23.**

## DEV-177 — CLOSED (2026-08-09). Already enforced; the ledger was never updated

**Verified at `076b4dc`.** The reproducer from this entry is rejected:

```text
$ starkc check repro.stark
Error: [E0204] generic parameter 'T' duplicates another generic parameter in scope
  --> repro.stark:4:15
 4 |     fn choose<T>(self, value: T) -> T {
   |               ^ 'T' is already declared by the enclosing impl
   = related: repro.stark:3:6: 'T' first declared here
check EXIT=1
```

**Negative control:** the same program with distinct names (`fn choose<U>`) still compiles — `OK`,
exit 0. So the rule is enforced, not the construct broken. The other arm of NAME-SHADOW-001,
`fn f<T, T>`, is rejected identically.

**Fixed by `78bd84c` — "DEV-177: enforce NAME-SHADOW-001, which was never enforced at all".** The
repair landed and this entry was never moved to CLOSED.

**Consequence for C10-Q, and it is the largest single change to the claim.** DEV-177 was the *only*
population-A deviation that **accepted a program the specification forbids**. With it closed, **no
open deviation makes a conformance claim false** — every remaining one either refuses what the spec
allows or executes an accepted program wrongly. The C10-Q derivation moves from *"PASS is not
supported because a claim would be FALSE"* to *"PASS is not supported because 84 rules are
unattributed"*, which is a different and weaker objection.

## DEV-181 — CLOSED (2026-08-09). Already fixed; the ledger was never updated

**Verified at `076b4dc`.** The reproducer compiles and runs:

```text
$ starkc check dev181.stark     OK, exit 0
$ starkc run   dev181.stark     prints 1, exit 0
```

`x = x.method()` — the everyday idiom this entry called its own worst consequence — works.

**Fixed by `57ff6b9` — "DEV-181: `x = x.method()` was refused by the borrow checker".**

**Consequence:** the row C10's disposition register flagged as *"the highest user-friction item"*
does not exist.

---

# Ledger hygiene finding (C10-Q preparation, 2026-08-09)

**SUPERSEDED 2026-08-09 by the C10-Q reproduction pass: the figure is SEVEN of twenty-three, and
`DEV-005` does not belong in the twenty-three at all — it owns no live heading here. See
`STARKLANG/docs/compiler/audits/C10-Q-REPRODUCTION-PASS.md`.**

**Three of twenty-three population-A deviations do not reproduce**: `DEV-005` (removed by AS2's
one-pipeline consolidation), `DEV-177` (`78bd84c`), `DEV-181` (`57ff6b9`).

**13% of the "open" list was fiction**, and all three were named in a drafted release claim.

The cause is structural, not careless: this ledger is **append-only**, so closing a deviation
requires a deliberate new entry, and a repair landed under a different work packet has no mechanism
forcing that entry to be written. Each of these three was fixed by a commit whose message names the
DEV number — the information existed; nothing connected it to the ledger.

**The rule this produces, and C10-Q depends on it:** a deviation may not be named in a release
claim on the strength of its ledger entry. It must be **reproduced at the candidate head**, or
closed. OD-7 imposed exactly this on `DEV-005` and it found one; applying it to only one entry was
the mistake.

## DEV-083 — CLOSED, does not reproduce (C10-Q reproduction pass, 2026-08-09)

The entry's verbatim reproducer — `impl<T> Pair<Option<T>, Int32>` with
`let p = Pair { x: Some(5), y: 42 }; p.tag();` — compiles and runs, printing 42. It predicted
`E0302 method 'tag' not found for type 'Pair<Option<_infer_4>, _infer_5>'`.

Checked against three receiver shapes, including `Vec::new()` and a bare `None`, which are the most
unresolved a receiver argument can be. All resolve.

**Incidentally repaired.** No commit names DEV-083. `5b5edd3` — "AS3 Boundary 4 EXIT: `find_method`
and `find_impl_fn` no longer exist" — rewrote exactly the one-way matching the entry describes.
This is the class of closure a git-log audit cannot find, because fixing it was not the point of the
work that fixed it.

Note this entry was adjudicated OPEN under OD-7 earlier the same day, on the strength of its
description rather than a re-run.

## DEV-122 — CLOSED, does not reproduce (C10-Q reproduction pass, 2026-08-09)

The symptom was a fault in one file reported against another. A division-by-zero inside a module is
now reported at `src/helper.stark:3:5`, with the right line rendered, under **both** the HIR oracle
and the native binary.

`Span` now carries `pub source: SourceId` (`src/source.rs:19`), and `Span::in_source` is the
constructor. That is precisely the "platform correction (mandatory `SourceId` on every span,
resolution total by construction)" the entry filed as a separate future WP; it landed under AS1b.
Incidental, as with DEV-083 — no commit names DEV-122.

**A residual survives this closure and is not tracked by it.** `SourceFile::line_col` still clamps
(`offset.min(self.src.len() as u32)`), and compile-time and runtime rendering remain separate paths.
The clamp was dangerous because a span could be resolved against the wrong file; that is now
prevented structurally, which is why this closes. The further hardening the entry asked for —
`start <= end`, a column inside its line, one shared `resolve_span` — has not been done, and
`debug_assert!(lo <= hi)` is not a release check. Whether that deserves its own number is an owner
call.

## DEV-161 — CLOSED, does not reproduce (C10-Q reproduction pass, 2026-08-09)

`stark build` with `CARGO_TARGET_DIR` set builds and runs, and the hijack directory is never
created. The builder passes `--target-dir` explicitly rather than clearing the variable, so the path
the build writes and the path the builder reads derive from one value — documented at
`src/backend/generated_rust/build.rs:104-119`.

## DEV-162 — CLOSED, and already recorded as closed elsewhere (C10-Q reproduction pass, 2026-08-09)

A partial move followed by a read of a live sibling runs correctly. More to the point,
`COMPILER-STATE.md:2176` has recorded `DEV-162   read through a whole-value accessor   CLOSED
(CD-372)` since CD-372. This entry was backfilled as OPEN under OD-7 on 2026-08-09, contradicting
the state file on the same day. **Two sources of record disagreed and the ledger was the wrong
one.**

## DEV-178 — CLOSED, does not reproduce (C10-Q reproduction pass, 2026-08-09)

`b39c49d` — "DEV-178: a function value carries the instantiation it was created with" — is the
repair the entry's own Resolution section prescribed.

Verified behaviourally rather than by commit title, because a title is not evidence. `size_of::<T>()`
read from inside an associated function of a generic impl returns **4** for `Holder::tsize(1)` and
**1** for `Holder::tsize(true)`. A result that discriminates `Int32` from `Bool` cannot be produced
without the generic environment. The function-value path returns 4 for
`let f: fn() -> UInt64 = type_size::<Int32>;`. Both causes named in the entry are covered.

## DEV-157 — reproduces, but not in the shape this entry named (OPEN, C10-Q reproduction pass, 2026-08-09)

Kept OPEN, with its reproducer corrected.

The shape the entry names — `Err(_) => panic(..)` in match-arm value position — now builds and runs
natively, including when the panicking arm is taken. **The defect is nonetheless alive**:

```stark
fn main() { let x: Int32 = panic("p"); println(x); }
```

fails with `native build does not yet support this program: MirTy Never has no C5.3a generated-Rust
representation yet`, and so does `panic` in argument position.

This was one probe away from being filed as a false closure in a release claim. It is the reason
every non-reproducing verdict in this pass was re-tested across shape variants.

## DEV-159 — not settled by the reproduction pass (OPEN, C10-Q reproduction pass, 2026-08-09)

Remains OPEN, and this is not a confirmation. A native build racing its own dependency build is
non-deterministic; a single successful build does not falsify it, and no fix commit exists. Counted
as open conservatively. Settling it requires repeated cold builds of an HTTPS program.

## DEV-180 — scheduled: its own packet, immediately after C10-Q closes (OPEN, owner ruling 2026-08-09)

Still OPEN and still reproducing. This records **when** it is repaired and what the packet already
knows, so the packet does not restart the investigation.

**Owner ruling (2026-08-09): DEV-180 is taken as its own packet once C10-Q closes, and not before.**

The reason is not difficulty. Binding a genuine reference for `&mut self` changes what the HIR
oracle MEANS by a mutable receiver, and the oracle is the reference every engine-agreement claim in
C10 is measured against. Landing it before C10-Q invalidates the evidence package; landing it after
costs a re-run of the comparison, which is the cheaper and more honest order. This supersedes the
entry's earlier "sequenced before A4 resumes" only as to ordering against C10-Q — A4 still follows
this repair, not the other way round.

### The three questions the entry required answered first — answered

1. **Why DEV-070 excluded `&mut self` when `&self` moved to genuine references.** Because
   take/write-back needs the caller's slot emptied to bind the value as `self`, and the `&self` arm
   never does: it binds `Value::Ref(receiver_place.clone())`, since a shared borrow need not own.
   The asymmetry was left deliberately and commented — "(`&mut self` keeps its take/write-back
   model.)" **It was a deferral, not a design.**
2. **Whether that limitation still holds. It does not.** `Place` and `place_slot_mut` already write
   through a projection; that is exactly what the `&self` path uses today. The machinery the
   deferral was waiting for exists.
3. **Whether the returned-reference test depends on rebasing out of the method frame.** This is the
   real risk and the reason the entry's third forbidden repair exists. A returned `&mut` currently
   points into method-frame temporary storage; binding a real reference makes it point into caller
   storage, which is correct but is not what
   `mut_reference_returned_from_mut_self_method_writes_through` was written against. That test must
   be re-derived from the specification, **not adjusted until it passes.**

### The shape, and the tell that it is the right one

Bind `Value::Ref(receiver_place.clone())` for `hir::Receiver::RefMut`, let mutability come from the
static `Ty` and the borrow checker, and **delete the write-back and its error-path restoration
entirely** — including the "the `Drop` receiver disappeared" internal error. That code exists only
to service the flattening. A repair that leaves it in place has not made the receiver a reference.

The entry's three forbidden repairs stand unchanged: no `&mut T → bare T` in `value_matches_ty`, no
receiver-specific validator exception, and no synthetic reference to method-local storage.

### Expected cost, recorded before the work so it can be checked against

The five named receiver tests move, plus whatever the three-engine suite finds — and it will find
something, because this changes the oracle. A packet that reports *no* engine movement has probably
not changed the semantics it set out to change.

## DEV-186 — the LSP transport now bounds its allocation (CLOSED, 2026-08-09)

`Content-Length` is a claim made by the peer, and it was honoured by allocating that many bytes
before reading any of them and before the JSON parser ran.

Two changes, and both were needed. `MAX_CONTENT_LENGTH` (64 MiB) rejects an absurd claim outright.
`take` + `read_to_end` then grows the buffer as bytes ACTUALLY ARRIVE, so a peer that advertises the
maximum and sends ten bytes causes ten bytes of growth. **A cap alone would still allocate the cap**
— that is the half a naive fix misses, and it has its own test.

Evidence: `lsp::server::tests::dev186_*` — an absurd length, the boundary derived from the constant
rather than hard-coded, an under-cap lie, and a control proving an ordinary `initialize` is still
served. Without the control every other assertion is satisfiable by a server that refuses
everything.

Unchanged: this is the TRANSPORT's authority. `json::MAX_DEPTH` bounds recursion inside the parser
and could not have helped, because the parser never runs.

## DEV-156 — `stark fmt` keeps field doc comments on their fields (CLOSED, 2026-08-09)

The formatter measured a flat rendering of the field list into a scratch buffer, so its item printer
was forbidden from consuming comments — a consumed comment would be discarded with the measurement.
`field_def` therefore never emitted them, and field docs survived only as unconsumed trivia, flushed
AFTER the struct. A documented struct came back as a one-line struct followed by its own orphaned
docs. Worse than the original one-line entry recorded: the reproduction pass found the comments
detached from what they document, not merely relocated.

`struct_field_list` skips the flat measurement when a comment lies inside the braces, which is both
why the broken form is chosen and why consuming comments there is safe. With no interior comment it
defers to `delimited_list` and the flat form is byte-identical.

Evidence: `tests/formatter.rs::dev156_*` — four cases. The ordering one matters most: the defect
PRESERVED every comment and merely moved it, so a test asserting the text still appears somewhere
would have passed against the broken formatter. `dev156_an_undocumented_struct_is_still_flat` is the
negative control; without it, forcing every struct broken would satisfy the rest. The full formatter
corpus sweep is unchanged.

**Recorded limitation, pre-existing and unchanged:** a blank line before an UNDOCUMENTED field is
still dropped. `delimited_list` never preserved blank lines between fields; DEV-156 did not move
that in either direction, and the test asserts the behaviour as it is rather than as it ideally
would be.

## DEV-172 — every signed minimum is now writable (CLOSED for the signed widths, 2026-08-09)

`let a: Int8 = -128;` and the minimum of every other signed width compile, run and build natively.
A negative literal is a unary minus applied to a POSITIVE literal, so the magnitude reaching the
range check was `128`, out of range for `Int8` — and the identical argument refused the minimum of
every signed width.

`-<int literal>` is now folded into ONE literal in **three** places, because three phases each
performed the two-step evaluation independently:

```text
typecheck   the literal arm range-checks the negated value
HIR interp  eval_expr folds the pair; eval_lit_signed carries the sign into the suffix check
MIR lower   emits a single negative constant
```

Evidence: `three_engine_differential::dev172_signed_minimums_agree` — all three engines, including
the suffixed `-128i8` form.

**The MIR half was found by that test and by nothing else.** A native build with optimisation ON
succeeded, because the MIR optimiser happened to const-fold the shape; the same program with
`--no-mir-opt` emitted `((128i8) as i128).checked_neg()` and rustc rejected the crate. The engines
disagreed on whether a program was BUILDABLE, and a hand-run `stark build` reported success. Had the
fix stopped at the interpreter, the deviation would have been recorded closed on a passing manual
check.

`dev172_negating_the_minimum_still_traps` pins that the fold is NARROW: only a literal directly
under a unary minus folds, so `-a` where `a` is `Int8::MIN` still traps `IntegerOverflow` in all
three engines. A fold that reached further, or that quietly widened the type, completes instead and
fails that test.

**Residual, deliberately not taken:** an UNSUFFIXED literal above `Int64`'s range is still typed
against the signed default, so `let u: UInt64 = 18446744073709551615;` is refused. Unlike the signed
minimums, this has a working form — `18446744073709551615u64` produces the correct value — so it is
an ergonomic gap, not an unwritable value. Fixing it means changing which literals may enter an
inference variable, which is DEV-015's area and affects defaulting for every unsuffixed literal.
Recorded rather than bundled into an unrelated repair.

## DEV-120 — CLOSED, RECLASSIFIED AS DOCUMENTED LIMIT (post-C10 repair programme, P8, 2026-08-10)

The entry above stands unedited. This heading records the disposition §14 required, reached by
reproduction rather than by re-reading the entry.

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`. No compiler source changed.

### The five questions §14.1 required, answered by reproduction

Reproducer — unbounded self-recursion with no base case:

```stark
fn down(n: Int32) -> Int32 { return down(n + 1); }
fn main() { println(down(0)); }
```

```text
Does the normative contract require unbounded recursion?   NO — LIMIT-RESOURCE-001 makes capacities
                                                           implementation/target-defined.
Does it require graceful resource-limit reporting?         ONLY "when the host permits".
Does native execution fail gracefully or abort?            ABORTS. exit 134 (SIGABRT),
                                                           "fatal runtime error: stack overflow".
Do HIR and MIR report the limit deterministically?         YES. Classified host/resource failure,
                                                           "call depth limit reached (512 frames)",
                                                           stable exit 2, no process abort.
Is native stack exhaustion bounded before host abort?      NO. The generated binary recurses on the
                                                           host stack and the host terminates it.
```

### Why this closes rather than repairing

The gap between exit 2 and exit 134 is real and is **not** being denied. It closes because the rule
that governs it already permits the divergence twice over: capacities are implementation-defined, so
512 frames and a stack-shaped native capacity are not required to agree, and the reporting duty is
qualified by "when the host permits" — a signalled stack overflow is the host declining. **No claim
is made that the two capacities match**, and resource exhaustion is excluded from engine comparison
by construction, so the three-engine claim is untouched.

Owner ruling D4 (WP-C7.9) already decided the repair question in the other direction: record the
boundary rather than instrument the backend. Bounding it natively means per-call depth
instrumentation in every generated function — paid by every program, to report a host-defined
condition it still could not fully cover, since host stack growth from the runtime or a provider
stays invisible to it. This disposition does not reopen D4; it stops carrying a settled decision as
an open defect.

**§14's explicit warning was observed: the number was not raised.** `MAX_CALL_DEPTH` is unchanged.

**Residual, stated rather than hidden:** a program that recurses without a base case is reported
cleanly under `stark run` and dies by signal when built natively. That asymmetry is now a documented
limit, not an open deviation. If native execution ever acquires a reason to bound call depth for its
own sake, this reopens as a repair — the original entry's "owning gate: none scheduled" still holds.

**Evidence:** `starkc/tests/resource_exhaustion.rs` (interpreter side, pre-existing); the native
side reproduced above at the baseline SHA.

## DEV-167 — CLOSED, CE1 DECIDED: the method form is a non-promise, not a gap (2026-08-10)

The entry above stands unedited. It correctly identified this as blocked on a
blanket-implementation decision and correctly called that decision CE-shaped. **The owner made it.**

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`.

### Reproduced first

```stark
fn show<T: Display>(value: &T) -> String { return value.to_string(); }
```

```text
[E0302] method 'to_string' not found for type 'T'
        no trait in scope declares a method named 'to_string'
```

Refused at **name resolution**, identically under all three engines. This is not an engine
disagreement and not a backend gap — nothing to reconcile, because the language does not have it.

### Why there was no conformance gap to repair

`06-Standard-Library.md:817` declares `trait ToString { fn to_string(&self) -> String; }` and
`:446` gives `str::to_string`. **Neither promises that every `Display` type carries the method
form.** The implementation matches the specification exactly. The only live question was whether
Core v1 *should* make that promise — a change to the language contract, **CE1**, Charter §2.3.

### Decision (owner, CE1, 2026-08-10): keep the free function; close as a documented non-promise

The two alternatives and why neither was taken:

```text
impl<T: Display> ToString for T   Permitted by the spec, but it CLOSES the trait. See below.

resolver branch on "to_string"    Reintroduces exactly the two-tier trait model DEV-166 removed
                                  (RESOLVED, DEV-DISPLAY-DISPATCH) — method visibility depending
                                  on whether a trait is compiler-known. Trades a closed defect
                                  for ergonomics.
```

### Correcting the original entry's reason for the blanket option

The entry above says "Core v1 has neither blanket implementations nor extension traits." **The
blanket half is wrong, and the real objection is stronger.** `03-Type-System.md`
TRAIT-COHERENCE-002 permits them: "Blanket implementations are permitted only when the overlap test
proves them disjoint from every other implementation."

The decisive clause is the one before it: **"Positive trait bounds never make unifying heads
disjoint."** The head of `impl<T: Display> ToString for T` is `ToString for T`, and `T` unifies with
every type. The `T: Display` bound does **not** narrow it for overlap purposes — disjointness is
proved only by incompatible nominal constructors, unequal concrete types, or different trait
identities, none of which apply.

So the blanket impl is admissible only while it is the **only** `ToString` implementation in the
resolved package graph, and it permanently forecloses any other: a user writing
`impl ToString for MyType` — to render a type differently from its `Display` form, which is a
legitimate thing to want — would be rejected by coherence, in a package that may not even contain
the blanket impl. Core v1 has no specialization, no negative implementations and no
declaration-order priority to escape with. **That is the cost: not implementation difficulty, but
converting `ToString` from a trait users may implement into a closed trait derived from `Display`.**

### What is supported

`stark_fmt::to_string<T: Display>(value: &T) -> String` — `packages/stark-fmt/src/lib.stark:75`.
It is a real workaround, not a notional one: exercised on a **user-defined** type by
`packages/stark-fmt/src/tests.stark::test_to_string_free_function` (`to_string(&p)` → `"(0,0)"`)
and by the consumer at `packages/stark-fmt-consumer/src/main.stark:46`.

### The decision is pinned by tests, not only by this paragraph

`starkc/tests/dev_display_dispatch.rs`:

```text
to_string_on_a_display_bound_is_refused_by_decision
    positive  the refusal is E0302 and names `to_string`
    negative  it must NOT be the missing-bound diagnostic — `Display` is already bounded, so
              "requires the bound" would be advice the user has already taken, and would mean
              resolution believed `to_string` was reachable from some bound

the_display_bound_still_contributes_fmt_after_the_to_string_decision
    negative  `fmt()` — the method a Display bound DOES contribute — still dispatches. A change
              that reached `to_string` by widening compiler-known bound contribution, or that
              narrowed contribution to exclude it, fails one of these two.
```

The first test fails the moment someone adds the name-keyed resolver branch, which forces the CE1
decision to be reopened deliberately rather than reversed quietly for ergonomic reasons. That is
the whole point of pinning a decision rather than documenting one.

**Residual:** none as a defect. `value.to_string()` on a `T: Display` remains refused **by
decision**. If Core v1 ever acquires blanket implementations for independent reasons, this becomes
a natural consequence of that feature and should be revisited then — not before, and not on its own.

## DEV-220 — a diverging arm captured the join's inference variable (RESOLVED, post-C10 P3, 2026-08-10)

Found while building §9.1's `Never` position matrix for DEV-157, and registered separately under
the repair programme's §19.4 rather than absorbed into it: the root cause is in **typecheck
inference**, not in MIR or backend representation, which is what DEV-157 is about.

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`.

### Normative expectation

`!` coerces to every type. It therefore **constrains nothing** — joining a diverging arm with an
inhabited one yields the inhabited arm's type.

### Current behaviour at baseline: an internal compiler error

```stark
fn main() { let x: Int32 = if true { 1 } else { panic("p") }; println(x); }
```

```text
Error: internal compiler error: DEV-121 representation mismatch at an expression result:
       expected `Never`, found `Int`
```

`return` reaches it identically, and that shape is not an error path at all:

```stark
let x: Int32 = if true { 1 } else { return; };
```

### Cause, and its relationship to DEV-218

**DEV-218 (CLOSED 2026-08-09) created the precondition, and correctly so.** It made a block produce
`!` when its reachable path diverges. What no rule then covered is that `!` must not be *bound to*
an open variable. `infer.rs::unify` matched its `(Ty::Infer(id), other)` arm before
`(Ty::Never, _) | (_, Ty::Never)`, so `unify(?T, Never)` bound `?T := Never`. The expression's
recorded type became `Never` while the value produced at run time was the inhabited arm's, and
**DEV-121's representation guard — closed, and working exactly as designed — caught it.** The ICE
is the invariant doing its job; the defect is upstream of it.

**Why DEV-218's evidence did not catch this.** Its three programs put the inhabited arm where it
resolved the variable *first*; `Never` then met an already-concrete type and the correct
no-op arm applied. Reversing a match's arm order is sufficient to reproduce. The pre-existing
match-arm probe in DEV-157's own entry passes for exactly this reason — it is `Ok(n) => n` first.

### Repair

```text
typecheck/infer.rs   the `(Ty::Never, _) | (_, Ty::Never)` arm moves ABOVE the `Infer` arms, and
                     records the open variable rather than binding it
typecheck/state.rs   `never_coerced_vars`
typecheck/infer.rs   `default_never_coerced_vars`, run from `items.rs` AFTER integer-literal
                     defaulting
```

**The fallback pass is not optional, and its ordering is load-bearing.** Dropping the binding alone
merely moved the failure: a variable constrained by nothing else stayed open and reached MIR as
`type Infer(TypeVarId(0))` — the exact escape `default_unconstrained_int_literals` documents itself
as preventing. It defaults to `Never` only after integer defaulting has had its turn, so
`let x = if c { 1 } else { panic(..) };` still yields `Int32`.

### Evidence, and it was proved capable of failing

`three_engine_differential.rs`, five cases, all three engines:

```text
dev220_if_join_with_a_diverging_else                 diverging arm SECOND
dev220_match_join_with_the_diverging_arm_first       diverging arm FIRST — the variable is fully
                                                     open when `Never` arrives
dev220_if_join_with_an_early_return                  `return`, not `panic`
dev220_the_diverging_arm_still_traps_when_taken      NEGATIVE: taking the diverging path still
                                                     traps `Panic`. A "fix" treating the arm as an
                                                     ordinary value of the join type passes the
                                                     first three and fails this
dev220_an_unannotated_join_still_defaults_to_int32   NEGATIVE: literal defaulting still wins over
                                                     the `!` fallback. A fallback that ran first,
                                                     or bound eagerly at the unify site, yields
                                                     `Never` here and fails
```

**With the arm moved back below `Infer`, all five fail; with the repair, all five pass.** Verified
by reverting the repair in place rather than by assertion.

Green alongside: 576 lib, 132 conformance, 114 three-engine (before these five), 23
dev_display_dispatch, layer audit, mir_differential. `cargo fmt --check` clean.

**Residual:** none for this defect. The `Never` positions that remain unbuildable natively are
DEV-157's, not this one — see its entry.

## DEV-157 — CLOSED, REPAIRED: `!` has a native representation, and it is uninhabited (post-C10 P3, 2026-08-10)

The entry above stands unedited, including its warning that this was "one probe away from being
filed as a false closure." That warning was earned twice more here: the position matrix §9.1
required found **two** further defects the entry did not name, one of them an internal compiler
error.

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`.

### The position matrix, built by probe

22 programs, each run under the HIR oracle and built natively. Classified per §9.1:

```text
SUPPORTED AT BASELINE      statement expression; match arm (inhabited arm first);
                           return/tail position
REFUSED EARLY, CORRECTLY   nothing — see "accepted set" below
INTERNAL COMPILER ERROR    if/else join; match with the diverging arm FIRST; `else { return; }`;
                           unannotated join                              -> DEV-220
MIR LOWERING FAILURE       if/else join (`block in value position yielded no value`)
BACKEND FAILURE            local initialiser; argument position; tuple, array and struct
                           elements; nested composite                    -> this entry
```

### Three distinct repairs, in three phases

**1. `typecheck` — registered separately as DEV-220.** A diverging arm captured the join's
inference variable. Not absorbed here: the root cause is inference, not representation, and §19.4
requires a defect class in an unrelated phase to be registered on its own.

**2. `mir/lower.rs` — the else arm now tolerates a diverging block.** The then arm already did
(`if let Some(v) = then_value`); the else arm routed through `lower_expr_to_operand` and demanded
a value. `if c { x } else { return; }` — not an error path, ordinary control flow — could not be
built natively. Nothing is assigned on a path that never reaches the join, which is the rule the
then arm already relied on.

**3. `backend` — `MirTy::Never` is `core::convert::Infallible`, an EMPTY enum.**

§9.2's rule was that no runtime storage may be invented for an uninhabited value. None is:

```text
emit_types::emit_ty_at    MirTy::Never -> core::convert::Infallible (uninhabited, zero-sized)
emit_types::mentions_never a composite with an uninhabited component is itself uninhabited
emit_bodies               such a local is declared UNINITIALISED, never default-initialised
```

`default_value_expr` still refuses `Never` and is right to — there is no value to fabricate. The
local is the result place of a diverging expression; control never reaches its assignment.
**rustc's own definite-assignment analysis is the standing check on that claim**: a `Never` local
that were genuinely read fails to compile rather than reading fabricated storage.

**4. `mir/verify.rs` — the never-coercion allowance is now STRUCTURAL.** `expect_ty` permitted
`Never` only at the top level, so `(1, panic("p"))` into a `(Int32, Int32)` place was rejected
`MIR-0004: expected Tuple([Int32, Int32]), found Tuple([Int32, Never])`. `03-Type-System.md` says
`!` coerces to *any* type; a composite with an uninhabited component is itself uninhabited and no
value of it reaches the assignment. `never_coercible` recurses structurally — **only `Never` is
permissive; every other mismatch still fails.** Found by the three-engine harness and by nothing
else: `stark build` alone accepted the program.

### The accepted program set changed, and it changed TOWARD the specification

```stark
let x: Int32 = 1 + panic("p");
```

was refused `[E0500] type '!' does not satisfy operator trait 'Num'`, and is now accepted (and
traps, in all three engines). **That refusal was an artefact of DEV-220**, not a rule: the literal's
variable had been bound to `Never`, so the operator check ran against `!` instead of `Int32`.
`03-Type-System.md` line 67 is unqualified — "An expression of type `!` coerces to any other type"
— so accepting it is conformance, not widening. Recorded explicitly because a change to the
accepted set is normally CE1/CE2 territory; this one required no decision because the
specification already stated the outcome.

### Evidence

`three_engine_differential.rs`, seven cases, all three engines agreeing on trap category AND exact
trap line:

```text
dev157_never_in_a_local_initialiser          dev157_never_inside_an_array
dev157_never_in_argument_position            dev157_never_in_a_struct_field
dev157_never_inside_a_tuple

dev157_a_diverging_call_still_diverges       NEGATIVE (§9.3): a backend that gave `Never` real
                                             storage and let control fall through completes here
dev157_code_after_a_diverging_initialiser_is_unreachable
                                             NEGATIVE: statements after an unreachable initialiser
                                             never run
```

**Proved capable of failing:** with the `MirTy::Never` representation removed, 6 of 7 fail. The
seventh is `..._still_diverges`, which correctly passes either way — it guards a *different* wrong
fix, and a control that failed for the absence of the repair would not be controlling anything.

Green: 576 lib, 132 conformance, 126 three-engine, 23 dev_display_dispatch, 8 mir_differential,
layer audit, adversarial accepted-surface audit. `cargo fmt --check` clean; clippy
`--workspace --all-features --all-targets -D warnings` clean.

**Residual:** `loop { }` with no `break` has type `!` per TYPE-LOOP-001; it builds, and was
verified by building rather than running, because running it correctly never terminates.

## DEV-168 — CLOSED, REPAIRED: the qualified core-trait call lowers (post-C10 P4, 2026-08-10)

The entry above stands unedited.

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`.

### Reproduced

```stark
impl Display for P { fn fmt(&self) -> String { return "P".to_string(); } }
let s: String = Display::fmt(&p);
```

```text
stark run   -> P                                       (front end and HIR oracle, both fine)
stark build -> native build does not yet support this program: callee form (C4.5)
```

### Cause

`lower_call` matched `Res::TraitMember(trait_item, member)` and reached into
`hir::ItemKind::Trait` for the signature. **A compiler-known trait has no such item** — the same
fact DEV-166 was about — so `Display::fmt` resolves to `Res::CoreTraitMember` instead, matched no
arm, and fell through to the catch-all.

### Repair, and what it deliberately does NOT add

§10.2 forbade a second trait-dispatch authority. None was added:

```text
check_qualified_core_trait_call   ALREADY selects the impl member and publishes it, through
                                  publish_operator_use -- the same publisher `a == b` uses,
                                  with DispatchProvenance::CoreTrait
operator_callable_key             ALREADY consumes exactly that provenance, and already handles
                                  both binding times: Static for a concrete nominal, Bound for a
                                  bounded generic parameter
```

The new `Res::CoreTraitMember` arm **reads that published answer** and emits the call. No impl
scan, no unification, no trait name special-cased, no generic substitution rediscovered in the
backend — the arm is routing, not selection. The receiver still weakens `&mut` to `&` when the
selected member takes `&self` (C6.1f-b2), and, as in the user-trait arm, no auto-borrow applies
because the receiver is written explicitly.

### Evidence

`three_engine_differential.rs`:

```text
dev168_qualified_core_trait_call_on_a_concrete_nominal
dev168_qualified_core_trait_call_through_a_generic_impl   impl<T: Display> Display for W<T> --
                                                          fails if lowering re-derives the
                                                          callable instead of reading the
                                                          checker's substitution
dev168_qualified_eq_matches_the_operator_it_spells        a DIFFERENT core trait, so the repair is
                                                          not keyed to `Display`; `Eq::eq(&a,&b)`
                                                          and `a == b` select the same impl
```

**`dev_display_dispatch.rs::qualified_calls_disambiguate_the_two_traits` is upgraded from
front-end-and-oracle to full three-engine agreement.** That test is the one this entry named as its
evidence, and its doc comment recorded the gap; it now proves the gap closed rather than describing
it. It exercises both spellings — `Display::fmt` (repaired here) and `OtherFormat::fmt` (the user
trait path, which already lowered) — so a future divergence between them is pinned by the
interleaved output.

**Negative control:** `a_qualified_core_trait_call_without_an_impl_is_still_refused`. A type with no
`Display` impl is still refused `E0500`, naming both the type and the trait. Because the repair
reads a publication rather than scanning, a type that never produced one cannot reach lowering at
all — a backend-side impl scan would, and would fail this.

**Proved capable of failing:** with the arm removed, all three `dev168_*` cases fail.

Green: 576 lib, 132 conformance, 129 three-engine, 24 dev_display_dispatch, 17 as3_display_plan,
5 as3_invocation_authority, 8 mir_differential, layer audit, adversarial accepted-surface audit.

### Residual — a SEPARATE front-end gap, not this one

```stark
fn show<T: Display>(x: &T) -> String { return Display::fmt(x); }
```

is refused `[E0500] type 'T' does not implement 'Display'`. `check_qualified_core_trait_call`
selects by scanning impls for the receiver's nominal type and never consults the parameter's
BOUNDS, so a bounded generic receiver finds nothing. This is a **front-end over-rejection**,
pre-existing and independent of this repair — DEV-168 was explicitly "type-checks and runs under
the oracle, MIR refuses", and this shape does not type-check at all. Registered here rather than
absorbed, per §19.4. The ordinary method form (`x.fmt()`) works, so it has a working spelling.

## DEV-140 — reproduced, no consumer, repair deferred (OPEN, post-C10 P6 §12.1 assessment, 2026-08-10)

Assessed individually as §12.1 requires. Not grouped with DEV-141..145 — §12.2's rule is satisfied
by no pair of them.

```text
reproduced          YES. layer_audit L7153, an ENFORCING gate since CD-342: it fails on an
                    unregistered finding and equally on a registered one that stops reproducing,
                    so a green run is the reproduction evidence.
user-visible shape  `v.insert(0u64, 2)` -- a `Vec` method outside the implemented lowering set.
consumer            NONE. No first-party package calls insert/extend/truncate/sort/reverse/
                    contains/dedup/split_off/drain/retain on a Vec.
missing layer       There is no `RuntimeFn::VecInsert`. Adding it spans FOUR layers: the RuntimeFn
                    enum, a `stark_runtime` implementation, backend emission, and the MIR
                    verifier's runtime-callee signature table (V-RT-1/MIR-0012).
radius              Feature addition, not a repair. Each new method repeats all four layers.
disposition         DEFERRED. §12.1 step 6 permits repair only if bounded; this is not, and §24
                    directs that application pressure drive it. §12.3 forbids widening native
                    claims beyond tested shapes, and there is no shape to test against a consumer.
```

Front end and HIR oracle accept and run it; MIR refuses before any code is emitted. **No
soundness impact** — an accepted-but-unbuildable refusal, the E0105 class, caught at compile time.

## DEV-141 — reproduced, a std-full profile boundary (OPEN, post-C10 P6 §12.1 assessment, 2026-08-10)

```text
reproduced          YES. layer_audit L8093.
user-visible shape  `HashMap<Int32, D>` where `D` implements `Drop`.
consumer            NONE. No first-party package declares a `HashMap` of any element type.
missing layer       Lowering has no drop elaboration for map values carrying a destructor.
radius              **This is a PROFILE boundary, not only an implementation gap.** The refusal
                    text names it: "reserved -- std-full". `06-Standard-Library.md` defines
                    `core-min` and `std-full`; the built profile is core-min, which does not carry
                    these collections. Repairing it means implementing part of a profile this
                    build does not claim.
disposition         DEFERRED, and the reason is different from its five siblings: closing it is
                    not a defect repair but a profile expansion. Reclassifying it as a documented
                    profile boundary is arguably more accurate than "layer defect" and is left as
                    an owner decision rather than taken here.
```

## DEV-142 — reproduced, needs generated lifetimes, deferred (OPEN, post-C10 P6 §12.1 assessment, 2026-08-10)

```text
reproduced          YES. layer_audit L9130.
user-visible shape  `(String, &str)` printed as a tuple -- a droppable composite that also carries
                    a borrowed element.
consumer            NONE.
missing layer       The drop plan for a composite mixing an owned droppable and a borrow needs
                    GENERATED LIFETIMES in the emitted Rust -- a later C6.3e slice.
radius              The largest of the six. Generated lifetimes are a backend capability, not a
                    method; nothing else in the six needs them.
disposition         DEFERRED.
```

## DEV-143 — reproduced, no consumer, repair deferred (OPEN, post-C10 P6 §12.1 assessment, 2026-08-10)

```text
reproduced          YES. layer_audit L5346.
user-visible shape  `assert_eq(x, y)` where the operands are a user struct with `impl Eq`.
consumer            NONE, and this was checked closely because tests are where it would bite
                    first. Every `assert_eq` across the first-party packages compares a scalar or
                    a string -- `.len()`, `.as_str()`, an index projection's field. Not one
                    compares a user-defined type.
missing layer       TWO phases. The checker types `assert_eq` as `fn(T, T) -> Unit` with no `Eq`
                    requirement and publishes NO selection, so there is nothing for MIR to read;
                    MIR then refuses the user-nominal operand. A repair must add the publication
                    (as `println` does for Display, via `display_checks`/`record_display_plan`)
                    and then dispatch through it.
radius              Bounded-ish and the most tractable of the six -- it would reuse the same
                    published-selection path DEV-168 now routes through. Still a two-phase change
                    to a builtin's contract, and it moves an existing refusal from MIR to the
                    checker, which is the shape CD-294 records as not always cheap (E0106 was
                    reverted for exactly that reason).
disposition         DEFERRED, with a note that this is the one to do FIRST if application pressure
                    appears. `a == b` on a user nominal already works in all three engines, so the
                    working spelling exists.
```

## DEV-144 — reproduced, no consumer, repair deferred (OPEN, post-C10 P6 §12.1 assessment, 2026-08-10)

```text
reproduced          YES. layer_audit L3698: `type Core(ValuesIter, [Primitive(Int32)]) (C4.5)`.
user-visible shape  `for` driving an iterator that is neither a range nor a `Vec` cursor.
consumer            NONE. No first-party package writes `for x in <expr>()` over such an iterator.
missing layer       Lowering implements the range and `Vec` cursors; other iterables reach an
                    unsupported site. `iterator_fn_key` already exists to read the checker's
                    `Iterator::next` selection, so the authority is present -- what is missing is
                    the per-cursor lowering for each Core iterator type.
radius              One cursor type at a time. Not shared with any other deviation here.
disposition         DEFERRED.
```

## DEV-145 — reproduced, no consumer, repair deferred (OPEN, post-C10 P6 §12.1 assessment, 2026-08-10)

```text
reproduced          YES. layer_audit L6450: `method to_uppercase on String`.
user-visible shape  A method call whose receiver auto-derefs to a type lowering does not carry.
consumer            NONE. No first-party package calls to_uppercase/to_lowercase/trim/replace/
                    starts_with/ends_with/find/split_at/repeat.
missing layer       Same four-layer shape as DEV-140 -- RuntimeFn, runtime implementation, backend
                    emission, verifier signature -- but on `String` rather than `Vec`.
radius              Feature addition per method.
disposition         DEFERRED.
```

## P6 — the six layer defects: individually assessed, repair deferred (2026-08-10)

**Not a bulk deferral: six separate assessments, above.** Recorded together only to state the
finding that applies across them, and the two conclusions that required all six to see.

**Finding: zero application pressure.** Not one of the six shapes is used by any first-party
package. §12.1 step 3 asks whether a real package needs the shape; for all six the answer is no.

**§12.2's grouping rule is satisfied by no pair.** The six name FOUR different missing authorities:

```text
C4.5e runtime-method sub-slice      DEV-140 (Vec), DEV-145 (String)
std-full collections                DEV-141
C6.3e generated lifetimes           DEV-142
Eq-impl dispatch for a builtin      DEV-143
C4.5 iterator cursor lowering       DEV-144
```

DEV-140 and DEV-145 come closest — the same four-layer shape — and still fail the rule: different
receiver type, different runtime functions, different negative controls. "Both are in generated
Rust" and "both are a method sub-slice" are exactly the insufficient groupings §12.2 names.

**Why they are not repaired here.** §12.1 step 6 permits repair only if bounded; none is. §12.3
forbids widening the native claim beyond tested shapes, and with no consumer there is no shape to
test. §24 is explicit that `application hits it -> reproduce -> repair boundedly` is preferable to
`deviation exists -> redesign backend until count is zero`.

**These six continue to DEFINE the supported native subset**, which is what the original CD-342
registration was for. The layer audit keeps them honest in both directions: it fails on an
unregistered finding AND on a registered one that stops reproducing.

**Owner decision, 2026-08-10:** record the assessment and defer. Population unchanged at 9.

## DEV-159 — RESOLVED: the generated-crate directory is now mutually excluded (post-C10 P5, 2026-08-10)

Both earlier headings stand unedited, including the C10-Q pass's refusal to call it settled. That
refusal was right: it was never confirmed OR falsified, only carried.

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`.

### Reproduced, and not marginally

§11.1's rule is that a single successful build settles nothing. So: six concurrent `stark build`
invocations of ONE program, from a cold artifact directory, forty iterations.

```text
BEFORE   73 failures / 240 builds   (N=6 concurrent, debug)
```

Two distinct signatures, both from the same cause:

```text
"the STARK native backend generated a crate that Cargo could not build"
"could not install native artifact ... No such file or directory (os error 2)"
```

### Cause — and the content-addressed directory is not the fix, it is the collision

§11.2 lists "content-addressed build directory" as a candidate remedy. **It was already there**
(`compute_build_key`, since CD-044/055/067 — long before this deviation was reported) and it is
precisely why two builds of the same program land in the same directory. That reuse is deliberate
and worth keeping.

What had no sequencing was everything this compiler does *around* Cargo, all against that one
directory:

```text
reject_stale_artifact_version    can remove_dir_all the directory
write_file                       std::fs::write -- truncate-then-write, not atomic
cargo build                      Cargo locks its OWN target dir, so this part was never the problem
reading the produced binary      done by the CALLER, after build_and_link returns
```

### Repair

`BuildLock`, held from before the stale check until the artifact is dropped — which is **after the
caller installs the binary**, because that read is one of the two failure signatures. The guard
travels out in `NativeArtifact` for exactly that reason.

**The mutual exclusion is `create_dir`, not a sleep.** Creating a directory is an atomic
test-and-set: two callers cannot both succeed. The backoff is only how a loser waits, and
correctness does not depend on its duration — §11.2's "do not rely on sleeps" forbids sleeping
*instead of* synchronising, which is the opposite of this. No `unsafe` (the crate forbids it) and
**no new dependency**, which matters for a compiler whose dependency surface is `sha2` and its own
crates.

Scope is one build key. Two builds of different programs have different keys and never contend —
§11.2 permits global serialisation only if narrower isolation is impossible, and it is not. An
abandoned lock (a build killed by Ctrl-C, a CI timeout, an OOM) is broken by age, so the repair
cannot trade a race for a hang.

The lock lives BESIDE the crate directory, not inside it: `reject_stale_artifact_version` may
remove that directory while the lock is held, and a lock deleted by that removal would release
itself mid-build.

### Measured result

```text
BEFORE   73 failures / 240 builds   (N=6, debug)
AFTER     0 failures / 240 builds   (N=6, debug)
AFTER     0 failures / 200 builds   (N=8, release)
```

### Negative control — §11.3

A stress run is not a regression test: it is slow and its failure rate is a probability, so a
broken lock could pass one. What is pinned instead is the property the stress run depends on.

`build::dev159_build_lock::two_acquisitions_of_one_build_directory_cannot_overlap` — eight threads,
25 acquisitions each, counting overlaps between the guard's fences. **With the exclusion neutered
(`acquire` returning a guard without creating the directory) it fails immediately and
deterministically**, which is what §11.3 asks for; the stress run's sensitivity is a probability,
this one's is not. Two further controls pin that the guard releases on drop, and that different
build keys do not contend — a lock keyed on anything coarser would serialise the whole compiler.

Green: 579 lib, 132 conformance, 129 three-engine, 24 dev_display_dispatch, 8 dev160_call_site_thunk,
15 c64_platform_matrix. `cargo fmt --check` clean; clippy `--workspace --all-features --all-targets
-D warnings` clean.

**Residual:** measured on macOS, one machine, with a small program. The mechanism is filesystem
atomicity rather than anything program- or platform-specific, and CI covers the three Tier-1
platforms, but the STRESS NUMBERS above are from this host and are not claimed for others.

## DEV-180 — RESOLVED (post-C10 P1, repair commit `1db9760`, 2026-08-10)

The owner's ruling that this follows C10-Q was observed: the repair landed after the gate closed.

Receiver materialisation already bound `Value::Ref(caller_place)` for `&mut self`. What remained
was the epilogue from when it did not — an error path that took the callee's receiver local and
wrote it back into the caller's place, an algorithm whose only purpose was to simulate a mutable
reference by taking the value and putting it back. With a genuine reference bound, the caller's
place is never emptied, so there is nothing to restore, and the value it would have restored is
itself a `Value::Ref`. `&mut self` also joins `&self` in leaving the frame before cleanup: a
borrowed receiver must not be among the locals destroyed at method exit.

`rebase_frame_refs` is untouched — a returned `&mut` still needs rebasing out of the method frame,
and the entry's own list of forbidden repairs names removing it as the mistake to avoid.

Evidence: `as3_receiver_materialization` 7/7, `interp::tests` 144/144, and the negative controls the
entry named (`audit_10c_a_mut_self_receiver_must_keep_place_identity`, by-value self still consumes,
shared receiver does not consume).

## DEV-160 — the record corrected: `E0502` DOES reach the user (OPEN, post-C10 P2, 2026-08-10)

§8.1 asked for the exact live shape and a classification before choosing a repair layer. Both are
below, and the reproduction contradicts what this deviation was believed to be.

**Baseline SHA:** `689d26d26990399d1de3026c13c271c403a45032`.

### Correction to the record

The C10-Q-era reading — carried into the post-C10 reproduction pass — was that DEV-160's remaining
gap is *backend completeness with the boundary enforced by name*, never delegated to rustc. The
`emit_call_thunk` module header says exactly that: b, c and d are "refused by name ... because the
alternative is `E0502` inside this generated module".

**That is not true for at least one shape.** Reproducer:

```stark
struct Req { url: String, body: String }
fn send(u: &str, b: String) -> UInt64 { return u.len() + b.len(); }
fn main() {
    let r = Req { url: "http://x".to_string(), body: "hi".to_string() };
    println(send(r.url.as_str(), r.body));
}
```

```text
stark run   -> 10
stark build -> error: the STARK native backend generated a crate that Cargo could not build
               generated crate: error[E0502]: cannot borrow `_1` as mutable because it is also
               borrowed as immutable  --> src/main.rs:103:96
```

**This is the failure §8 names as the one the architecture must not have:** STARK accepts, generated
Rust is emitted, rustc rejects it, and the user is shown a borrow error about a line they did not
write.

### Classification (§8.1): D, with a DETECTION gap

Not A or B — the front end is right, the accesses are disjoint. Not C. It is D, and the specific
mechanism is that the refusal written for this shape is **unreachable for it**:

```text
plan_for_call
  absorbable_borrows(...)      derived from the call's OWN block
  if !conflicts(...) -> None   <-- returns here for a cross-block borrow
  ...
  DEV-160b refusal             <-- written for `builder.url.as_str()`, never reached
```

`as_str()` runs in an earlier block, so its `&str` is not among this block's borrows; the argument
list looks like a single access to `_1`; no thunk is planned; and the named refusal downstream never
runs.

### A repair was implemented, measured, and REVERTED

Making `conflicts` consult `borrow_provenance` (which does trace across blocks) makes the refusal
reachable, and the reproducer above then fails with the named DEV-160b message and its workaround
instead of `E0502`. All compiler suites stayed green, including `dev160_call_site_thunk` 8/8.

**It over-refuses shipping code.** `stark-get` — the HTTPS application — stopped building:
`stark_http_client::follow@[]` was refused, and `follow` is a path that builds today and is already
written in the workaround form this deviation recommends. Provenance over-approximates by design: it
reports that a value derives from a slot, not that a borrow of that slot is still live where rustc
looks. Narrowing it to exactly rustc's answer means modelling rustc's borrow checker over generated
code, which is §19.3's "repair requires architectural expansion".

Reverted. `stark-get` builds again. **The measurement is kept because it is the finding:** the cheap
fix for the detection gap is not admissible, and that is worth more than an untested claim that it
would work.

### Disposition

OPEN. §23's exit criterion 3 for this deviation — "its exact supported boundary is enforced by STARK
rather than delegated accidentally to rustc" — is **NOT met**, and this entry now records why
rather than asserting that it is.

What is now known that was not before: the boundary has a hole; the hole is a reachability gap in
`plan_for_call`, not a missing refusal; the obvious closure over-refuses real code; and the honest
closure needs either cross-block absorption (DEV-160b's own deferred work package, owner ruling
2026-08-03) or a liveness-accurate provenance that provenance is not.

**Residual for users:** the shape has a working spelling — bind the fields to locals before the
call — which is what `stark-http-client` already does and documents at `src/lib.stark`.

## DEV-160 — the rustc leak is sealed; the capability half remains (OPEN, post-C10 P2 continued, 2026-08-10)

Supersedes the disposition immediately above, which recorded that no bounded closure was
admissible. **That was true of the repair attempted there, and not true in general.** The entry is
left standing because the failed attempt is the reason this one is correct.

### What the first attempt got wrong

Making `conflicts` consult `borrow_provenance` over-refused `stark_http_client::follow`. The cause
was one propagation rule, not the idea:

```rust
Rvalue::Use(Operand::Copy(p) | Operand::Move(p)) => vec![p.local.0],
```

**A move transfers ownership.** `follow` does `let mut url = builder.url;` and then
`send_once(.., url.as_str(), headers, body)`. After that move, `url` owns its own slot, and
borrowing it does not borrow `builder` — but provenance propagated through the move, so the
reference looked like a borrow of `builder`, whose siblings were also moved. The reproducer is
genuinely different: it borrows `r.url` **in place**, while `r` still owns it.

### Repair

```text
borrow_provenance   a MOVE severs provenance; only copies and references propagate
conflicts           consults provenance, so a borrow arriving from an EARLIER block is visible
                    before the early return that was skipping the DEV-160b refusal
```

The refusal was never missing. It was **unreachable for its own case**: `plan_for_call` returned
`None` first, because a cross-block borrow is not among the call block's own borrows.

### Measured, in both directions — the direction that caught the first attempt

```text
reproducer `send(r.url.as_str(), r.body)`   E0502 in generated code -> REFUSED BY NAME
stark-get (HTTPS application)               builds, unchanged
5 consumer applications                     build, unchanged
dev160_call_site_thunk                      8/8, incl. the DEV-160d under-refusal guard
```

**The accepted program set is unchanged.** Eight borrow/move shapes built with the pre- and
post-repair compilers and classified:

```text
in-block borrow + sibling move        BUILDS -> BUILDS
fields moved to locals first          BUILDS -> BUILDS      (the `follow` shape)
borrow + Copy sibling                 BUILDS -> BUILDS
borrow outliving the call             REFUSED -> REFUSED    (DEV-160d, unchanged)
two sibling moves                     BUILDS -> BUILDS
borrow + sibling .len() in one call   BUILDS -> BUILDS
cross-block `as_str()` + move         E0502  -> REFUSED BY NAME
same, with a trailing field read      E0502  -> REFUSED BY NAME
```

Only the two leaks changed, and each became a named refusal. **Nothing that built stops building
and nothing newly builds** — §12.3 is satisfied without a subset claim being widened.

### What this does and does not settle

§23's exit criterion 3 — "the exact supported boundary is enforced by STARK rather than delegated
accidentally to rustc" — is now **MET for the demonstrated shape**. `E0502` no longer reaches a
user for it.

DEV-160 **stays OPEN** for the capability half: these programs are valid STARK and still do not
build. Closing that is DEV-160b's cross-block absorption — the thunk absorbing the borrow-producing
call across the block boundary — which remains its own deferred work package under the owner ruling
of 2026-08-03.

### The residual risk, stated rather than hidden

Provenance answers "this value may derive from that slot", not "a live borrow of that slot reaches
this call". Those differ, and the first attempt is proof that the gap has teeth. Severing on moves
removes the false positive that mattered, and the corpus and package evidence above found no other
— **but absence of a counterexample is not a proof of precision.**

The precise mechanism is a backward walk of the reference's own def-use chain (producer call →
its reference argument → the `RefOf` that seeded it), admitted only when the producer dominates the
consumer on one straight-line path with nothing observable between. That analysis is needed anyway
for cross-block absorption to decide what it may absorb, and it should replace the provenance
heuristic here when it lands — one authority deciding both, which is the property this module
already holds elsewhere. Recorded now so the interim is retired deliberately rather than forgotten.

# Reconciliation: sixteen IDs COMPILER-STATE.md tracked and this ledger never did

`c10-deviation-populations.py` reports a bucket named "POPULATION A — named in COMPILER-STATE.md,
owning no heading here". It held sixteen IDs. They were never open defects hiding from the tool;
they were resolved in `COMPILER-STATE.md` prose and never given a heading the ledger could
classify. The consequence was narrow and real: **"population A is N" was only ever true of the
ledger-derived set**, and a reader of `COMPILER-STATE.md` got a different number.

Each is closed below with the evidence that settled it. **Probed, not taken on the prose's word** —
this repository's own record is that 7 of 23 deviations did not reproduce at the C10-Q anchor, and
that DEV-157 was "one probe away from being filed as a false closure". A paragraph asserting a fix
is exactly the thing that needs checking.

Probes run at `9a0557f`, outside the repository tree.

## DEV-091 — out-of-range float→int cast at 64-bit widths (RESOLVED, reconciled 2026-08-10)

```stark
let f: Float64 = 1.0e19; let n: Int64 = f as Int64;
```

```text
HIR     Error: runtime error: numeric cast out of range
native  error: runtime trap: cast failure
```

Traps in both engines. `1.0e19` exceeds `Int64::MAX` (~9.22e18) and is precisely the shape the
defect admitted, because both sides compared against `max as f64`, which rounds UP at 64-bit width.

## DEV-096 — the trap CATEGORY for an out-of-range cast (RESOLVED, reconciled 2026-08-10)

Settled by the same probe, and it is a different claim from DEV-091's: the oracle reported every
out-of-range cast as an arithmetic overflow. It now says **"numeric cast out of range"**, and native
says **"cast failure"** — the cast category, not the arithmetic one. Recorded separately because a
probe that only checked "it traps" would have passed while the category was still wrong.

## DEV-097 — bounds-check span blame (RESOLVED, reconciled 2026-08-10)

```stark
let v: [Int32; 3] = [1, 2, 3]; let i: UInt64 = 7u64; println(v[i]);
```

Traps `index out of bounds` in HIR and native. **Bounded claim:** this probe confirms the trap
fires and agrees across engines; the entry's specific complaint was that the two ends of one bounds
check blamed different columns, and a full column-level re-check is not claimed here.

## DEV-099 — a layout query on an ARRAY type (RESOLVED, reconciled 2026-08-10)

The entry read as a live pre-existing defect: `size_of::<[Int32; 4]>()` "reaches lowering and dies
with 'field type form (C4.5)'".

```text
HIR     16
native  builds, prints 16
```

**It does not reproduce.** Fixed at some point between 2026-07-23 and now, and never recorded.
This is the one of the sixteen that most needed a probe rather than a reading.

## DEV-092 — symbol sanitization injectivity (RESOLVED, reconciled 2026-08-10)

`backend::generated_rust::mangle` states injectivity as its purpose in three places and carries the
round-trip-through-a-decoder test the entry called for. `mangle::` suite green, 9 passed.

## DEV-095 — the generated-crate build key (RESOLVED, reconciled 2026-08-10)

Recorded as a WP-C5.3 opening condition and explicitly NOT fixed at the time. It is fixed now:
`build.rs`'s test module is titled "DEV-095's cache-invalidation coverage — every semantic input
that can affect generated code must change the build key", with one test per input so a failure
names the input that stopped being covered. `build::tests` green, 24 passed.

## DEV-101 — cross-package generic typecheck provenance (RESOLVED, reconciled 2026-08-10)

`tests/cross_package_generics.rs` green, 11 passed.

## DEV-093 — native success-path tests observed no computed values (RESOLVED, reconciled 2026-08-10)

Both were recorded FIXED in `COMPILER-STATE.md` when found, and neither is a user-visible language
behaviour a probe can reach: DEV-093 was native success-path tests asserting only `exit == 0`
(fixed by making them observe computed values, which is what every three-engine case now does), and
DEV-094 was the version-mismatch message naming the wrong version on each side. Closed on the
recorded evidence, and this heading says so rather than implying a probe was run.

## DEV-098 — `Operand::Copy` on a `&mut` reference (ACCEPTED-INDEFINITELY, reconciled 2026-08-10)

Never a defect. A deliberate, verifier-accepted MIR shape that the `Copy` classification does not
describe, recorded as NOT a regression when found. Dispositioned rather than closed, on OD-7's
distinction: it needs an owner and a statement, never a repair.

## DEV-002 — stale conformance counts (RESOLVED, reconciled 2026-08-10)

A tooling-hygiene finding. `check-conformance.py` now warns on `missing` entries that still carry
`source`/`tests` fields and on likely-semantic-rejection rules with zero recorded tests.

## DEV-094 — the version-mismatch message named the wrong version on each side (RESOLVED, reconciled 2026-08-10)

`version::check` assigned the LINKED runtime's `RUNTIME_VERSION` to `expected_runtime_version` and
the generation-time value to the other side, so each half of the message named the wrong one.
Recorded FIXED in `COMPILER-STATE.md` when found. Not a language behaviour a probe can reach, and
this heading says that rather than implying one was run.

## DEV-158 — install through a whole-value accessor (RESOLVED, CD-371, reconciled 2026-08-10)

Already closed under its own CD in `COMPILER-STATE.md`; what was missing was a heading here. Its
sibling DEV-162 closed under CD-372, and DEV-160a under CD-374 — the same family, and the only one
of the four still live is DEV-160's capability half.

## DEV-163 — a read timeout did not report as a timeout on Unix (RESOLVED, CD-375, reconciled 2026-08-10)

```text
Unix     SO_RCVTIMEO expires -> EAGAIN       -> ErrorKind::WouldBlock -> Interrupted
Windows  SO_RCVTIMEO expires -> WSAETIMEDOUT -> ErrorKind::TimedOut
```

One platform reported the wrong classification for the same condition — the platform-divergence
shape `stark-layout-verification` exists for. Closed under CD-375.

## DEV-164 — closed in the same packet as DEV-163 (RESOLVED, CD-375, reconciled 2026-08-10)

## DEV-182 — a parser decoded escaped non-BMP characters to the empty string (RESOLVED, reconciled 2026-08-10)

Closed, and the reason it is still cited is worth keeping: **both sides reported success and only
the VALUE was wrong**, so it passed protocol validation. That is why C10-B promises diagnostic
codes, spans and text SEPARATELY from determinism — byte-identical output for the same source says
nothing about whether the output is right.

## DEV-165 — `connect_timeout` accepted and ignored — POPULATION B, not A (reconciled 2026-08-10)

**The one of the sixteen that is genuinely open — and it is not Population A.** It is an
HTTP-client defect, not a compiler one; the audit that found it said so explicitly. It belongs to
**Population B (release/distribution)**, which constrains public wording rather than conformance,
is frozen by hand per OD-3, and is already owned: *deferred to the networking roadmap*.

Given a heading here so the tool stops reporting it as unclassified, **not** to move it into
Population A. It is not a compiler-track defect and must not inflate that count.

**This heading deliberately carries no bare `OPEN`, and therefore sorts to ADJUDICATE.** That is the
correct answer, not a dodge: the tool's own note says adjudicate is where a human decides and a
regex that guessed would be doing it badly. Which population an ID belongs to is exactly such a
decision. It is OPEN — in Population B, which OD-3 freezes by hand and which this file cannot
derive.

## DEV-221 — a qualified core-trait call on a BOUNDED generic parameter (OPEN, registered 2026-08-10)

Registered so it stops being a residual paragraph inside DEV-168 with no number of its own.

```stark
fn show<T: Display>(x: &T) -> String { return Display::fmt(x); }
```

```text
[E0500] type 'T' does not implement 'Display'
```

`check_qualified_core_trait_call` selects by scanning impls for the receiver's nominal type and
never consults the parameter's BOUNDS, so a bounded generic receiver finds nothing. A **front-end
over-rejection**: `T: Display` says T implements Display, so TYPE-METHOD-001's qualified form
should be available.

Distinct from DEV-168, which was "type-checks and runs under the oracle, MIR refuses" — this shape
does not type-check at all, in any engine. Found while proving DEV-168's repair.

**Working spelling exists:** the ordinary method form `x.fmt()` works, which is what DEV-166's
repair delivered. Severity is ergonomic, not a correctness leak.

## DEV-165 — RESOLVED: `connect_timeout` is applied, not accepted and ignored (2026-08-10)

The reconciliation heading above recorded this as Population B, open, and deferred to the
networking roadmap. **It is repaired instead.**

### The defect went one layer deeper than the entry said

The entry named the HTTP client, and the client was not the cause. `stark_net::connect` — the one
connect API that takes a deadline — was:

```stark
pub fn connect(address: SocketAddress, timeout: Duration) -> Result<TcpStream, NetworkError> {
    if !timeout.is_zero() { return Err(NetworkError::Unsupported); }
    connect_socket_address(&address)
}
```

**It refused every non-zero timeout**, and succeeded only for a zero duration — which reads as "no
timeout" and is the opposite of what passing a `Duration` means. So `stark-http-client` used
`connect_no_timeout` and was *correct to*: calling `connect` with its configured deadline would
have failed every connection. Switching only the client would have broken it.

`connect_to_any`'s own doc comment promised the behaviour the code did not have — "**timeout
budget: PER ADDRESS ... each gets the deadline the caller asked for**" — sitting directly above the
untimed call.

### Repair, across four layers

```text
starkc/providers/stark-net-native.json   `stark_tcp_stream_connect_timeout`: buffer_in,
                                          scalar_in u64, handle_out
stark-net/native/src/lib.rs               the entry point, over TcpStream::connect_timeout;
                                          plus the linkage extern and BOTH symbol-set lists
stark-net/src/lib.stark                   `connect` routes through it
stark-http-client/src/lib.stark           `connect_to_any` passes `config.connect_timeout`
```

`nanos` follows the convention `set_read_timeout` already established rather than inventing a
second one — **zero means no timeout, and a non-zero duration that rounds to zero is raised to
1ns** rather than silently becoming "block forever". At the STARK layer a zero duration is
*rejected*, the same rule `timeout_nanos` applies to the read and write setters: zero is the one
value where intent is ambiguous, and a caller who wants no bound says so by name with
`connect_no_timeout`.

The address is parsed in the provider rather than resolved there: `connect_timeout` takes one
`SocketAddr` by construction, so resolving inside it would hide the per-address budget. Resolution
and ordering stay in `stark-net`'s STARK surface where the caller can see them.

### Evidence — a control someone else wrote, which fired

`stark-net-resource-consumer` carried this, written when the defect was recorded:

```text
"Pinned rather than hidden: when a connect timeout lands, this assertion fails and forces this
 consumer to be updated"
```

It required `TcpStream::connect(peer(), duration_seconds(5u64))` to FAIL. Against a live peer it
now panics with *"connect-with-timeout works now — implement it properly and update this"*. **A
control that was passing only because the feature was broken.** Its polarity is corrected: a real
deadline connects, and a **new negative control** requires a zero duration to be refused — without
which a repair that ignored the duration entirely would still pass.

**The deadline is real, and measured.** Connecting to 203.0.113.1 — TEST-NET-3, RFC 5737,
reserved for documentation and never routed, so a connection attempt has nowhere to go:

```text
connect(target, duration_seconds(2u64))   returned in 2.479s
```

Green: `stark-net-native` 11/11 (including both manifest/symbol-set gates, which caught two lists
this repair had to update); starkc lib 579; a10 provider suites verify/resolve/bind/emit/call/
resource; `stark-net-resource-consumer` prints `STARK_NET_RESOURCE_OK` against a live peer;
`stark-http-client-consumer`, `stark-get` and `stark-tls-consumer` all build.

### Residual

`connect_no_timeout` remains, and is still the honest spelling for an unbounded connect. The HTTP
client no longer uses it; `stark-net-resource-consumer` does, so the declared-surface gate is
still satisfied.

---

## DEV-222 — a pattern naming a variant that does not exist compiles, and silently never matches (OPEN, registered 2026-08-11)

Found by `stark-cookie` at `2cd4a08`. **This is a wrong-code defect, not an over-rejection.**

```stark
enum Colour { Red, Green }

fn describe(c: &Colour) -> String {
    match *c {
        Colour::Blu => String::from("blue"),   // TYPO. `Colour` has no variant `Blu`.
        Colour::Red => String::from("red"),
        _other => String::from("wildcard"),
    }
}

fn main() {
    println(describe(&Colour::Green).as_str());
}
```

```text
probe: OK
wildcard
```

`stark check` reports OK. The misspelled arm is treated as a pattern that never matches, and the
value falls to the wildcard. The same holds for a variant path on a **struct**, which can have no
variants at all:

```stark
struct Thing { value: Int64 }
// ...
match *r {
    Thing::Missing(n) => println(n),          // `Thing` is a struct
    _other => println("fell through to the wildcard"),
}
```

### The diagnostic pathology makes it worse

Remove the wildcard and the program *is* rejected — but by the wrong diagnostic:

```text
Error: [E0303] non-exhaustive pattern match
```

E0303 points at the `match`, not at the typo. The obvious response to "non-exhaustive" is to add a
wildcard arm — which converts a caught bug into a silent one. **The diagnostic leads the developer
into the failure mode.**

### Where it is not

`resolve.rs`'s three pattern branches are correct and already guard for this:

- `ast::PatKind::Path` (L1388) emits `E0200 undefined pattern path` when `res == Res::Err`
- `ast::PatKind::TupleVariant` (L1404) emits `E0202 undefined enum variant` when `res == Res::Err`
- `ast::PatKind::Struct` (L1421) emits `E0202 undefined struct/variant` when `res == Res::Err`

None of them fires, so **`resolve_path` is returning something other than `Res::Err` for
`Type::NonexistentName`.** The guards are right; their input is wrong. That is the single site to
repair, and repairing it should light up all three branches at once.

### Blast radius

Any misspelled variant in a `match` with a wildcard arm. It found `stark-cookie` exactly that way:
after `CookieAttribute` changed from an enum to a struct, a test still carrying the old
`CookieAttribute::MaxAge(seconds)` pattern kept compiling and began failing at runtime instead of
at the type error that should have caught it. A refactor that renames or restructures an enum gets
no help from the compiler — the arms silently stop matching.

No package workaround exists and none is needed: this is a missing rejection, not a shape to code
around.

### Precedent: this is the same class as DEV-053/054

`DEV-053` — *"a bare `None` pattern never matched by value; it silently acted as an unconditional
wildcard"* — is the same failure: a pattern path that does not resolve to a variant becomes
something that silently does not match, with wrong runtime output and no diagnostic. C2's exit
report calls DEV-053/054 **"the most severe finding to date"** in the compiler track.

DEV-053 was fixed for the specific case of a *bare identifier* resolving to a builtin
(`lower_pattern`'s `ast::PatKind::Binding` arm). **The general case was never closed**: a
QUALIFIED path naming a variant that does not exist still resolves to something that is not
`Res::Err`, and the three pattern branches' guards therefore never fire. DEV-222 is the same
defect class recurring one resolution path over.

## DEV-223 — a variant sharing a name with an in-scope type is reported non-exhaustive (OPEN, registered 2026-08-11)

Found by `stark-cookie` at `2cd4a08`.

```stark
enum Policy { A, B }

enum Attr {
    Flag,
    Policy(Policy),     // variant name == type name
}

fn render(attr: &Attr) -> String {
    match *attr {       // [E0303] non-exhaustive pattern match
        Attr::Flag => String::from("flag"),
        Attr::Policy(p) => match p {
            Policy::A => String::from("a"),
            Policy::B => String::from("b"),
        },
    }
}
```

Renaming the variant — changing nothing else — compiles and runs correctly. The match is
exhaustive as written; the rejection is spurious.

**Probably the same root cause as DEV-222**, and filed separately only because the observable
symptom and the developer's experience differ. The hypothesis, stated as a hypothesis because it
was inferred from behaviour rather than read out of `resolve_path`: the path `Attr::Policy`
resolves its final segment to the *type* `Policy` rather than to the variant, the arm therefore
carries a non-`Res::Variant` resolution, exhaustiveness does not count it, and the remaining arms
do not cover the enum. If that is right, one fix in `resolve_path` closes both.

**DEV-053's history is a direct precedent and a warning.** That entry *originally* recorded a
"spurious `E0303` non-exhaustive" — DEV-223's exact symptom — and investigation found the cause was
not the exhaustiveness algorithm at all but pattern resolution in `lower_pattern`, with a silent
wildcard as its other face. The same reading should be applied here before assuming this one is
merely cosmetic.

**Severity as observed: over-rejection, fail-safe.** It fails loudly at compile time and can never
produce wrong output. Its cost is that `Enum::Variant(SameNamedType)` — an ordinary and idiomatic shape —
is unavailable, forcing a name that exists only to dodge a compiler bug. `stark-cookie` carries
`CookieAttributeKind::SameSitePolicy` for this reason; `SameSite` is the name that belongs there.

## DEV-224 — native: an enum carrying a non-Copy payload cannot be matched through a shared reference (OPEN, registered 2026-08-11)

Found by `stark-cookie` at `2cd4a08`. A **native backend gap**, not a front-end defect: the
front end and both interpreters accept the program, and `stark build` refuses it.

```stark
enum Attr {
    Flag,
    Text(String),
    Num(Int64),
}

fn kind_only(a: &Attr) -> Int64 {
    match *a {
        Attr::Flag => 0i64,
        Attr::Text(_) => 1i64,      // `_`, binding nothing
        Attr::Num(_) => 2i64,
    }
}
```

```text
error: native build does not yet support this program:
       binding a non-Copy scrutinee through a shared reference
```

**Even `_` patterns fail.** The rejection is about the scrutinee — a non-`Copy` enum read through
`&` — not about what the arms bind, so there is no arm spelling that avoids it. An enum is
non-`Copy` as soon as one variant carries a `String`.

### Why it matters more than its wording suggests

`enum { A(String), B(Int64), C }` held in a `Vec` and read by reference is the ordinary shape for
an AST node, a JSON value, a config entry, a protocol attribute — anything tagged. Under this gap
none of them is natively compilable, and native compilation is the shipping path for
capability-backed programs. Interpreter-only verification will not surface it.

### Package-level alternative, and its cost

A tagged struct compiles and runs natively:

```stark
struct Attr { kind: Kind, text: String, num: Int64 }   // `Kind` fieldless, therefore Copy
```

`match a.kind` reads a `Copy` field through the shared reference and is accepted. `stark-cookie`'s
`CookieAttribute` is built this way for exactly this reason.

The cost is real and is in a shipped public API: a sum type makes an invalid combination
*unrepresentable*, while a tagged struct makes it merely unconstructible-by-convention. A
`Secure` attribute carrying a value cannot exist in the enum formulation; in the struct
formulation it is prevented only because every caller goes through a constructor.

## DEV-223 — REVISED: it is not fail-safe. The constructor face type-checks and fails at RUNTIME (2026-08-11)

The original entry, registered earlier the same day, called this "over-rejection, fail-safe … can
never produce wrong output". **That is wrong, and this heading corrects it.** The name collision has
a second face in EXPRESSION position that the pattern reproducer hid:

```stark
enum Policy { A, B }
enum Attr { Flag, Policy(Policy) }

fn build() -> Attr { Attr::Policy(Policy::A) }   // an ordinary enum constructor
```

```text
probe: OK
Error: runtime error: item is not callable
  --> probe/src/main.stark:5:22
```

`stark check` passes. The failure is deferred to **runtime**. A valid enum constructor is
unusable, and nothing in the front end says so.

Severity is therefore **not** ergonomic. Pattern position over-rejects loudly (spurious `E0303`);
expression position accepts and traps.

### Root cause, read out of the source rather than inferred

`resolve_path_relative` (`starkc/src/resolve.rs`), subsequent-segment loop. For `Attr::Policy`,
segment 0 resolves `Attr` to `Res::Item`, and because `Attr` is not a submodule `current_mod` stays
the *enclosing* module. Segment 1 then hits this branch first:

```rust
} else if let Some(&res) = self.modules[current_mod.0 as usize].items.get(name_str) {
```

`Policy` IS an item of that module — the enum type — so it wins, and `current_res` becomes
`Res::Item(policy_enum)`. The enum-variant branch below it is never reached:

```rust
} else if let Some(Res::Item(item_id)) = current_res {
    match self.item_details.get(&item_id) {
        Some(ItemDefDetail::Enum { variants }) => { /* the variant lookup, never reached */ }
```

**The module-item lookup is consulted before the qualifying item's own variants**, so any
module-level name shadows a same-named variant of the enum being qualified. This supersedes the
earlier entry's hypothesis that the final segment resolved to the type "somehow"; it is an
ordering bug, and the ordering is visible in the source.

### Relationship to DEV-222 — they are NOT the same defect

The earlier entry guessed one shared root cause. Reading the function shows two distinct ones in
the same loop:

- **DEV-223** is the branch ORDER above: module items before the qualifier's variants.
- **DEV-222** is the enum/struct fallback below it: a name that is not a variant becomes
  `Res::AssociatedFn(item_id, span)`, not `Res::Err`.

```rust
Some(ItemDefDetail::Enum { variants }) => {
    if let Some(variant_idx) = variants.iter().position(|v| v == name_str) {
        current_res = Some(Res::Variant(item_id, variant_idx as u32));
    } else {
        current_res = Some(Res::AssociatedFn(item_id, segment.span));   // DEV-222
    }
}
Some(ItemDefDetail::Struct { .. }) => {
    current_res = Some(Res::AssociatedFn(item_id, segment.span));       // DEV-222, struct face
}
```

**That fallback is correct for expressions and must stay.** `Duration::from_seconds`,
`Instant::now`, `Line::new` and `UnixTimestamp::from_unix_seconds` are user-declared associated
functions reached exactly this way — over sixty call sites across `packages/`. Making these arms
return `Res::Err` would break every one of them.

DEV-222 is therefore **not** a defect in `resolve_path`. `resolve_path` is answering the question it
was asked. The defect is that *pattern* lowering accepts a resolution that cannot be a pattern:
`Res::AssociatedFn` is never a valid pattern, and `lower_pattern`'s three branches test only
`res == Res::Err` before emitting `E0200`/`E0202`. The repair belongs in that guard — widening it
from "is it `Res::Err`" to "is it a resolution a pattern may name" — or in a pattern-aware
resolution entry point. It does not belong in the shared expression path.

---

## DEV-222 — RESOLVED (2026-08-11): pattern lowering asks what a pattern may name

Repaired where the defect was, which was **not** `resolve_path`. That function answers for
EXPRESSION position, and `Res::AssociatedFn` for a name that is not a variant is the correct answer
there — `Duration::from_seconds`, `Instant::now` and `Line::new` reach their definitions through it,
sixty-odd call sites across `packages/`. Returning `Res::Err` from those arms would have broken
every one.

`resolve.rs` gains `resolution_is_pattern_legal`, exhaustive over `Res`, consulted by all three
pattern branches through one `reject_non_pattern_resolution` helper. The branches previously asked
only `res == Res::Err`. The diagnostic now names the path — `'Colour::Blu' is not a pattern; no such
variant exists` — instead of `E0303 non-exhaustive` pointing at the `match`, which was the wording
that invited a wildcard and made the bug silent.

Regression: `starkc/tests/dev222_pattern_only_resolutions.rs`, six tests. Verified to fail against
the unfixed compiler and to pass against the repair, with the accepting side (valid unit and tuple
variants, associated functions in expression position, `Some`/`None`) pinned as controls.

## DEV-223 — RESOLVED (2026-08-11): the qualifier's associated namespace is searched first

`resolve_path_relative`'s subsequent-segment loop consulted the enclosing module's items before the
variants of the item being qualified. Now a qualifier that owns associated names answers first, via
`qualified_associated_name`, and a module qualifier still falls through to the module namespace —
tracked explicitly with `current_is_module`, because `crate`/`super` stash a placeholder `Res::Item`
that the first version of this repair misread as a real type (caught by DEV-148's suite).

Both faces close: the exhaustive match compiles, and `Attr::Policy(Policy::A)` constructs instead of
trapping at runtime with `item is not callable`.

Regression: `starkc/tests/dev223_variant_shadowed_by_a_type.rs`, six tests. The constructor face is
EXECUTED through the interpreter rather than stopping at `typecheck`, because that face passed both
front-end stages and failed at runtime — a test that stopped earlier passed against the defect.

## DEV-225 — RESOLVED on arrival (2026-08-11): associated-name precedence, beyond enum variants

Found by auditing outward from DEV-223 rather than by hitting it. NAME-RESOLVE-001 in
`04-Semantic-Analysis.md` says: *"Associated names are searched only after resolving their
qualifying type or trait."* DEV-223 was the enum-variant face; the rule covers structs, traits and
models, and all of them lost to the enclosing module.

```stark
struct Foo { a: Int64 }
impl Foo { pub fn new() -> Foo { Foo { a: 1i64 } } }
fn new() -> Int64 { 99i64 }
fn main() { let f = Foo::new(); println(f.a); }
```

Before: `E0001 cannot access field 'a' on non-struct type 'Int64'` — `Foo::new` had resolved to the
module-level `new`. After: prints `1`.

**This is a conformance deviation against a numbered normative rule**, not a preference. The repair
is the same `qualified_associated_name` helper DEV-223 introduced, generalised from enum variants to
every qualifier that owns associated names.

## DEV-226 — RESOLVED on arrival (2026-08-11): only builtin CONSTRUCTORS are patterns

Every `Res::Builtin` was accepted as a pattern. Most builtins are functions.

```stark
match v { Vec::new(x) => println(x), _other => println("fell through") }
```

`stark check` reported OK and the program printed `fell through` — DEV-222's failure mode one
namespace over. `hir::builtin_is_pattern_constructor` is now exhaustive over `Builtin`: `Some`,
`None`, `Ok`, `Err`, the three `Ordering` variants and the five `IOError` variants are constructors;
everything else is a function or a constant expression and is refused in pattern position.

## DEV-227 — RESOLVED on arrival (2026-08-11): a bare identifier is a value pattern only for a variant or a constant

Every `Res::Item` was taken by value in the bare-identifier pattern branch, so a name binding a
FUNCTION became a value pattern that could never equal anything:

```stark
fn helper() -> Int64 { 1i64 }
fn main() { let n = 3i64; match n { helper => println("x"), _other => println("fell") } }
```

compiled and printed `fell`.

`02-Syntax-Grammar.md` SYN-PATTERN-001 states the rule: a single identifier pattern "that resolves
to a unit enum variant or a constant in scope matches by value; otherwise it introduces a new
binding." **The repair is therefore to BIND, not to reject** — an audit suggestion to make this an
error would have contradicted the grammar. `ItemDefDetail` gains a `Const` variant so a constant
stays distinguishable from a function, which it was not before.

Same-session regressions for DEV-225/226/227:
`starkc/tests/dev225_227_resolution_namespaces.rs`, eight tests including the module-path and
`super::` controls the precedence reorder could most plausibly have broken.

## DEV-228 — the resolver has ONE namespace map where NAME-RESOLVE-001 specifies four (OPEN, registered 2026-08-11)

**Architectural, and the reason DEV-222/223/225 were all easy to reach.**

NAME-RESOLVE-001: *"Core has distinct module, type, value, and associated-item namespaces… The same
spelling may coexist in different namespaces, but two declarations in one namespace and scope are
duplicates."*

`Resolver::ModuleData` carries `items: HashMap<String, Res>`, and `declare_items` puts functions,
structs, enums, traits, constants, aliases, modules and models into that one map. So the coexistence
the rule permits is rejected:

```stark
struct Pair { a: Int64 }
fn Pair() -> Int64 { 5i64 }
```

```text
Error: [E0204] duplicate definition of 'Pair' in the same scope
```

A type and a value sharing a spelling is legal per the rule and is refused.

**No downstream validator can recover this.** Once both declarations collapse into one map entry,
the namespace distinction is gone before any later pass can consult it — which is why this is filed
as architecture rather than as another precedence exception. DEV-223 and DEV-225 were both repaired
by moving one lookup ahead of another; a third and fourth exception on the same single map is the
wrong direction. The resolver should carry the namespaces the rule names.

Not repaired here. It is a resolver-model change, not a bug fix, and it wants a compiler-track
decision rather than a package-derived session.

## DEV-229 — qualified builtin paths are matched by SPELLING before name resolution runs (OPEN, UNCONFIRMED, registered 2026-08-11)

`resolve_path_relative` opens with a `match self.path_to_string(path).as_str()` over roughly thirty
hard-coded spellings — `"String::from"`, `"Vec::new"`, `"HashMap::new"`, `"Ordering::Less"`,
`"IOError::Other"` and the rest — returning `Res::Builtin` before any module, import or item lookup
happens.

NAME-RESOLVE-001 gives no precedence to builtin spellings over declared names, so a user declaration
occupying one of those spellings can never win. Whether these names are *reserved* is not stated in
the specification; the implementation currently assumes a reservation the language does not express.

**Filed UNCONFIRMED, deliberately.** A probe declaring `enum Ordering { Less, Equal, Greater }` and
matching `Ordering::Less` behaved correctly, but that test cannot separate "the user's enum won"
from "the builtin won and happened to agree", so it establishes nothing either way. What is certain
is the code path; what is not is whether any reachable program observes a wrong answer. Closing this
needs either a specification statement that the spellings are reserved, or a probe that distinguishes
the two resolutions.
