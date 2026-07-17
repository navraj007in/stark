# STARK Compiler — Repository and Pipeline Map

WP-C0.1 deliverable. Ground truth as of head `6fa8c15b94bd1376a847132498d31dd356524180`
(2026-07-17). Every claim below is cited to `path:line`; where a claim could not be verified to
that level it is marked accordingly. This document supersedes the module-layout tables in
`starkc/README.md`, which are stale (see `COMPILER-STATE.md` CD-003 and WP-C0.2).

Contents:
1. Module table (purpose, input, output, entry points)
2. Shared vs. duplicated compiler entry points
3. Global state, filesystem/process access, nondeterministic collections
4. Source-file and module provenance flow
5. Test files per subsystem
6. Stubs and deliberately incomplete handlers
7. Archived code that must not be modified

---

## 1. Module table

`starkc/src/lib.rs` declares 21 `pub mod`s as the crate root. Three binaries build on top of the
library: `starkc` (`src/main.rs`), `stark` (`src/bin/stark.rs`), `starkide`
(`src/bin/starkide.rs`).

### Front end / core pipeline

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `lexer.rs` | 1,141 | Hand-written lexer per `01-Lexical-Grammar.md`: maximal munch, keyword/reserved/identifier classification, literal-suffix rules, nested block comments, raw strings, recovery lexing. | `&SourceFile` | `(Vec<Token>, Vec<Diagnostic>)`; with comments: `(Vec<Token>, Vec<Comment>, Vec<Diagnostic>)` | `tokenize(file)` (`lexer.rs:255`), `tokenize_with_comments(file)` (`lexer.rs:264`) |
| `parser.rs` | 3,548 | Recursive-descent parser for `02-Syntax-Grammar.md`; per-precedence-level expression parsing (16 levels), struct-literal restriction, generic `>` splitting, multi-file/package parsing incl. `mod` submodule loading. | `&SourceFile` or `&PackageGraph` | `(Ast, Vec<Diagnostic>)` | `parse(file, mode)` (`parser.rs:37`, Core-only), `parse_with_options(file, mode, options)` (`parser.rs:43`), `parse_project(root_file, options)` (`parser.rs:59`), `parse_package_graph(graph, options)` (`parser.rs:91`), `parse_with_options_into(...)` (`parser.rs:330`, low-level per-file entry; owns diagnostic file-provenance backfill, see §4) |
| `ast.rs` | 746 | Arena-allocated AST (typed IDs, not references/lifetimes; names as `Span`s; no grouping-paren nodes). | n/a (populated by parser) | `Ast { types, exprs, stmts, items, pats, blocks, dims, root, item_files: HashMap<ItemId, Arc<SourceFile>>, synthetic_spans }` (`ast.rs:32-46`) | `alloc_*`/accessor methods (`ast.rs:660-707`) |
| `resolve.rs` | 2,209 | Name resolution + AST→HIR lowering (PLAN.md M2.1): module tree, path/import/glob resolution, builtin/primitive/core-trait/core-type resolution, tensor-extension name gating. | `&Ast`, `Arc<SourceFile>`, `LanguageOptions` | `(Hir, Vec<Diagnostic>)` | `resolve(ast, file)` (`resolve.rs:84`, Core-only), `resolve_with_options(ast, file, options)` (`resolve.rs:89`), `is_tensor_builtin(b)` (`resolve.rs:1961`) |
| `hir.rs` | 680 | High-level IR — the desugared representation every pass after resolution uses exclusively. | n/a (populated by resolve) | `Hir { types, exprs, stmts, items, pats, blocks, root, item_files, ... }` | allocator/accessor methods mirroring `ast.rs` (`hir.rs:597-637`) |
| `typecheck.rs` | 8,791 | Type checking, mutability, tensor-extension dimension/dtype/device unification (PLAN.md M2.2); invokes `flow::check` and `borrowck::check` as sub-passes at the end. | `&Hir`, `Arc<SourceFile>`, `LanguageOptions` | `TypeCheckResult { diagnostics, tables: TypeTables { expr_types, local_types, local_mutability } }` (`typecheck.rs:144-155`) | `check(hir, file)` (`typecheck.rs:157`), `check_with_options(hir, file, options)` (`:162`, wraps `analyze_with_options(...).diagnostics`), `analyze(hir, file)` (`:170`), `analyze_with_options(hir, file, options)` (`:174`, true root entry) |
| `flow.rs` | 403 | Definite-assignment / assignment-mutability data-flow analysis, kept separate from type inference. | `&Hir`, `Arc<SourceFile>` (**unused**, see §4), `&HashMap<ExprId, Ty>` | `Vec<Diagnostic>` | `check(hir, _file, expr_types)` (`flow.rs:21`) |
| `borrowck.rs` | 900 | Borrow checker / ownership pass (PLAN.md M2.4): active-borrow and moved-place tracking, one-`&mut`-XOR-many-`&`. | `&Hir`, `Arc<SourceFile>` (single file, see §4), `&HashMap<ExprId,Ty>`, `&HashMap<LocalId,Ty>` | `Vec<Diagnostic>` | `check(hir, file, expr_types, local_types)` (`borrowck.rs:47`, whole-crate), `check_fn(...)` (`:68`), `check_snippet(...)` (`:89`) |
| `interp.rs` | 3,464 | Gate-3 tree-walking interpreter over typed HIR; deterministic `BTreeMap`-backed `HashMap`/`HashSet` runtime value (see §3). | `&Hir`, `Arc<SourceFile>`, `&TypeTables` | `Result<Execution { output: String }, RuntimeError { message, span }>` | `run(hir, file, tables)` (`interp.rs:397`, runs `main`), `run_item(hir, file, tables, item)` (`:412`, used by `test_runner`) |

### ONNX (Gate 4 M4.5)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `onnx/mod.rs` | 39 | "Bounded ONNX signature import and declaration verification … No graph node or initializer data is interpreted or executed" (`onnx/mod.rs:1-4`). Re-exports submodules. | — | — | — |
| `onnx/importer.rs` | 902 | Decodes an ONNX protobuf model's I/O signature into a `ModelSignature`; generates a deterministic STARK `model` declaration. | ONNX bytes (`&[u8]`) or `&Path` | `Result<ModelSignature, OnnxError>` or generated declaration `String` | `decode_signature(bytes, limits)` (`:222`), `read_signature(path)` (`:598`), `import_file(input, output, force)` (`:624`), `format_declaration(signature, stem, hash)` (`:642`) |
| `onnx/verifier.rs` | 615 | Compares a decoded `ModelSignature` against a STARK `model` declaration and reports drift. | `&ModelSignature` + declaration `&str`, or artifact/declaration `&Path`s | `Result<VerificationReport, OnnxError>` | `verify_declaration_file(artifact, declaration, model_name)` (`:91`), `verify_declaration_source(...)` (`:111`), `escape_json(s)` (`:53`, also reused by `main.rs` JSON diagnostics) |

### Deploy (Gate 5)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `deploy/mod.rs` | 183 | Top-level Gate-5 orchestration: runs the tensor front end over a pipeline, verifies against a live ONNX artifact, lowers to `DeploymentProgram`, emits a host crate. | pipeline `&Path`, model `&Path`, entry fn name, output dir | `Result<DeploymentSummary, DeployError>` (side-effecting) / `Result<DeploymentProgram, DeployError>` (pure) | `deploy(pipeline, model, entry, out, force)` (`:39`), `lower_pipeline(pipeline, model, entry)` (`:89`, documented pure — never writes files/invokes Cargo/downloads) |
| `deploy/ir.rs` | 302 | Deployment IR: a bounded, typed, backend-oriented representation of one checked inference pipeline (straight-line tensor/model ops, symbolic dims). Deliberately not a general STARK→Rust IR. | — | `DeploymentProgram`/`DeploymentFunction` value/op nodes, each carrying an originating `Span` | `as_literal()` (`:44`), `is_symbolic()` (`:53`), `entry_fn()` (`:269`) |
| `deploy/lower.rs` | 1,189 | HIR → Deployment IR lowering (Gate 5 M5.1): lowers only the entry function's acyclic reachable call graph and a bounded op set; rejects everything else with an `E06xx` diagnostic, never falls back to the interpreter. | `&Hir`, checked `TypeTables`, entry item/name | `Result<DeploymentProgram, Vec<Diagnostic>>` (internal `lower_reachable`, `pub(crate)`) | public surface is `deploy::mod::lower_pipeline` |
| `deploy/emit.rs` | 617 | Deterministic host-project emission: `DeploymentProgram` → compilable Rust host crate (pipeline + checked-in runtime + pinned Cargo files + README). Documented pure function of the program — no timestamps/absolute paths/random IDs. | `&DeploymentProgram` | `Vec<EmittedFile>`, then written to disk | `emit(program)` (`:34`), `write_project(out, files, force)` (`:71`) |

### Formatter (WP8.2)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `formatter/mod.rs` | 114 | Specification-driven, idempotent source formatter; walks the parsed AST (never raw text), re-attaches comment trivia by position; formats each file independently of its package. | `&SourceFile`, `LanguageOptions` | `Result<String, Vec<Diagnostic>>` | `format_file(file, options)` (`:29`) |
| `formatter/comments.rs` | 58 | Position-ordered cursor over collected comment trivia. | `&[Comment]` | stateful cursor | `new`, `take_before`, `peek`, `advance`, `is_empty`, `take_rest` (`:17-53`) |
| `formatter/precedence.rs` | 178 | Re-derives from AST tree shape where source text needed parentheses (AST has no grouping-paren nodes). | `&Ast`, `ExprId`/`BinOp`/`AssignOp` | `Level`/`bool`/`&'static str` | `bin_op_level` (`:46`), `bin_op_symbol` (`:62`), `assign_op_symbol` (`:86`), `level` (`:108`), `needs_parens` (`:127`), `head_is_struct_lit` (`:163`) |
| `formatter/printer.rs` | 1,238 | AST-to-text renderer; canonical, idempotent output (4-space indent, 100-col soft wrap). | `&Ast`, `&SourceFile`, `&[Comment]` | `String` | `format(ast, file, comments)` (`:36`) |

### Doc generator (WP8.5)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `doc_gen/mod.rs` | 118 | Orchestrates `stark doc`: extraction → markdown render → syntax highlight → static HTML site + search index. | extracted `Vec<DocItem>`, package name, output dir | generated HTML site on disk; `usize` doc-item count | `generate_from_items(items, package_name, output_dir)` (`:24`), `validate_examples(examples, src)` (`:46`, compiles doc-comment `stark` fences) |
| `doc_gen/extract.rs` | 511 | Associates each public item with its preceding `///` run (doc comments aren't stored in AST/HIR — mirrors formatter's separately-collected trivia). | `&Ast`, `&SourceFile`, `&[Comment]` | `Vec<DocItem>` | `extract(...)` (`:107`), `collect_examples(items)` (`:134`), `extract_stark_fences(doc)` (`:154`) |
| `doc_gen/highlight.rs` | 109 | Syntax-highlights `stark` code blocks using the real lexer (not regex). | `&str` | `String` (HTML) | `highlight(code)` (`:11`), `escape(s)` (`:93`) |
| `doc_gen/html.rs` | 316 | Generates per-item pages, package index, search UI. | `&[DocItem]`, package name, output dir | files on disk | `item_url(item)` (`:12`), `write_site(items, package_name, output_dir)` (`:22`) |
| `doc_gen/markdown.rs` | 144 | Deliberately minimal, non-CommonMark Markdown→HTML renderer for the doc-comment subset actually used. | `&str` | `String` (HTML) | `render(doc)` (`:13`) |
| `doc_gen/search.rs` | 86 | Builds the flat JSON search index for client-side search. | `&[DocItem]`, package name | `Vec<SearchEntry>`, written as JSON | `build_index(items, package_name)` (`:17`), `write_index(entries, output_dir)` (`:51`) |

### LSP (WP8.1)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `lsp/mod.rs` | 23 | Stdio-based LSP server entry, wires `protocol`/`server`/`state` together. | stdin/stdout | `io::Result<()>` | `run()` (`:16`) |
| `lsp/protocol.rs` | 417 | LSP message types + minimal hand-rolled JSON encoder/decoder ("no external dependencies"). | raw message text / `JsonValue` tree | `JsonValue` / `Message` | `parse_json(s)` (`:100`), `parse_message(content)` (`:315`) |
| `lsp/server.rs` | 507 | Message routing and document-sync loop. | generic `R: BufRead`, `W: Write` (abstracted over stdio for testability) | `std::io::Result<()>` | `new()` (`:21`), `run(reader, writer)` (`:28`) |
| `lsp/state.rs` | 172 | Server-side state: open documents + per-URI/version compilation-result cache. | stateful | stateful | `new`, `set_root_uri`, `open_document`, `update_document`, `close_document`, `get_document`, `cache_compilation_result`, `get_cached_result`, `clear` (`:45-110`) |

**Duplication note:** `lsp/protocol.rs` implements its own `JsonValue`/`parse_json`, entirely
separate from `package.rs`'s own `JsonValue`/`parse_json` (`package.rs:5-58`). Two independent
hand-rolled JSON implementations exist in the crate. Not a correctness bug, but a candidate for
a future simplification pass — flagged here rather than acted on, per Charter rule "avoid broad
refactors not required by the active WP."

### Test runner (WP8.3)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `test_runner/mod.rs` | 193 | `stark test` discovery/execution. Core v1 has no attribute syntax, so tests are discovered by naming convention (`fn test_*()`, `fn test_ignored_*()`). | `&Hir`, `&SourceFile`, `&TypeTables` | `Vec<TestCase>` / `TestResult { outcome, output, duration }` | `discover_tests(hir, root_file)` (`:55`), `run_test(hir, root_file, tables, test)` (`:133`), `filter_by_name(tests, name_filter)` (`:170`) |

### Package (CLI-facing)

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `package.rs` | 1,108 | `starkpkg.json` manifest parsing, dependency resolution (path + registry sources), workspace-root/boundary checks, `stark.lock` read/write, hand-rolled JSON parser. | filesystem paths, manifest/lockfile text | `Package`, `PackageGraph`, `Lockfile` | `parse_json(input)` (`:30`), `Package::from_manifest(path)` (`:362`), `find_package_root(start_dir)` (`:509`), `PackageGraph::load_from_root_with_modes(path, locked, offline)` (`:787`), `PackageGraph::load_from_root(path)` (`:1092`), `Lockfile::parse(content)` (`:541`), `Lockfile::serialize(&self)` (`:603`), `calculate_dir_sha256(dir)` (`:690`) |

### Options / extensions

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `options.rs` | 177 | Explicit, deterministic extension gating. `LanguageOptions::default()` enables no extensions, so every entry point is Core-only unless a caller opts in; extensions are threaded as a value, never read from global state. | CLI `--extension` name strings | `LanguageOptions` (`ExtensionSet`) | `ExtensionId::from_name(name)` (`:29`), `LanguageOptions::with_tensor()` (`:81`), `.tensor()` (`:93`), `options_from_extension_flags(names)` (`:122`) |
| `extensions/mod.rs` | 8 | "Optional, non-Core language extensions." Re-exports `tensor`. | — | — | — |
| `extensions/tensor/mod.rs` | 7 | `tensor` v0.1 extension namespace, isolated from Core `Ty` behind a registration/capability boundary. Re-exports `dim`, `types`. | — | — | — |
| `extensions/tensor/dim.rs` | 434 | Pure algebraic core for dimension expressions as multivariate polynomials; equality via canonical normal-form comparison — "the only arithmetic fact the checker knows." | — | `Poly` value type | `zero/constant/var/add/neg/sub/mul/as_constant/is_provably_nonnegative/equal/iter_terms` (`:82-191`) |
| `extensions/tensor/types.rs` | 972 | Tensor element/device types and their unification, kept out of the Core `Ty` enum behind an opaque handle. | — | `Shape`, `UnifyCtx` | `Shape::new/with_spans/rank/is_static` (`:95-109`), `UnifyCtx::new/fresh_dim/rigid_dim/fresh_device/rigid_device/fresh_dtype/rigid_dtype/provenance` (`:291-341`) |

### Support

| Module | Lines | Purpose | Input | Output | Key entry points |
|---|---|---|---|---|---|
| `source.rs` | 136 | `SourceFile` (name + text + precomputed line-start table); `Span` (half-open byte range, **no file identity** — see §4). | raw source text | `SourceFile`; `(usize,usize)` 1-based line/col | `SourceFile::new(name, src)` (`:42`), `line_count()`, `line_col(offset)` (`:69`), `line_text(line)` (`:80`), `Span::new/point/to` (`:16-29`) |
| `diag.rs` | 212 | Dependency-free diagnostic renderer matching `04-Semantic-Analysis.md` format exactly. | `Diagnostic` (severity, code, message, span, label, helps, notes, optional `Arc<SourceFile>`) + fallback `&SourceFile` | `String` | `Diagnostic::error/warning(message, span)` (`:51-75`), `.with_file/with_code/with_label/with_help/with_note` (`:77-100`), `render(default_file)` (`:104`) |
| `ast_dump.rs` | 758 | Stable indented AST tree dump for snapshot tests and `starkc parse --dump`; kept out of `ast.rs` so that stays a pure data model. | `&Ast`, `&SourceFile` | `String` | `dump(ast, file)` (`:18`) |

### Binaries

| Binary | Lines | Purpose |
|---|---|---|
| `starkc` (`src/main.rs`) | 745 | Single-file compiler CLI: `check`/`run`/`parse`/`lex`/`lsp`/`import`/`verify`/`deploy` (`main.rs:9-33` USAGE). Bare `SourceFile`, no package graph, supports `--extension` directly. |
| `stark` (`src/bin/stark.rs`) | 699 | Package-oriented CLI: `check`/`build`/`run`/`test`/`fmt`/`doc` (`bin/stark.rs:14-38` USAGE). Always requires a `starkpkg.json` package root; always Core-only (`LanguageOptions::CORE` hardcoded at `:109,330,606`); uses `PackageGraph`/`parse_package_graph`. |
| `starkide` (`src/bin/starkide.rs`) | 1,941 | Terminal IDE/TUI, not a pipeline stage. Raw-mode terminal control via `Command::new("stty")` (see §3). |

---

## 2. Shared vs. duplicated compiler entry points

No `pub fn compile`/pipeline helper exists anywhere in `starkc/src` (`grep -rn "pub fn
compile\|pub fn pipeline"` → no matches). Each binary's `main` independently re-sequences the
same library calls rather than delegating to one shared driver — this is the situation Charter
rule 18 ("cross-tool compiler behaviour must converge... rather than subtly different
pipelines") is written to prevent, and it has already partially happened.

**`starkc check`** (`main.rs:551-602`, `cmd_check`):
1. `parse_with_options(&file, mode, options)` (`:558`)
2. gate: `diags.iter().all(|d| d.severity != Severity::Error)` — **warnings do not block** (`:562-564`)
3. `resolve::resolve_with_options(&tree, file_arc.clone(), options)` (`:567`)
4. gate: same all-errors check (`:570-572`)
5. `typecheck::check_with_options(&hir, file_arc.clone(), options)` (`:574`) — discards type tables, no interp

**`starkc run`** (`main.rs:702-745`, `cmd_run`):
1. `parse_with_options(&file, ParseMode::Program, options)` (`:707`)
2. gate: `diagnostics.is_empty()` — **any diagnostic of any severity blocks**, not just errors (`:709`)
3. `resolve::resolve_with_options(&tree, file.clone(), options)` (`:711`)
4. gate: `diagnostics.is_empty()` again (`:713`)
5. `typecheck::analyze_with_options(&hir, file.clone(), options)` (`:714`)
6. `interp::run(&hir, file.clone(), &checked.tables)` (`:725`)

**`stark check`/`build`/`run`** (`bin/stark.rs:40-176`, shared fallthrough in `main`):
1. `find_package_root(&current_dir)` (`:93`)
2. `PackageGraph::load_from_root_with_modes(&manifest_path, locked, offline)` (`:101`)
3. `options = LanguageOptions::CORE` — hardcoded; **`stark` has no `--extension` flag at all** (`:109`)
4. `parse_package_graph(&graph, options)` (`:110`) — multi-file, unlike `main.rs`'s single-file `parse_with_options`
5. gate: `diags.iter().all(|d| d.severity != Severity::Error)` (`:129`)
6. `resolve::resolve(&ast, root_file.clone())` (`:130`) — this is `resolve()`, not
   `resolve_with_options()` directly, but `resolve()` is a one-line wrapper
   (`resolve.rs:84-86`: `resolve_with_options(ast, file, LanguageOptions::CORE)`), so
   functionally identical here since `options == CORE` regardless.
7. gate: same check (`:133`)
8. `typecheck::analyze_with_options(&hir, root_file.clone(), options)` (`:134`) — used for
   **both** check/build and run, unlike `starkc check`'s discard-tables wrapper
9. if `cmd == "run"`: `interp::run(&hir, root_file.clone(), &checked.tables)` (`:153`)

**Verdict:**
- The underlying library functions genuinely are shared — `parse_with_options`/
  `parse_package_graph`, `resolve::resolve_with_options`, `typecheck::analyze_with_options`,
  `interp::run` are the same functions called from both binaries. There is no separate
  reimplementation of lex/parse/resolve/typecheck/interp logic.
- But the two binaries have **drifted in policy**, three ways:
  1. **Gating disagreement inside `starkc` itself**: `starkc check` allows warnings through
     (`severity != Error`), while `starkc run` blocks on any diagnostic (`is_empty()`). A program
     that produces a parse/resolve **warning** with zero errors will report `starkc check` → OK
     (exit 0) but `starkc run` on the identical file refuses to execute at all (falls through to
     the diagnostic-rendering/`ExitCode::FAILURE` path, `main.rs:741-744`). This is a real,
     user-visible inconsistency within one binary, independent of the cross-binary question.
  2. **Extension support asymmetry**: `starkc` supports `--extension tensor`; `stark` has no
     extension flag and is Core-only by construction. This is by design (`stark` is the
     package-facing everyday CLI; tensor/ONNX work goes through `starkc`), but it means the two
     binaries are not interchangeable front ends over "the same compilation," they compile
     genuinely different units (bare file vs. package graph) under genuinely different default
     language options.
  3. **Type-table handling**: `starkc check` throws away type tables via `check_with_options`;
     `stark check`/`build` always compute them via `analyze_with_options`. Harmless in practice —
     `check_with_options` is confirmed to be a thin wrapper (`typecheck.rs:162-168`:
     `analyze_with_options(...).diagnostics`) — but it is a second code path nonetheless.
- Confirmed `typecheck::check_with_options` (`typecheck.rs:162-168`) and `typecheck::check`
  (`:157-159`) are both thin wrappers over `analyze_with_options`/`analyze` — no separate
  type-checking logic exists between "check" and "run/analyze" modes.

**Disposition:** finding (1) above is a genuine bug-shaped inconsistency, not just documentation
debt — it changes accept/reject behavior of the same binary across two subcommands on the same
input. Recorded as DEV-SEED (see `COMPILER-STATE.md`, and `starkc/docs/conformance/
KNOWN-DEVIATIONS.md` once WP-C0.4 runs). It should be triaged under WP-C1.x, not fixed here —
WP-C0.1's job is to map and report, not repair.

---

## 3. Global state, filesystem/process access, nondeterministic collections

### Global/static state
No `lazy_static!`, `OnceCell`, or `OnceLock` usage anywhere in `starkc/src`. All `static`
occurrences are either `&'static str` idioms or one genuine immutable lookup table:
`static TENSOR_OPS: &[TensorOpDescriptor] = &[...]` (`typecheck.rs:5936`) — fixed data, not
mutable global state. One `static N: AtomicUsize` exists at
`deploy/template/runtime.rs.in:644`, but this is inside a **template emitted into the generated
deployment host crate** (Gate 5 output) — it is part of the generated program's runtime, not the
compiler's own state.

### Process execution
- `bin/stark.rs:685,689,693` — `open_in_browser()`, invoked only from `stark doc --open` to
  launch the OS browser on generated docs. Not part of the compiler pipeline.
- `bin/starkide.rs:1144,1189,1207` — `Command::new("stty")` to query/restore terminal raw-mode
  state for the TUI editor. Not compiler-internal.
- No `Command::new`/`std::process::Command` in any of `lexer.rs`, `parser.rs`, `resolve.rs`,
  `typecheck.rs`, `interp.rs`, `flow.rs`, `borrowck.rs`, `deploy/*`, `onnx/*`, `doc_gen/*`,
  `formatter/*`, `lsp/*`. (Test files spawn `Command::new(env!("CARGO_BIN_EXE_starkc"))` — test
  harness only, e.g. `tests/gate4_tensor.rs:8`.)

### HashMap/BTreeMap ordering and output determinism
- `interp.rs:67` implements STARK's own `std::collections::HashMap` value as `BTreeMap<Value,
  Option<Value>>` — deliberately deterministic runtime iteration order.
- `deploy/emit.rs:283,341,421` and `doc_gen/html.rs:67-68` use `BTreeMap` for output-order-
  critical data (IR value naming, doc-tree grouping), matching `deploy/emit.rs`'s own
  determinism claim (`:5-9`: "generating twice is byte-identical").
- `package.rs:603-639` (`Lockfile::serialize`) iterates a `HashMap<String, LockfilePackage>` but
  explicitly sorts before emitting (`:608-609`, `:626-627`), so `stark.lock` output is
  deterministic **despite** unordered map iteration — but only because of this downstream sort;
  a future refactor that inlines or skips it would silently reintroduce nondeterminism.
- `package.rs:844,925` — dependency-graph walk order over `HashMap<String, Package>`/
  `HashMap<String, DependencySource>` is unordered; safe for the final lockfile (sorted at
  serialize time) but can affect *which* of several simultaneously-failing dependencies is
  reported first — a minor, real nondeterminism in error ordering, not correctness.
- **Confirmed nondeterminism, not just a risk pattern**: `resolve.rs` glob-import handling
  (`use module::*`) copies items from `ModuleData::items: HashMap<String, Res>` (`resolve.rs:45`)
  via unsorted `.iter()` at two call sites — `resolve.rs:475-479` (absolute-path glob) and
  `:536-540` (relative-path glob). `insert_module_item` (`:571-595`) raises `E0204` ("duplicate
  definition... in the same module scope") when a name collides with a different `Res`. Because
  Rust's default `HashMap` uses a randomized per-process hash seed, **which of two
  glob-colliding names is treated as "first" (silently wins) vs. "second" (flagged E0204) is
  nondeterministic across runs**, and diagnostics are never sorted before rendering (`grep -n
  "sort" diag.rs` → no hits; every render call site iterates `&diags` in push order). This
  affects reproducibility of `starkc check`'s printed diagnostic output for any program using
  `use mod::*` with colliding re-exports.
- No other output-relevant `HashMap` iteration (diagnostics, generated code, doc HTML) was found
  unguarded by a downstream sort.

**Disposition:** the glob-import nondeterminism is a real finding that belongs in the WP-C0.4
known-deviations ledger (governance/reproducibility impact — Charter definition-of-done rule "no
new... nondeterministic iteration... introduced in compiler paths," and "generated output is
deterministic across two runs") and should be triaged for a fix under WP-C1.2 (name
resolution/modules) rather than fixed inline here.

---

## 4. Source-file and module provenance flow

**Core fact: `Span` carries no file identity.** `source.rs:10-13`:
```rust
pub struct Span { pub lo: u32, pub hi: u32 }
```
A bare half-open byte range, meaningless without externally supplied file context. There is
**no `FileId`/`SourceId` type anywhere in the crate** (`grep -rn "FileId\|SourceId"` across
`starkc/src` → zero hits). Provenance is instead reconstructed ad hoc via `Arc<SourceFile>`
threaded by value plus a `HashMap<ItemId, Arc<SourceFile>>` side table (`ast.rs:44`,
`hir.rs:217`) — item-granularity, not expression/statement-granularity. Coverage differs sharply
by stage:

1. **Parse — correct.** `parse_with_options_into` (`parser.rs:330-365`) unconditionally
   back-fills every parser/lexer diagnostic with its originating file before returning:
   `if diag.file.is_none() { diag.file = Some(file_arc.clone()); }` (`:359-363`). Parse-stage
   diagnostics for any file in a package (root or `mod`-loaded submodule) always render against
   the correct file.
2. **Typecheck — correct, via deliberate per-item backfill.** `TypeChecker` swaps `self.file`
   per item during both passes of `check_crate`, using `hir.item_files.get(&item_id)`
   (`typecheck.rs:1916-1919` Pass 1, `:2065-2068` Pass 2), and back-fills any diagnostic still
   missing a file after each item (`:2041-2044`, `:2050-2054`, `:2126-2129`, `:2437-2440`). Two
   sites use explicit `.with_file(file)` for cross-item diagnostics referencing a *different*
   item's file (Copy/Drop conflicts referencing another struct) — `:2455`, `:2498`. All 185
   `Diagnostic::error/warning` sites in the file occur inside the per-item loops this backfill
   covers.
3. **Resolve — genuinely lossy.** `resolve.rs` has 20 `Diagnostic::error/warning` construction
   sites and **zero** `.with_file()` calls anywhere (exhaustive grep). `resolve.rs` populates
   `item_files` itself (`:1728-1729`) and has full access to per-module file identity via
   `ModuleData::file: Arc<SourceFile>` (`:44`), but never attaches it to its own diagnostics.
   `resolve()`/`resolve_with_options()` return `Vec<Diagnostic>` directly (`:84-93`), and neither
   `main.rs` (`:567-568,711-712`) nor `bin/stark.rs` (`:130-131,358-359`) post-hoc backfill
   `diag.file` before rendering. **Consequence:** for a multi-file `stark` package, a
   name-resolution error inside a non-root `mod`-loaded file renders against the caller-supplied
   `default_file` (always the **root** file — `bin/stark.rs:138,172,362,374`), producing a `-->`
   header with the wrong filename and byte offsets mapped against the wrong file's `line_starts`
   table (`diag.rs:104-106`). `SourceFile::line_col` clamps out-of-range offsets
   (`source.rs:70`: `offset.min(self.src.len())`), so this does not panic — it silently produces
   a plausible-looking but incorrect line:col and filename.
4. **Flow/borrowck — provenance structurally dropped.** `flow::check`'s file parameter is
   explicitly unused: `pub fn check(hir: &Hir, _file: Arc<SourceFile>, ...)` (`flow.rs:21-24`,
   note `_file`) — flow diagnostics never carry any file. `borrowck::check`/`check_fn`/
   `check_snippet` (`borrowck.rs:47,68,89`) take a single `file` argument, store it once as
   `self.file`, and use it for every diagnostic regardless of which file the flagged item
   actually came from — no per-item `item_files` lookup exists in `borrowck.rs` (zero
   `item_files`/`with_file` hits). Both are invoked at the end of
   `typecheck::analyze_with_options`, passing the checker's own top-level (root/entry) file
   (`typecheck.rs:221-222`). **For multi-file packages, every flow and borrow-check diagnostic
   for a non-root-file item is misattributed to the root file.**
5. **Interp.** `interp::run`/`run_item` take a single `file: Arc<SourceFile>` used to build
   `RuntimeError { message, span }` on trap — same single-file assumption as borrowck.

**Summary:** parse and typecheck are provenance-correct for multi-file packages; resolve, flow
analysis, and borrow checking are not — they structurally assume single-file compilation for
diagnostic rendering. This is a real, demonstrable regression whenever `stark check`/`build`/
`run`/`test` (the only paths that use multi-file `PackageGraph`s) is exercised on a package with
more than one `.stark` file and an error/warning originates outside the entry file.

**Disposition:** this is one of the most significant findings in this map. It is scoped as a
WP-C1.2 (name resolution/modules/visibility — "source-file-correct diagnostics" is explicitly
one of that WP's matrix rows) and WP-C1.4 (ownership/borrowing) fix, not something to repair
under WP-C0.1. Recorded for the WP-C0.4 known-deviations ledger with normative expectation
("diagnostics must retain the correct file, module, package... provenance," Charter rule 17)
vs. current behavior (misattribution to root file for 3 of 5 pipeline stages in multi-file
builds).

---

## 5. Test files per subsystem

27 top-level files under `starkc/tests/`, plus `tests/common/mod.rs` (shared scaffolding, not a
test file itself) and `tests/fixtures/` (non-Rust reference/comparator assets: Python reference
scripts, Rust comparator cases, manifests — used by the two `#[ignore]`d hermetic-boundary
tests).

| File | Primary module(s) exercised |
|---|---|
| `conformance.rs` | `lexer.rs`, `parser.rs`, `resolve.rs`, `typecheck.rs` — spec-fixture manifest harness |
| `diag_format.rs` | `diag.rs`, `parser.rs` — end-to-end diagnostic rendering |
| `doc_gen.rs` | `doc_gen/extract.rs`, `doc_gen/markdown.rs`, `doc_gen/highlight.rs`, `doc_gen/mod.rs` |
| `formatter.rs` | `formatter/mod.rs`, `formatter/printer.rs`, `formatter/precedence.rs` — golden-file + idempotence + re-parse sweep |
| `gate2_package.rs` | `package.rs`, `parser.rs` (`parse_package_graph`), `resolve.rs`, `typecheck.rs` |
| `gate2_valid.rs` | `resolve.rs`, `typecheck.rs` — single-file semantic checker |
| `gate3_execution.rs` | `interp.rs` (+ resolve/typecheck prerequisites) |
| `gate3_package_resolution.rs` | `package.rs`, `resolve.rs` — multi-file resolution |
| `gate4_onnx.rs` | `onnx/importer.rs`, `onnx/verifier.rs` |
| `gate4_semantics.rs` | `typecheck.rs` (tensor-extension semantics), `resolve.rs` |
| `gate4_tensor.rs` | `extensions/tensor/*`, `resolve.rs`, `typecheck.rs` — via `starkc check --extension tensor` subprocess |
| `gate4a_prelude_traits.rs` | `resolve.rs`, `typecheck.rs`, `interp.rs` — Core prelude/traits |
| `gate4b_string_vec.rs` | `resolve.rs`, `typecheck.rs`, `interp.rs` — String/Vec stdlib |
| `gate4c_collections.rs` | `resolve.rs`, `typecheck.rs`, `interp.rs` — HashMap/HashSet/iterators |
| `gate5_codegen.rs` | `deploy/*` end-to-end with real ONNX Runtime backend — `#[ignore]`d, non-hermetic |
| `gate5_defects.rs` | `deploy/lower.rs`, `deploy/mod.rs` — diagnostic-guard corpus |
| `gate5_emit.rs` | `deploy/emit.rs` |
| `gate5_lower.rs` | `deploy/lower.rs`, `deploy/ir.rs` — drives `deploy::lower_pipeline` |
| `gate5_semantic_gaps.rs` | `typecheck.rs` — Core semantic gap closure |
| `gate7_defects.rs` | `deploy/lower.rs`, `onnx/*` — Gate 7 defect-corpus guard |
| `gate7_deploy.rs` | `deploy/*` — symbolic-shape lowering |
| `gate7_frontend.rs` | `resolve.rs`, `typecheck.rs`, `extensions/tensor/*` |
| `gate7_semantic.rs` | `typecheck.rs` — tensor value-range semantics |
| `phase4e_math_random_io.rs` | `resolve.rs`, `typecheck.rs`, `interp.rs` — Math/Random/IOError stdlib (see §6) |
| `robustness.rs` | `lexer.rs`, `parser.rs` — deterministic pseudo-fuzz (fixed-seed LCG) |
| `snapshots.rs` | `parser.rs`, `ast_dump.rs` — byte-for-byte AST dump comparison |
| `test_framework.rs` | `test_runner/mod.rs` |

No dedicated `tests/gate6_*.rs` suite exists — Gate 6 (old numbering) evidence lives in
`starkc/tests/results/gate6/` (data) and `starkc/tests/fixtures/gate6/` (comparator harness)
instead of as `cargo test` cases.

Full pass/fail counts (2026-07-17 audit run): **383 passed, 0 failed, 2 ignored** — see
`COMPILER-STATE.md` Repository baseline for the per-binary breakdown.

---

## 6. Stubs and deliberately incomplete handlers

Exhaustive grep across `starkc/src` for `todo!()`, `unimplemented!()`, `// stub`/`// STUB` (any
case), `TODO`/`FIXME`/`XXX`/`HACK` (any case) returned **zero matches**. There are no
marker-tagged stubs in the compiler source.

`unreachable!()` — 8 sites, all genuine exhaustiveness guards, not hidden stubs: `package.rs:73`
(JSON fallthrough), `typecheck.rs:6513`, `interp.rs:3232` (`AssignOp::Assign`, handled by an
earlier path — this arm covers only compound-assign operators), `onnx/importer.rs:409,789`
(protobuf field-kind fallthrough), `formatter/printer.rs:642,657,665`.

"Unsupported"/"not supported" diagnostics exist but represent deliberate, spec-referenced,
test-covered scope restrictions, not incompleteness:
- `interp.rs:1519` — `"tensor operations are not supported in the Core interpreter"`: the
  tree-walking interpreter (Gate 3, Core-only) does not execute tensor-extension operations by
  design; tensor programs are compiled via `deploy/lower.rs` instead, never interpreted
  (`deploy/lower.rs:6-11`: "Nothing is silently skipped and nothing falls back to the
  interpreter").
- `deploy/lower.rs` — a large, deliberate family of `E0605`/`E0606`-coded "unsupported in a
  deployment pipeline" diagnostics (over 20 sites), matching the documented restriction that
  Gate 5 lowering only accepts a bounded straight-line op set (`:6-9`) and rejects everything
  else with a diagnostic rather than silently mis-compiling. Covered by `gate5_defects.rs`/
  `gate7_defects.rs`, which assert these exact rejections.
- `onnx/importer.rs:176,391-418,562` — bounded ONNX decoding deliberately rejects unsupported
  protobuf wire types/graph ports/tensor element types, consistent with the stated scope
  (`onnx/mod.rs:1-4`).

**The one genuine, self-documented incompleteness item** is the `resolve.rs` extension-gating
bug, confirmed present in the current tree:

Commit `e58e948` ("implement Phase 4E: Math, Random, and IOError (File struct deferred)")
documents it directly in the commit message: bare `min`/`max` resolve to the tensor extension's
builtin (`Builtin::TensorMin`/`TensorMax`) unconditionally via `resolve_unqualified`, which is
missing the `options.tensor()` gate that `resolve_path_relative` has. Confirmed in the current
tree:

- **Correct, gated pattern** in `resolve_path_relative` (`resolve.rs:682-685`):
  ```rust
  } else if let Some(builtin) = resolve_builtin(name_str) {
      if !is_tensor_builtin(builtin) || self.options.tensor() {
          resolved = Some(Res::Builtin(builtin));
      }
  }
  ```
- **Ungated path** in `resolve_unqualified` (`resolve.rs:1854-1876`): after scope/module-item/
  primitive lookups fail, it calls `resolve_builtin(name)` with **no `options.tensor()` check**
  (`:1866-1867`). `resolve_builtin` maps bare `"min"`/`"max"` unconditionally to
  `Builtin::TensorMin`/`TensorMax` (`:1908-1909`).
- **Trigger surface**: `resolve_unqualified` has exactly two call sites —
  `resolve.rs:659` (resolving `self`) and `:1000` (struct-literal shorthand-field lowering). The
  concrete failure mode: in **Core-only mode** (no `--extension tensor`), a struct-literal
  shorthand field named exactly `min` or `max` with no local/module item of that name in scope —
  e.g. `Point { min }` intended as a plain local-variable shorthand — silently resolves to the
  tensor builtin instead of correctly failing with "undefined variable 'min' (shorthand field)."
  If a local named `min`/`max` genuinely exists, the scope-lookup loop (`:1855-1859`) takes
  precedence, so this only fires in the no-such-local case.
- **Confirmed still unfixed**: `git log --oneline -- starkc/src/resolve.rs` shows `e58e948`
  (Phase 4E) is the most recent commit touching the file; `git blame -L 1854,1876
  starkc/src/resolve.rs` attributes `resolve_unqualified`'s body to commit `86afc2e` ("Implement
  Gate 2 semantic checker and terminal IDE," 2026-07-15 — predates the tensor extension
  entirely), meaning the later tensor `min`/`max` builtin overload (Gate 4) was never retrofitted
  into this pre-existing function's gating logic, and no later commit has touched it.

**Disposition:** carries no `TODO`/stub marker in source (only in the commit message that
introduced the surrounding Phase 4E work), so it would not have been caught by the marker grep
alone — confirming the value of WP-C0.4's independent-verification requirement rather than
trusting seeded/reported issues. Owner: WP-C1.2 (name resolution) fix; WP-C0.4 ledger entry in
the interim.

---

## 7. Archived code that must not be modified

Per `CLAUDE.md` (repo root): archived material is `STARKLANG/docs/archive/`, `web-docs/`,
`STARKLANG/compiler/`, `Practice/`. Note: `web-docs/` is at the **repository root**, not under
`STARKLANG/` — `STARKLANG/web-docs/` does not exist. Confirmed contents (existence + rough
content type only; not deep-read, per their archived/off-limits status):

- **`STARKLANG/docs/archive/`** — pre-pivot design directories (`00-Overview` through
  `09-Examples`, plus a `README.md` documenting the archive-vs-spec conflict table). Markdown
  design docs for the pre-pivot language (actors, hybrid GC, lowercase `i32`/`f32`,
  `Package.stark` TOML manifest, ML pipeline DSL, cloud annotations).
- **`web-docs/`** (repo root) — a static HTML documentation website
  (`README.md`, `index.html`, `template.html`, `styles.css`, plus `architecture/`,
  `concurrency/`, `deployment/`, `overview/`, `spec/`, `stdlib/`, `syntax/`, `types/`
  directories) — pre-pivot marketing/docs site.
- **`STARKLANG/compiler/`** — a Python prototype: `type_checker.py`, `type_system.py`,
  `test_simple.py`, `test_type_checker.py`, `README.md`. Explicitly called out in `CLAUDE.md`:
  "must not be extended for Core v1 work."
- **`Practice/`** — `run_all_practice.sh`, `test_file.txt`, and subdirectories `Algorithms/`
  (6 `.stark` exercises), `Basics/` (`hello.st`, `funcex.st`), `Conforming/` (6 `.stark` files —
  early/legacy conformance-style examples), `Interpreter/` (`starkvm.py`, `parser.py`,
  `interpreter.py` — a second, separate Python prototype interpreter, distinct from
  `STARKLANG/compiler/`'s Python type checker).

**Aside relevant to WP-C0.2 (documentation reconciliation):** `CLAUDE.md` itself is internally
inconsistent — its opening summary states "semantic analysis and execution do not yet [exist]"
while its own "Implementation Status" section ~100 lines later says the opposite ("front end
plus semantic analysis and execution done... Gates 1-3 are closed"). Both cannot be current; the
opening-summary line is stale. Flagged here for WP-C0.2 to fix, not fixed in this document.

---

## Cross-references

- `COMPILER-STATE.md` — DEV-SEED-004 (this bug), DEV-SEED-005 (starkc/stark drift) originate
  from this map; Follow-ups section tracks disposition.
- `starkc/docs/conformance/KNOWN-DEVIATIONS.md` (WP-C0.4, not yet written) — will carry the
  formal ledger entries for: the `check`/`run` gating inconsistency (§2), the glob-import
  nondeterminism (§3), the multi-file provenance loss in resolve/flow/borrowck (§4), and the
  `resolve.rs` tensor-builtin gating bug (§6).
