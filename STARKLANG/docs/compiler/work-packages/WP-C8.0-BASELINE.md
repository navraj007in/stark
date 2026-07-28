# WP-C8.0 Baseline — Semantic Language Services

Date: 2026-07-28

## Scope

This baseline audits the current STARK language-server and VS Code integration before C8 repair work. The audited implementation is:

- LSP CLI entry point: `starkc lsp` via `starkc/src/main.rs`
- Server implementation: `starkc/src/lsp/server.rs`
- Server state: `starkc/src/lsp/state.rs`
- Protocol support: `starkc/src/lsp/protocol.rs`
- Shared compiler analysis/query APIs: `starkc/src/analysis.rs`, `starkc/src/analysis/query.rs`
- VS Code client: `editors/vscode/src/lspClient.ts`
- VS Code subprocess diagnostics path: `editors/vscode/src/extension.ts`, `editors/vscode/src/compiler.ts`, `editors/vscode/src/diagnostics.ts`

## Current Pipeline

The LSP server is a stdio JSON-RPC server reached through the main compiler binary. It stores full text for open documents, tracks document versions, and recompiles the current open document on open/change/save.

Current LSP analysis uses:

```text
open document text
    -> SourceFile(uri, text)
    -> analyze_project(ProjectInput::program(source), options)
    -> ProjectAnalysis
    -> diagnostics / semantic query handlers
```

This is compiler-backed for diagnostics, formatting, and semantic queries. C8 now loads the nearest `starkpkg.json` for real file URIs inside a package, analyzes the package graph through the shared compiler pipeline, and overlays currently open editor buffers onto package entry/module reads. Unsaved/non-file buffers still use the single-buffer program path.

The compiler already has a shared semantic query foundation in `ProjectAnalysis` and `analysis::query`, including stable analysis-scoped handles for syntax/HIR nodes, symbol lookup at byte offsets, definitions, references, type rendering, signatures, document symbols, workspace symbols, and diagnostic batches. C8.0A wired LSP position/range conversion through a centralized UTF-16 mapper. C8.2/C8.3 now use this query surface for hover, definition, and references.

## Capability Truthfulness

| Capability        | Advertised | Implemented semantically | Placeholder | Tested | Editor validated |
| ----------------- | ---------: | -----------------------: | ----------: | -----: | ---------------: |
| Diagnostics       | Yes, via `publishDiagnostics` notifications | Partial: compiler-backed for current open buffer or discovered package snapshot with open-buffer overlays | No | Unit test | Not recorded |
| Hover             | Yes | Yes, for symbols and typed HIR nodes in the current analysis snapshot | No | Unit test | Not recorded |
| Definition        | Yes | Yes, via resolved symbol identity in the current analysis snapshot | No | Unit test | Not recorded |
| References        | Yes | Yes, via resolved symbol identity in the current analysis snapshot | No | Unit test | Not recorded |
| Completion        | Yes | Partial: indexed semantic item completion from the current analysis snapshot | No | Unit test | Not recorded |
| Signature help    | Yes | Partial: resolved HIR call expressions with active parameter from argument spans | No | Unit test | Not recorded |
| Rename            | Yes | Partial: safe top-level symbols in the current analysis snapshot | No | Unit test | Not recorded |
| Document symbols  | Yes | Yes: compiler symbol index for current analysis snapshot | No | Unit test | Not recorded |
| Workspace symbols | Yes | Partial: cached open-document analysis snapshots | No | Unit test | Not recorded |
| Semantic tokens   | Yes | Partial: full-document tokens from compiler symbol classification | No | Unit test | Not recorded |
| Inlay hints       | No | No | No | No | Not recorded |
| Formatting        | Yes | Partial: uses parser/formatter over live buffer, not semantic analysis | No | Unit test | Not recorded |

Unsupported inlay hints are not advertised. Initial audit found hover, definition, and references advertised despite lacking semantic implementation; C8.0 repaired the advertised capability set, and C8.2/C8.3 re-advertised those features after backing them with compiler semantic queries. C8.4 advertises completion for the bounded semantic-symbol subset now implemented and signature help for resolved HIR call expressions. C8.5 advertises rename for safe top-level symbols plus document and workspace symbols backed by the compiler symbol index. C8.6 advertises full-document semantic tokens without delta support.

## Inventory Findings

- Language-server binary and entry point: no separate binary; `starkc lsp` starts `starkc::lsp::run`.
- JSON-RPC transport: manual stdio framing with `Content-Length`; minimal in-repo JSON parser/serializer.
- Initialization capabilities: initially advertised full sync, hover, definition, references, and document formatting. C8.0 truthfulness repair temporarily removed unsupported hover/definition/references. C8.2/C8.3 re-enabled them after semantic implementation landed.
- Document open/change/save/close: full document sync only; open/change/save compile the document; close clears diagnostics and removes state.
- Project-root and package discovery: real file URIs inside a discovered `starkpkg.json` package use `PackageGraph::load_from_root_with_modes` and `ProjectInput::package_with_overlays`; unsaved/non-file buffers fall back to single-buffer analysis.
- Source-text storage and version tracking: open documents store URI, version, and full text; compilation cache is versioned by URI.
- Compiler invocation path: in-process `analyze_project(ProjectInput::program(...))` for standalone LSP buffers and `analyze_project(ProjectInput::package_with_overlays(...))` for package LSP buffers; VS Code separately spawns `starkc check --stdin --filename` only for the explicit manual check path.
- Diagnostic publication: compiler-backed `DiagnosticBatch` converted to LSP diagnostics, including severity, code, range, related information, source, version data, package/provenance data, extension data, notes/help/rule metadata in `data`.
- Hover: uses `ProjectAnalysis::symbol_at`, `signature`, `type_of`, and `hir_at`; returns no hover for unresolved positions.
- Definition: uses `ProjectAnalysis::symbol_at` and `definition`; returns LSP `Location` for resolved symbols.
- References: uses `ProjectAnalysis::symbol_at` and `references`; respects `includeDeclaration`.
- Completion: LSP handler returns deterministic completion items from `ProjectAnalysis::completion_candidates`, derived from indexed semantic symbols and shared signature rendering.
- Signature help: LSP handler uses `ProjectAnalysis::signature_help_at`; currently limited to resolved HIR function calls and parameter labels from compiler-owned source spans.
- Rename: LSP handler uses `ProjectAnalysis::rename_edits` over resolved symbol identity; currently limited to indexed top-level symbols, rejects invalid names and same-snapshot top-level collisions.
- Document symbols: LSP handler returns `SymbolInformation` from `ProjectAnalysis::document_symbols`.
- Workspace symbols: LSP handler searches cached open-document analyses via `ProjectAnalysis::workspace_symbols`.
- Semantic tokens: full-document LSP handler encodes compiler semantic token classifications; range and delta requests are not advertised.
- Inlay hints: no LSP handler and no advertised capability.
- Formatting integration: server formats the live buffer with `formatter::format_file`; VS Code invokes standard format provider on save when configured.
- VS Code activation: starts LSP in trusted workspaces when extension is enabled; passes configured extensions in initialization options; restarts on relevant configuration changes.
- VS Code subprocess use: manual `STARK: Check Current File` still runs `starkc check --message-format json --stdin --filename`. Automatic open/save/type subprocess diagnostics are gated by `stark.diagnostics.subprocess`, default `false`, so LSP diagnostics are the default authority.
- Placeholder or hard-coded responses: no advertised placeholder responses remain for hover, definition, or references; no hard-coded developer paths found in the LSP client path.
- Tests: unit tests capture truthful advertised capabilities, semantic hover, resolved definition/references, UTF-16 position conversion, and versioned compiler-backed diagnostic publication.

## Baseline Evidence

Permanent baseline tests added in `starkc/src/lsp/server.rs`:

- `initialize_advertises_only_semantically_supported_handlers`
- `hover_uses_compiler_symbol_signature`
- `definition_and_references_use_resolved_symbol_identity`
- `completion_returns_indexed_semantic_symbols`
- `signature_help_uses_resolved_callee_and_argument_spans`
- `rename_uses_resolved_symbol_identity_for_safe_top_level_symbols`
- `document_and_workspace_symbols_use_compiler_symbol_index`
- `semantic_tokens_are_encoded_from_compiler_classification`
- `json_rpc_transport_handles_initialize_unknown_method_and_shutdown`
- `json_rpc_transport_publishes_diagnostics_for_open_document`
- `package_lsp_analysis_uses_open_document_overlays`
- `json_rpc_transport_publishes_package_overlay_diagnostics`
- `json_rpc_transport_change_clears_diagnostics_and_close_publishes_clear`
- `json_rpc_transport_malformed_message_does_not_stop_later_request`
- `json_rpc_transport_exit_notification_stops_without_response`
- `json_rpc_transport_serves_semantic_feature_requests`
- `json_rpc_transport_handles_cancellation_formatting_workspace_symbol_and_rename_failure`
- `lsp::position::tests::*`
- Existing `publishes_shared_diagnostics_with_document_version`

Verification:

```text
cargo check -p starkc
cargo test -p starkc --lib lsp:: -- --nocapture
npm run compile --prefix editors/vscode
```

Latest results: `cargo test --lib lsp:: -- --nocapture` (34 tests), `cargo check`, and `npm run compile --prefix editors/vscode` pass.

## Risks To Repair In Later C8 Steps

- Hover coverage is currently bounded to symbols and typed HIR nodes already indexed by `ProjectAnalysis`; richer symbol display and documentation remain C8.2 follow-up work.
- Definition and references are semantic within one current analysis snapshot. Package snapshots are now used for discovered package files, but broader workspace indexing and external package source policy remain follow-up work.
- Completion is currently a safe semantic-symbol subset. Local-scope completion, receiver member completion, enum variants, package paths, and type/value-position filtering remain C8.4 follow-up work.
- Signature help is a bounded first pass. Method receivers, generic substitutions, overload-like ambiguity, and richer callable forms remain follow-up work.
- Document symbols are flat `SymbolInformation` rather than hierarchical `DocumentSymbol`; hierarchy remains a C8.5 follow-up.
- Workspace symbols currently search cached open-document analysis snapshots. Package-backed snapshots improve each opened package file, but there is still no independent full-workspace index.
- Rename remains a conservative subset. Local rename, external/package rename, richer collision checks, prepareRename, and workspace-wide consistency checks remain follow-up work.
- Semantic tokens are currently limited to classifications available from indexed symbols and resolved occurrences. Additional lexical categories, modifiers beyond `declaration`, and inlay hints remain follow-up work.
- LSP analysis now has a coherent package snapshot path for real files under `starkpkg.json`, including module files, dependency files, source identities, document versions for open files, diagnostics, and query indexes. Remaining gaps are workspace-wide lifecycle triggers and richer dependency/config invalidation.
- The VS Code legacy subprocess checker remains available for an explicit manual check and an opt-in automatic path. Real-editor validation must confirm the default LSP-only diagnostic authority in VS Code.
- Stale result suppression is limited by synchronous per-message analysis and versioned cache lookup. C8.1 must preserve this property when analysis becomes asynchronous or project-wide.
- Formatting is advertised but is parser/formatter-backed rather than semantic. It should remain advertised only with that documented scope unless C8 chooses to make formatting part of the semantic pipeline.

## C8.0 Closure

C8.0 is closed for baseline classification: every currently advertised capability has been classified, placeholder behavior is captured by tests, and the immediate semantic-query/project-snapshot requirements for C8.0A are identified.
