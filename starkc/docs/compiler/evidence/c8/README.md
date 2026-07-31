# Gate C8 Evidence Registry

Date: 2026-07-31
Qualified commit: `6556a0d`
Status: `C8-CANDIDATE-COMPLETE`

## Verification Commands

```text
cargo test --lib lsp:: -- --nocapture
```

Result: pass, 36 tests.

Coverage included:

- truthful `initialize` capability advertisement;
- compiler-backed diagnostics with document versions;
- package snapshot diagnostics with open-document overlays;
- UTF-16 position mapping;
- hover from compiler symbol signatures and typed HIR nodes;
- definition and references through resolved symbol identity;
- semantic completion candidates;
- signature help for resolved HIR calls;
- rename edits for safe top-level symbols;
- document and workspace symbols from the compiler symbol index;
- full-document semantic tokens from compiler classifications;
- raw JSON-RPC transport behavior for initialize, unknown methods, malformed messages, shutdown, exit, cancellation, formatting, workspace symbols, and rename failures.

```text
cargo check
```

Result: pass.

```text
npm run compile
```

Directory: `editors/vscode`

Result: pass.

## Toolchain

- `rustc 1.93.0 (254b59607 2026-01-19)`
- VS Code extension package target: `engines.vscode = ^1.91.0`
- VS Code extension package: `stark-language` version `0.2.0`

## Capability Evidence Matrix

| Capability | Evidence |
| --- | --- |
| Diagnostics | `json_rpc_transport_publishes_diagnostics_for_open_document`, `publishes_shared_diagnostics_with_document_version`, `json_rpc_transport_publishes_package_overlay_diagnostics` |
| Hover | `hover_uses_compiler_symbol_signature` |
| Definition | `definition_and_references_use_resolved_symbol_identity` |
| References | `definition_and_references_use_resolved_symbol_identity` |
| Completion | `completion_returns_indexed_semantic_symbols` |
| Signature help | `signature_help_uses_resolved_callee_and_argument_spans` |
| Rename | `rename_uses_resolved_symbol_identity_for_safe_top_level_symbols` |
| Document symbols | `document_and_workspace_symbols_use_compiler_symbol_index` |
| Workspace symbols | `document_and_workspace_symbols_use_compiler_symbol_index` |
| Semantic tokens | `semantic_tokens_are_encoded_from_compiler_classification` |
| Formatting | `json_rpc_transport_handles_cancellation_formatting_workspace_symbol_and_rename_failure` |
| Package overlays | `package_lsp_analysis_uses_open_document_overlays`, `package_build_returns_openable_uris_and_whole_file_results` |

## Editor Validation

The VS Code extension was build-validated with `npm run compile`.

No interactive Extension Development Host run was recorded in this environment. That is the only reason this gate is marked `C8-CANDIDATE-COMPLETE` instead of `C8-CLOSED`.

