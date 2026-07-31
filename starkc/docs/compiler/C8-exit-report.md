# Gate C8 Exit Report — Semantic Language Services

Status: `C8-CANDIDATE-COMPLETE`
Date: 2026-07-31
Qualified commit: `6556a0d`

## Claim

Gate C8 provides compiler-backed semantic language services for the documented STARK project and package configurations. Advertised diagnostics, hover, navigation, references, symbols, completion, signature and rename capabilities are derived from shared compiler analysis and have been validated through protocol tests. Known limitations and unsupported configurations remain explicitly listed.

This is candidate-complete rather than closed because real VS Code Extension Development Host validation has not been recorded.

## Toolchain

- `rustc 1.93.0 (254b59607 2026-01-19)`
- VS Code extension package target: `^1.91.0`
- VS Code extension package: `stark-language` `0.2.0`

## Capability Matrix

| Capability | Advertised | Compiler-backed | Verified |
| --- | ---: | ---: | ---: |
| Diagnostics | Yes | Yes | Yes |
| Formatting | Yes | Parser/formatter-backed | Yes |
| Hover | Yes | Yes | Yes |
| Definition | Yes | Yes | Yes |
| References | Yes | Yes | Yes |
| Completion | Yes | Bounded semantic-symbol subset | Yes |
| Signature help | Yes | Bounded resolved-call subset | Yes |
| Rename | Yes | Safe top-level subset | Yes |
| Document symbols | Yes | Yes | Yes |
| Workspace symbols | Yes | Cached analysis snapshots | Yes |
| Semantic tokens | Yes | Full-document semantic classifications | Yes |
| Inlay hints | No | No | N/A |

No advertised semantic capability is a placeholder.

## Architecture

The language server analyzes editor documents through the shared compiler pipeline:

```text
document/package input
  -> ProjectInput
  -> analyze_project
  -> ProjectAnalysis
  -> analysis::query
  -> LSP response
```

For real file URIs under a discovered `starkpkg.json`, C8 uses package graph analysis with open-buffer overlays. Unsaved or non-file buffers use the single-buffer program path. Position and range conversion is centralized through the UTF-16 mapper in `starkc/src/lsp/position.rs`.

## Package And Extension Behavior

Package-backed snapshots preserve source identities, document versions, openable file URIs, diagnostics, and semantic query indexes for package files and module files. Compiler extensions are passed through initialization options using the same extension names as the CLI `--extension` path; unknown names do not prevent server startup.

## Verification

Evidence registry: `starkc/docs/compiler/evidence/c8/README.md`.

Passed:

- `cargo test --lib lsp:: -- --nocapture` from `starkc` — 36 tests passed.
- `cargo check` from `starkc` — passed.
- `npm run compile` from `editors/vscode` — passed.

## Known Limitations

- No interactive VS Code Extension Development Host validation record is present.
- Completion is a semantic-symbol subset, not a complete local/member/package-path completion engine.
- Signature help is limited to resolved HIR call expressions.
- Rename is intentionally conservative and limited to safe top-level symbols in the current analysis snapshot.
- Workspace symbols search cached open-document analysis snapshots rather than a separate persistent workspace index.
- Semantic tokens are full-document only; range and delta semantic-token requests are not advertised.
- Inlay hints are not implemented and are not advertised.

## Carried Items

- Record real VS Code Extension Development Host validation for diagnostics, hover, definition, references, completion, signature help, rename, symbols, semantic tokens, formatting, and default LSP-only diagnostic authority.
- Decide whether to expand completion, rename, workspace indexing, semantic-token modifiers, or inlay hints in a later gate.

## Reconciliation

C8 changes are scoped to language-server, semantic-query, editor integration, and C8 documentation paths. Existing unrelated working-tree edits in `stark-base64/**` were not touched.

## Exit Conclusion

`C8-CANDIDATE-COMPLETE`

