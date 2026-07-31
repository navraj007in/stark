# WP-C9.1 Extension Isolation Conformance

Date: 2026-07-31

## Status

Closed for Part A. Permanent regression coverage now covers compiler CLI, package CLI, formatter,
doc generation, package analysis, single-file analysis, LSP single-file/package sessions, invalid
configuration, sequential analysis, and parallel analysis.

## Required Behaviour Matrix

| Entry point | Core default | Tensor disabled rejects | Tensor enabled accepts | Builtins isolated | Unknown extension | Duplicate extension | Session isolation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `starkc check` | Yes | Yes | Yes | Existing Gate 4 tests | Reject | Reject | N/A |
| `starkc run` | Yes | Shared parser/options | Shared parser/options | Shared resolver | Reject | Reject | N/A |
| `stark check` | Yes | Yes | No language-extension surface | Fixed Core | Reject flag by usage | Reject flag by usage | N/A |
| `stark build` | Yes | Fixed Core | No language-extension surface | Fixed Core | Reject flag by usage | Reject flag by usage | N/A |
| Formatter | Yes by default | Yes | Yes through explicit API options | Shared parser/resolver | Caller policy | Caller policy | Per call |
| Doc generator | Yes | Yes | No language-extension surface in `stark doc` | Shared parser | Fixed Core | Fixed Core | Per call |
| Verifier | ONNX artifact verifier, no extension flag | N/A | N/A | N/A | Fixed surface | Fixed surface | Per call |
| Deploy path | Tensor/ONNX-specific command | Pipeline must check under tensor rules | Yes | Shared analysis | Fixed surface | Fixed surface | Per call |
| LSP single-file | Yes | Yes | Yes | Shared analysis | Reject initialize | Reject initialize | Yes |
| LSP package mode | Yes | Yes | Yes | Shared analysis | Reject initialize | Reject initialize | Yes |
| Test helpers | Yes by default | Yes | Yes | Shared analysis | Reject when using parser | Reject when using parser | Yes |

## Policy

Unknown extension names are rejected. Duplicate extension declarations are rejected. The policy is
implemented by `options_from_extension_flags` and now applies consistently to CLI and LSP
initialization options.

## Permanent Evidence

- `starkc/tests/c91_extension_isolation.rs`
- `starkc/src/lsp/server.rs::tests::initialize_rejects_unknown_and_duplicate_extensions_like_cli`
- `starkc/src/lsp/server.rs::tests::shutdown_clears_lsp_extension_session_state`
- `starkc/src/lsp/server.rs::tests::package_lsp_sessions_keep_tensor_extension_isolated`
- Existing Gate 4 parser/resolver/typechecker tests for tensor-disabled diagnostics and tensor
  enablement.

## Current Implementation Notes

LSP `ServerState::clear` resets `LanguageOptions::CORE`, so shutdown/reinitialize in one process
does not retain tensor enablement. `ProjectAnalysis` stores the exact options used for that
analysis, and C9.1 tests sequential and parallel analyses with different extension sets.

## Closeout

All Part A isolation surfaces identified in the baseline have either direct tests or are classified
as having no extension configuration surface. Core-only remains the default. Unknown and duplicate
extension declarations reject deterministically. No mutable global extension registry was found.
Part B remains blocked.
