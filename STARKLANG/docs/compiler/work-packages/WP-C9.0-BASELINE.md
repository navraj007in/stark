# WP-C9.0 Baseline — Extension Isolation Governance

Date: 2026-07-31

## Status

C9 is open. Part A is authorised. Part B is blocked pending a second independent artifact
implementation with working evidence. No artifact-provider generalisation is authorised in C9.0.

## Extension Selection Inventory

| Entry point | Default | Configuration source | Unknown | Duplicate | Session lifetime | Shared state | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `starkc parse` | Core-only | `--extension <name>` | Reject usage error | Reject usage error | One process invocation | None identified | `options` unit tests |
| `starkc check` | Core-only | `--extension <name>` | Reject usage error | Reject usage error | One process invocation | None identified | `c91_extension_isolation` |
| `starkc run` | Core-only | `--extension <name>` | Reject usage error | Reject usage error | One process invocation | None identified | covered by shared option parser |
| `starkc import` | ONNX artifact tool | ONNX path only | No extension surface | No extension surface | One process invocation | None identified | ONNX importer tests |
| `starkc verify` | ONNX artifact verifier | ONNX path + declaration | No extension surface | No extension surface | One process invocation | None identified | Gate 5/7 verifier tests |
| `starkc deploy` | Tensor/ONNX deployment path | ONNX model path + STARK pipeline | No extension surface | No extension surface | One process invocation | build cache only | Gate 5/7 deploy tests |
| `starkc lsp` | Core-only | `initializationOptions.extensions` | Reject initialize params | Reject initialize params | Server session; reset on shutdown | per-server state | `lsp::server` tests |
| `stark check` | Core-only | package/command path; no extension flag currently inventoried | Fixed Core | Fixed Core | One process invocation | None identified | existing package/build tests |
| `stark build` | Core-only for source analysis | package manifest/provider config | Fixed Core for language extensions | Fixed Core | One process invocation | build cache | C7 build tests |
| Formatter | Core unless caller passes options | `LanguageOptions` argument | Via option parser at caller | Via option parser at caller | Single format call | None identified | formatter tests |
| Doc generator | Core unless caller passes options | compiler options path | Via option parser at caller | Via option parser at caller | Single generation call | None identified | doc-gen tests |
| Test harness helpers | Core unless helper opts into tensor | helper-owned `LanguageOptions` | Via option parser if used | Via option parser if used | Test call | None identified | Gate 4/C9 tests |

## Active Architecture

Extension selection is represented by `starkc::options::LanguageOptions`, containing an immutable
`ExtensionSet`. `LanguageOptions::CORE` and `Default` are Core-only. `LanguageOptions::with_tensor`
is the only extension-enabled convenience constructor.

The extension option is threaded into parser, resolver, type checker, formatter, package analysis,
and LSP analysis. No environment-variable or process-global extension switch was found. The known
global/process state near C9 is unrelated to extension selection: analysis ids, build-cache files,
and package/provider registries that are selected from explicit inputs.

## Baseline Commands

Required full baseline commands are recorded in `starkc/docs/compiler/evidence/c9/README.md`.
Targeted implementation commands may be used during C9.1 and C9.2, with the full command set rerun
at Part A close.

## Exit

C9.0 baseline inventory is recorded. The next implementation packet is C9.1 extension-isolation
conformance. C9 Part B remains blocked.
