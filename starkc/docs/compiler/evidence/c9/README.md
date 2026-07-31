# Gate C9 Evidence

Date: 2026-07-31

## Status

Part A is complete for C9.0-C9.2. Part B is blocked pending second-artifact evidence.

## Implemented Evidence

- LSP invalid extension initialization now rejects unknown and duplicate names with `-32602`,
  matching CLI/internal option policy.
- LSP shutdown clears language options back to Core-only.
- `c91_extension_isolation` proves Core-only default rejection, explicit tensor acceptance,
  sequential session isolation, parallel analysis isolation, CLI invalid configuration refusal,
  fixed-Core package/format/doc behavior, and package-module isolation.
- LSP package-mode tests prove separate Core-only and tensor-enabled server sessions do not share
  extension state.

## Commands

Targeted commands for the current packet:

```text
cargo test -p starkc --test c91_extension_isolation
cargo test -p starkc --lib lsp:: -- --nocapture
```

Full C9.0 baseline / Part A close commands:

```text
cargo check --workspace
cargo test --workspace
cargo test -p starkc --lib lsp:: -- --nocapture
npm run compile --prefix editors/vscode
```

Latest command results are recorded in the task closeout rather than this static evidence index.
