# Gate C9 Evidence

Date: 2026-07-31

## Status

Part A is in progress. Part B is blocked pending second-artifact evidence.

## Implemented Evidence

- LSP invalid extension initialization now rejects unknown and duplicate names with `-32602`,
  matching CLI/internal option policy.
- LSP shutdown clears language options back to Core-only.
- `c91_extension_isolation` proves Core-only default rejection, explicit tensor acceptance,
  sequential session isolation, parallel analysis isolation, and CLI invalid configuration refusal.

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
