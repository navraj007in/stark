# WP8.4 — VS Code Extension Implementation

**Status:** Complete (interactive VS Code testing not possible in this
environment — see Testing below)
**Module:** `editors/vscode/`

## Overview

`editors/vscode/` already had a working extension (syntax highlighting,
snippets, and subprocess-based `starkc check` diagnostics/run/open-in-IDE
commands) — none of it used the LSP server built in WP8.1. This WP adds
the LSP client integration plus the remaining plan deliverables, without
disturbing the existing subprocess-based diagnostics flow (both run side
by side).

## What was added

- **LSP client** (`src/lspClient.ts`, `vscode-languageclient`): starts
  `starkc lsp` automatically in trusted workspaces, alongside the existing
  subprocess checks. Gives hover/go-to-definition/find-references/format
  "for free" — VS Code wires its standard editor commands (`F12`,
  `Shift+F12`, hover) to whatever capabilities the server declares, no
  custom command code needed for those three.
- **`stark.formatCurrentFile`** command (`Ctrl+Shift+F` / `Cmd+Shift+F` on
  `.stark` files) plus `stark.formatOnSave`, both driving VS Code's
  built-in `editor.action.formatDocument` / `executeFormatDocumentProvider`
  against the LSP's `textDocument/formatting`.
- **`stark.toggleTensorMode`** command, flipping a new
  `stark.tensorExtensionEnabled` setting and restarting the compiler/LSP
  integration so the new extension set takes effect immediately.
- **Status bar item** showing LSP state (starting/running/error); click to
  restart.
- **`stark.testOnSave`**: runs `stark test` in the current file's package
  on save, output to a new "STARK Test" channel.
- **New settings**: `stark.package.path` (the `stark` binary, distinct
  from `stark.compiler.path`'s `starkc`), `stark.extensionEnabled`,
  `stark.lspLogLevel` (wired to the LSP client's protocol trace level),
  `stark.formatOnSave`, `stark.tensorExtensionEnabled`, `stark.testOnSave`.
- **`stark.showLanguageServerOutput`** / **`stark.restartLanguageServer`**
  commands for troubleshooting (the existing `stark.restartCompiler` now
  also restarts the LSP client, not just diagnostics).

## A real bug found and fixed along the way (separate commit `4eda834`)

Wiring the tensor toggle through to the LSP surfaced that `starkc lsp`
hardcoded `LanguageOptions::default()` (Core-only) in both
`compile_document` and `handle_formatting` — a client requesting the
`tensor` extension would still get "requires extension `tensor`" errors
on every tensor file, regardless of what it asked for. Fixed in
`starkc/src/lsp/`: `ServerState` now carries the session's
`LanguageOptions`, set once from `initialize`'s `initializationOptions`
(`{"extensions": ["tensor"]}`) and reused for every parse. Verified via a
raw JSON-RPC script (bypassing VS Code) that a tensor-only construct
(`model M<B: Dim> { ... }`) formats successfully once `extensions:
["tensor"]` is sent at initialize, and fails without it.

## Deliberately not built

- **`stark.generateDocs`** — the plan lists it, but WP8.5 (Documentation
  Generator) doesn't exist yet; `starkc doc` isn't a real command. Adding
  a menu entry that fails with "unknown command" would be worse than not
  having it — added when WP8.5 ships.
- **Custom keybindings for hover/go-to-definition/find-references** — the
  plan lists `F12`/`Shift+F12`/`Ctrl+K Ctrl+I`, but these are already VS
  Code's own default keybindings for those exact editor commands; once
  the LSP client registers hover/definition/references providers, they
  work without any extension-side keybinding registration. Only
  **Format Document** got an explicit binding, since VS Code's own default
  for that (`Shift+Alt+F`) differs from what the plan specifies.

## Testing

- `npx tsc --noEmit` — clean.
- `npm run lint` (ESLint) — clean.
- `npm run compile` (esbuild) — builds `dist/extension.js` successfully.
- `npm test` (existing TextMate grammar snapshot tests) — 4/4 pass,
  unaffected by this change.
- **LSP protocol-level verification** (raw JSON-RPC over stdio, bypassing
  VS Code entirely — the same technique used in WP8.1/WP8.2's manual
  tests): confirmed `initialize` with `initializationOptions.extensions:
  ["tensor"]` is accepted and correctly enables tensor parsing for
  `textDocument/formatting`, and that a session without it correctly
  rejects tensor-only syntax.
- **Not verified**: actual behavior inside a running VS Code Extension
  Development Host (status bar rendering, command palette entries,
  format-on-save firing on a real save, hover popups) — no `code` CLI is
  available in this environment to launch one. Every layer below the VS
  Code UI itself (TypeScript correctness, bundling, the LSP protocol
  exchange the client depends on) is verified; the UI wiring on top of it
  is not interactively confirmed. Flagging this explicitly rather than
  claiming full verification.

## Next steps

Real interactive testing (install the packaged `.vsix`, or run the
Extension Development Host) is the natural next step whenever a VS
Code-capable environment is available. WP8.5 (Documentation Generator)
will want to come back and add `stark.generateDocs` once `starkc doc`
exists.
