# Changelog

## 0.2.0

- Language server integration (`starkc lsp`, WP8.1): hover, go-to-definition,
  find-all-references, and document formatting, wired through
  `vscode-languageclient`. Starts automatically in trusted workspaces
  alongside the existing subprocess-based diagnostics.
- New command: **STARK: Format Document** (`Ctrl+Shift+F` / `Cmd+Shift+F` on
  `.stark` files), backed by the language server's formatter (`stark fmt`'s
  rules). `stark.formatOnSave` runs it automatically before saving.
- New command: **STARK: Toggle Tensor Extension**, flipping
  `stark.tensorExtensionEnabled` and restarting the compiler/language-server
  integration with the new extension set.
- New commands: **STARK: Show LSP Output**, **STARK: Restart Language
  Server**.
- Status bar item (visible on `.stark` files) showing language server state;
  click to restart.
- `stark.testOnSave` runs `stark test` in the current file's package on
  save, reporting results to a new **STARK Test** output channel.
- New settings: `stark.package.path`, `stark.extensionEnabled`,
  `stark.lspLogLevel`, `stark.formatOnSave`, `stark.tensorExtensionEnabled`,
  `stark.testOnSave`.
- Fix: `stark.testOnSave` no longer forwards `--extension` to `stark test`
  (the CLI doesn't accept it and would fail every run when a language
  extension like `tensor` was configured); a note is logged to the
  **STARK Test** channel instead of silently dropping the flags.
- Fix: `stark.testOnSave` now runs from the saved file's own directory
  rather than the workspace folder, so `stark test` finds the correct
  package in workspaces containing more than one STARK package.
- Fix: a missing/invalid `stark.package.path` binary now reports a clear
  error (via a new child-process `error` handler) instead of an unhandled
  exception.
- Fix: `engines.vscode` raised to `^1.91.0` to match the actual minimum
  required by the `vscode-languageclient` dependency (was `^1.75.0`,
  understating the real requirement).

## 0.1.0

- Initial release.
- Syntax highlighting for STARK Core v1 (per `01-Lexical-Grammar.md`) and the
  optional `tensor` v0.1 extension (types, `model` declarations, contextual
  keywords).
- Language configuration: comments, brackets, auto-closing/surrounding pairs,
  doc-comment continuation.
- Snippets for common items and a tensor `model` declaration.
