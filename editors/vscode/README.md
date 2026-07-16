# STARK Language — VS Code extension

Syntax highlighting, language configuration, and snippets for the
[STARK programming language](../../STARKLANG/docs/spec/). The grammar follows
the normative Core v1 lexical grammar
(`STARKLANG/docs/spec/01-Lexical-Grammar.md`) and additionally recognizes the
optional `tensor` v0.1 extension
(`STARKLANG/docs/extensions/Tensor-Model-Types.md`).

## Features

- **Syntax highlighting** (`.stark` files) covering:
  - keywords — control flow, declarations, storage/visibility modifiers, and
    the word operators `in`/`as`;
  - primitive types (`Int8`…`UInt64`, `Float32`, `Float64`, `Bool`, `String`,
    `Char`, `Unit`, `str`) and the tensor extension types (`Float16`,
    `BFloat16`, `Tensor`, `TensorDyn`, `TensorAny`, `Dim`, `DType`, `Cpu`,
    `Cuda`);
  - integer literals (decimal/hex/octal/binary with `i8`…`u64` suffixes and
    `_` digit separators), float literals (`f32`/`f64`), strings with escape
    sequences, raw strings (`r"..."`), and character literals;
  - line (`//`), block (`/* */`, **nested**), and doc (`/** */`) comments;
  - `model` declarations and the `input`/`output`/`device` contextual keywords;
  - reserved words (`async`, `await`, `unsafe`, `dyn`, …) flagged as reserved so
    they stand out — they are not valid in Core v1 programs.
- **Language configuration**: comment toggling, bracket matching, auto-closing
  and surrounding pairs, and doc-comment continuation.
- **Snippets**: `main`, `fn`, `struct`, `enum`, `trait`, `impl`, `match`,
  `let`, `model`, and more.

This extension is presentation-only: it ships no language server, so there is
no live type-checking, completion, or diagnostics. Use `starkc check` for
semantic analysis. (An LSP server is out of scope per the roadmap.)

## Scope

This is **Phase V1** (the language package) of
`starkc/docs/VSCODE_EXTENSION_PLAN.md` — syntax, language configuration, and
snippets, all of which work without a compiler and in untrusted workspaces.
The plan's other phases are intentionally **not** included here:

- **V0** — a versioned `starkc check --message-format json --stdin` protocol;
- **V2** — compiler discovery and native VS Code diagnostics;
- **V3** — Check / Run / Open in STARK IDE commands;
- **V4** — packaging, cross-platform validation, and release.

Those phases require the compiler-integration work and are a separate effort.

## Grammar tests

The TextMate grammar is snapshot-tested against the real VS Code TextMate
engine (`vscode-textmate` + `vscode-oniguruma`), per the plan's §4.4:

```bash
npm install
npm test            # verify scope snapshots
npm run test:update # regenerate after an intentional grammar change
```

Fixtures live in `test/fixtures/`; golden scope snapshots in `test/snapshots/`.

## Install locally

From this directory:

```bash
# One-time: install the VS Code extension packager
npm install -g @vscode/vsce

# Package into a .vsix
vsce package

# Install the packaged extension
code --install-extension stark-language-0.1.0.vsix
```

Or, for quick iteration without packaging, symlink/copy this folder into your
VS Code extensions directory:

```bash
ln -s "$(pwd)" ~/.vscode/extensions/stark-language-0.1.0
```

Then reload VS Code and open any `.stark` file.

## Development

The grammar lives in `syntaxes/stark.tmLanguage.json`. To iterate:

1. Open this folder in VS Code.
2. Press `F5` to launch an Extension Development Host.
3. Open a `.stark` file; edit the grammar and reload the host to see changes.

Use **Developer: Inspect Editor Tokens and Scopes** (command palette) to see
the TextMate scope assigned to any token.

## License

MIT — see the repository root.
