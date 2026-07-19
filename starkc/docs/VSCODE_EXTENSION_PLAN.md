# STARK VS Code extension implementation plan

This plan delivers a useful, maintainable VS Code experience without building
an LSP prematurely. The first release is a thin client over `starkc`; a later
language server can replace the compiler-process adapter without rewriting the
extension UI.

## 1. Goals and boundaries

### Initial release goals

- recognize `.stark` files;
- provide accurate syntax highlighting, comments, brackets, indentation, and
  auto-closing;
- check the active document with `starkc` and publish native VS Code
  diagnostics;
- support unsaved buffers without temporary-file path bugs;
- expose explicit Check, Run, and Open in STARK IDE commands;
- support Core-only and opt-in `tensor` extension checking;
- locate local, PATH-installed, and user-configured compiler binaries;
- work on macOS, Windows, and Linux;
- package as a normal VSIX with no platform-specific compiler bundled.

### Explicitly deferred

- completion, hover, go-to-definition, references, rename, and semantic tokens;
- a language server or compiler daemon;
- debugger/debug adapter support;
- tensor-shape visualization;
- formatter and code actions beyond compiler-provided help text;
- automatic compiler download or update;
- Marketplace publication and signing;
- bundling `starkc` binaries inside the VSIX.

These deferred features require stable semantic query APIs or an LSP and should
not be simulated by regex-based source analysis.

## 2. Proposed repository layout

Create the extension as a separately buildable package:

```text
editors/vscode/
  .vscodeignore
  CHANGELOG.md
  LICENSE
  README.md
  package.json
  package-lock.json
  tsconfig.json
  esbuild.mjs
  language-configuration.json
  syntaxes/stark.tmLanguage.json
  src/
    extension.ts
    compiler.ts
    configuration.ts
    diagnostics.ts
    commands.ts
    protocol.ts
  test/
    unit/
    fixtures/
    suite/
```

Keep Node dependencies and build artifacts inside this directory. Do not turn
the Rust crate into a Node workspace. Commit `package-lock.json`; ignore
`node_modules`, `dist`, test downloads, coverage, and generated VSIX files.

Use TypeScript in strict mode and bundle the extension host entry with esbuild.
Use the current supported VS Code engine baseline selected at implementation
time and document it in the extension README.

## 3. Phase V0 — compiler integration contract

Do this before wiring diagnostics into VS Code. Human-readable diagnostic text
is presentation output and must not become an extension protocol.

### 3.1 Machine-readable checking

Add a stable compiler invocation equivalent to:

```text
starkc check --message-format json \
  [--extension tensor] \
  --stdin --filename /absolute/workspace/file.stark
```

Source arrives on standard input. `--filename` provides the logical path for
diagnostics and relative module resolution. The compiler must not overwrite the
file. Human-readable output remains the default for terminal users.

Write protocol JSON to stdout and operational/logging text to stderr. Emit one
complete JSON document per invocation rather than interleaving JSON and prose.
Exit status remains meaningful:

- `0`: check completed without errors;
- `1`: source diagnostics were produced;
- `2`: invalid CLI usage;
- another nonzero value: compiler/I/O/internal failure.

Warnings do not make the process fail unless existing compiler policy says
otherwise.

### 3.2 Versioned diagnostic schema

Use a versioned envelope. The exact Rust type names may vary, but the wire
contract must contain equivalent data:

```json
{
  "schemaVersion": 1,
  "tool": "starkc",
  "toolVersion": "0.1.0",
  "diagnostics": [
    {
      "severity": "error",
      "code": "E0212",
      "message": "tensor shape mismatch",
      "file": "/workspace/main.stark",
      "range": { "startByte": 12, "endByte": 18 },
      "labels": [
        {
          "message": "expected [B, 3, 224, 224]",
          "file": "/workspace/model.stark",
          "range": { "startByte": 30, "endByte": 52 }
        }
      ],
      "notes": [],
      "help": "permute the input axes"
    }
  ]
}
```

Requirements:

- ranges are zero-based UTF-8 byte offsets with an exclusive end, matching the
  compiler's existing spans;
- every primary diagnostic includes a file and range;
- related labels may refer to another file;
- absent codes/help use `null` or are omitted consistently;
- file paths are absolute and normalized, while the extension handles Windows
  drive-letter comparison case-insensitively;
- malformed compiler JSON is treated as an integration failure, not as an
  empty successful check.

The extension converts byte offsets to VS Code's UTF-16 line/character ranges
using the exact document text associated with that check. Test ASCII,
multibyte UTF-8, combining characters, and astral Unicode.

### 3.3 Compatibility and compiler tests

Add Rust CLI tests proving:

- human output remains unchanged without `--message-format json`;
- valid, invalid, warning-only, malformed CLI, and missing-file cases use the
  documented status codes;
- JSON escaping and Unicode spans are valid;
- stdin source is checked instead of stale disk content;
- relative module lookup uses `--filename` correctly;
- Core and `--extension tensor` modes remain isolated;
- parser, resolver, type, flow, and borrow diagnostics serialize through the
  same schema.

Suggested commit: `Add versioned JSON diagnostics for editor clients`

## 4. Phase V1 — language package

This phase works without invoking a compiler and remains available in
untrusted workspaces.

### 4.1 Extension manifest

Create a package manifest with:

- language id `stark`;
- display name `STARK`;
- file extension `.stark`;
- aliases `STARK` and `stark`;
- grammar and language-configuration contributions;
- command and configuration contributions defined below;
- activation on the STARK language and STARK commands, not `*`.

Do not claim `.st` by default because it is commonly associated with
Smalltalk. Users may add their own `files.associations` entry.

### 4.2 TextMate grammar

Build grammar scopes from the live lexical and syntax specifications, not from
examples alone. Cover:

- line and block comments, including nested-block behavior only if VS Code's
  grammar can represent the compiler behavior accurately;
- strings, character literals, escapes, numeric literals, and suffixes;
- Core keywords, primitive types, booleans, and reserved words;
- function/type/trait/enum/struct/const/module declarations;
- generics, attributes if supported, operators, punctuation, and paths;
- tensor dtypes/devices and contextual `model`, `input`, and `output` words;
- invalid/unterminated literals where reliable.

Contextual tensor words must not be colored as unconditional Core keywords in
ordinary identifier positions. Prefer conservative highlighting over incorrect
semantic claims.

### 4.3 Language configuration

Configure:

- `//` and `/* ... */` comments;
- `{}`, `[]`, and `()` brackets;
- auto-closing and surrounding pairs for brackets and quotes;
- indentation around brace-delimited blocks;
- folding markers if the language defines region comments;
- word pattern suitable for identifiers and paths.

### 4.4 Grammar validation

Add tokenization fixtures for Core programs, tensor/model declarations,
comments, strings, malformed literals, and contextual keywords. Snapshot
meaningful scope sequences rather than screenshots. Open representative files
in VS Code for a manual visual pass before release.

Suggested commit: `Add STARK syntax support for VS Code`

## 5. Phase V2 — compiler discovery and diagnostics

### 5.1 Settings

Contribute these settings:

```text
stark.compiler.path             string, default "starkc"
stark.compiler.extensions       string[], default []
stark.check.onSave              boolean, default true
stark.check.onType              boolean, default false
stark.check.debounceMs          integer, default 500, minimum 100
stark.check.clearOnClose        boolean, default true
stark.trace.server              "off" | "messages" | "verbose"
```

Initially accept only compiler-known extension names. Do not enable `tensor`
by default because Core-only behavior is the language default. Workspace-folder
settings override user settings in the normal VS Code manner.

### 5.2 Compiler resolution

Resolve `stark.compiler.path` as follows:

1. an absolute configured executable;
2. a relative configured path resolved against the owning workspace folder;
3. a command name resolved by the child process through `PATH` without a
   shell;
4. in Extension Development mode only, optionally detect the repository's
   `starkc/target/debug/starkc` to ease contributor testing.

Never build a shell command string. Spawn the executable with an argument
array, `shell: false`, a bounded output buffer, and the document/workspace
directory as the working directory. Quote handling is the operating system's
job. Show an actionable “compiler not found” message with links to settings and
the extension output channel, but avoid repeated notification spam.

### 5.3 Workspace Trust

Syntax support always works. Do not start configured executables or run code in
an untrusted workspace. Explain that checking/running requires trust and resume
normally if trust is later granted.

### 5.4 Check lifecycle

Maintain one diagnostic collection owned by the extension. For each document:

- check on save by default;
- allow opt-in debounced checking on type;
- cancel or supersede an older process when a newer document version is
  scheduled;
- discard results whose document version no longer matches;
- send the current in-memory document through stdin;
- clear diagnostics after a successful clean result;
- optionally clear them when the document closes;
- terminate child processes during extension shutdown.

Use independent state per URI and support multi-root workspaces. Cap compiler
runtime and output size with documented constants; on timeout/overflow, cancel
the process and report an integration error in the STARK output channel.

Convert primary diagnostics for the active URI into VS Code diagnostics.
Attach cross-file labels using `DiagnosticRelatedInformation`. Combine notes
and compiler help in a readable diagnostic message without losing the primary
message/code. Map compiler error/warning/note severities explicitly.

### 5.5 Commands

Contribute:

- `STARK: Check Current File`;
- `STARK: Restart Compiler Integration`;
- `STARK: Show Output`.

Check works for unsaved buffers. Restart cancels children, clears cached
compiler state, rereads settings, and rechecks visible STARK documents.

### 5.6 Tests

Unit-test protocol validation, byte-to-UTF-16 conversion, executable argument
construction, path normalization, extension flags, cancellation, stale result
discarding, and severity mapping. Use a deterministic fake compiler executable
for extension integration tests; do not require a globally installed `starkc`.

Add VS Code host tests for activation, settings, diagnostics, multi-root file
ownership, save/on-type behavior, document close, malformed JSON, compiler-not-
found, and untrusted workspace behavior.

Suggested commit: `Integrate starkc diagnostics with VS Code`

## 6. Phase V3 — Run and terminal IDE commands

These commands are user-initiated and require a trusted workspace.

### 6.1 Run current file

Add `STARK: Run Current File`. Require the document to be saved, then reuse or
create a named VS Code terminal and invoke `starkc run <absolute-file>`.
Respect the configured compiler path. Do not attempt to capture or parse
interactive program output as diagnostics.

Before running, optionally perform a check and stop when it reports errors.
Make this policy a documented setting only if users demonstrably need to
disable it; avoid unnecessary configuration initially.

### 6.2 Open in terminal IDE

Add `STARK: Open Current File in STARK IDE`. Locate `starkide` beside an
absolute `starkc` binary first, otherwise resolve it through PATH. Launch it in
a dedicated integrated terminal with the saved absolute file path.

Do not assume Unix path syntax or shell escaping. VS Code terminal APIs do
accept command text, so use VS Code's platform-aware shell quoting helper or a
small exhaustively tested quoting function for Windows PowerShell/cmd and
POSIX shells. If reliable shell selection cannot be determined, present the
exact command for the user instead of executing a potentially incorrect one.

### 6.3 Editor UI

Add editor-title or context-menu entries sparingly. Commands must remain
available through the Command Palette; avoid persistent toolbar clutter. A
status item may show checking/error state only while a STARK document is
active.

Suggested commit: `Add STARK run and terminal IDE commands`

## 7. Phase V4 — packaging and release readiness

### 7.1 Documentation

The extension README must document:

- supported features and explicit non-features;
- installation from VSIX;
- installing/building `starkc` and `starkide` separately;
- compiler path and extension settings;
- workspace trust behavior;
- Core versus tensor checking;
- troubleshooting and how to open the output channel;
- supported operating systems and VS Code versions.

Add a changelog, repository/license metadata, an extension icon only if a
project-approved asset exists, and a minimal sample workspace.

### 7.2 Build scripts

Provide reproducible scripts:

```text
npm run compile
npm run lint
npm run test:unit
npm run test:integration
npm run package
```

`npm run package` produces a VSIX with source maps and development-only files
excluded. Inspect VSIX contents and enforce a reasonable size ceiling.

### 7.3 Cross-platform validation

Before release, validate on current supported macOS, Windows, and Linux runners:

- clean `npm ci` and build;
- unit tests;
- extension host smoke tests;
- compiler discovery with spaces and Unicode in paths;
- `.stark` activation;
- JSON diagnostics and Unicode ranges;
- Check and Run commands;
- VSIX creation and installation smoke test.

The VSIX remains platform-neutral because it does not bundle Rust binaries.
Test it against separately built `starkc` release packages.

### 7.4 Publication boundary

Creating the VSIX is part of implementation. Publishing to the VS Code
Marketplace, creating publisher credentials, signing in, or uploading a
release requires explicit user authorization and is not automatic.

Suggested commit: `Package the STARK VS Code extension`

## 8. Validation matrix

Run the Rust validation when compiler protocol code changes:

```bash
cd starkc
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo doc --no-deps
```

Run extension validation from `editors/vscode/`:

```bash
npm ci
npm run compile
npm run lint
npm run test:unit
npm run test:integration
npm run package
```

From the repository root, finish with `git diff --check` and inspect the full
changed-file list. Do not modify or weaken existing compiler, IDE, or Gate
tests to accommodate the extension.

## 9. Release acceptance criteria

The first VS Code release is ready when:

- `.stark` files activate and highlight correctly without workspace trust;
- human `starkc` diagnostics remain backward compatible;
- the JSON diagnostic protocol is versioned and fully tested;
- saved and unsaved documents receive accurate native diagnostics;
- Unicode byte spans map correctly to VS Code UTF-16 ranges;
- stale checks cannot overwrite newer results;
- compiler lookup works with spaces and Unicode on all three operating systems;
- Core mode remains default and tensor mode is an explicit setting;
- Check, Run, and Open in STARK IDE commands behave predictably;
- no command executes in an untrusted workspace;
- all Rust and extension validation passes;
- the generated VSIX contains no bundled compiler, secrets, test downloads,
  or unrelated repository files;
- installation and setup are documented for a new user.

## 10. Recommended execution order

1. Finish and stabilize Gate 4 compiler semantics.
2. Implement V0 JSON/stdin compiler protocol.
3. Implement V1 syntax package in parallel only after the language tokens are
   stable.
4. Implement V2 compiler diagnostics.
5. Add V3 explicit run/terminal IDE commands.
6. Complete V4 packaging and cross-platform validation.
7. Gather usage feedback before designing an LSP milestone.

## 11. Starter prompt for an implementation agent

> Implement the STARK VS Code extension exactly according to
> `starkc/docs/VSCODE_EXTENSION_PLAN.md`. Begin with Phase V0 only. Read
> `AGENTS.md`, inspect the live dirty worktree, and preserve unrelated changes.
> Do not parse human-readable diagnostics and do not begin the TypeScript
> extension until the versioned JSON/stdin compiler contract and its Rust tests
> pass. At the end of V0, run focused and full Rust validation, inspect the
> complete diff, and report changed files, test results, deviations, and a
> suggested commit message. Do not commit or begin V1 without explicit
> instruction.
