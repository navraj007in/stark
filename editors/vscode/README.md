# STARK Language — VS Code Extension

Official Visual Studio Code extension for the **STARK** programming language, providing TextMate syntax highlighting, code snippets, and live semantic and typechecking diagnostics powered by the `starkc` compiler.

---

## Features

### 1. Syntax Highlighting (`.stark` files)
- **Core v1 Grammar**: Keywords, variables, control flow (`if`, `else`, `match`, `while`, `loop`, `return`), and operator keywords (`in`, `as`).
- **Tensor and Range Extensions**: Highlights tensor primitive/generic types (`Tensor`, `TensorDyn`, `TensorAny`, `Dim`, `DType`, `Cpu`, `Cuda`) and range-type qualifiers/constraints (`ByteRange`, `UnitRange`, `Normalized`, `device =`, `range =`).
- **Literals and Comments**: Handles decimal, hex, octal, and binary integer literals, float literals, escape sequences, raw strings, and nested block comments.

### 2. Live Compiler Diagnostics
- **Workspace Diagnostics**: Automatically reports errors, warnings, and type-mismatch details directly inside the editor.
- **Unsaved Buffer Checking (stdin)**: Feeds document edits directly to the compiler on the fly, showing diagnostic results without needing to write temporary files to disk.
- **Related Information**: Maps multi-file diagnostics, compiler help messages, and notes to editor tooltips.
- **Save and Change Triggers**: Supports running checks automatically on file save, or dynamically while typing with configurable debouncing.
- **Race Prevention**: Monotonically tracks and automatically cancels active/superseded compiler processes when check-on-type triggers again.

### 3. Integrated Commands
- **Check Current File**: Runs manual semantic checks on the active buffer and populates the diagnostics view.
- **Run Current File**: Automatically saves changes and runs the program in a dedicated interactive terminal, forwarding any active compiler extension flags (e.g. `--extension tensor`).
- **Open in STARK IDE**: Opens the current file in the native STARK interactive compiler IDE.
- **Restart Compiler Integration**: Clears active diagnostics and triggers a fresh check of the active document.

### 4. Configuration Options
Adjust configurations in your User or Workspace `settings.json`:
- `stark.compiler.path`: Command or absolute path to your `starkc` compiler binary (defaults to `"starkc"`).
- `stark.compiler.extensions`: List of active compiler extensions to pass (e.g. `["tensor"]`).
- `stark.check.onSave`: Triggers checking when a `.stark` file is saved (default: `true`).
- `stark.check.onType`: Triggers checking in the background while typing (default: `false`).
- `stark.check.onTypeDebounceMs`: Debounce delay before background type-checking starts (default: `500`).

---

## Workspace Trust & Security
This extension enforces VS Code's **Workspace Trust** boundary:
- In untrusted workspaces, all process execution (running, checking, or opening in STARK IDE) is disabled to prevent arbitrary code execution on untrusted files.
- Safe syntax highlighting, language configuration, and code snippets remain active.

---

## Local Development & Testing

### 1. Grammar & Snapshot Tests
The TextMate grammar is snapshot-tested using the actual VS Code TextMate and Oniguruma tokenizers:
```bash
npm install
npm test            # Run tokenization tests
npm run test:update # Update snapshots after modifying the grammar
```

### 2. Packaging the Extension
To package the extension into a portable `.vsix` archive:
```bash
# Compiles the TS code and minifies the bundle into dist/extension.js
npm run package

# Creates the portable vsix package
npx @vscode/vsce package --no-dependencies
```

### 3. Installing Locally
Installs the VSIX package directly into your VS Code editor:
```bash
code --install-extension stark-language-0.1.0.vsix
```

---

## License
MIT — see the repository root.
