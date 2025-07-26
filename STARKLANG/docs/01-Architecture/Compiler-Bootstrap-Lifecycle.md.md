# STARKLANG Compiler Bootstrap Lifecycle

This document defines the end-to-end lifecycle of how STARKLANG source code is transformed from `.stark` source files into executable bytecode packages consumed by the STARKVM. It also outlines the CLI interface, compiler phases, build artifacts, and tooling integration required for seamless development and deployment.

---

## 📦 Compiler Pipeline Stages

```text
.stark Source Files  →  Lexer → Parser → AST → Semantic Analyzer → IR → Optimizer → Bytecode Generator → VM Bundle
```

### 1️⃣ **Lexical Analysis (Lexer)**
- Tokenizes source code into a stream of lexemes.
- Detects and categorizes keywords, identifiers, literals, symbols, etc.

### 2️⃣ **Parsing (Parser)**
- Converts tokens into Abstract Syntax Tree (AST).
- Validates grammar rules and syntax structure.

### 3️⃣ **Semantic Analysis**
- Type checking
- Scope resolution
- Function/Module binding
- Identifier resolution

### 4️⃣ **Intermediate Representation (IR) Generation**
- AST is converted into IR (typed, portable instruction tree).
- Easier to optimize and analyze than AST.

### 5️⃣ **Optimization Phase (Optional)**
- Constant folding, dead code elimination, inlining.
- IR-level optimizations for performance.

### 6️⃣ **Bytecode Generation**
- IR is converted to STARK Bytecode (.sb)
- Encoded instruction set format defined by STARKVM spec.

### 7️⃣ **Package Bundling**
- Bundles:
  - Bytecode files (`.sb`)
  - Metadata (`starkpkg.json`)
  - Dependency resolution
  - Deployment descriptors (if @serverless annotated)

---

## ⚙️ CLI Lifecycle & Commands

### `stark init`
- Initializes a new project with folder structure & manifest (`starkpkg.json`).

### `stark build`
- Triggers full compiler pipeline
- Outputs:
  - `/build/*.sb` (bytecode)
  - `/build/manifest.json`
  - `/build/deploy.yaml` (if applicable)

### `stark run`
- Executes compiled bytecode via STARKVM.
- Usage: `stark run main.sb`

### `stark serve`
- Runs bytecode in **server mode** (microservice/serverless entrypoint).

### `stark deploy`
- Builds and deploys the containerized application to cloud target.
- Auto-generates container image, push scripts, YAML/bicep.

### `stark fmt`
- Formats `.stark` code using standard style rules.

### `stark test`
- Executes test files with assertion tracking.

### `stark docgen`
- Generates markdown/HTML API docs from annotated source code.

---

## 📁 Build Artifacts Structure

```
/my-project
├── src/
│   └── main.stark
├── starkpkg.json
├── build/
│   ├── main.sb
│   ├── manifest.json
│   └── deploy.yaml
```

---

## 🛠 Developer Tooling Hooks
- Compiler exposes hooks at each stage:
  - `onToken(token)`
  - `onASTNode(node)`
  - `onIRBlock(ir)`
  - `onBytecodeInstruction(instr)`

These allow integration with:
- IDE plugins (LSP)
- Visual debuggers
- Static analyzers

---

## 🔄 Future Enhancements
- Incremental Compilation
- Watch Mode (`stark dev`)
- Hot-reload for local VM execution
- Bytecode-to-IR reverse tool (`stark disasm`)

---

## ✅ Summary
STARKLANG’s compiler lifecycle is designed for clarity, extensibility, and cloud-first packaging. With modular stages and CLI tooling, it ensures developers can go from **source code to cloud-executable** in a single command pipeline.

