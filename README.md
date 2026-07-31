# STARK Language

> A statically typed language for safe, verifiable AI inference pipelines—built on an ownership-safe general-purpose core.

**Status: pre-alpha · active development**

STARK is an experimental programming language designed to catch errors in AI deployment pipelines before inference begins.

Its general-purpose Core provides static typing, ownership, borrowing, structured error handling and predictable execution semantics. The optional tensor extension adds compile-time checks for tensor shapes, element types, devices and imported model signatures.

STARK currently includes a working Rust compiler, semantic checker, borrow checker, interpreter, ONNX signature importer, multi-file module system, package management with semantic versioning, native compilation, and compiler-backed language services.

### Where the compiler actually is

Two gate sequences run in this repository and they answer different questions. The **Gate 1–7** table further down covers the original tensor/deployment track, whose Gate 6 and Gate 7 decisions (*REVISE*, *RETAIN AS RESEARCH*) still stand. The **Gate C0–C10** track is a separate, evidence-first re-closure of Core v1 conformance and execution, and it is where current work happens:

| Track gate | Position |
| --- | --- |
| C0–C6 | Closed. C6 closed with a *qualified* native subset — 59 of 87 audited standard-library methods have a verified invocation, 28 explicitly refused or excluded |
| **C7** | **Closed.** Native compilation with debug and release profiles, build cache, MIR optimiser, and an HTTP/JSON REST workload qualified on Linux, macOS and Windows |
| **C8** | **Candidate-complete.** Compiler-backed language services (LSP + VS Code extension). Held open by interactive editor validation: hover, go-to-definition and find-references are confirmed in a real session; the other seven capabilities are protocol-tested only |
| **C9** | Open. Extension isolation, and a conditionally authorised artifact-provider generalisation whose second half is blocked pending evidence from a second artifact format |

Programs are compared across four execution configurations — the HIR reference interpreter, the MIR interpreter, native debug and native release — with each case's expected result pinned against the specification rather than against another engine's output. `COMPILER-STATE.md` is the authoritative position; this table is a summary and can lag it.

## Why STARK?

AI inference pipelines often connect components that are individually valid but incompatible when assembled:

* a model expects NCHW input but receives NHWC;
* preprocessing produces the wrong element type;
* a tensor is placed on an incompatible device;
* an ONNX artifact changes after its declaration was generated;
* a dynamic dimension is treated as statically known;
* postprocessing assumes an incorrect output shape.

These errors are commonly discovered at runtime.

STARK aims to make them visible earlier:

```text
STARK source
    → parsing and name resolution
    → type, ownership and borrow checking
    → tensor shape/dtype/device verification
    → ONNX artifact signature verification
    → generated native inference host
```

The initial validation target is deliberately narrow: a reliable, typed computer-vision inference pipeline using an existing backend rather than a new tensor runtime.

## Example

```stark
model Resnet50<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output probabilities: Tensor<Float32, [N, 1000]>;
}

fn preprocess(
    image: Tensor<UInt8, [1, 224, 224, 3]>
) -> Tensor<Float32, [1, 3, 224, 224]> {
    image
        .permute::<[0, 3, 1, 2]>()
        .cast::<Float32>()
}

fn infer(
    model: Resnet50,
    raw: TensorAny
) -> Result<Tensor<Int64, [1]>, String> {
    let image = raw.refine::<UInt8, [1, 224, 224, 3]>()?;
    let input = preprocess(image);
    let output = model.predict(&input);

    Ok(output.softmax::<1>().argmax::<1>())
}
```

An incompatible shape, dtype or device is rejected before model execution.

## Current capabilities

### STARK Core

The Core language currently supports:

* primitive numeric, Boolean, character and string types;
* functions, recursion, constants and local type inference;
* structs, enums, tuples, arrays and pattern matching;
* `Option`, `Result` and the `?` operator;
* generics, traits, associated types and inherent implementations;
* ownership, moves, partial moves and reinitialisation;
* shared and mutable lexical borrows;
* receiver auto-borrow and auto-dereference;
* checked integer operations and bounds-checked indexing;
* loops, ranges, `if`, `match`, `while` and `for`;
* deterministic destruction and `Drop`;
* `Vec`, `String`, `Box`, ranges and basic file I/O.

Core programs pass through:

```text
Source
  → Tokens
  → AST
  → resolved HIR
  → type and ownership analysis
  → typed-HIR interpreter
```

### Tensor extension

The optional `tensor` extension provides:

* tensor element types;
* static and symbolic dimensions;
* shape unification;
* dtype and device checking;
* broadcasting validation;
* permutation, reshape and reduction checks;
* model declarations;
* typed model inputs and outputs;
* explicit dynamic-to-static refinement;
* isolation from Core-only compilation.

Tensor features must be enabled explicitly:

```bash
cargo run -- check --extension tensor program.stark
```

### ONNX integration

STARK can inspect ONNX model metadata and generate a typed model declaration:

```bash
cargo run -- import model.onnx --out model.stark
```

It can also verify that an artifact still matches its declaration:

```bash
cargo run -- verify model.onnx --declaration model.stark
```

Verification detects differences such as:

* input or output count;
* port names and ordering;
* element types;
* tensor ranks;
* static dimensions;
* dynamic dimensions;
* dimension identity;
* artifact checksum drift.

The importer reads bounded model metadata only. It does not execute graph nodes.

### Native deployment prototype

Gate 5 lowers a supported STARK inference program into a bounded deployment IR and generates a self-contained Rust host project.

```bash
cargo run -- deploy \
  --extension tensor \
  pipeline.stark \
  --model model.onnx \
  --out generated-host
```

The generated project is designed to include:

* translated preprocessing and postprocessing operations;
* ONNX Runtime integration;
* exact model port bindings;
* fail-fast artifact hash validation;
* pinned Rust dependencies;
* a committed lockfile;
* deterministic generated source;
* no generated `unsafe` code.

Gate 5's measured demonstration is complete (see [`starkc/docs/gate5-exit.md`](starkc/docs/gate5-exit.md)); the follow-on Gate 6/7 decision checkpoints subsequently recorded REVISE and RETAIN AS RESEARCH LANGUAGE respectively for further tensor-track productisation — see the Delivery gates table below and [`starkc/docs/gate7-decision.md`](starkc/docs/gate7-decision.md).

## Quick start

### Requirements

* Rust stable
* Cargo
* Git

The compiler currently declares Rust 1.85 as its minimum supported version.

Clone the repository:

```bash
git clone https://github.com/navraj007in/stark.git
cd stark/starkc
```

Run the test suite:

```bash
cargo test
```

### Installing the compiler

To use STARK outside this checkout, install the binaries **and the crates the native backend
links against**. The compiler finds those by a fixed layout relative to its own executable, so
the directory structure below is not a suggestion — it is what `stark build` looks for.

```bash
# From starkc/
cargo build --release --bins

PREFIX="$HOME/.local"                       # must be on PATH
install -m 755 target/release/stark    "$PREFIX/bin/"
install -m 755 target/release/starkc   "$PREFIX/bin/"
install -m 755 target/release/starkide "$PREFIX/bin/"

# The runtime every generated binary links, plus its own path dependency.
rsync -a --exclude target stark-runtime/       "$PREFIX/lib/stark/stark-runtime/"
rsync -a --exclude target stark-provider-abi/  "$PREFIX/lib/stark/stark-provider-abi/"
```

Three binaries, not one: `stark` is the package driver, `starkc` the single-file CLI and language
server, `starkide` the terminal IDE. The VS Code extension defaults to `starkc`.

**Provider-backed capabilities** — clock, filesystem, environment and TCP — need their provider
crates too, in a root that mirrors the repository's shape:

```bash
PROV="$PREFIX/lib/stark/providers"
rsync -a --exclude target stark-provider-abi/ "$PROV/starkc/stark-provider-abi/"   # from starkc/
for p in stark-time stark-env stark-file stark-net; do
  rsync -a --exclude target "../$p/native/" "$PROV/$p/native/"
done
```

Without that root, a package declaring a capability builds only from inside a checkout. Discovery
is deliberately environment-free: no variable is consulted, and the search is the enclosing
checkout first, then the installed toolchain's own directory.

To depend on a first-party STARK package from anywhere, install its sources and point at them
with an absolute path:

```json
{ "dependencies": { "stark_time": { "package": "stark-time", "path": "/absolute/path/stark-time" } } }
```

### Single-file workflow

Parse a program:

```bash
cargo run -- parse examples/gate3/01_hello.stark
```

Type-check and borrow-check it:

```bash
cargo run -- check examples/gate3/01_hello.stark
```

Execute it:

```bash
cargo run -- run examples/gate3/01_hello.stark
```

Check a tensor-enabled program:

```bash
cargo run -- check \
  --extension tensor \
  examples/gate4/valid_pipeline.stark
```

### Project workflow (multi-file)

Create a project with `starkpkg.json`:

```bash
mkdir myapp && cd myapp
mkdir -p src
cat > starkpkg.json << 'EOF'
{
  "name": "myapp",
  "version": "0.1.0",
  "entry": "src/main.stark"
}
EOF
```

Build and run:

```bash
stark check                    # Check the project and dependencies
stark run                      # Execute the entry point
stark build                    # Compile a native debug executable
stark test                     # Run tests
stark check --locked           # Use existing lock file (CI/CD)
stark check --offline          # Use cache only (offline mode)
```

## Command overview

### Single-file CLI (starkc binary)

```bash
# Core language
cargo run -- lex file.stark
cargo run -- parse file.stark
cargo run -- check file.stark
cargo run -- run file.stark

# Parse a block-body snippet
cargo run -- parse --snippet --dump file.stark

# Tensor extension
cargo run -- check --extension tensor file.stark

# ONNX integration
cargo run -- import model.onnx --out model.stark
cargo run -- verify model.onnx --declaration model.stark

# Deployment prototype
cargo run -- deploy \
  --extension tensor \
  pipeline.stark \
  --model model.onnx \
  --out generated-host
```

### Project CLI (stark binary)

Run from any directory in a STARK project (looks up to `starkpkg.json`):

```bash
# Project-oriented commands
stark check                     # Check package and dependencies
stark build                     # Build target/stark/debug/<package>
stark run                       # Run entry point with the reference interpreter
stark test                      # Run tests

# Build modes
stark check --locked            # Use existing stark.lock (reproducible, CI/CD)
stark check --offline           # Use cache only (no network)
stark check --locked --offline  # Both (maximum strictness)
```

`stark build` requires Rust 1.85 or newer and uses the locally installed
`stark-runtime` crate without network access. Cross-platform release archives
for macOS, Linux, and Windows include the `stark`, `starkc`, and `starkide`
binaries, that runtime, and platform installers; see
[`starkc/README.md`](starkc/README.md#release-binaries).

## Editor support

### Language server and VS Code extension

`starkc lsp` speaks LSP over stdio, backed by the compiler's own analysis rather than a parallel
model of the language — hover reads the compiler's symbol table, and navigation uses resolved
symbol identity. The extension lives in [`editors/vscode/`](editors/vscode/):

```bash
cd editors/vscode
npm run compile
npx @vscode/vsce package -o stark-language.vsix
code --install-extension stark-language.vsix
```

**Confirmed in a real editor session:** hover, go-to-definition, find-references. Rename,
diagnostics, formatting, completion, signature help, document symbols and semantic tokens are
advertised and protocol-tested, but have no interactive record yet — which is why Gate C8 is
candidate-complete rather than closed.

One setup trap worth knowing: the extension defaults `stark.compiler.path` to `starkc` on `PATH`,
and a VS Code launched from a desktop environment does not inherit a shell `PATH`. If the server
does not start, set that setting to an absolute path.

### Terminal IDE

The repository also includes `starkide`, a dependency-free terminal workbench inspired by classic Turbo-style development environments.

```bash
cargo run --bin starkide
```

Or open an existing source file:

```bash
cargo run --bin starkide -- ../Practice/Basics/hello.st
```

It provides:

* a Unicode-aware source editor;
* multiple buffers;
* project and recent-file navigation;
* search;
* undo and redo;
* compiler diagnostics;
* build and run output;
* keyboard-driven menus.

Important keys:

| Key       | Action            |
| --------- | ----------------- |
| `F2`      | Save              |
| `F4`      | Visit diagnostics |
| `F9`      | Check program     |
| `Ctrl+F9` | Run program       |
| `F10`     | Open menus        |
| `Ctrl+Q`  | Quit              |

## Project maturity

STARK is an advanced prototype, not a production-ready language.

The following areas are working:

* normative Core v1 specification (Phases 0–5);
* lexer and parser with full conformance (Gate 1);
* multi-file module system with cross-file resolution (Phase 1);
* package manifests (`starkpkg.json`) with local path dependencies (Phase 2);
* semantic versioning and reproducible dependency resolution (Phase 3);
* structured diagnostics with source spans;
* name and module resolution across files;
* type checking and local inference;
* generics, traits and associated types;
* ownership and borrow checking;
* trait default-method body checking (Phase 5);
* cross-package trait coherence (Phase 5);
* typed-HIR execution;
* Core runtime with standard traits, collections and iterators (Phases 4A–4B);
* tensor shape, dtype and device analysis (Gate 4–7);
* ONNX signature import and verification (Gate 4);
* symbolic shape arithmetic and value-range semantics (Gate 7);
* native inference deployment with ONNX Runtime (Gate 5);
* native compilation of ordinary Core programs, debug and release, on Linux, macOS and Windows
  (Gate C7) — over a *qualified* subset: Gate C6 audited 87 standard-library methods and verified
  an invocation for 59, and makes no claim of full Core or standard-library native conformance;
* compiler-backed language services over LSP (Gate C8, candidate-complete);
* lock files (`stark.lock`) with SHA-256 content hashing;
* offline and locked build modes.

The following areas remain incomplete or intentionally deferred:

* a complete standard library (Phase 4 started; Phase 4+ ongoing);
* **iterator combinators and by-value `Vec` iteration** — `map`, `filter`, `count`, `collect`,
  `fold`, `reduce`, `any`, `all` and `find` are **refused by the front end** with `E0105`. They
  ran in the reference interpreter while no compiler could lower them, and a program the language
  accepts but no engine can build is worse than one it refuses. Iterate a borrow (`v.iter()`) in a
  `for` loop; implementing the combinators needs MIR adapter types and is scheduled work, not a
  rejection of the feature;
* networking libraries;
* public package registry (Phase 3+ defined; not yet implemented);
* interactive editor validation beyond hover, definition and references (Gate C8's open item);
* editor integrations beyond VS Code;
* qualification of every *usage shape* of a supported library method — an audited method has at
  least one verified invocation, which is not the same as every valid use of it working;
* stable debugging and profiling tools;
* mature FFI;
* capturing closures (deferred, not in Core v1);
* training and automatic differentiation (deferred, extension);
* GPU kernel generation (deferred, extension);
* a custom tensor runtime (using ONNX Runtime instead);
* broad platform and architecture validation;
* API and language stability guarantees.

Expect breaking changes.

## Delivery gates

STARK development is organised around evidence-based gates.

**This table is the original tensor/deployment track and stops at its Gate 7 decision.** It is not
the project's current position: the Gate C0–C10 compiler track has continued past it and is
summarised in [Where the compiler actually is](#where-the-compiler-actually-is), with
[`COMPILER-STATE.md`](COMPILER-STATE.md) as the authority. Both are live records — they answer
different questions, and neither supersedes the other.

| Gate   | Scope                                              | Status      |
| ------ | -------------------------------------------------- | ----------- |
| Gate 1 | Core lexer, parser and fixture conformance         | Complete    |
| Gate 2 | Resolution, type checking, ownership and borrowing | Complete    |
| Gate 3 | Executable Core path and `core-min` runtime        | Complete    |
| Gate 4 | Tensor frontend and ONNX signature integration     | Complete    |
| Gate 5 | Native inference deployment prototype              | Complete    |
| Gate 6 | Go, revise or stop decision based on evidence      | Decision recorded — REVISE |
| Gate 7 | Symbolic-shape + semantic tensor deployment experiment | Decision recorded — RETAIN AS RESEARCH |

Gate 5 is intended to produce one reproducible computer-vision deployment and measure:

* output correctness;
* artifact size;
* startup time;
* peak memory;
* steady-state latency;
* integration complexity;
* the quality of compile-time diagnostics.

The project will expand only if that evidence demonstrates a meaningful advantage over a library, schema generator or existing compiler.

## Design principles

### Safety without garbage collection

STARK uses ownership, moves and borrowing to manage values without requiring a tracing garbage collector.

### Explicit semantics

The language avoids implicit numeric conversions. Integer overflow traps, indexing is bounds checked and pattern matches are expected to be exhaustive.

### Small Core, optional extensions

Tensor and model concepts are implemented as an explicit extension rather than being embedded throughout the Core language.

### Existing inference backends

STARK does not attempt to implement convolution kernels, GPU drivers or a new ML runtime. The current prototype generates a host using ONNX Runtime.

### Specification and implementation remain aligned

When implementation work exposes a specification defect, the project updates the normative specification, generated documents, fixtures and compiler together rather than silently diverging.

## Repository layout

```text
STARKLANG/
  docs/spec/              Normative STARK Core v1 specification
  docs/extensions/        Optional extension specifications
  docs/ROADMAP.md         Evidence-based delivery gates
  docs/PLAN.md            Engineering plan and technical decisions
  tests/spec-fixtures/    Extracted specification conformance corpus

starkc/
  src/                    Rust compiler and interpreter
  src/extensions/tensor/  Tensor extension implementation
  src/onnx/               ONNX metadata import and verification
  src/deploy/             Deployment IR and host generation
  examples/gate3/         Executable Core examples
  examples/gate4/         Tensor and ONNX examples
  tests/                  Unit, integration and conformance tests
  docs/                   Gate exit reports and technical documentation

Practice/                  Early language experiments
```

## Testing and conformance

Run the complete suite:

```bash
cargo test --all-targets --all-features
```

Additional validation:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --all-targets
cargo doc --no-deps
```

The repository includes:

* 121 extracted specification fixtures;
* parser and semantic conformance tests;
* valid-program suites;
* exact-output interpreter tests;
* borrow and ownership negative tests;
* deterministic pseudo-fuzz robustness tests;
* tensor semantic tests;
* ONNX malformed-input and boundary tests;
* deployment lowering and emission tests.

Passing tests demonstrate the bounded behaviour covered by the current corpus. They do not yet constitute a language stability or production-readiness guarantee.

## Documentation

Start with:

* [`STARKLANG/docs/index.md`](STARKLANG/docs/index.md)
* [`STARKLANG/docs/spec/STARK-Core-v1.md`](STARKLANG/docs/spec/STARK-Core-v1.md)
* [`STARKLANG/docs/ROADMAP.md`](STARKLANG/docs/ROADMAP.md)
* [`STARKLANG/docs/PLAN.md`](STARKLANG/docs/PLAN.md) (currently tracks only through Gate 5; see `COMPILER-STATE.md` for current status)
* [`starkc/docs/gate1-exit.md`](starkc/docs/gate1-exit.md)
* [`starkc/docs/gate2-exit.md`](starkc/docs/gate2-exit.md)
* [`starkc/docs/gate3-exit.md`](starkc/docs/gate3-exit.md)
* [`starkc/docs/gate4-exit.md`](starkc/docs/gate4-exit.md)
* [`starkc/docs/gate5-exit.md`](starkc/docs/gate5-exit.md)
* [`starkc/docs/gate6-memo.md`](starkc/docs/gate6-memo.md) — decision: REVISE
* [`starkc/docs/gate7-decision.md`](starkc/docs/gate7-decision.md) — decision: RETAIN AS RESEARCH LANGUAGE
* [`COMPILER-STATE.md`](COMPILER-STATE.md) — current compiler-track governance position (Gate C0–C10, a newer, independent evidence-first re-closure track; see [`STARKLANG/docs/compiler/COMPILER-CHARTER.md`](STARKLANG/docs/compiler/COMPILER-CHARTER.md))
* [`starkc/docs/dev/compiler-map.md`](starkc/docs/dev/compiler-map.md) — module-by-module compiler pipeline map

## Contributing

STARK is currently best suited to contributors interested in:

* compiler implementation;
* programming-language semantics;
* ownership and borrow analysis;
* diagnostics;
* conformance testing;
* tensor type systems;
* ONNX tooling;
* reproducible native AI deployment.

Useful contributions include:

* minimal reproducible compiler bugs;
* specification ambiguities or contradictions;
* missing positive and negative test cases;
* diagnostic quality improvements;
* ONNX metadata edge cases;
* documentation corrections;
* carefully bounded Gate 5 deployment work.

Before proposing a large language feature, review the roadmap and current non-goals. New Core features should be supported by a concrete requirement that cannot be addressed cleanly through the existing language or a library.

## Influences

STARK’s design draws inspiration from:

* Rust for ownership and borrowing;
* Swift for TensorFlow and Mojo for typed ML language exploration;
* Julia for numerical programming;
* conventional ahead-of-time deployment toolchains.

STARK does not aim to reproduce any of these languages. Its initial focus is narrower: verifiable, reproducible native AI inference deployment.

## License

STARK is available under the MIT License.

See [`LICENSE`](LICENSE).
