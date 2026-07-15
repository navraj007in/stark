# STARK Language

> A statically typed language for safe, verifiable AI inference pipelines—built on an ownership-safe general-purpose core.

**Status: pre-alpha · active development**

STARK is an experimental programming language designed to catch errors in AI deployment pipelines before inference begins.

Its general-purpose Core provides static typing, ownership, borrowing, structured error handling and predictable execution semantics. The optional tensor extension adds compile-time checks for tensor shapes, element types, devices and imported model signatures.

STARK currently includes a working Rust compiler, semantic checker, borrow checker, interpreter, ONNX signature importer and an early native deployment pipeline.

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

Early Gate 5 work can lower a supported STARK inference program into a bounded deployment IR and generate a self-contained Rust host project.

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

This deployment path is still under active development. The complete measured Gate 5 demonstration has not yet been declared finished.

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

## Command overview

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

## Terminal IDE

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

* normative Core v1 specification;
* lexer and parser;
* structured diagnostics;
* name and module resolution;
* type checking and local inference;
* generics and traits;
* ownership and borrow checking;
* typed-HIR execution;
* minimal Core runtime;
* tensor shape, dtype and device analysis;
* ONNX signature import and verification;
* initial deployment lowering and host generation.

The following areas remain incomplete or intentionally deferred:

* production native code generation for ordinary Core programs;
* a broad standard library;
* networking;
* package management and a package registry;
* a language server and mainstream editor integrations;
* stable debugging and profiling tools;
* mature FFI;
* capturing closures;
* training and automatic differentiation;
* GPU kernel generation;
* a custom tensor runtime;
* broad platform and architecture validation;
* API and language stability guarantees.

Expect breaking changes.

## Delivery gates

STARK development is organised around evidence-based gates.

| Gate   | Scope                                              | Status      |
| ------ | -------------------------------------------------- | ----------- |
| Gate 1 | Core lexer, parser and fixture conformance         | Complete    |
| Gate 2 | Resolution, type checking, ownership and borrowing | Complete    |
| Gate 3 | Executable Core path and `core-min` runtime        | Complete    |
| Gate 4 | Tensor frontend and ONNX signature integration     | Complete    |
| Gate 5 | Native inference deployment prototype              | In progress |
| Gate 6 | Go, revise or stop decision based on evidence      | Pending     |

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
* [`STARKLANG/docs/PLAN.md`](STARKLANG/docs/PLAN.md)
* [`starkc/docs/gate1-exit.md`](starkc/docs/gate1-exit.md)
* [`starkc/docs/gate2-exit.md`](starkc/docs/gate2-exit.md)
* [`starkc/docs/gate3-exit.md`](starkc/docs/gate3-exit.md)
* [`starkc/docs/gate4-exit.md`](starkc/docs/gate4-exit.md)

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
