# STARK Language

> A statically typed language for safe, verifiable AI inference pipelines—built on an ownership-safe general-purpose core.

**Status: pre-alpha · active development**

STARK is an experimental programming language designed to catch errors in AI deployment pipelines before inference begins.

Its general-purpose Core provides static typing, ownership, borrowing, structured error handling and predictable execution semantics. The optional tensor extension adds compile-time checks for tensor shapes, element types, devices and imported model signatures.

STARK currently includes a working Rust compiler, semantic checker, borrow checker, interpreter, ONNX signature importer, multi-file module system, package management with semantic versioning, native compilation, compiler-backed language services, 28 first-party packages — among them an HTTP/1.1 and HTTPS client written in STARK — and a release installer for macOS, Linux and Windows.

### Where the compiler actually is

Two gate sequences run in this repository and they answer different questions. The **Gate 1–7** table further down covers the original tensor/deployment track; its tensor verdicts still stand (*REVISE*, then technical POSITIVE with productisation DEFER), while the project-wide "research language" policy Gate 7 also carried was [superseded on 2026-08-04](starkc/docs/gate7-superseded.md). The **Gate C0–C10** track is a separate, evidence-first re-closure of Core v1 conformance and execution, and it is where current work happens:

| Track gate | Position |
| --- | --- |
| C0–C6 | Closed. C6 closed with a *qualified* native subset — 59 of 87 audited standard-library methods have a verified invocation, 28 explicitly refused or excluded |
| **C7** | **Closed.** Native compilation with debug and release profiles, build cache, MIR optimiser, and an HTTP/JSON REST workload qualified on Linux, macOS and Windows |
| **C8** | **Candidate-complete.** Compiler-backed language services (LSP + VS Code extension). Held open by interactive editor validation: hover, go-to-definition and find-references are confirmed in a real session; the other seven capabilities are protocol-tested only |
| **C9** | Open. Extension isolation, and a conditionally authorised artifact-provider generalisation whose second half is blocked pending evidence from a second artifact format |

Programs are compared across four execution configurations — the HIR reference interpreter, the MIR interpreter, native debug and native release — with each case's expected result pinned against the specification rather than against another engine's output. `COMPILER-STATE.md` is the authoritative position; this table is a summary and can lag it.

Two tracks ran alongside the compiler gates and are now closed. The **HTTP client track (HC0–HC13)**
closed on 2026-08-03 with a qualified HTTP/1.1 and HTTPS client written in STARK — see
[First-party packages](#first-party-packages). **Installer Phase I** closed the same day: release
archives, platform installers, a versioned install tree and `stark doctor`. Neither makes STARK
releasable, and [Project maturity](#project-maturity) says exactly what still stands in the way.

Forward work is governed by [`ROADMAP.md`](ROADMAP.md) (repository root), the single live plan for
package, application and platform work. It does not supersede the compiler gate track.

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

Gate 5's measured demonstration is complete (see [`starkc/docs/gate5-exit.md`](starkc/docs/gate5-exit.md)); the follow-on Gate 6/7 decision checkpoints recorded REVISE and then a positive technical verdict with productisation **deferred** pending evidence from external developers — see the Delivery gates table below and [`starkc/docs/gate7-decision.md`](starkc/docs/gate7-decision.md). **The tensor track has not moved since**, and nothing in the platform work below authorises restarting it.

### First-party packages

The repository carries 28 packages written in STARK under [`packages/`](packages/), each with its
own `starkpkg.json`, lock file and test suite, and each exercised by a consumer package that must
actually *call* the surface it declares.

| Area | Packages |
| --- | --- |
| Encoding and text | `stark-ascii`, `stark-base64`, `stark-hex`, `stark-percent`, `stark-checksum`, `stark-uuid` |
| Formatting | `stark-fmt` |
| Data formats | `stark-json`, `stark-csv`, `stark-form`, `stark-mime`, `stark-query` |
| Paths and URLs | `stark-path`, `stark-glob`, `stark-url` |
| Host access | `stark-time`, `stark-env`, `stark-io`, `stark-random` |
| Command line | `stark-args` |
| Versioning | `stark-semver` |
| Networking | `stark-net` (TCP + DNS), `stark-tls` |
| HTTP | `stark-http-core`, `stark-http-parser`, `stark-http-serialize`, `stark-http-client` |

**Host access is capability-declared and provider-backed.** A package that needs the outside world
names the capability in its manifest, and a native provider crate satisfies it at build time:

| Capability | Provider crate | Declared by |
| --- | --- | --- |
| `clock` | `stark-time/native` | `stark-time` |
| `filesystem` | `stark-file/native` | `stark-io` |
| `process.env`, `process.args` | `stark-env/native` | `stark-env` |
| `random` | `stark-random/native` | `stark-random` |
| `tcp`, `dns` | `stark-net/native` | `stark-net`, `stark-http-client` |
| `tls` | `stark-tls/native` | `stark-tls`, `stark-http-client` |

**Capability-backed packages run only through `stark build`.** The reference and MIR interpreters
have no host access at all — they cannot open a socket or read a clock — so `stark run` will not
execute a program that reaches the network or the filesystem. That is a deliberate boundary, not a
gap in the interpreters.

The HTTP client is the deepest thing built in the language so far: HTTP/1.1 over TCP, HTTPS with
verified TLS from the URL alone, chunked and content-length framing, bounded redirects with
credential stripping across origins, and JSON convenience helpers. It was qualified against peers
that are adversarial on the wire — 42 executed cases on Linux, macOS and Windows — and the closing
packets found four defects, two of which were remote-abort vulnerabilities rather than parse errors.

A request is a `fetch` against a client, and HTTPS differs from HTTP only in the URL — there is no
second API and no application-level switch:

```stark
use stark_http_client::default_config;
use stark_http_client::error_text;
use stark_http_client::fetch;
use stark_http_client::new_client;

fn main() {
    let client = new_client(default_config());

    match fetch(&client, "https://example.com/health") {
        Ok(response) => {
            if response.status == 200u16 {
                println("healthy");
            }
        }
        Err(error) => {
            println(error_text(&error).as_str());
        }
    }
}
```

The package declaring this needs `"capabilities": ["tcp", "dns", "tls"]` in its `starkpkg.json`, a
path dependency on `stark-http-client` installed beside it (see
[Installing the toolchain](#installing-the-toolchain)), and `stark build` — not `stark run`.

What it does **not** do is stated as plainly: no HTTP/2 or HTTP/3, no connection reuse, no
decompression, no proxies, no cookie jar, no client certificates, no streaming bodies, and a
`connect_timeout` that is accepted and ignored (DEV-165). The full list is
[`STARKLANG/docs/http-client/HC13-KNOWN-LIMITATIONS.md`](STARKLANG/docs/http-client/HC13-KNOWN-LIMITATIONS.md),
which is the shorter and more useful companion to the qualification report.

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

### Installing the toolchain

There are two paths: a release package, or a hand-built install from this checkout. The first is
the supported one.

#### From a release package

Build a package for the current host — the builder has no dependencies beyond Python and Cargo:

```bash
# From starkc/
python3 scripts/build-release.py        # py -3 on Windows
```

Packages are written to `target/packages/` — `.tar.gz` on macOS and Linux, `.zip` on Windows — and
each carries `stark`, `starkc`, `starkide`, the native runtime, the provider ABI, a `manifest.json`
of SHA-256 hashes, the license and installers. Every operating system and CPU architecture needs its
own package; cross-compilation is validated and then refused, with its reason.

Extract the package and install from inside it:

```bash
./install.sh                            # defaults to ~/.local
./install.sh --prefix /custom/prefix
~/.local/lib/stark/uninstall.sh         # uninstall
```

```powershell
.\install.ps1                           # defaults to %LOCALAPPDATA%\Programs\STARK; updates user PATH
.\install.ps1 -Prefix C:\Tools\STARK -NoPathUpdate
& "$env:LOCALAPPDATA\Programs\STARK\lib\stark\uninstall.ps1"
```

The installer stages the payload, verifies it against `manifest.json` before anything is published,
then writes a **versioned tree** — `lib/stark/versions/<version>` with `lib/stark/current` pointing
at it — and places `stark` in `bin/` as a symlink on Unix or a copy on Windows.

Check an installation at any time:

```bash
stark doctor                            # re-hashes every manifest-listed file
stark doctor --json                     # machine-readable, for CI
stark doctor --root /path/to/package    # inspect an extracted package, including one for another platform
```

`stark doctor` establishes **integrity, not authenticity.** It detects corruption and partial
extraction. It cannot tell you the manifest came from a STARK release — anyone who can replace the
payload replaces the manifest with it. Release archives are unsigned; a public distribution still
needs a signed manifest, a trusted release key, signature verification before installation, and
platform notarisation. None of that exists yet.

The package also does **not** carry the first-party STARK packages or their provider crates. A clean
machine can build ordinary Core programs; building an HTTP or TLS program still means obtaining
those sources separately, as below.

#### By hand, from this checkout

Install the binaries **and the crates the native backend links against**. The compiler finds those
by a fixed layout relative to its own executable, so the directory structure below is not a
suggestion — it is what `stark build` looks for.

```bash
# From starkc/
cargo build --release --bins

PREFIX="$HOME/.local"                       # must be on PATH
install -m 755 target/release/stark    "$PREFIX/bin/"
install -m 755 target/release/starkc   "$PREFIX/bin/"
install -m 755 target/release/starkide "$PREFIX/bin/"

# The installed tree MIRRORS THE REPOSITORY. That is load-bearing, not tidiness: the runtime
# depends on `../stark-provider-abi` and each provider crate on `../../starkc/stark-provider-abi`,
# so only a repository-shaped root makes both resolve to ONE copy. Cargo refuses a lockfile that
# names one package at two paths — even when the second is a symlink to the first.
L="$PREFIX/lib/stark"
rsync -a --exclude target stark-runtime/      "$L/starkc/stark-runtime/"
rsync -a --exclude target stark-provider-abi/ "$L/starkc/stark-provider-abi/"
```

Three binaries, not one: `stark` is the package driver, `starkc` the single-file CLI and language
server, `starkide` the terminal IDE. The VS Code extension defaults to `starkc`.

**Provider-backed capabilities** — clock, filesystem, environment, random, TCP/DNS and TLS — need
their provider crates too, in a root that mirrors the repository's shape. All six, or the missing
ones fail at build time rather than at install time:

```bash
# Under `packages/`, exactly as in a checkout — they share the ABI crate installed above.
for p in stark-time stark-env stark-file stark-net stark-random stark-tls; do
  rsync -a --exclude target "../packages/$p/native/" "$PREFIX/lib/stark/packages/$p/native/"
done
```

The `packages/` level is load-bearing, not decoration. Each provider crate reaches the ABI through
`../../../starkc/stark-provider-abi`, which resolves only at that depth — install one level up and
Cargo looks for the ABI where it is not.

Without that root, a package declaring a capability builds only from inside a checkout. Discovery
is deliberately environment-free: no variable is consulted, and the search is the enclosing
checkout first, then the installed toolchain's own directory.

To depend on a first-party STARK package, install its sources **beside your own package** and name
it by path. The path may be absolute or relative, but it must resolve inside the workspace, and the
workspace is the *parent directory of your package* — nothing above it is reachable:

```text
projects/
  myapp/            <- your package; the workspace is `projects/`
    starkpkg.json
  stark-time/       <- a sibling, therefore reachable
  stark-json/
```

In this repository that arrangement is `packages/`, which is why every first-party manifest names
its dependencies as plain siblings (`../stark-percent`) and why moving them all together changed
none of those paths.

```json
{ "dependencies": { "stark_time": { "package": "stark-time", "path": "../stark-time" } } }
```

A path pointing outside that root is refused by name — `resolves to '…' which is outside the
permitted workspace` — rather than being silently resolved. It is the reason a single shared
package directory somewhere else on the machine does not work today, and a registry is the
scheduled answer.

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
stark build --release           # Optimised profile, same STARK-observable semantics
stark run                       # Run entry point with the reference interpreter
stark test                      # Run fn test_* functions, tests/ programs and examples/
stark test http --show-output   # Filter by substring; print output from passing tests too
stark fmt                       # Format the package
stark fmt --check               # Report non-canonical files without rewriting them (exit 1)
stark doc --open                # Generate API documentation for public items
stark cache status              # Report the bounded build cache
stark cache clean               # Clear it
stark doctor                    # Verify the installed toolchain against its manifest

# Build modes
stark check --locked            # Use existing stark.lock (reproducible, CI/CD)
stark check --offline           # Use cache only (no network)
stark check --locked --offline  # Both (maximum strictness)
stark build --no-build-cache    # Discard the generated crate afterwards (the qualification path)
stark build --no-mir-opt        # Compile MIR exactly as lowered, to bisect a suspected optimiser defect
```

The build cache reuses whole content-addressed generated crates and their Cargo artefacts. It is
not fine-grained incremental compilation, and it is not trying to be.

`stark build` requires Rust 1.85 or newer and uses the locally installed
`stark-runtime` crate without network access. Release archives for macOS, Linux
and Windows carry the `stark`, `starkc` and `starkide` binaries, that runtime,
the provider ABI, a hash manifest and platform installers — but not the
first-party packages or their providers; see
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
advertised and protocol-tested, but have no interactive record yet. Gate C8 **closed** on 2026-08-06
(CD-385) with exactly that limit stated rather than removed, and `DEV-012` stays open for those
seven features.

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
* compiler-backed language services over LSP (Gate C8, closed with DEV-012 open for
  seven features);
* lock files (`stark.lock`) with SHA-256 content hashing;
* offline and locked build modes;
* 28 first-party packages written in STARK, each with a consumer package that calls its declared
  surface;
* manifest-declared host capabilities backed by native provider crates — clock, filesystem,
  environment, random, TCP/DNS and TLS — with cross-provider ownership transfer and affine host
  resources;
* an HTTP/1.1 and HTTPS client written in STARK, qualified against adversarial peers on Linux,
  macOS and Windows (HC0–HC13);
* release packaging and installation — archives, platform installers, a versioned install tree,
  uninstall, and `stark doctor` manifest verification (Installer Phase I).

The following areas remain incomplete or intentionally deferred:

* a complete standard library (Phase 4 started; Phase 4+ ongoing);
* **iterator combinators and by-value `Vec` iteration** — `map`, `filter`, `count`, `collect`,
  `fold`, `reduce`, `any`, `all` and `find` are **refused by the front end** with `E0105`. They
  ran in the reference interpreter while no compiler could lower them, and a program the language
  accepts but no engine can build is worse than one it refuses. Iterate a borrow (`v.iter()`) in a
  `for` loop; implementing the combinators needs MIR adapter types and is scheduled work, not a
  rejection of the feature;
* **a releasable distribution.** Installer Phase I is implemented, but the payload does not carry
  the first-party packages or their providers, a clean machine cannot yet build an HTTP or TLS
  program offline, and the archives are unsigned — `manifest.json` establishes integrity, never
  authenticity;
* networking beyond an HTTP/1.1 client — no server, no HTTP/2 or HTTP/3, no connection reuse, no
  proxies, no streaming bodies, and `connect_timeout` is accepted and ignored (DEV-165);
* structured concurrency and persistent storage (scheduled in [`ROADMAP.md`](ROADMAP.md));
* public package registry (Phase 3+ defined; not yet implemented) — dependencies are local paths;
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
| Gate 7 | Symbolic-shape + semantic tensor deployment experiment | Tensor verdict recorded — technical POSITIVE, productisation DEFER. Its project-wide "research language" policy is [superseded](starkc/docs/gate7-superseded.md) |

Gate 5 is intended to produce one reproducible computer-vision deployment and measure:

* output correctness;
* artifact size;
* startup time;
* peak memory;
* steady-state latency;
* integration complexity;
* the quality of compile-time diagnostics.

That test governs the **tensor track**, and it has not been passed: the project will expand there
only if the evidence demonstrates a meaningful advantage over a library, schema generator or
existing compiler. The general-purpose language took a different route, answered by the Gate C0–C10
evidence and the platform work above, and that is what
[`starkc/docs/gate7-superseded.md`](starkc/docs/gate7-superseded.md) records.

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
ROADMAP.md                 The single live forward plan (packages, applications, platform)
COMPILER-STATE.md          Authoritative compiler-track position (Gate C0–C10)

STARKLANG/
  docs/spec/               Normative STARK Core v1 specification
  docs/extensions/         Optional extension specifications
  docs/compiler/           Compiler governance: charter, roadmap, work packages
  docs/http-client/        HC0–HC13 evidence, limitations and release checklist
  docs/ROADMAP.md          Historical record of the closed Gate 1–7 sequence
  docs/PLAN.md             Historical engineering plan (tracks only through Gate 5)
  docs/archive/            Pre-pivot design and superseded roadmaps — not current
  tests/spec-fixtures/     Extracted specification conformance corpus

starkc/
  src/                     Rust compiler and interpreter
  src/extensions/tensor/   Tensor extension implementation
  src/onnx/                ONNX metadata import and verification
  src/deploy/              Deployment IR and host generation
  src/bin/                 stark, starkc and starkide entry points
  stark-runtime/           Runtime crate the native backend links against
  stark-provider-abi/      Native provider ABI
  dist/                    Platform installers and uninstallers
  scripts/build-release.py Dependency-free release packager
  examples/gate3/          Executable Core examples
  examples/gate4/          Tensor and ONNX examples
  tests/                   Unit, integration and conformance tests
  docs/                    Gate exit reports and technical documentation

packages/                  First-party packages written in STARK
  stark-<name>/
    src/lib.stark          Package source
    native/                Native provider crate, where the package needs one
  stark-<name>-consumer/   The package's consumer, which must call its declared surface

editors/vscode/            Language-server client and syntax support
website/                   starklang.com — React + Vite static site
Practice/                  Early language experiments — pre-pivot, not current
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

* 116 extracted specification fixtures, each triaged in
  [`STARKLANG/tests/spec-fixtures/manifest.toml`](STARKLANG/tests/spec-fixtures/manifest.toml);
* parser and semantic conformance tests;
* valid-program suites;
* exact-output interpreter tests;
* four-engine differential conformance — reference interpreter, MIR interpreter, native debug and
  native release — with each expected result pinned against the specification;
* borrow and ownership negative tests;
* deterministic pseudo-fuzz robustness tests;
* tensor semantic tests;
* ONNX malformed-input and boundary tests;
* deployment lowering and emission tests;
* per-package STARK test suites run by `stark test`, plus consumer packages that must call each
  declared public surface;
* HTTP client qualification against controlled adversarial peers — malformed, oversized and
  stalling — asserted on the named error rather than on failure alone.

Passing tests demonstrate the bounded behaviour covered by the current corpus. They do not yet constitute a language stability or production-readiness guarantee.

## Documentation

Start with:

* [`STARKLANG/docs/index.md`](STARKLANG/docs/index.md)
* [`STARKLANG/docs/spec/STARK-Core-v1.md`](STARKLANG/docs/spec/STARK-Core-v1.md)
* [`ROADMAP.md`](ROADMAP.md) — **the single live forward plan**, August 2026 – February 2027
* [`COMPILER-STATE.md`](COMPILER-STATE.md) — the authoritative compiler-track position
* [`STARKLANG/docs/http-client/HC13-QUALIFICATION-REPORT.md`](STARKLANG/docs/http-client/HC13-QUALIFICATION-REPORT.md)
  and [`HC13-KNOWN-LIMITATIONS.md`](STARKLANG/docs/http-client/HC13-KNOWN-LIMITATIONS.md) — what the
  HTTP client does, and what it does not

Historical records, retained for their citations and non-goals rather than as plans:

* [`STARKLANG/docs/ROADMAP.md`](STARKLANG/docs/ROADMAP.md) — the closed Gate 1–7 sequence
* [`STARKLANG/docs/PLAN.md`](STARKLANG/docs/PLAN.md) (tracks only through Gate 5)
* [`starkc/docs/gate1-exit.md`](starkc/docs/gate1-exit.md)
* [`starkc/docs/gate2-exit.md`](starkc/docs/gate2-exit.md)
* [`starkc/docs/gate3-exit.md`](starkc/docs/gate3-exit.md)
* [`starkc/docs/gate4-exit.md`](starkc/docs/gate4-exit.md)
* [`starkc/docs/gate5-exit.md`](starkc/docs/gate5-exit.md)
* [`starkc/docs/gate6-memo.md`](starkc/docs/gate6-memo.md) — decision: REVISE
* [`starkc/docs/gate7-decision.md`](starkc/docs/gate7-decision.md) — tensor verdict: technical POSITIVE, productisation DEFER
* [`starkc/docs/gate7-superseded.md`](starkc/docs/gate7-superseded.md) — retires that memo's project-wide "research language" policy (2026-08-04), leaving its tensor verdicts intact

For compiler work specifically:

* [`STARKLANG/docs/compiler/COMPILER-CHARTER.md`](STARKLANG/docs/compiler/COMPILER-CHARTER.md) — the
  governance rules the Gate C0–C10 track runs under
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
* reproducible native AI deployment;
* libraries written in STARK itself, and the native providers behind them.

Useful contributions include:

* minimal reproducible compiler bugs;
* specification ambiguities or contradictions;
* missing positive and negative test cases;
* diagnostic quality improvements;
* ONNX metadata edge cases;
* documentation corrections;
* adversarial cases against the first-party packages, asserted on the named error;
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
