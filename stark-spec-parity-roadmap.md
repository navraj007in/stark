# STARK Specification-Parity Roadmap

**Baseline:** Latest reviewed repository state at commit `19db8c2`  
**Objective:** Close the gap between the current STARK implementation and the normative Core v1 and tensor v0.1 specifications.

The implementation already has a strong Core parser, semantic checker, ownership analysis, interpreter, tensor front end, ONNX importer, verifier, and bounded native deployment path. The remaining work is concentrated in multi-file compilation, packages, the full standard library, semantic completeness, general native compilation, complete tensor lowering, and production tooling.

---

## Phase 0 — Establish an Auditable Conformance Baseline (Status: Complete — 2026-07-16)

### Objective

Convert the specification-to-implementation gap into a machine-maintained inventory.

### Deliverables

- Create `STARKLANG/conformance/core-v1-coverage.toml`.
- Assign every normative rule a stable identifier, for example:
  - `LEX-001`
  - `TYPE-034`
  - `MEM-012`
  - `STD-VEC-008`
  - `PKG-014`
- Record one status for every rule:
  - `implemented`
  - `partial`
  - `missing`
  - `intentionally-deferred`
  - `spec-defect`
- Link every implemented rule to:
  - compiler source;
  - positive tests;
  - negative tests where applicable.
- Add a CI check that fails when normative rules have no recorded status.
- Generate a chapter-by-chapter coverage report.
- Record all known implementation deviations explicitly.

### Exit Criteria

- Every normative Core v1 requirement is represented.
- Every claimed implementation has executable evidence.
- No feature is considered complete merely because its syntax parses.
- A generated report shows coverage by specification chapter.

### Relative Size

Small.

---

## Phase 1 — Complete Multi-File Modules (Status: Complete — 2026-07-16)

### Objective

Make `mod foo;` genuinely compile, type-check, diagnose, and run code from other files.

### Architecture

Refactor the compiler from a single-root AST model into a source-module graph:

```text
Package
  └── ModuleGraph
        ├── ModuleId
        ├── SourceFile
        ├── AST
        ├── parent module
        ├── child modules
        └── resolved items
```

### Deliverables

- Retain each loaded module's AST.
- Recursively declare and lower items from:
  - `foo.stark`
  - `foo/mod.stark`
- Support nested file modules.
- Maintain a global source map for diagnostics.
- Ensure diagnostics point to the correct source file.
- Detect:
  - missing module files;
  - duplicate module declarations;
  - conflicting `foo.stark` and `foo/mod.stark`;
  - invalid module paths.
- Make `check`, `run`, the terminal IDE, and VS Code diagnostics use the same module graph.
- Complete straightforward `pub use` re-exports.

### Required Tests

```text
main.stark
math.stark
network/mod.stark
network/http.stark
```

Test:

- cross-file function calls;
- types and traits across files;
- private and public access;
- nested `super`, `self`, and `crate` paths;
- cross-file generic calls;
- cross-file ownership errors;
- cross-file runtime execution.

### Exit Criteria

This must work end to end:

```stark
// main.stark
mod math;
use crate::math::add;

fn main() {
    println(add(2, 3));
}
```

```stark
// math.stark
pub fn add(a: Int32, b: Int32) -> Int32 {
    a + b
}
```

Both `starkc check` and `starkc run` must succeed.

### Relative Size

Medium to large.

---

## Phase 2 — Implement Package Manifests and Local Dependencies (Status: Complete — 2026-07-16)

### Objective

Deliver a usable project model before attempting a public registry.

### Manifest

Implement `starkpkg.json`:

```json
{
  "name": "my-app",
  "version": "0.1.0",
  "entry": "src/main.stark",
  "dependencies": {
    "utilities": {
      "path": "../utilities"
    }
  }
}
```

### Deliverables

- Package-root discovery.
- Manifest parsing and validation.
- Default entry file: `src/main.stark`.
- Custom entry-file support.
- Package-name validation.
- Version validation.
- Path dependency loading.
- Canonical path checks.
- Workspace-boundary enforcement.
- Package dependency graph.
- Direct and transitive cycle detection.
- External package import roots.
- Package-aware diagnostic paths.

### CLI

Introduce project-oriented commands:

```bash
stark build
stark check
stark run
stark test
```

Retain file-oriented commands for experimentation:

```bash
starkc check file.stark
```

### Exit Criteria

A three-package local workspace must compile:

```text
app
 ├── depends on image-pipeline
 └── depends on utilities

image-pipeline
 └── depends on utilities
```

The compiler must reject:

- missing manifests;
- invalid entries;
- dependency cycles;
- duplicate package names;
- imports from undeclared dependencies;
- path dependencies outside the permitted workspace.

### Relative Size

Large.

---

## Phase 3 — Reproducible Dependency Resolution (Status: Complete — 2026-07-16)

### Objective

Move from local workspaces to deterministic versioned packages.

This phase should begin only after local dependency support is stable.

### Deliverables

- Semantic-version parser.
- Exact, caret, and comparator constraints.
- Dependency solver.
- Deterministic highest-compatible-version selection.
- Package cache.
- Lock file, for example:

```text
stark.lock
```

- Locked and offline build modes:

```bash
stark build --locked
stark build --offline
```

- Dependency provenance in build output.
- Content hashes for resolved packages.
- Registry abstraction:
  - filesystem registry first;
  - remote registry later.
- Protection against dependency substitution and path traversal.

### Exit Criteria

- Identical manifest and lock file produce identical dependency graphs.
- Resolution remains stable across machines.
- Conflicting constraints produce actionable diagnostics.
- Offline builds succeed when the cache is complete.
- Corrupted cached packages are rejected.

### Relative Size

Large.

---

## Phase 4 — Complete the Core Standard Library

The current `core-min` profile is sufficient for the prototype but not for full Core v1 parity. Deliver this phase incrementally.

---

### Phase 4A — Prelude and Core Traits

Implement and validate:

- `Clone`
- `Hash`
- `Default`
- `Display`
- `Error`
- `From`
- `Into`
- `TryFrom`
- `Index`
- `IndexMut`
- `Iterator`
- `FromIterator`

Also implement:

- `size_of`
- `align_of`
- `swap`
- `replace`
- `take`

---

### Phase 4B — Complete `String`, `str`, and `Vec`

Add missing APIs such as:

- `chars`
- `bytes`
- `contains`
- `starts_with`
- `ends_with`
- `find`
- `replace`
- `split`
- `trim`
- case conversion
- `Vec::append`
- `Vec::extend`
- `Vec::iter`
- `Vec::as_slice`
- complete mutable-reference-returning methods

---

### Phase 4C — Collections

Implement:

- `HashMap<K, V>`
- `HashSet<T>`
- keys, values, and entry iteration
- ownership-safe mutable lookup
- deterministic behavioral tests

---

### Phase 4D — Iterators

Implement:

- `map`
- `filter`
- `fold`
- `reduce`
- `collect`
- `count`
- `find`
- `any`
- `all`

Borrow-carrying iterator types must interact correctly with the borrow checker.

---

### Phase 4E — Math, Random, and I/O

Implement:

- logarithmic and exponential functions;
- trigonometry;
- rounding;
- `Random`;
- `File`;
- `IOError`;
- stderr output;
- structured file operations.

### Exit Criteria

- Every normative standard-library signature has an implementation.
- Every documented behavioral guarantee has tests.
- Borrow-carrying iterators cannot outlive their source.
- The standard library is accessible through a real `std` package/module tree rather than only hardcoded resolver names.

### Relative Size

Very large; deliver incrementally.

---

## Phase 5 — Close Remaining Core Semantic Gaps

### Objective

Move from representative Core support to a credible Core v1 conformance claim.

### Trait Completeness

- Fully type-check trait default-method bodies.
- Validate generic substitutions inside default methods.
- Check default methods against implementing types.

### Re-Export Completeness

- Complete `pub use`.
- Resolve multi-level re-export graphs.
- Detect ambiguous re-exports.
- Enforce privacy through re-export chains.
- Support re-exports across packages.

### Match Analysis

Add:

- unreachable match-arm warnings;
- redundant pattern warnings;
- usefulness analysis;
- nested-pattern usefulness;
- range-pattern overlap analysis where applicable.

### Cross-Package Coherence

- Apply orphan and overlap rules across package boundaries.
- Prevent duplicate trait implementations across dependencies.
- Produce diagnostics identifying both conflicting implementations.

### Borrow Checker Validation

Stress the borrow checker against:

- iterators;
- collection views;
- nested generic wrappers;
- methods returning references;
- cross-module and cross-package APIs.

### Exit Criteria

- All intentional Gate 2 semantic deferrals are closed.
- Every semantic rule has positive and negative tests.
- Public APIs are checked across package boundaries.
- No known normative semantic deviation remains.

### Relative Size

Medium to large.

---

## Phase 6 — Build System and General Native Compilation

### Objective

Decide whether STARK remains an interpreted Core language with a deployment backend or fulfills its compiled-language positioning.

### Recommended Architecture

Introduce a backend-independent middle layer:

```text
AST
 → HIR
 → typed HIR
 → MIR
 → interpreter
 → native backend
```

Keep the interpreter as the reference implementation.

### Backend Sequence

1. Cranelift for scalar Core native compilation.
2. Generated Rust or C as a bootstrap alternative.
3. LLVM only if optimization evidence justifies the additional complexity.

### Deliverables

- MIR with explicit:
  - control-flow blocks;
  - moves;
  - drops;
  - calls;
  - aggregates;
  - panic paths.
- Native code generation for:
  - primitives;
  - functions;
  - structs and enums;
  - control flow;
  - references;
  - standard-library calls.
- Debug symbols and stack traces.
- Build profiles:

```bash
stark build
stark build --release
```

- Target selection.
- Interpreter/native differential testing.

### Exit Criteria

- The same Core conformance suite passes under interpreter and native execution.
- Drop order and runtime failure semantics match.
- A normal multi-package Core application produces a standalone executable.
- Native execution is not limited to tensor deployment.

### Relative Size

Very large.

---

## Phase 7 — Complete Tensor Specification Execution

The tensor front end is strong, but not every checked operation is executable through the deployment backend.

### Deliverables

Implement lowering for the full normative tensor operation set:

- `zeros`, `ones`, `full`, `from_vec`;
- elementwise operations;
- static broadcasting;
- `broadcast_to`;
- `matmul`;
- `batch_matmul`;
- `reshape`;
- `permute`;
- `transpose`;
- `concat`;
- slicing;
- reductions;
- `softmax`;
- `argmax`;
- `cast`;
- `to_device`;
- runtime `refine`;
- value-range transitions:
  - `scale_255`
  - `normalize`

### Backend Capability Model

The compiler must distinguish:

```text
supported by frontend
supported by selected backend
unsupported by selected backend
```

A checked operation must never silently reach a backend that cannot execute it.

### Device Work

- Real CPU implementation.
- CUDA only after a concrete backend is selected.
- Explicit device-transfer lowering.
- Runtime validation of loaded model placement.

### Exit Criteria

- Every tensor v0.1 operation is either executable or documented as an explicit conformance deviation.
- Dynamic refinement is runtime-tested.
- Shape, dtype, device, and range guarantees remain intact after lowering.
- Multiple models and multi-input/multi-output models are tested.

### Relative Size

Very large.

---

## Phase 8 — Production Tooling

### Objective

Make the completed language usable without requiring knowledge of compiler internals.

### Language Server

Implement:

- completion;
- hover types;
- go to definition;
- find references;
- rename;
- document symbols;
- workspace symbols;
- signature help;
- semantic tokens.

### Formatter

```bash
stark fmt
stark fmt --check
```

The formatter should be specification-driven and idempotent.

### Test Framework

```bash
stark test
```

Support:

- unit tests;
- integration tests;
- package test discovery;
- filtered runs;
- structured output.

### Package-Aware VS Code Support

- project-root detection;
- dependency-aware diagnostics;
- cross-file navigation;
- external-package source navigation;
- test commands;
- build profiles.

### Documentation

```bash
stark doc
```

Generate package and API documentation.

### Exit Criteria

A new developer can run:

```bash
git clone ...
stark build
stark test
stark run
```

without manually invoking Cargo or locating `starkc`.

### Relative Size

Large.

---

## Phase 9 — Full Conformance and Release Qualification

### Objective

Make an evidence-backed Core v1 and tensor v0.1 implementation claim.

### Deliverables

- Generated conformance dashboard.
- All specification examples machine-classified.
- Full API tests for the standard library.
- Multi-file and multi-package conformance tests.
- Interpreter/native differential tests.
- Property and fuzz testing.
- Malformed manifest and dependency corpus.
- Security review of:
  - package resolution;
  - archive extraction;
  - build scripts;
  - compiler process execution.
- Performance baselines.
- Compatibility policy.
- Deprecation policy.
- Versioned release artifacts.
- Reproducible builds.

### Exit Criteria

A release can state precisely:

```text
STARK Core v1: conforming
Tensor extension v0.1: conforming
Known deviations: none
```

Alternatively, every remaining deviation must be listed explicitly.

---

# Recommended Sequencing

## Track 1 — Genuine Core v1 Usability

```text
Phase 0
 → Phase 1
 → Phase 2
 → Phase 4
 → Phase 5
 → Phase 9
```

## Track 2 — Package Ecosystem

```text
Phase 2
 → Phase 3
 → Phase 8
 → Phase 9
```

## Track 3 — General Compiled Language

```text
Phase 4
 → Phase 5
 → Phase 6
 → Phase 9
```

## Track 4 — Full Tensor-Extension Execution

```text
Phase 1
 → Phase 2
 → Phase 7
 → Phase 9
```

---

# Recommended Immediate Scope

The next implementation block should contain only:

1. Phase 0 — conformance inventory.
2. Phase 1 — multi-file module completion.
3. Phase 2 — manifest and local path dependencies.

Do not begin the public registry, lock-file solver, native compiler, or full tensor runtime until those three foundations are complete.

## Proposed Next Gate

> **Gate 8 — Multi-File Core and Local Packages:** A three-package STARK workspace must parse, resolve, type-check, and execute with correct cross-file diagnostics, visibility, ownership behavior, and dependency-cycle detection.
