# Gemini execution brief — Gate 4 tensor front end and ONNX import

This document is the implementation contract for completing **Gate 4** of the
STARK roadmap. It is written for an implementation agent working directly in
this repository. Follow it milestone by milestone; do not reinterpret Gate 4
as permission to build a tensor runtime, training system, or generalized ML
framework.

## 1. Mission

Implement the narrowest useful slice of the STARK `tensor` v0.1 extension so
that the compiler can:

1. parse tensor-specific syntax only when the extension is enabled;
2. type-check tensor dtype, rank, shape, symbolic dimensions, and device;
3. parse and validate nominal `model` declarations;
4. inspect an ONNX artifact and generate a STARK model declaration;
5. verify an artifact against a declaration before inference;
6. reject deliberately broken model calls with actionable, provenance-rich
   diagnostics.

Gate 4 is complete only when one representative computer-vision ONNX model can
be imported without a handwritten signature and the negative shape/dtype/device
cases fail before model execution.

## 2. Sources of truth and precedence

Read these files before changing code:

1. `STARKLANG/docs/ROADMAP.md`, especially Gate 4.
2. `STARKLANG/docs/PLAN.md`, especially section 4 and decisions T1–T12.
3. `STARKLANG/docs/extensions/Tensor-Model-Types.md` in full. Sections 3–9
   are normative for this gate.
4. `starkc/docs/gate1-exit.md`, `gate2-exit.md`, and `gate3-exit.md` to
   understand what must remain working.
5. Current compiler code under `starkc/src/` and tests under `starkc/tests/`.

Precedence is:

```text
current normative Core/extension specs
    > ROADMAP.md and PLAN.md
    > gate exit reports and current code/tests
    > root assistant-context files
    > archived documents and old prototypes
```

`AGENTS.md` and `CLAUDE.md` currently contain stale implementation-status
statements. They are context only, not proof that Gate 2 or Gate 3 is missing.
The current Rust compiler has completed Gates 1–3. Never extend the archived
Python prototype under `STARKLANG/compiler/`, and never copy archive-era
syntax such as lowercase `f32`, GC, actors, or `Package.stark` into Core.

## 3. Non-negotiable boundaries

- Preserve all Core-only behavior and all existing tests.
- Tensor support is an **optional extension**, never an unconditional Core
  feature. Core-only invocations must reject extension syntax with a message
  naming `tensor`.
- Keep extension-owned types and rules out of the Core type representation as
  far as practical. Use a registration/capability boundary, not scattered
  `if tensor` checks throughout the checker.
- Do not build tensor kernels, a custom VM, GPU execution, autodiff, training,
  quantization, named axes, layouts, distributed execution, or Gate 5 host
  code.
- Do not add ONNX Runtime execution in Gate 4. Gate 4 reads and verifies model
  metadata. Actual inference and native host generation belong to Gate 5.
- Do not broaden the language to solve hypothetical models. Implement one
  representative CV signature first, then only the static/dynamic dimension,
  multiple-input, and multiple-output behavior required by the normative
  extension.
- No unrelated IDE work. The terminal IDE may gain a tensor-extension toggle
  only after the CLI/compiler API is stable, and that toggle is not a Gate 4
  exit criterion.
- Do not edit `STARKLANG/docs/spec/STARK-Core-v1.*` directly. If a Core spec
  defect truly blocks the extension, stop and document it before changing the
  normative individual Core source files.
- Preserve unrelated dirty or untracked files. Stage only files belonging to
  the current milestone.
- No unsafe Rust. Keep strict Clippy clean.

## 4. Required architecture

The existing pipeline is:

```text
Source -> Tokens -> AST -> HIR -> semantic checks -> typed-HIR interpreter
```

Gate 4 extends the front end and semantic layers. It must not make the Core
interpreter responsible for tensor execution.

### 4.1 Extension configuration

Introduce one shared configuration passed through parse, resolve, and type
analysis rather than reading global process state:

```rust
pub struct LanguageOptions {
    pub extensions: ExtensionSet,
}

pub enum ExtensionId {
    Tensor,
}
```

Exact names may differ, but the properties must hold:

- default is Core-only;
- configuration is explicit and deterministic in unit tests;
- `starkc check --extension tensor file.stark` enables tensor parsing/checking;
- `starkc parse --extension tensor ...` supports extension syntax;
- duplicate or unknown extension ids produce a CLI usage error;
- `run` either rejects `--extension tensor` with a clear “frontend only in
  Gate 4” message or checks then refuses tensor execution. It must never
  pretend to execute tensors.

Prefer compatibility wrappers such as the existing `parse(...)` and
`typecheck::check(...)` defaulting to Core-only, with new option-aware entry
points. This minimizes churn in existing tests and consumers.

### 4.2 Extension-owned semantic representation

Do not add many tensor-specific variants directly to the Core `Ty` enum.
Introduce an extension type payload/registry boundary, for example:

```rust
Ty::Extension(ExtensionTyId)
```

with tensor definitions in a dedicated module. The exact design is open, but
Core equality, display, substitution, unification, and diagnostics must
delegate through one extension interface. A Core-only checker must initialize
and run without knowing tensor constructors.

Suggested module boundaries:

```text
starkc/src/extensions/mod.rs
starkc/src/extensions/tensor/mod.rs
starkc/src/extensions/tensor/dim.rs
starkc/src/extensions/tensor/types.rs
starkc/src/extensions/tensor/ops.rs
starkc/src/extensions/tensor/model.rs
starkc/src/onnx/mod.rs
starkc/src/onnx/import.rs
starkc/src/onnx/verify.rs
```

Use fewer files if that is clearer. Avoid a single oversized `tensor.rs` and
avoid distributing tensor rules across `typecheck.rs` match arms.

### 4.3 Provenance

Shape diagnostics are a product feature. Every dimension fact must be able to
retain or recover:

- its source span;
- its origin category: literal, generic parameter, refinement, imported model
  port, or operation result;
- a human-readable origin label;
- the constraint that introduced or compared it.

Do not bolt provenance onto diagnostic strings at the end. Design it into the
dimension/type representation before implementing operations.

## 5. Milestone plan

Complete milestones in order. Each milestone ends with focused tests, the
full quality suite, a small documentation/status update, and a separate commit.
Do not start the next milestone while the repository is red.

### M4.0 — Baseline, decisions, and fixtures

Goal: establish a reproducible baseline and eliminate architectural guessing.

Tasks:

1. Run the full existing validation suite and record the test count.
2. Inspect `ast.rs`, `parser.rs`, `hir.rs`, `resolve.rs`, `typecheck.rs`,
   `diag.rs`, and `main.rs`. A design note already exists at
   `starkc/docs/gate4-design.md` (untracked) describing the chosen extension
   boundary — review it against the current code and extend or correct it in
   place. Do not overwrite it from scratch, and do not commit unrelated
   untracked files alongside it.
3. Correct only the stale Gate 2/Gate 3 checkboxes/status statements in active
   roadmap-facing documentation, including the one-line status block in the
   root `CLAUDE.md` (which still reads "Gate 2 next" though Gates 1–3 are
   closed). Do not rewrite historical exit reports.
4. Select a representative CV model class, preferably a small ResNet-family
   model with a signature equivalent to:

   ```text
   input  Float32 [B, 3, 224, 224]
   output Float32 [B, 1000]
   ```

5. Do not commit a large production ONNX binary. Use two fixture tiers:
   - a tiny deterministic ONNX fixture committed under
     `starkc/tests/fixtures/onnx/` for automated tests;
   - a documented external representative-model URL plus SHA-256 and a helper
     script or command for the Gate 4 exit demonstration.
6. Evaluate maintained Rust ONNX decoding options. Prefer a narrowly scoped
   protobuf/ONNX dependency over a full inference engine. Record the choice,
   version, maintenance signal, transitive cost, and why rejected alternatives
   were not selected. Do not add ONNX Runtime merely to read metadata.
7. Create a Gate 4 fixture manifest covering parse-pass, Core-only rejection,
   semantic-pass, and semantic-error cases.

Exit:

- the baseline is green;
- architecture and dependency decisions are documented;
- the reference signature and fixture strategy are fixed;
- no tensor implementation has been mixed into Core yet.

Suggested commit: `Plan Gate 4 tensor extension architecture`

### M4.1 — Extension gate and syntax

Goal: represent all Gate 4 surface syntax without applying tensor semantics.

Tasks:

1. Add explicit extension configuration to library and CLI entry points.
2. Extend lexer/parser/AST for:
   - `Float16` and `BFloat16` only when `tensor` is enabled;
   - `Dim` and `DType` generic bounds;
   - shape generic arguments: `[B, 3, H * 2, 224]` and `[]`;
   - dimension expressions using integer literals, identifiers, `+`, `-`,
     `*`, and parentheses;
   - const index-list generic arguments;
   - named generic argument `device = D` without breaking associated-type
     bindings;
   - `model Name<...> { input ...; output ...; }` items.
3. Add AST and HIR nodes that preserve spans for each dimension expression,
   model port, and generic argument.
4. Update AST dumping so every new node has deterministic output.
5. Update resolution:
   - model names enter item namespaces;
   - model generic parameters are in scope for all ports;
   - port types and dimension identifiers resolve predictably;
   - model generics reject non-`Dim` parameters later in semantic checking.
6. When the extension is disabled, recognize enough of D1–D5 to produce a
   focused diagnostic such as “this syntax requires extension `tensor`” rather
   than a cascade of generic parse errors.
7. Even with the extension enabled, reject the reserved constructs of extension
   §11 with a focused diagnostic that names them as reserved in `tensor` v0.1:
   quantized dtypes (`QInt8<scale, zero_point>`), named axes
   (`[batch: B, ...]`), memory-layout types (`NCHW`, row/column-major), and
   range/inequality dim constraints (`B <= 64`). Conformance §12 requires this;
   a bare “unknown identifier” is not sufficient. If M4.6 narrows the claimed
   conformance instead of rejecting a given construct, that narrowing must be
   explicit in `gate4-exit.md`.

Required parser cases:

- static, symbolic, scalar, and device-qualified tensors;
- nested/parenthesized dim expressions;
- empty and trailing-comma shape lists;
- model with one/multiple inputs and outputs;
- malformed model port, malformed dim expression, invalid index-list syntax;
- every extension construct rejected in Core-only mode;
- each reserved §11 construct rejected as reserved even with the extension on;
- all existing 121 conformance fixtures unchanged.

Exit:

- extension syntax round-trips through AST/HIR dumping;
- Core-only and tensor-enabled modes are independently tested;
- no type rule is implemented by parser heuristics.

Suggested commit: `Add gated tensor extension syntax`

### M4.2 — Dimension algebra and tensor types

Goal: implement the semantic foundation independently of operations and ONNX.

Tasks:

1. Implement dimension expressions as a canonical **multivariate** polynomial
   normal form over integer coefficients and symbolic variables. Products of
   distinct variables (`B*H`) and higher-degree monomials must normalize — a
   linear-only normalizer is insufficient because `reshape`'s element-count
   equality (§6.4) compares products of every dim.
2. Normalization must prove identities such as:

   ```text
   B * (H + 1) == B*H + B
   2 * (N + 3) == 2*N + 6
   ```

3. Equality is normal-form equality only. Do not add a general solver or
   inequality/range reasoning.
4. Enforce non-negative dimensions exactly as specified:
   - negative literal results are rejected;
   - subtraction involving symbols is rejected unless non-negativity follows
     from literal constants alone;
   - overflow/representation limits yield diagnostics, never panics.
5. Add distinct semantic kinds for type, `Dim`, `DType`, and device parameters.
   Reject use of a dim in type position, mixed/trait bounds on dims, and
   undeclared dimension variables.
6. Add extension-owned representations for:
   - dtype including `Float16`/`BFloat16`;
   - `Tensor<T, Shape, Device>`;
   - `TensorDyn<T>` and `TensorAny`;
   - `Cpu` and `Cuda<N>`;
   - implicit fresh device variables when `device` is omitted.
7. Implement shape and device unification:
   - rank must be statically known and equal;
   - dimensions compare positionally by canonical equality/unification;
   - dtype never coerces implicitly;
   - devices must unify;
   - tensor values are `Move`, never `Copy`.
8. Define existential refinement facts and scopes. This is a purely **static**
   obligation in Gate 4: after `refine::<S>`, each unbound variable in `S`
   becomes a fresh existential symbolic dim for the rest of the enclosing
   block, and reuse of an already-bound variable asserts equality (spec §4.3).
   Do **not** build a runtime `refine` execution path — ONNX signature
   verification (M4.5) is metadata comparison and does not need one. Runtime
   refinement belongs to Gate 5.

Required unit tests:

- polynomial normalization/equality and non-equality;
- malformed and negative dimension expressions;
- symbolic unification, rank mismatch, dtype mismatch, and device mismatch;
- implicit device polymorphism;
- distinct existential symbols from separate refinements;
- same-symbol consistency within/reused after a refinement;
- tensors rejected as `Copy` and accepted as owned `Move` values;
- property tests or bounded generated tests for normalization stability,
  commutativity of equality, and “never panic” behavior.

Exit:

- tensor types can be checked in declarations and function signatures;
- dimension equality is deterministic and independently tested;
- diagnostics include both compared shapes and their origins.

Suggested commit: `Implement tensor dimensions and types`

### M4.3 — Table-driven tensor operation typing

Goal: type the v0.1 operation surface needed by the CV demonstration without
implementing numerical execution.

Implement operation constraints as data plus reusable shape transforms. Do not
write an unrelated bespoke method checker for every operation.

Required operations:

- construction: `zeros`, `ones`, `full`, `from_vec`;
- elementwise: add/sub/mul/div/min/max and comparisons;
- explicit `broadcast_to` plus the normative trailing-dimension broadcast rule;
- `matmul` and `batch_matmul`;
- `permute`, `reshape`, `concat`, `slice_axis`, and rank-2 `transpose`;
- `sum_axis`, `mean_axis`, `argmax`, `sum`, and `softmax`;
- `cast` and `to_device`.

Rules to test explicitly:

- broadcasting aligns from the trailing dimension;
- only literal `1` broadcasts; two unrelated symbols do not;
- matmul inner dimensions must unify;
- permutation is complete, unique, and in range;
- reshape requires polynomially equal element counts;
- concat agrees on every non-axis dimension;
- reduction axes are literal and in range;
- cast changes dtype only; transfers change device only;
- multi-operand device mismatch is a compile-time error;
- unprovable constraints are rejected, not deferred to runtime.

Diagnostics must name the operation, expected constraint, actual values, and
origins. Add a suggested fix only when mechanically justified (`permute`,
`cast`, `broadcast_to`, `refine`, or `to_device`). Never suggest a transform
that the checker has not proved would solve the mismatch.

Exit:

- the preprocessing chain in extension Appendix A type-checks through the
  model call boundary using declared signatures;
- one-line mutations for shape, dtype, and device fail with snapshot-tested
  diagnostics;
- no tensor operation reaches the Gate 3 interpreter.

Suggested commit: `Type-check tensor operations and diagnostics`

### M4.4 — Model declarations and generated interface

Goal: make model signatures nominal and callable.

Tasks:

1. Validate `model` declarations:
   - at least one input and output;
   - ports remain ordered;
   - port names are unique in the appropriate namespace;
   - every port type is `Tensor` or `TensorDyn` as allowed by the spec;
   - model generic parameters are all `Dim` parameters;
   - every symbolic dimension is declared by the model or validly bound.
2. Generate semantic signatures for nominal `load` and `predict` behavior
   without manufacturing source text inside the parser.
3. `predict` accepts one argument per input, preserves port order, unifies dim
   parameters independently per call, and returns either the sole output or a
   tuple of outputs in declaration order.
4. Model declarations are nominal even if structurally identical.
5. A model call mismatch must point both to the call-site tensor origin and the
   corresponding imported/declaration port.

Required tests:

- single and multiple input/output models;
- model generic batch dimension inferred independently at two call sites;
- zero-input/output, duplicate port, illegal port type, illegal generic kind;
- nominal model distinction;
- model call shape/dtype/device mismatch diagnostics.

Exit:

- hand-written model declarations are fully checked and callable in semantic
  analysis;
- the generated interface exactly follows extension section 7.2;
- no inference backend is required.

Suggested commit: `Add typed model declarations`

### M4.5 — ONNX signature import and verification

Goal: derive declarations from artifacts and detect signature drift before
execution.

CLI contract:

```text
starkc import model.onnx --out model.stark
starkc verify model.onnx --declaration model.stark [--model Name]
```

Equivalent well-documented names are acceptable, but import output must be
deterministic and verification must be scriptable with meaningful exit codes.

Importer requirements:

1. Decode ONNX protobuf safely with bounded/clear failure diagnostics.
2. Read graph inputs/outputs in declared order while excluding initializer-only
   graph inputs where required by ONNX semantics.
3. Map supported ONNX element types exactly to STARK dtypes. Unsupported types
   must produce a focused error, not a guessed mapping.
4. Map static dimensions to integer literals.
5. Map dynamic dimensions to fresh, stable `Dim` names. Reuse a name only when
   ONNX metadata establishes identity; sanitize collisions deterministically.
6. Reject missing-rank, invalid/negative dimensions, unsupported sparse/sequence/
   optional/map ports, and malformed artifacts with actionable messages.
7. Generate formatted, deterministic ordinary STARK source suitable for
   version control. Include a generated-file comment with tool version and
   artifact hash, but no timestamps or absolute paths.
8. Refuse to overwrite an existing output unless an explicit `--force` flag is
   supplied.

Verification requirements from extension section 7.3:

- port count, order, names, and element types match exactly;
- a declaration literal matches only the same artifact static dimension;
- a declaration variable matches only an artifact dynamic dimension;
- artifact-static under a declaration variable is a mismatch because the
  declaration over-promises flexibility;
- report all useful signature differences in one run where practical;
- return nonzero on mismatch or malformed input.

Fixture strategy:

- generate tiny ONNX files deterministically from checked-in test code or a
  fixture generator; do not hand-edit protobuf bytes;
- include static, symbolic batch, multiple-output, initializer, unsupported
  dtype, missing-shape, and corrupt-file cases;
- keep large reference artifacts outside Git and verify their checksum before
  use.

Security/robustness:

- never execute an ONNX graph while importing;
- impose reasonable file-size/decode limits or clearly document library limits;
- no path traversal from model metadata;
- fuzz or mutation-test the importer boundary if the chosen decoder permits it;
- malformed input must never panic.

Exit:

- the tiny fixture and one representative CV model generate correct declarations;
- generated declarations parse and type-check with `--extension tensor`;
- a golden generated declaration is stable byte-for-byte;
- every section 7.3 drift class fails verification before inference.

Suggested commit: `Import and verify ONNX model signatures`

### M4.6 — Gate integration, evidence, and closure

Goal: demonstrate the gate, harden regression coverage, and make completion
reproducible.

Tasks:

1. Add `starkc/examples/gate4/` containing:
   - imported declaration for the representative signature;
   - valid preprocessing/model-call program;
   - minimal broken variants for shape, dtype, and device;
   - artifact/declaration drift verification case.
2. Add `starkc/tests/gate4_tensor.rs` and `gate4_onnx.rs` integration tests.
3. Add diagnostic snapshots or exact assertions for the four headline defect
   classes. Ensure source spans point at the useful operand/port.
4. Add `starkc/docs/gate4-exit.md` mapping every roadmap deliverable and exit
   criterion to code, tests, and reproducible commands.
5. Update active README/roadmap status only after all criteria pass.
6. Document deviations from tensor v0.1. A deviation cannot silently be called
   complete; either implement it, explicitly narrow the claimed conformance,
   or obtain a spec decision.
7. Run the full validation matrix below from a clean checkout-equivalent state.

Exit demonstration should be no more complicated than:

```bash
cd starkc
cargo run -- import path/to/reference.onnx --out /tmp/reference.stark
cargo run -- check --extension tensor /tmp/reference.stark
cargo run -- check --extension tensor examples/gate4/valid_pipeline.stark
cargo run -- check --extension tensor examples/gate4/bad_shape.stark
cargo run -- verify path/to/reference.onnx --declaration /tmp/reference.stark
```

The valid commands must succeed; the deliberate negative command must fail
with the expected diagnostic.

Suggested commit: `Close Gate 4 tensor frontend`

## 6. Test and validation contract

Run focused tests continuously. At the end of every milestone run:

```bash
cd starkc
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo doc --no-deps
git diff --check
```

Also prove Core isolation explicitly:

```bash
cargo test --test conformance
cargo test --test gate2_valid
cargo test --test gate3_execution
cargo run -- check examples/gate3/01_hello.stark
cargo run -- run examples/gate3/01_hello.stark
```

Required test layers:

- unit tests for dimension algebra, type unification, ONNX mapping, and
  declaration formatting;
- parser/AST tests for every syntax delta;
- semantic positive/negative fixtures;
- exact or snapshot diagnostic tests;
- CLI integration tests for flags, import, verification, overwrite behavior,
  and exit status;
- robustness tests for malformed dimension syntax and corrupted ONNX data;
- regression tests proving Core behavior is identical with no extension.

Never weaken an existing test, remove a diagnostic assertion, or update a
snapshot merely to make it green. Explain why expected output changed.

## 7. Diagnostic quality examples

Bad:

```text
type mismatch
```

Minimum acceptable:

```text
error: matmul inner dimensions do not match
  required: K == K2
  left:  768, from `features` dimension 2
  right: 512, from `weights` dimension 1
```

For a model call:

```text
error: input `image` of model `ResNet50` has incompatible shape
  expected: [B, 3, 224, 224], from imported port `image`
  found:    [B, 224, 224, 3], from `images`
  help: permute NHWC input with `.permute::<[0, 3, 1, 2]>()`
```

For artifact drift:

```text
error: model declaration over-promises dynamic dimension
  declaration: `B`, from output `class` dimension 0
  artifact:    static dimension 1
  note: a declaration variable may match only an ONNX dynamic dimension
```

Use the repository's structured `Diagnostic` facilities and spans. These are
examples of information content, not mandatory byte-for-byte wording.

## 8. Git and collaboration protocol

Before every edit:

1. inspect `git status --short`;
2. identify pre-existing user changes;
3. avoid touching those files unless they are part of the approved milestone.

At each milestone, this repo works work-package-by-work-package: **present, do
not auto-commit.**

1. show the exact changed-file list;
2. run focused and full validation;
3. inspect `git diff --check` and the staged diff;
4. present the staged milestone diff and the suggested commit message, then
   **wait for explicit approval before committing.** Do not commit
   speculatively across milestones;
5. commit only the approved milestone files (never unrelated untracked work);
6. report the commit id, test count, and any documented deviation.

Do not use destructive Git commands. Do not reset, discard, clean, or silently
rewrite user work. Do not push unless explicitly asked.

If another agent modifies overlapping files midway, stop, inspect the combined
diff, reconcile intentionally, and rerun the complete suite. Never assume a
partially generated or conflict-marked file is valid.

## 9. Stop and escalate conditions

Stop implementation and present a concrete decision memo when any of these
occurs:

- the normative extension is internally contradictory;
- a required syntax change conflicts with Core grammar or existing valid code;
- an ONNX construct cannot be mapped without inventing semantics;
- the chosen dependency requires unsafe code in this crate or effectively
  imports a whole execution backend;
- a representative model has unknown rank, unsupported port kinds, or metadata
  ambiguity not addressed by section 7;
- tensor isolation would require a broad rewrite of the Core checker;
- a Gate 5 concern becomes necessary to claim Gate 4 completion;
- the same failure persists after three evidence-based attempts.

The memo must include: observed facts, minimal reproducer, relevant spec text,
options, trade-offs, and a recommended decision. Do not guess through a
language-design ambiguity.

## 10. Explicitly deferred work

Do not implement these during this assignment:

- tensor numerical storage or kernels;
- ONNX Runtime execution or generated Rust/C host glue;
- native executable generation for inference;
- image decoding, resize, normalization kernels, or dataset APIs;
- performance benchmarking against Python;
- semantic labels such as color space/range/frame;
- quantization, layouts, named axes, training, or autodiff;
- debugger, LSP, package registry, cloud tooling, or more IDE features.

These belong to Gate 5, Gate 6 follow-up experiments, or later ecosystem work.

## 11. Definition of done

Gate 4 is done when all statements below are true:

- [ ] Core-only mode remains the default and rejects D1–D5 by naming the
      required `tensor` extension.
- [ ] Reserved §11 constructs are rejected as reserved even with the extension
      enabled (or the conformance claim is explicitly narrowed in
      `gate4-exit.md`).
- [ ] Tensor syntax parses and lowers with source spans intact.
- [ ] `Dim`, `DType`, shape, dtype, and device rules are statically checked.
- [ ] Polynomial dimension equality and symbolic unification are tested.
- [ ] Required tensor operations type-check without numerical execution.
- [ ] `model` declarations are nominal, validated, and callable.
- [ ] A deterministic importer generates a STARK declaration from ONNX.
- [ ] Static and dynamic ONNX dimensions map according to section 7.4.
- [ ] Artifact/declaration verification enforces every section 7.3 rule.
- [ ] One representative CV model imports without a handwritten signature.
- [ ] Incorrect shape, dtype, and device calls fail before execution with
      actionable diagnostics and correct spans.
- [ ] Artifact signature drift fails at verification/load time.
- [ ] All existing Core, Gate 2, Gate 3, and IDE tests still pass.
- [ ] Formatting, strict Clippy, tests, release build, docs, and diff checks
      pass.
- [ ] `gate4-exit.md` provides reproducible evidence for every roadmap exit
      criterion.
- [ ] No Gate 5 or deferred feature was smuggled into the implementation.

## 12. Starter prompt for Gemini

Use this prompt together with this file:

> Implement Gate 4 exactly according to
> `starkc/docs/GEMINI_GATE4_IMPLEMENTATION.md`. Begin with M4.0 only. Read all
> sources of truth listed in section 2, inspect the live repository and dirty
> worktree, run the baseline validation, then present the M4.0 design and file
> plan before editing implementation code. Work milestone-by-milestone, keep
> the tensor extension isolated from Core, preserve unrelated changes, and do
> not enter Gate 5. After each milestone, run the full validation contract and
> commit only the milestone files. If a stop condition is reached, provide the
> requested decision memo instead of inventing semantics.

