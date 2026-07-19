# Gemini execution plan — complete STARK Gate 4 (M4.3–M4.6)

This is the implementation contract for the remaining Gate 4 work. Implement
the milestones in order and do not begin Gate 5 runtime execution work.

The normative language source is
`STARKLANG/docs/extensions/Tensor-Model-Types.md`. Existing repository behavior,
`starkc/docs/gate4-design.md`, and passing tests are compatibility constraints.
If this plan and the normative extension disagree, stop and report the conflict
instead of silently choosing new semantics.

## 1. Required repository discipline

Before editing:

1. Read `AGENTS.md`, the normative tensor extension, `starkc/PLAN.md` if
   present, `starkc/docs/gate4-design.md`, and the live M4.1/M4.2 code.
2. Run `git status --short` and preserve every pre-existing user change.
3. Establish the baseline with the focused Gate 4 tests and the full suite.
   There are currently 187 tests; report the observed count rather than
   hard-coding it in completion evidence.
4. Do not add an ONNX dependency during M4.3 or M4.4.

Work milestone by milestone. At the end of each milestone:

- show the exact changed-file list;
- run focused tests and the full validation matrix;
- run `git diff --check`;
- inspect the complete diff for conflict markers, generated debris, accidental
  edits, and weakened assertions;
- present the result and suggested commit message, but do not commit until the
  user explicitly requests it.

Never reset, clean, discard, or overwrite unrelated work. If another agent
changes an overlapping file, stop, inspect the combined diff, reconcile it
deliberately, and rerun validation.

## 2. Architectural decisions

### 2.1 Extension-owned semantic types

Generalize the current tensor-only extension wrapper without representing a
model as `Ty::Struct`:

```rust
pub enum Ty {
    // Core variants remain unchanged.
    Extension(Box<ExtensionTy>),
    // ...
}

pub enum ExtensionTy {
    Tensor(TensorKind),
    Model(ModelTy),
    ModelError,
}
```

`ModelTy` must carry nominal identity, normally the model declaration's
`ItemId`. Struct construction, field access, struct equality, and ordinary
struct trait behavior must never apply to it. Core unification, formatting,
substitution, ownership classification, and exhaustive matches delegate
extension values to extension helpers.

Model dimension parameters describe each `predict` signature. They are
freshened per call; they must not turn the loaded model receiver into a
permanently shape-instantiated structural type.

### 2.2 ONNX decoder

M4.5 uses a small safe-Rust Protobuf wire reader. Do not use `onnx-pb`, a
build-time `protoc`, ONNX Runtime, `tract`, generated host code, `unsafe`, or an
external command to decode a model.

The reader only needs the metadata path used by Gate 4:

```text
ModelProto.graph
  GraphProto.input / output / initializer
    ValueInfoProto.name / type
      TypeProto.tensor_type
        elem_type / shape
          TensorShapeProto.dim
            dim_value / dim_param
```

Unknown fields must be skipped according to their wire type. Support skipping
wire types 0, 1, 2, and 5; reject group wire types and malformed tags with an
actionable error. All cursor and length arithmetic must be checked.

### 2.3 Hashing

The generated-file comment includes the lowercase SHA-256 of the exact ONNX
bytes. Add a maintained, pure-Rust SHA-256 crate in M4.5, with default features
disabled when practical. Do not hand-write SHA-256 and do not invoke
`shasum`, `sha256sum`, or another platform program from `starkc`.

## 3. M4.3 — table-driven tensor operation typing

### Goal

Type the complete normative v0.1 tensor operation surface without numerical
storage, kernels, bytecode, or interpreter execution.

### 3.1 Resolution and call forms

All operations live in the extension library's `tensor` module and become
available through the supported `use tensor::*`/prelude behavior when the
`tensor` extension is enabled. Core-only mode must not acquire these names.

Add compiler builtin resolution for every supported standalone function, not
only constructors. Method-call sugar may share the same descriptor, but it
must not be the sole supported call form where the normative API defines a
function.

Required operations:

- construction: `zeros`, `ones`, `full`, `from_vec`;
- elementwise: `add`, `sub`, `mul`, `div`, `min`, `max`, and comparison
  operations producing `Tensor<Bool, ...>`;
- explicit broadcasting: `broadcast_to`;
- matrix: `matmul`, `batch_matmul`;
- shape: `permute`, `reshape`, `concat`, `slice_axis`, rank-2 `transpose`;
- reduction: `sum_axis`, `mean_axis`, `argmax`, `sum`, `softmax`;
- conversion/placement: `cast`, `to_device`.

Use the comparison names already established by the grammar/library design.
If the normative document does not uniquely name them, stop with a decision
memo instead of inventing public API names.

### 3.2 Descriptor design

Create one descriptor table and reusable rule handlers. The exact Rust names
may differ, but each descriptor must carry equivalent information:

```rust
struct TensorOpDescriptor {
    name: &'static str,
    call_forms: CallForms,
    receiver: ReceiverMode,
    arity: Arity,
    generics: &'static [GenericSlot],
    operands: &'static [OperandMode],
    dtype_rule: DTypeRule,
    device_rule: DeviceRule,
    shape_rule: ShapeRule,
    result_rule: ResultRule,
    suggested_fixes: &'static [SuggestedFixKind],
}
```

The descriptor selects reusable validation/transform functions. It is
acceptable for complex transforms such as reshape or concat to have dedicated
helpers; it is not acceptable to build an unrelated method-checking branch for
every operation.

The descriptor path must validate receiver/argument arity, generic argument
count and kind, borrowing mode, dtype constraints, device constraints, shape
constraints, and result construction. Invalid calls must not fall through to
generic “unknown method” diagnostics.

### 3.3 Required rules

- Binary elementwise broadcasting aligns trailing dimensions. A position is
  valid only when dimensions are provably equal or one side is literal `1`.
  Two unrelated symbolic dimensions do not broadcast.
- Elementwise arithmetic preserves dtype. Comparisons return `Bool`. No
  implicit dtype coercion is allowed.
- Multi-operand operations require devices to unify.
- `matmul` is rank 2 and requires equal inner dimensions.
- `batch_matmul` is rank 3 and requires equal batch and inner dimensions.
- `permute` must contain each index `0..rank` exactly once. Diagnose missing,
  duplicate, and out-of-range indices.
- `reshape` requires polynomial equality of source and destination element
  counts. An unprovable equality is an error.
- `concat` requires a literal in-range axis, equal rank, and equality of every
  non-axis dimension; its axis result is the normalized sum.
- `slice_axis` enforces the exact §6.4 constraint, checks the axis, and replaces
  that dimension with `LEN`.
- `transpose` accepts rank 2 only and swaps dimensions 0 and 1.
- Reduction axes are literal and in range. `sum_axis`/`mean_axis` remove the
  axis, `argmax` removes it and returns `Int64`, full `sum` returns shape `[]`,
  and `softmax` preserves shape.
- `cast` changes dtype only. Gate 4 does not implement or test runtime
  saturation behavior.
- `to_device` changes device only.
- `zeros`, `ones`, and `full` obtain dtype/shape from generic arguments or
  expected-type inference. `full` checks the scalar value dtype.
- `from_vec` follows its normative `Result<Tensor<...>, ShapeError>` signature;
  do not claim compile-time proof of a runtime vector length.
- Any constraint that is false or cannot be proven is rejected statically.

Suggested fixes are emitted only after proving the suggested transformation
would satisfy the failed constraint. Allowed fix families are `permute`,
`cast`, `broadcast_to`, `refine`, and `to_device`.

### 3.4 Diagnostics and tests

Each diagnostic names the operation, expected constraint, actual values, and
both available provenance origins. Add focused tests for every operation and
at least these failures:

- trailing broadcast success and failure;
- literal-1 versus symbolic broadcasting;
- matmul and batch-matmul inner mismatch;
- incomplete, duplicate, and out-of-range permutations;
- reshape unequal and unprovable element counts;
- concat rank, axis, and non-axis mismatch;
- slice/reduction invalid axes;
- dtype and device mismatch;
- wrong arity and wrong generic kind;
- a mechanically valid suggestion and a case where no suggestion is safe.

Prove that no tensor operation reaches the Gate 3 interpreter.

Suggested commit: `Type-check tensor operations and diagnostics`

## 5. M4.4 — nominal model declarations and calls

### Goal

Make hand-written model declarations validated, nominal, loadable at the type
level, and callable at the type level without adding an inference backend.

### 4.1 Validation

For every `hir::ItemKind::Model`:

- require at least one input and one output;
- preserve declaration order independently for inputs and outputs;
- reject duplicate port names;
- allow only `Tensor` and `TensorDyn` port types;
- require every model generic to have kind `Dim`;
- reject undeclared dimension identifiers and invalid bindings;
- retain declaration and individual port spans for later diagnostics.

Structurally identical model declarations remain distinct nominal types.

### 4.2 Generated semantic interface

Add `Res::ModelLoad(ItemId)` for `ModelName::load`. It accepts one borrowed
string path and returns `Result<ModelType, ModelError>`. This is a semantic
signature only in Gate 4; do not add interpreter execution for `load`.

Type `.predict(...)` on `ExtensionTy::Model`:

- require exactly one borrowed tensor argument per input port in input order;
- freshen all model dimension and device variables for every call;
- unify each input against its corresponding port;
- return the single output directly or a tuple in output declaration order;
- preserve tensor ownership by borrowing the receiver and inputs;
- point mismatches at both the call operand and declaration port origin.

Update all `Res` and `Ty` consumers exhaustively, including resolver, type
checker, borrow checker, flow analysis, interpreter rejection paths, AST/HIR
dumping, and display/substitution helpers.

### 4.3 Tests

Cover:

- one and multiple inputs/outputs;
- zero inputs and zero outputs;
- duplicate port names;
- illegal port type and illegal generic kind;
- undeclared dimensions;
- model nominal inequality;
- `ModelName::load` success and incorrect argument calls;
- `predict` arity, shape, dtype, and device failures;
- two calls to one model with independently inferred batch dimensions;
- receiver/input borrowing and later reuse;
- output tuple order and declaration/call-site diagnostic spans.

Suggested commit: `Add typed nominal model declarations`

## 6. M4.5 — deterministic ONNX signature import and verification

### Goal

Read ONNX metadata without executing the graph, generate reviewable STARK
declarations, and detect all §7.3 signature drift.

### 5.1 CLI contract

```text
starkc import <model.onnx> --out <model.stark> [--force]
starkc verify <model.onnx> --declaration <models.stark> [--model <Name>]
```

These commands enable tensor declaration parsing internally; users do not add
`--extension tensor`. Unknown, duplicated, or missing flags are usage errors
with exit code 2. I/O, malformed artifact, malformed declaration, and signature
mismatch failures are nonzero and distinguishable in diagnostics. Success is
zero and script-friendly.

`verify` selects `--model Name` when supplied. Without it, exactly one model
declaration must exist; otherwise report the available names and require
selection.

### 5.2 Decoder limits and robustness

Use named constants and test their boundaries:

- maximum ONNX file size: 256 MiB;
- maximum individual length-delimited field: 256 MiB and never beyond the
  enclosing message;
- maximum message nesting depth: 32;
- maximum graph inputs: 4096;
- maximum graph outputs: 4096;
- maximum initializers: 1,000,000;
- maximum tensor rank: 64;
- maximum UTF-8 name length: 4096 bytes.

Read the bounded file as bytes, not UTF-8. Never allocate from an unchecked
wire length. Reject truncated varints, varints longer than ten bytes, integer
overflow, invalid field zero, invalid UTF-8 metadata names, negative static
dimensions, missing tensor type, and missing rank. Zero static dimensions are
valid because STARK dimensions are non-negative.

Reject sparse, sequence, optional, map, and other unsupported port types with
a focused error. Malformed input must never panic. Add mutation/property tests
over small byte sequences if practical.

Preserve graph input/output order. Remove a graph input only when its exact
name is present in `GraphProto.initializer`. Reject empty or duplicate external
port names after filtering.

### 5.3 ONNX-to-STARK mapping

Maintain an explicit element-type mapping table. Map only supported STARK
dtypes and reject every other ONNX element type; never guess or coerce.

Static non-negative dimensions become integer literals. Dynamic dimensions:

- reuse a sanitized symbol only when a non-empty ONNX `dim_param` establishes
  identity;
- anonymous dynamic dimensions receive fresh stable names by first occurrence;
- sanitize deterministically and resolve keyword/collision conflicts using
  `_2`, `_3`, ... suffixes;
- preserve identity across inputs and outputs when metadata proves it.

For verification, symbol spelling itself is not significant, but equality
relationships are. A repeated declaration dimension must map to one proven
artifact dynamic identity. If anonymous metadata cannot prove the repetition,
report an unprovable signature match. Conversely, reject a declaration that
uses independent variables where the artifact explicitly ties the dimensions,
because it over-promises flexibility.

### 5.4 Deterministic source generation

Derive the model name from the input filename stem by splitting non-identifier
characters, converting to UpperCamelCase, prefixing `Model` if the result is
empty or starts with a digit, and suffixing `_2`, `_3`, ... for collisions.
Use one shared identifier sanitizer for model, port, and dimension identifiers,
with STARK keyword avoidance.

Generated output must:

- contain ordinary formatted STARK model source;
- include a generated-file comment with the `starkc` version and lowercase
  SHA-256 artifact hash;
- contain no timestamp, absolute path, random identifier, or platform-specific
  newline;
- end with exactly one newline;
- be byte-for-byte stable for identical input and compiler version;
- refuse an existing output unless `--force` is present;
- use an atomic same-directory temporary file plus rename so a failed write
  cannot corrupt the previous declaration.

Never use a model-provided name as an output path.

### 5.5 Verification semantics

Parse and typecheck the selected declaration through the normal tensor-enabled
front end. Accumulate useful differences rather than returning after the first:

- input/output counts;
- order and exact original port names;
- element dtype;
- rank;
- each static dimension value;
- static-versus-dynamic classification;
- dynamic identity constraints described above.

A declaration literal matches only the same artifact static value. A
declaration variable matches only an artifact dynamic dimension. Artifact
static under a declaration variable is always drift.

The generated hash is provenance information, not a substitute for signature
comparison. Verification must compare the live artifact metadata.

### 5.6 Fixtures and tests

Generate tiny valid Protobuf fixtures deterministically in checked-in Rust test
code or a dedicated fixture generator—never by manually editing opaque bytes.
Cover:

- static dimensions;
- named and anonymous dynamic dimensions;
- repeated dynamic identity;
- multiple inputs and outputs;
- initializer-only input filtering;
- zero dimension;
- unsupported dtype and unsupported port kind;
- missing type, missing shape, negative dimension;
- duplicate/empty ports;
- truncated and corrupt wire data;
- every decoder limit boundary;
- deterministic golden source;
- parse/typecheck of generated source;
- overwrite refusal and `--force`;
- single-model selection, explicit model selection, and ambiguous selection;
- all §7.3 drift classes and accumulated reporting;
- stable CLI exit statuses.

Keep the large ResNet reference outside Git. Its checksum-verified import is an
opt-in exit demonstration, not a network-dependent requirement of the
hermetic automated suite.

Suggested commit: `Import and verify ONNX model signatures`

## 6. M4.6 — integration and Gate 4 closure

Add `starkc/examples/gate4/` containing:

- the representative imported declaration;
- a valid preprocessing and model-call program;
- minimal shape, dtype, and device failures;
- an artifact/declaration drift example.

Add integration tests in `starkc/tests/gate4_tensor.rs` and
`starkc/tests/gate4_onnx.rs`, plus exact/snapshot diagnostic assertions where
appropriate. CLI tests must cover import, verify, flags, overwrite behavior,
model selection, output text, and process exit status.

Create `starkc/docs/gate4-exit.md` mapping every normative deliverable to code,
tests, and reproducible commands. Record deviations honestly. Update active
README/roadmap status only after every exit criterion passes.

The exit report must demonstrate:

```bash
cd starkc
cargo run -- import path/to/reference.onnx --out /tmp/reference.stark
cargo run -- check --extension tensor /tmp/reference.stark
cargo run -- check --extension tensor examples/gate4/valid_pipeline.stark
cargo run -- check --extension tensor examples/gate4/bad_shape.stark
cargo run -- verify path/to/reference.onnx --declaration /tmp/reference.stark
```

It must also prove Core isolation with the existing Core, Gate 2, Gate 3, and
IDE tests. Tensor builtins must not appear in Core-only mode, and no Gate 3
runtime behavior may regress.

Suggested commit: `Close Gate 4 tensor and ONNX integration`

## 7. Full validation matrix

Run from `starkc/` after every milestone and again from a clean-checkout-
equivalent worktree at Gate closure:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo doc --no-deps
git diff --check
```

Also run focused Core isolation commands documented in the existing Gate 4
brief. Do not weaken tests, remove assertions, or update snapshots merely to
make validation green. Explain every intentional expected-output change.

## 8. Explicitly deferred to Gate 5 or later

Do not implement:

- tensor numerical storage or kernels;
- runtime `refine` checks;
- ONNX graph execution or ONNX Runtime integration;
- interpreter execution of tensor operations, `Model::load`, or `predict`;
- generated Rust/C inference glue or native inference binaries;
- image decoding, resize, normalization, datasets, training, or autodiff;
- benchmarking claims;
- layout/named-axis/quantization semantics;
- LSP, debugger, registry, cloud, or additional IDE work.

## 9. Stop conditions

Stop and provide a decision memo instead of guessing if:

- normative operation names or semantics are ambiguous;
- Core grammar or existing valid Core behavior would need to change;
- model generic or dynamic-dimension identity rules conflict with the spec;
- an ONNX construct cannot be mapped without inventing semantics;
- safe extension isolation requires a broad Core checker rewrite;
- a dependency introduces `protoc`, native runtime linkage, `unsafe` in this
  crate, or a graph execution backend;
- the representative model contains unsupported/ambiguous port metadata;
- Gate 5 execution work becomes necessary to claim Gate 4 completion;
- the same failure persists after three evidence-based attempts.

The memo must include observed facts, a minimal reproducer, relevant normative
text, available options, trade-offs, and a recommendation.

## 10. Definition of done

Gate 4 is complete only when:

- the full normative tensor operation surface is statically typed through the
  unified descriptor system;
- actionable shape, dtype, and device errors carry useful origins;
- model declarations are validated, nominal, and callable with fresh per-call
  dimensions;
- ONNX import is safe, bounded, deterministic, and hermetically tested;
- verification enforces every §7.3 drift rule and reports useful differences;
- generated declarations parse and typecheck in tensor mode;
- one checksum-verified representative CV model imports successfully;
- Core-only, Gate 2, Gate 3, and IDE behavior remains green;
- formatting, strict Clippy, all tests, release build, docs, and diff checks
  pass;
- `gate4-exit.md` provides reproducible evidence and lists every deviation;
- no deferred Gate 5 feature has been included.
