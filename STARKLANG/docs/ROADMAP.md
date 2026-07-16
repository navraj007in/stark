# STARK Implementation Roadmap

**Status:** active planning document  
**Current objective:** prove that STARK can catch deployment-pipeline defects
before execution while producing a practical native inference program.

This roadmap defines the evidence required to advance the project. It does not
assign calendar dates: later work begins when the preceding gate is satisfied,
not when an estimate expires.

## 1. Product Wedge

STARK's first validation target is a narrow computer-vision inference pipeline:

```text
ONNX artifact
    -> generated, typed model signature
    -> typed preprocessing and postprocessing
    -> compile-time shape/dtype/device checks
    -> native executable using an existing backend
```

The prototype should demonstrate errors that ordinary deployment stacks often
discover only at runtime. It is not intended to prove every proposed STARK use
case or to establish a new tensor runtime.

## 2. Design Boundaries

The roadmap is guided by five concerns: **shape**, **meaning**, **location**,
**ownership**, and **constraints**. These are a review lens, not a requirement
that one universal type expose five independent parameters.

For the current wedge:

- Core v1 supplies ownership and general language safety.
- The `tensor` extension supplies shape, dtype, device, and model-signature
  checking.
- Semantic meaning such as color space, value range, and coordinate frame may
  be explored after the base tensor pipeline works.
- New Core features require evidence that the prototype cannot be implemented
  cleanly without them.

`Tensor` remains the primary multidimensional type. Domain types such as
`Image`, `Signal`, and `PointCloud` should begin as library-level types. A more
general storage abstraction is considered only after several implemented
domains independently require the same model.

## 3. Delivery Sequence and Gates

### Gate 1 — Core front end

Build a lexer and parser for the normative Core v1 grammar.

Deliverables:

- source locations and useful syntax diagnostics;
- an AST suitable for semantic analysis;
- automated parser tests derived from the 121 fixtures in
  `STARKLANG/tests/spec-fixtures/`;
- fixtures classified as expected-pass, expected-semantic-error, or
  intentionally illustrative where necessary.

Exit criteria:

- every classified parsing fixture has a deterministic expected result;
- valid Core examples parse without implementation-specific grammar changes;
- discovered specification ambiguities are resolved in the normative source,
  with generated combined documents kept in sync.

### Gate 2 — Core semantic checker

Implement name resolution, type checking, and ownership/borrow analysis before
adding tensor-specific semantics.

Deliverables:

- module and lexical-scope name resolution;
- type checking and local inference for the Core v1 surface;
- move and borrow checking;
- structured diagnostics with source spans;
- positive and negative conformance tests.

Exit criteria:

- representative valid Core programs pass semantic analysis;
- invalid programs demonstrate the safety rules claimed by Core v1;
- no tensor-only rule is embedded in the Core checker.

### Gate 3 — Minimal execution path

Execute enough Core STARK to support the inference prototype. An interpreter,
bytecode engine, or direct lowering path is acceptable; building a novel VM is
not a prerequisite.

Deliverables:

- functions, control flow, structs/enums, errors, and required collection and
  I/O operations;
- the `core-min` standard-library profile needed by the prototype;
- repeatable build and test commands.

Exit criteria:

- small Core programs compile and execute end to end;
- execution semantics agree with the Core v1 conformance summary;
- the selected path can host or call the model backend without a new runtime.

### Gate 4 — Tensor front end and ONNX import

**Status: complete (2026-07-15).** Reproducible evidence is recorded in
`starkc/docs/gate4-exit.md`.

Implement the narrowest slice of the `tensor` v0.1 extension required to type
an imported computer-vision model.

Deliverables:

- `Dim`, `DType`, tensor shapes, devices, and `model` signatures;
- ONNX signature inspection and generated STARK declarations;
- load-time artifact/signature verification;
- compile-time diagnostics that trace shape, dtype, and device mismatches to
  their origins.

Exit criteria:

- at least one representative ONNX CV model imports without a handwritten
  signature;
- static and dynamic ONNX dimensions map according to the extension spec;
- intentionally incorrect calls fail before model execution with actionable
  diagnostics.

### Gate 5 — Go/no-go deployment prototype

**Status: complete (2026-07-16).** Reproducible evidence is recorded in
`starkc/docs/gate5-exit.md`.

Build one realistic inference program covering input refinement,
preprocessing, model invocation, and postprocessing.

The evaluation corpus must include both a valid pipeline and deliberately
broken variants covering:

- incompatible tensor dimensions;
- incorrect element type;
- incompatible device placement;
- artifact/declaration signature drift.

The program must produce a deployable native artifact through an existing
backend such as IREE, ONNX Runtime with generated host glue, or generated
Rust/C.

Exit criteria:

- correct output agrees with the reference implementation within a documented
  numerical tolerance;
- all four defect classes above are caught at compile time or, for artifact
  drift, during fail-fast loading before inference;
- build steps, artifact size, startup time, peak memory, and steady-state
  latency are measured reproducibly;
- backend integration does not require STARK to implement tensor kernels or a
  new ML runtime;
- the prototype and its diagnostics are small enough to explain in a focused
  technical demonstration.

### Gate 6 — Decision checkpoint

**Status: decision recorded — REVISE (2026-07-16).** Evidence and rationale are
in `starkc/docs/gate6-memo.md` (measured comparators: `baseline_defects.py`,
`rust-comparator/`; consolidated in `starkc/tests/results/gate6/results_summary.md`).
Measured outcome — defects caught before inference: STARK 5/5, Python/ORT 2/5,
strongest generated typed-Rust host 5/5 *on the Gate 5 CV-preprocessing
pipeline*. Against the operational (Python) baseline STARK's advantage is
decisive; against the strongest typed-Rust comparator it reaches only parity on
this pipeline — parity that cost the Rust host a hand-rolled re-implementation of
STARK's tensor/device typing and holds only because this pipeline uses no shape
arithmetic (stable Rust hits the `generic_const_exprs` wall on reshape/matmul/
conv/broadcast). The differentiator — general shape arithmetic and semantic CV
annotations — is not exercised by the current demonstrator, so REVISE (not GO,
not STOP): re-scope the next experiment to isolate it. That experiment requires
its own §5 proposal before any work.

After Gate 5, record a go, revise, or stop decision based on evidence.

Proceed when the prototype shows a material safety or deployment advantage
that is not erased by integration complexity. If successful, the next narrow
experiment is semantic CV annotations—initially color space, value range, and
coordinate frame—with explicit propagation and conversion rules.

Revise or stop when the same guarantees are more simply obtained through a
library, schema generator, or existing compiler; when backend boundaries make
the type guarantees misleading; or when implementation cost outweighs the
demonstrated deployment benefit.

Performance is recorded at this checkpoint, but no fixed speedup is a gate
until a controlled baseline and workload exist. STARK must not claim a
performance multiple solely from eliminating Python orchestration.

### Gate 7 — Symbolic Shape and Semantic Tensor Deployment

**Status: proposed (authorised by the Gate 6 `REVISE` decision).** Bounded scope,
non-goals, defect corpus, comparator requirements, measurements, and exit/stop
criteria are defined in
`STARKLANG/docs/proposals/GATE7_SYMBOLIC_SEMANTIC_DEPLOYMENT.md`. This is the
narrow experiment the Gate 6 decision called for: one realistic detector-head
pipeline that requires *computed* tensor dimensions (which the Gate 5 pipeline
lacked) plus exactly one semantic tensor property (image value range), deployed
natively through the existing ONNX Runtime host and compared against a **real,
end-to-end** typed-Rust host (correcting the Gate 6 stubbed comparator). It tests
whether STARK provides a practical language-level advantage when a pipeline
carries computed shape relationships and semantic state; it does not presume the
answer. Implementation of each work package (G7-01…G7-07) begins only after the
proposal is reviewed for scope. No VM, LSP, or broad language work is authorised.

## 4. Explicit Non-Goals Before the Decision Checkpoint

The following work is outside the critical path:

- training, automatic differentiation, and optimizer APIs;
- a custom tensor kernel language, VM, GPU stack, or graph compiler;
- a universal `Buffer<...>` abstraction;
- full physical-unit algebra and derived-unit inference;
- worst-case execution-time or compile-time deadline guarantees;
- audio-specific dimension arithmetic such as integer division;
- robotics middleware, ROS integration, and real-time robotics runtimes;
- cloud deployment syntax, package registry infrastructure, and distributed
  execution;
- broad positioning beyond the proven CV deployment wedge.

These items are not rejected permanently. Each requires a separate proposal
backed by a concrete use case after the go/no-go prototype.

## 5. Roadmap Governance

- The normative Core and extension specifications define language behavior;
  this document defines implementation order only.
- A completed checkbox or implementation milestone does not by itself expand
  project scope.
- Any proposed Core change must identify the blocking prototype requirement,
  alternatives considered, and compatibility impact.
- Deferred work moves onto the active roadmap only with an owner, a bounded
  experiment, and measurable exit criteria.
- Status changes should update this document and the compact summary in the
  repository README in the same change.
