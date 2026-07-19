# Claude execution brief — STARK Gate 5 go/no-go deployment prototype

**Audience:** Claude Code implementing this repository milestone  
**Status:** implementation-ready plan; Gate 4 is complete  
**Objective:** produce and measure one native ResNet50 inference deployment
without turning STARK into a tensor runtime or general-purpose transpiler

---

## 0. Read this first

Work milestone by milestone. Do not implement all of Gate 5 in one unreviewed
change. At the start of every milestone:

1. read `AGENTS.md` and inspect `git status --short`;
2. preserve every pre-existing user change and untracked file;
3. run the baseline tests relevant to the files you will touch;
4. state the exact files you intend to change;
5. stop if a condition in section 15 is reached.

After each milestone, show the changed-file list and validation evidence. Do
not commit until explicitly asked. Use the suggested commit boundaries in
section 14, and never stage unrelated dirty files.

### Critical correction to the older Gate 5 proposal

Do **not** reimplement any of the following; Gate 4 already provides them:

- nominal `model` HIR/types;
- `Model::load` resolution and static type checking;
- `model.predict(...)` input/output typing;
- fresh per-call model dimensions;
- ONNX signature import and verification;
- shape, dtype, device, refinement, and tensor-operation type checking.

The older pasted proposal also suggested a general STARK-to-Rust transpiler.
That is too broad. Gate 5 requires a **bounded deployment IR** for one proven
inference pipeline. Unsupported reachable HIR must receive a focused codegen
diagnostic; it must never be silently ignored or guessed.

---

## 1. Sources of truth, in priority order

Read these completely before M5.0 work:

1. `STARKLANG/docs/ROADMAP.md`, Gate 5 and Gate 6;
2. `STARKLANG/docs/PLAN.md`, especially T1, T3, T10–T12 and Gate 5;
3. `STARKLANG/docs/extensions/Tensor-Model-Types.md`, especially sections
   4.3, 6, 7, 8, 10, Appendix A;
4. `starkc/docs/gate4-exit.md`;
5. `starkc/src/typecheck.rs`, `hir.rs`, `resolve.rs`, `interp.rs`;
6. `starkc/src/onnx/` and `starkc/tests/gate4_onnx.rs`;
7. `starkc/examples/gate4/` and `starkc/tests/gate4_tensor.rs`;
8. `starkc/tests/fixtures/onnx/fetch-reference.sh`;
9. `starkc/scripts/build-release.py` for packaging conventions only.

Backend APIs are time-sensitive. At M5.0, verify the current official
documentation and record exact versions in `docs/gate5-backend-decision.md`.
As of this plan, the latest documented `ort` release is `2.0.0-rc.12`, it can
target ONNX Runtime API versions 1.17–1.24, and its recent releases require
Rust 1.88. Treat this as a starting hypothesis, not permission to skip the
spike.

Primary references:

- <https://github.com/pykeio/ort>
- <https://docs.rs/ort/latest/ort/>
- <https://onnxruntime.ai/docs/>
- <https://onnxruntime.ai/docs/performance/tune-performance/>
- <https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html>

---

## 2. Gate 5 definition of done

Gate 5 closes only when all of these are true:

- a real `resnet50-v1-7.onnx` artifact travels through import, verification,
  checked STARK preprocessing/postprocessing, host generation, native build,
  and CPU inference;
- the generated binary accepts an ordinary image, performs documented
  ImageNet preprocessing, runs ONNX Runtime, and emits deterministic JSON
  containing top class/probability and timing data;
- output agrees with a separately implemented Python ONNX Runtime reference
  within a tolerance fixed before results are inspected;
- shape, dtype, and device defects fail at STARK compile time with existing
  origin-rich diagnostics;
- artifact drift fails before inference—during generation and again at
  runtime loading when an external model is supplied;
- executable size, model/bundle size, startup time, peak RSS, warm inference
  latency, and steady-state latency are measured reproducibly;
- the backend integration uses an existing runtime and contains no custom
  convolution or model kernels;
- the normal compiler/IDE remains free of an ONNX Runtime dependency;
- Core-only, Gates 2–4, IDE, formatting, strict Clippy, release build, docs,
  and diff checks remain green;
- `docs/gate5-exit.md` records commands, environment, results, deviations,
  and a link to a prepared Gate 6 decision template.

Performance is measured, not gated. Do not invent or advertise a speedup
target.

---

## 3. Explicit non-goals

Do not implement:

- a general STARK-to-Rust compiler;
- arbitrary Core/HIR code generation;
- a new VM, graph compiler, tensor allocator, or tensor kernel library;
- ONNX parsing or signature rules beyond the completed Gate 4 importer;
- training, autodiff, datasets, quantization, named axes, or layout types;
- CUDA, CoreML, DirectML, TensorRT, IREE, or multi-provider selection before
  the CPU baseline closes;
- automatic model optimization, quantization, or ORT format conversion;
- a web service, package registry, LSP, debugger, or additional IDE work;
- cross-compilation infrastructure for every platform as a Gate 5 blocker.

The mandatory prototype is one native CPU deployment on the development host.
A second operating system is useful evidence but is not allowed to expand the
gate into a release-engineering project.

---

## 4. Locked architecture

```text
pipeline.stark + model.onnx
        │
        ├─ normal tensor-enabled parse / resolve / typecheck
        ├─ Gate 4 live artifact/declaration verification
        └─ bounded deployment lowering
                    │
                    ▼
             Deployment IR
                    │
          deterministic Rust emitter
                    │
                    ▼
       generated Cargo host project
       ├─ safe generated glue
       ├─ small checked host runtime
       ├─ pinned ort + ndarray + image dependencies
       └─ model/signature/hash manifest
                    │
                    ▼
       native CPU binary + model bundle
```

### 4.1 Dependency boundary

`starkc/Cargo.toml` must **not** gain `ort`, `ort-sys`, `ndarray`, `image`, or
other inference dependencies. The compiler emits text and metadata using its
existing dependency-light implementation. Backend dependencies belong only
to the generated host template/project.

This protects:

- the compiler's current Rust 1.85 MSRV;
- `starkide` and compiler release sizes;
- Core-only builds from native runtime downloads/linkage;
- hermetic ordinary tests.

The generated host may declare a newer Rust version if the pinned backend
requires it. Record that split explicitly rather than silently raising the
compiler MSRV.

### 4.2 CPU first

Use ONNX Runtime's CPU execution provider for the mandatory baseline. Static
device checking still demonstrates the CUDA/CPU defect at compile time, but
Gate 5 does not need a CUDA machine or GPU backend to close.

### 4.3 Input boundary

The generated executable owns image I/O:

1. decode JPEG/PNG with bounded dimensions and decoded-byte limits;
2. resize the shorter side to 256 while preserving aspect ratio;
3. center-crop 224×224;
4. expose owned NHWC `UInt8 [1, 224, 224, 3]` data to the generated pipeline
   as `TensorAny`;
5. let the STARK entry function perform `refine`, `permute`, `cast`, ImageNet
   mean/std normalization, prediction, softmax, and argmax.

Image decoding and resizing are host I/O adapters, not new STARK language
semantics. The `TensorAny.refine` in STARK remains the explicit trust boundary.

### 4.4 Artifact binding

At generation time:

- decode and verify the live ONNX signature with the Gate 4 implementation;
- compute the exact artifact SHA-256;
- emit the expected signature/hash into a deterministic deployment manifest.

At binary startup, before session creation:

- hash the supplied model and compare it to the manifest;
- produce a focused nonzero `ArtifactMismatch` error on disagreement;
- only then create the ONNX Runtime session and run inference.

Exact hash binding is deliberately stronger than signature-only matching for
this prototype. The exit report must say so. Do not claim that hash comparison
is a general replacement for Gate 4 signature verification.

---

## 5. Command-line contracts

### 5.1 Compiler command

Add:

```text
starkc deploy <pipeline.stark>
  --model <model.onnx>
  --entry <function>
  --out <directory>
  [--force]
```

Rules:

- tensor mode is enabled internally;
- unknown, duplicated, or missing arguments exit 2 and print usage;
- parse, type, verification, lowering, I/O, and generation failures exit 1;
- exactly one selected model declaration must be reachable from the entry;
- `--force` replaces only files owned by a previously generated project;
- never recursively delete an arbitrary user directory;
- generation is atomic where practical and leaves the previous output intact
  on failure;
- success prints the output directory, model SHA-256, entry, and next build
  command;
- generation itself never downloads dependencies or invokes Cargo.

Cargo remains an explicit second step:

```bash
cargo build --release --locked \
  --manifest-path target/gate5-host/Cargo.toml
```

### 5.2 Generated binary

The generated binary contract is:

```text
stark-resnet50
  --model <model.onnx>
  --image <image.jpg|image.png>
  [--warmup <N>]
  [--iterations <N>]
  [--json]
```

JSON success output must have a versioned stable schema:

```json
{
  "schema": 1,
  "model_sha256": "...",
  "input": {"width": 224, "height": 224, "channels": 3},
  "top1_index": 0,
  "top1_probability": 0.0,
  "startup_ms": 0.0,
  "inference_ms": {"iterations": 30, "min": 0.0, "median": 0.0, "p95": 0.0}
}
```

Do not put diagnostics or progress text on stdout in `--json` mode. Errors go
to stderr and return nonzero. Validate iteration ranges and cap them to avoid
accidental unbounded runs.

---

## 6. Deployment IR contract

Create an extension-owned module:

```text
starkc/src/deploy/mod.rs
starkc/src/deploy/ir.rs
starkc/src/deploy/lower.rs
starkc/src/deploy/emit.rs
starkc/src/deploy/template/
```

The IR is typed and backend-oriented. It must not store source text and
reparse it. Every node retains an originating `Span` for diagnostics.

Minimum data model:

```text
DeploymentProgram
  compiler_version
  entry
  selected model declaration/signature
  artifact SHA-256
  reachable functions in deterministic order

DeploymentFunction
  typed parameters/results
  locals
  straight-line blocks
  operations

DeploymentOp
  Refine
  TensorPermute
  TensorCast
  TensorFull
  TensorConcat
  TensorSub
  TensorDiv
  ModelPredict
  Softmax
  ArgMax
  UserCall
  Return / Result propagation
```

Only add operations required by the checked-in Gate 5 program. If a smaller
set proves sufficient, keep the smaller set.

### 6.1 Entry ABI

For Gate 5, require the selected entry to be equivalent to:

```stark
fn infer(
    model: Resnet50V17,
    input: TensorAny,
) -> Result<Tensor<Int64, [1]>, String>
```

Equivalent output may include probabilities as a tuple if tuple output is
implemented deliberately. Do not infer CLI behavior from arbitrary function
signatures. Reject a nonconforming entry with one focused deployment error.

### 6.2 Reachability

Lower only the entry's acyclic reachable call graph. Support small helper
functions such as `preprocess`; reject:

- recursion;
- dynamic dispatch or unresolved calls;
- reachable loops/complex control flow not needed by the prototype;
- unsupported collections, structs, traits, I/O, or mutation;
- tensor operations outside the deployment descriptor set.

Use an `E06xx` deployment-diagnostic range after checking that it does not
collide with normative codes. Each error names the unsupported construct and
its source span. Never fall back to the interpreter or emit placeholder code.

---

## 7. Representative STARK pipeline

Add `starkc/examples/gate5/valid_pipeline.stark`. It should remain readable
and close to Appendix A. The intended shape is:

```stark
model Resnet50V17<N: Dim> {
    input data: Tensor<Float32, [N, 3, 224, 224]>;
    output resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>;
}

fn infer(
    model: Resnet50V17,
    raw: TensorAny,
) -> Result<Tensor<Int64, [1]>, String> {
    let nhwc = raw.refine::<UInt8, [1, 224, 224, 3]>()?;
    let nchw = nhwc.permute::<[0, 3, 1, 2]>().cast::<Float32>();

    // Build broadcastable [1, 3, 1, 1] channel constants from `full` + `concat`.
    // Mean:  [123.675, 116.28, 103.53]
    // Std:   [58.395, 57.12, 57.375]
    let normalized = nchw.sub(&mean).div(&std);
    let logits = model.predict(&normalized);
    let probabilities = logits.softmax::<1>();
    Ok(probabilities.argmax::<1>())
}
```

The final code must spell out construction of `mean` and `std` using already
supported tensor operations and `f32` literals. First prove the source checks
with `--extension tensor`; do not add a new tensor-literal syntax merely for
convenience.

The model's documented preprocessing is: RGB, shorter-side resize to 256,
224 center crop, channel normalization using the values above, then HWC→CHW.

---

## 8. Generated Rust host requirements

### 8.1 Determinism and safety

Generated project contents must:

- contain no timestamp, absolute source/model path, random identifier, or
  host-specific newline;
- use stable identifier sanitization and deterministic ordering;
- end text files with exactly one newline;
- include compiler version, source hash, model hash, selected entry, and
  expected signature in a manifest;
- contain no generated `unsafe` block;
- use checked size arithmetic and bounded image dimensions/decoded bytes;
- avoid shell command construction and path use derived from model metadata;
- refuse overwrites without `--force`.

### 8.2 Runtime representation

Use a small safe Rust tensor wrapper over the exact array type selected in
M5.0. Implement only:

- owned UInt8, Float32, and Int64 tensors;
- runtime dtype/shape inspection and `refine`;
- `permute`, saturating cast, `full`, axis concat, broadcast `sub`/`div`,
  softmax, and argmax;
- conversion to/from ONNX Runtime input/output values.

Do not implement general tensor algebra. Each operation gets unit tests for
shape, values, dtype, overflow/NaN behavior where applicable, and failure
paths. Model inference itself remains an ONNX Runtime call.

### 8.3 Dependency pinning

M5.0 must select and pin exact versions and feature flags. Expected starting
point, subject to the spike:

- `ort = "=2.0.0-rc.12"` with CPU/std/API features only;
- the `ndarray` version compatible with that exact `ort` release;
- `image` with `default-features = false`, enabling only JPEG and PNG;
- `sha2` for runtime artifact binding;
- a minimal serialization option, or manual JSON if this keeps the dependency
  surface smaller and fully escaped.

Commit the generated host's `Cargo.lock`. Build exit commands use `--locked`.
Record licenses and native-runtime packaging behavior. If the backend's MSRV
is 1.88, set that on the host project only.

---

## 9. Milestones

### M5.0 — backend spike and frozen decision

**Goal:** prove the external runtime/toolchain boundary before compiler work.

In a disposable directory, create the smallest safe Rust program that:

1. builds with an exact `ort` version/features;
2. loads the pinned ResNet50 artifact;
3. creates a `[1, 3, 224, 224]` Float32 input;
4. runs one CPU inference;
5. extracts `[1, 1000]` Float32 output;
6. reports session creation and inference duration;
7. identifies dynamic-library dependencies using `otool -L`, `ldd`, or the
   Windows equivalent.

Record in `starkc/docs/gate5-backend-decision.md`:

- exact crate/runtime versions, features, MSRV, licenses, transitive count;
- whether ORT is static or which native files must be packaged;
- supported host triples actually tested;
- session/value APIs used;
- observed output shape and failure behavior;
- chosen ndarray/image/JSON approach;
- rejected alternatives and why;
- the final generated-host `Cargo.toml` dependency block.

Do not add ORT to `starkc`. Delete or leave the disposable spike outside Git;
commit only the decision record.

**Exit:** one real inference succeeds and the packaging/MSRV boundary is
understood. If it does not, stop with the decision memo described in section
15.

### M5.1 — deployment contract and IR lowering

**Files:** `src/deploy/{mod,ir,lower}.rs`, `src/lib.rs`, focused tests.

Implement:

- entry/model selection;
- tensor-enabled normal frontend invocation;
- Gate 4 live artifact verification before lowering;
- reachable-call-graph collection and recursion detection;
- exact Deployment IR nodes required by the prototype;
- `E06xx` diagnostics with spans for unsupported reachable HIR;
- no dependency on the backend crate.

Tests cover valid lowering, wrong entry ABI, multiple/no models, recursive
helpers, unsupported control flow/op, source spans, stable IR ordering, and a
regression proving Core-only behavior is unchanged.

**Exit:** the valid Gate 5 source lowers deterministically; every unsupported
construct fails explicitly.

### M5.2 — deterministic host emitter and CLI

**Files:** `src/deploy/emit.rs`, `src/deploy/template/`, `src/main.rs`, CLI
integration tests.

Implement `starkc deploy` and emit:

```text
<out>/Cargo.toml
<out>/Cargo.lock
<out>/src/main.rs
<out>/src/generated_pipeline.rs
<out>/src/runtime.rs
<out>/deployment.json
<out>/README.md
```

Use checked-in templates and generated fragments rather than a giant string
inside one Rust function. Golden-test every emitted text file. Generate twice
and compare byte-for-byte. Test overwrite refusal, `--force`, partial-write
cleanup, argument errors, and paths containing spaces.

The normal hermetic suite may syntax-check emitted source, but it must not
download ORT. Backend compilation remains an opt-in test until dependencies
are cached or CI explicitly provisions them.

**Exit:** deterministic project emission is complete and ordinary compiler
tests remain network-independent.

### M5.3 — generated CPU runtime and real inference

Implement and test the safe runtime template:

- bounded image decode/resize/crop;
- dynamic `TensorAny` representation and refinement;
- required tensor preprocessing/postprocessing operations;
- artifact hash check before session creation;
- ONNX Runtime session, named input/output, output validation;
- stable human and JSON output;
- warmup/iteration validation and latency collection.

Tests are split:

- hermetic unit tests for all local tensor/image-boundary logic using tiny
  arrays and generated images;
- opt-in real backend test using `STARK_GATE5_MODEL` and
  `STARK_GATE5_IMAGE`;
- runtime negative test that swaps one byte/model and proves inference never
  starts after the hash mismatch.

**Exit:** the generated release binary performs one correct end-to-end CPU
inference on the real model.

### M5.4 — reference oracle and defect corpus

Add:

```text
starkc/examples/gate5/valid_pipeline.stark
starkc/examples/gate5/bad_shape.stark
starkc/examples/gate5/bad_dtype.stark
starkc/examples/gate5/bad_device.stark
starkc/examples/gate5/artifact_drift.stark
starkc/tests/gate5_codegen.rs
starkc/tests/snapshots/gate5_*.stderr
starkc/tests/fixtures/gate5/fetch-input.sh
starkc/tests/fixtures/gate5/reference.py
starkc/tests/fixtures/gate5/requirements.txt
```

Requirements:

- source defects are minimal one-line mutations and have exact diagnostics;
- artifact drift uses a deliberately different tiny signature fixture for
  hermetic tests and a copied/mutated real artifact only in opt-in tests;
- select a small redistributable sample image with source, license, bytes,
  and SHA-256 recorded, or generate a deterministic synthetic image for the
  hermetic layer;
- pin Python reference dependencies exactly;
- Python and generated host use identical resize, crop, channel order,
  mean/std, softmax, and input image;
- emit raw logits/probabilities for comparison, not only top-1.

Fix the numerical agreement rule before examining results:

- identical output shape `[1, 1000]`;
- finite outputs;
- same top-1 index;
- maximum absolute error ≤ `1e-4` **or** maximum relative error ≤ `1e-4`
  per element, with both maxima reported.

If preprocessing-library interpolation differs, do not loosen tolerance
silently. Compare the preprocessed tensor first and align documented resize
semantics or record a justified deviation.

**Exit:** the valid result agrees with the independent reference, and all
four defects are demo-ready.

### M5.5 — packaging and measurement harness

Add a dependency-light script such as:

```text
starkc/scripts/run-gate5-evaluation.py
```

It must:

- verify model and input checksums;
- invoke `starkc deploy` and `cargo build --release --locked`;
- run a fixed warmup and measured iteration count;
- run the Python reference with the same parameters;
- collect executable size, model size, total bundle size, session startup,
  median/p95 inference latency, and peak RSS;
- capture `git rev-parse HEAD`, dirty status, OS/arch, CPU, RAM, rustc/cargo,
  Python, ORT, and dependency versions;
- write machine-readable `results.json` plus a Markdown summary;
- preserve raw command output for audit;
- distinguish unavailable metrics from zero values.

Suggested default protocol:

- 5 warmup runs;
- 30 measured runs for developer checks;
- 100 measured runs for the recorded exit result;
- one process for steady-state latency;
- a fresh process measurement for startup;
- peak RSS from `/usr/bin/time -l` on macOS, `/usr/bin/time -v` on Linux, or
  a documented Windows equivalent.

Do not compare numbers captured under different providers, inputs, thread
settings, power modes, or build profiles.

**Exit:** one command recreates the build, correctness comparison, and
measurement report from checksum-pinned inputs.

### M5.6 — exit evidence and Gate 6 handoff

Create:

```text
starkc/docs/gate5-exit.md
starkc/docs/gate6-decision-template.md
```

The exit report maps every roadmap criterion to code/tests/commands and
records:

- exact backend/toolchain decision;
- model/input provenance and checksums;
- generated project/binary structure;
- numerical agreement results;
- four-defect evidence;
- complete measurement environment and results;
- binary/native-library packaging behavior;
- known limitations and every deviation;
- reproducible validation commands.

The Gate 6 template must have three outcomes—go, revise, stop—and questions
about safety value, integration complexity, artifact size, runtime overhead,
diagnostic clarity, maintainability, and whether a library/schema generator
would deliver the same value more simply.

Only after every criterion passes, update `README.md` and
`STARKLANG/docs/ROADMAP.md` to mark Gate 5 complete and Gate 6 next.

---

## 10. Test architecture

### Hermetic default suite

Must run with no network, Python packages, large ONNX model, or native ORT
download:

- Deployment IR unit tests;
- deterministic emitter goldens;
- CLI parsing/status/overwrite tests;
- generated runtime tensor-op tests using tiny arrays;
- defect diagnostic snapshots;
- tiny artifact drift/hash fixtures;
- all existing Core, Gate 2, Gate 3, Gate 4, and IDE tests.

### Opt-in real-backend suite

Use explicit environment variables:

```bash
STARK_GATE5_MODEL=/path/resnet50-v1-7.onnx \
STARK_GATE5_IMAGE=/path/sample.jpg \
cargo test --test gate5_codegen -- --ignored --nocapture
```

The test verifies checksums before doing expensive work. It builds with
`--locked`, runs inference, compares reference output, and reports paths to
retained logs. It never downloads implicitly.

### Full compiler matrix

Run after every milestone and at closure:

```bash
cd starkc
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo doc --no-deps
git diff --check
```

Also rerun:

```bash
cargo test --test conformance
cargo test --test gate2_valid
cargo test --test gate3_execution
cargo test --test gate4_semantics
cargo test --test gate4_tensor
cargo test --test gate4_onnx
```

---

## 11. Required end-to-end demonstration

The final documented flow should be approximately:

```bash
# From repository root: obtain checksum-pinned model.
starkc/tests/fixtures/onnx/fetch-reference.sh /tmp/stark-gate5

cd starkc

# Static checks and generation.
cargo run -- check --extension tensor examples/gate5/valid_pipeline.stark
cargo run -- deploy examples/gate5/valid_pipeline.stark \
  --model /tmp/stark-gate5/resnet50-v1-7.onnx \
  --entry infer \
  --out target/gate5-host \
  --force

# Native build and inference.
cargo build --release --locked \
  --manifest-path target/gate5-host/Cargo.toml
target/gate5-host/target/release/stark-resnet50 \
  --model /tmp/stark-gate5/resnet50-v1-7.onnx \
  --image /tmp/stark-gate5/sample.jpg \
  --warmup 5 \
  --iterations 30 \
  --json

# Reproducible evaluation.
python3 scripts/run-gate5-evaluation.py \
  --model /tmp/stark-gate5/resnet50-v1-7.onnx \
  --image /tmp/stark-gate5/sample.jpg \
  --iterations 100
```

The three source defect checks and artifact mismatch command must be shown
immediately after this in `gate5-exit.md`, with expected nonzero statuses.

---

## 12. Review checklist

Before presenting any milestone as done, review:

- [ ] existing Gate 4 functionality was reused, not duplicated;
- [ ] `starkc` itself has no ORT/image/ndarray dependency;
- [ ] compiler MSRV is unchanged unless separately approved;
- [ ] generated code contains no `unsafe` and no absolute paths/timestamps;
- [ ] unsupported HIR fails with a span instead of being skipped;
- [ ] artifact verification occurs before inference;
- [ ] image limits prevent decompression bombs/unbounded allocation;
- [ ] preprocessing matches the documented reference exactly;
- [ ] JSON stdout is stable and clean;
- [ ] dependencies and native libraries are pinned/documented/licensed;
- [ ] ordinary tests remain hermetic;
- [ ] real-model tests are checksum-gated and opt-in;
- [ ] no Gate 6 conclusion is written before measurement evidence exists;
- [ ] unrelated dirty files are unstaged and preserved.

---

## 13. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| `ort` pre-release API churn | Pin an exact release and lockfile after M5.0; isolate APIs inside the generated runtime template. |
| `ort` requires Rust newer than `starkc` | Keep separate host MSRV/toolchain; do not raise compiler MSRV. |
| Native runtime packaging is unclear | M5.0 inspects linked libraries before codegen work. |
| General transpiler scope explosion | Lower only the entry's typed reachable subset; reject everything else. |
| Preprocessing mismatch dominates output error | Compare preprocessed tensors before model outputs; keep one documented algorithm. |
| Large model makes tests slow/non-hermetic | Keep model outside Git; opt-in checksum-gated backend suite. |
| Image license/provenance unclear | Use a generated fixture or a clearly redistributable image with source/license/hash. |
| Artifact swapped after generation | Runtime SHA-256 check before session creation. |
| Performance numbers become marketing claims | Record protocol/environment/raw results; no fixed speedup gate. |
| Cross-platform work expands scope | Close on one native CPU host; record other platforms as follow-up evidence. |

---

## 14. Suggested commit boundaries

Present each for approval; do not auto-commit:

1. `Record Gate 5 ONNX Runtime backend decision`
2. `Lower checked inference pipelines to deployment IR`
3. `Generate deterministic Rust inference hosts`
4. `Run ResNet50 through the generated CPU host`
5. `Add Gate 5 correctness and defect corpus`
6. `Measure and package the Gate 5 deployment`
7. `Close Gate 5 deployment prototype`

Never combine a speculative backend spike with compiler architecture changes.

---

## 15. Stop conditions

Stop and produce a decision memo rather than guessing when:

- the chosen `ort` release cannot run the pinned model on the host;
- native packaging requires unreviewed system installation or cannot produce
  a portable bundle;
- the compiler itself would need to depend on ORT/native inference libraries;
- implementing the representative program requires a general transpiler,
  arbitrary control-flow lowering, or new language syntax;
- a tensor operation's runtime semantics are ambiguous against the normative
  extension;
- the reference and generated preprocessing cannot be made identical without
  changing the agreed algorithm;
- output disagreement remains after independently comparing input tensor,
  ORT inputs, raw logits, and postprocessing;
- the required model/image has unsupported metadata or unclear licensing;
- CUDA/GPU work becomes necessary to make the CPU prototype function;
- a dependency requires `unsafe` in generated STARK-owned code rather than
  inside an audited external FFI crate;
- completing the work would silently weaken Gate 4 checks or Core isolation;
- the same blocking failure persists through three evidence-based attempts.

The memo must contain observed facts, minimal reproducer, commands/output,
relevant source-of-truth text, options, trade-offs, and a recommendation.

---

## 16. Starter prompt for Claude

Use this prompt with this file:

> Implement STARK Gate 5 according to
> `starkc/docs/CLAUDE_GATE5_IMPLEMENTATION_PLAN.md`. Begin with M5.0 only.
> Read every source of truth in section 1, inspect the dirty worktree, run the
> existing baseline, and verify current backend documentation before changing
> compiler code. Do not redo Gate 4 model typing and do not build a general
> transpiler. Prove one real CPU ONNX Runtime inference in a disposable spike,
> then write `gate5-backend-decision.md` and present the evidence for review.
> Work milestone by milestone, preserve unrelated changes, keep ordinary tests
> hermetic, and stop with a decision memo if any section 15 condition occurs.
> Do not commit until explicitly asked.
