# Gate 6 decision memo — evidence collected during Gate 5

**Status:** Complete. Evidence measured (G6-04 + G6-05); decision recorded.

**Decision:** **REVISE** (owner-confirmed 2026-07-16). Re-scope the demonstrator to isolate the differentiator, rather than GO or STOP. Rationale and next experiment in §6.

---

## 1. Questions Evaluated

Does STARK's compile-time tensor/model checking deliver a **material safety or deployment advantage** over the strongest technical comparator (a generated Rust typed host) and is that advantage sufficient to justify the adoption and maintenance cost of a new language? (ROADMAP.md Gate 6.)

A second independent question is: Is `stark verify` useful as a low-friction CI tool even when users do not adopt the STARK language?

---

## 2. What was built (Gate 5 Evidence Inherited)

* **Representative model**: `resnet50-v1-7.onnx` (SHA-256 pinned: `af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c`).
* **Pipeline**: `import` → checked signature → STARK preprocess/classify → generated Rust host calling ONNX Runtime → single native binary.
* **Defect corpus**: the four roadmap classes as one-line source mutations.

---

## 3. Evidence — safety (the STARK thesis)

| Defect class | Caught when | Diagnostic | Evidence |
| --- | --- | --- | --- |
| Incompatible tensor dimensions | compile time | `E0212` | `bad_shape.stark:6:1`: literal dimension mismatch at axis 3 (found 100, expected 224) |
| Incorrect element type | compile time | `E0212` | `bad_dtype.stark:18:32`: type mismatch (expected Float32, found UInt8) |
| Incompatible device placement | compile time | `E0212` | `bad_device.stark:25:32`: device mismatch at `model.predict` (expected Cpu, found Cuda<0>) |
| Artifact/declaration drift | build/deploy time | `deploy error` | `artifact_drift.stark`: output dimension 1 mismatch (artifact 1000, declaration 999) |

* **Baseline Comparison**: The candidate defects are all caught before execution by STARK (5/5). Both comparators have now been measured: the Python/ORT operational baseline (G6-04, §3.1) and the strongest generated typed-Rust host (G6-05, §3.2).

### 3.1 Measured — Python/ONNX Runtime operational baseline (G6-04)

Harness: `starkc/tests/fixtures/gate6/baseline_defects.py`; raw results:
`starkc/tests/results/gate6/python-baseline.json`. Each row was produced by
running the defect against ResNet50 v1.7 (SHA pinned) + ONNX Runtime `1.27.0`
on macOS/arm64 (providers: CoreML, Azure, CPU — **no CUDA**). Nothing below is
asserted; every outcome is observed.

| Defect class | STARK | Python/ORT — when | Python/ORT — observed behaviour |
| --- | --- | --- | --- |
| Incompatible tensor dimensions | compile time (`E0212`) | runtime, at `session.run` | **runtime error** `InvalidArgument: Got invalid dimensions for input` |
| Incorrect element type | compile time (`E0212`) | runtime, at `session.run` | **runtime error** `InvalidArgument: Unexpected input data type. Actual: tensor(uint8), expected: tensor(float)` |
| Incompatible device placement | compile time (`E0212`) | **never** | **silent fallback**: requested `CUDAExecutionProvider`, ORT emitted a warning, fell back to `['CPUExecutionProvider']`, and ran to completion with no error |
| Artifact/declaration signature drift | build/deploy time (deploy error) | **never (structural)** | **uncatchable**: Python generates no typed declaration, so there is nothing to diff the artifact against |
| Runtime artifact swap (Gate 5 defect #5) | load time (`ArtifactMismatch`, SHA gate) | **never** | **silent wrong output**: a different-SHA model loaded and ran with no integrity check; top-1 went 258 → 739 |

Reading: the two shape/dtype defects Python *does* catch, but only at
`session.run` — after the program is built, deployed, started, the model
loaded, and the input preprocessed; STARK rejects them at compile time. The
device and both drift defects Python does **not** catch at all (silent fallback
/ silent wrong output / structurally uncatchable).

### 3.2 Measured — strongest generated typed-Rust host (G6-05)

Harness: `starkc/tests/fixtures/gate6/rust-comparator/` (`run.py` + `lib.rs` +
`cases/`); raw results: `starkc/tests/results/gate6/rust-comparator.json`;
`rustc 1.93.0` stable. `lib.rs` is the strongest plausible *generated* host: it
pushes the STARK tensor contract into Rust's type system — element type as `T`,
shape as per-rank const generics (`Tensor4<T,N,C,H,W,Dev>`), device as a phantom
tag. Every row is rustc's actual verdict.

| Defect class | STARK | typed-Rust — when | typed-Rust — observed |
| --- | --- | --- | --- |
| Incompatible tensor dimensions | compile (`E0212`) | **compile** | `E0308`: `expected 224, found 100` (`cases/d1_dim.rs`) |
| Incorrect element type | compile (`E0212`) | **compile** | `E0308`: `Tensor4<f32,..>` vs `Tensor4<u8,..>` (`cases/d2_dtype.rs`) |
| Incompatible device placement | compile (`E0212`) | **compile\*** | `E0308`: `expected ..Cpu, found ..Cuda<0>` (`cases/d3_device.rs`) |
| Declaration/signature drift | build/deploy | load | types generated from signature; stale decl caught by the same digest gate |
| Runtime artifact swap | load (SHA gate) | load | `ArtifactMismatch … refusing to run inference`, exit 1 (`cases/d4b_artifact.rs`) |

**Result: the strongest typed-Rust host reaches parity with STARK — 5/5 caught
before inference — on this pipeline.** Two measured/structural caveats keep
STARK's advantage real but narrow it sharply from a can/can't to a cost/generality
argument:

* `*` **Device typing is non-idiomatic.** Compile-time device catching required a
  phantom-device tensor API that re-implements STARK's device typing. An ordinary
  `ort` host configures device via execution providers at session build — a
  runtime concern — so it would catch this at runtime or silently, like Python.
  "Strongest comparator" here means "a host that paid to re-implement STARK's
  type system in generated Rust."
* **The shape-arithmetic wall (measured).** `cases/limit_reshape.rs` is a *valid*
  program (flatten `[1,3,224,224] → [1, C*H*W]`) that stable Rust cannot even
  express: `error: generic parameters may not be used in const operations`.
  Computed output dims (reshape/flatten/matmul/conv-stride/broadcast) need the
  unstable `generic_const_exprs`. This CV *preprocessing* needs only permutation
  + cast + elementwise ops (no arithmetic), which is exactly why Rust reaches
  parity here. Any pipeline that reshapes before a head, does conv-stride math,
  broadcasts, or carries more than one symbolic dim forces the stable-Rust host
  to drop those dims to runtime. STARK types all of them.

---

## 4. Evidence — deployment cost (measured)

Measured results for the STARK-generated native host on Apple Silicon (arm64):

* **Reference-output agreement**: Class 258 (Samoyed, prob 0.947176); matches Python reference within 1e-3 (Top-5 index and ordering matches exactly).
* **Artifact size**: `23.67 MB` for the generated host release binary (excludes model bytes).
* **Model size**: `97.83 MB`.
* **Total bundle size**: `121.50 MB`.
* **Inference Latency (median over 100 runs)**: `13.479 ms` (min: 12.674 ms, p95: 19.180 ms).
* **Peak RSS**: `374.00 MB` (includes loaded model and ONNX Runtime session state).
* **Warmup iterations**: 5.
* **Warm pipeline E2E latency**: `13.479 ms`.
* **Cold E2E latency**: `304.61 ms` (dominated by ORT model load time).
* **Build steps**: Fully automated cargo build (statically linked via `ort-sys`).

*Note: the safety comparison against the Python baseline (G6-04, §3.1) and the typed-Rust comparator (G6-05, §3.2) is now measured. Comparative deployment-cost metrics (binary size / latency / RSS of a hand-written Rust host vs. the STARK-generated host) were not separately measured; both compile to a single ORT-backed native binary, so no material deployment-cost delta is claimed in either direction.*

---

## 5. Principal Technical Comparator

The principal comparator for STARK's language safety model is:
```text
ONNX signature importer
+ generated Rust model types
+ typed tensor/layout API
+ artifact checksum verification
+ ONNX Runtime host
+ Rust compiler and Cargo
```
Python remains an operational baseline, not the strongest safety comparator.

---

## 6. Decision synthesis and recommendation

Both comparators are measured (§3.1 Python, §3.2 typed-Rust; consolidated in
`starkc/tests/results/gate6/results_summary.md`). Defects caught before
inference: **STARK 5/5, Python 2/5, strongest typed-Rust 5/5.**

**Against the operational baseline (Python), STARK's advantage is decisive:** 5/5
vs 2/5, and Python's two are caught only at `session.run`, after full deploy,
load, and preprocessing. **Against the strongest comparator (a generated typed-
Rust host), STARK reaches only parity on this pipeline** — and that parity cost
the Rust host a bespoke, re-implemented STARK-in-Rust type system, and holds only
because this CV preprocessing avoids shape arithmetic (the measured
`generic_const_exprs` wall).

Mapping to ROADMAP.md Gate 6:

* **GO** wants "a material safety or deployment advantage not erased by
  integration complexity." The as-built wedge does not clear this bar against the
  *strongest* comparator: the same four/five defect guarantees are obtainable via
  a schema generator + typed Rust for this pipeline. Deployment cost is a wash
  (both are one ORT-backed binary).
* **STOP** wants the guarantees to be "more simply obtained" elsewhere, or the
  type guarantees to be misleading at the backend boundary. Not met: the Rust
  parity path is *not* simpler (it re-implements the type system by hand and hits
  a hard wall on any shape-arithmetic pipeline), and STARK's checks are sound at
  the ORT boundary (verified in Gate 5).
* **REVISE (recommended).** STARK's real, un-replicated edge is **generality**
  (arbitrary shape arithmetic that stable Rust cannot type), **ergonomics** and
  **diagnostics**, and headroom for **semantic CV annotations** (color space,
  value range, coordinate frame — ROADMAP.md's proposed next experiment) that
  have no Rust-type equivalent at all. The current demonstrator does not exercise
  any of these, so it under-sells the differentiator and reads as parity.

**Recommended next narrow experiment (owner to approve before any work):**
re-scope the demonstrator to a pipeline that (a) requires shape arithmetic —
e.g. a detector/decoder with reshape/flatten/broadcast where the Rust comparator
is forced to drop dims to runtime — and (b) carries one semantic CV annotation
end to end with propagation + conversion rules. That is the experiment that turns
"parity" into a demonstrated, non-replicable advantage.

**Decision recorded: REVISE (owner-confirmed 2026-07-16).** The next narrow
experiment (the re-scoped demonstrator above) requires its own roadmap-governed
proposal — owner, bounded scope, measurable exit criteria (ROADMAP.md §5) —
before any implementation. No LSP work or language expansion is authorized by
this decision.
