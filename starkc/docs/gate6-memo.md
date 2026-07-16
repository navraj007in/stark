# Gate 6 decision memo — evidence collected during Gate 5

**Status:** Evidence collection in progress

**Decision:** Not yet made

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

* **Baseline Comparison**: The candidate defects are successfully caught before execution by STARK. The Python/ORT operational baseline has now been measured (G6-04, below); the typed-Rust comparator (G6-05) remains outstanding, so a language-level advantage over a *generated Rust typed comparator* is still not established.

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
/ silent wrong output / structurally uncatchable). This fixes the weakest
comparator; whether the *strongest* comparator (typed Rust) closes the gap is
the open G6-05 question.

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

*Note: Controlled comparisons against a Python baseline and a typed Rust comparator will be performed in G6-04 and G6-05 to gather concrete comparative metrics. Unmeasured estimates for these baselines are not cited.*

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

## 6. Current Status & Next Steps

Evidence collection is in progress. Gate 6 decisions are not yet made. No strategy decisions, LSP work, or language expansions are authorized.

* **G6-04 — Python/ORT baseline defect matrix: DONE.** Measured and recorded in §3.1 (`baseline_defects.py` + `python-baseline.json`). Result: Python catches 2/5 defects and only at `session.run` runtime; it misses device placement and both drift classes entirely.
* **G6-05 — Typed Rust comparator: OUTSTANDING (next WP).** Build a generated typed-Rust host (ONNX signature import → generated model types → typed tensor/layout API → artifact checksum verify → ORT) and run the same defect matrix to establish how many of the five a strong statically-typed comparator catches at compile time. This is the number that decides go vs. revise.
* **Decision write-up — after G6-05.** Only once both comparators are measured can §1's material-advantage question be answered.
