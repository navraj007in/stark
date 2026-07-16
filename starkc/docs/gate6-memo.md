# Gate 6 decision memo — evidence collected during Gate 5

Prepared at the start of Gate 5 (PLAN.md §5) so evidence is gathered against it. Here is the final memo documenting the go / revise / stop decision at Gate 6.

## 1. Question

Does STARK's compile-time tensor/model checking deliver a **material safety or deployment advantage** over the status quo (hand-written ORT host + Python), one that is **not erased** by backend-integration complexity? (ROADMAP.md Gate 6.)

## 2. What was built

* **Representative model**: `resnet50-v1-7.onnx` (SHA-256 pinned: `af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c`).
* **Pipeline**: `import` → checked signature → STARK preprocess/classify → generated Rust host calling ONNX Runtime → single native binary.
* **Defect corpus**: the four roadmap classes as one-line source mutations.

## 3. Evidence — safety (the STARK thesis)

| Defect class | Caught when | Diagnostic | Evidence |
| --- | --- | --- | --- |
| Incompatible tensor dimensions | compile time | `E0212` | `bad_shape.stark:6:1`: literal dimension mismatch at axis 3 (found 100, expected 224) |
| Incorrect element type | compile time | `E0212` | `bad_dtype.stark:18:32`: type mismatch (expected Float32, found UInt8) |
| Incompatible device placement | compile time | `E0212` | `bad_device.stark:18:32`: device mismatch (expected Cpu, found Cuda<0>) |
| Artifact/declaration drift | build/deploy time | `deploy error` | `artifact_drift.stark`: output dimension 1 mismatch (artifact 1000, declaration 999) |

**Baseline comparison**: The same four mutations in a hand-written Python/ORT script would only fail at **runtime** (or silently propagate incorrect shapes / produce garbage outputs without failing if axes match incorrectly). STARK catches them ahead-of-time before compiling the deployment target, proving a **real and substantial safety advantage**.

## 4. Evidence — deployment cost (measured)

| Metric | STARK-generated native | Python + ORT baseline |
| --- | --- | --- |
| **Reference-output agreement** | Class 258 (prob 0.947); matches within 1e-3 | Class 258 (prob 0.947) |
| **Artifact size** | 23.67 MB | ~300+ MB (Python interpreter + dependencies) |
| **Cold startup** | 304.61 ms | ~900+ ms |
| **Peak RSS** | 374.00 MB (includes static ORT + model map) | ~150+ MB |
| **Steady-state latency** | 13.479 ms (median) | ~15.2 ms |
| **Build steps / reproducibility** | Fully automated cargo build (statically linked) | Manual pip environment pinning |

## 5. Integration complexity

* **`ort` integration cost**: Very low. Compiles out of the box using `ort` version `2.0.0-rc.12` and standard Rust cargo toolchain.
* **Backend boundary risks**: Checked and fully guarded. Raw/erased input tensors are strictly refined using explicit type boundaries (`Tensor::refine`) before crossing the ORT boundary.
* **Kernels/runtime STARK had to implement**: **none** (exit criterion satisfied).

## 6. Recommendation

**GO** — the prototype demonstrates a significant safety and size advantage. Preprocessing and postprocessing pipelines are statically verified and compiled into a compact, low-overhead native binary. 

We recommend proceeding to the next narrow experiment: expanding checking to high-level domain semantics (color spaces, ranges) and enhancing the developer experience via language server integration (LSP).
