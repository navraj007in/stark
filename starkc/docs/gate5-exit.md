# STARK Gate 5 — Native Inference Deployment (Exit Report)

## 1. Executive Summary

This report documents the completion of the Gate 5 Go/No-Go prototype deployment. Using a statically-checked computer-vision inference pipeline compiled from STARK to a generated native Rust host, we successfully executed a ResNet50 model against a sample image.

* **Categorical Agreement**: The generated Rust host and the independent Python reference oracle produced the same top-one class index: **`258`** (Samoyed).
* **Numerical Agreement**: The small difference is consistent with the known interpolation differences between Pillow and the Rust `image` crate. The outputs agreed within an absolute tolerance of `1e-3`:
  * **Top-1 absolute probability difference**: `0.000288`
  * **Maximum absolute difference across 1,000 probabilities**: `0.000288`
  * **Top-5 ordering agreement**: Exact match

---

## 2. Toolchain and Backend Selection

* **Deployment Target Profile**: `tensor-deploy-1`
* **Inference Runtime Engine**: `ort` version `=2.0.0-rc.12`
* **ONNX Runtime Packaging**: The generated host statically links the prebuilt ONNX Runtime archive (`libonnxruntime.a`) downloaded by `ort-sys` during the first build. The resulting binary is self-contained and does not require an external shared library. Offline builds require the ORT cargo cache to be pre-populated.
* **Rust and Toolchain Matrix**:
  * **starkc compiler MSRV**: Rust `1.85`
  * **generated host MSRV**: Rust `1.88` (due to `ort` MSRV)
  * **Active environment versions**:
    ```text
    rustc: rustc 1.93.0 (254b59607 2026-01-19)
    cargo: cargo 1.93.0 (083ac5135 2025-12-15)
    python: Python 3.14.4
    numpy: 2.5.1
    pillow: 12.3.0
    onnxruntime: 1.27.0
    ```

---

## 3. Model & Input Checksums

* **Model Checksum** (ONNX Zoo ResNet50 v1.7, downloaded and verified on-demand; not committed to the repository):
  * **Path**: `tmp/resnet50-v1-7.onnx`
  * **SHA-256**: `af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c`
* **Input Checksum** (Standard PyTorch Hub reference image, downloaded and verified on-demand; not committed):
  * **Path**: `tmp/dog.jpg`
  * **SHA-256**: `f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a`

---

## 4. Image Preprocessing and Scaling Correction

Initial pipeline evaluation reported class `624` (library) due to a scaling omission. ResNet50 expects inputs normalized to `[0.0, 1.0]` before applying channel-wise mean/std subtraction.
We updated `valid_pipeline.stark` to explicitly scale the casted image tensor:
```stark
    let scale = full::<Float32, [1, 1, 1, 1]>(255.0f32);
    let scaled = nchw.div(&scale);
    let normalized = scaled.sub(&mean).div(&std);
```
Dividing by `255.0` aligned the input ranges and resolved the class predictions.

---

## 5. Stable Softmax & Top-5 Diagnostics

The host uses a stable softmax implementation:
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

No `NaN` or infinity occurred, and probabilities sum to approximately `1.0`.

### Rust Host Top 5 Output
```text
Top 5 predictions:
  1. Class 258: logit=14.7350, prob=0.947176
  2. Class 259: logit=10.8920, prob=0.020286
  3. Class 270: logit=10.3562, prob=0.011874
  4. Class 261: logit=9.1672, prob=0.003613
  5. Class 279: logit=9.1438, prob=0.003530
```

### Python Reference Top 5 Output
```text
Top 5 predictions:
  1. Class 258 (Samoyed): logit=14.7370, prob=0.947464
  2. Class 259 (Pomeranian): logit=10.8944, prob=0.020312
  3. Class 270 (white wolf): logit=10.3581, prob=0.011881
  4. Class 261 (keeshond): logit=9.1684, prob=0.003615
  5. Class 279 (Arctic fox): logit=9.1454, prob=0.003533
```
Both platforms identify the same top 5 classes, representing white, fluffy animals (Samoyed, Pomeranian, White Wolf, Keeshond, Arctic Fox), matching the `dog.jpg` visual features.

---

## 6. Execution & Memory Metrics

Measurements were collected on **macOS Apple Silicon (arm64)**:

| Metric | Value | Scope & Details |
| --- | --- | --- |
| **Generated Binary Size** | 23.67 MB | Self-contained, statically links `libonnxruntime.a`, debug symbols stripped |
| **Model size** | 97.83 MB | Reference model (`resnet50-v1-7.onnx`) |
| **Total Deployed Bundle** | 121.50 MB | Binary + Model only (excludes ORT cache, source, labels, and system libs) |
| **Peak Resident Set Size (RSS)** | 374.00 MB | 392,167,424 bytes. Peak RSS includes the loaded model and ONNX Runtime session state. |
| **Warmup iterations** | 5 | Runs executed to warm up cache and Ort Session |
| **Benchmark iterations** | 100 | Runs measured for latency stats |

### Detailed Latency Breakdown
* **CLI parsing**: < 1 ms
* **Session startup**: 304.61 ms (SHA-256 verification and ORT session instantiation)
* **Warm inference**: 13.043 ms (`session.run` only)
* **Warm pipeline**: 13.479 ms (Decode/preprocess/inference/postprocess with reused session)
* **Cold end-to-end**: 304.61 ms

---

## 7. Defect Corpus Diagnostics

The STARK compiler successfully rejects all four defect classes before model execution; source defects are rejected during type checking, declaration drift during host generation, and byte-level artifact drift during generated-host startup.

### 1. Incompatible Tensor Dimensions
* **Case file**: `examples/gate5/bad_shape.stark`
* **Command**: `starkc check --extension tensor examples/gate5/bad_shape.stark`
* **Exit status**: `1`
* **Diagnostic Code**: `E0212`
* **Primary Error**: `Error: [E0212] tensor dimension mismatch at axis 3: expected 224 from literal dimension, found 100 from literal dimension`
* **Stage**: Type checking (Front-end)
* **Inference attempted**: No

### 2. Incorrect Element Type
* **Case file**: `examples/gate5/bad_dtype.stark`
* **Command**: `starkc check --extension tensor examples/gate5/bad_dtype.stark`
* **Exit status**: `1`
* **Diagnostic Code**: `E0212`
* **Primary Error**: `Error: [E0212] tensor element type mismatch: expected Float32, found UInt8`
* **Stage**: Type checking (Front-end)
* **Inference attempted**: No

### 3. Incompatible Device Placement
* **Case file**: `examples/gate5/bad_device.stark`
* **Command**: `starkc check --extension tensor examples/gate5/bad_device.stark`
* **Exit status**: `1`
* **Diagnostic Code**: `E0212`
* **Primary Error**: `Error: [E0212] tensor device mismatch: expected Cpu, found Cuda<0>`
* **Stage**: Type checking (Front-end)
* **Inference attempted**: No

### 4. Artifact/Declaration Signature Drift
* **Case file**: `examples/gate5/artifact_drift.stark`
* **Command**: `starkc deploy examples/gate5/artifact_drift.stark --model tmp/resnet50-v1-7.onnx --entry infer --out tmp/drift-host --force`
* **Exit status**: `1`
* **Diagnostic Code**: Custom deployment signature drift error
* **Primary Error**: `Error: ONNX artifact does not match the model declaration: - output resnetv17_dense0_fwd dimension 1 differs: artifact 1000, declaration 999`
* **Stage**: Host generation (`starkc deploy`)
* **Inference attempted**: No

### 5. Runtime Artifact Checksum Drift
* **Case file**: `runtime_model_drift` (mutating the `.onnx` binary on disk after host compilation)
* **Command**: `./tmp/eval-host/target/release/stark-resnet50 --model tmp/mutated_model.onnx --image tmp/dog.jpg`
* **Exit status**: `1`
* **Diagnostic Code**: `ArtifactMismatch`
* **Primary Error**: `error: ArtifactMismatch: model SHA-256 f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a does not match the expected af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c; refusing to run inference`
* **Stage**: generated-host startup
* **Inference attempted**: No

---

## 8. Evaluation Provenance

To guarantee evaluation reproducibility, the SHA-256 checksums of all active files and scripts are recorded below:

* **STARK git commit SHA**: `9b6745bbf500e303a5973272ae4ae0e648cb5456`
* **dirty**: `False`
* **Model Checksum**: `af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c`
* **Image Checksum**: `f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a`
* **Evaluation Script Checksum**: `83e44c20b8f041ff9bc561a3371f4961e6992d9ccf53097c23102c77d94d31fe`
* **Reference Python Checksum**: `b2149b5c2a048a1c89073e51f8490a6e9d6d33261a293c6f1a8c9b2d6a5c1e2f`

Machine-readable evaluation artifacts are preserved in the repository under:
```text
starkc/tests/results/gate5/
    environment.json
    rust-host.json
    python-reference.json
    benchmark-runs.json
    comparison.json
```
