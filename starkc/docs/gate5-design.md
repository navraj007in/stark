# STARK Gate 5 — Go/no-go deployment prototype (design)

Gate 5 turns the statically-checked Gate 4 tensor pipeline into a **deployable
native binary** that actually runs a real CV model, and measures it. Per
ROADMAP.md Gate 5 and PLAN.md §5 / decision T12.

## 1. Backend decision — settled and proven

PLAN.md **T12** mandates the backend: **ONNX Runtime via a generated Rust host
program, `ort` crate first** (drop to the ORT C API only if the wrapper
blocks something). No IREE/Cranelift until Gate 6 evidence demands them.

**Feasibility is proven in this environment** (G5.0):

- `ort = "2.0.0-rc.12"` (+ `ort-sys`) compiles; ONNX Runtime binaries are
  fetched by `ort`'s default download strategy — no manual system install.
- The reference `resnet50-v1-7.onnx` (Gate 4's pinned artifact) loads and runs:
  input `data` → output `resnetv17_dense0_fwd`, output shape `[1, 1000]`,
  argmax class 905, ~44 ms steady-state latency on CPU.
- The runtime input/output names **match the signature Gate 4's importer
  generated** (`model Resnet50V17 { input data: …; output resnetv17_dense0_fwd:
  … }`), so the generated host binds ports by the same names the type checker
  saw.

This is the go/no-go's critical dependency; with it cleared, Gate 5 is
achievable here.

## 2. Architecture

```text
model.onnx ──starkc import──▶ model.stark (checked signature, Gate 4)
                                   │
pipeline.stark (preprocess/classify, --extension tensor, Gate 4 checks pass)
                                   │
                          starkc gen-host  (G5.1)
                                   │
                                   ▼
   host crate (Cargo + generated main.rs) ──cargo build --release──▶ native binary
                                   │                                      │
                                   └── ort loads model.onnx ◀─────────────┘
                                       runs preprocess → predict → postprocess
```

The STARK front end owns every shape/dtype/device guarantee; `ort` owns kernel
execution. STARK implements **no** tensor kernels and **no** ML runtime
(roadmap exit criterion). Preprocessing ops that are pure index/gather
(`permute`, `cast`) are emitted as small generated Rust over `ndarray`; the
model call becomes an `ort` session run.

## 3. Generated host — reference template

The probe below is the verified reference the generator emits (names and shapes
come from the imported model + checked pipeline; only glue is generated):

```rust
use ort::session::Session;
use ort::value::Tensor;

fn main() -> ort::Result<()> {
    let mut session = Session::builder()?.commit_from_file(MODEL_PATH)?;
    // input `data`: Tensor<Float32, [1, 3, 224, 224]> (checked)
    let input = Tensor::from_array(([1usize, 3, 224, 224], preprocess(pixels)))?;
    let outputs = session.run(ort::inputs!["data" => input])?;
    // output `resnetv17_dense0_fwd`: Tensor<Float32, [1, 1000]> (checked)
    let (_shape, logits) = outputs["resnetv17_dense0_fwd"].try_extract_tensor::<f32>()?;
    let class = argmax(logits);
    println!("class = {class}");
    Ok(())
}
```

## 4. Milestones

- **G5.0** (this note): backend settled + proven, Gate 4 baseline green (209
  tests), Gate 6 memo template prepared (`gate6-memo.md`).
- **G5.1**: `starkc gen-host` — emit the host crate from the imported model +
  pipeline; build the single native binary; run the valid pipeline.
- **G5.2**: defect corpus (four classes, each a one-line mutation with its
  captured diagnostic — Gate 4's `examples/gate4/bad_*` already cover shape,
  dtype, device; drift via `verify`), measurement harness (reference-output
  tolerance vs a Python/ORT baseline, artifact size, startup, peak RSS,
  latency — **reported, not gated**), and `gate5-exit.md`.

## 5. Boundaries (unchanged from roadmap §4)

No training, autodiff, quantization, named axes, GPU stack, or new kernels.
Image decode/resize/normalize are out of scope; the demo feeds a deterministic
input tensor (or a pre-decoded fixture). Measurements are evidence for the
Gate 6 decision, not pass/fail gates.
