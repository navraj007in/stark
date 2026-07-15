# Gate 5 backend decision (frozen at M5.0)

Per `CLAUDE_GATE5_IMPLEMENTATION_PLAN.md` §9 M5.0 and PLAN.md **T12**. This
record freezes the inference-backend boundary **before** any compiler code is
written, so later milestones integrate against pinned facts rather than a
moving target. The spike is disposable and lives outside Git
(`scratchpad/ortprobe`); only this record is committed.

## 1. Decision

**Backend: ONNX Runtime via the `ort` crate, pinned `=2.0.0-rc.12`, statically
linked, CPU execution provider.** The generated host crate owns this
dependency. `starkc` itself gains **no** inference dependency (plan §4.1).

Rejected alternatives and why:

- **ORT C API directly** — unnecessary; the `ort` safe wrapper runs the pinned
  model out of the box, so the C API is a fallback only (T12), not needed now.
- **`tract` (pure-Rust inference)** — would make STARK ship a model runtime,
  violating the roadmap exit criterion "no custom convolution or model
  kernels" and the "existing runtime" requirement (§2).
- **IREE / Cranelift / tvm** — explicitly deferred until Gate 6 evidence
  demands them (T12, plan §3 non-goals).
- **Dynamic ORT (`load-dynamic`)** — rejected for the prototype: it needs a
  `libonnxruntime.{dylib,so}` shipped and discovered at runtime. The default
  static download strategy produces a self-contained binary (see §4), which is
  the stronger portability story for a go/no-go demo.

## 2. Proven inference (spike result)

Environment: macOS (Darwin 25.5.0), `aarch64-apple-darwin`, rustc/cargo
**1.93.0**. Model: `resnet50-v1-7.onnx`, SHA-256
`af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c` (Gate 4's
pinned artifact; 98 MB; kept outside Git).

The minimal safe program (no `unsafe`) performed one real CPU inference:

```
loaded: input `data` -> output `resnetv17_dense0_fwd`
output shape = [1, 1000]
argmax class = 905, logit = 10.0339
inference latency = 46.2 ms
```

Critically, the runtime input/output **names match the signature Gate 4's
importer generated** (`model Resnet50V17 { input data: …; output
resnetv17_dense0_fwd: … }`), so the generated host will bind ports by the exact
names the type checker saw. Output shape `[1, 1000]` matches the checked
signature. This clears the go/no-go's critical external dependency.

## 3. APIs used (freeze these inside the generated runtime template)

| Purpose | Call |
| --- | --- |
| Session build + load | `Session::builder()?.commit_from_file(path)?` |
| Input port name | `session.inputs()[0].name()` (method, not field) |
| Output port name | `session.outputs()[0].name()` |
| Build input value | `Tensor::from_array(([1usize,3,224,224], vec))?` — `(shape, Vec)` tuple, **not** an ndarray |
| Run | `session.run(ort::inputs![name => input])?` |
| Extract output | `outputs[name].try_extract_tensor::<f32>()?` → `(&Shape, &[f32])` |

These are isolated behind the generated `runtime.rs` so `ort`'s pre-release API
churn touches exactly one template file (risk §13).

## 4. Packaging / linkage (the M5.0 unknown, now resolved)

`otool -L` on the release binary lists **only system libraries** — no
`libonnxruntime.dylib`:

```
/usr/lib/libc++.1.dylib
/System/Library/Frameworks/Foundation.framework/.../Foundation
/usr/lib/libSystem.B.dylib
/System/Library/Frameworks/CoreFoundation.framework/.../CoreFoundation
/System/Library/Frameworks/CoreML.framework/.../CoreML
/usr/lib/libobjc.A.dylib
```

- **ONNX Runtime is statically linked.** `ort-sys` (build-deps `ureq` +
  `lzma-rust2` + `hmac-sha256`) downloads and SHA-verifies a prebuilt static
  `libonnxruntime.a` (70 MB `ar` archive), cached under
  `~/Library/Caches/ort.pyke.io/dfbin/aarch64-apple-darwin/<hash>/`, and links
  it into the binary. **No native ORT file must be packaged alongside the
  binary.**
- `CoreML.framework`, `Foundation`, `libobjc` are macOS **system** frameworks
  (ORT's CoreML EP is compiled in but the CPU EP is used); always present, not
  shipped.
- Binary size ≈ **23 MB** (unstripped, debug=false release). The ~70 MB static
  ORT archive is mostly dead-stripped by the linker to the kernels referenced.

Reproducibility caveat: the static lib is fetched once at first build and
cached; `--locked` pins crate versions, and the download is content-hash
verified by `ort-sys`. Offline CI must pre-warm the cache — recorded as a
known constraint for M5.5, not a blocker.

## 5. Versions, MSRV, licenses

| Crate | Version | License | Notes |
| --- | --- | --- | --- |
| `ort` | `=2.0.0-rc.12` | MIT OR Apache-2.0 | rust-version **1.88** |
| `ort-sys` | `=2.0.0-rc.12` | MIT OR Apache-2.0 | rust-version **1.88**; downloads static ORT |
| `ndarray` | `0.17.2` | MIT OR Apache-2.0 | version `ort` rc.12 pulls; use the same |
| ONNX Runtime (upstream binary) | 1.x (bundled by ort-sys) | MIT (Microsoft) | statically linked, not vendored in-tree |

Transitive dependency count for the spike (`ort` + `ndarray`, no-dev): **47**.

**MSRV split (plan §4.1, §8.3):** the pinned backend requires **Rust 1.88**.
This applies to the **generated host project only**. `starkc`'s own MSRV
(1.85) is **not** raised — the compiler emits text/metadata and never depends
on `ort`. The generated `Cargo.toml` sets `rust-version = "1.88"`.

## 6. Final generated-host `Cargo.toml` dependency block

Frozen starting point for M5.2's emitter (features finalized when the runtime
template lands in M5.3; CPU/std/API + static download only, no GPU providers):

```toml
[package]
name = "stark-resnet50"
version = "0.1.0"
edition = "2021"
rust-version = "1.88"   # backend MSRV; NOT the compiler's

[dependencies]
ort = "=2.0.0-rc.12"                              # default features: static download-binaries + ndarray; CPU EP
ndarray = "=0.17.2"                               # same version ort rc.12 resolves
image = { version = "=0.25", default-features = false, features = ["jpeg", "png"] }
sha2 = "=0.10"                                    # runtime artifact hash binding (§4.4)
# JSON output is emitted by hand (bounded, fully escaped) — no serde, to keep the surface minimal (§8.3).
```

`Cargo.lock` is committed with the generated project; exit build commands use
`--locked` (plan §8.3, §11).

## 7. Stop-condition check (plan §15)

None triggered. The pinned `ort` release runs the pinned model on the host; no
system install is required beyond the automatic static download; the compiler
does not depend on ORT; no `unsafe` was needed in the spike; no ambiguity in
tensor runtime semantics arose at this layer. Cleared to proceed to M5.1.
