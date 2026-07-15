# Gate 4 exit — tensor front end and ONNX signatures

Gate 4 closed on 2026-07-15. The optional `tensor` extension now carries a
source file through syntax, resolution, static tensor/model checking, and ONNX
signature verification without changing Core-only behavior:

```text
STARK source → tensor-enabled AST/HIR → shape/dtype/device checks
ONNX bytes   → bounded metadata decoder → generated model declaration
ONNX + declaration → normal tensor front end → accumulated signature drift
```

Gate 4 inspects metadata only. It does not execute ONNX graphs or tensor
operations; native numerical execution remains Gate 5 work.

## Deliverables and evidence

| Deliverable | Implementation | Evidence |
| --- | --- | --- |
| Isolated extension syntax | `options.rs`, parser/AST/HIR tensor gates | parser, resolver, conformance, and `gate4_tensor` Core-isolation tests |
| Dimensions, dtypes, devices | `extensions/tensor/dim.rs`, `types.rs`, unified type checker | polynomial, unification, shape/dtype/device, ownership, and refinement unit tests |
| Tensor operation surface | descriptor table in `typecheck.rs` | constructor, elementwise, broadcast, matrix, transform, reduction, cast, and device tests |
| Nominal model declarations | parser/HIR/type checker model support | model validation, fresh call dimensions, multi-input identity, `load`, and `predict` tests |
| Deterministic ONNX import | `onnx/importer.rs`, `starkc import` | golden source, exact SHA-256, initializer filtering, limits, malformed input, and overwrite tests |
| Artifact verification | `onnx/verifier.rs`, `starkc verify` | count/order/name/dtype/rank/static/dynamic/identity drift and model-selection tests |
| Demonstration corpus | `examples/gate4/` | `tests/gate4_tensor.rs` and the commands below |

The decoder reads only the protobuf fields needed for graph-port signatures.
It enforces the documented file, field, depth, port, initializer, rank, and
name limits; rejects malformed wire data and unsupported port/dtype kinds; and
never executes graph nodes. Generated files contain the compiler version and
SHA-256 of the exact artifact bytes, but no path or timestamp.

## Representative CV artifact

The reference is ONNX Model Zoo `resnet50-v1-7.onnx`:

- expected bytes: `102583340`;
- pinned SHA-256:
  `af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c`;
- imported input: `data: Tensor<Float32, [N, 3, 224, 224]>`;
- imported output:
  `resnetv17_dense0_fwd: Tensor<Float32, [N, 1000]>`.

The large binary remains outside Git. Fetching is opt-in and checksum-gated:

```bash
cd ..
starkc/tests/fixtures/onnx/fetch-reference.sh /tmp
cd starkc
cargo run -- import /tmp/resnet50-v1-7.onnx \
  --out /tmp/resnet50-v1-7.stark --force
cargo run -- check --extension tensor /tmp/resnet50-v1-7.stark
cargo run -- verify /tmp/resnet50-v1-7.onnx \
  --declaration /tmp/resnet50-v1-7.stark
STARK_GATE4_REFERENCE_ONNX=/tmp/resnet50-v1-7.onnx \
  cargo test --test gate4_onnx \
  imports_and_verifies_checksum_pinned_reference_model -- --ignored
```

`examples/gate4/reference_resnet50.stark` is the byte-for-byte declaration
generated from that checksum-pinned artifact by `starkc 0.1.0`.

## Demonstrate the defect classes

The valid preprocessing/model-call example uses typed NHWC-to-NCHW
permutation, dtype conversion, and model prediction:

```bash
cargo run -- check --extension tensor examples/gate4/valid_pipeline.stark
```

Each source defect fails before numerical execution:

```bash
# Each command below is expected to exit 1 with E0212.
cargo run -- check --extension tensor examples/gate4/bad_shape.stark
cargo run -- check --extension tensor examples/gate4/bad_dtype.stark
cargo run -- check --extension tensor examples/gate4/bad_device.stark
```

The diagnostics identify the incompatible argument span and corresponding
model port. The artifact drift example also exits 1 and reports all useful
differences in one run:

```bash
cargo run -- verify /tmp/resnet50-v1-7.onnx \
  --declaration examples/gate4/artifact_declaration_drift.stark
```

It reports the two dynamic-versus-static batch dimensions and the `1000`
versus `999` output dimension together.

## Core isolation and regression evidence

Core remains the default. Running the Gate 4 pipeline without the extension
fails with focused messages naming `tensor`:

```bash
cargo run -- check examples/gate4/valid_pipeline.stark
```

The full suite includes the 121-fixture Core conformance corpus, Gate 2 valid
programs, Gate 3 execution and I/O, terminal IDE behavior, tensor-mode
semantics, ONNX robustness, and explicit Core isolation. Focused reproduction:

```bash
cargo test --test conformance
cargo test --test gate2_valid
cargo test --test gate3_execution
cargo test --test gate4_semantics
cargo test --test gate4_tensor
cargo test --test gate4_onnx
cargo run -- check examples/gate3/01_hello.stark
cargo run -- run examples/gate3/01_hello.stark
```

## Closure validation

Run from `starkc/`:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release --all-targets
cargo doc --no-deps
git diff --check
```

## Deviations and intentional boundaries

There are no known deviations from the bounded Gate 4 conformance claim.
Unsupported ONNX dtypes and non-tensor graph ports are rejected rather than
coerced, as required. The following are intentional gate boundaries, not
silent omissions:

- tensor storage, kernels, numerical operation execution, and runtime
  `refine` checks;
- ONNX Runtime integration, graph execution, generated host glue, and native
  inference binaries;
- image decoding/resizing/normalization kernels, datasets, training, and
  autodiff;
- performance claims, layouts, named axes, quantization, and semantic image
  annotations.

Those concerns remain in Gate 5 or later. Gate 5 is now the active roadmap
objective: build and measure one native inference deployment using an existing
backend without implementing a new tensor runtime.
