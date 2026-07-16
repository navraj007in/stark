# Gate 7 frozen workload — Tiny YOLOv2 (freeze document)

This is the G7-01 freeze: the exact model, inputs, signatures, pre/post
processing, expected outputs, and tolerance that every later Gate 7 work
package (G7-02…G7-07) is measured against. Nothing here may drift without a new
freeze.

## Model origin and licence

| Field | Value |
| --- | --- |
| Model | Tiny YOLOv2 (Pascal VOC, 20 classes) |
| Source | ONNX Model Zoo — `validated/vision/object_detection_segmentation/tiny-yolov2` |
| Licence | **MIT** |
| ONNX opset | 8 (`ai.onnx`) |
| Artifact | `tinyyolov2-8.onnx` (~60 MB; **not committed** — fetched by checksum) |
| Fetcher | `examples/gate7/fetch-artifacts.py` (verifies SHA-256, exits non-zero on mismatch) |

The model was **not** modified in any way. It was chosen because its raw
detection head naturally requires computed shape relationships (the
differentiator the Gate 6 decision called for), not to create an artificial
limitation for any comparator.

## Pinned checksums (SHA-256)

| Artifact | SHA-256 |
| --- | --- |
| `tinyyolov2-8.onnx` | `583fb7fdc948435ceac9fa82efc7708701efe8382a859a3dd46526b155f5f2ae` |
| `test_data_set_0/input_0.pb` | `7df6dfefe116d2024a2678a3b015b98256ab8e15205b6b67e5291c8e7889cb5c` |
| `test_data_set_0/output_0.pb` | `59bd363150b98190fe3957e98905a645feec39b16c7f543e8e96d9e8e595c727` |
| `dog.jpg` (reference image) | `f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a` |

## Tensor signatures (verified)

```text
input  image : Float32 [None, 3, 416, 416]   (None = dynamic batch -> symbolic B)
output grid  : Float32 [None, 125, 13, 13]
```

The **batch dimension is genuinely dynamic** in the artifact, so `B` is a real
symbolic dimension, not a synthesized one.

## The computed shape relationship (why this workload)

`125` is not a literal to be copied through — it is **`A * (5 + C)`** with
`A = 5` anchors and `C = 20` classes. The decode splits and reorders it:

```text
grid    : [B, 125, 13, 13]        # B symbolic, 125 == A*(5+C)
reshape : [B, 5, 25, 13, 13]      # 125 == 5 * 25 (product equality must hold)
permute : [B, 5, 13, 13, 25]      # per-cell box+score vectors
```

Verified by `reference.py`: `channel_relationship_holds = true`,
`reshape_product_equality_holds = true`. This is exactly the shape arithmetic
the Gate 5 ResNet pipeline lacked and that stable Rust cannot type without the
unstable `generic_const_exprs` feature (measured in Gate 6).

## Preprocessing (frozen)

```text
load image -> resize 416x416 (bilinear) -> RGB -> NCHW float32
value range: RAW [0,255]  (Tiny YOLOv2 expects raw pixels: NO /255, NO mean/std)
```

Determined empirically (`reference.py`): `[0,255]` gives coherent detections
(max score 0.633); `[0,1]` gives noise (max score 0.004). **Implication for
G7-04:** this model's input contract is a *ByteRange* tensor, so the value-range
semantic experiment demonstrates the common bug of *accidentally normalising*
input for a model that wants raw pixels — the inverse of the ResNet case. This
is honest and realistic; the value-range state machine (`ByteRange` /
`UnitRange` / `Normalized`) is unchanged, only which state the model requires.

## Postprocessing (frozen)

```text
per cell/anchor: (bx,by) = (grid + sigmoid(t_xy)) * 32
                 (bw,bh) = exp(t_wh) * anchor * 32
                 score   = softmax(class_logits) * sigmoid(objectness)
filter: score >= 0.3 ; then class-wise NMS at IoU 0.45
anchors: [1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52]
```

## Expected outputs (reference image `dog.jpg`)

Recorded in `tests/results/gate7/reference.json`:

| Rank | Class | Score | Box (cx,cy,w,h) |
| --- | --- | --- | --- |
| 1 | cat | 0.6330 | (148,177,217,390) |
| 2 | dog | 0.4036 | (177,233,259,376) |

(The reference image is a Samoyed; the weak tiny model reads it as cat+dog.
This is a correctness *anchor for STARK-vs-Python agreement*, not a claim about
detection quality.)

**Model-correctness oracle:** on the ONNX-bundled conformance tensor
(`input_0.pb`), ORT output agrees with the publisher's `output_0.pb` to
`max_abs_diff = 2.19e-05`.

## Numerical tolerance

`1e-3` absolute (oracle and STARK-vs-reference comparisons).

## Current-compiler starting point (informational, scopes G7-02)

Checked with the committed `starkc` (`--extension tensor`):

* `examples/gate7/model.stark` → **OK** (symbolic-`B` model signature already
  parses and checks).
* `examples/gate7/valid_pipeline.stark` → **OK**: the frontend *already* types
  the symbolic reshape + permute and propagates `B` across `decode`/`infer`.
* A wrong reshape product (`5*26 != 125`) is **already rejected** with `E0212
  cannot prove reshape element counts equal`.
* **Gap for G7-02:** that diagnostic surfaces an internal symbol (`21125*d3`)
  instead of the source-level dim `B` and the offending axis — §7 of the Gate 7
  instructions forbids exposing internal polynomial IDs. G7-02's work is
  therefore primarily *diagnostic quality* and covering the remaining negative
  cases (broadcast, concat, cross-symbol confusion), not building product
  equality from scratch.

## Files

```text
examples/gate7/model.stark            canonical model signature (checks OK alone)
examples/gate7/valid_pipeline.stark   self-contained decode pipeline (checks OK)
examples/gate7/fetch-artifacts.py     fetch + SHA-verify model/testdata/image
tests/fixtures/gate7/reference.py     independent Python/ORT reference + oracle
tests/results/gate7/reference.json    frozen expected shapes/outputs/provenance
```

## Reproduce

```bash
cd starkc
python examples/gate7/fetch-artifacts.py           # fetch + verify (into tmp/gate7)
python tests/fixtures/gate7/reference.py           # oracle + decode + reference.json
```

Requires the packages in `tests/fixtures/gate7/requirements.txt`.
