"""Gate 7 (G7-01) — independent Python/ONNX Runtime reference for Tiny YOLOv2.

Establishes the authoritative reference for the frozen workload:

  1. MODEL CORRECTNESS ORACLE: runs the pinned model in ONNX Runtime on the
     ONNX-bundled random conformance tensor and checks the raw output against
     the ONNX-bundled output_0.pb (an oracle authored by the model publisher,
     independent of this decode). This proves the model runs correctly.
  2. REAL DETECTION REFERENCE: runs the real reference image through the actual
     preprocessing (resize 416, CHW float32 in [0,255] — Tiny YOLOv2 expects
     RAW pixel values, NOT normalised), then the detector decode that exercises
     the *computed* shape relationships Gate 7 is about, exposing every
     intermediate shape:
        [B, A*(5+C), H, W]  ->  reshape [B, A, 5+C, H, W]
                            ->  permute [B, A, H, W, 5+C]
                            ->  decode boxes (grid + anchor broadcast) -> NMS
  3. writes a machine-readable reference (shapes, oracle result, detections,
     tolerance, dynamic provenance) to tests/results/gate7/reference.json.

Nothing that can be measured is hardcoded: checksums, shapes and detections are
computed from the fetched artifacts at run time.

Usage:
    reference.py            # oracle + image decode + write reference.json
"""

import hashlib
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
STARKC = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEST = os.path.join(STARKC, "tmp", "gate7")
RESULTS = os.path.join(STARKC, "tests", "results", "gate7")

MODEL = os.path.join(DEST, "tinyyolov2-8.onnx")
INPUT_PB = os.path.join(DEST, "test_data_set_0", "input_0.pb")
OUTPUT_PB = os.path.join(DEST, "test_data_set_0", "output_0.pb")
IMAGE = os.path.join(DEST, "dog.jpg")

# Frozen decode constants for Tiny YOLOv2 / Pascal VOC.
SIDE = 416           # input H == W
GRID = 13            # output H == W
STRIDE = SIDE // GRID  # 32
N_ANCHORS = 5        # A
N_CLASSES = 20       # C
BOX = 5 + N_CLASSES  # 5 box params (tx,ty,tw,th,to) + C class scores == 25
ANCHORS = np.array(
    [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52],
    dtype=np.float32,
).reshape(N_ANCHORS, 2)
VOC = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CONF_THRESHOLD = 0.3
NMS_IOU = 0.45
TOLERANCE = 1e-3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pb(path):
    t = onnx.TensorProto()
    with open(path, "rb") as f:
        t.ParseFromString(f.read())
    return numpy_helper.to_array(t)


def preprocess(image_path):
    """Tiny YOLOv2 preprocessing: resize to 416x416, CHW float32 in [0,255]."""
    img = Image.open(image_path).convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)          # HWC, [0,255]
    return np.transpose(arr, (2, 0, 1))[None]      # 1,3,416,416


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def decode(raw, shapes):
    """raw: [B, A*(5+C), H, W] -> list of detections. Records intermediate shapes."""
    b, _, h, w = raw.shape
    shapes["model_output"] = list(raw.shape)

    x = raw.reshape(b, N_ANCHORS, BOX, h, w)          # split A*(5+C) -> [A, 5+C]
    shapes["reshape_B_A_box_H_W"] = list(x.shape)
    x = np.transpose(x, (0, 1, 3, 4, 2))              # [B, A, H, W, 5+C]
    shapes["permute_B_A_H_W_box"] = list(x.shape)

    cols = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)
    rows = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
    tx, ty, tw, th, to = (x[..., i] for i in range(5))
    cls = x[..., 5:]

    bx = (cols + sigmoid(tx)) * STRIDE
    by = (rows + sigmoid(ty)) * STRIDE
    bw = np.exp(tw) * ANCHORS[:, 0].reshape(1, N_ANCHORS, 1, 1) * STRIDE
    bh = np.exp(th) * ANCHORS[:, 1].reshape(1, N_ANCHORS, 1, 1) * STRIDE
    scores = softmax(cls, axis=-1) * sigmoid(to)[..., None]
    shapes["decoded_scores_B_A_H_W_C"] = list(scores.shape)

    best_cls = np.argmax(scores, axis=-1)
    best_score = np.max(scores, axis=-1)
    dets = []
    for (bi, ai, yi, xi) in np.argwhere(best_score >= CONF_THRESHOLD):
        dets.append(
            {
                "class": VOC[int(best_cls[bi, ai, yi, xi])],
                "class_index": int(best_cls[bi, ai, yi, xi]),
                "score": float(best_score[bi, ai, yi, xi]),
                "cx": float(bx[bi, ai, yi, xi]),
                "cy": float(by[bi, ai, yi, xi]),
                "w": float(bw[bi, ai, yi, xi]),
                "h": float(bh[bi, ai, yi, xi]),
            }
        )
    return nms(dets)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx0, by0, bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def nms(dets):
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        if all(d["class_index"] != k["class_index"] or iou(d, k) < NMS_IOU for k in kept):
            kept.append(d)
    # round for stable JSON
    for d in kept:
        for key in ("score", "cx", "cy", "w", "h"):
            d[key] = round(d[key], 4 if key == "score" else 2)
    return kept


def dump_permuted(out_path):
    """Run the model on the ONNX-bundled input and dump the reshape+permute
    result [1, A, H, W, 5+C] as raw little-endian Float32 — exactly the tensor
    the STARK `decode` pipeline produces — so the generated host's output can be
    compared numerically on the identical input. Also dumps the same input as a
    raw Float32 tensor for the host to consume."""
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    x = load_pb(INPUT_PB).astype(np.float32)
    raw = sess.run([out_name], {in_name: x})[0]
    permuted = raw.reshape(1, N_ANCHORS, BOX, GRID, GRID).transpose(0, 1, 3, 4, 2)
    np.ascontiguousarray(x, dtype=np.float32).tofile(out_path + ".input")
    np.ascontiguousarray(permuted, dtype=np.float32).tofile(out_path)
    print(f"dumped input {list(x.shape)} and permuted {list(permuted.shape)}")


def main():
    if "--dump-permuted" in sys.argv:
        dump_permuted(sys.argv[sys.argv.index("--dump-permuted") + 1])
        return

    for p in (MODEL, INPUT_PB, OUTPUT_PB, IMAGE):
        if not os.path.exists(p):
            sys.exit(f"missing artifact {p}; run examples/gate7/fetch-artifacts.py first")

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # 1. Model-correctness oracle on the ONNX-bundled conformance tensor.
    conf_in = load_pb(INPUT_PB)
    expected = load_pb(OUTPUT_PB)
    conf_out = sess.run([out_name], {in_name: conf_in})[0]
    max_abs_diff = float(np.max(np.abs(conf_out - expected)))
    oracle_ok = max_abs_diff <= TOLERANCE

    # 2. Real detection reference on the reference image.
    img_in = preprocess(IMAGE)
    raw = sess.run([out_name], {in_name: img_in})[0]
    shapes = {}
    dets = decode(raw, shapes)

    computed_channels = N_ANCHORS * (5 + N_CLASSES)
    channel_ok = raw.shape[1] == computed_channels
    reshape_product_ok = (
        int(np.prod(shapes["reshape_B_A_box_H_W"])) == int(np.prod(shapes["model_output"]))
    )

    report = {
        "workload": "tiny-yolov2 (ONNX Model Zoo, MIT)",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "platform": sys.platform,
        },
        "provenance": {
            "model_sha256": sha256(MODEL),
            "conformance_input_sha256": sha256(INPUT_PB),
            "conformance_output_sha256": sha256(OUTPUT_PB),
            "image_sha256": sha256(IMAGE),
        },
        "signature": {
            "input": {"name": in_name, "shape": list(img_in.shape), "dtype": str(img_in.dtype)},
            "output": {"name": out_name, "shape": list(raw.shape), "dtype": str(raw.dtype)},
        },
        "preprocessing": {
            "resize": [SIDE, SIDE],
            "layout": "NCHW",
            "value_range": "[0,255] raw (Tiny YOLOv2 expects RAW pixels; NO /255, NO mean/std)",
        },
        "computed_shape_relationships": {
            "formula": "A*(5+C)",
            "A": N_ANCHORS,
            "C": N_CLASSES,
            "value": computed_channels,
            "artifact_channel_dim": int(raw.shape[1]),
            "channel_relationship_holds": bool(channel_ok),
            "reshape_product_equality_holds": bool(reshape_product_ok),
        },
        "intermediate_shapes": shapes,
        "oracle": {
            "source": "ONNX-bundled test_data_set_0/output_0.pb",
            "max_abs_diff": round(max_abs_diff, 8),
            "tolerance": TOLERANCE,
            "agrees": bool(oracle_ok),
        },
        "postprocessing": {"conf_threshold": CONF_THRESHOLD, "nms_iou": NMS_IOU},
        "detections": dets,
    }

    os.makedirs(RESULTS, exist_ok=True)
    out_json = os.path.join(RESULTS, "reference.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"oracle agrees: {oracle_ok} (max_abs_diff={max_abs_diff:.2e}, tol={TOLERANCE})")
    print(f"channel A*(5+C)={computed_channels} == artifact dim {raw.shape[1]}: {channel_ok}")
    print(f"reshape product equality holds: {reshape_product_ok}")
    print("intermediate shapes:")
    for k, v in shapes.items():
        print(f"  {k}: {v}")
    print(f"detections after NMS (>= {CONF_THRESHOLD}): {len(dets)}")
    for d in dets:
        print(f"  {d['class']:12} score={d['score']:.4f} "
              f"box=({d['cx']:.0f},{d['cy']:.0f},{d['w']:.0f},{d['h']:.0f})")
    print(f"wrote {out_json}")

    if not (oracle_ok and channel_ok and reshape_product_ok):
        sys.exit("reference verification FAILED")


if __name__ == "__main__":
    main()
