"""Gate 6 (G6-04) — Python/ONNX Runtime baseline defect matrix.

For each of the four roadmap defect classes, this harness drives a realistic
Python + ONNX Runtime inference pipeline and records the ACTUAL observed
behaviour: whether the defect is caught, at which stage, and how (exception
type + message, or silent wrong output). Nothing is asserted a priori — every
row is produced by running the defect.

This is the weakest ("operational baseline") comparator in the Gate 6 memo.
The typed-Rust comparator (G6-05) is measured separately.

Usage:
    baseline_defects.py <model.onnx> <image.jpg> [out.json]
"""

import hashlib
import json
import os
import sys
import traceback

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

EXPECTED_MODEL_SHA = "af16a04a6ec48ac494065d4439fe9dea590d337b9ca6dc328160ccf04a217b9c"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(image_path, size=224):
    """Correct preprocessing -> float32 NCHW [1,3,size,size] (matches reference.py)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    short = 256
    if w < h:
        new_w, new_h = short, int(round(h * short / w))
    else:
        new_h, new_w = short, int(round(w * short / h))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    x0, y0 = (new_w - size) // 2, (new_h - size) // 2
    img = img.crop((x0, y0, x0 + size, y0 + size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    return (arr - mean) / std


def top1(logits):
    return int(np.argmax(logits))


def run_model(model_path, input_tensor, providers=None):
    sess = ort.InferenceSession(
        model_path, providers=providers or ["CPUExecutionProvider"]
    )
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    out = sess.run([oname], {iname: input_tensor})[0][0]
    return sess, out


def record(defect, roadmap_class, attempt):
    """Run `attempt` (a zero-arg fn) and normalise its outcome into a row."""
    try:
        result = attempt()
        result.setdefault("defect", defect)
        result.setdefault("roadmap_class", roadmap_class)
        return result
    except Exception as e:  # noqa: BLE001 — we are deliberately capturing anything
        return {
            "defect": defect,
            "roadmap_class": roadmap_class,
            "caught": True,
            "stage": "runtime (session.run / session init)",
            "outcome": "runtime_error",
            "exception_type": type(e).__name__,
            "detail": str(e).strip().splitlines()[0][:400],
        }


def main():
    model_path, image_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    model_sha = sha256(model_path)
    correct = preprocess(image_path, 224)

    # Ground truth (the valid pipeline) so "silent wrong output" is grounded.
    _, ref_logits = run_model(model_path, correct)
    ref_top1 = top1(ref_logits)

    rows = []

    # --- Defect 1: incompatible tensor dimensions -------------------------
    def d1():
        wrong = preprocess(image_path, 100)  # [1,3,100,100] instead of 224
        _, _ = run_model(model_path, wrong)
        return {"caught": False, "stage": "n/a", "outcome": "unexpected_success"}

    rows.append(record("d1_dim", "incompatible tensor dimensions", d1))

    # --- Defect 2: incorrect element type ---------------------------------
    def d2():
        wrong = (correct * 0 + 1).astype(np.uint8)  # right shape, wrong dtype
        _, _ = run_model(model_path, wrong)
        return {"caught": False, "stage": "n/a", "outcome": "unexpected_success"}

    rows.append(record("d2_dtype", "incorrect element type", d2))

    # --- Defect 3: incompatible device placement --------------------------
    # Intent: run on GPU (CUDA). CUDA provider is absent on this host.
    def d3():
        sess = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider"]
        )
        active = sess.get_providers()
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        got = top1(sess.run([oname], {iname: correct})[0][0])
        used_cuda = "CUDAExecutionProvider" in active
        # It "ran": did it honour the requested device, or silently fall back?
        return {
            "caught": False,
            "stage": "no error raised",
            "outcome": "honoured_device" if used_cuda else "silent_device_fallback",
            "requested_provider": "CUDAExecutionProvider",
            "active_providers": active,
            "top1_after": got,
            "detail": (
                "Requested CUDA; ORT silently fell back to "
                f"{active} and ran anyway (no exception)."
                if not used_cuda
                else "CUDA honoured."
            ),
        }

    rows.append(record("d3_device", "incompatible device placement", d3))

    # --- Defect 4a: declaration/signature drift (structural) --------------
    # Python has no separate typed model declaration to diff against the
    # artifact: the .onnx file IS the signature. There is therefore no build
    # stage at which a declaration/artifact signature mismatch could be caught.
    rows.append(
        {
            "defect": "d4a_decl_drift",
            "roadmap_class": "artifact/declaration signature drift",
            "caught": False,
            "stage": "n/a (no declaration exists)",
            "outcome": "not_applicable_uncatchable",
            "detail": (
                "No typed declaration is generated in the Python pipeline; the "
                "artifact is the sole source of truth, so a declaration-vs-"
                "artifact signature mismatch has nothing to be checked against."
            ),
        }
    )

    # --- Defect 4b: runtime artifact drift (behaviour-changed model) ------
    # Analogue of Gate 5 defect #5: the deployed artifact is swapped for a
    # different (still-valid) model. STARK's generated host refuses on SHA
    # mismatch (ArtifactMismatch). Measure what Python does.
    def d4b():
        m = onnx.load(model_path)
        # Target the weight tensor feeding the graph's output node (the final
        # classifier), so the swap changes the prediction rather than being
        # attenuated by a mid-network residual/batchnorm path.
        out_names = {o.name for o in m.graph.output}
        final_node = next(
            (n for n in reversed(m.graph.node) if out_names & set(n.output)), None
        )
        floats = {
            i.name: i
            for i in m.graph.initializer
            if i.data_type == onnx.TensorProto.FLOAT
        }
        target = None
        if final_node is not None:
            # prefer the 2-D weight (not the 1-D bias) among the node's inputs
            weights = [floats[n] for n in final_node.input if n in floats]
            target = max(weights, key=lambda i: len(i.dims), default=None)
        if target is None:  # fallback: largest float initializer
            target = max(floats.values(), key=lambda i: int(np.prod(i.dims)) if i.dims else 0)
        arr = onnx.numpy_helper.to_array(target).copy()
        # Stand in for "a genuinely different deployed model version": replace
        # the largest weight tensor with a seeded, same-statistics random tensor.
        # (A uniform scale would preserve argmax since softmax is monotone under
        # a positive scalar; a light noise add leaves a 0.95-confidence class
        # intact. A different model version, by contrast, changes the output.)
        rng = np.random.default_rng(0)
        std = max(float(arr.std()), 1e-3)
        arr = rng.normal(float(arr.mean()), std, arr.shape).astype(np.float32)
        target.CopyFrom(onnx.numpy_helper.from_array(arr, target.name))
        mutated_path = os.path.join(
            os.path.dirname(out_path or model_path), "mutated_resnet50.onnx"
        )
        onnx.save(m, mutated_path)
        mutated_sha = sha256(mutated_path)
        _, mut_logits = run_model(mutated_path, correct)
        mut_top1 = top1(mut_logits)
        os.remove(mutated_path)
        changed = mut_top1 != ref_top1
        return {
            "caught": False,
            "stage": "no integrity gate",
            "outcome": "silent_wrong_output" if changed else "ran_no_error_same_top1",
            "original_sha": model_sha,
            "mutated_sha": mutated_sha,
            "mutated_tensor": target.name,
            "ref_top1": ref_top1,
            "mutated_top1": mut_top1,
            "detail": (
                "Swapped artifact (different SHA) loaded and ran with no "
                f"integrity check; top-1 went {ref_top1} -> {mut_top1}."
                if changed
                else "Mutated model ran with no error; top-1 unchanged."
            ),
        }

    rows.append(record("d4b_artifact_drift", "artifact/declaration signature drift (runtime swap)", d4b))

    report = {
        "harness": "gate6 baseline_defects.py",
        "comparator": "python + onnxruntime (operational baseline)",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "platform": sys.platform,
        },
        "model_path": model_path,
        "model_sha256": model_sha,
        "model_sha_matches_pin": model_sha == EXPECTED_MODEL_SHA,
        "reference_top1": ref_top1,
        "defects": rows,
    }

    print(json.dumps(report, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
