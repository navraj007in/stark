#!/usr/bin/env python3
"""Gate 7 (G7-05) — native deployment evaluation.

Reproducibly drives the frozen Tiny YOLOv2 symbolic-shape pipeline end to end
(STARK source -> deploy -> cargo release build -> real ONNX Runtime execution),
runs the independent Python reference, exercises the defect corpus, records
operational measurements, and writes machine-readable evidence to
starkc/tests/results/gate7/. Fails if the tracked working tree is dirty.

Usage:
    scripts/run-gate7-evaluation.py [--allow-dirty]
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STARKC = os.path.dirname(HERE)
RESULTS = os.path.join(STARKC, "tests", "results", "gate7")
TMP = os.path.join(STARKC, "tmp", "gate7")
MODEL = os.path.join(TMP, "tinyyolov2-8.onnx")
IMAGE = os.path.join(TMP, "dog.jpg")
VENV_PY = os.path.join(STARKC, "tmp", "venv", "bin", "python3")


def sh(args, cwd=None):
    t0 = time.time()
    res = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return res, (time.time() - t0) * 1000.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_with_rss(args):
    osn = platform.system()
    if osn == "Darwin":
        cmd = ["/usr/bin/time", "-l"] + args
    elif osn == "Linux":
        cmd = ["/usr/bin/time", "-v"] + args
    else:
        cmd = args
    res = subprocess.run(cmd, capture_output=True, text=True)
    rss = None
    for line in res.stderr.splitlines():
        low = line.lower()
        if "maximum resident set size" in low:
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                rss = int(digits)
                if osn == "Linux":  # KB -> bytes
                    rss *= 1024
    return res, rss


def git_head_and_dirty():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=STARKC).decode().strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "-uno"], cwd=STARKC
    ).decode().strip()
    return head, len(status) > 0


def starkc(args):
    return sh(["cargo", "run", "-q", "--"] + args, cwd=STARKC)


def code_of(text):
    import re
    m = re.search(r"\[E0[0-9]+\]|ArtifactMismatch|does not match the model declaration", text)
    return m.group(0) if m else None


def main():
    allow_dirty = "--allow-dirty" in sys.argv
    head, dirty = git_head_and_dirty()
    if dirty and not allow_dirty:
        print("error: Gate 7 evaluation requires a clean tracked tree (use --allow-dirty for dev)")
        sys.exit(1)
    for p in (MODEL, IMAGE):
        if not os.path.exists(p):
            print(f"error: missing {p}; run examples/gate7/fetch-artifacts.py first")
            sys.exit(1)
    os.makedirs(RESULTS, exist_ok=True)

    # --- 1. deploy + build the frozen pipeline ------------------------------
    out_dir = os.path.join(STARKC, "tmp", "gate7-eval-host")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    print("=== deploy ===")
    res, deploy_ms = starkc([
        "deploy", "examples/gate7/valid_pipeline.stark",
        "--model", MODEL, "--entry", "infer", "--out", out_dir, "--force",
    ])
    if res.returncode != 0:
        print(res.stderr)
        sys.exit("deploy failed")
    print(f"deploy: {deploy_ms:.0f} ms")

    print("=== cargo release build ===")
    res, build_ms = sh([
        "cargo", "build", "--release", "--locked",
        "--manifest-path", os.path.join(out_dir, "Cargo.toml"),
    ], cwd=STARKC)
    if res.returncode != 0:
        print(res.stderr)
        sys.exit("host build failed")
    print(f"host build: {build_ms/1000:.1f} s")
    binary = os.path.join(out_dir, "target", "release", "stark-inference-host")

    # --- 2. run the host (json) with RSS ------------------------------------
    print("=== native inference ===")
    res, rss = run_with_rss([
        binary, "--model", MODEL, "--image", IMAGE,
        "--warmup", "5", "--iterations", "100", "--json",
    ])
    if res.returncode != 0:
        print(res.stderr)
        sys.exit("inference failed")
    host = json.loads(res.stdout.strip())
    print(f"host result: {host['result']['dtype']} {host['result']['shape']} "
          f"checksum {host['result']['checksum']}")

    # --- 3. independent Python reference ------------------------------------
    print("=== python reference ===")
    ref_res, ref_ms = sh([VENV_PY, os.path.join(STARKC, "tests", "fixtures", "gate7", "reference.py")])
    if ref_res.returncode != 0:
        print(ref_res.stderr)
        sys.exit("reference failed")
    with open(os.path.join(RESULTS, "reference.json")) as f:
        reference = json.load(f)

    # --- 4. defect corpus ---------------------------------------------------
    print("=== defect corpus ===")
    defects = []

    def record(case, command, res, stage, inference_attempted, expected):
        actual = code_of(res.stdout + res.stderr)
        defects.append({
            "case": case,
            "command": " ".join(command),
            "exit_code": res.returncode,
            "diagnostic_code": actual,
            "detection_stage": stage,
            "inference_attempted": inference_attempted,
            "expected_result": expected,
            "actual_result": "rejected" if res.returncode != 0 else "ACCEPTED (unexpected)",
        })

    source_defects = [
        ("d1_reshape_product", "incorrect reshape product"),
        ("d2_symbol_relationship", "broken symbolic dimension relationship"),
        ("d3_broadcast", "invalid broadcast"),
        ("d4_range_missing", "missing range conversion"),
        ("d5_range_double", "duplicate range conversion"),
        ("d6_dtype", "wrong dtype"),
        ("d7_device", "wrong device"),
    ]
    for name, _ in source_defects:
        path = f"examples/gate7/defects/{name}.stark"
        cmd = ["cargo", "run", "-q", "--", "check", "--extension", "tensor", path]
        res, _ = sh(cmd, cwd=STARKC)
        record(name, ["starkc", "check", "--extension", "tensor", path],
               res, "compile-time (type checking)", False, "compile-time rejection")

    # d8: declaration/artifact drift (deploy-time)
    drift_out = os.path.join(STARKC, "tmp", "gate7-drift")
    cmd = ["deploy", "examples/gate7/defects/d8_declaration_drift.stark",
           "--model", MODEL, "--entry", "infer", "--out", drift_out, "--force"]
    res, _ = starkc(cmd)
    record("d8_declaration_drift", ["starkc"] + cmd, res,
           "deploy-time (artifact verification)", False, "deploy-time rejection")

    # d9: runtime artifact swap (mutate the model on disk; host SHA check refuses)
    mutated = os.path.join(TMP, "mutated_tinyyolov2.onnx")
    data = bytearray(open(MODEL, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(mutated, "wb").write(data)
    res, _ = sh([binary, "--model", mutated, "--image", IMAGE, "--warmup", "0", "--iterations", "1"])
    record("d9_runtime_artifact_swap", [os.path.basename(binary), "--model", "<mutated>"],
           res, "runtime (generated-host load, before inference)", False, "runtime refusal")
    os.remove(mutated)

    caught = sum(1 for d in defects if d["exit_code"] != 0)
    print(f"defects caught before inference: {caught}/{len(defects)}")

    # --- 5. correctness comparison ------------------------------------------
    host_shape = host["result"]["shape"]
    ref_shape = reference["intermediate_shapes"]["permute_B_A_H_W_box"]
    comparison = {
        "evaluation_commit": head,
        "git_dirty": dirty,
        "host_output_shape": host_shape,
        "reference_decoded_shape": ref_shape,
        "output_shape_matches": host_shape == ref_shape,
        "model_correctness_oracle": {
            "source": "ONNX-bundled output_0.pb",
            "max_abs_diff": reference["oracle"]["max_abs_diff"],
            "tolerance": reference["oracle"]["tolerance"],
            "agrees": reference["oracle"]["agrees"],
        },
        "computed_shape_relationships": reference["computed_shape_relationships"],
        "host_output_checksum": host["result"]["checksum"],
        "note": (
            "The host reproduces the symbolic reshape+permute deterministically "
            "(stable checksum) and its output shape matches the Python reference's "
            "decoded intermediate. Numerical model correctness is anchored by the "
            "ONNX-bundled oracle; a full host-vs-Python tensor diff is bounded by "
            "image-resize algorithm differences (Rust `image` crate vs Pillow)."
        ),
    }

    # --- 6. write evidence ---------------------------------------------------
    rustc_v = subprocess.check_output(["rustc", "--version"]).decode().strip()
    cargo_v = subprocess.check_output(["cargo", "--version"]).decode().strip()
    binary_size = os.path.getsize(binary)
    model_size = os.path.getsize(MODEL)

    write = lambda name, obj: json.dump(obj, open(os.path.join(RESULTS, name), "w"), indent=2)

    write("environment.json", {
        "stark_commit": head,
        "git_dirty": dirty,
        "os": platform.system(),
        "arch": platform.machine(),
        "rustc_version": rustc_v,
        "cargo_version": cargo_v,
        "python_version": platform.python_version(),
        "onnxruntime_version": reference["environment"]["onnxruntime"],
        "model_sha256": sha256(MODEL),
        "image_sha256": sha256(IMAGE),
    })
    write("stark-host.json", {
        "entry": "infer",
        "pipeline": "examples/gate7/valid_pipeline.stark",
        "deploy_ms": round(deploy_ms, 1),
        "host_build_ms": round(build_ms, 1),
        "binary_size_bytes": binary_size,
        "model_size_bytes": model_size,
        "total_bundle_size_bytes": binary_size + model_size,
        "startup_ms": host["startup_ms"],
        "inference_ms": host["inference_ms"],
        "peak_rss_bytes": rss,
        "result": host["result"],
    })
    write("benchmark-runs.json", {
        "iterations": len(host["latencies_ms"]),
        "runs_ms": host["latencies_ms"],
    })
    write("defect-matrix.json", {
        "evaluation_commit": head,
        "defects_caught_before_inference": caught,
        "defect_count": len(defects),
        "defects": defects,
    })
    write("comparison.json", comparison)

    # exit summary markdown
    with open(os.path.join(RESULTS, "results_summary.md"), "w") as f:
        f.write("# Gate 7 (G7-05) — native deployment evidence\n\n")
        f.write(f"Commit `{head}` (dirty: {dirty})\n\n")
        f.write(f"- Deploy: {deploy_ms:.0f} ms · host build: {build_ms/1000:.1f} s\n")
        f.write(f"- Binary: {binary_size/1e6:.2f} MB · model: {model_size/1e6:.2f} MB\n")
        f.write(f"- Startup: {host['startup_ms']:.1f} ms · warm median: "
                f"{host['inference_ms']['median']:.2f} ms · peak RSS: "
                f"{(rss or 0)/1e6:.1f} MB\n")
        f.write(f"- Host output: {host['result']['dtype']} {host_shape} "
                f"(checksum {host['result']['checksum']}), shape matches reference: "
                f"{comparison['output_shape_matches']}\n")
        f.write(f"- Model oracle max_abs_diff: {reference['oracle']['max_abs_diff']:.2e} "
                f"(tol {reference['oracle']['tolerance']})\n")
        f.write(f"- Defects caught before inference: **{caught}/{len(defects)}**\n\n")
        f.write("| defect | stage | code | caught |\n|---|---|---|---|\n")
        for d in defects:
            f.write(f"| {d['case']} | {d['detection_stage']} | {d['diagnostic_code']} | "
                    f"{d['exit_code'] != 0} |\n")

    print(f"\nwrote evidence to {RESULTS}")
    ok = (comparison["output_shape_matches"] and reference["oracle"]["agrees"]
          and caught == len(defects))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
