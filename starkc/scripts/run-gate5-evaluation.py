#!/usr/bin/env python3
import os
import sys
import json
import time
import platform
import subprocess
import shutil

def get_git_info():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
        is_dirty = len(status) > 0
        return head, is_dirty
    except Exception:
        return "unknown", True

def get_sys_info():
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    return info

def run_command(args, cwd=None):
    t0 = time.time()
    res = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    t1 = time.time()
    return res, (t1 - t0) * 1000.0

def run_with_rss(args, os_name):
    if os_name == "Darwin":
        cmd = ["/usr/bin/time", "-l"] + args
    elif os_name == "Linux":
        cmd = ["/usr/bin/time", "-v"] + args
    else:
        cmd = args
        
    res = subprocess.run(cmd, capture_output=True, text=True)
    rss = None
    # Parse RSS from stderr
    for line in res.stderr.splitlines():
        if "maximum resident set size" in line:
            parts = line.strip().split()
            if parts:
                rss = int(parts[0]) # In bytes on macOS
        elif "Maximum resident set size" in line:
            parts = line.strip().split(":")
            if len(parts) > 1:
                rss = int(parts[1].strip()) * 1024 # Convert KB to bytes on Linux
    return res, rss

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    starkc_dir = os.path.dirname(script_dir)
    
    model_path = os.path.join(starkc_dir, "tmp", "resnet50-v1-7.onnx")
    image_path = os.path.join(starkc_dir, "tmp", "dog.jpg")
    venv_python = os.path.join(starkc_dir, "tmp", "venv", "bin", "python3")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run download first.")
        sys.exit(1)
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}. Run download first.")
        sys.exit(1)
        
    out_dir = os.path.join(starkc_dir, "tmp", "eval-host")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        
    print("=== 1. STARK Deploy ===")
    args = [
        "cargo", "run", "--", "deploy",
        "examples/gate5/valid_pipeline.stark",
        "--model", model_path,
        "--entry", "infer",
        "--out", out_dir,
        "--force"
    ]
    res, deploy_time = run_command(args, cwd=starkc_dir)
    if res.returncode != 0:
        print("Deploy failed:")
        print(res.stderr)
        sys.exit(1)
    print(f"STARK Deploy completed in {deploy_time:.1f}ms")
    
    print("=== 2. Rust Build ===")
    args = [
        "cargo", "build", "--release", "--locked",
        "--manifest-path", os.path.join(out_dir, "Cargo.toml")
    ]
    res, build_time = run_command(args, cwd=starkc_dir)
    if res.returncode != 0:
        print("Build failed:")
        print(res.stderr)
        sys.exit(1)
    print(f"Rust release build completed in {build_time:.1f}ms")
    
    binary_path = os.path.join(out_dir, "target", "release", "stark-resnet50")
    binary_size = os.path.getsize(binary_path)
    model_size = os.path.getsize(model_path)
    
    print("=== 3. Running STARK Native Inference (with RSS collection) ===")
    run_args = [
        binary_path,
        "--model", model_path,
        "--image", image_path,
        "--warmup", "5",
        "--iterations", "100",
        "--json"
    ]
    os_name = platform.system()
    res, rss = run_with_rss(run_args, os_name)
    if res.returncode != 0:
        print("Inference run failed:")
        print(res.stderr)
        sys.exit(1)
        
    try:
        host_results = json.loads(res.stdout.strip())
    except Exception as e:
        print("Failed to parse JSON output from compiled host:")
        print(res.stdout)
        sys.exit(1)
        
    print("=== 4. Running Python ORT Reference ===")
    ref_args = [
        venv_python,
        os.path.join(starkc_dir, "tests", "fixtures", "gate5", "reference.py"),
        model_path,
        image_path
    ]
    ref_res, ref_time = run_command(ref_args)
    if ref_res.returncode != 0:
        print("Python reference failed:")
        print(ref_res.stderr)
        sys.exit(1)
        
    # Get top-1 class and prob from python stdout
    py_class = None
    py_prob = None
    for line in ref_res.stdout.splitlines():
        if line.startswith("top-1 class :"):
            py_class = int(line.split(":")[1].strip())
        elif line.startswith("probability :"):
            py_prob = float(line.split(":")[1].strip())
            
    git_head, git_dirty = get_git_info()
    sys_info = get_sys_info()
    
    # Save standard results files under docs/
    results_json = {
        "metadata": {
            "git_commit": git_head,
            "git_dirty": git_dirty,
            "system": sys_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "artifacts": {
            "binary_size_bytes": binary_size,
            "model_size_bytes": model_size,
            "total_bundle_size_bytes": binary_size + model_size
        },
        "performance": {
            "host": host_results,
            "peak_rss_bytes": rss,
            "reference_python": {
                "top1_class": py_class,
                "top1_probability": py_prob,
                "duration_ms": ref_time,
                "stdout": ref_res.stdout.strip()
            }
        }
    }
    
    with open(os.path.join(starkc_dir, "docs", "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
        
    # 5. Populate the exact files in starkc/tests/results/gate5/
    results_dir = os.path.join(starkc_dir, "tests", "results", "gate5")
    os.makedirs(results_dir, exist_ok=True)
    
    # Get rustc and cargo versions
    rustc_ver = subprocess.check_output(["rustc", "--version"]).decode().strip()
    cargo_ver = subprocess.check_output(["cargo", "--version"]).decode().strip()
    
    # environment.json
    env_data = {
        "stark_commit": git_head,
        "git_dirty": git_dirty,
        "os": sys_info["os"],
        "arch": sys_info["arch"],
        "processor": sys_info["processor"],
        "rustc_version": rustc_ver,
        "cargo_version": cargo_ver,
        "python_version": sys_info["python_version"],
        "onnxruntime_version": "1.27.0"
    }
    with open(os.path.join(results_dir, "environment.json"), "w") as f:
        json.dump(env_data, f, indent=2)
        
    # rust-host.json
    rust_host_data = {
        "binary_size_bytes": binary_size,
        "model_size_bytes": model_size,
        "total_bundle_size_bytes": binary_size + model_size,
        "startup_ms": host_results["startup_ms"],
        "image_decode_ms": None, # Kept placeholder for now
        "latency_min_ms": host_results["inference_ms"]["min"],
        "latency_median_ms": host_results["inference_ms"]["median"],
        "latency_p95_ms": host_results["inference_ms"]["p95"],
        "peak_rss_bytes": rss
    }
    with open(os.path.join(results_dir, "rust-host.json"), "w") as f:
        json.dump(rust_host_data, f, indent=2)
        
    # python-reference.json
    py_ref_data = {
        "top1_index": py_class,
        "top1_probability": py_prob,
        "latency_ms": ref_time
    }
    with open(os.path.join(results_dir, "python-reference.json"), "w") as f:
        json.dump(py_ref_data, f, indent=2)
        
    # benchmark-runs.json
    runs_data = {
        "iterations": len(host_results["latencies_ms"]),
        "runs_ms": host_results["latencies_ms"]
    }
    with open(os.path.join(results_dir, "benchmark-runs.json"), "w") as f:
        json.dump(runs_data, f, indent=2)
        
    # comparison.json
    idx_match = host_results["top1_index"] == py_class
    prob_diff = abs(host_results["top1_probability"] - py_prob)
    tol_passed = prob_diff <= 1e-3
    comparison_data = {
        "top1_index_match": idx_match,
        "prob_diff": prob_diff,
        "absolute_tolerance_passed": tol_passed
    }
    with open(os.path.join(results_dir, "comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)
        
    # Markdown summary
    summary_md_path = os.path.join(starkc_dir, "docs", "results_summary.md")
    with open(summary_md_path, "w") as f:
        f.write("# STARK Gate 5 Evaluation Summary\n\n")
        f.write(f"**Timestamp:** {results_json['metadata']['timestamp']}\n")
        f.write(f"**Commit:** `{git_head}` (dirty: {git_dirty})\n\n")
        f.write("## System Environment\n")
        f.write(f"- OS: {sys_info['os']} ({sys_info['arch']})\n")
        f.write(f"- Python: {sys_info['python_version']}\n")
        f.write(f"- rustc: `{rustc_ver}`\n")
        f.write(f"- cargo: `{cargo_ver}`\n\n")
        f.write("## Artifact Metrics\n")
        f.write(f"- Generated Host Release Binary: {binary_size / 1024 / 1024:.2f} MB\n")
        f.write(f"- ONNX Model: {model_size / 1024 / 1024:.2f} MB\n")
        f.write(f"- Total Bundle: {(binary_size + model_size) / 1024 / 1024:.2f} MB\n\n")
        f.write("## Inference Performance\n")
        f.write(f"- Top-1 Class: {host_results['top1_index']}\n")
        f.write(f"- Top-1 Probability: {host_results['top1_probability']:.6f}\n")
        f.write(f"- Host Session Startup: {host_results['startup_ms']:.2f} ms\n")
        f.write(f"- Peak RSS: {rss / 1024 / 1024:.2f} MB ({rss} bytes)\n")
        f.write(f"- Latency (100 iterations):\n")
        f.write(f"  - Min: {host_results['inference_ms']['min']:.3f} ms\n")
        f.write(f"  - Median: {host_results['inference_ms']['median']:.3f} ms\n")
        f.write(f"  - P95: {host_results['inference_ms']['p95']:.3f} ms\n")
        
    print("Evaluation completed successfully! Results written to docs/results.json and docs/results_summary.md")

if __name__ == '__main__':
    main()
