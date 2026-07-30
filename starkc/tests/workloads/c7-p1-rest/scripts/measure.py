#!/usr/bin/env python3
"""Reproducible local measurement harness for the C7 P1 workload."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from pathlib import Path


def run(command: list[str], cwd: Path) -> tuple[float, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return time.perf_counter() - started, completed.stdout


def line_count(paths: list[Path]) -> int:
    return sum(len(path.read_text().splitlines()) for path in paths)


def main() -> int:
    script = Path(__file__).resolve()
    package = script.parents[1]
    compiler_root = script.parents[4]
    compiler = compiler_root / "target" / "debug" / "stark"
    build = [str(compiler), "build", "--emit-rust", "--verbose"]

    cold_seconds, build_output = run(build + ["--no-build-cache"], package)
    generated_line = next(
        line for line in build_output.splitlines() if "generated crate:" in line
    )
    generated_root = Path(generated_line.split("generated crate:", 1)[1].strip())
    generated_rust = generated_root / "src" / "main.rs"
    warm_samples = [run(build, package)[0] for _ in range(3)]
    e2e_samples = [
        run(["python3", str(package / "scripts" / "e2e.py")], package)[0]
        for _ in range(5)
    ]
    source_files = sorted((package / "src").glob("*.stark"))
    binary = package / "target" / "stark" / "debug" / "c7-p1-rest"
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=compiler_root.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rustc = subprocess.run(
        ["rustc", "-Vv"], check=True, capture_output=True, text=True
    ).stdout.strip()
    result = {
        "schema": "stark.c7.p1.measurement.v1",
        "compiler_commit": git_sha,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "rustc": rustc,
        "profile": "debug",
        "request_count": 24,
        "concurrency": 1,
        "source_lines": line_count(source_files),
        "generated_rust_lines": line_count([generated_rust]),
        "binary_bytes": binary.stat().st_size,
        "cold_build_seconds": cold_seconds,
        "no_change_build_seconds": {
            "samples": warm_samples,
            "median": statistics.median(warm_samples),
        },
        "functional_run_seconds": {
            "samples": e2e_samples,
            "median": statistics.median(e2e_samples),
            "requests_per_second_median_run": 24 / statistics.median(e2e_samples),
        },
        "warmup": "one cold build before three cached builds; each e2e run launches a fresh server",
        "correctness": "every timed e2e sample validates all 24 raw responses byte-for-byte",
        "generated_crate_preserved": True,
    }
    output = package / "measurements" / "latest.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
