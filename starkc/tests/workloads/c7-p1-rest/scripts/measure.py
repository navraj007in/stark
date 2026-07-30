#!/usr/bin/env python3
"""Reproducible local measurement harness for the C7 P1 workload."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
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


def executable(path: Path) -> Path:
    """`path` with the platform's executable suffix."""
    return path.with_name(path.name + ".exe") if sys.platform == "win32" else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("debug", "release"),
        default="debug",
        help="STARK build profile to measure. C7.5's debug/release runtime ratio needs both.",
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        default=None,
        help="path to the `stark` binary; defaults to the repo's release build",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the record; defaults to measurements/<profile>.json",
    )
    args = parser.parse_args()

    script = Path(__file__).resolve()
    package = script.parents[1]
    compiler_root = script.parents[4]
    compiler = args.compiler or executable(compiler_root / "target" / "release" / "stark")
    build = [str(compiler), "build", "--emit-rust", "--verbose"]
    if args.profile == "release":
        build.append("--release")

    cold_seconds, build_output = run(build + ["--no-build-cache"], package)
    generated_line = next(
        line for line in build_output.splitlines() if "generated crate:" in line
    )
    generated_root = Path(generated_line.split("generated crate:", 1)[1].strip())
    generated_rust = generated_root / "src" / "main.rs"
    warm_samples = [run(build, package)[0] for _ in range(3)]
    # `sys.executable`, not "python3": Windows CI has no `python3` on PATH, and a measurement
    # harness that only runs on two of three Tier-1 platforms cannot close a cross-platform row.
    e2e = [sys.executable, str(package / "scripts" / "e2e.py"), "--profile", args.profile]
    e2e_samples = [run(e2e, package)[0] for _ in range(5)]
    source_files = sorted((package / "src").glob("*.stark"))
    binary = executable(package / "target" / "stark" / args.profile / "c7-p1-rest")
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
        "profile": args.profile,
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
    output = args.output or package / "measurements" / f"{args.profile}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
