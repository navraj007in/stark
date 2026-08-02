#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# **STARK's output contract is UTF-8 bytes; this script's stdout must be too.**
#
# The subprocess is already read as UTF-8, so `completed.stdout` is correct text. Writing it out is
# where it broke: Python encodes stdout using the CONSOLE's encoding, which on Windows is cp1252,
# and a STARK program that prints an emoji then dies here with UnicodeEncodeError -- in the script
# reporting the result, not in the program under test. Linux and macOS never showed it because
# their default is already UTF-8.
#
# `errors="replace"` rather than strict: this is a reporting path, and a byte it cannot render must
# not fail a qualification run that otherwise passed. The comparison against `expected_stdout`
# happens on the decoded text above, so substitution here cannot mask a real mismatch.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class PackageCase:
    package: str
    consumer: str
    expected_stdout: str
    # CD-347 — EXECUTED SURFACE. Names the resource types the package exposes, each of which must
    # have its acquire / use / close observed by a NATIVE consumer run.
    #
    # This exists because CD-345 found `stark-net` passing all seven steps while `connect`, `read`,
    # `write` and `close` had never been called by anything. The consumer only formatted addresses,
    # so a build-breaking defect (DEV-146) sat in the package undetected: nothing had ever lowered
    # a call into the raw bindings. A gate that a resource package can pass without exercising its
    # resources is not a gate for resource packages.
    #
    # Why a SEPARATE native consumer rather than folding it into the ordinary one: step 5 runs
    # `stark run`, and the interpreter has no provider layer — any consumer touching a bound
    # resource dies with "provider binding not lowered". So the resource exercise is native-only by
    # construction, and the split is forced by the toolchain, not chosen.
    #
    # Empty means "this package holds no host resources", which is the honest state for the ten
    # pure packages and must stay easy to declare.
    resources: tuple[str, ...] = ()
    resource_consumer: str | None = None
    resource_expected_stdout: str | None = None


CASES = [
    PackageCase(
        package="stark-json",
        consumer="stark-json-consumer",
        expected_stdout='{"name":"stark","items":[1,true,null],"unicode":"\U0001f600"}\n',
    ),
    PackageCase(
        package="stark-url",
        consumer="stark-url-consumer",
        expected_stdout="q=stark%20url&tag=compiler&tag=language&emoji=%F0%9F%98%80\n",
    ),
    PackageCase(
        package="stark-base64",
        consumer="stark-base64-consumer",
        expected_stdout="Zm9vYmFy\n",
    ),
    PackageCase(
        package="stark-hex",
        consumer="stark-hex-consumer",
        expected_stdout="48656c6c6f\n",
    ),
    PackageCase(
        package="stark-uuid",
        consumer="stark-uuid-consumer",
        expected_stdout="f81d4fae-7dec-11d0-a765-00a0c91e6bf6\n",
    ),
    # The HTTP substrate (CD-304), added CD-326. Nothing in CI had ever run these five — not
    # their tests, not `fmt --check`, not a native build — which is how three of them stayed
    # unformatted from the day they landed until CD-325, and how `stark-mime`, `stark-query`
    # and `stark-form` shipped with ZERO tests until CD-320.
    #
    # Their `expected_stdout` is a marker line rather than a computed result, because these
    # consumers are smoke tests for the package graph: what they prove is that the package
    # checks, tests, formats, and that its consumer runs identically through the interpreter
    # and as a native binary. The per-function behaviour is asserted by each package's own
    # `test_*` suite, which the `stark test` step above runs.
    PackageCase(
        package="stark-ascii",
        consumer="stark-ascii-consumer",
        expected_stdout="ASCII_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-percent",
        consumer="stark-percent-consumer",
        expected_stdout="PERCENT_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-mime",
        consumer="stark-mime-consumer",
        expected_stdout="MIME_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-query",
        consumer="stark-query-consumer",
        expected_stdout="QUERY_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-form",
        consumer="stark-form-consumer",
        expected_stdout="FORM_CONSUMER_OK\n",
    ),
    PackageCase(
        package="stark-net",
        consumer="stark-net-consumer",
        expected_stdout="STARK_NET_CONSUMER_OK\n",
        resources=("TcpStream",),
        resource_consumer="stark-net-resource-consumer",
        resource_expected_stdout="STARK_NET_RESOURCE_OK\n",
    ),
]


def run(cmd: list[str], cwd: Path, expected_stdout: str | None = None) -> None:
    label = f"{cwd.name}: {' '.join(cmd)}"
    print(f"::group::{label}", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sys.stdout.write(completed.stdout)
        if completed.returncode != 0:
            raise SystemExit(f"{label} failed with exit status {completed.returncode}")
        if expected_stdout is not None and completed.stdout != expected_stdout:
            raise SystemExit(
                f"{label} stdout mismatch\n"
                f"expected: {expected_stdout!r}\n"
                f"actual:   {completed.stdout!r}"
            )
    finally:
        print("::endgroup::", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stark", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[2], type=Path)
    parser.add_argument("--exe-suffix", default=".exe" if os.name == "nt" else "")
    args = parser.parse_args()

    stark = args.stark.resolve()
    repo_root = args.repo_root.resolve()

    for case in CASES:
        package_dir = repo_root / case.package
        consumer_dir = repo_root / case.consumer
        run([str(stark), "check"], package_dir)
        run([str(stark), "test"], package_dir)
        run([str(stark), "fmt", "--check"], package_dir)
        run([str(stark), "check"], consumer_dir)
        run([str(stark), "run"], consumer_dir, expected_stdout=case.expected_stdout)
        run([str(stark), "build", "--no-build-cache"], consumer_dir)
        artifact = consumer_dir / "target" / "stark" / "debug" / f"{case.consumer}{args.exe_suffix}"
        run([str(artifact)], consumer_dir, expected_stdout=case.expected_stdout)

        # CD-347: the executed-surface requirement. A package that declares resources must ship a
        # native consumer that acquires, uses and closes each one. `stark run` is deliberately NOT
        # part of this sequence -- the interpreter has no provider layer.
        if case.resources:
            if case.resource_consumer is None:
                raise SystemExit(
                    f"{case.package} declares resources {case.resources} but names no "
                    f"resource_consumer. A resource package must exercise acquire/use/close "
                    f"natively; see CD-345 for what a happy-path-only gate concealed."
                )
            resource_dir = repo_root / case.resource_consumer
            if not resource_dir.is_dir():
                raise SystemExit(
                    f"{case.package}: resource consumer {case.resource_consumer} does not exist"
                )
            run([str(stark), "check"], resource_dir)
            run([str(stark), "fmt", "--check"], resource_dir)
            run([str(stark), "build", "--no-build-cache"], resource_dir)
            resource_artifact = (
                resource_dir
                / "target"
                / "stark"
                / "debug"
                / f"{case.resource_consumer}{args.exe_suffix}"
            )
            run(
                [str(resource_artifact)],
                resource_dir,
                expected_stdout=case.resource_expected_stdout,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
