#!/usr/bin/env python3
"""Run the P1 pure-STARK tests without loading host capability declarations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_compiler(root: Path) -> Path:
    """The `stark` binary, whichever profile is built.

    Defaulting to `target/debug/stark` made this script depend on a profile the caller never
    promised: CI builds `--release`, and the script died with `FileNotFoundError` before running a
    single test. Release is preferred (it is what a qualification run builds), debug is accepted,
    and a missing binary is reported as itself rather than as a stack trace.
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [
        root / "target" / profile / f"stark{suffix}" for profile in ("release", "debug")
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "no `stark` binary found; build one first. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler", type=Path, default=None, help="path to the `stark` binary"
    )
    args = parser.parse_args()
    script = Path(__file__).resolve()
    package = script.parents[1]
    # Resolved to an absolute path: the tests run from a temporary directory, so a relative
    # `--compiler` would be interpreted against the wrong root and fail as "not found".
    compiler = (
        args.compiler.resolve() if args.compiler else find_compiler(script.parents[4])
    )
    with tempfile.TemporaryDirectory(prefix="stark-c7-p1-pure-") as temporary:
        root = Path(temporary)
        shutil.copytree(package / "src", root / "src")
        (root / "starkpkg.json").write_text(
            json.dumps(
                {
                    "name": "c7-p1-rest-pure-tests",
                    "version": "0.1.0",
                    "entry": "src/pure.stark",
                    "dependencies": {},
                },
                indent=2,
            )
            + "\n"
        )
        (root / "stark.lock").write_text('{"packages":[]}\n')
        completed = subprocess.run(
            [str(compiler), "test", "--show-output"],
            cwd=root,
            check=False,
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
