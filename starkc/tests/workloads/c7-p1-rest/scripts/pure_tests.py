#!/usr/bin/env python3
"""Run the P1 pure-STARK tests without loading host capability declarations."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve()
    package = script.parents[1]
    compiler = script.parents[4] / "target" / "debug" / "stark"
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
