#!/usr/bin/env python3
"""Regenerate the non-normative combined Core v1 Markdown, HTML, and PDF views.

`--check` verifies the combined Markdown on disk matches what the normative sources
would regenerate, without writing anything or invoking pandoc/weasyprint (exit 1 on
drift). CI uses this for the C3-ENTRY "spec regeneration consistency" baseline check;
the HTML/PDF views are excluded because their toolchain output is not byte-reproducible.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "spec"
SOURCES = [
    SPEC / "00-Core-Language-Overview.md",
    SPEC / "01-Lexical-Grammar.md",
    SPEC / "02-Syntax-Grammar.md",
    SPEC / "03-Type-System.md",
    SPEC / "04-Semantic-Analysis.md",
    SPEC / "CORE-V1-ABSTRACT-MACHINE.md",
    SPEC / "05-Memory-Model.md",
    SPEC / "06-Standard-Library.md",
    SPEC / "07-Modules-and-Packages.md",
    SPEC / "CORE-V1-FUTURE-BOUNDARIES.md",
]
COMBINED = SPEC / "STARK-Core-v1.md"


def main() -> None:
    missing = [str(path) for path in SOURCES if not path.is_file()]
    if missing:
        raise SystemExit("missing Core source(s): " + ", ".join(missing))

    combined = "\n\n\n---\n\n".join(path.read_text(encoding="utf-8").rstrip() for path in SOURCES)

    if "--check" in sys.argv[1:]:
        on_disk = COMBINED.read_text(encoding="utf-8") if COMBINED.is_file() else ""
        if on_disk != combined + "\n":
            raise SystemExit(
                f"{COMBINED} is out of sync with the normative sources; "
                "rerun STARKLANG/tools/build-core-spec.py"
            )
        print(f"{COMBINED} is in sync with the normative sources")
        return

    COMBINED.write_text(combined + "\n", encoding="utf-8")

    subprocess.run(
        [
            "pandoc",
            str(COMBINED),
            "--standalone",
            "--metadata",
            "title=STARK-Core-v1",
            "-o",
            str(COMBINED.with_suffix(".html")),
        ],
        check=True,
    )
    subprocess.run(
        [
            "pandoc",
            str(COMBINED),
            "--pdf-engine=weasyprint",
            "--metadata",
            "title=STARK-Core-v1",
            "-o",
            str(COMBINED.with_suffix(".pdf")),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
