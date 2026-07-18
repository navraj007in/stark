#!/usr/bin/env python3
"""Regenerate the non-normative combined Core v1 Markdown, HTML, and PDF views."""

from pathlib import Path
import subprocess


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
