#!/usr/bin/env python3
"""Regenerate the non-normative combined Core v1 Markdown, HTML, and PDF views."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "spec"
SOURCES = sorted(SPEC.glob("0[0-7]-*.md"))
COMBINED = SPEC / "STARK-Core-v1.md"


def main() -> None:
    if len(SOURCES) != 8:
        raise SystemExit(f"expected 8 numbered Core sources, found {len(SOURCES)}")

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
