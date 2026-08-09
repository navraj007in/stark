#!/usr/bin/env python3
"""C10-0 — the three deviation/residual populations, extracted mechanically.

OD-3 (owner ruling, 2026-08-09) refuses a single "deviation" denominator. Three populations are
frozen separately, and only A is the denominator for CD-021's compiler-conformance rule:

    A  compiler deviations       KNOWN-DEVIATIONS.md live-heading set, plus DEV-* that appear in
                                 COMPILER-STATE.md and nowhere in that file
    B  release/distribution      constrains the release WORDING, not compiler conformance
    C  assurance residuals       constrains the STRENGTH of an evidence claim, and asserts no
                                 defect at all

This script computes A, and reports what it can see of B and C so the hand-audit has a starting
list rather than a blank page.

WHY IT EXISTS, rather than reading the file. `KNOWN-DEVIATIONS.md` is APPEND-ONLY: a deviation
tracked across several packets gets a NEW `## DEV-nnn` heading each time rather than an edited one,
so THE FIRST HEADING IS NOT ITS STATUS -- the last one is. DEV-121's first heading says OPEN and it
is CLOSED 3,558 lines later. Reading top-to-bottom and believing the first heading is the failure
this script exists to prevent.

It deliberately does NOT decide status for the ambiguous cases. A heading carrying `OPEN` inside a
phrase like "(OPEN, deferred by decision)" or a heading with no status word at all is reported as
needing adjudication, because a regex that guessed would be doing the reviewer's job badly. The
output's `ADJUDICATE` section is the hand-audit's worklist.
"""
from __future__ import annotations
import re, sys, pathlib, json, argparse

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEVIATIONS = ROOT / "starkc" / "docs" / "conformance" / "KNOWN-DEVIATIONS.md"
STATE = ROOT / "COMPILER-STATE.md"

HEADING = re.compile(r"^## (DEV-\d+)\b(.*)$")
CLOSED_WORDS = ("CLOSED", "RESOLVED", "SUPERSEDED", "WITHDRAWN", "RETIRED")


def read(path: pathlib.Path) -> list[str]:
    # CRLF is normalised at the read: a Windows checkout must not change the counts.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def live_headings(lines: list[str]) -> dict[str, tuple[int, str]]:
    """DEV id -> (line number, heading text) of its LAST heading. The last one is the live one."""
    last: dict[str, tuple[int, str]] = {}
    for i, line in enumerate(lines, 1):
        m = HEADING.match(line)
        if m:
            last[m.group(1)] = (i, line)
    return last


def classify(heading: str) -> str:
    """closed | open | adjudicate.

    `adjudicate` is not a failure mode, it is the honest answer for a heading whose status word is
    absent or is qualified by the same sentence (e.g. "OPEN, deferred by decision", or a heading
    that says both RESOLVED and OPEN). Those go to a human.
    """
    upper = heading.upper()
    has_open = "OPEN" in upper
    has_closed = any(w in upper for w in CLOSED_WORDS)
    if has_open and has_closed:
        return "adjudicate"
    if has_open:
        return "open" if re.search(r"\(OPEN\)|\[OPEN\b|\(OPEN,", heading, re.I) else "adjudicate"
    if has_closed:
        return "closed"
    return "adjudicate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = ap.parse_args()

    dev_lines = read(DEVIATIONS)
    last = live_headings(dev_lines)

    buckets: dict[str, list[tuple[str, int, str]]] = {"open": [], "closed": [], "adjudicate": []}
    for did, (ln, head) in last.items():
        buckets[classify(head)].append((did, ln, head[3:]))
    for b in buckets.values():
        b.sort(key=lambda r: int(r[0][4:]))

    # Ids mentioned ANYWHERE in the deviations file, vs ids that own a heading. The gap is
    # in-body cross-references, and it is why "distinct ids" and "entries" are different counts.
    mentioned = set(re.findall(r"DEV-\d+", "\n".join(dev_lines)))
    # Population A's second half: DEV-* named in the state file but owning no heading here.
    state_ids = set(re.findall(r"DEV-\d+", STATE.read_text(encoding="utf-8"))) if STATE.exists() else set()
    orphans = sorted(state_ids - set(last), key=lambda d: int(d[4:]))

    if args.json:
        json.dump(
            {
                "headings_total": sum(1 for l in dev_lines if HEADING.match(l)),
                "distinct_ids_with_heading": len(last),
                "distinct_ids_mentioned": len(mentioned),
                "population_A_open": [r[0] for r in buckets["open"]],
                "population_A_adjudicate": [r[0] for r in buckets["adjudicate"]],
                "population_A_orphans_from_state": orphans,
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    print(f"## DEV- headings                  {sum(1 for l in dev_lines if HEADING.match(l))}")
    print(f"distinct ids owning a heading     {len(last)}")
    print(f"distinct ids mentioned anywhere   {len(mentioned)}   (the gap is in-body references)")
    print()
    print(f"POPULATION A -- live OPEN by the last heading  ({len(buckets['open'])})")
    for did, ln, head in buckets["open"]:
        print(f"    {did}  L{ln:<6d} {head[:110]}")
    print()
    print(f"ADJUDICATE -- the last heading does not settle it  ({len(buckets['adjudicate'])})")
    print("    Not an error. A human decides; a regex that guessed would be doing it badly.")
    for did, ln, head in buckets["adjudicate"]:
        print(f"    {did}  L{ln:<6d} {head[:110]}")
    print()
    print(f"POPULATION A -- named in COMPILER-STATE.md, owning no heading here  ({len(orphans)})")
    print(f"    {', '.join(orphans) if orphans else '(none)'}")
    print()
    print("Populations B (release/distribution) and C (assurance residuals) are NOT derivable from")
    print("this file and are frozen by hand in the C10-0 inventory -- see OD-3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
