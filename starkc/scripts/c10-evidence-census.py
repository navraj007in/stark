#!/usr/bin/env python3
"""C10-A1 -- the conformance-evidence integrity census.

THE QUESTION. For each normative rule C10 might make a claim about: does executable evidence
exist, at what precision, and can that evidence disagree with the implementation?

THE DENOMINATOR, declared in C10-0 BEFORE this ran and not chosen after seeing the result:

    the 161 granular IDs in semantic-freeze/CORE-V1-COMPLETENESS.md, the inventory of record.
    NO EXCLUSIONS -- rules classed intentionally-deferred / prohibited / spec-defect stay IN the
    denominator and are bucketed N/A with the reason, so the denominator cannot shrink to flatter
    the result.

FOUR BUCKETS, and the one that matters is the second:

    PRECISE     positive AND negative evidence at test-FUNCTION precision
                (core-v1-c2.11-evidence.toml)
    AGGREGATE   evidence exists but only as a file-level or aggregate-runner citation. The legacy
                database's own header concedes it "does not distinguish positive from negative
                coverage, and often cites only the aggregate starkc/tests/conformance.rs
                fixture-corpus runner with no per-rule attribution within it" -- DEV-017
    ABSENT      checked, and confirmed to have no evidence in that category
    N/A         not a claimable behaviour: prohibited, deliberately-unspecified,
                intentionally-deferred, or a recorded spec defect

AGGREGATE IS NOT COVERAGE. It is UNCLASSIFIED. Reporting it as covered is the specific error this
census exists to prevent, and promoting a rule from AGGREGATE to PRECISE requires NAMING A TEST
FUNCTION -- never inspection, never inference from a file name.

THE FORCING MECHANISM IS THE ENUMERATION, NOT THE YIELD. This census may legitimately conclude that
zero classifications changed. What it must demonstrate is that it actually enumerated the intended
population, so `--self-test` injects a rule citing a test function that does not exist and requires
the census to report it. A census with no negative control is the same error EI2 made: it read the
machinery instead of enumerating the corpus, and missed a control that was sitting in the tree.
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INVENTORY = ROOT / "STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md"
C211 = ROOT / "STARKLANG/conformance/core-v1-c2.11-evidence.toml"
LEGACY = ROOT / "STARKLANG/conformance/core-v1-coverage.toml"

# An ID is TWO OR MORE dash-separated segments then a three-digit number. The first version
# of this regex allowed exactly two segments and silently dropped every three-segment ID --
# `NUM-INT-ARITH-001`, `NUM-FLOAT-OP-001` and their kin -- which understated the denominator
# by five and made five c2.11 rules look like citations to a nonexistent inventory entry.
# Caught by cross-checking c2.11's ids against this file rather than by reading the regex.
ROW = re.compile(r"^\|\s*([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+-\d{3})\s*\|(.*)$")
NA_CLASSES = ("prohibited", "deliberately-unspecified", "intentionally-deferred", "spec-defect")


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def inventory_rows() -> list[dict]:
    """Every granular ID, with the columns the census reads. One row per ID, in file order."""
    rows, seen = [], set()
    for line in read(INVENTORY).split("\n"):
        m = ROW.match(line)
        if not m:
            continue
        rid = m.group(1)
        if rid in seen:      # an ID appearing twice is itself a finding; keep the first
            continue
        seen.add(rid)
        cells = [c.strip() for c in m.group(2).split("|")]
        rows.append({
            "id": rid,
            "question": cells[0] if len(cells) > 0 else "",
            "status_class": cells[1] if len(cells) > 1 else "",
            "home": cells[2] if len(cells) > 2 else "",
            "evidence": cells[3] if len(cells) > 3 else "",
            "decision": cells[5] if len(cells) > 5 else "",
        })
    return rows


def c211_precise() -> dict[str, dict]:
    """Rules with test-FUNCTION-precision positive and negative evidence."""
    out, cur = {}, None
    for line in read(C211).split("\n"):
        if line.strip() == "[[rule]]":
            cur = {}
            continue
        if cur is None:
            continue
        m = re.match(r'\s*id\s*=\s*"([^"]+)"', line)
        if m:
            cur["id"] = m.group(1); out[cur["id"]] = cur; continue
        for field in ("positive_tests", "negative_tests"):
            m = re.match(rf"\s*{field}\s*=\s*\[(.*)\]", line)
            if m:
                cur[field] = re.findall(r'"([^"]+)"', m.group(1))
    return out


def citation_exists(citation: str) -> tuple[bool, str]:
    """`path` must exist; `path::fn_name` must also contain `fn fn_name`.

    This is the check that makes a citation evidence rather than a promise, and it is the same
    rule check-conformance.py applies -- a renamed or deleted TEST, not merely a renamed file.
    """
    if "::" in citation:
        path, fn = citation.split("::", 1)
    else:
        path, fn = citation, None
    p = ROOT / path
    if not p.exists():
        return False, f"path does not exist: {path}"
    if fn:
        if p.is_dir():
            return False, f"function citation on a directory: {citation}"
        if not re.search(rf"\bfn\s+{re.escape(fn)}\b", read(p)):
            return False, f"no `fn {fn}` in {path}"
    return True, ""


def classify(row: dict, precise: dict) -> tuple[str, str]:
    rid = row["id"]
    if rid in precise:
        return "PRECISE", "core-v1-c2.11-evidence.toml"
    cls = row["status_class"].lower()
    if any(k in cls for k in NA_CLASSES):
        return "N/A", row["status_class"]
    ev = row["evidence"].lower()
    if ev in ("none", "none; none", "", "-"):
        return "ABSENT", "evidence column records none"
    if "u17" in ev or re.search(r"\b(lex|syn|type|sem|mem|std|mod)-\d{3}\b", ev):
        return "AGGREGATE", "legacy citation only (DEV-017)"
    if "none" in ev:
        return "AGGREGATE", "one side recorded none; the other is a legacy citation"
    return "AGGREGATE", "unrecognised evidence form -- treated as unclassified, never as covered"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="inject a citation to a test function that does not exist and require "
                         "the census to report it")
    args = ap.parse_args()

    rows = inventory_rows()
    precise = c211_precise()

    if args.self_test:
        print("NEGATIVE CONTROL -- injecting a citation to a function that does not exist\n")
        precise = dict(precise)
        precise["__INJECTED__"] = {
            "id": "__INJECTED__",
            "positive_tests": ["starkc/tests/conformance.rs::a_function_that_does_not_exist"],
            "negative_tests": [],
        }

    # Every citation in the PRECISE set is verified to resolve. This is the census's own control:
    # an unresolvable citation must be REPORTED, not silently counted as precise evidence.
    broken = []
    for rid, rule in sorted(precise.items()):
        for field in ("positive_tests", "negative_tests"):
            for c in rule.get(field, []):
                ok, why = citation_exists(c)
                if not ok:
                    broken.append((rid, field, c, why))

    buckets: dict[str, list] = {"PRECISE": [], "AGGREGATE": [], "ABSENT": [], "N/A": []}
    for row in rows:
        b, why = classify(row, precise)
        buckets[b].append((row["id"], why, row["home"]))

    if args.json:
        json.dump({
            "denominator": len(rows),
            "buckets": {k: [r[0] for r in v] for k, v in buckets.items()},
            "broken_citations": [{"rule": r, "field": f, "citation": c, "why": w}
                                 for r, f, c, w in broken],
        }, sys.stdout, indent=2)
        print()
        return 1 if broken and not args.self_test else 0

    print(f"DENOMINATOR: {len(rows)} granular rules "
          f"(semantic-freeze/CORE-V1-COMPLETENESS.md; declared in C10-0, no exclusions)\n")
    total = len(rows)
    for b in ("PRECISE", "AGGREGATE", "ABSENT", "N/A"):
        n = len(buckets[b])
        pct = 100.0 * n / total if total else 0.0
        print(f"  {b:<10} {n:>4}   {pct:5.1f}%")
    print()
    print("AGGREGATE is UNCLASSIFIED, not covered. Do not add it to PRECISE to make a percentage.")
    print()
    print(f"CITATION INTEGRITY -- every PRECISE citation resolved to a real `fn`: "
          f"{'FAIL' if broken else 'PASS'}")
    for rid, field, c, why in broken:
        print(f"    {rid:<24} {field:<15} {c}\n        {why}")
    if args.self_test:
        hit = any(r == "__INJECTED__" for r, _, _, _ in broken)
        print()
        print(f"NEGATIVE CONTROL: {'PASS -- the injected citation was reported' if hit else 'FAIL -- the census did not notice'}")
        return 0 if hit else 1
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
