#!/usr/bin/env python3
"""C10-A2 — generate the conformance dashboard from resolved evidence.

**This does not transcribe C10-A1's buckets, and that prohibition is the packet's central rule.**
A1 measured the INVENTORY's citation state; A1-F3 proved those states are not evidence states
(`EXT-ISOLATION-001` records `none; none` while nine tests run in CI). Every row here is resolved
against the TREE by `c10-a2-resolve.py`, or it says `UNRESOLVED` — never `none`.

Columns are plan §6.5's, in its order. Two are deliberately not auto-filled:

    mutation/challenge status   inherited from AS8 where a trial exists; otherwise NOT-CHALLENGED.
                                Nothing here invents a challenge that was not run
    independent control         `none` is a legitimate and expected value -- six of eleven ESF
                                authorities are INVISIBLE to all three engines. A dashboard with
                                no `none` cells has been massaged

NO SINGLE HEADLINE PERCENTAGE IS EMITTED. Plan §7.3 forbids mixing precise and unclassified rows
into one number, and §6.5 forbids a glossy figure that hides evidence quality.
"""
from __future__ import annotations
import json, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
INVENTORY = ROOT / "STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md"
ESF = ROOT / "STARKLANG/docs/compiler/engine-shared-fate.json"
OUT_JSON = ROOT / "STARKLANG/conformance/c10-dashboard.json"

ID = r"[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+-\d{3}"
ROW = re.compile(r"^\|\s*(" + ID + r")\s*\|(.*)$")

#: AS8 trials, by the normative rule family they targeted. Inherited verbatim -- C10 re-runs
#: nothing that is FRESH under plan §8.2a, and C10-0 verified all 12 authority files and 13 control
#: suites hash identically at `e7bb95d` and at the baseline.
AS8_CHALLENGED = {
    "OWN-COPY-001": "AS8-MUT-003/005/006 SURVIVED the differential; 009/010/011 KILLED by c61f_structural_copy",
    "OWN-DROP-001": "AS8-MUT-002 KILLED (25) — via the Copy+Drop CONTRADICTION, not a wrong destructor set",
    "TYPE-PRIM-001": "AS8-MUT-013 SURVIVED with the spec fixtures selected (AS8-R5)",
    "TRAIT-CORE-001": "AS8-MUT-014/015 SURVIVED — ESF-TRAIT-001 has NO control of any kind (AS8-R10)",
    "TRAP-CATEGORY-001": "AS8-MUT-007 KILLED (4) by the HIR oracle; MUT-008 the honest no-op (AS8-R2)",
}


def inventory_rows() -> dict[str, dict]:
    rows, seen = {}, set()
    for line in INVENTORY.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"):
        m = ROW.match(line)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        cells = [c.strip() for c in m.group(2).split("|")]
        rows[m.group(1)] = {
            "question": cells[0] if cells else "",
            "status_class": cells[1] if len(cells) > 1 else "",
            "home": cells[2] if len(cells) > 2 else "",
        }
    return rows


def esf_by_rule() -> dict[str, list[str]]:
    """ESF authorities, keyed by the normative rule each names, so a row can carry its shared fate."""
    try:
        data = json.loads(ESF.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    blob = json.dumps(data)
    for rid in set(re.findall(ID, blob)):
        entries = [e for e in json.dumps(data, indent=0).split("\n") if rid in e]
        if entries:
            out.setdefault(rid, []).append("see engine-shared-fate.json")
    return out


def main() -> int:
    inv = inventory_rows()
    resolved = json.loads(
        subprocess.run(
            ["python3", str(ROOT / "starkc/scripts/c10-a2-resolve.py"), "--json"],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    esf = esf_by_rule()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=ROOT).stdout.strip()

    rows = []
    for rid, meta in inv.items():
        r = resolved.get(rid, {})
        sites = r.get("function_level", [])
        rows.append({
            "rule": rid,
            "home": meta["home"],
            "question": meta["question"],
            "status_class": meta["status_class"],
            "implementation": sorted({p for p in r.get("implementation_only", [])})[:4],
            "evidence_state": r.get("state", "UNRESOLVED"),
            "positive_negative_sites": [f"{h['file']}::{h['fn']}" for h in sites][:6],
            "engines": sorted({h["class"] for h in sites}) or ["-"],
            "evidence_class": evidence_class(sites, r.get("state", "UNRESOLVED")),
            "shared_fate": esf.get(rid, []),
            "independent_control": independent_control(sites),
            "challenge": AS8_CHALLENGED.get(rid, "NOT-CHALLENGED"),
            "deviation": "",
            "last_verified": {"commit": head, "toolchain": "rustc stable (CI floats)",
                              "platforms": "see C10-0 §3 and plan §14.1"},
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"head": head, "population": len(rows), "rows": rows},
                                   indent=2) + "\n", encoding="utf-8")

    by_state: dict[str, int] = {}
    for r in rows:
        by_state[r["evidence_state"]] = by_state.get(r["evidence_state"], 0) + 1
    print(f"wrote {OUT_JSON.relative_to(ROOT)}  ({len(rows)} rows, head {head[:7]})")
    for k, v in sorted(by_state.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:>4}")
    print("\nNo aggregate percentage is emitted, by plan §7.3.")
    return 0


def evidence_class(sites, state) -> str:
    """Charter §5.2's vocabulary, chosen from where the evidence actually lives."""
    if not sites:
        return "UNCLASSIFIED" if state == "UNRESOLVED" else "CONF"
    classes = {h["class"] for h in sites}
    if "ENGINE(correlated)" in classes:
        return "DIFF"
    if "SPEC" in classes:
        return "SPEC"
    if "UNIT(cfg-test)" in classes:
        return "UNIT"
    return "REG"


def independent_control(sites) -> str:
    """A control must be able to CONTRADICT the implementation.

    A differential suite compares engines that inherit the same front-end decision, so for a shared
    authority it cannot disagree however many engines agree — EI0's frozen rule. `none` here is a
    real answer, not a gap in this script.
    """
    classes = {h["class"] for h in sites}
    if classes & {"INTEGRATION", "UNIT(cfg-test)", "SPEC"}:
        return "yes — front-end/hand-authored site"
    if classes == {"ENGINE(correlated)"}:
        return "none — differential only (CROSS_ENGINE_DERIVED)"
    return "none"


if __name__ == "__main__":
    raise SystemExit(main())
