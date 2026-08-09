#!/usr/bin/env python3
"""AS8-R3 — enumerate the test corpus for controls, per shared-fate authority.

WHY THIS EXISTS. EI2 concluded "no independent control exists" for `ESF-COPY-001`, EI4 ranked it
critical on that basis, and EI5's selected-test columns repeated it. All three were wrong: thirteen
hand-authored tests in `c61f_structural_copy.rs` kill the mutations that survived every differential
suite. The error was not carelessness, it was METHOD -- the audit read the differential machinery and
the register instead of enumerating the corpus, so the controls it missed were precisely the
front-end tests no differential suite runs.

THE KEY IS THE NORMATIVE RULE ID, NOT THE FUNCTION NAME. `c61f_structural_copy.rs` never mentions
`copy_eligible_types`; it cites OWN-COPY-001. A symbol census finds tests that touch the
IMPLEMENTATION and misses every test that pins the RULE -- which is the only kind that can act as a
control, because a control must be able to contradict the implementation.

STATED LIMITATION, because a census that overstates its reach is worse than none. This finds tests
that CITE a rule ID. A control that pins the rule without naming it is invisible here, and the
spec-fixture corpus carries no rule IDs at all -- so absence of a hit is NOT proof of absence of a
control. Where this script reports NONE, the honest next step is a mutation trial, not a residual.
"""
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Authority -> the normative rules a control for it would have to pin.
AUTHORITIES = {
    "ESF-COPY-001": ["OWN-COPY-001"],
    "ESF-COPY-002": ["OWN-COPY-001"],
    "ESF-DROP-001": ["OWN-DROP-001", "DROP-EXACT-001"],
    "ESF-DROP-002": ["DROP-ORDER-001", "DROP-COLLECTION-001", "DROP-LOOP-001"],
    "ESF-TRAP-001a": ["TRAP-CATEGORY-001"],
    "ESF-TRAP-001b": ["TRAP-CATEGORY-001"],
    "ESF-TYPE-001": ["TYPE-PRIM-001"],
    "ESF-TRAIT-001": ["TRAIT-DEF-001", "TRAIT-LAW-001", "TRAIT-ASSOC-001"],
    "ESF-RES-001": ["RES-", "HostResource"],
    "ESF-NUM-001": ["NUM-", "TYPE-INT-001"],
}


def classify(path, src):
    """A control must be able to DISAGREE with the engines. A differential suite compares engines
    to each other, so it cannot; a front-end test asserting a diagnostic can."""
    if not path.endswith(".rs"):
        return "corpus-case"
    if "differential" in src or "three_engine" in src or "agree(" in src:
        return "ENGINE"
    if "typecheck::" in src or "starkc::parser" in src or "resolve(" in src:
        return "FRONT-END"
    return "other"


def main():
    files = {}
    for base in ("starkc/tests", "STARKLANG/tests"):
        for root, _, names in os.walk(os.path.join(ROOT, "..", base)):
            for n in names:
                if n.endswith((".rs", ".toml", ".stark")):
                    p = os.path.join(root, n)
                    try:
                        files[p] = open(p, encoding="utf-8", errors="ignore").read()
                    except OSError:
                        pass

    gaps = []
    for auth, rules in AUTHORITIES.items():
        by = collections.defaultdict(list)
        for p, src in files.items():
            if any(r in src for r in rules):
                by[classify(p, src)].append(os.path.basename(p))
        front = sorted(set(by.get("FRONT-END", [])))
        engine = sorted(set(by.get("ENGINE", [])))
        corpus = by.get("corpus-case", [])
        print(
            f"{auth:<15} FRONT-END {len(front):>2}   ENGINE {len(engine):>2}   corpus {len(corpus):>3}"
        )
        if front:
            print(f"{'':<15} control(s): {', '.join(front)}")
        else:
            print(f"{'':<15} control(s): none found by rule citation — MUTATE TO DECIDE")
            gaps.append(auth)

    print()
    print(f"authorities with a cited front-end control : {len(AUTHORITIES) - len(gaps)}")
    print(f"authorities with none found                : {len(gaps)}  {gaps}")
    print()
    print("Absence of a hit is NOT evidence of absence of a control (see the module docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
