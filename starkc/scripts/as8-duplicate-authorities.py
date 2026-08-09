#!/usr/bin/env python3
"""AS8 — find semantic rules DUPLICATED across consumers, which is the inverse of shared fate.

The shared-fate register tracks rules DECIDED ONCE and consumed by every engine, where agreement
between engines is meaningless because there is nothing to disagree with. This script looks for the
opposite hazard: the same rule written out TWICE, where the copies can drift apart silently and the
second copy is nobody's control.

CD-065 recorded exactly this and fixed it -- `mir_ty_is_copy` "had been written out here
identically" in the backend before being consolidated -- and CD-062 recorded the same for
destruction order. Consolidating was right. This script asks which duplicates were never found.

METHOD. Extract every top-level `fn` DEFINITION by brace matching, normalise whitespace, hash the
body, and report bodies that are byte-identical under different names.

TWO FALSE-POSITIVE TRAPS, both hit on the first run and both fixed here:
  * a trait method DECLARATION has no body, so a naive matcher runs on to the next `{` in the file
    and reports sixteen "identical" bodies in one trait. Signature scanning now stops at a `;` at
    paren depth zero.
  * short bodies collide for uninteresting reasons, so bodies under four lines are skipped.

WHAT THIS DOES NOT DO. It finds only TEXTUALLY identical bodies. A rule reimplemented with
different names or a different match order is invisible here, so a clean report is not evidence of
no duplication. As everywhere in AS8: absence of a hit is not a finding.
"""
import collections
import hashlib
import os
import re
import sys

MIN_BODY_LINES = 4


def fn_definitions(src):
    for m in re.finditer(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn (\w+)\s*\(", src, re.M):
        depth, open_brace = 0, None
        for j in range(m.end() - 1, len(src)):
            c = src[j]
            if c in "(<[":
                depth += 1
            elif c in ")>]":
                depth -= 1
            elif c == ";" and depth <= 0:
                break  # a declaration, not a definition
            elif c == "{" and depth <= 0:
                open_brace = j
                break
        if open_brace is None:
            continue
        depth = 0
        for j in range(open_brace, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    yield m.group(1), src[open_brace : j + 1]
                    break


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    bodies = collections.defaultdict(list)
    for dirpath, _, names in os.walk(root):
        for n in names:
            if not n.endswith(".rs"):
                continue
            path = os.path.join(dirpath, n)
            src = open(path, encoding="utf-8", errors="ignore").read()
            for name, body in fn_definitions(src):
                if body.count("\n") + 1 < MIN_BODY_LINES:
                    continue
                key = hashlib.sha1(re.sub(r"\s+", " ", body).strip().encode()).hexdigest()
                bodies[key].append((name, os.path.relpath(path, root), body.count("\n") + 1))

    dups = [v for v in bodies.values() if len({n for n, _, _ in v}) > 1]
    for group in sorted(dups, key=lambda g: -g[0][2]):
        print(f"{group[0][2]:>4} lines")
        for name, path, _ in group:
            print(f"       {name:<28} {path}")
    print()
    print(f"fn definitions scanned              : {sum(len(v) for v in bodies.values())}")
    print(f"identical bodies under different names: {len(dups)}")
    print()
    print("A textual match only. A rule reimplemented differently is invisible here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
