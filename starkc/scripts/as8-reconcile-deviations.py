#!/usr/bin/env python3
"""AS8 — reconcile KNOWN-DEVIATIONS.md entries against executable evidence.

AS8's work section asks to "reconcile deviation statuses with executable evidence". The deviations
file has ~187 entries and NO uniform status field, so "the status" is prose. This script does not
invent one. It cross-references three independent sources and reports where they DISAGREE:

    the deviations file   does the entry's own text say it is closed/fixed/ruled?
    the decision record   does COMPILER-STATE.md record a CD closing it?
    the test corpus       does any test NAME it?

A disagreement is not automatically an error — a deviation can be legitimately open with a test that
reproduces it, and a closed one can keep its regression test forever. What the script produces is
the SHORT LIST worth a human read, instead of 187 entries worth.

The one combination that is always worth attention is printed first: **the record says closed and
nothing tests it.** That is a closure with no executable evidence behind it.
"""
import collections
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVFILE = os.path.join(ROOT, "starkc/docs/conformance/KNOWN-DEVIATIONS.md")
STATE = os.path.join(ROOT, "COMPILER-STATE.md")
TESTS = os.path.join(ROOT, "starkc/tests")

# Scoped to the entry's HEADLINE, not its body. The first version searched the whole entry and
# included "RULING", so it matched 115 of 189 entries and reported nothing usable. A check that
# flags two thirds of its input is not a check -- it is the same false-positive failure the AS8
# scanners keep hitting, and the same fix applies: narrow until the hits are worth reading.
CLOSED_WORDS = re.compile(r"\b(CLOSED|FIXED|RESOLVED|REPAIRED|SUPERSEDED|WITHDRAWN)\b")
HEADLINE_CHARS = 240


def entries():
    text = open(DEVFILE, encoding="utf-8").read()
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"^#{2,3} (DEV-[A-Z0-9-]+)", text, re.M)]
    for k, (pos, name) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(text)
        yield name, text[pos:end]


def main():
    state = open(STATE, encoding="utf-8").read()
    archive = ""
    adir = os.path.join(ROOT, "STARKLANG/docs/compiler/state-archive")
    for f in os.listdir(adir):
        if f.endswith(".md"):
            archive += open(os.path.join(adir, f), encoding="utf-8").read()
    record = state + archive

    tested = collections.Counter()
    for dirpath, _, names in os.walk(TESTS):
        for n in names:
            if not n.endswith((".rs", ".stark", ".toml")):
                continue
            src = open(os.path.join(dirpath, n), encoding="utf-8", errors="ignore").read()
            for dev in set(re.findall(r"DEV-[A-Z0-9-]+", src)):
                tested[dev] += 1

    rows = []
    multi = collections.defaultdict(list)
    text = open(DEVFILE, encoding="utf-8").read()
    for m in re.finditer(r"^#{2,3} (DEV-[A-Z0-9-]+)([^\n]*)", text, re.M):
        line = text[: m.start()].count("\n") + 1
        multi[m.group(1)].append((line, m.group(0)))
    multi = {k: v for k, v in multi.items() if len(v) > 1}

    for name, body in entries():
        says_closed = bool(CLOSED_WORDS.search(body[:HEADLINE_CHARS]))
        in_record = bool(re.search(rf"{re.escape(name)}\b", record))
        rec_closed = bool(re.search(rf"{re.escape(name)}[^\n]*\b(CLOSED|FIXED|RULED|repaired)", record))
        rows.append((name, says_closed, rec_closed, in_record, tested.get(name, 0)))

    def show(title, hits, note, fmt):
        print(f"\n{title}  ({len(hits)})")
        print(f"    {note}")
        for h in hits[:40]:
            print("      " + fmt(h))
        if len(hits) > 40:
            print(f"      ... and {len(hits) - 40} more")
        return hits

    print(f"deviation entries: {len(rows)}   entries named by at least one test: "
          f"{sum(1 for r in rows if r[4])}")

    a = show("CLOSED IN THE RECORD, NAMED BY NO TEST",
             [r for r in rows if r[2] and r[4] == 0],
             "A closure with no executable evidence behind it. Always worth a read.",
             lambda r: f"{r[0]:<22} tests={r[4]}")

    # A deviation tracked across several append-only entries states its status MORE THAN ONCE, and
    # the FIRST statement is the one a reader meets first. DEV-121 opens with "(OPEN; instance fixed
    # CD-305, class open)" at line 2813 and is CLOSED at line 6371 -- 3,500 lines later. That is the
    # same hazard AS8 fixed in COMPILER-STATE.md: the current position is not discoverable without
    # reconstructing the chronology.
    stale = []
    for name, heads in multi.items():
        first, last = heads[0], heads[-1]
        f_closed = bool(CLOSED_WORDS.search(first[1]))
        l_closed = bool(CLOSED_WORDS.search(last[1]))
        if f_closed != l_closed:
            stale.append((name, first, last, l_closed))
    show("FIRST HEADING CONTRADICTS THE LAST",
         stale,
         "A reader who stops at the first entry gets a status the file later reverses.",
         lambda h: f"{h[0]:<12} L{h[1][0]:<6} -> L{h[2][0]:<6} now={'CLOSED' if h[3] else 'OPEN'}")

    show("NOT MENTIONED IN ANY DECISION RECORD OR ARCHIVE",
         [r for r in rows if not r[3]],
         "Filed and never ruled on. Fine for a seed entry; suspicious for a numbered one.",
         lambda r: f"{r[0]:<22} tests={r[4]}")

    # DELIBERATELY NOT REPORTED: "entry reads closed but the record does not confirm". The first
    # version of this script printed that bucket and it held 115 of 189 entries, because the
    # record-side regex needs the DEV id and the closing word on ONE LINE and most records do not
    # write them that way. The bucket measured the regex, not the corpus. Removed rather than
    # tuned: a check nobody can act on is worse than no check, and this file has spent a packet
    # learning that.

    print("\nA disagreement is not automatically an error — an open deviation may keep a")
    print("reproducing test, and a closed one may keep its regression test forever.")
    return 1 if a else 0


if __name__ == "__main__":
    sys.exit(main())
