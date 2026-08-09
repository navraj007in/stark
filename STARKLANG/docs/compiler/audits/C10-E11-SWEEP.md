# C10 — E11 cross-reference sweep

**Exit criterion:** E11. **Status: DRAFT — the sweep is re-run against the final head before C10-Q.**
**Date:** 2026-08-09. **Swept at:** `dbddf21`.
**Method:** the `stark-doc-sweep` project skill — this repo's own commands, not generic advice.

> *"A release qualification that cites nonexistent or stale evidence fails."* — plan §16.2, and a
> stop condition rather than polish.

**Why this is a draft.** E11 must be clean **at the head C10-Q qualifies**, and that head does not
exist yet — PR #16 is unmerged. Every finding below is fixed; the sweep is re-run after the merge,
because a merge can reintroduce exactly what a sweep removed.

---

# 1. What was checked, and what it found

| # | Check | Result |
| --- | --- | --- |
| 1 | **Counts and enumerations** | **2 stale counts, both fixed** — §2 |
| 2 | **Table vs prose** | **1 unreconcilable count sentence, fixed** — §3 |
| 3 | **Cross-document pointers resolve** | 6 flagged, **1 real, fixed**; 4 legitimately future/generated, 1 correctly historical — §4 |
| 4 | **Canonical source before downstream** | clean — §5 |
| 5 | **Retired decisions must not read as current** | clean — §5 |
| 6 | **SHA / run-id references resolve** | clean — 10 apparent misses, all false positives or cross-repo — §6 |
| 7 | **Platform claims** | clean — both "three platforms" claims verified against `ci.yml` — §7 |
| 8 | **Deviation IDs and residual ranges** | clean — §8 |

---

# 2. Counts — two stale, both mine, both fixed

```text
C10-DEVIATION-DISPOSITIONS.md   header said 23 live-OPEN, body said 24 in TWO places
                                DEV-005 closed on reproduction ~1 hour earlier; I updated the
                                header and missed the body. FIXED

C10-0-OPENING-INVENTORY.md      "117 spec fixture files" against README's "116 fixtures"
                                Both true of different things: 117 FILES in the directory, 116
                                MANIFEST ENTRIES (the 117th is manifest.toml itself). Relabelled
                                to 116 fixtures with the discrepancy stated. FIXED
```

**The first is the canonical failure this sweep exists for**: a count updated in one place and not
the other, within an hour, by someone who knew the rule.

---

# 3. Table vs prose — the sentence that went stale twice in one day

`C10-F-COMPATIBILITY-POLICY.md`'s summary sentence:

```text
first    "Eight commitments, eight non-commitments, two not-applicable"
         went stale the moment the capability-vocabulary axis moved from NOT APPLICABLE

second   "Nine commitments, nine non-commitments, one not-applicable — across fourteen axes"
         counted ITEMS in the summary block while claiming FOURTEEN AXES. Two different
         granularities that do not reconcile: 9+9+1 = 19, not 14

now      "Fourteen axes — six COMMITTED, seven UNCOMMITTED, one NOT APPLICABLE", counted per
         §1.x SECTION HEADING, with the grep that checks it printed beside it
```

**One sentence produced two findings in one day.** The fix is not a third number — it is making the
number mechanically checkable, so the next reader can verify it in one command instead of trusting
it.

---

# 4. Pointers — six flagged, one real

```text
REAL, FIXED
  C10-D-DIFFERENTIAL.md     the plan's §18 artefact list promised it; C10-D produced ONE document
                            (C10-MUTATION-LEDGER.md) because its differential and metamorphic
                            findings are three paragraphs. Splitting three paragraphs into a second
                            file to satisfy a list is filing, not evidence. The LIST was corrected
                            rather than a stub written to make the pointer resolve

LEGITIMATELY ABSENT
  GATE-C10-CLOSURE.md       C10-Q's output. Does not exist yet, by design
  C10-RELEASE-STATEMENT.md  likewise
  build.json, manifest.json generated build/installer artifacts, not repo files

CORRECTLY HISTORICAL — and this one is the interesting classification
  c6_corpus_cases.rs        cited in state-archive/C5-C7-closed-detail.md and the closed
                            WP-C6.5.md. The target never existed (C10-0's finding F5), and I fixed
                            the LIVE corpus README earlier today. These two are HISTORICAL RECORDS,
                            which §16.1 forbids rewriting. Left as written — deliberately
```

**That last row is the sweep doing its harder job.** A sweep that "fixes every dangling pointer"
would have rewritten two historical records to make a grep clean, destroying the provenance §16.1
exists to protect. The right answer distinguishes a live pointer from a preserved one.

---

# 5. Canonical order, and retired policies

**Canonical-first: clean.** No C10 document describes behaviour its canonical source has not
scoped. C10-F explicitly demotes the capability-vocabulary axis to PENDING while it sat on an
unmerged branch, which is this rule applied rather than merely satisfied.

**Retired policies: clean.** Every live occurrence of `RETAIN AS RESEARCH LANGUAGE` is phrased as
superseded (`COMPILER-CHARTER.md` §1.5's dated update, `COMPILER-ROADMAP.md`'s "SUPERSEDED on
2026-08-04"), and the remaining hit is inside a dated `COMPILER-STATE.md` record — preserved as
written, correctly.

---

# 6. SHAs and run ids — ten apparent misses, zero real

Every 7–40 hex string in the C10 documents was resolved against the repository:

```text
2097152                    2 MiB in bytes (a stack size), matched by the hex pattern
254b59607                  rustc's commit hash — external
31292404920, 31292404936   GitHub Actions RUN IDS, not SHAs
31294314143, 31295224000
b3b28e7…, 5cac025…         stark-samples commits — a DIFFERENT REPOSITORY, and both are the
                           sample-suite pins, recorded with their provenance in C10-0
```

**Zero unresolved in-repo SHAs.** The cross-repo ones are exactly the case §14.1b's *relationship to
baseline* field exists for, and C10-0 records both pins and the reason the pin moved.

---

# 7. Platform claims — both verified against the workflow

```text
C10-THREAT-MODEL S09    "release package smoke on three platforms"     ci.yml: linux, macos,
                                                                       windows — TRUE
C10-P §1.2a             "green evidence on all three platforms"        full-matrix runs
                                                                       31294314143 + 31295224000
```

Checked because C10-0's finding F1 established that spec-fixture conformance, the C6.5 mutation
controls and the external sample suite are **linux-x64 only** — so a "three platforms" claim is not
automatically safe in this repository.

---

# 8. What the sweep did NOT find

Stated because a sweep reporting only successes is indistinguishable from one that did not run:

```text
no dangling DEV or CD id            every cited id exists and its LIVE heading matches the citation
no stale gate status                C9 Part B "deferred", C10 "open" agree across every document
no residual range overrun           AS8-R1..R15 and C10-R1..R2 match their highest allocated ids
no test-count inflation             no scoped run's number is presented as a tree total
no release wording yet              C10-Q has not run, so there is nothing to cross-check
```

---

# 9. Re-run before C10-Q

```bash
python3 starkc/scripts/c10-deviation-populations.py     # population counts
python3 starkc/scripts/c10-a2-resolve.py                # dashboard buckets
grep -oE "^## 1\.[0-9]+[a-z]? [^—]*— \*\*[A-Z ]+" C10-F-COMPATIBILITY-POLICY.md | ...   # axis count
```

**Plus every check in §1, at the merged head.** This draft is evidence that the sweep was run; it is
not evidence that the final head is clean.
