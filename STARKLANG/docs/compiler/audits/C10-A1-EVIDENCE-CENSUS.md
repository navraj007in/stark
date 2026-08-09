# C10-A1 — Conformance-evidence integrity census

**Packet:** C10-A1, the first half of WP-C10.1. Split from the dashboard by plan §6.1 so the
dashboard's *inputs* are established before its *rows* are written.
**Date:** 2026-08-09. **Baseline:** `f12ececca6d4bdabf828d657c4a4f719a7f9c39a`.
**Tool:** `starkc/scripts/c10-evidence-census.py` (`--json`, `--self-test`).

**The question A1 answers:** for each normative rule C10 might claim, does executable evidence
exist, at what precision, and can that evidence disagree with the implementation?

---

# 1. The headline, and the sentence that qualifies it

```text
DENOMINATOR   168 granular rules   semantic-freeze/CORE-V1-COMPLETENESS.md, no exclusions

PRECISE      36    21.4%    positive AND negative evidence at test-FUNCTION precision
AGGREGATE    85    50.6%    evidence cited only at file level or via the aggregate runner
ABSENT       42    25.0%    the inventory's evidence column records `none`
N/A           5     3.0%    prohibited / deliberately-unspecified / deferred / spec-defect
```

> **This census measures CITATION, not the test tree.** `ABSENT` means *the inventory of record
> cites no evidence for this rule*. It does **not** mean the rule is untested — and §4 shows that
> for at least two rules it is flatly false.

That distinction is the packet's main result and the reason the raw percentages must never be
published on their own.

---

# 2. A1-F1 — the denominator was wrong, and the seven missing rules are not random

**Every prior document, including the C10 plan's own §1.4, cites "161 granular rules". The number
is 168.**

The undercount is a **method** defect, not a data one. Rule IDs were matched as two dash-separated
segments plus a three-digit number (`LEX-IDENT-002`), and **seven IDs have three segments**:

```text
NUM-INT-TYPE-001     what widths and value ranges do integer types have
NUM-INT-ARITH-001    what happens on integer add/subtract/multiply/negate OVERFLOW
NUM-INT-DIV-001      signed division/remainder rounding, zero, and MIN/-1
NUM-FLOAT-OP-001     float operation semantics
NUM-FLOAT-FORMAT-001 float formatting
NUM-FLOAT-TRAIT-001  float trait behaviour
NUM-FLOAT-REPRO-001  float reproducibility
```

**All seven are the numeric-semantics rules** — integer overflow, division by zero, float behaviour.
For a language whose headline guarantee is *"integer overflow, division by zero, out-of-bounds
indexing and failing casts always trap, in every build mode"*, these are among the most
load-bearing rules in the specification, and the counting method used to plan this campaign could
not see them.

**Caught by cross-checking, not by reading.** The first run reported five `core-v1-c2.11-evidence.toml`
rules citing IDs "not in the inventory" — `NUM-INT-ARITH-001` among them. The obvious reading was
"the evidence file has dangling references". The correct reading was "my regex cannot see those
IDs". Verified by a second, independently written enumerator: **168 distinct IDs, zero duplicates,
exactly 7 three-segment IDs** — precisely the 168 − 161 gap.

**The population did not change; the enumeration of it was faulty.** Per plan §7.2 this is recorded
as a dated correction line, not an edit of the C10-0 declaration.

```text
metric        per-rule conformance evidence classification
population    the granular IDs in semantic-freeze/CORE-V1-COMPLETENESS.md
frozen at     f12ecec
declared as   161 (C10-0, 2026-08-09)
CORRECTED to  168 (C10-A1, 2026-08-09) — the population is unchanged; the ENUMERATOR was
              undercounting three-segment IDs. Verified by two independent enumerators.
```

**This is the third instance in two packets of the same failure mode:** a measurement whose method
was never checked against something it should not find. F1/F3/F5 in C10-0, and now A1-F1.

---

# 3. A1-F2 — citation integrity PASSES, and the negative control proves the check works

```text
every PRECISE citation resolves to a real `fn` in a real file        PASS  (36 rules, 0 broken)
```

Verified the same way `check-conformance.py` does it: a `path::fn_name` citation requires
`fn fn_name` to exist in that file, so a **renamed or deleted test** is caught, not merely a
renamed file.

**Negative control — the forcing mechanism required by plan §6.4:**

```bash
python3 starkc/scripts/c10-evidence-census.py --self-test
```

injects a rule citing `starkc/tests/conformance.rs::a_function_that_does_not_exist` and requires the
census to report it:

```text
CITATION INTEGRITY: FAIL
    __INJECTED__  positive_tests  starkc/tests/conformance.rs::a_function_that_does_not_exist
        no `fn a_function_that_does_not_exist` in starkc/tests/conformance.rs
NEGATIVE CONTROL: PASS -- the injected citation was reported
```

**A clean citation-integrity result is believed because the check was shown able to fail.** Per the
owner's amendment 2, no finding count was expected or targeted: zero broken citations and twenty
would have been equally legitimate outcomes. What had to be demonstrated is that the census
enumerated the population and that its check can disagree.

---

# 4. A1-F3 — `ABSENT` does not mean untested, and this is EI2's error in mirror image

Two spot-checks are enough to establish it:

```text
EXT-ISOLATION-001   inventory evidence column: `none; none`
                    IN THE TREE: starkc/tests/c91_extension_isolation.rs, 9 test functions,
                    running in CI on every push. C9.1 built an extension-isolation matrix

OWN-PARTIAL-001     inventory evidence column: `none;`
                    IN THE TREE: the as4_* suites (destructor authority, hostile combinations,
                    property adversaries, reference rule) and the c61f_* structural-copy suites
```

**The inventory's `none` is a statement about the inventory, frozen into a data file in 2026-07-18
and not maintained as the tree grew.** EI2 read the differential machinery and concluded
`ESF-COPY-001` had no control while `c61f_structural_copy` sat in the tree; the inventory records
`none` for rules whose suites are green in CI right now. Same error, different medium.

**Consequence for C10-A2, and it is a hard rule:** the dashboard may **not** transcribe this
census's buckets as its evidence column. Each `ABSENT` and each `AGGREGATE` row must be resolved
against the *tree* — by naming a test function or by confirming absence — before the dashboard
states anything. Where that work is not done, the row says **UNRESOLVED**, never `none`.

## 4.1 Where ABSENT concentrates

```text
EXEC    8    evaluation order, aggregates, assignment, place expressions, temporaries, `for`,
             exactly-once
REF     5    reference identity, projection, return, slices, borrow-carrying values
FLOW    4    definite initialisation, mutability, bounds, `?`
AM      4    abstract-machine locals, objects, owners, temporaries
FUTURE  3    closure/FFI/thread exclusions — largely N/A in substance
OWN 2  DROP 2  PAT 2  PKG 2  STD 2  LIMIT 2  SYN 1  TYPE 1  MOVE 1  NUM 1  LAYOUT 1  EXT 1
```

The concentration in `EXEC`, `REF`, `FLOW` and `AM` — 21 of 42 — is a statement about which parts
of the specification were never re-cited at C2.11 precision. It is **not** a statement that
execution semantics are untested: the three-engine differential, the 89-case corpus and the frozen
execution snapshots all exercise exactly this surface. **They are simply not attributed per rule**,
which is DEV-017 exactly.

---

# 4.2 A1-F5 — the gap is COHERENT, not rot, and that tells A2 where to look

Cross-tabulating the census against `core-v1-rule-id-map.toml`, which maps the 59 legacy broad
rules onto granular IDs:

```text
granular IDs                              168
mapped from a legacy rule                  98
UNMAPPED — no legacy predecessor at all     70

                total    of which UNMAPPED
  PRECISE          36                   15
  AGGREGATE        85                   13
  ABSENT           42                   41      <--
  N/A               5                    1
```

**Forty-one of the forty-two ABSENT rules have no legacy predecessor either.** They are not rules
whose evidence decayed — they are rules **C2.6 created when it split 59 broad IDs into a granular
inventory, which C2.11 then never reached.** C2.11 stopped at 36 rules, by its own header: it
covered "the high-cost frozen semantic surface", not the whole inventory.

So the shape is:

```text
59 legacy rules   ->  98 granular IDs   ->  36 re-cited at function precision by C2.11
                      70 granular IDs   ->  no legacy ancestor; 41 of them uncited anywhere
```

That is a **known, explainable, bounded** gap rather than an unknown one, and it changes what C10-A2
has to do. The 41 are not scattered across the specification: §4.1 shows them concentrated in
`EXEC`, `REF`, `FLOW` and `AM` — the execution and reference semantics that C2.6 articulated in the
Abstract Machine chapter after the legacy database was written.

**And per A1-F3 they are still UNRESOLVED, not untested.** The three-engine differential, the 89-case
corpus and the frozen execution snapshots exercise exactly this surface. Nobody has attributed that
exercise to these rule IDs. **That attribution is C10-A2's central task**, and A1's contribution is
to say precisely which 41 rows it starts with rather than leaving A2 to rediscover them.

---

# 5. What this means for the release claim

Stated plainly, because C10-Q must not discover it late:

```text
21.4%  of normative rules carry per-rule positive AND negative evidence at function precision
50.6%  are cited only through an aggregate runner with no per-rule attribution (DEV-017)
25.0%  are cited as having no evidence — a claim about the INVENTORY, unverified against the tree
```

**A "Core v1: conforming" claim cannot rest on this census as it stands**, and the reason is not
that the compiler is weak — it is that the *attribution* is. The tree contains substantially more
evidence than the inventory cites. C10-A2's job is to close that gap row by row, and where it
cannot, the dashboard says so.

**No conformance percentage may be published that mixes PRECISE with AGGREGATE** (plan §7.3).
AGGREGATE is unclassified.

---

# 6. Findings

| ID | Finding | Consequence |
| --- | --- | --- |
| **A1-F1** | The denominator is **168**, not the 161 every prior document cites. Seven three-segment `NUM-*` IDs — **all the numeric-semantics rules, including integer overflow and division by zero** — were invisible to the counting method | C10-0's declaration corrected by a dated line; the population is unchanged |
| **A1-F2** | Citation integrity **PASSES** (36/36 resolve to real `fn`s), and the check is proved able to fail by an injected non-existent citation | The clean result is believed |
| **A1-F3** | **`ABSENT` means "the inventory cites nothing", not "nothing tests it."** `EXT-ISOLATION-001` records `none; none` while 9 tests run in CI | **Binding on C10-A2:** never transcribe these buckets into the dashboard. Resolve against the tree, or say UNRESOLVED |
| **A1-F4** | 85 of 168 rules (50.6%) are cited only through an aggregate runner | The DEV-017 debt, quantified against the correct denominator for the first time |
| **A1-F5** | **41 of the 42 ABSENT rules have no legacy predecessor either.** The gap is exactly the granular rules C2.6 created and C2.11 never reached — coherent and bounded, not decay | Names the precise 41 rows C10-A2 starts from |

---

# 7. Reproducing

```bash
python3 starkc/scripts/c10-evidence-census.py              # buckets + citation integrity
python3 starkc/scripts/c10-evidence-census.py --json       # machine-readable
python3 starkc/scripts/c10-evidence-census.py --self-test  # the negative control
```

Exit status is non-zero when a citation does not resolve, so the census is CI-able as a guard
against a renamed or deleted test silently orphaning a rule's evidence.

---

# 8. What C10-A1 does NOT claim

```text
NOT a conformance measurement      it measures CITATION QUALITY. A rule in AGGREGATE may be
                                   thoroughly tested; a rule in PRECISE may be tested by two
                                   weak assertions
NOT a coverage measurement         coverage is a separate axis and lives nowhere in this file.
                                   Coverage is not conformance: typecheck/traits.rs is 82.77%
                                   covered and ESF-TRAIT-001 has no control of any kind
NOT a judgement on the 42          `ABSENT` rows are UNRESOLVED against the tree, not condemned
NOT a target                       no percentage here is a goal, and none should be inferred
```
