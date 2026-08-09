# C10-D — mutation ledger

**Packet:** C10-D, WP-C10.3. **Date:** 2026-08-09.
**Baseline:** `51ca1af` (merge of `origin/develop` `1d20123` into `wp-c10/execution-plan`).
**Harness discipline:** plan §8. **Freshness:** §8.2a.

---

# 1. Population — declared BEFORE any trial ran (plan §8.2)

```text
IN SCOPE
  claims C10 intends to publish (OD-2: Core v1 Compiler Stable, Native Systems Preview)
  whose supporting evidence has NOT already been mutated in AS8's 39 trials
  and whose evidence C10-A2 flagged as thin or unattributed

OUT OF SCOPE
  anything already trialled in AS8 with a recorded verdict that is still FRESH  (§8.2a)
  ESF-TRAIT-001, ESF-TRAP-001a and the other AS8-R residuals — trialled, recorded, unchanged
  anything whose claim C10 does not intend to publish
```

**Inherited evidence re-verified against this baseline, not assumed:** all 12 mutation-authority
files and all 13 control suites hash identically to `e7bb95d`. **Every one of AS8's 39 trials
remains citable and none is re-run.** That is the freshness rule paying for itself rather than
adding ceremony.

---

# 2. C10D-CTL-001 — the `RuntimeFn` parity control (owner-ruled, built)

**Not a mutation — a control that did not exist.** The AS8 owner ruling on `AS8-DA-002/003/004`
was to keep both implementations and **build the missing cross-check**.

**Why it was missing.** AS8 mutated each interpreter/verifier pair one-sided and both sides died —
but the kill *messages* showed neither copy killed the other: copy A died to `mir_differential`,
copy B to an `unreachable!()` elsewhere. The redundancy was real; the cross-check was imaginary.

```text
location   starkc/src/mir/mod.rs, mod c10d_runtime_fn_parity
scope      all 100 RuntimeFn variants — the closed, versioned runtime surface (contract §7)
families   vec, box, slice, AND map
change     six classifiers `fn` -> `pub(super) fn`. Visibility only, no behaviour change;
           the two implementations stay independent by design
```

Three tests: exhaustive parity, an `ALL`-completeness check, and a **non-empty proper-subset**
check so that emptying both tables cannot pass as agreement (the `AS8-MUT-003` shape, where
`copy_canon_matrix` covers its target completely and controls nothing).

**Exhaustiveness is enforced at compile time.** `_exhaustiveness_witness` matches every variant with
no catch-all, so adding a `RuntimeFn` breaks the build rather than silently shrinking the
population — the failure `AS8-DA-006` names as "the sixth `MirTy` catch-all to swallow this
variant".

## 2.1 Negative control — the pass is believed because the failure was demonstrated

```text
injected   verify.rs's slice table loses `SliceIsEmpty` (one-sided drift)
result     FAILED: "slice: SliceIsEmpty — interp says true, verify says false"
restore    verified byte-identical; re-run green
```

## 2.2 A fourth pair the register does not catalogue

`AS8-DA` lists Vec (`DA-002`), Box (`DA-003`) and Slice (`DA-004`).
**`is_map_runtime` / `is_map_runtime_fn` is a fourth pair of identical shape**, found by
enumerating the classifiers rather than by reading the register.

`AS8-DUPLICATE-AUTHORITIES.md` says of itself: *"this is a lower bound, not an inventory… a rule
reimplemented with different names, a different match order, or an equivalent-but-not-identical
expression is invisible to it."* **This is that warning coming true.** The parity control covers all
four, so the gap is closed as well as recorded.

### `C10-DA-001` — allocated here, and deliberately NOT called `AS8-DA-007`

**Owner ruling, 2026-08-09.** The ID-deferral rule is about `CD-*`/`DEV-*` numbers that could
collide with the parallel toolchain branch. This is neither: it is a new register entry in a
namespace C10 owns, so it is allocated now.

**And it is not `AS8-DA-007`. AS8 is closed, and AS8 did not find this.** Numbering it into AS8's
sequence would rewrite who discovered what.

```text
C10-DA-001   Which `RuntimeFn`s are Map operations
             A   mir::interp::is_map_runtime
             B   mir::verify::is_map_runtime_fn
             intended relationship   as AS8-DA-002/003/004 — the verifier checks what the
                                     interpreter executes, and an independent table is what lets
                                     it disagree
             found by                C10-D, exhaustive enumeration of the classifiers
             disposition             KEEP SEPARATE + exhaustive parity control (C10D-CTL-001)
```

**This extends the known duplicated-authority population from six to seven without rewriting the
closed AS8 result.** `AS8-DA-001..006` stand exactly as recorded; `C10-DA-001` is the seventh, and
its provenance says so.

---

# 3. C10D-MUT-001 — and it refutes a claim C10-A2 made

| field | value |
| --- | --- |
| **authority/rule ID** | `LEX-KEYWORD-001` — *"Which Core words always tokenize as keywords?"* |
| **target** | `starkc/src/lexer.rs::keyword`, the arm `"mut" => Mut` |
| **mutation** | delete the arm: `mut` lexes as an ordinary identifier |
| **prediction** | **KILLED** by `lexer.rs`'s own unit tests — C10-A2 counted 26 test fns and 32 error assertions there and called lexical negative evidence "dense" |
| **selected control** | `--lib lexer` (26 tests); `--test conformance`; `--test gate2_valid` |
| **expected result** | a lexer-level assertion fails |
| **actual result** | **UNEXPECTED.** `--lib lexer`: **26 passed, 0 failed.** `conformance`: FAILED. `gate2_valid`: 11 of 56 FAILED |
| **killer(s)** | `conformance::spec_conformance`; 11 `gate2_valid` cases — **all by programs ceasing to parse**, none by a lexical assertion |
| **residual** | **C10-R1** (below) |
| **freshness** | `n/a — run at 51ca1af` |
| **restore verification** | `diff` byte-identical; `--lib lexer` re-run green |

## 3.1 The finding

**`lexer::tests::keywords_reserved_and_idents` — a test named for exactly this rule — passed while
`mut` was not a keyword.**

That is the `copy_canon_matrix` shape in a new file: a test that *names* the rule and does not
*control* it. It is also a direct refutation of C10-A2's reasoning, which inferred control from a
**count of error assertions**:

> *"lexical negative evidence is DENSE (`lexer.rs`, 26 test fns / 32 error assertions) … these rules
> are controlled; the attribution is missing."*

**The count was real and the inference was wrong.** Those 32 assertions cover literal forms, escapes
and malformed input — not keyword identity. A2 measured the wrong property and reasoned from it, in
the same session that recorded EI2 making precisely that error.

## 3.2 C10-R1 — the residual

```text
C10-R1   Keyword identity is controlled only COARSELY, by programs failing to parse.

         What that catches:   a keyword that stops being a keyword, because the grammar then
                              rejects every program using it
         What it would MISS:  a keyword mis-mapped to a DIFFERENT keyword, or a reserved word
                              silently promoted to a keyword, where the program still parses.
                              Nothing in the tree pins the token a word maps TO
         Disposition:         population C (assurance residual). Owner: C10-Q states the lexical
                              claim as controlled by acceptance/rejection, not by token identity
```

**No DEV is allocated.** A survivor — or here, a kill by the wrong mechanism — means the evidence
cannot detect the defect, not that the defect is present. The AS8 owner ruling, inherited verbatim.

---

# 4. Metamorphic — the twelve families, and what C10-D did not add

`starkc/tests/c6-corpus/metamorphic.py` carries **M01–M12**, each with a recorded precondition and
a named normative rule, and each transform asserting that it actually rewrote the source (a
transformation that changes nothing is a fake pair).

**C10-D adds none, and that is a scope decision rather than an omission.** The plan's candidate list
(formatter idempotence, harmless parenthesisation, equivalent import forms) each needs a normative
rule stating the equivalence *before* the relation may be added — §10.2 — and none of the three has
one written down today. **Adding a relation on intuition is exactly what §10.2 forbids**, and
sourcing three new normative equivalences is spec work, not qualification.

Recorded as **C10-R2**: the metamorphic surface is twelve families wide and could be wider; the
blocker is normative, not technical.

---

# 5. What C10-D establishes, and what it does not

```text
ESTABLISHES   the interpreter/verifier runtime classifications now have a total cross-check,
              proved able to fail, over the whole closed enum and all FOUR families
ESTABLISHES   keyword identity is controlled coarsely, by parse failure — measured, not assumed
ESTABLISHES   AS8's 39 trials are FRESH at this baseline and need no re-running

DOES NOT      re-open any AS8 residual. R2, R5, R10, R13 and the rest stand as recorded
DOES NOT      add metamorphic relations without normative backing
DOES NOT      allocate DEVs for evidence gaps — a gap is not a defect
DOES NOT      challenge the other 163 NOT-CHALLENGED rules. C10-D challenged the claims C10-A2
              flagged as thin; the rest remain NOT-CHALLENGED in the dashboard, honestly labelled
```

---

# 6. C10-R — the §8.2a re-run after the toolchain integration

**Date:** 2026-08-09. **Candidate:** `29ce610` (`develop` `eb60dec` merged into the C10 branch).
**Raw record:** `evidence/c10/c10r-rerun-trials.json`.

## 6.1 The freshness rule fired for real

Re-running §8.2a against merged `develop` after PR #15:

```text
AUTHORITY moved   src/mir/lower.rs            src/resolve.rs
CONTROL moved     three_engine_differential.rs   a11_host_resource.rs
                  c788_resource_lifecycle.rs

31 of 41 trials STALE      6 by clause 1 (the mutated code moved)
                          28 by clause 2 (the killing suite moved)
                          10 remain FRESH
```

**Yesterday this rule returned "all FRESH" twice and looked like ceremony.** Today it stopped C10-Q
from citing 31 measurements of a compiler that no longer exists.

## 6.2 The subset re-run, and why not all 31 (owner decision)

Plan §8.2's population rule is that mutation applies to **claims C10 intends to publish**. C10-F
marks Core language compatibility **UNCOMMITTED** (56 of 168 rules function-evidenced), so trials
backing unpublished Core rules do not need refreshing to support a claim nobody is making.

```text
RE-RUN (9)   all 6 clause-1 trials — the mutated code itself moved, the strongest reason
             + MUT-017, MUT-037   host-resource typing, backing the COMMITTED provider claims
             + MUT-036            MIR verifier, backing the COMMITTED rejection-on-mismatch claim

HISTORICAL   the remaining 22, all clause-2-only against `three_engine_differential`, backing Core
(22)         rules C10-F does not commit to. Their AS8 verdicts stand as recorded, and the ledger
             marks them HISTORICAL rather than current
```

**The disclosure this creates:** C10-Q's mutation evidence is **9 current + 22 historical + 10
fresh-by-hash**. Any claim resting on one of the 22 must say the measurement predates the
toolchain integration.

## 6.3 Result — 9/9 reproduce AS8's recorded outcome

```text
trial           AS8 recorded   C10-R re-run   killers   agreement
AS8-MUT-007     KILLED         KILLED         4         MATCH
AS8-MUT-017     KILLED         KILLED         3         MATCH
AS8-MUT-023     KILLED         KILLED         1         MATCH
AS8-MUT-034     SURVIVED       SURVIVED       0         MATCH
AS8-MUT-035     SURVIVED       SURVIVED       0         MATCH
AS8-MUT-036     KILLED         KILLED         3         MATCH
AS8-MUT-037     SURVIVED       SURVIVED       0         MATCH
AS8-MUT-038     KILLED         KILLED         2         MATCH
AS8-MUT-039     SURVIVED       SURVIVED       0         MATCH
```

**Killer counts match too**, not merely the verdicts — the same suites kill by the same margin.

> **Read the harness's `UNEXPECTED` label carefully.** It compares against `expect`, which encodes
> **EI5's original prediction**, not against what AS8 measured. `MUT-034/035/037/039` are printed
> UNEXPECTED because EI5 predicted KILLED and they survived — **which is exactly what AS8 recorded**
> (`AS8-R13`, `AS8-R14`). Reporting those as new surprises would be misreading the tool, and it is
> the kind of misreading a ledger exists to prevent.

## 6.4 What this establishes

```text
ESTABLISHES  the toolchain integration changed resolve.rs and mir/lower.rs WITHOUT changing any
             measured control relationship. The evidence survived the merge, verified rather than
             assumed
ESTABLISHES  AS8-R13 (non-`pub` re-export visibility has no control anywhere) and AS8-R14
             (may_need_drop's HostResource arm unguarded) still hold at the candidate — both
             residuals are current, not stale
ESTABLISHES  batch 0 calibrates in both directions on the merged tree, so these results are
             interpretable at all

DOES NOT     refresh the 22 historical trials
DOES NOT     re-open any AS8 residual — all nine outcomes are unchanged
```

## 6.5 Harness change

`as8-mutate.py` gains `--only`, selecting trial ids **across** batches. Needed because trials made
stale by a merge do not line up with AS8's batch structure, and **duplicating their definitions into
a new C10 batch would create a second copy that can drift from the first** — the `AS8-DA-*` failure
mode, in the tool that exists to detect it.

Restore verified after the run: `git status` shows only the harness edit, no mutation residue.
