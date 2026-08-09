# C10-Q — evidence package and derived release claim

**Status: DRAFT FOR OWNER DECISION. Not a decision.**
Charter §2.3 **CE8** makes any Core conformance or release claim an owner escalation, and §2.2
forbids a session claiming Core v1 conformance on its own authority. **C10 proposes; the owner
authorises.**

**Candidate head:** `076b4dc` — merge of PR #16 into `develop`, carrying every C10 packet.
**Date:** 2026-08-09.

---

# 1. Exit criteria — all eleven met

| | Criterion | Evidence |
| --- | --- | --- |
| E1 | opening inventory, contradictions resolved, OD-1…OD-9 ruled | `C10-0-OPENING-INVENTORY.md` |
| E2 | dashboard over a declared population, no blank control cells | `C10-CONFORMANCE-DASHBOARD.md`, 168 rows |
| E3 | robustness over the declared targets, harness proved able to fail | `C10-B-ROBUSTNESS.md` |
| E4 | security surface frozen first, every defence names a falsifier | `C10-THREAT-MODEL.md`, `C10-C-SECURITY-REVIEW.md` |
| E5 | mutation ledger complete, ten fields, freshness re-checked | `C10-MUTATION-LEDGER.md` |
| E6 | metamorphic relations declared before running | ledger §4 — **none added, and why** |
| E7 | performance over the frozen workload set | `C10-E-PERFORMANCE-BASELINE.md` |
| E8 | every compatibility axis COMMITTED / UNCOMMITTED / N/A | `C10-F-COMPATIBILITY-POLICY.md`, 14 axes |
| E9 | all three populations dispositioned | `C10-DEVIATION-DISPOSITIONS.md` — 23 A, 5 B, 20 C |
| E10 | CI green at the qualified head, overlap recorded, claims mapped | `C10-E10-CI-EVIDENCE.md` — **28/28** |
| E11 | cross-reference sweep clean | `C10-E11-SWEEP.md` |

---

# 2. The evidence, stated as it is

## 2.1 Conformance — the number that bounds everything

```text
POPULATION            168 granular rules   (corrected from the long-cited 161 — A1-F1)

PRECISE-C211           36   positive AND negative evidence at test-function precision
RESOLVED-BY-TREE       20   a test function cites the rule; the inventory never recorded it
CORPUS-OR-FILE-LEVEL   27   real evidence, not per-rule attributed
IMPLEMENTATION-ONLY     1
UNRESOLVED             84   NO citation found — and A1-F3 proved this does NOT mean untested
```

**56 of 168 rules carry function-precision evidence.** The 84 are unresolved *attribution*, not
demonstrated absence: `EXT-ISOLATION-001` recorded `none` while nine tests ran in CI on every push.

## 2.2 Execution and platform

```text
28/28 CI jobs green at 076b4dc, both workflows, overlap recorded
conformance / differential / mutation-control SUITES run on ALL THREE platforms (213 binaries
    on Windows) — the "linux-only" claim C10 made three times was WRONG and understated this
C6.4 tier-1 qualification RECORD and C6.5 corpus replay: linux + macos ONLY
external sample suite (the one EXTERNALLY_DERIVED control): linux only, 39/39
tier-3 x86_64-apple-darwin: PACKAGED with archive and installers, exercised by NO job
```

## 2.3 Assurance

```text
mutation evidence   9 current + 22 historical + 10 fresh-by-hash
                    9/9 re-runs reproduced AS8's recorded outcome, killer counts included
new control         RuntimeFn parity, exhaustive over 100 variants and 4 families, proved able
                    to fail by one-sided drift
robustness          T1/T2/T4/T5/T6/T8 pass over ~2,000 cases; harness proved able to fail
                    T3 (package graphs) and T7 (malformed artifacts) NOT RUN
security            no class-B finding; 7 surfaces carry a defence with NO falsifier
performance         one platform (Darwin-arm64); ONNX appendix EMPTY (inputs unavailable offline)
```

## 2.4 Deviations

```text
A  21 live-OPEN compiler deviations, every one owned and dispositioned
   NONE of them accepts what the specification forbids — see the correction below
B   5 release/distribution — DEV-165, standalone toolchain, offline build, signing, tier-3
C  20 assurance residuals — including AS8-R2/R10/R13/R14, all still true at the candidate
```

> ### CORRECTED 2026-08-09 — THREE of twenty-three deviations do not reproduce
>
> This package's first draft named 23 deviations, and stated that **DEV-177 accepts a program the
> specification forbids** — the one subtraction that made a conformance claim FALSE rather than
> narrow. **DEV-177 was already fixed** by `78bd84c`, whose message is literally *"DEV-177: enforce
> NAME-SHADOW-001, which was never enforced at all"*. The ledger was never updated.
>
> ```text
> DEV-005   does not reproduce — removed by AS2's one-pipeline consolidation
> DEV-177   does not reproduce — 78bd84c. E0204 is emitted, with a related span
> DEV-181   does not reproduce — 57ff6b9. `x = x.method()` compiles and runs
> ```
>
> **13% of the "open" list was fiction, and all three were named in a drafted release claim.**
>
> The cause is structural: the ledger is append-only, so closing an entry needs a deliberate new
> heading, and a repair landing under a different work packet has nothing forcing that heading to
> be written. All three fixes name their DEV number in the commit message — the information
> existed and nothing connected it to the ledger.
>
> **The rule this produces: a deviation may not be named in a release claim on the strength of its
> ledger entry. It must be REPRODUCED at the candidate head, or closed.** OD-7 imposed exactly
> that on `DEV-005` and it found one; applying it to a single entry was the mistake.
>
> **Reproduction status of the remaining 21 — stated honestly, because it bounds this package:**
>
> ```text
> REPRODUCED at 076b4dc   DEV-172   Int8::MIN and Int64::MIN rejected, E0008 — still real
> NOT YET REPRODUCED      the other 20. Their entries are inherited, and three of the last
>                         twenty-three inherited entries were wrong
> ```

---

# 3. Deriving the claim — §17.2's procedure, applied

**Step 2: the strongest claim the evidence supports, per class.** **Step 3: subtract.**

## 3.1 Subtractions, each with its source

```text
DEV-177            WITHDRAWN — it does not reproduce. NO REMAINING DEVIATION ACCEPTS WHAT THE
                   SPECIFICATION FORBIDS, so no subtraction now makes a claim FALSE. Every
                   remaining one makes a claim NARROW
84 + 27 rules      per-rule conformance cannot be claimed over the full 168; 56 carry
                   function-precision evidence
DEV-140..145       six CD-342 layer defects DEFINE the supported native subset. A native claim
                   is a claim about their complement
T3, T7 not run     robustness excludes package-graph and malformed-artifact inputs
C6.4/C6.5 records  no Windows tier-1 qualification record, no Windows corpus-replay arm
tier-3             packaged, never executed — excluded from every claim
AS8-R2/R10/R13/R14 trap-category VOCABULARY, Core trait contracts, non-`pub` re-export
                   visibility, and the verifier's HostResource drop arm have NO control. Claims
                   over them rest on no mutation evidence
S16                INTEGRITY, not AUTHENTICITY. No distribution claim
7 security surfaces defence present, falsifier absent — UNVERIFIED, not defended
performance        one platform. No cross-platform performance claim
22 trials          mutation evidence predating the toolchain integration
```

## 3.2 What survives — the derived claim

**This is what the evidence supports, not what anyone hoped for:**

```text
STARK Core v1 front end, interpreter, MIR, and native backend: conforming for the listed
platform matrix, with deviations DEV-083, DEV-120, DEV-122, DEV-140..145, DEV-156, DEV-157,
DEV-159..162, DEV-167, DEV-168, DEV-172, DEV-178, DEV-180, DEV-186.
(DEV-177 and DEV-181 REMOVED — they do not reproduce. 21 deviations, not 23.)

Platform matrix: aarch64-apple-darwin and x86_64-unknown-linux-gnu (Tier-1);
x86_64-pc-windows-msvc (Tier-2, no tier-1 qualification record).
x86_64-apple-darwin (tier-3) is PACKAGED AND UNEXERCISED and is excluded.

Conformance is claimed per-rule for the 56 of 168 granular rules carrying function-precision
positive and negative evidence; the remaining 112 are exercised but not per-rule attributed.

NAME-SHADOW-001 IS conformant — DEV-177 was fixed by `78bd84c` and the ledger had not caught up.
No rule is violated by acceptance.

Language services: compiler-derived, protocol-conformant, and interactively validated by the
owner in the recorded environment.

Distribution: integrity-verified, NOT authenticated. Unsigned archives.

Tensor extension v0.1: deferred research. No claim.
```

## 3.3 The gate decision this implies

Charter §5.3's vocabulary. **The owner chooses; this is the derivation, not the choice.**

```text
PASS                    NOT SUPPORTED — but the REASON changed. It is no longer "a claim would be
                        FALSE" (DEV-177 is closed); it is that 84 of 168 rules are unattributed
                        and 21 deviations remain. A bare "conforming" claim would OVERSTATE, not
                        lie
PASS-WITH-DEVIATIONS    SUPPORTED, on §3.2's wording, IF the 20 unreproduced deviations are
                        verified first — see the condition below
REVISE                  SUPPORTABLE, and now more defensible than when this package was drafted:
                        three of twenty-three named deviations did not exist, so the claim's own
                        deviation list is not yet trustworthy
BLOCKED                 not indicated
```

## 3.4 The condition this package now carries

**Recommendation: PASS-WITH-DEVIATIONS, conditional on reproducing the remaining 20 deviations at
`076b4dc` first.**

A release claim naming 21 deviations is only as good as the list. **Three of the last
twenty-three were fiction**, and each would have been published as a known limitation of a
compiler that does not have it. Naming a deviation that no longer exists is its own false claim —
the same argument OD-7 used for DEV-005, now with a 3-in-23 base rate behind it.

**REVISE is the honest alternative** if the owner would rather not spend that pass: it says the
evidence is sound and one bounded verification remains, which is exactly the situation.

---

# 4. What C10 would be claiming falsely if it said less carefully

The three phrasings the roadmap forbids, and why each is wrong here:

```text
"STARK Core v1: conforming.  Known deviations: none."
    FALSE. 23 open compiler deviations, one of them an over-acceptance

"conforming on the listed platform matrix"  without the Tier-2 qualifier
    OVERSTATES. Windows has no tier-1 qualification record and no corpus-replay arm

"three independent implementations"
    PROHIBITED. One front end, three execution strategies, a named reference engine, and six
    authorities INVISIBLE to all three (EI6's calibrated wording, reaffirmed by Campaign B)
```

---

# 5. What C10 got wrong about itself, and corrected

Recorded because a qualification campaign's credibility rests on it having challenged its own
conclusions, not only the compiler's.

```text
the denominator          161 -> 168. The seven invisible rules were ALL the numeric-semantics
                         rules — integer overflow, division by zero, float behaviour
"ABSENT means untested"  false. EXT-ISOLATION-001 recorded `none` with nine tests running in CI
"lexical evidence is     refuted by mutation. `keywords_reserved_and_idents` did not detect a
 dense, so controlled"   keyword ceasing to be a keyword
the S10 correction       withdrawn — I read package.rs from a parallel session's unmerged branch
"linux-x64 only"         wrong in THREE documents, and it UNDERSTATED the evidence
DEV-005                  named as a deviation for months; does not reproduce. AS2 removed it
C10-F's summary count    went stale twice in one day
```

**Six of these were found by challenge rather than review**, which is the argument for the
campaign's method over an audit.

---

# 6. What the owner is being asked to decide

```text
1  the gate decision            PASS-WITH-DEVIATIONS, or otherwise
2  the exact release wording    §3.2 as drafted, or amended
3  whether DEV-177 must close   it is the only subtraction that falsifies rather than narrows
   before any conformance
   claim is published
4  whether T3/T7 robustness     both are declared-but-not-run; C10-Q can name them or require them
   and the 7 unverified
   security surfaces are
   acceptable as named
   residuals
```

**Nothing in this package should be read as authorising a `develop -> main` promotion.** That
follows the decision; it does not accompany it.
