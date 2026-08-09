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
A  23 live-OPEN compiler deviations, every one owned and dispositioned
   ONE of them ACCEPTS WHAT THE SPEC FORBIDS: DEV-177 (NAME-SHADOW-001)
   the other 22 either REFUSE what the spec allows, or execute an accepted program wrongly
B   5 release/distribution — DEV-165, standalone toolchain, offline build, signing, tier-3
C  20 assurance residuals — including AS8-R2/R10/R13/R14, all still true at the candidate
```

---

# 3. Deriving the claim — §17.2's procedure, applied

**Step 2: the strongest claim the evidence supports, per class.** **Step 3: subtract.**

## 3.1 Subtractions, each with its source

```text
DEV-177            NAME-SHADOW-001 is VIOLATED BY ACCEPTANCE. No unqualified conformance claim
                   may cover it — this is the only subtraction that makes a claim FALSE rather
                   than narrow
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
DEV-159..162, DEV-167, DEV-168, DEV-172, DEV-177, DEV-178, DEV-180, DEV-181, DEV-186.

Platform matrix: aarch64-apple-darwin and x86_64-unknown-linux-gnu (Tier-1);
x86_64-pc-windows-msvc (Tier-2, no tier-1 qualification record).
x86_64-apple-darwin (tier-3) is PACKAGED AND UNEXERCISED and is excluded.

Conformance is claimed per-rule for the 56 of 168 granular rules carrying function-precision
positive and negative evidence; the remaining 112 are exercised but not per-rule attributed.

NAME-SHADOW-001 is NOT conformant: DEV-177 accepts a program the specification forbids.

Language services: compiler-derived, protocol-conformant, and interactively validated by the
owner in the recorded environment.

Distribution: integrity-verified, NOT authenticated. Unsigned archives.

Tensor extension v0.1: deferred research. No claim.
```

## 3.3 The gate decision this implies

Charter §5.3's vocabulary. **The owner chooses; this is the derivation, not the choice.**

```text
PASS                    NOT SUPPORTED. DEV-177 accepts what the spec forbids, and 84 rules are
                        unattributed. A bare "conforming" claim would be false
PASS-WITH-DEVIATIONS    SUPPORTED by the evidence above, on the wording in §3.2
REVISE                  supportable if the owner judges DEV-177 or T3/T7 must close first
BLOCKED                 not indicated — no evidence gap prevents a credible NARROW claim
```

**Recommendation: PASS-WITH-DEVIATIONS**, on §3.2's wording. Offered as a recommendation because
CE8 reserves the decision, and because §17.2 forbids choosing the wording before the evidence — the
wording above was derived after E1–E11, not before.

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
