# Campaign B — exit report

**Date:** 2026-08-09. **Gate:** Campaign B passes when AS5–AS8 are complete or explicitly deferred
with owner-approved evidence. **Its report is a prerequisite for C10 release qualification, and it
does not itself make a stability or conformance claim.**

**Verdict: PASS.**

---

## 1. The packets

| Packet | Status |
| --- | --- |
| AS5 | **CLOSED 2026-08-07** — `AS-SPRINT2-CLOSEOUT.md` |
| AS6 | **CLOSED 2026-08-08 (CD-390)** — tensor extension quarantine; the extension's semantic authority moved behind `TensorCheckCtx`, one-directional by construction |
| AS7 | **CLOSED 2026-08-08 (CD-391); criterion 2 RE-QUALIFIED 2026-08-09 (CD-393)** — the type checker split into eleven modules with an executable, cycle-free dependency graph |
| AS8 | **CLOSED 2026-08-09 (CD-394)** — assurance against the frozen AS6/AS7 result |
| Sprint 4 Tier-3 | **PASS** — `AS-SPRINT4-CLOSEOUT.md` |

## 2. What the campaign actually established

**Structure.** `typecheck.rs` was 14,588 lines and is now eleven modules whose dependency direction
is *enforced by a test*, not documented in prose. The tensor extension no longer holds Core
semantic authority.

**Assurance.** 39 compiler-source mutation trials — the first in this repository to touch compiler
source at all. `c6_mutation` had established that the *comparator* was sensitive; nothing had ever
shown that the *evidence base* could detect a wrong rule.

```text
26 trials CONFIRMED the prediction
13 trials FALSIFIED it, IN BOTH DIRECTIONS
```

## 3. The finding a release qualification should carry forward

**Three of the four evidence documents this campaign consumed were wrong about what the tree can
detect, and the errors were not careless — they were methodological.** EI2 audited the evidence base
by *reading the differential machinery and the register*. Every control it missed was a front-end
test that no differential suite runs, and every over-claim it made was a suite that runs but cannot
disagree.

```text
ESF-COPY-001   recorded as having NO control      ->  c61f_structural_copy is a real one
ESF-TRAIT-001  recorded as controlled by a matrix ->  the matrix enumerates FROM the rule it checks
ESF-TYPE-001   recorded as controlled by fixtures ->  the fixtures classify ACCEPTANCE, not identity
ESF-PROV-001   recorded as "two engines only"     ->  mir/verify.rs is a control, and is not an engine
```

**The transferable lesson for C10: an evidence claim that has not been mutated is an opinion.** A
suite that passes tells you it agrees with the implementation. Only a suite that *fails when the
implementation is wrong* tells you it can disagree — and thirteen times out of thirty-nine, the
documented expectation did not survive contact with that test.

**Coverage independently made the same point.** `typecheck/traits.rs` is 82.77% covered and
`ESF-TRAIT-001` has no control at all: Core trait contracts can be declared arbitrarily wrong —
`Eq::eq` by value, `Ord::cmp` returning `Bool` — and every selected suite passes. Coverage says a
line ran. It does not say anything would have noticed the line being wrong.

## 4. What Campaign B explicitly does NOT claim

Stated plainly because the gate's own wording requires it.

```text
NOT a stability claim        the campaign's subject is architecture and assurance
NOT a conformance claim      C10 is where conformance is qualified; nothing here substitutes
NOT "three independent       one front end, three execution strategies, a named reference
    implementations"         engine (the HIR oracle), and six INVISIBLE authorities. EI6's
                             calibrated public wording stands, amended once by measurement:
                             "trap categories" narrowed to "the trap category vocabulary",
                             because a one-sided mis-assignment IS caught
NOT tensor-track progress    the tensor track remains deferred research on Gate 7's terms.
                             AS6 quarantined the extension; it did not advance it
```

## 5. Open, and carried forward

```text
DEV-213        LSP whole-package analysis cached per open URI, invalidated per URI. Demonstrated
               at HEAD. Owner-ruled non-blocking; next bounded LSP correctness packet.
               Until it closes, `workspace/symbol` correctness under MULTI-FILE EDITING is a
               qualified claim
DEV-012        C8's interactive editor validation, seven features unvalidated (CD-385)
Gate C9 Part B blocked pending second-artifact evidence; ONNX alone authorises no generalisation
AS8-R1..R15    mutation and evidence residuals; R3 and R15 discharged
AS8-DA-001..6  duplicate-authority dispositions SETTLED and scheduled after Sprint 4:
               DA-001/DA-005 consolidate; DA-002/003/004 remain separate and gain an exhaustive
               parity/drift test over the closed `RuntimeFn` set; DA-006 unchanged
branch coverage unavailable from this toolchain; not fabricated, not claimed
```

## 6. Landing

**Both stages landed 2026-08-09.**

```text
develop  b33b3e7
  +-- PR #10  Sprint 3 (AS0-AS5)       merged 645997d   MERGE COMMIT, 2 parents
  +-- PR #11  Sprint 4 (AS6/AS7/AS8)   merged d79ad03   MERGE COMMIT, 2 parents
develop  d79ad03   841 commits
```

No rebase, squash or cherry-pick, by owner ruling. **Verified after the fact rather than assumed:**
every packet SHA the governance records cite still resolves —

```text
6050efa  977b7a3  4c4311a  31246c0  5190d1b  e7bb95d  645997d     all OK
```

Gate evidence at each stage:

```text
stage 1   CI and C7.8 both green on 6050efa; merge-tree clean; develop carried one commit
          (ROADMAP.md) that the branch did not, so a merge commit was REQUIRED, not preferred
stage 2   exact-head Tier-3 CI run 31290518438 on 4eea128 — 24 jobs, 0 failures
```
