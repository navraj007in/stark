# STARK Compiler STATE

# Current position

*Charter §2.4 position line. Updated 2026-08-10. **Read this block, not the chronology below.***

```text
Gate: POST-C10 (no gate active)  Active packet: WP-ARCH-CLOSE (CD-400/401) — **AC1, AC2 and AC3
                                 MET**. AC3's two runs are GREEN on frozen SHA 915e565 (attempts 1
                                 and 2, 24/24 each, no rerun-to-green), so the §8 PRE-ALPHA COHORT
                                 GATE IS OPEN. AC5 IN PROGRESS, zero Class-D in the swept
                                 categories. AC4/AC6/AC7 open. AC5-F1 is now DEV-236, RULED under
                                 CE1. Architecture closure remains PROVISIONAL. Prior: AC2 MET, AC3 repair
                                 landed, AC3 exit NOT met — the two-run count is RESET: run 1
                                 at cd6732f is historical under §13, and CI at d300d3d FAILED
                                 (AS2 guard, mine, repaired). AC1 COMPLETE — DEV-160 RESOLVED,
                                 probe verdict POSITIVE. **AC5 COMPLETE — zero Class-D**; F4/F5/F7
                                 open and owned. **AC4: all 11 authorities addressed**, F1/F2/F3/
                                 F5/F7 resolved, F4/F6 awaiting owner disposition. **AC6 COMPLETE**
                                 — one overstatement corrected. Reopen rule not yet committed.
                                 Population A 8
Blocked: none — C10 CLOSED PASS-WITH-DEVIATIONS at 076b4dc (CD-397).
                `develop -> main` AUTHORISED (CD-399, compiler tree 860e33a, CI 24/24 green).
                CD-398's authorisation was SPENT on PR #21 and covered tree 5967a42, 18 commits
                behind; CD-399 is the new decision that tree change required
Compiler baseline: Core=done  MIR=done  Native=done — qualified subset, CI 24/24 + C7.8 green
                at 860e33a
Population A: 8 open — DEV-140..145 (the supported-subset boundary), DEV-221,
              DEV-233 (the interpreter loses output written before a trap). DEV-236 RESOLVED
              2026-08-12: `println` now enforces its own `T: Display` bound at the generic
              definition, answered by the bound authority — CD-401's architecture test PASSES.
              DEV-160 RESOLVED
              2026-08-12 under WP-ARCH-CLOSE AC1 step 2 — the thunk absorbs the call that produced
              the borrow, so the cross-block programs build and run. DEV-235 RESOLVED
              2026-08-12 under WP-ARCH-CLOSE AC3: the promotion-gating check failed because an
              accepted socket inherited the listener's non-blocking flag on macOS, not because of
              timing — see CD-400. DEV-229 RESOLVED
              2026-08-11: the prelude spellings are a fallback, not a pre-emption, which the
              namespaces made expressible. DEV-228 RESOLVED 2026-08-11: the
              resolver now carries the module/type/value namespaces NAME-RESOLVE-001 specifies,
              so the precedence exceptions have nothing left to order. DEV-232 and DEV-234 both RESOLVED
              2026-08-11: the `Copy` bound was repaired in both halves, which gave DEV-232's
              rejection a legal spelling and let it re-land. DEV-231, found by the audit's scope C
              and RESOLVED the same day. DEV-224 was mis-scoped on filing, REVISED,
              and RESOLVED 2026-08-11 as a capability increase; DEV-230 found by the resolution
              audit and RESOLVED the same day.
              DEV-222/223 RESOLVED 2026-08-11, plus DEV-225/226/227 found by the same audit and
              resolved on arrival
Primary remaining compiler capability: NONE. DEV-160 — the only one any written code reached —
                 is RESOLVED 2026-08-12. The remaining eight are the six supported-subset
                 boundaries, one ergonomic entry, one interpreter-observation entry
Next strategic milestone: standalone toolchain / C9 Part B second artifact
Optional tracks: ArtifactInfra=blocked (C9 Part B, second artifact)
                 TensorExpansion=blocked (Gate 7 DEFER, unchanged)
```

## What is true right now

| | |
| --- | --- |
| **Active packet** | **WP-ARCH-CLOSE — AUTHORISED AND ACTIVE (CD-400, 2026-08-12)**, `plans/WP-ARCH-CLOSE-ARCHITECTURE-CLOSURE-QUALIFICATION.md`. Three outcomes: PASS / INCOMPLETE / FAIL-ARCHITECTURE. **AC2 MET** — `starkc/docs/conformance/NATIVE-CONFORMANCE-MATRIX.md` is generated from a live compiler run and drift-gated in both directions (falsified both ways before being trusted); DEV-140..145 all present as executable boundary probes. **AC3's repair has landed — DEV-235 RESOLVED, population A 10 -> 9 — and AC3's exit is NOT met**: it also requires two complete clean CI runs with no rerun-to-green. **AC1 step 1 DONE and POSITIVE** — the borrow-origin analysis moved from the native emitter to `starkc/src/mir/borrows.rs` (owner, CE3), which is a repair at an owning authority with no exception required; DEV-160's capability half is untouched and population A stays 9. Next is AC1 step 2 (cross-block absorption), then AC4. The AC7 triage field and the `FINAL_REPAIR_SHA` freshness rule bind from CD-400 onward. *Previously:* **the post-C10 deviation repair programme CLOSED 2026-08-10** (`STARKLANG/docs/compiler/audits/POST-C10-DEVIATION-REPAIR-REPORT.md`). Baseline `689d26d`, final candidate `5967a42`, **population A 13 -> 8**. RESOLVED: DEV-180, DEV-220, DEV-157, DEV-168, DEV-159, DEV-165 (population B); DEV-120 reclassified as a documented limit, DEV-167 closed by owner decision under CE1. DEV-160's **rustc E0502 leak is SEALED** — a STARK-owned named refusal replaces the generated-Rust error — while its cross-block capability stays OPEN. DEV-140..145 assessed individually and **repair DEFERRED by owner decision**: not one of the six shapes is used by any first-party package, and they continue to define the supported native subset. DEV-221 newly registered (`Display::fmt` on a bounded generic receiver, ergonomic; `x.fmt()` works). CI and C7.8 Native Capabilities both green at the final candidate across Linux, macOS and Windows |
| **Promotion** | **`develop -> main` AUTHORISED — CD-399 (owner, CE8, 2026-08-11)** for compiler tree `860e33a`, CI `31468998989` 24/24 green. It states the fact the decision turns on: population A is 8 on `main` and 11 here, and the three added were DISCOVERED, not introduced — `main` has DEV-228/229/233 too and has never recorded them, while carrying ten defects this tree closed. Below is the SPENT authorisation it replaces. **`develop -> main` AUTHORISED — CD-398 (owner, CE8, 2026-08-10)**, the separate decision CD-397 said would follow it. Candidate **PR #21**, compiler tree `5967a42`, merge commit preserving history. **Conditional on `main`'s required `CI complete` going green for this tree** — a red required check withdraws the authorisation rather than inviting an override. It promotes a branch and **does not upgrade any claim**: PASS-WITH-DEVIATIONS, unsigned distribution, and all eight open population-A entries survive the merge unchanged |
| **Superseded packet** | *(historical)* **C10-0, P, A1, A2, B, C, D, E, F ALL COMPLETE**; DEV-012/213/214 all closed. Toolchain branch INTEGRATED (PR #15, develop `eb60dec`); **§8.2a re-run DONE** — 31 stale, 9 re-run, 9/9 reproduced. **ALL ELEVEN EXIT CRITERIA MET** at `076b4dc`. PR #16 merged; E9 (23 A + 5 B + 20 C, all owned), **E10 green in 28/28 jobs**, E11 swept. **C10-Q DECIDED.** Reproduction pass COMPLETE over all of population A (`audits/C10-Q-REPRODUCTION-PASS.md`): **7 of the 23 deviations at the anchor do not reproduce** — DEV-083/122/161/162/177/178/181 — so **population A is 16**, every entry observed at the candidate rather than inherited (DEV-159, a build race, carried conservatively). NO deviation accepts what the spec forbids. **DECIDED: PASS-WITH-DEVIATIONS (owner, CE8, 2026-08-09) — GATE C10 CLOSED.** **DEV-180 SCHEDULED (owner ruling 2026-08-09): its own packet, immediately after C10-Q closes and not before** — binding a genuine reference for `&mut self` changes what the HIR oracle means by a mutable receiver, and the oracle is what every engine-agreement claim is measured against. Its three prerequisite questions are answered in the ledger so the packet does not restart the investigation |
| **Active branch** | `develop` — Sprint 3 and Sprint 4 both landed as merge commits (`645997d`, `d79ad03`) |
| **Gates C0–C8** | CLOSED. C8 closed short on one requirement by owner ruling (CD-385) and DEV-012 stays open for seven features |
| **Gate C9** | Part A CLOSED (C9.0/C9.1/C9.2). **Part B DEFERRED** pending second-artifact evidence; no provider generalisation from ONNX alone (CE7). **Does NOT block C10** — CD-395, OD-1 |
| **Gate C10** | **CLOSED 2026-08-09 — PASS-WITH-DEVIATIONS** (owner decision under CE8). Claim, residuals and derivation: `C10-Q-EVIDENCE-PACKAGE.md` §3.2/§3.2a. **16 deviations**, every one reproduced at the candidate rather than inherited — seven of the 23 at the anchor did not reproduce and were closed. Named residuals, accepted rather than required: robustness targets **T3 and T7 declared and NOT RUN** (no robustness claim over either), and **seven security surfaces with a defence and no falsifier** (R-S03/05/06/09/11/14/15) — named, not claimed. Conformance is per-rule for 56 of 168 granular rules. Distribution is integrity-verified, NOT authenticated. **This did not authorise `develop -> main`** — CD-398 did, separately, on 2026-08-10. **The 16 is the count AT THE C10 DECISION and is not current** — the post-C10 repair programme took population A to **8**; see the Active packet row |
| **Sprint 4** | **CLOSED.** AS6 (CD-390), AS7 (CD-391, criterion 2 re-qualified CD-393), AS8 (CD-394), Tier-3 closeout PASS |
| **Campaign B** | **EXITED PASS 2026-08-09** — `audits/CAMPAIGN-B-EXIT-REPORT.md`. It gates C10 and makes no stability or conformance claim itself |
| **Native backend** | SELECTED — generated Rust, behind verified MIR, Cranelift kept open as a C7-gated migration (CD-026) |
| **Tensor track** | Deferred research on Gate 7's own terms. Platform progress is **not** permission to reopen it |

## Where to look

```text
this file, top             the current position — you are reading it
this file, CD-398 down     the append-only decision record, newest first
starkc/docs/conformance/    NATIVE-CONFORMANCE-MATRIX.md — is a construct supported natively?
                            Generated, drift-gated, and the answer an external developer needs
AS8-MUTATION-FINDINGS.md   what the mutation trials actually showed
ENGINE-SHARED-FATE-...     which semantic rules no engine can independently check
state-archive/             closed gate detail and session records, verbatim
ROADMAP.md (repo root)     the one live forward plan
```

## Known open, at a glance

**SUPERSEDED FOR POPULATION A, 2026-08-10; the count moved again on 2026-08-12.** The census below
is the state at the C10 anchor and is kept for provenance. **Population A is 9 today** —
DEV-140/141/142/143/144/145, DEV-160, DEV-221, DEV-233 — per the position line at the top of this
file, which is the authority. It reached **8** through
`STARKLANG/docs/compiler/audits/POST-C10-DEVIATION-REPAIR-REPORT.md` (2026-08-10), rose to 11 and
then fell to 10 as the 2026-08-11 resolution audits discovered and closed entries, and fell to 9
when CD-400 resolved DEV-235. Every one of those figures was computed with
`python3 starkc/scripts/c10-deviation-populations.py`, which remains the only trustworthy way to
ask. Populations B and C below are
unchanged **except** DEV-165, which the repair programme resolved. Read the entries below as history;
read the report for what is open.

*Prior correction, 2026-08-09 (CD-395, finding F2): this block listed TWO deviations; there were
THIRTY-TWO. It was not wrong about the two, and it says of itself that it is a summary — but a
qualification session that trusted it would have carried 2 instead of 32. The three populations are
frozen separately by OD-3.*

```text
POPULATION A — as at the C10 anchor (SUPERSEDED; now 8)       26 OPEN + 1 accepted + 1 dormant

  live OPEN by the last ledger heading                                               26
                            OD-7 adjudicated all 8 unsettled entries and BACKFILLED the 6 that
                            lived only here. `c10-deviation-populations.py` now reports
                            ADJUDICATE = 0 — population A is fully resolved
    DEV-012 interactive editor validation, 7 of 10 features    -> C10-P, needs a human
                                                                  in an editor (MANUAL evidence)
    DEV-140/141/142/143/144/145   the six CD-342 "layer defect" registrations —
                                  these BOUND THE SUPPORTED SUBSET and are load-bearing
                                  for any native-conformance claim
    DEV-120 native call-depth exhaustion      DEV-122 span source-identity gap
    DEV-167 Display::fmt has no to_string()   DEV-168 qualified core-trait call, no MIR lowering
    DEV-172 no signed type expresses its min  DEV-177 generic-parameter shadowing accepted
    DEV-178 generic context not retained      DEV-180 HIR flattens &mut self receivers
    DEV-181 assignment-RHS borrow blocks it   DEV-186 LSP unbounded Content-Length
    DEV-214 a left-associative operator chain ABORTS the compiler with a stack overflow
            (65 terms on a 2 MiB thread stack). Found by C10-B. OWNER CALL: every fix
            changes the accepted set, the architecture, or only moves the cliff

  OPEN HERE, owning NO heading in KNOWN-DEVIATIONS.md                                 6
    DEV-156 stark fmt evicts member doc comments   DEV-157 no MirTy::Never in the backend
    DEV-159 native build races its dependency      DEV-160 whole-value projection borrows
    DEV-161 ambient CARGO_TARGET_DIR breaks builds  DEV-162 read through a whole-value accessor
    ^^ a C10.7 check reading only the ledger would not see these. CD-395 finding F3

  ADJUDICATED by OD-7 (owner, 2026-08-09) — none remain unsettled
    CLOSED / retired    DEV-010 (superseded by C8)   DEV-020 (confirmed design)
                        DEV-021 (verified correct)   DEV-196 (Core(File) not lowerable at all;
                                                     the reachability test is KEPT as a premise
                                                     guard)
    OPEN, accepted      DEV-005 (CLI warning-policy drift; needs ONE current-head reproduction
                                 before C10-Q, because the entry is old enough that a later
                                 change may already have removed it)
                        DEV-083 (impl-head concrete position vs unresolved receiver argument;
                                 constrains the Core Stable claim, does not block C10)
    ACCEPTED-INDEF.     DEV-011 (doc comments as trivia — no normative requirement demands
                                 otherwise; a representation preference is NOT a conformance
                                 defect, and it was NOT "fixed")
    DORMANT             DEV-179 (unreachable while iterator map/filter is refused by E0105;
                                 not closed, because the hazardous code remains; not counted
                                 live, because nothing can reach it)

POPULATION B — release/distribution (constrains WORDING, not conformance)
    DEV-165 connect_timeout accepted and ignored -- RESOLVED 2026-08-10 (`1913b19`, `5967a42`);
    standalone toolchain PARTIAL;
    offline package build NOT PROVEN; signed distribution NOT PROVEN (integrity, not
    authenticity); x86_64-apple-darwin tier-3 PACKAGED AND NEVER EXECUTED (F1)

POPULATION C — assurance residuals (constrain CLAIM STRENGTH; assert NO defect)
    AS8-R1/R2/R4/R5/R6/R8/R9/R10/R12/R13/R14 live; R3, R15 DISCHARGED; R7 is a method finding
    AS8-DA-001..006  DA-001/005 CONSOLIDATE (outside C10); DA-002/003/004 REMAIN SEPARATE and
                     owe a RuntimeFn parity/drift test (test-only, permitted in C10-D);
                     DA-006 KEEP
    RA-LAYOUT unmeasured; RA-LINTS suppresses two deny-by-default lints in generated code
    DEV-017   the coverage DB cannot express per-rule +/- evidence. C10-A1 measured the
              consequence against the CORRECTED denominator: 85 of 168 granular rules are
              cited only through an aggregate runner
    branch coverage unavailable from this toolchain — not claimed, not fabricated

Gate C9 Part B  DEFERRED, and does NOT block C10 (CD-395, OD-1)
(PR #10 / #11   RESOLVED — both merged as merge commits; every cited packet SHA still resolves)
```

**This block is a summary and is not authoritative over the records below it.** Where they
disagree, the dated record wins and this block is stale — fix it in the same change.

---

## CD-401 — AC3's two-run protocol fixed on one frozen SHA; AC5-F1 ruled under CE1 (2026-08-12)

Two owner decisions, taken together because the second creates the repair the first must precede.

### Decision 1 — AC3 is a reliability experiment, so the TREE IS HELD CONSTANT

**Same-SHA reruns count. Two distinct trees are NOT required, and requiring them would be
methodologically weaker.** AC3 asks *does this qualification tree pass reliably, or is there latent
intermittency?* Holding the code constant isolates that variable; demanding a second landing
introduces a code-change confound and turns one intermittency test into two unrelated observations.

```text
AC3 qualification tree = 915e565

RUN 1   complete full CI execution, this SHA, first attempt, must finish SUCCESS
RUN 2   full CI execution again, same SHA, initiated only AFTER run 1 succeeded,
        every job executed again, must finish SUCCESS

FORBIDDEN
        rerunning failed jobs to obtain green
        selective rerun
        retrying a failed RUN 2 until two greens exist
```

A fresh workflow invocation with its own run id is preferred where the workflow can be dispatched at
a SHA; **a full "rerun all jobs" of the already-successful run 1 is equally valid** — it is not
rerun-to-green, it is asking a green tree to reproduce its result. The attempt number is recorded
explicitly either way.

**The failure rule is the part that makes this evidence rather than sampling:**

```text
run 1 green, run 2 red  ->  AC3 = INCOMPLETE. Investigate the red.
                            DO NOT run a third hoping for green.
```

If run 2 exposes a reliability defect whose repair changes the tree, a **new freeze SHA** is
established and the two-run count resets. That is what stops successful executions being
cherry-picked.

### Decision 2 — AC5-F1 (now **DEV-236**), under CE1: enforce `Display` at the generic DEFINITION

**The obligation is checked where the generic is written. Interpolation is NOT weakened.** The
normative text substantially settles it: `PRINT-DISPLAY-001` defines `print`/`println`/`eprint`/
`eprintln` as ordinary generic functions constrained `T: Display`, **not syntax hooks**; Core states
interpolation uses ordinary trait resolution and that a generic parameter must carry a bound
actually supplying `fmt`; and `TYPE-METHOD-003` says a parameter's trait capabilities come from its
declared bounds *and their resolved identities*.

```text
fn show<T>(x: T)          { println(x); }   REJECT at the definition
fn show<T: Clone>(x: T)   { println(x); }   REJECT at the definition
fn show<T: Display>(x: T) { println(x); }   ACCEPT   -- Core Display identity
trait Display { ... }                       REJECT unless the bound resolves to
fn show<T: Display>(x: T) { println(x); }            the CORE Display identity
```

The current `println` behaviour is not preserved: it lets the front end accept a generic body whose
declared constraints are insufficient and pushes the failure into monomorphisation, which is the
acceptance-boundary defect class this programme exists to remove.

**The repair must NOT be a `println` special case.** It belongs in the authority that checks generic
callee obligations:

```text
println<T: Display>  --called with U-->  caller environment must prove  U : CoreDisplay
                                          yes -> publish/use callable
                                          no  -> FRONT-END diagnostic
```

If `println` bypasses ordinary generic-bound checking because it is represented as a builtin, it is
**routed through the existing bound-obligation mechanism** rather than gaining another
`if callee == println`.

> **This repair is itself an architecture test, and that is deliberate.** If an apparently simple
> obligation cannot be expressed through the existing generic-call/bound authority, **that is a more
> serious architecture finding than F1** — and it is exactly §4's "consumer patched because the
> owning authority cannot express the rule". It would be recorded as such rather than worked around.

**Registered as `DEV-236`, and the AC5 audit's own classification was corrected with it.** F1 was
first filed Class C on the reasoning that the divergence was a policy choice needing an owner. It is
not a choice: `PRINT-DISPLAY-001` and `TYPE-METHOD-003` already decide both halves, and both were
verified verbatim before the correction. **A rule the specification already settles is
non-conformance, not architectural residue** — it belongs in the deviation ledger with a DEV number,
not in an A/B/C/D class. Population A 8 -> 9.

Triage records `Architecture trigger: NONE`, with the reason stated rather than assumed: the bounds
are present, resolved and identity-carrying; the obligation checker is simply not consulted for this
callee. **One authority not consulted is not a missing authority**, so it is not AC7-D. If a repair
attempt shows the authority cannot express the obligation, that classification is revisited on the
evidence — which is Decision 2's own architecture test.

**Blast radius measured before endorsing the breaking change: zero.** No first-party generic function
prints; every `println` under `packages/` is on a concrete type.

### AC3 RESULT — both runs GREEN on the frozen SHA. The condition is SATISFIED

```text
AC3 qualification tree      915e565

RUN 1   CI 31575087419   conclusion=success   ATTEMPT 1   24 jobs   0 failed
        C7.8 31575087391 conclusion=success   ATTEMPT 1    4 jobs   0 failed
RUN 2   CI 31575087419   conclusion=success   ATTEMPT 2   24 jobs   0 failed
        full rerun of every job, same SHA, initiated only after run 1 succeeded

no failed job was rerun to obtain either result
no selective rerun
no third run
```

Run 2 shared run 1's concurrency group — the workflow keys it on the tested SHA — so it **queued
rather than raced**, which is what kept it off the fixed ports 39187-39191 that a parallel run
collides on. That is why the docs push was held until both runs finished: a push during either run
would have started a DIFFERENT SHA in a DIFFERENT group, in parallel, and `Address already in use`
would have reddened a green tree at the cost of the whole freeze.

**What this evidence licenses, stated precisely.** Two greens is a real result and a weak one. At the
flake rates this repository has actually observed, two samples pass with roughly 64% probability
*despite* a live intermittency; at 1-in-20 it is 90%. So the claim is **"no intermittency was
observed in two samples of this tree"** — not "there is no intermittency". Both run ids and both
attempt numbers are recorded above so a later reader can check rather than trust this summary.

### AC3 EXIT MET, and the cohort gate is OPEN

```text
AC2   MET   cd6732f -- the generated, drift-gated native conformance matrix
AC3   MET   DEV-235 repaired, and two clean CI runs on 915e565 with no rerun-to-green
            -> the §8 pre-alpha cohort gate is OPEN
```

**The release state must say exactly this while the cohort runs** (CD-401):

```text
Architecture closure:      PROVISIONAL
AC5:                       IN PROGRESS
Class-D findings:          none IN THE SWEPT CATEGORIES
Unswept categories:        listed in AC5-PATCHWORK-AUDIT.md §5
Known limitation:          DEV-236 -- `println` on a generic parameter does not enforce its own
                           `T: Display` bound. `fn show<T>(x: T) { println(x); }` compiles and
                           then fails at MIR. Workaround: write the bound, `T: Display`
```

Acceptable for a controlled pre-alpha cohort, because the cohort is itself discovery evidence.
**Not** acceptable for WP-ARCH-CLOSE PASS or any unconditional public architecture-stability claim.

**DEV-236 belongs in the cohort's known-limitations document before anyone is invited.** It is the
first generic a newcomer writes, and today it produces a correct compiler error about the wrong
layer.

### Sequencing, and what today's runs can and cannot be reused for

```text
1  run 1 at 915e565 completes
2  if green, full second CI run at the SAME SHA
3  both green -> AC3's two-run reliability condition SATISFIED
4  AC2 -- ALREADY MET (cd6732f, the generated drift-gated conformance matrix).
   The cohort gate waits on AC3 alone
5  release the freeze; implement F1 at the generic-bound authority
6  continue AC5's unswept categories, engine-local reconstruction first
```

**Today's two runs are valid evidence for COHORT ENTRY and cannot be reused as the final
WP-ARCH-CLOSE freshness runs.** F1 and the remainder of AC5 will land compiler-affecting repairs
after them, so final closure reruns from the eventual `FINAL_REPAIR_SHA`. This preserves §13 without
holding the cohort behind work the cohort gate does not require.

### What the release state must say while the cohort runs

```text
Architecture closure:      PROVISIONAL
AC5:                       IN PROGRESS
Class-D findings:          none IN THE SWEPT CATEGORIES
Unswept categories:        explicitly listed (AC5-PATCHWORK-AUDIT.md §5)
```

Acceptable for a controlled pre-alpha cohort, because the cohort is itself discovery evidence.
**Not** acceptable for declaring WP-ARCH-CLOSE PASS, or for any unconditional public
architecture-stability claim.

## AC4 OPENED — pattern legality had an unguarded arm (2026-08-12)

`STARKLANG/docs/compiler/audits/AC4-ADVERSARIAL-CAMPAIGN.md`. Trials run through AS8's harness,
extended rather than replaced, because it enforces the evidence invariant structurally: **a trial
declares KILLED or SURVIVED before it runs**, and the harness reports CONFIRMED / UNEXPECTED.

**Mapping AS8's 26 trials onto AC4's eleven authorities: 2 covered, 7 partial, 2 with NO trial.**

**The finding.** Pattern legality — chosen first because DEV-222/223/225/226/227 all landed in one
day, every one a pattern that compiled, reported nothing and silently never matched. Its authority
`resolution_is_pattern_legal` has three arms; two were guarded and **the `Res::Item` arm had no
control at all**. Replacing it with `Res::Item(_) => true` left the whole suite green.

```text
mod m { pub fn f() -> Int32 { 1 } }
match n { m::f => .. , _ => .. }

unmutated   E0200             rejected
mutated     no diagnostic     ACCEPTED as a pattern that never matches -- DEV-227 restored
```

A control was added and the mutation now dies. **`mir::borrows`'s three trials all CONFIRMED**,
including one declared SURVIVED — the aggregate filter is precautionary with no control reaching it,
which is now an explicit classification in the harness rather than a doc-comment residual.

**Three invalid probes preceded the valid one**, and the campaign record says so: a bare identifier
in pattern position BINDS a fresh variable and never reaches the authority, so the probe reported no
diagnostics both mutated and unmutated and was briefly misread as a live defect. **The tell is the
unmutated run agreeing with the mutated one** — the same signal that caught an unreaching program in
the `mir::borrows` trials.

**AC4 is NOT met.** Seven authorities are PARTIAL, one (generic specialization environment) still has
no trial, and the shared-fate register has not been reconciled. **Do not read "2 covered, 7 partial"
as 82% done** — pattern legality would have counted as nearly covered, and its unguarded arm is the
argument against counting.

## AC6 COMPLETE — a public claim overstated by a dropped clause (2026-08-12)

`STARKLANG/docs/compiler/audits/AC6-PUBLIC-CLAIM-SWEEP.md`. Six public surfaces swept; the
prohibited *"three independent implementations"* claim remains absent everywhere.

**The finding is a truncation, not an invention.** EI6 approved wording ending *"…and by recorded
residuals where no control yet exists"*. The website shipped it without that clause:

```text
shipped   "Those rules are listed in a public register and checked separately."
```

The dropped clause is what made the sentence true. Without it the claim is that ALL such rules are
checked — and the paragraph names three. AC4 measured them:

```text
Copy eligibility          c61f_structural_copy, 13 tests            CHECKED
destructor eligibility    independent_evidence: "none"              risk CRITICAL
trap category vocabulary  "none, and none is constructible"
```

**Two of the three named examples had no independent control; one cannot have one.** Corrected to
restore EI6's meaning, including saying out loud that one rule admits no constructible check —
`ESF-TRAP-001a`, where a wrong `TrapCategory` enum makes every engine and the corpus manifest wrong
together.

**No claim about MIR verification needed correcting, because none is published.** `ESF-VERIFY-001`
was created during the AC4 reconciliation so that a future *"independently verified MIR"* claim
would have a sensor. The sensor exists **before** the claim, which is the right order.

**Residual, owned rather than rediscovered:** nothing mechanically prevents the next consuming
change from truncating an approved sentence the same way. A drift gate shaped like
`cohort_limitations_are_current.rs` — assert the register's verdict for each named rule against what
the public copy says — is the durable fix and is not built.

## AC5 COMPLETE — every category swept, zero Class-D (2026-08-12)

`STARKLANG/docs/compiler/audits/AC5-PATCHWORK-AUDIT.md`. Every category on AC5's search list has a
stated denominator and a classification.

```text
findings, by disposition
  DEV-236   left the A/B/C/D scheme entirely -- the spec already settles it. RESOLVED same day
  F2 F4 F5 F6 F7   Class C. F3 and F6 REPAIRED in the audit; F4, F5, F7 owned and open
  AS8-DA-002/003/004, has_user_destructor   Class B, deliberate, both copies mutation-killed
  is_copy, precedence, builtin-keyed dispatch   Class A, legitimate
  CLASS D                                    ZERO
```

**Zero Class-D is the condition for AC5 not to force FAIL-ARCHITECTURE. It is not a PASS** — §14
gates that on AC4's campaign too.

**Two results are worth more than the findings.** The category most likely to hold a Class-D —
engine-local reconstruction, where AC1 found a genuine instance — came back clean *on measurement*:
the MIR interpreter has ONE HIR reference in 2,878 lines, and every backend reference is an identity
payload `MirTy` embeds rather than a lookup. And `packages/`, never previously examined, holds
eleven historical notes that were correctly updated when their defect was fixed, and exactly one
live workaround — whose own comment named the condition that had just been met.

**What "complete" does NOT claim**, stated in the audit rather than left to a reader: every category
was examined, not every instance found. A patchwork using none of the searched vocabulary would not
appear.

**F7 is the finding most likely to matter next.** `layer_audit.rs` enforces reachability at the
front end -> lowering boundary; **nothing asks the same question at lowering/verify -> backend**. 109
refusal sites, nine citing work packages closed long ago, and no test distinguishes "unreachable arm
with stale text" from "unregistered acceptance boundary missing from the conformance matrix". This
category already produced a real defect today — F3, DEV-160b's refusal still describing a mechanism
removed hours earlier.

## DEV-236 RESOLVED — the architecture test in CD-401's Decision 2 PASSES (2026-08-12)

`println` enforces its own `T: Display` bound at the generic definition, per the CE1 ruling.
**Population A 9 -> 8.**

**The architecture test is the result worth recording, not the fix.** CD-401 required the repair to
land at the authority that checks generic callee obligations, and named the alternative: had the
obligation been inexpressible there, that would have been a **more serious finding than the
deviation** — §4's *"consumer patched because the owning authority cannot express the rule"*.

It was expressible. The defect was one line —

```rust
Ty::Param(_) => true, // discharged by the caller's own bound
```

— asserting a discharge that never happened, since `builtin_type` types the print family's parameter
as a bare inference variable and no obligation was ever attached. It now calls
`param_declares_bound(name, "Display", Some(Res::CoreTrait(CoreTrait::Display)))`, an authority that
already existed and already compares resolved identities. **No `if callee == println` anywhere.**
`Architecture trigger: NONE` is therefore CONFIRMED by the repair rather than predicted at triage.

**The repair exposed a second defect, and the codebase had already written down the rule it broke.**
The first version rejected `fn show<T: Display>(x: T) { println(x); }` — a bound plainly written —
because answering `Ty::Param` from declared bounds made a Pass-3 obligation **scope-sensitive** while
it still carried no scope. `DeferredDisplayPlan`'s doc comment states the general rule: *"a deferred
obligation may read resolved types freely, but any scope-sensitive question it asks is a question
about a scope that no longer exists. Capture the scope with the obligation."* `display_checks` now
carries `generic_scope` as the plan queue always did.

```text
revert the repair                     killed by 4 tests
keep the check, drop the IDENTITY     killed by exactly 1 -- the identity test
```

**Consequence for AC3, stated rather than left to be discovered.** This is a compiler-affecting
repair, so under §13 the two green runs on `915e565` are now historical **for closure purposes**.
They remain valid for **cohort entry**, exactly as CD-401 anticipated. Final closure still reruns
from the eventual `FINAL_REPAIR_SHA`.

## CD-400 — WP-ARCH-CLOSE AUTHORISED as the active packet; AC3's first repair has landed (2026-08-12)

**Owner decision, 2026-08-12.** `WP-ARCH-CLOSE` — the final compiler architecture closure
qualification — is the **active packet**, executing under
`STARKLANG/docs/compiler/plans/WP-ARCH-CLOSE-ARCHITECTURE-CLOSURE-QUALIFICATION.md`. Until this
entry the position line said no packet was active while a closure package was being written, which
is exactly the state the record exists to prevent.

**This entry authorises the packet and takes none of its verdicts.** WP-ARCH-CLOSE's outcome is a
CE8 claim — architecture STABILIZED, and a published native conformance contract, are both public
technical claims requiring an evidence audit. Nothing below upgrades any claim.

```text
packet             WP-ARCH-CLOSE          AUTHORISED, ACTIVE
scope              AC1 DEV-160 probe · AC2 executable native conformance contract ·
                   AC3 qualification reliability · AC4 adversarial authority campaign ·
                   AC5 patchwork audit · AC6 public wording · AC7 reopen sensor
not a gate         it is neither a C-gate under COMPILER-ROADMAP.md nor a ROADMAP.md track
outcome model      PASS | INCOMPLETE | FAIL-ARCHITECTURE  -- three states, not two
                   difficulty produces INCOMPLETE; only an architecture finding produces
                   FAIL-ARCHITECTURE, and Architecture Stabilization is NOT reopened by
                   unfinished work
execution order    AC3, AC2, [pre-alpha cohort may start], AC1, AC4, AC5, AC6, AC7,
                   FINAL_REPAIR_SHA, rerun all evidence, decide
```

### Two rules bind from today, not at closure

- **The AC7 triage sensor is live.** Every new substantive compiler DEV entry carries an
  `Architecture trigger:` field — `NONE`, `AC7-A`..`AC7-F`, or `PENDING-CLASSIFICATION` — and **may
  not leave triage still pending**. The twenty-defect observation count is then derived from the
  ledger mechanically instead of being reconstructed from memory later. DEV-235's resolution entry
  carries the first one.
- **Final evidence must postdate `FINAL_REPAIR_SHA`.** No closure evidence may predate the last
  repair affecting the claim it supports. Evidence generated earlier is historical and cannot close
  the package. This is what stops closure evidence predating the defects it claims to cover.

### AC3 — DEV-235 RESOLVED, and the exit is NOT yet met

`DEV-235` is **RESOLVED** (ledger, 2026-08-12). The registration blamed "loopback socket timing";
the cause was a socket flag. `EchoServer`'s listener is non-blocking so its accept loop can poll a
stop channel, and **on macOS and the BSDs an accepted socket inherits `O_NONBLOCK` where on Linux
it does not** — so the echo thread's first read returned `WouldBlock` in microseconds, the harness
discarded the error, and the closed connection reached the client as EOF in place of its echo. A
standalone probe on macOS arm64 measured the inheritance directly (`READ Err(WouldBlock) after
3.667µs`); the five-second deadline was never consulted. **No timeout would have repaired this**,
which is why the package refuses timeout-tuning as a flake response.

`packages/stark-tls`'s peer already carried the fix and a comment describing the same divergence.
The two harnesses were written from one shape and only one was repaired.

The harness now also **reports its own failures** (`HarnessError::Echo`) instead of discarding
them, so this class of defect accuses the harness rather than the provider under test. Falsified by
removing the repair: both the new regression test and the original detach test fail deterministically
on macOS, the second now naming `WouldBlock` explicitly. Restored: 12/12 green, three consecutive runs.

```text
population A       10 -> 9      DEV-140..145, DEV-160, DEV-221, DEV-233
AC3 exit           NOT MET      it additionally requires two complete clean CI runs with no
                                failed job rerun to green, on all Tier-1 platforms. Neither has
                                been run at this tree
```

**This is one mechanism removed from one harness.** It is not a claim that the qualification lanes
are free of timing dependence: `send_frame`'s call-site deadlines and the TLS peer's remain,
unexercised and unchanged by this repair.

### AC2 — the native conformance contract is EXECUTABLE and drift-gated

`starkc/docs/conformance/NATIVE-CONFORMANCE-MATRIX.md` is **generated from a live compiler run** by
`starkc/tests/native_conformance_matrix.rs`, which validates it on every CI run on all three Tier-1
platforms. It answers, without `COMPILER-STATE.md` or the deviation ledger: *is this construct
supported natively, and if not, what happens when I write it?*

```text
20 boundary constructs measured
     6  SUPPORTED           all executed through HIR, MIR, native debug AND native release,
                            compared on the full normative observation by the same comparator
                            the C6 differential suites use
     8  REFUSED-BY-DESIGN   each with the STARK diagnostic that refuses it, by code
     6  KNOWN-DEVIATION     DEV-140..145, every one present, each a STARK-owned refusal
                            arriving before any code is emitted
```

**The drift gate was falsified in both directions before it was trusted**, which is the only reason
to believe it:

```text
published matrix hand-edited            -> FAILS, naming the line, published vs measured
   (DEV-140's row flipped to SUPPORTED)

compiler mutated, matrix untouched      -> FAILS
   (integer negation made to refuse)       "registered as lowering cleanly, but it was
                                            refused by MIR lowering"
```

The mutation was reverted; `git diff` on `lower.rs` is empty.

**No second classifier was created.** The probe inventory moved from `layer_audit.rs` into
`tests/support/layer_probes.rs`, and the audit's three-way verdict is now *projected from* the same
staged measurement the matrix renders — one traversal of the compiler, two readings. A matrix built
beside the audit would have been exactly the duplicate-authority shape AC5 exists to find.

**What the matrix does not claim**, stated in the file itself: it is a boundary inventory, not a
census of the language; a `KNOWN-DEVIATION` row is valid STARK refused by this compiler rather than
by the specification; and `SUPPORTED` means the four configurations agreed *on that probe*, with the
probe's own expectation pinned separately by `layer_audit` so agreement alone cannot carry a row.

```text
AC2 exit           MET for the boundary inventory. An external developer can determine the
                   status of these constructs without reading any governance document
open              the inventory covers the edges; extending it as new boundaries appear is
                   ordinary maintenance of the generator, not a new packet
```

### AC3's two-run requirement: RUN 1 of 2 recorded

```text
tree               cd6732f (this packet's AC2+AC3 landing)
CI                 31563159250   24/24 jobs   ATTEMPT 1
C7.8 Native Cap.   31563159221    4/4  jobs   ATTEMPT 1
reruns             NONE. No failed job was rerun to obtain either result, and no required
                   check was waived. `attempt: 1` on both, read from the API rather than
                   asserted
Tier-1             `fmt, clippy, test` green on linux-x64, macos-arm64 AND windows-x64
```

Two claims this run is the first evidence for, both of which could only be tested off this machine:

- **The conformance matrix is platform-independent, as its own Tier-1 section says.** It was
  generated on macOS; Linux and Windows regenerated it and matched. A construct behaving differently
  on one of them would have failed the job there, naming the differing line.
- **The DEV-235 repair holds on Linux and Windows.** Linux never inherited `O_NONBLOCK`, so the new
  call is a no-op there — but the regression test forces the adverse interleaving on every platform,
  which had never been exercised deliberately before.

**One green C7.8 run is NOT evidence the flake is gone.** DEV-235 was intermittent and runs looked
exactly like this before it was found. The evidence that the mechanism was removed is the
falsification — deterministic failure with the repair backed out — not the colour of this run.

**A second clean run is still required, and it must not be manufactured.** Re-pushing this tree to
harvest a second green would defeat the rule's purpose, which is to catch intermittency; the second
run should come from the next commit that lands work here.

### AC1 — the architecture probe's FIRST result is POSITIVE, and DEV-160 stays OPEN

The borrow-origin analysis moved out of the native emitter into `starkc/src/mir/borrows.rs`, by
**owner decision under CE3 (2026-08-12)**. The placement was put to the owner rather than taken:
MIR-level module (chosen), keep it in the backend, or publish from `borrowck.rs` (rejected for now —
it answers a different question, returns only diagnostics, and works on HIR places whose mapping to
MIR locals is unconfirmed).

**This is NOT an architecture finding, and the reasoning matters more than the verdict.** A backend
computing borrow provenance looks like §4's *"semantic information reconstructed downstream because
the authoritative phase discarded it"*, which is why AC1 flagged it. On inspection it is not: the
HIR borrow checker asks *is this program legal*; this asks *what does this value borrow in the
lowered form*. Different questions, and the second has an owner. **Repaired at an owning authority,
no exception required** — which is the positive evidence AC1's hypothesis predicted.

**The concrete architecture evidence: a consumer patch disappeared because the authority became
correct.** The heuristic propagated a call's arguments into its result unconditionally, so
`send(u: &str, b: String) -> UInt64` recorded a scalar as borrowing the aggregate — and every
consumer carried a type check to undo it. Two such checks and a `by_value_tys` map built solely to
feed one of them are deleted. A fourth copy of AS4's `stores_a_reference` (the backend's private
`may_carry_borrow`, complete with the property-bearing wildcard that authority refuses) is deleted
with them.

**The mutation trials corrected themselves, and both readings are recorded:**

```text
FIRST TRIAL, simple shape       call-result KILLED by 2; move SURVIVED; dest guard SURVIVED
AFTER a (String, &str) shape    call-result KILLED by 3; move KILLED by 1
                                dest guard SURVIVED; aggregate filter SURVIVED
```

The first trial's survivors were an unreaching program, not weak rules — they masked one another.
**Three tests written as controls were not controls**, and would have been reported as coverage.
Two rules remain uncontrolled and are labelled precautionary in the module; both survived a mutation
verified to have applied.

```text
DEV-160            STILL OPEN. The capability half is untouched: the cross-block programs are
                   valid STARK and still do not build. Population A remains 9
AC1                NOT MET. This is step 1 of the owner's two-step: precise analysis first, then
                   reassess cross-block absorption with it in hand
```

**One control was withdrawn rather than counted.** `stark check --target-native` over the packages
does not reach `plan_for_call` at all — it scans runtime functions — so the 67-package sweep it
produced is not evidence about this change. The 33 first-party APPLICATIONS built natively are, and
they include `stark-get`, whose dependency `stark_http_client::follow` is what the first repair
attempt broke.

### CI run at `d300d3d` FAILED, and the failure was mine

7 of 24 jobs red, attempt 1, all tracing to **one** root cause:
`as2_one_pipeline::phase_calls_below_a_test_marker_are_a_known_set`.

`mir::borrows`'s inline `#[cfg(test)]` module assembles the front end by hand — parse, resolve,
typecheck, lower — to observe one body's lowered form. AS2's guard pins the exact set of files
allowed to do that below a test marker, and a new one appeared unregistered.

```text
fmt, clippy, test        linux-x64, macos-arm64, windows-x64     all three
C6.4 tier-1 qualification linux-x64, macos-arm64                 `-p starkc --test as2_one_pipeline`
C6.4 tier-1 agreement / CI complete                              downstream of the above
```

**Repaired the way the guard itself directs** — its failure message reads *"A new test that builds a
pipeline by hand is fine — add it to TEST_ONLY"*. It is a visibility registry, not a prohibition:
the point is that the scan's blind spot stays small, named and reviewed. `src/mir/borrows.rs` is now
in `TEST_ONLY` with the reason it needs to be there.

**Why local verification missed it.** Targeted suites were run — `--lib`, the DEV-160 suites, the
differentials — but not the source-scanning architecture guards, which live in test files that no
targeted invocation touches. There are eleven of them and **every one runs in under a second**:

```text
as1b_source_registry   as2_one_pipeline    as3_invocation_authority
as6_core_module_vocabulary   as6_core_session_isolation   as7_module_dependencies
dev121_boundary_inventory    dev121_view_producer_audit   dev135_field_move_paths
operand_move_inventory       c10c_security
```

All green after the repair. **Any change that adds a module, a test module, or a phase call should
run that batch before pushing** — it is seconds of local work against a full CI cycle.

**Not counted as a DEV.** No compiler behaviour changed and no program's meaning moved; an
architecture guard correctly refused an unregistered addition. Under CD-400's §12.3 list this is
test infrastructure and does not count toward AC7's twenty.

### AC1 step 2 — DEV-160 RESOLVED, and the probe's verdict is POSITIVE

**The capability half is closed.** `send(r.url.as_str(), r.body)` builds, runs, and agrees across
all four engine configurations, as does the originally reported three-argument shape. The thunk now
absorbs the call that produced the borrow. **Population A 9 -> 8.**

**The design was decided by a soundness argument that ruled out the cheap repair.** Laundering the
reference through a raw pointer at the call site would have needed no new plan structure and no
detector — and it is unsound. `slot.rs` says the thunk's `&'a mut` is *"what anchors every reference
it hands on"*; under Stacked Borrows, taking that `&mut` invalidates tags derived from any earlier
borrow of the slot, so a reference created before the thunk was entered is dead inside it however it
travelled. It must be created INSIDE, which brings the producing call with it.

**AC1's hypothesis is confirmed.** The capability landed inside the existing borrow architecture:

```text
engine-local semantic dispatch            NONE -- one new ThunkArg variant on the existing plan
downstream semantic reconstruction        NONE -- the origin relation is MIR's, from step 1
duplicated ownership authorities          NONE -- one detector, one plan, three consumers
backend-specific language restrictions    ONE REMOVED (DEV-160b), none added
precedence/order exceptions               NONE -- MIR argument order is unchanged; CD-007 holds
```

**Two defects this repair found in itself.** The first attempt built and then panicked at run time —
the call site still passed `_9.unwrap()` for a local nothing assigned any more, because the
`by_value` entry survived the `plan_args` replacement. And the Miri fixture for the new shape was
initially **unguarded**, reintroducing the exact failure the existing guard exists to prevent; worse,
that guard's extraction took the first token of each line and would have silently dropped the
`field_ref_raw` nested inside `as_str(...)`. The extraction now scans nested calls, a second guard
covers the new fixture, and it was falsified.

```text
Miri, CI's exact flags       27/27, incl. the_absorbed_producer_shape_is_sound_under_stacked_borrows
fixture guards               both pass; the new one falsified by shortening its declared shape
dev160_call_site_thunk       9    DEV-160c/d guards UNCHANGED; ordinary_calls_plan_nothing holds
ac1_dev160_probe             4    incl. both over-refusal controls
mir_differential           132    three_engine_differential 129
native ownership/linkage    36    33 first-party APPLICATIONS built natively, 0 failures
```

**What is NOT closed.** DEV-160c (provider argument sequences) and DEV-160d (a borrow outliving the
call) are unchanged and still refused by name — deliberately, since a repair that quietly widened
them would have been invisible. And **only one producer may be absorbed per thunk**: two would each
need their own predecessor edge, and the admission test allows one.

### AC3's two clean runs cannot be collected while repairs are still landing

`f780bb3`'s CI was **fully green — 24/24 and 4/4, attempt 1, no reruns**, all three Tier-1 platforms
plus the Miri lane. It still does not count, for the same reason `cd6732f` stopped counting: this
DEV-160 landing postdates it, and §13 disqualifies evidence predating the repairs it covers.

That is not a defect in the runs. It is the structure of the package: **the two clean runs are
collectable only once repairs stop**, which is exactly what §17 step 9 prescribes — establish
`FINAL_REPAIR_SHA`, then rerun all final evidence. Recording a running tally mid-packet and resetting
it on each landing is motion, not progress, and the count is better read as "not yet started" until
the repair sequence ends.

### AC3's run 1 no longer counts toward closure, by the rule this packet wrote

The `d300d3d` failure is not the only consequence. §13's freshness rule says closure evidence may not
predate the last repair affecting the claim it supports, and the AC1 landing plus this repair are
both later than `cd6732f`. **Run 1 at `cd6732f` is now historical evidence.** It was recorded above
with that caveat already stated — *"not a claim that it will survive to closure"* — and it has not.

```text
AC3 two-run requirement    reset. Both clean runs must come from the qualifying tree
§17 step 9                 unchanged: ALL final evidence is rerun at the end anyway
cohort gate                still shut
```

### The freshness rule already applies to this evidence, and says so

Under §13, these runs count toward **AC3's exit** — which gates cohort entry — and are **not** final
closure evidence. `FINAL_REPAIR_SHA` is not yet established, and any later compiler-affecting repair
(AC1's DEV-160 work is the obvious candidate) makes `cd6732f` historical. §17 step 9 therefore stands
unchanged: **all** final qualification evidence is rerun at the end, including these two runs. Recording
run 1 here is not a claim that it will survive to closure.

## CD-399 — `develop -> main` PROMOTION AUTHORISED for compiler tree `860e33a`, on a count that went UP (2026-08-11)

**Owner decision under CE8: the `develop -> main` promotion is AUTHORISED** (2026-08-11), taken
against the facts below after they were drafted by the session that produced the tree. CD-398 does
**not** cover this tree: it was over compiler tree `5967a42`, 18 commits behind, and it says in
terms that a commit changing a compiler input requires a new decision. This is that decision.

```text
promotion          AUTHORISED (owner, CE8, 2026-08-11)
candidate          develop @ 860e33a -> main
merge shape        MERGE COMMIT, history preserved -- as PR #12, #19 and #21 were
condition          main requires `CI complete`. This authorisation is for THIS tree on GREEN;
                   a red required check withdraws it rather than inviting an override
release wording    UNCHANGED. CD-397's PASS-WITH-DEVIATIONS still governs. This promotes a
                   branch and upgrades no claim
```

**The tree is the one CI verified.** `git diff --name-only 860e33a HEAD` is empty — HEAD *is* the
tree the run covered. CI `31468998989`: **24 of 24 jobs success**, including `fmt, clippy, test`,
`first-party package qualification` and `C6.4 tier-1 qualification` on all three platforms, plus
`C7.8 Native Capabilities` (`31468998772`) green separately.

### The count went UP, and that is the fact this decision turns on

`main` records population A as **8**. This tree records **11**. A promotion that raises the number
of known deviations deserves an explicit argument rather than a summary, so:

- **The 8 are unchanged and still present**: DEV-140..145, DEV-160, DEV-221. Not one degraded.
- **The 3 added — DEV-228, DEV-229, DEV-233 — were discovered, not introduced.** `main`'s compiler
  has all three; it has never recorded them. DEV-228 (one namespace map where NAME-RESOLVE-001
  specifies four) is as true of `main` as of this tree.
- **Ten defects were closed that `main` still carries, live and unlabelled**: DEV-222, 223, 224,
  225, 226, 227, 230, 231, 232, 234. Four of them are wrong-code — programs that compile, emit no
  diagnostic, and take the wrong branch. `main` has every one of them today.

So on defects *present*, this tree is strictly better. On defects *known*, it is three worse, and
the three are shared. **The number rose because knowledge rose.** Refusing the promotion does not
remove DEV-228 from `main`; it only removes the record of it.

### What the promotion would carry, named rather than implied

- **DEV-228 open and architectural.** The resolver has one namespace map where the specification
  names four, so `struct Pair` alongside `fn Pair()` is refused though NAME-RESOLVE-001 permits it.
  This tree adds the second precedence exception to that map. A third would be the wrong direction,
  and the model is an owner decision that this promotion neither takes nor forecloses.
- **DEV-229 UNCONFIRMED**, deliberately. The code path is certain; no probe yet separates "the
  user's declaration won" from "the builtin won and agreed".
- **DEV-233**, the interpreter losing output written before a trap. A debugging tax, not a
  correctness one.
- **DEV-140..145** unchanged: accepted-but-unbuildable, owner-deferred, no first-party consumer.
- **Distribution remains integrity-verified, NOT authenticated.** Archives unsigned. Unchanged by
  this promotion and not improved by it.

### What else is in the delta, beyond the compiler

Two package changes and one gate change, none of which alter a claim:

- `stark-cookie` added and then revised to a sum type; `stark-urlencoded` replacing `stark-query`
  and `stark-form`. 31 libraries plus `stark-get`.
- **Qualification coverage 25 -> 31 of 31 libraries.** Six had no case at all; `stark-csv` carried
  the largest test suite in the repository and had never been built natively. All six now build and
  run natively on all three platforms, verified by this CI run.

### The honest caveat about this session's evidence

CI failed once on this work, at `ab8da4d`, on a real defect: a borrow-check repair that broke a
compiler test the local sweep never ran. The repair was reverted, the blocking gap (DEV-234) was
found and fixed, and the repair re-landed verified against the full corpus with `--no-fail-fast`
— 231 binaries, 2,903 tests, 0 failures. As with CD-398, that CI caught something first is part of
why the final candidate is trusted rather than an argument against it. It is recorded here because
a decision should see the failed attempt, not only the passing one.

---

## CD-398 — `develop -> main` PROMOTION AUTHORISED, on a compiler that is strictly better than the one it replaces (2026-08-10)

**Owner decision under CE8: the `develop -> main` promotion is AUTHORISED.** This is the separate
decision CD-397 said would follow and did not accompany it. Candidate: `develop` at the tip of
**PR #21**, whose **compiler tree is the post-C10 repair programme's final candidate `5967a42`**.

The compiler tree, not a branch SHA, is what this authorisation is over. Verified rather than
asserted: `git diff --name-only 5967a42 HEAD` lists exactly two paths, both Markdown —
`COMPILER-STATE.md` and the repair report — so **no compiler input has changed since the tree CI
went green on**. A commit that changes one requires a new decision, not this one.

```text
promotion          AUTHORISED (owner, CE8, 2026-08-10)
candidate          develop @ PR #21 tip, compiler tree = 5967a42 -> main
merge shape        MERGE COMMIT, history preserved — as PR #12 and PR #19 were
condition          main is protected and requires `CI complete`. The authorisation is for THIS
                   tree on GREEN. A red required check withdraws it; it does not survive an
                   override
release wording    UNCHANGED. CD-397's PASS-WITH-DEVIATIONS still governs
```

**What the authorisation rests on.** The tree being promoted is not new work — it is the tree
`main` already carries, with a repair programme layered over it and nothing else. `main`'s compiler
is `689d26d`, which is precisely the programme's *starting baseline*, so the comparison is a strict
one rather than a trade: **population A 13 -> 8**, no deviation opened that was not also closed,
and no engine disagreement remains in the repaired set. CI and C7.8 Native Capabilities were both
green at `5967a42` across Linux, macOS and Windows — fmt, clippy, full tests, first-party packages,
C6.4 qualification, C6.5 corpus replay, mutation controls, the P1 REST workload, release package
smoke, external samples, and the DEV-160 Miri guard. That CI failed twice first, on real defects in
the new work, is part of why the final candidate is trusted rather than a caveat against it.

**The one change that is a boundary change, not a repair.** DEV-160's cross-block borrow shape used
to escape native lowering and reach the user as a raw `rustc E0502` inside `mod stark_proj` — the
exact outcome CD-374 claimed the named refusal prevented, and the finding CD-397 recorded and
deliberately did not act on. It is now a STARK-owned named refusal. **The capability limitation is
unchanged; only the compiler's boundary is.** Promoting is therefore not a claim that the shape
works. It is a claim that the compiler no longer fails as a Rust compiler when it meets it.

**What this decision does NOT do.** It is a promotion of a branch, not an upgrade of a claim.

```text
it does not strengthen the conformance claim   PASS-WITH-DEVIATIONS, per-rule for 56 of 168
                                               granular rules, stands exactly as CD-397 wrote it
it does not authenticate distribution          archives remain unsigned. Integrity, not
                                               authenticity
it does not close DEV-140..145                 they still define the supported native subset. The
                                               repair report's per-entry table records each as
                                               "assessed, deferred — no consumer"; DEV-141 is a
                                               `std-full` PROFILE boundary and not a defect at all
it does not close DEV-160 or DEV-221           population A is 8 after this merge, not 0
it does not discharge T3 / T7 or the seven     they remain NAMED RESIDUALS, unchanged
   security surfaces
it does not close C9 Part B                    that still needs a second artifact
it does not reopen the tensor track            Gate 7 productisation DEFER stands
```

**Why now rather than after more repair.** Of the eight deviations that remain, the repair report
finds only DEV-160's capability half is reached by any code anyone has written; the other seven are
supported-subset boundary and one ergonomic residual. Holding a strictly better compiler off `main`
to shrink a population that no consumer is hitting would trade real qualified improvement for a
smaller number. The next compiler campaign starts from the promoted tree.

**Next:** the standalone toolchain / C9 Part B second artifact. No compiler gate is active.

---

## CD-397 — GATE C10 CLOSED: PASS-WITH-DEVIATIONS, on a deviation list that was verified rather than inherited (2026-08-09)

**Owner decision under CE8: PASS-WITH-DEVIATIONS**, on the claim wording in
`C10-Q-EVIDENCE-PACKAGE.md` §3.2 as drafted. Recorded in §3.2a. Gate C10 is CLOSED.

```text
gate decision      PASS-WITH-DEVIATIONS
release wording    §3.2 as drafted
T3 / T7            accepted as NAMED RESIDUALS; no robustness claim over either
7 surfaces         accepted as NAMED RESIDUALS; named, not claimed
DEV-177 question   WITHDRAWN — the reproduction pass answered it
promotion          NOT authorised. `develop -> main` follows the decision separately
```

**What the claim rests on that it did not at first draft.** Every one of the 16 deviations it names
was observed failing at the candidate. Seven of the 23 present at the anchor did not reproduce and
were closed — DEV-083, DEV-122, DEV-161, DEV-162, DEV-177, DEV-178, DEV-181 — and each would
otherwise have been published as a known limitation of a compiler that does not have it. DEV-177 in
particular was the only subtraction that made a conformance claim FALSE rather than narrow, so its
closure moved the objection to PASS from "a claim would be false" to "84 of 168 rules are
unattributed", which is weaker and more honest.

**The residuals are inside the claim text, not beneath it.** Nine robustness targets were declared
before measurement and two are reported unrun; the population was not trimmed to seven after the
fact, which is the denominator manipulation the plan forbids. The seven security surfaces are named
individually rather than counted, because a residual a reader must go looking for is not a
disclosure.

**Two findings recorded and deliberately not acted on.** DEV-160's named-refusal boundary does not
cover the shape `stark-http-client` works around — it reaches rustc and surfaces `E0502` inside
`mod stark_proj`, the outcome CD-374 says the named refusal prevents. And DEV-122 closes with a
surviving clamp in `line_col`; what made that clamp dangerous is now prevented structurally by
`SourceId` on every span, so the entry closes, but the hardening it also asked for was never done.

**Method note, because it is the transferable part.** The ledger could not be trusted to a git-log
audit: three of the seven non-reproducing entries were repaired *incidentally* by unrelated
consolidations (AS3 method resolution, AS1b span identity), so no commit names them. Only re-running
a reproducer finds those. DEV-157 was one probe from a false closure in the other direction — the
shape its entry named now builds, while the defect is alive in other `Never` positions.

**Next:** the post-C10 deviation repair programme, running P0 (reproduction) then P1 (DEV-180),
per the owner's ruling that DEV-180 follows C10-Q and does not precede it.

**Post-C10 repair programme, P0 + P8 (2026-08-10, baseline `689d26d`).** Reproduction pass
extended over the whole remaining population — `audits/POST-C10-DEVIATION-REPRODUCTION.md`.
**No deviation failed to reproduce**; the ledger is accurate at this baseline. **DEV-120 CLOSED —
RECLASSIFIED AS DOCUMENTED LIMIT** (population A 13 -> 12): the interpreters classify call-depth
exhaustion cleanly (exit 2, 512 frames) and a native binary dies by SIGABRT (exit 134), which
`LIMIT-RESOURCE-001` permits twice over — capacities are implementation-defined and the reporting
duty is qualified by "when the host permits". Owner ruling D4 (WP-C7.9) had already decided the
repair question; the entry was carrying a settled decision as an open defect. `MAX_CALL_DEPTH`
unchanged. **DEV-167 RAISED AS CE1** rather than resolved: `06-Standard-Library.md` declares
`ToString` but never promises every `Display` type has the method form, so there was no
conformance gap — only a language question the packet must not answer by implementation
convenience. **DECIDED (owner, CE1, 2026-08-10): keep the free function; CLOSED as a
documented non-promise** (population A 12 -> 11). Neither alternative was taken — blanket
impls are a language feature, and a name-keyed resolver branch would reintroduce the two-tier
trait model DEV-166 removed. The decision is pinned by two tests in
`tests/dev_display_dispatch.rs`, so reversing it fails CI rather than passing unnoticed.

**Post-C10 repair programme, P3 + P4 + P6 (2026-08-10, baseline `689d26d`). Population A 11 -> 9.**

- **DEV-220 NEW, REGISTERED AND REPAIRED.** Found while building §9.1's `Never` position matrix,
  and registered separately per §19.4 because the root cause is inference, not representation. A
  diverging arm CAPTURED the join's inference variable: `unify` tried its `Infer` arm before its
  `Never` arm, so `unify(?T, Never)` bound `?T := Never` and the expression claimed a type no value
  of it ever had. **DEV-121's representation guard — closed, and working exactly as designed —
  caught it as an internal compiler error on `let x: Int32 = if c { 1 } else { panic("p") };` and
  on the far more ordinary `else { return; }`.** DEV-218 (CLOSED 2026-08-09) created the
  precondition correctly by making diverging blocks produce `!`; nothing then stopped `!` binding a
  variable. Its three programs put the inhabited arm first, which is why reversing a match's arm
  order reproduces. Repair: the `Never` arm moves ABOVE the `Infer` arms and records the open
  variable; `default_never_coerced_vars` settles it AFTER integer-literal defaulting, so
  `let x = if c { 1 } else { panic(..) };` still yields `Int32`. Five three-engine cases, two of
  them negative controls; all five fail with the repair reverted in place.
- **DEV-157 CLOSED, REPAIRED.** Four repairs across three phases, not one — DEV-220 in typecheck,
  a diverging-else tolerance in MIR lowering, `MirTy::Never` as `core::convert::Infallible` (an
  EMPTY enum, with the local declared uninitialised so no storage is invented for an uninhabited
  value), and a STRUCTURAL never-coercion allowance in the verifier. That last was found by the
  three-engine harness and by nothing else: `stark build` alone accepted the program the verifier
  rejected. **One accepted-set change, toward the specification:** `1 + panic("p")` was refused
  `E0500` as an artefact of DEV-220 and is now accepted, per `03-Type-System.md` line 67's
  unqualified "an expression of type `!` coerces to any other type".
- **DEV-168 CLOSED, REPAIRED, with no second trait-dispatch authority.**
  `check_qualified_core_trait_call` already publishes the selection through the same publisher
  `a == b` uses; `operator_callable_key` already consumes that provenance. The new
  `Res::CoreTraitMember` lowering arm reads that answer. `qualified_calls_disambiguate_the_two_traits`
  — the test this deviation named as its evidence, whose comment recorded the gap — is upgraded
  from front-end-and-oracle to full three-engine agreement. **Residual registered separately:**
  `Display::fmt(x)` on a BOUNDED generic parameter is refused `E0500` at the front end, because
  selection scans impls and never consults bounds. Pre-existing and independent.
- **DEV-140..145 ASSESSED INDIVIDUALLY, REPAIR DEFERRED (owner decision).** §12.1 step 3's question
  answered for each: **not one of the six shapes is used by any first-party package.** They name
  FOUR different missing authorities, so §12.2's grouping rule is satisfied by no pair. Each needs
  a multi-layer feature addition, not a bounded repair; DEV-141 is a `std-full` PROFILE boundary
  rather than a defect at all. §24's `application hits it -> reproduce -> repair boundedly` governs.
  They continue to define the supported native subset, kept honest in both directions by the
  enforcing layer audit.

**Post-C10 repair programme, P1 + P5 + P2 (2026-08-10). Population A 9 -> 7.**

- **DEV-180 RESOLVED**, repair commit `1db9760`, after C10-Q as the owner ruled.
- **DEV-159 RESOLVED, and it was reproduced rather than argued.** Six concurrent `stark build`
  invocations of ONE program from a cold artifact directory: **73 failures in 240 builds**, in two
  signatures — a generated crate Cargo could not build, and an artifact that had vanished by the
  time it was installed. The content-addressed build directory §11.2 lists as a remedy was already
  there and is *why* they collide; what had no sequencing was everything this compiler does around
  Cargo (the stale check's `remove_dir_all`, non-atomic writes, and the caller's read of the
  binary). `BuildLock` excludes on `create_dir` — an atomic test-and-set, no `unsafe`, no new
  dependency — scoped to one build key, held until the artifact is dropped so the caller's install
  is covered. **0/240 debug, 0/200 at eight-way release.** The §11.3 control is a unit test that
  fails deterministically when the exclusion is neutered, because a stress run's sensitivity is
  only a probability.
- **DEV-160 STAYS OPEN, and the record is corrected.** The C10-Q-era reading — carried into the
  post-C10 reproduction pass — was that the boundary is refused by name and never delegated to
  rustc. **It is not, for at least one shape.** `send(r.url.as_str(), r.body)` reaches the user as
  `error[E0502]` inside the generated crate: `plan_for_call` returns `None` before the DEV-160b
  refusal written for exactly that shape, because a borrow arriving from an earlier block is not
  among the call block's own borrows. A repair was implemented (make `conflicts` consult
  `borrow_provenance`) — it works on the reproducer, keeps every suite green, and **over-refuses
  `stark_http_client::follow`, breaking the `stark-get` build**. Reverted; the measurement is kept
  because the finding is that the cheap closure is not admissible. §23's exit criterion 3 for
  DEV-160 is NOT met, and the entry now says so instead of asserting it is.
- **DEV-160 UPDATE, same day: the rustc leak IS sealed.** The first attempt's over-refusal had one
  cause, not a fatal one — `borrow_provenance` propagated across `Rvalue::Use(Operand::Move(p))`.
  A move TRANSFERS OWNERSHIP: `follow` does `let mut url = builder.url;`, after which borrowing
  `url` does not borrow `builder`. Severing provenance on moves fixes the false positive, and
  `conflicts` can then consult provenance so a borrow arriving from an earlier block is visible
  before the early return that was making the DEV-160b refusal unreachable for its own case.
  Measured with the pre- and post-repair compilers over eight borrow/move shapes: **the two E0502
  leaks became named refusals and nothing else moved** — nothing that built stops building, nothing
  newly builds, so no subset claim widened. §23 criterion 3 is met for the demonstrated shape.
  **DEV-160 stays OPEN for the capability half**: the programs are valid STARK and still do not
  build, which is DEV-160b's cross-block absorption under the 2026-08-03 owner ruling. Residual
  recorded: provenance answers "may derive from", not "a live borrow reaches here", and the precise
  def-use walk that cross-block absorption needs anyway should replace this heuristic when it lands.

**Bookkeeping reconciliation (2026-08-10). The "named here, owning no heading in the ledger"
bucket is now EMPTY — it held sixteen.**

Those sixteen were never open defects hiding from the tool: they were resolved in this file's prose
and never given a heading `KNOWN-DEVIATIONS.md` could classify. The consequence was narrow and real
— **"population A is N" was only ever true of the ledger-derived set**, and a reader of this file
got a different number. Each now carries a closing heading with the evidence that settled it.

**Probed, not taken on the prose's word**, because this project's own record is that 7 of 23
deviations did not reproduce at the C10-Q anchor. DEV-091/096 (out-of-range 64-bit float→int cast:
traps, and as a CAST failure rather than an arithmetic one), DEV-097 (bounds check), DEV-099
(`size_of::<[Int32; 4]>()`) were probed in both engines; DEV-092, DEV-095 and DEV-101 were settled
against their live test suites (mangle 9, build::tests 24, cross_package_generics 11).

**DEV-099 is the one that justified probing rather than reading.** It was recorded as a live
PRE-EXISTING defect — a layout query on an array type failing to lower — and it does not reproduce.
Fixed at some point after 2026-07-23 and never recorded.

Two are not simple closures:

- **DEV-098 → ACCEPTED-INDEFINITELY.** Never a defect: a deliberate, verifier-accepted MIR shape
  the `Copy` classification does not describe.
- **DEV-165 → ADJUDICATE, deliberately.** It is OPEN, but in **Population B** (release/distribution
  wording), is an HTTP-client defect rather than a compiler one — the audit that found it said so —
  and is already deferred to the networking roadmap. Its heading carries no bare `OPEN` precisely
  so it does not inflate Population A. Which population an ID belongs to is a human decision, which
  is what ADJUDICATE is for.

**DEV-221 REGISTERED.** The DEV-168 residual now has a number instead of being a paragraph inside
another entry: `Display::fmt(x)` on a BOUNDED generic parameter is refused `[E0500] type 'T' does
not implement 'Display'`, because selection scans impls and never consults bounds. A front-end
over-rejection, distinct from DEV-168 (which type-checked and failed at lowering). Ergonomic — the
ordinary method form `x.fmt()` works.

**Population A is 8**: DEV-140..145, DEV-160's capability half, and DEV-221.

**PROGRAMME CLOSED (2026-08-10). Final report:
`STARKLANG/docs/compiler/audits/POST-C10-DEVIATION-REPAIR-REPORT.md`.**

```text
baseline  689d26d      final  5967a42      population A  13 -> 8
CI        run 31353396547 SUCCESS, 24/24 jobs, all three Tier-1 platforms
          run 31353396543 SUCCESS (C7.8 Native Capabilities)
```

**DEV-165 REPAIRED** (population B, so it does not move the count). It went a layer deeper than its
entry said: `stark_net::connect` refused every non-zero timeout, so the HTTP client's
`connect_no_timeout` was correct and switching only the client would have failed every connection.
A new provider operation, its native implementation, `stark_net::connect` and the client were
changed together. A control written when the defect was recorded — requiring the connect to FAIL —
fired against a live peer; its polarity is corrected and a zero-duration refusal control added.
Measured: a 2-second deadline against RFC 5737 TEST-NET-3 returned in 2.479s.

**CI failed twice before it passed, and both failures were this programme's own work.** The record
keeps them rather than showing only the green run. (1) `windows-x64`: the DEV-159 build lock treated
Windows' pending-delete `PermissionDenied` as fatal — a real portability defect no macOS run could
reach. (2) all three platforms: `stark fmt --check`, STARK's own source formatter, which is separate
from `cargo fmt` and had been clean throughout.

**§23 criterion 3 is PARTIAL, and is recorded as partial.** DEV-160's `E0502` leak is sealed, so the
boundary is enforced by STARK rather than delegated to rustc; the capability half — those programs
are valid STARK and still do not build — remains DEV-160b's own deferred work package.

**DEV-156/172/186 REPAIRED after the decision (population A 16 -> 13; now 9 — see the post-C10 programme entries above).** The formatter no longer evicts field doc comments; every signed minimum is writable (folded in typecheck, the HIR interpreter AND MIR lowering — the MIR half was visible only with `--no-mir-opt`, because the optimiser const-folded the shape and a hand-run `stark build` therefore passed); the LSP transport bounds its allocation before reading, and still reports a truncated frame as `UnexpectedEof` rather than buying the bound with the failure signal.

## CD-396 — installed toolchains carry an explicit library set and resolve it locally (2026-08-09)

> **RENUMBERED from CD-395 at integration, 2026-08-09.** Both this branch and Gate C10 allocated
> `CD-395` on the same day, to different decisions, while working in parallel. C10's landed on
> `develop` first (PR #13, merge `1d20123`), so **C10 keeps 395 and this record moves to 396** —
> the rule is arrival order on the trunk, not authorship order. Nothing about the decision below
> has changed; only its number.

**Owner-approved implementation of WP-PKG-TOOLCHAIN-ROOT.** Three distribution choices are now
load-bearing rather than inferred from directory names:

1. A package ships only when its manifest says `"distribution": { "toolchain": true }`. The
   marker names toolchain bundling rather than registry publication. The current marked set is the
   27 packages whose entry is `src/lib.stark`; the `stark-get` application and all 25 consumer
   fixtures do not ship.
2. The bundled version must satisfy the requested constraint. An incompatible request is refused
   with both requested and carried versions.
3. Fresh resolution precedence is explicit `path`, then the workspace registry, then the
   executable-relative toolchain root. A compatible workspace-registry package shadows a bundled
   package with one warning. A lockfile does not re-run precedence: its `registry` or `toolchain`
   source remains authoritative.

Toolchain lock entries record `source: "toolchain"`, version and content hash, never the absolute
installation prefix. Toolchain packages remain local in `--offline` mode and contribute the same
transitive capability envelope as path and registry packages. The installed tree mirrors
`packages/<name>/`, preserving existing sibling path dependencies.

`stark doctor` now has independent named `provider_crates` and `packages` checks. The provider
check derives its required set from the compiler's built-in provider registry, so deleting a crate
and also deleting its manifest hash cannot produce a false OK.
## CD-395 — Gate C10 OPENED; six opening decisions ruled; the "at a glance" block was short by thirty (2026-08-09)

**`STARKLANG/docs/compiler/plans/WP-C10-COMPILER-RELEASE-QUALIFICATION.md` is the execution plan,
APPROVED WITH AMENDMENTS. `work-packages/C10-0-OPENING-INVENTORY.md` is the freeze.** Campaign B
exited; C10 is the release-qualification campaign, and it is a QUALIFICATION campaign — its purpose
is to determine what the existing compiler can legitimately claim, not to improve it.

### The sequencing question, answered from the documents rather than by preference

The position line read as *"C9 Part B blocks C10"*. **The roadmap does not say that anywhere.**
§4.5 admits "blocked on second-artifact evidence" as one of three PERMITTED explicit statuses;
WP-C10.7 requires C0–C8 plus the mandatory native path; the Core v1 Compiler Stable class requires
"C7, C8, and C10". C9 is excluded in three independent places, and Charter §2.4 scopes the
`Blocked:` field to the CURRENT gate. C9 Part A already supplies the extension-isolation and
tensor-stage inputs C10.1 consumes.

```text
OD-1  APPROVED               C9 Part B does not block C10. Part A CLOSED, Part B DEFERRED
                             pending second-artifact evidence; ONNX alone authorises nothing (CE7)
OD-2  APPROVED               evaluate Core v1 Compiler Stable + Native Systems Preview ONLY.
                             STARK v1 General-Purpose Stable is a wider claim on much the same
                             evidence and remains a separate owner act (CD-022)
OD-3  APPROVED W/REFINEMENT  THREE separately countable populations, not one denominator:
                             A compiler deviations (the CD-021 denominator)
                             B release/distribution (constrains WORDING)
                             C assurance residuals (constrains CLAIM STRENGTH, asserts no defect)
OD-4  MODIFIED               DEV-012 and DEV-213 are CLOSED during C10, not carried. Neither
                             blocks opening; both gate the claim. New packet C10-P, new gate
                             C10-G before C10-Q. Neither reopens C8
OD-5  APPROVED               see the superseding note below
OD-6  APPROVED               ROADMAP.md §0.1 corrected; §6.0's gate text preserved and marked
                             satisfied
```

### Three amendments the owner made to the plan's METHOD

1. **The plan's five inputs are not the whole authority.** The normative Core and extension
   specifications retain theirs under Charter §1.6/§1.9. A work-package plan sits at level 5 of the
   source-of-truth hierarchy and cannot demote the specification beneath itself.
2. **No expected finding count, and no expected falsification rate.** The plan had said A1 "should
   expect to correct some non-zero number" and that C10 "should expect a comparable rate" to AS8's
   13/39. Both create investigator bias. **0 corrections and 20 are equally legitimate**; what must
   be demonstrated is that the census enumerated the intended population — so the forcing mechanism
   is an injected mis-citation the census must report, not a yield. **AS8's lesson was not "a third
   of audits are wrong"; it was "do not infer evidence strength from reading the machinery."**
3. **Inherited mutation evidence gets a freshness rule** (plan §8.2a). A prior trial may be cited
   only while the targeted authority AND the claimed killing evidence are unchanged; otherwise the
   result is HISTORICAL and the trial re-runs. Without it, C10 could cite `AS8-MUT-025` as evidence
   that `mir/verify.rs` controls provider signatures long after that verifier was rewritten.

### OD-5 — the superseding note. CD-394 IS NOT REWRITTEN.

**CD-394's evidence line ("coverage baseline published as `--lib` only and labelled as such") and
`AS8-EXIT-QUALIFICATION.md` §5's `AS8-R15` row were correct when written and were overtaken by
`0bc9aee`.** Both are preserved exactly as written. The live figures:

```text
full corpus      regions 83.05%   functions 84.92%   lines 83.64%
AS8-R15          DISCHARGED
branch coverage  unavailable from this toolchain — NOT CLAIMED, NOT FABRICATED
live document    STARKLANG/docs/compiler/AS8-COVERAGE-BASELINE.md
```

### What C10-0 measured, and the three things nobody had checked

**F3, the most consequential.** Six open deviations exist **only** in this file, owning no heading
in `KNOWN-DEVIATIONS.md` at all:

```text
DEV-156  `stark fmt` evicts member doc comments
DEV-157  the native backend has no representation for `MirTy::Never`
DEV-159  a native build can race its own dependency build
DEV-160  place-granular borrows, whole-value projections
DEV-161  an ambient `CARGO_TARGET_DIR` breaks every native build
DEV-162  READ through a whole-value accessor
```

A C10.7 check reading only the ledger would not have seen them. OD-3's second clause — "plus any
`DEV-NNN` present in `COMPILER-STATE.md` but absent there" — was written for exactly this and
caught six.

**F2.** The `Known open, at a glance` block listed **two** deviations. Population A has **thirty-two**
(18 live-OPEN in the ledger + 6 state-only + 8 needing adjudication). The block was not lying — it
says of itself that it is a summary — but a qualification session that trusted it would have carried
2 instead of 32. **Corrected below, forward-only.**

**F1.** `target-matrix.json` names **four** targets, not three. `x86_64-apple-darwin` is **tier-3,
packaged with an archive and both installers, and exercised by no CI job whatsoever.** The C10 plan
listed three and missed it. **C10-Q may not include tier-3 in any conformance claim.**

**F6, the positive one, and it is measured rather than assumed.** Every inherited AS8 mutation
result is FRESH: all 12 mutated authority files and all 13 control suites hash identically at
`e7bb95d` and at `f12ecec`. All 39 trials are citable without re-running. Re-checked at C10-Q.

Also found: four individually-correct and non-interchangeable counts of "how many deviations"
(186 headings / 170 ids owning one / 178 ids mentioned / 190 reconciler entries), and
`tests/c6-corpus/README.md` citing `c6_corpus_cases.rs` twice — including a runnable
`cargo test --test c6_corpus_cases` — **for a target that does not exist** (the enforcer is
`c6_generated_corpus.rs`).

### Baseline

```text
qualification baseline   f12ececca6d4bdabf828d657c4a4f719a7f9c39a
CI                       run 31292404920 (CI) + 31292404936 (C7.8), both conclusion success,
                         zero non-succeeding jobs — queried per job, not read off a badge
execution branch         wp-c10/execution-plan
```

**Next:** C10-P (DEV-213 repair, DEV-012 interactive validation) alongside C10-A1 (the 161-rule
evidence census). **C10-Q remains an owner decision under CE8** — Charter §2.2 forbids a session
claiming Core v1 conformance on its own authority. C10 proposes; the owner authorises.

### C10-P, same day — DEV-213 CLOSED; DEV-012 cannot be closed by a session

`audits/C10-P-LANGUAGE-SERVICES.md`. The LSP now invalidates per **package**, not per URI:
`CompilationResult` records the `package_root` its whole-package analysis was built against, and
`invalidate_package_of` sweeps every sibling sharing it — from `open`, `update` **and** `close`,
because all three change the overlay set the analysis is computed from.

**The pass is believed because the failure was demonstrated first.** With the sibling sweep
disabled the flipped test fails with the defect's exact signature — `["alpha_symbol",
"renamed_symbol"]`, both names present — and the control was then removed and the restore verified
byte-identical. AS8's test is renamed and polarity-flipped rather than deleted, exactly as its own
assertion message instructed. `c10-deviation-populations.py`, which knows nothing about the repair,
independently reports population A's live-OPEN set dropping 18 → 17.

Evidence: 48 LSP tests, 569 unit tests, clippy `--workspace --all-features --all-targets -D warnings`
exit 0, `fmt --check` clean.

**C10-0's freshness prediction held.** It stated in advance that an LSP-confined repair would
disturb none of the 12 mutation-authority files or 13 control suites; the files changed are
`src/lsp/state.rs` and `src/lsp/server.rs`. Every AS8 trial remains citable. Re-verified at C10-Q
regardless.

**The residual this packet creates, stated rather than left to be found:** invalidation is now more
eager, so a package with many open URIs recompiles more often. AS8 measured the old duplication's
cost as immaterial and the defect was never about cost — but **no before/after was taken here and
none is claimed.** If an LSP workload enters C10-E's frozen set, this is the change to measure.

### C10-A1, same day — the denominator itself was wrong, and the seven missing rules are the numeric ones

`audits/C10-A1-EVIDENCE-CENSUS.md`. Tool: `scripts/c10-evidence-census.py`.

```text
DENOMINATOR   168 granular rules      (every prior document, this one included, said 161)

PRECISE      36    21.4%    positive AND negative evidence at test-FUNCTION precision
AGGREGATE    85    50.6%    cited only via a file or the aggregate runner  (DEV-017)
ABSENT       42    25.0%    the INVENTORY cites nothing — see below
N/A           5     3.0%
```

**A1-F1.** Rule IDs were matched as two dash-separated segments plus a number, and **seven have
three** — `NUM-INT-ARITH-001`, `NUM-INT-DIV-001`, `NUM-FLOAT-OP/FORMAT/TRAIT/REPRO-001`,
`NUM-INT-TYPE-001`. **All seven are the numeric-semantics rules**: integer overflow, division by
zero, float behaviour. For a language whose headline guarantee is that overflow and division by
zero always trap, the counting method used to plan this campaign could not see the rules that say
so. Caught by cross-checking c2.11's ids against the inventory, not by reading the regex; confirmed
by a second independently written enumerator (168 distinct, 0 duplicates, exactly 7 three-segment).
**The population is unchanged; the enumeration was faulty** — recorded as a dated correction line
per plan §7.2, not as an edit.

**A1-F3, and it is binding on C10-A2.** `ABSENT` means *the inventory cites nothing*, **not** *nothing
tests it*. `EXT-ISOLATION-001` records `none; none` while `starkc/tests/c91_extension_isolation.rs`
runs nine tests in CI on every push; `OWN-PARTIAL-001` records `none` while the `as4_*` and `c61f_*`
suites exercise it. **This is EI2's error in mirror image** — EI2 read the machinery and missed a
control sitting in the tree; the inventory froze the same mistake into a data file in 2026-07-18 and
never revisited it. A2 may not transcribe these buckets; every ABSENT and AGGREGATE row is resolved
against the TREE or the dashboard says UNRESOLVED.

**A1-F5.** 41 of the 42 ABSENT rules have no legacy predecessor either. The gap is exactly the
granular rules C2.6 created when it split 59 broad IDs into 168, and which C2.11 — by its own header,
covering "the high-cost frozen semantic surface" — never reached. **Coherent and bounded, not decay**,
and it names the precise 41 rows A2 starts from.

**A1-F2.** Citation integrity PASSES: all 36 PRECISE citations resolve to a real `fn`, checked the
way `check-conformance.py` checks them so a renamed TEST is caught, not merely a renamed file. The
clean result is believed because `--self-test` injects a citation to a function that does not exist
and the census reports it. Per the owner's amendment, no finding count was expected or targeted.

### C10-B, same day — the robustness gate FAILS on one target, and the guard that should have caught it works

`audits/C10-B-ROBUSTNESS.md`. Suite: `tests/c10b_robustness.rs`, 12 tests.

**DEV-214 — a left-associative operator chain aborts the compiler with a stack overflow.** The
parser HAS the right guard (`MAX_DEPTH = 200`, *"this code is nested too deeply to parse"*) and it
bounds SYNTACTIC nesting, because that is what recurses in recursive descent. A chain does not
recurse there — `parser.rs` implements the 16-level precedence table as one `loop` per level — so
the counter never moves. **The AST is still n deep**, and the walks after the parser descend it.

```text
(((((...1...)))))  300 deep   ->  REJECTED cleanly     <- the bounded failure the gate asks for
1 + 1 + ... + 1     65 terms  ->  SIGABRT              <- DEV-214
```

**The severity is in the stack size.** 8 MiB (a main thread): 240 OK, 250 aborts. **2 MiB (Rust's
default for a SPAWNED thread, and what `cargo test` gives each test): 60 OK, 65 aborts.** ~30 KB of
stack per AST level. The LSP analyses on a server thread, so an embedding sits on the low number.
Found because `cargo test` overflowed where `cargo run` had not — the difference between them IS the
finding, and a suite run only on a main thread would have reported a threshold four times too
generous.

**Not repaired, by rule rather than reluctance.** Counting chain depth would start rejecting
200–245-term expressions that compile today — a change to the accepted/rejected program set, CE1/CE2
and plan stop condition 5. Converting the walks to a worklist is the broad refactoring §3.2 forbids.
Raising the stack moves the cliff without removing it. **Owner call.**

Targets T1, T2, T4, T5, T6, T8 and both determinism checks PASS over ~2,000 cases. **T3 (package
graphs) and T7 (malformed artifacts) were NOT RUN** and are named as such rather than dropped from
the population — C10-Q may not claim robustness over them. DEV-186 is confirmed and characterised.

The passes are believed because `aaa_harness_self_test_detects_an_injected_panic` runs first and
proves, in both directions, that the driver reports a panic and does not fire on ordinary input.

### C10-F, same day — eight commitments, eight non-commitments, and two MSRVs

`C10-F-COMPATIBILITY-POLICY.md`. Candidate `33a8608`; every claim read from that commit per §14.1b.
Each of the fourteen axes is COMMITTED with evidence, UNCOMMITTED with what would be needed, or
NOT APPLICABLE with the reason — no fourth option, no partial credit, and a promise with no evidence
citation is **deleted rather than softened**.

**The three version axes are COMMITTED as REJECTION promises, not stability promises**, which is the
useful shape for a 0.1.0 compiler: MIR runtime surface (`MIR-0017`, tested by
`rejects_unsupported_runtime_surface` stamping `0.1-A999`), runtime ABI (generated prologue,
`c63_closure_evidence`), and the Native Provider ABI — the strongest, requiring **exact version AND
content hash**, whose own source says *"'close enough' is how a build stops being repeatable"*.

**Core language compatibility is UNCOMMITTED, and that is C10-A2's number arriving where it
matters:** 56 of 168 rules carry function-precision evidence. A Core compatibility promise would be
a promise about all 168.

**Two different MSRV claims, and only one is enforced.** `MINIMUM_RUSTC_VERSION = "1.85.0"` makes
`stark build` refuse a *user's* older rustc — a real check with a diagnostic. `rust-version = "1.85"`
in `Cargo.toml` claims **starkc itself** still builds on 1.85, and **nothing verifies it**: CI uses
`dtolnay/rust-toolchain@stable`, which resolved to 1.93.0 today, so the compiler could have adopted
a 1.90 feature unnoticed. The enforced check is committed; the Cargo field is UNCOMMITTED until one
CI job builds on 1.85.

**C10-0's F1 lands here as policy.** `x86_64-apple-darwin` is tier-3: packaged with an archive and
both installers, exercised by **no CI job**. UNCOMMITTED, and the release notes must say what tier-3
means rather than let a reader infer support from a packaged artifact.

**Diagnostics: determinism is not stability.** C10-B verified byte-identical diagnostics for the same
source; DEV-182 (a value silently wrong while both sides reported success) and C10-R1 (keyword
identity unpinned) are why codes, spans and text must be promised separately or not at all.

### C10-E, same day — the phase split, and DEV-213's residual measured rather than argued

`audits/C10-E-PERFORMANCE-BASELINE.md`; data `benchmarks/c10/darwin-arm64.json`. Candidate
`01ba608`, Darwin-arm64, rustc 1.93.0. **Baselines only — no threshold proposed, nothing optimised.**

**Workload integrity is verified before any timing is taken**: 7 workloads, 16 files, 0 drift
against `FROZEN.json`. The harness exits rather than measure a drifted workload, because a baseline
against a changed workload looks comparable and is not.

**E2 — the phase split `c7-baseline.py` cannot produce.** Type checking dominates at **44–59%**;
lexing is a flat ~6.5% everywhere. `w06_multi_package` is the only workload where resolution rises
(26.8%) and checking falls — the one workload with a package graph.

**E2.1 — scaling is mildly superlinear, and recorded rather than acted on.** 64x the input costs
**117x** the time (~`O(n^1.17)`), with the growth concentrated in checking: its share climbs 52% ->
75% from 100 to 6,400 functions. 234 ms for 6,400 functions blocks nothing; investigating it is a
separate approved packet (§12.3).

**E4 — the finding. An edit now costs about a cold open.**

```text
modules   cold open   edit -> diagnostic   workspace symbol
      4      548 us              510 us               1 us
      8      937 us            1,051 us               3 us
     16    1,808 us            1,869 us               4 us
     32    3,162 us            3,185 us               8 us
```

`edit -> diagnostic` tracks `cold open` at every size. **That is C10-P's residual, quantified**:
package-scoped invalidation drops every cached analysis of the package, so the first query after an
edit pays for a full re-analysis. C10-P declared this before measuring it — *"invalidation is now
more eager… no before/after was taken here and none is claimed"* — and at 3.2 ms for 32 modules it
is inside interactive latency, so **nothing here argues for reverting or optimising the repair.**

**AS8's 22 ms / 181 ms are NOT used as a before/after.** They describe a different architecture (one
analysis per URI, invalidated per URI) measured with a different harness on a differently-shaped
package. The numbers here are smaller; that is not evidence of a speedup and is not claimed as one.

**E6 — one platform.** Darwin-arm64 measured; Linux-x64 and Windows-x64 **NOT MEASURED**. The
harnesses are portable (platform in the filename, RSS unit in the payload, because macOS reports
bytes and Linux kilobytes) but no CI job runs them. **C10-Q may not generalise these to the platform
matrix**; the concrete next step is a CI job uploading `benchmarks/c10/<platform>.json`.

**OD-8 appendix — EMPTY, and the reason is availability rather than scope.** No `.onnx` file exists
in the tree: `gate4/manifest.toml` records that ONNX fixtures "are generated deterministically by
test code", and `gate5/fetch-input.sh` downloads its reference image from the network, with deploy
additionally needing ONNX Runtime. Import/verify could be timed by driving the generator; **deploy
cannot be timed offline at all.** Measuring a third of the appendix and labelling it
"import/verify/deploy" would be worse than measuring none. **OWNER DECISION:** authorise driving the
generator for import/verify and record deploy as unmeasurable offline, or accept the empty appendix
with this explanation as the reason. Either satisfies OD-8; guessing does not.

### C10-R, same day — the freshness rule fired, and the evidence survived the merge

`C10-MUTATION-LEDGER.md` §6. Candidate `29ce610` (`develop` `eb60dec` merged in).

**PR #15 moved five files on the mutation-evidence list**, and §8.2a marked **31 of 41 trials
STALE** — 6 because the mutated code moved (`mir/lower.rs`, `resolve.rs`), 28 because the killing
suite moved (`three_engine_differential`, `a11_host_resource`, `c788_resource_lifecycle`).

**Yesterday the rule returned "all FRESH" twice and looked like ceremony. Today it stopped C10-Q
citing 31 measurements of a compiler that no longer exists.**

**Owner decision: re-run the subset backing published claims, not all 31.** Plan §8.2 scopes
mutation to claims C10 intends to publish, and C10-F marks Core language compatibility UNCOMMITTED —
so trials backing unpublished Core rules need not be refreshed. Nine re-run: all six clause-1
trials, plus MUT-017/037 (host-resource typing, backing the COMMITTED provider claims) and MUT-036
(MIR verifier, backing the COMMITTED rejection claim). **Twenty-two are recorded HISTORICAL**, and
C10-Q owes the disclosure that its mutation evidence is 9 current + 22 historical + 10 fresh-by-hash.

**Result: 9/9 reproduce AS8's recorded outcome, killer counts included.** The integration changed
`resolve.rs` and `mir/lower.rs` without changing any measured control relationship — verified rather
than assumed. `AS8-R13` and `AS8-R14` both still hold at the candidate.

**One reading trap, recorded because it would be easy to misreport.** The harness prints UNEXPECTED
against `expect`, which encodes **EI5's original prediction** — not against what AS8 measured. So
`MUT-034/035/037/039` print UNEXPECTED while matching AS8 exactly. Calling those new surprises would
be misreading the tool.

`as8-mutate.py` gains `--only` (select trial ids ACROSS batches), because stale trials do not line
up with AS8's batches and duplicating their definitions would create the second copy that drifts —
the `AS8-DA-*` failure mode, inside the tool built to detect it.

### C10-D, same day — a control built, and a C10-A2 claim refuted by measurement

`C10-MUTATION-LEDGER.md`. Baseline `51ca1af`.

**Population declared before running**, per plan §8.2: claims C10 intends to publish whose evidence
is not already mutated and which A2 flagged as thin. **All 12 AS8 authority files and 13 control
suites hash identically to `e7bb95d` at this baseline, so all 39 inherited trials are citable and
none was re-run** — the freshness rule paying for itself rather than adding ceremony.

**C10D-CTL-001 — the `RuntimeFn` parity control, built as the owner ruled.** AS8 found that killing
both sides of the interpreter/verifier pairs proved nothing: the kill messages showed copy A dying
to `mir_differential` and copy B to an `unreachable!()` elsewhere, so the redundancy was real and
the cross-check imaginary. Now exhaustive over all 100 variants of the closed `RuntimeFn` surface,
across **four** families. Exhaustiveness is a compile-time guarantee — a no-catch-all witness match,
so a new variant breaks the build rather than shrinking the population. Proved able to fail:
dropping `SliceIsEmpty` from the verifier's table alone yields *"slice: SliceIsEmpty — interp says
true, verify says false"*; restore verified identical. Six classifiers went `fn` -> `pub(super) fn`
— visibility only, no behaviour change, the two implementations stay independent by design.

**A fourth pair the register does not list.** `AS8-DA` catalogues Vec/Box/Slice.
`is_map_runtime` / `is_map_runtime_fn` is a fourth of identical shape, found by enumerating the
classifiers rather than reading the register — exactly what `AS8-DUPLICATE-AUTHORITIES.md` predicts
when it calls itself "a lower bound, not an inventory". Allocated as **`C10-DA-001`**, not
`AS8-DA-007`: AS8 is closed and did not find this, and numbering it into AS8's sequence would
rewrite who discovered what. **The known duplicated-authority population goes from six to seven
without touching the closed AS8 result.** (The ID-deferral ruling below is about `CD-*`/`DEV-*`
numbers that could collide with the parallel branch; this namespace is C10's own.)

**C10D-MUT-001 refutes a C10-A2 claim.** A2 concluded from counting that lexical negative evidence
is "DENSE … these rules are controlled". Deleting `"mut" => Mut` from the keyword table left **all
26 `lexer.rs` tests passing — including `keywords_reserved_and_idents`, a test named for exactly
that rule.** The kill came from `conformance` and `gate2_valid`, by programs ceasing to parse.

```text
C10-R1   keyword identity is controlled only COARSELY, by parse failure.
         CATCHES  a keyword that stops being one — every program using it then fails
         MISSES   a keyword mis-mapped to a DIFFERENT keyword, where the program still parses.
                  Nothing pins the token a word maps TO
```

**The count was real; the inference was wrong.** Those 32 assertions cover literal forms, escapes
and malformed input, not keyword identity — measuring the wrong property and reasoning from it, in
the same session that recorded EI2 doing the same. A2's section is corrected in place. **No DEV
allocated:** a kill by the wrong mechanism means the evidence cannot detect the defect, not that the
defect is present.

**C10-R2** — no metamorphic relation was added. The plan's candidates (formatter idempotence,
harmless parenthesisation, equivalent import forms) each need a normative rule stating the
equivalence before §10.2 permits them, and none has one written. The blocker is normative, not
technical.

### Integration hazard, recorded 2026-08-09 — CD/DEV namespace collision with unmerged work

**Do not renumber anything yet. Owner ruling: IDs are assigned at consolidation time, against a
frozen `develop`, and no reservation is made now.** C10-D/E/F may allocate more before the parallel
branch integrates, so reserving today would be a guess that ages badly.

Measured against `origin/develop` (`1d20123`) and `fix/release-package-ships-providers`:

```text
CD-395    develop 4 files  |  parallel 1 file   COLLISION — C10 owns it on develop
DEV-214   develop 5 files  |  parallel 1 file   COLLISION — C10 owns it on develop
DEV-215..218                  parallel only     no collision today; inside a range C10-D/E/F may want
divergence                    16 behind, 2 ahead
```

**C10 authoritatively owns `CD-395` and `DEV-214` on `develop`.** The parallel branch's records
carrying those numbers are the ones that get renumbered at integration — after freezing `develop`,
enumerating every allocated CD and DEV, and assigning the next free ones.

### C10-Q is NOT next, and the reason is this branch (owner ruling, 2026-08-09)

`fix/release-package-ships-providers` is no longer a distribution patch. Against current `develop`
it modifies package resolution, capability semantics, provider resolution, native support
classification, flow/interpreter/MIR behaviour, manifests, lockfiles and distribution —
`package.rs`, `flow.rs`, `interp.rs`, `mir/lower.rs`, `native_build.rs`, `provider_resolve.rs`,
`typecheck/body.rs`, many tests, and nearly every first-party manifest.

**A C10-Q decision followed by merging that branch would qualify a compiler that is not the
compiler being promoted to `main`.** The order is therefore:

```text
C10-D -> C10-E -> C10-F -> DEV-012 editor evidence
      -> integrate the parallel work
      -> §8.2a freshness + requalify every AFFECTED C10 check
      -> C10-Q against that ONE consolidated SHA
      -> develop -> main
```

D/E/F evidence that stays fresh is retained; anything whose authority or falsifier moves is re-run.

### C10-A2, same day — the dashboard, and a tool that needed three corrections

`C10-CONFORMANCE-DASHBOARD.md` + `conformance/c10-dashboard.json` (168 rows, generated).
Resolver: `scripts/c10-a2-resolve.py`; generator: `scripts/c10-dashboard.py`.

**A1's buckets were NOT transcribed** — that was the packet's binding rule, and A1-F3 is why. Every
row is resolved against the TREE, keyed on the normative rule id rather than the symbol
(`as8-control-census.py`'s method, generalised from 11 authorities to 168 rules).

```text
PRECISE-C211           36     RESOLVED-BY-TREE       20     CORPUS-OR-FILE-LEVEL   27
IMPLEMENTATION-ONLY     1     UNRESOLVED             84
```

**Function-precision evidence rose 36 -> 56 of 168 with no test written.** The twenty were already
in the tree and the inventory never recorded them.

**The resolver needed three corrections, and every one was found by reading NAMES rather than
counts** — the counts were plausible at every stage:

```text
1  implementation citations counted as evidence   `interp.rs::eval_expr` offered as evidence for
                                                  TYPE-PRIM-001. An implementation cannot be its
                                                  own control — AS8-R4 exactly
2  `///` citations attributed BACKWARDS           a doc comment above a `#[test]` named the
                                                  PRECEDING function; five attributions came back
                                                  shifted by one, every name a real function in
                                                  the right file
3  corpus cases counted at function precision     a `.stark` case has `fn main`, and attributing a
                                                  rule to "main" is precision theatre
```

Each correction LOWERED the numbers. **A dashboard citing the wrong test is worse than one citing
none, because it looks checked.** Four rows still attribute to `main` because the rule id sits
inside an embedded STARK string in a Rust test — the file is right, the function name is not, and
that is documented rather than engineered away.

**`EXT-ISOLATION-001` is the exemplar and is now fixed.** The inventory recorded `none; none`; C10-A1
found nine tests running in CI; and the resolver could not find them either, because nothing in
`c91_extension_isolation.rs` named the rule it pinned. One module-header note plus five per-test
attributions moved it to RESOLVED-BY-TREE with its five real test functions named — no test
written, no behaviour changed. That is the shape of the remaining work.

**A finding beyond attribution.** The spec-fixture corpus is a POSITIVE-evidence corpus: 64
`parse-pass`, 27 `notation`, 17 `semantic-error`, 7 `lex-pass`, and **exactly one `parse-fail`**.
Charter §1.6 rule 15 requires positive and negative evidence to travel together. Measuring the two
families separately: lexical negative evidence is dense (`lexer.rs`, 26 test fns / 32 error
assertions), syntactic is thin (`parser.rs`, 47 test fns / 5 asserting a rejection). **No C10-Q
claim of syntactic conformance may rest on the fixture corpus alone.**

### C10-C, same day — no class-B finding, and the probe that was nearly vacuous

`audits/C10-C-SECURITY-REVIEW.md`, model `C10-THREAT-MODEL.md` (16 surfaces, FROZEN in C10-0 §9
before any finding was reviewed), probes `tests/c10c_security.rs`.

**No security vulnerability found, and no `SEC-C10-*` allocated.** The reason is structural rather
than lucky: the compiler has no shell (all four `Command::new` sites build argument vectors), one
dependency (`sha2`, default-features off), generated Rust escaped by Rust's own escaper and then
compiled by rustc — so a literal breaking out is a build failure, not a silent injection — and
interpreters with no host access at all, which is what keeps "no LSP workspace trust" in class D
rather than class B.

**The finding worth recording is about my own test.** S01's first probe asserted only that a hostile
`mod` name produced *some* diagnostic. It passed, and it was nearly vacuous — `mod ordinary_name;`
also produces one, because the file is missing. Reading the actual messages exposed it:

```text
mod ordinary_name;         ->  "file not found for module 'ordinary_name'"   <- reached the FS
mod ../../../etc/passwd;   ->  "expected a module name, found `..`"          <- rejected by GRAMMAR
```

The defence is that hostile forms never reach the filesystem layer, so the assertion is now that
they never produce the file-not-found diagnostic — a claim about WHICH path ran. **A passing test is
not evidence until you know what it would have to see to fail.** That cost one `println!`.

Classified: **S16 integrity-not-authenticity** (class C — governs the release wording; `stark doctor`
detects corruption, not provenance). **S12 PATH-resolved cargo/rustc**, **S11 no LSP workspace
trust**, **S02 build reads its package** — class D, accepted, and S12 must be STATED in the release
notes rather than left implicit. **Seven surfaces carry a defence with NO falsifier** and are
recorded UNVERIFIED, not claimed; R-S03 is the one worth closing first, because there the defence
itself is unknown rather than merely untested.

**No CE9 decision is requested:** the packet reviewed those surfaces and changed none of them.

### OD-7 and OD-8 — owner rulings, 2026-08-09

**OD-7.** All eight unsettled deviations adjudicated (above), and the six that lived only in this
file are **backfilled into `KNOWN-DEVIATIONS.md` as forward-only headings** with status carried
across unchanged. Nothing historical was rewritten. `c10-deviation-populations.py` was taught OD-7's
status vocabulary — `ACCEPTED-INDEFINITELY` and `DORMANT` are neither open nor closed, and
collapsing either into `open` would inflate the count CD-021's release rule ranges over. It now
reports **ADJUDICATE = 0**.

**OD-8 — ONNX timings: INCLUDE, QUARANTINED.** C10-0 recommended exclusion; the owner declined,
because WP-C10.6 explicitly lists ONNX import/verify/deploy and excluding it would need an override
of C10's own contract. It enters C10-E as a **separately scoped optional-extension appendix**, never
aggregated into a compiler-performance number, with a verbatim sentence stating that it qualifies the
frozen maintenance surface only and supports no tensor-capability claim. **C10-E's LSP latency
baseline also absorbs DEV-213's residual** — post-fix numbers, and AS8's pre-fix figures are
historical context that may NOT be called a before/after unless harness and workload are
demonstrably identical.

### DEV-012 CLOSED — owner verification, 2026-08-09. C10-G passes on both arms.

Seven protocol-only features exercised in a real editor: VS Code 1.132.0,
`starklang.stark-language@0.2.0` **built from the C10 candidate `37a0a03`**, release binaries from
the same candidate wired explicitly rather than resolved from `PATH`, macOS 26.5.2 arm64. The build
was verified to carry the C10 work first — a 250-term chain produced `[E0209] … (250 levels; the
limit is 200)`, which only the DEV-214 repair emits.

**Evidence class MANUAL, and the record is a VERDICT rather than a per-feature value transcript.**
Recorded that way deliberately: `GATE-C8-CLOSURE.md` §4 exists because DEV-182 — a parser silently
decoding every escaped non-BMP character to the empty string — **passed** protocol validation, since
both sides reported success and only the value was wrong. The owner is the only party who can
produce this evidence and is the authority on their own session, so it closes; C10-Q should word it
as *interactively validated by the owner in the recorded environment*.

**C10-G: both arms satisfied.** DEV-213 closed by repair, DEV-012 by validation — OD-4's preferred
route, not its fallback. The Core v1 Compiler Stable language-services claim need not be narrowed
for either.

**The DEV-012 session specification is retained** in `C10-P-LANGUAGE-SERVICES.md` §3.2 as the
procedure a future re-validation should follow. It needs a person exercising seven
features in a real editor — MANUAL evidence under Charter §5.2. `C10-P-LANGUAGE-SERVICES.md` §3
specifies the session so it can be done in one sitting, and records the rule that matters: check
**values, not verdicts**, because DEV-182 passed protocol validation while silently decoding every
escaped non-BMP character to the empty string.

## CD-394 — AS8 CLOSED, qualification PASS; the evidence base was overstated in three documents (2026-08-09)

**`STARKLANG/docs/compiler/audits/AS8-EXIT-QUALIFICATION.md` is the record.** All five exit criteria
met. Campaign B's last packet closes; the Sprint 4 Tier-3 closeout is what remains.

### What AS8 measured, and what it changed

39 compiler-source mutation trials across every rule family the packet's scope names — ownership,
trap, drop, resolver, MIR verifier — plus paired one-sided trials on duplicated authorities.

```text
26 CONFIRMED the prediction    13 FALSIFIED it, IN BOTH DIRECTIONS
```

**That ratio is the packet's most useful single number.** EI5's predictions were reasoned from EI2's
*reading* of the evidence base, and were wrong a third of the time — which is the argument for
mutation over audit, made by the audit failing rather than by assertion.

Four register entries were wrong before AS8 measured them, and one was wrong in the compiler's
favour:

```text
ESF-COPY-001   "no control in-tree"       ->  c61f_structural_copy IS one (MUT-009/010/011);
                                              critical -> high
ESF-TRAP-001   one entry, INVISIBLE       ->  split 001a (vocabulary, INVISIBLE, no control
                                              constructible) / 001b (assignment, PARTIALLY_VISIBLE,
                                              caught by the oracle in 4 tests); high -> medium
ESF-TYPE-001   "spec fixtures control it" ->  they do not (MUT-013); medium -> HIGH
ESF-PROV-001   "two engines, not three"   ->  mir/verify.rs is an in-tree control (MUT-025).
                                              EI2 counted ENGINES and the verifier is not one
```

### The one defect, and the fifteen residuals

**DEV-213 — the LSP caches one whole-package `ProjectAnalysis` per open URI and invalidates only the
edited one.** Demonstrated by a passing test at HEAD: rename a symbol in one open file and
`workspace/symbol` returns both the new name and the name that no longer exists.

It was found by a work item framed as a PERFORMANCE question — *"replace the duplication where
measurement shows material cost"*. Measured cost is 22 ms for one analysis and 181 ms for eight open
URIs, which is not material. **Answering the question as written closes the item and finds nothing.**
The duplication's real consequence is N copies with independent invalidation.

**No DEV was allocated for any mutation survivor**, per the owner ruling: a survivor means the
evidence cannot detect a defect, not that the defect is present. AS8-R1..R15 carry those.

### Duplicated authorities — a new dimension, deliberately NOT a register category

Owner ruling: EI0's vocabulary answers *what kind* of authority something is; duplication answers
*how many implementations exist and what relationship they have*. `AS8-DA-001..006` live outside the
frozen enum.

The ruling also corrected the instinct to consolidate, and measurement then refined the ruling: all
three interpreter/verifier pairs were killed on both sides, **but the kill messages show neither
copy is checking the other** — copy A by the differential, copy B by an `unreachable!()` elsewhere.
So consolidating would lose no control that exists today, and create shared fate where there is
none. **Left as an owner call.** `AS8-DA-005` is unambiguous: `scalar_name` can drift silently.

### Governance surfaces repaired

```text
COMPILER-STATE.md      12,979 -> 6,681 lines; `# Current position` is now FIRST. `## Position` had
                       been at line 5,456 describing Gate C5 closing four gates earlier, and a
                       2,808-line section was still headed IN PROGRESS. Verified lossless:
                       57/57 CD sections retained, zero lines lost
KNOWN-DEVIATIONS.md    the same defect, sharper: append-only, so DEV-121's FIRST heading says OPEN
                       and it is CLOSED 3,558 lines later. Index added naming the live heading
lib.rs                 cited PLAN.md as the live plan and "Gates 1-3 ... interpreter", four gates
                       after native compilation closed
```

### Evidence

Clippy `--workspace --all-features --all-targets -D warnings` clean; `fmt --check` clean; 581 tests
across 5 targets, 0 failed; pinned samples **34/34** at `b3b28e757f38d691e7309f168d1209e28ac459af`;
coverage baseline published as `--lib` only and labelled as such.

**Next:** the Sprint 4 Tier-3 closeout, then the Campaign B exit report. **Unresolved and owner-owned:**
the PR #10 / PR #11 merge topology — Sprint 4 cannot close without deciding how AS7 and sprint-4 land
on `develop`.

## CD-393 — AS7 criterion 2 re-qualified; CD-391's PASS is superseded, its closure stands (2026-08-09)

**Owner-directed correction. CD-391 is not rewritten — it is preserved as written and superseded
here.** AS7 was closed on a criterion-2 PASS that the forcing test was not capable of producing.

### What CD-391 claimed, and why it was not true

> *"dependency direction documented and cycle-free — PASS, regenerated at the exact head"*

`as7_module_dependencies` derives module ownership for each method by parsing `impl` blocks, and
its parser recognised an ENUMERATED LIST of visibility prefixes: `fn`, `pub fn`, `pub(crate) fn`.
It did not recognise **`pub(super) fn`** — the visibility essentially every method extracted by AS7
was given. The ownership map therefore covered **36 of 234 methods, about 15%**, and the other 85%
of the graph was unobserved. The test was green because it was looking at almost nothing.

Five real violations were live under that green:

```text
traits -> convert     the Packet 7 cycle the owner ruling was supposed to have broken
infer  -> convert
state  -> body
state  -> infer
state  -> traits
```

**The Packet 7 edge is the serious one.** The owner ruling on Packet 7 rejected a permitted cycle
and added `bounds` as an orchestration layer. That ruling was correctly applied to the EXPLICIT
REFERENCES and never applied to the METHODS: conversion-dependent trait machinery stayed in
`traits` and kept calling `convert_hir_type`, so `convert <-> traits` was never actually gone. The
audit's own text records the ruling as discharged. It was not.

### Why the forcing test could not catch its own blindness

Packet 1 proved the checker fails on an injected violation — by adding `use super::X`. That
exercises the **explicit-reference** detector. The **method-ownership** detector was never injected
against, so the one half of the check that mattered for a packet that moves methods was the half
with no negative control. **A check with two mechanisms needs a negative control per mechanism.**

That proof now exists. Injecting `self.convert_hir_type(id)` into `traits.rs` produces
`typecheck/traits.rs -> typecheck/convert.rs is not permitted`, and green returns on removal.

### The repair

`trait_contracts.rs` (1,191 lines) carries the eight methods that must know what a WRITTEN type
means — `validate_impl_rules`, `check_core_trait_impl`, `associated_fn_type`,
`build_trait_impl_index`, `contract_ty`, `declared_member_signature`, `trait_member_signature`,
`assoc_binding_map` — and sits above `convert`. Trait identity and impl selection stay in `traits`
and convert nothing; `core_method_signature` is deliberately not among the movers.

The three `state` edges were **my own misclassification, and the more instructive failure.** The
publication family was put in `state` because "publication writes storage" — classifying those
functions BY THEIR EFFECT rather than BY WHAT THEY NEED. Each resolves, instantiates or selects
trait candidates before writing. The decision of what to publish belongs with the caller, and
those functions ARE that decision. Five `publish_*` moved to `body`, `ty_to_string` and
`format_nominal` to `infer`, `instantiate_sig` to `body` where its only caller is.

```text
types <- state <- infer <- traits <- convert <- {bounds, trait_contracts} <- patterns/body <- items <- mod

body 4,842 | traits 2,043 | mod 1,846 | types 1,153 | trait_contracts 1,191
infer 1,156 | items 986 | convert 901 | state 569 | patterns 429 | bounds 116
```

Eleven modules, not ten. `state` fell from 896 to 569, which measures how much non-storage work had
been put there. *(Figures are `wc -l`. CD-391 additionally quoted "mod.rs 596 production"; that
number was produced by a method the record does not state and I could not reproduce — excluding
mod.rs's four inline `#[cfg(test)]` modules by brace matching gives 474 at both heads. The totals
are stated here instead, because they are unambiguous.)*

### Criterion 2 — re-qualified

```text
2 dependency direction documented and cycle-free   PASS   with the REPAIRED detector: 234 of 234
                                                          methods owned, eleven modules, 4/4 green
```

Criteria 1, 3, 4 and 5 are unaffected and re-checked rather than assumed:

```text
1 no semantic behaviour change      PASS   PROVED, not asserted: the whitespace-stripped line
                                           multiset over all of typecheck/ across the repair is
                                           identical but for imports, module docs, `impl` wrappers
                                           and comments. A pure move preserves the multiset
3 internals not accidentally public PASS   public API of crate::typecheck: 31 -> 31, IDENTICAL SET
4 build/dependency surface          PASS   no manifest change
5 size reported, never the criterion       the repair ADDED a module and grew the total
```

Local evidence at the repaired head: 759 tests across 18 targets, 0 failed; clippy
`--workspace --all-features --all-targets -- -D warnings` clean; conformance baseline clean.

### What AS8 inherits from this, beyond the five edges

**CD-391's own "finding AS8 inherits" section listed four blind checkers and drew the right
lesson — and this is a fifth instance of that same lesson, found after the record was written.**
The sentence it offered, *a check that does not cover the thing being claimed cannot support the
claim*, was correct and insufficiently applied: nobody asked what fraction of the input the check
covered. **Coverage of the checker itself is now a required measurement, not an inference from a
green run.** 36-of-234 is the number that should have been on the qualification and was not.

**Next:** AS8 resumes. EI trap classification, the EI2 evidence inventory and the EI4/EI5 rankings
are corrected under the same instruction and recorded separately.

## CD-392 — WP-ENGINE-INDEPENDENCE approved, transferred to AS8; AS0 stays closed (2026-08-09)

**Owner ruling. A governance gap, resolved prospectively rather than by rewriting history.**

`WP-ENGINE-INDEPENDENCE.md` was specified as an **AS0 assurance subpacket** and AS0's work section
instructed its execution, describing the shared-fate register, evidence audit and engine-risk
profiles as AS0 outputs. **Those outputs were omitted from AS0's binding exit criteria**, so AS0
closed without them — while AS8 explicitly requires *consuming* them and forbids repeating the
inventory under a second taxonomy. AS8's opening inventory (73c562d) found the packet still
`PROPOSED` with none of its five artefacts in existence.

```text
AS0 historical closure                  KEEP CLOSED
WP-ENGINE-INDEPENDENCE status           APPROVED, without redesign
Execution context                       transferred administratively to the AS8 assurance phase
Existing EI0-EI6 taxonomy and outputs   remain canonical
AS8 mutation target selection           BLOCKED until EI5 publishes the ranked list
A second AS8 engine taxonomy            PROHIBITED
Other AS8 evidence work                 MAY PROCEED in parallel
```

**This ruling changes scheduling and ownership only.** It does not alter compiler semantics and does
not reopen Campaign A.

**The full packet runs, not a minimal subset.** EI0 freezes the classification vocabulary before the
register is built; EI5's ranking includes `RUSTC_ASSUMPTION` and rustc-sensitive lowering, which
EI3 supplies. Executing only EI1/EI2/EI4/EI5 would leave EI3 dangling and reproduce this same
dependency debt at C10.

```text
EI0 vocabulary -> EI1 register -> EI2 evidence audit -> EI3 rustc/backend assumptions
    -> EI4 risk profiles -> EI5 ranked mutation targets    ===> AS8 mutation work unblocked
EI6 public claim calibration completes during AS8/C10 closure
```

**AS8 trials reference EI authority IDs.** AS8 may assign its own trial IDs; it may not invent a
semantic classification independent of the shared-fate register.

**The evidence invariant is now binding on AS8**, promoted from the requirement recorded at AS8
opening:

> **No evidence mechanism may support a claim until its ability to distinguish success from failure
> has itself been demonstrated.**

For mutation testing, before any real batch is reported: a **known-live mutant MUST be killed** and
a **known-equivalent/no-op mutant MUST survive** — proving the harness detects a real semantic
disturbance *and* does not merely fail whenever source changes. Likewise a coverage number declares
its target crates/files first, an LSP benchmark names and pins its representative projects first,
and a mutation kill rate declares its exact target population and tags first. Otherwise the
denominator is movable and the number is weak evidence.

This matters here specifically because EI5 itself warns that even a *killed* shared-authority
mutation is insufficient unless the killing evidence is independently derived.

## CD-391 — AS7 CLOSED, qualification PASS (2026-08-08)

> **SUPERSEDED IN PART by CD-393 (2026-08-09).** Criterion 2's PASS below was produced by a
> forcing test that observed 36 of 234 methods; five violations were live under it. The
> closure stands and the record is preserved as written — read CD-393 for the corrected
> criterion-2 result and the repaired module graph.

**`STARKLANG/docs/compiler/audits/AS7-EXIT-QUALIFICATION.md` is the record.**

The type checker is split by semantic ownership. `typecheck.rs` was 14,432 lines; `typecheck/mod.rs`
is now **596 production lines** — a facade carrying the module declarations, the four entry points
and AS6's `TensorCheckCtx` impl. Ten modules, dependency graph documented and **executable**:

```text
types <- state <- infer <- traits <- convert <- bounds <- patterns/body <- items <- mod

body 4,396 | traits 3,200 | infer 1,262 | types 1,153 | items 986
convert 901 | state 896 | patterns 429 | bounds 116 | mod 596 production
```

```text
1 no semantic behaviour or diagnostic structure changes  PASS  ZERO fixture changes, ten packets
2 dependency direction documented and cycle-free         PASS  regenerated at the exact head
3 internals not accidentally public                      PASS  API 31 -> 31; removed one that was
4 default build/dependency surface unchanged             PASS  no manifest change at all
5 file-size reduction reported as outcome                reported, never used as the criterion
```

CI green on the exact head: **CI 24/24 and C7.8 Native Capabilities 4/4 on both `977b7a3` and
`4c4311a`, zero failing**, including `fmt, clippy, test` on windows-x64 — the platform no local
evidence in this packet covered — and both C6.4 tier-1 qualification runs.

**The graph was revised once, under the packet's stop condition.** `convert` and `traits` were
genuinely strongly connected: converting `HashMap<K,V>` must prove `K: Hash + Eq`, and proving
`Iterator<Item = Foo>` must convert the written `Item`. The owner ruling rejected a permitted cycle
and added `bounds` as the missing orchestration layer. The pre-move dependency check found this
**before** the split fossilised it, which is the argument for writing such a check first.

**Ambient state.** All eight fields now enter and leave through named scoped operations that save
and restore; two latent defects were fixed — `current_fn_ret` was cleared rather than restored, and
`current_module` was never restored. Both were correct only under "item checking never nests", the
invariant the splitting was most likely to break.

**The finding AS8 inherits.** Four defects in AS7's own verification — `find` vs `rfind`,
trait-impl methods conflated with inherent ones, "prose" references that were machine-checked, and
`cargo check --lib` not compiling `#[cfg(test)]` code — plus one in the qualification's own
measurement. One sentence covers them: *a check that does not cover the thing being claimed cannot
support the claim.* The discipline that worked: *introduce the violation on purpose and watch the
check fail.* AS8 is entirely an evidence packet and should adopt it as a standing requirement — a
mutation suite that kills nothing looks identical to one that works until you try.

**Recorded limits:** `pub(super)` is wide inside `typecheck` and narrowing it is later work;
`body.rs` is 4,396 lines and a further split was not attempted because the approved decomposition
names one `body`; 35 historical `.md` files still say `typecheck.rs` and are preserved as written.

**Next:** AS8 may open. It is assurance against the frozen AS6/AS7 result. Sprint 4 then closes with
its Tier-3 closeout, and Campaign B exits before C10. **Unresolved:** PR #11 is a draft carrying 121
commits against `develop` purely to trigger CI; PR #10 (sprint-3) is the merge path for AS0-AS6.
Sprint 4 cannot close without deciding how AS7 and sprint-4 actually land.

## CD-390 — AS6 CLOSED, qualification PASS (2026-08-08)

**`STARKLANG/docs/compiler/audits/AS6-EXIT-QUALIFICATION.md` is the record.**

AS6 passes all five published exit criteria on head `6050efa`. CI fully green on all three Tier-1
platforms — CI 24/24 and C7.8 Native Capabilities 4/4, zero failing jobs — plus 629 local tests,
0 failed, and `cargo fmt --check` clean.

```text
1 Core-only sessions load no tensor-owned name or rule       PASS  two-directional, per surface
2 no open-ended tensor spelling tables in central Core       PASS  + a forcing test that pins it
3 tensor behaviour and ONNX verification unchanged           PASS  ZERO fixture changes
4 no public extension/plugin/provider API introduced         PASS  zero pub items; compiler-enforced
5 Part B artifact-provider work remains blocked              PASS  C9.3 evidence still absent
```

**What moved.** Three catalogues and six vocabulary tables left Core: the resolver's builtin
catalogue, `hir::Builtin`'s 33 tensor variants, `TENSOR_OPS` and the rule types, the parser's 21
spellings, the resolver's 15-name type table, and the checker's kind/device/value-range classifiers
*with the diagnostic phrases that recite them*. The tensor semantic authority — 1,276 lines of
dtype/shape/device/schema/broadcasting rules — sits behind a **fifteen-service context that does not
contain `check_expr`**, so the extension may consume checked types but cannot cause Core expression
checking. `TypeChecker`'s members are private to its module, so that boundary is compiler-enforced.

**Recorded residue, four entries across three files**, held as a set-equality ledger rather than a
skip-list: `ast::Primitive::name`'s two element-type spellings (a closed Core enum; sealing it is
the `hir::Builtin` cut applied to `Primitive`, wider than AS6 scoped), `deploy/ir.rs`'s `DeployTy`
Display, and `deploy/emit.rs`'s generated-Rust host type name.

**The finding.** Criterion 2 is the only one of the five with **no behavioural signature**, and it
failed twice after its surfaces had been declared clean — a spelling table returns one arm at a
time and every test still passes. AS7's criterion 2 ("dependency direction documented and
cycle-free") has the same shape; its executable check should exist **before** the modularisation
starts.

**Next:** AS7 may open. It requires exclusive tree ownership — a 14,000-line pass split cannot
survive a parallel session in the same file — and takes its ambient-state conversions first, as
separate commits, before the file splitting.

## CD-389 — Campaign B approved, Sprint 4 in execution, AS6 open (2026-08-08)

**Where compiler work actually stands. Read this before anything below it.**

```text
Campaign A          PASS — closed, CI-confirmed. Do not reopen.
Campaign B          APPROVED for execution, 2026-08-08 (AS6-AS8 as designed)
Current sprint      Sprint 4 — AS6, then AS7, then AS8
Current packet      AS6, extension quarantine — IN EXECUTION, NOT CLOSED
                    (superseded by CD-390: AS6 CLOSED; and CD-391: AS7 CLOSED)
Active branch       wp-arch-stability/sprint-4
Gate track          C0-C10: see the CD records below; C8 CLOSED (CD-385)
```

**AS6's position.** Six packets have landed. The tensor extension's *semantic authority* and its
*surface vocabulary* are out of Core:

```text
architectural discovery        DONE   46ae2ec
builtin/catalogue quarantine   DONE   fe80129
runtime/lowering boundary      DONE   33cb0a7
tensor type-system boundary    DONE   62ef6b0 (rules), 9147073 (authority, 15-service context)
parser residual audit          DONE   5190d1b   CI green, 28/28 jobs
exit-criterion cleanup         DONE   6050efa   model-decl split, vocabulary tables, forcing lint
AS6 exit qualification         DONE   PASS — see CD-390 above
```

**AS6 is NOT closed and nothing downstream may say it is.** Closure requires the published exit
criteria and Tier-2/CI qualification on the exact head. The owner approval of Campaign B ratifies
the landed work as *execution*; it does not substitute for that evidence.

**The finding this packet contributes.** Exit criterion 2 — "central Core modules do not contain
open-ended tensor spelling tables or method catalogues" — is the only one of AS6's five with **no
behavioural signature**. Criteria 1 and 3 are pinned in both directions by the session-isolation
suite; criterion 2 is not pinned by anything that runs. It therefore decayed silently, and residue
survived three packets that each reported their surface clean: `resolve.rs` kept a 15-name
vocabulary table, and a census found the tensor type-constructor spellings in **three** places and
the element-type spellings in **four**. `starkc/tests/as6_core_module_vocabulary.rs` is the
structural check that now pins it — the deliverable the work package listed and no implementation
packet had built.

**Carry into AS7:** a structural criterion needs a structural check committed with the cut, not a
procedure and not a reviewer's grep. AS7's criterion 2 ("dependency direction documented and
cycle-free") has the same shape and should get its executable check before the modularisation
starts.

**Branch note.** Sprint 4's AS6 packets began on `wp-arch-stability/sprint-3` to preserve the active
execution state after Campaign A closed; from `6050efa` execution continues on
`wp-arch-stability/sprint-4`. History is retained, not rewritten. See
`WP-ARCHITECTURE-STABILIZATION.md` §1.

## CD-385 — Gate C8 CLOSED, deliberately short on one requirement (2026-08-06)

**Owner ruling. `STARKLANG/docs/compiler/GATE-C8-CLOSURE.md` is the record.**

C8 had been CANDIDATE-COMPLETE since 2026-07-31 on a single blocking reason: missing interactive
editor validation. The 2026-07-31 session answered it for **three of ten** advertised features —
hover, go-to-definition, find-references. The other seven (diagnostics, formatting, completion,
signature help, rename, document symbols, semantic tokens) are protocol-tested only.

**The gate closes with that limit stated, not removed.** The execution plan's §17.2 lists ten
requirements for `C8-CLOSED`; nine are met and item 8, "real VS Code validation passes", is
**partially met and overridden by owner decision**. The closure document marks it that way so an
auditor reads "deliberately closed short" rather than "met". **DEV-012 stays OPEN, narrowed** to the
seven unvalidated features — the deviation carries the residue, exactly as C7 closed without a
steady-state runtime-performance claim.

Why close rather than hold: the blocking reason was environmental, holding the gate open does not
itself produce the missing evidence, and the ambiguity — not the risk — was blocking AS5 and AS8.

**A limit this ruling adds that the exit report did not have.** CD-384 (DEV-182, the same day) found
the LSP parser decoding every escaped non-BMP character to the empty string. WP-C8.7 is "Protocol
and editor validation" and that defect passed it, because a protocol test asserting a well-formed
exchange records agreement when both sides say "ok". C8's protocol evidence demonstrates *verdict*
agreement, not *value* agreement. Recorded as a standing limit; AS5's conformance corpus is where
value-level agreement gets established.

**Consequences.** AS5 and AS8 take their "C8 closes first" branches and are unblocked.
`WP-ARCHITECTURE-STABILIZATION.md`, the C8 exit report, the C8 evidence README, the C9 plan's
starting-position note and `KNOWN-DEVIATIONS.md` were all reconciled in the same change.

## CD-384 — DEV-182, the LSP parser silently ate every non-BMP character (2026-08-06)

Taken as a live-defect pre-emption under `WP-ARCHITECTURE-STABILIZATION.md` §3, on its own branch
with its own evidence, rather than absorbed into the Sprint 1 architecture work that found it.

`parse_json_string` read the four hex digits of a `\u` escape, called `char::from_u32`, and
**discarded the escape when that failed**. It always fails for a surrogate half — a surrogate is not
a Rust `char` — and there was no pairing step. So:

```text
"😀"   ->  ""      (should be U+1F600)
"\ud83d"         ->  ""      (should be REJECTED: unpaired surrogate)
"\u00zz"         ->  ""      (should be REJECTED: malformed escape)
```

Every case **parsed successfully** and returned a value the input did not denote. That is why it
survived: a verdict-based comparison records agreement, because both sides say "ok". It was found by
the AS0 manifest-strictness audit only because that audit compares **values, not just verdicts** —
the third delta the work-package added after the first draft compared verdicts alone.

Any editor sending a non-BMP character in a completion label, file path, or diagnostic message lost
it silently. Emoji in a string literal is the ordinary case.

The repair pairs surrogates per RFC 8259 §7 and rejects what cannot be paired: a lone high or low
surrogate, a high surrogate not followed by a `\u` low surrogate, a malformed or truncated hex
escape. `read_hex4` returns `None` on a non-hex digit rather than contributing nothing, so a bad
escape now fails the parse instead of vanishing from a value reported as good.

Eight tests, five of which failed before the repair (`surrogate_pairs_decode_to_one_scalar` returned
`Some("")`). `cargo test --lib` 531 passed / 0 failed; `cargo fmt --check` and
`cargo clippy --all-targets -D warnings` clean.

**Not fixed here, and still open for AS5.** The same parser accepts trailing input after a complete
value, and both it and `package.rs` accept raw control characters in strings; `escape_json_string`
does not escape control characters on output; `package.rs` rejects all `\u` escapes and all exponent
numbers. Those are the tightening and compatibility axes recorded in
`AS0-MANIFEST-STRICTNESS-AUDIT.md`, not this defect.

## CD-383 — two over-acceptances, one of them a double destruction (2026-08-05)

An external audit separated the open defects by direction: programs the checker REJECTS that it
should accept (DEV-167, DEV-168, DEV-172) against programs it ACCEPTS that the language forbids.
The second kind ships; the first is visible the moment you compile. Both of the second kind are
fixed here.

### DEV-169 — and the answer to the question the ledger left open

The record said "whether the drop flag suppresses the second run is unverified". **It does not.**

```stark
let mut resource = Resource { id: 1 };
resource.drop();
println("after");
```
```text
dropped
after
dropped
```

The destructor runs **twice on one value**. For a resource-bearing type that is a double release.
This was not an over-acceptance with cosmetic consequences; it was a soundness violation, and the
ledger had classified it as "potentially safety-significant" precisely because nobody had run it.
Running it took one minute and should have happened when the defect was recorded.

03-Type-System.md, "Copy and Drop" is unambiguous: "`Drop::drop` MUST NOT be called explicitly; use
the free function `drop(value)`." The check now runs at IMPL-MEMBER SELECTION — when a call
resolves into an `impl Drop for T` block — rather than on the method's NAME. That distinction is
the difference between a correct fix and a broken one: an inherent method called `drop` is an
ordinary method and stays callable, which a name-keyed check would have refused.

### DEV-171 — operator bounds by identity, not spelling

```stark
mod fake { pub trait Eq { fn unrelated(&self) -> Int32; } }
use fake::Eq;
fn compare<T: Eq>(a: T, b: T) -> Bool { a == b }   // was accepted
```

`param_declares_bound` compared the bound's SOURCE TEXT against `"Eq"`. Written qualified
(`T: fake::Eq`) the same program was rejected — the tell that the answer depended on spelling. It
now resolves each bound through `hir::resolved_bound_trait`, the identity path CD-379 established,
and compares it to the Core trait the operator requires. `Eq`, `Ord` and `Num` all route through
that one branch, so all three are covered by construction rather than by enumeration.

`satisfies_bound_parts` keeps its name-addressable form deliberately: DEV-118's built-in
obligations have no `TraitRef` to resolve. Only the generic-parameter branch changed — the one that
had a written bound all along.

**This rejects programs that previously compiled**, which is the intent and is stated as a
behaviour change rather than described as a pure bug fix.

### What was NOT taken on, and why

* **DEV-121's class closure** (a runtime representation contradicting the checked type) is a
  representation-invariant extension across loop bindings, call arguments and every
  reference-producing intrinsic. Different mechanism, different pass; bundling it here would make
  neither reviewable.
* **DEV-165** (`connect_timeout` accepted and ignored) is an HTTP-client defect, not a compiler
  one, and the audit itself said it does not belong in a compiler correctness packet.

Both remain open with their priority recorded.

### stark-args

Verified and included in the same commit at the owner's request: `stark check`, 9 package tests,
`stark fmt --check`, **10 of 10 declared callables called**, consumer check/run/build, and
byte-identical output from the interpreter and the native binary (`7|alpha|--literal`). Counts moved
25 → 26 packages and 22 → 23 consumers across README, CLAUDE.md, AGENTS.md, the website and the
sweep skill, and the README package table gained a Command line row.

### Evidence

`starkc/tests/over_acceptance_audit.rs` — 8 tests. Each fix is paired with the cases that must keep
working, because the risk in both repairs is over-rejection: a method merely NAMED `drop`, the free
function `drop(value)` destroying exactly once, automatic destruction still firing, and genuine
`Eq`/`Ord`/`Num` bounds including a user `impl Eq`.

Regression: adversarial_trait_impls, c62d_operator_coretrait, gate2_valid, conformance all green.

### Status

DEV-169 and DEV-171 CLOSED. **Every known accepted-invalid program in the compiler checker is now
rejected**; DEV-121 (accepted-valid program, wrong execution) and DEV-165 (accepted configuration,
no effect) remain open and are different categories.


## CD-382 — DEV-173 CLOSED: a nested string literal inside an interpolation field (2026-08-05)

```stark
println(f"{choose(\"yes\", \"no\", true)}");
println(f"{lookup(\"name\")}");
```

The form the original WP-FMT-001 acceptance matrix asked for, and the one CD-380 refused. With it,
interpolation admits ordinary expressions rather than ordinary-expressions-minus-string-literals,
and WP-FMT-001 is no longer "v0.1 partial".

### Why it was hard, stated exactly

A nested string literal inside a field must be written `\"a\"` — the enclosing literal is delimited
by `"`, so its quotes have to be escaped — and `\"` is not expression syntax. The field cannot be
lexed from the file as written.

CD-380 tried parsing a DECODED copy and retagging the resulting spans to the field's span. That
produced the WRONG STRING: a literal read its value from its span, so a retagged literal read the
field's raw source back and rendered `\"slice\"` where the program said `slice`. Refusing was the
right call at the time. A later attempt at proper span REMAPPING failed for a second reason worth
recording: spans are embedded throughout the AST — in `Path`, `PathSegment`, names — not only on
nodes, so remapping node spans left identifiers pointing at file offset 0.

### The repair, in two halves, both required

1. **A length-preserving stand-in.** The field is parsed against a copy of the whole file in which
   each `\"` inside that field becomes ` "` when it opens a nested literal and `" ` when it closes
   one. Every byte offset is unchanged, so every span the sub-parse produces is already a real file
   span and nothing is remapped — which is what sidesteps the embedded-span problem entirely.

   **Which side the space lands on is load-bearing.** Blanking the closing backslash in place puts
   the space INSIDE the literal: `f"{choose(\"yes\", ..)}"` renders `yes ` with a trailing space.
   Observed while running it, not reasoned about afterwards.

2. **Literals carry their decoded value.** `Ast::str_lits`/`Hir::str_lits` hold every string
   literal's value, interned at parse time from whatever buffer the parser was reading, and
   `Lit::Str` names its entry. **Spans are now purely diagnostic.** This is the architectural half
   the DEV-173 record predicted, and it is what makes the stand-in sound: without it a literal would
   still read its value back from the real file's `\"a\"`.

### What is still refused, and why it is a different thing

An escape other than `\"` in a field source belongs to the enclosing literal and *changes* the
inner text — `\\` means one backslash, `\n` means a newline — so blanking it would silently alter
the value. Those fields are refused with that reason. This is not the old blanket refusal narrowed
by convenience: it is the boundary between "the escape is punctuation the outer literal forced" and
"the escape is data".

### Evidence

`tests/wp_fmt_001_interpolation.rs` — 40 tests. `a_field_may_contain_a_nested_string_literal`
covers six forms including a `:` and a `}` INSIDE a nested string (both of which the field scanner
must read as text, not structure), a struct literal, and a format specification applied to a nested
literal. `a_field_may_not_contain_an_escape_other_than_a_quote` pins the remaining refusal.

Regression: three_engine_differential 109, mir_differential 132, dev_display_dispatch 21,
dev_bound_trait_identity 15, adversarial_stderr 11, gate2_valid 56, conformance 3, robustness 6,
span_integrity 2.

### Status

DEV-173 CLOSED.

**WP-FMT-001 is complete for the defined Core v1 interpolation surface.** Plain nested string
literals are supported; a nested literal carrying a DATA-BEARING escape (`\n`, `\t`, `\\`,
`\u{...}`) remains an explicitly rejected future extension.

That wording is a correction. This entry first said "complete ordinary-expression interpolation",
which is literally broader than the implementation: `f"{lookup(\"a\nb\")}"` is a valid ordinary
expression and is refused. **That is the third claim in this work package stated wider than what was
built** — after §6's version claim and the `println(f"...")` evidence gap — and the pattern is the
finding, not the individual wordings. Each was caught by review rather than by me, and each was a
summary written from what the change was *for* rather than from what it *does*.

The restriction is retained in the specification (01-Lexical-Grammar, LEX-FORMAT-004) rather than
living only in a defect record, so a reader of the grammar sees it.

Tier-1 qualification: `7e41a1e` was green across all three lanes, which closed CD-381's outstanding
item. **That run does not cover this commit** — `c5b7581` changed the parser, AST/HIR literal
storage, the interpreter and MIR lowering — so it carries its own CI result, and the evidence claim
here rests on that one.


## CD-381 — WP-FMT-001 correction packet: six defects, one of them mine to admit (2026-08-04)

An external review of `987369b` reopened WP-FMT-001. It was right on every point, and one of them is
a **false statement in my own closure report**: §6 said the MIR runtime-surface version had changed.
It had not. `MIR_RUNTIME_SURFACE` was still `0.1-A13` while twelve `RuntimeFn` members had been
added across CD-378 and CD-380.

### The six

1. **Runtime surface unversioned.** `0.1-A14` now covers all twelve additions — CD-378's seven
   `Fmt*` members as well as CD-380's five `Fmt*Spec` ones. Both work packages added runtime surface
   without advancing the constant, so a consumer built against A13 would have accepted a program it
   cannot represent instead of rejecting it (V-SURFACE-1). `MIR_VERSION` stays `0.3`: additive
   surface, not a structural change.
2. **The field scanner did not know comments.** `f"{value /* } */}"` mis-scanned, because the
   scanner skipped escapes, strings and char literals but not `//` or `/* */`. A field delegates to
   the ordinary expression parser, so it must admit ordinary comments — including NESTED block
   comments, which 01-Lexical-Grammar §6 requires and which a one-character patch would not handle.
3. **The verifier typed the specification operands but did not require them to be constants.**
   Verified MIR could therefore have carried `FmtIntSpec(v, computed_word, computed_fill)` — dynamic
   formatting beneath a feature defined as statically specified, with a specification word no front
   end had validated. **MIR-0037** now requires both operands to be constants, requires the word to
   decode to a valid specification with zero unused bits, and bounds width and precision by
   LIMIT-FMT-WIDTH/PRECISION. A side effect worth stating: `Spec::unpack`'s defaults for unknown
   align/sign/kind encodings are now unreachable in verified MIR rather than silently normalising
   malformed compiler output.
4. **Inert flags were accepted.** `f"{42:#}"` set `alternate` and rendered nothing different;
   `f"{42:0}"` set `zero_pad` with width zero; `f"{1.25:f}"` asked for fixed-point with no
   precision; `f"{n:<06}"` wrote an alignment that zero-padding then overrode. LEX-FORMAT-003 says
   an implementation must reject a specification it does not act on — these were exactly the case it
   names, and they are refused now.
5. **LIMIT-FMT-SEGMENTS had an off-by-one hole.** The check ran after each field, so the trailing
   literal segment could push the count one past the limit. Checked once over the finished list.
6. **`println(f"...")` was only ever tested through `.as_str()`.** Proving the advertised form
   failed immediately — which is the point. See DEV-174.

### DEV-174, found by fixing the test rather than the code

`eprint`/`eprintln` were typed `&str` while 06-Standard-Library declares
`fn eprintln<T: Display>(value: T)` and PRINT-DISPLAY-001 names all four output functions together.
`eprintln(s)` with an owned `String` was rejected; `println(s)` was accepted. The stderr half of the
runtime surface has carried the full display family since 0.1-A13 and lowering already redirects by
channel — **only the signature lagged**, and no test had ever passed `eprintln` anything but a
`&str`. Both pairs are now typed alike and both go through the same deferred `Display` check.

Testing the convenient form instead of the advertised one is how a gap survives a suite that looks
thorough. Worth remembering beyond this work package.

### Scope, corrected

WP-FMT-001 is **IMPLEMENTED — v0.1 partial interpolation**, not closed. DEV-173 blocks a field
containing an escape sequence, and the original acceptance matrix included
`f"{choose(\"yes\", \"no\")}"`. Declaring closure while that is refused was an overclaim; the
closure report now says so at the top rather than having the claim edited away. Tier-1 qualification
is also still unobserved — the first push failed CI on clippy before reaching those lanes.

### Evidence

`tests/wp_fmt_001_interpolation.rs` — 39 tests, adding the direct `println(f"...")`/`eprintln`
form, comments inside fields (including a nested block comment and `}`/`:` inside one), and the six
inert-flag refusals.

### CI found three more, and one of them is a guard working as designed

The first push of this packet failed CI on three targets:

* `a10_provider_call::runtime_surface_is_current` and
  `a11_host_resource::the_mir_version_records_every_shape_amendment` both PIN the surface constant.
  Bumping to A14 fired them, which is their purpose. **But note what that means:** these guards
  pin the constant, not the surface, so they fail when the constant moves and stay silent when the
  surface grows without it. They were green through CD-378 and CD-380 while twelve `RuntimeFn`
  members were added unannounced. Recorded in `a10_provider_call.rs` rather than left implicit; a
  guard that could fail for the right reason would have to derive something from the `RuntimeFn`
  set itself.
* `adversarial_stderr::the_eprint_family_accepts_only_str_today` — the WP-C7.9 test that pinned
  DEV-174's restriction. Its own doc comment said the lowering already supported every `Display`
  shape, that widening would need "only a signature change and cases", and that this test "fails
  the day that happens, which is the right moment to add them." It did, and they are added: the
  three shapes it rejected now render byte for byte on stderr, plus a negative pinning that a type
  without `Display` is still refused there.

A recorded limitation carrying a test that fails when it is lifted is worth more than a to-do
comment. It turned this repair into one commit instead of a rediscovery.

### Status

DEV-174 CLOSED. DEV-173 remains open and is what stands between "v0.1 partial" and complete
ordinary-expression interpolation. The architecture was not the problem and is unchanged.


## CD-380 — WP-FMT-001: interpolated string literals (2026-08-04)

```stark
let message = f"pkg={name} n={count:04} r={ratio:.2} ok={ok}";
```
```text
pkg=stark n=0042 r=0.76 ok=true
```

**STARK has a complete, compile-time-checked string formatting feature.** `f"..."` produces an
ordinary owned `String` through the `Display` architecture CD-378 and CD-379 established. It is not
a macro, not a variadic call, and not a runtime-parsed format string: segments are split at parse
time, every specification is validated against its field's type at type checking, and no format
string exists in a running program.

### One implementation of the rules, three engines

Alignment, fill, width, sign, radix, alternate prefix and fixed precision live in
`stark_runtime::fmt_spec` and nowhere else. `starkc` already depends on that crate and every native
binary links it, so the HIR oracle, the MIR interpreter and generated code call the SAME functions —
the arrangement that already keeps `x.fmt()` and `println(x)` from drifting. A specification reaches
the runtime as a packed `UInt64` word plus a `Char` fill, both compile-time constants.

MIR gained **five** operations — `FmtPad`, `FmtIntSpec`, `FmtUIntSpec`, `FmtFloat64Spec`,
`FmtFloat32Spec` — not one per syntax combination. Everything about *how* a value renders is in the
word; the operation only says which value family it is.

### Tokenizing without a token soup

`f"..."` is ONE token, scanned exactly like a cooked string. The parser splits it and, for each
field, lexes **the original file over that field's own byte range** (`lexer::tokenize_range`) and
parses it with the ordinary expression parser. So spans inside an interpolation are real file spans
— `tests/span_integrity.rs` now asserts a field's expression is spanned inside its literal — and
nesting, calls, indexing, struct literals and paths are handled by the parser that already handles
them. No source text is ever reconstructed or rewritten.

The scanner tracks depth over `(`/`[`/`{` and consumes escapes whole, so `Point { x: 1, y: 2 }`'s
`:` and `}` stay inside the expression, `module::CONST`'s `::` is not a specification separator, and
`\u{1F600}`'s braces are an escape rather than a field.

### Rulings taken, and why

* **Width counts Unicode scalars**, not bytes and not terminal cells — the only choice that renders
  identically on every platform. It never truncates.
* **Odd centring puts the extra fill on the right**, so `{"x":^4}` is `| x  |`.
* **Sign, then prefix, then zero-padding, then digits**: `-00042`, `0x000000ff`.
* **A negative value in another base keeps its sign and renders its magnitude**: `-255` in hex is
  `-ff`. The host's two's-complement pattern is never exposed.
* **`0x` prefixes both hex cases** — the prefix names the base, the type character chooses digit case.
* **Rounding is half-to-even**; `Float32` renders at its declared width, never widened first;
  non-finite values ignore precision (`NaN`, not `NaN.00`).
* **Precision on a string is REFUSED.** It could only mean truncation, and Core v1 has no ruling on
  scalar-versus-grapheme-versus-byte cutting. Refusing beats guessing.
* **Alignment without a width is refused** — it is a no-op, and almost always a typo.
* **A numeric mode on a generic `T: Display` is refused.** `Display` does not prove integer
  formatting, and inventing a numeric bound to make it compile was explicitly out of scope.

### Ownership

A field **borrows**. `Display::fmt` is `&self`, so a place expression is read, not moved: `f"{x}"`
twice then `use_value(x)` is legal, for a non-`Copy` value and for an affine `Drop`-bearing one. A
temporary field is destroyed exactly once after its bytes are appended. Fields evaluate strictly
left to right, exactly once each — never a second time to discover a width.

### Spec first

**LEX-FORMAT-001/002/003** (01-Lexical-Grammar), **EXPR-FORMAT-001** (02-Syntax-Grammar) and
**STD-FORMAT-002…005** (06-Standard-Library) state the grammar, evaluation order, ownership,
type/spec compatibility, byte-exact rendering, and that interpolation is human-readable formatting
rather than an escaping mechanism for JSON, HTML, SQL, shell or URLs. Compiled spec regenerated; the
fixture corpus is now **114** blocks, the new one triaged in the same change.

### Evidence

- `starkc/tests/wp_fmt_001_interpolation.rs` — 36 tests. Every positive case runs the three-engine
  comparator with stdout pinned in the test; plus debug-vs-release native agreement, and a
  generated-source check that interpolation reaches `stark_runtime::fmt_spec` and that the crate
  contains no `format!`, `write!`, `writeln!`, `std::fmt::Display`, `std::fmt::Debug` or
  `#[derive(Debug`.
- `stark_runtime::fmt_spec` — 16 unit tests on the rules themselves, including `i64::MIN`, `-ff`,
  half-to-even, `Float32` width preservation and scalar-counted width.
- `src/format_syntax.rs` — 12 unit tests on the scanner and specification grammar, including the
  malformed inputs that must diagnose rather than panic.
- `stark fmt` round-trips interpolated literals byte-identically; `packages/stark-fmt` is unchanged
  and green, and interpolation needs no dependency on it.

### Deliberate exclusions, recorded rather than implied

* **No multiline interpolated form.** STARK has no multiline string literal to prefix; §2.5 said not
  to invent one.
* **No raw interpolated form** (`rf"..."`) — deferred, as §2.5 permits.
* **The source formatter reprints an interpolated literal verbatim** rather than re-formatting its
  embedded expressions. Reconstructing the literal risks changing what the program prints, which is
  a semantic difference, not a formatting one. §19 permits this trade and asks that it be recorded.

### Opened

- **DEV-172 — no signed type can express its own minimum value.** `let a: Int8 = -128;` is
  rejected: the magnitude is range-checked before the unary minus. Pre-existing, unrelated to
  formatting, found while testing that formatting a minimum does not overflow. The RENDERER handles
  `i64::MIN` correctly; no STARK program can produce the value to hand it.
- **DEV-173 — an interpolation field may not contain an escape sequence.** A nested string literal
  inside a field necessarily carries the outer literal's escapes, and parsing a decoded copy makes a
  string literal read its own source back (`\"slice\"` for `slice`) because literals read their value
  from their span. Refused rather than mis-parsed; workaround is to bind the value first.

### Status

WP-FMT-001 CLOSED for FMT-0 through FMT-5 as scoped. Formatting is sufficient for CLI output and
structured log lines; see the closure report for the REST-server assessment.


## CD-379 — DEV-BOUND-TRAIT-IDENTITY: a bound denoted whatever trait was spelled the same (2026-08-04)

**A follow-up to CD-378, and a correction to it.** CD-378 unified method candidate *collection*
across user and compiler-known traits. The step before that — deciding WHICH trait a bound denotes —
was still done by spelling, in two passes, and below the front end execution did not use the answer
at all.

### Four failures, all reproduced before any code changed

`typecheck::resolve_bound_trait` and `borrowck::bound_method_receiver` each took
`text(bound.path.span)` and scanned every HIR item for a trait declared with that name.

1. **A qualified bound matched nothing.** `T: traits::Render` compared `"traits::Render"` against
   the declaration's name `"Render"`. The bound contributed no methods, and `value.render()` was
   rejected with *"method 'render' requires the bound 'T: Render'"* — on a function whose signature
   already wrote exactly that bound. Every bound on a trait a package exports through a module was
   unusable.
2. **An unrelated trait captured the name.** `mod unrelated { pub trait Display { fn other(&self); } }`
   anywhere in the program took over every `T: Display` bound. CD-378's own §2 stated this as a
   design — "a user trait of the same spelling wins, exactly as `resolve_path` does" — which was the
   defect written down as a rule: `resolve_path` resolves against the bound's module and imports; a
   global name scan does not.
3. **Declaration order decided ownership.** Two same-named traits, one `&self` and one `self`: the
   borrow checker returned whichever appeared FIRST in HIR item order. The same program compiled or
   failed E0100 depending only on the order its two trait declarations were written in. The
   regression test is that pair, both halves of which must compile.
4. **Execution ignored the identity entirely.** Even with the front end fixed, both engines selected
   an implementation by method NAME on the receiver's nominal, so a type implementing two same-named
   `Render` traits ran the same body for both bounds. The type checker was right and every engine
   below it was wrong the same way — which is exactly what three-engine agreement cannot detect.

**Failures 1 and 2 are refusals. Failures 3 and 4 are acceptances of the wrong program** — an
order-dependent move check, and a call executing a different trait's body than the one type checking
approved. That is the more serious half of this entry.

### The repair: one identity, read from the resolver

`hir::resolved_bound_trait(hir, bound)` reads `TraitRef::res` and nothing else, with exhaustive
matches over `Res` and `ItemKind` — a new resolution or item category forces a decision here rather
than falling into a `_ => None`. `hir::BoundTrait` moved out of `typecheck` so both front-end passes
consume the same type and the same answer. **No spelling-based bound lookup remains in either pass.**

Below the front end: the checker records the selected trait per call site
(`TypeTables::bound_trait_calls`, `Res::Item` or `Res::CoreTrait`); the HIR interpreter passes it to
`find_method`'s already-existing trait filter, and MIR lowering passes it to a new one on
`find_impl_fn`. A filtered lookup considers only that trait's impl — never an inherent method and
never another trait's — exactly as a qualified call does.

**Canonical symbols now carry the trait's module path.** `impl left::Render for Item` and
`impl right::Render for Item` both produced `Item::Render::tag@[]`; the C5.4a linkage preflight
refused the program as "one symbol, two identities", and it was right to. A top-level trait's prefix
is empty, so every pre-existing symbol is byte-identical.

### What CD-378 got right, kept

Candidate collection, selection, ambiguity, the single Core-trait signature table, the `&self`
ruling for `Display::fmt`, and the missing-bound diagnostic all stand unchanged. All 21 cases in
`tests/dev_display_dispatch.rs` pass unmodified; `stark-fmt`'s public API, its 7 tests and both
consumer paths are unchanged.

### Evidence

- `starkc/tests/dev_bound_trait_identity.rs` — 15 tests: qualified bounds through nested generics
  and an impl head; two same-named traits in two modules dispatching to `L` and `R` (which pins
  which BODY ran, not merely that it compiled); an unrelated `Display` failing to capture a Core
  bound and an imported one correctly winning; receiver identity across `&self`, `&mut self` and
  `self`; the declaration-order pair; and a direct assertion that `resolved_bound_trait` returns the
  resolver's own `Res::Item`.
- Correction appended to `starkc/docs/compiler/WP-DEV-DISPLAY-DISPATCH.md` — append-only, stating
  what that report examined and what it did not, rather than editing it to look prescient.

### Opened

- **DEV-171 — an unrelated trait satisfies an OPERATOR bound by spelling.** `use fake::Eq;` then
  `fn compare<T: Eq>(a: T, b: T) -> Bool { a == b }` is ACCEPTED; written qualified (`T: fake::Eq`)
  it is correctly rejected. `ty_satisfies_operator_bound` compares the bound's text against `"Eq"`.
  Not fixed here: it is bound *satisfaction* rather than method identity, the same function also
  serves built-in obligations that have no `TraitRef` (DEV-118), and the repair decides what a
  user trait shadowing a Core trait's name means for operators — a semantics ruling, CE2-shaped.

### Status

DEV-170 CLOSED. **DEV-DISPLAY-DISPATCH (CD-378) is now fully closed**: it was closed for the
property it stated and open on one it did not state — that a bound denotes the trait the resolver
selected. Both hold.


## CD-378 — DEV-DISPLAY-DISPATCH: a compiler-known trait bound was not a trait bound (2026-08-04)

**`fn show<T: Display>(x: T) -> String { x.fmt() }` was rejected.** `[E0302] method 'fmt' not found
for type 'T'` — while the identical shape over a user-declared trait compiled and ran. The bound was
*checked*; it contributed nothing to method resolution.

### The defect is the trait model, not formatting

`typecheck.rs::resolve_method`'s bounded-generic branch resolved each bound by searching
`hir::ItemKind::Trait` items for a matching name. A compiler-known trait has no declaration item —
`resolve.rs` turns `Display` into `Res::CoreTrait(CoreTrait::Display)` and there the trait ends — so
the search returned `None`, the loop fell through, and the impl scan below could not match a
`Ty::Param` receiver either. Method visibility depended on whether a trait happened to be
compiler-known. That is two trait models, and the same hole covered `Ord::cmp`, `Clone::clone`,
`Hash::hash`, `Iterator::next` and `Into::into` on a bounded parameter. `Display` is where it was
noticed only because a `Display` bound has no purpose except calling `fmt`.

**DEV-023 (WP-C2.11) recorded that `Display`/`Hash` as bounds were "already correctly recognized".**
That was true of bound CHECKING and false of everything downstream. It fixed the concrete half
(`"hi".fmt()`) and left the generic half open, and nothing in the entry distinguished the two claims.

### Two more defects were in the same branch, and both are pre-existing

* **The move checker had no bounded-generic receiver at all.** `borrowck.rs::method_receiver`
  returned `None` for `Ty::Param`, and its caller's `None` arm CONSUMES the receiver. Every `&self`
  method reached through any bound moved it — for USER traits too:
  `fn f<T: Named>(x: T) { x.name(); x.name(); }` failed E0100 "use of moved value". Confirmed
  empirically before the fix. This had to be fixed here, because "format a value and keep using it"
  is the property the work package exists to establish.
* **Bound order was a resolution rule.** The branch returned on the first bound supplying the name,
  so `T: A + B` with both declaring `m` picked `A` silently instead of reporting ambiguity.

### What landed: one candidate path

`BoundTrait` makes both kinds of trait *an identity a bound resolves to* — `User(ItemId)` or
`Core(CoreTrait)`. Candidates are collected additively from every bound, de-duplicated by trait
identity, and then ONE selection runs: zero is a missing-bound diagnostic, one is checked, more than
one is E0203 naming both traits. Argument checking, `Self` substitution, associated-type
normalisation and diagnostics are shared from that point. A user trait of the same spelling wins,
the same precedence `resolve_path` already applies.

**No second signature registry was added.** `core_trait_contract` — WP-C7.9 Packet B's table for
checking user `impl` blocks against a Core trait's required shape — already carried
`fmt / Some(Ref) / [] / String`. A bound now reads that table. What a bound makes callable is by
construction what an implementation must provide. The filter on which Core methods a bound exposes
is `receiver.is_some()`, a property of the contract: `Default::default` and `From::from` have no
receiver and therefore no method spelling to resolve. **No method-name branch exists anywhere in the
change** — nothing keys on the string `"fmt"`.

### The missing-bound diagnostic

```text
[E0302] method 'fmt' requires the bound 'T: Display'
   |
   |     x.fmt()
   |     ^^^^^^^ 'T' has no bound that declares 'fmt'
```

Derived from the traits actually in scope, user and compiler-known alike, so it also names a user
trait and says nothing when no trait declares the name (that case keeps the plain "not found"
wording). Fires for `fn bad<T>(..)` and for `fn bad<T: Named>(..)` identically.

### The concrete tail: primitives had no `Display` impl to find

Monomorphisation grinds `T` down before MIR sees it. For a user nominal the ordinary impl path
resolved `fmt` already; for a PRIMITIVE there is no impl item, because 06 declares
`impl Display for Int32` "and similar for other types" and no source file writes those blocks. Seven
`RuntimeFn` variants (`FmtInt64`, `FmtUInt64`, `FmtBool`, `FmtFloat64`, `FmtFloat32`, `FmtChar`,
`FmtUnit`) are the lowering of exactly those declarations, sharing `stark_runtime::format`'s
renderers with the `Print*` family — so `x.fmt()` and `println(x)` cannot disagree in any engine.
`String`/`str` reuse `StringAsStr`/`StrToString`. The `RuntimeFn` matches in `emit_runtime.rs`,
`mir/verify.rs` and `mir/interp.rs` are exhaustive, so all three were forced open by the addition.

### Spec first

**TYPE-METHOD-003** (03-Type-System.md) states that a generic parameter's candidates come from its
bounds and nowhere else, that collection is additive, that written order is not a selection rule,
and that a compiler-known trait contributes through the same collection with no priority.
**STD-TRAIT-002** (06-Standard-Library.md) states the same property from the library side and names
the program a conforming implementation must accept. STD-FORMAT-001 gained the sentence the
ownership work depends on: `Display::fmt`'s receiver is `&self`; formatting borrows and never
consumes, which is what makes `Display` usable at all for an affine type. Compiled spec regenerated.

### Evidence

- `starkc/tests/dev_display_dispatch.rs` — 21 tests. Every positive case goes through the shared
  three-engine comparator with stdout pinned in the test rather than taken from an engine: all
  `Display` primitives through one generic function, user impls, non-`Copy` and affine values used
  after formatting, nested `outer<T>`→`inner<U>` forwarding, both bound orders, an impl-head bound,
  and debug-vs-release native agreement. Negative: missing bound, wrong bound, unknown method,
  non-`Display` concrete type, arity, and ambiguity in BOTH orders.
- `native_selects_stark_formatting_not_rusts` reads the generated crate and requires
  `stark_runtime::format::fmt_i64` present and `format!`, `std::fmt::Display`, `std::fmt::Debug`,
  `#[derive(Debug`, `ToString` absent.
- `packages/stark-fmt` + `packages/stark-fmt-consumer` — the proof workload, registered in the
  qualification gate. `Line::value<T: Display>` and `to_string<T: Display>` are the whole surface.
  7 package tests; consumer runs identically under the interpreter and as a native binary.
- Full report: `starkc/docs/compiler/WP-DEV-DISPLAY-DISPATCH.md`.

### One transitional compromise, stated plainly

`core_trait_contract` is not an ordinary trait DECLARATION, and the preferred architecture asks for
one. Core trait method metadata must eventually be derived from real prelude trait items carrying a
lang-item-like classification, at which point that table and `BoundTrait::Core` both disappear and
`BoundTrait` collapses to a single `ItemId`. That is a resolver-bootstrap change — the prelude has
no source file today — and is a tracked follow-up, not part of this work package.

### Opened

- **DEV-167** — no method-form `to_string()`; needs blanket implementations. `stark-fmt` ships the
  free function. Deferred by decision, NOT by resolver special-casing.
- **DEV-168** — `Display::fmt(&x)` has no MIR lowering ("callee form (C4.5)"). TYPE-METHOD-001 names
  this call as the way to disambiguate an ambiguous trait method, and it runs in one engine of three.
  Found while proving the ambiguity this work package introduces is resolvable.
- **DEV-169** — an explicit `.drop()` call type-checks. Pre-existing, in the CONCRETE path;
  `Drop::drop` was included in the bound surface so the generic path matches it rather than
  disagreeing for no stated reason. Needs a spec-vs-implementation ruling.
- Untracked follow-up: `Clone::clone`, `Hash::hash`, `Iterator::next` and `Into::into` are now
  callable through a bound at the front end, and their concrete lowering is uneven — a program using
  them generically now fails at LOWERING rather than at type checking. Worse diagnostic position for
  shapes that were rejected outright before; not a regression in what compiles.

### Status

DEV-166 CLOSED. The REST server's formatting prerequisite is **met** for rendering values into text;
see the work-package report §8 for the two limits to scope around (no format strings; `Display` is
not a serialisation format — use `stark-json` for payloads).


## CD-377 — installer Phase I: the layout the compiler could not find (2026-08-03)

**The installed toolchain could not build anything on macOS or Windows.** CI caught the symptom on
Linux, where it was a stale path assertion; underneath it was a real defect that Linux alone would
never have shown.

### The installer and the compiler disagreed about the layout

The installer now writes a VERSIONED tree — `lib/stark/current` → `versions/<v>`, payload beneath —
and puts a **symlink** (Unix) or a **copy** (Windows) at `<prefix>/bin/stark`. `discover_runtime`
searched only

```text
<bin>/../lib/stark/starkc/stark-runtime
<bin>/../lib/stark/stark-runtime
```

neither of which exists in that tree.

**It worked on Linux by accident.** `current_exe()` there resolves `/proc/self/exe`, so invoking the
`bin/` symlink already reported the real location and the flat form matched. macOS does not resolve
it; Windows installs a copy, so there is no link to resolve. Same package, three platforms, one
working — the DEV-163 shape exactly.

Reproduced without a Windows machine, by invoking both paths:

```text
/tmp/prefix/bin/stark build                      -> runtime installation is missing
/tmp/prefix/lib/stark/current/bin/stark build    -> Built app
```

Fixed by teaching `discover_runtime` the versioned forms FIRST, so the lookup no longer depends on
the exe path having been resolved through a symlink.

**My earlier "verified end to end" missed this because I never set
`STARK_REQUIRE_INSTALLED_RUNTIME=1`.** Without it the compiler falls back to a source checkout, so
every one of those builds was proving the checkout worked. The environment variable is the whole
experiment.

### `stark doctor`, hardened

Three findings from external review, all confirmed before fixing:

- **Windows executable name.** `("bin", "bin/stark")` was hardcoded, and `install.ps1` runs
  `stark.exe doctor --root` during staging and throws on failure — so a correct Windows package was
  rejected with "staged STARK installation failed manifest verification". Install-blocking, and
  invisible on Unix. Now read from the manifest's `host_target`, which also makes `doctor --root`
  work when inspecting a package built for another platform.
- **The manifest reader was formatting-dependent, and its failure mode was silence.** It split the
  file array on the literal `"\n    {"`. A compact manifest yielded zero entries — and the old
  binary reports that as `manifest_files: ok (0/0 files verified)`. **A verifier that silently
  checks nothing and calls it a pass is worse than one that errors.** Replaced with a real
  recursive-descent parser: escapes including surrogate pairs, bounded nesting, duplicate keys
  rejected rather than last-wins, and sizes that must be whole and non-negative.
- **Manifest paths are now validated.** Relative, no `..`, no drive or absolute form, and unique
  after case folding — Windows and macOS filesystems are case-insensitive, so two entries differing
  only in case name one file and the second certifies whatever the first wrote. A path escaping the
  root would let a manifest certify a file the package never installed.

`serde_json` was recommended and is **not** taken: `starkc` has three dependencies, and adding
`serde` plus a proc-macro chain is a supply-chain decision for the owner, not a code fix. The
defect is closed either way. Nine adversarial tests cover the parser and the path rules.

### Classification — Phase I, not a distribution

```text
Installer Phase I / compiler distribution   IMPLEMENTED
Standalone first-party toolchain            PARTIAL      packages are not in the payload
Offline package/provider build              NOT PROVEN
Public signed distribution                  NOT PROVEN   integrity, not authenticity
```

`manifest.json` detects corruption. It does not establish that the manifest came from a STARK
release — anyone who can replace the payload replaces the manifest with it. Signing, a trusted key,
verification before installation and notarisation are all outstanding.

## CD-376 — HC13 correction: two remote aborts, and a timeout claim counted wrong (2026-08-03)

**External review of `bfceaa0`. Every point was correct and every one is verified in the code
below, not accepted on assertion.** The HTTP client is reclassified **feature-track complete, not
security-release complete**.

### SEC-HTTP-001 and SEC-HTTP-002 — availability vulnerabilities, not parse errors

STARK traps on integer overflow in **every build mode**, so an arithmetic boundary in the parser is
not a wrong error message — it is a **remote process abort**. A hostile server choosing its own
response could stop any client reading it.

```text
SEC-HTTP-001   Content-Length: 18446744073709551616
               guard rejected `value > 1844674407370955161` but ADMITTED the boundary, then added
               a digit up to 9 on top of ...610
SEC-HTTP-002   chunked: "1\r\nx\r\nFFFFFFFFFFFFFFFF"
               `FFFFFFFFFFFFFFFF` accumulates to exactly u64::MAX WITHOUT overflowing, so the size
               parses legitimately; `body.len() + size` then overflowed on any non-empty body
```

**Why HC13's eleven malformed routes missed them.** Both sit exactly where the magnitude guard
stops and the final accumulation still happens. `not-a-number` and `zz` are refused long before the
boundary, so ordinary malformed-input coverage cannot reach either. Adversarial infrastructure is
necessary and not sufficient; the routes have to be aimed at the arithmetic.

Fixed: the Content-Length guard now checks the final digit at the boundary, and the cumulative chunk
check is a **subtraction** — with a `>=` guard first, because a subtraction that underflows traps
exactly as an addition that overflows does, and swapping one for the other would have moved the
defect rather than fixed it.

**Falsified.** Reverting either fix makes its test fail with `integer overflow`. Two new wire routes
(`/bad-length-overflow`, `/bad-chunk-cumulative-overflow`) prove the same against a live peer — and
there, reaching the *next line of output at all* is half the assertion, because before the fix the
process died.

### The timeout evidence was counted wrong, by me

Three stalling routes prove **two** phases, not three: `/slow-headers` and `/slow-body` both report
`ReadResponse`. The report said "three different phases" while its own case inventory showed
otherwise, and the limitations document said "two are not proved" and then listed three.

Worse than the arithmetic: **two of those were filed as "unproven" when they are not implemented.**

```text
ReadResponse    PROVEN
TlsHandshake    PROVEN
WriteRequest    UNPROVEN          deadline installed; no peer fills a receive window
Connect         NOT IMPLEMENTED   DEV-165
Resolve         ABSENT            no mechanism could produce it
```

**DEV-165 — `ClientConfig.connect_timeout` is advertised and never enforced.** The client calls
`connect_no_timeout`, and `stark-net::connect` refuses every non-zero timeout with `Unsupported`.
A caller setting it gets no error and no effect. My limitations document claimed it "IS applied to
the socket" — that was simply false, and it is the worst kind of documentation error because it
reads as reassurance. Deferred to the networking roadmap (it needs a non-blocking connect and a
poll, i.e. a provider ABI change), but the false claim is removed now.

**`Resolve` is ABSENT.** `stark-net::resolve` takes a host, a port and size/count limits, and passes
no duration to the provider. Filing it under "unproven" invited a reader to assume it merely lacked
a test.

### Status

```text
HC0-HC12 feature programme        CLOSED
HC13 adversarial qualification    CLOSED (corrected here)
HTTP client FEATURE track         COMPLETE
SEC-HTTP-001, SEC-HTTP-002        CLOSED (this)
DEV-163, DEV-164                  CLOSED (CD-375)
DEV-165                           OPEN -- deferred to the networking roadmap
PUBLIC RELEASE readiness          BLOCKED -- DEV-165, and no installer exists
```

Evidence: 42 executed cases (13 malformed, 4 oversized, 3 stalls), 36 parser unit tests, 16
packages through the full gate.

## CD-375 — HC13 CLOSED: adversarial peers, DEV-163 and DEV-164 (2026-08-03)

**The HTTP client track is complete: HC0–HC13.** HC13's job was to prove the client **fails
correctly**, which is a different property from proving it works and the one that had never been
tested end to end.

### The finding, which is the point of the packet

**DEV-163 — a read timeout did not report as a timeout on Unix.**

```text
Unix     SO_RCVTIMEO expires -> EAGAIN       -> ErrorKind::WouldBlock -> NetworkError::Interrupted
Windows  SO_RCVTIMEO expires -> WSAETIMEDOUT -> ErrorKind::TimedOut   -> NetworkError::TimedOut
```

So `stark-http-client` reported **"the connection failed"** on Linux and macOS and **"timed out
reading the response"** on Windows — identical peer, identical STARK source. An operator reading the
Unix message would have gone to look at the network instead of at the peer deliberately holding the
socket.

The deadline always worked. Only its **report** was wrong, which is exactly why nothing caught it:
every test through HC0–HC12 used a peer that *answers*, and a timeout that never fires cannot
misreport. It was found within an hour of a peer existing that stalls.

Fixed in `stark-net`'s native provider, where the socket mode is known. A provider stream is always
blocking — the only `set_nonblocking(true)` in that file is the test harness's listener — so
`WouldBlock` from a read or write can mean one thing: the deadline expired. Both platforms now
report `STATUS_TIMED_OUT`.

### What was built

```text
11 malformed routes    status line, version, header name, obs-fold, bare LF, two lengths,
                       length+TE, length value, transfer coding, chunk size, chunk terminator
 4 oversized routes    status line, header line, header count, body ceiling
 3 stalls              slow headers, slow body, a TCP peer that never speaks TLS
```

`stark-http-parser` already had 34 unit tests over an error type with 23 variants — but they assert
that malformed input is *rejected*, not *which* error it produces, and they hand the parser a
literal rather than delivering it over a socket. **No test anywhere would have noticed a bare LF
being reported as a chunk-size error.** These eighteen do, on the wire.

**Each case asserts the NAMED reason, not merely failure.** Eighteen cases all reporting "the
response was bad" would also pass against a client that rejected the valid responses above them in
the same run. The reason is what distinguishes a parser from a wall.

Two design points worth keeping:

- `/big-body` declares 12 MiB and **actually sends it**, against a lowered ceiling. The limit is
  enforced on total bytes read, not on the parsed body, so a peer cannot evade it by under-declaring
  `Content-Length`. A header check alone would pass the peer that lies.
- `tls_stall_peer` **holds** each accepted connection rather than closing it. Closing gives a
  connection error, which is a different outcome from a handshake that never progresses — and the
  wrong one to be testing.

### A second finding, from my own test: DEV-164

Adding the DEV-163 regression test made `a_detached_socket_is_live_and_this_provider_has_forgotten_it`
fail about **one run in five**. It was green in twelve consecutive runs without the new test, so
this was mine — not a flake I merely uncovered.

`stark-net`'s provider table is process-global and `cargo test` runs tests in parallel in one
process. Two tests hand a raw socket OUT of the provider (`detach` -> `into_raw_fd` -> `adopt`),
which leaves a live fd outside any Rust owner for a window; a third test opening and closing sockets
alongside them makes that window observable. The symptom was a detached socket that connected,
accepted its writes, and then reported `UnexpectedEof` instead of the echo.

**The product is not at fault** — `next_id` is monotonic under the lock so handle ids are never
reused, and `detach` consumes the stream with `into_raw_fd` so nothing closes the fd. The defect is
in the test suite's sharing of process-global state. Fixed with the writer-serialising mutex this
repository already uses for the TLS lifecycle tests: **every test that opens a socket takes it, not
only the ones that assert on the table.** A test that merely opens a socket perturbs a table
assertion just as much as one that reads it.

Verified 0/20 with the guard against ~1/5 without.

Two things went wrong on the way to that, and both are worth recording because they cost the most
time:

- I first blamed the echo harness's 5-second handler timeout and **tested it** — raised it to 60s,
  and the failure persisted. Refuted in one run; had I "fixed" it instead, I would have shipped a
  change that did nothing and claimed a cause.
- My first version of the test parked a thread for three seconds. Removing that made the failure
  *rarer* rather than gone, which is the worst possible outcome and was only caught by running the
  suite twenty times instead of once.

### One criterion is partial, and says so

Three of five timeout phases are proved on the wire. `Connect` and `Resolve` are not: a loopback
cannot black-hole a SYN deterministically, and a flaky negative test is worse than an absent one
because it teaches people to re-run the suite. They are recorded as **unproven**, not as working —
which is the distinction DEV-163 exists to justify.

Marking the criterion ✅ on three of five would be exactly the overstatement this packet punished.

### Evidence

```text
40  executed cases for stark-http-client, all native, all against live loopback peers
    (18 of them new: 11 malformed, 4 oversized, 3 stalls)
16  first-party packages through the full gate, exit 0
 3  Tier-1 platforms, no platform gating anywhere in the harness
10  of 10 required fixture servers built
 5  evidence documents
```

Every peer **asserts its bind rather than attempting it**. A skipped peer would silently downgrade
lifecycle evidence to lowering evidence while the gate still reported success — which matters most
on Windows, where a loopback TLS listener is likeliest to fail to bind.

### Status

```text
HC0-HC12  CLOSED
HC13      CLOSED -- one acceptance criterion partial and reported as partial
DEV-163   CLOSED (this)
DEV-164   CLOSED (this) -- provider test suite shared process-global state
```

Still open from the track, each deliberately in its own packet: dot-segment resolution in
`stark-url`; making `Header`/`HeaderMap.entries` private (an API break); and **no installable
toolchain** — no release has been published and `build-release.py` does not stage provider crates,
so a package release would produce a client nobody can run.

## CD-374 — DEV-160: the call thunk, and the shapes it still refuses (2026-08-03)

**Owner ruling accepted (2026-08-03): DEV-160 is NOT closed as a family.** The call-thunk
architecture, the Miri evidence mechanism and the named-refusal boundaries are approved; the defect
splits into four, of which one closes here. Cross-block absorption is DEFERRED to its own work
package, and the HTTP workaround is KEPT.

```text
DEV-160a  same-block direct-call disjoint projections      CLOSED (this)
DEV-160b  borrow returned by an EARLIER call               OPEN / DEFERRED
DEV-160c  conflicting provider-call argument sequence      OPEN / DEFERRED
DEV-160d  borrow surviving beyond the sibling move/call    OPEN / DEFERRED
```

**b, c and d are over-refusals, not unsound execution.** Each is refused by name before rustc, which
is the correct outcome for a shape the backend cannot emit: without it they reach the user as
`E0502` inside `mod stark_proj` — a correct compiler error about code they never wrote.

### What a thunk is

One generated function per conflicting call site, in `mod stark_proj` beside the wrappers it calls:

```rust
pub fn stark_thunk_23main_40_5b_5d_23bb2<'a>(
    s0: &'a mut stark_runtime::slot::ValueSlot<stark_ty_230_40_5b_5d>,
) -> u32 {
    let p0: *mut stark_runtime::slot::ValueSlot<stark_ty_230_40_5b_5d> = s0;
    unsafe {
        let a0 = stark_refraw_23struct_230_23f0::<'a>(p0);
        let a1 = stark_moveraw_23struct_230_23f1(p0);
        let a2 = stark_copyraw_23struct_230_23f2(p0);
        stark_consume_40_5b_5d(a0, a1, a2)
    }
}
```

The slot arrives ONCE through a real `&'a mut`, one raw pointer is derived from it, and every
operand is evaluated through that pointer **in MIR order**. There is one `&mut` in existence, so
there is nothing left to conflict with; `'a` comes from a real reference, so a borrow the thunk
hands on has honest provenance. The call site is `stark_proj::NAME(&mut _1)` — one safe call, §7.8
intact.

### The part that was not in the plan: absorbing the borrow

The conflicting `&` is usually **not in the argument list at all**. `f(&p.name, p.body)` lowers to a
`RefOf` STATEMENT filling a temporary, and only the temporary is an argument. A thunk that took over
the argument list alone would leave that borrow live beside its own `&mut` and change nothing.

So the thunk takes over the borrow's statements too, and `emit_bodies` suppresses them. Three
conditions gate it, and each was needed:

```text
same block          moving the RefOf inside the thunk must not move it past a branch
projected base      a whole-slot borrow has no raw twin (and STARK rejects it beside a move anyway)
every read is here  the definition is suppressed, so nothing may need the value afterwards
```

Delaying the borrow is sound because the front end has already proved nothing between it and the
call can mutate what is borrowed. A *disjoint* sibling may be moved in that gap — and re-deriving
through a raw projection reads the untouched field either way, which is exactly what a whole-value
accessor could not do.

**`let r = &p.name;` lowers to a PAIR** — `_8 = &_1.0` then `_7 = copy _8` — and only `_7` reaches
the call. Following that chain, and suppressing every statement along it, is the difference between
absorbing the reported idiom and absorbing nothing.

### DEV-160d and DEV-160b, and why each is a refusal rather than a gap

**DEV-160d — a borrow that outlives the call.** `let r = &p.name; f(r, p.body); use(r);` cannot be absorbed —
suppressing the definition breaks the later read — and cannot be left alone. Refused, naming the
local and the field.

**DEV-160b — a borrow arriving through an earlier call.** This is the shape DEV-160 was reported as:

```text
send_once(builder.url.as_str(), builder.headers, builder.body)
```

`as_str` runs in an earlier block and returns a `&str` borrowing `builder`. By the outer call it is
an ordinary non-slot local carrying no sign of where it came from, so the backend now **traces
borrow provenance** — `RefOf` seeds it, copies and borrow-carrying aggregates propagate it
(OWN-CARRY-001 makes provenance structural), and a call's result inherits its arguments', which is
STARK's own shortest-input rule read as may-alias. A by-value argument whose type could carry a
borrow and whose provenance meets a participating slot is refused by name, with the workaround
stated.

Absorbing it means absorbing the intermediate CALL, across a block boundary, turning that block's
terminator into a `goto`. That is a second mechanism, not an extension of this one, and it changes
control flow — **flagged for an owner ruling rather than taken unilaterally.** The HTTP client's
`send()` workaround therefore stays, and the comment above it in `stark-http-client/src/lib.stark`
now says which half of DEV-160 closed and which did not.

Provenance over-approximates, filtered by type: without the type filter,
`consume(p.taken, p.kept.len())` would be refused, because `len` takes `&p.kept` and the relation
propagates — but the result is a `UInt64` and borrows nothing.

### DEV-160c — the provider audit

A provider call never reaches `emit_call`. It is emitted as a statement SEQUENCE — one
`let __prov_aN = ...;` per argument (A10/CD-200) — which has the SAME conflict: `__prov_a0` holding
a shared borrow is a live local when `__prov_a1` moves a sibling through `&mut`. The thunk does not
apply (there is no single expression to replace, and the ABI's out-parameters and handle transfers
are not arguments a thunk could carry), so it is refused by name. This was the audit the addendum
asked for, and it found a real path rather than confirming an unreachable one.

### What bounds the change

A call that does not conflict reaches none of this. Both mechanisms that could touch it — the plan
lookup in `emit_call`, the statement suppression in `emit_one_block` — are gated on a plan existing,
so `ordinary_calls_plan_nothing` asserts the detector stays silent on four shapes each ONE condition
away from conflicting: two `Copy` reads, a lone borrow, a lone move, a whole-value move.

One plan, three consumers. `emit_projections::collect` skips the argument lists the plans cover,
`emit_projections::emit` renders the helpers each plan names, and `emit_call` looks its plan up. The
addendum required this and it is not decoration: DEV-162 shipped an `E0425` precisely because the
emitter named a helper the collector had never been asked for.

### Miri, and keeping the fixture honest

A thunk is generated code, and Miri cannot run what has not been generated. So `stark-runtime`
carries a hand-written one and a pinned CI job (`nightly-2026-07-20`,
`-Zmiri-strict-provenance`) runs the slot primitives under it — the only check here that can tell a
sound raw projection from one that merely happens to work.

That arrangement has an obvious failure mode: the generator changes, the fixture does not, and the
Miri job keeps proving something about code the compiler no longer emits. So the fixture publishes
`GENERATED_THUNK_SHAPE`, and `the_miri_fixture_matches_what_the_generator_emits` derives the same
sequence from a freshly generated thunk by resolving each wrapper to the primitive inside it. The
two must agree. **Neither check is worth much without the other.**

`-Zmiri-ignore-leaks` is required and does not weaken the aliasing check: three `should_panic` tests
hold heap values when the panic aborts them, which is what those tests are for.

### Evidence

```text
8   DEV-160 tests -- 4 executed through HIR, MIR, native-debug AND native-release,
                     2 refusal assertions, 1 bounding invariant, 1 fixture-drift check
26  stark-runtime slot tests, all green under Miri with strict provenance
23  suites re-run green locally: MIR lowering/verification/differential, the
    three-engine differential, ownership, aggregates, generics, function values,
    providers, host resources, and DEV-135/150/154/162
```

Local runs are scoped by design; `cargo test --workspace` belongs to CI, which is what the totals
should be read from.

The four executed cases are compared across engines rather than each asserted to exit 0 — including
the ordering case, where the `Copy` read is deliberately the THIRD argument, after the move, so a
reordered thunk would read storage a sibling had already left.

### Status

```text
DEV-158   install through a whole-value accessor      CLOSED (CD-371)
DEV-162   read through a whole-value accessor         CLOSED (CD-372)
DEV-160a  same-block conflicting evaluation           CLOSED (this)
DEV-160b  borrow through an earlier call              REFUSED by name; DEFERRED by ruling
DEV-160c  provider-call argument sequences            REFUSED by name; DEFERRED by ruling
DEV-160d  borrow outliving the call                   REFUSED by name; DEFERRED by ruling
```

### Why DEV-160b is a work package and not a follow-up commit

It is not an extension of the thunk. It has to absorb an EARLIER call terminator, replace that
terminator with a `goto`, preserve the failure and control-flow behaviour of the call it absorbed,
preserve the returned reference's provenance, potentially span several blocks, and coordinate more
than one call result. Every one of those is a property the current mechanism does not touch.

## CD-373 — DEV-160 foundation: the raw-slot primitives, and an order finding (2026-08-03)

**Owner ruling accepted (raw-pointer call-site thunk; argument reordering PROHIBITED because CD-007
freezes left-to-right evaluation).** This lands the foundation only. **DEV-160 is still OPEN** — no
thunk is generated yet and the HTTP workaround stays.

### What landed

Four `unsafe` raw-pointer primitives on `ValueSlot`: `field_ref_raw`, `move_field_raw`,
`copy_field_raw`, `take_raw`. They take `*mut ValueSlot<T>` and never form a reference to the slot,
so a borrow of one field and a move of a disjoint sibling can be live together — which the
`&self`/`&mut self` forms cannot express, because each borrows the whole slot. That inexpressibility
IS DEV-160.

**The ruling's lifetime point was decisive and I would have got it wrong.** My plan was to change
the existing helpers to take raw pointers and keep returning `&F`. A safe function returning a
reference derived from a raw pointer alone has no lifetime source — the borrow would be unbounded
and the signature a lie. So these are `unsafe`, carry an explicit `'a`, and are callable only from a
generated thunk that takes the slot ONCE through a real `&'a mut ValueSlot<T>`, which is what
anchors every reference it hands on.

The aliasing rule is written into the module: inside such a thunk no `&ValueSlot` or
`&mut ValueSlot` may be reconstructed after a field reference has been derived. Every access goes
through the raw pointer for the thunk's whole body.

Four tests, including the shape that motivates the whole thing — a field borrow and a sibling move
alive simultaneously — plus dead-slot and partial-slot refusals through the raw path, so the checks
are not skipped merely because the caller holds a pointer.

### A finding the thunk design has to absorb

Working through the emission, the thunk cannot take only the CONFLICTING slot and receive the other
arguments pre-evaluated. Evaluating a non-conflicting argument at the call site would place it
BEFORE the projections performed inside the thunk, which is the argument-order change the ruling
prohibits.

So the thunk must take **every distinct local an argument reads**, by `&mut ValueSlot<..>`, and
perform **every** operand read inside itself, in MIR order. That is consistent with the ruling's
"performs the fixed disjoint accesses internally ... in MIR order, and invokes the callee" — stated
here because it is a bigger obligation than "hand the thunk the conflicting slot", and it decides
the thunk's signature.

Constants and non-slot scalar locals may still be passed by value: their reads are unobservable and
order-insensitive.

### Remaining for DEV-160

```text
conflict detector       same slot base in >= 2 argument places, at least one requiring &mut
thunk plan identity     body/call-site, callee identity + signature, base slot type,
                        ordered argument modes, ordered projection chains, return type
thunk generation        into mod stark_proj, safe signature, raw body, MIR order, callee call
call-site emission      one safe call, no unsafe in the generated MIR body
negative controls       the owner's fifteen, incl. drop-exactly-once, overlap refusals,
                        indirect/runtime/provider call audit, debug AND release agreement
```

### Evidence

30 runtime tests, 161 across MIR verification, the three-engine differential and DEV-162's
regression, clippy clean. Nothing behaves differently yet: the primitives are unreferenced by any
emission path.

## CD-372 — DEV-162 CLOSED; DEV-160's obvious fix does not work, and here is why (2026-08-03)

### DEV-162 — reading a sibling field of partially-moved storage

Sibling of DEV-158, same root cause. Once a field is moved out the storage is `Partial`, and a read
of an UNTOUCHED sibling was emitted as `&slot.get().f1`, where `get` requires a complete value:

```text
_7.reinit(stark_proj::stark_move_23struct_230_23f0(&mut _1));
_13 = (&_1.get().f1);   // aborts: the slot is PARTIAL
```

`copy_field` already covered the `Copy` case by value (WP-C6.1b). This is the rest: a non-`Copy`
field, borrowed. `ValueSlot::field_ref` reads through a raw projection, so it never materialises a
reference to the surrounding value. `HelperOp::Ref` joins Move/Copy/Drop/Write, and the emitted form
is `(*stark_proj::stark_ref_…(&_1))` — dereferenced, because callers in `Borrow` mode prepend their
own `&` and need a place expression.

**The part missed first.** `Rvalue::RefOf` carries a PLACE, not an operand, so `rvalue_operands`
returns nothing for it and the collector never generated the helper the emitter had already named.
That surfaces as `E0425` inside the generated crate — a name error in code nobody wrote — not as any
diagnostic the compiler produces. Collector and emitter must agree and nothing but a build proves
it, which is now what the regression test does, across three engines.

### DEV-160 — the obvious fix does NOT work, recorded before anyone tries it

```stark
consume(p.url.as_str(), p.headers, p.body)   // accepted by STARK, E0502 in generated Rust
```

The instinct — and my own first plan — is to hoist each argument into a temporary before the call:

```rust
let __a0 = stark_ref_…url(&_1);
let __a1 = stark_move_…headers(&mut _1);   // still E0502
```

**It does not help.** `__a0` holds a shared borrow of `_1` that stays live until the call consumes
it, so every later `&mut _1` still conflicts. Sequencing the statements changes nothing about the
borrow's extent.

Two options actually remain, and both have a real cost:

```text
reorder    emit every `&mut` argument BEFORE any borrow that lives into the call. Sound here —
           a borrowed field and a moved field are necessarily disjoint or MIR would have refused
           the program — but it changes ARGUMENT EVALUATION ORDER, which CD-007 fixes. Needs a
           decision, not just an edit.

raw ptr    give the helpers `*mut ValueSlot<T>` parameters, which do not participate in borrowck.
           Conflicts with §7.8's rule that generated MIR bodies contain no `unsafe` of their own,
           unless the unsafety is pushed entirely inside `mod stark_proj`.
```

Recorded rather than attempted. Getting this wrong quietly changes evaluation order for every
call in the language.

### Where the family stands

```text
DEV-158  install through a whole-value accessor      CLOSED (CD-371)
DEV-162  read through a whole-value accessor         CLOSED (this)
DEV-160  whole-slot borrows, disjoint projections    OPEN — needs an evaluation-order ruling
```

Two of three closed. The remaining one is not a bug to fix but a decision to take.

### Evidence

378 across the MIR, native, ownership and aggregate suites; 26 runtime; clippy clean; all 16
packages green. `dev162_partial_field_read.rs` compares three engines rather than asserting each
exits 0 separately, and covers the `Copy` sibling alongside the non-`Copy` one so a regression in
either is visible against the other.

## CD-371 — DEV-158 CLOSED; the diagnosis was wrong twice before it was right (2026-08-03)

**Assigning over a struct field whose old value is a drop unit now works natively.** Both HTTP
workarounds are removed and the packages still build and pass.

### The defect was in TWO places, and I had only found the second

I documented DEV-158 twice as "no operation returns a slot from `Partial` to `Whole`". True, and not
the abort. Reading the generated Rust — which is what I should have done first — showed it:

```rust
_7.reinit(stark_proj::stark_move_23struct_231_23f0(&mut _3));  // slot -> PARTIAL
_3.get_mut().f0 = _6.take();                                   // <- ABORTS: get_mut needs WHOLE
```

The INSTALL uses a whole-value accessor on storage the matching move-out just made partial. The
missing state transition is real and necessary, and it runs after a line that never completed. So
the fix is two halves:

```text
ValueSlot::write_field    a raw-projection field write, valid over partially-moved storage
ValueSlot::mark_whole     the state transition, guarded by MIR's drop flags
```

`HelperOp::Write` joins Move/Copy/Drop in the projection-wrapper generator, and `emit_assignment`
routes a projected destination in slot-backed storage through it. `ptr::write`, not assignment: the
field is uninitialised at every generated call site because CD-012 requires the old unit to be moved
out first, so there is nothing to drop and assigning would drop garbage.

### A gate copied without re-deriving its reason

`emit_storage_whole` was written by copying `emit_storage_dead`'s gating, including its no-op
drop-plan check. That check is right for `finish_partial` — a no-drop slot is written with `reinit`,
which has no dead-slot check for a storage END to satisfy — and **wrong** for `mark_whole`: a slot is
made partial by a field MOVE, and `take` aborts on it whether or not the whole-type plan is a no-op.

Worse, a no-op whole-type plan is the COMMON case, because MIR decomposes an aggregate's drop into
per-unit flag-guarded drops. So the copied gate suppressed emission for exactly the shape DEV-158 is
about. The reproducer's `Config` reported `plan noop = true` and `mark_whole` was never emitted — the
statement was in the MIR and produced no code. Found by instrumenting rather than by reading it
again.

### The guard is correct in both directions

Proved, not asserted. A struct with two droppable fields, one moved out and never restored, the
other assigned: the guard must NOT fire, and a wrong fire is observable rather than silent — the
scope-end `finish_partial` would hit a WHOLE slot and abort by name. It passes.

### What this did NOT fix

**Reading one field of partially-moved storage still aborts.** `t.b.as_str()` after `t.a` was moved
out goes through `get()`, which requires WHOLE. Same family — a whole-value accessor over partial
storage — and the same family as DEV-160's whole-slot borrows. Filed as DEV-162.

That makes three in one class, and they want one fix rather than three:

```text
DEV-158  install through a whole-value accessor        CLOSED
DEV-162  READ through a whole-value accessor           OPEN
DEV-160  whole-slot borrows for disjoint projections   OPEN
```

### Evidence

3 new runtime tests (write_field's siblings survive, mark_whole in all three states), 315 across the
MIR and native suites, 26 runtime, clippy clean, all 16 packages green. The original reproducer and
the HC11 three-field shape both run natively and agree with the interpreter, and both HTTP
workarounds are deleted rather than merely marked removable.

### The process note

Three diagnoses, two wrong, and the two wrong ones were both reasoning from the source rather than
from the artefact. The generated Rust was available the whole time and named the failing line in one
read. When a backend defect is about what the backend EMITS, read the emission first.

## CD-370 — the diagnostic-injection hole I opened while closing the wire one; DEV-161 (2026-08-03)

**From a second Codex review of CD-369. Both findings were right, and the first is a mistake worth
naming precisely.**

### The repair reintroduced the injection one layer out

CD-369's commit message argued that a rejected VALUE must never be echoed, because it is
attacker-influenced and echoing it moves the injection into the log. Correct — and then the same
commit carried the rejected NAME verbatim into the error text. An invalid name may itself contain
CRLF. My own regression test asserted the reported name was exactly `X-Test\r\nInjected`.

So the reasoning was right and the code did the opposite of it, in the adjacent case.

Fixed **structurally** rather than by escaping:

```stark
InvalidHeaderName            carries NOTHING — the name is what failed, so there is no safe
                             version of it to report
InvalidHeaderValue(name)     carries the name, safe HERE and only here because the name is
                             checked FIRST and this variant is unreachable until it passed
```

The order of the two checks is the safety argument, and it is stated in the source. Escaping was
rejected as the primary fix: a sanitiser is something a future call site can forget, whereas a
variant carrying no string cannot leak one. The new test renders the error and scans it for control
bytes, so it asserts the property rather than the shape.

### `Content-Type` gets the same singleton policy as `Location`

`json_checked` used `get_first`; two `Content-Type` headers are two contradictory claims about the
same bytes, which is the same class of silent choice. Now `AmbiguousContentType`. And
`RequestBuilder::json` REFUSES when the caller already set one, rather than appending a second —
appending would put the contradiction on the wire and leave the winner to the server.

### DEV-161 — an ambient `CARGO_TARGET_DIR` breaks every native build

Cargo's default output is `<manifest dir>/target`, which is where the backend looks. An exported
`CARGO_TARGET_DIR` overrides it, the child inherits it, the build SUCCEEDS elsewhere, and the
backend reports "Cargo succeeded but the expected binary is missing" — naming neither the cause nor
the variable. `CARGO_TARGET_DIR` is a common global setting, so any developer with it exported could
not `stark build` at all.

Fixed by passing `--target-dir` explicitly, with the read path reusing the same value, so nothing
about the environment can separate where the build writes from where the backend looks.

**How it was found is the uncomfortable part.** It broke `mir_statement_consumers` and
`c788_resource_lifecycle`, and I reported both as pre-existing environmental failures unrelated to
my changes — twice. The second time I "confirmed" it by stashing every change and re-running. **That
control was worthless: the stashed run had the same variable exported.** Controlling for the code
while holding the environment fixed proves nothing about the environment. The review pushed back on
the dismissal, which is the only reason it got looked at.

Both suites now pass, including under the hostile variable. `StorageWhole`'s handling by every
statement consumer is therefore execution-evidenced, not merely compile-evidenced — which was the
review's specific concern.

### Still open, unchanged

```text
dot-segment reference resolution   bounded RFC 3986 resolver, belongs in stark-url
Header/HeaderMap field privacy     an API break, its own change
DEV-158                            lowering + runtime guard, the hard half
DEV-160                            field-granular generated projections
DEV-159                            native build racing its own dependency build
HC13                               not started
```

## CD-369 — HC12.1: a proven CRLF-injection hole closed, plus two P1 redirect gaps (2026-08-03)

**From an external Codex review of CD-368. All three findings were real; the first is a security
defect that predates HC12 and I verified it by exploit before fixing it.**

### P0 — header validation was bypassable, and it was reachable

`stark-http-core::header()` validates on construction, and the serializer trusted that. But
`Header`'s fields and `HeaderMap.entries` are PUBLIC, so a header can come into being without ever
touching the constructor. Written as a probe and run:

```text
value: "safe\r\nInjected: yes"

GET / HTTP/1.1\r\n
Host: a.test\r\n
X-Test: safe\r\n
Injected: yes\r\n        <- a header the caller never wrote
Connection: close\r\n
```

CRLF header injection, from safe STARK, no `unsafe` and no provider.

**The invariant is now enforced where the bytes are emitted**, because that is the only place that
cannot be bypassed by constructing the value differently. `SerializeError::InvalidHeader(name)`
carries the NAME only — a value rejected for containing CRLF is attacker-influenced by definition,
and echoing it into a log moves the injection one layer out instead of stopping it.

The regression test IS the exploit, plus bare CR, bare LF, NUL, and four invalid name shapes — and
one control asserting a well-formed hand-built header still serializes, so the repair rejects what
cannot be written rather than everything built without the constructor.

**Still open, recorded not fixed:** making `Header`/`HeaderMap.entries` private behind validated
accessors. That is an API break and belongs in its own change; this closes the hole.

### P1 — two URI-reference forms were silently mis-resolved

| base + Location | was | now |
| --- | --- | --- |
| `/one/two?q=1` + `?page=2` | `/one/?page=2` — a DIFFERENT resource | `/one/two?page=2` |
| `/one/two` + `ftp://other.test/f` | `http://a.test/one/ftp://other.test/f` | refused |

The first silently requested something the server did not name. The second fell through to the
relative-path branch: not dialling FTP is not the same as being correct. Fragment-only references
are refused too — they address a position in the current document, so there is nothing to fetch.

The scheme check follows RFC 3986 (`ALPHA *( ALPHA / DIGIT / "+" / "-" / "." ) ":"`, colon before
any `/?#`), so ordinary paths containing a colon still resolve — pinned by test, because a check
that swallowed `a/b:c` would be its own bug.

### P1 — a duplicate `Location` was first-wins

Now `get_singleton`, and `AmbiguousLocation`. Two `Location` headers are two destinations, and
picking one silently is a choice between things the server said — the class of disagreement request
smuggling is built on. `headers_for_next_hop` also propagates a validation failure instead of
silently omitting the header, since "the second request quietly lost a header" is indistinguishable
from a bug at the far end.

### Still open from the review

**Dot-segment removal (`.` / `..`) is not implemented.** Codex is right that the real answer is a
bounded RFC 3986 resolver in `stark-url` rather than a second URL implementation growing inside the
HTTP client, and a half-written normaliser is worse than none. Recorded for HC13's packet.

### DEV-158 — the fix is in progress, not landed

`ValueSlot::mark_whole` exists and is proven (3 tests: partial→whole with the field written back and
every whole-value operation working afterwards, idempotent on whole, refused on dead).
`Statement::StorageWhole` is defined and wired through the verifier (MIR-0036), the interpreter
(inert), linkage and the emitter. **Lowering does not emit it yet**, so nothing behaves differently
and the workarounds stay.

One finding while sizing it: the cheap static shortcut — "if the assigned place covers all the
local's drop units, wholeness follows" — is TOO WEAK. `RequestBuilder` has three droppable fields,
so `out.body = body` covers one of three: exactly HC11's case, still broken. It would have looked
like a fix and left the motivating instance failing. The real emission needs the runtime conjunction
of the local's drop flags, which is what remains.

## CD-368 — HC12 CLOSED: safe redirects; DEV-160 found (2026-08-03)

**Redirect support is opt-in, bounded, and cannot silently forward credentials to another origin.**
All three words are separate mechanisms. Full record:
`STARKLANG/docs/http-client/HC12-REDIRECT-EVIDENCE.md`.

```text
opt-in        follow_redirects defaults false; off, a 3xx is RETURNED, not errored — a redirect
              is a valid answer and hiding it would misreport what the server said
bounded       max_redirects (5) AND loop detection over every visited URL — two different faults,
              two different errors, because raising the limit should fix one and not the other
not silently  Authorization and Cookie stripped on any origin change; opting out is possible and
              is named `preserve_authorization_same_origin_only`
```

### Two rulings worth stating

**301/302 rewrite POST to GET**, contradicting a literal reading of the RFCs and matching every
browser and `curl -L`. The letter would send a POST body to a target the origin server redirected a
POST *away* from — both surprising and the more dangerous reading. 307/308 preserve and replay,
which is safe only because a body is a buffered `Vec<UInt8>`.

**Origin comparison uses the EFFECTIVE port**, so `https://h/` and `https://h:443/` are one origin.
Otherwise a redirect that merely spelled the port differently would strip credentials for no reason,
and callers would learn to turn the stripping off — which is how a safety default dies.

### A bug the pure tests could not have found

The 303 case asserts against what the PEER received. Method and body were already correct, and
`Content-Type: text/plain` was still riding along on a bodyless `GET` — a claim about content that
is not there. Dropping a body now drops every header that describes one. The rewrite-table test
alone would have passed; the echo route reflecting the actual wire is what caught it.

### DEV-160 — place-granular borrows, whole-value projections (OPEN)

STARK's borrow checker is place-granular (DEV-154) and correctly accepts disjoint-field borrows in
one call:

```stark
send_once(client, builder.method, builder.url.as_str(), builder.headers, builder.body)
```

The generated projections take `&slot` and `&mut slot`, losing that granularity, so **rustc rejects
the generated code**:

```text
error[E0502]: cannot borrow `_2` as mutable because it is also borrowed as immutable
```

A correct program refused by the backend. Worked around by moving the fields into locals first.

**This is the same shape as DEV-158** — the slot abstraction is whole-value while the ownership
model is place-granular — and it is the third defect in that family. Whatever fixes the
`Partial`/`Whole` transition should be scoped to look at projection granularity generally rather
than at field assignment alone.

### Evidence

38 `stark-http-client` tests (9 new) and 22 consumer cases (10 new), all against live peers. The
credential case reads the WIRE rather than the policy flag: the cleartext peer redirects to the TLS
peer, and the echo route reflecting `GET|-|-|` proves the header was absent on the second request.
The bound and the loop are proved separately — `/r-loop` revisits one target, `/r-hopN` walks an
ever-lengthening chain of distinct ones.

## CD-367 — HC11 CLOSED: JSON convenience, and a strict UTF-8 decoder (2026-08-03)

**Common JSON REST calls no longer require manual byte conversion or header construction, and HTTP
core still knows nothing about JSON.** Full record:
`STARKLANG/docs/http-client/HC11-JSON-EVIDENCE.md`.

```text
stark-http-core     TextDecodeError, decode_utf8, HttpResponse::body_text
stark-http-client   RequestBuilder::json, HttpResponse::json / json_checked, JsonBodyError
```

The split is forced: `stark-http-core` must not depend on `stark-json`, or everything that parses a
header pulls in a JSON parser. `body_text` lives in core because `HttpResponse` is declared there.

### The substantial part was a UTF-8 decoder

There is no `String::from_utf8` in the core surface, so HC11 wrote one. The accepted set is explicit
RANGES, not "leading byte then N continuations", because the short form accepts three things it must
not: overlong forms (`C0 80` is NUL in two bytes, invisible to a checker scanning decoded text),
surrogates (`ED A0 80`–`ED BF BF`), and anything above `U+10FFFF`. Each is a documented
parser-differential bug class — two components disagreeing about what a byte string means is how a
filter gets bypassed.

Strict also means **no replacement characters**. An invalid sequence is an error carrying the byte
offset; substituting `U+FFFD` would hand a caller a body that differs from what the server sent,
undetectably.

**The gap was found twice, independently.** Before HC11 there was no `body_text`, and two people —
the author of the first consumer and an outside reviewer writing their own client — each looked for
the obvious method, did not find it, and copied the same manual
`Char::from_u32(body[i] as UInt32)` loop out of an existing consumer. That loop is Latin-1: it
treats each byte as a code point, so `é` returns as two garbage characters. Fine for ASCII, silently
wrong otherwise. Two people reaching the same wrong idiom is the argument the helper had to exist.

### Independent corroboration of HC10, recorded but NOT gate evidence

An outside reviewer built their own client at the HC10 HEAD and ran it against real hosts:
`GET https://api.github.com/rate_limit` returned 200 over TLS validated against the **system** trust
store, headers reaching the server and response headers parsed back. That covers the one direction
the offline tests cannot — `SystemRoots` is tested here NEGATIVELY, since the fixture CA is in no
machine's store. HC13 forbids qualification depending on internet services, so it stays
corroboration.

### A coherence hazard, noted not exploited further

STARK permits an inherent `impl` on a FOREIGN type — verified, and that is how `HttpResponse::json`
is declared from the client package. Rust forbids it. Nothing stops two packages adding a `json`
method to the same foreign type and colliding. Harmless today, and it is what let the roadmap's
frozen call shape be matched exactly, but it is a real gap in the orphan rule.

### DEV-158 hit a SECOND time, and that is the finding

`RequestBuilder::json` did `out.body = body` — assigning over a `Vec<UInt8>`, a drop unit. Green
under the interpreter, aborted natively. Same workaround: build the struct as a literal from moved
fields.

**Two workarounds for one defect in one work package, both caught only by a native run.** The
three-engine divergence means the cheap engine cannot be trusted to find it, and every future
package writing `x.field = <owned value>` is exposed. This is the argument for prioritising the fix
recorded in CD-366.

### DEV-159 — a native build can race its own dependency build

Reported by the same outside reviewer: a first native build of an HTTPS program FAILED and succeeded
on retry, the generated crate having raced its `aws-lc-rs` dependency build. A user hitting this
sees a confusing failure. At minimum the diagnostic should say to retry; better, the build should
not race.

### Evidence

29 `stark-http-core` tests (10 new, the decoder's), 29 `stark-http-client` tests (8 new), and a
twelfth consumer case that encodes a value containing a four-byte scalar, POSTs it over verified
TLS, and re-encodes what comes back. Comparing RE-ENCODED values rather than destructuring is
deliberate: it exercises decode and encode together, so a decoder and an encoder wrong in the same
direction cannot agree their way past it.

## CD-366 — HC10 CLOSED: HTTPS from the URL alone; DEV-158 found (2026-08-03)

**`Client::send` now selects HTTP or HTTPS from the scheme, and there is no other way to ask.** No
per-request TLS switch, no insecure flag, no route to `https://` without certificate and hostname
verification. Full record: `STARKLANG/docs/http-client/HC10-HTTPS-EVIDENCE.md`.

`SystemRoots` is implemented (`rustls-native-certs` 0.8.2) and is `default_config()`'s policy —
CD-361's point delivered: the platform's trust anchors WITHOUT handing the protocol to a platform
TLS stack. `BundledRoots` stays refused; vendoring a CA list is a distribution decision nobody has
taken, and falling back to the system store would give a caller the opposite of what they asked for.

### DEV-158 — assigning over a drop-unit field aborts natively (OPEN)

```stark
enum Policy { None, Explicit(String) }
struct Config { policy: Policy, tag: UInt32 }

fn with_roots(pem: String) -> Config {
    let mut config = base();               // base() yields Policy::None
    config.policy = Policy::Explicit(pem);
    config                                 // aborts here
}
```

```text
generated-code invariant violated: mutable access to a dead slot: the slot is PARTIAL
```

**Cause — and note this is NOT `drop_field_with`, which was the first guess.** `lower_overwriting_assign`
(`mir/lower.rs`) implements CD-012's rule that the new value installs *before* the old is destroyed:

```text
1. save each covered drop unit into a temp   Assign(tmp, Move(unit_place))   <- slot -> PARTIAL
2. install the new value                     Assign(place, rhs)
3. drop the saved temps, flag-guarded
4. set the covered units' drop flags true
```

Step 1's move-out is what marks the slot `Partial`, via `move_field`. Step 2 writes the field back.
But **no operation returns a slot from `Partial` to `Whole`**: the API has `write`, `reinit`,
`take`, `drop_value`, `move_field`, `drop_field_with` and `finish_partial`, and the last goes to
`Dead`. The slot stays `Partial`, and the next whole-struct use hits the guard.

**Why it is not a one-line fix.** A slot may return to `Whole` only when EVERY drop unit is live.
Writing back the unit this assignment covers does not establish that: a SIBLING unit may have been
moved out earlier, in which case the slot is legitimately still partial. Per-unit liveness lives in
MIR's drop flags rather than in the slot, and `slot.rs`'s own docs record the owner review that
caught those two being conflated — the three-state design is that repair. A naive `restore_whole()`
reintroduces exactly the unsoundness it exists to prevent.

**The candidate fix, for whoever takes this.** MIR already holds per-unit liveness as ordinary
locals, so the backend CAN see it: after step 4, emit a `mark_whole()` guarded by the conjunction of
all of the local's drop flags. That is sound on MIR's own record of liveness rather than on a guess,
and it needs no cross-block analysis — the whole sequence is emitted by one function. What it does
need is a new runtime operation, emission for it, tests, and a soundness review. That is a compiler
work package, not an HC10 edit, which is why it is filed rather than patched.

**Bisected to one shape:**

| | native |
| --- | --- |
| assign over a NON-drop field (`config.tag = 9u32`) | fine |
| build the whole struct as ONE literal | fine |
| assign over a DROP-UNIT field, then use the struct | **aborts** |

**The worst property: the interpreter accepts the same program.** `stark test` and `stark run` are
green and only the native build fails, at runtime, as an abort. Any package writing
`config.field = <owned value>` over a pre-existing struct is exposed.

HC10's workaround is one struct literal instead of a field assignment — same semantics, same API,
recorded inline. Remove it when this closes.

### A language question, raised not resolved

Core v1 has no mutable binding of an enum payload in a pattern. So
`enum Transport { Plain(TcpStream), Secure(TlsStream) }` cannot carry a `&mut self` method —
`E0400 mutable method receiver requires a mutable place` — and with no trait objects and no
closures either, the plain and secure request flows are written out TWICE. That is a **language**
decision for the owner, not a defect, and the duplication is deliberate and commented rather than
hidden behind something that looks abstract and is not.

### Two process notes

**An experiment run only under the interpreter proves only the interpreter.** The enum-payload move
that DEV-158 eventually broke on was validated with `stark run` early in HC9 and never natively,
which is why it surfaced three layers later in an HTTPS build rather than in a 25-line probe.

**A stale copy of the harness cost real time.** Several minutes of chasing a hostname mismatch ended
at a `/tmp` snapshot of `qualify-first-party-packages.py` taken before the fixture change. The code
under test had been correct the whole time. Regenerate the filtered copy, or run the real script.

## CD-365 — HC9 CLOSED: verified TLS, and CD-360's rule found in a fourth place (2026-08-03)

**A STARK program can now establish a verified TLS 1.2/1.3 stream over a `stark-net` TCP connection
and release both layers exactly once, without touching a raw ABI symbol.** rustls 0.23.43 over
aws-lc-rs 1.17.3, Profile N, exactly the versions CD-361 observed.

Full record: `STARKLANG/docs/http-client/HC9-TLS-EVIDENCE.md`.

### CD-360's rule had a FOURTH site, and it was the verifier

CD-360 found the transfer-ownership rule implemented in three places and fixed each separately. The
MIR verifier was a fourth. It stayed hidden because CD-360's fixture built its
`ValidatedProviderCall` by hand and emitted from it — never running the verifier over a transfer.
HC9's first native build:

```text
MIR-0005 stark_tls::connect bb53: call argument:
  expected HostResource(… provider: "stark-std-tls", resource: "tcp_stream"),
  found    HostResource(… provider: "stark-std-net", resource: "tcp_stream")
```

**The planner was right and the verifier was wrong** — a correct program refused by the compiler,
which is the worse of the two ways to be inconsistent. The rule now lives in ONE function,
`mir::provider_sig::owner_of`, which both callers use. A fifth site cannot restate it slightly
differently, and a test asserts the planner's actual type and the verifier's expected type are the
same value rather than each being separately plausible.

**The lesson is about the fixture, not the code.** A hand-built `ValidatedProviderCall` skips every
stage between planning and emission. Three sites were fixed, the ruling was recorded as implemented,
and the first real caller found the fourth immediately.

### A package can now NAME another package's resource

The gap CD-360 did not reach: the derived signature for `stark_tls_stream_connect` takes a
`TcpStream`, which is `stark-net`'s nominal, so derivation failed with
`UnboundResourceInSignature`. A transfer was declarable in a *provider* manifest and not in a
*package* one.

```json
"foreign_resources": { "tcp_stream": { "package": "stark_net", "nominal": "TcpStream" } }
```

Resolves to `stark_net::TcpStream` and **synthesizes nothing**. Binding it as an ordinary resource
instead would generate a SECOND `enum TcpStream {}` — a distinct `ItemId`, the same spelling, and a
handle the program could not pass anywhere. Inferring the owner from the graph would make a typo
resolve to nothing far from its cause. So it is declared, names the owner, and is refused if the
alias is not a dependency, if the resource is also owned, or if it is a Core type.

### How the socket physically crosses

CD-360 conveyed ownership but not the object: a `RawResourceHandle` indexes the OWNER's private
table. `stark_provider_abi::RawOsHandle` now documents a detach convention —
`stark_<resource>_detach(handle, *mut RawOsHandle)` — resolved **by the linker**, since every
provider is statically linked into one binary. No Cargo edge, no path assumption, and deliberately
NOT in the provider manifest: a manifest describes the STARK-callable surface, and `detach` is
callable by no package and emitted by no lowering.

**Open, recorded rather than rediscovered:** a missing detach symbol is a LINK error naming a
symbol, not a compiler diagnostic.

### The ordering inside `connect` is the cleanup story

```text
detach the socket FIRST  ->  validate  ->  handshake
```

The handle is consumed whatever the function returns, so any early return before the socket is
adopted strands it in the net provider's table. Detaching first makes every later error path a plain
Rust drop. There is no cleanup code in that function, and its absence is the design.

### Evidence

19 provider tests (the full certificate matrix, both protocol versions distinguishable, handshake
deadline, peer-close, fragmented records, leak-freedom on every failure path), 16 new compiler tests,
8 package tests, and `stark-tls` as the **16th package** in the qualification gate — declared surface
14 callables, all called. All provider-related starkc suites re-run green.

**CD-360's runtime proving case is closed by this**: a real transfer against a live peer, both
outcomes, release observed exactly once.

### DEV-156 — `stark fmt` evicts member doc comments (OPEN)

A doc comment on a struct FIELD is relocated to after the struct; one on an `impl` METHOD is
relocated INSIDE the body. Reproducer:

```stark
pub struct Config {
    pub first: UInt32,
    /// PROBE DOC
    pub last: UInt32,
}
```

becomes `pub struct Config { pub first: UInt32, pub last: UInt32 }` followed by a dangling
`/// PROBE DOC`.

Cause: `printer::field_def` never consumes leading comments, so they survive only via
`CommentStream::take_rest`'s no-loss net, which flushes at the next position the printer does
attach. `item_seq` calls `emit_leading_comments` correctly, which is why top-level items are fine.
Fixing it needs `measure_flat` to snapshot the comment cursor, a member comment to force the
multi-line branch, and per-member emission in that branch.

**Both forms are idempotent after one pass, so `fmt --check` passes and the gate never noticed.**
`stark-net` has its method commentary inside method bodies — almost certainly this defect, absorbed
rather than reported.

Not fixed under HC9: it changes canonical form repo-wide, so every affected package must be
reformatted in the same commit, and this checkout is shared. `stark-tls` uses the surviving
placement with an inline note pointing at this entry.

**On reducing it:** three attempts reported "PRESERVED" falsely, because the baseline copy was kept
INSIDE the package directory and `stark fmt` formats every `.stark` file in a package — mangling the
baseline identically and emptying the diff. A formatter reducer must keep its baseline outside.

### Two other findings

* **DEV-157** — the native backend has no representation for `MirTy::Never`, so
  `Err(_) => panic(..)` in match-arm VALUE position checks and then fails to build. Known C5.3 gap;
  `stark-tls-consumer` nests instead, as `stark-net-resource-consumer` already does.
* `c788_resource_lifecycle::build_driver_selects_closes_for_bound_resource_nominals` fails in this
  checkout with "Cargo succeeded but the expected binary is missing". **Verified pre-existing on
  HEAD** by stashing every HC9 change. Environmental, tied to the shared `target/`.

### Not claimed

`SystemRoots`/`BundledRoots` are declared and REFUSED — HC10's, and refused by name rather than
silently substituted. Profile F is not qualified: it needs CMake and Go, neither present. HTTPS is
HC10.

## CD-364 — `crate_location` deleted; P0.2 complete (2026-08-03)

**The last piece of the mechanism that made every native capability a compiler-source change is
gone.** A provider's crate location now comes from its manifest, resolved against a root the caller
supplies — the compiler's own root for a built-in, the manifest's directory for an external one.

```rust
// before: a hardcoded match over five names
crate_location("stark-net-native", repo_root) -> repo_root/stark-net/native

// after: the manifest says
repo_root.join(&provider.crate_path)
```

`crate_path` is constrained at parse time to be relative and free of `..`, so the join cannot escape
the root. For an external provider that root is the only containment there is.

**`built_in_crate_location` is not `crate_location` returning under a new name.** It is a lookup
OVER the manifests, so the path data still lives in exactly one place and the function cannot
disagree with it. Adding a provider means adding a manifest and nothing else. Built-in only, by
design: an external provider's root comes from the application's declaration, which is what makes it
containable.

### P0.2 exit criteria

| | |
| --- | --- |
| a provider supplied outside the compiler repo is discovered, validated, linked | DONE |
| `first_party()` expressed the same way an external provider is | DONE (CD-362) |
| ABI mismatch, unsupported target, duplicate capability, missing checksum each refused by name | DONE |
| a provider not enabled in the application manifest cannot be activated by a dependency | DONE |
| release builds record provider hashes | `AdmittedProvider` carries identity, version and hash; **wiring into build metadata remains** |

### Verification

`cargo clippy --all-targets` clean (0 warnings), `cargo fmt --check` clean, the four P0.2/CD-360
suites green (51 tests), and the 15-package gate green — including native builds and live-peer
resource lifecycles, which is the evidence that matters, since it exercises the new location path
end to end.

### Two process notes worth keeping

**Clippy earned its place three times in this stretch alone** — `derivable_impls`, then two rounds of
`crate_location` callers that `cargo build` and targeted `cargo test` never compiled. `--all-targets`
is the only local command that compiles what CI compiles.

**Twice I let a partial signal stand in for a complete one.** A fix loop that grepped for ONE error
kind reported "no more sites" when the build had failed for a different reason; and I chased
`crate_location` callers one clippy run at a time — three four-minute runs to find five callers that
`grep -rn` listed in one second. The compiler's output is deliberately truncated; the tree is not.
Ask the source directly.

## CD-363 — P0.2 external provider discovery, trust tiers, and `crate_path` containment (2026-08-03)

### The crate-location ruling

> **A provider's manifest declares its crate path, resolved against a root the caller supplies —
> the compiler's own root for a built-in, the manifest's directory for an external one.**
> `crate_location()` is deleted.

One RULE, two roots. The alternative considered and rejected was keeping a layout convention for
built-ins: that is the `first_party()` shape again — a hardcoded path surviving beside a declared
one — merely moved rather than removed. The root differs by how the provider was ADMITTED, which is
already a first-class distinction (`ProviderTrust`), so it is a visible parameter rather than a
hidden special case.

Neutral on an existing fragility, not worse: built-in `crate_path` values are repo-layout relative,
which is exactly what `crate_location`'s match arms already assumed. Moving it from Rust to JSON
makes it visible and fixable without a compiler change — worth something given a stale install
layout has dropped a provider before.

### `crate_path` containment — a gap found while implementing, not while designing

Nothing in the ruling as chosen constrained `crate_path` to be relative. An external manifest is
written by a third party BY DEFINITION, and `"crate_path": "/etc"` or `"../../elsewhere"` would
escape the root it was admitted under — **the only containment this mechanism has.**

Now refused, and stricter than the obvious form:

* enforced at BOTH the parse and the resolution entry point, so neither is a route around the other;
* checked on the STRING, not the joined path — `provider/../../elsewhere` normalises into something
  that looks contained, so canonicalising first is how the check gets defeated, and a symlink beats
  post-hoc canonicalisation anyway. Refusing the components does not depend on the filesystem's
  cooperation;
* Windows drive prefixes refused on every host, since a manifest may have been written elsewhere.

### Trust is explicit, not enforced

```text
pure STARK package             no native code, no provider
first-party native provider    ships with the compiler, versioned with it
approved third-party provider  declared by the APPLICATION, pinned by version AND checksum
untrusted / local provider     path-based, development only, never in a release build
```

**No sandboxing is attempted** — a partial isolation story invites misplaced confidence, whereas a
visible tier is honest and achievable now. What the mechanism guarantees is that native third-party
code cannot enter a build BY ACCIDENT: every route in is deliberate, recorded, pinned and refusable.

Four properties, all refusal-tested:

1. **off by default** — declaring a provider is not enough;
2. **no transitive activation** — only the application may activate one. A library must not pull
   native code into a program that never asked for it, which is the difference between a dependency
   graph and an attack surface;
3. **pinned exactly** — version and checksum both, or the provider on disk is not the provider that
   was approved. Both hashes are reported so the reader can tell which artefact moved;
4. **development trust does not survive release** — an unpinned path provider works while developing
   and is refused in a release build.

Every failing provider is reported, not just the first: an application pinning three wrongly should
learn all three in one build.

### Evidence

32 tests across `p02_provider_manifest.rs` (11) and `p02_external_provider_trust.rs` (21). The
15-package gate is green through the manifest path, including native builds and live-peer resource
lifecycles.

### Still open in P0.2

Wiring discovery into `native_build.rs` and deleting `crate_location`, which has four real callers.
Deliberately not sprinted: that path produced two red CI runs this session, and it is the wrong
place for blind edits. The discovery surface exists and is tested; the old path still works; nothing
is half-rewired.

## CD-361 — joint HC9/CRYPTO0 decision: rustls + aws-lc-rs (2026-08-03)

> **Select `rustls` with `aws-lc-rs` as STARK's TLS and general native-cryptography foundation.
> Reject `native-tls` for the first-party TLS provider.**

Recorded in `WP-CRYPTO0-TLS-BACKEND.md` — which also CREATES the CRYPTO0 record, since none
existed. HC9's roadmap section is updated at source: **backend selection is no longer part of the
HC9 estimate.**

### Why not native-tls

It is not one TLS implementation — SChannel, Secure Transport and OpenSSL by platform. That would
give STARK three error surfaces, three certificate behaviours, three security policies and three
FIPS stories, and it is directly contrary to what this track has spent its effort on: one rule every
engine satisfies by construction. It also multiplies CD-347/348's obligation, since lifecycle
evidence would be needed per platform stack rather than once. Permitted later as an external
provider under WP-EXTERNAL-PROVIDERS; not as the first-party implementation.

### The sharpest point in the ruling

```text
trust-anchor source  ≠  TLS implementation
```

System roots can be used without handing the protocol to a platform stack. That defuses the only
strong argument for native-tls, and it mirrors a separation this codebase already makes —
`crate_location`'s doc: a crate's path is a property of the checkout, its name a property of the
program. HC9's fixture uses `ExplicitRoots` with a test CA; `SystemRoots` is HC10's concern.

### Verified before freezing, not carried over

The external claims were fetched and checked rather than transcribed. **The ruling held up**, with
two refinements:

| claim | result |
| --- | --- |
| FIPS 140-3 certificate **#4816**, AWS-LC-backed | confirmed exactly |
| rustls 0.23.42 | documentation had already moved to **0.23.43** |
| aws-lc-rs 1.17.x | confirmed, 1.17.3, released 2026-07-17 |
| normal build needs a C/C++ compiler; FIPS adds CMake and Go | confirmed — CMake/Go/bindgen are *never* needed for Profile N |
| a Cargo feature alone is not a FIPS claim | confirmed, and **more specific** than stated |

The version drifting between the ruling and the check, within one day, is itself the argument for
the pin-exactly policy. Recorded as versions OBSERVED; the pin comes from HC9's qualification
output, because you pin what you qualified.

**Profile F is a two-step activation, not a flag:** install `default_fips_provider().install_default()`
and verify `ClientConfig::fips()` at runtime. Both are checkable, so they belong in Profile F's
qualification criteria rather than in prose.

**A correction to my own objection:** I had called the build cost understated. Verification showed
the ruling's split was accurate — Profile N needs only a C/C++ compiler. The residual point stands
but is smaller: providers link statically into the generated workspace, so that compiler is required
of every user building a TLS program, not only of the provider's authors. Recorded as a named cost.

### Two things recorded so they are not rediscovered

* A provider manifest's `targets` field declares triples but **cannot express toolchain
  prerequisites**, so a provider may declare a target it cannot build on without extra tooling.
  Belongs to WP-EXTERNAL-PROVIDERS.
* `stark-http-client::parse_http_url` refuses `https://` outright today, deliberately. **HC10 turns
  that refusal into scheme dispatch** — the visible edge of this decision in already-shipped code.

## CD-360 — cross-provider transfer ruled and implemented; P0.1 closed (2026-08-03)

**Ruling, from the language owner:**

> A cross-provider `HandleConsumed` transfer consumes the source handle regardless of whether the
> provider operation succeeds or fails. Failure does not restore the source resource. The consuming
> provider is responsible for releasing any underlying native resource when it fails before
> producing the destination handle.

`HandleConsumed<T>` therefore keeps the meaning it has always had — ownership leaves the caller
unconditionally — which is precisely why this needed **no change to drop elaboration**, no
branch-dependent move state, and no place live on one result arm and dead on another. Option B
would have required conditional move restoration across provider boundaries; that is ownership
machinery, and it is not justified by making failed handshakes recoverable. It remains available as
a future extension.

Recorded in `native-provider-abi-v0.1-CD360-amendment-2.md`.

### Three enforcement sites, not one

The packet predicted a validator amendment. Implementation found the rule enforced in **three**
places, and only reading the first two would have shipped a P0 that could not lower:

| site | what it checked | change |
| --- | --- | --- |
| `provider_abi::validate` | a provider may only name resource types it declares | foreign types nameable in `HandleConsumed` position only, carrying no close obligation |
| `ProviderSet::select` | (nothing — could not see across providers) | a foreign consumption resolves to EXACTLY ONE owner, and to the owner the consumer named |
| `provider_bind` planner | handle type id and MirTy derived from the CALLING provider | for a transferred handle both come from the OWNER |

The third is the one that mattered. `mir/lower.rs`'s `HandleConsumed` arm already carried a comment
stating CD-360's rule verbatim — written for A11 §8, long before the question was asked — so the
move semantics and drop behaviour genuinely were already correct. **But the call could not be
planned at all**: `UndeclaredResourceType`. Nothing had ever lowered a transfer.

A handle carries its OWNER's type id, because it was created with it, and the consuming provider
must present it unchanged. Deriving it from the consumer would hand the provider a tag naming a
different resource. `ValidatedProviderCall` now carries `ForeignResourceCall` for that reason.

### Why the declaration is explicit

`foreign_resources` is declared, not inferred. Treating "any handle type I did not declare" as
foreign would silently accept `HandleConsumed { resource_type: "tcp_strem" }` and defer the typo to
a link failure. Naming the owning provider keeps the check at the three-part identity
`{nominal, provider, resource}` the type system already uses — which is also why
`ForeignResourceOwnerMismatch` exists: a matching resource NAME under a different owner is a
DIFFERENT resource.

### Evidence

19 tests. `cd360_cross_provider_transfer.rs` — 11 declaration rules (2 allowed, 9 refused) and 4
resolution rules; `cd360_transfer_lowering.rs` — 4 lowering assertions on a synthetic net→wrap
transfer, deliberately not TLS, so the proving case does not wait on a certificate chain.

**The fixture earned its keep twice.** It caught the planner refusal, and it caught a bad assertion
of my own: the first double-release check grepped the whole generated file and failed on the
`extern "C"` declaration rather than a call. That form would have passed for the wrong reason had
the code been broken differently — a declaration is not an invocation, and the test now cuts the
extern block before checking the body.

All ten provider suites re-run green (132 tests).

### What P0.1 does NOT include

The **runtime** proving case — a transfer executed against a live peer, both outcomes, release
observed exactly once — remains open and belongs with HC9, since it needs a TLS peer with a
controlled certificate chain. §3 of the amendment (a failing provider must leave no live native
resource) is a provider-author obligation **no compiler check can enforce**; it is recorded so
review can carry it.

**P0.2 (external provider discovery) is now the critical path.**

## CD-359 — HC9 paused; two P0 platform-architecture packets opened (2026-08-03)

**Two items previously carried as backlog are release-architecture blockers, and HC9 must not be
implemented before the first is frozen.** Recorded by the language owner; packets written to scope,
deliberately NOT combined.

### Revised priority

```text
P0  Cross-provider resource-transfer ABI      WP-PROVIDER-HANDLE-TRANSFER.md
P0  External provider discovery/registration  WP-EXTERNAL-PROVIDERS.md
P1  HC9 TLS implementation                    DESIGN-BLOCKED by P0.1
P1  Database provider foundation              blocked by P0.2
P1  HC10 HTTPS                                blocked by HC9
P2  HC11-HC13
```

The two design tracks may run in parallel. **DB0 (STARK-facing value, error, connection, transaction
and cursor contracts) may proceed now** — it is pure STARK and does not prejudge either decision.

### Why HC9 stops

TLS wraps TCP, so the TLS provider must take a `TcpStream` the net provider created. The ABI has no
way to express that, and without a frozen rule an implementation would duplicate ownership, smuggle
raw handles, bypass the validator, fuse TCP and TLS into one provider, or leave Drop authority
unclear. Each weakens the resource model A11/CD-234/CD-237/CD-240 exist to guarantee.

### The scope finding that shrinks P0.1

Probing `provider_abi::validate` established that **most of the transfer contract already exists**:

| already true | consequence |
| --- | --- |
| resource identity is structural over `{nominal, provider, resource}` | provider identity is part of the TYPE; a transfer is a genuine type change |
| `HandleOut` writes its slot only on success | the destination's failure disposition is settled |
| close is selected per resource, and a closeless resource is refused | "which provider releases" is answered structurally |
| every function returns `ProviderStatus`, no direct returns | `Result<HandleOut<TlsStream>, TlsError>` is the shape it already has |

So the packet does not design a mechanism. It authorizes **one referencing rule** — a provider may
name a foreign resource type in `HandleConsumed` position without inheriting its close — and freezes
**one failure rule**.

The two existing refusals are CORRECT and must survive; the new rule sits alongside them:

```text
ResourceTypeMissingClose      declaring a foreign type would give it a second, competing close
HandleResourceTypeUndeclared  a provider may only reference types it declares
```

### The hard question, and the recommendation

What happens to the SOURCE handle when a transfer fails. Three candidates are set out in the packet;
the recommendation is **(A) failure also consumes the source**, because it is the only option
requiring **no change to drop elaboration** — `HandleConsumed` keeps meaning exactly what it means
today, unconditionally consumed. Returning ownership on failure would make ownership depend on a
runtime value, which is precisely the class of conditional invariant this compiler has repeatedly
failed to get right first time. It also states the real-world truth: a failed handshake does not
leave a usable socket.

### Why P0.2 is broader than databases

`first_party()` is a hardcoded `Vec` and `crate_location` a hardcoded `match`. Providers are
compiler-integrated extensions, not an ecosystem mechanism: every native capability needs a compiler
change, nobody outside the repo can publish one, provider versioning is welded to compiler releases,
and trust policy is implicit because we wrote everything that exists. **The public package system is
incomplete for host capabilities.**

The packet keeps static linking and changes only DISCOVERY — manifests instead of hardcoded tables,
with `provider_abi::validate` unchanged and merely fed from a different source. Trust is made
explicit rather than enforced: four tiers, external providers off by default, no transitive
activation, exact version and checksum, no sandboxing attempted.

Its exit criterion is an executable claim:

> Adding PostgreSQL, MongoDB, MySQL or SQL Server requires no compiler-source change.

## CD-358 — the file-provenance audit, and borrow conflicts made place-granular (2026-08-03)

Two items from the post-CD-357 list, plus a CI failure that CD-357 caused and this fixes.

### 1. The provenance audit closed the class by EXERCISE, not by inspection

`self.text(span)` slices the file currently being CHECKED. A name belonging to a DECLARATION —
an impl's generic parameter, a signature's, a trait default's return type — belongs to the file
that declared it. Across a module boundary those differ, and the failure is **silent**: the
comparison succeeds against garbage.

The same bug has now been repaired at six sites across four decisions:

| | site | found by |
| --- | --- | --- |
| DEV-069 | a trait method's name | a trait default across files |
| DEV-101 | cross-package generic typecheck | a package consumer |
| DEV-148 | an associated function's name, then its generic parameters | `stark-url` calling its own `Url::parse` |
| **DEV-155** | a METHOD's impl generics, and a trait default's signature TYPES | **this audit** |

**Inspection was the wrong tool and had already failed four times.** There are ~90 `self.text`
calls in `typecheck.rs`, most legitimately reading the file under check. Classifying them by eye is
exactly the process that missed this repeatedly. A probe that actually compiles two-file packages
found the remaining live site in ONE run:

```text
*w.get() != 11   ->  E0001 expected 'S', found an integer literal
```

`'S'` is `T`'s offset in `lib.stark` landing on an `S` in `inner.stark`.

The repair is a `decl_text` helper that resolves against `foreign_sig_item` when a declaring item is
in scope — a helper rather than a habit, precisely because remembering `item_text` at 29 sites is
what has not worked. `tests/cd358_cross_module_provenance.rs` drives every construct across a module
boundary, so a future site added without it fails there rather than in a package months later.

**The near miss worth recording:** `item_text` returns `"?"` for an out-of-range span, so two
mis-sliced parameter names could COLLIDE on one key and substitute each other's types — a WRONG
program rather than a rejected one. Every failure seen so far was a refusal; that one would not
have been. A two-parameter test pins it.

**Also answered:** associated TYPES resolve correctly across a module boundary — the open question
DEV-148 left behind.

### 2. DEV-154: borrow conflicts compare PLACES

OWN-BORROW-001 has always said "Disjoint field projections do not overlap". Every comparison in
`borrowck` tested `b.local == local`, so a borrow of `p.a` blocked a read of `p.b`. The `Borrow`
record now carries the borrowed place, and every comparison — creation, assignment, move, method
receiver, read — goes through `places_overlap`, field-precise since DEV-135.

**This repair makes the checker accept more, so the refusals are the load-bearing half.** Identity,
parent-over-child, whole-local-over-field, two exclusive borrows of one field, assignment to a
borrowed place, and move-out-of-borrowed-storage all stay refused. The move check is deliberately
stricter than the read check — it rejects under ANY live borrow, shared included, because moving
invalidates storage a live view still points into — and going place-granular did not weaken it.

### 3. CD-357 broke the AST snapshots, and blessing them would have hidden it

CI went red on `tests/snapshots`. Inserting OWN-BORROW-002's example as `03-Type-System__19`
shifted every later fixture by one, and the snapshot cases name fixtures **by number** — so
`__20`, `__31`, `__37`, `__40` silently came to mean different constructs.

`UPDATE_SNAPSHOTS=1` would have gone green while repointing each snapshot at a different construct.
The cases were RENUMBERED to follow their content instead, and the `.ast` files renamed with them —
**the snapshot contents did not change**, which is the proof the mapping is right. A comment on
`CASES` now records that renumbering, not re-blessing, is the correct response.

The extractor's "manifest is in sync" check covers the manifest only; the snapshots are a second
artefact keyed to the same numbering, with no such check. That gap is real and remains open.

**Verification:** 15/15 packages qualify, external sample suite 39/39, and the three new suites
(8 provenance + 10 place-granular + the CD-357 15) are green. Full workspace coverage is CI's.

## CD-357 — DEV-150 ruled: uniform rejection, hoisting required (2026-08-02)

**Ruling (B), from the language owner. Now normative as OWN-BORROW-002 in `03-Type-System.md`:**

> A call may not create an exclusive borrow of a place while another argument in the same call
> reads from or borrows an overlapping place. Such reads must be evaluated into locals before the
> exclusive borrow is created.

```stark
fill(&mut buffer, buffer.len());   // rejected
let count = buffer.len();          // hoist
fill(&mut buffer, count);          // accepted
```

Uniform in the base — a local, a place reached through `&mut`, a field projection, an index, a free
function or a method receiver — and independent of argument order. **Core v1 therefore does not
define argument evaluation as providing two-phase borrow semantics**, and says so; adopting them
stays reserved and this ruling stays reversible.

Chosen over blessing the accepted case because that would have required accepting the LOCAL case
too — widening the borrow rule into two-phase borrows, with evaluation-order machinery and a real
semantics commitment. (B) keeps one backend-neutral rule every engine satisfies by construction.

### What had to change

The rule already existed and already fired for a local base. It stopped one indirection away:
passing a `&mut`-typed place REBORROWS, which registers no active borrow, so the read that followed
saw nothing to conflict with.

`check_argument_overlap` now runs as its own pass over the whole argument list — **a method
receiver included**, since `v.push(v.len())` is the same conflict as `push(&mut v, len(&v))` —
BEFORE the left-to-right walk. It has to be a separate pass: a check that falls out of the walk can
only ever catch the borrow-first order, and the ruling is order-independent. `exclusive_borrow_of`
treats an explicit `&mut place` and a `&mut`-typed place alike, which is the whole repair. A
report-once set keeps one mistake to one diagnostic, rather than the new check and the old one both
reporting the same read in different words.

### Livability, checked rather than assumed

**All 15 first-party packages pass the gate under the rule with zero new diagnostics.** The only
site that ever hit it was `stark-http-parser`'s four `take_line` calls, hoisted when the defect was
first found. A rule that had broken working code across the tree would have been the wrong rule to
implement without saying so.

### Engine agreement is by construction

The front end rejects, so nothing reaches the HIR oracle or MIR. `check`, `run` and `build` all
refuse the previously-accepted program with the same diagnostic — which is the point: the old
behaviour was accepted by the checker, executed correctly by the oracle, and refused by rustc.

### Evidence

`tests/dev150_argument_overlap.rs` — 15 tests, negatives varying the base and the order, positives
for every hoisted and non-overlapping form (different locals, literals, successive borrows,
successive reborrows of a parameter, two shared reads). Plus spec fixture
`03-Type-System__19.stark`, classified `semantic-error` with `errors = "E0101"`, so **the spec's own
example is an executable test of the rule.**

Supersedes `dev150_argument_conflict_through_reference.rs`, which pinned the INCONSISTENCY while the
ruling was open. Its own doc required it to be rewritten around whichever ruling landed, and both of
its "the two bases disagree" tests went red the moment they agreed — the mechanism working as
designed, twice in two commits now.

### One defect uncovered on the way: DEV-154

CD-357's overlap check is place-granular and correctly declined to fire on `f(&mut p.a, p.b)`. The
OLDER `check_read_borrow_conflict` then reported it anyway, because it compares only the LOCAL and
ignores projections — so **disjoint field projections over-reject, contradicting OWN-BORROW-001's
"Disjoint field projections do not overlap".** Pre-existing; visible only because two checks in the
same area now disagree about granularity. Filed OPEN and deliberately NOT bundled here: loosening a
borrow check is its own change with its own negative controls, and must not ride along with a ruling
that tightens one.

## CD-356 — DEV-148 CLOSED: the name was sliced out of the wrong file (2026-08-02)

**Filed as a language limitation about associated functions. It was a text bug, and the gate built
one commit earlier is what forced it into the open.**

`Wrap::make(2)` from a submodule of its own package failed with "associated function 'make' not
found". Path resolution was correct — it reached `Res::AssociatedFn`. `typecheck`'s lookup then
compared member names with `self.text(span)`, which slices **the file currently being checked**,
while a member's name span belongs to the file that declared the `impl`. Instrumented, `impl Wrap`'s
two members read back as:

```text
member name_text="rap:"  has_receiver=false     // `make`'s offsets applied to the other file
member name_text="?"     has_receiver=true      // a span running past the shorter file's end
```

No candidate could ever match. **Methods were unaffected because method lookup selects on the
receiver's TYPE rather than by slicing a name** — and that asymmetry is the whole reason this looked
like a rule about associated functions instead of a bug about files.

### A second site, one layer down

Fixing the comparison made plain associated functions work and immediately exposed the same defect
in generics:

```text
error: [E0500] type 'r' does not satisfy operator trait 'Eq'
```

`'r'` is `T` sliced from the wrong file. The substitution map's keys and the `Ty::Param`s they
substitute into must be read from the SAME file or substitution silently fails to fire, so
`foreign_sig_item` now carries the declaring item across the whole signature conversion. Note also
that `item_text` yields `"?"` for an out-of-range span, so several mis-sliced parameter names could
COLLIDE on one key and substitute each other's types; a two-parameter test pins that they cannot.

### The rule was already written down twice

DEV-069 fixed exactly this for trait methods — "the trait's method names belong to the TRAIT's
declaring file" — and `build_assoc_projections` converts "against the impl's own file". This site
simply missed it. The general statement, worth keeping where someone will read it: **`self.text` is
correct only for spans from the file under check; every lookup that reads a name off a foreign
declaration needs `item_text`.** Worth auditing the remaining `self.text` call sites against that.

### What closing it unblocked, and what it then found

The three items CD-355 recorded as `surface_blocked` became callable, and **the gate refused its own
stale records** — the self-cleaning rule firing for real rather than in principle:

```text
stark-url: these are recorded as blocked, but are now called:
      Url::parse
```

With the records removed and the three exercised, **all 15 packages qualify with zero blocked
items: every public callable in the tree is now called by its package's own tests or consumers.**

**And calling `TcpStream::connect` for the first time found a third dead API.** It refuses EVERY
non-zero timeout with `Unsupported`:

```stark
pub fn connect(address: SocketAddress, timeout: Duration) -> Result<TcpStream, NetworkError> {
    if !timeout.is_zero() { return Err(NetworkError::Unsupported); }
    connect_socket_address(&address)
}
```

So the natural connect API — the one that takes a deadline — has never connected to anything. It
succeeds only for a ZERO duration, which reads as "no timeout" and is the opposite of what passing a
`Duration` means. There is no connect-timeout in the provider ABI to implement it against: the
declaration ran ahead of the capability. Pinned in the consumer as a required failure, so landing a
real connect timeout forces the assertion to change.

That makes **three dead APIs in `stark-net` found by calling things nothing had called** —
`shutdown_write` (permanent stub), `connect`-with-timeout (unsupported for every meaningful
argument), and the timeout setters (unbuildable at a call site, DEV-151). All three concern
timeouts or lifecycle, and all three were invisible for the same reason.

**Evidence:** `tests/dev148_associated_fn_across_modules.rs` — 7 tests over a real two-file package
graph, because a single-file fixture cannot reproduce a provenance bug. Vacuity-checked by reverting
the repair: the three cross-boundary positives go RED, all four controls stay green.

## CD-355 — the gate now requires that a package's declared surface is CALLED (2026-08-02)

**The gap this closes has cost three separate stretches, each time closing the instance and leaving
the class open:**

| | what happened | what was fixed |
| --- | --- | --- |
| CD-345 | `stark-net` passed all seven steps while `connect`/`read`/`write`/`close` had never been called, hiding a build-breaking defect (DEV-146) | that package's consumer |
| CD-347/348 | resource LIFECYCLES made executable, against a live peer | the resource category |
| CD-354 (DEV-151) | the same failure one level in: `set_read_timeout` was declared under CD-346, qualified, documented and **unbuildable at every call site**, because nothing had ever called it | that one method |

Each round fixed an instance. **The class is: the gate proves a package builds and its consumer
runs; it never proved that what a package DECLARES is reached by anything.**

### The check

`qualify-first-party-packages.py` gains a step: every public callable must be CALLED by the
package's own tests or its own consumers. The declared surface comes from `stark doc` — the
compiler's own AST walk — not a regex over `pub fn`, so it cannot drift from the source.

**The bar is the package's OWN evidence**, not "called by something, anywhere in the tree". A
downstream caller can be deleted, and proves nothing about the package in isolation.

**Matching is textual and deliberately biased toward FALSE PASSES.** Comments are stripped first, so
prose never counts as a call; but an alias or a generic dispatch can credit a call that does not
happen. That bias is chosen: a false FAILURE would push someone to add a fake call to satisfy the
gate, which is worse than a missed one, because it teaches that gate output is noise.

### What it found immediately

**12 uncalled public callables across 3 of 15 packages** — and the concentration is the finding:

- `stark-net`: **all seven** `impl TcpStream` methods. The entire method surface was dead
  end-to-end; every consumer used the free functions instead. DEV-151 was one instance of a block
  that had never been called at all. Now exercised by the native resource consumer against the echo
  peer, including the DEV-151 reproducer as a real call site.
- `stark-mime`: four `MediaType` methods, wrapping free functions the tests already covered. A
  wrapper no test calls is not a thinner API — it is a second implementation nobody has run.
- `stark-url`: `Url::parse`.

Also surfaced: **`shutdown_write` is a stub** that always returns `Unsupported`. Calling it is what
made that visible. The consumer now asserts it fails, so implementing it forces the assertion to be
updated rather than letting a permanently-broken promise sit in the surface.

### Blocked items are counted, not waived

Three of the twelve are ASSOCIATED functions and cannot be called at all — DEV-148. They are
recorded per package with the defect that blocks them, and **the gate refuses a record whose item
has become callable**. A fix to DEV-148 therefore forces the records out rather than letting them
rot; the same self-cleaning rule as the sample suite's "an unexpected PASS is a failure". The
purpose is to make the cost of an open defect countable instead of invisible.

### Two compiler defects had to be fixed first

- **DEV-152** — `doc_gen::extract` silently DISCARDED the methods of any `impl` whose type had no
  page-level item. A synthesized resource nominal (CD-234) has none, so all seven `stark-net`
  methods were absent from its documentation. A surface gate built on that extractor would have
  certified the package as fully covered. It also explains part of why nobody called them: the docs
  did not say they existed.
- **DEV-153** — `hir_field_ty` had no arm for an unsized slice, so `owned.write_all(input)` refused
  to lower while `write_all(&mut owned, input)` built. This is **DEV-151's second-order cost**:
  opening method dispatch on a resource receiver routed declared parameter types through that
  conversion for the first time, and met a form it had never had to handle. A repair that widens
  what is reachable will expose whatever the newly reachable path never handled — that is the price
  of the DEV-151 class, not an argument against paying it.

### DEV-148's scope was wrong

Filed as cross-PACKAGE; it is cross-MODULE, which is strictly wider. A submodule of the same
package cannot call `Wrap::make` either, and neither can the fully qualified `super::Wrap::make`.
So a package cannot even TEST its own associated functions. The failure is not in the resolver —
the path reaches `Res::AssociatedFn` — but in `typecheck.rs`'s associated-function lookup. Methods
are unaffected because method lookup goes by the receiver's TYPE rather than by path resolution,
which is exactly why the two diverge.

**Status: 15 packages qualify with the surface check enforcing**, 3 items recorded blocked.

## CD-354 — three compiler defects found by qualifying HC7/HC8; one escalated, not repaired (2026-08-02)

**Writing two packages and running them through the gate found three compiler defects and one
semantics question. None was found by a reproducer; every one was found by executing something that
had never been executed.**

| | what | disposition |
| --- | --- | --- |
| DEV-149 | a `&self` method on a `&mut` base is neither weakened nor reborrowed | FIXED |
| DEV-150 | the argument read-conflict rule does not fire through a reference base | **ESCALATED** |
| DEV-151(a) | a method on a host-resource receiver did not lower | FIXED |
| DEV-151(b) | a written-out `()` lowered to `Tuple([])`, not `Unit` | FIXED |

### DEV-149 is my own DEV-147 repair, narrowed on the wrong axis

DEV-147 taught the four `borrow_*_receiver` sites to reborrow rather than move, then gated the
repair on the mutability the METHOD wants. The gate belongs on the mutability the BASE has:

```stark
fn count(v: &mut Vec<UInt8>) -> UInt64 { v.len() }   // check: OK, run: 1, build: REFUSED
```

Two failures from one omission — MIR-0005 (the `&mut` handed over unweakened) and MIR-0007 (the
caller's reference moved). One reborrow fixes both, because `&*base` from a `&mut` base IS the
weakening. The shape it blocked is "measure a caller's buffer, then modify it", which is what
`stark-http-parser::drop_front` does and what surfaced it.

### DEV-151 is CD-345's lesson one level down

CD-345 found `stark-net` passing all seven gate steps while `connect`/`read`/`write`/`close` had
never been called. CD-347 fixed that by requiring a native consumer to exercise each resource's
lifecycle. **This is the same failure one level in: a declared surface whose CALL SITES were still
unexecuted.**

CD-346 ruled that a resource operation moving a cursor or consuming bytes takes `&mut self`.
`stark-net` declared `set_read_timeout`/`set_write_timeout` as methods on that ruling and qualified.
Lowering refused every method call on a host-resource receiver, so **CD-346's ruling was
unbuildable at every call site** — and nothing learned that, because nothing had called one.
`stark-http-client` was the first caller and failed immediately.

The refusal was a missing match arm, not a missing capability: `HostResourceTy.nominal` already
holds the item the `impl` hangs off. Fixing it then exposed (b) — a written `()` reaching the tuple
arm and producing `Tuple([])` where every synthesized site uses `MirTy::Unit`, so
`fn f() -> Result<(), E>` declared a return type no constructed value could match. `Result<(), E>`
is a very common signature; it took two unexecuted paths crossing to make the divergence reachable.
The structural test now asserts no lowered signature or local carries an empty tuple, which catches
a divergence that has not yet MET a conflicting value.

**What this says about the gate.** Both halves of DEV-151 were reachable only by CALLING a declared
surface natively. The seven steps check that a package builds and its consumer runs; they do not
check that everything the package DECLARES is called by something. That is the next gap of the same
family, and it is not closed by this CD.

### DEV-150 is escalated, not repaired

`f(&mut x, x.field)` is refused for a local base and accepted through a `&mut` parameter; the
interpreter runs the accepted form and the native backend emits Rust that rustc refuses (E0503).
Two defensible rulings:

- **(A)** the checker is right and evaluation should be sequenced — close to Rust's two-phase
  borrows, but it requires the LOCAL case to start being accepted too, so it widens the borrow rule;
- **(B)** the checker is wrong and the rule must fire uniformly — conservative and matches the spec
  as written, but `f(buf, buf.len())` stops compiling.

They disagree about whether a real program is sound, so this is a language decision rather than a
repair-commit decision, and it is left OPEN for the language owner. The test suite pins the
INCONSISTENCY rather than either ruling: whichever lands, the test contradicting it fails and the
entry must be revisited. `stark-http-parser`'s four `take_line` call sites were rewritten to hoist
the read, which is required under either ruling.

**Evidence:** `tests/dev149_shared_receiver_over_mutable_base.rs` (13),
`tests/dev150_argument_conflict_through_reference.rs` (4),
`tests/dev151_resource_method_dispatch.rs` (4). All three include negative controls, because each
repair's own risk is handing out access that was never held.

## CD-353 — HC7 and HC8 delivered and qualified; the gate grows an HTTP peer (2026-08-02)

**`stark-http-parser` (HC7) and `stark-http-client` (HC8), both qualified through the seven-step
gate. All 15 first-party packages now qualify, with both resource-bearing ones observed against
live peers.**

### HC7 — the parser, and the exit criterion that shaped its tests

The roadmap's HC7 exit criterion is "the parser can consume any legal fragmentation pattern without
socket knowledge". That is not a claim a few hand-picked splits support, so the suite parses each
message at EVERY two-part split and requires every result to agree — n-1 boundaries per message,
each landing mid-token somewhere different: inside `HTTP/1.1`, between CR and LF, inside a header
name, inside a chunk size. The consumer does the same rather than parsing one buffer, because a
one-buffer consumer would prove nothing this package is for.

34 tests. Ten states, four framings (fixed, chunked, close-delimited, none), 1xx skipping, HEAD
responses, and the rejection half: bare LF, obs-fold, conflicting `Content-Length`, `Content-Length`
with `Transfer-Encoding`, unsupported codings, malformed chunk sizes and terminators, truncated
bodies, and each limit.

**Two real parser defects, both found by the whole-vs-fragmented differential rather than by any
single case:** the OWS-skip after a header colon used an `n + 1` sentinel that destroyed the index
it had just found, so every header value was mis-sliced; and the `UntilClose` transition returned
before the drain arm could run, dropping every close-delimited body that arrived in one buffer.

### HC8 — the client, and what a capability-bearing package can be tested with

Every useful operation in `stark-http-client` requires a provider, and `stark test` runs on the HIR
interpreter, which has no provider layer. So the split is forced, not chosen:

- **`stark test` (14 tests)** covers what is decidable without a socket — URL targeting, config
  budgets, builders.
- **`stark-http-client-consumer`** is native-only and requires a live HTTP peer. It PANICS without
  one rather than reporting success, per CD-348.

Step 5 (`stark run`) is therefore unreachable for this package. The gate now has an
`interpreter_exempt` flag that skips it with a printed reason — and REFUSES to accept the flag
unless the case also declares resources and a resource consumer, so an exempt package is executed
MORE than an ordinary one, never less. Validated in code rather than left to reviewer discipline,
because CD-345 is the record of what an unexecuted step costs.

**Two real client defects, found by the tests:** URL fragments reached the request target (an
information leak — a fragment is client-side only and must never go on the wire), and an empty or
invalid authority was accepted, including `http://h:/` silently defaulting to port 80 rather than
being reported as the typo it is.

### The HTTP peer

`qualify-first-party-packages.py` grows `http_peer()` beside `echo_peer()`, serving four routes that
each pin a response shape the client must handle differently: `/fixed`, `/chunked`, `/fragmented`
(head and body split across several writes with pauses), and `/close-early`. The last two matter
most — they are what a client that assumes one `recv()` per response, or that treats a short body as
complete, gets wrong. Binding is asserted, never skipped.

Observed natively, end to end: resolve, connect, set timeouts, write, read across fragmentation,
decode chunks, detect an early close, release the stream.

### Also recorded

DEV-148 (a cross-package associated function is unresolvable) was found mid-sprint and is OPEN.
`Type::new()` is simply unavailable to a consumer, so every first-party package exposes free
constructors instead — a convention adopted without anyone recording why, which is how a defect
becomes a house style. `stark-time` gained `duration_seconds`/`duration_millis`/`duration_nanos` as
the forced workaround.

## CD-348 — CD-347's claim was stronger than its evidence; the gate now earns it (2026-08-02)

**CD-347 said the gate requires a consumer to "acquire, use and close each resource". It did not.
The checked-in consumer connected to a port expected to REFUSE, so on the path CI actually took:**

```text
connect fails
  -> no TcpStream acquired
  -> write_all never executed
  -> close never executed
  -> drop-release never executed
```

The program compiled and linked every one of those branches, so it was valid evidence that the
source type-checks, provider calls lower, symbols link, the executable starts, and the failure path
runs. **It was not evidence that an acquired resource is used and released.** The honest claim for
that version was:

> every resource-bearing package ships a native consumer that COMPILES AND LINKS its
> acquire/use/release surface and EXECUTES AT LEAST ONE provider path.

Recorded because the gap between that sentence and CD-347's is precisely the gap CD-345 was about:
a claim satisfied by a path that never calls the product.

**The fix is the peer, not a weaker sentence.** `qualify-first-party-packages.py` now starts a
loopback echo listener before running a resource consumer, so the full lifecycle executes:

```text
acquire -> write -> read -> EXPLICIT close      the affine release the package exposes
acquire -> write -> IMPLICIT drop release       MIR drop elaboration emitting the close
```

**It cannot silently degrade.** If the port cannot be bound, qualification FAILS with a message
saying why, rather than falling back to the failure path — falling back would restore the weaker
claim while still reporting success. And the consumer PANICS if the peer is absent: verified, exit
code 101, so a peerless run can never be mistaken for a pass.

### EXECUTED SURFACE, by package category

The standing rule needed this precision, or a future team satisfies it through an expected error:

| Category | Bar |
| --- | --- |
| pure package | the ordinary consumer executes each principal public behaviour |
| function-shaped provider | the native consumer SUCCESSFULLY invokes each capability family |
| resource-shaped provider | the native consumer SUCCESSFULLY acquires, uses and releases every resource type — BOTH release paths |
| failure-only environment | a deterministic negative path is allowed, but must be LABELLED lowering/linking evidence, never lifecycle evidence |

The fourth row is the important one: it keeps the escape hatch open for environments where success
genuinely cannot be arranged, while making it impossible to use one and call the result lifecycle
evidence.

EVIDENCE: hardened eleven-package gate exit 0 with the peer, `STARK_NET_RESOURCE_OK` observed;
the consumer exits 101 with no peer; echo peer refuses to skip on a bind failure; fmt clean.
FILES: starkc/scripts/qualify-first-party-packages.py, stark-net-resource-consumer/src/main.stark,
COMPILER-STATE.md.
NEXT: unchanged — HC3/HC4 and the OPS resource items are unblocked; HC5/HC6 and the pure fills
never were.

## CD-346 / CD-347 — DEV-146 repaired with its ruling; the gate's surface coverage made executable (2026-08-02)

**The two toll items. Resource-track work (HC3/HC4, OPS stdio/signals/process) unblocks on these;
HC5/HC6 and the pure OPS fills never depended on either and should not have waited.**

### CD-346 — DEV-146, and the layer the first diagnosis got wrong

`weaken_ref_to` was never the problem. Its mutability arm is type-agnostic and would have handled
`HostResource` fine. **Provider calls never reached it**: the `HandleBorrowed` arm of
`lower_provider_call` pushed its operand with no expected-type coercion at all.

DEV-133 routed SIX coercion sites through `weaken_ref_to` and its comment warned that "whichever
site was forgotten would keep this defect". Provider calls were the seventh. It stayed invisible
because no first-party package called a resource function until `stark-net` did — the same
blindness CD-345 found in the gate, one layer down.

**THE RULING, which is what outlives the repair:**

```text
AbiParam::HandleBorrowed   always derives a SHARED reference   (ABI fact, unchanged)
package surface            may declare &mut; the compiler weakens
```

The two need not match, so the surface question is answered by SEMANTICS rather than by the ABI:

- an operation that consumes or produces bytes, or moves a cursor, takes `&mut` — a shared borrow
  would let a caller hold two readers of one stream, making byte-consumption order non-local and
  unreviewable;
- a purely observational operation stays `&`;
- neither choice changes what crosses the ABI.

Settled once, here, rather than re-litigated per package: io v0.2 streams, signals, process
handles and crypto keys all face it. **Recorded caveat:** the ruling was made from what the ABI
verifiably does. The CRYPTO0 convergence was NOT in evidence when it was written and should be
checked against it before the first crypto package declares a surface — if CRYPTO0 says something
narrower, this ruling yields to it.

**Negative control, because the risk is weakening the wrong way.** If `&R` could satisfy a `&mut R`
parameter the repair would hand out exclusive access from a shared borrow — an aliasing hole worse
than the defect. Pinned.

`stark-net`'s `&mut` signatures are restored, with the ruling recorded at their definition.

### CD-347 — the gate's executed-surface requirement

A `PackageCase` now declares the resource types it exposes, and a package that declares any must
ship a NATIVE consumer whose run acquires, uses and closes each one. Missing consumer, missing
directory, or a failing run all fail qualification.

**The split is forced, not chosen.** Step 5 is `stark run`, and the interpreter has no provider
layer — any consumer touching a bound resource dies with "provider binding not lowered". So the
resource exercise cannot live in the ordinary consumer, and the gate runs the resource consumer
without a `stark run` step.

`stark-net-resource-consumer` is the first: acquire+close, acquire+use+close (through the `&mut`
path DEV-146 broke, so the package would not have built before CD-346), and acquire-then-let-drop-
release. Deterministic in CI — it needs no peer, because what it proves is that the resource path
LOWERS, LINKS and EXECUTES, which is exactly what was unobserved.

Verified to bite: removing the resource consumer fails the gate with a message naming CD-345.

### The standing rule for the Codex lane

**Definition of done now includes executed surface, stated in each directive.** CD-344's failure
was not Codex writing wrong code — the behaviour was sound, and the end-to-end run proves it. It
was the lane's evidence standard being satisfiable by a consumer that never called the product.
Every future package directive names its required consumer exercises the way the repair packets
named their must-pass sets. One paragraph per directive; the difference between two lanes having
one discipline or two.

EVIDENCE: `dev146_resource_borrow_weakening` 3 cases; `mir_verify`/`mir_lowering`/
`c788_lifecycle_e2e`/`conformance`/`gate3_execution` 87 green; provider and resource suites
(`a10_provider_call`, `a11_host_resource`, `c786_tcp`, `c788_resource_lifecycle`) green; hardened
eleven-package gate exit 0 with `STARK_NET_RESOURCE_OK` observed; end-to-end native TCP client
`wrote / 5 / closed` against a listener that received `b'PING\n'`; fmt clean.
FILES: starkc/src/mir/lower.rs, starkc/tests/dev146_resource_borrow_weakening.rs (new),
starkc/scripts/qualify-first-party-packages.py, stark-net/src/lib.stark,
stark-net-resource-consumer/ (new), starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: HC3/HC4 and the OPS resource items unblock. HC5/HC6 and the pure OPS fills were never blocked.

## CD-345 — HC1 and HC2 qualified with evidence; HC2 was qualified in name only (2026-08-02)

**HC1 (`stark-url`) and HC2 (`stark-net`) landed as plain commits with no CD entry and no evidence
statement. Both are in the eleven-package gate and both pass it. For HC1 that means what it sounds
like. FOR HC2 IT DID NOT.**

### The finding: a happy-path gate cannot qualify a resource-holding package

`stark-net` is the first first-party package that holds host resources. The seven-step gate ran
`check` / `test` / `fmt` on the package, then `check` / `run` / `build` / execute on its consumer —
and **the consumer only formatted addresses**:

```stark
let address = socket_address(ipv4(127u8, 0u8, 0u8, 1u8), 1u16);
if socket_address_text(&address).as_str() != "127.0.0.1:1" { panic(..) }
```

The package's own tests are two cases, both address formatting. So `connect`, `read`, `write`,
`write_all`, `close` and `shutdown_write` — the entire reason the package exists — were qualified
**in name only**. Nothing had ever lowered a call into the raw bindings.

### What that concealed: DEV-146, a build-breaking defect, on develop

The CD-344 signature change (`&TcpStream` -> `&mut TcpStream` on `read`/`write`/`write_all`,
Codex's work, committed by me) makes any program that CALLS those functions fail to build:

```text
MIR-0005 call argument: expected Ref { mutable: false, inner: HostResource(tcp_stream) },
                        found    Ref { mutable: true,  inner: HostResource(tcp_stream) }
```

`weaken_ref_to` does not weaken `&mut T` to `&T` when `T` is a `HostResource`. Accepted by the
front end, refused by MIR verification — the DEV-132/DEV-133 class, third mechanism. Registered as
**DEV-146**; the signatures are reverted to shared borrows with the defect named at their
definition.

**My CD-344 verification was insufficient and I can name how.** I ran `stark check` (front end
only, which accepts) and the package gate (whose consumer never calls the affected functions). I
also checked that no package consumes the changed API — true, and exactly why nothing caught it.
The check I did not run is the one that matters for a resource package: build a program that
actually calls the thing.

### First end-to-end observation of the resource path

Never done before. A native client against a real loopback listener:

```text
client: wrote / 5 / closed          exit 0
server: received b'PING\n' / closed
```

connect, write, read, close all work. The package's behaviour is sound; only the build was broken.

**Drop elaboration verified, not assumed.** `close()`'s comment claims MIR emits
`stark_tcp_stream_close` exactly once for an owned stream. Confirmed in the generated Rust:
`_7.drop_with(|__v| unsafe { stark_tcp_stream_close(__v.as_raw()) })` for a program that never
calls `close`.

**Affine lifecycle negatives, observed for the first time:**

| Shape | Outcome |
| --- | --- |
| double `close` | REFUSED, E0100 |
| use after `close` | REFUSED, E0100 |
| never closed | accepted — drop elaboration emits the close (verified above) |
| closed on one branch only | accepted — same |
| stream stored in a `Vec` | accepted |

### A STRUCTURAL LIMIT OF THE GATE, which is the durable finding

I tried to close the hole by making the consumer exercise the resource path. **It cannot.** Step 5
is `stark run` on the consumer, and the interpreter has no provider layer — any consumer touching a
bound resource dies with "provider binding not lowered". So the seven-step gate is CONSTITUTIONALLY
unable to qualify a resource path, for `stark-net` or any future resource package.

That is not a `stark-net` problem and should not be patched inside one. Resource lifecycle belongs
in a native-only test alongside the existing provider e2e suites (`a10_*`, `c788_lifecycle_e2e`) and
the C7.8 native-capabilities workflow. **Filed as the recommended next step, not done here** — it
is a gate change, and gate changes need their own scope.

### HC1 by contrast

`stark-url` is genuinely qualified: 19 tests, 9 exercising the new absolute-URL surface, and a
consumer that calls `parse_url`/`Url::parse`. Pure parsing, no resources, nothing deferred.

### Correction

Commit messages CD-337 … CD-344 say "all ten packages qualify". It has been **eleven** since
`56a78b4` added `stark-net`, which landed between CD-336 and CD-337. The RUNS covered all eleven
and passed; the descriptions were stale. Corrected here.

EVIDENCE: eleven-package qualification exit 0; `stark-net` check/test/fmt clean; end-to-end native
TCP client against a Python listener; generated-Rust inspection for drop elaboration; five affine
lifecycle probes.
FILES: stark-net/src/lib.stark (signatures reverted, DEV-146 named),
starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-146), COMPILER-STATE.md.
NEXT: a native resource-lifecycle test for `stark-net` before HC3/HC4 consume these APIs, and
DEV-146 before the `&mut` signatures can be restored.

## CD-343 — WP-DEV-134-139 final report; programme complete pending CI (2026-08-02)

**All six CD-334 defects repaired, all infrastructure tasks delivered. §17 report at
`STARKLANG/docs/compiler/work-packages/WP-DEV-134-139-FINAL-REPORT.md`.**

**Recommendation: release, CONDITIONAL on CD-340/341/342 reporting green.** WP §15's gate is
otherwise satisfied, including its DEV-135 branch: the inventory proved parent poisoning
unacceptable, and the precision a DEV-135b would have built already existed.

**The qualification that must not be smoothed over.** CD-337 NEVER WENT GREEN — it failed
`clippy::collapsible_match`, the fix landed in CD-338, and every commit from there is green, so
DEV-136's code is transitively covered. But "aggregate CI green" is a release-gate item, and CI is
the SOLE workspace authority since CD-337 dropped local workspace runs. CD-341 and CD-342 each add
a required job to `ci-complete` and neither has yet been observed passing.

**Root cause of that miss, which matters more than the lint.** The repo pins `channel = "stable"`;
CI's resolves to 1.97.0, this machine's had gone stale at 1.93.0. Every "clippy clean" before
CD-338 was against an older lint set than CI's. Gate is now `cargo +1.97.0 clippy`.

**What the programme actually found.** None of the six needed a design change; four were a single
wrong line or a single missing consultation, and DEV-135 — sized by the work package as "full
field-sensitive move paths" — was one enum variant, because the move model was already
field-precise and only field IDENTITY was broken. Four of six were WIDER than filed, and in every
case the extra half was found by the repair's own must-pass tests rather than by the reproducer.

**Residual, all registered, none from CD-334:** DEV-121 stays open with its blind spot now named
(INV-VALUE-REP-001 checks `let` bindings; a for-loop binding is not a `let`, and both known
instances were loop items). DEV-140…145 registered at CD-342. DEV-083 open. `types_equal`'s
missing `Ty::Param` arm is symptomless and unowned. `?` conversion semantics is a language-design
question with no owner.

FILES: STARKLANG/docs/compiler/work-packages/WP-DEV-134-139-FINAL-REPORT.md, COMPILER-STATE.md.
NEXT: owner review; then the DEV-121 invariant extension is the highest-value unowned item, since
it would close a class rather than another instance.

## CD-342 — the layer audit is an enforcing gate; its six findings are now registered (2026-08-02)

**WP-DEV-134-139 §11. The audit reported and passed unconditionally, so a NEW layer defect could
appear and the suite would stay green — it could only ever be read by a human who happened to
look. It now fails on any UNREGISTERED finding.**

**The bar is not zero findings.** Six reachable lowering refusals exist and are NOT repaired by
this programme. They are now numbered, which is the actual change: CD-331 found and printed them
and they had carried no deviation number since.

| DEV | Probe | Reachable lowering refusal |
| --- | --- | --- |
| DEV-140 | L7153 | `Vec::` method outside the implemented lowering set |
| DEV-141 | L8093 | `HashMap` over a user-`Drop` value type |
| DEV-142 | L9130 | droppable composite carrying a borrowed element |
| DEV-143 | L5346 | `assert_eq` on a user-defined type |
| DEV-144 | L3698 | `for` over a non-range, non-`Vec` iterator |
| DEV-145 | L6450 | method on a peeled type outside the implemented slice |

Every probe now declares the disposition it is expected to have — `FrontEnd`, `Lowers`, or
`KnownDev("DEV-xxx")` — and the test compares actual against registered.

**It fails in BOTH directions, which is the part worth stating.** A registered defect that stops
reproducing fails too, because that means either the DEV was fixed and its registration is stale,
or the probe no longer reaches the construct it was written for. Both need a human decision; both
are invisible to a test that only looks for regressions. The failure was verified by deliberately
mis-registering one probe and confirming the gate reports "registered as Lowers but actually
KnownDev".

**Disposition of the six is unscheduled and per-site, not global.** Two repair shapes exist —
raise the refusal into semantic analysis (E0105) or teach lowering the construct (DEV-132,
DEV-133). CD-294 is the precedent for why raising is not always cheap: E0106 was reverted because
`v[i]` appears in value AND place positions that only later phases distinguish.

Local: `cargo test --test layer_audit` green; negative case verified by mis-registration.
FILES: starkc/tests/layer_audit.rs, starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: reconciliation and the §17 report — the last two programme tasks.

## CD-341 — the external sample suite is published, pinned, and gated in CI (2026-08-02)

**WP-DEV-134-139 §10.1/§10.2. The suite that found all six CD-334 defects is now a repository CI
clones at a fixed SHA, not a directory on one machine.**

```text
repo   navraj007in/stark-samples   (public)
pin    b3b28e757f38d691e7309f168d1209e28ac459af
job    external-sample-suite  ->  required via ci-complete
```

**Kept EXTERNAL on purpose (§10.2).** The fixture corpus and the generated C6 corpus both grow
from this compiler's own model of what programs look like. The sample suite grew from TASKS — sort
a vector, walk a graph, parse an expression — and that independence is why it found six defects the
in-tree suites did not. Absorbing it would reorganise it around compiler subsystems and destroy
the property that makes it useful.

**PINNED BY SHA, not tracking a branch.** §10.3 requires that when a compiler fix changes an
external task's expected outcome, the suite's manifest is updated and the pin moves to the commit
carrying that update, in the same logical change set. A floating `main` would let the suite drift
green or red for reasons unrelated to the commit under test — precisely the confusion the pin
prevents. The job also asserts the resolved HEAD equals the pin, because `ref:` accepts a branch
name and would otherwise float silently.

**It runs the BUILT artifacts**, never `cargo run`, so what is tested is what would ship.

**A machine-readable manifest now exists (§10.1)**, which the suite previously lacked —
`run-all.sh` printed pass/fail and nothing more. `manifest.json` records 39 cases with, per case:
id, description, linked DEV, and the expected outcome for EACH engine (front end, HIR, MIR,
native), using an explicit vocabulary that distinguishes `not_reached` (rejected earlier) from
`not_supported` (the engine lacks the construct) from `not_exercised` (this case does not drive
it). `verify.py` drives it, writes `results.json`, and CI uploads both as evidence.

**An unexpected PASS fails the job.** A reproducer that silently starts working means an
expectation went stale, not that the suite is healthy — the six `defects/` cases are exactly this
shape, since every one of them now does the OPPOSITE of what its file header describes.

Local: `verify.py` — 39/39 cases matched, 1.8s. CI YAML validated; `ci-complete` now needs twelve
jobs, and forgetting to add one remains visible there rather than silently unprotected.
FILES: .github/workflows/ci.yml, COMPILER-STATE.md. Suite content lives in its own repository.
NEXT: §11 layer-audit inventory enforcement, then reconciliation and the §17 report.

## CD-340 — DEV-138 CLOSED as a DEV-121 instance; all six CD-334 defects repaired (2026-08-02)

**WP-DEV-134-139 Part F. The classification came first and decided the repair, as §9 required.**

```text
declared item type   &str            06-Standard-Library.md: SplitIter / String::split / &str
HIR runtime value    Value::String   OWNED  <- the defect
value_is_copy        Value::Str -> true, Value::String -> false
front end            ACCEPTS (sees a Copy shared reference)
MIR / native         VACUOUS - both refuse SplitIter outright (C4.5)
```

**The MIR and native rows are vacuous, not confirming**, and are recorded that way rather than
counted as agreement: those engines do not implement `SplitIter`, so they could not have
disagreed. §9.3's "treat as distinct" test requires MIR to emit `Move` for a Copy shared-reference
item AND all engines to consume it. Neither holds; every testable fold criterion does.

**Producer-specific, which is what identifies it as DEV-121 rather than something new.** Six shapes
were probed: `&Vec<String>`, `&Vec<Int32>`, `chars()`, and a plain `&str` outside a loop were
already correct. Only `split` was wrong — and `trim`/`substring`, with the SAME declared return
type, already yielded `Value::Str`. The repair makes `split` consistent with its siblings rather
than introducing a rule. One line, no new `Value` variant, no amendment.

**THE MORE USEFUL FINDING IS WHY THE INVARIANT MISSED IT.** INV-VALUE-REP-001 exists precisely to
catch this class, and checks at every **`let`** that a binding declared `&str`/`&[T]` does not hold
owned storage. A **for-loop binding is not a `let`**. Both known DEV-121 instances —
`String::bytes()` at CD-305 and `String::split()` here — were reachable through a loop item, and
both were found by a user-facing program rather than by the invariant. Extending it to loop
bindings and call arguments is what would close the class; finding a third instance by hand would
not. Recorded against DEV-121, unowned.

**ALL SIX CD-334 DEFECTS ARE NOW REPAIRED.** Three were soundness holes; none required a design
change, and four turned out to be a single wrong line or a single missing consultation:

| DEV | Root cause in one line |
| --- | --- |
| 134 | operand and return type were never compared |
| 137 | a condition is neither a block nor a statement, so nothing popped its borrows |
| 136 | move state merged syntactic children instead of reaching predecessors |
| 135 | a field was identified by the span it was written at |
| 139 | two bound lookups each read half the generic environment |
| 138 | one producer returned an owned value for a borrowed type |

Local:
- cargo test --test dev138_iterator_item_representation -- 10 cases, green
- cargo test --test c63a_string --test copy_canon_matrix --test three_engine_differential --test
  exec_snapshots --test conformance --test gate2_valid --test gate3_execution -- 209 green
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34, and its `defects/05` reproducer now runs correctly
- full workspace NOT run, per the amended evidence policy

CI:
- the aggregate gate is the authority for this commit

FILES: starkc/src/interp.rs, starkc/tests/dev138_iterator_item_representation.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: the three non-defect programme tasks — in-tree regression manifest (§10.1), pinned
external-suite CI (§10.2, BLOCKED: the suite has no git remote), layer-audit inventory enforcement
(§11) — then final reconciliation and the §17 report.

## CD-339 — DEV-139 CLOSED: a method body reads the impl's bounds, not only its own (2026-08-02)

**WP-DEV-134-139 Part E. Five of six defects closed; only DEV-138 remains.**

**Wider than filed: it was TWO lookups, and the second was deferred.** The entry names operator
desugaring, but `satisfies_bound` — ordinary trait-bound satisfaction — had the identical gap. Each
kept its OWN copy of the parameter lookup and each consulted `current_fn_generics` alone; they
agreed only by coincidence. And the trait-bound half is deferred: DEV-067(a) captures "the generic
environment this obligation was recorded in" and replays it at drain, and that capture was also
fn-generics-only — so an obligation raised inside `impl<T: Ord> Pair<T>` replayed against half its
environment and still failed after the operator half was repaired. Two of this defect's own tests
found that second half, which is the argument for writing the must-pass set before assuming the
first fix was the whole fix.

**Nothing new was brought into scope.** WP-C6.2b-F5 already installed impl-head generics in
`current_impl_generics` for method bodies. The lookups never asked. This repair is a READ, not a
new binding — it cannot change which names are in scope, only which declared bounds are found.

**Two helpers, each written once:**

```
param_declares_bound(param, required)   both lookups call it
current_generic_env()                   the deferred capture calls it
```

DEV-128 and DEV-130 are both "the rule was written twice and the copies drifted". This was already
two copies; it is now one each.

**Negative controls, because WIDENING an environment risks discharging obligations never
declared:** no bound at all, `Eq` where `Ord` is required, `Ord` where `Num` is required, a bound
on a DIFFERENT parameter (pins that the lookup still matches on parameter NAME rather than finding
any bound in scope), an unbounded method-level parameter, and an undischarged callee obligation.

**DEV-083 is NOT closed by this.** It is impl-head *matching* — a concrete position in an impl head
against an unresolved receiver type argument. This was impl-head *bounds being read*. Different
mechanism; DEV-083 remains OPEN and unowned.

**What class of program is now prevented from failing:** any generic CONTAINER whose methods use
the bounds its impl declares — `Heap<T: Ord>`, `SortedVec<T: Ord>`, a `max` method. The rule is on
the environment, not on `Ord` or on operators, so it covers `Eq`/`Num`/user traits, inherent and
trait impls, and trait-bound obligations as well as operator desugaring.

Local:
- cargo test --test dev139_impl_generic_bounds -- 16 cases (10 accept, 6 reject), green
- cargo test --test c62b_f5_impl_bounds --test c62b_f6_self_normalisation --test
  c62c_associated_types --test c62d_operator_coretrait --test cross_package_generics --test
  native_c6_2_generics_traits -- 60 green; the generics/bounds subsystem closest to this change
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 69 green
- cargo test --test dev134_try_error_type --test dev135_field_move_paths --test
  dev136_terminating_path_moves --test dev137_while_condition_borrows -- 62 green, all four
  previously closed defects in this programme
- cargo test --test mir_verify --test three_engine_differential -- 160 green
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34, and its `defects/06` reproducer now checks OK
- full workspace NOT run, per the amended evidence policy

CI:
- the aggregate gate is the authority for this commit, including clippy on CI's own stable

FILES: starkc/src/typecheck.rs, starkc/tests/dev139_impl_generic_bounds.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: DEV-138 — build the engine matrix first and apply WP §9.3's decision rules, rather than
assuming it is independent.

## CD-338 — DEV-135 CLOSED: a field is one place however many times it is written (2026-08-02)

**WP-DEV-134-139 Part B. Also carries a `collapsible_match` fix for CD-337 that CI caught and the
local gate did not, and two Codex documentation updates to the frozen P1 workload.**

**THE ESTIMATE WAS WRONG AND THE RECORD IS CORRECTED, NOT REWRITTEN.** The CD-334 inventory said
"the gap is in the front end's `moved_places`, which is keyed on whole locals". It is not.
`moved_places` is a `HashSet<Place>`, `Place` already carries `projections`, and `places_overlap`
already does prefix matching. The front end was ALREADY field-precise: moving `pair.left` already
left `pair.right` live, and moving the parent afterwards was already refused.

**The actual defect was field IDENTITY, one enum variant wide:**

```rust
Projection::Field(name.lo, name.hi)   // the SPAN the name was written at
```

Two mentions of one field sit at different byte offsets, so `owner.handle` on line 5 and
`owner.handle` on line 6 were two DIFFERENT projections that `places_overlap` correctly reported as
disjoint. Nothing was missing from the move model; the comparison could never succeed. Storing the
resolved NAME fixes it. Same class as DEV-122 — identity taken from a span rather than from what
the span denotes.

**So the WP's two-stage model was never entered, and that is a real outcome rather than a shortcut.**
§5.2 split this into a conservative "DEV-135a parent poisoning" gate and a "DEV-135b precision"
follow-on. The inventory ruled poisoning out — sibling survival is asserted by the conformance
fixture set and four differential suites. But the precision DEV-135b was meant to BUILD already
existed. The repair is neither stage. **No DEV-135b is filed and none is owed**: sibling survival,
nested paths, parent/child ordering, and exactly-once drop are all covered, which is exactly what
DEV-135b's closure criteria asked for. WP §15's release gate resolves on its second branch.

**What class of program is now prevented:** any program that moves the same owned field, tuple
element, or nested field out twice — and, by the same prefix rule that already worked, any that
moves a parent after a field or reads a field after the parent. The check is on the PLACE, not on
syntax, so it holds however the field is reached.

**A CI-vs-local gate divergence, worth more than the lint it caught.** CD-337 went red on
`clippy::collapsible_match`. The lint is real; the reason it was missed is that the repo pins
`channel = "stable"` and CI's stable resolves to **1.97.0** while this machine's stable had gone
stale at **1.93.0**. Every "clippy clean" reported earlier in this programme was against an OLDER
lint set than CI's. Corrected here and going forward: the gate is `cargo +1.97.0 clippy`. This
matters disproportionately now that CD-337 made CI the sole workspace authority — a local gate that
silently differs from CI undermines exactly that arrangement.

**Codex changes included at the owner's instruction, reviewed not rubber-stamped.** Two docs on the
frozen P1 REST workload: the plan's status moves to `IMPLEMENTED — TIER-1 QUALIFIED`, and the report
identifies `P1-COMPILER-001` as a local label for the already-governed `DEFECT-C788-LOOP-TEMP`
(discharged by MIR amendment A12) while demoting a stale `P1 PARTIAL` handoff to quoted history.
Cross-references verified: `a12_storage_end_shapes.rs`, `mir-amendment-A12-storage-end.md`, and
CD-263/264/265/273 all exist. **CD-269 is cited and is absent from this file** — it is a real
decision (commit `28a9ad1`, cited in five other documents), so the Codex text is correct and the gap
is in this ledger. Recorded, not silently patched.

Local:
- cargo test --test dev135_field_move_paths -- 16 cases (6 reject, 10 accept), green
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 65 green
- cargo test --test mir_verify --test mir_differential --test three_engine_differential -- 292 green
- cargo test --test dev134_try_error_type --test dev136_terminating_path_moves --test
  dev137_while_condition_borrows -- 46 green, all three previously closed defects
- cargo test --test c61f_reference_boundary --test c61f_structural_copy --test
  native_c6_1_ownership --test operand_move_inventory --test copy_canon_matrix -- 51 green
- cargo test --test native_c5_3_aggregates_enums --test c6_generated_corpus -- 27 green; these are
  the suites that assert partial-move field precision at the MIR and native layers
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- full workspace NOT run, per the amended evidence policy
- `cargo +1.97.0 clippy --workspace --all-targets --all-features -- -D warnings` was IN FLIGHT when
  this was committed, at the owner's instruction to let CI decide. It is NOT claimed as passing.

CI:
- aggregate workspace gate is the authority for this commit, including clippy on CI's own stable

FILES: starkc/src/borrowck.rs, starkc/tests/dev135_field_move_paths.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md, and the two Codex P1 documents.
NEXT: DEV-139 — merge impl-level generics into the obligation environment the operator check reads.

## CD-337 — DEV-136 CLOSED: only branches that reach a join contribute to it (2026-08-02)

**WP-DEV-134-139 Part D, and the second of four milestone points. `if flag { return out; }
out.push('a');` compiles.**

**Layer: `borrowck.rs`.** The `If` arm unioned the then-branch's move set into the post-state
unconditionally; the `Match` arm extended the merged set from every arm. Neither asked whether the
branch reaches the join. A branch that `return`s is not a predecessor of the statement after the
`if`, so its moves were being attributed to a path they never happened on.

**Divergence is read from existing authorities, not re-derived from syntax.** A
`Return`/`Break`/`Continue` statement anywhere in the sequence, plus the type checker's own
`Ty::Never` for `panic(..)` and any call returning `!`. Composite forms recurse: an `if` diverges
only when both sides do, a `match` only when every arm does.

**THE DIRECTION OF CONSERVATISM IS THE ENTIRE SAFETY ARGUMENT.** The predicate answers "does this
definitely NOT reach the join?":

```
wrong `true`   -> drops a real move from the join -> accepts use-after-move -> UNSOUND
wrong `false`  -> keeps the old false positive     -> merely annoying
```

So it reports `true` only on evidence, and anything unrecognised falls through to `false`. `loop`
without a reachable `break` is deliberately NOT treated as diverging — judging it needs
reachability analysis the checker does not have, and guessing would land on the unsound side.

**Two merge subtleties, both found while writing the repair and both pinned:**

| Case | Naive answer | Correct answer |
| --- | --- | --- |
| `if` with no `else`, branch terminates | branch's move set | the state from BEFORE the `if` |
| `match` where ALL arms terminate | empty merged set | the pre-match state |

The second is the dangerous one: an empty merge would silently resurrect a value moved BEFORE the
`match`, turning a false positive into a false negative. `a_move_before_an_all_diverging_match_
is_still_rejected` exists precisely for it.

**Drop obligations, not just diagnostics.** `a_droppable_value_survives_a_terminating_branch`
executes both paths and asserts each `Guard` is destroyed exactly once — the false path drops at
end of scope, the true path moves into a callee that drops it there.

**What class of program is now prevented from failing:** any program whose move happens only on a
path that leaves the function or the loop. The rule is on the control-flow edge, not on the
syntax, so it covers `return`, `break`, `continue`, `panic`, nested `if`, and `match` arms
uniformly — and it does NOT cover a branch that can fall through, which is what keeps
maybe-moves rejected.

Local:
- cargo test --test dev136_terminating_path_moves -- 14 cases (9 accept, 5 reject), green
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 65 green
- cargo test --test mir_verify --test mir_differential --test three_engine_differential -- 292 green
- cargo test --test dev134_try_error_type --test dev137_while_condition_borrows -- 32 green,
  the two previously closed defects in this programme
- cargo test --test c61f_reference_boundary --test native_c6_1_ownership --test
  operand_move_inventory --test copy_canon_matrix --test exec_snapshots --test snapshots -- 43 green
- cargo clippy --release --lib --tests --all-features -- -D warnings -- clean
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34
- FULL WORKSPACE (milestone 2 of 4) -- see the commit for the recorded result

CI:
- aggregate workspace gate PENDING

FILES: starkc/src/borrowck.rs, starkc/tests/dev136_terminating_path_moves.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: DEV-135b — full field-sensitive move paths, which the inventory established is the
release-gating repair rather than the DEV-135a poisoning gate.

## CD-336 — DEV-137 CLOSED: condition-only borrows end at the branch boundary (2026-08-02)

**WP-DEV-134-139 Part C. `while i < v.len() { v[i] = 5; }` compiles. So does the `if` form, which
had the identical defect and was found by this repair's own test.**

**The layer, recorded before the repair as the work package required: `borrowck.rs`.** Not MIR,
not liveness, not the back-edge. `active_borrows` is a stack scoped by exactly two mechanisms —
`check_block` truncates at block end, `check_stmt` truncates after each expression statement. A
CONDITION is neither: it is an expression evaluated outside any statement of its own. So

```rust
hir::ExprKind::While { cond, body } => {
    self.check_expr(*cond);      // pushes the auto-borrow `values.len()` takes
    self.check_block(*body);     // records its entry depth AFTER that push
}
```

left the condition's temporaries on the stack for the whole body, and `check_block` restored to a
depth that already included them. Nothing popped them until the enclosing statement ended.

**Wider than filed, same mechanism.** `if` conditions were identical. The growing-vector must-pass
case is what exposed it: `if values.len() < 5u64 { values.push(1); }` inside a loop body was
refused for the same reason. One repair, `check_condition`, written ONCE and used by both arms —
DEV-128 and DEV-130 are both "the rule was written twice and the copies drifted".

**The scope boundary is the whole design, and it is not "loop and branch headers".** `match`
scrutinees and `for` iterators are deliberately NOT routed through `check_condition`:

| Position | Borrow must | Why |
| --- | --- | --- |
| `while` / `if` condition | END at the branch | value is consumed by the branch |
| `match` scrutinee | SPAN the arms | PAT-BIND-001 binds payloads by reference into it |
| `for` iterator | SPAN the body | yields references into the iterated value |

Truncating either of the bottom two would hand out references to storage the checker had stopped
tracking. Both are pinned by negative controls that fail if someone later generalises the repair.

**Why depth-based rather than clearing the borrow set.** A borrow created before the loop
(`let view = &values;`) sits at a shallower depth than the snapshot, so the truncate cannot reach
it and a body mutation through its owner is still refused. That is
`borrow_predating_the_loop_stays_live`, and it is the difference between modelling a region and
just forgetting.

**Execution, not merely acceptance.** `a_growing_vector_re_evaluates_its_condition` runs through
the oracle and asserts output. It also settles a question the workaround raised: hoisting
`let n = v.len()` was a SEMANTIC change, not a stylistic one — that loop grows the vector it is
measuring, so a hoisted bound stops early. The samples that carried the hoist workaround were
working around a defect at the cost of a different meaning.

**What class of program is now prevented from failing:** any program that reads a receiver in a
condition and mutates it in the guarded branch. The fix is on the borrow REGION, not on `len` or
on `Vec`, so it holds for every method, every receiver type, `&mut` parameters included, and for
indexed place reads (`while values[0] < 3`) as well as method calls.

Local:
- cargo test --test dev137_while_condition_borrows -- 16 cases (12 accept, 4 reject), green
- cargo test --test conformance --test gate2_valid --test gate3_execution -- 65 green
- cargo test --test mir_verify --test mir_differential --test three_engine_differential
  -- 292 green (78s for the three-engine differential)
- cargo test --test c61f_reference_boundary --test c61f_nested_refs --test native_c6_1_ownership
  --test dev132_vec_index_place --test operand_move_inventory -- 44 green, the borrow/ownership
  subsystem closest to this change
- cargo clippy --release --lib --tests --all-features -- -D warnings -- clean
- cargo fmt --all -- --check -- clean; rustfmt on the two touched files only
- qualify-first-party-packages.py over all ten packages -- exit 0
- external task-shaped suite -- 34/34
- full workspace NOT run for this commit, per the 2026-08-02 evidence ruling; the next milestone
  run is after DEV-136

CI:
- aggregate workspace gate PENDING

FILES: starkc/src/borrowck.rs, starkc/tests/dev137_while_condition_borrows.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, COMPILER-STATE.md.
NEXT: DEV-136, then the milestone full-workspace run.

## CD-335 — DEV-134 CLOSED: `?` now relates its operand to the return type (2026-08-02)

**WP-DEV-134-139 Part A. `?` required exact error-type compatibility; it now does. The ruling is
REJECT, not convert — recorded as a decision, because it is a language question and not an
implementation detail.**

```text
`?` requires exact error-type compatibility.
Implicit From-based propagation is not part of this repair.
```

**The defect was one missing relation, not two.** The `Try` arm asked "is the enclosing return
type `?`-capable?" and "is the operand `?`-capable?" as INDEPENDENT questions and never compared
them. That single omission produced two symptoms, and the repair work found the second:

| Accepted before | Propagated value | Caller sees |
| --- | --- | --- |
| `Result<_, Low>?` in a fn returning `Result<_, High>` | a `Low` | tag from another enum |
| `Option<_>?` in a fn returning `Result<_, _>` | a `None` | tag from another enum |
| `Result<_, _>?` in a fn returning `Option<_>` | an `Err` | tag from another enum |

The constructor half was NOT in DEV-134 as filed. It is the same mechanism and the same repair, so
it widened the existing entry rather than taking a new number (WP §2, one mechanism one repair).

**Deferred, like `display_checks`, and for the same reason.** The operand's error type is routinely
an inference variable while the body is being checked (`Err(make())?`), so an eager comparison
would either reject valid code or force a premature binding. `check_try_compatibility` is recorded
during checking and drained after inference settles.

**E0006 widened rather than a new code allocated.** The spec's E0006 now covers the whole
return-type contract for `?` — wrong constructor, mismatched error type, or a function that
returns neither. One code per concept, normative table stable, conditions distinguished by
message. `non_result_return_reports_once_not_twice` pins that the pre-existing condition still
reports once rather than twice.

**A LATENT GAP FOUND BY THIS WORK'S OWN NEGATIVE CONTROL, and deliberately not repaired.**
`types_equal` has no `Ty::Param` arm: two occurrences of the same type parameter compare unequal
and fall to `_ => false`. Its existing callers are coherence and overlap paths where `Ty::Param`
is pre-handled or where a conservative `false` is safe, so it has no demonstrated symptom there —
but it made the first version of this repair reject

```stark
fn low<E>(e: E) -> Result<Int32, E> { Err(e) }
fn same<E>(e: E) -> Result<Int32, E> { let v = low(e)?; Ok(v) }
```

which `error_type_as_a_generic_parameter_is_accepted` caught before it could ship. Widening a
shared coherence primitive to fix a symptomless gap was rejected as out of scope; instead the
structural walk takes the `Ty::Param` behaviour as a PARAMETER (`types_equal_inner`) — written
ONCE, reached by two entry points, because DEV-128 and DEV-130 are both "the rule was written
twice and the copies drifted". Whether `types_equal` itself should be widened is unowned and gets
a DEV number only if a symptom is found.

**What class of program is now prevented, which is the closure question rather than "the
reproducer passes":** no program can propagate a value into a return type that cannot represent
it. The check is on the TYPES at the propagation site, not on any syntactic shape, so it holds for
`?` in any position — nested helpers, generic bodies, chained propagation — and it cannot be
evaded by adding a `From` impl, which is the shape a reader coming from Rust would expect to work.

EVIDENCE (all run locally, at this head):
`cargo test --test dev134_try_error_type` — 16 cases, 7 reject / 9 accept, green.
`cargo test --test conformance --test gate2_valid --test gate3_execution` — 65 green.
`cargo test --test c788_synth --test a10_provider_call --test c788_source_time_e2e` — 32 green;
these are the provider paths that use `?` most heavily and were the main over-rejection risk.
`qualify-first-party-packages.py` over all ten packages — exit 0.
External task-shaped suite — 34/34, unchanged.
`cargo fmt --all -- --check` — clean. `rustfmt` was run on the two touched files only, never
tree-wide, because this checkout is shared with parallel sessions.
Spec regenerated (`build-core-spec.py`) and the 112-block fixture corpus re-extracted: manifest in
sync, no block added or renumbered.
LEFT TO CI: `cargo clippy --workspace --all-targets --all-features -- -D warnings` and
`cargo test --workspace --all-targets --all-features` were still running locally when this was
prepared; both are required and are the aggregate gate.
FILES: starkc/src/typecheck.rs, starkc/tests/dev134_try_error_type.rs (new),
starkc/docs/conformance/KNOWN-DEVIATIONS.md, STARKLANG/docs/spec/04-Semantic-Analysis.md and the
regenerated STARK-Core-v1.{md,html,pdf}, COMPILER-STATE.md.
NEXT: DEV-137, per the work package's required order.

**PROGRAMME STATUS — two counts, deliberately kept apart.** They are not the same number and
conflating them misreports progress in both directions: the defect count understates the work
(regression manifest, external-suite CI, and layer-audit hardening are none of them defects), and
the task count understates release readiness (only the defects gate the release).

```text
Defects (WP-DEV-134-139 Parts A-F)     ALL SIX CLOSED (134, 135, 136, 137, 138, 139)
                                       DEV-135b NOT FILED — see CD-338
                                       DEV-138 closed as a DEV-121 instance; that class stays OPEN
                                       DEV-135b conditional on the DEV-135 inventory

Programme tasks (WP-DEV-134-139)       16 of 16 complete; release recommendation is
                                       CONDITIONAL on CD-340/341/342 CI going green
                                       includes the six defect repairs plus the in-tree
                                       regression manifest (§10.1), the pinned external-suite
                                       CI job (§10.2), layer-audit inventory enforcement (§11),
                                       and final reconciliation (§16)
```

**LOCAL EVIDENCE POLICY, owner ruling 2026-08-02, in force from DEV-137 onward.** Per-commit local
evidence is TARGETED — fmt, clippy over affected targets, the dedicated DEV suite, closest
subsystem suites, MIR differential/verifier when lowering or ownership changes, affected package
qualification, the external sample suite, and a clean `git status --short`.

The original ruling additionally required full local workspace runs at four milestones.
**AMENDED the same day after measurement: local workspace runs are DROPPED entirely after
CD-337.** Each takes ~17 minutes, two more were scheduled, and they duplicate a gate CI already
enforces on every pushed commit. Milestones 1 (CD-335, DEV-134) and 2 (CD-337, DEV-136) were run
and their results are recorded; no further local workspace run is required, including at
programme completion.

**CI's aggregate required check is therefore the SOLE workspace authority**, which makes the merge
gate strictly more important rather than less. A targeted local run supports a commit's evidence
statement and never replaced CI, but from CD-338 onward nothing local covers the workspace at all.
Any commit whose CI run is red is unverified regardless of how much local evidence it carries.

**A procedural rule learned the hard way (CD-337): while a workspace run is in flight, nothing
else may touch cargo.** The first attempt at milestone 2 was invalidated because a
`cargo build --bin starkc` for the NEXT defect landed mid-run, and ~49 test files invoke the
compiler through `CARGO_BIN_EXE` — so an unknown number of suites ran against a binary carrying an
unrelated change. Two other measurement errors in the same session point the same way: a
`head -40` that truncated a run and made `head`'s exit code look like cargo's, and a
`grep '^test result'` that counted tests NAMED `result_*` as suite summaries. Capture the tool's
own exit code, and validate counts against a known total rather than trusting a summary line.

## CD-334 — six defects filed from an external sample suite; three are soundness (2026-08-02)

**An 18-package sample suite was written OUTSIDE this repository, against the release binaries, to
answer "what does it feel like to write ordinary STARK today?". It found six defects, numbered
DEV-134…DEV-139. Three of them are soundness gaps the fixture corpus does not reach.**

| DEV | One line | Class |
| --- | --- | --- |
| 134 | `?` neither converts the error type nor requires a conversion to exist | **soundness** — type confusion |
| 135 | moves of individual struct FIELDS are not tracked; second move surfaces as an ICE | **soundness** (bounded by the oracle) |
| 136 | a move on a `return`ing path is treated as unconditional (E0100) | false positive |
| 137 | a receiver auto-borrow in a `while` CONDITION is live across the body (E0101) | false positive |
| 138 | an iterator-yielded `&str` is consumed by its first use | **soundness-adjacent**; candidate DEV-121 instance |
| 139 | impl-level generic bounds are invisible to operator desugaring (E0500) | false positive |

Full structured entries — normative expectation, reproducer, engine behaviour, impact, workaround,
disposition — are in the canonical ledger, `starkc/docs/conformance/KNOWN-DEVIATIONS.md`. This
record is the index and the finding about method, not a second copy.

**DEV-134 is the one that needs an owner decision rather than an implementation.** The spec does not
scope a `From` conversion at the propagation site, so "convert" would be new semantics — CE-shaped —
while "reject" is the conservative half and can land alone. Filing it does not presume which.

**DEV-138 is filed as a hypothesis, not a finding.** It is plausibly an instance of the still-open
DEV-121 value-representation class rather than an independent defect; INV-VALUE-REP-001 is the
instrument that would settle it, and it has not been run against this reproducer. Recorded that way
deliberately, so the count is not inflated by a duplicate.

**Why an external suite found things the corpus did not, which is the durable point.** The fixture
corpus and the generated C6 corpus both grow from the compiler's own model of what programs look
like. The sample suite grew from *tasks* — sort a vector, walk a graph, parse an expression, encode
a run-length string — and the defects cluster exactly where those two diverge:

```
DEV-136, DEV-137   ordinary imperative loop and early-return shapes
DEV-135, DEV-138   ownership of things the corpus rarely uses twice
DEV-139            generic CONTAINERS with methods, not generic functions
```

DEV-137 is the most disruptive in practice: `while i < v.len()` is how an indexed loop is written,
and every in-place algorithm hits it. Its workaround — hoist the length — **fails when the length
changes**, so a growing queue must track its length by hand. That is worth weighting above the other
two false positives when this is scheduled.

**A limitation found alongside them, filed as neither defect nor deviation because it may be
intended:** `Box<T>` cannot be dereferenced, so a recursive tree built with `Box` can be constructed
but never walked by reference — traversal requires consuming it with `Box::into_inner`. The suite
routes around this with an arena (nodes in one `Vec`, children as indices), which is a legitimate
technique rather than a workaround. If by-reference traversal of a boxed tree is meant to be
possible, this is a seventh defect; if not, it is a documented consequence of having no `Deref`.
The owner's call, which is why it carries no DEV number.

EVIDENCE: six runnable reproducers, one per DEV, verified against
`starkc/target/release/{stark,starkc}` at this head — each shown to reproduce its stated
`starkc check` and `starkc run` outcomes. The suite itself is 18 packages / 55 files / ~5,500 lines
and passes 34 of 34 checks with the six workarounds applied and commented.
SCOPE: **documentation only.** No compiler source, test, or fixture was modified under this CD, and
no defect was repaired. Nothing here is gate evidence.
FILES: starkc/docs/conformance/KNOWN-DEVIATIONS.md (DEV-134…139 appended), COMPILER-STATE.md.
NEXT: owner triage — DEV-134's reject-vs-convert ruling, and whether DEV-138 folds into DEV-121.

## CD-294 — E0106 reverted: the layer migration was not the cheap kind (2026-07-31)

**CD-293 moved `v[i]`-on-a-non-`Copy`-element from MIR verification into semantic analysis as
E0106, on the E0105 precedent that acceptance and executability should agree. It broke three
working programs and is reverted.**

| What broke | Why it was never a move |
| --- | --- |
| `holder.key == values[0]` | dispatches through `Eq::eq(&self, &Key)` — auto-borrowed |
| `vs[idx()].push(arg())` | method receiver — borrowed |
| `v[1u64] = Loud { id: 20 }` | assignment target — never read at all |

**The premise was wrong, not the execution.** MIR reaches `VecIndexGet` only from the value-read
path; a receiver, an assignment target, a borrow, and an auto-borrowed comparison operand all index
a Vec and never arrive there. The front end sees only the syntax `v[i]`. Refusing on the syntax
refuses four things to catch one.

Scoping it correctly means enumerating every place context — `&`, receiver, field base, assignment
LHS, comparison operand, nested index, match scrutinee — and **missing one breaks working code**,
which is what happened three times. That is a value-context analysis the checker does not have, and
building it is a design change, not a diagnostic fix.

**The ergonomic win survives where it can tell the two apart.** MIR-0016 now reads: "…`v[i]` reads
by value, which would move the element out of the Vec; borrow it instead with `v.get(i)` (yields
`Option<&T>`), `&v[i]`, or read it in place by iterating with `for x in &v`". A message cannot
produce a false positive.

**CD-293's other two changes stand and are untouched:** `for x in &v` and `&v[i]` are pure
additions — they can only make previously-rejected programs work, never break working ones.

**WHAT THIS MEANS FOR THE LAYER AUDIT, which is the durable finding.** The estimate of "1–2 days,
each fix small" for batch-migrating MIR refusals into semantic analysis is wrong. E0105 was cheap
because by-value `Vec` iteration has exactly ONE syntactic form. E0106 was not, because `v[i]`
appears in value and place positions that only later phases distinguish. The audit's output
therefore needs a fourth classification beyond reachable/unreachable:

```
CHEAP     one syntactic form, unambiguous            (E0105-shaped)
EXPENSIVE appears in value AND place positions;      (E0106-shaped)
          needs context analysis the checker lacks
```

Which one a site is cannot be read off its message. It takes a probe per site.

**Also recorded: how this was missed locally.** Two of the three failures are unit tests inside
`src/interp.rs`, which `cargo test --test <name>` never compiles. Only
`cargo test --workspace --all-targets` sees them — the same command CI runs.

EVIDENCE: full-workspace verification was still running when this was pushed, at the owner's
direction. fmt clean and clippy clean under CI's exact flags
(`--workspace --all-targets --all-features -- -D warnings`) were confirmed before the revert.

## CD-293 — the three Vec ergonomics edges, and the guard that was a name filter (2026-07-31)

**Found by writing CD-292's file surface against the real API, not by review.** Reading a
`Vec<DirectoryEntry>` — a Vec of owning structs, which is the shape of most real data — failed
three ways in a row, and only the fourth spelling worked.

| Spelling | Was | Now |
| --- | --- | --- |
| `for x in &v` | E0001 "requires an iterable value" | **works** — same cursor as `v.iter()` |
| `for x in v` | E0105, names `.iter()` | unchanged (by-value moves elements out) |
| `v[i]`, non-`Copy` | **MIR-0016 at verification** | **E0106 in semantic analysis**, with a help |
| `&v[i]` | unrepresentable | **works** — `VecGetRef`, `None` arm traps |

**`for x in &v`** was not an architectural limit — it was a missing arm, in three engines: the type
checker, MIR lowering (builds the cursor with the same `VecIterNew` the method call emits), and the
HIR oracle (builds `Value::VecIter` at the same place). The differential harness caught the oracle
half; a two-engine change would have shipped a divergence.

**`v[i]` on a non-`Copy` element is correctly refused** — it would move the element out of a place
the Vec still owns. What was wrong was WHERE: it type-checked, ran in the oracle, and died at MIR
verification. **An accepted program no compiler could build** — precisely the defect class WP-C7.9
Packet E fixed for by-value `Vec` iteration (E0105) and left unfixed for indexing. E0106 now raises
during semantic analysis and its help names both borrowing reads, because there are two and neither
is guessable from "requires a Copy element type". Added to the normative spec; compiled spec
regenerated.

**`&v[i]` was unrepresentable, and that part IS architectural.** A Vec is `MirTy::Core` — an opaque
runtime type, not a projectable place — so there is no `Projection::Index` to borrow, which is why
`&a[i]` on an ARRAY always worked and `&v[i]` never did. Closing it needed no representation change:
`VecGetRef` already yields `Option<&T>`, and the `None` arm IS the out-of-bounds case, so it raises
`IndexOutOfBounds` — same category, same observable behaviour as `v[i]`, reached by another route.

**One column of disagreement, caught by the harness.** MIR blamed the index expression, the oracle
blamed the enclosing `&`. Same category, same line. MIR now matches the oracle: three-engine
agreement is the authority on provenance, and either span alone would have been defensible.

**CD-292's CI failure: a name filter standing in for a semantic rule.**
`no_environment_mutating_function_is_declared` scanned EVERY first-party provider for
`set|put|unset|remove|clear|exec|spawn`. It failed on `stark_iofile_set_len` and `stark_iofile_remove`
while `stark_iofile_write` and `stark_file_create` — which mutate no less — sat beside them and
passed, because their names miss the list. Packet 5's rule is that the **process environment** is
read-only, not that no provider mutates anything; a filesystem provider that cannot write is not a
filesystem provider. Split accordingly: `exec`/`spawn` stay whole-registry (nothing runs a process),
the mutation list applies to `process.*` capabilities. The guard tests the rule again instead of the
spelling.

EVIDENCE: `c63c_iterators` **22/22** three-engine, including the out-of-bounds trap case.
`c783_args_env` 9/9, `c788_starkc_build` 6/6, `c784_file` 11/11. fmt clean.

## CD-292 — the rest of the file surface, executed rather than declared (2026-07-31)

**`stark-io` had four types with nothing behind them.** `OpenOptions`, `SeekFrom`, `FileMetadata`
and `DirectoryEntry` were declared and referenced by no function — `OpenOptions` had validation but
no `open_with_options`, so append/truncate/create-new were unreachable. The API looked like it had
open options; it did not.

Provider: 6 `io_file` symbols → 19. Adds open-with-options, seek, durable sync, set-length, metadata
(by handle and by path), path existence, remove, rename, copy, directory create/remove/list.
Package: ~20 functions consuming all four types, plus `path_join`.

**Encodings, each chosen rather than defaulted** — ABI §10 admits no aggregate parameter:
- open options travel as a **bitmask**; an unknown bit is `InvalidInput`, not a dropped mode;
- a seek origin is a **discriminant byte**; a `Start` offset above `Int64::MAX` is refused rather
  than cast, because a failing `as` traps in every build mode;
- metadata is a **row of out-slots**, with each timestamp a (seconds, valid) pair. No sentinel:
  every `Int64` is a real instant, so there is no in-band value meaning "absent" — the same defect
  CD-277 found in the clock reading, avoided by construction;
- a directory listing is a **bounded NUL-separated snapshot** into a caller buffer, not a cursor.
  A cursor would be a second resource type with its own lifecycle to get wrong; a snapshot owns
  nothing past return. Truncation is reported and raised as `LimitExceeded` rather than returned as
  a short list, which would be indistinguishable from a directory that small.

**Deliberately absent:** recursive directory creation and recursive delete. Both are unbounded
effects from a single call, and the second is the most destructive filesystem primitive there is.
Callers walk with `read_dir` and act on what they have actually seen.

**FOUR LANGUAGE SHARP EDGES, found by writing real code against a real API** — the evidence P1 was
expected to produce, arriving early:
1. `if v.len() > 0 && !f(v[v.len()-1]) { v.push(..) }` — **E0101**. The index in the condition holds
   a borrow across the mutation in the body. Reading the byte into a local first ends it.
2. `entries[i]` where the element is a non-`Copy` struct — **MIR-0016**. `VecIndexGet` requires a
   `Copy` element, and borrowing the indexed place does not help. **A `Vec` of non-`Copy` structs is
   not readable by index at all.**
3. `for x in &v` — **E0001**, `&Vec<T>` is not iterable.
4. `for x in v` — **E0105**, but the diagnostic names the fix: "iterate over a borrow with
   `v.iter()`". The one refusal here that teaches instead of only refusing.

(2)+(3)+(4) compound: a `Vec` of owning structs is reachable **only** through `.iter()`, and two of
the three natural spellings fail first. Worth `WP-C7.8-RB0`'s attention, or its own ergonomics item.

**A behavioural correctness point the test caught:** `set_length` on a handle from `open_file`
fails, because `open_file` is read-only. The API was right and the test was wrong; the test now uses
`open_with_options` with `write`, which exercises that path too.

EVIDENCE: new `io_expanded_surface_executes_from_source_through_stark_io_package` — seek positions,
metadata lengths, listing composition and cleanup asserted on **observed values**, not on absence of
error, so an operation that silently did nothing fails it. `c788_starkc_build` 6/6,
`c788_provider_api_manifest` 10/10, `a11_host_resource` 38/38, `c784_file` 11/11. fmt clean.
Workspace and clippy left to CI.

## CD-291 — file IO works, because the package stopped asking for Core's identity (2026-07-31)

**`io_minimal_executes_from_source_through_stark_io_package` passes.** Ordinary STARK source opens,
writes, reads and closes a real file through the first-party provider in a natively built binary.
CD-290 shipped that test `#[ignore]`d; this removes the attribute rather than the guards.

**The question was never "how do we let a package use Core `file`".** It was "why does a package
need Core's resource identity at all", and the answer is that it does not. `stark-io`'s type is
`NativeFile`, not `File`. The only thing binding it to `file` bought was the provider's existing
symbols — and it cost every guard that protects Core `File`'s single destruction path.

`stark-io` now binds **`io_file`**: a second resource type on the same provider, with its own
symbols (`stark_iofile_*`) and its own handle tag. Consequences, none of them exemptions:

| Guard | Why it passes now |
| --- | --- |
| CD-224 — a package may not claim a Core resource | `io_file` is absent from `ResourceRegistry::builtin()` |
| MIR-0027 — a Core-owned resource may not be a `HostResource` | `io_file` is not `LegacyCore` |
| A11 §5 rule 4 — MIR owns a resource's only close | `io_file` is wholly on the `HostResource` path |

**The verifier caught the one real defect in this design, which is the point.** With `io_file` a
genuine resource, `stark-io`'s `file_close` calling the provider close directly became MIR-0033: a
second destruction path. Correct — drop elaboration already emits that close. `file_close(file:
NativeFile)` now takes the handle by value and calls nothing; taking ownership IS the close. The
signature keeps its `Result` but can only return `Ok`, because a destructor has nowhere to put an
error — `file_flush` is where a flush failure is observable. That is a real API consequence and it
is documented rather than papered over.

**What this does NOT do:** migrate Core `File`. That remains open, and having traced it, it is a
three-engine change — the reference interpreter implements `File` natively (`Builtin::FileOpen`,
`FileCreate`), so checker, interpreter and backend must move together and be requalified for
agreement. An earlier estimate of "a session or two" in this session's discussion was wrong. Nothing
in `stark-io` waits on it any more; only the spelling `File` does.

**Also fixes the CI failure CD-290 caused.** The C6.4 qualification harness rejects any `#[ignore]`
not registered in `CLASSIFIED_IGNORES` — "either the observation is required, in which case fix the
test, or classify it with a reason". CD-290's unclassified ignore failed `C6.4 tier-1 qualification`
on both tier-1 platforms and the dependent agreement job, while all three `fmt, clippy, test` jobs
passed, which is exactly the split that rule exists to produce. Taking the first branch — fixing the
test — removes the deviation at its cause.

EVIDENCE: `c788_starkc_build` **5 passed / 0 ignored** (was 4 + 1 ignored), `c788_provider_api_manifest`
10/10, `a11_host_resource` 38/38, `stark-file/native` 8/8, provider suites green. fmt clean, clippy
clean. No file under `starkc/src/mir/` or `package.rs` is touched by this change.

## CD-290 — WP-IO.1 lands with its guards intact; its e2e is blocked on Route B (2026-07-31)

**The minimal native file-IO slice is committed. Its end-to-end test is `#[ignore]`d, deliberately,
and that is the honest state rather than a failure to finish.**

The slice binds `stark-io`'s nominal `NativeFile` to the provider resource `file`. That resource is
Core-owned (`ResourceRegistry::builtin()` maps it to `LegacyCore(CoreType::File)`). The slice first
ran by removing three compiler guards:

| Guard | Removal |
| --- | --- |
| CD-224, `package.rs` — a package may not declare a Core resource | deleted outright |
| MIR-0027, `verify.rs` — a Core-owned resource may not be a `HostResource` | `&& !(provider == "stark-std-file" && resource == "file")` |
| A11 §5 — MIR owns a resource's only close | early `continue` on the same string pair |

Together those put `file` on the `HostResource` path for selected rules while it kept legacy
direct-close semantics: **one resource name, two MIR representations, two destruction paths.** That
is the half-migration SELECT-C exists to refuse and that CD-235's `partially_migrated_core` was
written to catch. The in-code comment beside the first exemption argues specifically against
named exemptions, because one "would still exempt a program that HAD migrated, which is the very
state the guard exists to catch" — a string pair is strictly weaker than the form it rejects.

**All three guards are restored, and `a_package_may_not_declare_a_core_resource` is restored to its
original assertion** (it had been inverted to `a_package_may_declare_the_file_resource_for_stark_io`).
`io_minimal_executes_from_source_through_stark_io_package` is `#[ignore]`d with the reason on it.

**What unblocks it:** migrating `file` off the legacy path WHOLLY — Route B's representation and
lifecycle work. A complete migration is already permitted; only the partial one is refused. When it
lands, delete the attribute; nothing in `stark-io` needs to change. Recorded in
`stark-io/BLOCKERS.md`, whose "Closed in the minimal native slice" heading was corrected to
"Written" — it claimed a closure the guards do not allow.

Also in this change, from the same work: cross-file item resolution in `native_build.rs`
(`resolve_resource_items` read spans against the entry file), a non-panicking `span_text`, `pub` on
the synthesized resource nominal, and the `stark-file` status vocabulary. The `IOError::` labels in
that vocabulary name variants Core's `IOError` does not have; they are consumed only as text in a
generated-code comment (`emit_provider.rs`), so this is a naming defect, not a conformance break —
left as-is and recorded here rather than fixed silently inside another change.

EVIDENCE: `a11_host_resource` 38/38, `c788_provider_api_manifest` 10/10, `c788_starkc_build` 4 passed
/ 1 ignored, `c64_platform_matrix` 15/15. fmt clean, clippy clean.

## CD-289 — a guard test matched the layout the guard's own decision introduced (2026-07-31)

**CI had been red for four consecutive commits (CD-284, CD-285, CD-286, CD-287) on one test**, on
linux-x64, macos-arm64 and windows-x64. Not one of those changes caused it in the sense the run
implied, and the switch under test worked correctly throughout.

`portability_installed_runtime_requirement_refuses_the_checkout_fallback` asserted that under
`REQUIRE_INSTALLED_RUNTIME` no attempted path ends with `starkc/stark-runtime` — an accurate way to
name the checkout fallback until **CD-284 introduced the installed MIRROR layout**,
`<prefix>/lib/stark/starkc/stark-runtime`, which mirrors the repository precisely so that the runtime
crate and the provider crates resolve `stark-provider-abi` to one path. That legitimate installed
candidate ends with the same two components, so the guard began rejecting the very layout the
decision it guards had just added.

Fixed at the assertion, which now compares against the checkout path itself
(`<starkc manifest dir>/stark-runtime`) rather than a suffix, plus a check that the installed
locations are still attempted and reported. A suffix was always the wrong instrument for identifying
one specific directory.

EVIDENCE: `c64_platform_matrix` 15/15 locally; the three-platform result is the CI run on this commit.

## CD-287 — every `MirTy` predicate that asserts a property is now exhaustive (2026-07-31)

**Generalises CD-240 from the instance to the shape.** CD-240 found `MirTy::HostResource` classified
`Copy` by a `_ => true` arm and fixed that arm, while recording the real finding in its message:
"THIRD TIME A MirTy CATCH-ALL HAS SWALLOWED THIS VARIANT." By the time A11 was working the count was
six, each found by an e2e observing generated code, none by a test — because the type checker cannot
notice a new variant falling into a wildcard.

**The rule applied.** A wildcard is safe when its arm DECLINES to handle a type (`unsupported(...)`,
`unreachable!`, "don't fold this"). It is unsafe when the arm ASSERTS A PROPERTY — `Copy`, `Noop`,
"needs no drop", "carries no borrow" — because it then makes that claim on behalf of every variant
nobody has classified yet, silently and with the suite green. Twelve predicates were of the second
kind and are now exhaustive; the ~40 decline-shaped wildcards in `lower.rs` are deliberately
untouched.

| Site | Was |
| --- | --- |
| `mir/mod.rs` `TypeContext::is_copy` | `_ => true` |
| `mir/lower.rs` `FnLowerer::is_copy` | `_ => true` |
| `mir/lower.rs` `ty_needs_drop` | `_ => false` |
| `mir/lower.rs` `ty_has_user_drop_guarded` | `_ => false` |
| `mir/lower.rs` `ty_mentions_user_nominal` | `_ => false` |
| `mir/lower.rs` `ty_carries_ref` | `_ => false` |
| `mir/verify.rs` `may_need_drop` | `_ => false` |
| `mir/verify.rs` `mir_needs_drop` | `_ => false` — **latent defect, see below** |
| `mir/drop_plan.rs` `plan_for` | `_ => Ok(DropPlan::Noop)` |
| `backend/…/emit_types.rs` `ty_carries_reference` | `_ => false` |
| `backend/…/emit_types.rs` `ty_contains_ref` | `_ => false` |
| `backend/…/emit_types.rs` `nominal_needs_lifetime` | `_ => false` |

**THE SEVENTH SWALLOWED INSTANCE, AND THE FIRST FOUND BY THE COMPILER.** `verify::mir_needs_drop`
was still classifying `MirTy::HostResource` as needing no drop. The verifier therefore held two
copies of "does this need dropping" that disagreed about resources — `may_need_drop` said true,
`mir_needs_drop` said false — with nothing in the suite distinguishing them. Every previous instance
of this defect was found by an e2e observing generated code, after a leak; this one was found by
making the match exhaustive, which is the entire argument for doing it.

**WHAT IT GOVERNED IS NARROWER THAN THAT DISAGREEMENT SUGGESTS, and the first draft of this entry
overstated it.** `mir_needs_drop` has exactly ONE consumer: V-COPY-1's rule that `VecClear` requires
a non-droppable element type, because clearing discards elements without running their glue. It does
not participate in the `Drop` terminator path at all — that path runs through `drop_plan::plan_for`
and `may_need_drop`. So the wrong answer was **latent, not active**: reachable only through a
`Vec<HostResource>`, where `clear()` would have discarded live handles without closing them, which
is exactly the leak MIR-0016 exists to prevent. A real defect and the right fix, but it was not
mislowering resource drops today, and this entry should not be read as saying it was.

**A second behavioural fix:** `ty_contains_ref` did not recurse into `MirTy::Core`'s arguments, so a
`Vec<&T>` was reported reference-free. Also surfaced by exhaustiveness.

**Three answers the wildcards were hiding, now written down rather than decided quietly:**
- `ty_needs_drop`'s `Core` arm is asymmetric — `VecIter`/`KeysIter`/`Iter` need glue,
  `CharsIter`/`SplitIter`/`ValuesIter`/`MapIter`/`FilterIter` do not. Preserved exactly as it stood.
  Whether the second group is right is a question for iterator lowering, and it is now visible.
- `ty_carries_ref` (lowering) and `ty_carries_reference` (backend) disagree on `FnPtr`: the backend
  descends into params/return, lowering calls every fn value borrow-free. Defensible — a Rust
  `fn(&T)` is higher-ranked and needs no lifetime parameter, which is all this guards — but the two
  copies had never been checked against each other.
- `drop_plan::plan_for` on `String` returns `Noop` because the backend lowers it to a Rust `String`
  whose own destructor reclaims it. Previously that answer came from the wildcard.

**The duplication is still the defect, and it is worse than CD-240 recorded.** Twelve sites are
twelve implementations of four rules: "is Copy" lives in two places, "needs drop" in **four**
(`lower::ty_needs_drop`, `lower::ty_has_user_drop_guarded`, `verify::may_need_drop`,
`verify::mir_needs_drop`), "carries a reference" in three. Each has historically been corrected
separately, after a leak, and the `mir_needs_drop` finding above is what a fourth undiscovered copy
costs. Exhaustiveness makes the next omission a compile error at every copy; it does NOT make the
copies agree — `ty_carries_ref` and `ty_carries_reference` still disagree on `FnPtr`. Unifying them
is a design change and is deliberately NOT this one, but it now has a concrete cost attached.

**Why now, and not after C7.8.** Route B (`OwnedResourceHandle`, MIR-owned exactly-once close,
`resource_type` on `HandleOut`) reshapes `MirTy`. That is the exact event that has cost six silent
leaks. This converts the seventh into a compile error before the variant is added, not after.

**BEHAVIOURAL QUALIFICATION, per the owner's ruling that an implementation-predicate fix does not
close a disagreement.** Two tests in `a11_host_resource.rs`, deliberately separate because they prove
different things:
- `vec_clear_on_a_host_resource_element_is_rejected` — the regression guard, placed at the corrected
  arm's one real consumer. `VecClear` over `Vec<HostResource>` must raise MIR-0016. **This fails on
  the pre-CD-287 code**, which is what makes it a regression test rather than a restatement.
- `the_verifier_accepts_a_drop_emitted_for_a_host_resource` — the anchor the ruling asked for: with a
  close recorded, a `Drop` terminator on a resource local is accepted by the real verifier, not
  merely planned. It passes before AND after CD-287, because the `Drop` path never consults
  `mir_needs_drop`. Its doc comment says so explicitly so it is not later mistaken for the guard.

The inverse guard the ruling suggested already exists and was left alone:
`dropping_a_resource_with_no_recorded_close_fails` (a resource with no close must not plan) and
`rejects_vec_clear_on_droppable_element` in `mir_verify.rs` (the same rule for a user `Drop` type).

EVIDENCE: `cargo check --lib` clean across all twelve sites — which for this change is the load-
bearing check, since a missed variant is a compile error and nothing else would report one.
`cargo test --test a11_host_resource`: **38 passed / 0 failed**, including both new tests. `cargo fmt
--check` clean; `cargo clippy --test a11_host_resource` clean. Full workspace verification left to
CI: the shared checkout currently holds a parallel session's in-flight WP-IO.1 edit that does not
compile (`c788_provider_api_manifest.rs` calls an undefined `pkg`), so a workspace run right now
would report their breakage, not this change's.
FILES: starkc/src/mir/{mod,lower,verify,drop_plan}.rs,
starkc/src/backend/generated_rust/emit_types.rs, starkc/tests/a11_host_resource.rs (its doc comment
described the wildcard in the present tense), COMPILER-STATE.md.
NOT MINE, PRESENT IN THE SAME SHARED CHECKOUT: `stark-io/`, `starkc/src/provider_registry.rs` and
`WP-IO.1-Minimal-Native-File-IO.md` belong to a parallel session's WP-IO.1 work and must not be
staged with this change. The `native_build.rs` refactor that was also in this tree was that
session's, and landed as CD-286 (`manual_find` clippy fix) while this entry was being written —
which is why this is CD-287.

## Gate C9 — OPEN (2026-07-31)

Active WP: C9 Part A closeout.

Part A is complete for C9.0 baseline/governance, C9.1 extension-isolation conformance, and C9.2
tensor/ONNX provider map. Part B is blocked pending second-artifact evidence; no provider
generalisation is authorised from ONNX alone.

Current policy recorded for C9.1: Core-only is the default; `tensor` must be explicitly enabled;
unknown and duplicate extension configuration is rejected at CLI/internal/LSP configuration
surfaces.

## DEV-012 — interactive editor validation, partially recorded (2026-07-31)

The first interactive record this deviation has ever had. It has been open since Gate C1 with the
text "VS Code extension UI never interactively verified", and what closed that gap is a person
using the editor rather than another protocol test.

**Setup.** Extension `starklang.stark-language@0.2.0`, built from this tree with esbuild, packaged
and installed with `--force`. Compiler `662842c`, binaries installed to `~/.local/bin`
(`stark`, `starkc`, `starkide`). VS Code 1.130.0 on macOS 26.5.2 arm64. Workspace
`~/Desktop/stark-extension-test`, a real STARK package that checks and runs.

**One thing worth carrying forward:** the extension defaults `stark.compiler.path` to `starkc`, and
VS Code launched from Finder does not inherit a shell `PATH` — so a `~/.local/bin` install is
invisible to it unless the setting is given an absolute path. The test workspace pins it. Anyone
validating from a fresh install will hit this first.

**Confirmed by the owner, interactively:**

| Feature | Result |
| --- | --- |
| Hover | works |
| Go-to-definition | works |
| Find-references | works |

**Not exercised, and therefore still unverified in an editor:** rename, diagnostics (on save and on
type), formatting, completion, signature help, document symbols, semantic tokens. Each is covered by
protocol tests only, which is what DEV-012 exists to distinguish from.

**Gate C8 remains CANDIDATE-COMPLETE, and closing it is the owner's call.** Its exit report names
missing interactive validation as the single reason it is not closed. That reason is now partly
answered: the three core navigation queries are confirmed against real compiler analysis. Whether
three of ten features is the record the gate's claim requires is a governance decision, not one this
entry makes.


## WP-C7.9 — CLOSED (CD-275…CD-278)

Three-engine adversarial conformance correction. Detail archived at
`STARKLANG/docs/compiler/state-archive/C5-C7-closed-detail.md`.

## Gate C6 — CLOSED (CD-183)

Closure detail archived at `STARKLANG/docs/compiler/state-archive/C5-C7-closed-detail.md`.
Gate-level evidence remains in the C6 exit report.

## WP-C7.8 — CLOSED with Gate C7 (CD-274)

Native capabilities: providers, capability declaration, resource lifecycle, loopback. Detail
archived at `STARKLANG/docs/compiler/state-archive/C5-C7-closed-detail.md`. The live evidence is
the `C7.8 Native Capabilities` CI workflow, which runs on three platforms every push.

## Gate C7 — CLOSED 2026-07-31 (CD-274)

Native compilation through generated Rust, debug and release, on three Tier-1 platforms, over a
qualified standard-library subset. **It closed without a steady-state performance claim** (CD-273),
and that limit is part of the closure rather than a footnote to it.

The six ruling sections recording how it got there — CD-195 exit assessment, CD-261 reassessment,
CD-262 qualification-blocked ruling, CD-264/CD-265 on DEFECT-C788-LOOP-TEMP, and CD-273/CD-274 —
are archived verbatim at `STARKLANG/docs/compiler/state-archive/C5-C7-closed-detail.md`.

## Position — superseded

The C5-era position block (522 lines, last accurate 2026-07-23) is archived at
`STARKLANG/docs/compiler/state-archive/C5-C7-closed-detail.md`. **The live position is the
`# Current position` block at the top of this file.**

## Repository baseline
- Last completed transition: WP-C2.13 (Gate C2 exit and Core v1 semantic freeze). Verdict
  **CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS** — all 24 high-cost open
  questions (CORE-Q-001..024) approved, 166-row completeness inventory has zero
  absent/contradictory/unclassified rows (6 remain `pending-owner-approval` governance
  bookkeeping only, behavior already implemented/tested), 33 deviations closed this gate
  (seventeen WP-C2.2 runtime-semantics defects, six WP-C2.11 items, DEV-036, seven
  post-WP-C2.11 correction-pass items, DEV-053/054), 8 remained open and non-soundness-relevant
  at gate close (current open set after the post-Gate-C2 correction brief: DEV-005/010/011/012,
  DEV-017 partial, DEV-060 — see the open index below).
  Full report: `starkc/docs/compiler/C2-exit-report.md`. C3-entry is the active transition
  before WP-C3.1.
- Transition base commit: `c268d7c` (`Add systems ecosystem roadmap`), after the post-Gate-C2
  correction-brief commit that resolved DEV-051, DEV-052, and DEV-055 and opened DEV-060.
- Amendment base commit: `60b49e2` (`CD-021 function-value native validation...`) — the head
  this state revision was written against. (Field renamed from "Current committed head" under
  CD-022: a commit cannot record its own SHA, so that framing was permanently one behind;
  the live head is always `git log`, never this file.) Commit only on explicit user request.
- Rust toolchain: `starkc/rust-toolchain.toml` pins `channel = "stable"` (no version number, tracks
  stable) with `rustfmt`/`clippy` components. Active environment measured: `cargo 1.93.0
  (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607 2026-01-19)`. `starkc/Cargo.toml` declares
  `rust-version = "1.85"` (crate MSRV). The Gate-5 *generated deployment host* (not `starkc`
  itself) separately requires Rust 1.88 due to the `ort` crate's MSRV
  (`starkc/docs/gate5-backend-decision.md:107-110`) — this does not raise `starkc`'s MSRV.
- Latest verified code baseline: `cargo test --workspace --all-targets --all-features`
  (starkc/, post-CD-025, 2026-07-19):
  **597 passed, 0 failed, 2 ignored** (594 → 596 from DEV-060's fix: one new typecheck
  regression test, one new interp execution test, one existing test rewritten in place; 596 →
  597 from CD-025's `corpus_lock_matches_frozen_snapshot` integrity test)
  across **4 unittest binaries** (`src/lib.rs`,
  `src/main.rs`, `src/bin/stark.rs`, `src/bin/starkide.rs`) **+ 32 integration-test files**
  (`find starkc/tests -maxdepth 1 -type f -name '*.rs' | wc -l`,
  re-counted against the
  post-WP-C2.7 tree — the
  "3 unittest binaries + 31/32 files" figure quoted in several prior session records below was
  never actually verified against `ls`/`cargo test`'s own "Running ..." lines and had drifted;
  not chasing down exactly which prior WP's arithmetic first went wrong, since that would need
  checking out old commits for no real benefit — this line is now the corrected, directly-counted
  baseline going forward). Up from 383/0/2 at Gate C0 close (file count at that point not
  re-verified for the same reason). WP-C1.1 added `span_integrity.rs` + 12 tests, WP-C1.2 added
  15 more across `resolve.rs`'s inline tests and `gate2_package.rs`, WP-C1.3 added 8 more across
  `typecheck.rs`'s and `interp.rs`'s inline test modules, WP-C1.4 added 11 more across
  `gate2_valid.rs` and `gate3_execution.rs`, WP-C1.5 added 21 more to `gate2_valid.rs`, WP-C1.6
  added `conformance_report.rs` (new file) + 4 tests.
  Both ignored tests are
  intentionally opt-in (a checksum-pinned live ONNX artifact test in `tests/gate4_onnx.rs`, and
  a live-ORT-download inference test in `tests/gate5_codegen.rs`). Full per-file breakdown
  recorded in `starkc/docs/dev/compiler-map.md` (WP-C0.1; not re-regenerated for the WP-C1.1/
  C1.2/C1.3 deltas — see that file's own scope note).
  Latest recorded validation also has `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`, and conformance
  validation/reporting clean.
- Core spec revision: `STARKLANG/docs/spec/` files 00-07 plus
  `CORE-V1-ABSTRACT-MACHINE.md` and `CORE-V1-FUTURE-BOUNDARIES.md`, normative per
  `CLAUDE.md`. Spec fixture corpus:
  `STARKLANG/tests/spec-fixtures/manifest.toml`, 113 entries (parse-pass 65,
  semantic-error 16, notation 27, lex-pass 4, parse-fail 1). WP-C2.7 removed 28 stale,
  duplicative memory-model examples and now contains 13 abstract-machine adversarial examples
  after its correction pass. WP-C2.8 appended five static-semantics review fixtures without
  renumbering existing examples.
- Tensor spec revision: `STARKLANG/docs/extensions/Tensor-Model-Types.md` (extension `tensor`
  v0.1), `AI-Extensions.md` (non-normative sketches).
- Conformance DB: `STARKLANG/conformance/core-v1-coverage.toml`, 59 `[[rule]]` entries.
  **Integrity-audited under WP-C0.3 (2026-07-17)**: no duplicate rule IDs, no references to
  nonexistent spec chapters (both now mechanically checked, see `starkc/scripts/
  check-conformance.py`). Post-correction counts: 53 implemented, 6 partial, 0 missing.
  Pre-correction counts (53 implemented, 2 partial, 4 missing) were **stale**, not accurate — see
  DEV-002. `starkc/scripts/check-conformance.py` now also warns (non-fatal) on `missing` entries
  that still carry a `source`/`tests` field and on likely-semantic-rejection rules with zero
  recorded tests, as a heuristic staleness signal for future audits. Known representational gap:
  the schema's single `tests` array does not distinguish positive from negative test evidence, so
  Charter rule 15 ("positive and negative evidence travel together") cannot be mechanically
  verified from this database alone for every rule. **WP-C1.6** (closed 2026-07-18) addressed
  this with a richer schema (`positive_tests`/`negative_tests`, function-level `path::function`
  citations) and populated it for 20 of 59 rules with real evidence; the remaining 39 still rely
  on the single aggregate `tests` citation and are reported as "unclassified" by the new
  `generate-conformance-report.py`, not silently treated as verified — see DEV-017.
  **Coverage percentages remain provisional**: "implemented" status
  for any individual rule is not re-verified at Core v1 rule-completeness depth until WP-C1.x; see
  governing rule in `COMPILER-CHARTER.md` §1.5 rule 14 and the explicit no-percentage-trust
  statement this state file and the WP-C0.5 exit report both carry.
  WP-C2.6 adds `STARKLANG/conformance/core-v1-rule-id-map.toml`, a mechanically validated
  transition from every one of those 59 broad IDs to the stable granular inventory IDs. It does
  not inherit broad implementation status; C2.11 must classify evidence and status per granular
  rule.

## Current compiler pipeline
- Source -> lexer (`lexer.rs`) -> parser (`parser.rs`) -> AST (`ast.rs`) -> resolve (`resolve.rs`)
  -> HIR (`hir.rs`) -> type/flow/borrow check (`typecheck.rs`, `flow.rs`, `borrowck.rs`) ->
  interpreter (`interp.rs`).
- Extension front end: `extensions/tensor/` (dim algebra, tensor/model types), gated by
  `options.rs` (`LanguageOptions`/`ExtensionSet`).
- Artifact path: `onnx/` (bounded ONNX signature import/verify, no graph execution) ->
  `deploy/` (Gate-5 lowering to a generated Rust host calling ONNX Runtime via the `ort` crate).
- Additional entry points (three separate binaries, non-overlapping command sets — see
  `starkc/docs/dev/compiler-map.md` for full detail):
  - `starkc` (`main.rs`): `check`, `run`, `parse`, `lex`, `lsp`, `import`, `verify`, `deploy`.
  - `stark` (`bin/stark.rs`): `check`, `build`, `run`, `test`, `fmt`, `doc`.
  - `starkide` (`bin/starkide.rs`): interactive terminal IDE, no CLI subcommands.
  - `lsp/` module backs `starkc lsp`; `formatter/` backs `stark fmt`; `doc_gen/` backs
    `stark doc`; `test_runner/` backs `stark test`.
- **Known duplication requiring WP-C0.1 tracing**: `starkc` and `stark` each implement their own
  `check`/`run`, and neither binary exposes the full command surface — a caller needing
  `deploy`/`verify`/`import`/`lsp` together with `build`/`test`/`fmt`/`doc` must invoke both
  binaries. Whether these two `check`/`run` implementations share one pipeline or have drifted is
  unverified; resolve in WP-C0.1 (this is exactly the "shared vs. duplicated entry points"
  question that WP is scoped to answer, and directly bears on Charter rule 18 — cross-tool
  convergence).

## Decision log — append-only
- CD-001 [WP-C0.0] Adopted the "C0-C10" gate numbering from
  `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet.md` as a **new, independent**
  sequence, not a renumbering of the repo's pre-existing (non-prefixed) Gate 1-7 track. The two
  numbering systems now coexist; `COMPILER-ROADMAP.md` carries a note at its top explaining the
  relationship. Rationale: the brief's own gate definitions (front end conformance closure,
  reference execution contract, compiled-language decision spike, MIR, native backend, language
  services, extension isolation, release qualification) do not map one-to-one onto the old
  gates, which were scoped around a single tensor/ONNX vertical-slice demonstrator rather than
  general Core conformance. Renumbering the old track retroactively would rewrite closed
  historical evidence, which Charter §1.5 rule 2 and WP-C0.2 ("do not rewrite historical gate
  evidence to match later implementation") forbid.
- CD-002 [WP-C0.0] Recorded that the strategic question Gate C3 (Compiled-Language Decision
  Spike) exists to answer has **already been examined once**, under the old gate track, and
  closed with a non-GO outcome:
  - `starkc/docs/gate6-memo.md`: Decision **REVISE** (owner-confirmed 2026-07-16) — comparator
    evidence was 5/5 vs 2/5 defects caught pre-inference against Python/ORT baseline, and parity
    (5/5 vs 5/5) against "the strongest typed-Rust host" comparator; recommendation was to
    re-scope the demonstrator, not GO or STOP outright.
  - `starkc/docs/gate7-decision.md`: Decision **RETAIN AS RESEARCH LANGUAGE** (owner-confirmed
    2026-07-16), tensor-track technical verdict POSITIVE, tensor productisation verdict DEFER,
    language thesis UNRESOLVED. Explicitly authorizes only a `stark verify` external-validation
    track as next work and states "No LSP work or language expansion is authorized" (superseded
    for LSP specifically by the subsequent WP8.1-8.5 work, all committed after gate7-decision.md
    per `git log`; that expansion was evidently owner-authorized outside this decision doc's
    text, but the state file flags the textual contradiction for WP-C0.2 to reconcile formally).
  - Disposition: Gate C3 must treat gate6-memo.md/gate7-decision.md as **directly relevant prior
    evidence about interpreter-vs-native tradeoffs**, not reopen the question from zero. This is
    scoped as a C3-entry consideration, not a C0 decision — C0 does not skip ahead of C1/C2. Set
    `Conditional tracks: Native=deferred` above to reflect that the most recent owner decision on
    a related (ONNX-vertical) native-deployment question was non-GO; C3 will need fresh evidence
    for the *general* Core compilation question, which the old gates never tested (old Gate 5's
    "native" path is code generation to a *generated Rust host*, not general Core-to-native
    compilation — it has no bearing on scalar/loop/struct/enum native lowering that C3-C7 would
    need to evaluate).
- CD-003 [WP-C0.0] Confirmed two stale root-adjacent status documents exist and require
  correction under WP-C0.2 (not fixed in this WP — C0.0 is bootstrap-only, per its own "Done
  when" — but recorded now so the fix isn't lost):
  - `CLAUDE.md:110-113,137` states "Gates 1-3 are closed... next: Gate 4" — contradicted by
    `starkc/docs/gate4-exit.md` through `gate7-decision.md`, all closed, and by the root
    `README.md`'s own delivery-gates table which correctly lists all seven gates as
    Complete/Decision-recorded.
  - `starkc/README.md:4` states "Gate 4 (tensor front end and ONNX signatures) is complete" with
    no mention of Gates 5-7, and its module "Layout" table omits `deploy/`, `lsp/`, `formatter/`,
    `doc_gen/`, `test_runner/` — five of the crate's fifteen `pub mod`s are undocumented there.
  - `STARKLANG/docs/PLAN.md:5` says "The roadmap defines what evidence advances the project
    (Gates 1-6)" and has no Gate 7 section, while `STARKLANG/docs/ROADMAP.md` has a full,
    evidence-cited Gate 7 section matching `gate7-decision.md` exactly. PLAN.md was last
    substantively updated for Gates 1-5.
  - By contrast, root `README.md` is internally consistent with all seven gate exit/decision
    docs and is the most reliable of the pre-existing status documents.
- CD-004 [2026-07-17, outside any single WP — a mid-session governance update triggered by a new
  source document] The user provided a revised master brief,
  `STARKLANG/docs/STARK-Compiler-Build-Brief-Revised-Sonnet(1).md` (title: "... (Native Compiler
  Required)"), which supersedes the original `STARK-Compiler-Build-Brief-Revised-Sonnet.md` this
  track was bootstrapped from (WP-C0.0). **This is a real, deliberate scope change, not a
  clarification**: the original brief framed Gate C3 as an open, evidence-based question — GO,
  REVISE, DEFER, or STOP on whether STARK needs a general native Core compiler at all, explicitly
  naming DEFER/STOP as valid, non-failure outcomes. The revised brief removes that question
  entirely: general native Core compilation is now a **mandatory** completion requirement (new
  §1.2 "Guaranteed compiler completion state" in `COMPILER-CHARTER.md`), Gate C3 is renamed
  "Native Compiler Architecture and Backend Selection Spike" and now only selects *how* (backend
  strategy: SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), never *whether*. An
  interpreter-only release is explicitly "not an allowed C3 completion outcome," and Gates
  C4-C7 change from *conditional* on a GO decision to *mandatory* after C3 selects an
  architecture. Diff confirmed Gates C0-C2 and C6/C8/C9 are textually unchanged; the change is
  scoped to §1 (framing/rules), the `COMPILER-STATE.md` template in §2.4, Gate C3's outcome
  vocabulary, Gate C4/C5's conditionality headers, Gate C10's release-statement requirements,
  §4's dependency map (native path folded into the single mandatory path, no more separate
  "native compiler path" branch), §5.3's gate-decision vocabulary (adds `BLOCKED`), §7's session
  budget (single ~57-86 session mandatory-path figure, replacing the old bifurcated
  "interpreter-only 31-48 / full-native 58-88" framing), and §8's strategic-outcome list.
  Regenerated `COMPILER-CHARTER.md` and `COMPILER-ROADMAP.md` in full from the new brief text
  (same extraction method as WP-C0.0) rather than hand-patching, to guarantee fidelity; updated
  this file's Position-line schema (`Mandatory compiler path: Core=/MIR=/Native=` +
  `Optional tracks: ArtifactInfra=/TensorExpansion=`, replacing the old `Conditional tracks:`
  line) and renamed the `## Backend decision` section to `## Native backend selection` with the
  new status vocabulary (`not evaluated | SPIKING | SELECTED | REVISE | BLOCKED` + a `Selected
  strategy` field, replacing `GO | REVISE | DEFER | STOP`). CD-002's own text is **not** rewritten
  (append-only) but is now superseded in one specific respect: its framing that "Gate C3 will
  need fresh evidence for the general Core compilation question" remains true, but its implicit
  suggestion that a DEFER/STOP-style outcome remains available for general native compilation no
  longer holds — see the correction notes added inline in `COMPILER-CHARTER.md` §1.5 and
  `COMPILER-ROADMAP.md`'s header relationship note, both of which point back to this entry.
  Gates C0-C2 work already completed (this entire session, through WP-C1.2) required **no
  rework** — none of it touched native-compilation framing. Both brief files are left on disk
  as-is (the original for historical reference, the "(1)" revision as the new live source); this
  is a content decision, not a file-management one, and neither file was deleted or renamed.
- CD-006 [2026-07-18, WP-C1.5] **SUPERSEDED 2026-07-26 by CD-139 — succession of authority, NOT
  reversal on the merits. Do not cite this decision for float behaviour.** It arbitrated wording in
  `03-Type-System.md` that WP-C2.9 replaced the same day (08:47 → 17:29) with the explicit paired
  rules NUM-INT-DIV-001 (integer zero division traps) and NUM-FLOAT-OP-001 (floating zero division
  does not). The sentence it read is gone from the spec; its integer half survives under
  NUM-INT-DIV-001. Original entry follows.
  Resolved a spec-internal tension in `03-Type-System.md`'s Numeric
  Semantics section, found during the WP-C1.5 audit and flagged to the user rather than resolved
  unilaterally (CE2-shaped): the section states both "Division or modulo by zero is a runtime
  error and MUST trap" and, in an adjacent bullet, "Floating-point operations follow IEEE-754
  semantics (NaN, +/-Inf)" — the current implementation traps on `0.0 / 0.0` (a literal reading
  of the first bullet), which is in tension with the second bullet's implied NaN/Inf behavior for
  floats specifically. **User decision: keep trapping (current behavior); no code change.** The
  "MUST trap" rule applies uniformly across all numeric types including floats; the IEEE-754
  bullet is read narrowly (governing ordinary float arithmetic results — e.g. overflow producing
  `+Inf`, not division by zero specifically, which STARK treats as an error condition like any
  other div-by-zero). No spec or code edits made under this decision; recorded so the question is
  not re-litigated in a future WP. `interp.rs`'s Float `BinOp::Div`/`Rem` arms are unchanged.
- CD-007 [2026-07-18, WP-C2.1] Settled a spec-silent gap found while writing
  `STARKLANG/docs/compiler/reference-execution.md` §1: the spec addressed almost no
  subexpression evaluation order (binary operands, call arguments, method receiver-vs-arguments,
  aggregate-literal fields, assignment lhs-vs-rhs, index base-vs-index). Flagged to the user
  rather than resolved unilaterally (CE1/CE2-shaped, per WP-C2.1's own scope-control answer).
  **User decision: adopt the interpreter's observed left-to-right order as normative.** Added a
  new "Evaluation Order (Core v1)" subsection to `03-Type-System.md` (after "Operators and
  Traits," before "Copy and Drop") stating: strict left-to-right evaluation for binary operands
  (non-short-circuit), call arguments, struct/tuple/array literal fields, and index base-before-
  index; short-circuit semantics for `&&`/`||` (already spec-derivable, now stated explicitly);
  condition/scrutinee-before-branches for `if`/`match` (also already spec-derivable); receiver-
  before-arguments for method calls; and right-hand-side-before-left-hand-side-place for
  assignment (explicitly flagged as the most surprising rule, since many C-family languages
  evaluate the LHS place first). `STARK-Core-v1.md`/`.html`/`.pdf` regenerated in the same change.
  No interpreter code changes needed — `interp.rs` already implements exactly this order
  throughout (confirmed during WP-C2.1's own drafting); this decision closes the spec-vs-
  implementation gap from the spec side, not the code side.
- CD-008 [2026-07-18, WP-C2.1] Settled a second spec-silent gap found in the same document, §10.3:
  `HashMap`/`HashSet` iteration order was unaddressed by any normative spec text, while the only
  related prose (`06-Standard-Library.md`'s non-normative "Performance Notes" — "HashMap<T> uses
  open addressing with Robin Hood hashing") implied unordered iteration, in tension with the
  interpreter's actual `BTreeMap`/`BTreeSet`-backed fully-sorted-deterministic behavior. Flagged
  to the user rather than resolved unilaterally (CE1/CE2-shaped). **User decision: adopt
  sorted-deterministic (ascending key order) as normative.** Added a new "Iteration Order (Core
  v1)" subsection to `06-Standard-Library.md` immediately after the `HashSet<T>` API block,
  stating `HashMap::keys`/`values`/`iter`, `HashSet::iter`, and `for`-loops over either MUST visit
  entries in ascending key order per the key type's `Ord` impl, regardless of internal storage
  strategy. Reworded the "Performance Notes" line to remove the implication of unordered
  iteration (now frames storage strategy as implementation-defined but explicitly subordinate to
  the iteration-order requirement — an open-addressing implementation would need to sort at
  iteration time to conform). `STARK-Core-v1.md`/`.html`/`.pdf` regenerated in the same change
  (shared with CD-007). No interpreter code changes needed — `interp.rs`'s `BTreeMap`/`BTreeSet`
  representation already satisfies this rule exactly.
  **Correction (CD-009, same day, external review):** CD-008 as originally written is broken —
  `HashMap<K, V>`/`HashSet<T>` only bound `K`/`T: Hash + Eq` (confirmed:
  `06-Standard-Library.md` lines 271, 293), never `Ord`, so "ascending key order per the key
  type's `Ord` impl" can require an implementation that isn't guaranteed to exist. It is also
  inaccurate to describe the interpreter as already satisfying this rule: `interp.rs`'s
  `BTreeMap`/`BTreeSet` sort by `Value`'s own internal structural `Ord` (a Rust-level total order
  over the runtime representation), not by dispatching to the STARK key type's own `Ord`
  implementation (which, per DEV-027 found in this same WP, cannot even be written today). CD-008
  is left as-is above (append-only — a record of what was decided, even though wrong), superseded
  by CD-009.
- CD-009 [2026-07-18, WP-C2.1 correction pass, external review] Corrects CD-008. **User decision:
  `HashMap`/`HashSet` iterate in first-insertion order**, not sorted-by-key order — no `Ord` bound
  needed (matches the actual `Hash + Eq` bound), still fully deterministic. Reworded
  `06-Standard-Library.md`'s "Iteration Order (Core v1)" subsection accordingly (insert appends to
  iteration order; re-inserting an existing key keeps its position; remove-then-reinsert moves it
  to the end) and reworded "Performance Notes" to match. `STARK-Core-v1.md`/`.html`/`.pdf`
  regenerated. **This is now a real, confirmed WP-C2.2 deviation, not a no-op**: `interp.rs`'s
  `BTreeMap`/`BTreeSet` representation does not track insertion order at all (it sorts by
  structural `Value::Ord`), so it does not satisfy the corrected rule — recorded as DEV-032.
- CD-010 [2026-07-18, WP-C2.1 correction pass, external review] Refines CD-007. **User decision:
  keep "the method receiver evaluates before any argument" as normative** (matching user-defined
  method dispatch and common OOP convention), rather than changing the rule to match a narrower
  implementation detail. However, re-reading `interp.rs::call_core_method` (the dispatch path for
  builtin/stdlib-type methods — `Vec`, `String`, `HashMap`, etc., as opposed to user-defined
  nominal types) during the same review found it evaluates argument expressions *before*
  resolving the receiver — the exact opposite of `call_method`/`call_user_method`'s order for
  user-defined types. CD-007's original claim "no interpreter changes are needed... `interp.rs`
  already implements exactly this order throughout" is therefore **incorrect** for this one path;
  left as-is above (append-only), corrected here. Recorded as a new WP-C2.2 deviation, DEV-033 —
  `call_core_method` needs to resolve the receiver before evaluating arguments, to match the now-
  confirmed-normative rule and `call_method`'s own behavior for user-defined types.
- CD-011 [2026-07-18, WP-C2.1 correction pass, external review] DEV-029 (struct/enum field drop
  order is alphabetical-by-field-name, not declaration order) was recorded as a confirmed
  deviation, but `05-Memory-Model.md`'s Drop Order section only ever demonstrated reverse-
  declaration-order for sibling `let` bindings — it never actually stated a rule for a struct's
  own field-internal drop order; DEV-029's framing called reverse-declaration-order "the only
  coherent extension" (an inference, not a citation). Flagged to the user rather than left as an
  inferred deviation (CE1/CE2-shaped). **User decision: amend the spec to state it explicitly.**
  Added two sentences plus a short example to `05-Memory-Model.md`'s Drop Order section extending
  the existing rule to struct/enum-variant fields (reverse declaration order). `STARK-Core-v1.md`/
  `.html`/`.pdf` regenerated (this addition included a new `stark` code block, requiring a spec-
  fixture re-triage: `05-Memory-Model__22.stark` through `__27.stark` renumbered to `__23`
  through `__28`, new `__22.stark` triaged `parse-pass`/`program`; verdict census updated to 68/
  122; `extract-spec-examples.sh` confirms the manifest is back in sync). DEV-029 is now a
  confirmed, spec-backed deviation rather than an inferred one — its ledger entry updated to cite
  the new normative text instead of describing the rule as inferred.
- CD-012 [2026-07-18, WP-C2.7] Approved CORE-Q-006 and the normative Core v1 abstract machine.
  Runtime authority moves from scattered operational prose to
  `CORE-V1-ABSTRACT-MACHINE.md`. Evaluation is exactly once; assignment evaluates RHS before
  destination, installs the new value before destroying the old; normal early transfers clean
  exited scopes; language traps abort without unwinding, including during destination resolution
  and partial aggregate construction. Reference identity is abstract and survives legal
  ownership/call transfers; returned receiver-derived references designate caller objects and
  range slices are live views. CORE-Q-020 is approved only for runtime ownership/destruction of
  existing Core patterns, and CORE-Q-017 only for the language-trap boundary; C2.8/C2.9 retain
  their remaining portions. This decision defines semantics but deliberately defers compiler/
  interpreter alignment and adversarial rule evidence to C2.11.
- CD-013 [2026-07-18, WP-C2.7 correction] Corrected CD-012's CORE-Q-006 approval scope.
  CORE-Q-006 is approved for runtime abstract-machine semantics only; static place legality,
  borrow coexistence/regions, temporary-reference escape, and returned-reference legality remain
  pending under C2.8. This supersedes only CD-012's phrase "Approved CORE-Q-006", not its
  runtime decisions or its C2.11 implementation-alignment deferral.
- CD-014 [2026-07-18, WP-C2.8] Approved the Core v1 static-semantics freeze. Type aliases are
  transparent; values are finitely sized with only `str`/`[T]` unsized behind references;
  inference is deterministic and function-local; trait selection is source-order-independent
  with no specialization; borrows have conservative lexical regions and no temporary
  extension; patterns use deterministic exhaustiveness/usefulness analysis; and constants use
  a closed side-effect-free evaluator. Standard-library hooks are recognized by canonical item
  identity only. CORE-Q-002/003/004/005A/006/007/015/020 are approved. CORE-Q-005 is partially
  approved because C2.9 still supplies canonical package/version identity. Numeric results,
  float trait participation, layout-query results, and resource-limit classification likewise
  remain C2.9 inputs. Compiler/interpreter alignment and granular evidence remain C2.11 work.
- CD-015 [2026-07-18, WP-C2.9] Approved the numeric, target, text, process, package, and
  standard-library contract freeze. Integers are fixed-width and checked; primitive floats use
  reproducible IEEE operations but do not implement `Eq`/`Ord`/`Hash`; text is valid UTF-8 with
  byte offsets and Unicode 15.1 casing. Package identity is relocation-stable and lock-backed,
  with one selected version per source/name/major line. Only `size_of`/`align_of` expose
  target layout and Core promises no ABI. Four no-argument `main` signatures have deterministic
  status/stream mappings. `core-min` is mandatory and `std-full` is optional but indivisible.
  Resource, compiler-limit, API-error, language-trap, and host/process failures are distinct.
  CORE-Q-005, Q008–Q014, Q017–Q019, Q021, Q023, and Q024 are approved; alignment remains C2.11.
- CD-016 [2026-07-18, WP-C2.10] Approved CORE-Q-016 and the Core v1 future-extension
  boundary. Core execution is safe and single-threaded; capturing closures, explicit lifetime
  syntax/reference fields, trait objects, concurrency, macros, unsafe, and general FFI remain
  outside Core. Future callables must preserve ownership/capture/Drop semantics. Host access is
  limited to metadata-bound approved native providers with explicit identity, integrity, ABI,
  target, provenance, capability, and verification. Extensions require explicit stable
  identity/version enablement and cannot change Core-only behavior. No future feature is
  implemented by this decision; C2.11 owns exclusion/isolation enforcement evidence.
- CD-017 [2026-07-18, C2.8/C2.9 correction] Clarified nine pre-C2.11 freeze points.
  Generic fields may instantiate with references and recursively propagate borrow provenance;
  constant patterns never invoke user `Eq`; positive bounds never prove unifying impl heads
  disjoint. Canonical package names are distinct from identifier-valid aliases, each alias
  selects exactly one major line, and all packages remain library-importable while executable
  mode selects the root `main`. Floating `**` is rejected. Standard hash values use canonical
  FNV-1a encodings and primitive Display bytes are exact. `std-full` freezes availability and
  explicitly stated behavior only; unstated method edge cases are not conformance claims.
- CD-018 [2026-07-18, roadmap amendment before WP-C3.1] Adopted the post-C2 roadmap correction
  brief without replacing the core C3-C7 sequence. Inserted mandatory `C3-ENTRY — Native
  Readiness and Carry-Forward Closure` before WP-C3.1; made pending-owner-approval rows,
  DEV-051/052/055 ownership, WP-C2.12 generated-corpus/cross-backend transfer, versioned corpus
  freeze, and native-path CI baseline explicit. Strengthened C3.1's frozen workload with
  generics, trait dispatch, default trait sibling calls, references/slices, Drop-bearing trait
  dispatch, opaque host resources, and provider-boundary file I/O. Added Native Provider ABI
  v0.1 to C5.1, removed C5.4's "where supported" generic-call escape hatch, introduced platform
  tiers, added real systems workloads to C7 measurement, and created
  `STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md` with S0-S7 plus the post-C6 P1 Native Systems
  Baseline checkpoint. This is a sequencing and evidence-governance amendment; it does not
  reopen C2 or change Core v1 semantics.
- CD-019 [2026-07-19, C3-entry follow-up amendment] Tightened the post-C2 roadmap amendment
  before WP-C3.1. DEV-060 is now owned by C3-ENTRY and must be disposed before the workload
  freeze. P1 now gates C7.5/C7.7 closure and is required for Native Systems Preview and
  STARK v1 General-Purpose Stable claims, while Core v1 Compiler Stable may describe compiler
  maturity without claiming systems-platform maturity. The C3 provider/resource experiment is
  explicitly disposable and non-normative; C5.1 remains the first stable Native Provider ABI.
  Systems S6 is split into joint concurrency tracks for language proposal, compiler
  implementation, runtime/provider work, and ecosystem validation. `COMPILER-STATE.md`'s
  load-bearing header now points at `c268d7c`, the 594/0/2 verified code baseline, and the
  remaining C3-entry blockers.
- CD-020 [2026-07-19, C3-entry governance-repair pass — no semantic or compiler change]
  Repaired the governance surface before C3-ENTRY closure work begins. (a) Created
  `work-packages/WP-C3-ENTRY.md` — the transition's executable WP: named exit artifact
  (`starkc/docs/compiler/C3-entry-exit.md`), mechanical corpus-freeze definition
  (`corpus.lock`, SHA-256 per file, version-bump rules), per-blocker owners, "Done when";
  roadmap C3-ENTRY section now points at it. (b) Amended WP-C4.4/C5.6/C6.5 in
  `COMPILER-ROADMAP.md` to carry their transferred WP-C2.12 generated-corpus/cross-backend
  obligations in the receiving WP text (previously stated only in the C3-ENTRY bullet list,
  invisible to the charter's minimal session-input packet). (c) CI baseline delta:
  `.github/workflows/ci.yml` commands widened to the C3-ENTRY forms, added spec-regeneration
  check (new `--check` mode in `STARKLANG/tools/build-core-spec.py`, Markdown-only since
  pandoc/weasyprint output is not byte-reproducible) and a named execution-snapshot step;
  local fmt + exec_snapshots verified green, full CI run pending. (d) Accuracy corrections:
  `KNOWN-DEVIATIONS.md` tail summary (claimed DEV-009/022/023/024 open; all four resolved by
  WP-C2.11 per their own entries — stale paragraph from C2.6 time), state header current-head
  (`c268d7c` → `9e85396`) and spec-fixture census (112/parse-pass-64 → directly re-counted
  113/parse-pass-65; evidence-inventory "121-fixture" figure also corrected), charter §1.5/§2.4
  "roadmap §5.3" dangling references (vocabulary lives in charter §5.3), charter §2.1 step 10
  commit policy (owner convention: commit only on explicit request), WP-C6.4 tier label ("Core
  v1 Stable" → "Core v1 Compiler Stable" matching the C10 release class), and a new
  "Relationship to the compiler roadmap's P1 checkpoint" section in
  `STARKLANG/docs/ecosystem/SYSTEMS-ROADMAP.md` (CD-018 described P1 as living there but the
  file never mentioned it; S5 is now explicitly identified as the P1-completing stage).
  (e) Compressed this file from 3,145 to ~700 lines per charter §2.4: deviation seed sections,
  C0/C1-era file inventory, completed follow-ups, and session records through Post-Gate-C2
  Issue 5 moved **verbatim** to `STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md`;
  decision log, conformance summary, gate exit summaries, open-deviation index, and the
  Issues 6-8 session record retained inline. Charter/roadmap edits under this entry are
  governance/bookkeeping repairs, not meaning changes to the extracted brief.
- CD-021 [2026-07-19, owner-approved roadmap amendment — function-value native validation,
  P1 trap report, release deviation sweep] Origin: an external-review debate established that
  non-capturing `fn(...) -> ...` function types are **existing frozen Core v1 capability**
  (`03-Type-System.md:198-200,999`; stdlib contract `06-Standard-Library.md:243-244,260-262,
  663-666`; `interp.rs:260` `Value::Function(ItemId)`), not a future closure feature — so the
  native path must validate them explicitly rather than leave them implicit. Three changes,
  same style/class as CD-018's workload strengthening: (a) WP-C3.1's frozen workload gains
  items 16-21 (typed function-value local; indirect invocation; `Option::map`/`Result::map`
  with a function value; function value in a struct field; cross-package function reference;
  monomorphised-generic function value with an explicit record-the-boundary fallback) — any
  item failing against the current implementation becomes a DEV entry before backend
  selection, deliberately; C4 gains explicit indirect-call ownership (WP-C4.1 MIR
  function-value constants/indirect-call representation, WP-C4.3 indirect-call signature
  verification, WP-C4.5 function-value lowering with provenance); WP-C5.1's runtime ABI list
  gains function-value/code-pointer representation, indirect calling convention,
  cross-package function-symbol identity, and function values in aggregates. (b) P1's exit
  list (roadmap §4.2) and S5's requirements (`SYSTEMS-ROADMAP.md`) gain a documented
  trap-abort operational report — deliberately trap one handler, record the effect on
  in-flight connections/resources/buffered output/process state; evidence input for any
  future fault-isolation proposal, explicitly no semantic change. (c) WP-C10.7 gains a
  release-blocking deviation sweep: every open deviation needs an owning gate/WP or a
  recorded accepted-indefinitely disposition. Related but not enacted here: the planned
  paper-only "Callable ABI and Future Closure Compatibility Spike" memo (existing-capability
  section + future-closure-compatibility section, outcomes GO/REVISE-ABI/
  DEFER-ESCAPING-BORROWS/ANNOTATIONS-LIKELY/NO-CURRENT-DESIGN) remains a separate proposal to
  be drafted before WP-C5.1; it is a recommendation, not yet approved work.
- CD-022 [2026-07-19, owner-approved follow-up amendment — external review of CD-020/CD-021
  commits] Three changes. (a) **Release-class coherence repair, preserving CD-019.** External
  review correctly found two superimposed models: C7.7 requires P1 (CD-019), Core v1 Compiler
  Stable requires C7, so its "must not claim systems-platform maturity unless P1 is complete"
  conditional was vacuous and General-Purpose Stable's "+P1" added no evidence. Resolution
  keeps CD-019's C7 gating (its motive — no toy-workload performance report — stands) and
  recasts the two stable classes as differing in **claim scope, not evidence**: Compiler
  Stable necessarily carries P1 evidence but asserts compiler maturity only; General-Purpose
  Stable adds no evidence gate and is the class permitted to assert systems-platform
  maturity. The reviewer's alternative (decouple C7 from P1) was considered and rejected as a
  CD-019 reversal. (b) **Function-value property validation.** WP-C3.1 gains workload items
  22 (repeated indirect invocation through one local — spec-guaranteed by function values
  being `Copy`, `03-Type-System.md` §Copy and Drop; DEV-060 is this bug class for default
  trait methods) and 23 (`Copy` aggregate with a function-value field, copied, both copies
  invoked), plus a pre-backend-selection requirement to settle the two genuinely open
  properties — `Eq`/`Ord`/`Hash` participation and monomorphised-generic function-value
  identity — from the frozen spec or by CE1/CE2 escalation, never by MIR/ABI accident. The
  reviewer's broader open-question list (Copy? repeated calls? Drop?) was narrowed: those are
  already frozen by the spec's Copy rule. (c) **State-header field rename**: "Current
  committed head" → "Amendment base commit" (self-referential staleness by construction).
  Outstanding from the same review, not part of this entry: a demonstrated green CI run
  (requires pushing to origin; no run exists yet).
- CD-023 [2026-07-19, owner-approved] Approved all six `pending-owner-approval` completeness
  rows (`LEX-COMMENT-001`, `LEX-ERROR-001`, `STD-OPTION-001`, `STD-RESULT-001`, `STD-ITER-001`,
  `STD-VEC-001`) as-is — the behavior each row describes has been implemented and exercised
  throughout Gate C2; the gap was governance bookkeeping only (C2 exit report). All six flipped
  to `settled` in `CORE-V1-COMPLETENESS.md` (`LEX-ERROR-001` keeps its DEV-017 note — an
  evidence-citation-precision gap, not a behavior question). C2-exit-report.md gained a dated
  post-gate update note per the same convention as the DEV-051/052/055 correction, rather than
  rewriting historical gate-close evidence. This closes the first of C3-ENTRY's four blockers;
  DEV-060, the corpus freeze, and the green CI run remain open.
- CD-024 [2026-07-19, owner-approved disposition: fix now] Closed DEV-060 (repeated call to an
  un-overridden trait default method wrongly flagged as a move). Root cause: `borrowck.rs`'s
  `method_receiver` — consulted by the `Call` handler to decide whether a method receiver is
  moved, borrowed, or mutably borrowed — only ever searched `ImplItem::Fn` overrides, with no
  equivalent to `typecheck.rs::resolve_method`'s `default_fallback` (WP-C1.3/DEV-013). A call to
  an un-overridden trait default method therefore returned `None` from `method_receiver`, and
  the `Call` handler's `None => self.check_expr(*base)` arm ran instead of the `Some(Receiver::
  ..)` arms — `check_expr`'s `Path` arm unconditionally consumes (moves) any `Local`/`SelfValue`
  place, regardless of the method's real receiver kind. Fixed by adding the matching
  trait-default-body fallback to `method_receiver` itself, mirroring `typecheck.rs`'s search but
  returning the method's declared `sig.receiver`. Verified both the `&self` case (original
  repro) and a new `&mut self` companion case (the `RefMut` arm wasn't exercised by the original
  repro alone — two sequential calls must register two non-conflicting borrows, not a move), and
  that the original repro now executes with correct output twice, not just "no diagnostic".
  Full workspace suite: 596 passed / 0 failed / 2 ignored (up from 594 — one new typecheck test,
  one new interp execution test, one existing test rewritten in place from
  documenting-the-defect to asserting success). `cargo fmt --all -- --check` and `cargo clippy
  --workspace --all-targets --all-features -- -D warnings` both clean. Full writeup:
  `KNOWN-DEVIATIONS.md`'s DEV-060 entry. This closes the second of C3-ENTRY's four blockers; the
  corpus freeze (now unblocked — WP-C3-ENTRY.md's procedure required this fix to land first) and
  the green CI run remain open.
- CD-025 [2026-07-19] Froze the WP-C2.12 execution-snapshot corpus and closed C3-ENTRY. Blocker
  3 (corpus freeze): `starkc/tests/exec_snapshots/corpus.lock` created at `corpus_version =
  1.0.0`, base commit `3d12f45`, SHA-256 per corpus file (48 files: 31 `.stark` + 17 `.snap`
  incl. `metamorphic/`); lock digest
  `8cda2df5e26aa35dfc8eb222f1e073eb4ea2336297e91ecc4e62b8fbd27dc0dc`. New integrity test
  `corpus_lock_matches_frozen_snapshot` (exec_snapshots.rs) enforces hash-match + no-missing +
  no-unlisted, negatively verified (tampering one `.snap` fails it with the expected message;
  restore passes). Freeze taken after DEV-060's fix per WP-C3-ENTRY.md procedure. Blocker 4 (CI):
  green on `origin/main` @ `3d12f45`, owner-confirmed. With blockers 1 (CD-023) and 2 (CD-024)
  already closed, **C3-ENTRY is closed** — exit artifact `starkc/docs/compiler/C3-entry-exit.md`
  written, Position line flipped to `Gate: C3  Next: WP-C3.1  Blocked: none`. Any future corpus
  change must bump `corpus_version` with a dated note here; a bare `UPDATE_SNAPSHOTS=1`
  regeneration is a freeze violation the integrity test catches. No semantic or Core behavior
  change.

- CD-026 [2026-07-19, WP-C3.4, owner CE5 decision] **Backend selection: `SELECT-GENERATED`.**
  Generated Rust is the initial production backend behind verified MIR; the MIR contract is to be
  designed backend-neutrally so `SELECT-DIRECT` (Cranelift) remains a live C7-gated migration
  (charter §1.6 rule 9). Basis: WP-C3.2 (generated-Rust) reached 8/17 frozen-corpus breadth
  cheaply with zero mismatches and trap parity, the shortest/lowest-risk path to correct broad
  native compilation (charter §1.6 rule 7); WP-C3.3 (direct Cranelift) is correct and self-
  contained (no rustc dep) but owns monomorphization/layout/drop/runtime up front — the better
  *eventual* backend if the self-contained-compiler goal becomes primary, which is a C7 judgment.
  Neither `REVISE` (missing data — exe size/startup, MIR-level comparison — is inherent to
  sequencing, needs C4-C7, not a bounded pre-C4 follow-up) nor `BLOCKED` (both paths demonstrated
  correct native execution). Accepted trade: `stark build` permanently requires a rustc toolchain
  and is slower; acceptable for STARK-as-research-language, re-evaluated at C7. Full three-way
  analysis + the required architecture commitments (MIR boundary, runtime/ABI, targets, debug
  mapping, unsupported-MVP closure, why-direct-rejected-as-initial):
  `starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`. Gate C3 closes; next is
  Gate C4 (MIR contract, CE3). This decision selects a backend strategy only — it does not build
  MIR, define the MIR contract, or fix the runtime ABI (those are C4/CE3 and C5.1/CE4).
- CD-027 [2026-07-19, owner-approved: two CE freezes + a correction-pass authorization] Settled
  the two CD-022 carry-forward function-value properties and repaired the fn-value feature
  cluster found by executing CD-021 workload items 16-22 against the interpreter for the first
  time. **(a) CE1 — TYPE-FN-001** (new normative rule, `03-Type-System.md` §Function Types):
  function values are `Copy`/`Clone`, never `Drop`, and do **not** implement `Eq`/`Ord`/`Hash`
  in Core v1 (float-precedent); consequence: function-value identity is unobservable, so the
  monomorphised-generic-identity question collapses to deterministic symbol naming (C6.2), not
  language semantics. **(b) CE2 — TYPE-FN-002** (same section): a generic fn coerces to a
  concrete fn type only when the expected type fully determines every generic argument;
  semantics = instantiate at the coercion site. Combined spec regenerated; no new code blocks so
  no fixture re-triage; two granular rows (TYPE-FN-001/002) added to `CORE-V1-COMPLETENESS.md`
  (166 → 168 rows — the fn-value questions were a genuine inventory gap). **(c) Pre-C4.1
  correction pass (authorized fix-now):** DEV-061 (indirect calls through fn-value locals/params
  never executed — missing `Res::Local|SelfValue` arm in interp call dispatch; the machinery
  existed one arm below), DEV-062 (fn values not `Copy` in borrowck/typecheck — `Ty::Fn`
  explicitly misclassified against the spec's Copy list), DEV-063 (`Option::map`/`and_then`,
  `Result::map`/`map_err`/`and_then` absent from the method table despite the normative §Option/
  §Result APIs) — all three FIXED with 5 new regression tests; the semantic oracle can now
  execute workload items 16-22. One new narrow deviation found and deliberately not fixed in
  this pass: DEV-064 (undetermined-generic fn coercion accepted; TYPE-FN-002 requires rejection;
  owner C4.5). Note: these settlements landed after CD-026's backend selection but before any
  MIR/ABI work — the selection is unaffected (identity-unobservability removes the one property
  that could have differentiated the candidates' ABIs).

- CD-029 [2026-07-19, review-directed correction pass before C4.5 breadth] Four corrections
  from the external review of the C4.1-C4.4 foundation, applied before they could embed across
  complete-Core lowering. (a) **Trap provenance**: `MirRunError::Trap` was discarding
  `SourceInfo` — a right-category trap at the wrong location would have passed the C4.4
  differential; outcomes now carry full `TrapInfo`, mir.md §6 amended to make provenance part
  of the observable trap outcome, and the differential compares user-origin trap spans exactly
  against the oracle (synthetic origins compare classification). Both existing trap tests pass
  with exact span equality. (b) **TypeContext contract treatment**: formally amended into
  mir.md §2 as part of the in-memory MIR compilation unit (additive, not dump-serialized, MIR
  stays v0.1) — resolving the governance debt the WP-C4.3 record flagged. (c) **Verified-MIR
  wrapper**: `verify_program` returns `VerifiedMirProgram<'_>`; `run_program` (and eventually
  the generated-Rust backend) consumes only that — "no backend bypasses MIR validation" is now
  an API property. (d) **Differential-independence caveat**: the shared `canonical_float`
  formatter is structurally invisible to the HIR/MIR differential; claim qualified everywhere
  going forward ("no difference in lowering and MIR execution for the tested subset, with some
  runtime algorithms intentionally shared") and compensated by new spec-derived golden +
  round-trip property tests (`tests/canonical_float.rs`, incl. NaN/±inf/-0.0/notation
  boundaries at exponent 15↔16 and -4↔-5/subnormals/max-min finite). Also adopted the review's
  C4.5 increment ordering + honest maturity calibration (architecture ~90%, implementation
  breadth ~35-45%, validation ~70%) into WP-C4.5.md.

- CD-030 [2026-07-19, owner-approved disposition of the external C4.5c-head review] The review
  (written against `82211f6`, before WP-C4.5d landed) found three validation holes plus two
  warnings. Disposition: **fold the load-bearing items into C4.5e as its entry step
  (WP-C4.5e-0)** — (1) IndexProof definite-initialization dataflow (the global name→base map
  alone accepted MIR whose check ran on only one branch; slices in C4.5e build directly on the
  proof discipline), (3) V-REF-1 write-through-shared-reference rejection (MIR-0014), (4)
  partial-output-before-trap comparison in the differential (C4.5e's panic/assert paths are
  exactly where it matters; both engines now expose pre-failure stdout —
  `interp::run_with_partial_output`, `MirFailure`), plus the review-warned user-`impl Copy`
  misclassification, confirmed real (valid Copy-struct programs failed MIR verification as
  use-after-move) and fixed as **DEV-068**. **Deferred with owners** (defense-in-depth only,
  no observable-behavior risk in the current subset): frame-generation identities in the MIR
  interpreter (owner: C4.5f, before cross-package call graphs grow frames) and
  projected-move take-and-poison (owner: C4.5e proper, alongside the runtime values that make
  aggregates bigger; the unit-flag design makes the current clone-not-take unobservable, and
  the stale interp comment claiming whole-local verifier conservatism was corrected). Review's
  wording caution accepted: C4.5c externally = "top-level generic monomorphisation and static
  bound dispatch" (generic *methods*/impls and user-nominal Eq/Ord operator lowering remain
  later-increment work). The review's C4.5d checklist was already fully implemented by the
  WP-C4.5d commit it had not seen, except the two deferred items above.
- CD-031 [2026-07-19, CE3 — owner-approved MIR v0.1 Amendment A1] Approved
  `STARKLANG/docs/compiler/mir-amendment-A1-strings-runtime.md` (rev. 3) as a **narrow additive
  amendment to MIR v0.1**, runtime surface `0.1-A1` — the contract prerequisite the C4.5e-main
  body needs before lowering strings/collections. Additions, all additive (no existing construct
  reinterpreted): `Constant::Str(String)` = a decoded immutable UTF-8 literal typed `&str`
  (owned `String` only via runtime `StringFromStr`; literal identity unobservable);
  `Terminator::Trap { message: Option<Operand> }` for `panic`/`assert` messages (participates in
  every operand analysis, not just typing); `String`/`Vec`/`VecIter` become drop-elaborated
  runtime values (**always** buffer-reclaim glue; element-destructor execution conditional on
  `T`; `Vec<T>` element drop in **reverse index order**, matched empirically to the frozen oracle
  `interp.rs::drop_value`); a versioned `RuntimeFn` appendix (30 ops lowered in C4.5e + a reserved
  group activated later only by a dated enumeration bumping the surface id); one new in-memory
  `TypeContext` field (`copy_types`) and two new `MirProgram` fields (`mir_version`/
  `runtime_surface`, consumer-checked before any body); new verifier codes MIR-0015 (V-STR-1/2,
  Trap.message typing), MIR-0016 (V-COPY-1: `VecIndexGet`/`VecIterNext` require `T: Copy`;
  `VecClear` requires non-droppable `T`). Two owner-mandated honesty rules: no `RuntimeFn` ever
  runs a user element destructor (those run only at visible `Drop` terminators — `clear()` on
  droppable `T` lowers to a pop-and-drop loop; `v[i]=x` uses `VecReplace(...)->T` so the caller
  drops the old value); and a backend doing explicit reverse-order element destruction must
  suppress any automatic (Rust) element drop. Three rev cycles (rev. 1 direction approved; rev. 2
  eight corrections; rev. 3 four final corrections) recorded in the doc's §11. `mir.md` §5/§7
  carry pointers to the amendment; `MIR_VERSION` stays `0.1`. This decision approves the contract
  only — no code is written by it; the C4.5e main body implements it next.

- CD-032 [2026-07-19, owner decision — A1 iteration correction, folded into C4.5f] The
  WP-C4.5e-2 implementation surfaced that Amendment A1's by-value `VecIterNext -> Option<T>`
  ("the `for x in v` desugar") has **no STARK source trigger**: STARK has no by-value
  `for x in v`; the only iteration form is `for x in v.iter()`, and `Vec::iter()` binds the
  loop variable as `&T` (stdlib `iter(&self) -> VecIter<T>`). So all Vec/collection iteration
  in STARK is **by-reference** — an interior reference into a runtime container, which is the
  work A1 §5d already reserved and tied to C4.5f's frame-generation hardening. **Owner
  decision: fold iteration into C4.5f.** A1's by-value iteration ops are struck from surface
  `0.1-A1` (they were never added to the `RuntimeFn` enum, so `0.1-A1` as implemented is
  unchanged — no bump); by-reference iteration (`VecIterNew`/`VecIterNext` yielding
  `Option<&T>`) is a C4.5f deliverable activated by a future dated `0.1-A2` surface bump,
  alongside `VecGetRef`/`StringSubstring` interior views and the frame-generation identities.
  Amendment doc updated (rev. 4): §5c iteration rows struck, §5e reframed as the C4.5f
  carry-forward design, rev-4 log added. No code change; strings (e-1) and the Vec data
  surface (e-2) are untouched. `collection_iter__01`'s `for value in values.iter()` stays
  clean-Unsupported until C4.5f; its push/index/len half lowers under e-2.
- CD-033 [2026-07-19, owner disposition of the WP-C4.6 gate-exit audit] **Gate C4 stays
  open under the strict reading: "every normative Core construct required by C5" means the
  full normative Core language plus the `core-min` stdlib profile, NOT a representative-
  workload subset** (which would weaken the gate and let known language gaps transfer into C5
  merely because the chosen app avoids them). `core-min` is the C5 baseline, not std-full.
  **Required before C4 exit:** A1 (generic impls/assoc fns/trait methods/generic Drop), A2
  (general + nested pattern lowering), A3 (user `Eq`/`Ord` operator dispatch — `Eq` may
  proceed independently, but the `Ordering` runtime-surface amendment must be drafted for CE3
  review before the `Ord` portion is implemented), A4 (`core-min` ops: chars iteration,
  `Vec::get`/`get_mut`, slices, `size_of`/`align_of`, first-class integer ranges, and the
  `core-min`-classified Option/Result operations — via a required dated runtime-surface
  amendment), A5 (bit/shift/pow operators), A6 (non-Copy Vec iteration — the Copy restriction
  is an implementation compromise, not a language rule), A7 (normative expression forms).
  **May remain reserved beyond C4** unless separately required by the stable Core contract:
  std-full ops (`HashSet`, `HashMap::values`/`remove`, `Vec::contains`). **Front-end
  prerequisites with explicit owners:** DEV-069 is a prerequisite for the C5 multi-file/
  multi-package application claim (parallel front-end WP allowed, but C5 must not claim normal
  multi-file support while declaration spans read against the wrong file); DEV-067, `Box`
  deref, and the primitive `Ordering::cmp` surface get explicit owners and are resolved where
  `core-min` requires. **Implementation order (dependency-aware, not smallest-first):**
  (1) A5+A7 mechanical coverage; (2) A6 borrowed Vec iteration; (3) A3 `Eq`, then the CE3
  `Ordering` decision, then `Ord`; (4) A4 runtime/`core-min` surface; (5) A2 general pattern
  lowering; (6) A1 generic impl monomorphisation. The WP-C4.6 exit report is updated after
  each class with positive, negative, verifier, and HIR/MIR differential evidence; C4 closes
  only when all required classes are green and no normative Core or `core-min` construct
  required by C5 remains silently unsupported.

- CD-034 [2026-07-19, CE3 — owner-approved MIR Amendment A2 with clarifications] Approved
  `EnumRef::CoreOrdering` as the MIR representation of the prelude `Ordering` enum (three
  fieldless variants, logical discriminants Less=0/Equal=1/Greater=2 — logical MIR only, not a
  physical ABI; C5.1 owns physical layout) and the ordered-operator lowering (`<`/`<=`/`>`/`>=`
  on a non-generic user nominal → `Ord::cmp` call + discriminant compare; operands borrowed
  left-to-right, never moved). Additive; **runtime surface stays `0.1-A3`, `MIR_VERSION` stays
  `0.1`.** Five clarifications required and applied: (1) renamed "Ordering as a Runtime Value" →
  "Ordering as a Logical MIR Enum" (avoid confusion with the `RuntimeFn` surface); (2)
  discriminants logical-only; (3) recorded the **C4-open additive-amendment versioning policy**
  in `mir.md` (until C4 closes, CE3-approved additive shape amendments stay in v0.1 and are
  recorded in the contract; after C4 exit any shape change needs a version bump) and reflected
  `CoreOrdering` in the contract's `EnumRef` description; (4) `println(Ordering)` is out of A2
  (Display is A4) — the round-trip test verifies construct/return/match only; (5) DEV-070
  accepted as correctly classified and owned by A2. Implemented in the same session with
  full lowering/verify/interp/dump coverage; the invalid-variant guard (v3 → MIR-0008) satisfies
  the CE3 requirement #8. Amendment doc `mir-amendment-A2-ordering.md` marked APPROVED.

- CD-035 [2026-07-20, WP-C4.7-1 — **PROPOSED, awaiting owner CE3 ratification**] **MIR Amendment
  A3 (arithmetic completion), recorded post-hoc.** CD-033 approved class A5 (bit/shift/pow
  operators) and WP-C4.6 implemented it, but the `mir.md` versioning policy also requires each
  additive *shape* amendment to be individually CE3-approved and recorded in the contract, and
  that step was missed. The record now exists in `mir.md` §"A3 shape amendment": pure
  `MirBinOp::BitAnd/BitOr/BitXor` (integer-only; same-width two's-complement results are always
  representable, so no range check is owed and §5 totality holds; `~x` → `x ^ mask` rather than a
  new `MirUnOp`), `CheckedOp::Pow` (NUM-INT-ARITH-001), `CheckedOp::Shl`/`Shr` activated
  (NUM-SHIFT-001; no masking or count reduction), and `TrapCategory::InvalidShift` held distinct
  from `IntegerOverflow`, with the reference interpreter's `CheckedOutcome::Trap(Some(cat))`
  category override specified as a rule backends must reproduce. Additive; `MIR_VERSION` stays
  `0.1` and no runtime-surface identifier changes (A3 adds no `RuntimeFn`). **The ask is
  ratification of the record, not approval of new code — the code shipped in WP-C4.6 A5.**
  Consequence if ratified: WP-C4.7-3's layout amendment is **A4** (`mir-amendment-A4-layout.md`),
  renumbered from the plan's "A3" to avoid a collision.

- CD-041 [2026-07-21, owner decision — DEV-089 close-out + Gate C4 closure] **User `Display`
  dispatch through `print`/`println`/`eprint`/`eprintln`, in both engines; then close C4, open C5.**
  The owner ruled that a user type's own `Display::fmt` must execute (06 treats `Display` as an
  ordinary trait, not a syntax hook), rejecting both the previous oracle debug rendering and the
  MIR refusal. **(a) Spec:** `print`/`println`/`eprint`/`eprintln` respecified as
  implementation-provided generic `<T: Display>` functions; **PRINT-DISPLAY-001** (06-Standard-
  Library) states the nine-point contract (evaluate arg once; select the unique coherent `Display`
  by ordinary resolution; invoke `fmt` once; print exactly the returned bytes; `*ln` appends one
  `0x0A`; destroy the formatting `String` after submission; the argument follows by-value call
  ownership; a trap in `fmt` propagates with no newline/partial result; no fallback for a type
  lacking `Display` — E0500). STD-FORMAT-001 and the prelude/IO signatures updated; compiled spec
  and fixtures regenerated (manifest in sync, 112 blocks). **(b) Oracle:** `display_text` +
  `finish_display` run the impl and drop the by-value argument after its bytes are submitted; the
  internal aggregate rendering is retained only as a diagnostic facility. **(c) MIR:**
  `lower_print_display` emits an ordinary static `Callee::Instance` call to the selected `fmt`,
  then the existing `StringAsStr` + `Print(ln)Str` runtime ops, then visible `Drop`s of the
  formatting `String` and the argument. **No new MIR shape, no new `RuntimeFn`, no runtime-surface
  bump** (`MIR_RUNTIME_SURFACE` stays `0.1-A8`); `fmt` is a normal instance call so user code,
  traps and provenance stay visible. Generic user types and `T: Display`-bounded generic functions
  are supported at their monomorphised instances. **(d) DEV-090** (split from DEV-086): by-value
  iteration over a non-`Copy` array element is rejected in the front end (`E0104`, `borrowck.rs`);
  full ownership-transferring non-`Copy` array iteration is an accepted limitation outside the C5
  baseline, scheduled later. **(e) DEV-088 use-site:** using a `const` declared in another file is
  rejected in the checker (`E0215`), deferred to the front-end/multi-file completion package with
  DEV-083. **(f) Closure:** the six-clause stopping rule (CD-040(c)) now holds in full — clause 3
  satisfied by DEV-089's resolution — so **Gate C4 is CLOSED and Gate C5 (native compilation) is
  OPEN**, 2026-07-21. Evidence: `mir_differential.rs::dev089_*` (8 tests),
  `gate2_valid.rs::printing_requires_display` / `::rejects_by_value_iteration_over_non_copy_array`
  / `::accepts_by_value_iteration_over_copy_array` / `::cross_file_const_use_is_rejected`.

- CD-038 [2026-07-20, CE3 — owner-approved MIR Amendment A5] **`Projection::ConstIndex(u64)`.**
  A statically known array element: valid only on `Array<T, N>`, the verifier checks `index < N`
  itself, no `CheckIndex` terminator and no `IndexProof` local, invalid on `Vec`/slice, dynamic
  indexing unchanged. It participates PRECISELY in move analysis, which is the point — a
  proof-backed `Index` names no statically-known sub-place, so moving one element out poisoned
  every sibling and made consuming array patterns over droppable elements unrepresentable
  (lowering emitted them; verification rejected them). The same decision required **typed internal
  paths**: move-dataflow and drop-unit paths are now typed components (field / variant field /
  constant index) rather than raw `u32` sequences, so distinct projection kinds cannot compare
  equal, and fixed-length arrays decompose into PER-ELEMENT drop units. Additive; `MIR_VERSION`
  stays `0.1`; runtime surface untouched (`0.1-A8`). Recorded in `mir.md` as amendment A5.
  **Narrowed, not closed:** by-value iteration over a non-`Copy` array element — the loop index is
  a runtime counter, so no `ConstIndex` names the consumed element; reading by copy would be
  unsound (double free of a `String` in a real backend), so it is refused cleanly. Closing that
  needs unrolling or runtime-indexed drop flags, a separate design question.

- CD-039 [2026-07-20, WP-C4.7 post-exit-report, owner-directed] **Corpus 1.1.0 → 1.2.0**, completing
  the compact refresh to the six workloads §4 of the owner's directive specified. Adds a
  **multi-file** case (cross-file structs, methods, trait default + override, a cross-file `Drop`,
  and source provenance; its `helper.stark` is a corpus FILE but not a CASE, having no `main`) and
  folds DEV-086's consuming array pattern into the array/slice case. Lock regenerated (58 → 61
  files), base commit updated, and the version assertion in `exec_snapshots.rs` updated in the same
  change. A bump rather than an amendment of 1.1.0 because the array case's bytes changed, which
  the freeze rules treat as a corpus change. **All 48 hashes from 1.0.0 remain byte-identical**, so
  the original baseline survives inside 1.2.0. Writing the multi-file case found **DEV-088**
  (cross-file `const` initializers evaluated against the entry file); the declaration-time half was
  fixed, the use-site half recorded, and the case reduced to its subject per the owner's
  scope-discipline instruction.

- CD-040 [2026-07-20, owner decisions closing out WP-C4.7] Four dispositions.
  **(a) Runtime-surface ratification, post hoc:** A1 rev. 11 (`BoxNew`/`BoxIntoInner`, `0.1-A7`)
  and rev. 12 (exclusive slice views, `0.1-A8`) are ratified. Documentation and the active
  constant agree (`MIR_RUNTIME_SURFACE = "0.1-A8"`), so no implementation change was requested or
  made. **(b) DEV-083 deferred:** *"DEV-083 is deferred to a dedicated post-C5-front-end work
  package. The eventual design must use candidate-local inference snapshots and
  declaration-order-independent candidate evaluation. It must not mutate global inference state
  while probing candidates."* Provisionally assigned to `WP-C6.x Method Resolution Completion`;
  must stay visible in the ledger and in release/conformance reporting.
  **(c) Gate interpretation amended:** C4 exit does not require correcting every recorded
  front-end over-rejection before native-backend work. The stopping rule is: accepted programs
  produce valid verified MIR; unsupported programs reject cleanly; no known mislowering, ownership
  unsoundness or engine divergence remains; MIR contains the concepts C5 needs; the required
  C5/Core baseline lowers; and remaining narrow front-end over-rejections are documented and
  scheduled. **Condition 3 does not silently waive condition 2** — DEV-083 is owner-approved as
  outside the mandatory C5 lowering baseline because it is a front-end inference-completeness
  issue with a workaround and no MIR/backend effect, and that is recorded as a scope decision
  rather than an exemption. **(d) Scope discipline:** no further open-ended C4 audit; only the
  bounded final validation.

- CD-037 [2026-07-20, WP-C4.7-9, owner-directed] **Frozen execution corpus bumped 1.0.0 →
  1.1.0 — ADDITIVE ONLY.** Five new primary cases cover the constructs WP-C4.6's Class-A campaign
  and WP-C4.7 added, every one of which the differential suite exercised but NO frozen case did:
  `ownership_drop__03_discarded_values_and_nested_patterns` (unwrap_or discarding at the call,
  nested-pattern drop order, shorthand bindings), `collection_iter__03_slice_views_and_array_
  iteration` (shared + exclusive slices, write-through to the base, array iteration),
  `struct_enum_trait__05_generic_methods_and_impl_heads` (method-own generics, non-bare impl
  heads, trait-default generics), `primitive__04_bitwise_shift_pow_and_ordering` (A5 operators,
  compound forms, primitive/`Char`/`String` `cmp`, the float operator/trait split), and
  `option_result__03_box_and_layout_queries` (`Box` new/into_inner + drop timing, a recursive type
  through `Box`, layout queries, expected-typed literals). `corpus.lock` regenerated: 48 → 58
  files, base commit updated, and the version assertion in `exec_snapshots.rs` updated in the same
  change per the freeze procedure. **Verified additive:** all 48 hashes from 1.0.0 are byte-identical
  in the new lock and no pre-existing corpus file was modified, so the 1.0.0 baseline survives
  inside 1.1.0 and comparisons taken against it remain valid. All 22 cases agree across the HIR
  and MIR engines. Writing the slice case found **DEV-087** (the oracle treated a slice reference
  as non-`Copy`, so passing one to a function consumed it) — closed in the same change.

- CD-036 [2026-07-20, CE3 — owner-approved MIR Amendment A4, as drafted] Approved
  `Rvalue::LayoutQuery { kind: SizeOf | AlignOf, ty: MirTy }` — a **pure** rvalue typed `UInt64`
  that PRESERVES the queried type, replacing WP-C4.6 A4-1's type-erasing `Const 8` lowering of
  `size_of`/`align_of`. Rationale: 06-Standard-Library classifies them as *target-layout queries*
  and 07's LAYOUT-QUERY-001 makes them the only Core layout observations, so a backend must be
  able to answer them from MIR alone (charter §1.2). Approved with the drafted scope: consumers
  answer through a single layout service; the C4 reference implementation returns `(8, 8)` for
  every type, so **behavior is unchanged and the HIR oracle is not touched** — real per-target
  numbers are C5.1's, since CD-015 fixed none and LAYOUT-ABI-001 makes them target-/
  version-dependent. Not a `RuntimeFn` (type-only input, cannot trap, compile-time knowledge).
  Verifier owns one rule (dest `UInt64`, MIR-0004); `Sized`-ness stays the front end's property.
  Additive: `MIR_VERSION` stays `0.1`, runtime surface stays `0.1-A6`. Alternatives (a) record as
  a deviation, (b) real numbers now, and (c) defer to C5 were presented and declined — (c) would
  have needed a MIR version bump, since C4 exit freezes v0.1 for backend consumption.

- CD-042 [2026-07-21, owner CE4 decision] **`WP-C5-ENTRY.md` APPROVED at its recommended choices;
  WP-C5.1 implementation cleared to begin.** The entry plan (`STARKLANG/docs/compiler/
  work-packages/WP-C5-ENTRY.md`) freezes the Gate C5 supported subset, the generated-Rust
  representation contract, the ownership/move/Drop strategy, the `LayoutQuery` strategy, the
  minimal runtime and Native Provider ABI v0.1 scope, the generated-crate topology, `stark build`
  behaviour, the C5.1-C5.6 work-package sequence, the native differential test matrix, stop/
  escalation rules, and the Gate C5 exit-report format. Owner accepted the §19 decision table as
  drafted (generated Rust backend, debug-only profile, concrete-monomorphised-instances-only
  generics, `MaybeUninit<ManuallyDrop<T>>`-style non-`Copy` storage, explicit MIR-directed Drop
  glue with no automatic Rust `Drop`, isolated unsafe helpers only, Cargo invoked internally by
  `stark build`, local/pinned generated dependencies, Native Provider ABI v0.1 specified but not
  required to execute in the MVP). Status flipped `PROPOSED` → `APPROVED` in the entry-plan
  document itself. Outstanding before WP-C5.1a code lands: name the frozen C5 reference workspace
  (§4), record its green HIR/MIR baseline snapshot, and record the first host target and Rust
  toolchain versions — these are execution-time deliverables of WP-C5.1a/b, not additional
  approval gates.

- CD-043 [2026-07-21, WP-C5.1a, owner decision] **C5.1a representation decision closed: exact
  `MirTy` matrix enumerated, host target for the first native proof pinned to BOTH
  `aarch64-apple-darwin` (primary/local) and `x86_64-unknown-linux-gnu` (secondary/CI), not a
  single target as the entry plan's default allowed.** Full record: `STARKLANG/docs/compiler/
  work-packages/WP-C5.1.md`. The `MirTy` matrix (enumerated against `starkc/src/mir/mod.rs` and
  `starkc/src/hir.rs::CoreType`) marks IN: all integer/float/`Bool`/`Char`/`Unit`/`Never`/`Str`/
  `String` primitives, `Struct`, user `Enum`, `Option`/`Result` (and structurally `Ordering`),
  `Tuple`, `Array`, narrow `Ref`, `FnPtr`; marks OUT by default: `Slice`, and every
  `Core(CoreType::*)` payload except that `String`/`Option`/`Result`/`Ordering` never actually
  route through `MirTy::Core` (they lower to `MirTy::String`/`MirTy::Enum` directly) — so the real
  OUT set is `Vec`, `Box`, `HashMap`/`HashSet`, `Range`/`RangeInclusive`, all iterator `CoreType`s,
  `Random`, `IOError`/`File`. **Scope consequence recorded for C5.4d:** the frozen reference
  workspace's required "a loop" (§4.1) must be a `while`/array loop, not a `for x in a..b` range
  loop or Vec/HashMap iteration, since every range/iterator `CoreType` is OUT unless a minimal path
  is separately approved first. Owner chose the dual-target option over a single first-proof
  target specifically to avoid a later cross-platform retrofit, matching the project's existing
  dual-toolchain-version validation habit (1.93/1.97). Non-`Copy` storage, move/Drop invariants,
  enum/`Option`/`Result` representation, function-pointer representation, and the layout-query rule
  are all confirmed against the already-approved §6–10 (CD-042) with no changes. WP-C5.1a CLOSED;
  next is WP-C5.1b (backend/runtime skeleton).

- CD-044 [2026-07-21, WP-C5.1b] **Backend/runtime skeleton delivered; empty `fn main() { }`
  compiles and runs as a real native executable — the C5.1b proof, and the project's first
  generated-Rust output that is not a disposable spike.** Full record: `STARKLANG/docs/compiler/
  work-packages/WP-C5.1.md` §C5.1b. New workspace member `starkc/stark-runtime/` (dependency-free,
  §11.3); `starkc/src/backend/{mod,version}.rs` +
  `starkc/src/backend/generated_rust/{mod,emit_program,emit_types,emit_bodies,emit_places,
  emit_runtime,mangle,source_map,build}.rs`. Real logic lands in `output.rs`/`version.rs`
  (runtime) and `emit_program`/`emit_types`/`emit_bodies`/`mangle`/`build` (backend); `trap.rs`/
  `value.rs`/`provider_abi.rs` (runtime) and `emit_places.rs`/`emit_runtime.rs`/`source_map.rs`
  (backend) are doc-only placeholders by design (§5.1: "a responsibility map, not a requirement to
  create every file immediately") — nothing is hidden behind them, there is simply nothing to
  lower yet at C5.1b's scope. Entry point discovered via the literal symbol `"main@[]"`, the same
  convention `mir::interp::run_program` already uses (kept identical, not reinvented, per §5.2).
  Test: `starkc/tests/native_c5_1b_skeleton.rs::empty_main_compiles_and_runs_natively` — full
  pipeline (parse→resolve→typecheck→lower→verify→`emit_native_debug`→`cargo build
  --offline`→run), asserts exit 0 and empty stdout. **Proven on the primary target
  (`aarch64-apple-darwin`) this session; the secondary target (`x86_64-unknown-linux-gnu`) is
  proven by the next CI run — no separate CI job was needed since the test runs under the
  existing `cargo test --workspace --all-targets --all-features` step.** Validation: `cargo fmt`
  clean, `cargo clippy -D warnings` clean, full workspace suite green (0 failures across ~1050
  lines of test output), `cargo test --test exec_snapshots` green (4/4) — the C3-ENTRY CI
  baseline is unaffected by the new workspace member. One real defect found and fixed during
  bring-up (not a DEV#, an in-WP implementation correction, not a semantic defect): the initial
  `emit_trivial_unit_body` assumed a body has exactly one block; the real lowered MIR for an
  empty `main` has two (`bb0` real, `bb1` a synthetic dead `Unreachable` block from WP-C4.5's
  return-slot elaboration) — fixed to read `body.entry` specifically and require every other
  block be trivially dead, discovered by dumping real MIR rather than assumed. WP-C5.1b CLOSED;
  next is WP-C5.1c (Native Provider ABI v0.1 specification).

- CD-045 [2026-07-21, WP-C5.1c] **Native Provider ABI v0.1 document DRAFTED (status `PROPOSED`)
  with a compile-time validator and mock fixtures delivered; owner CE4 review of the document's
  technical content is still open — this is NOT a closure entry.** CD-042 approved *writing* a
  v0.1 ABI document as one of `WP-C5-ENTRY.md`'s recommended §19 choices; it did not pre-approve
  this document's actual design, which is new substantive content drafted in this WP (the same
  distinction WP-C4.1's `mir.md` draft-then-CE3-review-then-CD-028-approval sequence already
  established as the pattern for this project — a design document is not self-approving just
  because writing one was authorized). Full record: `STARKLANG/docs/compiler/
  native-provider-abi-v0.1.md` (17/17 of §10.1's required points covered) and `STARKLANG/docs/
  compiler/work-packages/WP-C5.1.md` §C5.1c. Delivered: the document itself; real `#[repr(C)]`
  ABI types in `starkc/stark-runtime/src/provider_abi.rs` (`ResourceHandle`, `BorrowedBuffer`,
  `BorrowedBufferMut`, `ProviderStatus`); a compile-time metadata validator in `starkc/src/
  backend/provider_abi.rs` (`validate(&ProviderMetadata) -> Result<(), Vec<AbiViolation>>`,
  returns every violation found, not just the first, matching the MIR verifier's own convention);
  a fictional illustrative `example-kv` mock provider plus 6 deliberately-invalid fixtures, one
  per violation class — 7/7 tests pass. No provider feature expansion beyond the document +
  validator + fixtures (§10.2): no dynamic loading, no real `extern "C"` linkage, no file/network
  provider implementation. **One cross-reference defect found and fixed before this entry was
  written, not after:** the document's own §10.1-point citations drifted during drafting (three
  headings cited the wrong point number against the entry plan's 17-item list — §10 cited "point
  16" instead of 17, §15 cited "points 14 and 15" instead of "14 and 16", §16 cited "point 14"
  instead of 15); caught by a deliberate grep-and-recount sweep against the source list before
  commit, not by the owner. Validation: `cargo fmt`, `cargo clippy -D warnings`, full workspace
  suite, and `exec_snapshots` all green. **WP-C5.1c: document/validator/fixtures DELIVERED; the
  design itself awaits owner CE4 review before WP-C5.1 overall can close** (provider execution is
  not required for the C5 MVP, so this blocks only the design-review checkbox, not
  implementation).

- CD-046 [2026-07-21, owner CE4 decision] **Native Provider ABI v0.1 (`STARKLANG/docs/compiler/
  native-provider-abi-v0.1.md`) APPROVED AS DRAFTED, no changes required.** Closes the review gate
  CD-045 opened. Owner reviewed the document's actual technical choices — the C-ABI-idiom error
  convention (§11: status code + out-parameters, chosen to avoid a hand-rolled unsafe tagged
  union), the no-borrowed-handle-in-v0.1 decision (§8), and the closed `AbiType` vocabulary (§6/
  §10) as the single mechanism enforcing both the callback prohibition and the
  no-generated-Rust-aggregate-crossing rule — and approved as drafted, the same draft-then-CE4-
  review outcome `mir.md` reached under CD-028 (there: approve-with-required-changes; here:
  approve outright). Document status flipped `PROPOSED` → `APPROVED`. **WP-C5.1c CLOSED; WP-C5.1
  (Runtime ABI and Layout Design) CLOSED in full — all of C5.1a/b/c done.** Per `WP-C5-ENTRY.md`
  §14's exit checklist: CE4 decision recorded (CD-042 representation contract + CD-046 provider
  ABI), one verified empty/scalar MIR program is a standalone executable on both pinned targets,
  runtime/backend/compiler version checks demonstrated, no language semantics hidden in the
  runtime. Next: WP-C5.2 (scalar native lowering) — primitive values/constants (C5.2a), locals/
  places/copies/moves (C5.2b), operations/control flow (C5.2c), direct functions/calls (C5.2d),
  trap path (C5.2e).

- CD-047 [2026-07-21, WP-C5.2a] **Constant emission delivered — `emit_types::emit_constant`
  covers every primitive `Constant` variant.** Full record: `STARKLANG/docs/compiler/
  work-packages/WP-C5.2.md` §C5.2a. `Bool`/`Unit` direct; `Int` with the integer-suffix reused
  from `emit_ty`; `Int(codepoint, MirTy::Char)` (the `Char` constant's actual MIR encoding, per
  `mir::lower`'s f-3b) reconstructed via `char::from_u32(...).unwrap()` since Rust has no `char`
  literal suffix; `Float` via `f64`'s `Debug` formatting (guaranteed round-trip, always a decimal
  point/exponent so it parses back as a float literal) with `NaN`/`Infinity`/`-Infinity` handled
  as named `f64::` constants since they have no Rust literal syntax. **Real bug caught before
  commit:** the first version unconditionally appended an `f64` suffix, producing invalid
  `f64::NANf64` for the NaN case — caught by the test harness (every emitted expression is
  round-tripped through a real `rustc --edition 2021 --crate-type lib` parse/typecheck, not just
  string-shape-asserted), fixed by making the NaN/Infinity branches return an already-fully-typed
  expression the caller does not re-suffix. 5/5 tests pass. Validation: `cargo fmt`, `cargo
  clippy -D warnings`, full workspace suite, `exec_snapshots` — all green. **Process note:** the
  owner flagged that running the full workspace suite after every small change was slowing
  development; going forward, scoped `cargo test --lib`/`--test <file>` runs during iteration,
  full-suite runs reserved for WP/gate closure points (recorded for future sessions in memory,
  not just here). WP-C5.2a CLOSED; next is WP-C5.2b (locals/places/copies/moves).

- CD-048 [2026-07-21, WP-C5.2b] **Real locals/places/assignments/copies delivered —
  `emit_body` (renamed from and fully replacing C5.1b's `emit_trivial_unit_body`) declares every
  body local and lowers `Use`-rvalue assignments; `emit_place` supports bare locals.** Full
  record: `STARKLANG/docs/compiler/work-packages/WP-C5.2.md` §C5.2b. Locals declared `let mut _N:
  T;` uniformly (uninitialised, `mut` regardless of reassignment — cheap given the generated
  file's blanket `#![allow(unused)]`, and leaving them genuinely uninitialised means a
  lowering-bug read-before-write is caught by rustc's own definite-assignment analysis, not
  silently given a fabricated default). `Operand::Copy`/`Operand::Move` both emit the same bare
  place reference — sound because `emit_ty` only admits primitive `MirTy`s and every primitive is
  `Copy` by construction; real non-`Copy` move/liveness tracking stays deferred to WP-C5.3+. The
  entry's Unit-return check moved from inside the body emitter to `emit_program.rs` specifically
  (a Rust-`fn main()` constraint, not a general body-emission one), so `emit_body` stays reusable
  for an arbitrary-return-type function once WP-C5.2d lifts the single-body-program restriction.
  Two new end-to-end native tests (`native_c5_2b_locals.rs`: real `Int32`/`Bool`/`Char`/
  `Float64`/`UInt8` locals + a copy; separate `Float32`/`Float64` locals) plus the existing
  `native_c5_1b_skeleton.rs` empty-`main` proof re-run unchanged as a regression check that the
  generalized emitter still handles the C5.1b shape. One STARK-level (not backend) snag caught
  writing the test: an unsuffixed `2.5` float literal defaults `Float64` and does not coerce to a
  `Float32`-typed `let` (`E0001`) — fixed in the test source. Validation: `cargo fmt`, `cargo
  clippy -D warnings`, scoped tests (`backend::` 16/16, new test 2/2, regression 1/1),
  `exec_snapshots` 4/4 — full workspace suite not re-run this WP, per the new test-run-frequency
  policy (last green at WP-C5.2a; this WP's changes are additive and narrowly scoped to
  `backend::generated_rust`). WP-C5.2b CLOSED; next is WP-C5.2c (operations and control flow).

- CD-049 [2026-07-21, WP-C5.2c] **Real operations and arbitrary control flow delivered —
  arithmetic (with correct overflow/div-by-zero/shift trapping), comparisons, bitwise ops,
  `if`/`else`, and `while` loops now compile and run natively, matching `mir::interp::eval_checked`
  (the oracle) exactly.** Full record: `STARKLANG/docs/compiler/work-packages/WP-C5.2.md`
  §C5.2c. `emit_bodies.rs` restructured to a block-index dispatch loop (`let mut __bb: u32 =
  <entry>; loop { match __bb { 0 => {...}, ... } }`) — the standard technique for emitting an
  arbitrary MIR basic-block graph without recovering structured `if`/`while` shapes, since Rust
  has no `goto`; `Goto`/`SwitchInt` both reduce to `__bb = target; continue;`, so loops need no
  special-casing versus branches. Checked ops widen to `i128`, use Rust's native `checked_*`,
  then range-filter against the DESTINATION type — provably equivalent to native narrow-width
  checked arithmetic for `Add`/`Sub`/`Mul`/`Div`/`Rem`/`Neg`/`Pow`, but NOT optional for `Shl`
  (native `checked_shl` only validates the shift count, silently dropping overflowed bits, which
  would violate STARK's always-trap semantics for left-shift overflow specifically). Trap
  categories read directly from the terminator's own `TrapInfo` rather than re-derived, matching
  `mir::interp`'s own "terminator's category, with the `Shl`/`Shr` bad-count `InvalidShift`
  override" rule exactly. New `stark_runtime::trap::abort_minimal` is an explicitly MINIMAL,
  not-yet-final abort (stderr category + nonzero exit) — the real trap ABI (source spans, §13.2
  canonical format) stays WP-C5.2e's job; this exists now only because "overflow and silently
  continue" would be unsound to leave unimplemented. **Real soundness bug caught and fixed before
  commit, not cosmetic:** WP-C5.2b's "leave locals uninitialised, let rustc's definite-assignment
  analysis catch a lowering bug" strategy silently breaks the moment a body has more than one
  block — rustc treats each `match __bb { N => {...} }` arm as an independent branch of one
  ordinary match with no notion that arm 1 only runs after arm 0 already assigned a local (that
  fact lives in data flowing through `__bb`, invisible to rustc across `continue`). The first
  real multi-block test programs failed to compile with `E0381` immediately, not hypothetically;
  fixed by default-initialising every local (`emit_types::default_value_expr`), the standard fix
  for this codegen pattern, trading away C5.2b's "free" lowering-bug-catch property (MIR's own
  V-MOVE-1 verifier remains responsible for that instead) — WP-C5.2b's own record was revised to
  say so rather than left stale. Five new end-to-end native tests
  (`native_c5_2c_operations.rs`: full arithmetic/comparison suite, an `Int32` overflow trap, a
  division-by-zero trap, `if`/`else`, a `while` loop to 5) plus the C5.1b/C5.2b proofs re-run
  unchanged as regressions. Validation: `cargo fmt`, `cargo clippy -D warnings`, scoped tests
  (`backend::` 16/16, new test 5/5, prior regressions 3/3), `exec_snapshots` 4/4 — full workspace
  suite not re-run per the test-run-frequency policy. WP-C5.2c CLOSED; next is WP-C5.2d (direct
  functions and calls).

- CD-050 [2026-07-21, WP-C5.2d] **Multi-function programs, real parameters, and direct calls
  delivered — `emit_program.rs`'s single-body restriction (present since WP-C5.1b) is lifted.**
  Full record: `STARKLANG/docs/compiler/work-packages/WP-C5.2.md` §C5.2d. Every body in
  `program.bodies` is emitted as its own Rust item (`lower_program`'s own doc comment already
  guarantees the set is self-contained and transitively-reachable, so no separate linking logic
  was needed); the entry instance stays specially wrapped as Rust's literal `fn main()` with the
  version-check prologue, every other body goes through new `emit_bodies::emit_function`.
  `emit_param_list` maps each `body.params[j]` to the local whose `LocalKind` is `Param(j)` (a
  local's position and its parameter index are NOT the same number) and emits it as a `mut` Rust
  parameter under that local's own `_N` name, so ordinary statement emission needs no
  special-casing to read a parameter. `Terminator::Call` with `Callee::Instance` lowers to an
  ordinary Rust call, using `mangle::function_name_for_symbol` as the one naming authority for
  both defining and calling a function (entry symbol → `main`, everything else → its sanitized
  form) rather than two conventions that could drift apart. `Callee::FnValue`/`Callee::Runtime`
  stay deferred to WP-C5.4c and wherever the first `RuntimeFn` group lands, respectively. **No
  bug this time** — unlike C5.2b/c, the one real hazard this WP's design raised (declaring a
  `Param`-kinded local a second time in the block body would silently shadow the real argument
  with a fabricated default) was caught in review before writing the test (`emit_block_body`'s
  default-init loop explicitly `continue`s past `Param`-kinded locals), not discovered by a
  failing build. Two new end-to-end native tests (`native_c5_2d_calls.rs`: a two-parameter `add`
  call, and a three-parameter `clamp` helper feeding an `if` plus a second `Float64`/`Bool`
  helper) passed on the first run, plus the C5.1b/C5.2b/C5.2c proofs re-run unchanged as
  regressions. Validation: `cargo fmt`, `cargo clippy -D warnings`, scoped tests (`backend::`
  18/18, new test 2/2, prior regressions 8/8), `exec_snapshots` 4/4 — full workspace suite not
  re-run per the test-run-frequency policy. WP-C5.2d CLOSED; next is WP-C5.2e (trap path).

- CD-051 [2026-07-21, WP-C5.2e] **Real trap ABI delivered — every checked-operation trap now
  reports its category and an exact source file/line on stderr, exit code 101 (matching `stark
  run`'s own established convention).** Full record: `STARKLANG/docs/compiler/work-packages/
  WP-C5.2.md` §C5.2e. `stark_runtime::trap::abort(category, file, line, column) -> !` replaces
  C5.2c's `abort_minimal` placeholder outright. Source location is resolved at COMPILE TIME
  (`SourceFile::line_col` against `MirProgram::files`, both already available to the backend) and
  baked into the generated call site as literals — a documented, deliberate simplification of
  §13.1's compact-span-ID-plus-runtime-lookup-table design (that design exists to deduplicate
  span data for large programs; inlined literals are simpler and exactly as correct at MVP
  scale), not an oversight. `emit_abort_call` is the one place that assembles a trap-abort call,
  used for both a terminator's default category and the `Shl`/`Shr` `InvalidShift` override, so
  the two trap sites within one checked operation cannot independently drift. Category messages
  are NOT claimed to match the HIR interpreter's own ad hoc per-call-site strings byte-for-byte —
  no canonical table exists there to match, and the differential comparator (§15.1) checks
  category plus source file/line, not stderr text. C5.2c's own two trap tests were retrofitted
  from a loose `assert_ne!` to the exact `assert_eq!(status, Some(101))` now that the precise
  contract exists. Four new tests (`native_c5_2e_traps.rs`): an overflow trap asserting an EXACT
  `file:line` match (not a loose check), plus division-by-zero/invalid-shift/cast-failure each
  asserting category message and exit code. Validation: `cargo fmt`, `cargo clippy -D warnings`,
  scoped tests (`backend::` 18/18, new test 4/4, all prior native regressions including the two
  retrofitted), `exec_snapshots` 4/4. **WP-C5.2e CLOSED. WP-C5.2 (scalar native lowering) is
  NOT YET claimed closed**: §14's exit condition explicitly requires three-engine (HIR/MIR/
  native) automated agreement, and every `native_c5_2*.rs` test to date asserts on the native
  engine's own output in isolation, not an automated diff against the other two engines the way
  `mir_differential.rs` already does for HIR-vs-MIR. This gap is recorded here deliberately
  rather than treated as satisfied by "native looks right" reasoning. Building the three-engine
  differential harness (§15.1/§15.2) is the next open decision — whether it lands as a C5.2-
  closing addendum or defers to WP-C5.6 (which already co-owns cross-backend snapshot replay per
  the WP-C4.4/CD-018 carry-forward) is for the owner to decide, not resolved here.

- CD-052 [2026-07-21, WP-C5.2 review response] **External review of head 37828a07 raised seven
  findings; all seven verified as REAL against the code (no false positives). Four fixed here
  (DEV-091/092/093/094), one recorded as a C5.3 opening condition (DEV-095), two escalated to the
  ABI's owner as a CE4 amendment.** Writing the regression tests for the first finding surfaced an
  eighth, previously unknown defect (DEV-096) that the review did not name.

  - **DEV-091 — float→int casts accepted out-of-range values at 64-bit widths, in BOTH the MIR
    interpreter and the native backend. FIXED.** Both compared the truncated value against
    `max as f64`, which ROUNDS UP at 64-bit widths: `u64::MAX as f64` is 2^64 and `i64::MAX as
    f64` is 2^63. Exactly 2^64 therefore passed the guard, and the subsequent saturating `as`
    clamped it to `u64::MAX` — silently producing a value where 03-Type-System.md requires a
    trap. Same defect at 2^63 for `Int64`. Fixed in both engines with a half-open test against an
    EXACT bound: every `max + 1` is a power of two and so exactly representable as `f64`
    (`mir/interp.rs`'s `Cast` arm; `emit_bodies.rs`'s new `int_float_bounds_tokens`, deliberately
    separate from `int_bounds_tokens`, whose inclusive pair remains correct for the exact-`i128`
    checked-arithmetic path). The HIR ORACLE was already correct here — it truncates to `i128`
    and range-checks in exact integer arithmetic — so this was a genuine engine divergence, not a
    shared misreading of the spec. The reason it survived: no corpus or inline case had ever
    exercised a 64-bit cast boundary. Seven new boundary cases in `mir_differential.rs` (2^64,
    greatest f64 below 2^64, 2^63, greatest below 2^63, -2^63 inclusive, below -2^63, truncation
    ordering) plus three native ones in `native_c5_2c_operations.rs`.
  - **DEV-092 — symbol sanitization was not injective, while its own doc comment asserted that
    it was. FIXED.** `sanitize_symbol` hex-encoded disallowed bytes as `_hh` but passed `_`
    through unchanged, so encoded output was indistinguishable from source text that already
    spelled an escape: `pkg::f` and a legally-named STARK function `pkg_3a_3af` both encoded to
    `stark_pkg_3a_3af...`. Reachable from ordinary source, because `key_symbol` puts a
    `::`-joined module/package path in every symbol, and materially relevant since C5.2d, where
    every MIR body became its own Rust function. Fixed by making `_` the escape introducer and
    escaping it as `__`; the encoding is now decodable, hence injective, and stays readable
    (`my_fn` → `my__fn`) rather than hex-encoding every byte. Tests: a pairwise-distinctness
    sweep over 17 adversarial symbols (`::`/`_3a` at package and module boundaries, `@`/`_40`,
    `[`/`_5b`, literal-vs-escaped underscores, the `, ` type-argument separator, and non-ASCII
    identifiers) plus a round-trip-through-a-decoder test that states injectivity directly rather
    than sampling for collisions.
  - **DEV-093 — native success-path tests observed no computed values. FIXED.** The arithmetic,
    branch, loop and direct-call tests computed results and asserted only `exit == 0`; a backend
    returning zero from every function would have passed most of the suite. All success-path
    tests now assert IN the STARK program via `assert_eq`/`assert` (native `println` is still
    WP-C5.3), covering every arithmetic result, both branch directions, loop trip count AND body
    effect, zero-iteration loops, call return values, and parameter order. This required
    implementing `Terminator::Trap` in the backend (message-less form — what `mir::lower` emits
    for `assert`/`assert_eq`/`assert_ne`), which was still `Unsupported` at CD-051 and is
    properly WP-C5.2e's own deliverable; `Trap` carrying a user `&str` message remains WP-C5.3.
    A NEGATIVE CONTROL (`a_false_assertion_traps_natively`) proves a false assertion really does
    reach the trap ABI and exit 101 — without it, "exit 0" would remain ambiguous between
    "assertions held" and "assertions compiled away".
  - **DEV-094 — the version-mismatch message named the wrong version on each side. FIXED.**
    `version::check` assigned the LINKED runtime's `RUNTIME_VERSION` to `expected_runtime_version`
    and the generation-time recorded value to `actual_`, while the generated crate prints them as
    "generated for runtime {expected}, linked against {actual}". Fixed at the source (the field
    assignment, not the message) so the names read correctly for any future consumer, with a test
    that pins the field-to-side assignment rather than merely that a mismatch is detected.
  - **DEV-095 — the generated-crate build key omits nominal type context and the Drop map.
    RECORDED as a WP-C5.3 opening condition, NOT fixed here.** `compute_build_key` hashes
    `program.dump()`, and `dump()` emits only the version header and bodies; the MIR contract
    states the nominal type context and destructor map are in-memory parts of the compilation
    unit that the textual dump does not serialize. Changing a struct's fields or its `Drop`
    metadata could therefore leave the build key unchanged and silently reuse a stale generated
    crate. This CANNOT bite before aggregates and Drop exist, which is exactly WP-C5.3, so it is
    a C5.3 entry condition rather than a C5.2 defect: before aggregates land, build identity must
    cover a deterministic encoding of the nominal type context, the Drop implementation map, the
    source table, package graph identity, the entry instance, all bodies, and the backend/
    runtime/toolchain versions.
  - **DEV-096 — the HIR oracle reported every out-of-range cast as an ARITHMETIC OVERFLOW trap,
    at every width. FIXED. Not named by the review; found by DEV-091's new boundary tests, which
    failed on category mismatch rather than on the bound.** Both cast arms in `interp.rs`
    (int→int and float→int) routed through `check_integer_range`, whose message is hardcoded
    `"integer overflow"`, so the oracle disagreed with the MIR interpreter and the native backend
    — both of which classify a failing cast as `TrapCategory::CastFailure` — for every
    out-of-range cast, not merely at 64-bit boundaries. 03-Type-System.md enumerates overflow and
    failing `as` casts as DISTINCT always-trap causes, and the oracle's own non-finite float case
    already used the cast-specific message, so this was an implementation artifact of a shared
    helper rather than a semantic question. Split into `check_cast_range` (cast failure) and
    `check_integer_range` (overflow) over one shared width predicate, so the two can never drift
    on WHICH values are in range while differing, correctly, on which trap they raise. Two
    narrow-width regression tests pin the category independently of any float rounding.
  - **Escalated to the owner as a CE4 amendment, NOT changed here** (the Native Provider ABI
    v0.1 is owner-approved under CD-046, so amending it is the owner's decision):
    `STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md` documents two
    contradictions between the approved document and its own validator — the return-shape
    contradiction (§11 says every provider function returns `ProviderStatus` with results via
    out-parameters, but `FunctionDecl` has `returns: AbiType` with no out-parameter
    representation, and the validator's own "valid" fixture has `kv_open` returning
    `ResourceHandle`), and `ResourceHandle` deriving `Clone`/`Copy` against §12's exclusive-
    ownership and close-exactly-once rules. Both are cheap to correct now because no provider
    executes in the C5 MVP; neither is corrected without owner sign-off.
  - **Also observed, not filed as defects**: no integer literal above `Int64::MAX` is expressible
    (an unsuffixed literal types as `Int64` first, so even `let x: UInt64 = 18446744073709549568;`
    is rejected), `Int64::MIN` has no literal spelling, and an unsuffixed literal in argument
    position does not receive expected-type propagation from a sibling argument. These shaped how
    the boundary tests are written (documented at the test) but are pre-existing front-end
    behaviours unrelated to native lowering.
  - **The review's one process observation did NOT hold up.** It reported that the "CI green"
    claim was unverifiable because its GitHub connector exposed no workflow run for head
    37828a07. `gh run list` shows the `CI` workflow completed with conclusion `success` on
    37828a07 (and on 5af7ad7/56b5202/c9eaa53 before it), so the claim was accurate and the gap was
    in the connector's visibility, not in the evidence. Worth recording for its own reason,
    though: CI was green on the very commit carrying DEV-091's semantic defect. `fmt`, `clippy`
    and the full workspace suite all passed because **no test exercised a 64-bit cast boundary** —
    a green pipeline bounds the risk to what the corpus covers, and this pass is a direct
    demonstration of that limit.
  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, `mir_differential` 132/132 (up from 123 — the frozen
    corpus plus nine new cast cases: seven boundary, two category), all five `native_c5_*` suites
    green (19 tests, up from 13), `exec_snapshots`/`conformance`/`gate3_execution` green, and the
    full workspace suite green.

- CD-053 [2026-07-21, WP-C5.2 closure + CE4 amendment direction] **Owner directive, four parts:
  build the three-engine differential harness NOW as the WP-C5.2 closure addendum (not deferred to
  WP-C5.6); do NOT approve CE4 Amendment 1 as submitted — revise and resubmit before either
  `provider_abi.rs` changes; keep the ABI version at `0.1`; keep DEV-095 (build-key completeness)
  as a mandatory WP-C5.3 opening condition.** All four executed.

  - **Part 1 — the three-engine differential harness. BUILT; WP-C5.2 CLOSED.**
    `starkc/tests/three_engine_differential.rs` implements `WP-C5-ENTRY.md` §15.1's **three-engine
    pipeline**, comparing traps in **normalised** form for C5.2 (raw stderr byte equality is NOT
    compared — the HIR oracle has no canonical stderr format to compare against, only ad hoc
    per-call-site strings; what is compared is what those bytes mean, i.e. category plus exact
    file/line/column): one source string per case, run through the HIR interpreter (oracle), the MIR
    pipeline (lower → verify → execute) and the native binary (lower → verify → emit → cargo build
    → run), each result **normalised into one common `Outcome`** — `Completed { stdout, exit }` or
    `Trapped { category, file, line, column, stdout_before }` — and all three required equal. The
    normalisation is the substance: the oracle raises prose plus a byte span, MIR raises a
    `TrapCategory` plus a `SourceInfo`, and the native binary writes stderr text and a process exit
    code, so agreement is only mechanically checkable once all three are projected onto one type.
    Compared per case: completion-vs-trap, exit status, trap category, exact trap file/line/column,
    and observable output. 20 tests, all green.
    - Coverage against §14's six required dimensions: scalar arithmetic (all operators, widths,
      precedence, negative-operand division/remainder, `Float64`); branches (both directions of
      each `if`/`else`, an `else if` chain taking middle and final arms, nested, no-`else`, `if`
      as an expression, `&&`/`||`/`!`); loops (zero-iteration in two shapes; accumulate,
      `continue`, `break`, nested); direct calls (multi-function, argument order via a
      non-commutative callee, no-arg, `Unit`-returning, nested-call arguments, recursion, call in
      a loop); successful checked operations (arithmetic landing exactly on `Int32::MAX`/`MIN`,
      shift counts at width-1, in-range casts at the narrower type's exact boundary, widening,
      int↔float); and every admitted trap category (`IntegerOverflow`, `DivideByZero` for both `/`
      and `%`, `InvalidShift`, `CastFailure`, `AssertFailure` for both `assert_eq` and bare
      `assert`). `IndexOutOfBounds`, `UnwrapNone`/`UnwrapErr` and message-carrying `Panic` are not
      reachable from the C5.2 surface and the oracle-normalisation function panics explicitly on
      them rather than guessing.
    - CD-052 regressions re-pinned as three-engine agreement rather than per-engine assertions:
      **DEV-091** (four cases — in-range boundary conversions, exactly 2^64 → `UInt64`, exactly
      2^63 → `Int64`, first f64 below `Int64::MIN`; both sides of every bound), **DEV-096** (a
      case only a category comparison can hold, since all three engines exit 101 either way),
      **DEV-092** (the source-level consequence, not just the encoding: `mod m { pub fn f() }`
      versus a top-level `fn m_3a_3af()` — one Rust identifier under the old encoding — with both
      called and both return values observed), and the **negative control** proving a false
      assertion really does fail the run in all three engines, without which every
      assertion-observed completing case would be decorative.
    - **Mutation-tested before being trusted.** A comparator that passes proves nothing until it
      has been shown to fail. Two mutations were injected into the native backend and reverted:
      `checked_add` → `checked_sub` (result: `MIR/NATIVE DISAGREEMENT`, MIR `Completed` vs. native
      `Trapped { AssertFailure }` — the value dimension is live) and native trap `line` → `line +
      1` (result: same category and file, line 4 vs. 5 — the location dimension is live,
      independently of category). `git diff` confirms neither survives.
    - Honest handling of the output dimension: native `println` is `Unsupported` until WP-C5.3, so
      values are observed through in-program `assert`/`assert_eq`. Rather than quietly excluding
      stdout from the comparison, `NATIVE_STDOUT_SUPPORTED: bool = false` gates a precondition
      **enforcing** that every case is output-free, which is what makes full three-way `Outcome`
      equality total. Flipping that constant when native output lands drops the precondition and
      starts comparing real bytes, with no other change.
    - One production change only: `stark_runtime::trap::TrapCategory::message()` became `pub`, so
      the harness normalises native stderr against the runtime's own category table instead of a
      second copy in a test file that would drift the first time a message's wording changed.
    - Per-engine tests (`native_c5_2*.rs`, `mir_differential.rs`) remain and remain useful, but
      per the owner's direction they are **supplementary** and do not themselves satisfy §14. What
      stays with WP-C5.6 is cross-backend replay of the frozen `exec_snapshots` corpus (the
      WP-C4.4/CD-018 carry-forward); what moved out of it is the comparator.

  - **Part 2 — CE4 Amendment 1 NOT approved as submitted; revised and resubmitted.** The owner
    approved five principles (every physical provider function returns `ProviderStatus`; result
    values travel through explicit output channels; the owning resource representation is not
    `Clone`/`Copy`; a raw C-compatible `Copy` handle may remain inside the isolated FFI boundary;
    the owning wrapper must NOT implement Rust `Drop` — verified MIR keeps the exactly-once close
    obligation) and named four issues revision 1 omitted. Revision 2
    (`STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md` revision 2) resolves all four:
    (a) `BorrowedBuffer`/`BorrowedBufferMut` are borrowed call-duration views, so §8's
    ownership-transfer language is corrected to cover handles only — as written it made *reading
    the buffer you just passed to `kv_get`* a use-after-transfer; (b) the v0.1 prohibition on
    borrowed handles is lifted, because consuming-only handles made §17's own mock provider
    unexpressible (`kv_get` would consume the store it reads); (c) every handle parameter and
    handle output names its declared resource type, so the validator can enforce §13's
    wrong-resource-type rule it currently cannot see; (d) direction and ownership are separated —
    revision 1's `Direction × AbiType` product is **rejected**, since of its 15 combinations six
    are meaningful, three are one case spelled three times, and the distinction that matters
    (borrowed vs. consumed handle) is the one it cannot express. Replaced by a closed `AbiParam`
    enum over exactly the seven owner-enumerated forms, plus a `RawResourceHandle`
    (`Copy`, boundary-only) / `OwnedResourceHandle` (non-`Copy`, non-`Clone`, no `Drop`) split, a
    close-function rule requiring exactly one consumed handle of the declared type and no ordinary
    value output, two new violation classes, and a corrected `valid_example_kv` fixture. One
    discretionary reading is flagged for the owner rather than assumed (may a close function take
    additional pure inputs?). **Neither `provider_abi.rs` changes until revision 2 is approved.**

  - **Part 3 — ABI version stays `0.1`.** Nothing has shipped or executed against this ABI, so
    correcting a pre-execution contract is an amendment, not a version bump. Recorded as CE4
    Amendment 1 to v0.1.

  - **Part 4 — DEV-095 confirmed as a mandatory WP-C5.3 OPENING condition.** WP-C5.3 may not begin
    aggregate or Drop-bearing native generation until every semantic input affecting generated
    code — nominal type context and the Drop map included — is in the build key and covered by
    cache-invalidation tests. Recorded in Follow-ups as a blocking entry condition, not a
    to-do.

  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, `three_engine_differential` 20/20, `mir_differential`
    and all five `native_c5_*` suites green, and **`cargo test --workspace` green: 884 passed /
    0 failed / 2 ignored across 52 test binaries**.
    **Correction to the figure first recorded here (818 across 40 binaries):** that was an
    undercount of the *same* green run, not a different result — the background capture of that
    run lost its first 24 lines to output buffering, so 12 suites never reached the tally. Caught
    by re-running with a complete capture and noticing the suite count disagreed. Recorded rather
    than quietly overwritten, because "the number moved and nobody said why" is worse than the
    original error.

- CD-054 [2026-07-21, CE4 Amendment 1 approved and implemented] **The owner approved revision 2's
  design with five required changes, ruled the flagged close-function question, and directed that
  the amendment, the approved ABI document, both implementation files, the fixtures and the
  violation tests land in one commit. Done — CE4 Amendment 1 to Native Provider ABI v0.1 is
  APPROVED (ABI version stays `0.1`) and applied.**

  - **Approved from revision 2**: the closed `AbiParam` model; the fixed physical `ProviderStatus`
    return; explicit output channels; typed borrowed/consumed/output handles; borrowed buffer
    semantics; the `RawResourceHandle`/`OwnedResourceHandle` separation; owning handles being
    non-`Clone`, non-`Copy` and without Rust `Drop`; version `0.1`; the corrected example-provider
    shapes.

  - **The close-function ruling.** A close function takes **exactly one parameter** —
    `HandleConsumed { resource_type: rt }` — and nothing else. Revision 2's permissive reading
    (additional pure inputs such as a `flush: Bool` allowed) is withdrawn. The reason is
    architectural: **MIR's `Drop(place)` terminator supplies only the resource being dropped**, so
    a close function with a second parameter is one the generated code cannot call — every extra
    argument would have to be invented by the backend. The consequence is a design rule, not just
    a validation rule: any flush/completion/fallible operation needing arguments must be a
    separate provider function invoked BEFORE Drop.

  - **Four new normative rules** (amendment §4.6-§4.9, landed as ABI doc §8, §11.1, §13.2, §6.1):
    - **Consumed-handle error rule.** Ownership transfers at call ENTRY; a `HandleConsumed` value
      is dead regardless of what `ProviderStatus` reports. Ownership returning on failure would
      make a handle's liveness depend on a runtime value, so use-after-transfer could not be
      decided by MIR verification and exactly-once close would stop being a static property. An
      operation wanting ownership back on failure declares an explicit `HandleOut` (a *fresh*
      handle, not a resurrected one) or borrows instead.
    - **Output initialisation rule.** `ScalarOut`/`HandleOut` storage is uninitialised before the
      call and valid only on success: allocate through `MaybeUninit`, never read or wrap on
      failure, and validate a successful raw handle's resource type before constructing the owning
      wrapper. `ScalarInOut`/`BufferInOut` stay caller-initialised and caller-owned across the
      call. The asymmetry is the point — an `Out` slot is a promise kept only on success; an
      `InOut` slot is the caller's own memory, lent for one call.
    - **Close-failure rule.** A close function's nonzero status cannot become a recoverable
      `Result::Err`, because a `Drop` terminator has no result destination. It is a distinct fatal
      provider-close/host failure: abort without unwinding, do not retry, treat the handle as
      consumed, run no further pending Drop glue. Recoverable work (flush/commit/sync) must be a
      separate operation performed before close.
    - **Physical ABI mapping.** Every `AbiParam` variant mapped to its exact C parameter, plus the
      requirement that all raw↔owned conversions go through isolated reviewed boundary helpers,
      never generated ad hoc field access. Two pairs are physically identical and deliberately
      distinct in metadata: `ScalarOut`/`ScalarInOut` (both `*mut T`, differing in the
      initialisation contract) and `HandleBorrowed`/`HandleConsumed` (both a raw handle by value,
      differing in the ownership contract) — the C signature cannot carry either difference, which
      is exactly why the declaration must.

  - **Implemented in one commit**, per the directive: the ABI document updated (§6 rewritten, §6.1
    /§11.1/§13.1/§13.2 added, §7/§8/§10/§12/§17/§18 amended, each marked *(amended, CD-054)*);
    `starkc/src/backend/provider_abi.rs` (`ScalarTy`, `AbiParam`, `returns`-less `FunctionDecl`,
    `HandleResourceTypeUndeclared` and `CloseFunctionShape`/`CloseShapeProblem` violations, and
    the two new validator rules); `starkc/stark-runtime/src/provider_abi.rs` (the raw/owning split
    and the three boundary helpers, with resource-type validation inside `from_raw_checked` so it
    cannot be skipped by a call site that forgets it); and the fixtures rewritten to conform.
    **`example-kv` now works as an example**: `kv_open` writes its handle into a `HandleOut`,
    `kv_get` borrows the store and has somewhere to put the value it retrieves, and `kv_close`
    consumes exactly one handle. Tests: 14 in the validator module, up from 7 — five new
    negatives (an undeclared handle resource type, and one per close-shape problem: an extra
    parameter, an added output, a borrowed rather than consumed handle, a consumed handle of the
    wrong resource type) plus two new positives (ordinary operations borrow rather than consume;
    every value result is an explicit output form) — and 3 in the runtime module.

  - **What is NOT claimed.** No provider executes; §10.2's boundary is unchanged. Every rule in
    the four new sections is a statement about code that does not exist yet — the validator, the
    type definitions and the fixtures are what exist. The call-site generation that must obey the
    output-initialisation and boundary-helper rules belongs to whichever package first makes a
    provider execute. `WP-C5.1.md` records which four of its own C5.1c statements this
    supersedes, rather than being silently edited.

  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, **`cargo test --workspace` green: 894 passed / 0 failed
    / 2 ignored across 52 test binaries** (up from 884 — the seven new validator tests and three
    new runtime tests).

- CD-055 [2026-07-21, DEV-095 discharged — WP-C5.3 entry condition] **The generated-crate build
  key now covers every semantic input that can affect generated code, with cache-invalidation
  tests. WP-C5.3's blocking entry condition (CD-053 part 4) is DISCHARGED; aggregate and
  Drop-bearing native generation may begin.**

  - **The defect.** `compute_build_key` hashed `program.dump()` plus the eight version axes, and
    `dump()` serializes only the version header and the bodies. The MIR contract is explicit that
    the **nominal type context and the destructor map are in-memory parts of the compilation unit
    the textual dump does not serialize**. So two programs with byte-identical dumps but different
    struct fields, different enum variants, a different `Drop` impl, or different `Copy`
    classification hashed to the SAME key — and the second build would silently reuse the first's
    generated crate. Unreachable while the backend admitted only primitives; live the moment
    WP-C5.3 lands aggregates and `Drop`, which is why it was fixed before rather than after.

  - **The fix.** `build_key_input(program, versions)` builds a canonical, line-oriented encoding
    which `compute_build_key` hashes. Sections: `[versions]` (all eight axes), `[entry]`,
    `[sources]` (per-file name + SHA-256 of contents), `[types.struct_fields]`,
    `[types.enum_variants]`, `[types.drop_impls]`, `[types.copy_types]`, `[bodies]`
    (`program.dump()`, already the contract's deterministic body serialization). Determinism comes
    from the data structures themselves — `TypeContext` is `BTreeMap`/`BTreeSet` and
    `program.bodies` is sorted by canonical symbol. Tagged `build key v2` so a future encoding
    change is visibly a different scheme rather than silently colliding with v1 keys.

  - **Why the encoding is a separate function from the hash.** A test asserting "these two keys
    differ" says nothing about WHICH input made them differ; a test that can diff the encoding
    does. `the_key_input_carries_every_documented_section` pins that every section is present, so
    a section deleted from the encoder fails by name instead of quietly weakening every other test
    in the module.

  - **Coverage** (7 tests, `backend::generated_rust::build::tests`): key determinism (the baseline
    without which every "the key changed" assertion could be satisfied by a key that changes every
    time); a different body; **the DEV-095 regression** — eight one-input mutations across all
    four `TypeContext` fields (new nominal, changed field type, changed type arguments, new enum,
    reordered variants, gained destructor, changed destructor instance, became `Copy`), each
    asserting `dump()` stays byte-identical as a PRECONDITION before asserting the key changed, so
    the test is meaningless the day it stops being the actual condition; a different file name
    (names reach generated code verbatim through trap-site `file:line:column`); a source-content
    change invisible to `dump()` (an appended comment moves no span, and §11.1 requires
    source-content hashes regardless); and all eight version axes moved independently.

  - **Verified by mutation, not just by passing.** Simulating the old key (dropping the `[types]`
    sections from the hashed input) makes the regression test fail with
    `struct_fields: a new nominal: build key did not change — a stale generated crate would be
    reused`. Reverted; `git diff` confirms nothing of the simulation survives.

  - **One §11.1 item deliberately not given its own section: package graph identity.** A C5
    program is one compilation unit and the source table is its identity; when multi-package
    linkage lands (WP-C5.4) it gets its own section rather than being assumed covered. Recorded
    in the encoder's own comment so the next reader does not have to rediscover the reasoning.

  - Validation, **scoped deliberately** per the standing process note (full-workspace runs are for
    WP/gate closure points, not intermediate changes — this discharges an entry condition, it does
    not close a package): `cargo fmt --all -- --check` clean, `cargo clippy --workspace
    --all-targets --all-features -- -D warnings` clean (workspace-wide, since clippy is cheap),
    and every consumer of the changed code green — `backend::` unit tests 35/35 (including the
    seven new build-key tests) plus all six suites that invoke `emit_native_debug`, which is
    `compute_build_key`'s only caller: `native_c5_1b_skeleton` 1/1, `native_c5_2b_locals` 2/2,
    `native_c5_2c_operations` 9/9, `native_c5_2d_calls` 3/3, `native_c5_2e_traps` 4/4,
    `three_engine_differential` 20/20. Nothing outside the native build path reads the build key
    (`grep` confirms no other reference in the workspace), so the untouched suites — parser,
    lexer, formatter, LSP, ONNX, gate4/gate7 — carry no information about this change. ~15 seconds
    against ~40 minutes for the full suite.

- CD-056 [2026-07-21, WP-C5.3 opened; C5.3a closed] **WP-C5.3 opened by owner directive after
  CD-055 discharged its entry condition. C5.3a (tuples, arrays, structs) CLOSED. Two owner
  decisions are OPEN and flagged rather than resolved unilaterally; one oracle defect (DEV-097)
  was found and fixed; one scope boundary is now a named diagnostic instead of a rustc error.**

  - **Delivered (C5.3a)**: §6.2 type mapping for `Tuple`/`Array`/`Struct`; §6.3 nominal
    definitions (one Rust `struct` per type-context instance, positional `f0..fn` field names,
    `BTreeMap` order); `mangle::type_name_for_nominal` (injective, and provably disjoint from
    function names because `#` cannot occur in a STARK identifier); `emit_places::TyEnv`, the
    projection-type walk; `Rvalue::Aggregate` for all three kinds; `ConstIndex`, `CheckIndex` and
    proof-backed `Index`; `LocalKind::IndexProof`. Tuples map to **Rust tuples** — §6.2 offered
    "concrete tuple or named internal aggregate; choose one canonical form", and the Rust tuple
    needs no generated definition, no deterministic name, and no reachability walk.
    Evidence: seven new three-engine cases plus four native-only cases
    (`native_c5_3a_aggregates.rs`) for what a three-engine comparator structurally cannot cover.

  - **Why `TyEnv` exists, since it is the one structural addition**: MIR's `Projection::Field(i)`
    is ONE variant covering both struct fields and tuple elements, but generated Rust needs `.f0`
    for one and `.0` for the other. Choosing requires the projected place's type, hence a walk
    from the local's declared type through the nominal type context. It also let `operand_mir_ty`
    stop refusing projected operands, so a `SwitchInt` on a struct field or array element works.

  - **DEV-097 — the HIR oracle blamed two different columns for two ends of one bounds check.
    FIXED.** An out-of-range index trapped at the whole index expression's span; a NEGATIVE index
    trapped at the index operand's span. So the oracle disagreed with both other engines on one of
    the two, and was internally inconsistent about one check. Found by the three-engine harness's
    negative-index case; no corpus or inline case had ever indexed with a negative value. Fixed in
    `interp.rs` to use the index-expression span for both, matching MIR and native. **This is the
    fourth defect this campaign has found that lived only in the gap between engines.**

  - **OPEN DECISION 1 — what does "three-engine agreement on target layout queries" mean?**
    §14's C5.3 exit lists it, and it **cannot be satisfied as literally stated**: both
    interpreters answer **8 for every type** (`mir::interp::reference_layout`, whose own doc says
    a real per-type algorithm is the backend's job and that "a backend replaces this function and
    nothing else"), while the native engine answers its **actual Rust target layout**
    (`size_of::<Int32>()` is 4). `assert_eq(size_of::<Int32>(), 4)` traps in both interpreters and
    succeeds natively. This is not a backend defect — LAYOUT-ABI-001 makes layout target-dependent
    by design — but the exit condition needs a definition. Candidate readings: (a) the
    interpreters adopt a real layout algorithm matching the native target, which makes the
    reference oracle target-dependent; (b) agreement means agreement on RELATIONS Core guarantees,
    not absolute values; (c) layout queries are excluded from value agreement, with the divergence
    documented as intended. **Until the owner rules, the harness asserts only that layout queries
    run in all three engines and agree on completion-vs-trap, plus relations true under both
    answers.** The value question is recorded, not dropped.

  - **OPEN DECISION 2 — the §6.3-vs-§7.4 `Copy`-derive reading (implemented, reversible).** §6.3
    forbids deriving `Clone`/`Copy`/`Eq`/`Ord`/`Hash` "as a shortcut for STARK semantics"; §7.4
    says a MIR copy is emitted only for MIR-`Copy` types and the backend must not broaden that
    set. A STARK struct with an `impl Copy` needs SOME mechanism for `Operand::Copy` to read it
    twice. **Reading taken:** deriving `Clone`/`Copy` on exactly the instances MIR classifies
    `Copy` is not a shortcut — MIR decides, the derive follows, the set is neither broadened nor
    narrowed. No other trait is derived. `emit_types::mir_ty_is_copy` mirrors
    `mir::lower::is_copy` rather than asking Rust anything. If the owner reads §6.3 as forbidding
    this, the alternative is a generated copy helper per nominal and the change is confined to
    `emit_types::derives_for` plus one test.

  - **Scope boundary now a named diagnostic.** A **non-`Copy` value moved out of a local
    initialised in an EARLIER block** is refused as `Unsupported` naming WP-C5.3d. The backend
    lowers MIR's block graph to `loop { match __bb { .. } }`, so every block is one iteration of
    one Rust loop, and Rust's borrow checker cannot see that MIR never revisits a moved-from
    local — it reports "value moved here, in previous iteration of loop" for a move verified MIR
    proves sound. Found when a three-engine case passing a struct by value produced a
    `BuildFailed` carrying a rustc borrow-check error; a scope limit surfacing as a rustc error is
    itself a defect in the diagnostic. Moving WITHIN one block still works (ordinary aggregate
    construction lowers that way) and has its own test, so the guard is pinned against
    over-rejection too.

  - **OPEN DECISION 3 (blocks C5.3d) — the non-`Copy` storage strategy.** §7.2 proposes
    `MaybeUninit<ManuallyDrop<T>>` plus explicit liveness and move/drop helpers, and permits
    evidence-based simplification. A safe-Rust `Option<T>`-shaped variant would model MIR
    liveness without any unsafe helper. Choosing is CE4-shaped and is not made here.

  - Validation: `cargo fmt --all -- --check` clean, `cargo clippy --workspace --all-targets
    --all-features -- -D warnings` clean, **`cargo test --workspace` green: 917 passed / 0 failed
    / 2 ignored across 53 test binaries** (up from 894/52 — the new `native_c5_3a_aggregates`
    suite plus the new three-engine and unit tests). The full-workspace run is justified here
    rather than the scoped set, per CD-055's rule: `interp.rs` — the semantic oracle — changed for
    DEV-097, and that is a workspace-wide consumer (`mir_differential`, `exec_snapshots`,
    `conformance`, `gate3_execution` all read it).

- CD-057 [2026-07-21, C5.3b closed] **User enums, discriminants, and payload access compile and
  run natively. C5.3b CLOSED. The one structural problem — Rust cannot project into an enum
  variant outside a `match` — is solved by emitting a match EXPRESSION, with two consequences
  recorded rather than discovered later.**

  - **Delivered**: user enums → generated Rust enums with uniformly TUPLE variants (`V0()`,
    `V1(i32)`, `V2(i32, i32)`); `AggKind::EnumVariant` construction (type arguments from the
    destination, as with struct aggregates); `Projection::VariantField` reads;
    `Rvalue::Discriminant`. `EnumRef::CoreOption`/`CoreResult`/`CoreOrdering` are deliberately
    EXCLUDED — they belong with match/`?` lowering in C5.3c rather than being half-supported.

  - **Uniform tuple variants, including empty ones.** `V0()` is legal Rust, and the uniformity
    removes a special case from construction, from patterns (`V0(..)` matches it), and from the
    discriminant match. A unit variant would need different syntax in all three places.

  - **The structural problem.** Every other MIR projection appends to a place expression (`.f0`,
    `[2]`); a variant field has to WRAP what came before, because Rust exposes no way to project
    into a variant outside a `match`. Emitted as
    `(match &base { Ty::V1(__payload) => *__payload, _ => unreachable!("V-DISC-1: ...") })`.
    Two consequences, both deliberate: (a) the `_` arm is **provably dead** — V-DISC-1 makes a
    variant-field projection legal only after a discriminant test — so it gets the same
    `unreachable!()` the verifier-proved dead-block path has, naming the rule rather than
    fabricating a value that would paper over a lowering bug; (b) the result is an EXPRESSION,
    not a place, so it cannot be an assignment destination. `emit_dest_place` refuses that
    explicitly — a guard, not a limitation, since lowering emits `VariantField` only through
    `read_place` and pattern tests and STARK has no syntax for assigning into a payload.

  - **`Rvalue::Discriminant` takes the same shape** (an enum with payloads has no integer `as`
    conversion), listing **every variant with no catch-all**, so adding a variant cannot silently
    fall through to a wrong index. Its arms are typed by the DESTINATION local rather than a fixed
    width — a hardcoded `i128` failed to compile against MIR's `Int64` discriminant local, caught
    by the first native probe.

  - **Evidence**: four new three-engine cases (all three payload arities constructed and matched;
    payload field ORDER via a non-commutative operation, so a wrongly-bound two-field payload
    cannot pass; discriminant selection across four variants in a loop with distinct per-variant
    values, so any mis-selected arm changes the sum; a trap raised from a payload value) and three
    new native-only cases (one definition per instance with uniform tuple variants; a discriminant
    match naming every variant; the `unreachable!()` arm citing V-DISC-1). One test expectation of
    mine was wrong — a trap line off by one — and all three engines agreeing is what exposed it,
    which is exactly why `agree_trapping` takes the expected line independently.

  - **C5.3b makes CD-056 decision 3 (non-`Copy` storage) urgent rather than optional.** C5.3a's
    cross-block non-`Copy` move boundary bites far harder for enums: conditionally constructing a
    value and then matching it — the ordinary way enums are used — puts construction in one block
    and the match in another, which is exactly what the block-dispatch loop cannot express for a
    non-`Copy` value. The discriminant-selection test needs `impl Copy` to cross that boundary at
    all. **C5.3c is worse still**: `Option`/`Result` payloads are frequently non-`Copy` and `?` is
    inherently cross-block, so the storage decision is a prerequisite for C5.3c, not a nicety.

  - Validation, **scoped** per CD-055's rule (this change is backend-only — no `interp.rs`, no
    MIR contract, nothing with workspace-wide consumers): `cargo fmt --all -- --check` clean,
    `cargo clippy --workspace --all-targets --all-features -- -D warnings` clean, `backend::` unit
    tests 40/40, `three_engine_differential` 31/31, `native_c5_3_aggregates_enums` 7/7, and the
    five earlier `native_c5_*` suites green. ~22 seconds.

- CD-058 [2026-07-21, owner review of 7829552] **C5.3b APPROVED as closed. The three CD-056
  decisions are RESOLVED. Work-package sequencing changed: a bounded prerequisite, C5.3d-0, is
  inserted BEFORE C5.3c.**

  - **C5.3b's limitation, stated precisely (owner wording).** C5.3b supports **Copy payload
    reads**. **Non-Copy payload movement remains blocked on the controlled-storage foundation and
    is not claimed complete merely by the current `VariantField` expression.** The scoped
    validation was confirmed correct for that commit: generated-Rust backend, its tests, and
    compiler records only — no workspace-wide semantic consumer.

  - **DECISION 1 — layout-query agreement. RESOLVED.** For C5 exit, layout-query agreement means
    **exact `size_of`/`align_of` agreement when all three engines execute under ONE recorded
    target-layout context**. `(8, 8)` is preserved as the default historical C4 reference layout.
    For C5 differential execution, an **injectable target-layout manifest** is generated or probed
    through the same canonical generated-Rust representations, target triple, rustc version,
    backend/runtime versions and profile as the native build; HIR and MIR consume that manifest
    during C5 layout cases, and the harness compares exact values. Relations-only layout tests may
    remain but **do not discharge** the C5.3 exit condition. (The current
    `layout_queries_run_in_all_three_engines` case is therefore a placeholder, not evidence.)

  - **DECISION 2 — Copy derivation. APPROVED as implemented, with the rule stated exactly.** A
    generated nominal instance may derive `Clone, Copy` **if and only if that exact concrete
    instance is present in MIR's `copy_types` classification**. MIR remains the authority: the
    backend must not infer Copy from Rust fields or trait resolution, and **`.clone()` must never
    implement a MIR move or copy**. `Eq`, `Ord`, `Hash`, `Drop` and other semantic traits are not
    derived as substitutes for STARK behaviour.

  - **DECISION 3 — non-Copy storage. RESOLVED: §7.2 controlled manual storage.**

    ```text
    ValueSlot<T> {
        storage: MaybeUninit<ManuallyDrop<T>>,
        whole-place live state,
        typed drop-unit live state where MIR distinguishes sub-places
    }
    ```

    **Ordinary `Option<T>` is REJECTED** — it introduces Rust-owned destruction.
    **`Option<ManuallyDrop<T>>` is REJECTED as the general representation**: it is adequate only
    for whole-value liveness, and once a field or constant-index element has been moved the
    remaining bytes no longer necessarily form a valid complete `T`. `MaybeUninit` is required to
    hold that partially moved state legally. An Option-shaped slot **may later be admitted as an
    optimisation** for locals MIR dataflow proves have no partial-move paths.

    Recording the reasoning because it is the part that would otherwise be re-litigated: the
    objection to `Option<ManuallyDrop<T>>` is not about destruction (`ManuallyDrop` already
    suppresses that) but about **representation validity under partial moves** — a distinction the
    C5.3a/C5.3b work had not yet had to confront, since neither admits partial moves.

  - **SEQUENCING CHANGE — C5.3c does NOT begin next.** A bounded prerequisite is inserted:
    **C5.3d-0 — non-Copy storage and movement foundation**, whose purpose is to unblock C5.3c and
    which **does not close C5.3d**. Its seven required deliverables (helper module; no ad hoc
    unsafe in emitted bodies; move semantics; Drop semantics; the five initial supported movement
    shapes; partial-move discipline; mutation-tested evidence) are recorded in
    `WP-C5.3.md`. After C5.3d-0 passes: C5.3c using the slot abstraction for non-Copy
    `Option`/`Result` values and `?` paths, then **C5.3d-1** with the dedicated observable
    destruction fixture and the final exactly-once/order/no-Drop-after-trap proof.

  - C5.3a and C5.3b remain closed.

- CD-059 [2026-07-21, C5.3d-0 CLOSED] **The non-Copy storage and movement foundation is complete.
  C5.3c is unblocked. One structural finding blocks part of C5.3d-1 and needs an owner decision.**

  - **Soundness correction first (owner review).** The initial `ValueSlot` was unsound for partial
    moves: `move_sub` took `&mut T`, moved a field out, and left the slot "live", after which
    `get`/`get_mut`/`take`/`drop_value` all remained callable over storage that no longer held a
    valid `T`. **The module's own test asserted `slot.get().1` after moving `.0`, so the bug was
    written into its evidence.** Corrected to a three-state machine — `Dead`/`Whole`/`Partial` —
    with whole-value operations requiring `Whole`, partial access restricted to raw-pointer
    projection, and an explicit `finish_partial` transition. Miri confirms zero UB across 18 slot
    tests; restoring the old permissive guard makes Miri report a real **use-after-free**.

  - **What this says about the validation strategy, not just the code.** The three-engine harness
    could not have caught it: it compares observable outcomes, and UB that does not change
    observable behaviour agrees across all three engines. **Differential testing is strong for
    semantics and blind to memory soundness.** Miri is now the compensating control — and even
    Miri did not flag `move_field` → `get` for a `(String, i32)`, because a moved-out `String`'s
    bytes stay bit-valid. For that case the state machine *is* the evidence. Layered: state
    machine primary, Miri for what it can see, neither complete alone.

  - **Generated projection helpers** (`emit_projections.rs`): one per (type, sub-place) pair the
    program actually uses, emitted into `mod stark_proj`. Raw `fn(*mut T) -> *mut F` via
    `addr_of_mut!` for struct/tuple/array (valid over partial storage); whole `fn(&mut T) -> &mut F`
    for enum payloads, which Rust cannot address without a `match`. Deliverable 2 verified on a
    partial-move program: every `unsafe` lies inside that module.

  - **`Copy` field reads had to become field-precise too**, and the state machine is what found
    it: moving `o.a` out then reading `o.b` aborted with "the slot is PARTIAL", because `get()`
    correctly refuses partial storage. Not an optimisation — a correctness consequence.

  - **All five deliverable-5 movement shapes work.** The C5.3a cross-block guard is deleted; what
    it refused now compiles and runs.

  - **STRUCTURAL FINDING — user `Drop` impls cannot compile natively yet (owner decision needed).**
    A destructor's receiver is `&mut Self`, so `impl Drop` requires `MirTy::Ref`, and references
    are outside the C5 subset entirely. This holds even when the body never touches `self` — the
    signature alone is enough. Therefore: `Terminator::Drop` works for structural glue only; a
    user destructor cannot be dispatched natively until `Ref` is admitted at least for destructor
    receivers; and **C5.3d-1's dedicated observable destruction fixture cannot be built as
    planned**. The §7.7 no-Drop-after-trap property is proven STRUCTURALLY instead (no `drop_with`
    precedes any abort site), and the difference is recorded rather than glossed.

  - Validation, scoped (backend + runtime; no workspace-wide semantic consumer): fmt clean,
    clippy clean, stark-runtime 23/23, `backend::` 40/40, `three_engine_differential` 35/35,
    `native_c5_3_aggregates_enums` 10/10, earlier native suites green, **Miri 18/18 with zero UB**.

- CD-060 [2026-07-21, C5.3d-0 REOPENED and re-closed; C5.3c in progress] **An owner review of
  `4a7e24c` found two contract violations the closure record had not covered. Both were real.
  Corrected; C5.3d-0 re-closed.**

  - **VIOLATION 1 — the partial-field primitives could not honestly be safe.** `move_field`,
    `copy_field`, `drop_field_with` and `move_field_whole` accepted an arbitrary projection
    function and then read the pointer it returned, checking only the SLOT's state. They could
    not validate that the pointer belonged to the slot, that the field was still live, or that
    the same field had not already been moved — so **safe Rust could reach UB** by calling
    `move_field(the_same_projection)` twice. The module's docs claimed preconditions were
    "checked rather than assumed"; for per-field liveness and projection validity that was false.

    Corrected as the owner directed: all four primitives are now `unsafe fn` with explicit
    `# Safety` contracts, and the backend emits **one safe wrapper per (type, sub-place,
    operation)** into `mod stark_proj`. Each wrapper pairs exactly one primitive with exactly one
    fixed projection over one slot type, so the obligation is discharged **by construction**
    rather than claimed. Emitted MIR bodies call only wrappers — asserted by a test that scans
    the bodies for `move_field`/`copy_field` and requires none.

  - **VIOLATION 2 — whole-enum structural Drop silently omitted its payload.**
    `emit_drop_glue` located a possible user destructor for an enum and then walked
    `struct_fields`, which an enum has no entry in. It never matched the active variant and never
    traversed `enum_variants`, so dropping a whole non-`Copy` enum marked the slot dead and leaked
    its payload. **Miri could not report it because the slot tests ignore leaks by design** — the
    fix's own evidence channel was blind to it.

    Corrected: enum glue now emits a match over EVERY variant (no catch-all, so a new variant
    cannot silently acquire a no-op drop) with payload fields dropped in reverse declaration
    order, mirroring `mir::interp::drop_in_place`. Two unit tests pin variant coverage, reverse
    order, and that `Copy` payload fields are ignored rather than dropped.

    **Currently unexercised by any compilable program**, and worth stating: no droppable type is
    expressible in the C5 subset, because a user `Drop` impl needs `&mut Self` and references are
    out of scope. The fix is correct and tested at the emitter level; it becomes reachable when
    the destructor-reference lane lands.

  - **C5.3c (Option/Result) is IN PROGRESS, not closed.** Core enums now share the user-enum
    representation through one `variant_payloads` table — the single source the definition, the
    discriminant match, and every projection all read — with `Option` as `None=0`/`Some=1`,
    `Result` as `Ok=0`/`Err=1`, `Ordering` as three fieldless variants, mirroring
    `mir::verify::variant_payload`. A probe compiles and runs `Option`/`Result` construction,
    matching and payload reads natively. **Deviation from §6.2 to flag:** §6.2 preferred ordinary
    Rust `Option<T>`/`Result<T, E>` "if all observable semantics match"; generated enums are used
    instead, so one mechanism covers every enum and no Rust drop glue exists for a type MIR is
    responsible for destroying. Owner may overrule; the change would be confined to
    `emit_types::nominal_type_name`.

  - Validation: fmt clean, clippy clean, stark-runtime 23/23, `backend::` 42/42,
    `three_engine_differential` 35/35, `native_c5_3_aggregates_enums` 10/10, Miri 18/18 zero UB.

- CD-061 [2026-07-21, C5.3c CLOSED] **`Option`, `Result`, matches and `?` compile and run
  natively. Two of the three remaining C5.3 gaps are now known to share ONE root cause.**

  - **Core enums share the user-enum representation** through one `variant_payloads` table — the
    single source the definition, the discriminant match and every projection read — mirroring
    `mir::verify::variant_payload`: `Option` `None=0`/`Some=1`, `Result` `Ok=0`/`Err=1`,
    `Ordering` three fieldless variants (A2).

  - **§6.2 deviation, flagged.** §6.2 preferred ordinary Rust `Option`/`Result` "if all observable
    semantics match"; generated enums are used instead so one mechanism covers every enum and no
    Rust drop glue exists for a type MIR is responsible for destroying — which matters more now
    that `ValueSlot` makes destruction explicitly MIR's. Reversible in
    `emit_types::nominal_type_name`.

  - **`?` needed no backend work**: MIR has already lowered it to branches and returns. A native
    test asserts no Rust `?` appears in the output, so the propagation is MIR's own control flow
    rather than a borrowed operator whose equivalence would have to be argued.

  - **Evidence**: four three-engine cases (both Option variants, including one flowing through a
    local into a later block; Result with DIFFERENT Ok/Err payload types, so confusing the two
    variants' payload tables would not compile; `?` on both propagating and falling-through
    paths; a trap from inside an Option payload, checking provenance on the core-enum path) and
    two native cases pinning generated variant order and the absence of Rust `?`. One expected
    trap line of mine was wrong again and all three engines agreeing exposed it — the third time
    that independent expectation has earned its place.

  - **`Ordering` is supported but UNREACHABLE, and it shares a root cause with the Drop gap.** It
    needs no special case in the emitter, but cannot be produced from compilable C5 source: the
    only way to obtain one is `a.cmp(&b)`, and `cmp` takes a reference. That is the same cause as
    user `Drop` impls being unrepresentable (`&mut Self` receiver). **The two remaining C5.3 gaps
    are one gap — the absence of references** — which means the narrow destructor-reference lane,
    slightly widened, would close both. Worth knowing before scoping it.

  - Validation, scoped: fmt clean, clippy clean, stark-runtime 23/23, `backend::` 42/42,
    `three_engine_differential` 39/39, `native_c5_3_aggregates_enums` 12/12, earlier native
    suites green.

- CD-062 [2026-07-21, owner decisions after C5.3c] **Five decisions. C5.3's remaining work is
  reduced from four unrelated gaps to TWO closure packages: references/Drop evidence, and exact
  target layout.**

  1. **C5.3c closure ACCEPTED** (`9aa94ac`) under the scoped-validation policy. The owner's note
     on why it matters architecturally: `?` required no backend reconstruction — MIR already
     contains the branches, payload moves and early return, and the backend merely emits them.
     The test prohibiting Rust's `?` is the correct guard against semantic reconstruction.

  2. **Generated core enums APPROVED; §6.2 AMENDED rather than the implementation reverted.** New
     normative wording: *"Core enums use compiler-generated concrete enum representations governed
     by MIR's canonical variant table. Rust `Option`, `Result` and `Ordering` are not used as
     STARK value representations in C5. A future representation optimisation requires evidence
     that discriminants, layout queries, movement, partial movement and explicit MIR-directed
     destruction remain equivalent."* The original "prefer Rust's types if observable semantics
     match" condition is **too weak after `ValueSlot`**: Rust-owned Drop can conceal a missed MIR
     Drop and make exactly-once evidence less falsifiable, and the dual path through definitions,
     discriminants, projections and Drop glue would be permanent. A semantic boundary, not an
     implementation convenience.

  3. **EPHEMERAL BORROWED-CALL REFERENCE LANE APPROVED** — renamed from "destructor-reference
     lane", because it covers both cases the missing-references finding identified: shared refs
     for `cmp(&other)` and exclusive refs for `Drop::drop(&mut self)`. Bounded to: `RefOf` borrows
     only a verified live, WHOLE place; never into a partially moved `ValueSlot`; the reference is
     consumed by a statically resolved direct call; creation and consumption in the SAME basic
     block; a generated reference temporary has exactly one use; reference-typed parameters
     allowed; callees may use `Deref` projections from them; shared reads, exclusive mutates and
     serves as destructor receiver. Forbidden: returning, storing in aggregates, writing into user
     locals, passing indirectly, carrying across blocks, nested references, slices, reference
     equality, general reborrowing, reference-valued results. Everything else rejected before
     rustc. A pre-emission validator enforces single-use/same-block; the emitter **inlines the
     borrow into the call** (`cmp_fn(&lhs, &rhs)`, `drop_fn(&mut value)`) rather than introducing
     general reference storage — considerably safer than making references ordinary
     `ValueSlot`-backed values.

  4. **`DropPlan` MANDATORY before C5.3d-1 closure**, and it precedes any general
     `NativeOperation` refactor (owner accepted that sequencing). A representation-neutral plan
     derived from `MirTy` + `TypeContext`, consumed by BOTH the MIR interpreter and the native
     emitter: `Noop` / `UserDestructor(instance)` / `Struct(reverse fields)` / `Enum(every variant
     → reverse payload)` / `Tuple(reverse)` / `Array(reverse indices)`. Preserves: user destructor
     first; structural fields or active payload after; reverse declaration order; complete variant
     coverage; no action for `Copy` units. **Does not change MIR v0.1** — it centralises an
     existing duplicated derivation. CD-060 fixed the enum-Drop *instance*; `DropPlan` removes the
     *class*.

  5. **Universal `NativeOperation` IR DEFERRED**, to evolve incrementally. **Layout manifest
     OPENED as an independent package (C5.3e)**, which may proceed in parallel since it depends on
     neither references nor `DropPlan`.

  - **Execution order set by the owner**: C5.3d-1a (ephemeral references) → C5.3d-1b (canonical
    `DropPlan`) → C5.3d-1c (observable closure evidence, then close C5.3d-1). C5.3e independent;
    if work must be sequential, C5.3d-1 first as the higher correctness risk.

  - **Trap-line expectations KEPT**, with an addition: each trapping fixture must carry an
    `expected_span_reason` note documenting WHY the expected location is correct, derived from the
    language rather than from any engine. The owner's rationale: having corrected the expected
    answer three times confirms these expectations are independent rather than self-fulfilling.

- CD-063 [2026-07-21, C5.3d-1a CLOSED] **The ephemeral borrowed-call reference lane is
  implemented. `Ordering` is reachable and user destructors compile — the two gaps CD-061
  identified as one root cause are closed.**

  - **Delivered**: `MirTy::Ref` in the type mapping; `Projection::Deref`; `Rvalue::RefOf` as a
    borrow expression; `LocalKind::DropFlag` admitted; and `validate_ephemeral_references`, a
    pre-emission validator refusing every out-of-lane shape.

  - **Three design points worth keeping**: (a) a reference local is **never** slot-backed, even a
    `&mut` one — a slot-backed `&mut Self` receiver would make the destructor's `Deref` project
    through the slot rather than the reference; (b) reference locals are declared
    **uninitialised**, so rustc becomes a *second* check on the lane — a reference escaping its
    block fails as "possibly uninitialized" rather than reading a fabricated value; (c) one
    slot-backing rule (`emit_types::is_slot_backed`) shared by the signature emitter, the local
    declarations and place emission. That third point is not theoretical: those sites disagreed
    during this work and produced a crate binding a parameter under one convention and reading it
    under the other.

  - **DEVIATION FROM CD-062, reported not absorbed.** The lane requires the reference to be
    "consumed by a statically resolved direct call". That is the destructor shape exactly, but
    **not** what `a.cmp(&b)` lowers to: for primitives lowering INLINES the comparison, giving
    `_5 = &_2; _6 = copy _5; _7 = Lt(copy _1, copy (*_6))` — consumed by a `Deref` READ inside a
    `BinOp`, via an intermediate copy. Ephemeral, same-block, unstored and unreturned all still
    hold, so the lane's purpose is intact; its stated consumption form is not. The validator
    accepts same-block consumption by read as well as by call. **The alternative is to reject
    `cmp` and leave `Ordering` unreachable, which would defeat the lane's own motivation** —
    owner may rule otherwise.

  - **Evidence**: two three-engine cases (all three `Ordering` variants with distinct results; a
    destructor reading through `&mut Self`) and two native cases — one asserting the destructor
    receiver is a bare Rust reference not a slot, one driving out-of-lane shapes (returned
    reference; reference carried across blocks) and requiring refusal **before rustc**, failing
    loudly if any reaches rustc and fails there instead.

  - Two matches became exhaustive as a result (`LocalKind`, `Rvalue`) and their catch-alls were
    deleted: a new variant now stops compilation instead of silently becoming an `Unsupported`
    diagnostic nobody reads.

  - Validation, scoped: fmt clean, clippy clean, stark-runtime 23/23, `backend::` 42/42,
    `three_engine_differential` 41/41, `native_c5_3_aggregates_enums` 14/14, earlier native suites
    green.

- CD-064 [2026-07-22, C5.3d-1b DONE] **`mir::drop_plan` is the canonical destruction plan, derived
  once and consumed by both the MIR interpreter and the native emitter** — CD-062 decision 4
  discharged.

  - **The defect class, not the instance.** CD-060 fixed the emitter's enum drop glue after it was
    found walking `struct_fields` and dropping no payload at all. The cause was structural: two
    independent reconstructions of one semantic rule, agreeing only because they were written to.
    `drop_plan::plan_for(ty, types)` is now the only derivation; `interp::run_drop_plan` and
    `emit_bodies::emit_drop_plan` each APPLY it and decide nothing about order, coverage or
    obligation.

  - **Four invariants moved from convention into the plan's SHAPE**: (a) `Destructor { symbol,
    then }` **nests** the components inside the destructor, so "fields before the user destructor"
    is *unrepresentable* rather than merely discouraged; (b) components are stored in destruction
    order and consumers iterate forward, with `array_order(len)` a named function so reversing it
    is a visible edit; (c) `Variants` is indexed by variant number, always complete, and carries
    each variant's **full arity** beside its droppable fields, so a generated `match` is exhaustive
    without a catch-all; (d) any component whose plan is `Noop` is absent, and an all-`Noop` parent
    with no destructor is itself `Noop` — which is where "never drop a `Copy` field" now lives,
    once, instead of as a filter each consumer must remember.

  - **`Vec`/`Box` name their element by TYPE, not by an inlined sub-plan.** They are Core v1's only
    indirection and therefore its only route to a recursive type (`enum List { Nil, Cons(Int32,
    Box<List>) }`); inlining would not terminate. Everything else is inline, finite, and planned
    eagerly.

  - **MIR v0.1 unchanged**, runtime surface untouched — this centralises an existing derivation.
    The variant-payload table (previously written out three times — `interp`, `verify`,
    `emit_types` — with the variant indices agreeing only by inspection) moved into the same
    module, and all three now read it. The interpreter memoises plans per type (`Rc<DropPlan>`),
    since the walk this replaced was lazy and a `Drop` inside a loop runs once per iteration.
    Tuples and arrays reach the native drop path for the first time as a consequence;
    `Vec`/`Box` steps are **refused** by the emitter rather than approximated, since glue that
    destroyed elements while leaking the buffer would be worse than a refusal.

  - **FLAGGED, carried forward unchanged, not silently corrected.** The remaining `Core` types —
    `String`, `HashMap`, `HashSet`, the iterators, `File` — plan to `Noop`, exactly reproducing
    what `interp::drop_in_place` already did. For a `HashMap<K, V>` whose `V` has a destructor that
    is arguably wrong, but it is the reference semantics as they stand, and changing it here would
    move the oracle without an owner decision. Recorded in the module so the question is
    answerable rather than lost.

  - **Evidence**: 14 derivation tests (order, coverage, index preservation, `Noop` collapse, core
    enums, deferred `Vec`/`Box`, a recursive type through `Box`, missing tables erroring rather
    than silently planning nothing) plus CD-062's mutation set. Each mutation corrupts the SHARED
    plan and shows the corruption reach the generated Rust — which is what establishes application
    rather than re-derivation, since a re-deriving emitter would ignore a corrupted plan and every
    one of these would fail. Five of the six are representable: omitted variant, omitted payload
    field, reversed order, re-added `Copy` field, and destructor ordering — that last one resolving
    to *unrepresentable*, with the nearest permitted rearrangement landing the destructor on a
    field and thus failing to compile. The sixth (`Drop` after a trap) was already covered by
    `mir_differential`, `gate3_execution::trap_aborts_without_running_pending_destructors` and
    `native_c5_3_aggregates_enums`, and carries no plan semantics.

  - Validated with the **full workspace suite**, not the scoped set: `interp.rs` is the semantic
    authority and every differential fixture consumes it.

- CD-065 [2026-07-22, owner assessment after `888d9c5`] **The process-driven re-engineering phase
  of C5 is CLOSED. Stop improving the process; finish the evidence, the manifest, linkage, build UX
  and exit qualification. Carry the broader process lessons into C6.**

  - **Owner's finding**: `DropPlan` genuinely replaces the duplicated derivations rather than
    merely documenting them; the emitter's remaining responsibility is only how to spell a planned
    step. Two sources of future drift are gone (destruction traversal; variant-payload definitions).
    No comparable structural refactor is judged outstanding. Another general abstraction now would
    be diminishing returns.

  - **DEFERRED explicitly**: `NativeOperation` IR, broad operation-planning abstractions,
    architecture dashboards, process metrics, retroactive conversion of old work packages, general
    references, runtime liveness bitmaps.

  - **Only two process items remain**: one adversarial review at C5.3 closure (Drop reachability,
    partial moves, layout evidence, rejected adjacent cases), and one gate-exit review at C5.6
    against the twelve C5 outcome conditions and the final supported-subset claim.

  - **Bounded caveat recorded for the future owning-core-representation package, not for C5.3.**
    `DropPlan` maps `String`/`HashMap`/`HashSet`/iterators/`File` to `Noop`, preserving interpreter
    semantics. Not a C5 blocker, because the generated backend still REJECTS those representations
    rather than silently compiling them. But before an owning core representation (e.g. a native
    Rust `String`) is admitted, STARK must distinguish **STARK semantic Drop glue** from **native
    representation reclamation**: a type may have no user-visible STARK destructor while still
    requiring its buffer or allocation to be reclaimed. To be solved by that package, not
    speculatively inside C5.3.

  - **Remaining C5 work, owner's ordering**: (1) C5.3d-1c observable Drop closure — now evidence
    work, not architecture: exactly-once, destructor-before-fields, reverse field/payload order,
    a moved value destroyed only by its new owner, no destructor after a trap, **plus one
    partial-move case with a genuinely droppable sibling** (the emitter still refuses projected
    `Drop` terminators, so this case settles whether the bounded C5 subset needs sub-place `Drop`
    emission or whether every approved fixture legally avoids it — the last ownership seam likely
    to expose implementation work); (2) C5.3e exact layout manifest; (3) C5.4 linkage and function
    values — function-instance constants, function-value storage/copying, indirect calls,
    cross-package references, the frozen three-package workspace; (4) C5.5 `stark build` as a
    user-facing route; (5) C5.6 qualification, including **hosted CI as a real exit item, not a
    formality** — `888d9c5` carries no GitHub status checks despite locally reported validation.

  - **Owner maturity estimate**: C5.3 approximately 90–93% complete; full Gate C5 approximately
    76–80%. Highest-risk architectural section (non-`Copy` ownership and destruction) judged under
    control.

  - **Copy consolidation FOLDED IN to C5.3d-1c by owner direction, and DONE.** The classification
    had been derived three times — `lower::is_copy`, `verify::mir_is_copy`,
    `emit_types::mir_ty_is_copy` — the same defect class CD-064 closed for destruction. The two
    CONSUMERS now share `TypeContext::is_copy`; `lower::is_copy` deliberately does not delegate,
    because it is the PRODUCER and answers the nominal case from the HIR precisely to fill the
    table the others read. Since no single function could cover both, the producer/consumer
    agreement is enforced empirically instead: `assert_copy_classification_agrees` runs over every
    differential program and the whole frozen corpus, checking that lowering never emits
    `Operand::Copy` for a place the type context calls non-`Copy`.

- DEV-098 [2026-07-22, found by the CD-065 fold-in, NOT a regression] **`Operand::Copy` on a `&mut`
  reference is a deliberate, verifier-accepted MIR shape that the `Copy` classification does not
  describe.**

  - The producer/consumer agreement check, run unrestricted, flagged **exactly 11 sites** across
    the corpus and the full differential suite — and **every one was `Ref { mutable: true, .. }`,
    no other type at all**. That uniformity is the result: the two classifiers agree everywhere the
    question is the same one.

  - It is not a defect in either. A `&mut` handed to a callee or a bounds check is **reborrowed**,
    not moved, or MIR would lose the reference; `is_copy` answers a different question about the
    same type ("does binding it elsewhere consume it?" — yes). Both answers are correct for their
    own question. `Operand::Copy` therefore means "read without consuming", which for `&mut` is a
    reborrow rather than a duplication.

  - **Why it matters to C5 and where it is contained.** The native backend does not slot-back
    references, so `Operand::Copy` on a `&mut` local emits a plain Rust read — which for `&mut`
    is a *move* in Rust, not a copy. A second read of the same reference local would therefore not
    compile. Contained today by the C5.3d-1a lane's single-use/same-block validator (refusal before
    rustc) and by rustc itself as the backstop. Flagged for the C5.3 adversarial review rather than
    changed: renaming or splitting `Operand::Copy` would be a MIR contract change.

  - The check is scoped to exclude `&mut` and is retained as a live guard for every other type.

- CD-066 [2026-07-22, C5.3d-1c DONE; C5.3d-1 CLOSED] **The observable destruction closure is
  evidence for seven properties across three engines — and it exposed a missing backend operation
  that was wider than the partial-move seam it was aimed at.**

  - **The observation channel is a real constraint, stated rather than worked around.** Native
    `println` does not exist (`Callee::Runtime` is wholly unsupported until WP-C5.4c) and
    `NATIVE_STDOUT_SUPPORTED` is still `false`; STARK has no globals and no reference fields, so a
    destructor cannot record its own firing for a later assertion either. The cases therefore use a
    **trapping destructor as a position probe**: traps abort, so the first destructor to run is the
    one that traps and the trap's exact line names it. Each case is built so one ordering question
    decides the reported line, and destructors that must not both fire get different types so they
    occupy different lines. This reads out one bit of order per run. Full native destruction
    *tracing* is blocked on `RuntimeFn` and belongs to WP-C5.4c.

  - **Seven properties, eight three-engine cases**: own destructor before fields; fields in reverse
    declaration order; active-variant payload only (a MIRRORED pair, because one case alone would
    be satisfied by an engine that always destroyed variant 0); a moved value destroyed by its new
    owner (the caller's assertion is deliberately false, so caller-scope destruction would report a
    different line — a probe, not a tautology); no destructor after a trap; exactly once; and the
    partial move with a droppable sibling. Every `expected_line` is derived from the language rule
    and carries an `expected_span_reason` note, per CD-062.

  - **Exactly-once is the one property a trap probe cannot show** (a trap aborts on the first
    destruction, so a second is never reached). Stated as a completing case instead, and what makes
    completion meaningful is engine-specific: the MIR interpreter poisons a local's slot on `Drop`
    and the native `ValueSlot` asserts `Whole` in `drop_with`, so a second destruction is a
    violation in both rather than a silent repeat.

  - **THE FINDING — per-unit (sub-place) destruction was missing, and not only for partial moves.**
    Two fixtures failed to build. MIR's drop elaboration decomposes an aggregate with several drop
    units into **one flag-guarded `Drop` per unit on a projected place** — `drop _1.1` then
    `drop _1.0`, each behind its own `Bool [dropflag]` — so a plain two-droppable-field struct with
    no destructor of its own arrives projected. The backend refused all projected `Drop`s, so that
    struct could not compile natively at all. **The refusal was correct, not merely conservative**:
    collapsing per-unit drops into a whole-local one would destroy a unit MIR's flags say is
    already gone (§7.6).

  - **Closed with a real operation, not a relaxation.** `HelperOp::Drop` generates one wrapper per
    (base type, projection) around `ValueSlot::drop_field_with` — the primitive already existed
    from C5.3d-0 — with the unit's `DropPlan` **baked into the wrapper**, since a wrapper is
    already per-(type, projection) and that fixes the field type and hence the plan. Call sites
    stay plain safe calls, so an emitted body still contains no `unsafe` and no destruction logic.
    A projected `Drop` of an **enum payload** is refused with a stated reason: an enum's payload is
    destroyed by the whole-enum plan's variant match, and the `&mut T` projection form needs a
    complete value the drop is in the middle of dismantling.

  - **What the emitter does NOT decide.** MIR sequences the units and MIR's flags skip the
    moved-out one; the emitter follows. Per-unit liveness stays MIR's, per §7.6.

  - **C5.3e is now the ONLY remaining C5.3 exit condition** — every other §14 item is discharged.

- DEV-099 [2026-07-23, found while scoping C5.3e, PRE-EXISTING] **A layout query on an ARRAY type
  fails to lower.** `size_of::<[Int32; 4]>()` reaches lowering and dies with "field type form
  (C4.5)" — `hir_field_ty` does not handle an array type in a turbofish position. Every other
  queryable shape works: primitives, tuples, structs, user enums, `String`, and a monomorphised
  generic parameter. Not introduced by C5.3e; recorded because arrays are inside the C5.3a subset,
  so the gap is visible from the layout-query exit condition. Bounded front-end work, not a
  semantic question.

- CD-067 [2026-07-23, owner decision, RECOMMENDATION OVERRULED] **The generated crate must NOT
  cross-check the STARK layout contract against Rust's physical layout, and generated internal
  nominals must NOT be `#[repr(C)]` for that purpose.**

  - **The authority analysis stands**: the named versioned `TargetLayout` contract is the
    observable result; physical representation stays unobservable and backend-private. Native
    lowering emits `4u64`, not `core::mem::size_of::<i32>() as u64`.

  - **Why the recommendation was wrong.** The proposed assertion enforces a stronger, different
    rule — *the target contract must equal the generated-Rust backend's physical representation* —
    which Core v1 does not require. It would (a) make the contract **backend-dependent**, so a
    later Cranelift backend using a different representation while implementing the same contract
    would be obstructed; (b) **conflate three separate contracts** — the observable language
    layout contract, the internal backend representation, and the separately versioned provider
    ABI — when LAYOUT-ABI-001 explicitly says equal `size_of`/`align_of` does not establish
    interoperation compatibility, so blanket `#[repr(C)]` could later be misread as an internal ABI
    commitment; (c) **sacrifice representation freedom for no Core-visible gain**, since field
    reordering and niche optimisation are unobservable and forcing a full `Option` discriminant
    pays a physical cost for no normative guarantee; and (d) **not actually validate the abstract
    contract** — it checks agreement with one Rust representation, not that the algorithm is
    internally coherent, that arrays follow the declared stride, that alignment combinators hold,
    that enum formulas cover every variant, that all three engines use the same named target, or
    that the manifest matches its recorded contract version.

  - **The concern about unfalsifiability was valid; the remedy was not.** Falsifiability comes
    from making the **declared algorithm and manifest independently testable**, not from
    redefining the contract as "whatever Rust physically chose".

  - **Required instead**: one versioned `TargetLayout`; one deterministic combinator
    implementation; an explicit target-contract identifier (`target_contract`,
    `layout_contract_version`, `compiler_layout_revision`); exact FROZEN values for the C5 layout
    matrix (primitives, tuples, arrays, structs, user enums, `Option`, `Result`, function values,
    and every other admitted C5 value); independent HIR-type and MIR-type walks; native constants
    from the same manifest; mutation tests that alter a primitive, an aggregate rule, or a manifest
    entry and break agreement; manifest identity in the build key and build report; and rejection
    when the requested target and manifest identity disagree.

  - **A host-layout comparison may later exist as a non-normative diagnostic** (`--audit-host-layout`)
    that REPORTS rather than rejects, unless a particular representation explicitly declares
    `physical_layout_matches_target_contract = true` — useful for provider-ABI types, serialization
    buffers, memory-mapped structures, or a backend optimisation deliberately relying on physical
    equivalence. Never for ordinary internal STARK values.

  - **DEV-099 is promoted to a MANDATORY C5.3e prerequisite**, not an adjacent limitation: arrays
    are in the approved C5 aggregate subset and the exit matrix explicitly requires fixed-array
    layout coverage, so a deterministic front-end failure on a required layout shape would leave
    C5.3e incomplete.

  - **Plan correction required** in `WP-C5-ENTRY.md`: replace the language saying generated Rust
    answers layout queries from its actual generated representation with — "`size_of<T>` and
    `align_of<T>` return values from the selected versioned STARK `TargetLayout` contract. HIR, MIR
    and native execution consume that same contract. A backend's internal physical representation
    is not observable and need not equal those values unless a separate representation contract
    explicitly requires equivalence."

## C5.3e — target-layout manifest — CLOSED (CD-067)

Closed with Gate C5 on 2026-07-23 (CD-077). The 2,808-line working detail — still headed
**IN PROGRESS** four gates after it closed — is archived verbatim at
`STARKLANG/docs/compiler/state-archive/C5-C7-closed-detail.md`. The versioned `stark-64-v1`
layout contract and exact layout-query values are frozen in `starkc/docs/compiler/C5-exit-report.md`.

## Conformance summary
- Lexical: WP-C1.1 requalification complete (2026-07-17). Strengthened: all 15 reserved words
  now tested by name (was 3), reserved-word rejection confirmed in non-expression positions,
  nested-comment depth tested to 4 levels (was 2) with a matching unterminated-at-depth negative
  case. Found and closed one real bug in the process (DEV-014). Found and recorded, but did not
  fix, a real gap outside this rule's own scope (DEV-015, literal overflow never checked).
- Syntax: WP-C1.1 requalification complete. Strengthened: `>>`/`>>=`/`>=` generic-closing-token
  splitting (added the previously-untested `GtEq`→`Eq` split arm and a bare-shift-expression
  contrast case), multi-file `mod` layout (added missing-file, duplicate-declaration, and
  circular-reference cases — the missing-file case is DEV-014's regression test), depth-limit
  boundary behavior (added exact-latch and false-positive-floor assertions, `starkc/tests/
  robustness.rs`), diagnostic determinism across repeated parses of identical input, and AST
  span-containment (new `starkc/tests/span_integrity.rs`, DEV-018 — first-ever programmatic
  span-invariant check in the codebase, covering `Expr`/`Block` nodes across the full parseable
  fixture corpus).
- Types: WP-C1.3 requalification complete (2026-07-17). The equality/trait-dispatch closure the
  roadmap flags is now **fully resolved** (DEV-008 closed — real `Eq::eq` dispatch implemented,
  plus a companion fix so `Ty::Core` container types satisfy Eq/Ord bounds at all). STD-004
  (standard traits) exhaustiveness audit closed (DEV-013) with 2 real bugs found and fixed:
  `.clone()` was entirely non-functional on every compiler-builtin type (String/Vec/Option/
  Result/HashMap/HashSet/Range/IOError), and trait default method bodies were never used as a
  fallback when unoverridden — both now fixed with regression tests. `Error`/`Hash`/`Display`/
  `Clone` as generic *bounds* were already correctly recognized throughout (the DEV-013 seed's
  worry about `Error` support was checking the wrong function). Two new deviations found and
  recorded but deliberately not fixed to keep scope bounded: DEV-023 (`Display`/`Hash` share
  Clone's old "missing as a callable method on builtins" bug, not yet fixed) and DEV-024 (`From`
  trait `Type::from(value)` associated-function calls fail to resolve, root cause not yet
  isolated). Local inference boundaries, generic substitution, associated types, orphan/overlap,
  and conflicting-impl diagnostics were spot-checked against existing tests
  (`gate5_semantic_gaps.rs`, `typecheck.rs`'s own test module) and found adequately covered —
  not subjected to the same exhaustive research-agent audit as WP-C1.1/C1.2 given the WP's time
  budget was consumed by the two substantial bug-fix cycles above; a future pass could still
  deepen this if warranted.
- Semantics: old Gate 2/3 coverage; pending WP-C1.3-C1.5.
- Memory: old Gate 2 M2.4 (ownership/borrows); pending WP-C1.4 full positive/negative corpus
  construction — not yet confirmed to exist at that depth.
- Modules/packages compiler surface: old Gate 2/Phase 1-3 (multi-file modules, `starkpkg.json`
  manifests, dependency resolution/locking per `git log` Phase 1-3 commits). `PKG-004`/`PKG-005`/
  `PKG-006` were incorrectly `missing` in the coverage database — corrected to `partial` under
  WP-C0.3 with real source/test citations; see DEV-002. WP-C1.2 requalification complete
  (2026-07-17): name resolution, module/visibility rules, imports, and re-exports strengthened
  across the full 10-item roadmap matrix; 3 real bugs found and fixed (DEV-004, DEV-006 resolve
  half, DEV-007); 1 new significant finding recorded but not fixed (DEV-019, E-code collisions);
  cross-package coherence checking (SEM-007) and cross-package diagnostic file attribution both
  went from "unverified" to "confirmed working" with real two-package-workspace tests (DEV-021).
  STARK's visibility model confirmed stricter than Rust's (private = exact defining module only,
  no descendant inheritance) — see the dedicated "Design fact pinned down by WP-C1.2" note below.
- Tensor extension: old Gate 4 (`gate4-exit.md`, closed 2026-07-15, "no known deviations")
  covers syntax/resolution/static checking + bounded ONNX metadata decode. Old Gate 7
  (`gate7-decision.md`) added symbolic/computed dimensions and value-range semantics with a
  13/13 defect-detection result. Both predate the new C-numbering; WP-C1.x does not re-audit
  extension code (Core-only scope), but WP-C9.1/C9.2 will need this as input later.

## Known deviations — open index
Canonical ledger (full structured entries): the file now carries **108 distinct numbered
deviations** as of 2026-08-02, counted as unique `## DEV-NNN` headings (DEV-121 has two — an
original and an UPDATE — and is counted once). CD-334 added six. NOTE: this line previously read
"97 numbered deviations as of 2026-08-01", which did not match the file then either (102 by the
same count); the discrepancy predates CD-334 and is recorded rather than silently rewritten,
because whichever convention produced 97 may be the intended one. Path:
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`. The per-deviation narrative that used to live in
this file (seed list + WP-C1.1/C1.2/C1.3 addition sections) is archived verbatim in
`STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (CD-020); the ledger remains the
single source of truth.

> **STALE — do not read this list as current (noted 2026-08-11).** It is dated 2026-08-02 and has
> not tracked the ledger since. It lists DEV-005, DEV-011, DEV-012 and DEV-017 as open; the ledger
> has DEV-005 and DEV-012 CLOSED, DEV-011 ACCEPTED-INDEFINITELY, and DEV-017 PARTIALLY CLOSED. It
> also predates everything from DEV-140 onward.
>
> **The position block at the top of this file is the current open set**, and the ledger's own
> "last heading wins" rule is the authority behind it: derive the set by taking each `DEV-NNN`'s
> LAST heading in `KNOWN-DEVIATIONS.md` and keeping the ones that are not a closure.
>
> Kept rather than deleted because the narrative below is a record of what was believed on
> 2026-08-02, and rewriting it would erase that. It is annotated instead — the same treatment the
> "97 numbered deviations" discrepancy above got, and for the same reason.

Open as of 2026-08-02. Entries DEV-005…DEV-017 are long-standing and unscheduled, and no open
deviation belongs to the C4 track. **DEV-134…DEV-139 were opened 2026-08-02 by CD-334 and are new,
not long-standing** — three of them are soundness gaps and none has an owning gate yet.
- DEV-005 — `starkc` vs `stark` check/run warning-gating drift. Open, unowned since Gate C1.
- DEV-011 — doc comments are lexer trivia, not AST/HIR metadata. Unscheduled; needs a scoped
  proposal.
- DEV-012 — VS Code extension UI interactively verified **in part** (2026-07-31). Hover,
  go-to-definition and find-references were exercised by the owner in a real VS Code session and
  behaved. **Rename, diagnostics-on-save/on-type, formatting, completion, signature help, document
  symbols and semantic tokens were NOT exercised** and remain protocol-tested only. Open for that
  remainder; owner: post-C8 editor validation.
- DEV-017 — 39 of 59 legacy coverage rules still lack function-level positive/negative evidence
  classification (tooling exists; classification unscheduled).
- DEV-134 — CLOSED CD-335 (WP-DEV-134-139 Part A). `?` now requires exact error-type and
  constructor compatibility; the ruling was REJECT, not convert. Whether Core v1 should gain
  `From` conversion at `?` is a separate, still-open language-design question with no owner.
- DEV-135 — CLOSED CD-338 (WP-DEV-134-139 Part B). The move model was already field-precise; the
  defect was field IDENTITY taken from a span. No DEV-135b was filed: the precision that follow-on
  would have built already existed.
- DEV-136 — CLOSED CD-337 (WP-DEV-134-139 Part D). Move state now merges only from predecessors
  that reach the join; `loop` without a reachable `break` is deliberately still treated as
  reaching, because proving otherwise needs reachability analysis the checker lacks.
- DEV-137 — CLOSED CD-336 (WP-DEV-134-139 Part C). Condition-only borrows now end at the branch
  boundary, for `if` as well as `while`; `match` scrutinees and `for` iterators deliberately keep
  theirs.
- DEV-138 — CLOSED CD-340 as a CONFIRMED DEV-121 instance (WP-DEV-134-139 Part F). DEV-121's
  class stays OPEN, and its blind spot is now named: INV-VALUE-REP-001 checks `let` bindings, and
  a for-loop binding is not a `let`, so no loop item is covered.
- DEV-139 — CLOSED CD-339 (WP-DEV-134-139 Part E). Both the operator and trait-bound lookups now
  read the combined impl+method environment. DEV-083 is a different mechanism and remains OPEN.
- Informational, not owed a fix: DEV-SEED-008 (two hand-rolled JSON parsers), DEV-SEED-014
  (no attribute syntax — deliberate scope fact).

Closed 2026-07-31: **DEV-010** (C8 candidate closeout) — LSP hover, definition, and references
are no longer protocol stubs. They are backed by `ProjectAnalysis` semantic queries and covered by
`hover_uses_compiler_symbol_signature` and
`definition_and_references_use_resolved_symbol_identity`.
Closed 2026-07-20: DEV-070 (WP-C4.6 A2, both engines); DEV-074 (numbered by WP-C4.7-1 and closed
at creation — the A4-2e oracle slice-message alignment, a governance gap, not a code defect);
**DEV-069** (WP-C4.7-4 — per-item file resolution in typecheck/borrowck/oracle; this also
DISCHARGES CD-033's C5 multi-file prerequisite); **DEV-072** and **DEV-073** (WP-C4.7-5 —
move-out-of-borrow via match bindings, now rejected E0101; generic impls matched through
`match_impl_type` for operator and iterable bounds); **DEV-067** and **DEV-071** (WP-C4.7-7 —
bounded-parameter bounds behind references and at intra-generic call sites; `Ordering`
exhaustiveness); **DEV-077** (WP-C4.7-6.1 — oracle `Box::into_inner` double-drop); **DEV-078**
(WP-C4.7-6.3 — integer literals adopt their expected type); **DEV-075** (the DEV-075 increment —
`Char` ordered by Unicode scalar value, `Bool` not `Ord`, plus normative `PRIM-TRAIT-001`);
**DEV-076** (WP-C4.7-8.1a — the oracle's `unwrap_or` double-drop).
Closed 2026-07-19: DEV-060 (CD-024); DEV-061/062/063 — the function-value cluster — in the
CD-027 pre-C4.1 correction pass; DEV-064 (undetermined-generic rejection, WP-C4.5c, E0004);
DEV-065/066 (C4.5b oracle fixes). See `KNOWN-DEVIATIONS.md`.

## Design fact pinned down by WP-C1.2 (not a deviation, recorded so it isn't re-discovered)
STARK's visibility model is **stricter than Rust's**: per `07-Modules-and-Packages.md` §Visibility
("items are private to their defining module by default"), a private item is visible **only**
within its exact defining module — there is no Rust-style "visible to the defining module and
all its descendants." Confirmed by the pre-existing `module_paths_imports_and_visibility_are_
enforced` test (root cannot access a private item of its own direct child module) and by three
new WP-C1.2 tests (`super_and_crate_navigate_correctly_from_a_nested_module`,
`private_item_is_not_visible_from_a_descendant_module`,
`pub_use_single_level_reexport_is_visible_from_outside`) — the first drafts of the latter two
tests were written assuming Rust-style descendant-inherits-privacy semantics and failed against
the real implementation, which is what surfaced this. Any future WP writing STARK test fixtures
involving nested modules and private items should assume this stricter model.


## Architecture decisions
- AD-001 [pre-existing, old Gate 5] Native artifact-deployment backend is **ONNX Runtime via the
  `ort` crate**, pinned `=2.0.0-rc.12`, statically linked, CPU execution provider only
  (`starkc/docs/gate5-backend-decision.md:11`). IREE/Cranelift/TVM explicitly considered and
  deferred at that time. This is a decision about the *tensor artifact deployment* backend, not
  a decision about general Core native compilation — the two must not be conflated (see CD-002).
- AD-002 [pre-existing] ONNX decoding uses a hand-written protobuf reader with zero new runtime
  dependencies beyond `sha2` (for checksum verification); `ort`, `tract-onnx`, and `onnx-pb`
  crates were evaluated and rejected (`starkc/docs/gate4-design.md:158-169`). `starkc`'s own
  `Cargo.toml` has exactly one dependency, `sha2`, and forbids `unsafe_code` at the lint level.
- AD-003 [pre-existing] Both CLI binaries (`starkc`, `stark`) hand-roll argument parsing against
  a `USAGE` const rather than using `clap` or another CLI-parsing crate (confirmed: no `clap`
  entry anywhere in `Cargo.toml`/`Cargo.lock`).

## Native backend selection
- Status: **SELECTED** (WP-C3.4, owner CE5 decision, 2026-07-19).
- Selected strategy: **generated Rust/C** — generated Rust as the initial production backend
  behind verified MIR, with a **backend-neutral MIR contract that keeps `SELECT-DIRECT`
  (Cranelift) open as a C7-gated migration** (charter §1.6 rule 9, no lock-in). Decision +
  full three-way analysis: `starkc/docs/compiler/spikes/WP-C3.4-backend-selection-analysis.md`;
  recorded as CD-026.
- Architecture commitments (roadmap WP-C3.4): emitter consumes **verified MIR** (not typed HIR);
  small STARK runtime library (print/panic/trap glue); Rust owns MVP value layout + calling
  convention; Native Provider ABI (C5.1) as `extern "C"` provider calls from generated Rust;
  Tier-1 targets first (linux-x64, macos-arm64) via rustc; debug/trap file:line via a STARK-span
  → generated-Rust-line → rustc-debug-info table; unsupported-MVP closure (floats/`?`/tuple
  patterns/traits/Drop/refs/Vec/HashMap/fn-values) tracked into C4.5/C5/C6.
- **Accepted trade (recorded):** `stark build` requires a full `rustc` toolchain as a permanent
  build dependency, and builds are slower than the direct backend. Acceptable for STARK-as-
  research-language; **re-evaluate the backend choice at C7** if the self-contained-compiler /
  systems-platform goal becomes primary (same evidence-gated pattern as the LLVM decision).
- Workload: 23-item frozen set (`NATIVE-CORE-ARCHITECTURE.md` §5), items 1-10 mapped to the
  frozen `exec_snapshots` corpus v1.0.0 (semantic oracle), items 11-23 specified reference
  programs. Two properties (fn-value Eq/Ord/Hash participation, monomorphised-generic fn-value
  identity) must be settled from the frozen spec or by CE1/CE2 before selection (CD-022).
- Spike evidence so far:
  - **WP-C3.2 generated-Rust (done):** 4/17 frozen corpus cases lower and match the interpreter
    exactly (arithmetic/precedence, loops/for/break/continue, multi-width ints, Int8-overflow
    trap→abort parity); 0 semantic mismatches on supported cases; 13/17 cleanly reported
    unsupported; mean rustc 87 ms/case. Liabilities unresolved (not falsified): rustc
    build-dependency weight, compile-time scaling, exe size, debug-info trap mapping, and the
    unsupported breadth (aggregates/generics/traits/refs/Drop/fn-values). Report:
    `starkc/docs/compiler/spikes/WP-C3.2-generated-rust.md`; artifact `tests/spike_genrust.rs`
    (isolated, disposable).
  - **WP-C3.3 direct Cranelift (done):** 3/17 frozen corpus cases lower and match the interpreter
    exactly (arithmetic, loops/for/break/continue, Int8-overflow trap→abort parity); 0 semantic
    mismatches; 14/17 unsupported (same families as C3.2 plus unsigned ints — spike is
    signed-only, hence 3 vs C3.2's 4). Produces a real standalone native executable (Cranelift
    object + `cc` link). Codegen ~2 ms/case (phase-only), link ~47 ms/case; **defensible
    end-to-end ~49 ms vs rustc ~87 ms ≈ 1.8× on this tiny workload — explicitly NOT a general
    performance multiple** (charter caution; see the report's timing caveat — the raw 2-vs-87
    codegen ratio is not like-for-like). No rustc build dependency. Finding: Cranelift 0.133 needs
    rustc ≥1.94 (>1.93 here) → pinned 0.110, an MSRV-churn maintenance cost. Higher glue than
    generated-Rust (we own CFG/SSA/overflow/Drop/layout); weaker out-of-box debug-info; but the
    bigger beneficiary of the mandatory MIR (MIR ≈ Cranelift's own block/terminator model).
    Report: `starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md`; artifact
    `tests/spike_cranelift.rs` + dev-only Cranelift deps (isolated, disposable).
- **Breadth run (2026-07-19):** generated-Rust extended to structs/impl-methods/generics/
  Option/Result/match/String → **8/17** frozen corpus cases (all matching), via ~250 lines of
  mechanical text emission (rustc absorbs monomorphization/layout/ABI/Drop). Cranelift breadth
  **measured at the struct boundary, not fully implemented** — struct-by-value needs stack-slot
  layout + field offsets + sret ABI; enums need tagged-union layout; generics need a
  monomorphization engine; String/Vec need a runtime — each a subsystem the direct backend owns.
  Cranelift stays 3/17. **Key WP-C3.4 caveat: most of that direct-backend breadth cost is
  mandatory MIR work anyway (Gate C4), so the HIR-level comparison overstates the direct
  backend's long-run cost.** Full head-to-head:
  `starkc/docs/compiler/spikes/WP-C3-breadth-comparison.md`. (Implementing Cranelift
  struct-by-value is a bounded ~150-200-line follow-up if an exact struct head-to-head number is
  wanted.)
- Both spikes done; the tradeoff is symmetric and matches the §4 hypothesis: generated-Rust =
  low glue + free cross-platform/debug-info + broad correctness cheaply + heavy rustc dep; direct
  = fast builds + no rustc + ABI control + biggest MIR beneficiary, but owns monomorphization/
  layout/drop/runtime. Neither falsified nor cleared; WP-C3.4 selects (CE5, owner).
- Evidence: see CD-002 for the closest existing evidence (old Gate 6/7 tensor/ONNX-deployment
  track) — informative precedent for methodology, not a substitute (CD-004).

## Diagnostic codes allocated or changed
- **MIR-0001..MIR-0013** [WP-C4.3, 2026-07-19] First allocation of the `MIR-xxxx`
  compiler-internal namespace (charter §5.1): 0001 target OOB, 0002 local OOB, 0003 projection
  type, 0004 assignment/operand type, 0005 call/checked signature, 0006 bare unsized, 0007
  possibly-moved use, 0008 discriminant/variant misuse, 0009 drop/drop-flag, 0010 index-proof
  discipline, 0011 FnPtr arithmetic/comparison, 0012 reserved (runtime-set violation —
  structurally impossible while RuntimeFn is a closed enum; reserved for serialized MIR), 0013
  invalid FileId in SourceInfo. These are internal invariant failures (lowering bugs), never
  user-source diagnostics. Full map: `src/mir/verify.rs` header + WP-C4.3.md.
- **MIR-0036** [WP-COPY-CANON Phase 3, CD-311] INV-MOVE-001: a `Move` operand from a place whose
  type is `Copy`. A `Copy` type's contract is that reading leaves the source intact; `Move` empties
  it and transfers drop responsibility. Emitting both about one value lets every consumer believe
  whichever it prefers. Unconditional, with no exemption mechanism — see CD-311 for why an
  "unobservable move" escape hatch was refused. Found four latent defects on its first runs
  (DEV-124, DEV-125, DEV-127).
  **This section is stale between MIR-0013 and MIR-0036**: MIR-0014..MIR-0027 and MIR-0034/0035
  were allocated by later WPs (A1/A5/A11/A12) and recorded only in `src/mir/verify.rs`'s header
  map, which is the working registry. Reconciling them here is unscheduled and is noted rather
  than silently papered over by this entry.
- **E0008** [WP-C1.5] Integer literal out of range for its type (suffixed literal exceeds its
  suffix's representable range, or an unsuffixed literal exceeds `Int64`). See DEV-015.
- **E0009** [WP-C1.5] Array repeat count (`[value; count]`) is not a compile-time constant
  expression.
  Both registered in `04-Semantic-Analysis.md`'s normative Error Categories table
  (`STARK-Core-v1.md` regenerated in the same change). No codes allocated or changed by any other
  WP under this governance framework yet. Existing (pre-governance-framework) normative
  `E####`/`W####` codes are inventoried as part of WP-C0.1 (`starkc/src/diag.rs`), not duplicated
  here.

## Evidence inventory
- `starkc/docs/gate1-exit.md` through `gate7-decision.md` — old-numbering gate evidence, see CD-001/CD-002.
- `STARKLANG/tests/spec-fixtures/manifest.toml` — 113-entry spec-fixture corpus (directly
  re-counted 2026-07-19; the "121-fixture" figure this line carried from the C0 audit had
  drifted), verdict census in
  Repository baseline above.
- `cargo test --workspace --all-targets --all-features` output (2026-07-17 audit run) — 383
  passed / 0 failed / 2 ignored, full per-suite breakdown to be carried into
  `starkc/docs/dev/compiler-map.md` (WP-C0.1).
- `STARKLANG/conformance/core-v1-coverage.toml` — 59 rules, 53 implemented / 6 partial / 0
  missing, **integrity-audited under WP-C0.3** (duplicate-ID check, spec-chapter-validity check,
  4 stale `missing` entries corrected with cited evidence). `python3 starkc/scripts/
  check-conformance.py` output (2026-07-17, post-correction): 0 errors, 0 warnings.


## File inventory for current gate
C3-ENTRY (active transition): `STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md` (transition
work package, created 2026-07-19 under CD-020), `.github/workflows/ci.yml` (baseline widened
under CD-020), `STARKLANG/docs/compiler/state-archive/C0-C2-closed-detail.md` (new archive).
Closed-gate file inventories (C0/C1): archived verbatim in the state-archive file; per-gate
evidence in the C0/C1/C2 exit reports.

## Follow-ups
- [ ] WP-C0.2 carry-forward (governance-process question, unresolved): gate7-decision.md's "No
      LSP work or language expansion is authorized" text was apparently overridden for WP8.1-8.5,
      but no explicit owner override record exists. Owner should either backfill a decision
      record or confirm WP8.x was tooling, not "language expansion" in Gate 7's sense.
- [ ] DEV-005: pick one warning-gating policy for `starkc check`/`run` vs `stark` — still
      unowned; candidate for C3-ENTRY or a small pre-C3 correction.
- [x] WP-C8.2/C8.3: implement real LSP hover/definition/references (DEV-010). Closed by C8
      candidate closeout; semantic query tests pass.
- [ ] Post-C8: interactive VS Code Extension Development Host validation (DEV-012). C8 is
      candidate-complete until this record exists.
- [ ] WP-C1.1 follow-up (not blocking): underscore-placement rules for binary/octal literals
      untested; no max-value-per-suffix positive test for the 8 int / 2 float suffixes.
- [ ] DEV-017 remainder: classify the 39 unclassified legacy coverage rules (unscheduled).
- [x] **DEV-095 — WP-C5.3 opening condition. DISCHARGED 2026-07-21, CD-055.** The build key was
      hashing `program.dump()`, which omits the nominal type context and the Drop map, so a
      changed struct field or `Drop` impl could leave the key unchanged and silently reuse a stale
      generated crate. The key now covers all eight version axes, the entry symbol, the source
      table (names + content hashes), all four `TypeContext` fields, and the bodies — with seven
      cache-invalidation tests, mutation-verified against the old behaviour. **WP-C5.3's blocking
      entry condition is satisfied; aggregate and Drop-bearing native generation may begin.**
- [x] **Native Provider ABI v0.1 — CE4 Amendment 1. CLOSED 2026-07-21, CD-054**: approved at
      revision 3 and applied in full (ABI document, both `provider_abi.rs` files, fixtures,
      violation tests). Revision 1 was not approved; revision 2's design was approved with five
      required changes; revision 3 incorporates them. The close-function question was ruled —
      exactly one parameter, the consumed handle, nothing else, because MIR's `Drop(place)`
      supplies no argument list. ABI version stays `0.1`. Record:
      `STARKLANG/docs/compiler/native-provider-abi-v0.1-CE4-amendment-1.md`.
- [x] DEV-060: dispose before C3 workload freeze (C3-ENTRY blocker). **Closed 2026-07-19,
      CD-024 — fixed in `borrowck.rs::method_receiver`.**
Completed follow-ups through Gate C2 are archived verbatim in the state-archive file.

## Gate exit summaries
- C0: **PASS** (2026-07-17). Bootstrap, current-state audit, and authority repair complete. Full
  report: `starkc/docs/compiler/C0-exit-report.md`. Four stale documents corrected (`CLAUDE.md`,
  root `README.md`, `starkc/README.md`, `STARKLANG/docs/PLAN.md`); conformance database
  integrity-audited with 4 staleness errors fixed (DEV-002, closed); 10 confirmed deviations
  recorded with full structured detail in `starkc/docs/conformance/KNOWN-DEVIATIONS.md`; module-
  by-module compiler map produced (`starkc/docs/dev/compiler-map.md`). Explicit non-claim: no
  conformance percentage from this gate is trusted for Core v1/tensor v0.1 conformance purposes
  — see exit report's "No conformance percentage is trusted" section. Next: Gate C1.
- C1: **CORE-FRONTEND-CONFORMING-WITH-LISTED-DEVIATIONS** (2026-07-17/18). Full report:
  `starkc/docs/compiler/C1-exit-report.md`. Six requalification WPs closed (lexical/syntax, name
  resolution/modules/visibility, types/generics/traits, ownership/borrowing/drop checking,
  control flow/patterns/constants/numerics, conformance evidence generator); 12 of 23 deviations
  closed, 2 partially closed, 9 open and non-soundness-relevant. This entry backfilled during
  WP-C2.13's consistency sweep — not recorded here at the time of C1's own close. Next: Gate C2.
- C2: **CORE-V1-SEMANTIC-FOUNDATION-FROZEN-WITH-LISTED-DEVIATIONS** (2026-07-18). Full report:
  `starkc/docs/compiler/C2-exit-report.md`. Reference-execution contract, abstract machine, and
  future-boundaries specifications written from scratch; all 24 high-cost open questions
  approved; 166-row completeness inventory has zero absent/contradictory/unclassified rows (6
  pending-owner-approval governance-only); 33 deviations closed this gate (the largest body of
  runtime-semantics fixes in the compiler track's history, including DEV-053/054 — a bare `None`
  pattern silently matching any value with wrong runtime output, the most severe finding to
  date), 8 remained open and non-soundness-relevant at gate close (see the open index above
  for the current set). WP-C2.12's differential corpus is
  representative, not exhaustive — explicitly disclosed, not disqualifying (cross-backend replay
  is blocked behind Gate C3 by the roadmap's own dependency order). Next: Gate C3, WP-C3.1.

---

## Session records

Records for WP-C0.0 through Gate C8 are archived verbatim at
`STARKLANG/docs/compiler/state-archive/session-records-C0-C8.md` (C0–C2 were archived earlier
under CD-020). **Sprint 4's records stay in this file** — a compression target is not a reason
to archive a record still being worked against.

---

### 2026-08-11 — three deviations registered from package work (`stark-cookie`)

Not a compiler packet. `stark-cookie` v0.1 was implemented on `develop` at `2cd4a08` under a
package brief that forbids compiler changes, and hit three compiler-track findings. All three are
recorded in `starkc/docs/conformance/KNOWN-DEVIATIONS.md` with minimal reproducers. **No compiler
source was modified.** Population A 8 -> 11.

- **DEV-222 — WRONG-CODE, and the one that matters.** A pattern naming a variant that does not
  exist type-checks clean and silently never matches, falling to the wildcard. `stark check`
  reports OK. Without a wildcard the program is rejected by `E0303 non-exhaustive`, which points
  at the match rather than the typo — so the natural fix (add a wildcard) converts a caught bug
  into a silent one. `resolve.rs`'s three pattern branches already guard for this correctly
  (`E0200`/`E0202` on `res == Res::Err`); the fault is that `resolve_path` does not return
  `Res::Err` for `Type::NonexistentName`. **Same class as DEV-053/054**, which C2's exit report
  calls "the most severe finding to date": DEV-053 closed it for a bare identifier resolving to a
  builtin, and the qualified-path case was never closed.
- **DEV-223 — REVISED the same day; it is NOT fail-safe.** A variant whose name matches an
  in-scope type makes an exhaustive match report `E0303 non-exhaustive` — and, in expression
  position, makes an ordinary constructor `Attr::Policy(Policy::A)` pass `stark check` and then
  **fail at runtime** with `item is not callable`. Root cause read out of
  `resolve_path_relative`: the subsequent-segment loop consults `current_mod`'s module items
  BEFORE the qualifying item's own variants, so a module-level name shadows the variant. **Not
  the same defect as DEV-222** — two distinct faults in the same loop; see the REVISED heading in
  the ledger.
- **DEV-224 — native gap.** An enum carrying a non-`Copy` payload cannot be matched through a
  shared reference; even `_` arms are refused, because the rejection is about the scrutinee. This
  blocks the ordinary tagged-value shape (`enum { A(String), B(Int64) }` in a `Vec`, read by
  reference) on the native path, which is the shipping path for capability-backed programs.
  `stark-cookie` uses a tagged struct instead, which costs it the unrepresentability a sum type
  would have given.

None of the three was worked around by changing the compiler; DEV-222 has no package-level
workaround because it is a missing rejection rather than a shape to avoid.

---

### 2026-08-11 (later) — DEV-222/223 repaired, and an audit outward from them found three more

Not a chartered packet: a repair taken directly from the `stark-cookie` findings, plus an external
audit of the resolver that widened the scope. **Uncommitted at the time of writing.** Population A
stays at 11 — two resolved, two registered.

**Repaired.** DEV-222 in pattern lowering (`resolution_is_pattern_legal`, exhaustive over `Res`,
reached through one `reject_non_pattern_resolution` helper from all three pattern branches) and
DEV-223 in `resolve_path_relative` (`qualified_associated_name` ahead of the module lookup, with
`current_is_module` so `crate`/`super`'s placeholder `Res::Item` is not misread as a type). The
expression resolver is untouched: `Res::AssociatedFn` keeps its meaning, which sixty-odd
associated-function call sites across `packages/` depend on.

**Found by the audit, and resolved in the same change:** DEV-225 (associated-name precedence for
structs, traits and models, not just enum variants — a NAME-RESOLVE-001 conformance deviation),
DEV-226 (every `Res::Builtin` accepted as a pattern, so `Vec::new(x)` matched nothing silently),
DEV-227 (every `Res::Item` accepted as a by-value pattern, so a bare function name matched nothing;
repaired by BINDING per SYN-PATTERN-001 rather than by rejecting, which an audit suggestion would
have got backwards).

**Registered OPEN, not repaired:** DEV-228 — `ModuleData::items` is one `HashMap<String, Res>` where
NAME-RESOLVE-001 specifies four namespaces, so a type and a value sharing a spelling is rejected
with `E0204`. This is the common cause behind how easily 222/223/225 were reached, and it cannot be
recovered downstream: the distinction is gone once both declarations collapse into one entry. Two
precedence exceptions have now been added to that single map; a third would be the wrong direction.
**Recommend a compiler-track decision on the resolver's namespace model before more precedence work
lands.** DEV-229 — thirty hard-coded builtin path spellings are matched before name resolution runs;
filed UNCONFIRMED because no probe yet distinguishes "the user's declaration won" from "the builtin
won and agreed".

Considered and NOT filed: an audit suggestion to add a central `Res::Variant` well-formedness
invariant. It is reasonable hardening, but no reproducer was produced and nothing observed depends
on it; filing it as a deviation would put a design preference in the defect ledger.

Validation: `cargo fmt --check` clean; `cargo clippy --workspace --all-features --all-targets --
-D warnings` exit 0; starkc lib 579; adversarial_patterns, conformance, module/import/provenance,
DEV-148 and native enum suites green; 20 new regression tests across three files, each verified to
fail against the unfixed compiler; every first-party library package suite green; native debug and
release consumers correct. Not run: full `cargo test --workspace` (shared checkout).
