# WP-ARCH-CLOSE — Final Compiler Architecture Closure Qualification

**Status:** **AUTHORISED AND ACTIVE — CD-400 (owner, 2026-08-12).** **AC1 and AC2 are MET**;
AC4, AC5, AC6 and AC7 are open. AC3's repair has landed (DEV-235 RESOLVED, population A 10 -> 9) but its *exit*
additionally requires two complete clean CI runs with no rerun-to-green, and neither has been run —
so the §8 cohort gate is **not** open. AC1, AC4, AC5, AC6 and AC7 are untouched.
**Authored:** 2026-08-12, against `develop` at `2462b80`.
**Consolidates:** the WP-ARCH-CLOSE package and its *Final Amendments*, into one document.
The amendments are binding and supersede the corresponding base wording; where they did, the base
wording was **replaced, not appended**. §18 records every such replacement, plus three
reconciliations against what the tree actually contains.
**Owning track:** compiler, under `STARKLANG/docs/compiler/COMPILER-CHARTER.md`.
**Relationship to the AS programme:** this is **not** a fifth architecture-stabilization sprint and
does not reopen Sprints 1–4 (packets AS0–AS8), which are closed.

---

## 0. How to execute this package without chat history

A session executing this package needs exactly six inputs:

```text
1  STARKLANG/docs/compiler/COMPILER-CHARTER.md     standing rules, escalations CE1-CE9,
                                                   evidence classes, gate vocabulary
2  COMPILER-STATE.md (repo root)                   current position, append-only decision record
3  starkc/docs/conformance/KNOWN-DEVIATIONS.md     the DEV ledger — DEV-160, DEV-235, DEV-140..145
4  this file                                       the package
5  STARKLANG/docs/compiler/ENGINE-SHARED-FATE-REGISTER.md   AC4's starting map
   STARKLANG/docs/compiler/ENGINE-PUBLIC-CLAIM-CALIBRATION.md  AC6's approved wording (EI6/CD-392)
6  STARKLANG/docs/compiler/audits/AS8-*.md         the mutation and duplicate-authority baseline
                                                   AC4 extends rather than repeats
```

Source-of-truth order is the charter's, unchanged: normative specification first, decision records
second, gate evidence and executable tests third, roadmap fourth, **this file fifth**. Where this
package and a normative specification disagree, the specification wins and this package is
defective — fix it in the same change.

---

## 1. Purpose

This is the final qualification package for STARK compiler architecture stabilization.

Its purpose is to establish, through **adversarial evidence rather than architectural
self-description**, that:

1. the remaining reachable compiler capability gap is closed;
2. the native supported subset is explicit, executable and externally usable;
3. qualification infrastructure produces trustworthy evidence;
4. major semantic rules have identifiable owners and are not maintained through hidden
   engine-local patches;
5. the architecture-stabilization programme can remain closed under a written, falsifiable
   reopen rule.

This package does **not** attempt to make the language feature-complete. It establishes whether
the existing architecture is sufficient for continuing development without an active architecture
programme.

---

## 2. Outcome model — three states

```text
PASS
    All closure criteria satisfied.
    Architecture Stabilization remains CLOSED.

INCOMPLETE
    One or more required work items or qualification criteria remain unmet,
    but no architecture-invalidating finding has occurred.

    WP-ARCH-CLOSE remains OPEN.
    Architecture closure remains PROVISIONAL.
    Architecture Stabilization is NOT reopened.

FAIL-ARCHITECTURE
    A Class-D finding, governing repair-rule violation, or AC7 architecture
    trigger demonstrates that existing architectural boundaries are
    insufficient.

    Architecture Stabilization = REOPENED.
```

Worked examples:

```text
DEV-235 still flaky                                  -> INCOMPLETE
conformance matrix not fully generated               -> INCOMPLETE
DEV-160 implementation unfinished                    -> INCOMPLETE

DEV-160 requires separate HIR and MIR borrow
    authorities                                      -> FAIL-ARCHITECTURE
new resolver precedence exception required           -> FAIL-ARCHITECTURE
consumer patched because the semantic owner
    cannot express the rule                          -> FAIL-ARCHITECTURE
```

**A work item being difficult is not evidence of architectural instability.** Difficulty produces
INCOMPLETE; only an architecture finding produces FAIL-ARCHITECTURE.

There is **no PASS-WITH-UNCLASSIFIED-FINDINGS outcome**. Every finding is classified before the
package reports any state at all.

---

## 3. Baseline

Compiler position at authoring, read from `COMPILER-STATE.md` (2026-08-12, `develop` `2462b80`):

```text
Core        done — qualified subset
MIR         done — qualified subset
Native      done — qualified subset

Architecture Stabilization, Sprints 1-4 (AS0-AS8)   CLOSED
C10 Release Qualification                           CLOSED, PASS-WITH-DEVIATIONS
Campaign A / Campaign B                             EXITED PASS

Population A — 10 open at authoring; **9 after AC3's repair** (§18.6):

  DEV-140..145   the supported-native-subset boundaries (repair DEFERRED by owner decision)
  DEV-160        cross-block borrow absorption — rustc E0502 leak SEALED, capability OPEN.
                 Of the ten, the ONLY one any written code reaches
  DEV-221        ergonomic (Display::fmt on a bounded generic receiver)
  DEV-233        the interpreter loses output written before a trap
  DEV-235        a promotion-gating check that fails on loopback socket timing
                 — RESOLVED 2026-08-12 under AC3 (CD-400). The cause was not timing
```

---

## 4. Governing repair rule

Every defect found by this package MUST first identify its semantic owner.

```text
defect
  ↓
identify normative rule
  ↓
identify owning compiler authority
  ↓
repair authority
  ↓
verify downstream consumers
```

**A repair MUST NOT be accepted merely because it makes the failing program pass.**

The following are architecture findings:

```text
same semantic repair required independently in multiple engines

new precedence/order exception required to preserve behaviour

semantic information must be reconstructed downstream because
the authoritative phase discarded it

new engine-local dispatch mechanism required for a language-level rule

new hard-coded source spelling introduced to bypass normal resolution

consumer patched because the owning authority cannot express the rule
```

An architecture finding **blocks this package** until repaired at its owner, or escalated to
reopening the architecture programme (§2, FAIL-ARCHITECTURE).

---

## 5. AC1 — DEV-160 as the live architecture probe

DEV-160 is in this package for an **architectural** reason, not a feature-completeness one. It is
the strongest currently available live probe of whether the existing borrow architecture can absorb
a substantial semantic capability extension without architectural violence.

The hypothesis under test:

```text
The existing borrow/move authority can represent precise
cross-block reference provenance and liveness without:

    engine-local semantic dispatch
    downstream semantic reconstruction
    duplicated ownership authorities
    backend-specific language restrictions
    precedence/order exceptions
```

### 5.1 Required movement

```text
FROM                            TO
rustc leak sealed               cross-block borrow represented by STARK
capability open                 precise STARK-owned analysis
heuristic provenance detection  valid program builds natively
                                no generated-Rust borrow error
```

The current `borrow_provenance` *"may derive from"* heuristic **MUST NOT** become the permanent
implementation. It must either disappear or be reduced to a non-semantic diagnostic/helper role.

> **Where it lived was itself part of the probe, and that half is now ANSWERED.** `borrow_provenance`
> was defined in `starkc/src/backend/generated_rust/emit_call_thunk.rs` — in the native emitter,
> downstream of the borrow authority — which looks like §4's *"semantic information reconstructed
> downstream because the authoritative phase discarded it"*.
>
> **It was not that shape.** The HIR borrow checker asks *is this program legal*; this analysis asks
> *what does this value borrow in the lowered form*. Different questions, and the second has an
> owner: MIR. The analysis moved to `starkc/src/mir/borrows.rs` on 2026-08-12 by owner decision
> under CE3, with **no exception required** — positive evidence for §5's hypothesis. The
> capability half of DEV-160 remains open; see §18.6.

Expected implementation direction, already identified by the deviation investigation:

```text
reference use
    ↓
backward def-use walk
    ↓
precise reference origin
    ↓
cross-block liveness / absorption
    ↓
place-granular borrow relationship
```

### 5.2 Invariants the implementation must preserve

```text
p.a borrowed, p.b usable where disjoint
mutable/shared overlap rejected
move from a live borrowed place rejected
Copy reads remain legal
non-Copy moves remain ownership checked
```

### 5.3 AC1 evidence

```text
original DEV-160 reproducer          borrow terminated before later mutation
cross-block shared borrow            borrow propagated through locals
cross-block mutable borrow           a move that severs provenance
disjoint field controls              generic forms
overlapping field refusal            nested projections

HIR/MIR/native agreement where execution applies
native debug and native release
Tier-1 platforms
```

### 5.4 AC1 outcomes

```text
lands cleanly inside existing borrow architecture
    -> strong positive architecture evidence; DEV-160 RESOLVED

implementation remains unfinished
    -> WP-ARCH-CLOSE INCOMPLETE (not a failure)

requires an architectural exception prohibited by §4
    -> FAIL-ARCHITECTURE
```

**A named refusal of valid STARK is not sufficient for this package.** The sealed E0502 leak is the
starting position, not the exit.

---

## 6. AC2 — the executable native conformance contract

One authoritative developer-facing matrix for the native compiler, so that

```text
Native = done — qualified subset
```

stops requiring institutional knowledge to interpret.

### 6.1 It must not be a hand-maintained table

Every externally meaningful matrix cell marked `SUPPORTED`, `REFUSED-BY-DESIGN` or
`KNOWN-DEVIATION` must derive from an executable corpus probe or qualification case:

```text
source probe
    ↓
expected semantic result
    ↓
compiler execution
    ↓
recorded normalized result
    ↓
generated conformance cell
```

For supported constructs the probe must exercise the applicable stages:

```text
front end · HIR · MIR lower · MIR verify · MIR execute · native debug · native release
```

For refused constructs the probe must verify the **STARK-owned** refusal and its expected
diagnostic/category — not merely that the build failed.

### 6.2 Columns and statuses

Per externally relevant Core construct or rule:

```text
Parser · Resolver · Typecheck · HIR execution · MIR lowering · MIR verification ·
MIR execution · Native debug · Native release · Tier-1 qualification ·
Status · Limitation / DEV
```

Allowed statuses:

```text
SUPPORTED · REFUSED-BY-DESIGN · DEFERRED · KNOWN-DEVIATION · NOT-APPLICABLE
```

No ambiguous `partial` cell without an explanation.

### 6.3 Drift gating

The published matrix MUST be generated from probe results, and CI MUST regenerate or validate it:

```text
compiler behaviour changes + published matrix does not change   ->  CI FAILURE
published matrix edited    + probe evidence disagrees           ->  CI FAILURE
```

The matrix is therefore a **drift-gated executable developer contract**, not documentation.

### 6.4 DEV-140..145

Each of the six must appear explicitly, each with executable boundary probes, and each as either

```text
SUPPORTED
```

or

```text
documented unsupported native boundary
+ deterministic STARK-owned refusal
+ external-facing limitation
```

They do **not** need to be implemented to make the matrix green. The objective is an honest
contract, and §15 keeps their implementation out of scope.

### 6.5 AC2 exit

An external developer can determine whether a valid STARK construct is supported natively without
reading `COMPILER-STATE.md`, implementation source, or historical deviation records.

> **Status, 2026-08-12: MET.** `starkc/docs/conformance/NATIVE-CONFORMANCE-MATRIX.md` is generated
> by `starkc/tests/native_conformance_matrix.rs` from a live compiler run and validated by that test
> on all three Tier-1 platforms. 20 boundary constructs: 6 `SUPPORTED` (each executed through all
> four engine configurations and compared on the full normative observation), 8
> `REFUSED-BY-DESIGN` with the diagnostic code that refuses them, 6 `KNOWN-DEVIATION` — DEV-140..145,
> every one present with an executable boundary probe, as §6.4 requires. **The drift gate was
> falsified in both directions before it was trusted**: a hand-edited row fails naming the line, and
> a mutated lowering fails naming the probe. The mutation was reverted.
>
> The probe inventory lives in `starkc/tests/support/layer_probes.rs` and is **shared with
> `layer_audit`**, whose three-way verdict is projected from the same staged measurement the matrix
> renders. A matrix with its own classifier would have been the duplicate-authority shape §10 exists
> to find.

> **Build on the existing generators, not beside them.** `starkc/scripts/generate-conformance-report.py`
> (over `STARKLANG/conformance/core-v1-coverage.toml`) and `starkc/scripts/c10-dashboard.py` already
> establish the house pattern: every column resolved against the tree at generation time, never
> hand-asserted prose. A second, differently-shaped generator is itself a duplicate authority.

---

## 7. AC3 — close DEV-235 and make red mean red

Qualification infrastructure is part of the compiler's evidence architecture. A required check that
fails on network timing teaches everyone to re-run it, and a gate that is re-run reflexively has
stopped being a gate.

The loopback/socket gate must not depend on timing luck. Prefer synchronization on observable
readiness/state over:

```text
sleep · timeout guess · retry until green
```

The repaired test must demonstrate that **the failure mechanism has been removed**, rather than
that a timeout was increased. The lifecycle claim still needs a live peer — CD-347/CD-348 require a
resource-shaped provider to successfully acquire, use and release — so deleting the socket is not
an available answer.

Qualification candidate requirements:

```text
complete CI execution #1     GREEN
complete CI execution #2     GREEN

no failed job manually rerun to obtain either result
no required check waived
all Tier-1 platforms covered
```

A failure followed by a successful rerun is evidence to investigate, not qualification evidence.

**AC3 exit:** `DEV-235 RESOLVED`, and required CI failures once again carry information about the
candidate tree.

> **Status, 2026-08-12: the repair has landed; the exit has NOT been met.** DEV-235 is resolved —
> the cause was an accepted socket inheriting `O_NONBLOCK` on macOS/BSD, not timing, and no timeout
> would have fixed it (CD-400, and the ledger's resolution entry). **Run 1 of the two clean CI runs
> is recorded** at `cd6732f`: CI 24/24 and C7.8 4/4, attempt 1 on both, no failed job rerun, all
> three Tier-1 platforms. **One clean run remains**, so AC3 is not complete and the §8 cohort gate
> is not open.
>
> Two cautions, both recorded in CD-400 rather than left to be rediscovered. A green C7.8 run is not
> itself evidence the flake is gone — it was intermittent, and runs looked like this before it was
> found; the falsification is the evidence. And under §13 these runs count toward AC3's exit but are
> **not** final closure evidence: `FINAL_REPAIR_SHA` is unset, and §17 step 9 reruns everything at
> the end regardless.

---

## 8. Cohort entry gate

AC2 and AC3 are launch-critical for the external pre-alpha cohort. **WP-ARCH-CLOSE PASS is not
required before cohort entry.**

```text
AC2 executable conformance contract       COMPLETE
AC3 qualification reliability / DEV-235   COMPLETE
        ↓
    PRE-ALPHA COHORT MAY START
```

These may continue while the controlled cohort is active:

```text
AC1  DEV-160 probe
AC4  adversarial architecture campaign
AC5  patchwork audit
AC7  long-running reopen instrumentation
```

The cohort then becomes an additional adversarial evidence source. Externally discovered cases
enter the normal DEV triage and architecture-trigger classification process (§12).

**DEV-160 is therefore not on the critical path to pre-alpha participation.**

---

## 9. AC4 — adversarial architecture validation

Do not validate architecture by reading its comments or its ledger. **Attack the authorities.**

Semantic boundaries to exercise, at minimum:

```text
resolution / namespaces          borrow / move ownership
pattern legality                 Drop determination
type identity and Copy           MIR lowering
trait / bound dispatch           MIR verification
generic specialization env.      runtime-function classification
                                 provider/resource ownership
```

For every authority selected:

```text
1. identify the architectural claim
2. mutate or deliberately break the authority
3. identify what SHOULD detect the break
4. run that control
5. record whether the mutation was killed
6. classify the evidence as independent or shared-fate
```

At least one adversarial mutation per critical authority. **A test that derives its expected result
from the mutated authority does not count as independent evidence.**

Existing resolver audit evidence may be reused, but the resolver alone is not sufficient to close
this package. AS8's mutation trials and `starkc/scripts/as8-mutate.py` are the baseline to extend.

**AC4 exit:** every critical authority has either an `independent falsifier`, or an explicit
shared-fate classification with an identified alternative control. **No authority may be described
as independently verified solely because HIR, MIR and native inherit the same answer.**

---

## 10. AC5 — patchwork / special-case audit

A targeted source audit for architectural residue. Search specifically for:

```text
hard-coded semantic source spellings      backend-specific acceptance rules
name-based semantic dispatch              precedence exceptions
engine-local reconstruction of            special handling keyed to individual builtins
    type/generic information              copy/paste semantic tables
duplicated semantic classifiers           consumer/package workarounds for
TODO / FIXME / workaround / temporary          compiler limitations
```

Every finding receives exactly one classification:

```text
A — legitimate language/runtime special case
B — deliberate independent verifier implementation
C — architecture debt, safe and explicitly tracked
D — patchwork / semantic authority violation
```

**Class D blocks closure** (§2, FAIL-ARCHITECTURE). Class C requires a named owner and a
disposition, but does not automatically block closure.

Known examples — builtin fallback handling, and the deliberate MIR interpreter/verifier redundancy
recorded as `AS8-DA-002/003/004` — are to be **classified**, not automatically removed.

The objective is not "zero special cases". The objective is:

```text
zero unclassified semantic exceptions
zero known symptom patches
zero accidental duplicate authorities
```

---

## 11. AC6 — the public architecture claim

The project MUST NOT claim

```text
three independent compiler implementations agree
```

unless the rule under discussion is actually independently implemented. The accurate — and
technically stronger — claim is that STARK has one shared semantic front end, a HIR execution
oracle, verified MIR lowering, an independently checking MIR verifier, a MIR execution engine, and
a native execution path; that observable outcomes are differentially qualified across those
boundaries; and that shared semantic authorities and their shared-fate limitations are explicitly
tracked.

> **This criterion is largely pre-satisfied, and AC6 is scoped accordingly.** EI6 (CD-392,
> `ENGINE-PUBLIC-CLAIM-CALIBRATION.md`) approved the calibrated wording, and its three corrections
> and the rustc addition **have landed** — `website/src/content.ts` now reads *"Three engines, four
> configurations, one answer"*, carries the scoped agreement paragraph, and describes rustc's role;
> `CLAUDE.md` and `AGENTS.md` say *"four engine configurations"*. The prohibited phrase occurs
> nowhere. See §18.3.

**AC6 is therefore a verification sweep, not a rewrite:**

```text
1. re-run EI6's searches over README.md, website/src/content.ts, CLAUDE.md, AGENTS.md,
   ROADMAP.md, STARKLANG/docs/ROADMAP.md, and any surface added since 2026-08-09
2. reconcile the published wording with AC4's findings — if AC4 demotes an authority
   previously described as separately checked, the public copy is now overstated
3. record the result, including "no change required" if that is what the sweep finds
```

---

## 12. AC7 — the architecture reopen rule, with a mechanical sensor

Architecture Stabilization remains closed after this package unless subsequent evidence triggers
the reopen rule.

### 12.1 The rule

For the next **20 substantive compiler defects**, reopen Architecture Stabilization if any of:

```text
A. Two defects require changing the same language-level semantic
   rule independently in multiple execution engines.

B. Two defects require new precedence/order exceptions rather than
   representation of the namespace/semantic distinction involved.

C. Any defect requires introducing a new engine-local language
   dispatch authority.

D. Any defect shows that an authoritative compiler phase discarded
   semantic information that downstream phases must reconstruct.

E. An adversarial test proves a claimed independent checker actually
   shares the mutated authority and no independent control exists.

F. Two defects attributed to one authority demonstrate that the
   authority boundary itself is insufficient rather than merely
   containing implementation bugs.
```

Ordinary local defects do not reopen the programme:

```text
pattern arity validation missing                     -> ordinary compiler defect
borrow checker forgot one projection case            -> ordinary compiler defect,
                                                        if the repair belongs cleanly there
HIR and MIR require unrelated trait-selection
    implementations                                  -> architecture finding
resolver requires another hand-written
    precedence exception                             -> architecture finding
```

### 12.2 The sensor — triage tagging

Every new substantive compiler DEV entry created after WP-ARCH-CLOSE begins MUST carry an
architecture-trigger field **at triage**:

```text
Architecture trigger:

    NONE
    AC7-A · AC7-B · AC7-C · AC7-D · AC7-E · AC7-F
    PENDING-CLASSIFICATION
```

**A DEV entry may not leave triage with `PENDING-CLASSIFICATION`.** The twenty-defect observation
count is then derived **mechanically from the ledger**, not reconstructed from memory.

`starkc/scripts/c10-deviation-populations.py` already parses that ledger to compute populations and
is the natural place for the count and the `PENDING-CLASSIFICATION` prohibition to become
executable, rather than a second parser beside it.

### 12.3 What counts toward the twenty

Counts:

```text
a demonstrated compiler semantic, lowering, execution, ownership,
resolution, verification, or backend correctness defect
```

Does not count:

```text
documentation · formatting · test infrastructure · CI infrastructure ·
developer ergonomics with no semantic consequence · release packaging
```

Ambiguous cases are decided **at DEV triage and recorded there**, never retrospectively when the
twenty-defect rule is being evaluated.

### 12.4 After twenty

After twenty qualifying substantive compiler defects without an AC7 trigger:

```text
Architecture closure = NORMAL STANDING ASSUMPTION
```

rather than provisional.

---

## 13. Final-evidence freshness rule

**No WP-ARCH-CLOSE qualification evidence may predate the final repair affecting the claim it
supports.**

Before any final PASS determination, establish

```text
FINAL_REPAIR_SHA
```

— the latest commit containing any semantic, architectural, qualification, conformance-contract or
AC-required repair produced by this package.

All final qualification evidence MUST be generated from `FINAL_REPAIR_SHA`, or from a descendant
containing no subsequent compiler-affecting repair. This includes:

```text
two clean CI qualification runs        native conformance generation
full differential suite                Tier-1 qualification
adversarial/mutation results           first-party package/provider qualification
                                       patchwork-audit final result
```

Evidence produced before a later relevant finding or repair is **historical evidence** and cannot
satisfy final closure. This is what prevents closure evidence from predating the defects or repairs
it is claimed to cover.

---

## 14. Final qualification

WP-ARCH-CLOSE reaches **PASS** only when every line is true:

```text
[ ] AC1 architecture probe complete and DEV-160 resolved
[ ] AC2 generated executable conformance contract complete
[ ] DEV-140..145 represented by executable boundary probes
[ ] AC3 / DEV-235 resolved
[ ] AC4 adversarial architecture campaign complete
[ ] AC5 patchwork audit complete
[ ] no Class-D finding remains
[ ] every Class-C architecture debt explicitly owned
[ ] shared-fate register reconciled
[ ] public architecture wording corrected
[ ] AC7 triage tagging operational
[ ] reopen rule committed
[ ] FINAL_REPAIR_SHA established
[ ] all closure evidence postdates FINAL_REPAIR_SHA
[ ] two complete clean CI runs from the qualifying tree
[ ] no rerun-to-green used for those runs
[ ] full compiler corpus green
[ ] HIR/MIR/native differential qualification green
[ ] first-party package/provider qualification green
[ ] Tier-1 native debug/release qualification green
```

If requirements remain incomplete **without** an architecture violation:

```text
WP-ARCH-CLOSE = INCOMPLETE
Architecture Stabilization = CLOSED, PROVISIONAL
```

If an architecture trigger occurs, the package does **not** patch around it. It records:

```text
WP-ARCH-CLOSE = FAIL-ARCHITECTURE
Architecture Stabilization = REOPENED
```

and creates the smallest architectural repair packet necessary.

If every criterion passes:

```text
WP-ARCH-CLOSE = PASS

Compiler architecture = STABILIZED
Architecture Stabilization = CLOSED
The provisional label remains governed only by AC7's twenty-defect observation period
```

---

## 15. Explicit non-goals

Do **not** include, merely to obtain a cleaner deviation count:

```text
DEV-221 ergonomic Display syntax
DEV-233 interpreter pre-trap output behaviour
implementation of all DEV-140..145 capability boundaries
Cranelift migration
tensor expansion
new language features
new package ecosystem work
general compiler refactoring
code cleanup unrelated to a demonstrated finding
```

Those continue through their appropriate tracks.

---

## 16. Successful final state

A PASS closeout permits the project state to say:

```text
Compiler architecture: STABILIZED

Architecture Stabilization:
    CLOSED
    independently closure-qualified by WP-ARCH-CLOSE

Known reachable compiler capability gaps:
    none

Native language coverage:
    explicitly documented by a generated, drift-gated conformance matrix

Qualification infrastructure:
    reliable — red required checks are actionable

Semantic authority policy:
    every demonstrated defect repaired at its owning authority

Architecture reopen policy:
    active for the next 20 substantive compiler defects

Next work:
    standalone toolchain / C9 Part B / application platform
```

At that point continuous compiler-architecture work ends, and future compiler defects are ordinary
maintenance unless the reopen rule fires.

**On an INCOMPLETE closeout none of the above may be written.** The state file records
`WP-ARCH-CLOSE OPEN, architecture closure PROVISIONAL`, names the unmet criteria, and says so in
the position line.

---

## 17. Execution order

The package is now to be executed rather than further redesigned.

```text
1. AC3 — DEV-235 / trustworthy qualification
2. AC2 — executable native conformance contract

        ── PRE-ALPHA COHORT MAY START ──

3. AC1 — DEV-160 architecture probe
4. AC4 — adversarial authority campaign
5. AC5 — patchwork/special-case audit
6. AC6 — public architecture wording
7. AC7 — triage sensor + reopen instrumentation
8. establish FINAL_REPAIR_SHA
9. rerun ALL final qualification evidence
10. PASS / INCOMPLETE / FAIL-ARCHITECTURE decision
```

No further architecture-package refinement is planned unless execution exposes a genuine ambiguity
or an architectural finding.

---

## 18. Consolidation record

### 18.1 What the amendments replaced

| Base package wording | Superseded by |
| --- | --- |
| §1 "ends in one of two states: PASS or FAIL" | §2 three-state model. `INCOMPLETE` is new, and is the state an unfinished DEV-160 produces |
| §4 AC1 framed as "close DEV-160 completely" | §5 AC1 framed as a **live architecture probe**, with the hypothesis stated and three named outcomes. The exit is unchanged; the reason for including it is now explicit, and difficulty is evidence-producing rather than failure |
| §5 AC2 "create one authoritative matrix" | §6 AC2 **executable and CI-drift-gated**: every meaningful cell derives from a probe, and CI fails on either direction of drift |
| §6 AC3 (unchanged in substance) | §7, plus §8: AC2+AC3 are the **cohort entry gate**, and PASS is not required for cohort entry |
| §10 AC7 rule with no measurement mechanism | §12, with a mandatory ledger triage field, a prohibition on leaving triage `PENDING-CLASSIFICATION`, and an explicit counts/does-not-count list |
| — (absent) | §13 **final-evidence freshness rule** and `FINAL_REPAIR_SHA` |
| §11 final qualification (15 lines) | §14 (20 lines), which adds the freshness, triage-sensor and executable-probe criteria |
| §13 successful final state | §16, plus the explicit statement of what may **not** be written on an INCOMPLETE closeout |

### 18.2 Reconciliation — the AS programme's own numbering

The base package says "AS1–AS4". In the tree the architecture-stabilization programme ran as
**Sprints 1–4**, whose packets are numbered **AS0–AS8**
(`STARKLANG/docs/compiler/audits/AS-SPRINT1-CLOSEOUT.md`, `AS-SPRINT2-CLOSEOUT.md`,
`AS-SPRINT4-CLOSEOUT.md`, `SPRINT-4-CLOSURE.md`; packets AS6/AS7/AS8 closed under CD-390/391/394).
Consolidated as "Sprints 1–4 (packets AS0–AS8)". No scope change is implied — the closed programme
is the same one.

### 18.3 Reconciliation — AC6 is largely already done

The base package's AC6 reads as if the public claim still needs correcting. It does not. EI6
(CD-392) measured it on 2026-08-09 and found **the prohibited phrase was never published**; its
three corrections and the rustc addition have since landed in `website/src/content.ts`. AC6 is
therefore consolidated as a **verification sweep with a reconciliation duty against AC4's
findings**, which is the part that is genuinely still open — AC4 can demote an authority and make
today's accurate copy overstated tomorrow.

### 18.4 Reconciliation — population A is 10, and the baseline says so

The base package's §2 lists DEV-160, DEV-140..145, DEV-221, DEV-233 and DEV-235. That is exactly
the ten `COMPILER-STATE.md` reports at `2462b80`, so §3 states the count explicitly rather than
leaving it to be recounted. DEV-228/229/232/234 closed on 2026-08-11 and are **not** in scope here.

### 18.5 Governance home — SETTLED by CD-400

This package is neither a gate under `COMPILER-ROADMAP.md` nor a track under `ROADMAP.md` §0, so it
needed a home in the decision record before AC3 could start. **CD-400 (owner, 2026-08-12)** is that
home: it authorises WP-ARCH-CLOSE as the active packet, records the three-state outcome model, and
puts two rules into force immediately rather than at closure — the AC7 triage field (§12.2) and the
`FINAL_REPAIR_SHA` freshness rule (§13). `COMPILER-STATE.md`'s position line and its "Active
packet" row both name the packet as of that entry.

### 18.6 Execution log

| Date | Item | Result |
| --- | --- | --- |
| 2026-08-12 | Package consolidated and authorised | CD-400 |
| 2026-08-12 | **AC3 repair** — DEV-235 | **RESOLVED.** Cause was `O_NONBLOCK` inherited by the accepted socket on macOS/BSD, not timing. Falsified by removing the repair (deterministic failure); 12/12 green restored, three consecutive runs. Population A 10 -> 9 |
| 2026-08-12 | **AC2** — executable native conformance contract | **MET.** Generated matrix, drift-gated in both directions and falsified both ways. 20 constructs: 6 SUPPORTED / 8 REFUSED-BY-DESIGN / 6 KNOWN-DEVIATION. DEV-140..145 all present as executable boundary probes. Probe inventory shared with `layer_audit`, so no second classifier |
| 2026-08-12 | AC3 **exit**, run 1 of 2 | **RECORDED.** `cd6732f` — CI `31563159250` 24/24, C7.8 `31563159221` 4/4, **attempt 1 on both, no rerun-to-green**, all three Tier-1 platforms. First off-machine test of the matrix's platform-independence claim and of the DEV-235 repair on Linux/Windows |
| 2026-08-12 | CI at `d300d3d` | **FAILED, 7 of 24, attempt 1 — and the failure was ours.** All seven trace to one cause: `mir::borrows`'s inline test module assembles the pipeline by hand and was not registered in AS2's `TEST_ONLY`. Repaired as the guard's own message directs. Local verification had run the targeted suites but not the eleven source-scanning architecture guards, each of which runs in under a second — that batch is now the pre-push step |
| — | AC3 **exit** | **NOT MET, and the two-run count is RESET.** Under §13 the `cd6732f` run is historical: the AC1 landing and the AS2 repair both postdate it. Both clean runs must come from the qualifying tree, and per §17 step 9 all final evidence is rerun at the end regardless. The §8 cohort gate stays shut |
| 2026-08-12 | **AC1 step 1** — borrow-origin analysis | **DONE, POSITIVE.** Moved from the native emitter to `starkc/src/mir/borrows.rs` (owner, CE3). Not a §4 finding — the HIR checker answers a different question, so nothing was reconstructed, only misplaced. Two consumer type checks and a `by_value_tys` map deleted because the authority became correct; a fourth copy of AS4's `stores_a_reference` deleted with them. Controls: 5 unit + 4 AC1 probe + 8 DEV-160 + 132 MIR-diff + 129 three-engine + 584 lib, and **33 first-party applications built natively** |
| 2026-08-12 | AC1 mutation trials | **4 rules mutated; 2 controlled, 2 not.** First trial reported 1 of 3 killed and was wrong — the program did not reach the rules, which masked one another. Adding a `(String, &str)` shape made the move rule falsifiable. The statement dest guard and aggregate filter survived verified-applied mutations and are labelled precautionary in the module rather than counted as verified |
| 2026-08-12 | **AC1 step 2** — cross-block absorption | **DONE. DEV-160 RESOLVED, population A 9 -> 8.** The thunk absorbs the call that produced the borrow; the reported shape and two variants build, run, and agree across all four engine configurations. The cheap repair — laundering the reference through a raw pointer at the call site — was **ruled out on Stacked Borrows grounds**: the thunk's `&'a mut` invalidates tags derived from any earlier borrow, so the reference must be created inside. Miri passes on the new shape under CI's flags; both fixture guards pass and the new one was falsified |
| — | **AC1 exit** | **MET.** DEV-160 resolved, and the probe's verdict is POSITIVE: no engine-local dispatch, no downstream reconstruction, no duplicated authority, no precedence exception — one backend-specific restriction REMOVED and none added. Residual, stated: DEV-160c and DEV-160d unchanged and still refused by name, and only one producer may be absorbed per thunk |
| 2026-08-12 | AC3's two-run count, structurally | **It cannot accumulate while repairs land.** `f780bb3` was fully green (24/24 + 4/4, attempt 1, all Tier-1 plus Miri) and is disqualified by §13 the moment DEV-160 lands, exactly as `cd6732f` was. This is the package's own design, not a defect in the runs: §17 step 9 collects the two runs AFTER `FINAL_REPAIR_SHA`. Read the count as "not yet started" until the repair sequence ends |
| — | AC4, AC5, AC6, AC7 | **NOT STARTED** |
