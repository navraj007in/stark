# WP-C10 — Compiler Release Qualification (execution plan)

**Status:** **APPROVED WITH AMENDMENTS (owner, 2026-08-09).** All six opening decisions are RULED —
see §2. C10-0 is authorised to begin **once a green exact-head CI baseline exists** (§1.2).
**Authored:** 2026-08-09, against `develop` at `f12ececca6d4bdabf828d657c4a4f719a7f9c39a`.
**Amended:** 2026-08-09 — owner rulings OD-1…OD-6 recorded; three methodological amendments
applied (authority of the normative specifications, §"How to execute"; no expected finding rate,
§6.4/§8.3; inherited-mutation freshness rule, §8.2a).
**Gate:** C10 — Compiler Release Qualification (`COMPILER-ROADMAP.md` §GATE C10).
**Owning track:** compiler, under `COMPILER-CHARTER.md`.

## How to execute C10 without chat history

A session executing this plan needs exactly five inputs:

```text
1  STARKLANG/docs/compiler/COMPILER-CHARTER.md    standing rules, escalations CE1-CE9,
                                                  evidence classes §5.2, gate vocabulary §5.3
2  COMPILER-STATE.md (repo root)                  current position, append-only decision record
3  STARKLANG/docs/compiler/COMPILER-ROADMAP.md    Gate C10 contract, release classes, §4.5
4  this file                                      the execution plan
5  the packet-specific documents named in §4      inherited evidence, by exact path
```

**These five are the required navigation and control inputs — they are not the whole authority.**

> **The normative Core and extension specifications retain their authority under
> `COMPILER-CHARTER.md` §1.6 rule 1 and §1.9.** This plan governs *how C10 is executed*; it does
> not and cannot govern *what STARK means*. Packet-specific evidence documents and the normative
> rules referenced by this plan must be read when the corresponding claim is qualified — C10-A1
> and C10-A2 are built on granular normative rule IDs and their exact spec homes, so the rest of
> this plan already behaves this way.

Source-of-truth order is the charter's §1.9, unchanged:

```text
1 approved normative Core or extension specification
2 approved decision record in COMPILER-STATE.md or a gate proposal
3 gate exit evidence and executable tests
4 the compiler roadmap
5 engineering plan / work-package documentation   <- THIS FILE SITS HERE
6 README, CLAUDE context, implementation notes, commit messages
7 archived pre-pivot documents — never authoritative
```

Where this plan and the roadmap disagree, the roadmap wins and this plan is defective — fix it in
the same change. Where this plan and a normative specification disagree, the specification wins and
this plan is defective in the same way.

---

# 1. Verified starting state

Everything in this section was read from `develop` at the head named above on 2026-08-09. Nothing
here is inherited from a summary.

## 1.1 Head and branch topology

```text
develop            f12ececca6d4bdabf828d657c4a4f719a7f9c39a
                   "Merge main into develop — sync only, zero file content"
Sprint 3 merge     645997d   PR #10, 2 parents, no rebase/squash/cherry-pick
Sprint 4 merge     d79ad03   PR #11, 2 parents, no rebase/squash/cherry-pick
```

The branch this plan was authored from, `wp-arch-stability/sprint-4` at `3f18e49`, is **fully merged
into `develop` and 57 commits behind it**. It carries nothing `develop` lacks. C10 executes on
`develop` (or a branch taken from it), never from `wp-arch-stability/*`.

## 1.2 CI at the head — NOT yet green, and this matters

At authoring time the CI run for `f12ecec` (run `31292404920`, push event) was **`in_progress`**:

```text
success   C6.5 corpus replay (linux-x64) (macos-arm64), C6.5 mutation controls,
          C6.5 corpus tier-1 agreement, spec fixture conformance,
          External sample suite (pinned), C6.4 windows tier-2 gap probe,
          DEV-160 raw slot primitives under Miri,
          release package smoke (linux-x64) (macos-arm64) (windows-x64),
          C7 P1 REST workload (all three), first-party package qualification (linux-x64)
                                                                              (windows-x64)
pending   fmt/clippy/test (all three), C6.4 tier-1 qualification (linux-x64) (macos-arm64),
          first-party package qualification (macos-arm64)
```

**C10-0 may not record "CI green at HEAD" until it has read a completed run for the exact SHA it
freezes.** Two runs race on every `develop` push (push event and the open `develop -> main` PR
event, deliberately queued rather than cancelled by the `concurrency:` group); C10-0 must name
**which run id** it read, not "CI".

## 1.3 Gate and campaign position, as the records state it

| Item | State at HEAD | Record |
| --- | --- | --- |
| Gates C0–C8 | CLOSED. C8 closed **deliberately short** on requirement 8 by owner override (CD-385) | `GATE-C8-CLOSURE.md` |
| Gate C9 | Part A closed (C9.0/C9.1/C9.2). **Part B blocked** pending second-artifact evidence | `COMPILER-STATE.md` current-position block |
| Campaign A | PASS, closed, CI-confirmed | `CAMPAIGN-A-EXIT-REPORT.md` |
| Campaign B | **EXITED PASS 2026-08-09.** Explicitly a prerequisite for C10; makes no stability or conformance claim | `CAMPAIGN-B-EXIT-REPORT.md` |
| AS6 / AS7 / AS8 | CLOSED — CD-390 / CD-391 (criterion 2 re-qualified CD-393) / CD-394 | the three audit files |
| Sprint 4 Tier-3 | PASS, exact-head CI run `31290518438` on `4eea128`, 24 jobs, 0 failures | `AS-SPRINT4-CLOSEOUT.md` |
| WP-ARCHITECTURE-STABILIZATION | **COMPLETE**, all four sprints closed | that file's header |
| Mandatory path | Core=done, MIR=done, Native=done | position line |
| Tensor track | Deferred research on Gate 7's terms. **Not reopened by C10.** | Charter §1.5, CLAUDE.md |

The user-supplied brief's programme summary is **confirmed in every particular** against these
records.

## 1.4 Measured inventories at HEAD

These are the raw populations C10's denominators will be drawn from. **They are measurements, not
targets, and none of them is yet a denominator** — §7 governs how a denominator is declared.

```text
integration test targets     210   top-level `.rs` files under starkc/tests/
                                   (AS8 recorded 209; +1 since. Reconcile in C10-0.)
test module dirs                   tests/common, tests/fixtures (21), tests/support (4)
spec fixture files           117   STARKLANG/tests/spec-fixtures/
legacy conformance rules      59   STARKLANG/conformance/core-v1-coverage.toml
granular semantic-freeze IDs 168   semantic-freeze/CORE-V1-COMPLETENESS.md  (stated as 161
                                   when this table was written; corrected by C10-A1 — see below)
granular rules with precise
  positive/negative evidence   36   STARKLANG/conformance/core-v1-c2.11-evidence.toml
C6.5 differential corpus      89   70 generated / 13 handwritten sentinels / 6 retained
metamorphic families          12   M01-M12, starkc/tests/c6-corpus/metamorphic.py
frozen perf workloads          7   w01-w07, starkc/benchmarks/c7-workloads/FROZEN.json
shared-fate authorities       11   ENGINE-SHARED-FATE-REGISTER.md (after the TRAP-001 split)
duplicated authorities         6   AS8-DA-001..006
AS8 mutation trials           39   26 CONFIRMED / 13 FALSIFIED, in both directions
AS8 residuals                 15   AS8-R1..R15; R3 and R15 discharged
deviation headings           186   `## DEV-` in KNOWN-DEVIATIONS.md
distinct DEV ids             178   in that file — the file is APPEND-ONLY; last heading wins
CI workflows                   2   ci.yml            14 job definitions -> 24 matrix-expanded
                                   c78-native-...     2 job definitions ->  4 matrix-expanded
                                                     28 checks total, matching CD-389's "28/28"
```

> **CORRECTED 2026-08-09 by C10-A1 (A1-F1): the denominator is 168, not 161.** Seven
> three-segment `NUM-*` IDs — all the numeric-semantics rules, including integer overflow and
> division by zero — were invisible to the counting method used here. The population is unchanged;
> the enumeration was faulty. Measured buckets: PRECISE 36, AGGREGATE 85, ABSENT 42, N/A 5.
> `audits/C10-A1-EVIDENCE-CENSUS.md`. **The paragraph below is preserved as written** — its
> argument is unaffected and its number is not.

**The 161-vs-36 gap is the single most important number in this table.** 161 granular semantic
rules exist; 36 carry positive/negative evidence at test-function precision. The remaining 125 are
cited only through the legacy 59-rule database, whose own header records that its `tests` field
"does not distinguish positive from negative coverage, and often cites only the aggregate
`starkc/tests/conformance.rs` fixture-corpus runner with no per-rule attribution within it" —
DEV-017. **C10.1 cannot produce a per-rule dashboard over 161 rules from evidence that exists for
36 without inventing attribution.** §6.2 and §7.2 are built around that constraint.

## 1.5 Versions and platform matrix at HEAD

```text
starkc                0.1.0        starkc/Cargo.toml
rust-version (MSRV)   1.85         starkc/Cargo.toml; CI uses dtolnay/rust-toolchain@stable
MIR_VERSION           "0.4"        starkc/src/mir/mod.rs:59
MIR_RUNTIME_SURFACE   separate constant, bumped independently of MIR_VERSION
layout contract       stark-64-v1  per-target, from starkc/target-matrix.json
target matrix         starkc/target-matrix.json, schema `stark-target-matrix-1`,
                      pinned to src/target.rs in both directions by
                      c64_platform_matrix.rs::target_matrix_json_matches_the_compiler
Tier-1                linux-x64, macos-arm64
Tier-2                windows-x64 (C6.4 row 25 REPORT-ONLY; a "windows tier-2 gap probe" job)
```

**There is no compiler-facing "language version" constant.** `starkc` 0.1.0 is a crate version.
C10.5 must not assume one exists; declaring the separation is part of C10.5's work, not an input.

## 1.6 Distribution position (inherited from the platform track, not the compiler track)

`ROADMAP.md` §1, quoting `HC13-RELEASE-CHECKLIST.md` §0:

```text
Installer Phase I / compiler distribution   IMPLEMENTED
Standalone first-party toolchain            PARTIAL      payload carries compiler, runtime and
                                                         provider ABI — NOT the first-party
                                                         package/provider set
Offline package/provider build              NOT PROVEN   a clean machine cannot build an
                                                         HTTP/TLS program without obtaining the
                                                         packages separately
Public signed distribution                  NOT PROVEN   the manifest establishes INTEGRITY,
                                                         not AUTHENTICITY
DEV-165                                     OPEN         connect_timeout accepted and ignored
```

This is **release/distribution weakness**, a category C10.4 must keep separate from compiler
correctness and from security vulnerability (§11.2). It constrains the *wording* C10.7 may
authorise; it does not by itself block a compiler-scoped release claim. **DEV-165 is not in
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`** — see §2, OD-3.

---

# 2. OPENING OWNER DECISIONS — ALL SIX RULED (2026-08-09)

**Every decision below is CLOSED.** Each is preserved in its original form — what the documents
say, whether they conflict, what is technically possible, the ruling requested — with the owner's
ruling appended. The analysis is kept rather than replaced by the answer, because the analysis is
what a later reader needs in order to know whether the ruling still holds.

Summary:

```text
OD-1  APPROVED               C9 Part B does not block C10
OD-2  APPROVED               evaluate Core v1 Compiler Stable + Native Systems Preview only
OD-3  APPROVED W/ REFINEMENT three separately countable populations, not one denominator
OD-4  MODIFIED               close both DEVs during C10 rather than carrying them; neither
                             reopens C8; unresolved state NARROWS the language-services claim
OD-5  APPROVED               dated superseding record; CD-394 and AS8's exit qualification
                             preserved untouched
OD-6  APPROVED               correct the live ROADMAP summary; §6.0's gate text preserved and
                             marked satisfied
```

**C10-0's first act** is to transcribe these six rulings into `COMPILER-STATE.md` as one dated
`CD-NNN` entry, and to make the two document corrections OD-5 and OD-6 authorise. CD numbers run
ahead of HEAD in this repository — read the maximum from `git log --all` before allocating, and
never force-push to relabel.

## OD-1 — Does C9 Part B block C10?

### What the documents actually say

**`COMPILER-STATE.md`, current-position block:**

```text
Gate: C9  Next: C10 release qualification  Blocked: Gate C9 Part B (second-artifact evidence)
```

**`COMPILER-CHARTER.md` §2.4** defines that line's schema as
`Gate: C<n>  Next: WP-C<n>.<m>  Blocked: <none|reason>`. The `Blocked:` field describes the
**current gate**, which the same line names as C9. It does not, on its own schema, say anything
about C10.

**`COMPILER-ROADMAP.md` §4.5 (release path)** — the only place the roadmap states C10's entry
condition:

> A compiler-track release qualification gate may open when:
> `C0–C8 are closed` + `P1 is complete for Native Systems Preview or STARK v1 General-Purpose Stable`
> + `C9 status is explicit (done, blocked on second-artifact evidence, or not required for this release)`
> + `tensor capability/deviation status is explicit`
>
> A release does not require every optional artifact or tensor-expansion track to be complete.

**`COMPILER-ROADMAP.md` WP-C10.7:**

> A compiler-track completion release requires C0–C8 and the mandatory native path C3–C7 to be
> closed.

**`COMPILER-ROADMAP.md`, release class "Core v1 Compiler Stable":** "Requires C7, C8, and C10". C9
is absent.

**`COMPILER-ROADMAP.md` §4.4 (artifact-infrastructure path):** C9.3 onward is gated on the
ecosystem second-artifact result. **Charter §1.3 and escalation CE7** forbid generalising from ONNX
alone.

### Do they conflict?

**No — the documents are consistent, and the conflict is in how the position line has been read.**

The roadmap names C9 as an **optional track whose status must be explicit**, and "blocked on
second-artifact evidence" is one of the three statuses it explicitly permits. C9 is excluded from
the mandatory set in three independent places (§4.5's `C0–C8`, WP-C10.7's `C0–C8`, the release
class's `C7, C8, and C10`). The state file's `Blocked:` field is scoped to the current gate by the
charter's own schema.

The residual risk is not contractual but **evidential**: WP-C10.1's dashboard scope includes
"extension isolation" and "tensor extension rules and backend capabilities". Both are supplied by
**C9 Part A**, which is closed (C9.1 extension-isolation conformance matrix, exercised at HEAD by
`starkc/tests/c91_extension_isolation.rs`; C9.2 tensor provider stage map). C10.1 therefore has its
inputs. **No C10.1 dashboard row may depend on C9.3–C9.6.**

### Can C10 proceed?

**Technically yes** — every input C10 consumes exists at HEAD. **Contractually yes**, on §4.5, and
only if the ruling below is recorded, because the current position line invites the opposite
reading and an unrecorded interpretation is exactly the drift this repository has been repairing.

### Smallest ruling needed

> **OD-1 ruling requested:** C9 Part B is an **optional track carried as an explicit deferral**, not
> a C10 entry blocker. C10 opens with C9 status recorded as *"Part A closed; Part B deferred pending
> second-artifact evidence; no provider generalisation authorised from ONNX alone (CE7)."* The
> `COMPILER-STATE.md` position line is amended to read
> `Gate: C10  Next: C10-0  Blocked: none` with C9 moved to the optional-tracks line as
> `ArtifactInfra=blocked (Part B, second artifact)`.
>
> **Nothing in C9's historical record is edited.** The amendment is a new dated entry.

### OWNER RULING — APPROVED (2026-08-09)

**C9 Part B does not block C10.** The owner records that this plan's reading supersedes the earlier
reading of the position line: C10's entry contract explicitly permits C9 to be blocked on
second-artifact evidence, and C9 Part A already supplies the extension-isolation and tensor-stage
inputs C10 consumes.

Record, as a **new dated decision** — C9's history is not rewritten:

```text
C9 Part A      CLOSED
C9 Part B      DEFERRED — second-artifact evidence required
               no generalisation from ONNX alone (CE7)

C10            MAY OPEN

COMPILER-STATE.md position line:
Gate: C10
Next: C10-0
Blocked: none

Optional tracks:
ArtifactInfra = blocked/deferred at C9 Part B
```

## OD-2 — Which release classes is C10 evaluating?

The roadmap defines four (Native Developer Preview, Native Systems Preview, Core v1 Compiler
Stable, STARK v1 General-Purpose Stable). C10-0 must freeze **which of them C10 gathers evidence
for**, because the evidence sets differ and a class chosen at exit is a class chosen after seeing
the result — forbidden by §7.

Recommended, but the owner's call:

```text
EVALUATE   Core v1 Compiler Stable          the class C10 exists to qualify
EVALUATE   Native Systems Preview           already satisfiable on C6+P1 evidence; cheap to
                                            state precisely and useful as a fallback claim
DO NOT     STARK v1 General-Purpose Stable  same evidence, wider claim (CD-022). Deciding to
                                            make the wider claim is a separate owner act, not a
                                            C10 finding
DO NOT     Native Developer Preview         subsumed
```

### OWNER RULING — APPROVED exactly as proposed (2026-08-09)

```text
EVALUATE          Core v1 Compiler Stable
                  Native Systems Preview

DO NOT EVALUATE   STARK v1 General-Purpose Stable
                  Native Developer Preview
```

The owner's reasons, recorded because they bind C10-Q's wording: the Systems Preview is a **useful
fallback claim**, and the General-Purpose claim is **materially wider even though it rests on much
of the same evidence** — it therefore requires a separate owner act and is not something a C10
finding can reach.

## OD-3 — What is the denominator for "every open deviation carries an owner"?

CD-021, carried into WP-C10.7: *"an open deviation with no owner blocks the release decision."*
**The plan cannot execute that rule until the population it ranges over is defined**, and at HEAD
there are at least three disjoint ledgers using one `DEV-NNN` namespace:

```text
starkc/docs/conformance/KNOWN-DEVIATIONS.md   186 headings, 178 ids, APPEND-ONLY
COMPILER-STATE.md                              the append-only decision record; DEV-165 appears
                                               here 6 times and in KNOWN-DEVIATIONS.md ZERO times
STARKLANG/docs/http-client/HC13-*.md           HTTP-track deviations incl. DEV-165, which
                                               ROADMAP.md §1 says "still blocks a public release"
AS8-R1..R15                                    residuals, deliberately NOT DEVs (owner ruling:
                                               a mutation survivor means the evidence cannot
                                               detect a defect, not that a defect is present)
AS8-DA-001..006                                duplicated authorities, deliberately outside the
                                               frozen ESF vocabulary
```

`as8-reconcile-deviations.py`'s standing findings compound this: **7 deviations closed in the record
are named by no test, and 44 are named in no decision record or archive at all.**

> **OD-3 ruling requested:** the C10.7 owner-coverage rule ranges over **`KNOWN-DEVIATIONS.md`'s
> live-heading set, plus any `DEV-NNN` that appears in `COMPILER-STATE.md` but not in that file**.
> Package/application-track deviations (DEV-165 and its kin) are in scope **only for the
> distribution/release-weakness register (§11.2, class C)** and are named in the release statement's
> exclusions, not treated as compiler-conformance deviations.

Without this ruling C10-0 cannot produce a countable inherited-deviation inventory, and C10.7 cannot
be executed at all.

### OWNER RULING — APPROVED WITH ONE REFINEMENT (2026-08-09)

**Do not force three conceptually different populations into one "deviation" denominator.** C10-0
freezes **three separately countable inventories**:

```text
A. COMPILER DEVIATIONS
     KNOWN-DEVIATIONS.md live-heading set
   + DEV-* present in COMPILER-STATE.md but absent from that file

B. RELEASE / DISTRIBUTION DEVIATIONS
     DEV-165 and its kin
   + installer / offline-build / authenticity / package-distribution limits (§1.6)

C. ASSURANCE RESIDUALS
     live AS8-R*
   + pending AS8-DA*
   + EI / RA residuals inherited into C10 (incl. RA-LAYOUT unmeasured, RA-LINTS)
   + any C10-R* this campaign allocates
```

**All three populations require an owner and a disposition before C10-Q.** They differ in what they
constrain:

```text
A   is the denominator for the compiler-conformance deviation rule (CD-021)
B   constrains the RELEASE / DISTRIBUTION WORDING — it does not gate compiler conformance
C   constrains the STRENGTH OF EVIDENCE CLAIMS — it does not assert a defect exists
```

> **This preserves the semantic distinction AS8 worked to establish: an evidence residual is not
> automatically a compiler defect.** A mutation survivor means the evidence cannot detect a wrong
> rule, not that the rule is wrong. Collapsing C into A would silently convert sixteen
> "we cannot see this" findings into sixteen "we are broken here" findings, and would have
> allocated DEV numbers the AS8 owner ruling explicitly refused.

**Consequences propagated through this plan:** §11.2's finding classes are aligned to these three
populations plus security (class B in that section is `SEC-C10-*`, which is a fourth register and
is not a deviation at all); §17.1's exit criterion E9 is restated per population; §6.3's opening
inventory freezes all three, not one.

## OD-4 — Are DEV-012 and DEV-213 repaired inside C10, before it, or carried?

Analysis and a recommendation are in §5.3. The ruling itself is the owner's. State it as one of:
`bounded pre-C10 repair packet` / `early C10 prerequisite packet` / `carried as a named release
deviation`. **C8 is not reopened under any of the three** — a DEV-213 repair is a new bounded LSP
correctness packet, which the AS8 record already anticipates ("fixed in the next bounded LSP
correctness packet").

### OWNER RULING — MODIFIED (2026-08-09). The plan's recommendation to carry both is REJECTED.

The plan recommended carrying both as named release deviations. **The owner rules otherwise, and
the reason is OD-2:** because C10 explicitly evaluates **Core v1 Compiler Stable**, both should be
*closed* rather than deliberately carried through C10-Q. The plan's own analysis is what makes this
cheap — DEV-012 is essentially one editor session, and DEV-213 is a **demonstrated HEAD correctness
defect**, not merely missing evidence.

```text
DEV-012   does not block C10-0
          execute interactive validation EARLY during C10
          no reopening of C8
          all seven features pass   -> CLOSE DEV-012
          any feature fails         -> allocate/use a bounded DEV, decide from the evidence

DEV-213   does not block C10-0
          bounded LSP correctness repair EARLY during C10
          no reopening of C8
          MUST be closed before an UNQUALIFIED Core v1 Compiler Stable
          language-services claim is authorised
```

**Neither blocks opening. Both gate the claim.** A new checkpoint is inserted before C10-Q:

```text
Core v1 Compiler Stable candidate
        |
        v
DEV-012 CLOSED  OR  an explicit, narrower language-services claim
DEV-213 CLOSED  OR  an explicit, narrower workspace-symbol claim
        |
        v
C10-Q
```

**The preferred route is to close them, not to weaken the release statement.** The narrower claim
is the fallback, not the plan.

**Scope discipline this ruling does not relax.** The DEV-213 repair is a bounded LSP
cache-ownership correction — the existing test
`as8_editing_one_file_leaves_other_uris_cached_analyses_stale` is written so a repair **flips its
polarity rather than deleting it**, and that is the repair's acceptance criterion. It is not
licence to redesign the LSP, and it does not reopen C8. DEV-012's validation is **MANUAL evidence**
under Charter §5.2 and must be disclosed as such; it must not be described as automated coverage.

**Both are scheduled as C10-P (§6.2), a bounded prerequisite packet running alongside C10-A1.**

## OD-5 — Two documents at HEAD contradict a third about AS8's coverage baseline

Found during this plan's opening inspection. **Not repaired here**, because repairing it means
editing historical records:

```text
AS8-MUTATION-FINDINGS.md:424       "AS8-R15 DISCHARGED. The full-corpus coverage run COMPLETED
                                    (83.05% regions, 84.92% functions, ...)"
AS8-COVERAGE-BASELINE.md           publishes the full-corpus headline 83.05% and RETIRES the
                                    --lib-derived correlation claim it had made
CAMPAIGN-B-EXIT-REPORT.md §5       "R3 and R15 discharged"  — agrees

AS8-EXIT-QUALIFICATION.md §5       "AS8-R15  the full-corpus coverage baseline was attempted and
                                    stopped on a disk floor; the published baseline is `--lib`
                                    only and says so"                              ← STALE
COMPILER-STATE.md CD-394 evidence  "coverage baseline published as `--lib` only and labelled as
                                    such"                                          ← STALE
```

Both stale statements were true when written and were overtaken by commit `0bc9aee` ("Owner rulings
applied; full-corpus coverage completed and RETIRED a claim this packet made").

> **OD-5 ruling requested:** add a **dated superseding note** to `COMPILER-STATE.md` (new CD entry)
> recording that CD-394's coverage-evidence line and AS8-EXIT-QUALIFICATION §5's AS8-R15 row were
> overtaken, and that `AS8-COVERAGE-BASELINE.md` is the live figure. **Do not edit CD-394 or the AS8
> exit qualification.** This is the documentation discipline C10 will be held to (§16); applying it
> to C10's own opening finding is the cheapest possible demonstration that it works.

### OWNER RULING — APPROVED (2026-08-09)

Add a dated superseding record stating that the later full-corpus run **completed** and that
`AS8-COVERAGE-BASELINE.md` carries the live figure. **Do not modify CD-394 or the historical AS8
exit qualification — they were correct when written.**

The corrected current state:

```text
full corpus      regions    83.05%
                 functions  84.92%
                 lines      83.64%

AS8-R15          DISCHARGED

branch coverage  unavailable from this toolchain / NOT CLAIMED
```

The owner records this as *"exactly the right prospective-correction discipline"*, and §16.1
generalises it: C10 corrects forward, never backward.

## OD-6 — `ROADMAP.md` §0.1's compiler/stabilisation row is stale

`ROADMAP.md` §0.1 states: *"Sprints 1 and 2 are complete; AS3 and AS4 remain"* and *"Campaign A is a
binding entry gate on §6"*. At HEAD all four sprints are CLOSED, Campaign A is PASS and Campaign B
has EXITED. `ROADMAP.md` is the live platform roadmap and this row is now wrong.

> **OD-6 ruling requested:** authorise a **current-state correction** to `ROADMAP.md` §0.1's
> architecture-stabilisation row (programme COMPLETE; Campaign A gate SATISFIED, so §6 Phase 4's
> entry gate is met) as part of C10-0. This is a summary line, not a historical decision — §16's
> rule permits updating current-state summaries in place. `ROADMAP.md` §6.0's binding gate text
> itself is a decision record and is **not** rewritten; it gains a dated "gate satisfied" note.

### OWNER RULING — APPROVED (2026-08-09)

`ROADMAP.md` is a **live current-state roadmap**, so its stale summary is corrected. The corrected
state:

```text
architecture stabilisation    COMPLETE
Campaign A                    PASS
Campaign B                    PASS
structured-concurrency gate   SATISFIED
```

**Do not remove the historical gate requirement from §6.0.** Add the fact that it has subsequently
been satisfied. The requirement and its satisfaction are two records, not one edited record.

---

# 3. Scope, governing principle, and non-goals

## 3.1 What C10 is

> **C10 determines what the existing compiler can legitimately claim.**

It is a qualification campaign. Its output is a release statement derived from evidence, plus the
evidence and the named residue. It is **not** an improvement programme that happens to end in a
release.

## 3.2 Non-goals — binding

C10 must not become any of these. Each is refused by citing this section plus the charter clause
named:

| Forbidden | Cite |
| --- | --- |
| compiler redesign | Charter §1.10 "avoid broad refactors not required by the active WP" |
| broad refactoring programme | as above; and WP-ARCHITECTURE-STABILIZATION is COMPLETE |
| new language-feature campaign | Charter §1.6 rule 4, §1.7 not-Core list, §2.2 |
| optimisation campaign | Charter §1.6 rule 7, §6 not-yet list; and WP-C10.6's own rule |
| provider-generalisation campaign | Charter §1.3, §1.6 rule 19, CE7; roadmap §4.4 |
| cleanup sweep | this section |
| reopening C8 | CD-385 closed it; a DEV-213 repair is a new packet, not a C8 reopening |
| reopening the tensor track | Gate 7 productisation DEFER stands; Charter §1.5 |
| moving the external sample-suite pin to make a result | §14.4 |

**Adding a test is not a scope violation. Changing what the compiler accepts or rejects is.**

## 3.3 Defect-handling rule

If qualification demonstrates an actual defect at HEAD:

```text
demonstrate the defect (a failing or wrongly-passing test at HEAD, not an argument)
  -> allocate or reuse a DEV number
  -> perform the SMALLEST bounded repair
  -> rerun the affected qualification evidence, and only that (see §15.2)
  -> return to the C10 packet that found it
```

If a finding does **not** invalidate the release claim under test:

```text
record it as a deviation or residual
  -> assign an owner and a disposition
  -> continue qualification
```

**Do not fix every weakness C10 discovers.** The AS8 precedent is binding: 16 mutation survivors
produced **zero** DEV numbers, because a survivor means the evidence cannot detect a defect, not
that the defect is present. C10 inherits that distinction exactly.

## 3.4 The AS8 lesson, as a binding C10 rule

Campaign B measured:

```text
39 compiler-source mutation trials
26 predictions CONFIRMED
13 predictions FALSIFIED, IN BOTH DIRECTIONS
```

Three of the four evidence documents AS8 consumed were **wrong about what the test tree can
detect**, and the errors were methodological, not careless: the audit was conducted by reading the
differential machinery rather than enumerating the corpus, so every control it missed was a
front-end test no differential suite runs, and every over-claim was a suite that runs but cannot
disagree.

> **C10 rule:** a material evidence claim is **not** established merely because its suite passes.
> Where practical, demonstrate that the claimed evidence **fails when the underlying rule is
> deliberately made wrong**.

Applied per §8. **Not applied mechanically to every trivial assertion** — §8.2 fixes the population
in advance.

---

# 4. Inherited evidence — the exact inputs C10 consumes

Every path below exists at `f12ecec`. A C10 packet citing anything else must add it here first.

## 4.1 Governance and position

```text
STARKLANG/docs/compiler/COMPILER-CHARTER.md
STARKLANG/docs/compiler/COMPILER-ROADMAP.md
COMPILER-STATE.md                                       (repo root)
ROADMAP.md                                              (repo root; platform, §0 authority bound)
STARKLANG/docs/compiler/work-packages/WP-ARCHITECTURE-STABILIZATION.md
```

## 4.2 Campaign B evidence base — the assurance inheritance

```text
audits/CAMPAIGN-B-EXIT-REPORT.md          the PASS, and §4's four explicit NON-claims
audits/AS8-EXIT-QUALIFICATION.md          five criteria, the 39-trial table  (see OD-5: §5 stale)
AS8-MUTATION-FINDINGS.md                  full analysis, Findings 1-8, AS8-R1..R15
AS8-DUPLICATE-AUTHORITIES.md              AS8-DA-001..006 + owner dispositions
AS8-COVERAGE-BASELINE.md                  full corpus 83.05% regions / 84.92% fn / 83.64% lines;
                                          --lib 46.69%/58.00%/48.34%; NO target, by work-item
ENGINE-SHARED-FATE-REGISTER.md            EI0 frozen vocabulary; 11 ESF entries
ENGINE-EVIDENCE-INDEPENDENCE.md           EI2 audit table; the 6 required questions
ENGINE-RISK-PROFILES.md                   EI4; per-engine roles; the corrected ranking
ENGINE-MUTATION-TARGETS.md                EI5 ranked targets
ENGINE-PUBLIC-CLAIM-CALIBRATION.md        EI6 — the approved public wording
RUSTC-ASSUMPTION-INVENTORY.md             EI3 — RA-* assumptions incl. RA-LAYOUT unmeasured
engine-shared-fate.json                   the machine-readable register (authoritative over the
                                          prose file's ESF-PROV-001 row)
starkc/scripts/as8-mutate.py              the mutation harness (686 lines)
starkc/scripts/as8-control-census.py      corpus census, keyed on normative rule IDs
starkc/scripts/as8-duplicate-authorities.py
starkc/scripts/as8-reconcile-deviations.py
```

## 4.3 Conformance and execution evidence

```text
STARKLANG/conformance/core-v1-coverage.toml         legacy, 59 rules, DEV-017 precision limit
STARKLANG/conformance/core-v1-rule-id-map.toml      59 legacy -> 161 granular
STARKLANG/conformance/core-v1-c2.11-evidence.toml   36 granular rules at fn precision
semantic-freeze/CORE-V1-COMPLETENESS.md             168 granular IDs (was cited as 161 until
                                                    C10-A1 measured it), the inventory of record
STARKLANG/tests/spec-fixtures/manifest.toml         117 fixtures, hand-triaged from the spec
starkc/tests/c6-corpus/                             89 cases, manifest+lock+generator, 12
                                                    metamorphic families, mutation controls
starkc/tests/exec_snapshots/                        the inherited frozen execution corpus v1.4.0
                                                    (a SEPARATE artifact; neither lock is valid
                                                    for the other tree)
starkc/scripts/check-conformance.py                 coverage-db internal consistency
starkc/scripts/generate-conformance-report.py       per-rule evidence report, JSON/Markdown
work-packages/C6-*.md                               ownership/reference/generics/drop/platform
                                                    /corpus-coverage matrices
```

## 4.4 Robustness, security, performance and platform inputs

```text
starkc/tests/robustness.rs                  deterministic fixed-seed pseudo-fuzz, lexer+parser,
                                            9 test fns, both ParseModes — the ONLY fuzz-shaped
                                            asset in the tree
starkc/tests/adversarial_*.rs (7 files)     boundaries, hash bounds, integer semantics, patterns,
                                            stderr, trait impls, accepted-surface audit
starkc/tests/as4_hostile_combinations.rs
starkc/tests/resource_exhaustion.rs
STARKLANG/docs/http-client/HC13-THREAT-MODEL.md     the METHOD C10.4 should copy: every defence
                                                   names the test that would fail if the defence
                                                   were removed
STARKLANG/docs/http-client/HC13-RELEASE-CHECKLIST.md   the distribution-weakness inheritance
starkc/benchmarks/c7-workloads/FROZEN.json  7 workloads, per-file SHA-256 + workload_hash,
                                            frozen at 4650d47, rustc 1.93.0, aarch64-apple-darwin
starkc/benchmarks/c7-workloads/c75-report-macos-arm64.json   ONE platform only
starkc/scripts/c7-baseline.py               --measure (STARK vs cargo/rustc split, measured not
                                            assumed) and --reproduce (two distinct abs paths)
starkc/target-matrix.json + scripts/target_matrix.py
work-packages/C6-PLATFORM-MATRIX.md         25 rows; 1-23 MET, 24 PASS, 25 REPORT-ONLY (Windows)
.github/workflows/ci.yml                    22 jobs; the `ci-complete` required check
.github/workflows/c78-native-capabilities.yml
starkc/scripts/build-release.py + test_build_release.py
```

## 4.5 What Campaign B explicitly does NOT hand C10

Reproduced verbatim from `CAMPAIGN-B-EXIT-REPORT.md` §4, because C10 is where each of these is
either established or explicitly declined:

```text
NOT a stability claim        the campaign's subject is architecture and assurance
NOT a conformance claim      C10 is where conformance is qualified; nothing there substitutes
NOT "three independent       one front end, three execution strategies, a named reference
    implementations"         engine (the HIR oracle), and six INVISIBLE authorities
NOT tensor-track progress    AS6 quarantined the extension; it did not advance it
```

---

# 5. Inherited residuals and deviations — classification

## 5.1 The classification each carried item must receive

Every item in §5.3 and §5.4 gets exactly these five answers, recorded in C10-0's inventory:

```text
blocks opening?                     yes / no
blocks a particular release claim?  which claim, precisely
must repair during C10?             yes / no
may remain as explicit deviation?   yes / no
outside C10?                        yes (name the owning track) / no
```

**Do not assume every carried item blocks C10.** Most do not.

## 5.2 How the inherited inventories are produced (not by reading the file)

Per **OD-3**, C10-0 freezes **three separately countable populations** — compiler deviations (A),
release/distribution deviations (B), assurance residuals (C). Only A is the denominator for CD-021's
compiler-conformance rule. All three need an owner and a disposition before C10-Q.

Population A is produced as follows.

`KNOWN-DEVIATIONS.md` is **append-only**: a deviation gets a new heading each time it is touched, so
**the first heading is not its status — the last one is.** DEV-121 opens "OPEN" and is CLOSED 3,558
lines later. 186 headings, 178 distinct ids.

C10-0 therefore produces the inventory mechanically:

```bash
python3 starkc/scripts/as8-reconcile-deviations.py        # regenerate the reconciliation
```

then hand-audits its two standing short lists (**7** closed-in-record/named-by-no-test, **44**
named in no decision record or archive), and only then applies §5.1 per live-heading deviation.
Reading the file top-to-bottom and believing the first heading is the failure this step exists to
prevent.

## 5.3 DEV-012 and DEV-213 — the two the brief singles out

### DEV-012 — interactive editor validation, seven features

**What it is.** C8 closed with interactive VS Code validation recorded for **3 of 10** advertised
features (hover, go-to-definition, find-references, confirmed by the owner on VS Code 1.130.0,
extension `starklang.stark-language@0.2.0`, macOS 26.5.2 arm64, 2026-07-31). Diagnostics,
formatting, completion, signature help, rename, document symbols and semantic tokens are
**protocol-tested only**. GATE-C8-CLOSURE §2a item 8 is an **explicit owner override**, labelled
"deliberately closed short", not "met".

**Assessment.** This is validation debt, not an implementation gap: all ten features share the same
analysis path and the same protocol layer, and C8's blocking reason was recorded as *environmental*
("no `code` CLI / Extension Development Host has been available"). The cost is one person-session
in an editor.

But GATE-C8-CLOSURE §4 adds a limit C10 must respect: **protocol validation checked verdicts, not
values, and a value defect survived it** — DEV-182, where the LSP JSON parser decoded every escaped
non-BMP character to the empty string, and both parse and response *succeeded*. So "protocol-tested"
is weaker evidence than it sounds, and a release claim over the seven features rests on it.

**RULED classification (OD-4, owner, 2026-08-09).** The plan proposed carrying this; the owner
ruled to close it.

```text
blocks opening?                     NO
blocks a release claim?             YES — any claim that language services are "validated"
                                    beyond the three navigation queries
must repair during C10?             YES — interactive validation runs EARLY in C10, as C10-P
may remain as explicit deviation?   ONLY as the fallback if validation is not obtained
outside C10?                        NO — C8 is not reopened
```

**Execution.** One interactive validation session over the seven protocol-only features, in the
recorded VS Code environment. **All seven pass -> CLOSE DEV-012.** Any feature fails -> allocate or
reuse a bounded DEV and decide from the evidence, not from the schedule.

It is **MANUAL evidence** under Charter §5.2 and must be disclosed as such — never described as
automated coverage. If the session cannot be obtained, the fallback stands: C10-Q states the
language-services claim as three interactively-confirmed navigation queries plus seven
protocol-conformant features, with DEV-012 named. **The fallback is not the plan.**

### DEV-213 — LSP multi-file workspace-symbol staleness

**What it is.** A demonstrated HEAD correctness defect. `ServerState::compilation_cache` is keyed by
URI and each value owns a whole-package `ProjectAnalysis`; `update_document` invalidates only the
edited URI; `handle_workspace_symbol` merges symbols from every cached analysis. Rename a symbol in
one open file of a package and `workspace/symbol` returns **both the new name and the name that no
longer exists**. Demonstrated by a **passing** test at HEAD:
`as8_editing_one_file_leaves_other_uris_cached_analyses_stale`, written so a repair flips its
polarity rather than deleting it.

**Scope.** Editor surface only. `stark build` and all three engines are unaffected — none uses this
cache. Owner-ruled non-blocking for Sprint 4; standing qualification recorded: *any claim that
`workspace/symbol` is correct under multi-file editing must be stated as qualified.*

**Recommended classification** (owner ruling required, OD-4):

```text
blocks opening?                     NO
blocks a release claim?             YES — precisely one: workspace/symbol correctness under
                                    multi-file editing
must repair during C10?             YES — bounded LSP correctness repair, EARLY in C10, as C10-P
may remain as explicit deviation?   ONLY as the fallback if the repair is not obtained
outside C10?                        NO — C8 is not reopened; this is a new bounded packet, which
                                    the AS8 record already anticipated
```

**RULED (OD-4, owner, 2026-08-09).** The plan proposed carrying this; the owner ruled to repair it,
because OD-2 evaluates Core v1 Compiler Stable and DEV-213 is a **demonstrated HEAD correctness
defect**, not merely missing evidence.

**Acceptance criterion, and the boundary of the repair.** The existing test
`as8_editing_one_file_leaves_other_uris_cached_analyses_stale` was deliberately written so that a
correct repair **flips its polarity rather than deleting it** — its own message says so. That flip
is the repair's acceptance criterion. The repair is a bounded cache-ownership correction: one
analysis per *package*, or invalidation across every URI of the affected package. It is **not**
licence to redesign the LSP, add incremental analysis, or revisit C8's scope.

**MUST be closed before an unqualified Core v1 Compiler Stable language-services claim is
authorised.** If it is not closed, C10-Q states the workspace-symbol claim narrowly, carrying the
standing qualification verbatim: *the LSP answers correctly for a single open file and for a freshly
opened one; the combination of several open URIs of one package and an edit to any of them is
wrong.* **The narrow claim is the fallback, not the plan.**

## 5.4 The remaining carried items

Classifications below are **recommendations** except where a row cites an owner ruling. Per OD-3
every row also carries a population letter: **A** compiler deviation, **B** release/distribution
deviation, **C** assurance residual.

| Item | Classification | Note |
| --- | --- | --- |
| **C9 Part B** | blocks opening: **NO — RULED, OD-1**. Blocks the *artifact-generality* claim only. Outside C10. | Roadmap §4.5 permits "blocked" as an explicit status |
| **AS8-R1, R4, R6, R8, R9, R13, R14** | **population C.** Evidence gaps, not defects. Feed §8's mutation population as **candidate targets**; each may become a C10-D control-construction item **only if** the claim it undermines is one C10 intends to make | No DEVs, per the AS8 owner ruling |
| **AS8-R2** (`ESF-TRAP-001a`, no control constructible) | **permanent residual.** Named in the release statement | MUT-008 is the honest no-op marking the boundary |
| **AS8-R5, R10** (`EV-SPEC-FIXTURES` does not control TYPE-PRIM-001; `ESF-TRAIT-001` has no control of any kind) | **highest-value C10-D targets.** `traits.rs` is 82.77% covered and Core trait contracts can be declared arbitrarily wrong with every selected suite passing | The clearest coverage-is-not-conformance exhibit in the tree |
| **AS8-R3, R15** | DISCHARGED | R15's discharge is contradicted by two stale records — OD-5 |
| **AS8-R7** (13/39 falsified) | a **method** finding, not a residual to close. It is the reason §8 exists | |
| **AS8-R11, R12** | R11 corrected in the compiler's favour; R12 (`scalar_name` can drift silently) is settled by the DA-005 CONSOLIDATE ruling | |
| **AS8-DA-001, DA-005** | owner ruling: **CONSOLIDATE**, "after Sprint 4". Sprint 4 is closed, so these are now schedulable | **Not C10 work** — they change compiler source. Route to a bounded post-C10 packet, or seek explicit owner authorisation to include. Default: outside C10 |
| **AS8-DA-002/003/004** | owner ruling: **REMAIN SEPARATE**, and "after Sprint 4, add an exhaustive parity/drift test over the closed `RuntimeFn` set" | The parity test is **test-only** and therefore permissible inside C10-D as a control-construction item. Recommended: do it, because C10 will otherwise make a backend-parity claim over three duplicated tables with no cross-check |
| **AS8-DA-006** | KEEP — the positive exemplar. No action | |
| **Branch coverage** | **unavailable from this toolchain**; llvm-cov reports region and line only, and the "branches" column is empty. Not fabricated, not claimed | C10 must not state a branch-coverage number. Ever |
| **Full-corpus coverage** | 83.05% regions / 84.92% functions / 83.64% lines, published with **no target**, per AS8's work item | C10 must not convert this into a threshold (§7.3) |
| **`RA-LAYOUT` unmeasured** (EI3) | rustc-assumption residual. In scope for C10-D only if a release claim depends on generated-struct layout | |
| **`RA-LINTS`** — two deny-by-default lints suppressed in generated code | narrows what rustc refuses, i.e. narrows the strength of "rustc is a genuine external control" | **Must be named in C10.1's shared-fate column** for every row whose independent control is rustc |
| **DEV-165** and the four distribution weaknesses | **population B.** Named as exclusions in the release statement; constrains release/distribution WORDING only | OD-3 RULED: B is counted separately and is not the CD-021 denominator |
| **DEV-017** (coverage-db precision) | **population C.** The reason 85 of 168 granular rules are cited only through an aggregate runner (C10-A1). **C10.1's central problem, not a side item** | §6.2; `audits/C10-A1-EVIDENCE-CENSUS.md` |

---

# 6. Packet decomposition and sequencing

## 6.1 The decomposition, and why it differs from the brief's sketch

The brief proposes C10-0, A, B, C, D, E, F, Q. This plan adopts that spine with **two changes**,
both justified:

1. **C10-A is split into A1 (evidence-integrity census) and A2 (dashboard construction).** The
   161-vs-36 gap means the dashboard's *inputs* have to be established before its *rows* can be
   written, and A1's output determines whether C10.1 is a 36-row precise dashboard with 125
   explicitly-unclassified rules, or a larger dashboard built by attributing the aggregate fixture
   runner (which would be invented attribution — forbidden). Merging them invites writing rows
   first and discovering the gap second, which is precisely how EI2 went wrong.

2. **C10-D runs mutations against claims C10-A2 has already drafted**, not against an
   independently-chosen target list. AS8's own §6 finding is that three of its five worst moments
   were **test-selection** failures — the control existed and was not selected, was in another crate
   and unrunnable, or was structurally incapable. Selecting mutation targets from the drafted claim
   set makes the selection auditable: every mutation is aimed at a sentence someone intends to
   publish.

Everything else follows the brief.

## 6.2 Packet table

| Packet | Roadmap WP | Purpose | Entry condition | Exit evidence |
| --- | --- | --- | --- | --- |
| **C10-0** | — | Opening inventory and qualification freeze | OD-1..OD-6 ruled; §1.2's CI run completed and read | `C10-0-OPENING-INVENTORY.md` (§6.3) |
| **C10-P** | — (OD-4) | Bounded prerequisite repairs: DEV-213 LSP cache ownership; DEV-012 interactive validation | C10-0 frozen | flipped-polarity test at HEAD; `C10-P-LANGUAGE-SERVICES.md` recording the MANUAL session |
| **C10-A1** | C10.1 | Conformance-evidence integrity census | C10-0 frozen | `C10-A1-EVIDENCE-CENSUS.md` |
| **C10-A2** | C10.1 | The dashboard | A1 complete; denominators declared | `C10-CONFORMANCE-DASHBOARD.md` + generated JSON |
| **C10-B** | C10.2 | Robustness and fuzzing | fuzz target population declared (§9.2) | `C10-B-ROBUSTNESS.md` + corpora + regression fixtures |
| **C10-C** | C10.4 | Security review | threat model / surface inventory FROZEN (§11.1) | `C10-C-SECURITY-REVIEW.md` + `C10-THREAT-MODEL.md` |
| **C10-D** | C10.3 | Differential, metamorphic, and selected mutations | A2 draft claims exist; relations declared (§10.2) | `C10-D-DIFFERENTIAL.md` + `C10-MUTATION-LEDGER.md` |
| **C10-E** | C10.6 | Performance baselines | workload set frozen (§12.1) | `C10-E-PERFORMANCE-BASELINE.md` + per-platform JSON |
| **C10-F** | C10.5 | Compatibility and version policy | A2, B, C, D, E complete enough to bound the promises | `C10-F-COMPATIBILITY-POLICY.md` |
| **C10-Q** | C10.7 | Exact-head release qualification and decision | all of the above; sweep §16.2 clean | `GATE-C10-CLOSURE.md` + `C10-RELEASE-STATEMENT.md` |

**Ordering is a partial order, not a chain.** B, C and E are independent of each other and of D;
they may run in any order or concurrently, subject to the WIP limits in `ROADMAP.md` §2.2. C10-P
runs alongside C10-A1 — it touches the LSP and the census touches neither. The hard edges are:

```text
C10-0  ->  everything
C10-A1 ->  C10-A2  ->  C10-D  ->  C10-F  ->  C10-Q
C10-B, C10-C, C10-E  ->  C10-F  (each contributes bounded promises)
C10-P                ->  C10-G  (the language-services gate)
every packet         ->  C10-Q
```

### C10-G — the language-services gate (OD-4), immediately before C10-Q

Not a packet; a **gate C10-Q may not pass without evaluating**:

```text
Core v1 Compiler Stable candidate
        |
        v
DEV-012 CLOSED  OR  an explicit, narrower language-services claim
DEV-213 CLOSED  OR  an explicit, narrower workspace-symbol claim
        |
        v
C10-Q
```

Either branch is a legitimate exit. What is **not** legitimate is reaching C10-Q with an unqualified
language-services claim and neither DEV closed.

C10-F is last-but-one deliberately: a compatibility promise made before the evidence exists is the
"accidental permanent promise" the roadmap warns about.

## 6.3 C10-0 — Opening Inventory and Qualification Freeze

**No substantive qualification work begins until this exists and its contradictions are resolved.**

`C10-0-OPENING-INVENTORY.md` must freeze, each with the command or file that produced it:

```text
1   exact develop HEAD (SHA), and the CI run id(s) read for it, with per-job conclusions
2   compiler/toolchain versions: starkc crate version, MSRV, the rustc used by CI and locally,
    MIR_VERSION, MIR_RUNTIME_SURFACE, layout contract, provider ABI version
3   supported platform matrix, read from target-matrix.json (not from prose), with the Tier-1 /
    Tier-2 split and what Tier-2 does NOT claim
4   release classes being evaluated (OD-2)
5   the THREE OD-3 populations, frozen and separately countable:
      A  compiler deviations      the reconciler output, hand-audited, each with §5.1's five
                                  answers. The CD-021 denominator
      B  release/distribution     DEV-165 + the four §1.6 weaknesses
      C  assurance residuals      live AS8-R*, pending AS8-DA*, EI/RA residuals, DEV-017
6   inherited mutation freshness: per §8.2a, the blob SHAs of every AS8 trial's authority file
    and killer files, at the trial commit and at the candidate head, with FRESH/STALE per trial
7   test/evidence inventory: every suite C10 will cite, by target name, with its Charter §5.2
    evidence class
8   mutation authority inventory inherited from AS8: the 11 ESF entries, the 6 DA entries, the
    39 prior trials and their verdicts — so C10 never re-runs a trial it already has
9   fuzz target population (§9.2), declared BEFORE any fuzzing
10  security surface (§11.1), frozen BEFORE any finding is reviewed
11  performance workload set (§12.1), frozen BEFORE any measurement
12  external pinned evidence: stark-samples repo + SHA (b3b28e7...), the C6.5 corpus lock and
    generator hashes, exec_snapshots corpus version
13  excluded scope: §3.2's list, restated as this campaign's refusals
14  expected artefacts from every C10 packet (§6.2's last column), by filename
```

**Forcing function.** C10-0 is not complete until a *fresh reader* can answer "what does STARK
claim today, and on what evidence" from this file alone. Test it by having the inventory name, for
each of the four release classes in OD-2, the evidence that class would require and whether that
evidence exists.

**Stop condition.** If §1.2's CI run for the frozen SHA does not complete green, C10-0 stops and
the failure is triaged before anything else. A qualification campaign opened on a red head has no
baseline.

## 6.4 C10-A1 — Conformance-evidence integrity census

**The question A1 answers:** for each normative rule C10 intends to make a claim about, does
executable evidence exist, at what precision, and can that evidence disagree with the
implementation?

Method — the one AS8 proved necessary, run in the order that makes selection auditable:

1. **Declare the population** (§7.2). **DONE — C10-0 declared it and C10-A1 corrected the count
   from 161 to 168 without changing the population.** The denominator is the **granular IDs** in
   `CORE-V1-COMPLETENESS.md`, because that is the inventory of record and the 59 legacy rules map
   onto it via `core-v1-rule-id-map.toml`. Declare it in C10-0, before measuring.
2. **Enumerate the corpus**, do not read the machinery. Use
   `starkc/scripts/as8-control-census.py`, which is keyed on normative rule IDs rather than
   function names and was built precisely because EI2's read-the-machinery method missed
   `c61f_structural_copy`.
3. **Classify each rule** into exactly one bucket:
   ```text
   PRECISE      positive and negative evidence at test-function precision (the 36, plus any
                A1 adds)
   AGGREGATE    cited only through starkc/tests/conformance.rs or a file-level `tests` entry
   ABSENT       checked and confirmed to have no evidence (an empty array, not an omission)
   N/A          intentionally-deferred / spec-defect / not-in-Core
   ```
4. **Do not promote AGGREGATE to PRECISE by inspection.** Promoting a rule requires naming the
   test function; if that means writing the test, that is a C10-A1 work item and is permitted
   (adding a test is not a scope violation) — but only for rules whose claim C10 intends to make.
5. **Flag every rule whose only evidence is `CROSS_ENGINE_DERIVED`** with no independent control,
   cross-referenced to the ESF register. Those rules cannot be claimed on differential agreement.

**Forcing function.** A1's output must include the count in each bucket **and** the list of rules
that moved bucket during A1 with the reason.

> **No expected finding count.** A1 may legitimately conclude **`0 previous classifications
> changed`**. Zero and twenty are equally valid results. What must be demonstrated is not that
> corrections were found but that the census **actually enumerated the intended population** — so
> the forcing mechanism is the enumeration, not the yield:
>
> ```text
> every rule in the declared population appears in the output exactly once
> every bucket assignment cites the file:function or the confirmed absence
> the enumerator is run twice and produces identical output (Charter §2.1 step 7)
> a deliberately mis-cited rule is injected and the census reports it as inconsistent
> ```
>
> That last line is the negative control. **A census with no negative control is EI2's error in a
> new costume** — and note precisely what AS8's lesson was and was not. It was **not** "one third of
> audits are wrong". It was: **do not infer evidence strength from reading the machinery.** An audit
> that enumerates correctly and finds nothing has done its job.

**Stop condition.** If ABSENT + AGGREGATE together exceed the population C10 can honestly claim
over, A1 stops and escalates the **claim scope** to the owner (CE8) rather than widening the
evidence.

## 6.5 C10-A2 — the dashboard

`C10-CONFORMANCE-DASHBOARD.md`, generated from a machine-readable source
(`STARKLANG/conformance/c10-dashboard.json`), with **one row per rule in the declared population**
and these columns:

```text
normative rule ID            granular ID; the legacy ID it maps from
normative home               exact spec file + section (from CORE-V1-COMPLETENESS.md)
implementation location      file(s); function where it is a single authority
positive evidence            test function, or AGGREGATE/ABSENT
negative evidence            test function, or AGGREGATE/ABSENT/N-A-with-reason
engines/configurations       which of {HIR, MIR, native-debug, native-release, front-end-only,
                             verifier} actually exercise it
evidence class               Charter §5.2 vocabulary: SPEC/UNIT/CONF/NEG/REG/PROP/FUZZ/DIFF/
                             PERF/EXT/MANUAL
shared-fate authority        ESF-* id if the rule sits behind one, with its visibility
independent control          the control if one exists; `none` if not; NEVER blank
mutation/challenge status    trial id + verdict, or NOT-CHALLENGED
open deviation/residual      DEV-*/AS8-R*/AS8-DA-*
last verified               commit SHA + toolchain + platform(s)
```

**Rules the dashboard must obey:**

- **No glossy percentage.** A single headline number that mixes PRECISE and AGGREGATE rows is
  forbidden. If a summary figure is published it must be per-bucket, and the AGGREGATE bucket must
  be reported as *unclassified*, never as covered. `generate-conformance-report.py` already makes
  this distinction and must not be undone.
- **`independent control: none` is a legitimate, expected value.** Six of eleven ESF authorities
  are INVISIBLE to all three engines. A dashboard with no `none` values has been massaged.
- **Coverage percentages do not appear in this dashboard at all.** They live in C10-E's neighbour
  document if anywhere. Coverage is not conformance: `typecheck/traits.rs` is 82.77% covered and
  `ESF-TRAIT-001` has no control of any kind.
- **The row's `engines` column must not say "three independent implementations".** EI6's calibrated
  wording is the only permitted phrasing (§10.1).

## 6.6 C10-B — robustness and fuzzing → §9

## 6.7 C10-C — security review → §11

## 6.8 C10-D — differential, metamorphic and selected mutations → §8, §10

## 6.9 C10-E — performance baselines → §12

## 6.10 C10-F — compatibility and version policy → §13

## 6.11 C10-Q — exact-head release qualification → §17

---

# 7. Measurement rule — denominators declared before measurement

## 7.1 The rule

> For every metric, the **denominator or target population is declared in writing before the
> measurement is taken**, and no denominator may be chosen after seeing the result.

Applies to: conformance counts, coverage, fuzz targets, mutation populations, security surfaces,
platform matrix, performance workloads, malformed-input corpora, differential cases.

## 7.2 How a denominator is declared

In C10-0 (or the packet's own opening section, before its first run), record:

```text
metric            what is being counted
population        the exact set, with the file/command that enumerates it
exclusions        what is out, and WHY — one line each, no bulk exclusions
frozen at         commit SHA
changed by        any later change to the population is a new dated line, never an edit
```

A population that changes mid-packet does not invalidate the packet; **silently changing it does.**

## 7.3 Standing prohibitions

```text
no branch-coverage number                  the toolchain does not report one (AS8, confirmed)
no coverage threshold or target            AS8's work item forbids it and MUT-003 shows why:
                                           copy_canon_matrix covers core_method_signature
                                           completely and controls nothing
no pooled mutation kill rate               EI5's rule, inherited: report by killer independence,
                                           not as a single percentage
no conformance percentage that mixes       §6.5
   PRECISE and AGGREGATE rows
no "N of M tests pass" as a conformance    581 tests across 5 targets was a SCOPED run at CD-394,
   claim                                   not the tree's test count. 210 integration targets exist
```

## 7.4 The line that must appear in every C10 evidence document

> **Coverage is not conformance. A line executing does not prove that anything would detect the
> line being semantically wrong.**

---

# 8. Mutation discipline

## 8.1 The harness

`starkc/scripts/as8-mutate.py` is reused, not rewritten. Its guarantees are the reason:

```text
declared prediction     each trial declares expect = KILLED | SURVIVED BEFORE it runs; the
                        harness reports CONFIRMED / UNEXPECTED against that declaration
clean target            it REFUSES to run on a dirty target (added after a broad `git add`
                        committed a live mutation to a pushed branch and failed every C6.5 job)
verified restore        the source file is ALWAYS restored — on interrupt, on build failure —
                        and the restore is verified, not assumed
kill provenance         it records WHY a mutation died, which is what revealed that in all three
                        AS8-DA pairs NEITHER copy was controlling the other
self-test               Batch 0 is a two-sided calibration (a real disturbance that must be
                        detected; a semantics-preserving edit that must NOT be) and `--batch 0`
                        refuses to be skipped silently
```

**C10 additions to the harness, permitted and expected:**

- **build scoping.** AS8's harness built all 209 test binaries per trial, relinking ~205 it would
  never run — that filled the disk to 99% and stretched one build to 32 minutes. C10 trials must
  build only the selected targets. Check `df` before a batch; use a scratch `CARGO_TARGET_DIR` that
  can be deleted (and note the C10-B/E interaction: an *exported* `CARGO_TARGET_DIR` breaks
  generated-crate builds — set it per-command).
- **selection recording.** Every trial records the exact `--test`/`--lib` selection. AS8's three
  worst moments were selection failures (MUT-005/006 the control existed and was not selected;
  MUT-016 the control was in another crate and unrunnable; MUT-013 the control was selected and
  structurally incapable). A trial whose selection is not recorded is not evidence.
- **`finally` integrity.** A killed batch once bypassed `finally` and left a mutation in the
  working tree. Every batch ends with `git status --porcelain` verified empty against the target
  files.

## 8.2 The mutation population — declared before running

**Mutation is applied to material semantic/release claims, not mechanically to every assertion.**
The population is:

> Every claim in the C10-A2 draft dashboard that C10 intends to publish as a **strong release
> claim**, whose supporting evidence has **not already been mutated in AS8's 39 trials**.

Which lands, by rule family:

```text
ownership / Copy / Drop          ESF-COPY-001/002, ESF-DROP-001/002 — partly done (AS8 batches
                                 1/1b/1c/2); AS8-R1 and AS8-R6 remain live gaps
trait contracts                  ESF-TRAIT-001 — AS8-R10: NO CONTROL OF ANY KIND. MUT-014/015
                                 declared Eq::eq by value and Ord::cmp returning Bool and both
                                 survived. HIGHEST-VALUE C10 target
resolver visibility              AS8-R13: non-`pub` re-export visibility has no control anywhere
                                 (MUT-035 and MUT-039 both survived). HIGH-VALUE
MIR verifier invariants          AS8-R14: may_need_drop's HostResource arm unguarded (MUT-037)
trap semantics                   ESF-TRAP-001a is a PERMANENT residual — no control constructible,
                                 and MUT-008 is the honest no-op that marks the boundary. Do NOT
                                 spend C10 time attempting one. ESF-TRAP-001b is controlled
provider / resource contracts    ESF-PROV-001 (mir/verify.rs IS an in-tree control, MUT-025);
                                 ESF-RES-001 (a_host_resource_is_never_copy, MUT-017)
backend parity                   AS8-DA-002/003/004: three duplicated RuntimeFn classifications
                                 with NO cross-check between them. The owner-ruled parity/drift
                                 test over the closed RuntimeFn set belongs here
type identity                    AS8-R5: EV-SPEC-FIXTURES does not control TYPE-PRIM-001; MUT-013
                                 survived with the fixtures selected
```

**Explicitly out of the mutation population:** anything whose claim C10 does not intend to publish;
anything already trialled in AS8 with a recorded verdict **that is still fresh under §8.2a**;
`ESF-TRAP-001a`.

## 8.2a Freshness rule for inherited AS8 mutation evidence (owner amendment, 2026-08-09)

Citing a prior trial instead of re-running it is only sound while the thing it measured has not
moved. **A mutation result is evidence about a specific compiler and a specific test tree, not a
permanent property of a rule.**

> **An inherited mutation result may be reused only if BOTH hold:**
>
> ```text
> 1  the SEMANTIC AUTHORITY targeted by the trial is unchanged since the recorded trial; and
> 2  the claimed KILLING / CONTROL evidence is unchanged in a way material to the trial
> ```
>
> **If either has changed:**
>
> ```text
> the prior mutation evidence is HISTORICAL
> rerun the trial against the C10 candidate head
> ```

**How it is enforced, cheaply.** For every inherited trial, C10-0 records the blob SHA of the
mutated source file and of each killing test file as they stood at the AS8 trial commit, then
compares against the C10 candidate head:

```bash
# per inherited trial: authority file + each killer file
git rev-parse <as8-trial-commit>:<path>     # recorded in the ledger
git rev-parse <c10-candidate-head>:<path>   # compared at C10-0 and again at C10-Q
```

Identical blobs -> **FRESH**, cite the trial. Differing blobs -> inspect the intervening commits; a
change that cannot be shown immaterial to the trial makes it **STALE**, and the trial re-runs.
`C10-MUTATION-LEDGER.md` carries a `freshness` column with `FRESH (blob match)` /
`FRESH (change reviewed, immaterial — reason)` / `STALE — rerun as C10-MUT-NNN`.

**Why this matters and is not bureaucracy.** Without it, C10 could cite `AS8-MUT-025` as evidence
that `mir/verify.rs` controls provider signatures long after that verifier has been substantially
rewritten — publishing a release claim backed by a measurement of a compiler that no longer exists.
The check is two `git rev-parse` calls per trial.

**Re-freshness is checked twice**: once at C10-0 when the inventory is frozen, and **again at C10-Q
against the final head**, because C10-P's repairs and any §3.3 defect repairs move source between
those two points.

## 8.3 Required record per trial

Every C10 trial records, in `C10-MUTATION-LEDGER.md`:

```text
authority/rule ID       ESF-*, AS8-DA-*, or the granular normative rule ID
target                  file + function + the exact find/replace
prediction              KILLED | SURVIVED, declared before the run
selected control        the exact test selection, verbatim
expected result         restatement of the prediction in the claim's terms
actual result           KILLED | SURVIVED
killer(s), if any       test names and the kill message
residual                if the prediction is falsified — a new AS8-style R-number in the C10-R
                        series, with what it means for the claim
restore verification    the harness's restore confirmation + `git status` clean
freshness               §8.2a — for an INHERITED AS8 trial: FRESH (blob match) /
                        FRESH (change reviewed, immaterial — reason) / STALE — rerun as
                        C10-MUT-NNN. For a trial run by C10 itself: `n/a — run at <head>`
```

**Ten fields. A trial missing any one of them is not evidence** — most often the missing one is
`selected control`, which is what made three of AS8's trials misleading until they were re-run.

**A falsified prediction is a finding, not an error.** Thirteen of thirty-nine were falsified in
AS8, in both directions, and that ratio is the strongest single argument that campaign produced.

> **No expected falsification rate.** C10 does not target, predict or benchmark against AS8's 13/39.
> A campaign in which **every** prediction is confirmed is a legitimate result, and so is one in
> which most are falsified. What must hold is the *mechanism*: predictions are declared before the
> run, the selection is recorded, the harness self-test passes in both directions, and no prediction
> is edited after the fact. **Do not tune predictions to look accurate, and do not tune them to look
> productively wrong either** — an expected-yield number corrupts a prediction in both directions.

## 8.4 What a survivor means, and does not

```text
SURVIVOR   the claimed evidence CANNOT DETECT a wrong rule.
           -> the CLAIM is weakened or withdrawn, or a control is constructed
           -> NO DEV is allocated (AS8 owner ruling: a survivor is not a defect)
KILL       the evidence can disagree. Record WHICH test killed it and whether that test is
           independent (HAND_AUTHORED / SPEC_DERIVED / EXTERNALLY_DERIVED) or correlated
           (CROSS_ENGINE_DERIVED / IMPLEMENTATION_GENERATED). A kill by a correlated control is
           weaker evidence than a kill by an independent one, and the ledger must say so
```

---

# 9. Fuzzing discipline (C10-B)

## 9.1 What exists at HEAD, and what does not

`starkc/tests/robustness.rs` is the **only** fuzz-shaped asset: a fixed-seed LCG driving random
character soup, random token soup, mutated fixtures and pathological nesting through
`parse(ParseMode::Program)` and `parse(ParseMode::Snippet)`, 9 test functions, on the stable
toolchain in ordinary CI. Its own header states the gate correctly: *"no panics, no hangs on
arbitrary input; grammar correctness is owned by the fixtures, not the fuzzer."*

**Nothing fuzzes the resolver, type checker, borrow checker, MIR verifier, artifact parsing, or the
LSP/diagnostic protocol.** `adversarial_*.rs` and `resource_exhaustion.rs` are hand-authored
adversarial cases, not generators.

**Constraint: stable Rust only** (Charter §1.10, rule 8). `cargo-fuzz`/libFuzzer requires nightly.
C10-B therefore extends the **deterministic seeded-generator** pattern already in the tree rather
than adopting a nightly fuzzer. This is a real limitation and must be stated in the release wording:
*bounded deterministic robustness testing*, not *fuzzing* in the libFuzzer sense.

## 9.2 Target population — declared in C10-0, before any run

```text
T1   lexer + parser                   robustness.rs, extended
T2   malformed-source corpus          truncation, encoding, BOM, mixed line endings, oversized
                                      identifiers (LEX-IDENT-002's 255 limit), deep nesting
T3   resolver / package / module      cyclic module graphs, cyclic package deps, missing entry,
       graphs                         duplicate module names, alias collisions, deep re-export
                                      chains, malformed starkpkg.json
T4   type checker                     generated ill-typed programs; deep generic instantiation;
                                      alias cycles; recursive types
T5   borrow checker                   generated ownership-hostile programs from the AS4 hostile
                                      combination shapes
T6   MIR verifier                     malformed MIR reached through generated source, plus
                                      direct verifier input where the API allows it
T7   malformed artifacts              ONNX inputs (truncated, wrong magic, oversized dims,
                                      hostile shape metadata); build.json / stark.lock /
                                      manifest.json / corpus.lock
T8   LSP / diagnostic / protocol      malformed JSON-RPC, wrong content-length, non-BMP escapes
       input                          (the DEV-182 shape), out-of-range positions, stale
                                      document versions, unknown methods
T9   hostile-input resource limits    time and memory bounds for every target above
```

**Declare per target, before running:** the generator, the seed(s), the case budget, and the
oracle (what counts as a failure). A target with no declared oracle is not runnable.

## 9.3 Qualification goals

```text
no panic                       including no `unreachable!()`, no arithmetic overflow panic in
                               the compiler itself, no unwrap on attacker-controlled input
no hang                        every target runs under a wall-clock bound; a timeout is a finding
bounded failure                a diagnostic or a clean error, never an unbounded allocation
deterministic outcome          the same seed produces the same diagnostics, byte for byte, where
   where required              the rule requires determinism (Charter §1.6 rule 16)
```

**Fuzzing does not need to prove random programs semantically meaningful.** A generated program
that is rejected is a pass, provided it is rejected the same way twice.

## 9.4 Seeds, corpora, minimisation and regressions

```text
seeds              fixed and recorded per target, in the packet document. A "random" seed is
                   not reproducible and is not evidence
reproducibility    every reported failure carries: target, seed, case index, and the minimised
                   input. Run deterministic commands twice (Charter §2.1 step 7)
corpus retention   generated corpora are NOT committed wholesale. Committed are: the generator,
                   its seed, and any MINIMISED case that found something
minimisation       shrink to the smallest input reproducing the behaviour before filing
regression policy  every minimised finding becomes a permanent fixture, following the C6.5
                   `retained/` precedent — a corpus whose machinery has never locked a real file
                   proves nothing about the machinery
lock discipline    if a case enters the C6.5 corpus, regenerate `corpus.lock` AND bump
                   `corpus_version` (minor = new cases). `c6_corpus_manifest.rs` enforces the
                   version line against a constant, so a regeneration cannot quietly redefine
                   the baseline
```

## 9.5 Forcing function

**The fuzz harness must be shown capable of finding something.** Before any "no panic, no hang"
claim is published, inject a deliberate panic into one target's path and demonstrate the harness
catches it, then verify restoration — the same two-sided calibration `as8-mutate.py --batch 0`
applies to mutations. An uncalibrated clean run is EI2's error in a new costume.

## 9.6 Stop condition

If a target produces findings faster than they can be triaged, **stop that target, record the
count, and escalate the claim** rather than fixing findings serially. C10 is not a hardening
campaign; §3.3's defect rule bounds repairs to those that invalidate a claim under test.

---

# 10. Differential and metamorphic discipline (C10-D)

## 10.1 The engine vocabulary — binding

**Never write "three independent implementations."** EI4 REPLACED that claim; EI6 calibrated the
public wording; Campaign B's exit report reaffirms it as an explicit non-claim. The permitted
phrasing is:

> **One front end and three execution strategies, differentially compared against a reference
> engine, over a shared semantic core.**

The distinctions C10-D must preserve in every artefact:

```text
HIR reference engine     src/interp.rs, 13,227 lines. THE ORACLE — the differential machinery
                         literally calls it that (`oracle_category`, "the oracle raised a trap
                         with no stated category"). Defines "expected". Not a shipping target.
                         NO HOST ACCESS AT ALL — providers/TCP/TLS/filesystem unexercised here
MIR interpreter          src/mir/interp.rs (2,870 lines) + supporting modules (16,097 total).
                         Proves the lowering executes and means what HIR meant. GATE, not product
native debug             generated Rust, debug profile
native release           generated Rust, release profile
verifier evidence        src/mir/verify.rs — a real in-tree control (MUT-025 proved it), and
                         NOT an engine. EI2 counted engines and missed this
shared front-end         one lexer, one resolver, one checker. Six ESF authorities are INVISIBLE
   authorities           to all three engines because the front end decides ONCE
```

And the hierarchy, which is why "peers" is the wrong word:

```text
HIR      defines expected behaviour     oracle
MIR      proves the lowering executes   gate
native   is the thing users run         product
```

## 10.2 Metamorphic relations — declared before running

**Only transformations the specification says are equivalent.** The existing 12 families
(M01–M12, `starkc/tests/c6-corpus/metamorphic.py`) already encode the two rules C10 inherits:

1. a transformation must be semantics-preserving **by a named normative rule**, not by intuition —
   each group records its precondition, and the preconditions are real (scope insertion only on
   bases with no `Drop` impl, because an extra block changes destruction timing under
   DROP-ORDER-001; arm reordering only on non-overlapping arms with no catch-all; loop-form
   equivalence only where ownership and Drop timing are identical);
2. **a transformation that changes nothing is a fake pair** — every transform asserts it actually
   rewrote the source.

C10-D's candidate additions, each requiring a named normative rule before implementation:

```text
formatter idempotence         format(format(x)) == format(x), and format(x) observes as x
                              — the formatter exists; this relation does not
harmless parenthesisation     only where 02-Syntax-Grammar's precedence makes it a no-op
equivalent import/path forms  only where 07-Modules-and-Packages says the forms are equivalent
declaration reordering        only for order-irrelevant items — NOT where Drop timing or
                              initialisation order is observable
debug/release equivalence     same observation across native debug and native release
repeated clean builds         reproducibility; c72_reproducibility.rs and c7-baseline.py
                              --reproduce (two DISTINCT absolute checkout paths) already exist
```

**A relation whose precondition cannot be stated as a normative rule reference is not added.**

## 10.3 What C10-D must not claim

- Agreement among engines that inherit the same authority is **not** corroboration. EI0's frozen
  rule: *a shared authority whose defects are INVISIBLE to differential comparison, and whose only
  supporting evidence is CROSS_ENGINE_DERIVED, is not independently evidenced — regardless of how
  many engines agree.*
- Provider behaviour has **two** engines, not three (the interpreters have no host access), and
  they share `mir::provider_sig`. `EV-PROVIDER-LOOP` (live peers, real sockets/processes) is the
  external control there, and it runs only on the native path.
- The differential detects a Copy **contradiction**, never a Copy **error** — every killing test in
  AS8 batches 1 and 1b was a *drop* test, and the two mutations with no drop consequence survived
  completely.

---

# 11. Security methodology (C10-C)

## 11.1 Freeze the surface before reviewing findings

`C10-THREAT-MODEL.md` is written and frozen **before** any finding is triaged. Copy the method that
already works in this repository — `HC13-THREAT-MODEL.md` states it exactly:

> *A threat model with no falsifier attached is a list of intentions.*

So every defence in C10's model names **the test that would fail if the defence were removed.** A
defence with no named falsifier is recorded as unverified, not as a defence.

Surface inventory, minimum (each row gets: asset, adversary, current behaviour, defence, falsifier,
class per §11.2):

```text
S01  source and module path traversal        resolving a module path outside the package root
S02  package/cache filesystem access         what `stark build` reads and writes, and where
S03  artifact parsing and limits             ONNX; build.json; stark.lock; manifest.json
S04  generated Rust/source escaping          the DEV-shaped hazard: manifest_paths_are_escaped_to
                                             _toml_rules_not_rust_debug_rules already exists; the
                                             general question is what user-controlled text reaches
                                             generated Rust and how it is escaped
S05  process execution                       `stark build` shells out to `cargo`; argv construction
S06  linker arguments                        what reaches the linker and from where
S07  environment propagation                 which vars are read/forwarded; CARGO_TARGET_DIR;
                                             STARK_REQUIRE_INSTALLED_RUNTIME
S08  temporary files/directories             C6.4 row 17: env::temp_dir + PID + counter, no shared
                                             root — with ONE known survivor outside the matrix (a
                                             gate-7 fixture using /tmp). Verify at HEAD
S09  archive extraction                      release archives; installer
S10  dependency/package provenance           path deps, stark.lock, first-party provider crates
S11  LSP workspace trust                     what the server reads on didOpen; executable paths
S12  executable/tool paths                   how cargo/rustc/linker are located; PATH trust
S13  denial-of-service inputs                shared with C10-B T9
S14  dependency vulnerabilities              the crate graph; note Charter §1.10's new-dependency
                                             rule
S15  licences                                MIT project; the dependency licence set
S16  installer/release authenticity          INTEGRITY not AUTHENTICITY today (§1.6). `stark
                                             doctor` re-hashes against manifest.json; anyone who
                                             can replace the payload can replace the manifest
```

**Escalation:** anything touching archive extraction, process execution, code generation, native
linking or trust boundaries is **CE9** (Charter §2.3) — owner decision, not an implementation call.

## 11.2 Findings are classified into four disjoint classes — never collapsed into DEV numbers

```text
A  COMPILER CORRECTNESS            the compiler does the wrong thing. -> DEV, §3.3's repair rule
B  SECURITY VULNERABILITY          an adversary gains something. -> SEC-C10-NNN, its own register,
                                   CE9 escalation, and a disclosure decision that is the owner's
C  RELEASE/DISTRIBUTION WEAKNESS   unsigned archives, non-standalone payload, offline build
                                   unproven, DEV-165. -> named in the release statement's
                                   exclusions; NOT a compiler-conformance deviation
D  ACCEPTED OPERATIONAL LIMITATION documented, owned, deliberately not fixed. -> named, with the
                                   reason and the owner
```

The HTTP track already uses a `SEC-*` namespace (SEC-HTTP-001/002), so `SEC-C10-NNN` is consistent
with existing practice and keeps class B out of the DEV ledger.

**Mapping onto OD-3's three frozen populations** — the two schemes are deliberately not merged,
because a security finding is not a deviation and a residual is not a defect:

```text
security class A  -> OD-3 population A   compiler deviations       (CD-021 denominator)
security class B  -> its OWN register    SEC-C10-*                 (not a deviation at all)
security class C  -> OD-3 population B   release/distribution      (constrains WORDING)
security class D  -> OD-3 population B or C, per the limitation's subject; if it is a limit on
                     what the EVIDENCE can show rather than on what the product does, it is C
```

An **assurance residual** (OD-3 population C) is never produced by C10-C. It comes from C10-A1 and
C10-D — from evidence that cannot disagree, not from an adversary who can gain something.

## 11.3 What C10-C does not do

It does not fix findings beyond §3.3's rule, does not harden speculatively, and does not add
dependencies (Charter §1.10 requires a necessity/maintenance/licence/security note per new
dependency — that is a separate decision, not a security-review side effect).

---

# 12. Performance methodology (C10-E)

## 12.1 Freeze the workloads first

`starkc/benchmarks/c7-workloads/FROZEN.json` already freezes **7 workloads** by per-file SHA-256
plus a `workload_hash`, at `4650d47`, rustc 1.93.0, aarch64-apple-darwin. That set is inherited.

**What it does not cover, and C10-0 must decide whether to extend it (a workload added after seeing
a number is forbidden by §7):**

```text
large-module scaling      the 7 workloads are small; w06 is the only multi-package one
multi-package scaling     beyond w06's app+lib
LSP change-to-diagnostic  AS8 measured ProjectAnalysis cost (4/8/16/32 modules: 1.4/1.8/7.7/22.0 ms
   latency                per analysis) but that is not change-to-diagnostic latency
ONNX import/verify/deploy included in WP-C10.6's list; the tensor track is DEFERRED, so C10-0 must
                          rule whether this row is measured (it is a measurement, not tensor
                          development) or explicitly excluded
```

**Extend the frozen set in C10-0 if at all**, re-freeze with new hashes, and record the extension
as a dated line. Never mid-packet.

## 12.2 Measure

Per workload, per platform, per profile (debug and release), using `c7-baseline.py --measure`:

```text
lex / parse / resolve / check   the phase split. NOTE: c7-baseline.py currently splits STARK work
                                from cargo/rustc work; a finer per-phase split may need harness
                                work, and that is a C10-E work item, not a compiler change
total compile                   cold
peak compiler memory            RSS
large-module scaling            if the frozen set is extended
multi-package scaling           if extended
LSP change-to-diagnostic        if the frozen set is extended
native debug build              time
native release build            time
binary size                     bytes
runtime                         only where the workload supports meaningful measurement — most of
                                w01-w07 run in ~2-3 ms and a runtime number there is noise
ONNX import/verify/deploy       per C10-0's ruling
```

**Method warning inherited from `c7-baseline.py`'s own header, and it is the useful kind of wrong:**
the first version of that harness timed `stark build` against `stark build --emit-rust`, assuming
the latter stopped before Cargo. It does not — `--emit-rust` only additionally writes the file — so
the two timings were the same run and the "host share" came out as noise, once at **-0.3%**. A
negative share is impossible, which is what exposed the method. **Any C10-E derived quantity must be
checked for impossible values before it is believed.**

## 12.3 The rule

> **Do not optimise merely because a number looks unattractive.**

Performance work belongs in a separate approved packet (Charter §1.6 rule 7, §6 not-yet list)
**unless a measured regression makes qualification impossible** — e.g. a workload no longer
completes, or a build exceeds CI's limits. A baseline is a baseline; C10.6's own text says
regression thresholds may be added only *after* stable baselines exist.

## 12.4 Platform honesty

The only inherited report is `c75-report-macos-arm64.json` — **one platform**. C10-E either produces
Linux-x64 and Windows-x64 reports through CI (§14.3) or states plainly that the baseline is
macOS-arm64 only. It does not generalise one platform's numbers to three.

---

# 13. Compatibility and version policy boundaries (C10-F)

## 13.1 The rule

> **Prefer narrow, evidence-backed commitments over "stable forever". Do not make accidental
> permanent promises.**

Every promise in C10-F must name the evidence that supports it and the packet that produced that
evidence. A promise with no evidence citation is deleted, not softened.

## 13.2 The axes — defined separately, never merged

```text
STARK language version           does not exist today as a constant. C10-F must either define one
                                 or state explicitly that Core v1 is identified by the normative
                                 spec set, not by a version number
compiler version                 starkc 0.1.0 (crate version). Pre-1.0
Core compatibility               what a future compiler must keep accepting/rejecting
optional extension compat        tensor v0.1 — DEFERRED track; the promise is "no promise",
                                 stated as such
MIR version                      MIR_VERSION "0.4". Already enforced: a consumer rejects a program
                                 whose mir_version it does not support (MIR-0017, V-SURFACE-1)
runtime ABI version              MIR_RUNTIME_SURFACE, bumped INDEPENDENTLY of MIR_VERSION.
                                 Compiler/runtime mismatch is already rejected before user code
                                 runs (C6.4 row 9)
Native Provider ABI version      native-provider-abi-v0.1 + CE4-amendment-1 + CD360-amendment-2.
                                 One stark-provider-abi must satisfy both the runtime's `../` and
                                 a provider's `../../../` — Cargo refuses a lockfile naming one
                                 package at two paths
generated artifact/build compat  build.json fields; build_key; the layout contract stark-64-v1
diagnostic compatibility         Charter §1.6 rule 16 makes diagnostics part of behaviour. State
                                 what is stable (codes? spans? text?) and what is not. Pre-1.0,
                                 the honest answer for TEXT is probably "not stable"
platform/toolchain support       Tier-1 linux-x64 + macos-arm64; Tier-2 windows-x64 (C6.4 row 25
                                 REPORT-ONLY). MSRV 1.85, CI on stable
deprecation policy               pre-1.0: what notice, if any, is promised
pre-1.0 vs stable                the governing distinction. `starkc` is 0.1.0 and this is a
                                 PRE-ALPHA language; C10-F should say so plainly
release signing/checksum/        INTEGRITY today, not AUTHENTICITY. A public distribution needs a
   authenticity                  signed manifest, a trusted release key, verification before
                                 installation, and platform notarisation. None exists
```

## 13.3 Forcing function

For each axis, C10-F states one of exactly three things — no fourth option:

```text
COMMITTED     with the evidence and the packet that produced it
UNCOMMITTED   explicitly, with what would be needed to commit
NOT APPLICABLE with the reason
```

---

# 14. CI and platform strategy

## 14.1 Do not assume "CI green" means a C10 requirement was exercised

`ci.yml` has 14 job definitions expanding to **24** matrix checks; `c78-native-capabilities.yml`
adds 2 definitions expanding to **4** — **28 checks in total**. **Every C10 release claim must name
the job(s) and platform(s) that actually execute it**, in the dashboard's `engines` and
`last verified` columns.

The mapping as it stands at HEAD (C10-0 must verify each row against the workflow file, not against
this table):

| Claim area | Jobs that execute it | Platforms |
| --- | --- | --- |
| fmt / clippy / unit + integration tests | `fmt, clippy, test` | linux-x64, macos-arm64, windows-x64 |
| spec-fixture conformance; spec regeneration in sync; fixture extraction in sync; coverage-db consistency; conformance evidence report | `spec fixture conformance` | **linux-x64 only** |
| Tier-1 platform qualification (C6.4) | `C6.4 tier-1 qualification` + `C6.4 tier-1 agreement` | linux-x64, macos-arm64 |
| Windows gap probe | `C6.4 windows tier-2 gap probe` | windows-x64 |
| Differential corpus replay + metamorphic families + package breadth | `C6.5 corpus replay` + `C6.5 corpus tier-1 agreement` | linux-x64, macos-arm64 |
| Comparator sensitivity (the C6.5 mutation controls) | `C6.5 mutation controls` | **linux-x64 only** |
| Slot primitives under Miri | `DEV-160 raw slot primitives under Miri` | **linux-x64 only** |
| Release packaging + installed-runtime isolation | `release package smoke` | all three |
| P1 REST workload, native debug + release, 24 byte-exact HTTP cases | `C7 P1 REST workload` | all three |
| First-party package qualification | `first-party package qualification` | all three |
| External pinned sample suite | `External sample suite (pinned)` | **linux-x64 only** |
| Provider metadata/unit/resource/loopback; TLS certificate matrix | `provider metadata/unit/resource/loopback` (`c78-native-capabilities.yml`) | all three |
| C7.8 qualification record comparison | `C7.8 qualification record comparison` | aggregator |
| the single branch-protection check | `ci-complete` | aggregator |

**Immediately visible consequence:** conformance-fixture evidence, the mutation controls and the
external sample suite are **single-platform**. Any release claim of the form "conforming on the
listed platform matrix" that rests on those jobs is claiming more than the evidence covers, unless
the claim is scoped to what the multi-platform jobs actually run. **C10-A2 must surface this per
row, and C10-Q must not paper over it.**

## 14.2 Tying evidence to the exact commit

```text
every claim cites   commit SHA + workflow run id + job name + platform
the required check  `ci-complete` is the single branch-protection check; it verifies every job
                    succeeded. Cite the RUN, not the check
two runs race       a develop push fires both the push trigger and the open develop->main PR
                    trigger for the same SHA; the concurrency group queues them by the tested
                    COMMIT so they do not overlap. BOTH report. Name which run id you read
artifacts           c64-evidence-*, c65-evidence-*, c65-mutation-evidence, c65-tier1-agreement,
                    c64-qualification-summary, c64-windows-gap-probe, conformance-evidence-report,
                    external-sample-suite-results. Table B of the platform matrix is filled from
                    CI artifacts, NEVER by hand
```

## 14.3 Where a platform cannot run a qualification locally

The implementing environment is macOS-arm64. For Linux-x64 and Windows-x64:

1. add the qualification as a **job** in `ci.yml` (or a new C10 workflow), not as a manual step;
2. have it upload its evidence as an artifact with a stable name;
3. cite `commit + run id + job + artifact` in the packet document;
4. **do not** transcribe numbers from a CI log into a document without the artifact behind them.

New workflow jobs are additive and must not change the `ci-complete` gating semantics without
saying so. Note the fixed-port constraint (39187–39191) that the concurrency group exists to
protect: a new job that binds those ports must join the same group.

## 14.4 Pins

```text
external sample suite   navraj007in/stark-samples @ b3b28e757f38d691e7309f168d1209e28ac459af
                        CI verifies the pin RESOLVED to that SHA (a `ref:` accepts a branch, so
                        an accidental branch name would float silently)
```

**Moving this pin during C10 is forbidden.** It is a deliberate act with its own review, and moving
it mid-qualification means the qualification measured two different things.

## 14.5 History

**Do not merge, rebase or rewrite evidence-bearing history for branch cleanliness.** Sprint 3 and
Sprint 4 landed as merge commits by owner ruling precisely so every cited packet SHA still resolves,
and that was verified after the fact rather than assumed. C10 inherits that rule.

---

# 15. Checkpoints, stop conditions, and re-run scope

## 15.1 Checkpoint contract

Each checkpoint states four things. The table is the contract; the packet sections give detail.

| Checkpoint | What may change before it | Failure means | Re-run scope if a defect is repaired |
| --- | --- | --- | --- |
| **C10-0** | nothing — it *is* the freeze | the campaign does not open | n/a |
| **C10-A1** | test additions only (new tests, no behaviour change) | the claim scope is wrong; escalate CE8 | re-run the census; it is cheap |
| **C10-A2** | as A1 | the dashboard cannot be built at the declared precision | rebuild affected rows only |
| **C10-B** | test/corpus additions; harness code | a panic/hang/nondeterminism at HEAD → §3.3 | the affected target, plus any dashboard row citing it |
| **C10-C** | threat-model text is FROZEN; findings are appended | a class-B finding → CE9, owner | the affected surface row; plus B if the finding is DoS-shaped |
| **C10-D** | test additions; mutation harness | a survivor → the CLAIM changes, not the compiler | the specific trial; and A2's row for that claim |
| **C10-E** | workload set FROZEN at C10-0 | a measurement is impossible or absurd (§12.2) | that workload on that platform |
| **C10-F** | nothing that changes a promise's evidence | a promise has no evidence → delete the promise | the affected axis |
| **C10-Q** | **nothing.** The head is frozen | see §17 | see §15.2 |

## 15.2 Does a repair invalidate earlier evidence?

**The rule:**

```text
a repair that changes COMPILER SOURCE       invalidates every packet's evidence that ran against
                                            the old source. Re-run: A2's affected rows, D's
                                            affected trials, B's affected targets, E's affected
                                            workloads, and the full CI matrix at the new head
a repair that changes TESTS ONLY            invalidates nothing already recorded; the new test is
                                            new evidence. Re-run: the affected suite
a repair that changes DOCUMENTATION ONLY    invalidates nothing. Re-run: the §16.2 sweep
a repair to the HARNESS                     invalidates results produced by the broken harness.
                                            Re-run those, and re-run the harness self-test first
```

**C10-Q re-runs everything at the final head regardless**, because a release claim is made about
one commit. The rule above governs the *interim* checkpoints, so that a mid-campaign repair does not
force a total restart.

## 15.3 Stop conditions — C10 halts and escalates

```text
1  the frozen head's CI does not go green                       C10-0 stops
2  A1 finds the claimable population is materially smaller       CE8 escalation on claim SCOPE
   than the release class assumes
3  a class-B security finding                                    CE9, owner, before continuing
4  a mutation survivor invalidates a claim the owner has         the CLAIM changes; if the owner
   already committed to publicly                                 wants the claim, that is new work
                                                                 and a new packet
5  a repair would require changing the accepted/rejected         CE1/CE2 — this is language design,
   program set                                                   not qualification
6  a finding requires generalising the provider abstraction      CE7 — refuse, cite Charter §1.3
7  a performance regression makes qualification impossible       owner decides; §12.3's exception
8  the packet's work would require reopening C8, the tensor      refuse, cite §3.2
   track, or a broad refactor
9  evidence cited by a C10 document is found not to exist        the citing document is defective;
                                                                 fix before proceeding (§16.2)
```

---

# 16. Documentation discipline

## 16.1 Historical records stay historical

If C10 discovers an earlier claim was wrong:

```text
DO      add a dated superseding record (a new CD entry in COMPILER-STATE.md, or a new dated
        section in the affected document)
DO      update current-state summaries (position lines, "known open at a glance", index tables)
DO      preserve the original record verbatim
DO NOT  rewrite the old decision
DO NOT  edit a historical record to manufacture consistency
```

This is not a style preference — it is how CD-393 handled AS7 (CD-391 "is not rewritten — it is
preserved as written and superseded here"), and OD-5 applies it to a finding this plan made at
opening.

## 16.2 Mandatory final cross-reference sweep

Before C10-Q may record a decision, run the sweep. It is the same discipline AS8's closeout used,
which caught four stale claims. Check:

```text
counts              every sentence stating a number: rule counts, trial counts, job counts,
                    platform counts, packet counts, deviation counts. Grep them
residual ranges     "AS8-R1..R15" and any new "C10-R1..Rn" — the upper bound must match the
                    highest allocated id
DEV IDs             every DEV cited exists, and its LIVE heading (last, not first) says what the
                    citing document claims
gate statuses       C0-C10, Campaigns A/B, Gate 7's split verdicts, the tensor track
branch/head refs    every SHA cited still resolves (`git cat-file -e <sha>`)
platform claims     every "on three platforms" is true of the job that produced the evidence
test counts         no scoped run's count is presented as the tree's count
evidence links      every cited path exists at HEAD; every cited section anchor exists in its
                    target
release wording     the statement in GATE-C10-CLOSURE.md is byte-identical to the one in
                    C10-RELEASE-STATEMENT.md and to any README/site copy
```

Concrete commands are in the `stark-doc-sweep` project skill; invoke it rather than re-deriving.

> **A release qualification that cites nonexistent or stale evidence fails.** This is a stop
> condition (§15.3 item 9), not a polish step.

---

# 17. C10 exit criteria and the release decision

## 17.1 Exit criteria

C10 may close when **all** of the following hold at one frozen head:

```text
E1   C10-0's inventory exists, its contradictions are resolved or explicitly carried, and every
     OD in §2 has a recorded owner ruling
E2   the conformance dashboard exists over a DECLARED population, with per-bucket counts, and no
     row has a blank `independent control` cell
E3   robustness qualification is complete over the DECLARED target population, the harness has
     been shown capable of finding an injected fault, and every finding is triaged into §11.2's
     classes or §3.3's defect rule
E4   the security surface was frozen before review; every defence names its falsifier; every
     finding carries a class (A/B/C/D), an owner and a disposition
E5   the mutation ledger is complete over the DECLARED population, every trial has all TEN
     required fields (§8.3) including freshness (§8.2a) re-checked against the FINAL head, and
     every survivor has either a weakened claim or a constructed
     control
E6   metamorphic relations were declared before running, each cites a normative rule, and no
     relation is a fake pair
E7   performance baselines exist over the FROZEN workload set, with the platform coverage stated
     honestly, and no optimisation was performed under C10 authority
E8   the compatibility policy states COMMITTED / UNCOMMITTED / NOT APPLICABLE for every axis in
     §13.2, each COMMITTED one citing its evidence
E9   ALL THREE OD-3 populations are dispositioned, each counted separately:
       E9a  population A (compiler deviations) — every live-heading deviation carries an owner
            and a disposition, or an explicitly recorded accepted-indefinitely decision. THIS is
            the CD-021 denominator, and an unowned entry here blocks the release decision
       E9b  population B (release/distribution) — every entry carries an owner and a disposition,
            and is named in the release statement's exclusions
       E9c  population C (assurance residuals) — every live AS8-R*, pending AS8-DA*, EI/RA and
            C10-R* residual carries an owner and a disposition, and each is reflected in the
            STRENGTH of the claim it constrains rather than as a defect
E9d  C10-G is evaluated: DEV-012 and DEV-213 are each CLOSED, or the corresponding claim is
     explicitly narrowed in the release statement (OD-4)
E10  CI is green at the frozen head, and every claim maps to the job/platform that executes it
E11  §16.2's sweep is clean
```

**E9a is the one that can block on its own.** An unowned **compiler** deviation blocks the final
release decision by CD-021, carried into WP-C10.7. E9b and E9c must be dispositioned too, but they
constrain *wording* and *claim strength* respectively rather than blocking — that separation is
OD-3's ruling and must not be collapsed at exit under time pressure.

## 17.2 The release decision procedure

**C10 must not decide its desired release wording at opening.** The statement is *derived* at exit.
Procedure:

```text
1  assemble the evidence: dashboard buckets, mutation ledger, robustness result, security classes,
   performance baselines, platform mapping, deviation register
2  for each release class frozen in OD-2, write the STRONGEST claim the evidence supports —
   drafted from the evidence, not from the class's aspirational description
3  subtract: every claim contradicted by a mutation survivor, an ABSENT-evidence rule, a
   single-platform job, or a named deviation
4  what remains is the claim. Choose the gate decision from Charter §5.3's vocabulary
5  escalate the claim to the owner under CE8 — "any Core conformance or release claim" is an
   owner decision requiring an evidence audit. C10 PROPOSES; the owner AUTHORISES
```

### Possible outcomes

```text
PASS                     a precise release claim is authorised
PASS-WITH-DEVIATIONS     authorised with named deviations
REVISE                   one bounded qualification or repair remains
BLOCKED                  the evidence cannot support a credible release claim
```

**Do not assume PASS.** These are the charter's own §5.3 terms, so no new vocabulary is introduced.
`DEFER`, `STOP` and `FAIL` also exist in §5.3 and remain available if the evidence points there.

### Every remaining deviation carries

```text
ID                          DEV-*/SEC-C10-*/C10-R*/AS8-R*/AS8-DA-*
current behaviour           what the compiler does today
impact                      on a user, stated concretely
release-claim consequence   which sentence of the release statement it narrows
owner                       a person or a named track
disposition                 repair / accept indefinitely / route to packet X
target packet/gate          or "accepted indefinitely", explicitly recorded
```

### The wording rules the roadmap already fixes

Permitted shape:

```text
STARK Core v1 front end, interpreter, MIR, and native backend: conforming for the listed
platform matrix
General native Core backend: production MVP, listed deviations X/Y
Tensor extension v0.1 frontend and verifier: conforming for listed scope
Tensor backend execution: capability-limited; see matrix
```

or

```text
STARK Core v1: conforming with deviations DEV-...
```

**Never publish**, unless a CE8 review confirms the evidence supports it:

```text
STARK Core v1: conforming
Known deviations: none
```

And, independently, **never** publish "three independent implementations" in any form (§10.1).

## 17.3 What C10 closure does not authorise

```text
it does not authorise a DISTRIBUTION claim      the archives are unsigned; the payload is not
                                                standalone; offline build is unproven
it does not close C9 Part B                     that needs a second artifact
it does not reopen the tensor track             Gate 7 productisation DEFER stands
it does not make STARK v1 General-Purpose       that is a separate owner act on the same evidence
   Stable's wider claim                          (CD-022)
```

---

# 18. Expected artefacts (the complete list)

```text
STARKLANG/docs/compiler/work-packages/C10-0-OPENING-INVENTORY.md
STARKLANG/docs/compiler/audits/C10-P-LANGUAGE-SERVICES.md      (DEV-213 repair + DEV-012 MANUAL)
STARKLANG/docs/compiler/audits/C10-A1-EVIDENCE-CENSUS.md
STARKLANG/docs/compiler/C10-CONFORMANCE-DASHBOARD.md
STARKLANG/conformance/c10-dashboard.json                     (generated; the dashboard's source)
STARKLANG/docs/compiler/audits/C10-B-ROBUSTNESS.md
STARKLANG/docs/compiler/C10-THREAT-MODEL.md                  (frozen before C10-C reviews anything)
STARKLANG/docs/compiler/audits/C10-C-SECURITY-REVIEW.md
STARKLANG/docs/compiler/audits/C10-D-DIFFERENTIAL.md
STARKLANG/docs/compiler/C10-MUTATION-LEDGER.md
STARKLANG/docs/compiler/audits/C10-E-PERFORMANCE-BASELINE.md
starkc/benchmarks/c10/<platform>.json                        (per-platform, from CI artifacts)
STARKLANG/docs/compiler/C10-F-COMPATIBILITY-POLICY.md
STARKLANG/docs/compiler/GATE-C10-CLOSURE.md
STARKLANG/docs/compiler/C10-RELEASE-STATEMENT.md
COMPILER-STATE.md                                            (new CD entries: the OD rulings, each
                                                             packet's close, the C10 decision)
```

Test/harness artefacts, as they arise: new fuzz generators under `starkc/tests/`, minimised
regression fixtures (C6.5 `retained/` if they enter the corpus, with `corpus.lock` regenerated and
`corpus_version` bumped), the `RuntimeFn` parity/drift test if OD/§5.4 authorises it, and any new
CI job with its artifact.

---

# 19. Register of contradictions and stale claims found at opening

Carried into C10-0 so none is lost. **None is repaired by this plan.**

| # | Finding | Where | Disposition |
| --- | --- | --- | --- |
| 1 | Position line reads as "C9 Part B blocks C10"; roadmap §4.5 says otherwise | `COMPILER-STATE.md` vs `COMPILER-ROADMAP.md` §4.5 / WP-C10.7 | **OD-1 — RULED: does not block** |
| 2 | AS8-R15 discharged in two documents, still open in two others | `AS8-MUTATION-FINDINGS.md`:424 + Campaign B §5 **vs** `AS8-EXIT-QUALIFICATION.md` §5 + `COMPILER-STATE.md` CD-394 | **OD-5 — RULED: APPROVED.** Dated superseding note; CD-394 and the AS8 exit qualification preserved untouched |
| 3 | `ROADMAP.md` §0.1 says "Sprints 1 and 2 are complete; AS3 and AS4 remain" | `ROADMAP.md` §0.1 | **OD-6 — RULED: APPROVED.** Current-state correction; §6.0's gate text preserved and marked satisfied |
| 4 | DEV-165 is called a release blocker in `ROADMAP.md` but is absent from `KNOWN-DEVIATIONS.md` | `ROADMAP.md` §1 vs the ledger | **OD-3 — RULED: three separately countable populations.** DEV-165 is population B |
| 5 | `KNOWN-DEVIATIONS.md` has 186 headings / 178 ids and is append-only; 7 closed-in-record are named by no test, 44 appear in no decision record | the file + `as8-reconcile-deviations.py` | §5.2 — mechanical inventory, hand-audited |
| 6 | Test-target count: 210 top-level integration targets at HEAD; AS8 recorded 209 | `starkc/tests/` | C10-0 reconciles and records the method |
| 7 | "581 tests across 5 targets" (CD-394) is a **scoped** run, and reads like a tree total | `COMPILER-STATE.md` CD-394 | §7.3 — never cite it as the tree's count |
| 8 | **168** granular rules (not 161 — A1-F1) vs 36 with precise evidence; **85 AGGREGATE, 42 ABSENT** | `CORE-V1-COMPLETENESS.md` vs `core-v1-c2.11-evidence.toml`; DEV-017 | MEASURED by C10-A1. And `ABSENT` means the INVENTORY cites nothing, not that nothing tests it (A1-F3) |
| 9 | Conformance fixtures, C6.5 mutation controls and the external sample suite are **linux-x64 only** | `ci.yml` | §14.1 — surfaced per dashboard row; must not be generalised to "three platforms" |
| 10 | The only performance report is macOS-arm64 | `c75-report-macos-arm64.json` | §12.4 |
| 11 | `RA-LINTS` suppresses two deny-by-default lints in generated code, narrowing rustc-as-control | `RUSTC-ASSUMPTION-INVENTORY.md`, EI3 | §5.4 — named in every dashboard row whose control is rustc |
| 12 | `ENGINE-SHARED-FATE-REGISTER.md`'s prose and `engine-shared-fate.json` disagreed on `ESF-PROV-001` until AS8; the JSON is authoritative | the register's own header | C10 reads the JSON, not the prose, for that row |
| 13 | CI at the frozen head was **in progress**, not green, when this plan was written | run `31292404920` | §1.2 — C10-0 must read a completed run |

---

# 20. Roadmap corrections recommended before execution

These are proposals to the owner, not edits. Each is a change to `COMPILER-ROADMAP.md`, which is a
governing document.

1. **WP-C10.3's bullet list says "HIR versus MIR interpreter; interpreter versus native backend"**
   without naming the oracle relationship or the verifier. Given EI2/EI4's finding that the
   differential has a *named reference engine* and that `mir/verify.rs` is an in-tree control that
   is **not** an engine, the roadmap text invites exactly the "three independent implementations"
   error the campaign retired. **Recommend:** add EI6's calibrated sentence to WP-C10.3 and a note
   that verifier evidence is a distinct class.

2. **WP-C10.3 lists "tensor deploy output versus frozen reference workloads"** while the tensor
   track is deferred research. **Recommend:** mark that bullet conditional on C10-0's ruling
   (§12.1's ONNX question), so a session does not read it as authorisation to work the tensor track.

3. **WP-C10.1 says "Generate a dashboard covering ... last verified commit and toolchain"** but does
   not require evidence *quality* columns. Given AS8, a dashboard without shared-fate, independent
   control and mutation-status columns would reproduce EI2's error at release scale. **Recommend:**
   fold §6.5's column list into WP-C10.1, or cite this plan from it.

4. **WP-C10.2's "lexer/parser fuzzing" etc. presumes a fuzzer** the charter's stable-Rust rule
   forbids. **Recommend:** amend to "deterministic seeded generation" and state the limitation, so
   the release wording does not over-claim.

5. **§4.5's release-path preconditions and WP-C10.7's preconditions differ** (§4.5 adds P1 and
   tensor status; C10.7 names C0–C8 + C3–C7). They are reconcilable but a fresh session must
   reconcile them itself. **Recommend:** make WP-C10.7 cite §4.5 explicitly.

6. **The release-class table says "Core v1 Compiler Stable ... Requires C7, C8, and C10"** and omits
   C0–C6 and C9. Correct in substance (C10 subsumes them via C10.7) but reads as an exhaustive
   list. **Recommend:** state it as "requires the mandatory path C0–C8 closed, plus C10", matching
   §4.5.

7. **Nothing in Gate C10 mentions the distinction between a compiler release claim and a
   distribution claim**, while `ROADMAP.md` §1 carries four distribution weaknesses that a reader
   would reasonably attach to "release". **Recommend:** add the §11.2 four-class separation to
   WP-C10.4, and an explicit line in WP-C10.7 that a compiler release claim does not imply an
   authenticated distribution.

---

**End of plan.**

**Approval state (owner, 2026-08-09):**

```text
WP-C10 EXECUTION PLAN — APPROVED WITH AMENDMENTS

OD-1  APPROVED         C9 Part B does not block C10
OD-2  APPROVED         evaluate Core v1 Compiler Stable + Native Systems Preview only
OD-3  APPROVED W/REF   three separately countable populations; all dispositioned by C10-Q
OD-4  MODIFIED         neither blocks opening; DEV-012 interactive validation and DEV-213
                       bounded repair both run EARLY in C10 as C10-P; neither reopens C8;
                       unresolved state NARROWS the final language-services claim
OD-5  APPROVED         prospective superseding record; CD-394 and the AS8 exit qualification
                       preserved
OD-6  APPROVED         correct the live ROADMAP summary; §6.0's gate preserved, marked satisfied

MANDATORY PLAN AMENDMENTS — ALL APPLIED
1  normative specifications remain authoritative        -> §"How to execute C10"
2  no expected audit or falsification finding rate      -> §6.4 forcing function, §8.3
3  inherited mutation evidence receives a freshness rule -> §8.2a
```

**C10-0 is authorised to begin once a green exact-head CI baseline exists (§1.2).** Its first act
is to transcribe the six rulings into `COMPILER-STATE.md` as one dated `CD-NNN` entry and to make
the two document corrections OD-5 and OD-6 authorise.

**C10-Q remains an owner decision under CE8** — "any Core conformance or release claim" is an
escalation, and Charter §2.2 forbids a session claiming Core v1 conformance on its own authority.
C10 proposes; the owner authorises.
