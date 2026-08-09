# WP-ARCHITECTURE-STABILIZATION — Compiler architecture consolidation programme

**Status:** **COMPLETE — all four sprints CLOSED. Campaign A PASS, Campaign B EXITED PASS
2026-08-09.**

```text
Sprint 1   CLOSED    AS-SPRINT1-CLOSEOUT.md
Sprint 2   CLOSED    AS-SPRINT2-CLOSEOUT.md          AS1b, AS5
Sprint 3   CLOSED    Campaign A PASS                 AS0-AS4
Sprint 4   CLOSED    AS-SPRINT4-CLOSEOUT.md, PASS    AS6 (CD-390), AS7 (CD-391, criterion 2
                     re-qualified CD-393), AS8 (CD-394)
Campaign B EXITED    CAMPAIGN-B-EXIT-REPORT.md, PASS — a prerequisite for C10, and it makes no
                     stability or conformance claim itself
```

Both sprints landed on `develop` as **merge commits** — `645997d` (Sprint 3) and `d79ad03`
(Sprint 4) — with no rebase, squash or cherry-pick, so every packet SHA these records cite still
resolves. Nothing in this programme remains open; **the next gate is C10 release qualification.**

**Date:** 2026-08-06, status reconciled 2026-08-08, **closed 2026-08-09**.
**Owning track:** compiler, under `COMPILER-CHARTER.md` and `COMPILER-ROADMAP.md`.
**Roadmap relationship:** this is a proposed compiler work-package programme, not a second live
project roadmap. `ROADMAP.md` remains the only live platform plan. If the later campaigns are
approved after AS0, the integration gates in §4 must be added to that roadmap and the current
compiler position must be recorded in `COMPILER-STATE.md`.

---

## 1. Approval record, and the decision now due

### Approved and delivered — Sprint 1 (2026-08-06)

Approved in session, executed on `wp-arch-stability/sprint-1`:

1. **Cranelift retirement** and the **manifest strictness audit** — audit-gated, neither changing
   compiler behaviour;
2. **AS0** — baseline, inventories and the entry-point characterization matrix. **Partial:** five
   work items remain (§7), so AS0's report exists but AS0 has not formally exited;
3. **AS1a** — the reproduced package source-identity/provenance defect, closed;
4. **AS2** — one compiler session; six hand-rolled pipelines removed. *Approved in session as part
   of Sprint 1's scope; it was not in this document's original approval boundary, which is why this
   section now records what was authorised rather than what is being asked for.*

What the evidence changed, which is the point of having run AS0 first: six bypassing assemblies not
four, three shipped binaries not two, AS5 re-sized from a tightening to a tightening plus a
compatibility correction plus a correctness defect, and the provenance half of the source-identity
defect found to be worse than recorded. All of it is in
`STARKLANG/docs/compiler/audits/AS0-BASELINE-AND-INVENTORY.md` and
`AS0-MANIFEST-STRICTNESS-AUDIT.md`.

### Still reserved

Campaigns A (remainder) and B are **not** approved. Their scope and ordering were always a proposal
to be resized after AS0, and the AS0 report now exists to resize them against.

### The second owner decision — TAKEN 2026-08-07

**The Campaign A gate is APPROVED and BINDING.** It is written into the live roadmap at
`ROADMAP.md` §6.0 (commit `b33b3e7` on `develop`), not left as a proposal here:

> Structured-concurrency compiler/runtime work may not begin until Campaign A exits green: AS0,
> AS1a, AS2, AS1b, AS3 and AS4 closed **and owner-reviewed**.

The evidence that justified making it real: AS0 inventoried six pipeline assemblies bypassing the
shared driver and several parallel provenance authorities; Sprint 2 then found four defects while
consolidating just two authorities, including `TRAIT-COHERENCE-001`'s cross-package clause — a
normative rule that had effectively never worked as specified (DEV-183).

AS3 and AS4 are the two foundations concurrency would amplify. AS3 closes the callable-use /
generic-instantiation gap that already has value-representation work *paused*; AS4 establishes one
authority for `Copy`, drop and borrow/reference containment. Concurrency asks exactly those
questions on day one — can this value cross into a task, is it moved or borrowed, who owns an affine
resource after spawn, when does its `Drop` run, what survives cancellation, can a reference outlive
the spawning scope. Building task semantics against several authorities would make concurrency
another *producer* of compensating mechanisms.

**The gate blocks the concurrency campaign only.** Packages, tooling, documentation and Phase 3's
synchronous REST server are unaffected; `ROADMAP.md` §6.0 states this so it cannot be read as a
general freeze.

### Sprint 3 — APPROVED, conditionally

AS3 → AS4 is approved, and **may not begin until Sprint 2's Tier-3 gate discharges** on green CI.
Scheduling decisions taken with it:

| Decision | Ruling |
| --- | --- |
| AS0's callable execution-site inventory as AS3's opening checkpoint | **APPROVED** — it completes AS0 item 6, establishes AS3's real scope, and prevents repeating A3c/A4's mistake of assuming the callable surface was complete |
| AS0's `WP-C7.8-RB0` predicate inventory as AS4's opening inventory | **APPROVED** |
| AS0's engine-independence inventory | **DEFERRED to AS8/C10** — it does not delay Sprint 3 |
| LSP transport hardening (DEV-186 + the request-id model) inside Sprint 3 | **NO** — DEV-186 is availability, not soundness or wrong-code, so it does not meet the live-defect pre-emption threshold, and §2.2's WIP limit argues against a second cross-cutting lane while AS3/AS4 are active |
| LSP transport hardening after Campaign A | **SCHEDULED** — a short inter-sprint packet between Campaign A green and Sprint 4, not a fifth architecture sprint |
| DEV-012's seven interactive validations | **Stays with AS8** — evidence work, not transport implementation |

The compiler's large-scale pipeline is retained:

```text
source/package
  -> AST
  -> resolved HIR
  -> type / flow / borrow / constant analysis
  -> typed-HIR execution
  -> monomorphised MIR
  -> optimisation
  -> MIR verification
  -> MIR execution or generated Rust
  -> native executable + runtime/providers
```

This programme is not a rewrite. It consolidates the contracts connecting those stages so that one
semantic fact has one authoritative representation and every compiler entry point uses the same
pipeline.

### Approval boundary

| Scope | Status | Integration gate |
| --- | --- | --- |
| Cranelift retirement, manifest strictness audit | **APPROVED, DELIVERED** (2026-08-06) | none; an isolated build/test-surface retirement and a read-only measurement, both audit-gated (§5) |
| AS0 | **CLOSED 2026-08-07** — item 6 `AS0-CALLABLE-EXECUTION-SITE-INVENTORY.md`, item 7 `AS0-RB0-PREDICATE-INVENTORY.md`, item 10 deferred to AS8/C10 by owner decision. All items done or explicitly deferred, which is AS0 §7's exit condition | discharged; Campaign A subsequently closed PASS |
| AS1a | **APPROVED, DELIVERED** | none; defect packet |
| AS2 | **APPROVED, DELIVERED** | none |
| Sprint 1 Tier-3 closeout | **PASS** — `AS-SPRINT1-CLOSEOUT.md`, CI 24/24 green on `7012080` | discharged; Sprint 2 may open |
| Sprint 2 Tier-3 closeout | **PASS** — `AS-SPRINT2-CLOSEOUT.md`; AS1b and AS5 closed, CI green 24/24 on `59bd1ca` | discharged; **Sprint 3 may open** |
| AS1b | **CLOSED 2026-08-07** — i, ii(a–e) and iii; owner-accepted at `a6107fb`. See `AS1B-OPENING-ANALYSIS.md` §9 | none |
| remainder of Campaign A (AS3, AS4) | **CLOSED** — Campaign A exits **PASS**, CI-confirmed. The AS0 report discharged the decision this row was waiting on | discharged; the structured-concurrency gate below is satisfied |
| AS5 | **CLOSED 2026-08-07** — a–g. See `AS-SPRINT2-CLOSEOUT.md` | none |
| Campaign B remainder (AS6–AS8) | **APPROVED for execution, 2026-08-08** — the whole of Campaign B as already designed, not AS6 alone. C8 is settled (CLOSED, CD-385), so AS8 is unblocked on that axis | before C10 release qualification |
| AS8 | **CLOSED 2026-08-09 — qualification PASS (CD-394).** All five exit criteria met. 39 compiler-source mutation trials across ownership, trap, drop, resolver and MIR-verifier rules: **26 confirmed the prediction, 13 falsified it in both directions.** Four shared-fate register entries were corrected by measurement (ESF-COPY-001 gains a control and drops to high; ESF-TRAP-001 splits into 001a/001b; ESF-TYPE-001 loses its recorded control and rises to high; ESF-PROV-001 gains one — `mir/verify.rs`). One defect filed, **DEV-213**, demonstrated at HEAD; no DEV for any survivor, per owner ruling. Coverage baseline established (tooling did not exist), pinned samples 34/34, `COMPILER-STATE.md` 12,979 -> 6,681 lines lossless. `AS8-EXIT-QUALIFICATION.md` | discharged; **Campaign B may exit after the Sprint 4 Tier-3 closeout** |
| AS7 | **CLOSED 2026-08-08 — qualification PASS (CD-391), CRITERION 2 RE-QUALIFIED 2026-08-09 (CD-393).** The original criterion-2 PASS was produced by a forcing test that observed 36 of 234 methods; five violations were live under it, including the `traits -> convert` cycle Packet 7's ruling was recorded as having broken. Repaired detector, `trait_contracts.rs` added, three `state` edges moved: the checker is **eleven** modules with an executable, cycle-free graph over 234 of 234 methods. Criteria 1/3/4/5 unaffected and re-checked. `AS7-EXIT-QUALIFICATION.md` §9 | discharged; **AS8 may open** |
| AS6 | **CLOSED 2026-08-08 — qualification PASS.** All five exit criteria met on `6050efa`; CI green on three Tier-1 platforms, zero failing jobs. `STARKLANG/docs/compiler/audits/AS6-EXIT-QUALIFICATION.md` (CD-390). Four residue entries and three limits recorded, not hidden | discharged; **AS7 may open** |

No calendar estimate is attached before the inventories exist. Planning is expressed in bounded
packets, and each packet exits only on its evidence. `ROADMAP.md` §2.2's work-in-progress limit
remains binding: only one major compiler/runtime packet is active at once.

### OWNER DECISION — 2026-08-08: Campaign B approved

> Campaign B remainder is **APPROVED for execution**.
>
> AS6 work already landed on the architecture-stabilization branch is **ratified as execution under
> this approval**; no implementation is being retroactively reclassified as qualification evidence.
>
> AS6 must still satisfy its published exit criteria and Tier-2/CI qualification before closure.
>
> AS7 may begin only after AS6 closes.
> AS8 remains ordered after AS7 and certifies the frozen AS6/AS7 result.
> Campaign B still exits only after AS6, AS7 and AS8 close.

Approved as **AS6–AS8, the already-designed Campaign B** — deliberately not AS6 alone, which would
have created a second artificial owner gate before AS7 without adding evidence. This is governance
reconciliation, not a change in technical scope.

**Why it was needed.** The reservation had become indefensible against the record: AS6 had several
landed implementation packets while the approval boundary still read `RESERVED`. A reader could not
tell whether the work was authorised. Resolving it *before* qualification keeps qualification
answering one question — does AS6 satisfy its technical exit contract? — rather than simultaneously
repairing who authorised it.

### Branch transition — Sprint 3 → Sprint 4

> Sprint 4 AS6 packets began on `wp-arch-stability/sprint-3` after Campaign A closure, to preserve
> the active architecture-stabilization execution state. From commit `6050efa`, Sprint 4 execution
> continues on `wp-arch-stability/sprint-4`. **History is retained rather than rewritten.**

The AS6 packets that landed on the Sprint 3-named branch are `46ae2ec`, `fe80129`, `33cb0a7`,
`62ef6b0`, `46e6cc8`, `ad8fce5`, `a84ee92`, `6037dfc`, `9147073` (2C), `5190d1b` (4C) and `6050efa`
(4D). Nothing about them is rebased, renamed or force-moved: making the branch history
aesthetically correct would create more governance risk than the one-line record it removes.

### Execution units

Campaigns are **evidence groupings**: they state what must be true before an outside commitment may
be made. Sprints are **execution units**: how the work is written and tested. The rhythm is *one
formal evidence closeout per sprint, multiple coherent commits inside the sprint, and targeted tests
at the semantic checkpoints each packet names* — never one enormous commit whose first meaningful
test runs days later. §7 defines the three evidence tiers; each packet's **Checkpoint evidence**
section names its own tier-2 boundaries.

| Sprint | Packets | Rhythm |
| ---: | --- | --- |
| 1 | Cranelift retirement, AS0, AS1a, AS2 | Isolated build-hygiene commit; targeted provenance and characterization checkpoints; then the driver-consolidation marathon and one formal sprint closeout |
| 2 | AS1b, AS5 | Long implementation runs, with SourceId diagnostic tests and the JSON corpus as separate checkpoints |
| 3 | AS3, then AS4 | Deliberately incremental. AS3's semantic-complete checkpoint authorises AS4; both close formally at the Sprint 3 closeout |
| 4 | AS6, AS7, then AS8 | **IN EXECUTION** (`wp-arch-stability/sprint-4`). Extension-isolation checkpoint, then the modularisation marathon, then assurance written against the finished result |

Which packets tolerate long uninterrupted implementation runs is a **property of the packet, not of
the sprint**. AS2 and AS7 are the two genuine refactoring marathons: their failure modes are largely
compiler-visible. AS1a, AS3 and AS4 change what the compiler decides and stay incremental. AS1b,
AS5 and AS6 permit long runs but change observable behaviour — diagnostic and trap locations,
accepted JSON, extension-conditioned semantics — so each carries mid-sprint checkpoints that
`cargo check` cannot stand in for. AS8 is assurance written against finished work, not
implementation to be batched with it.

**Sprints and campaigns intentionally do not nest.** AS5 belongs to Campaign B but executes in
Sprint 2, because its strict-parsing decisions are cheapest to settle alongside AS1b's source-map
work. This does not move the Campaign A exit gate: that gate is reached only when AS0, AS1a, AS2,
AS1b, AS3 and AS4 are closed and owner-reviewed — the end of Sprint 3 — regardless of AS5 having
landed earlier. The C8 gate decision that AS5 and AS8 were waiting on is made — CLOSED, CD-385 —
so neither is blocked on it; both inherit the limits §4 records.

Sprint 4 may not begin until Sprint 3 reports a green semantic boundary. AS8 must not certify work
that is still moving.

---

## 2. Why this programme exists

The architecture has a strong backbone: typed arena IRs, an executable HIR oracle, explicit MIR,
a verifier-gated backend boundary, generated-Rust native compilation, a separate runtime/provider
ABI, and multi-engine differential evidence.

The recurrent risk is **semantic-authority fragmentation**:

- a byte span does not carry the identity of its source;
- package roots can acquire both absolute and logical source names;
- explicit and implicit calls do not publish the same callable-selection metadata;
- type-to-runtime-value conformance is executable but cannot yet be enforced at every boundary;
- Copy, drop, reference-containment and related MIR properties have multiple implementations;
- CLI, package, test and tool entry points still assemble overlapping compiler pipelines;
- Core/tensor isolation is policy-tested but tensor knowledge remains embedded in central phases;
- JSON and version-surface rules are implemented in several partial authorities;
- the largest passes combine many responsibilities through large mutable contexts.

Fixing individual symptoms without consolidating their authority would preserve the defect class.

---

## 3. Programme rules

1. **No language redesign.** The accepted/rejected Core program set, ownership model, trap model,
   MIR semantics, ABI and backend remain unchanged unless separately approved through the charter.
2. **Behaviour before modularity.** Establish executable invariants and authority boundaries before
   splitting large files. Moving fragmented logic into smaller files does not consolidate it.
3. **One authority, independently challenged.** Producers and verifiers may remain independent
   consumers, but they must be tested against a declared semantic model rather than drifting local
   approximations.
4. **No generic extension framework.** C9's second-artifact evidence requirement remains binding.
   Tensor code may be quarantined behind existing internal boundaries; no public plugin/provider
   abstraction is authorised here.
5. **No full incremental compiler.** LSP work is limited to measured debounce, cancellation and
   cache ownership improvements. `COMPILER-CHARTER.md` §6 continues to defer full incrementality.
6. **No broad cleanup packet.** Each packet owns a named correctness or maintainability claim,
   adds evidence before deleting old mechanisms, and records adjacent findings as follow-ups.
7. **The live correctness stream continues.** Audit- or application-discovered soundness,
   double-destruction, over-acceptance and release-blocking defects are not queued behind this
   programme. A bounded defect repair may interrupt the active architecture packet. If the defect
   expands into a major compiler/runtime campaign, pause the architecture packet so the WIP limit
   still holds. At proposal time CD-383 is evidence that this lane is active, not historical.
8. **Shared-checkout discipline applies.** Each implementation packet declares its file ownership
   set and uses explicit-path staging if the owner requests a commit.

### Live-defect pre-emption rule

Before starting or resuming any packet, read the current `COMPILER-STATE.md` position and newest
audit entries. A newly reproduced memory-safety analogue, double destruction, ownership violation,
wrong-code result or accepted-invalid program with incorrect execution takes priority over roadmap
sequencing. The focused repair keeps its own DEV/CD evidence and does not get absorbed into a broad
architecture commit. Non-blocking adjacent findings remain follow-ups under the charter.

---

## 4. Dependency and integration map

```text
[SPRINT 1]
Cranelift retirement (isolated commit, audit-gated)
 |
 +--> AS0 Baseline, reproductions and characterization
        |
        +--> AS1a Canonical package source identity and provenance
               |
               +--> AS2 One compiler driver
                        (characterization baseline is AS2's mid-flight oracle)

[SPRINT 2]
 +--> AS1b SourceId-bearing spans
 |         <checkpoint: diagnostic/trap location tests>
 +--> AS5 Protocol and version contracts        [C8 gate decision MADE: CLOSED, CD-385]
           <checkpoint: JSON conformance corpus>

[SPRINT 3]
 +--> AS3 Callable-use totality -> resume value-representation enforcement
        |
        [SEMANTIC-COMPLETE CHECKPOINT — AS3 evidence green before AS4 opens]
        |
        +--> AS4 Semantic type-property authority

                  [CAMPAIGN A EXIT]
        AS0, AS1a, AS2, AS1b, AS3, AS4 closed and owner-reviewed
        required before structured-concurrency compiler/runtime work

        [SEMANTIC-GREEN GATE — Sprint 4 does not open until Campaign A is green]

[SPRINT 4]
 +--> AS6 Core/extension quarantine
 |         <checkpoint: Core-only and tensor-enabled session isolation>
 +--> AS7 Pass modularisation and compiler API boundary
 |
 |         [IMPLEMENTATION FREEZE — AS6/AS7 Tier-2 checkpoints green]
 |
 +--> AS8 Engine independence, tooling scale and governance closure
           [post-C8 tooling scale work: C8 gate decision MADE, CLOSED, CD-385]

                  [CAMPAIGN B EXIT]
                 required before C10
```

Sprint 1 (Cranelift retirement, AS0, AS1a, AS2) is approved and delivered. Everything after it is
reserved pending the second owner decision — see §1.

- **AS1b follows AS2** so SourceId is threaded through one pipeline rather than through several
  assemblies AS2 would immediately delete.
- **AS4 follows AS3**, and does not begin until AS3's semantic-complete checkpoint is green. AS3
  remains formally open until the shared Sprint 3 Tier-3 closeout. AS4 consolidates the
  type-property authorities that AS3's callable metadata feeds; building it on unvalidated metadata
  would mean re-doing it. This replaces the earlier proposal to design AS1a and AS4 in parallel.
- **AS2 requires AS0's characterization baseline.** The currently identified pipeline assemblies
  are not behaviourally identical, and AS0 must establish their exact count, so consolidation
  necessarily chooses a winner. Without a captured baseline, "the entry points now agree" is
  satisfiable by every assembly having silently changed.
- **C8's gate decision is made.** CD-385 closed it on 2026-08-06 with a stated limit — three of ten
  advertised features carry interactive evidence, and DEV-012 stays open and narrowed to the other
  seven (`GATE-C8-CLOSURE.md` §2). AS5 and AS8 therefore take their "C8 closes first" branches.

---

## 5. Campaign A — correctness foundations

### Sprint 1 opening — Cranelift retirement (not a packet)

A build/test-surface retirement with no production semantic claim, taken as one isolated commit
before AS0 so that subsequent clean test and all-target builds avoid obsolete dev-dependencies.

**Audit gate — all three must hold before the commit:**

1. no `src/` module references cranelift or `target-lexicon`;
2. the crates are `[dev-dependencies]` only, so the shipped compiler's dependency surface is
   unchanged (charter §1.10);
3. the only consumer is the disposable WP-C3.3 direct-backend spike, and its own dependency note
   authorises removing both the dependencies and `tests/spike_cranelift.rs` after Gate C3 selects a
   backend.

**Work:** remove the four `cranelift-*` crates and `target-lexicon` from default builds, and
remove `tests/spike_cranelift.rs`. Preserve the historical measurements and selection evidence in
the existing C3.3/C3.4 spike documents. **Two documents then hold dangling pointers and both must
be corrected in the same commit:** `WP-C3.3.md`'s deliverables list names the removed test file, and
`starkc/docs/compiler/spikes/WP-C3.3-direct-cranelift.md` carries a literal reproduction command
(`cargo test --test spike_cranelift`) that will no longer run. Replace each with an explicit
historical/non-runnable notice rather than deleting the record. The commit is build hygiene plus
test-surface retirement, not a dependency-only change.

**Checkpoint evidence:** `cargo check --all-targets`, `cargo test --lib`, and a recorded
before/after clean **test/all-target** build time to substantiate the benefit claim. Confirm the
staged ownership set contains the Cargo files, the disposable spike test and the C3.3 documentation
correction. No qualification cycle.

If any audit condition fails, stop and leave the dependencies in place; the retirement then returns
to AS7 where the boundary work can absorb it.

---

### Sprint 1 opening — manifest strictness audit (read-only, feeds AS5)

A read-only measurement taken in Sprint 1 because its result decides AS5's *shape*, and learning it
mid-Sprint-2 is expensive.

**Work:** parse every first-party `starkpkg.json` and lockfile under `packages/` with both the
current parser and a candidate strict RFC 8259 parser. The audit compares **values, not just
verdicts**, and records three deltas:

1. current accepts, strict rejects — invalid JSON such as trailing commas, unescaped control
   characters and malformed escapes;
2. strict accepts, current rejects — valid JSON the current implementation cannot consume;
3. both accept, values differ — silent corruption, where a parser produces a string the input did
   not denote.

The third is the dangerous cell, and the audit found it non-empty. `package.rs` rejects every `\u`
escape outright (`Unsupported escape`) — a compatibility gap that is still AS5's. The LSP parser
accepted `\u`, silently dropped any scalar `char::from_u32` refused, and never paired surrogates, so
a valid escaped emoji parsed to the empty string rather than failing. A verdict-only audit would
have called that agreement.

**That half is fixed.** It became DEV-182 / CD-384 (2026-08-06) and was repaired under the §3
live-defect pre-emption rule rather than waiting for AS5 — its own branch, its own evidence, merged
to `develop`. The cell is described here in the past tense on purpose: the *class* is what AS5
inherits, not the instance.

**Outcome, recorded in the AS0 report:** classify AS5 on all three axes. Existing repository files
rejected by the strict parser require a **repository migration** before tightening. Valid RFC 8259
inputs rejected by the current parser require a **compatibility correction**, even when the
checked-in corpus is strict-clean. Value divergence on commonly accepted input is a **correctness
defect** with its own DEV record, not an AS5 design choice. If none exists and the only delta is
invalid input currently accepted, AS5 is a tightening. Nothing is rewritten in Sprint 1.

---

### AS0 — Baseline, reproduction and authority inventory

#### Claim

Every later packet begins from a reproducible defect or a complete authority inventory, not from a
module-size impression.

#### Work

- Reproduce package-root source duplication with a package whose entry is outside the invoking
  process's current directory.
- Reproduce the provenance half separately: after logical package naming, verify whether
  `build_source_map` classifies real package files as `Module { package: None }` while the phantom
  absolute entry is the only `Root` record.
- Build the same package in two absolute checkout locations and compare source maps, MIR dumps and
  native build keys.
- Inventory every parse/resolve/typecheck pipeline assembly by entry point.
- **Capture a bounded characterization matrix** for those entry points. Each assembly runs every
  applicable row and records `NOT-APPLICABLE` for unsupported modes:
  - a valid Core package/program;
  - an invalid root source with ordered diagnostics;
  - an invalid dependency source proving source/provenance attribution;
  - a provider-overlay package proving synthesized-source handling;
  - Core and tensor language-option sessions, sequentially and in parallel where the entry point
    exposes extension configuration.
  Pin ordered diagnostics, source names/provenance, overlay handling, language options and whether
  the entry constructs its root `SourceFile` with a disk path. The exact assembly count is an AS0
  output, not a number assumed by this proposal. Divergences are findings, not test failures.
- Inventory every explicit and implicit user-callable execution site.
- Adopt the predicate inventory required by
  `WP-C7.8-RB0-MIR-Type-Property-Authority.md` rather than creating a second list.
- Inventory JSON parsers, serializers and accepted deviations from RFC 8259.
- Record baselines for check time, native build time, LSP change latency, compiler binary size and
  dependency count.
- Execute the approved AS0 scope of `WP-ENGINE-INDEPENDENCE.md` rather than inventing a second
  shared-fate vocabulary. Its register, evidence audit and engine-risk profiles are AS0 outputs;
  its rustc inventory and mutation recommendations feed AS5/C10 and AS8 respectively.
- Record and run the external `stark-samples` qualification suite, when available, pinned by commit
  hash and expected manifest. Treat it as independent application evidence, not as a normative
  source or an unversioned dependency on `~/Code/stark-samples`.

#### Exit criteria

1. The duplicate-identity and wrong-provenance halves are each reproduced or closed with contrary
   executable evidence.
2. The driver, callable, predicate and JSON inventories are exact-set checked where practical.
3. The entry-point characterization baseline is committed as executable tests, and every observed
   divergence between assemblies is recorded as a named finding AS2 must consciously resolve.
4. All three manifest-parser deltas are recorded, including value divergence on commonly accepted
   input, and AS5 is classified as tightening, compatibility correction, repository migration,
   correctness defect, or the applicable combination.
5. Performance commands and raw results are recorded and repeatable, including the clean-build
   before/after for Cranelift retirement.
6. The pinned samples-suite result is recorded, or its absence is explicit rather than silently
   reducing the independent evidence set.
7. Each later packet has a bounded ownership set and an identified rollback point.

#### Checkpoint evidence

`cargo test --lib`; the new characterization and reproduction tests; the two read-only audits. No
differential, native or package qualification — AS0 changes no compiler behaviour.

#### Stop condition

If relocation already produces one logical root identity and an invariant build key, close that
finding and do not manufacture an AS1 root fix. SourceId work remains independently justified by
`WP-SPAN-SOURCEID.md`.

---

### AS1a — Canonical package source identity and provenance

#### Dependencies

- AS0 reproduction of the duplicate-identity and wrong-provenance halves.

#### Work

- Give each physical source exactly one logical compiler identity.
- Keep canonical disk paths as loading metadata, never as source names.
- Make root/module/package provenance explicit rather than inferring it by comparing a logical
  source name with an absolute package-entry parent.
- Remove absolute checkout paths from MIR/build-key identity unless deliberately included in
  non-reproducible debug metadata.
- Use one helper for package entry `SourceFile` construction at every current call site; AS2 later
  makes the whole pipeline singular.

#### Exit criteria

1. One physical package root produces one `SourceRecord`.
2. The logical entry is the sole `Root`; every package module carries the correct non-empty package
   provenance.
3. Relocating identical source/package graphs preserves logical source maps, MIR dumps and build
   keys in two consecutive runs.
4. No canonical absolute checkout path participates in reproducible source identity.
5. Package, package-with-overlay and native-build paths share the same logical-entry helper and
   focused regression.

#### Checkpoint evidence

`cargo test --lib`; source-map and provenance regressions; the two-root relocation test comparing
logical source maps, MIR dumps and build keys; the AS0 characterization tests re-run to show which
recorded divergences this packet closed and which it left for AS2. Build keys change once here —
state that as an expected cache invalidation, not a regression.

#### Risks and escalation

AS1a is deliberately narrower than SourceId-bearing spans. Any proposed MIR debug-contract change
beyond removal of accidental absolute identity is a CE3 decision and is excluded from AS1a.

---

### AS2 — One compiler session and one pipeline

#### Claim

All tools observe the same package loading, overlays, language options, resolution, checking,
diagnostic and source-identity behaviour.

#### Work

- Define one internal `CompilerSession`/driver facade with explicit operations such as:

  ```text
  analyze
  check
  execute_hir
  lower_mir
  execute_mir
  build_native
  query
  ```

- Make package loading, provider overlays, language options, source maps and diagnostic collection
  session-owned inputs.
- Migrate `starkc check/run`, `stark check/run/test/build`, documentation example validation,
  deployment analysis and LSP package analysis.
- Keep command-line parsing and presentation outside the driver.
- Remove the superseded manual parse -> resolve -> typecheck assemblies only after an exact-set
  entry-point test proves migration completeness.

#### Exit criteria

1. A repository search finds no production entry point independently assembling the semantic
   pipeline outside the driver.
2. The same invalid package produces the same ordered diagnostic structures through compiler CLI,
   package CLI, test runner and LSP analysis.
3. Core/tensor language options remain per-session under sequential and parallel analysis.
4. Provider-backed packages use the same analysis result for checking and native building.
5. Existing unit, integration, fixture, package and differential suites remain green.

#### Checkpoint evidence

A refactoring marathon, but not a test-free one. `cargo check` is the inner loop; the checkpoints are
**per migrated entry point**, not one at the end: after each entry point moves onto the driver, run
`cargo test --lib` plus that entry point's characterization tests from AS0 and record whether its
behaviour was preserved or deliberately changed. A deliberate change needs a one-line justification
in the packet record — that is the difference between consolidating the assemblies and silently
replacing all but one of them.

#### Non-goal

This is not an incremental query engine and does not introduce persistent compiler state between
commands.

---

### AS1b — SourceId-bearing spans

**Status: CLOSED 2026-08-07** (i, ii a–e, iii). The packet as written under-estimated the work —
`SourceId` was allocated *after* the front end, so the identity a span needs did not exist when
spans are created — and it did not anticipate MIR's parallel `FileId` namespace. Both are recorded
in `AS1B-OPENING-ANALYSIS.md`; §9 carries the closure and the acceptance-criteria evidence.

#### Dependencies

- AS2 shared compiler session/driver.
- Existing `WP-SPAN-SOURCEID.md`, which remains the normative implementation packet.

#### Work

- Execute `WP-SPAN-SOURCEID.md` through the single AS2 pipeline.
- Route compile-time diagnostics and runtime trap locations through `SourceMap`.
- Remove ambient-file guesses and the interim wrong-source detector after total resolution exists.
- Prove that CLI, package, test, documentation and LSP consumers obtain the same SourceId-bearing
  diagnostics from the shared analysis result rather than adding per-entry-point plumbing.

#### Exit criteria

1. Dependency diagnostics and runtime traps resolve against the dependency's file and line table.
2. No AST/HIR/MIR/query diagnostic path accepts a bare byte range without source identity.
3. Span-to-location resolution is total through `SourceMap` in compile-time and runtime paths.
4. Existing diagnostic JSON remains deterministic.
5. Superseded diagnostic ambient-file guessing is removed only after exact-set migration evidence
   exists; item-to-file metadata used for separate module semantics is retained or removed on its
   own demonstrated purpose.

#### Checkpoint evidence

Long implementation runs are fine; the checkpoint is not optional. `cargo test --lib` plus the
diagnostic-location and trap-provenance suites at each phase boundary (AST, HIR, MIR, query paths).
**The type system cannot prove the correct `SourceId` was attached** — only that one was — so every
phase whose spans are converted needs a test that a diagnostic in a *dependency* file still resolves
to that file. Removing the interim wrong-source detector is its own checkpoint, gated on the
exact-set migration evidence in criterion 5.

#### Risks and escalation

Span representation is foundational but does not change language semantics. Any proposed MIR
debug-contract change beyond source identity is a CE3 decision and is excluded from AS1b.

---

### AS3 — Total callable-use metadata and oracle representation enforcement

#### Dependencies

- AS2 shared session/driver and AS1b source-aware semantic metadata path.
- Existing `WP-VALUE-REP-TOTAL.md` A0–A3c work.

#### Work

1. Author and approve `WP-CALLABLE-USE-TOTAL` before implementation.
2. Publish exactly one checker-selected `CallableUse` for every accepted explicit or implicit
   user-callable invocation, including:
   - selected callable identity;
   - explicit empty or populated generic environment;
   - receiver adjustment and binding mode;
   - argument and result types;
   - dispatch provenance, including compiler-known trait operations.
3. Make HIR execution and MIR lowering consume `CallableUse`; neither may reconstruct selection.
4. Add exact-set coverage across free calls, methods, associated functions, function values,
   trait defaults, qualified calls, equality, ordering, iteration and display.
5. Resume A4 of `WP-VALUE-REP-TOTAL` only after callable-use exactness passes.
6. Inventory and close the separately identified typed-mutation boundaries before closing the
   DEV-121 defect class.

#### Exit criteria

1. Every executable user-callable use has exactly one record; duplicates and omissions fail an
   invariant test.
2. Implicit and explicit dispatch install the checker-selected generic environment in the HIR
   oracle.
3. The total type-to-`Value` relation is enforced at parameters, returns, receiver boundaries,
   bindings and typed mutation without exemptions.
4. The frozen corpus and all engine comparisons remain green.
5. DEV-121 closes only with a class-level evidence statement, not one regression case.

#### Checkpoint evidence

**Deliberately incremental — this packet is not batched.** Each dispatch family in work item 4 (free
calls, methods, associated functions, function values, trait defaults, qualified calls, equality,
ordering, iteration, display) is its own checkpoint: `cargo test --lib`, the exact-set invariant
test, and the affected differential rows before the next family opens.

AS3 reaches its **semantic-complete checkpoint** when the full four-engine matrix and frozen corpus
are green; that checkpoint authorises AS4 to open. AS3 remains formally open until the shared Sprint
3 Tier-3 closeout satisfies the charter's work-package definition of done. This is the most
expensive checkpoint in the programme; budget it as wall-clock, not as a formality. DEV-121's class
closure is a class-level claim and may outlast the checkpoint; carry it explicitly rather than
declaring it satisfied by one green run.

#### Risks and escalation

Callable metadata is a semantic compiler contract. If the work changes overload selection, trait
semantics or the accepted/rejected program set, stop and use CE1/CE2 rather than folding the change
into this packet.

---

### AS4 — One authority for semantic type properties

#### Dependencies

- **AS3 at its green semantic-complete checkpoint.** AS4 consolidates authorities that AS3's
  callable metadata feeds; building it on unvalidated metadata would mean re-doing it. AS4 does not
  open until that checkpoint is green, while AS3's formal closure waits for the shared Sprint 3
  Tier-3 closeout.
- Execute the existing `WP-C7.8-RB0-MIR-Type-Property-Authority.md`; do not replace it with a fresh
  cleanup design.

#### Work

- Complete the required inventory for:
  - Copy classification;
  - runtime drop glue;
  - user-defined destruction;
  - stored-reference containment;
  - borrow-lifetime carrying;
  - user-nominal containment;
  - runtime representation.
- Distinguish differently worded semantic questions before consolidating implementations.
- Add equivalence/adversarial tests over the full type-variant set before deleting duplicates.
- Give lowering and backends one semantic authority surface.
- Preserve verifier challenge value: either use an independently implemented verifier predicate
  checked against the same declared matrix, or justify direct consumption where independence adds
  no evidence.
- Resolve or explicitly carry the iterator drop and function-pointer reference questions named by
  the existing packet.

#### Exit criteria

1. Every type property has one documented meaning and authority.
2. Near-neighbour predicates with different meanings are named so they cannot be substituted
   accidentally.
3. Adding a type/representation variant forces every applicable authority and evidence matrix to
   be updated.
4. Resource, iterator, reference, generic-drop and partial-move adversaries pass across HIR, MIR
   and native engines.
5. Any behavioural correction receives its own decision record; AS4 itself does not disguise one
   as refactoring.

#### Checkpoint evidence

Incremental, like AS3. Per property consolidated: the equivalence/adversarial matrix over the full
type-variant set **before** the duplicate predicate is deleted, then `cargo test --lib` and the
affected differential rows. Deleting a duplicate before its equivalence evidence exists is the
failure mode this ordering prevents. The reference-containment family (three known copies, one
already documented as disagreeing with the others on `FnPtr`) needs its disagreement resolved as a
recorded decision, not silently harmonised by whichever copy survives.

#### Campaign A exit gate

Campaign A passes only when AS0, AS1a, AS2, AS1b, AS3 and AS4 are complete and owner-reviewed. The
exit report must classify each criterion PASS, FAIL, DEFERRED-BY-DECISION or NOT-APPLICABLE and
include command-level evidence.

**Exit report drafted 2026-08-08: `audits/CAMPAIGN-A-EXIT-REPORT.md`, verdict CANDIDATE-PASS.**
All six packets are complete; 26 of 32 criteria PASS, four are PARTIAL and await an owner ruling
(§8 of the report), and the remaining two are carried from the sprint closeouts. The gate is **not**
passed and the binding restriction on structured-concurrency compiler/runtime work remains in force
until the owner rules and CI is green on the head commit.

**Reserved project-roadmap decision, not approved by Sprint 1:** the AS0 report now exists, so the owner
must decide whether to amend the project roadmap so structured-concurrency compiler/runtime
implementation may not begin until Campaign A passes. Package work not dependent on new compiler
semantics may continue under `ROADMAP.md`'s WIP limits.

---

## 6. Campaign B — maintainability and release readiness

### AS5 — Protocol, manifest and version-surface contracts

**Status: CLOSED 2026-08-07** (a–g). Sprint 1's classification held — tightening plus compatibility
correction plus correctness defect, no repository migration — but the packet counted *parsers*, and
the emit side carried three more defects (DEV-184) and the value model a fourth (DEV-185).
`AS5-OPENING-ANALYSIS.md` records the inventory, the CE9 decisions and the AS7 forward note.

#### Dependencies

- C8 is **CLOSED** (CD-385, 2026-08-06). It was CANDIDATE-COMPLETE at proposal time and this
  packet was gated on the decision; the decision is made, so AS5 is unblocked on that axis.
- EI3 of `WP-ENGINE-INDEPENDENCE.md` supplies the rustc/toolchain assumption inventory; AS5 decides
  how that proposal integrates with version and build-provenance contracts.
- **The "C8 closes first" branch applies.** AS5 preserves C8's protocol and interactive baseline
  while consolidating the shared JSON implementation used by non-LSP surfaces. It does not reopen
  C8's scope.
- **C8's protocol evidence has a stated limit AS5 must not lean on.** `GATE-C8-CLOSURE.md` §4
  records that protocol validation compared verdicts, not values, and that DEV-182 — the LSP parser
  decoding every escaped non-BMP character to the empty string — passed it. AS5's conformance corpus
  is where value-level agreement gets established; "C8's protocol tests pass" does not supply it.

#### Work

- Choose one strict JSON authority for package manifests, JSON-RPC/LSP, install manifests and
  compiler-generated JSON:
  - preferably a vetted library with dependency/risk review; or
  - one internal parser/serializer with RFC 8259 conformance tests.
- Reject trailing input and malformed escapes deterministically.
- Handle Unicode escapes and surrogate pairs correctly where JSON strings are accepted.
- Escape every required control character in every generated JSON surface.
- Preserve protocol-specific data models above the shared JSON layer.
- Replace manually remembered MIR/runtime-surface version bumps with a deterministic schema
  fingerprint or an exact-set test tied to the canonical surface.
- Add compatibility fixtures for old/new manifests and machine-readable diagnostics.

#### Exit criteria

1. Production code contains one JSON parser and one escaping authority.
2. A standard JSON test corpus and project-specific malformed cases pass.
3. C8's LSP protocol baseline proves rejection of trailing garbage and valid JSON for every
   diagnostic string; AS5's shared authority keeps that evidence green.
4. A runtime/MIR surface change cannot compile or pass tests without updating its compatibility
   identity.
5. Security-sensitive parsing decisions receive CE9 review where applicable.

#### Checkpoint evidence

Long implementation runs, with the corpus as a separate checkpoint from the consolidation. **AS5 is
behavioural**: stricter parsing changes which manifests and JSON-RPC messages are accepted, and no
type check reaches that. Checkpoints: the RFC 8259 conformance corpus; the first-party manifest set
re-parsed under the new authority with the Sprint 1 audit as the expected result; positive
round-trip tests for valid escapes and surrogate pairs; negative tests for invalid Unicode scalars
and unpaired surrogates; the LSP protocol suite, which C8's closure (CD-385) puts in scope.

If the Sprint 1 audit identified a repository migration, the manifest rewrite is its own commit with
its own checkpoint, taken before the parser tightens — not folded into the consolidation. A
compatibility correction likewise carries positive fixtures for every valid input class the current
parser rejected. Any value-divergence finding is repaired under its own DEV record with a
fails-before-the-repair test, not absorbed into the consolidation commit.

---

### AS6 — Quarantine extension-specific compiler knowledge

#### Dependencies

- Preserve the closed Part A behaviour of `WP-C9.1-EXTENSION-ISOLATION.md`.
- C9's second-artifact evidence gate remains closed unless independent evidence appears.

#### Work

- Inventory tensor/model/dtype/device branches in lexer/parser, resolver, checker, formatter,
  diagnostics and LSP.
- Move extension-owned names, type rules, methods and diagnostics behind sealed internal tensor
  modules/interfaces selected by the existing per-session `LanguageOptions`.
- Keep Core pass data structures extension-neutral where this can be done without a generic public
  abstraction.
- Add dependency/lint tests preventing new tensor imports in designated Core-only modules.
- Retain explicit frontend enablement and all C9.1 session-isolation tests.

#### Exit criteria

1. Core-only sessions load no tensor-owned name or semantic rule.
2. Central Core modules do not contain open-ended tensor spelling tables or method catalogues.
3. Tensor-enabled behaviour and ONNX verification remain unchanged for their documented scope.
4. No public extension/plugin/provider API is introduced.
5. Part B generic artifact-provider work remains blocked unless C9.3's independent evidence exists.

#### Checkpoint evidence

Long runs are acceptable; the risk is directional and needs both directions tested at each move.
Per surface quarantined (lexer/parser, resolver, checker, formatter, diagnostics, LSP): the C9.1
session-isolation suite, a Core-only session proving the tensor name is **absent**, and a
tensor-enabled session proving the same behaviour is **unchanged**. A quarantine that suppresses
tensor semantics passes the first test and fails the second; one that leaks passes the second and
fails the first. Neither is visible to `cargo check`.

AS6 completes before AS7 opens, so modularisation cuts on boundaries that are already clean.

---

### AS7 — Pass modularisation and compiler API boundary

#### Dependencies

Campaign A plus AS5 and AS6 must establish the boundaries first. AS7 does not invent them while
moving code.

#### Work

- Split the type checker by semantic ownership: inference, traits/method selection, patterns,
  ownership/borrowing, callable publication and extension checking.
- Split MIR lowering by calls, patterns, drop planning, intrinsics and metadata construction.
- Split the HIR interpreter into value model, executor, callable dispatch and Core-library
  operations.
- Replace ambient current-file/module/impl/generic state with scoped context objects where a
  missing restore can alter later work.
- Define a narrow supported compiler facade; make implementation modules `pub(crate)` unless an
  actual external consumer requires them.
- Move any remaining obsolete backend spikes out of default compiler builds, preserving historical
  evidence in documentation or a non-default spike crate. The Cranelift dependency retirement is
  taken in Sprint 1 as an isolated commit under its own audit gate (§5); it returns here only if
  that audit failed.

#### Exit criteria

1. No semantic behaviour or diagnostic structure changes in modularisation commits.
2. Dependency direction between submodules is documented and cycle-free.
3. Internal modules are not accidentally part of the supported public API.
4. Default dependency/build surfaces contain only active compiler architecture.
5. File-size reduction is reported as an outcome, not used as the acceptance criterion.

#### Checkpoint evidence

The second genuine marathon. Pure code motion is compiler-verified and needs only `cargo check`
between commits — but **ambient-state replacement is not code motion**. Replacing current-file,
current-module, current-impl and generic-environment state with scoped context objects can restore
the wrong context with every field present and every signature type-checking; this repo has already
paid for that class once, in file-provenance drift between `self.text` and item-level file metadata.

Split the packet accordingly: take the ambient-state conversions **first**, as separate commits with
`cargo test --lib` plus the provenance, generics and trait-resolution suites at each; then take the
file splitting as the marathon, where the checkpoint is genuinely `cargo check`.

**Exclusive tree ownership is a precondition.** Splitting a 14,000-line pass cannot survive a
parallel session editing the same file — take a worktree, or hold an explicit agreement that no
other session touches the declared ownership set for the duration.

---

### AS8 — Independent evidence, tooling scale and governance closure

#### Dependencies

- C8 gate exit: **done** (CD-385, 2026-08-06). AS8 is post-C8 performance/ownership work, not a
  substitute for C8's protocol and interactive semantic validation — and DEV-012 remains open for
  seven features, so AS8 must not be read as supplying that evidence either.

#### Work

- Consume the shared-fate register, evidence audit, engine-risk profiles and ranked mutation
  targets produced by `WP-ENGINE-INDEPENDENCE.md`; do not repeat its inventory under a second
  taxonomy.
- Add real compiler-source mutation trials for selected ownership, trap, drop, resolver and MIR
  verifier rules; observation/comparator mutation alone is insufficient.
- Establish line/branch coverage baselines for compiler crates and report uncovered semantic
  arms—without imposing an arbitrary percentage as a conformance claim.
- Run the external `stark-samples` suite as pinned independent application evidence. Record the
  suite commit and expectation manifest with the result; if it becomes a required CI gate, vendor
  or fetch an explicitly versioned artifact rather than depending on a developer home path.
- Profile LSP package analysis on representative multi-file projects.
- If evidence warrants it, add bounded debounce, cancellation and one-analysis-per-package cache
  ownership. Do not build full incrementality.
- Replace whole-package `ProjectAnalysis` duplication per open URI where measurement shows material
  cost.
- Compress `COMPILER-STATE.md` back toward the charter's current-state contract while preserving
  append-only history in an archive/ledger, and reconcile deviation statuses with executable
  evidence.
- Update `compiler-map.md`, `lib.rs` crate documentation and the canonical roadmaps at the campaign
  exit.

#### Exit criteria

1. Each differential claim names shared phases and at least one independent evidence source.
2. Selected source mutations are killed by the claimed suites; survivors are recorded as test
   gaps.
3. LSP changes are justified by before/after measurements and cancellation correctness tests.
4. Current compiler position is discoverable from the beginning of `COMPILER-STATE.md` without
   reconstructing chronology.
5. Architecture documentation matches production entry points and module ownership.

#### Checkpoint evidence

AS8 is assurance, not implementation, and cannot be batched with the work it challenges. It opens
only after AS6 and AS7 reach an implementation freeze with their Tier-2 checkpoints green. Its own
checkpoints are the mutation trials (a survivor is a recorded test gap, not a blocked packet), the
coverage baseline, the pinned samples-suite run, and the LSP before/after measurements. Sprint 4
becomes formally green only after AS8 and the Tier-3 closeout finish. Any LSP change AS8 makes on
the strength of its own measurements is ordinary implementation and takes ordinary checkpoints.

#### Campaign B exit gate

Campaign B passes only when AS5–AS8 are complete or explicitly deferred with owner-approved
evidence. Its report is a prerequisite for C10 release qualification, but it does not itself make a
stability or conformance claim.

---

## 7. Evidence cadence

Evidence runs at **three levels**. A full qualification cycle per packet would spend most of the
programme's time in gate ceremony; running nothing until a sprint ends would mean discovering on day
three that day one was wrong. The rule is: **one formal closeout per sprint, multiple coherent
commits inside it, and targeted tests at the checkpoints each packet names.**

### Tier 1 — inner loop (continuous)

- `cargo check`, or `cargo check --all-targets` when test code is in the ownership set;
- the one directly affected unit or integration test;
- no qualification, no differential, no native build.

This is the normal rhythm of a refactoring marathon. For AS2 and AS7's code-motion phase it is
almost the whole rhythm.

### Tier 2 — coherent checkpoint (at each packet's named boundaries)

- `cargo test --lib` — cheap when warm, and the default checkpoint everywhere;
- the focused integration tests named in that packet's **Checkpoint evidence** section;
- the affected four-engine differential rows where the packet changes semantics.

Checkpoints are not optional for packets that change observable behaviour, and they are not
substitutable by `cargo check`: source-location attachment (AS1b), pipeline selection (AS2), accepted
JSON (AS5), extension conditioning (AS6) and ambient-context restoration (AS7) are all invisible to
the type system. A local checkpoint commit is taken at each, so rollback costs one boundary rather
than a sprint.

### Tier 3 — sprint closeout (once per sprint)

- `cargo fmt --check`;
- `cargo clippy --all-targets -- -D warnings`;
- the full Rust suite through CI, or from an isolated clean worktree when explicitly required — not
  by running a broad shared-checkout command as an unrecorded substitute for CI;
- Core positive and negative fixture conformance;
- HIR/MIR/native debug/native release differential rows for affected semantics;
- tensor/extension tests when extension code is touched;
- deterministic outputs executed twice when identity, ordering or generated output is claimed;
- package/provider qualification when package loading, capabilities, runtime or build metadata is
  touched;
- the pinned external samples suite for sprints affecting accepted programs, ownership, execution,
  packages or engine agreement, when that suite is available;
- focused tests demonstrated to fail before the repair when the sprint closes a defect;
- updated deviations, coverage records, `COMPILER-STATE.md` and architecture documentation.

Provider-backed packages are built, not run through the interpreter. Provider crates receive their
own `--manifest-path` build/test rows where their sources are touched.

### Sprint-internal gates that are not closeouts

Three checkpoints carry gate authority without being sprint closeouts, because the work after them
would have to be redone if they were skipped:

| Gate | Position | Requirement |
| --- | --- | --- |
| AS3 semantic-complete checkpoint | inside Sprint 3, between AS3 and AS4 | full four-engine matrix and frozen corpus green before AS4 opens; AS3 remains formally open until Tier 3 |
| Semantic-green gate | Sprint 3 → Sprint 4 | Campaign A exit report accepted before AS6 opens |
| AS6+AS7 implementation freeze | inside Sprint 4, before AS8 | both packets' Tier-2 checkpoints green and the implementation ownership set frozen before assurance begins |

A sprint may also be interrupted at any point by the live-defect pre-emption rule in §3. A defect
repair taken mid-sprint keeps its own DEV/CD evidence and its own focused tests; it is not absorbed
into the sprint's closeout.

---

## 8. Programme success measures

The architecture-stabilisation programme succeeds when all of the following are demonstrable:

| Property | Measure |
| --- | --- |
| Source identity | one physical source has one logical identity; all spans resolve through `SourceId` |
| Relocation stability | identical package graphs at different roots produce identical logical MIR/build identities |
| Entry-point convergence | every tool reaches semantic analysis through one driver |
| Callable authority | every executable call has exactly one checker-published `CallableUse` |
| Runtime representation | every typed HIR boundary enforces the total `Ty`→`Value` relation |
| Type properties | Copy/drop/reference questions have documented, exact-set-tested authorities |
| Extension isolation | Core modules do not embed tensor-owned catalogues; no premature public framework exists |
| Protocol correctness | one strict JSON authority and mechanically checked compatibility surfaces |
| Maintainability | major passes have explicit ownership boundaries and a narrow public facade |
| Evidence independence | differential results state shared fate and are challenged by source mutation or independent fixtures |
| Tooling scale | LSP latency/cancellation/cache behaviour is measured and bounded without premature incrementality |
| Governance | current status, deviations and architecture documents agree with executable evidence |

No single metric, test count or green differential run is sufficient. The exit claim is that the
existing compiler architecture has stable, authoritative contracts—not that the language or public
toolchain has reached v1 stability.

---

## 9. Explicit non-goals

This programme does not authorise:

- new Core syntax or semantics;
- async/await, concurrency semantics or an HTTP server;
- a VM, JIT, LLVM migration or direct Cranelift backend;
- a new MIR, runtime ABI or value layout;
- a public compiler-plugin or generic artifact-provider framework;
- tensor productisation or broader tensor execution;
- full incremental compilation;
- compiler self-hosting;
- a Core conformance, stable compiler or public release claim.

Those remain governed by the specification, charter, C9/C10 gates and the consolidated project
roadmap.
