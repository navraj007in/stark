# AS8 opening inventory — what its dependencies actually are

**Read-only. No implementation.** AS8 is assurance written against the frozen AS6/AS7 result, so it
opens by establishing what already exists rather than by producing anything.

**Head:** `8712937`, immediately after AS7 CLOSED (CD-391). **Date:** 2026-08-08.

> **RESOLVED 2026-08-09 by owner ruling CD-392.** Option A approved: `WP-ENGINE-INDEPENDENCE` is
> **APPROVED without redesign** and its execution transferred administratively to the AS8 assurance
> phase. **AS0 remains closed and is not reopened.** The full EI0–EI6 packet runs, not a minimal
> subset — EI0 freezes the vocabulary before the register is built, and EI5's ranking includes
> `RUSTC_ASSUMPTION` and rustc-sensitive lowering, which EI3 supplies. AS8 may not select
> source-mutation targets until EI5 publishes the ranked list, and may not invent a second taxonomy.
> AS8's other evidence lanes proceed in parallel. The blocker analysis below is preserved as the
> record of why the ruling was needed.

## The blocker, stated first

AS8's **first work item** is:

> Consume the shared-fate register, evidence audit, engine-risk profiles and ranked mutation targets
> produced by `WP-ENGINE-INDEPENDENCE.md`; **do not repeat its inventory under a second taxonomy.**

```text
WP-ENGINE-INDEPENDENCE.md   Status: PROPOSED — owner approval required before execution
```

**It has never been executed, and none of its five execution outputs exist:**

```text
STARKLANG/docs/compiler/ENGINE-SHARED-FATE-REGISTER.md      absent
STARKLANG/docs/compiler/ENGINE-RISK-PROFILES.md             absent
STARKLANG/docs/compiler/RUSTC-ASSUMPTION-INVENTORY.md       absent
STARKLANG/docs/compiler/ENGINE-EVIDENCE-INDEPENDENCE.md     absent
STARKLANG/docs/compiler/engine-shared-fate.json             absent
```

Searched for the artefacts and for the packet identifiers `EI1`/`EI5` across every `.md`, `.toml`
and `.json` in the repository: the only mentions are inside the proposal itself and in
`WP-ARCHITECTURE-STABILIZATION.md`'s forward references to it.

**So AS8's mutation work has no ranked targets to consume, and the work item forbids inventing
them.** The two readings are:

```text
A  execute WP-ENGINE-INDEPENDENCE first (EI0-EI6, or the subset AS8 needs: EI1, EI2, EI4, EI5)
   then AS8 consumes its outputs as designed
B  AS8 selects its own mutation targets, which is exactly the second taxonomy the work item
   forbids and which WP-ENGINE-INDEPENDENCE §0 exists to prevent
```

**This was an owner decision, taken as option A — see the banner above.** It was not made here. It is not a technical obstacle — AS8 could
pick mutation targets tomorrow — it is a governance one: the packet says not to, and the reason it
says not to is that two competing engine-risk taxonomies would make every later independence claim
ambiguous.

Note also that `WP-ENGINE-INDEPENDENCE.md` is filed as an **AS0 subpacket**, and AS0 is CLOSED.
Executing it now under AS8 is a scope question in its own right.

## The other dependencies, measured

### Already satisfied — the pinned samples suite

AS8's work item:

> Run the external `stark-samples` suite as pinned independent application evidence. Record the
> suite commit and expectation manifest with the result; if it becomes a required CI gate, vendor
> or fetch an explicitly versioned artifact rather than depending on a developer home path.

`.github/workflows/ci.yml` already does this, and it is a required job:

```yaml
external-sample-suite:
  name: External sample suite (pinned)
  env:
    SAMPLES_REPO: navraj007in/stark-samples
    SAMPLES_SHA: b3b28e757f38d691e7309f168d1209e28ac459af
```

It checks out by SHA and then **verifies the pin resolved to the commit asked for**, because `ref:`
also accepts a branch and would silently float. No developer home path is involved. It was green on
`977b7a3` and `4c4311a`.

**AS8 inherits this satisfied, not open.** What remains is to *record the suite commit and
expectation manifest with the result* as evidence, which is reporting, not plumbing.

### Open — `COMPILER-STATE.md` compression

```text
COMPILER-STATE.md   12,816 lines
```

The work item is to compress it "back toward the charter's current-state contract while preserving
append-only history in an archive/ledger". The top of the file is now correct — CD-391 states the
current position without requiring chronology — so exit criterion 4 is *already* satisfiable by
reading; the compression is about the remaining 12,800 lines of dated records.

**Constraint carried from prior sessions: do not archive fresh or still-relevant records to hit a
size target.** The line count is not the goal; discoverability is, and criterion 4 measures that.

### Explicitly out of scope — DEV-012

```text
DEV-012 — VS Code extension UI interactively verified for 3 of 10 features (OPEN, NARROWED)
```

AS8's dependency note is unambiguous: AS8 is post-C8 performance/ownership work, **not** a
substitute for C8's protocol and interactive semantic validation, and **must not be read as
supplying DEV-012's evidence either**. Any LSP work AS8 does on the strength of its own measurements
is ordinary implementation taking ordinary checkpoints.

### Not yet measured

```text
line/branch coverage baselines      no tooling configured in CI; needs a decision on the harness
LSP package-analysis profiling      no benchmark exists; "representative multi-file projects"
                                    is undefined and needs naming before a before/after can mean
                                    anything
```

## What AS8 must adopt, carried from AS7

AS7's qualification recorded four defects in its own verification, plus two more found during
qualification and CI reading. The common shape:

> **A check that does not cover the thing being claimed cannot support the claim.**

and the compensating discipline that actually worked:

> **Introduce the violation on purpose and watch the check fail.**

**AS8 is entirely an evidence packet, which makes this its central risk rather than a footnote.** A
mutation suite that kills nothing looks identical to one that works. A coverage baseline computed
over the wrong target set looks identical to a correct one. Both would produce confident green
numbers that mean nothing — the same failure that produced four green-but-blind checks in AS7.

Concretely, AS8 should not report a mutation trial without first demonstrating that a **known-live
mutant is killed and a known-equivalent mutant survives**, and should not report a coverage figure
without stating which targets it was computed over.

## Recommended sequence, pending the ruling above

```text
0  OWNER DECISION on the WP-ENGINE-INDEPENDENCE dependency          <- blocks 2
1  record the pinned samples-suite evidence (satisfied; reporting only)
2  mutation trials, against EI5's ranked targets once they exist
3  coverage baselines, with the target set stated
4  LSP profiling, after naming the representative projects
5  COMPILER-STATE.md compression, preserving history
6  compiler-map.md / lib.rs docs / roadmap reconciliation at campaign exit
```

Items 1, 3, 4 and 5 are unblocked today. Item 2 is not.

## Branch

AS7 ran in the worktree `/Users/nexper/Documents/GitHub/stark-as7` on
`wp-arch-stability/as7-modularization`, which is where this inventory is written. **AS8 does not
need exclusive tree ownership** — it is assurance, not a 14,000-line file split — so whether it
continues on this branch or takes its own is an ordinary preference, not a precondition. Flagged
rather than decided.

The unresolved merge topology stands: PR #11 is a CI-only draft carrying 121 commits against
`develop`; PR #10 (`sprint-3`) is the merge path for AS0–AS6. Sprint 4 cannot close without settling
how AS7 and the sprint-4 branch land.
