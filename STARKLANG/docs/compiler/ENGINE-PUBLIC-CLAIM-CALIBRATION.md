# Public claim calibration

**Packet:** `WP-ENGINE-INDEPENDENCE.md` EI6, approved 2026-08-09 (CD-392). Inputs: EI1–EI5.

**Status: EI6 COMPLETE — approved wording published below; three corrections required, one
addition required.**

---

## The good news first: the prohibited claim is not made anywhere

```text
"STARK has three independent implementations of its semantics."
```

Searched `README.md`, `website/src/content.ts`, `CLAUDE.md`, `AGENTS.md` and both roadmaps for
"three independent", "independent implementation", "independent engine", "independently
implemented". **Zero occurrences.** Nobody has ever made the claim EI6 exists to prevent.

The language that *is* used is already careful in the right place:

> *"Programs are compared across four engine configurations … **each case pinned against the
> specification rather than against another engine**."* — `CLAUDE.md`, `AGENTS.md`
>
> *"Agreement alone is not the standard. Each case also pins its expected result against the
> specification, so three engines agreeing on the wrong answer fails."* — `website/src/content.ts`

That second sentence is the exact concern this packet formalised, written down before the packet
ran. EI6's job is therefore narrower than expected: three corrections and one addition, not a
rewrite.

## Correction 1 — "Four engines" is four *configurations*, and two of them are one engine

`website/src/content.ts` says **"Four engines, one answer"** and describes *"a reference
interpreter …, a mid-level IR interpreter, and native binaries built in debug and release."*

Native debug and native release are **the same engine at two optimisation levels**. They share the
lowering, the emitted Rust, and every authority in the register; they differ in a build profile.
Counting them as two engines inflates the independence claim by one.

`CLAUDE.md` and `AGENTS.md` already say **"four engine configurations"**, which is accurate. The
website should match.

```text
BEFORE  Four engines, one answer
AFTER   Three engines, four configurations, one answer
```

## Correction 2 — "each case pins its expected result against the specification" overstates runtime coverage

EI2 measured what is actually spec-derived:

```text
EV-SPEC-FIXTURES   SPEC_DERIVED, strongest control in the tree — but covers PARSE and SEMANTIC
                   CLASSIFICATION, not runtime semantics
EV-CORPUS-C6       handwritten + generated; every case states its expectation in the SHARED trap
                   vocabulary (mir::TrapCategory), so a corpus case cannot contradict ESF-TRAP-001
EV-DIFF-*          CROSS_ENGINE_DERIVED against the HIR oracle
```

So "each case pins its expected result against the specification" is true of the fixture corpus and
**not** uniformly true of runtime conformance cases. The claim should be scoped rather than dropped
— it is the strongest honest thing said in the public copy.

```text
BEFORE  Each case also pins its expected result against the specification, so three engines
        agreeing on the wrong answer fails.
AFTER   Agreement alone is not the standard: conformance cases are pinned against the
        specification, not against each other, so engines agreeing on the wrong answer fails.
        Where a rule is decided once and shared by every engine — Copy eligibility, destructor
        eligibility, trap categories — agreement cannot corroborate it, and those rules are
        tracked and separately checked rather than counted as three confirmations.
```

## Correction 3 — the supporting anecdote is load-bearing and should stay, verbatim

> *"That distinction is not theoretical. It is how a bound that every engine ignored equally, and an
> operation that completed where the specification required a trap, were both found and fixed."*

**Keep this.** It is a concrete instance of spec-pinning catching a shared-fate defect — precisely
the failure mode EI1 registers — and it is the public copy's strongest evidence that the
distinction is enforced rather than asserted. EI6 recommends no change.

## Addition — rustc's role is described nowhere, and criterion 2 requires it

Searched `README.md` and the website for "rustc", "generated Rust", "Rust toolchain": **zero
occurrences.** The native engine compiles generated Rust and inherits a trusted base from rustc
(EI3), and none of that is public.

Approved wording, from EI3's measurements:

> The native engine compiles STARK to safe Rust and builds it with `rustc`. That makes `rustc` an
> external check: it rejects generated code that violates Rust's borrow and move rules, and has
> caught real lowering defects that way. It is not a check on meaning — generated Rust can be valid
> and still say the wrong thing. Where Rust's rules differ from STARK's, STARK decides: arithmetic
> lowers to explicit checked operations rather than relying on the build profile, shifts do not use
> Rust's `checked_shl` because it validates only the shift count, and destruction order is STARK's
> own plan rather than Rust's drop order.

## Approved release language

For C10, release notes and announcement copy. Replaces the packet's suggested wording with one
calibrated to what EI1–EI5 measured:

> **STARK runs a program three ways, in four configurations: a reference interpreter that defines
> the semantics, a mid-level IR interpreter, and a native binary compiled through generated Rust in
> debug and release. Every maintained conformance case runs through all four and they must agree —
> on output, exit status, which destructor ran when, and the exact category and source location of
> any trap.**
>
> **The three paths have deliberately different roles rather than being independent
> reimplementations. They share one front end, and some semantic rules are decided once so the
> paths cannot drift apart. Where a rule is shared, agreement between paths cannot confirm it, so
> those rules are listed in a public register and checked separately — against the specification,
> by mutation testing, by executable gates, and by recorded residuals where no control yet exists.**

**A stronger claim may be made only when the register earns it** — that is, when the authorities
currently marked `INVISIBLE` acquire independent controls.

## Acceptance criteria

| Criterion | Status |
| --- | --- |
| public language matches measured independence | **Met by the corrections above.** The prohibited claim was never made; three overstatements identified, with replacements |
| rustc's role is accurately described | **Addition drafted** — currently absent from all public copy |
| shared-fate residuals are not concealed | **Met by design**: the approved wording states that shared rules exist, that agreement cannot confirm them, and that residuals are recorded. The register and residuals are in-repo and public |
| C10 and public announcement copy consume the approved wording | **OPEN — this is a handover, not something EI6 can close.** The wording is approved and published here; applying it to `website/src/content.ts` and C10 copy is the consuming change |

## Handover

```text
website/src/content.ts   corrections 1 and 2, plus the rustc addition
C10 release copy         consume the approved release language
CLAUDE.md / AGENTS.md    already accurate ("four engine configurations", spec-pinned) — no change
README.md                no engine-independence claim present — no change
```

EI6 does not edit the website in this commit: the packet's own instruction is that C10 and
announcement copy *consume* the approved wording, and changing marketing copy is a separate,
owner-visible act from approving what it should say.
