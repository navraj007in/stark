# Gate 7's project policy — superseded

**Date:** 2026-08-04
**Supersedes:** the *project-policy* portion of [`gate7-decision.md`](gate7-decision.md)
(RETAIN AS RESEARCH LANGUAGE, owner-confirmed 2026-07-16)
**Does not supersede:** that memo's tensor-track verdicts, which still stand — see §3.

---

## 1. What is retired

Gate 7 recorded three separable things under one heading. Only the third is retired:

```text
Tensor-track technical verdict      POSITIVE     STANDS  (§3)
Tensor productisation verdict       DEFER        STANDS  (§3)
Current project policy              RETAIN AS RESEARCH LANGUAGE   RETIRED (this document)
```

The retired policy carried a scope limit with it: that the decision authorised **only** a
`stark verify` external-validation track as further work. That sentence is the one doing active
harm, because it is repeated in documents a newcomer reads first, and it describes a project that
stopped nine months of work ago.

Gate 7 also recorded a fourth line — **broader language verdict: UNRESOLVED**, noting that
concurrency, backend and platform questions were outside what it tested. That line was not wrong.
It was answered, by evidence gathered after it.

## 2. What retired it

None of this was a reversal. The policy was overtaken incrementally, by work that was itself
authorised, and the label was simply never revisited.

| Evidence | Closed | What it settled that Gate 7 could not |
| --- | --- | --- |
| Gate C7 — native compilation, debug and release, on Linux, macOS and Windows | 2026-07-31 | STARK compiles and runs general-purpose programs natively on three Tier-1 platforms, over a qualified standard-library subset. Gate 7's native evidence was a single tensor deployment host. |
| Gate C8 — compiler-backed language services (LSP, VS Code) | candidate-complete | Editor integration exists and is backed by the compiler's own analysis, not a parallel model. |
| HC0–HC13 — an HTTP/1.1 and HTTPS client written in STARK | 2026-08-03 | The language is expressive enough to write a network client *in itself*, qualified against adversarial peers on three platforms. This is the single largest thing Gate 7 had no way to anticipate. |
| 24 first-party packages, capability-declared host access, native providers | ongoing through 2026-08-03 | There is a package ecosystem, a provider ABI, affine host resources and cross-provider ownership transfer. |
| Installer Phase I — release archives, platform installers, versioned install tree, `stark doctor` | 2026-08-03 | The toolchain can be installed and verified off a checkout. |
| [`ROADMAP.md`](../../ROADMAP.md) — the consolidated forward plan | adopted 2026-08-03 | The owner has adopted an application-platform programme: operability, security and artifacts, a REST server, structured concurrency, persistence, ecosystem. A research-only policy cannot coexist with an adopted roadmap that contradicts it. |

## 3. What still stands, and is not weakened here

**Gate 7's tensor-track verdicts are untouched.**

- **Technical: POSITIVE.** STARK carries runtime-symbolic dimensions, proves compile-time
  shape-relationship facts, detects artifact drift at deploy time, and needed far less application
  code than the strongest typed-Rust comparator, with both producing bit-exact identical inference.
  That evidence is unaffected by anything since.
- **Productisation: DEFER.** *This remains correct.* The gating condition Gate 7 named — external
  developers using the verifier — has still not been met. (Gate 7's memo cites "§13 of the
  proposal" for this; that pointer is wrong and predates this document. The protocol is
  `VERIFIER_VALIDATION_TRACK.md` **§8**, "Human-validation protocol", with the pass/fail bar in
  **§11**, "Exit criteria". §13 is that proposal's work-package list.)
  No external tensor adoption evidence has been gathered since 2026-07-16, and the tensor track has
  not moved. **Nothing in this document authorises tensor productisation**, and a reader looking for
  permission to start it will not find it here.

Retiring a project-wide label attached to a tensor-track experiment is not the same as changing
that experiment's findings, and this document deliberately does neither more nor less.

## 4. The position that replaces it

> **STARK is a pre-alpha general-purpose language with a working implementation, developed against
> an adopted application-platform roadmap.** Its compiler front end, semantic analysis, reference
> and MIR interpreters, and native compilation are working over a qualified subset. It has a
> first-party package ecosystem, capability-declared host access, and an installable toolchain. It
> is not production-ready, offers no stability guarantees, and expects breaking changes.
>
> The **tensor/model extension remains a deferred research track**, on Gate 7's own terms and
> awaiting Gate 7's own gating evidence.

Current status always comes from [`COMPILER-STATE.md`](../../COMPILER-STATE.md) for compiler work
and [`ROADMAP.md`](../../ROADMAP.md) for everything else. This document fixes a stale *policy*; it
is not a status page and must not be cited as one.

## 5. Scope discipline after this

Retiring "research language" removes a label, not the requirement for authorisation. Work outside
the current gate still needs a roadmap-governed proposal:
[`COMPILER-CHARTER.md`](../../STARKLANG/docs/compiler/COMPILER-CHARTER.md) §1.6/§6 for compiler
work, [`ROADMAP.md`](../../ROADMAP.md) §12 for current non-goals.

The `stark verify` external-validation track described in
[`VERIFIER_VALIDATION_TRACK.md`](../../STARKLANG/docs/proposals/VERIFIER_VALIDATION_TRACK.md)
remains a live proposal. Its authorisation cited the retired policy; it now stands or falls on the
roadmap's terms like any other proposal, and it is still the named gate for tensor productisation.
