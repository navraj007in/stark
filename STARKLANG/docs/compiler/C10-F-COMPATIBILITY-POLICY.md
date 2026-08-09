# C10-F — compatibility and version policy

**Packet:** C10-F, WP-C10.5. **Date:** 2026-08-09.
**Candidate SHA:** `33a8608` — descendant of `develop` `1d20123`. Every claim below was read from
that commit (plan §14.1b).
**Consumes:** C10-A2 (dashboard), C10-B, C10-C, C10-D, C10-E.

> **The rule (plan §13.1): prefer narrow, evidence-backed commitments over "stable forever".**
> Every promise names the evidence that supports it. **A promise with no evidence citation is
> deleted, not softened.**

Each axis is exactly one of **COMMITTED** (with evidence), **UNCOMMITTED** (with what would be
needed), or **NOT APPLICABLE** (with the reason). There is no fourth option and no partial credit.

---

# 1. The axes

## 1.1 STARK language version — **UNCOMMITTED**

**There is no language-version constant in the tree.** Core v1 is identified by the normative
specification set, not by a number a program or tool can read.

```text
to commit    define the identifier, decide whether it is observable to a program, and decide
             what a future Core v2 does to a v1 source. None of the three is decided today
```

**Deliberately not invented here.** C10 qualifies what exists; minting a version scheme is language
design, and §3.2 forbids it inside a qualification campaign.

## 1.2 Compiler version — **COMMITTED, narrowly**

```text
starkc 0.1.0                 starkc/Cargo.toml
```

**Commitment: pre-1.0 Cargo semantics — anything may change in any release.** That is a real promise
because it is the *absence* of one, stated explicitly rather than left to be assumed from a version
number that readers routinely over-read.

## 1.3 Core language compatibility — **UNCOMMITTED**

**This is the axis a release most wants to promise, and the evidence does not support one.**
C10-A2's dashboard, over the declared 168-rule population:

```text
PRECISE-C211           36     function-precision positive AND negative evidence
RESOLVED-BY-TREE       20     a test function cites the rule
CORPUS-OR-FILE-LEVEL   27
IMPLEMENTATION-ONLY     1
UNRESOLVED             84
```

**56 of 168 rules carry function-precision evidence.** A compatibility promise over Core would be a
promise about all 168.

```text
to commit    resolve the 84 UNRESOLVED against the tree (C10-A2 names them and shows the method),
             then decide which subset is promised. The subset can be narrow and still be worth
             stating — "these 56 rules will not change without a major version" is a real promise
```

## 1.4 Optional extension compatibility — **NOT APPLICABLE**

The tensor extension is deferred research on Gate 7's terms (productisation DEFER). **No
compatibility promise is made about `tensor` v0.1, and none is withheld pending evidence — the
track is not being productised.** C10-E's OD-8 appendix could not even measure it offline.

## 1.5 MIR version and runtime surface — **COMMITTED**

```text
MIR_VERSION           "0.4"        src/mir/mod.rs:59
MIR_RUNTIME_SURFACE   "0.1-A14"    src/mir/mod.rs:92 — bumped INDEPENDENTLY of MIR_VERSION
```

**Commitment: a consumer rejects a program stamped with an unsupported runtime surface, before any
body is verified.**

```text
evidence   MIR-0017 / V-SURFACE-1, src/mir/verify.rs:83
           tests/mir_verify.rs::rejects_unsupported_runtime_surface — stamps "0.1-A999" and
           requires the rejection
```

**This is a rejection promise, not a stability promise.** The versions may move freely; what is
promised is that a mismatch fails loudly instead of executing something the consumer does not
understand.

## 1.6 Runtime ABI — **COMMITTED, same shape**

A generated binary carries the runtime version it was generated for, and the prologue reports a
mismatch against the runtime actually linked
(`emit_program.rs:146`: *"stark-runtime version mismatch: generated for runtime {}, linked against
{}"*). Evidence: `tests/c63_closure_evidence.rs`.

**Again a rejection promise.** No claim is made that ABI 0.1 will be honoured by a future runtime.

## 1.7 Native Provider ABI — **COMMITTED, and the strongest of the three**

```text
ABI_VERSION "0.1"    stark-provider-abi/src/lib.rs:6
```

Provider discovery refuses both a version mismatch and a checksum mismatch
(`provider_manifest.rs`: `VersionMismatch`, `ChecksumMismatch`), and the source states the reason —
*"Reproducibility requires the two to agree exactly — 'close enough' is how a build stops being
repeatable"* and *"The provider on disk is NOT the provider that was approved."*

```text
evidence   tests/p02_external_provider_trust.rs
```

**Commitment: exact agreement is required — version AND content hash.**

## 1.8 Generated artifact and build compatibility — **UNCOMMITTED**

`build.json` records `build_key`, `compiler_version`, `mir_version` and target fields, and the build
cache keys on them. **What is absent is a statement of what a consumer may rely on across
versions** — the fields exist for reproducibility within a version, and no cross-version contract is
defined.

```text
to commit    decide which build.json fields are a stable contract and which are diagnostics
```

## 1.9 Diagnostic compatibility — **UNCOMMITTED, and C10 found two reasons why**

Charter §1.6 rule 16 makes diagnostics part of behaviour: code, primary span, related spans, notes,
help text and machine-readable form must remain testable and deterministic. C10-B verified
determinism (§9.3) — the same source twice yields byte-identical diagnostics.

**Determinism is not stability, and two C10 findings bear directly on any stability claim:**

```text
DEV-182 (C8)   the LSP JSON parser silently decoded every escaped non-BMP character to the empty
               string, and PASSED protocol validation, because both sides reported success. A
               diagnostic contract tested by verdict rather than by VALUE is not tested
C10-R1 (C10-D) keyword identity is controlled only coarsely, by parse failure. Nothing pins the
               token a word maps to, so nothing pins the diagnostic a wrong mapping would produce
```

```text
to commit    decide separately for CODES (plausibly stable), SPANS (plausibly stable), and TEXT
             (pre-1.0, almost certainly not). Promising all three together is how a project ends up
             unable to improve a message
```

## 1.10 Platform and toolchain support — **COMMITTED for Tier-1 only, and one target must be named as unexercised**

Read from `starkc/target-matrix.json`, whose compiler-owned fields are pinned to `src/target.rs` in
both directions:

| triple | tier | CI | promise |
| --- | --- | --- | --- |
| `aarch64-apple-darwin` | tier-1 | full | **COMMITTED** |
| `x86_64-unknown-linux-gnu` | tier-1 | full | **COMMITTED** |
| `x86_64-pc-windows-msvc` | tier-2 | partial — gap probe, smoke, P1, packages | **COMMITTED as Tier-2**, with the gap probe's scope |
| `x86_64-apple-darwin` | **tier-3** | **NONE** | **UNCOMMITTED — packaged and never executed** |

**C10-0's finding F1 lands here.** The release tooling builds an archive and both installers for
`x86_64-apple-darwin`, and no CI job has ever run a test on it. **C10-Q may not include tier-3 in
any conformance claim**, and the release notes must say what tier-3 means: named by the compiler,
packaged, unexercised.

### 1.10a Two different MSRV claims, and only one is enforced

```text
ENFORCED    MINIMUM_RUSTC_VERSION = "1.85.0"  (native_toolchain.rs:6)
            `stark build` refuses a USER's rustc older than 1.85 — a runtime check with a
            diagnostic

UNVERIFIED  rust-version = "1.85"  (starkc/Cargo.toml:8)
            the claim that STARKC ITSELF still builds on 1.85. Nothing checks it: CI uses
            `dtolnay/rust-toolchain@stable`, which resolved to 1.93.0 today, so the compiler could
            have adopted a 1.90 feature and no job would notice
```

**Commitment: the enforced check only.** The Cargo `rust-version` field is **UNCOMMITTED** until a
CI job builds on 1.85 — one job, named here as the concrete step.

**CI's toolchain floats**, which is fine for a gate and not fine for "built with". C10-Q records the
actual rustc of the run it cites, never the action name.

## 1.11 Deprecation policy — **UNCOMMITTED**

None exists. **Stated rather than drafted:** a deprecation policy is a promise about future
releases, and a project at 0.1.0 with 26 open compiler deviations has no basis for one yet.

## 1.12 Pre-1.0 versus stable — **COMMITTED**

**STARK is pre-alpha and pre-1.0, and C10 does not change that.** Gate C10 qualifies what the
compiler can claim; it does not confer stability. Every "COMMITTED" above is a commitment about
*current behaviour and its enforcement*, not a forward promise across versions — except where §1.5,
§1.6 and §1.7 explicitly promise *rejection on mismatch*, which is a property that gets more useful
as versions move, not less.

## 1.13 Release signing, checksum and authenticity — **UNCOMMITTED**

```text
COMMITTED    INTEGRITY. `stark doctor` re-hashes every payload file against manifest.json,
             detecting corruption and partial extraction
UNCOMMITTED  AUTHENTICITY. Anyone who can replace the payload can replace the manifest and its
             sidecar with it
```

C10-C class C. **A public distribution needs a signed manifest, a trusted release key, verification
before installation, and platform notarisation. None exists.** C10-Q may not describe the
distribution as verified or trusted.

## 1.14 Capability vocabulary — **UNCOMMITTED, and EVIDENCED AS BREAKING** (updated 2026-08-09)

This axis was recorded as NOT APPLICABLE while the work sat on an unmerged branch. **It is now
evidenced, and the evidence is a demonstrated break.**

Vocabulary v1 renames the pre-v1 capability names. The external sample suite — the register's only
`EXTERNALLY_DERIVED` control — failed on the integrated tree:

```text
error: no provider supplies capability `process.args` for target `aarch64-apple-darwin`
  STARK knows these capabilities: clock, environment-read, filesystem-read, filesystem-write,
  network-client, network-listen, randomness
```

**The rename is intentional and documented** (`spec/packages/capabilities.md` §Migration), and the
spec is explicit that a capability is never silently renamed *within* a vocabulary version — so
renaming *across* versions is legitimate, and the absence of an alias shim is deliberate rather than
an oversight.

```text
COMMITTED     `stark.lock` records `capability_vocabulary`, so the vocabulary a build resolved
              under is durable and a later mismatch is detectable rather than silent
UNCOMMITTED   what a compiler does with a lockfile written under an EARLIER vocabulary. No
              behaviour is defined, and the break above shows the question is not academic
to commit     decide the cross-vocabulary contract: reject, migrate, or interpret an older
              capability as the union of its successors — which §Migration already contemplates
```

**This is the first demonstrated compatibility break against an external consumer**, and C10-Q must
state it. It was invisible to every in-repo test, because the compiler's own 28 first-party packages
were migrated in the same branch that made the change.

---

# 2. Summary

```text
COMMITTED        compiler version (pre-1.0 semantics)
                 MIR version + runtime surface   — rejection on mismatch, tested
                 runtime ABI                     — rejection on mismatch, tested
                 Native Provider ABI             — exact version AND checksum, tested
                 platform support                — Tier-1 two targets; Tier-2 Windows
                 enforced minimum user rustc     — 1.85.0
                 pre-1.0 status
                 release INTEGRITY

UNCOMMITTED      STARK language version          — no identifier exists
                 Core language compatibility     — 56 of 168 rules function-evidenced
                 generated artifact contract     — no cross-version statement
                 diagnostic compatibility        — split codes/spans/text before promising
                 starkc's own MSRV               — declared 1.85, never built on 1.85
                 deprecation policy              — none
                 release AUTHENTICITY            — integrity only
                 tier-3 x86_64-apple-darwin      — packaged, never executed

                 capability vocabulary           — cross-vocabulary behaviour undefined, and
                                                   EVIDENCED AS BREAKING by the external sample
                                                   suite. §1.14. (Its lockfile stamp is committed,
                                                   listed above)

SPLIT            capability vocabulary           — the one axis that is BOTH: `stark.lock` records
                                                   the vocabulary a build resolved under
                                                   (COMMITTED), while what a compiler does with a
                                                   lockfile from an EARLIER vocabulary is
                                                   undefined (UNCOMMITTED)

NOT APPLICABLE   tensor extension compatibility  — deferred track, not productised
```

**Fourteen axes — six COMMITTED, seven UNCOMMITTED, one NOT APPLICABLE**, with capability
vocabulary split across both committed and uncommitted because it genuinely is both. The count is
per **§1.x section heading**, which is mechanically checkable:

```bash
grep -oE "^## 1\.[0-9]+[a-z]? [^—]*— \*\*[A-Z ]+" C10-F-COMPATIBILITY-POLICY.md | sed 's/.*\*\*//' | sort | uniq -c
```

*(Corrected twice on 2026-08-09. It first read "eight, eight, two", which went stale the moment the
capability-vocabulary axis moved. The replacement — "nine, nine, one" — counted ITEMS in the summary
block while claiming "across fourteen axes", two different granularities that do not reconcile. It
now counts sections, and names the command that checks it. **A summary sentence outliving its list
is the most common review finding there is, and this one produced two findings in one day.**)*

That ratio is the honest state of a 0.1.0 compiler, and every non-commitment names what would be
needed rather than deferring vaguely.

---

# 3. What C10-F does NOT do

```text
NOT drafting a deprecation policy    §1.11 — no basis at 0.1.0
NOT minting a language version       §1.1 — that is language design, forbidden by §3.2
NOT promising Core compatibility     §1.3 — the evidence covers 56 of 168 rules
NOT softening a promise to keep it   the rule is delete, not soften
NOT a release statement              C10-Q derives that from all packets, under CE8
```
