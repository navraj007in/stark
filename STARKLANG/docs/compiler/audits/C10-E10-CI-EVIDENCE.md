# C10 — E10 CI and platform evidence

**Exit criterion:** E10. **Date:** 2026-08-09.
**Qualified head:** `076b4dc` — the merge of PR #16 into `develop`, carrying every C10 packet.

> E10 is **not** "CI is green". It is *"CI is green at the head being qualified, with the mapping
> from claim to job recorded"* — and the second half is work no green run supplies.

---

# 1. The run, both workflows named

```text
31316841883   CI                        =>  success
31316841854   C7.8 Native Capabilities  =>  success

28 jobs, 0 non-success, across both workflows
```

**Both workflows are named because reading one for the other is an error I made twice today** —
reporting `dbddf21` green off its C7.8 result while its CI run was still in flight and about to
fail. §14.2 requires `commit + run id + job + platform`; the workflow name is part of that.

## 1.1 Overlap status — RECORDED, per §14.1a

```text
076b4dc      13:5x  ------------------------------  the qualified run
7e596b0      13:28 -> 13:49                          overlapping
f5e8a51      13:11 -> 13:30                          overlapping
681d339      13:06 -> 13:30                          overlapping
```

**Three branch runs overlapped this window**, all on `wp-c10/execution-plan`, all superseded. Per
§14.1a an overlapping **green** run is still green — and this one is green in every job, so no
re-run on a quiet branch is owed. Recorded rather than assumed, which is the whole point of the
rule.

---

# 2. The claim → job → platform map

Derived from `ci.yml`, not from memory.

| Job | linux-x64 | macos-arm64 | windows-x64 |
| --- | :-: | :-: | :-: |
| `fmt, clippy, test` | ✓ | ✓ | ✓ |
| `release package smoke` | ✓ | ✓ | ✓ |
| `C7 P1 REST workload` | ✓ | ✓ | ✓ |
| `first-party package qualification` | ✓ | ✓ | ✓ |
| `provider metadata/unit/resource/loopback` (C7.8) | ✓ | ✓ | ✓ |
| `C6.4 tier-1 qualification` | ✓ | ✓ | — |
| `C6.5 corpus replay` | ✓ | ✓ | — |
| `C6.4 windows tier-2 gap probe` | — | — | ✓ |
| `spec fixture conformance` | ✓ | — | — |
| `C6.5 mutation controls` | ✓ | — | — |
| `External sample suite (pinned)` | ✓ | — | — |
| `DEV-160 … under Miri` | ✓ | — | — |
| `C6.4 / C6.5 tier-1 agreement`, `ci-complete` | ✓ | — | — |

---

# 3. THE CORRECTION — a claim C10 made three times, and it understated the evidence

C10-0, the C10 plan and the E11 sweep all recorded:

> *"Spec-fixture conformance, the C6.5 mutation controls and the external sample suite are
> **linux-x64 only** … any release claim of the form 'conforming on the listed platform matrix'
> that rests on those jobs is claiming more than the evidence covers."*

**That is true of the dedicated JOBS and false of the TESTS.**

`build-and-test` runs `cargo test --workspace --all-targets --all-features` on **all three
platforms**. The Windows job of CI run `31312610019` ran **213 test binaries**, including every
suite the claim called single-platform:

```text
conformance.rs             three_engine_differential.rs   mir_differential.rs
c61f_structural_copy.rs    c6_generated_corpus.rs         c6_mutation.rs
c10b_robustness.rs         c10c_security.rs               dev214_expression_depth.rs
```

**What the linux-only jobs actually add is tooling and consistency checking, not conformance
testing:** spec-regeneration sync, fixture-extraction sync, coverage-database consistency, the
evidence report, Miri, and the pinned external suite.

## 3.1 The real platform gaps, which are narrower and different

```text
C6.4 tier-1 qualification    linux + macos. Windows gets the Tier-2 GAP PROBE instead
C6.5 corpus replay           linux + macos. So the cross-engine corpus REPLAY RECORD — and the
                             tier-1 agreement comparison built from it — has no Windows arm
External sample suite        linux only. The one EXTERNALLY_DERIVED control is single-platform
Miri                         linux only
```

**Windows therefore lacks the platform-qualification RECORD and the cross-engine corpus replay —
not the conformance suite.** That is a materially different, and much smaller, limitation.

## 3.2 Why this error is worth as much attention as an overstatement

**It understated the evidence, and C10-Q would have under-claimed.** A qualification campaign that
quietly claims less than it can prove is still reporting something untrue, and the correction was
only found because E10 required reading an actual Windows job log rather than inheriting a sentence.

Three greps returned false negatives on the way — a shell escape, a case mismatch, and an ANSI reset
sitting between `Running` and the path. **Two of them reported a confident `0`**, which would have
entrenched the wrong claim rather than corrected it.

---

# 4. What E10 establishes

```text
ESTABLISHES  076b4dc is green in all 28 jobs across both workflows, overlap status recorded
ESTABLISHES  the claim -> job -> platform map, derived from ci.yml
ESTABLISHES  conformance, differential and mutation-control SUITES run on all three platforms
CORRECTS     the "linux-x64 only" claim, in three documents, in the pessimistic direction

DOES NOT     make the C6.4/C6.5 qualification records multi-platform — Windows still has no
             tier-1 qualification record or corpus replay arm, and C10-Q must say so
DOES NOT     give the external sample suite a second platform
DOES NOT     extend to tier-3 x86_64-apple-darwin, which no job exercises at all (C10-0 F1)
```
