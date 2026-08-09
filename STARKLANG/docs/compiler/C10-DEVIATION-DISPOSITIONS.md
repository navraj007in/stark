# C10 — E9 deviation dispositions

**Exit criterion:** E9. **Date:** 2026-08-09. **Candidate:** `29ce610`.
**Authority:** CD-021, carried into WP-C10.7 — *"every open deviation carries either an owning
gate/WP or an explicitly recorded accepted-indefinitely disposition; an open deviation with no owner
blocks the release decision."*

**OD-3 requires three separately countable populations.** Only **A** is CD-021's denominator; **B**
constrains release wording; **C** constrains the strength of evidence claims and asserts no defect.

```text
A  compiler deviations      23 live-OPEN + 1 accepted-indefinitely + 1 dormant
                            (24 at first count; DEV-005 CLOSED on reproduction — §2.5)
B  release/distribution      5
C  assurance residuals      20
```

Every row carries the six fields WP-C10.7 requires: **ID, current behaviour, impact,
release-claim consequence, owner, disposition**.

---

# 1. The axis that organises this register

**Over-acceptance is qualitatively worse than refusal, and the register is sorted by it.**

```text
ACCEPTS WHAT THE SPEC FORBIDS   a wrong program compiles. INVISIBLE to the user, and it is the
                                only class that can make a conformance claim FALSE rather than
                                merely narrow
REFUSES WHAT THE SPEC ALLOWS    a right program is rejected. Visible, diagnosable, and it makes a
                                conformance claim NARROWER, not wrong
WRONG REPRESENTATION            correct acceptance, incorrect execution or storage
OPERATIONAL                     the compiler is right and the environment defeats it
```

Exactly **one** deviation is in the first class.

---

# 2. Population A — compiler deviations

## 2.1 OVER-ACCEPTANCE — the one that can falsify a conformance claim

| | |
| --- | --- |
| **ID** | **DEV-177** |
| **Current behaviour** | The checker accepts a method generic that duplicates its impl's generic |
| **Normative expectation** | `04-Semantic-Analysis.md` **NAME-SHADOW-001** forbids it |
| **Impact** | **A program the specification forbids is accepted.** It runs only because the duplicate happens to resolve consistently |
| **Release-claim consequence** | **A bare "conforming" claim over `NAME-SHADOW-001` would be FALSE.** Every other row narrows a claim; this one contradicts one |
| **Owner** | compiler track — bounded name-resolution packet |
| **Disposition** | **MUST be named explicitly in the release statement.** Not accepted-indefinitely: an over-acceptance that is left unnamed is how a conformance claim becomes untrue rather than incomplete |

## 2.2 REFUSES WHAT THE SPEC ALLOWS — narrows the claim, cannot falsify it

| ID | Current behaviour | Release-claim consequence | Owner | Disposition |
| --- | --- | --- | --- | --- |
| **DEV-172** | Signed minimums (`Int8::MIN` etc.) and `UInt64::MAX` cannot be written as literals | Numeric-literal conformance is narrower than `03-Type-System.md` states | compiler track — literal/parse packet | **NAMED DEVIATION.** A refusal, not an acceptance; no soundness impact |
| **DEV-168** | A qualified call to a compiler-known trait's method has no MIR lowering | Native/MIR execution is narrower than `TYPE-METHOD-001` | compiler track — MIR lowering | **NAMED.** Caught at lowering, before any code is emitted |
| **DEV-181** | A borrow taken by an assignment's own RHS blocks the assignment — `x = x.method()` is refused | Borrow-check conformance narrower; **the idiom is everyday**, so this is the most user-visible row here | compiler track — borrowck packet (same mechanism as DEV-137) | **NAMED, and flagged as the highest user-friction item** |
| **DEV-167** | `Display::fmt` has no method-form `to_string()` | Stdlib surface narrower | compiler track | **ACCEPTED — deferred by decision** (already recorded) |
| **DEV-140** … **DEV-145** | Six CD-342 "layer defects": `Vec::` methods outside the lowering set, `HashMap` over a user-`Drop` value, droppable composite with a borrowed element, `assert_eq` on a user type, `for` over a non-range/non-`Vec` iterator, method on a peeled type | **These six DEFINE the supported native subset.** Any native claim is a claim about their complement | compiler track — registered CD-342 | **NAMED AS A SET.** They are the boundary of the claim, not exceptions to it |

## 2.3 WRONG REPRESENTATION — accepted and executed incorrectly

| ID | Current behaviour | Release-claim consequence | Owner | Disposition |
| --- | --- | --- | --- | --- |
| **DEV-180** | The HIR interpreter flattens `&mut self` into owned receiver storage. **CONFIRMED, reachable from accepted Core v1 programs** | Constrains the **HIR-oracle** claim specifically — the engine that defines "expected" for the other two | compiler track — interpreter receiver lowering | **NAMED.** The oracle having a representation defect is worth stating plainly rather than folding into a list |
| **DEV-157** | The native backend has no representation for `MirTy::Never`; `Err(_) => panic(..)` in value position | Native execution narrower | compiler track | **NAMED** |
| **DEV-160**, **DEV-162** | Place-granular borrows: whole-value projections, and reads through a whole-value accessor | Borrow-check conformance narrower | compiler track — DEV-154 family | **NAMED.** DEV-160 is guarded in CI by the Miri job |
| **DEV-178** | Generic context not retained for associated-fn calls or function values | Generic conformance narrower | compiler track | **NAMED** |
| **DEV-122** | Span source-identity gap (guarded; instance fixed CD-306) | Diagnostic provenance, not program behaviour | compiler track | **NAMED, low severity** |

## 2.4 OPERATIONAL — the compiler is right, the environment defeats it

| ID | Current behaviour | Release-claim consequence | Owner | Disposition |
| --- | --- | --- | --- | --- |
| **DEV-161** | An ambient `CARGO_TARGET_DIR` breaks every native build | **None on conformance.** A documented environment requirement | compiler track / docs | **ACCEPTED-INDEFINITELY** with a documented requirement. It is a trap for tooling, not a defect in the language |
| **DEV-159** | A native build can race its own dependency build | Build reliability, not conformance | compiler track — build orchestration | **NAMED** |
| **DEV-120** | Native call-depth exhaustion is a bounded host limitation | Documented limit | WP-C7.9 Packet F | **ACCEPTED — documented** (already recorded) |
| **DEV-156** | `stark fmt` relocates a field's doc comment | Formatter, not the compiler | compiler track — formatter packet | **NAMED, tooling only** |
| **DEV-186** | The LSP transport allocates an unbounded `Content-Length` before parsing | **Also a C10-C class-C DoS surface (S13).** Editor-facing | compiler track — LSP packet | **NAMED.** C10-B's test is written so a future limit flips it |

## 2.5 Already dispositioned by OD-7, restated for completeness

| ID | Disposition |
| --- | --- |
| **DEV-005** | ~~OPEN, accepted~~ → **CLOSED 2026-08-09. IT DOES NOT REPRODUCE.** The condition OD-7 attached found exactly what it was for. `check` and `run` both gate on errors and both report the warning; `run` executes. Negative control: an error still refuses both, so the gate is intact and merely no longer fires on warnings. Removed as a side effect of **AS2's one-pipeline** work. **C10-Q must NOT name it** — naming a deviation that no longer exists is its own false claim |
| **DEV-083** | OPEN, **ACCEPTED-DEFERRED** — impl-head concrete position vs unresolved receiver argument. Constrains the Core Stable claim; does not block |
| **DEV-011** | **ACCEPTED-INDEFINITELY** — doc comments as trivia. No normative requirement demands otherwise |
| **DEV-179** | **DORMANT** — unreachable while iterator `map`/`filter` is refused by `E0105`. Not counted live |

**Population A: 24 live-OPEN, every one now carrying an owner and a disposition. CD-021 is
satisfied for A.**

---

# 3. Population B — release/distribution

| ID / item | Release-claim consequence | Owner | Disposition |
| --- | --- | --- | --- |
| **DEV-165** | `connect_timeout` accepted and ignored — named a public-release blocker by `ROADMAP.md` §1 | platform track (HTTP) | **NAMED EXCLUSION.** Not a compiler-conformance deviation |
| Standalone toolchain | PARTIAL — payload carries compiler/runtime/ABI, and now the provider set (PR #15) | platform track | **NAMED** |
| Offline package build | Improved by PR #15; not re-verified by C10 | platform track | **NAMED, unverified by C10** |
| Signed distribution | **INTEGRITY, not AUTHENTICITY.** No signed manifest, no release key, no notarisation | platform track | **NAMED EXCLUSION.** C10-Q may not describe the distribution as verified or trusted |
| **tier-3 `x86_64-apple-darwin`** | Packaged with an archive and both installers; **exercised by no CI job** | compiler track | **NAMED EXCLUSION.** Excluded from every conformance claim |

---

# 4. Population C — assurance residuals

**These constrain the STRENGTH of evidence claims and assert no defect.** Collapsing them into A
would convert "we cannot see this" into "we are broken here" — the distinction AS8 paid for.

| Residual | Constrains | Disposition |
| --- | --- | --- |
| `AS8-R2` | `ESF-TRAP-001a` has no control, **and none is constructible** | **PERMANENT.** Named in the release statement |
| `AS8-R10` | `ESF-TRAIT-001` has no control of any kind | Trait-contract claims rest on no mutation evidence |
| `AS8-R13` | Non-`pub` re-export visibility has no control anywhere. **Re-confirmed at the candidate by C10-R** | Visibility claims narrower |
| `AS8-R14` | `may_need_drop`'s HostResource arm unguarded. **Re-confirmed by C10-R** | Verifier claims narrower |
| `AS8-R1/R4/R5/R6/R8/R9/R12` | Assorted evidence gaps | Carried; no DEV, per the AS8 owner ruling |
| `C10-R1` | Keyword identity controlled only coarsely, by parse failure | Lexical claims stated as acceptance/rejection, not token identity |
| `C10-R2` | No metamorphic relation added — none had normative backing | Metamorphic surface is 12 families and could be wider |
| `C10-DA-001` | Fourth duplicated authority (Map `RuntimeFn`) | **CLOSED by C10D-CTL-001's parity control** |
| `RA-LAYOUT`, `RA-LINTS` | rustc-assumption residuals; two deny-by-default lints suppressed in generated code | Narrows "rustc is an external control" |
| `DEV-017` | The coverage DB cannot express per-rule ± evidence | Why 85 of 168 rules are AGGREGATE |
| Branch coverage | Unavailable from this toolchain | **Never stated** |
| **22 historical trials** | Mutation evidence predating the toolchain integration (C10-R §6.2) | Any claim resting on one must say so |

---

# 5. What E9 establishes

```text
SATISFIED   CD-021 for population A — 24 live-OPEN deviations, each with an owner and a
            disposition. No unowned deviation remains, so none blocks the release decision
SATISFIED   B and C dispositioned, counted separately, and mapped to what each constrains

THE FINDING Exactly ONE deviation (DEV-177) accepts what the specification forbids. Every other
            row either REFUSES something the spec allows, or executes an accepted program wrongly.
            That asymmetry is the single most useful sentence for C10-Q: a conformance claim can be
            made NARROW and true, and it is DEV-177 that decides whether it can be made at all
            over NAME-SHADOW-001

DONE        DEV-005's required current-head reproduction — and it CLOSED the deviation rather
            than confirming it. One of 24 named deviations turned out not to exist, which is the
            argument for requiring reproduction rather than inheriting a list
```
