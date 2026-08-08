# Engine evidence independence audit

**Packet:** `WP-ENGINE-INDEPENDENCE.md` EI2, approved 2026-08-09 (CD-392), executed as an AS8
prerequisite. **Vocabulary frozen at EI0; register at `ENGINE-SHARED-FATE-REGISTER.md`.**

**Status: EI2 COMPLETE.** One EI1 residual closed, three new residuals opened.

---

## The finding

**The three-engine differential is not a symmetric comparison. It has a named reference engine.**

`starkc/tests/support/differential.rs` calls the HIR interpreter **"the HIR oracle"** throughout —
`oracle_category`, `"the oracle raised a trap with no stated category"`, `"oracle failed without
trapping"`. MIR and native are checked *against* HIR's answer.

That is not a defect; a differential needs a reference. But it has a consequence the register makes
concrete:

> For any authority where the HIR engine's answer is itself **INDIRECT** — inherited from the same
> front-end decision MIR and native inherit — the oracle is not independent of the thing under
> test. Agreement is then a statement about the pipeline's internal consistency, not about
> correctness.

Six of the ten registered authorities are `INVISIBLE` to all three engines precisely because the
front end decides once and every engine consumes the decision. For those six, the differential
suites are **`CROSS_ENGINE_DERIVED` evidence with no independent control**, and EI0's frozen rule
applies: not independently evidenced, regardless of how many engines agree.

## EI1 residual closed

**`ESF-PROV-001` — resolved, and the answer is worse than `UNKNOWN`.**

`src/interp.rs` contains the string `provider` **zero times**. The HIR engine has no provider
dependency at all — by design: the interpreters have no host access, so capability-backed packages
build but cannot run under `stark run`.

```text
ESF-PROV-001   hir NONE (was UNKNOWN)   mir DIRECT   native INDIRECT
```

The register's `UNKNOWN` is discharged, but the consequence is sharper than the gap it replaces:
**provider behaviour has only two engines, and they share `mir::provider_sig`.** There is no third
engine to act as a control, so the "three-engine differential" claim does not extend to providers at
all. That is now a residual against the *claim*, not against the measurement.

## Audit table

| Evidence ID | Engines covered | Expectation source | Independence | Shared-fate risk | Independent control | Residual |
| --- | --- | --- | --- | --- | --- | --- |
| `EV-DIFF-3ENG` — `three_engine_differential` | hir, mir, native | **HIR oracle** — MIR/native compared against HIR's output | `CROSS_ENGINE_DERIVED` | **HIGH** — cannot detect any defect in the six `INVISIBLE` authorities | none in-tree | For `ESF-COPY-001`, `ESF-DROP-001`, `ESF-TRAP-001`, `ESF-RES-001`, `ESF-TYPE-001`, `ESF-TRAIT-001` this evidence is structurally unable to disagree |
| `EV-DIFF-MIR` — `mir_differential` | hir, mir | same oracle | `CROSS_ENGINE_DERIVED` | **HIGH** | none | as above, and covers one fewer engine |
| `EV-TRAP-CAT` — trap category comparison | hir, mir, native | `mir::TrapCategory`, read from the error rather than its prose | `CROSS_ENGINE_DERIVED` | **HIGH** | conformance fixtures assert a trap *occurs* | **A mis-categorised trap is invisible.** All three engines match on the same enum, so they agree on a wrong category. The mechanism that reads the category from the error rather than the message (a real improvement over prose-matching) does not address this |
| `EV-CORPUS-C6` — `c6-corpus`, kinds `handwritten` / `generated` / `retained` | hir, mir, native | mixed; `generated` carries a `generator_seed`, and every case carries `expected_trap_category` | `handwritten` = `HAND_AUTHORED`; `generated` = `SHARED_FIXTURE_GENERATOR` | **MEDIUM–HIGH** | manifest + generator hashes are pinned and checked; determinism is proven by re-running the generator | The corpus's **expectations are stated in the shared trap vocabulary**, so a corpus case cannot contradict `ESF-TRAP-001`. Generator determinism is verified; generator *correctness* is not independently derived |
| `EV-SPEC-FIXTURES` — `spec fixture conformance` | front end | **the specification**, hand-triaged into `manifest.toml` | `SPEC_DERIVED` | **LOW** | this is the control others lack | Covers parse/semantic classification, not runtime semantics — so it does not reach the runtime authorities in the register |
| `EV-COPY-MATRIX` — `copy_canon_matrix` | front end, mir | **enumerated from `typecheck/traits.rs`'s `core_method_signature` arms** | `IMPLEMENTATION_GENERATED` for the producer set; `HAND_AUTHORED` for the expected classification | **HIGH** | **partial** — the file deliberately includes ordinary-language producers (slice expressions, aliases, function returns) as controls, so the law is tested beyond the intrinsics it enumerates | Strong against **drift**, weak against a **wrong rule**: a wrong rule is enumerated faithfully. The file's own header concedes the shape — *"[the] test would pass just as happily if the reverse were true"* |
| `EV-SAMPLES` — `External sample suite (pinned)` | native | **external repository**, pinned by commit SHA with a resolution check | `EXTERNALLY_DERIVED` | **LOW** | strongest control in the tree | Covers whole-application behaviour, not specific authorities; a shared-authority defect would surface only if a sample happens to exercise it |
| `EV-PROVIDER-LOOP` — `C7.8 provider metadata/unit/resource/loopback` | mir, native | **live peers**, real sockets/processes | `EXTERNALLY_DERIVED` | **LOW–MEDIUM** | genuine external oracle | Two engines only (see `ESF-PROV-001`); no third-engine control exists for providers |
| `EV-STRUCTURAL` — `as6_core_module_vocabulary`, `as7_module_dependencies` | n/a — source structure | the source itself, plus a frozen declaration | `HAND_AUTHORED` | **LOW** | each was proved to fail on an injected violation | Structural, not semantic; says nothing about engine agreement |

## The six required questions, answered for the load-bearing sources

**1. Who or what generated the expectation?**
For the differential suites: the HIR engine. For the generated corpus: `generate.py`, pinned and
hash-checked. For `copy_canon_matrix`: a human, reading the checker's own match arms. For the spec
fixtures and the sample suite: sources outside the implementation.

**2. Does the expectation depend on the implementation being tested?**
Yes for `EV-DIFF-*`, `EV-TRAP-CAT` and `EV-COPY-MATRIX`. No for `EV-SPEC-FIXTURES`, `EV-SAMPLES`,
`EV-PROVIDER-LOOP`.

**3. Is the same fixture parser or semantic table used by multiple engines?**
Yes — `mir::TrapCategory` is the expectation vocabulary *and* the implementation vocabulary, for all
three engines and for the corpus manifest.

**4. Would the evidence detect a shared-authority defect?**
**No, for the six `INVISIBLE` authorities.** This is the audit's central answer and it is negative.

**5. Is there a negative control?**
Only for the structural tests, each of which was proved to fail on an injected violation. **No
semantic evidence source in this table has a demonstrated negative control** — none has been shown
to fail when the thing it checks is broken.

**6. Is there an independent spec-derived assertion?**
For the front end, yes (`EV-SPEC-FIXTURES`). For runtime semantics, **no**.

## Residuals opened

```text
EI2-R1  No semantic evidence source has a demonstrated negative control. Under CD-392's evidence
        invariant, none of them may yet be cited as establishing what it claims. This is the
        single largest gap the audit found and it is AS8's own to close, since AS8 is the packet
        that runs mutation trials.

EI2-R2  Provider evidence has two engines, not three. The three-engine independence claim must be
        stated with that exclusion or it overstates its scope.

EI2-R3  The trap vocabulary is simultaneously the implementation's, the differential's and the
        corpus manifest's. A mis-categorisation is invisible to every mechanism in the tree.
```

## What this means for EI4 and EI5

EI4 ranks risk; EI5 selects mutation targets. This audit says the ranking must weight
**`INVISIBLE` × no-independent-control** highest, because those are exactly the authorities where a
mutation will survive every existing suite — which is the definition of a test gap rather than a
passing grade.

It also means **EI5's mutation trials are the first real control this evidence base will have.**
That is a strong reason to run them, and a strong reason CD-392's invariant applies to them before
they are believed: a harness that kills nothing looks identical to one that works.
