# Sprint 4 — Tier-3 closeout

**Date:** 2026-08-09. **Branch:** `wp-arch-stability/as7-modularization`.
**Packets:** AS6 (CD-390), AS7 (CD-391, criterion 2 re-qualified CD-393), AS8 (CD-394).
**Range:** `b33b3e7..e7bb95d` — 45 commits after `6050efa`, which itself carries Sprint 3.

**Verdict: PENDING exact-head CI. Every other Tier-3 row is discharged below.**

---

## 1. What Sprint 4 changed

```text
typecheck.rs          14,588 lines -> ELEVEN modules with an executable, cycle-free DAG
extensions/tensor     the tensor extension's semantic authority extracted behind TensorCheckCtx
package.rs, parser.rs, resolve.rs, session.rs, source.rs, onnx/verifier.rs, test_runner
packages/stark-http-client
```

That surface decides which Tier-3 rows are required: the extension rows and the package/provider
rows are **not** optional here, because extension code and `package.rs` both changed.

## 2. Tier-3 checklist

| Row | Status |
| --- | --- |
| `cargo fmt --check` | **CLEAN.** Two of my own files needed formatting and were fixed with `rustfmt <paths>`, never a whole-tree `cargo fmt` — this checkout is shared |
| `cargo clippy --workspace --all-features --all-targets -- -D warnings` | **CLEAN.** Run in its own command and its exit read. An earlier piped run reported exit 0 while the build was failing; the pipe masks it |
| Full Rust suite through CI | **PENDING** — exact-head CI on `e7bb95d` |
| Core positive and negative fixture conformance | **CLEAN** — `conformance`, which drives `STARKLANG/tests/spec-fixtures` |
| HIR / MIR / native-debug / native-release differential rows | **CLEAN** via CI's `three_engine_differential`, `mir_differential`, C6.4/C6.5 |
| Tensor / extension tests | **REQUIRED and clean** — `extensions/tensor` changed under AS6; the AS6 boundary lint (`as7_module_dependencies::as7_does_not_reopen_the_as6_extension_boundary`) is green |
| Deterministic outputs executed twice | **CLEAN** — `replaying_a_shard_twice_produces_identical_observation_hashes` and `shard_assignment_is_content_addressed_and_stable` |
| Package / provider qualification | **REQUIRED and CLEAN.** `qualify-first-party-packages.py --stark <release>` exit **0**: 21 package test targets, 915 tests, 0 failed; 22 packages built and their consumers run, ending with `STARK_TLS_RESOURCE_OK` — a live TLS 1.3/1.2 session verified, used, closed, and an untrusted root rejected. Plus the C7.8 Native Capabilities workflow on three platforms |
| Pinned external samples suite | **34/34 PASSED** at `b3b28e757f38d691e7309f168d1209e28ac459af` (2026-08-02) |
| Focused tests shown to fail before the repair | **DISCHARGED, twice.** CD-393's injected `self.convert_hir_type(...)` producing `typecheck/traits.rs -> typecheck/convert.rs is not permitted`; and re-injecting the CD-393 blindness to make the coverage assertion print `36 of 235` and fail |
| Updated deviations, coverage records, `COMPILER-STATE.md`, architecture docs | **DONE** — see §4 |

## 3. Owner ruling on coverage (2026-08-09)

> **The `--lib` coverage baseline is accepted as sufficient for AS8.** The full-corpus run is
> supplemental, not gating.

Recorded here rather than by amending CD-394, per the ruling. The accepted position is:

```text
ACCEPTED    the scoped LINE and REGION baseline
            --lib TOTAL         regions 46.69%   functions 58.00%   lines 48.34%
DELIVERED   the full-corpus run COMPLETED after the ruling and is published as supplemental
            evidence, exactly as the ruling directed
            full corpus TOTAL   regions 83.05%   functions 84.92%   lines 83.64%
DEFERRED    branch coverage only. It is unavailable from this toolchain, was not fabricated,
            and is not claimed. AS8-R15 is otherwise DISCHARGED
```

**One wording correction, made here and not by rewriting the closed record.**
`AS8-EXIT-QUALIFICATION.md`'s work-item row says *"Line/branch coverage baselines … Done"*, while
`AS8-COVERAGE-BASELINE.md` states that branch coverage was unavailable from this toolchain and was
not silently substituted. **The baseline document is correct and the qualification row overstates
it.** No branch coverage was produced, none was fabricated, and none is claimed.

`llvm-cov` reports region and line coverage for this toolchain; its branches column is empty. The
baseline says so in its own "What this baseline is NOT" section, which is why the discrepancy is a
wording defect in one row rather than an evidence defect.

**The full-corpus run also retired a claim this packet made.** The `--lib` baseline asserted that
the two files holding the most `INVISIBLE` shared authorities were the two least covered, and called
that an independent corroboration of the mutation results. It was a `--lib` artefact:
`typecheck/traits.rs` is 82.77% and `typecheck/types.rs` 86.71% against a project total of 83.05%,
and `provider_synth.rs` moved from 0.00% to 96.05%. The replacement finding is stronger than the one
it retires — **`traits.rs` is 82.77% covered and `ESF-TRAIT-001` still has no control at all**
(`AS8-MUT-014/015` both survived). Coverage says a line ran; it does not say anything would have
noticed had the line been wrong.

## 4. Governance surfaces, before and after

```text
COMPILER-STATE.md      12,979 -> 6,681 lines. `# Current position` is now the FIRST section;
                       `## Position` had been at line 5,456 describing Gate C5 closing four gates
                       earlier. Verified lossless: 57/57 CD sections retained, zero lines lost to
                       either the file or the archive
KNOWN-DEVIATIONS.md    index of multi-entry deviations added — the file is append-only, so
                       DEV-121's FIRST heading reads OPEN and it is CLOSED 3,558 lines later
lib.rs                 crate docs cited PLAN.md as the live plan and "Gates 1-3 ... interpreter"
compiler-map.md        eleven modules, corrected counts, `trait_contracts` row
```

## 5. What Sprint 4 leaves open, deliberately

```text
DEV-213      LSP caches one whole-package analysis per open URI and invalidates only the edited
             one. Real HEAD defect, demonstrated by a passing test. OWNER RULING: does not block
             Sprint 4; fixed in the next bounded LSP correctness packet. Until it closes, any
             claim that `workspace/symbol` is correct under MULTI-FILE EDITING stays qualified
AS8-R1..R15  mutation and evidence residuals. No DEV allocated for any survivor: a survivor means
             the evidence cannot detect a defect, not that the defect is present
AS8-DA-*     duplicate-authority dispositions are SETTLED and scheduled AFTER Sprint 4 —
             DA-001/DA-005 consolidate; DA-002/003/004 remain separate and gain an exhaustive
             parity/drift test over the closed `RuntimeFn` set; DA-006 unchanged
Gate C9 B    blocked pending second-artifact evidence; ONNX alone authorises no generalisation
DEV-012      C8's interactive editor validation, seven features. Unchanged by this sprint
```

## 6. Landing

Owner-frozen sequence, no rebase, squash or cherry-pick — a merge commit preserves every CD/DEV
packet SHA, and squashing would destroy the chronology this programme spent the sprint making
auditable. Verified before execution:

```text
develop           b33b3e7
PR #10 head       6050efa    110 commits ahead of develop; develop has 1 commit not in it
                             (ROADMAP.md only) -> a merge commit is REQUIRED, not merely preferred
PR #11 head       e7bb95d    strict descendant of 6050efa by 45 commits, ZERO divergence
merge-tree        CLEAN for develop + PR #10; neither branch touches ROADMAP.md
```

```text
CD-394 exact-head CI GREEN -> PR #10 (integration stage 1, merge commit)
                           -> PR #11 (Sprint 4 integration) -> this closeout -> Tier-3 CI GREEN
                           -> merge PR #11 (merge commit) -> Campaign B exit report
```
