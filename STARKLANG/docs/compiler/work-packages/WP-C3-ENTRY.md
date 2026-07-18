# WP-C3-ENTRY — Native Readiness and Carry-Forward Closure (transition work package)

Status: **active** (opened 2026-07-19 under CD-020).
Governs the mandatory C3-ENTRY transition defined in `COMPILER-ROADMAP.md` (section
"C3-ENTRY — Native Readiness and Carry-Forward Closure", inserted by CD-018, tightened by
CD-019). This is a transition WP, not a semantic work package: it approves, freezes, and
transfers — it does not change Core v1 semantics, reopen Gate C2, or begin any C3 architecture
work.

## Exit artifact

`starkc/docs/compiler/C3-entry-exit.md` — the named completion record for this transition.
It must state, with evidence: each blocker's closure (who decided, when, where recorded), the
frozen corpus version and its lock digest, the CI run demonstrating the full baseline green,
and the exact next WP (WP-C3.1). C3 does not open without this artifact existing.

## Blockers and owners

### 1. Six completeness-row approvals — owner: project owner

Formally approve (or amend) the six rows still marked `pending-owner-approval` in
`STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md`:

```text
LEX-COMMENT-001   comment tokenization/attachment
LEX-ERROR-001     mandatory lexical rejection conditions
STD-OPTION-001    Option behavior/APIs
STD-RESULT-001    Result, propagation, combinators
STD-ITER-001      iterator protocol and termination
STD-VEC-001       Vec growth/indexing/bounds/ownership
```

All six are governance bookkeeping — behavior is already implemented and tested (C2 exit
report). Approval is recorded by flipping each row's decision state in the completeness ledger
and noting the batch in a dated `COMPILER-STATE.md` decision-log entry. Recommendations here
are explicitly unapproved until the owner acts.

### 2. DEV-060 disposition — owner: project owner (implementation: any session)

DEV-060 (repeated call to an un-overridden trait default method wrongly flagged as a move —
`starkc/docs/conformance/KNOWN-DEVIATIONS.md`) must be disposed **before the workload freeze in
WP-C3.1**, because C3.1 workload item 11 (default trait method calling another trait method)
sits directly on the affected dispatch path, and freezing a comparator workload over a known
defect muddies every later differential result. Allowed dispositions: (a) fix now with a
regression test (recommended), or (b) explicitly accept-and-document with a workload-item-11
carve-out note in the freeze record. Silence is not a disposition.

### 3. Versioned execution-corpus freeze — mechanical definition

The corpus to freeze is `starkc/tests/exec_snapshots/` (primary cases + `metamorphic/` pairs),
run by `cargo test --test exec_snapshots`.

Freeze procedure (perform only **after** blocker 2 is disposed, since a DEV-060 fix may
legitimately change corpus behavior):

1. Create `starkc/tests/exec_snapshots/corpus.lock` containing:
   - `corpus_version = "1.0.0"`;
   - the base commit hash the freeze was taken at;
   - one line per corpus file (every `.stark` and `.snap`, including `metamorphic/`), with its
     SHA-256.
2. Add a verification test (in `exec_snapshots.rs`) that recomputes the hashes and fails on
   any mismatch or on files present-but-unlisted / listed-but-absent. CI's "execution snapshot
   verification" step then covers both semantic replay and freeze integrity.
3. Version semantics: any post-freeze corpus change bumps `corpus_version` and requires a
   dated note in `COMPILER-STATE.md` naming the WP that authorized it (expected authors:
   C4.4/C5.6/C6.5 per their carried-forward WP-C2.12 ownership, or a recorded owner decision).
   Regenerating a `.snap` via `UPDATE_SNAPSHOTS=1` without a version bump is a freeze
   violation.
4. C3/C4 differential work references the corpus by `corpus_version`, never by "current
   contents".

### 4. CI baseline — delta status

`.github/workflows/ci.yml` existed before this transition and already covered fmt, clippy,
tests, fixture-extraction sync, conformance-database validation, and the evidence report. The
2026-07-19 governance-repair pass (CD-020) closed the gap to the C3-ENTRY required baseline:

- widened commands to `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets
  --all-features -- -D warnings`, `cargo test --workspace --all-targets --all-features`;
- added "Spec regeneration is in sync" via `python3 STARKLANG/tools/build-core-spec.py
  --check` (new `--check` mode; Markdown only — pandoc/weasyprint HTML/PDF output is not
  byte-reproducible and is deliberately excluded);
- added the named "Execution snapshot verification" step (`cargo test --test exec_snapshots`;
  redundant with the full test run by construction, kept as the explicitly named baseline
  evidence line).

Remaining to clear this blocker: one green run of the updated workflow on the repository's CI
(local verification of fmt + exec_snapshots passed 2026-07-19; the full matrix must be
demonstrated where CI actually runs). Once MIR/native work begins, CI must additionally grow
MIR verifier tests, native smoke builds, HIR/MIR/native differential cases, and the
supported-target compile matrix (roadmap, C3-ENTRY section).

### Already closed (recorded here for completeness)

- DEV-051/052/055: resolved 2026-07-19 with fixes and regression tests (see
  `KNOWN-DEVIATIONS.md` and the state file's Post-Gate-C2 Issues 6-8 session record).
- WP-C2.12 carry-forward ownership: written into the receiving WP definitions themselves
  (WP-C4.4, WP-C5.6, WP-C6.5 in `COMPILER-ROADMAP.md`) on 2026-07-19 under CD-020, so the
  charter's minimal session-input packet (charter + state + active WP) surfaces the obligation
  without needing this transition's text.

## Done when

- [ ] All six completeness rows carry an owner decision (approved or amended), recorded in the
      ledger and the state decision log.
- [ ] DEV-060 has an explicit disposition (fix landed with regression test, or documented
      acceptance with workload carve-out).
- [ ] `corpus.lock` exists at `corpus_version = "1.0.0"` with a passing integrity test.
- [ ] The updated CI workflow has one demonstrated green run.
- [ ] `starkc/docs/compiler/C3-entry-exit.md` exists, states all of the above with evidence,
      and names WP-C3.1 as next.
- [ ] `COMPILER-STATE.md` Position line reads `Gate: C3  Next: WP-C3.1  Blocked: none`.

## Scope-control answers (charter §2.6)

- **Exact claim this WP completes:** Gate C2's carry-forward obligations are closed or
  explicitly owned, and the native path starts from a versioned, CI-guarded baseline.
- **Later mechanism that would make results unattributable:** starting C3 spikes against an
  unversioned corpus — any later HIR/MIR/native disagreement could then be blamed on corpus
  drift instead of backend semantics.
- **Strongest existing comparator:** the C2 semantic-freeze preflight (same shape: audit,
  reconcile, transfer, produce a named artifact — its audit doc is the template for the exit
  artifact here).
- **Negative result that stops this WP:** the owner rejects one of the six completeness rows
  in a way that reopens a C2 semantic decision — that escalates (CE1/CE2) rather than being
  absorbed into this transition.
