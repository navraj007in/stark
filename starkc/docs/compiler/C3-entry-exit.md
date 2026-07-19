# C3-ENTRY Exit Report — Native Readiness and Carry-Forward Closure

Transition deliverable for the mandatory `C3-ENTRY` step in `COMPILER-ROADMAP.md` (inserted by
CD-018, tightened by CD-019, given an executable work package by CD-020 —
`STARKLANG/docs/compiler/work-packages/WP-C3-ENTRY.md`). Prepared 2026-07-19 at base commit
`3d12f45` (`CD-023/CD-024 close C3-ENTRY blockers 1-2`).

This is not a semantic gate. C3-ENTRY does not reopen Gate C2, change Core v1 semantics, or
begin native-architecture work. It exists to close or explicitly own Gate C2's carry-forward
obligations before backend-selection work starts, so that Gate C3 begins from a versioned,
CI-guarded baseline.

## Outcome

**C3-ENTRY CLOSED.** All four blockers are resolved with the evidence below. Gate C3 (Native
Compiler Architecture and Backend Selection Spike) opens; the next work package is **WP-C3.1**
(Architecture hypothesis and workload freeze, now a 23-item workload per CD-021/CD-022).

## Blocker closure

### 1. Six completeness-row approvals — CLOSED (CD-023, 2026-07-19)

The six rows previously `pending-owner-approval` in
`STARKLANG/docs/compiler/semantic-freeze/CORE-V1-COMPLETENESS.md` — `LEX-COMMENT-001`,
`LEX-ERROR-001`, `STD-OPTION-001`, `STD-RESULT-001`, `STD-ITER-001`, `STD-VEC-001` — were
approved as-is by the owner and flipped to `settled`. Each describes behavior already
implemented and exercised throughout Gate C2 (the C2 exit report records this); the gap was
governance bookkeeping, not missing or contradictory behavior. `LEX-ERROR-001` retains its
DEV-017 note (an evidence-citation-precision item, separately tracked, not a behavior
question). Zero rows remain `pending-owner-approval`. `starkc/docs/compiler/C2-exit-report.md`
carries a dated post-gate note recording this, rather than rewriting its gate-close evidence.
Decision record: `COMPILER-STATE.md` CD-023.

### 2. DEV-060 disposition — CLOSED by fix (CD-024, 2026-07-19)

DEV-060 (repeated call to an un-overridden trait default method wrongly flagged as a move of
the receiver) was root-caused and fixed rather than deferred, because C3.1 workload item 11
(default trait method calling another trait method) and items 22–23 (function-value repeated
invocation) sit on the affected path, and freezing a comparator workload over a known defect
would muddy every downstream differential result.

Root cause: `borrowck.rs`'s `method_receiver` searched only `ImplItem::Fn` overrides, with no
equivalent to `typecheck.rs::resolve_method`'s `default_fallback` (WP-C1.3/DEV-013). A call to
an un-overridden trait default method returned `None` from `method_receiver`, so the `Call`
handler's `None => self.check_expr(*base)` arm consumed (moved) the receiver regardless of its
real `&self`/`&mut self` kind. Fixed by adding the matching trait-default-body fallback to
`method_receiver`. Verified for the `&self` case, a `&mut self` variant (the `RefMut` arm was
not exercised by the original repro), and end-to-end execution. No soundness impact — this was
a rejection of legal code (availability), never an acceptance of illegal code. Full writeup and
regression-test list: `starkc/docs/conformance/KNOWN-DEVIATIONS.md` DEV-060. Decision record:
`COMPILER-STATE.md` CD-024.

### 3. Versioned execution-corpus freeze — CLOSED (CD-025, 2026-07-19)

The WP-C2.12 execution-snapshot corpus (`starkc/tests/exec_snapshots/`, 48 files: 31 `.stark` +
17 `.snap`, including `metamorphic/`) is frozen at:

```text
corpus_version = 1.0.0
base_commit    = 3d12f45f2388271d46e8d2a4e85f417237a13bc3
lock file      = starkc/tests/exec_snapshots/corpus.lock
lock sha256    = 8cda2df5e26aa35dfc8eb222f1e073eb4ea2336297e91ecc4e62b8fbd27dc0dc
```

The freeze was taken only after blocker 2 (DEV-060) was fixed, per WP-C3-ENTRY.md's procedure —
a fix could legitimately change corpus output, so freezing first would have locked in a value
about to change. `corpus.lock` records a SHA-256 per corpus file. A new integrity test,
`corpus_lock_matches_frozen_snapshot` in `starkc/tests/exec_snapshots.rs`, enforces the freeze
three ways: every listed hash must match, no listed file may be absent, and no `.stark`/`.snap`
file may exist that the lock does not list. The test was negatively verified (tampering with
one `.snap` makes it fail with the expected message; restoring makes it pass). Any intentional
corpus change must regenerate the lock AND bump `corpus_version` with a dated `COMPILER-STATE.md`
note; a bare `UPDATE_SNAPSHOTS=1` regeneration is a freeze violation the test catches. C3/C4
differential work references the corpus by `corpus_version`, never by "current contents."

### 4. Demonstrated green CI run — CLOSED (2026-07-19)

The CI baseline was widened to the C3-ENTRY required commands under CD-020 (`cargo fmt --all --
--check`; `cargo clippy --workspace --all-targets --all-features -- -D warnings`; `cargo test
--workspace --all-targets --all-features`; spec-regeneration `build-core-spec.py --check`;
fixture-extraction sync; conformance-database validation; named execution-snapshot step). The
push of base commit `3d12f45` to `origin/main` triggered the workflow, which the owner confirmed
succeeded. This is the demonstrated green run WP-C3-ENTRY.md requires. (Once MIR/native work
begins, CI must additionally grow MIR verifier tests, native smoke builds, HIR/MIR/native
differential cases, and the supported-target compile matrix, per the roadmap's C3-ENTRY
section.)

## Carry-forward ownership (recorded, not blocking)

The following were made explicit before this transition closed and are owned by later WPs; they
are not C3-ENTRY blockers:

- Unfinished WP-C2.12 generated-corpus work → WP-C4.4 and WP-C6.5 (written into those WP
  definitions under CD-020).
- Unfinished WP-C2.12 cross-backend replay → WP-C4.4, WP-C5.6, WP-C6.5 (same).
- The frozen corpus (`corpus_version = 1.0.0`) is the baseline those replay obligations run
  against.

## Local verification at close

```text
cargo build                                              clean
cargo test --workspace --all-targets --all-features      597 passed / 0 failed / 2 ignored
cargo fmt --all -- --check                               clean
cargo clippy --workspace --all-targets --all-features    clean (-D warnings)
cargo test --test exec_snapshots                         4 passed (incl. corpus_lock_matches_frozen_snapshot)
python3 starkc/scripts/check-conformance.py              clean (89.8% / 53-of-59, unchanged)
CI on origin/main @ 3d12f45                              green (owner-confirmed)
```

## Next

**Gate C3 — WP-C3.1: Architecture hypothesis and workload freeze.** Freeze the 23-item
representative workload (`COMPILER-ROADMAP.md`, WP-C3.1), define the measurement set, and write
`STARKLANG/docs/compiler/proposals/NATIVE-CORE-ARCHITECTURE.md`. Gate C3 selects *how* STARK
compiles natively (SELECT-GENERATED / SELECT-DIRECT / REVISE / BLOCKED), not *whether* — native
compilation is mandatory per CD-004.
