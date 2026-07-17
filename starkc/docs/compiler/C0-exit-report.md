# Gate C0 Exit Report — Bootstrap, Current-State Audit, and Authority Repair

WP-C0.5 deliverable. Prepared 2026-07-17 at head `6fa8c15b94bd1376a847132498d31dd356524180`.

## Decision

**PASS.** Gate C0's stated closing condition — "the project's current state can be stated
without relying on commit-message optimism or stale roadmap text" — is met. Every material
status claim audited across the repository's governance documents now agrees with
`COMPILER-STATE.md`, four stale documents were corrected in place, the conformance database's
integrity was audited and four real staleness errors were fixed, and ten confirmed compiler
deviations (plus two informational, not-owned items) are recorded with full structured detail
in `starkc/docs/conformance/KNOWN-DEVIATIONS.md`.

This PASS is a bootstrap/audit gate result, not a conformance claim. See "No conformance
percentage is trusted" below.

## Current head

```
6fa8c15b94bd1376a847132498d31dd356524180
```

Rust toolchain: `stable` (channel pinned in `starkc/rust-toolchain.toml`, no version number);
measured environment: `cargo 1.93.0 (083ac5135 2025-12-15)`, `rustc 1.93.0 (254b59607
2026-01-19)`. Crate MSRV: `rust-version = "1.85"` (`starkc/Cargo.toml:8`).

## Test and fixture counts

`cargo test --workspace --all-targets --all-features` (re-run 2026-07-17 at the end of Gate C0,
after all WP-C0.1-C0.4 changes — none of which touched Rust source):

```
383 passed, 0 failed, 2 ignored
```

Both ignored tests are intentionally opt-in and non-hermetic (a checksum-pinned live ONNX
artifact test in `tests/gate4_onnx.rs`; a live-ORT-download inference test in
`tests/gate5_codegen.rs`). Full per-binary/per-file breakdown: `COMPILER-STATE.md` Repository
baseline section and `starkc/docs/dev/compiler-map.md` §5.

Spec fixture corpus (`STARKLANG/tests/spec-fixtures/manifest.toml`): 121 entries — parse-pass 67,
semantic-error 18, notation 30, lex-pass 4, parse-fail 2 (verdict census re-verified during
WP-C0.0, matches the file's own header comment and `starkc/docs/gate1-exit.md`).

Conformance database (`STARKLANG/conformance/core-v1-coverage.toml`, post-WP-C0.3 correction):
59 rules — 53 implemented, 6 partial, 0 missing. `python3 starkc/scripts/check-conformance.py`
exits 0 with no errors and no warnings.

## Authoritative document list

In source-of-truth order (per `STARKLANG/docs/compiler/COMPILER-CHARTER.md` §1.8), as of this
exit:

1. **Normative Core v1 spec**: `STARKLANG/docs/spec/00`–`07` (individual files are the editing
   surface; `STARK-Core-v1.md`/`.html`/`.pdf` are generated, never edited directly).
2. **Approved decisions**: `COMPILER-STATE.md` (repo root) — decision log CD-001 through CD-003,
   deviation ledger DEV-002/004–013.
3. **Gate exit evidence**: `starkc/docs/gate1-exit.md` through `gate7-decision.md` (old
   numbering, all closed); this document (`C0-exit-report.md`, new numbering).
4. **Compiler roadmap**: `STARKLANG/docs/compiler/COMPILER-ROADMAP.md` (new C0–C10 gates) and
   `STARKLANG/docs/ROADMAP.md` (old Gates 1–7, still authoritative for that closed track).
5. **Engineering plan**: `STARKLANG/docs/PLAN.md` (accurate through Gate 5 work-package detail;
   Gates 6–7 recorded only as a summary note added under WP-C0.2, not full WP detail — see
   `COMPILER-STATE.md` CD-003 and the PLAN.md v0.6 changelog entry) and
   `stark-spec-parity-roadmap.md` (Phases 0–9; confirmed accurate, no corrections needed).
6. **README/context files**: `CLAUDE.md`, root `README.md`, `starkc/README.md` — all three
   corrected under WP-C0.2 (see below); `starkc/docs/dev/compiler-map.md` (new, WP-C0.1) is now
   the authoritative module-by-module reference, superseding `starkc/README.md`'s Layout table
   for that purpose.
7. **Archived, never authoritative**: `STARKLANG/docs/archive/`, `web-docs/` (repo root, not
   `STARKLANG/web-docs/` — a path error in the original WP-C0.1 task framing, corrected during
   research), `STARKLANG/compiler/` (Python prototype), `Practice/` (includes a second Python
   prototype under `Practice/Interpreter/`).

## Subsystem status matrix

| Subsystem | Old-numbering status | New-numbering (C-gate) status |
|---|---|---|
| Lexer | Gate 1 closed (`gate1-exit.md`) | Not yet requalified — WP-C1.1 pending |
| Parser | Gate 1 closed | Not yet requalified — WP-C1.1 pending |
| Name resolution | Gate 2 closed (`gate2-exit.md`) | Not yet requalified — WP-C1.2 pending; DEV-004, DEV-006 (partial), DEV-007 open against it |
| Type/trait checking | Gate 2 closed | Not yet requalified — WP-C1.3 pending; DEV-008, DEV-013 open against it |
| Ownership/borrow checking | Gate 2 closed (M2.4) | Not yet requalified — WP-C1.4 pending; DEV-006 (partial) open against it |
| Control flow/patterns/numerics | Gate 2 closed | Not yet requalified — WP-C1.5 pending |
| Interpreter (reference execution) | Gate 3 closed (`gate3-exit.md`) | Semantic-oracle contract not yet documented — WP-C2.1 pending; DEV-008, DEV-009 open against it |
| Tensor extension (front end) | Gate 4 closed (`gate4-exit.md`) | Out of C1 scope (Core-only); WP-C9.1/C9.2 will audit isolation |
| ONNX import/verify | Gate 4 closed (M4.5) | Maintained under Conditional Track T (T0); no new work authorized |
| Native deployment (ONNX Runtime host) | Gate 5 closed (`gate5-exit.md`) | Distinct question from Gate C3 (general native Core compilation) — see `COMPILER-STATE.md` CD-002; not itself a C-gate subsystem |
| Tensor productisation decision | Gate 6 REVISE, Gate 7 RETAIN AS RESEARCH LANGUAGE | Directly relevant prior evidence for Gate C3, not a substitute for it |
| Package/manifest/dependency system | Phase 1–3 closed (`stark-spec-parity-roadmap.md`) | `PKG-004/005/006` coverage-DB status corrected `missing`→`partial` (DEV-002, closed); WP-C1.2 will do the rule-level pass |
| Standard library | Phase 4A–4E closed/partial (`File` deferred) | `STD-003/004/005` coverage-DB citations corrected; DEV-009, DEV-013 open |
| Formatter | WP8.2 complete | Not yet in scope for any open C-gate |
| Test runner | WP8.3 complete | Not yet in scope for any open C-gate |
| Documentation generator | WP8.5 complete | Not yet in scope for any open C-gate |
| LSP | WP8.1/8.4 "foundation"/complete (self-described) | DEV-010 (hover/definition/references are stubs), DEV-012 (VS Code UI unverified) open — directly anticipated by WP-C8.2/C8.3/C8.7 |
| Multi-file diagnostic provenance | Not previously tracked as a subsystem concern | DEV-006 — new finding, WP-C0.1 |
| CLI behavior consistency (`starkc` vs `stark`) | Not previously tracked | DEV-005 — new finding, WP-C0.1 |

## Open deviation ledger

Ten confirmed deviations with full structured detail (normative expectation, current behaviour,
user impact, security/soundness impact, workaround, proposed disposition, owning gate) plus two
informational not-owned items: `starkc/docs/conformance/KNOWN-DEVIATIONS.md`. Summary:

| ID | One-line summary | Owning gate |
|---|---|---|
| DEV-002 | Coverage-DB staleness (4 rules wrongly `missing`) | **Closed** WP-C0.3 |
| DEV-004 | `resolve.rs` tensor-builtin gating bug (bare `min`/`max`) | WP-C1.2 |
| DEV-005 | `starkc check`/`run` warning-gating drift | WP-C1.x |
| DEV-006 | Multi-file diagnostic provenance loss (resolve/flow/borrowck) | WP-C1.2, WP-C1.4 |
| DEV-007 | Glob-import (`use mod::*`) nondeterminism | WP-C1.2 |
| DEV-008 | Structural equality, not `Eq` trait dispatch, at runtime | WP-C1.3 |
| DEV-009 | `File` has no runtime representation | unscheduled (candidate WP-C2.2) |
| DEV-010 | LSP hover/definition/references are stubs | WP-C8.2, WP-C8.3 |
| DEV-011 | Doc comments are trivia, not AST/HIR metadata | unscheduled (revisit at WP-C8.2 or a docs feature) |
| DEV-012 | VS Code extension UI never interactively verified | WP-C8.7 |
| DEV-013 | `STD-004` exhaustiveness unresolved (`Error` trait unconfirmed) | WP-C1.3 |

No deviation was found and left silently undocumented; each has an owning future WP or an
explicit "unscheduled" status with a candidate.

## No conformance percentage is trusted

Per Charter §1.5 rule 14 ("conformance claims require executable evidence") and this gate's own
closing requirement: **the 89.8% "implemented" figure `check-conformance.py` prints is not a
conformance claim.** It reflects a data-integrity-audited but not rule-completeness-audited
database. Specifically:

- Every `implemented` entry's `source`/`tests` paths were confirmed to exist on disk (mechanical
  check), but no entry's *behavioral completeness* against its full normative rule text was
  re-verified during Gate C0 — that is explicitly WP-C1.x's job, chapter by chapter.
- Four entries were found to have been simply wrong (`missing` when working, tested
  implementations existed) — if the database contained that class of error undetected for one
  audit cycle, further undetected errors in either direction (over- or under-claiming) cannot be
  ruled out until the WP-C1.6 rule-level conformance evidence generator exists and runs.
- The database's schema cannot currently distinguish positive from negative test evidence per
  rule (a WP-C1.6 follow-up), so Charter rule 15 ("positive and negative evidence travel
  together") is not yet mechanically verifiable from this data source.

Any future document that states a Core v1 or tensor v0.1 conformance percentage must derive it
from a WP-C1.6-generated report, not from this file, `COMPILER-STATE.md`'s snapshot counts, or
`check-conformance.py`'s current output.

## Exact next WP

**WP-C1.1 — Lexical and syntax requalification.**

Per the mandatory correctness path (`COMPILER-ROADMAP.md` §4.1): `C0 → C1 → C2`. Gate C0 is
closed; Gate C1 (Core v1 Conformance Closure) opens next. WP-C1.1 re-runs and strengthens
lexical/syntax fixture coverage, reserved-token handling, literal boundary/overflow lexing,
parser recovery guarantees, recursion/depth limits, AST span integrity, multi-file module
parsing, and extension-syntax Core-isolation — adding deterministic property/fuzz tests for no
panic/no hang/bounded resource use/stable diagnostics on malformed input.

Session budget note (`COMPILER-ROADMAP.md` §7): Gate C1 is sized at 7–10 sessions; this exit
report does not shorten that estimate merely because Gate C0 also produced early, real findings
in downstream areas (DEV-004 through DEV-013 span C1.2 through C8.x) — those findings are
inputs to future WPs, not substitutes for the requalification work each WP still owns.
