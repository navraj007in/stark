# WP-C1.6 — Conformance Evidence Generator

Gate: C1 (Core v1 Conformance Closure). Extracted verbatim in scope from
`STARKLANG/docs/compiler/COMPILER-ROADMAP.md`.

## Scope

Implement one command or CI job that emits, per rule:

```text
rule id
spec chapter/section
status
source implementation
positive tests
negative tests
deviation id if any
last verified commit
```

The report must be generated from machine-readable data, deterministic, and checked into CI as
an artefact or summary.

## Inherited findings (owned by this WP per prior WPs' disposition)

- **DEV-017** — `core-v1-coverage.toml`'s `tests` field cites files, not functions, and 40 of the
  59 rules cite only the aggregate `starkc/tests/conformance.rs` (the spec-fixture-corpus
  runner), which mixes positive and negative fixture coverage for every rule at once with no
  per-rule attribution. WP-C1.1 partially corrected the specific rules it touched
  (LEX-003/004/005/010/013, SYN-001/013) but left the schema itself unable to cite individual
  test functions, let alone distinguish positive from negative coverage structurally. Explicitly
  assigned to this WP: "this is exactly the rule-level-precision problem that WP exists to
  solve."

## Scope-control answers (Charter §2.6)

- **Exact compiler claim tested:** that a machine-generated, deterministic report can state, for
  every one of Core v1's 59 tracked rules, its implementation and test-evidence status without
  relying on hand-maintained prose that can drift from the actual source/tests (the failure mode
  DEV-002/DEV-017 both found real instances of).
- **Later mechanism that would make the result impossible to attribute:** a report field that is
  *computed* looking accurate but is actually hand-typed and can silently go stale (exactly
  DEV-002's failure mode) — every column must be either read directly from the coverage database
  or computed fresh from git/filesystem state at generation time, never asserted by prose.
- **Strongest existing comparator:** `starkc/scripts/check-conformance.py` (existing validator +
  aggregate chapter-level summary) — extended, not replaced; its path-existence and duplicate-ID
  checks remain load-bearing for this WP's new richer report.
- **Negative result that would stop this WP/gate:** a report that re-introduces DEV-002's failure
  mode by hand-asserting a column instead of deriving it, or that isn't actually deterministic
  (two runs against the same commit producing different output) — both would defeat the WP's own
  stated purpose.

## CE-escalation watch

Per the user's 2026-07-17 standing preference (`stark-ce-escalation-flagging` memory): flag any
CE1-CE9-shaped decision found in this WP *before* resolving it, not after. This WP is mostly
tooling/evidence-infrastructure work, lower CE risk than prior WPs, but watch for: any schema
change to `core-v1-coverage.toml` that would be a breaking change for downstream consumers (none
currently known, but flag if discovered); and any decision about how much of DEV-017's
40-rules-unclassified gap to actually close in this WP vs. defer further (a real scope/effort
tradeoff, not a pure bug fix) — flagged to the user before committing to a scope.

## Input packet

`COMPILER-CHARTER.md` + `COMPILER-STATE.md` + this file, `STARKLANG/conformance/
core-v1-coverage.toml`, `starkc/scripts/check-conformance.py`, `STARKLANG/tests/spec-fixtures/
manifest.toml`, `.github/workflows/ci.yml`.

## Execution log

See `COMPILER-STATE.md` session record `### WP-C1.6` for dated evidence, files touched, and
decisions.
