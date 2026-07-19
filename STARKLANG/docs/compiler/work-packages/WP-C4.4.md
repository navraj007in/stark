# WP-C4.4 — MIR Interpreter (Differential Validation)

Gate: C4. Scope from `COMPILER-ROADMAP.md` WP-C4.4: a MIR interpreter whose purpose is
differential validation, not a user-facing VM (charter §1.6 rule 11). The comparator:

```text
HIR interpreter output/failure == MIR interpreter output/failure
```

for each frozen workload the current lowering supports.

## Deliverables (done, 2026-07-19)

- `starkc/src/mir/interp.rs` — executes **verified** MIR: frames with option-slot locals
  (`Move` *takes* the value, so any verifier-missed use-after-move explodes loudly instead of
  reading stale data); place reads/writes through projections; total statements; `Checked`
  terminators implementing STARK trap semantics per integer width (overflow bounds per type,
  divide-by-zero incl. `MIN / -1` via range check, float div/rem-by-zero per CD-006, checked
  numeric casts); `SwitchInt` with the same `as u128` key wrap the lowering uses; direct,
  indirect (`FnValue`), and runtime calls; a fuel guard (50M steps) turning runaway-loop
  lowering bugs into clean internal errors rather than hangs. **Float printing calls
  `interp::canonical_float` — the HIR oracle's own formatter, exposed `pub` so both engines
  share one algorithm by construction (no drift possible).** Trap outcomes carry their
  `TrapCategory`; internal errors are loudly distinct from language-level traps.
- `starkc/tests/mir_differential.rs` — 7 differential tests, each running the full
  `lower → verify → execute` pipeline against the HIR oracle:
  - the 5 lowerable frozen-corpus cases (`corpus_version = 1.0.0`) — byte-equal stdout + equal
    status, and for `primitive__02` both engines trap with matching category;
  - function values (assignment, param-passing, `f(f(v))` — CD-021 items 16/17/22 executed
    through MIR);
  - `Option`/`Result` as logical enums end-to-end (construction → discriminant → match);
  - structs + tuples (aggregates, field projections);
  - division-by-zero trap agreement; mid-output trap agreement; recursion (`fib`) + loops.
  Trap correspondence is checked category↔oracle-message (`IntegerOverflow` ↔ "integer
  overflow", `DivideByZero` ↔ "division by zero", …).

## Findings

- One comparator-map bug caught by the harness itself (oracle says "division by zero"; the map
  said "divide by zero") — fixed; no engine disagreement found.
- **Zero semantic differences between HIR and MIR execution across the supported workload.**

## Done when (roadmap) — status

- [x] MIR interpreter exists for the supported subset.
- [x] `HIR output/failure == MIR output/failure` holds for every lowerable frozen workload.
- [x] Workspace 632/0/2, fmt + clippy clean.

## Next

WP-C4.5 — complete Core lowering (generics/monomorphisation, trait dispatch, full
`Option`/`Result` combinators, patterns, indexing via `CheckIndex` proofs, strings/Vec/runtime
calls, ownership/drop elaboration incl. `Drop` terminators, panic/trap paths, multi-package
symbol linkage) — expanding the same differential net as each construct lands. C4.6 gate exit
closes when the whole Core execution corpus runs equivalently through both interpreters.
