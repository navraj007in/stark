# WP-C2.7 — Abstract Machine and Execution Semantics

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation).

Status: **Completed 2026-07-18; correction pass applied after review.**

## Scope delivered

- normative `CORE-V1-ABSTRACT-MACHINE.md`, independent of interpreter frames, Rust values,
  future MIR, physical allocation, and backend ABI;
- abstract values, objects, storage, owners, locals, temporaries, places, projections,
  initialization, and moved-from state;
- exactly-once expression evaluation and a construct-complete order table, including
  function-valued callees;
- value/place contexts, copy/move reads, partial moves, reinitialization, and indexed-move
  prohibition;
- right-before-left assignment, install-before-old-drop replacement, compound assignment, and
  destination-trap behavior;
- partial aggregate ownership and separate normal-transfer versus aborting-trap cleanup;
- deterministic normal cleanup for temporaries, locals, parameters, fields, pattern residuals,
  loops, iterators, collections, replacement, and explicit `drop`;
- reference identity, live projection, caller-owned method returns, range-slice views, and
  borrow-carrying values;
- language-trap abort semantics and the backend differential observation comparator;
- authority cleanup in the type, semantic-analysis, and memory-safety chapters;
- thirteen manifest-triaged adversarial specification examples.

## Approved decisions

CORE-Q-006 is partially approved: C2.7 approves its runtime abstract-machine portion, while C2.8
retains static place legality, borrow coexistence and regions, temporary-reference escape, and
returned-reference legality. CORE-Q-020 is approved for runtime ownership and destruction of
the pattern forms present in Core v1; C2.8 retains static pattern typing, exhaustiveness, and
usefulness. CORE-Q-017 is approved only at the language-trap boundary: a specified trap aborts
without unwinding, while C2.9 still classifies resource, host, target, and process failures.

The assignment decision is: evaluate the RHS, resolve the destination, detach the old value,
install the new value, then destroy the old value. A trap during destination resolution aborts
without destroying the completed RHS. A trap during partial aggregate construction likewise
does not unwind completed fields. Normal `return`, `break`, `continue`, and `?` propagation do
perform deterministic cleanup.

## Prior runtime deviation mapping

| Deviation | Named rule violated |
| --- | --- |
| DEV-024, DEV-026, DEV-027 | `EXEC-DISPATCH-001` |
| DEV-028, DEV-041, DEV-042 | `REF-SLICE-001` |
| DEV-029 | `DROP-ORDER-001` |
| DEV-030 | `PAT-DROP-001` |
| DEV-031 | `EXEC-FOR-001` |
| DEV-032 | `OBS-COMPARE-001` plus the standard-library iteration contract |
| DEV-033 | `EXEC-EVAL-001` |
| DEV-034 | `EXEC-ONCE-001` |
| DEV-035 | `REF-RETURN-001` |
| DEV-037 | `REF-PROJECT-001` |
| DEV-038, DEV-043 | `EXEC-DISPATCH-001` |
| DEV-039 | `DROP-LOOP-001` |
| DEV-040 | `DROP-COLLECTION-001` |

Closed deviations remain regression evidence. Open deviations are implemented only in C2.11
after all C2.8–C2.10 decisions are complete.

## Evidence and scope control

The document was cross-checked against every Core expression form, the C2.1 reference-execution
audit, the corrected C2.2 interpreter regressions, the C2.6 granular inventory, and normative
chapters 02–06. The fixture extractor includes the new normative chapter; its thirteen examples
are manifest-triaged.

This work package defines semantics. It does not modify the Rust compiler/interpreter, implement
MIR, approve C2.8–C2.10 questions, or claim that current runtime evidence already covers the new
granular rules. C2.11 owns implementation and adversarial behavioral evidence.

## Post-completion correction

The correction pass after commit `3d64dfc` resolved three normative blockers and one governance
inconsistency found by external review:

- `HashMap`/`HashSet` insert and remove now specify ownership per key/value component and preserve
  the returned old map value as an owned `Option<V>`;
- function-valued callees evaluate exactly once before arguments and remain live through the
  call;
- pattern traversal, Copy-versus-move binding, reference projection, binding creation order, and
  residual destruction order are explicit;
- CORE-Q-006 is recorded as runtime-approved and static-part-pending rather than fully approved.

The same pass made borrowed-rvalue lifetimes through calls explicit, repaired the valid-slice
example, and added focused map replacement, function-callee, mixed binding, and nested binding-
order examples. It changes normative documentation and fixtures, not Rust interpreter behavior;
C2.11 still owns executable alignment.
