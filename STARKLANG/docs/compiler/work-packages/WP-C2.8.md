# WP-C2.8 — Type, Trait, Pattern, and Constant Semantics

Gate: C2 (Reference Execution Semantics and Compiler Service Foundation).

Status: **Completed 2026-07-18.**

## Scope delivered

- exact Core place and pattern syntax, namespaces, scope, resolution, and shadowing;
- primitive, nominal, reference, tuple, array, function, never, and transparent-alias identity;
- finite-sizedness, the `str`/`[T]` unsized boundary, recursive-type legality, and edge types;
- deterministic function-local inference, generic inference, coercion ordering, loop typing,
  method lookup, receiver adjustment, and ambiguity rejection;
- trait definition, associated-type, orphan, overlap, selection, normalization, and law rules,
  with no Core v1 specialization, negative implementations, or trait objects;
- static Copy, borrow coexistence, lexical-region, temporary escape, returned-reference, and
  borrow-carrying provenance rules completing the static remainder of `CORE-Q-006`;
- pattern typing, ownership restrictions, exhaustiveness, usefulness, and deterministic test
  order, complementing C2.7's runtime pattern ownership/destruction rules;
- a closed, side-effect-free constant-evaluation subset with deterministic dependency, cycle,
  trap, and failure behavior;
- the required Core trait set and a canonical-item-identity standard-library hook table;
- thirty-three stable granular rules transferred into their normative homes and mechanically
  checked against the completeness inventory.

## Approved decisions

Type aliases are transparent. Only `str` and `[T]` are unsized, and only immediately behind
references; direct value recursion is rejected while `Box` and `Vec` break sizedness cycles.
Inference is deterministic and function-local, and later uses of a binding within its lexical
region may constrain it. Trait implementation selection is independent of source order;
overlap is rejected by existential unification, bounds are not presumed disjoint, and Core v1
has no specialization.

Borrows use conservative lexical regions with no last-use shortening or temporary-lifetime
extension. Pattern exhaustiveness and usefulness use a deterministic pattern-matrix model.
Constants are immutable compile-time values evaluated by the specified closed subset.
Standard-library hooks are recognized only by canonical item identity; all other library
behavior uses ordinary resolution and dispatch.

`CORE-Q-005` remains partially approved because C2.9 owns the canonical package/version token
used by coherence and nominal identity. C2.9 also owns numeric results, float trait
participation, observable layout-query values, and resource-limit classification. Those
dependencies do not reopen the C2.8 algorithms.

## Evidence and scope control

The completeness and open-question registers were reconciled with the normative syntax, type,
semantic-analysis, and standard-library chapters. Five appended parser fixtures cover aliases
and recursion, generic inference, borrowed-rvalue regions, enum exhaustiveness, and constants;
existing fixtures retain their identities. The conformance validator requires every C2.8 rule
to occur exactly once in normative sources and to remain marked complete in the granular
inventory.

This work package defines static semantics and governance. It does not change the Rust
compiler/interpreter, allocate final diagnostic codes, or claim executable positive/negative
evidence for the new granular rules. WP-C2.11 owns implementation alignment, deviation closure,
and adversarial evidence after C2.9 and C2.10 complete.
