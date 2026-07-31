# WP-C7.8-RB0 — MIR Type Property Authority

**Status:** OPEN, not started
**Sequence position:** Route B **Packet 0** — after the Codex IO handoff, before Route B adds or
changes any `MirTy` representation
**Opened:** 2026-07-31, by owner ruling following CD-287/CD-288
**Scope discipline:** narrow semantic-authority change. **This must not become general MIR cleanup.**

## Why this exists

`MirTy` classification is answered in twelve places. CD-287 made all twelve exhaustive, so the next
new variant is a compile error rather than a silent default. That fixed the *omission* hazard. It did
nothing about the *divergence* hazard: twelve implementations of four rules that must agree and have
never been checked against each other.

The evidence that this is not theoretical:

- CD-240: `TypeContext::is_copy` classified `HostResource` as `Copy`. Every resource leaked.
- CD-287: `verify::mir_needs_drop` still classified `HostResource` as needing no drop, while
  `may_need_drop` — in the same file, answering the same question — said the opposite. Found by the
  compiler, not by a test, and only because exhaustiveness forced the arm to be written.
- Seven distinct sites have now swallowed that one variant, each corrected separately, most after a
  leak.

Route B (`OwnedResourceHandle`, MIR-owned exactly-once close, `resource_type` on `HandleOut`) is
exactly the kind of `MirTy` evolution that produced those seven. Entering another representation
migration with four definitions of "needs drop" outstanding is avoidable risk.

## The current duplication

| Rule | Implementations |
| --- | --- |
| is Copy | `TypeContext::is_copy` (`mir/mod.rs`), `FnLowerer::is_copy` (`mir/lower.rs`) |
| needs drop | `lower::ty_needs_drop`, `lower::ty_has_user_drop_guarded`, `verify::may_need_drop`, `verify::mir_needs_drop` |
| carries a reference | `lower::ty_carries_ref`, `emit_types::ty_carries_reference`, `emit_types::ty_contains_ref` |
| mentions a user nominal | `lower::ty_mentions_user_nominal` |

Target authority surface:

```text
is_copy
needs_runtime_drop
has_user_drop
carries_or_contains_reference
mentions_user_nominal
```

## Method — audit BEFORE consolidation

**The first deliverable is an inventory, not a refactor.** Some of these predicates only look like
duplicates. Where they genuinely answer different questions they must be *renamed and documented*,
not merged. Candidate distinctions already visible:

```text
contains_reference_in_storage      vs  carries_borrow_lifetime
requires_drop_glue                 vs  has_user_defined_destructor
```

These are not interchangeable. `HostResource` is the worked example: it requires drop glue
(its provider close) while having no user-defined destructor, which is why
`ty_has_user_drop_guarded` answers it `false` correctly while three other predicates answer `true`.

**Equivalence tests before deletion.** Each duplicate is removed only after a test demonstrates the
survivor agrees with it across the full `MirTy` variant set. A predicate that turns out to differ is
kept, renamed, and documented — deleting it would be the divergence, not the fix.

## Open semantic questions owned by this WP

Both were surfaced by CD-287, are recorded in code, and are **not** to be resolved by behavioural CD
until this WP's evidence exists. Preserving current behaviour was correct; do not "make them
consistent" because the names look similar.

### Q1 — iterator drop-glue asymmetry

`lower::ty_needs_drop` says `VecIter`/`KeysIter`/`Iter` need glue; `CharsIter`, `SplitIter`,
`ValuesIter`, `MapIter`, `FilterIter` do not. The asymmetry may be entirely valid.

Required evidence packet, per iterator type:

```text
stored fields
whether any field owns allocation / resource / drop-bearing state
whether it borrows another value
current lowering representation
current native representation
current drop plan
existing lifecycle tests
```

Then decide each iterator individually. Note the plausible outcomes are not uniform: an iterator
wrapping drop-bearing state may need glue; a pure cursor over borrowed storage may not; an iterator
nominal may need lifetime tracking without runtime drop; and a generated-Rust representation may
have host drop behaviour that MIR does not model.

### Q2 — `FnPtr` reference disagreement

`emit_types::ty_carries_reference` descends into a `FnPtr`'s parameters and return type;
`lower::ty_carries_ref` calls every fn value borrow-free.

Determine first whether the two ask the same question. A plain function pointer does not borrow the
values it will later receive, and a Rust `fn(&T)` is higher-ranked and needs no lifetime parameter —
which is the only thing the lowering predicate guards (E0106). But a function *value* representation
could carry an environment, lifetime-bearing metadata, or a bound receiver.

Expected outcome is one of:

```text
plain FnPtr carries no reference in both contexts  -> unify
the predicates answer different questions          -> rename, document, keep distinct
```

Do not force equality until the internal representation is traced.

## Status summary

```text
Iterator asymmetry:                      documented open semantic question
FnPtr disagreement:                      documented open semantic question
Blocking CD-287 / CD-288:                no
Blocking Route B representation change:  yes, through this WP
```

## Exit criteria

1. Inventory published: for each of the twelve predicates, the question it actually answers.
2. Every genuine duplicate replaced by one authority, each deletion preceded by an equivalence test.
3. Every non-duplicate renamed to say what it asks, and documented against its near neighbours.
4. Q1 and Q2 either resolved with evidence or explicitly carried forward with their packets filled in.
5. No behavioural change without its own CD. This WP is allowed to change *where* an answer comes
   from; changing *what* the answer is requires a separate decision.
