# WP-C7.9 Packet E — accepted-surface audit

**What was audited:** every language surface the front end accepts that MIR lowering refuses, so
that the reference interpreter is the only engine able to execute it.

**Why it matters:** for such a program, "do the engines agree?" has no answer — two of the three
cannot run it. It is neither a divergence nor a clean refusal, and it is the state the review found
five surfaces sitting in. The rule adopted here is that it may not be a steady state: a form is
either lowered, or refused by the front end.

**How the audit was performed:** every test in the suite that asserted the three-part shape
*(type-checks) ∧ (HIR runs it) ∧ (lowering refuses it)* was located — `c63c_iterators.rs`'s
`hir_only` helper and `c63d_map_key_identity.rs`'s — and each of their cases was dispositioned. The
`Iterator` combinator surface was then read out of `typecheck.rs`'s own signature table rather than
from the test list, which is what turned five known surfaces into nine.

**The guard:** `starkc/tests/adversarial_accepted_surface_audit.rs` asserts the implication
*accepted ⇒ lowerable* over the whole list. It holds whichever way a future package resolves any
individual row.

---

## Dispositioned surfaces

| # | Surface | Was | Disposition | Where |
| --- | --- | --- | --- | --- |
| 1 | by-value `Vec<T>` iteration (`for x in v`) | accepted, HIR-only | **REFUSED** `E0105` | `typecheck.rs`, `For` arm |
| 2 | `Iterator::map` | accepted, HIR-only | **REFUSED** `E0105` | `typecheck.rs`, `core_method_signature` |
| 3 | `Iterator::filter` | accepted, HIR-only | **REFUSED** `E0105` | as above |
| 4 | `Iterator::count` | accepted, HIR-only | **REFUSED** `E0105` | as above |
| 5 | `Iterator::collect` | accepted, HIR-only | **REFUSED** `E0105` | as above |
| 6 | `Iterator::fold` | accepted, HIR-only *(found by this audit)* | **REFUSED** `E0105` | as above |
| 7 | `Iterator::reduce` | accepted, HIR-only *(found by this audit)* | **REFUSED** `E0105` | as above |
| 8 | `Iterator::any` / `all` | accepted, HIR-only *(found by this audit)* | **REFUSED** `E0105` | as above |
| 9 | `Iterator::find` | accepted, HIR-only *(found by this audit)* | **REFUSED** `E0105` | as above |

Rows 6–9 were not in the review's list. They were found by reading the combinator table in
`core_method_signature` and checking each name against MIR lowering, which handles `next` and
nothing else. Had the audit been limited to the reported five, four surfaces would have remained in
exactly the state the packet exists to eliminate.

## Refusal, not implementation — and why

Owner ruling D3 for this work package: choose uniform front-end refusal rather than build new
iterator architecture. Implementing these means MIR representations for the adapter types
(`MapIter`, `FilterIter`) and method calls on non-nominal receivers — a work package, not a packet.
The refusal is therefore a scope decision and not a judgement that the combinators should not exist;
the review's own recommendation to implement them is recorded in the WP for whoever picks that up.

`for` loops over `v.iter()`, ranges, arrays and `s.chars()` are unaffected, and are pinned from the
other side by the guard file's `borrowed_iteration_is_unaffected`.

## Found, and deliberately NOT changed

| Surface | State | Why it stays |
| --- | --- | --- |
| `HashMap`/`HashSet` entries whose key or value type has a user `Drop` impl | accepted, HIR-only (`c63d_map_key_identity.rs`) | The refusal point is governed by **CE4 (CD-132)**, which decided these stay refused before MIR so that entry Drop order remains unobservable and therefore legitimately unspecified. Moving the refusal earlier is a change to a ruled decision, not an implementation detail, and WP-C7.9 does not reopen ruled decisions. **Recorded as an open row for whoever revisits CD-132.** |
| `match *r` where the matched type has a user `Drop` impl | accepted, HIR-only — lowering refuses with "match through a reference on a user-Drop type (front-end move-out-of-borrow gap)" | **Found during Packet C's qualification**, not by the audit pass: a drop-log case written to prove that a borrowed match leaves its referent alive could not be lowered at all. Same family as the row above and same governance (CE4/CD-132 territory), so it is recorded rather than changed. Pinned by `adversarial_patterns::a_borrowed_match_over_a_drop_type_is_refused_before_mir`, which fails if lowering ever starts accepting it — at which point it becomes a three-engine case with a drop log. |

This row is the reason the audit table exists rather than just the guard test: the guard asserts the
implication over the surfaces this packet owns, and this one is excluded by an explicit decision
rather than by omission.

## Diagnostic

`E0105` — *iteration form not supported by this implementation (deferred feature)*, allocated in
`04-Semantic-Analysis.md` beside `E0104`, which is the same class of refusal for by-value array
iteration. One code covers the whole set with a precise message per form, rather than nine codes
that would each name one row of a table that is expected to shrink.
