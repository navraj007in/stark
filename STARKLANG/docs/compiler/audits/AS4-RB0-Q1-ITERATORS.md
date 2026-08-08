# RB0 Q1 — the iterator drop-glue asymmetry: evidence packet and finding

**Packet:** AS4, Sprint 3. **Closes:** `WP-C7.8-RB0` Q1, carried forward untouched by AS0 §4.
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-08.
**Status:** **RESOLVED WITH EVIDENCE.** No behavioural change made — the finding is a
recommendation, and acting on it needs its own CD (RB0 exit criterion 5).

---

## 1. What RB0 asked for

> `lower::ty_needs_drop` says `VecIter`/`KeysIter`/`Iter` need glue; `CharsIter`, `SplitIter`,
> `ValuesIter`, `MapIter`, `FilterIter` do not. The asymmetry may be entirely valid.
>
> Required evidence packet, per iterator type: stored fields; whether any field owns
> allocation/resource/drop-bearing state; whether it borrows another value; current lowering
> representation; current native representation; current drop plan; existing lifecycle tests.
>
> Then decide each iterator individually.

## 2. The packet

| Iterator | Constructible as `MirTy::Core`? | Native representation | Owns allocation / resource? | Borrows? | Rust `Drop`? | Drop plan | `requires_drop_glue` |
| --- | :---: | --- | :---: | :---: | :---: | --- | :---: |
| `VecIter` | **yes** | `VecIter<'a, T> { slice: &'a [T], index: usize }` | no | yes | no | `Noop` | **true** |
| `KeysIter` | **yes** | `KeysIter<'a, K> { keys: &'a [K], index: usize }` | no | yes | no | `Noop` | **true** |
| `Iter` | **yes** | emits **as** `KeysIter` (DEV-116-B: the set's cursor IS the map's keys cursor) | no | yes | no | `Noop` | **true** |
| `CharsIter` | **yes** | `CharsIter<'a> { inner: std::str::Chars<'a> }` | no | yes | no | `Noop` | **false** (CD-387) |
| `SplitIter` | no | *no native struct* | — | — | — | `Noop` | true |
| `ValuesIter` | no | *no native struct* | — | — | — | `Noop` | true |
| `MapIter` | no | *no native struct* | — | — | — | `Noop` | true |
| `FilterIter` | no | *no native struct* | — | — | — | `Noop` | true |

Sources: `mir_ty`'s `MirTy::Core` constructions (constructibility); `stark-runtime/src/{vec,map,string}.rs`
(native structs); `emit_types::emit_ty_at` (emitted type); `drop_plan::plan_for`, where
`MirTy::Core(..) => DropPlan::Noop` for everything but `Vec` and `Box`; `c63c_iterators`,
`dev119_iterator_lifetime`, `dev138_iterator_item_representation`, `dev179_dormant_iterator_callbacks`
(lifecycle tests).

## 3. Finding: the asymmetry is **not** valid

Every constructible iterator is the same shape — **a borrowed cursor owning nothing**:

```text
VecIter    &'a [T] + index
KeysIter   &'a [K] + index
Iter       emits as KeysIter
CharsIter  std::str::Chars<'a>
```

None owns an allocation or a resource. None has a Rust `Drop`. All four plan as `Noop`. Yet
`CharsIter` answers `false` and the other three answer `true`, for **no reason visible in their
representation**. RB0 allowed that the asymmetry "may be entirely valid"; the evidence says it is
historical.

CD-387's reasoning for `CharsIter` — *a borrowed cursor; destroying it has no STARK-visible
destruction action and releases no owned language or provider resource* — applies verbatim to
`VecIter`, `KeysIter` and `Iter`.

The four unreachable iterators are a separate matter: they keep the historical answer under
AS4-DROP-AUTHORITY's rule for representations lowering cannot construct.

## 4. Why nothing is changed here

**The difference is observationally inert.** `drop_plan::plan_for` returns `Noop` for all of them,
and none has a native destructor, so `true` versus `false` costs a drop unit and a drop flag — not
correctness. Nothing leaks and nothing runs twice.

But flipping three iterators to `false` changes MIR shape (drop units, flags, and the terminators
around them), so it is a **behavioural** correction and owes its own decision record. RB0 exit
criterion 5 and AS4 work item 5 both say so, and this packet is not the place to smuggle one in.

**Recommendation:** classify `VecIter`, `KeysIter` and `Iter` as `false`, consistent with CD-387,
under a CD of their own. Expected effect: fewer drop units in generated MIR, no observable change to
any program.

## 5. Pinned

`mir::lower::as4_drop_predicate_inventory::rb0_q1_the_iterator_asymmetry_is_pinned_with_its_evidence`
asserts the current answers and fails, by name, if any of the three is changed — so the decision
above cannot be taken silently as part of an unrelated refactor.

## 6. RB0 status after this

```text
Q1  iterator drop-glue asymmetry   RESOLVED WITH EVIDENCE — asymmetry unjustified, change deferred to a CD
Q2  FnPtr reference disagreement   ANSWERED (AS4-REFERENCE-RULE.md): different questions, both kept
```
