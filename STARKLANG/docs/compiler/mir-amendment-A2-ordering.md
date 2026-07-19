# MIR v0.1 Amendment A2 — `Ordering` as a Logical MIR Enum (for user `Ord` dispatch)

Status: **APPROVED under CE3 with clarifications** (WP-C4.6 A3-Ord, CD-033), 2026-07-19.
Renamed from "Ordering as a Runtime Value" per the owner's clarification #1 — "runtime value"
could be confused with the separately versioned `RuntimeFn` surface; `Ordering` is a **logical
MIR enum**, not a runtime-op surface addition. The clarifications are folded into §2/§3/§5 below
and the implementation shipped against them.

**Clarifications applied (CE3 decision):**
1. `EnumRef::CoreOrdering` approved. Discriminants `Less=0/Equal=1/Greater=2` are **logical MIR
   discriminants only** — they do NOT fix a physical ABI; C5.1 chooses the physical layout while
   preserving the observable MIR semantics (§2).
2. Ordered-operator lowering approved as drafted; both operands evaluated left-to-right and
   borrowed `&Self`, never moved (§3).
3. Runtime surface stays `0.1-A3` (no runtime op added); `MIR_VERSION` stays `0.1`. The
   C4-open additive-amendment policy is recorded in the main MIR contract, and `CoreOrdering`
   is reflected in the contract's `EnumRef` description — not only here (§4, done).
4. Implementation scope + tests per the decision (§4/§5); `println(Ordering)` is **out of this
   amendment** (Display is A4) — the round-trip test verifies construct/return/match only (§5).
5. DEV-070 accepted as correctly classified and owned by A2 (§6).

Scope class: **narrow additive amendment to MIR v0.1** (like A1). It adds the representation and
lowering rules for one already-existing prelude enum (`Ordering`) as a first-class MIR value and
defines how the ordered comparison operators (`<`, `<=`, `>`, `>=`) on a user nominal lower to a
call of the type's `Ord::cmp`. It adds **no** `RuntimeFn` and **no** new `MirTy`. `MIR_VERSION`
stays `"0.1"`; the runtime surface identifier is **unchanged** (`0.1-A3`) because no runtime-op
surface changes — `Ordering` is an ordinary user-visible prelude enum, not a runtime container.

## 1. Why an amendment is needed at all

A1 §10 explicitly reserved "`Ordering` as a runtime value and user-nominal `Eq`/`Ord` impl
dispatch (own short design note when C4.5e reaches it)." This is that note. `Eq::eq` returns
`Bool`, which MIR already represents, so `Eq` dispatch needed no amendment (A3 shipped it). `Ord`
is different: `Ord::cmp(&self, &other) -> Ordering`, and the operator lowering must **inspect**
the returned `Ordering` to produce the `Bool` that `<`/`<=`/`>`/`>=` evaluate to. So MIR must:
(a) have a settled representation for an `Ordering` value, and (b) define the discriminant
inspection that maps an `Ordering` to the comparison's `Bool`.

## 2. `Ordering` representation

`Ordering` is the prelude enum
```
enum Ordering { Less, Equal, Greater }
```
It is **already** a normal user-visible enum in the front end (the HIR oracle represents it as
`Value::Ordering(std::cmp::Ordering)` and prints `Less`/`Equal`/`Greater`). In MIR it is
represented as an **ordinary three-variant fieldless enum**, exactly like any user
`enum { A, B, C }`:

- MIR type: `MirTy::Enum(EnumRef::CoreOrdering, [])` — a **new `EnumRef` variant**
  `CoreOrdering` (parallel to the existing `CoreOption`/`CoreResult`), so the enum has a stable
  MIR identity independent of any user `ItemId`. This is the only new enum-identity token.
- Variant discriminants (fixed logical, normative): `Less = 0`, `Equal = 1`, `Greater = 2`.
  These are **logical MIR discriminants only** (CE3 clarification #1): they fix the value
  `Rvalue::Discriminant` yields and the SwitchInt cases, so the HIR/MIR differential agrees, but
  they do **not** establish a stable external ABI or require C5.1 to use those exact physical
  bytes — C5.1 remains free to choose the physical enum layout while preserving the observable
  MIR semantics.
- Values are built with `Rvalue::Aggregate(AggKind::EnumVariant(EnumRef::CoreOrdering, v), [])`
  and inspected with `Rvalue::Discriminant` + `SwitchInt`, using the machinery that already
  exists for `Option`/`Result`.
- `Ordering` is `Copy`, carries no fields, and needs no drop glue. `println(Ordering)` (if ever
  lowered) prints the variant name; that path is **not** part of this amendment (it is A4
  Display surface) and stays unsupported until A4.

**Alternative considered and rejected:** representing `Ordering` as an `Int8` (−1/0/+1) in MIR.
Rejected because (a) it would diverge from the oracle's enum representation, breaking the
differential's `Discriminant`/value comparison, and (b) `Ordering` is a real user-constructible
prelude value (`let o = a.cmp(&b);` is legal Core), so it must round-trip as the enum the source
language sees, not a compiler-internal integer.

## 3. Lowering `<` / `<=` / `>` / `>=` on a user nominal

For `a OP b` where `OP ∈ {<, <=, >, >=}` and the operand type is a (non-generic) user nominal:

1. Find the type's `Ord::cmp` impl (`find_impl_fn(nominal, "cmp", receiverless = false)`), the
   same resolution path A3's `Eq` dispatch uses. If absent → clean `Unsupported` (the checker
   already requires the `Ord` bound, so in practice it is present).
2. Borrow both operands as `&Self` (left then right — evaluation order per EXEC-EVAL-001),
   reusing A3's `borrow_value_ref`.
3. Emit `Call Ordering::… cmp(&a, &b) -> ord_temp`, `ord_temp : Enum(CoreOrdering, [])`.
4. Read `Rvalue::Discriminant(ord_temp)` into an `Int64` and compute the `Bool` with a pure
   `MirBinOp` against the fixed discriminants, matching the oracle's mapping exactly:
   - `a < b`  ⟶ `disc == 0`               (Less)
   - `a <= b` ⟶ `disc != 2`               (Less or Equal)
   - `a > b`  ⟶ `disc == 2`               (Greater)
   - `a >= b` ⟶ `disc != 0`               (Greater or Equal)

   (The oracle computes the same predicate from the `std::cmp::Ordering` it gets back; these
   four expressions are exactly that predicate over discriminants 0/1/2.)

No new terminator, no new `RuntimeFn`. The `cmp` call is an ordinary user-method `Call`; the
discriminant read and compare reuse existing `Rvalue::Discriminant` and `MirBinOp::Eq`/`Ne`.

## 4. Verifier impact and MIR-contract versioning policy (done)

- `EnumRef::CoreOrdering` is added to the verifier's enum-typing so `Aggregate`/`Discriminant`
  on it type-check (three fieldless variants; **discriminant in 0..=2 — a v3 aggregate fails
  MIR-0008**, per CE3 requirement #8, tested by `rejects_invalid_core_ordering_variant`). No new
  error code.
- `Rvalue::Discriminant` already yields `Int64` (MIR-0008 unchanged); the follow-on
  `MirBinOp::Eq`/`Ne` against an `Int64` const is already legal.
- No surface-gate change (runtime surface stays `0.1-A3`).
- **Versioning policy (CE3 clarification #3), recorded in the main MIR contract `mir.md`:**
  *Until Gate C4 closes and MIR v0.1 is frozen for backend consumption, narrow additive shape
  amendments may remain within MIR v0.1 only when individually approved under CE3 and recorded
  in the MIR contract. After C4 exit, any MIR shape change requires a MIR version bump.* This
  amendment is such a CE3-approved additive amendment; `CoreOrdering` is reflected in the
  contract's `EnumRef` description, not only here.

## 5. Differential expectations (implemented; see the WP-C4.6 A3 tests)

- `a < b`, `a <= b`, `a > b`, `a >= b` on a user struct with an `Ord` impl agree with the oracle
  (Bool result), operands **borrowed** (no early drop), `Ord`-and-`Drop` type drops normally
  (`user_ord_all_operators_agree`, `user_ord_borrows_and_drops_normally_agree`).
- A directly-invoked `cmp` returns an `Ordering` value that round-trips through a `match` by
  variant, agreeing with the oracle (`ordering_value_round_trips_through_match_agree`).
  **Correction (CE3 clarification #4):** `println(Ordering)` is explicitly **out** of this
  amendment (Display is A4), so the round-trip test verifies **construction, return, and pattern
  matching only** — it does not require Display support. (A three-arm exhaustive `Ordering`
  match is a separate front-end exhaustiveness gap, DEV-071; the test uses an
  explicit-plus-wildcard match, which fully exercises the MIR `CoreOrdering` path.)

## 6. Dependency note (not part of the CE3 decision, recorded for sequencing)

Realistic user `Ord`/`Eq` impl bodies that `match *self` are currently blocked by **DEV-070**
(match on a shared-reference scrutinee moves it out), owned by WP-C4.6 A2. This amendment's
`Ord` dispatch mechanism is independent of DEV-070 (it calls `cmp`, whatever the body does);
end-to-end Ord tests whose `cmp` body matches `*self` will pass only after A2 lands. Tests will
be written with field-reading / `if`-chain `cmp` bodies (as in the `userord` lowering fixture)
until A2 lands, exactly as A3's `Eq` tests were.

## 7. Decision — APPROVED (CE3, 2026-07-19)

Approved with the five clarifications folded into the header and §2–§5 above. Implemented in the
same session: `CoreOrdering` in lowering/verify/interp/dump, `Ordering::Less/Equal/Greater`
construction, direct `cmp` calls, all four ordered operators on non-generic user nominals,
variant-index validity (v3 → MIR-0008), clean `Unsupported` for generic-nominal comparison, no
change to primitive comparison. Tests: `user_ord_all_operators_agree`,
`ordering_value_round_trips_through_match_agree`, `user_ord_borrows_and_drops_normally_agree`
(differential), `rejects_invalid_core_ordering_variant` + `accepts_valid_core_ordering_variants`
(verifier). A3 (Eq + Ord) is complete; DEV-070 remains open under A2.

### Original decision request (retained for the record)

Approve, under CE3, adding `EnumRef::CoreOrdering` (fixed discriminants Less=0, Equal=1,
Greater=2) as the MIR representation of the prelude `Ordering` enum, and the §3 ordered-operator
lowering, as an **additive MIR v0.1 amendment with no runtime-surface change** (stays `0.1-A3`).
On approval, the `Ord` portion of A3 is implemented against this design; until then it remains
clean `Unsupported`.
