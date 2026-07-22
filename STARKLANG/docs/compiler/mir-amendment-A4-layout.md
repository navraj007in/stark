# MIR v0.1 Amendment A4 — Type-Preserving Layout Queries (`size_of` / `align_of`)

Status: **APPROVED under CE3** (owner, 2026-07-20, WP-C4.7-3) — approved as drafted, i.e. the §7
recommendation: the shape addition now, the reference layout service returning the frozen (8, 8)
for every type, no oracle change, no spec edit, real target numbers left to C5.1. Logged as
**CD-036**. Implemented in the same session; §6's scope list is what shipped.

Numbering note: this amendment was called "A3" in the WP-C4.7 plan. WP-C4.7-1 recorded the
WP-C4.6 A5 arithmetic additions as MIR amendment **A3** (`mir.md`, CD-035, itself pending
ratification), so the layout amendment is **A4** to avoid two A3s.

Scope class: **narrow additive amendment to MIR v0.1**, in the same class as A1/A2/A3. It adds
one `Rvalue` variant. It adds **no** `MirTy`, **no** `RuntimeFn`, and **no** `TrapCategory`.
`MIR_VERSION` stays `"0.1"`; the runtime-surface identifier stays `0.1-A6` (nothing on the
runtime surface changes).

---

## 1. The defect being corrected

`06-Standard-Library.md` classifies `size_of`/`align_of` as **"target-layout queries"** and says
"C2.9 completes the target results of `size_of`/`align_of`". `07-Modules-and-Packages.md`
**LAYOUT-QUERY-001** makes them the *only* Core layout observations, requires them to be
"compile-time/runtime consistent", and requires them to "satisfy array/field placement needed by
safe execution". **LAYOUT-ABI-001** adds that the values may differ between named targets and
compiler versions, so no specific numbers are frozen by the spec.

What WP-C4.6 A4-1 actually implemented (`lower.rs`, the `Res::Builtin(SizeOf | AlignOf)` arm):

```rust
Res::Builtin(Builtin::SizeOf | Builtin::AlignOf) => {
    self.emit(Statement::Assign(dest,
        Rvalue::Use(Operand::Const(Constant::Int(8, MirTy::UInt64)))), …);
}
```

The queried type is **erased at lowering**. The HIR oracle does the same
(`interp.rs`: `Builtin::SizeOf | Builtin::AlignOf => Ok(Value::Int(8))`), so the differential
test `size_of_align_of_agree` passes — **but it passes because both engines share the same
placeholder, not because either is right.** Three concrete problems:

1. **A C5 backend cannot answer a target-layout query from this MIR.** By the time MIR exists,
   `T` is gone. The backend would have to re-derive it from the HIR, which defeats the charter's
   central rule (§1.2) that MIR is the *one* validated representation between the front end and
   every backend.
2. **It is not obviously spec-conformant even as a reference answer.** LAYOUT-QUERY-001 requires
   the values to satisfy "array/field placement needed by safe execution". A model in which
   `size_of::<Bool>()` and `size_of::<Int32>()` both report 8 is only defensible while *nothing
   consumes the answer*. It is not a target contract; it is a stub.
3. **The differential is vacuous here.** Two engines agreeing on a constant that neither derives
   from the query is not evidence of anything. Amendment A4 does not by itself fix this (both
   engines still return the same reference answers), but it makes the *representation* carry
   enough information for a real target answer to be substituted at exactly one place per engine.

Note what this amendment does **not** claim: it does not decide the target numbers. CD-015
(WP-C2.9) approved that only `size_of`/`align_of` expose target layout and that Core promises no
ABI; it did not fix per-type values, and LAYOUT-ABI-001 says they are target- and
version-dependent by design. **Choosing real numbers is C5.1's job** (the named target contract).
This amendment's job is to stop MIR from throwing away the question.

## 2. The shape addition

```text
Rvalue ::= …
         | LayoutQuery { kind: LayoutKind, ty: MirTy }    -- A4

LayoutKind ::= SizeOf | AlignOf
```

- **Typing:** the destination is always `UInt64` (matching 06's `fn size_of<T>() -> UInt64`).
  `ty` is the *queried* type and is unconstrained — any `MirTy`, including unsized ones, may be
  named (see §5 on `Sized`).
- **Purity:** `LayoutQuery` is a **pure `Rvalue`**, not a terminator. It cannot trap, cannot call
  user code, and cannot diverge, so §5's totality invariant holds unchanged. This is the reason
  it is an `Rvalue` rather than a `RuntimeFn` (see §3).
- **Monomorphisation:** because MIR is fully monomorphised, `ty` is always concrete. `size_of::<T>()`
  written inside a generic function is lowered with the active `param_subst` applied, so the
  body for `f@[Int32]` carries `LayoutQuery { SizeOf, Int32 }`. A `MirTy::Param` never appears —
  there is no such variant, so this is enforced by construction.
- **Dump grammar (§7 addition):**
  ```text
  _3 = layout_size_of(Int32)
  _4 = layout_align_of(Struct(#7)<Int32>)
  ```
  using the contract's existing `MirTy` rendering for the argument.

## 3. Why not a `RuntimeFn`

A `RuntimeFn` would be the smaller-looking change (no shape amendment at all), and it is the
wrong one:

- The runtime surface is a table of **value-level** operations with fixed or schematic
  signatures. `size_of` takes no value arguments — its only input is a *type*. Encoding it as a
  runtime call means either (a) inventing a type-as-operand encoding, which is a much larger
  change to `Operand` than one `Rvalue` variant, or (b) one `RuntimeFn` per queried type, which
  is not a closed set.
- Runtime ops are **terminators** (`Terminator::Call`), because a runtime op may trap. A layout
  query cannot trap. Making it a terminator would add basic-block structure to every use for no
  semantic reason and would misrepresent it to every analysis that reads "call" as "may abort".
- The runtime surface is the **backend's supplied-runtime contract** ("a backend knows exactly
  what runtime it must supply", §7). Layout is not something a backend *supplies at runtime*; it
  is something a backend *knows at compile time*. Putting it on the runtime surface would tell
  backend authors exactly the wrong thing.

## 4. Consumer contract (the single override point)

> **SUPERSEDED IN PART by CD-067 (WP-C5.3e, 2026-07-23).** This section's `reference_layout` no
> longer exists. The layout service it called for was built as `starkc/src/layout.rs`: a versioned
> named contract (`stark-64-v1`) that ALL THREE engines read — the HIR oracle included, which this
> amendment expected to leave untouched. A4's structural claim held exactly as written (the
> question survives into MIR, and one named service answers it); what changed is that the service
> is shared rather than per-consumer, and that a backend does **not** answer from "its target's
> real layout" in the sense of its own physical representation. Under CD-067 the contract is
> declared and the backend's representation is unobservable and need not equal it. This is the
> amendment's own option **(b)**, taken in C5 rather than C4 as it recommended.
>
> Read the rest of this section as the historical rationale for the shape, which is unchanged.

Each consumer answers `LayoutQuery` from a **layout service**, not inline:

- **The C4 reference interpreter** gets one function, `fn reference_layout(ty: &MirTy) -> (u64, u64)`
  returning `(size, align)`. In this amendment it returns **(8, 8) for every type** — i.e.
  **observable behavior is unchanged**; every existing differential and corpus expectation still
  holds, and the HIR oracle needs no change at all. The point is that the reference answer now
  lives in one named function that takes the type, instead of being a literal `8` smeared across
  the lowering.
- **A C5 backend** replaces exactly that one function with its target's real layout. Nothing else
  in lowering, verification, or the dump changes.
- **The verifier** checks one rule: `LayoutQuery`'s destination type is `UInt64` (MIR-0004
  otherwise). It does **not** validate the queried type — every `MirTy` is a legal question.

This is deliberately the minimum that makes the representation honest. It does not pretend to
give correct target numbers in C4; it makes the place where correct numbers go a single,
named, obvious one — and makes it impossible for a backend to receive MIR that has already
thrown the question away.

## 5. Interaction with `Sized`

LAYOUT-QUERY-001 scopes the guarantee to `Sized` `T`. Core v1's unsized types (`str`, `[T]`)
appear only behind a reference (contract V-TY-3), and the front end already rejects
`size_of::<str>()`-shaped calls upstream of MIR by ordinary type checking. This amendment adds
**no** MIR-level `Sized` check: MIR trusts the checked front end here, exactly as it does for
every other well-typedness property the checker owns. If that upstream check is ever found to be
missing, that is a front-end deviation to be numbered, not a verifier rule to add.

## 6. Scope of implementation if approved

1. `mir/mod.rs` — `Rvalue::LayoutQuery { kind, ty }` + `LayoutKind`; dump rendering.
2. `mir/lower.rs` — the `Res::Builtin(SizeOf | AlignOf)` arm reads the call's turbofish type
   argument (available on the HIR callee expr; converted with the existing written-HIR→`MirTy`
   path and the active `param_subst`) instead of emitting `Const 8`. Destination stays `UInt64`.
3. `mir/verify.rs` — one typing arm (dest must be `UInt64`).
4. `mir/interp.rs` — one `eval_rvalue` arm delegating to `reference_layout`.
5. `mir.md` — record the amendment (§5 rvalue grammar, §7 dump grammar, the versioning-policy
   amendment list), per the C4-open policy.
6. Tests: `size_of_align_of_agree` (existing differential — must stay green *unchanged*, which is
   the proof that behavior did not move); a new lowering/dump golden showing the queried type
   survives into MIR; a verifier negative `rejects_layout_query_with_non_uint64_dest`.
7. **No spec edit.** 06 and 07 already say what this amendment implements; nothing normative
   changes.

## 7. Decision requested

Approve `Rvalue::LayoutQuery { kind: SizeOf | AlignOf, ty: MirTy }` as an additive MIR v0.1
amendment (A4), typed `UInt64`, pure, answered by a per-consumer layout service whose C4
reference implementation returns the current (8, 8) for every type — representation fixed,
behavior unchanged, `MIR_VERSION` `0.1`, runtime surface `0.1-A6`.

Alternatives the owner may prefer instead:

- **(a) Do nothing, record as a known deviation.** Cheapest. Cost: C5 inherits a MIR that cannot
  answer a normative Core query, and the C4 exit report has to say so plainly — which is
  precisely the kind of finding the WP-C4.7 correction pass exists to prevent.
- **(b) Approve the shape AND require real reference numbers now** (a genuine per-type layout
  algorithm for the reference target, changing the HIR oracle too). This is a larger change: it
  makes `size_of::<Bool>()` return 1, breaks the current differential's shared placeholder in a
  way that must be re-pinned in BOTH engines, and needs an owner decision on the reference
  target's struct-packing rules — which is C5.1 material arriving early. Recommended only if the
  owner wants the layout contract settled inside C4.
- **(c) Defer the whole item to C5.** Then C5.1 does both the shape change and the numbers.
  Cost: the shape change lands after MIR v0.1 is frozen for backend consumption, so under the
  `mir.md` versioning policy it would require a **MIR version bump**, not an amendment.

Recommendation: **approve as drafted (the §7 paragraph)**. It is the only option that keeps the
representation honest without pulling C5.1's target contract into C4, and it is additive-now /
free-later, whereas (c) is additive-now / version-bump-later.
