# WP-VALUE-REP-TOTAL — a total type→representation mapping for the oracle

**Status:** ACTIVE. A0 and A1 landed; A2 next.
**Filed by:** CD-321, on owner direction, when INV-VALUE-REP-001 landed narrow.
**Owning track:** compiler (Gate C-series governance, `COMPILER-CHARTER.md` §1.6).
**Prerequisite deviations:** DEV-121 (narrowed, not class-closed).

---

## 1. What is already enforced, and what is not

WP-COPY-CANON's law:

> After expression typing, Copy/move behaviour — and the runtime representation that carries it —
> is determined exclusively by the normalized semantic type, never by the expression that produced
> the value.

The first half is enforced. INV-MOVE-001 (MIR-0036) rejects a `Move` operand from a `Copy` place
unconditionally, and it found four latent defects on its first runs (DEV-124, DEV-125, DEV-127).

The second half is enforced **in one direction, for one pairing**. INV-VALUE-REP-001 checks at every
`let` that a binding declared `&[T]` or `&str` does not hold an owned `Value::Vec`/`Value::String`.
That is exactly the direction DEV-121 broke — `let view = owner.bytes()` had `&[UInt8]` in the type
tables and owned storage at runtime, so passing it moved it and emptied the caller's binding.

Everything else is unchecked. There is no statement of what representation ANY other type must
have, so a mismatch in any other pairing is still invisible until a differential happens to run a
program shaped to expose it — which is how DEV-121, DEV-126 and DEV-129 were each found, and two of
those were found by CI rather than the corpus.

## 2. Why the narrow rule was landed rather than the total one

The oracle's value model is not currently total, and asserting that it is would produce firings on
correct programs. `&Int32` may legitimately arrive as the scalar itself through auto-deref;
`Value::Str` and `Value::String` both carry text and DEV-130 had to make comparison
representation-insensitive precisely because both occur where one type is declared.

A broad rule would therefore have to carry exemptions, and an invariant with exemptions is
advisory. The narrow rule always means something. That was the trade, taken deliberately.

**This package is the other half, done properly rather than bolted on.**

## 3. Scope

**In:**
- A declared mapping from normalized `Ty` to permitted `Value` representations, written down as a
  table before any code, because the disagreements are the point and they will not surface from
  reading the interpreter.
- Resolving the genuine ambiguities the narrow rule sidesteps, each as a decision with a reason:
  - `&T` for scalar `T` — is `Value::Ref(place)` required, or is the bare scalar permitted?
    Auto-deref currently produces both.
  - `Value::Str` versus `Value::String` — is the distinction meaningful, or should the oracle carry
    one representation for text? DEV-129 and DEV-130 are both consequences of it having two.
  - `Value::Slice` versus `Value::Vec` for `&[T]` — settled for `let` bindings, open elsewhere.
- Extending the check from `let` to the other binding sites: function parameters, match-arm
  bindings, field and element writes, and returns.
- INV-VALUE-REP-001 widened to the full mapping, or replaced by it.

**Out:**
- Changing the MIR or native value models. This is about the HIR oracle, which is the engine
  DEV-121 lived in and the one whose representation is least constrained.
- Performance. The oracle is a reference implementation.

## 4. Acceptance

1. The mapping exists as a written table, and every `Value` variant appears in it — including the
   ones no rule currently mentions. A variant absent from the table is the defect this package
   exists to prevent, the same shape as the `_ => true` wildcard that let `HostResource` be
   classified `Copy` (CD-240, DEV-128).
2. Each ambiguity in §3 is resolved with a recorded decision, not left to whichever site runs first.
3. The invariant covers every binding site listed, not only `let`.
4. It is unconditional. If a case cannot be made unconditional, that case is a DEFECT to fix or an
   amendment to approve — not an exemption to add.
5. Three-engine agreement is unchanged, and the frozen corpus is green.

## 5. Risk

The likely outcome is that the mapping cannot be made total without first CHANGING the oracle's
value model — probably collapsing `Str`/`String`, possibly making auto-deref produce a consistent
representation. That is a larger change than the invariant it enables, and this package should
expect to spend most of its effort there rather than on the check.

That is an argument for doing it deliberately, not for leaving it. The evidence from one session:
DEV-121, DEV-126, DEV-129, DEV-130 and DEV-131 were all consequences of the same untotal model, and
each was found by a different mechanism after reaching a different distance into the pipeline.

---

## 6. The representation matrix (A1)

Written before any widening of execution, because the disagreements are the point and they do not
surface from reading the interpreter. Every one of the **34** `Value` variants declared in
`interp.rs` appears below exactly once. There is deliberately no "other values are valid" row: a
variant absent from a table is the same defect shape as the `_ => true` wildcard that let
`HostResource` be classified `Copy` (CD-240, DEV-128).

`Value::kind()` in `interp.rs` matches on all 34 with **no wildcard arm**, so adding a variant is a
compile error until it is named there, and `every_value_variant_is_named_in_the_representation_matrix`
pins the count against this document.

### 6.1 What "permitted" means

The relation is `normalised Ty → closed set of permitted ValueKind`. It is a *relation*, not a
bijection: several types legitimately admit more than one representation today. Each such row states
the alternatives explicitly and says why, rather than being a wildcard wearing a rule's clothes.

The relation **observes**. It never converts `String`→`Str`, `Vec`→`Slice`, or dereferences a `Ref`
to make a check pass. A validator that repairs a mismatch destroys the evidence it exists to expose,
and the repair belongs at the producer.

### 6.2 Scalars and text

| Semantic `Ty` | Permitted `ValueKind` | Additional checks | Rationale |
| --- | --- | --- | --- |
| `Unit` | `Unit` | — | one value, one representation |
| `Bool` | `Bool` | — | |
| `Int8`…`Int64`, `UInt8`…`UInt64` | `Int` | representation family only | the oracle's `Int` is `i128` and does **not** carry width, so there is nothing about width to observe here. See 6.2.1 — this row is silent about the payload's numeric domain, not permissive about it. |
| `Float32` | `Float` | width is `F32` | `Value::Float` explicitly retains a `FloatWidth`, unlike `Value::Int`, so rejecting `Float(_, F64)` here checks information the model genuinely possesses |
| `Float64` | `Float` | width is `F64` | |
| `Char` | `Char` | — | |
| `String` (owned) | `String` | — | must never be `Str`: an owned value represented as a view would not move when moved |
| `Float16`, `BFloat16` | **none** | — | `tensor` extension element types (D3); not executable in Core v1 |

**Owned `String` has two spellings**, and A2's exhaustive match is what surfaced it: the resolver
maps the source name `String` to `Ty::Primitive(Primitive::String)`, while
`Ty::Core(CoreType::String, _)` also occurs in the type system. Both denote the same owned type and
both permit exactly `Value::String`. Handling only the `Core` spelling would have left every
ordinary `String` binding silently unvalidated — the precise failure this package exists to
prevent, caught by the compiler rather than by a differential.
| `str` standalone | **none** | — | unsized; a bare `str` value is an internal defect. Legal only behind a reference — see 6.4 |

#### 6.2.1 Representation is not numeric domain

Two invariants are easy to conflate, and this package owns only the first:

```text
representation invariant   Int32 → Value::Int                  (DEV-121)
numeric-domain invariant   the payload fits Int32              (checked arithmetic, literals, casts)
```

"Width is not checked" means the *representation* relation is silent about width, because
`Value::Int` is an `i128` that never carried one. It does **not** mean a `Value::Int` holding `1000`
behind a declared `Int8` is acceptable — that is a real defect, owned by checked arithmetic,
literals and casts, all of which trap. A reader who takes this row as licence to ignore an
out-of-domain payload has read it wrongly: the validator does not see a numeric defect, it does not
bless one.

### 6.3 Owned aggregates and collections

| Semantic `Ty` | Permitted `ValueKind` | Additional checks | Rationale |
| --- | --- | --- | --- |
| `Vec<T>` | `Vec` | payloads recursively `T` | never `Slice`/`Ref`: an owned vector represented as a view does not move |
| `Box<T>` | `Boxed` | payload `T` | |
| `[T; N]` | `Array` | exactly `N` slots | arity is the check that catches a truncated aggregate |
| `(T₁…Tₙ)` | `Tuple` | arity `n`, element types | |
| `Struct(item, args)` | `Struct` | **same `ItemId`**; field names | identity, not shape: two structs with identical fields are different types |
| `Enum(item, args)` | `Enum` | **same `ItemId`**; variant in range | |
| `Option<T>` | `Option` | payload `T` when present | |
| `Result<T, E>` | `Result` | payload `T`/`E` | |
| `HashMap<K, V>` | `HashMap` | key/value types | |
| `HashSet<T>` | `HashSet` | element type | |
| `Range<T>` / `RangeInclusive<T>` | `Range` | `inclusive` matches the core type | the two core types share one representation and are distinguished only by the flag |

**A whole value crossing a typed boundary must contain no moved-out `None` slot.** Partially moved
aggregates may exist inside the interpreter while being destructured or dropped; they may not cross
a parameter, return or assignment boundary.

### 6.4 References — the rows DEV-121 is about

| Semantic `Ty` | Permitted `ValueKind` | Additional checks | Rationale |
| --- | --- | --- | --- |
| `&mut T` (any `T`) | `Ref` | referent matches `T` | **never** flattened to a bare value. A mutable reference that is not a place cannot write through, and `take(&mut v)` needs the place itself |
| `&T`, `T` **non-Copy** | `Ref`, or a view (`Slice`) where `T` is unsized | referent matches `T` | flattening would copy a non-Copy value, which is the move-semantics violation |
| `&T`, `T` **Copy** | `Ref` **or** the permitted representation of `T` | if not `Ref`, must match `T`'s own row | **the one multi-valued row, and it is a semantic rule.** Copying a `Copy` `T` cannot consume, invalidate or run destruction on the referent, so the two representations are indistinguishable to any observation the oracle can make. The predicate is whether the **pointee** is Copy — never whether the reference is. It must not be extended to non-Copy `T` for convenience. |
| `&str` | `Str`, or `Ref` whose referent is `Str` or `String` | — | |
| `&str` → `String` | **FORBIDDEN** | — | **this is the DEV-121 ownership error.** Owned storage behind a borrowed type means passing it moves what it only borrows, emptying the caller's binding — on a program the checker and MIR both accept |
| `&[T]` | `Slice` | bounds within the referent | the preferred and normally the only form |
| `&[T]` → `Ref` to an `Array`/`Vec` | permitted **only** with an inventoried producer | that producer named in A6 | not open-ended: a row admitted because a real producer needs it, recorded with the producer |
| `&[T]` → `Vec`/`Array` immediately | **FORBIDDEN** | — | same ownership error as `&str`→`String`; this is the exact pairing INV-VALUE-REP-001 already rejects |
| `[T]` standalone | **none** | — | unsized; legal only behind a reference |

### 6.5 Iterators, resources and opaque values

Each is a distinct `CoreType`, and each has exactly one representation. They are listed
individually rather than as "iterators" because a class row is a wildcard.

| Semantic `Ty` | Permitted `ValueKind` | Rationale |
| --- | --- | --- |
| `CoreType::CharsIter` | `CharsIter` | |
| `CoreType::SplitIter` | `SplitIter` | |
| `CoreType::VecIter` | `VecIter` | place-backed: iterating a borrow must not copy the container |
| `CoreType::KeysIter` | `HashMapKeysIter` | |
| `CoreType::ValuesIter` | `HashMapValuesIter` | |
| `CoreType::Iter` | `HashMapIter` **or** `HashSetIter` | one core type serves both containers; the container decides. Both are named, so this is a closed pair rather than a wildcard |
| `CoreType::MapIter` | `MapIter` | |
| `CoreType::FilterIter` | `FilterIter` | |
| `CoreType::Random` | `Random` | |
| `CoreType::File` | `File` | a resource: identity matters, so it must never be copied to satisfy a check |
| `CoreType::IOError` | `IOError` | |
| `CoreType::Ordering` | `Ordering` | prelude enum with no HIR item; variants resolve to builtins |
| `Ty::Fn { .. }` | `Function` | |

**Iterator item representation is where DEV-138 lives.** `SplitIter` yields `&str` items and
`VecIter` yields `&T`; those items are governed by 6.4, not by the iterator's own row.

### 6.6 A SEPARATE table: types forbidden at runtime boundaries

**This is not part of the 34-row relation above and must not be read as an extension of it.** The
relation maps a runtime type to permitted representations; this table names types that may not reach
a value boundary at all, so they have no representations to permit.

Keeping them apart matters most for slices: `Value::Slice` is a *valid* runtime representation and
holds a row in 6.4, while standalone `Ty::Slice` is an unsized type that must never independently
cross a boundary. One name, two tables, opposite meanings.

Reaching a boundary with any of these is an internal compiler defect, reported as
`FailureClass::InternalInvariant` — never a trap:

| `Ty` | Why |
| --- | --- |
| `Never` | a value of `!` cannot exist; producing one means a diverging path returned |
| `Infer(_)` | inference did not complete before execution |
| `Error` | a type error reached runtime |
| `Param(_)` unsubstituted | the generic frame did not cover this parameter — the failure mode `concrete_runtime_ty` exists to catch |
| `Slice(_)` standalone | unsized; only `&[T]` is a value |
| `Extension(_)` | not executable in Core v1; reaching the oracle means extension gating failed. Tensor, model and model-error types live *inside* this variant rather than beside it — `Ty` has no separate `Tensor`/`Model`/`ModelError` arm |

### 6.7 Variant coverage

All 34 `ValueKind`s, each appearing in exactly one row of the **relation** (6.2–6.5). The forbidden
type list in 6.6 contributes no rows and is counted separately:

`Unit` `Bool` `Int` `Float` `Char` `Str` `String` `Tuple` `Array` `Struct` `Enum` `Vec` `Boxed`
`Option` `Result` `Range` `Slice` `Ref` `Function` `CharsIter` `SplitIter` `VecIter` `HashMap`
`HashSet` `HashMapKeysIter` `HashMapValuesIter` `HashMapIter` `HashSetIter` `MapIter` `FilterIter`
`Random` `IOError` `File` `Ordering`

### 6.8 Decisions recorded against §3's ambiguities

| Ambiguity | Decision | Reason |
| --- | --- | --- |
| `&T` for scalar `T`: `Ref` required, or bare scalar permitted? | **Both permitted, gated on the pointee being `Copy`** | Copying a Copy pointee is unobservable — it cannot consume, invalidate or destroy the referent. Stated as a semantic predicate on the pointee so it cannot decay into an exception list. |
| `Str` versus `String`: two representations or one? | **Two, kept — for now** | Collapsing them is the larger change and A0–A6 do not need it. The matrix makes the *distinction* enforceable, which is what tells us which producers are actually wrong. Collapsing before knowing that would erase the evidence. Revisit after A6's producer inventory. |
| `Slice` versus `Vec` for `&[T]` | **`Slice`; `Ref`-to-container only with an inventoried producer** | Settled for `let` already; the inventory requirement stops "permitted" from widening silently. |

### 6.9 Not yet enforced

The matrix is a statement of intent at A1. Only the `let`-binding subset of 6.4 is executed today
(INV-VALUE-REP-001). A2 implements the relation; A4 wires the boundaries; 121B covers typed
mutation. **DEV-121 stays open until then** — this document is the specification of the close, not
the close.

