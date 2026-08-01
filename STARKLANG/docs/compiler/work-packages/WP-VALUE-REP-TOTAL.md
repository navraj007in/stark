# WP-VALUE-REP-TOTAL — a total type→representation mapping for the oracle

**Status:** FILED, not started.
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
