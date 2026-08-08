# AS6 Packet 4 — the tensor type-system boundary, and the AS6→AS7 seam

**Owner ruling, 2026-08-08.** Recorded because the inventory that produced it arrived at the end of
a session, and the plan must survive into the next one intact.

## The finding that forced this

Splitting the checker's remaining references into code versus prose:

```text
typecheck.rs   946 in code,  37 in comments
parser.rs       98 in code,  31 in comments
```

The 946 are not stragglers. They are the tensor **type system**:

```text
TensorKind            61      dtype     155      model  48
tensor_ctx            60      device    139
TensorShapeRule       57
TensorGenericSchema   48
TensorDTypeRule       41
TensorDeviceRule      39
```

That is not Core *interacting* with the extension. It is extension semantics **implemented inside
the Core checker**, so AS6 cannot honestly close while it remains.

## The ruling

> Do not leave the tensor type system in `typecheck.rs` and declare AS6 complete. But also do not
> perform an independent ~946-reference AS6 extraction followed by AS7 re-cutting the same checker.
> Make the tensor type-system extraction the final AS6 packet **and simultaneously the first
> structural boundary AS7 inherits.**

**One cut, not two.** AS7 does not open partway through it, and AS7 does not touch that boundary
again unless evidence says it is wrong.

## The target

Core keeps the checker's *mechanisms* and provides them as services; the extension owns the tensor
*decisions*.

| Core provides | The extension decides |
| --- | --- |
| resolve an expression's type | can these shapes broadcast? |
| concretise a generic parameter | what dtype does this op produce? |
| publish an expression type | are these devices compatible? |
| emit a diagnostic | what does `reshape` do to the dimensions? |
| look up a generic argument | what schema does `matmul` require? |
| unify an ordinary `Ty` | how does a model's output shape propagate? |

## The one design decision that matters

`tensor_ctx`. Moving `TensorContext` and then handing it `&mut TypeChecker<'a>` would relocate the
code while preserving total dependency on the monolith — nominal movement, no boundary.

The dependency direction must be:

```text
Core checker machinery
        ↓ provides limited services
tensor checker
```

not the reverse. A narrow internal context — a `pub(crate)` struct of borrowed services, or an
internal trait — is the mechanism. **Not** a public trait/plugin framework, which the work package
forbids (exit criterion 4).

## Scope discipline

In scope: tensor-owned semantic state, tensor-owned semantic rules, tensor-specific checking entry
points.

**Out of scope, and belonging to AS7:** borrow-checker extraction, generic inference extraction,
flow analysis, callable checking, diagnostic redesign, general checker-context redesign. The high
reference count is not a licence to modularise the whole checker.

## The measure to optimise

Not:

```text
grep "Tensor" typecheck.rs == 0
```

but:

```text
no tensor semantic decision is owned by Core
```

`typecheck.rs` may legitimately end up containing `Builtin::Tensor(op) => tensor::check_builtin(op,
args, &mut tensor_ctx)` and references at the integration boundary. A zero-reference target would
push toward exactly the generic extension framework the programme rejects.

## Execution order

```text
4A  checker tensor semantic inventory — classify each site as
    extension rule / Core mechanism / boundary glue.  No implementation.
4B  extract the tensor type-system authority: TensorContext, shape/dtype/device
    rules, generic schemas, model checking. One or few sealed entry points.
    Generic typechecker mechanics stay in Core.
4C  parser residual audit (read-only first). Classify the 98 code references as
    enablement / grammar recognition / AST construction / semantic knowledge.
    Only (4) must move; centralised syntax dispatch is acceptable — parsing IS
    dispatch, and AS6 does not require a parser plugin architecture. The test is
    whether a new tensor syntax form requires edits in many unrelated places.
```

4A is **not** a separate owner checkpoint: inventory, cut, test and continue.

## The only owner-stop condition

```text
STOP only if tensor semantics cannot be separated without redesigning the
general Ty/inference model.
```

The tensor subsystem merely *using* many checker services is an implementation problem — design the
narrow context and continue.

## Revised AS6 status

```text
architectural discovery        DONE   (46ae2ec)
builtin/catalogue quarantine   DONE   (fe80129)
runtime/lowering boundary      DONE   (33cb0a7)
tensor type-system boundary    OPEN   — substantial
parser residual audit          OPEN   — probably small
qualification                  OPEN
```

AS6 is roughly **65–75% complete**, not the 90+% it looked before the inventory.

**The compensating point:** packet 4 pulls forward part of AS7's modularisation in the one place
AS6's semantics require it. AS7 inherits a large semantic-ownership boundary already cut, a
dependency direction already proven, and a large chunk already out of `typecheck.rs`. The total
sprint estimate does not rise as much as 946 references suggest.
