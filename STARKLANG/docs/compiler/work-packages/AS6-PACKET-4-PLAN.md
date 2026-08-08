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

## Packet 4A — the inventory (complete, 2026-08-08)

**The 946 reference count overstated the work, in the same direction packet 3 did.** Classified,
the checker's tensor code falls into three groups, and the integration surface is already narrow.

### 1. Extension rule, zero Core dependency — moves untouched

Lines 13026–13479, roughly 450 lines: `TensorGenericSchema`, `TensorDTypeRule`,
`TensorDeviceRule`, `TensorShapeRule`, `TensorResultRule`, `TensorOpDescriptor`, the `TENSOR_OPS`
static table, `BroadcastError`.

These are **standalone types and a table, not methods on `TypeChecker`**. Nothing holds them in Core
but the file they sit in.

### 2. Extension rule needing Core services — moves behind the context

`check_tensor_op` (1333 lines) and the tensor/model/device builders and unifiers:
`build_tensor_type`, `build_device`, `build_cuda_device`, `tensor_dtype`, `dtype_to_ty`,
`unify_tensor_types`, `emit_tensor_unify_error`, `ground_tensor_dims`, `check_tensor_refine`,
`check_model_def`, `check_model_method_call`, `enter/exit_tensor_param_scope`, `TensorParamScopes`.

### 3. Boundary glue — stays in Core

**Each entry point has exactly one caller:**

```text
check_tensor_builtin_call   1
check_tensor_method_call    1
check_tensor_refine         1
build_tensor_type           1
check_model_method_call     0
```

The integration surface is five call sites, not a web.

### The context design, measured rather than guessed

This is the question the ruling flagged as decisive — whether the extracted tensor checker would
need `&mut TypeChecker` and therefore constitute nominal movement. Across all 1333 lines of
`check_tensor_op`, the **Core** services actually used are:

```text
resolve                 unify                  check_expr
extract_const_int       extract_const_int_list extract_dim_generic
get_fix_suggestion      combine_value_range
```

Eight, each used one to four times. Everything else it calls — `build_shape`, `tensor_dtype`,
`dtype_to_ty`, `build_device`, `shape_volume`, `broadcast_to_check` — is a **tensor method calling
another tensor method**, and moves with it.

So the narrow internal context is genuinely narrow: about eight borrowed services. The dependency
direction the ruling requires is achievable without redesigning the `Ty`/inference model, which
means **the single owner-stop condition does not apply** and 4B proceeds as implementation.

### Revised 4B estimate

```text
~450 lines   move untouched (group 1)
~1400 lines  move behind an eight-service context (group 2)
5 call sites become the boundary (group 3)
```

Bounded work, not a 946-site migration. Note the lesson packet 3 already taught: reference counts
mislead in the *helpful* direction too — 175 sites collapsed to a handful of boundaries once the
requirement was "move it behind rather than respell it".

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
