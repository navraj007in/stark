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
4D  exit-criterion cleanup. NOT in the original order — added by owner ruling on
    2026-08-08, after a deliberate criterion-2 re-read found residual vocabulary
    tables in resolve.rs and typecheck.rs that 4A-4C had each reported clean.
    Split check_model_def, move the classification tables, re-run the census,
    and build the forcing test the work package asked for and no packet made.
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

### CORRECTION (2026-08-08): the service count was measured wrong

4A's "eight Core services" figure came from grepping `self.method(` inside `check_tensor_op`. That
pattern is blind to **field access and non-call uses of `self`**, so three helpers
(`shape_volume`, `broadcast_to_check`, `dtype_to_ty`) looked context-free, were moved as free
functions, and failed to compile — `shape_volume` reads `self.tensor_ctx`. The slice was reverted.

Re-measured over every `self.X` occurrence across all twelve tensor functions:

```text
36  distinct self.X
20  tensor-owned — move with the code
16  CORE surface
```

The Core surface is:

```text
fields    diags · hir · options
methods   check_expr · convert_hir_type · resolve · unify · text · ty_to_string
          extract_const_int · extract_const_int_list · combine_value_range
          build_value_range · generic_kind · get_fix_suggestion · allow_half_type
```

**Sixteen, not eight** — double the estimate, though still narrow beside `&mut TypeChecker`, which
exposes hundreds of members. The ruling's dependency-direction requirement remains achievable.

**The one design constraint this exposes: `check_expr`.** A tensor rule calling it means the
context must permit **re-entry into the general expression checker**, so the extracted module
cannot simply borrow inert data — it needs a callback into Core, and the borrow discipline around
that is the real work in group 2, not the line count.

**Method note, because this mistake has now recurred three times in one sprint:** measure a
dependency by attempting compilation, not by pattern-matching call syntax. `self.method(` is a
proxy for "uses the checker"; it is not the question. The same shape produced DEV-210's
`ends_with("Drop")`, DEV-212's `ty_has_user_drop`, and this.

### Revised 4B estimate

```text
~455 lines   moved untouched — DONE, 62ef6b0
~1400 lines  move behind a SIXTEEN-service context, one of which is `check_expr`
             and therefore requires re-entry into Core (group 2, open)
5 call sites become the boundary (group 3, open)
```

Bounded work, not a 946-site migration. Note the lesson packet 3 already taught: reference counts
mislead in the *helpful* direction too — 175 sites collapsed to a handful of boundaries once the
requirement was "move it behind rather than respell it".

## Revised AS6 status

*(Superseded by "Packet 4D — Status" at the end of this document; kept as the position after 4A.)*

```text
architectural discovery        DONE   (46ae2ec)
builtin/catalogue quarantine   DONE   (fe80129)
runtime/lowering boundary      DONE   (33cb0a7)
tensor type-system boundary    OPEN   — substantial
parser residual audit          OPEN   — probably small
qualification                  OPEN
```

AS6 was, at that point, roughly **65–75% complete**, not the 90+% it looked before the inventory.

**The compensating point:** packet 4 pulls forward part of AS7's modularisation in the one place
AS6's semantics require it. AS7 inherits a large semantic-ownership boundary already cut, a
dependency direction already proven, and a large chunk already out of `typecheck.rs`. The total
sprint estimate does not rise as much as 946 references suggest.


---

# Group 2A — the `check_expr` edge classification (complete, 2026-08-08)

**Result: the one-directional design survives the code. Three edges, zero semantic recursion.**

| Site | Classification |
| --- | --- |
| `check_tensor_op` +38 | **argument evaluation**, already hoisted — a `for arg in args` loop filling `actual_ops` before any rule runs |
| `check_tensor_refine` +9 | **error recovery** — the result is discarded, and the function then diagnoses that value arguments are not allowed |
| `check_model_method_call` +98 | **argument evaluation**, but interleaved with the compatibility check rather than hoisted |

## The ordering caution has a concrete answer

`check_tensor_op` validates the call's **form** before typing any argument, each failure returning
`Ty::Error` early:

```text
descriptor lookup             -> unknown tensor operation
receiver && !descriptor.method -> not a method
!receiver && !standalone       -> requires a receiver
────────────────────────────────  then check_expr on the arguments
```

That order is externally observable in diagnostics, and the phase split preserves it without
special handling: form validation and argument evaluation are **both** Core-side preparation. Only
the rules that follow move.

## The one site needing care

`check_model_method_call`'s `check_expr` sits inside a `zip` over instantiated model inputs, so
hoisting it must not reorder argument diagnostics against compatibility diagnostics. Per the
ruling, externally observable diagnostic behaviour is preserved — this site is staged, not simply
lifted.

## Consequence

2B proceeds as designed: `TensorCheckInput` carries already-typed arguments, no extension-owned
function receives an `ExprId` to typecheck, and `check_expr` is not in the context. The owner-stop
condition is **not** triggered — no tensor rule requires conditionally typechecking a previously
unchecked expression.

## The design rule this establishes

```text
A narrow dependency surface is necessary but not sufficient.

AS6 also requires one-directional semantic control:
Core may enter extension checking;
extension checking must not recursively enter the general Core expression checker.
```

`resolve`, `unify`, `ty_to_string`, constant extraction and value-range operations are **capability**
dependencies and are acceptable. `check_expr` is a **control-flow re-entry** edge and is not. Those
two were wrongly treated as equivalent members of one context, and separating them is the finding —
not the fact that the count doubled.


---

# Group 2B — the boundary located and measured (2026-08-08)

`check_tensor_op` is 1332 lines. **1292 of them follow argument evaluation as one contiguous
block, and `check_expr` does not appear in that remainder.** The phase boundary is therefore real
in the code, not merely in the design.

## The context, exactly

| Core services — 7 | Tensor-owned — 9, move with the block |
| --- | --- |
| `resolve`, `unify`, `diags` | `broadcast_shapes`, `broadcast_to_check` |
| `extract_const_int`, `extract_const_int_list` | `build_device`, `build_shape`, `dtype_to_ty` |
| `combine_value_range`, `get_fix_suggestion` | `shape_volume`, `tensor_ctx`, `tensor_dtype`, `extract_dim_generic` |

**Seven, against the sixteen-with-`check_expr` of the withdrawn design.** The split does not merely
remove the bidirectional edge — it more than halves the surface, because the services that existed
only to support *argument evaluation* (`check_expr`, `convert_hir_type`, `hir`, `options`, `text`,
`generic_kind`, `ty_to_string`, `allow_half_type`, `build_value_range`) stay in Core, where that
work now lives.

## What 2C moves

```text
Core keeps      operation lookup and call-form validation (uses TENSOR_OPS across
                the boundary), argument evaluation, and the final publish
Extension takes the 1292-line rule block, behind a seven-service context
```

Form validation stays Core-side because it must run **before** argument evaluation to preserve
diagnostic order (2A), and both are preparation. Only the rules move.

## Residual risk

The seven-service count is measured over `check_tensor_op`'s remainder alone. `check_tensor_refine`,
`check_model_def` and `check_model_method_call` have their own surfaces and were measured
separately in the 4A correction; they must be re-measured **over their post-evaluation remainders**
before 2C, by the same method. Measuring the whole function overstates the boundary, which is what
produced the sixteen figure.


## The remaining three entry points, measured over their remainders

```text
                          total  remainder  re-entry
check_tensor_refine          92         82     none
check_model_def              87         87     none   (no check_expr at all)
check_model_method_call     143         44     none
```

**No re-entry in any remainder**, so the one-directional design holds across the whole surface.

But the union of Core services is **14, not 7**, and the extra seven come almost entirely from one
place:

| Entry point | Core services in its remainder |
| --- | --- |
| `check_tensor_op` | resolve, unify, diags, extract_const_int, extract_const_int_list, combine_value_range, get_fix_suggestion |
| `check_tensor_refine` | build_value_range, text |
| `check_model_method_call` | diags, hir, resolve, ty_to_string |
| **`check_model_def`** | **convert_hir_type, generic_kind, options, resolve, ty_to_string, diags, text** |

### The finding: model DECLARATION checking is a different slice

`check_model_def` has no `check_expr` at all — its remainder is the whole function — and it needs
`convert_hir_type`, `generic_kind` and `options`. That is because declaring a model means
**converting written type syntax**, which is Core machinery by nature, not a tensor semantic
decision.

So 2C should not treat the four entry points as one extraction:

```text
tensor OPERATION rules        7 services, clean, extract first
model METHOD-call checking    4 services, subset of the above plus hir
tensor refine                 2 services, trivial
model DECLARATION checking    pulls in the type-conversion machinery — evaluate
                              separately; it may be more Core-mechanism than
                              extension-rule, and forcing it behind the same
                              context would widen the interface by 4 for one caller
```

Extracting the first three gives the boundary AS6 wants at a seven-service surface. Whether
`check_model_def` follows is a judgement about where model *declaration* validation belongs, and
should be decided on its own evidence rather than by grouping.


---

# Group 2C — the tensor semantic authority extracted (2026-08-08)

`starkc/src/extensions/tensor/check.rs`, 1736 lines. `typecheck.rs` 15,937 → 14,472.

## What moved, and what Core kept

| Entry point | Core keeps | Extension takes |
| --- | --- | --- |
| `check_tensor_op` | `TENSOR_OPS` lookup, call-form validation, argument evaluation, publish | the 1290-line post-evaluation remainder |
| `check_tensor_refine` | the `check_expr` loop and the "no value arguments" error | what a refinement produces (82 lines) |
| `check_model_method_call` | HIR walk, declaration scope, port conversion, freshening, argument evaluation | method surface, `.predict` arity, the per-argument borrowed-tensor rule, the result shape |

Also moved, because they are tensor semantics that only the rule block calls: `broadcast_shapes`,
`broadcast_to_check`, `can_broadcast_to`, `shape_volume`, `dtype_to_ty`, `get_fix_suggestion`.

`check_model_def` was left in Core, per 2B's finding.

> **SUPERSEDED 2026-08-08 by the owner ruling in packet 4D.** The recommendation below was
> **rejected**: `check_model_def` is *split*, not left in Core. The reasoning below measures the
> cost of moving the function **whole**, which was never the right cut — split by phase, the
> interface widens by **zero**. Kept as written because the error it contains is the finding.
> See "Packet 4D — Ruling 1".

Measured against the context 2C actually
built — rather than against 4A's guess — the case for leaving it is stronger than 2B thought:

```text
check_model_def   87 lines, 1 caller, no check_expr
services          diags · ty_to_string · resolve        already in TensorCheckCtx
                  convert_hir_type · generic_kind · options · text
                  enter_tensor_param_scope · exit_tensor_param_scope    SIX new members
```

Six new services — 15 → 21, a 40% wider interface — for one 87-line function with one caller, and
five of the six exist to *convert written type syntax and manage the declaration scope*, which is
Core machinery by the same rule that kept `build_shape` and friends in Core. Nothing in
`check_model_def` decides a dtype, shape, device or broadcast; it validates that a written model
declaration is well-formed. ~~**Recommendation: leave it in Core and close AS6 without it**~~ —
**overturned by the owner, 2026-08-08.** The six-service figure is the cost of moving the whole
function; the rules alone need three services the context already had.

## The context is fifteen services, not seven — and 2B's seven was measured wrong twice

2B's table listed seven Core services and nine tensor-owned helpers "moving with the block". Two of
those classifications did not survive compilation:

```text
get_fix_suggestion   listed CORE       — is pure tensor semantics (tensor_ctx + broadcast_to_check).
                                         Moved. Removes itself from the surface.
build_shape          listed tensor-owned, "moves with the block"
build_device         listed tensor-owned, "moves with the block"
tensor_dtype         listed tensor-owned, "moves with the block"
extract_dim_generic  listed tensor-owned, "moves with the block"
                                       — none of them can move. Each reads the generic-parameter
                                         scope tables (`dim_scope`/`dtype_scope`/`device_scope`),
                                         `hir`, `text`, `convert_hir_type` or `allow_half_type`,
                                         and each has callers outside the block.
```

The four that cannot move are **written-type-syntax conversion**, which the same plan already ruled
Core machinery when it deferred `check_model_def` ("converting written type syntax is Core
machinery"). So they stay in Core and are offered as services, and the classification is at least
consistent across the two decisions.

`TensorCheckCtx`, in full:

| Group | Services |
| --- | --- |
| diagnostics | `diags`, `tensor_error` |
| Core type machinery | `resolve`, `unify`, `ty_to_string` |
| constants and ranges | `extract_const_int`, `extract_const_int_list`, `extract_dim_generic`, `combine_value_range`, `value_range_of` |
| written syntax → tensor objects | `build_shape`, `build_refine_shape`, `build_device`, `tensor_dtype` |
| extension state held by the host | `tensor_state` |

**Fifteen — the same order of magnitude as the sixteen-service design 4A withdrew.** That is the
honest number, and it is also the reason the count was never the criterion. What separates this
design from the withdrawn one is not size:

```text
withdrawn   16 services, one of which was check_expr  -> bidirectional control
this        15 services, none of which is check_expr  -> one-directional
```

Every member reads Core state, converts written syntax, or emits. None of them re-enters expression
checking, and none of them can be made to: `TypeChecker`'s fields and methods are private to the
`typecheck` module, so `extensions::tensor::check` can reach **only** what the trait names. The
boundary is compiler-enforced, not conventional.

## Diagnostic order is preserved, and one site had to be staged to keep it

`check_tensor_op` and `check_tensor_refine` needed nothing: form validation and argument evaluation
were already a contiguous Core-side prefix.

`check_model_method_call`'s `check_expr` sat inside the `zip` over instantiated ports (2A's "one
site needing care"). Hoisting it would have reordered every argument diagnostic ahead of every port
diagnostic. Instead the **extension rule was made per-argument**: Core evaluates argument *i*, then
immediately calls `check_model_predict_arg` for argument *i*. Interleaving is unchanged, and
`check_expr` still never appears on the extension side.

One deliberate behaviour-neutral change: the "corresponding model port declared at …" note is now
built eagerly and passed in, rather than built lazily inside the failure branch. It is still
attached only on unification failure.

## How the move was verified — not by reading it

The remainder's 1276-line rule block was moved mechanically and then **proved unchanged** (the
other 14 lines are the `get_tensor_kind` closure, rewritten as the `tensor_kind_of` free function
because a closure capturing `cx` would have conflicted with the mutable borrows around it). Both the
original
(`git show HEAD:…`) and the extracted copy were normalised (comments stripped, rustfmt's line
breaks and trailing commas collapsed, the rename `self.X` → `cx.X` reversed) and compared as byte
strings.

```text
orig 22925   new 22925   IDENTICAL
```

`check_tensor_refine` was verified the same way (1690 == 1690, IDENTICAL, modulo the `range = R`
lookup folded into the new `value_range_of` service). `check_model_method_call` was rewritten by
hand and its diff is four relocations and two comments, nothing else.

This is the answer to the method note 4A recorded three times: a proxy for the question is not the
question. "It still compiles" would not have caught a dropped match arm in 1276 lines; a normalised
identity comparison does.

## The old path is gone, not merely bypassed

```text
get_fix_suggestion  dtype_to_ty  broadcast_shapes
broadcast_to_check  can_broadcast_to  shape_volume     0 occurrences in typecheck.rs
TensorShapeRule  TensorDTypeRule  TensorDeviceRule
TensorGenericSchema  TensorResultRule  BroadcastError  0 occurrences in typecheck.rs
TENSOR_OPS                                             1 occurrence  (the lookup at the boundary)
```

The glob `use crate::extensions::tensor::rules::*` was narrowed to `TENSOR_OPS` so a future rule
type cannot silently re-enter Core.

## Residual, and what 4C inherits

Case-insensitive `tensor|dtype|device|model` occurrences in `typecheck.rs`: **1152 → 698** (as at 2C; 4D moved four more tables out and added the `tensor_syntax::` call sites that replace them).

The 698 are one coherent slice, not scattered residue: **tensor type *construction* and model
*declaration*** — `build_tensor_type`, `build_shape`, `build_device`, `build_cuda_device`,
`tensor_dtype`, `enter`/`exit_tensor_param_scope`, `unify_tensor_types`,
`emit_tensor_unify_error`, `ground_tensor_dims`, `freshen_call_ty`, `check_model_def`, the
`convert_hir_type` tensor arms, and the `TensorCheckCtx` impl itself.

That is precisely the boundary the ruling said may legitimately remain (`typecheck.rs` "may
legitimately end up containing … references at the integration boundary"), plus the declaration
slice this packet recommends leaving there. **No tensor semantic *decision* — dtype, shape, device,
schema, broadcasting, model calling convention — is owned by Core any more**, which is the measure
the ruling said to optimise.

## Status

*(Superseded by "Packet 4D — Status" at the end of this document; kept as the position after 2C.)*

```text
architectural discovery        DONE   (46ae2ec)
builtin/catalogue quarantine   DONE   (fe80129)
runtime/lowering boundary      DONE   (33cb0a7)
tensor type-system boundary    DONE   (62ef6b0 rules, 2C authority)
  └─ model DECLARATION slice   recommendation later OVERTURNED — see packet 4D
parser residual audit          OPEN   — 4C
qualification                  OPEN
```


---

# Packet 4C — the parser residual audit (2026-08-08)

## The classification

`parser.rs` is 4027 lines and matches `tensor|dtype|device|model|shape` 227 times. **159 of those
are in code, and 80 of the 159 are inside the `#[cfg(test)]` module** that starts at line 3258. The
production surface is **79 references**, and they fall into the four groups 4C named:

| Group | What it is | Verdict |
| --- | --- | --- |
| **enablement** | seven `tensor_enabled()` gates | **stays** — this *is* the C9.1 per-session mechanism |
| **grammar recognition** | `ShapeGroupKind` and `shape_group_kind`'s bounded scan to the matching `]`; `single_bracket_elem_is_type`; `shape_arg`'s bracket/comma structure; the `dim_expr`/`dim_add`/`dim_mul`/`dim_atom` precedence family; `generic_args`' three-way bracket dispatch; `at_model_item`'s `Ident Ident` lookahead; `parse_model`/`parse_model_port`'s brace-and-semicolon structure | **stays** — parsing *is* dispatch |
| **AST construction** | `GenericArg::Shape`, `ShapeArg`, `DimExprKind::{Lit,Var,Binary,Error}`, `ItemKind::Model`, `ModelDef`, `ModelPort`, `PortDir`, `TypeKind::Primitive` | **stays** — `ast`-owned node kinds, and the harness already established criterion 1 holds for `ast.rs` |
| **semantic knowledge** | **21 hardcoded tensor spellings at 6 sites** | **moves** |

## The structural test: the parser passes it

> *Does a new tensor syntax form require edits in many unrelated places?*

**No.** Every one of the four groups is reached through a single function. A new shape form is one
edit in `shape_arg`; a new dimension operator is one edit in the `dim_*` family; a new model clause
is one edit in `parse_model`. The dispatch is centralised, which is the correct structure for a
recursive-descent parser and is explicitly not something AS6 asks to change (exit criterion 4
forbids the plugin architecture that "fixing" this would produce).

## But it failed criterion 2, and that is a different question

> *Central Core modules do not contain open-ended tensor spelling tables or method catalogues.*

The spellings, before:

```text
site                            spellings
parse_type reserved table        15   QInt8 QUInt8 QInt16 Quantized | NCHW NHWC RowMajor
                                      ColumnMajor TensorLayout | PeakMemory MemoryProfile |
                                      Gradient Grad Tape Autodiff
at_extension_primitive            2   Float16 BFloat16
single_ident_is_shape             1   Tensor
at_model_item                     1   model
parse_item dispatch               1   model
parse_model_port                  2   input output
```

The 15-entry reserved table is the same shape as the two tables AS6 has already moved — fe80129's
resolver spelling table and 62ef6b0's `TENSOR_OPS`. Reserving a name is a statement about the
extension's roadmap, not about Core's grammar: the forms parse identically either way, and every
future dtype, layout, deployment constraint or autodiff type added to the reservation list widened
a `match` inside Core's `parse_type`.

## The cut

`extensions/tensor/syntax.rs`, 95 lines at 4C (4D grew it to the extension's full vocabulary), five items:

```rust
MODEL_KEYWORD                  the model item's contextual keyword
port_direction(name)           -> Option<PortDir>
extension_primitive(name)      -> Option<Primitive>      Float16 / BFloat16
opens_shape_position(name)     -> bool                   which constructor opens a shape position
reserved_type_note(name)       -> Option<&'static str>   the v0.1 reservation list
```

Six call sites in `parser.rs`; **zero of the 21 spellings remain in its production code.** The
parser keeps every decision about *shape* — how a `[...]` group is disambiguated, how a dimension
expression associates, where a model's ports may appear. What it no longer keeps is the extension's
*vocabulary*.

`opens_shape_position` names exactly one constructor today. That is the point: a second
shape-carrying constructor is now added in `extensions/tensor/`, not in Core's `parse_type`.

## What this measurement does **not** say

`parser.rs`'s occurrence count went **up**, 225 → 227, because `tensor_syntax::` is itself a match.

That is the ruling's point made concrete, and it is worth recording as evidence rather than as an
aside: had the packet been optimising `grep Tensor parser.rs == 0`, this commit would have scored
as a regression. Against the measure the ruling actually set — *no tensor semantic decision is
owned by Core* — it moves 21 spellings out and the reference count is not evidence of anything.

## Status

*(Superseded by "Packet 4D — Status" at the end of this document; kept as the position after 4C.)*

```text
architectural discovery        DONE   (46ae2ec)
builtin/catalogue quarantine   DONE   (fe80129)
runtime/lowering boundary      DONE   (33cb0a7)
tensor type-system boundary    DONE   (62ef6b0 rules, 9147073 authority)
  └─ model DECLARATION slice   recommendation later OVERTURNED — see packet 4D
parser residual audit          DONE   (4C)
qualification                  OPEN
```


---

# Packet 4D — exit-criterion cleanup (owner-directed, 2026-08-08)

**Two owner rulings opened this packet, and both overturned a position recorded above.** They are
restated here in full because the sections they correct remain in the document.

## Ruling 1 — `check_model_def` is split, not left in Core

Group 2C recommended leaving `check_model_def` wholly in Core on the grounds that moving it would
widen `TensorCheckCtx` from 15 to 21. **Rejected.** The work package puts extension-owned names,
type rules, methods *and diagnostics* behind sealed tensor modules, and `check_model_def` still
decided extension policy: model generics must be `Dim`, ports may only be `Tensor`/`TensorDyn`,
port names must be distinct, and a model must have at least one input and one output.

The error in 2C's reasoning was to treat the function as indivisible. Split by phase — the same
split 2C itself established — the interface does not widen **at all**:

```text
Core                                   extensions::tensor
  enter generic scope
  classify each parameter        ──▶     generic-kind validity
  extract name / direction / span ──▶    duplicate port rule
  convert port type              ──▶     allowed port-type rule
                                         ≥1 input, ≥1 output
                                 ◀──    diagnostics, published by Core
```

The rules need `diags`, `resolve` and `ty_to_string` — **three services, all already among the
fifteen. Zero widening.** 2C's own measurement (six new services) was of moving the function
*whole*, which was never the right cut.

**Staged, not hoisted**, for the reason `check_model_method_call` was: `convert_hir_type` can emit
(a malformed `Tensor<...>` port), and the original per-port order is duplicate-name diagnostic →
conversion diagnostics → port-type diagnostic. Core drives the loop and calls `declare_port` before
its conversion and `check_port_type` after, reproducing that interleaving exactly rather than
approximating it.

## Ruling 2 — criterion 2 FAILED on `5190d1b`

`resolve.rs` still opened with a 15-name `extension_reserved_name` table — `Dim`, `DType`,
`Device`, `Float16`, `Tensor`, `TensorDyn`, `Cpu`, `Cuda`, the value ranges, `ModelError` — the
same architectural shape 4C had just removed from `parser.rs`. Three smaller vocabulary
authorities were also still in `typecheck.rs`: the `Dim`/`DType`/`Device` classifier, the
`Cpu`/`Cuda` classifier, and the value-range classifier.

The instruction was precise and is what made this bounded: **move classification tables, not the
functions that use them.** Core keeps HIR traversal, written-type conversion, scope lookup, const
extraction and inference; the extension owns `"Dim"` → kind, `"Cpu"` → device constructor,
`"ByteRange"` → value-range state, reserved name → description.

## What the census found beyond the ruling's list

4D-D re-ran criterion 2 across parser, resolve, typecheck, hir, interp and deploy/lower. It found
**four more** authorities the ruling had not enumerated:

| Site | What it was |
| --- | --- |
| `typecheck.rs` `build_tensor_type` | a four-arm **type-constructor table** — `"TensorAny"`, `"TensorDyn"`, `"Tensor"`, `"ModelError"` — dispatching on the spelling and repeating it in every arity diagnostic |
| `typecheck.rs` `ty_to_string` | `ExtensionTy::ModelError => "ModelError"` |
| `mir/lower.rs` | `ItemKind::Model(_) => "model"` |
| `deploy/lower.rs` | a **third** copy of the type-constructor spellings in `deploy_ty_from_ast`, and a **fourth** copy of the element-type spellings in `dtype_by_name` |

The type-constructor set existed in three places and the dtype-name set in four. Both are now one
table each, in `extensions/tensor/syntax.rs`, with round-trip tests pinning that the parse and
print directions cannot drift.

## The result

`extensions/tensor/syntax.rs` grew from 95 lines to the extension's full surface vocabulary:

```text
MODEL_KEYWORD  port_direction  port_keyword          model syntax
extension_primitive  dtype_by_name  dtype_of_primitive    element types
opens_shape_position  tensor_type_constructor             type constructors
tensor_param_kind  device_constructor  value_range_state  kinds, devices, ranges
extension_type_name  reserved_type_note                   owned and reserved names
TENSOR_PARAM_KIND_EXPECTATION  DEVICE_EXPECTATION
VALUE_RANGE_EXPECTATION                                   the phrases that recite them
```

The three `*_EXPECTATION` constants matter as much as the tables. A diagnostic that says
"expected `Cpu`, `Cuda<N>`, or a `Device` parameter" *is* the vocabulary, and leaving it in Core
would mean a fourth device constructor lands with Core still reciting a list of two.

## 4D-E — the forcing function AS6 asked for and never built

The work package lists a deliverable none of the four implementation packets produced:

> Add dependency/lint tests preventing new tensor imports in designated Core-only modules.

`starkc/tests/as6_core_module_vocabulary.rs` is it. Three tests over sixteen files:

1. **no Core module spells an extension name** — the vocabulary is read back out of
   `extensions/tensor/syntax.rs` itself, so a name added there is automatically a name Core may not
   spell;
2. **the three surfaces AS6's inventory found clean stay at zero**;
3. **AS6 introduced no `pub` item** in the modules it created (criterion 4).

It checks *string literals*, not reference counts, because 4C demonstrated that counts move the
wrong way: moving 21 spellings out of `parser.rs` **raised** its match count from 225 to 227.

**The exemption list is a ledger, not a skip-list.** The assertion is set equality, so a new
violation fails *and so does removing an accepted one without updating the list*. Four entries,
each with its reason:

```text
ast.rs "Float16"/"BFloat16"   Primitive::name — exhaustive rendering of a CLOSED Core enum.
                              Adding a dtype means adding a Primitive variant, which the compiler
                              forces everywhere; it is not a table that grows silently. Sealing it
                              is the cut fe80129 made for hir::Builtin's 33 variants and is wider
                              than AS6 scoped.
deploy/ir.rs "TensorAny"      Display for DeployTy, the deployment IR's own closed enum.
deploy/emit.rs "Tensor"       the GENERATED RUST host's type name. It coincides with the STARK
                              spelling; it is not one.
```

## Method note

The lint earned its place before it was committed. A shell census run first reported **zero**
spelling literals across eleven Core files; the loop had passed an unsplit `$FILES` variable (zsh
does not word-split unquoted variables), so `grep` matched nothing and the empty output read as a
clean result. The test found `ast.rs` immediately.

That is the fourth time in this sprint a proxy that resembled the question has produced a
confident wrong answer — after `self.method(`, `ends_with("Drop")` and `ty_has_user_drop`. The
compensating discipline is the same each time: make the check executable and let it run.

## Status

**This is the authoritative status block. The three above it are dated positions, each bannered.**

```text
architectural discovery        DONE   46ae2ec
builtin/catalogue quarantine   DONE   fe80129
runtime/lowering boundary      DONE   33cb0a7
tensor type-system boundary    DONE   62ef6b0 rules, 9147073 authority
  └─ model DECLARATION slice   DONE   4D-A — split, per owner ruling; 2C's
                                      leave-in-Core recommendation was rejected
parser residual audit          DONE   5190d1b (CI green, 28/28 jobs)
exit-criterion cleanup         DONE   6050efa
  ├─ resolver vocabulary       4D-B
  ├─ checker vocabularies      4D-C
  ├─ criterion-2 census        4D-D — four authorities beyond the ruling's list
  └─ forcing lint              4D-E — the work-package deliverable no packet had built
qualification                  OPEN   — on the 4D head, per owner instruction
```

Measured at 4D:

```text
extensions/tensor/check.rs    1862 lines      the semantic authority
extensions/tensor/syntax.rs    419 lines      the surface vocabulary
typecheck.rs                 14432 lines      from 15,937 at the start of packet 4
resolve.rs occurrences         101 -> 85
```

## The finding worth carrying into AS7

**Criterion 2 was the only one of the five that decays silently, and it is why residue survived
three packets after its surfaces were declared done.**

Criteria 1 and 3 are behavioural: the two-directional suite catches a regression in either, in both
directions, on every run. Criterion 2 has no behavioural signature at all. A spelling table does not
come back in one commit — it comes back one arm at a time, because somebody adds `"Float8"` to a
match in `parse_type` where the surrounding code already is, and every test still passes.

That is exactly what had happened: `resolve.rs` kept a 15-name table through three packets that each
reported their surface clean, and the census then found the type-constructor spellings in **three**
places and the element-type spellings in **four**. No test could have told anyone.

Two consequences for AS7, which will make many more ownership cuts than AS6 did:

1. **A structural criterion needs a structural check, committed at the same time as the cut** —
   not a procedure, and not a reviewer's grep. 4D-E is that check for AS6.
2. **Prefer criteria with behavioural signatures where the choice exists.** AS7's exit criterion 1
   ("no semantic behaviour or diagnostic structure changes") has one; its criterion 2 ("dependency
   direction is documented and cycle-free") does not, and should get its executable check written
   before the modularisation starts rather than after.

## AS6 closeout checklist — propagation order

Nothing below runs until **exact-head qualification PASSes**. No document may say `AS6 CLOSED`
before the evidence demonstrates it.

```text
evidence  ->  canonical state  ->  derived contributor documentation
```

not the reverse.

```text
After exact-head AS6 qualification PASS:

1. COMPILER-STATE.md                    canonical closure FIRST
2. WP-ARCHITECTURE-STABILIZATION.md     packet status, Sprint 4 progress
3. CLAUDE.md
4. AGENTS.md
5. compiler-map / architecture docs     if AS6 changed their claims
6. tensor-extension position/status docs
```

Items 3–6 are downstream summaries. They may lag; they must never lead. The reason for the
ordering is the failure this repo has already paid for: a downstream document describing behaviour
the canonical source has not scoped leaves implementers to pick between two answers, and whoever
ships first picks arbitrarily.

**Governance preconditions — both resolved 2026-08-08, before qualification, deliberately.**

```text
Campaign B approval    RESERVED -> APPROVED    WP-ARCHITECTURE-STABILIZATION.md §1
Branch identity        sprint-3 -> sprint-4    from 6050efa, history retained
```

They were resolved first so that qualification answers **one** question — does AS6 satisfy its
technical exit contract? — rather than simultaneously repairing who authorised the work and which
sprint is executing it.
