# AS3 Boundary 4 — Display characterization

**Packet:** AS3, `WP-CALLABLE-USE-TOTAL.md` Boundary 4.
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-07.
**Status:** COMPLETE. Measurement only — nothing was changed. **One design gate found, unresolved
by design.**

Ordered before the Display publication because `display_deep` recurses through composites at
runtime and reaches `Display::fmt` at any depth, so what the checker must publish depends on what
the renderer actually does. Measured with `examples/as3_display_probe.rs`.

---

## 1. What the compiler does today

Fixtures: `A` and `B` are distinct nominals with `Display`; `W<T>` is generic with `Display`.

| # | Case | Accepted | Output | `callable_uses` | of which Display |
| ---: | --- | :---: | --- | ---: | ---: |
| 1 | `println(a)` — top-level nominal | yes | `A!` | 0 | 0 |
| 2 | `println((a, b))` — two nominals | yes | `(A!, B!)` | 0 | 0 |
| 3 | `println((W<Int32>, W<Bool>))` | yes | `(W!, W!)` | 0 | 0 |
| 4 | `println(vec)` — `Vec<A>`, 2 elements | yes | `[A!, A!]` | 0 | 0 |
| 5 | `println(Some(a))` | yes | `Some(A!)` | 0 | 0 |
| 5 | `println(Ok(a))` — `Result<A, B>` | yes | `Ok(A!)` | 0 | 0 |
| 7 | `println(W<A>)` — nominal inside a generic | yes | `W!` | 0 | 0 |
| 8 | `println((a1, a2))` — same nominal twice | yes | `(A!, A!)` | 0 | 0 |
| 6 | **`fn show<T: Display>(x: T) { println(x); }`** | **yes** | `A!` | 2 | 0 |
| 6 | **`fn show2<P: Display, Q: Display>(x: (P, Q))`** | **yes** | `(A!, B!)` | 2 | 0 |

`display_uses = 0` throughout is expected: Boundary 4 has not published anything yet. What the
table establishes is the *rendering semantics the publication must match*.

---

## 2. Three facts the design depends on

### 2.1 The STOP rule is real (case 7)

`println(W<A>)` prints `W!`, not `W!` containing `A!`. The outer nominal's own `Display::fmt` runs
and the renderer does **not** descend into its fields.

So the checker's type walk must stop at the first nominal with a `Display` impl. Descending further
would publish uses the renderer never executes, and totality would then claim something false.

### 2.2 One expression really is many uses (cases 2, 3, 4, 8)

- case 2: one argument expression, **two** distinct `fmt` bodies;
- case 4: **one** use executed once per element;
- case 8: the same nominal at two tuple positions — one body, one environment, reached twice;
- case 3: the same *body* at two different instantiations.

Case 3 is the one that rules out choosing by nominal at runtime. `W<Int32>` and `W<Bool>` are the
same `ItemId` in a `Value::Struct { item, fields }` — the runtime value carries **no type
arguments** — so a nominal-keyed lookup cannot tell their environments apart. `DisplayPath` (the
static structural position) distinguishes them; the runtime does not have to.

### 2.3 The renderer's shape is the walk the checker must mirror

```text
primitive / String        no user callable
Tuple<A, B>               recurse at TupleField(0), TupleField(1)
Array<T, N> / Vec<T>      recurse at ArrayElement / VecElement
Option<T>                 recurse at OptionSome
Result<T, E>              recurse at ResultOk, ResultErr
user nominal + Display    publish, then STOP
```

This is not making the checker a renderer. It is producing the static dispatch plan for rendering
semantics `display_deep` and `emit_display_value` already implement.

---

## 3. The design gate: `Display` inside a generic body

**Both gate cases are accepted.**

```stark
fn show<T: Display>(x: T) { println(x); }        // accepted
fn show2<P: Display, Q: Display>(x: P, y: Q) { println((x, y)); }   // accepted
```

A generic body is checked **once**, with `T` unbound. So at the `println(x)` site *inside* `show`,
there is no `Display::fmt` body to name — the body is not determined until `show` is instantiated,
and one `show` may be instantiated at several types.

**Therefore `CalleeSelection::Static { declaration, body }` cannot express this site.** The model as
committed has exactly two variants, and neither fits:

| Variant | Why it does not fit |
| --- | --- |
| `Static { declaration, body }` | there is no body yet |
| `FunctionValue` | there is no function value; the dispatch is by trait bound |

### This is not a Display problem

It is the general shape of **bound dispatch**, which AS3 already has a name for:
`DispatchProvenance::Bound { trait_item }` exists, and `bound_trait_calls` is the checker's existing
record of "this call resolved through a generic parameter's bound". Display is simply the first
place `CalleeSelection` is forced to admit it.

### Options, not resolved here

1. **Extend `CalleeSelection` with a bound variant** — e.g. `Bound { trait_item, member }`, meaning
   *the body is determined by the instantiated `Self` at this use's environment*. The engines then
   resolve it through the instantiation they already carry, not by scanning.
2. **Reuse whatever AS3 establishes for bound method calls generally.** Boundary 2 published
   `Bound` provenance for method calls but always with a `Static` selection, because a method call
   through a bound *is* monomorphised before execution. Whether the same holds for Display inside a
   generic body needs the same treatment or an explicit difference.

**Deliberately not chosen here.** `WP-CALLABLE-USE-TOTAL.md` §7 exit criterion 5 requires
disagreements to be recorded rather than normalised, and this is the model being insufficient rather
than an engine disagreeing — a design decision, not an implementation detail. Inventing a
Display-specific runtime scan to avoid it would be the packet's own defect class.

---

## 4. What the publication must therefore do

1. Walk the **static** type, not the value, mirroring §2.3 and stopping per §2.1.
2. Key uses by `DisplayPath` under the root expression, so cases 2, 3 and 8 are distinguishable.
3. Handle the generic-body case per §3, once decided.
4. Publish the environment for the nominal's `Display` impl, instantiated — the same requirement
   the Iterator hardening established after publishing an empty one.

The interpreter then recurses value **and** static type together, using the existing
`concrete_runtime_ty` (which already substitutes through the active generic frame via
`typecheck::substitute_ty`), and looks up `(root_expr, path)`. No HIR scan, no name lookup, and no
generic arguments added to `Value`.

MIR is easier: `emit_display_value` already carries the full `MirTy` through its recursion, so the
same path index applies directly.
