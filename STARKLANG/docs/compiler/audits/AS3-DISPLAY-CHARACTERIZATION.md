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

### RESOLVED (2026-08-07): extend `CalleeSelection` with a real `Bound` variant

Owner ruling. Recorded in `WP-CALLABLE-USE-TOTAL.md` §8 and implemented at `6a1d4d3`/`0c99000`.

Option 2 above rested on a **false premise**, corrected on review: Boundary 2 does *not* publish a
`CallableUse` for bound dispatch at all. `resolve_method`'s bound branch records `bound_trait_calls`
and returns before the publication. So this is not Display finding an unusual corner — AS3 found a
**missing third binding time**, and `fn f<T: Speak>(x: T) { x.speak(); }` had the same hole.

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


---

## 5. Two further characterizations (2026-08-07)

Run before wiring the bound specialiser, on the principle that a data structure is not evidence
about what the language accepts.

| # | Case | Accepted | Output | `callable_uses` |
| ---: | --- | :---: | --- | ---: |
| G1 | trait **default** reached through a bound — `impl Describe for A2 {}` with no override, `x.text()` in `f<T: Describe>` | **yes** | `default` | 3 |
| G2 | **method generics** through a bound — `x.to::<Int32>()` in `g<T: Conv>` | **yes** | `1` | 3 |

### G1 — the index must carry effective targets, not written members

There is no impl member body. The executable target is the **trait default**, which A3b already
treats as executable and gives a `callable_types` entry. A `TraitImplIndex` recording only members
physically written inside an impl would find nothing here and the specialisation would fail on a
program that compiles and runs today.

So the index must record, per member name, the **effective** target:

```text
impl override exists   -> ImplMember
otherwise trait default -> TraitMember
otherwise               -> impossible for a checked impl
```

### G2 — `Bound` needs `method_args`, and this is a real gap

`CalleeSelection::Bound` carries `trait_args` but **no method arguments**, and this form is
accepted. A trait method may declare its own generics, and the turbofish at a bound call site
supplies them:

```stark
fn g<T: Conv>(x: T) -> Int32 { x.to::<Int32>() }
```

Without `method_args` the specialiser cannot produce the complete environment — it would bind the
impl's parameters and silently drop the method's. That is the same class as the Iterator hardening's
empty environment, found before it was built rather than after.

**Recorded, not inferred.** The alternative reading — "`Display::fmt` has no method generics, so this
need not enlarge the packet" — is only true of Display; the `Bound` machinery is general, and step 2
already published it for arbitrary user-trait bounds.

### Consequence for step 4

Both findings land on the same structure, so they are one change rather than two:

```text
IndexedImpl
    trait identity
    trait arguments          <- match with the SAME substitution map as Self
    parametric Self
    impl generic names
    effective members         <- impl override OR trait default
        declaration + body
    NO signature              <- callable_types[body] is the sole signature authority
```

and `CalleeSelection::Bound` gains `method_args` alongside `trait_args`.
