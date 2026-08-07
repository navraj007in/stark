# AS3 Boundary 4 — the selector census, and the deletion gate

**Packet:** AS3, `WP-CALLABLE-USE-TOTAL.md` Boundary 4.
**Branch:** `wp-arch-stability/sprint-3`. **Date:** 2026-08-07.
**Status:** OPEN — this is the gate, not a closure.

---

## 1. Why this document exists

After the MIR method and operator fallbacks were deleted (`c34cd34`), the remaining-work statement
said Display consumption plus interpolation would let both selector functions be deleted.

**That was wrong**, and the error is worth naming because it is a recurring one: the sites I had
worked on recently were vivid, and the older Boundary-2/3 sites had become invisible. A first census
using `grep "self\.find_method("` reinforced the mistake by **missing every multi-line call chain** —
`self\n    .find_method(` does not match that pattern, and the two qualified-trait callers are
written exactly that way.

The exit criterion for Boundary 4 is **structural**:

```text
fn find_method       DOES NOT EXIST
fn find_impl_fn      DOES NOT EXIST
```

not "Display no longer calls them". This document enumerates every live caller so that claim can be
checked rather than asserted.

**Method** (both forms, because the first one alone under-reports):

```bash
grep -nE "\.find_method\(|find_method\($"   starkc/src/interp.rs
grep -nE "\.find_impl_fn\(|find_impl_fn\($" starkc/src/mir/lower.rs
```

---

## 2. HIR interpreter — 7 live callers

| # | Site | What it selects | Published metadata available? |
| ---: | --- | --- | --- |
| 1 | `call_method` ordinary fallback | any method on a nominal | **Yes — not yet consumed.** MIR consumes the `Static` selection here; HIR does not. This is a live engine asymmetry |
| 2 | qualified user-trait call (`<T as Tr>::m()`) | trait member | `Qualified { trait_item }` provenance is published |
| 3 | qualified Core-trait call | core trait member | as above |
| 4 | nested/container `Eq` helper | `eq` for `contains`/`position` | **No.** Needs the originating expression threaded down so the checker can publish against it |
| 5 | `Iterator` fallback | `next` | partly — `resolve_user_iterator` exists |
| 6 | top-level Display (`display_text`) | `fmt` | **Yes** — `display_uses` as of `6919c84` |
| 7 | recursive Display (`display_deep`) | `fmt` at depth | **Yes** — keyed by `DisplayPath` |

### The asymmetry at site 1 is the notable finding

`static_selected_key` was added to MIR and has no HIR counterpart, so the two engines currently
answer an ordinary method call by different means: MIR reads the published body, HIR scans by name.
They agree today — the differential suites say so — but "two algorithms that agree on the inputs we
have" is precisely the state DEV-192 was hiding in.

---

## 3. MIR lowering — 5 live callers

| # | Site | What it selects | Published metadata available? |
| ---: | --- | --- | --- |
| 1 | associated-function lookup | `Type::assoc()` | provenance `Qualified`/`Direct` is published |
| 2 | `Iterator::next` | `next` | partly |
| 3 | nested/composite Display | `fmt` at depth | **Yes** — `display_uses` |
| 4 | interpolation Display | `fmt` for `f"{x}"` | **Yes** — interpolation now publishes through the same walk |
| 5 | top-level print Display | `fmt` | **Yes** |

---

## 4. The order this implies

```text
HIR Display consumption          sites 6, 7
MIR Display consumption          sites 3, 4, 5
census + mutation gate           re-run §1's greps; every Display site must be gone
HIR Static consumption           site 1 — closes the engine asymmetry in §2
qualified paths                  sites 2, 3 (HIR) and 1 (MIR)
Iterator                         site 5 (HIR), 2 (MIR)
nested Eq                        site 4 (HIR) — needs a publication first
delete find_method / find_impl_fn
```

Display is first because its metadata already exists. The residue is not Display's problem, and
finishing Display does **not** finish Boundary 4.

---

## 5. One rule for every consumption step

When publication says a body exists at a position, **absence at consumption is an internal
invariant violation, not a licence to improvise**.

Concretely, `display_deep` currently ends a failed lookup with:

```rust
find_method(...) else { return Ok(value.to_string()); }
```

That structural stringification must not survive for a checker-approved user nominal. A consumer
that quietly substitutes its own answer when the published one is missing is a fallback wearing a
different name — and DEV-192 is exactly what that costs: `==` through an `Eq` bound silently used
structural comparison and printed a wrong answer, undetected, because every fixture's `eq` happened
to agree with it.

---

## 6. Progress — HIR Display consumption (2026-08-07)

HIR sites 6 and 7 are closed. `display_text` and `display_deep` consume
`display_uses[(root, path)]`; neither calls `find_method`. **HIR: 7 → 5 live callers.**

The renderer's contract is now:

```rust
display_deep(root: ExprId, value, ty: Option<&Ty>, path: DisplayPath, span)
```

The static type is threaded because the value alone cannot decide the step: `Value::Array` and
`Value::Vec` are **one runtime shape and three static ones** (`ArrayElement`, `SliceElement`,
`VecElement`), and the checker keyed them apart. Guessing, or trying each key until one hits, would
let a genuine mismatch read as a hit.

`DisplayPath::child` is now the single constructor both the checker's walk and both engines' walks
use, so the two cannot drift.

### §5's rule, applied and then tested

Both defensive fall-throughs are gone:

| Site | Was | Now |
| --- | --- | --- |
| `display_deep` nominal arm | `return Ok(value.to_string())` | internal-invariant error |
| `display_text` nominal arm | fell through to `format_runtime_value` | internal-invariant error |

**The second one was only found by mutation.** Forcing the lookup to always miss, the first pass
showed `as3_display_plan` failing 14 of 17 — and `dev_display_dispatch` passing 21 of 21. The
survivor was not noise: `display_text`'s top-level nominal branch was still quietly answering for
itself, one level above the arm I had just fixed. Removing a fallback in the obvious place and
leaving its twin one frame up is exactly the shape of DEV-192.

With both removed, the mutation fails `as3_display_plan` (14/17), `wp_fmt_001_interpolation` (5/40)
and `c63e_formatting` (10/51).

`dev_display_dispatch` still survives it, and that is **correct**: its fixtures call `x.fmt()`
explicitly and print a `String`, so they exercise bound method dispatch and never reach the
renderer. Verified by reading the fixtures rather than by assuming either way — a surviving mutation
is a question, not a verdict.

---

## 7. Progress — MIR Display, HIR `Static`, and the qualified paths (2026-08-07)

| Engine | Callers | Closed in this pass |
| --- | ---: | --- |
| HIR interpreter | 7 → **2** | ordinary `call_method`, both qualified paths, both Display sites |
| MIR lowering | 5 → **2** | all three Display sites |

**MIR Display.** `emit_display_value` now carries `(root, path)` alongside its `MirTy`, and the
recursion builds the path mechanically — `TupleField(i)`, `ArrayElement`, `VecElement`,
`OptionSome`, `ResultOk`, `ResultErr`. `display_fn_key` resolves `Static` directly and `Bound`
through the shared specialiser with the concrete `Self` MIR already holds (DEV-189's rule). The
repeated-container semantics stay clean: one published position, executed once per runtime element,
no record per element.

**The engine asymmetry is closed.** `static_selected_callable` is the interpreter's counterpart to
MIR's `static_selected_key`, so both engines now answer an ordinary method call from the same
published body.

Before deleting the `call_method` fallback it was instrumented and run across nine suites — both
differentials, iterators, bound identity, adversarial trait impls, cross-package generics,
associated types and Display dispatch. It fired **zero** times. Deleted rather than annotated:
"unreached in the suites you ran" is not "unreachable", and an annotation saying exactly that is
what let DEV-191 hide behind a comment claiming it had been verified.

### Remaining — 4 callers

| Engine | Site | Blocker |
| --- | --- | --- |
| HIR | nested/container `Eq` helper | needs a publication: the originating `contains`/`position` expression must be threaded down so the checker can publish against it |
| HIR | `Iterator` fallback | — |
| MIR | associated-function lookup | — |
| MIR | `Iterator::next` | — |
