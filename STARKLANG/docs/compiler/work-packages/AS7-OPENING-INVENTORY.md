# AS7 opening inventory — the ambient state

**Read-only. No implementation.** AS7's checkpoint evidence requires the ambient-state conversions
*first*, as separate commits, before the file splitting. This is what they are converting.

**Head:** `81b1401`, immediately after AS6 CLOSED. **Date:** 2026-08-08.

## The packet's stated risk

> Replacing current-file, current-module, current-impl and generic-environment state with scoped
> context objects **can restore the wrong context with every field present and every signature
> type-checking**. This repo has already paid for that class once, in file-provenance drift between
> `self.text` and item-level file metadata.

So the inventory has to answer two different questions, and they are not the same question:

```text
1  does control ever leave between a save and its restore?   -> a LEAK
2  is a field ever written on a path that never saved it?    -> the WRONG CONTEXT
```

## The eight ambient fields

```text
field                    writes  saved  restored   position
current_self_ty              18      8         8   saved/restored
current_assoc_types           2      2         2   saved/restored
current_fn_generics           9      3         4   saved/restored
current_impl_generics         3      1         2   saved/restored
current_trait_id              2      1         1   saved/restored
allow_half_type               4      2         2   saved/restored
current_fn_ret                2      0         0   set-then-CLEARED, never restored
current_module                1      0         0   assigned per item, never restored
```

## Question 1 — leaks: none

Fourteen true save/restore pairs (`.replace()`, `.take()`, or a paired assignment with a matching
restore). For every one of them, **no `return` and no `?` occurs between the save and the restore**:

```text
3136 -> 3139  allow_half_type          6259 -> 6262  allow_half_type
3646 -> 3659  current_fn_generics      3878 -> 3893  current_fn_generics
3674 -> 3698  current_self_ty          4546 -> 4846  current_self_ty
3743 -> 3760  current_self_ty          8353 -> 8387  current_self_ty
3765 -> 3781  current_self_ty          8528 -> 8539  current_self_ty
3746 -> 3762  current_impl_generics    9469 -> 9486  current_self_ty
3766 -> 3782  current_trait_id         9610 -> 9628  current_self_ty
```

**A first pass reported two "NO RESTORE" sites, at 5290 and 5330. Both were false positives** — my
detector matched `let x = self.current_impl_generics.clone()` and
`let x = self.current_self_ty.clone()`, which are *reads*, not saves. Recorded because it is the
same failure shape AS6 hit four times: a pattern that resembles the question is not the question.
The corrected detector requires `.replace(`/`.take(`, or a plain assignment with a matching restore.

## Question 2 — the wrong context: two latent sites, no live defect

### `current_fn_ret` — cleared, not restored

Set at `typecheck.rs:5322`, set to `None` at `:5426`. `check_fn_def` has three call sites — free
functions (`:3732`), impl methods (`:3757`), trait default methods (`:3778`) — **all inside the
single Pass-2 item loop, none nested**. Clear-to-`None` is therefore correct *today*, but it is
correct **by the non-nesting invariant, not by construction**. The same applies to
`current_fn_generics` at `:5427`.

### `current_module` — assigned per item, never saved

Written once, at `typecheck.rs:3728`, at the top of the Pass-2 loop **before the item `match`**, so
it dominates every branch and no item can inherit a stale module. Also correct today, and also
correct only because that single assignment dominates every path that checks an item.

## The finding

**The ambient state has no defect today. Two of the eight fields are correct only because item
checking never nests — and AS7's splitting is the change most likely to break exactly that.**

A scoped context object that replaces "clear to `None`" with "restore the previous value" is
strictly safer and behaves identically while the invariant holds. That is the conversion to make,
and it is cheap. The risk is not in the conversion; it is in converting the six already-restored
fields and the two invariant-dependent ones *without noticing they are different cases*.

## What AS7 should do before it moves any code

Carried from AS6's exit qualification, whose finding was that **criterion 2 was the only criterion
with no behavioural signature, and it therefore failed twice after its surfaces were declared
clean**:

> AS7's exit criterion 1 — "no semantic behaviour or diagnostic structure changes" — has a
> behavioural signature and the existing suites pin it.
> AS7's exit criterion 2 — "dependency direction between submodules is documented and cycle-free" —
> **has none**, and nothing in the repo would fail if it regressed.

Two executable checks should exist **before** the split, not after:

1. **An ambient-state restoration check.** After checking a whole program, and after each item,
   every ambient field returns to its entry value. This has no behavioural signature today either —
   a wrongly restored context produces *plausible* wrong answers, not crashes.
2. **A dependency-direction check** over the intended module decomposition, so criterion 2 is
   pinned by something that runs rather than by a diagram.

Neither can be written until the target decomposition is named, which is the next AS7 step.

## Precondition, unresolved

AS7 requires **exclusive tree ownership** — the packet says so directly: *"Splitting a 14,000-line
pass cannot survive a parallel session editing the same file — take a worktree, or hold an explicit
agreement that no other session touches the declared ownership set for the duration."*

`typecheck.rs` is 14,432 lines. This checkout is shared. **This is an owner decision, not a
technical one, and it is not resolved.**
