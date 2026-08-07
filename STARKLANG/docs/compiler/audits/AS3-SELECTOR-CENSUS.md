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
