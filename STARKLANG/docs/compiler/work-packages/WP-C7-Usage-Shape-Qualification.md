# WP-C7 — Usage-Shape Qualification

**Status:** `PROPOSED` — opened by CD-183 as a C6 follow-on. Not a C6 blocker.
**Owner:** compiler track.

## Why

WP-C6.6's normative surface audit establishes that each admitted method has **at least one valid
invocation** that lowers and verifies. It does not establish that every valid *usage* of that method
works. DEV-119 is the proof that the gap is real rather than theoretical: `HashMap::keys`,
`HashSet::iter` and `Vec::iter` all passed the invocation audit while an ordinary post-loop mutation
failed native compilation, because the iterator's cursor held its borrow past the loop.

That is the same failure shape as DEV-115, one level down:

| | hidden by | exposed by |
| --- | --- | --- |
| DEV-115 | per-TYPE coverage (`str` covered, `str::bytes` broken) | reviewing an unrelated diff |
| DEV-116 | per-TYPE coverage (`HashSet<T>` one row, 8 methods unlowerable) | a coverage row with no case |
| DEV-119 | per-METHOD invocation (3 methods "executable", one usage broken) | verifying an unrelated fix |

Each was found by accident. This work package is the deliberate version.

## What this is NOT

**Not an exhaustive matrix.** Method × usage-shape explodes combinatorially — ownership and moves,
shared and exclusive borrows, stored references, exhaustion, `break`, `continue`, nesting, early
return, traps, Drop-bearing values, generics, package boundaries, and mutation before/during/after
use. Enumerating that product is neither affordable nor useful.

**Risk-based instead.** The shared semantic MECHANISMS are tested once across representative types,
not once per method. The output is a compact family matrix.

## Scope, in priority order

APIs that return or retain references and resources carry the risk; value-returning methods largely
do not.

### 1. Borrowed iterators
`Vec::iter`, `HashMap::keys`, `HashSet::iter`, and `HashMap::values` / `HashMap::iter` when
`WP-C7-HashMap-Completion` lands.

### 2. Direct reference-returning methods
`Vec::get`, `Vec::get_mut`, `HashMap::get`, `HashMap::get_mut` when it lands, and indexing /
mutable indexing.

### 3. Reborrow and storage shapes
A returned reference: used immediately; stored in a local; held across a mutation; reassigned;
passed through a function; returned from a function where the reference rules permit it.

### 4. Control-flow lifetime boundaries
Block exit, loop exhaustion, `break`, `continue`, early return, nested loops, and match arms.

## The invariant every case pins

DEV-119's fix could have been made by relaxing the borrow rules instead of shortening the cursor's
lifetime, which would have been worse than the defect. Every case therefore states which side of
this line it is on:

```text
the borrow is over        ->  the mutation MUST succeed
the borrow is still live  ->  the mutation MUST remain rejected, with the borrow diagnostic
```

A negative case must assert the SPECIFIC diagnostic (`E0101` for an active-borrow conflict) and that
the program otherwise parses and resolves — an "it was rejected somehow" assertion can pass for the
wrong reason, which is a mistake this track has already made once and corrected.
