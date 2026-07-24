# WP-C6.1g-b — Return-Source Lifetime Precision

**Track:** A (Claude)
**Status:** OPEN (owner disposition CD-097 item 2, 2026-07-24)
**Base:** `main` @ CD-096
**Blocks:** **Gate C6 native-conformance closure.** Does *not* block WP-C6.1f closure.

---

## 1. Why

C6.1f's reference-return support encodes OWN-RETURN-001's shortest-input rule with a **single shared
`'a`** across every reference parameter and the return. That is sound but **conservative**: it ties a
returned value to reference parameters it cannot possibly derive from, which **rejects valid Core
programs**.

Concretely, for `fn pick(a: &P, b: &P) -> &P { a }` the result provably derives from `a` alone, yet
the shared `'a` also ties it to `b` — so a caller passing a short-lived `b` and a long-lived `a`
cannot use the result past `b`'s region, although Core v1 permits it.

The present behaviour is **sound and may remain until this package lands**.

## 2. Requirement — as directed

**Return provenance must determine the relevant input lifetime set:**

- a result derived **only** from `a` MUST NOT be tied to an unrelated `b`;
- a result that **may** derive from either remains tied to **both** (OWN-RETURN-001 rule 3's
  shortest-of-all over the parameters it could derive from, and OWN-CARRY-001's "a merge of
  control-flow paths carries the union of possible source referents ... its region cannot exceed the
  intersection of their valid regions").

## 3. Notes carried forward from C6.1f

- The front end already computes derivation for the escape check (`borrowck::borrowed_local`
  recurses through `Block`, `If`, `Match`, calls and borrow-carrying aggregates). The per-path
  **source set** this package needs is a generalisation of that walk — from "is it a local?" to
  "which parameters can it derive from?" — so it should extend that logic rather than start fresh.
- The result must reach the backend, which today has no provenance input and therefore cannot do
  better than the shared `'a`. Threading it is part of this package.
- Elision still covers the zero- and one-reference-parameter cases; precision only matters at two or
  more.

## 4. Exit criteria

- A returned reference is tied to exactly the parameters it may derive from, on every path.
- Programs valid under OWN-RETURN-001 but rejected by the shared-`'a` encoding now compile, with
  native tests naming the previously-rejected shapes.
- Escape rejections (E0103) and the C6.1f negative corpus are unchanged — precision must not
  weaken provenance.
- Full workspace suite, `fmt --check`, strict `clippy` clean.
