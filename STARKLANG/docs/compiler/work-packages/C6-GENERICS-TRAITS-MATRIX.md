# C6-GENERICS-TRAITS-MATRIX — Track B / WP-C6.2

**Status:** C6.2a CLOSED (CD-086) — canonical callable identity corrected; the dispatch shapes it
unblocked are classified below. C6.2b…e remain open.
**Base:** `main`, post-WP-C6.1 closure (CD-085)
**Authorship:** the file is **Track B**-owned (`C6-FILE-OWNERSHIP.md §1`). The C6.2a section was
written by Track A under the owner's C6.2a ruling, because the defect was in *lowering identity*,
not method selection. Track B owns this file from C6.2b onward.
**Method:** probe-grounded — every row driven through the real pipeline
(`lower → verify → emit → run`), not assumed.

---

## 1. The C6.2a headline: one canonical-identity defect blocked all native dispatch

`Instance` identity is `(item, type_args, symbol)`, and the frozen contract
(`C6-SHARED-CONTRACTS.md §3`) requires a reference to carry the same triple as the body it names.

- **Bodies** derived `item` correctly from the `FnKey`: function item (`Top`), `impl_item`
  (`ImplFn`), `trait_item` (`TraitDefault`).
- **Call sites** passed the **receiver nominal** instead.

Result: one canonical symbol with two item identities, so the C5.4a linkage preflight (correctly)
refused **every** method, trait, operator and associated-function call before rustc. The full suite
stayed green only because no native test exercised an ordinary method call — destructors resolve
through `TypeContext::drop_impls`, a different path.

**Fix (owner ruling — a conformance correction, NOT a CE3):** one lowering-internal constructor,
`FnLowerer::instance_from_key(&FnKey) -> Instance`, is now the sole producer of every `Instance` —
the body's and all six `Callee::Instance` references. This removes the defect *class* rather than
its manifestations. No MIR shape, verifier rule, version, symbol scheme, or accepted-language
semantics changed; the linkage consistency check was **not** weakened.

---

## 2. Shape matrix (C6.2a)

`native` = builds and runs to exit 0 with three-engine agreement. Rows 6 and 11 were **added as
regressions** by the ruling's coverage list and were not in the original twelve-shape probe; of
those twelve, **nine** were refused by the identity defect, two already worked (12, 13), and one is
the separate lowering gap (14).

| # | Shape | Before | After | Evidence |
|---|---|---|---|---|
| 1 | inherent method | refused (identity) | ✅ native | `c62a_inherent_method` |
| 2 | associated function (`P::new`) | refused (identity) | ✅ native | `c62a_associated_function` |
| 3 | user-trait implementation method | refused (identity) | ✅ native | `c62a_user_trait_implementation_method` |
| 4 | default trait method (un-overridden) | refused (identity) | ✅ native | `c62a_default_trait_method` |
| 5 | bounded generic calling the bound method | refused (identity) | ✅ native | `c62a_bounded_generic_calls_the_bound_method` |
| 6 | method on a generic nominal | refused (identity) | ✅ native | `c62a_generic_nominal_method` |
| 7 | method-level generic (method's own generics) | refused (identity) | ✅ native | `c62a_method_level_generic_method` |
| 8 | associated type (`Self::Item`) | refused (identity) | ✅ native | `c62a_associated_type` |
| 9 | **`Eq` operator dispatch** — adversarial always-true impl | refused (identity) | ✅ native | `c62a_eq_operator_dispatch_uses_the_user_impl` |
| 10 | **`Ord` operator dispatch** — adversarial reversed impl | refused (identity) | ✅ native | `c62a_ord_operator_dispatch_uses_the_user_impl` |
| 11 | cross-package trait method call | refused (identity) | ✅ native | `c62a_cross_package_trait_method_call` |
| 12 | nested generic nominal | ✅ (already) | ✅ native | probe |
| 13 | generic nominal with `Drop` | ✅ (already) | ✅ native | probe |
| 14 | **fully-qualified call** `Shape::area(&q)` | `LOWER: callee form (C4.5)` | ❌ **still open** | **DEV-102** → C6.2b |

Rows 9 and 10 matter beyond dispatch: they early-discharge part of WP-C6-ENTRY **§20 (C6.2d)** —
**STARK's own impls must run, not Rust equivalents**. They do not close §20, which also covers
arithmetic/comparison desugaring, `Ordering` totality and operator trap behaviour. The evidence is
adversarial by construction: an always-true `Eq` and a reversed `Ord` both produce answers a Rust
`derive` or built-in comparison would contradict, so passing proves the user impl is the one invoked.

Earlier packages already established, and these keep green: concrete generic instances emit
exactly once with no Rust generic parameter list (C5.4b), generic function values (C5.4c), and
cross-package/dependency-to-dependency generic instantiation (DEV-101).

---

## 3. Identity invariant, asserted directly

Beyond relying on the linkage preflight, every C6.2a case asserts the invariant explicitly: for each
`Callee::Instance` reference, the body named by its canonical symbol carries the **same `symbol`,
`item` and `type_args`** (`assert_reference_identity_matches_bodies`). A deliberately mismatched
item is still refused (`a_mismatched_item_is_still_rejected`) — the check is intact.

---

## 4. Open for C6.2b and later

| Item | Owner |
|---|---|
| **DEV-102** — fully-qualified call form `Trait::method(&recv)`; a missing callee-lowering form, unrelated to the identity defect | C6.2b (method-resolution completion) |
| **DEV-083** — candidate-local inference snapshots / declaration-order-independent candidate evaluation (deferred by CD-040(b)) | C6.2b |
| the rest of §18's resolution matrix (`Self` in default methods, nested-reference receivers, ambiguity, privacy, inherent-vs-trait preference) | C6.2b |
| §19 associated types beyond the single-binding case (multiple bindings, bounds on associated types, projection in signatures) | C6.2c |
| §20 the remainder of operator/`CoreTrait` semantics (arithmetic and comparison desugaring, `Ordering` totality, operator trap behaviour) | C6.2d |
| generic-impl-head receiver inference recheck (WP-C6-ENTRY §2 carry-forward) | C6.2b |
| §21 deterministic instance identity under rebuild/relocation/dependency reorder for methods and trait instances | C6.2e |
| generic collections and iterators | C6.3 (Track C) interface |
