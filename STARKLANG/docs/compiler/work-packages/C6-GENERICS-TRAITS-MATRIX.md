# C6-GENERICS-TRAITS-MATRIX — Track B / WP-C6.2

**Status:** C6.2a CLOSED (CD-086). C6.2b IN PROGRESS — §18 matrix probed end-to-end (§5), DEV-102
closed (§6); F1–F6 dispositioned by owner ruling (§7). **F1 is a C6.2b blocker assigned to Track B**
and must be fixed before F2, F5, DEV-083 or C6.2c. C6.2c…e remain open.
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
| 14 | **fully-qualified call** `Shape::area(&q)` | `LOWER: callee form (C4.5)` | ✅ native (C6.2b) | `c62b_fully_qualified_*` |

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

## 4. Remaining C6.2 sub-package scope

The §18 matrix itself is now probed — see §5. What stays open beyond it:

| Item | Owner |
|---|---|
| **DEV-083** — candidate-local inference snapshots / declaration-order-independent candidate evaluation (deferred by CD-040(b)) | C6.2b |
| §19 associated types beyond the single-binding case (multiple bindings, bounds on associated types, projection in signatures) | C6.2c |
| §20 the remainder of operator/`CoreTrait` semantics (arithmetic and comparison desugaring, `Ordering` totality, operator trap behaviour) | C6.2d |
| §21 deterministic instance identity under rebuild/relocation/dependency reorder for methods and trait instances | C6.2e |
| generic collections and iterators | C6.3 (Track C) interface |
| the six C6.2b findings in §7 | **awaiting disposition** |


---

## 5. C6.2b — the §18 method-resolution matrix, probed

Every row driven `parse → resolve → typecheck → HIR-run → lower → verify → emit → native-run`.

| §18 row | Status | Note |
|---|---|---|
| inherent | ✅ native | C6.2a |
| user trait | ✅ native | C6.2a |
| CoreTrait | ✅ native | `Clone`, `Default`, `Eq`, `Ord` all dispatch to the user impl |
| default trait method | ✅ native | C6.2a; also reachable fully qualified |
| **fully qualified call** | ✅ native | **DEV-102 closed by C6.2b** — see §6 |
| generic-parameter method | ✅ native | `fn f<T: Sh>(t: T) { t.a() }` |
| **`Self` in default method** | ✅ native | `self`-typed locals and `Self::assoc()` in default bodies both work |
| shared-reference receiver | ⚠️ **blocked** | `let r = &p; r.get()` — **F3**, reference-lane |
| mutable-reference receiver | ✅ native | `p.bump()`; `let r = &mut p; r.bump(); p.get()` correctly rejected (E0101 — Core v1 borrows are lexically scoped, no NLL) |
| nested-reference receiver | ⚠️ **blocked** | **F4** — `&&T` unspellable; inferred `&&T` fails MIR verify |
| associated function | ✅ native | C6.2a |
| associated type | ✅ native | C6.2a |
| **ambiguity** | ✅ correct | two traits supplying `go` → **E0203**; inherent shadows trait; qualified form disambiguates |
| **privacy** | ❌ **under-rejects** | **F1** — private impl members and private fields are reachable cross-module |
| cross-package impl | ✅ native | C6.2a |

Recheck of the WP-C6-ENTRY §2 carry-forward: the **generic impl-head receiver-inference limitation
is still open** — **F5**.

---

## 6. DEV-102 CLOSED — fully qualified trait calls

TYPE-METHOD-001 requires the form ("Trait methods can always be called in fully-qualified function
form"; it "bypasses trait-name lookup but still requires a unique coherent impl"). The front end and
the HIR oracle accepted it; only MIR lowering had no `Res::TraitMember` callee arm.

Lowering now selects through `find_trait_impl_fn` — a **trait-filtered** lookup, deliberately
separate from `find_impl_fn`. `find_impl_fn` answers "what does `recv.m()` mean", so it prefers
inherent methods and accepts any in-scope trait; the qualified form must do neither. That separation
is what makes the form usable as the spec's own remedy for the E0203 ambiguity error.

Covered: plain call; **`A::go(&s)` vs `B::go(&s)` selecting different impls**; ignoring an inherent
method of the same name (while `s.go()` still prefers it); reaching a trait default body; extra
arguments; `&mut` receivers; a `Drop`-bearing receiver. E0203 and E0005 rejections are asserted to
persist. Because the receiver is written explicitly, no auto-borrow/auto-deref applies
(TYPE-METHOD-002 governs `recv.m()` only) — every argument lowers as an ordinary operand in source
order, which is why the arm is small.

Not covered, and **deliberately not broadened**: `Trait::assoc()` with no receiver (checker rejects
with E0005 — the implementing type is unrecoverable), and a trait implemented for a *specific*
instantiation of a generic nominal (`impl Get for W<Int32>` → E0500, a front-end limitation ahead of
lowering) — **F2**.

---

## 7. C6.2b findings — owner dispositions (2026-07-23)

Ordered by severity. F1 is the only one that **accepts invalid programs**; the rest are
over-rejections or unassigned scope.

**Dependency order (ruled): F1 → C6.1f/F3 → F4 → remaining F2/F5/F6 → C6.3b.**

| Finding | Disposition |
|---|---|
| **F1** | **CLOSED (CD-102)** — front-end privacy enforcement; `hir.item_modules` + `typecheck::check_member_visible` (E0207) at method/assoc-fn/field-read/field-construction. Originally **Track B, C6.2b BLOCKER** — fix before F2, F5, DEV-083 or C6.2c. No lease needed (`resolve.rs`, `typecheck.rs`, the C6.2 tests and this matrix are Track B-owned); request a narrow lease only if shared authority-bearing files become necessary. Enforce at the **semantic access point**, not the three discovered examples: field projection, method-call selection, associated-function selection, fully qualified calls to private impl members, generic and cross-package versions, defining-module access still accepted, public members of a private type not making it externally nameable, and inherent-member privacy kept distinct from trait-member accessibility. |
| **F2** | **CLOSED (CD-104)** — `default_int_literals_deep` defaults int literals inside the receiver type so a concrete-instance impl matches. Was: | After C6.1f. Track B. |
| **F3** | **New package `WP-C6.1f` — General Reference Storage, Reborrowing, and Provenance, Track A.** Not absorbed into C6.2b. See `WP-C6.1f.md`. |
| **F4** | **Split.** Nested-reference type parsing + MIR/reference representation → C6.1f (Track A); repeated auto-deref *selection* → Track B, after C6.1f. |
| **F5** | **CLOSED (CD-103)** — `current_impl_generics` consulted in bounded-`Ty::Param` method resolution. Was: | After C6.1f. Track B. |
| **F6** | After C6.1f. Track B. |

| # | Finding | Evidence | Normative basis |
|---|---|---|---|
| **F1** | **Privacy under-rejection.** A private inherent method, a private associated function, and a **private struct field** are all reachable from outside the defining module. Module-level items *are* enforced (private `fn`/`struct` correctly rejected in `resolve`), so the hole is specifically **impl members and fields**. | `m::S::secret()`, `s.hidden()`, `s.v` all run to exit 0 cross-module | MOD-VIS-001 ("Fields and enum variants follow their declarations' explicit visibility rules"); 07 §"Accessing a private item outside its module is a compile-time error"; TYPE-METHOD-001 step 5 |
| **F2** | **CLOSED (CD-104)** — `default_int_literals_deep` defaults int literals inside the receiver type so a concrete-instance impl matches. Was: | Trait impl on a *specific* instantiation of a generic nominal (`impl Get for W<Int32>`) is not seen: E0500 "trait is not implemented for receiver type". | `fq_generic_nominal` probe | 03 coherence/impl matching |
| **F3** | **A reference stored in a user local is refused by the backend** — "outside the C5 ephemeral reference lane". `let r = &p; r.get()` cannot be built, though the front end and HIR oracle accept it. | `recv_shared`, `generic_impl_head_inference` probes | C5 exit report §205 defers "general references" to "C6" **without a sub-package**; §18 nevertheless lists shared/nested-reference receivers as C6.2b rows |
| **F4** | Nested-reference receivers: `&&T` is **unspellable as a type** (parser: "expected a type, found `&&`"), and an inferred `&&T` receiver passes typecheck but fails MIR verify (MIR-0005). | `recv_nested_ref`, `F6_two_ref_via_arg` probes | TYPE-METHOD-002: auto-deref "repeatedly removes one leading `&`/`&mut`" — nested receivers are normative. Mostly downstream of F3 |
| **F5** | **CLOSED (CD-103)** — `current_impl_generics` consulted in bounded-`Ty::Param` method resolution. Was: | Impl-head bounds are invisible in method bodies: `impl<T: Sh> W<T> { fn go(&self) { self.v.a() } }` → E0302 "method 'a' not found for type 'T'". | `generic_impl_head_bounded` probe | The WP-C6-ENTRY §2 carry-forward, confirmed still open |
| **F6** | Impl signatures do not normalise `Self`: writing the concrete type where the trait declares `Self` (`fn make() -> G` for `fn make() -> Self`, or `o: &G` for `o: &Self`) is rejected E0500. Writing `Self` works. | `F4_impl_writes_concrete`, `F4_param_Self` probes | 03 impl/trait signature matching — spec does not obviously require the concrete spelling to be accepted |

**Doc defect found while grounding F4:** the repo `CLAUDE.md` summary said method calls "auto-deref
**one** reference level", but normative TYPE-METHOD-002 says auto-dereference "repeatedly removes one
leading `&`/`&mut`". **Corrected in `0873308`** — it stated a limitation the language does not have
and would have led a future agent to implement single-level auto-deref and treat nested-reference
receivers as out of scope.
