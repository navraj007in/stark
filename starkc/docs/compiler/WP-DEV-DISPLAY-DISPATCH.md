# WP-DEV-DISPLAY-DISPATCH — a compiler-known trait bound is an ordinary trait bound

**Closed 2026-08-04. Ledger entry: CD-378. Deviation: DEV-166 (resolved); DEV-167, DEV-168 and
DEV-169 opened.**

---

## 1. Root cause

The failing program:

```stark
fn show<T: Display>(x: T) -> String {
    x.fmt()
}
```

```text
[E0302] method 'fmt' not found for type 'T'
```

`T: Display` was recognised where a bound is *checked* and invisible where a method is *resolved*.
The two sites read the bound differently:

| site | how it reads a bound | compiler-known trait |
| --- | --- | --- |
| `satisfies_bound` / `ty_satisfies_operator_bound` | by trait NAME, against impls and the Core table | accepted |
| `resolve_method`, bounded-generic branch | by searching `hir::ItemKind::Trait` for that name | **found nothing** |

A compiler-known trait has no `hir::ItemKind::Trait` declaration item — `resolve.rs` turns the name
into `Res::CoreTrait(CoreTrait::Display)` and there the trait ends. So the bounded-generic branch's
`bound_trait_id` was `None`, the loop fell through, and the impl scan below could not match a
`Ty::Param` receiver either. The E0302 that followed was the truthful report of a candidate set
that had been built without ever consulting half of the bounds.

**This is a trait-model defect, not a formatting one.** The same hole covered `Ord::cmp`,
`Clone::clone`, `Hash::hash`, `Iterator::next` and `Into::into` on a bounded parameter. `Display`
is simply where it was noticed, because a `Display` bound has no purpose other than calling `fmt`.

Two further defects lived in the same branch and could not be left in place, because the shape the
work package requires does not work without them:

* **Ownership.** `borrowck.rs::method_receiver` had no `Ty::Param` arm. It returned `None`, and the
  `Call` handler's `None` arm CONSUMES the receiver. Every `&self` method reached through any bound
  therefore moved its receiver — for user traits too. `fn f<T: Named>(x: T) { x.name(); x.name(); }`
  failed E0100 "use of moved value" before this work package, and passes after it.
* **Ambiguity.** The branch `return`ed on the first bound that supplied the name, so
  `T: A + B` with both declaring `m` resolved by written order rather than being reported ambiguous.

---

## 2. Architecture implemented

The ruling was that `Display` must be an ordinary trait first and a compiler-known trait second,
with **one** candidate-resolution path. That is what landed.

```text
receiver type, after auto-dereference
        │
        ├── Ty::Param(T) ──► collect bounds
        │                      │
        │                      ├── BoundTrait::User(ItemId)   ──► find_trait_method_sig
        │                      └── BoundTrait::Core(CoreTrait) ──► core_trait_contract
        │                                     │
        │                                     ▼
        │                          ONE candidate list, de-duplicated by trait identity
        │                                     │
        │                          0 ──► missing-bound diagnostic (names the trait)
        │                          1 ──► check_bound_method_call
        │                         >1 ──► E0203, naming both traits
        │
        └── concrete type ──► the existing impl scan, unchanged
```

`BoundTrait` is the repair in one type: both kinds of trait are *an identity a bound resolves to*.
Selection, ambiguity, argument checking, `Self` substitution, associated-type normalisation and the
diagnostics are shared from that point on. `resolve_bound_trait` resolves a user trait of the same
spelling FIRST, so a program declaring its own `trait Display` gets its own — the same precedence
`resolve_path` already applies.

**No second signature registry was introduced.** `core_trait_contract` already existed: it is the
table `impl Display for Point` is checked against, and it already carried
`fmt / Some(Ref) / [] / String`. A bound now reads the same entry. What a bound makes callable is,
by construction, what an implementation must provide — one table, two readers. The filter on which
Core methods a bound exposes is `receiver.is_some()`, a property of the contract, not a list of
names: `Default::default` and `From::from` have no receiver and therefore no `x.default()` spelling
to resolve.

**No method-name branch exists anywhere in the change.** `grep -n '"fmt"' src/typecheck.rs` returns
the contract table entry and the pre-existing concrete-receiver entry, and nothing in the new code.

Enumeration over `CoreTrait` is done by `next_core_trait`, a match that is total over the enum, so
adding a variant is a compile error rather than a silent omission from a hand-maintained list. The
same no-wildcard discipline is applied to `FmtReceiver::of` over `MirTy`, to `resolve_bound_trait`
over `Res`, and to `trait_ref_type_args` over `GenericArg`. The `RuntimeFn` matches in
`emit_runtime.rs`, `mir/verify.rs` and `mir/interp.rs` were already exhaustive, and adding the seven
`Fmt*` variants forced each of them to be updated.

### The concrete tail

Monomorphisation grinds `T` down before MIR sees it, so `x.fmt()` inside `show<T: Display>` arrives
at lowering as `fmt()` on a concrete type. For a user nominal the ordinary impl path already
resolved it. For a PRIMITIVE there is no impl item to find — 06-Standard-Library declares
`impl Display for Int32` "and similar for other types" and no source file writes those blocks — so
`lower_method_call` refused the call with "method call on non-nominal receiver". Seven `RuntimeFn`
variants (`FmtInt64`, `FmtUInt64`, `FmtBool`, `FmtFloat64`, `FmtFloat32`, `FmtChar`, `FmtUnit`) are
the lowering of exactly those declarations. `String` and `str` reuse `StringAsStr`/`StrToString`.

---

## 3. Changed files

| file | change |
| --- | --- |
| `starkc/src/typecheck.rs` | `BoundTrait`/`BoundMethod`; `resolve_bound_trait`, `bound_method_candidates`, `check_bound_method_call`, `contract_ty_to_ty`, `check_call_arguments`, `traits_declaring_method`, `bound_trait_list`; `core_trait_bound_method`, `core_trait_method_receiver`, `trait_ref_type_args`, `next_core_trait`, `all_core_traits`; the rewritten bounded-generic branch and the missing-bound diagnostic |
| `starkc/src/borrowck.rs` | `current_generics`/`enclosing_generics`; `bound_method_receiver`; the `Ty::Param` arm in `method_receiver` |
| `starkc/src/interp.rs` | `call_core_method_at` (the Core surface entered with the receiver place already resolved); `receiver_is_type_param`; the generic-receiver fallback in `call_method` |
| `starkc/src/resolve.rs` | `resolve_core_trait` is `pub(crate)` |
| `starkc/src/mir/mod.rs` | seven `RuntimeFn::Fmt*` variants |
| `starkc/src/mir/lower.rs` | `FmtReceiver`; `lower_display_fmt`; the `fmt` arm in `lower_method_call` |
| `starkc/src/mir/verify.rs` | signatures for the seven variants |
| `starkc/src/mir/interp.rs` | execution of the seven variants, through `stark_runtime::format` |
| `starkc/src/backend/generated_rust/emit_runtime.rs` | emission of the seven variants |
| `starkc/stark-runtime/src/format.rs` | `fmt_i64`, `fmt_u64`, `fmt_bool`, `fmt_f64`, `fmt_f32`, `fmt_char`, `fmt_unit` |
| `starkc/tests/dev_display_dispatch.rs` | new — 21 tests |
| `starkc/scripts/qualify-first-party-packages.py` | `stark-fmt` registered |
| `STARKLANG/docs/spec/03-Type-System.md` | **TYPE-METHOD-003** |
| `STARKLANG/docs/spec/06-Standard-Library.md` | **STD-TRAIT-002**; STD-FORMAT-001's receiver sentence |
| `STARKLANG/docs/spec/STARK-Core-v1.{md,html,pdf}` | regenerated |
| `starkc/docs/conformance/KNOWN-DEVIATIONS.md` | DEV-166 resolved; DEV-167/168/169 opened |
| `packages/stark-fmt/`, `packages/stark-fmt-consumer/` | new |

---

## 4. Test matrix

### Positive — all through the three-engine comparator, stdout pinned independently

| case | proves |
| --- | --- |
| `generic_display_bound_makes_fmt_callable` | the reported defect is gone |
| `fmt_borrows_and_does_not_consume_an_owned_parameter` | `x.fmt()` twice on an owned `T` |
| `fmt_borrows_and_does_not_consume_a_non_copy_value` | a `String`-bearing nominal survives two formats and a third use |
| `fmt_borrows_and_does_not_consume_an_affine_value` | a `Drop`-bearing (therefore non-`Copy`, use-once) value is formatted and then moved into a consumer |
| `generic_dispatch_covers_the_display_primitives` | `Int8/16/32/64`, `UInt8/16/32/64`, `Float32`, `Float64`, `Bool`, `Char`, `String`, `&str` through ONE generic function |
| `generic_fmt_agrees_with_println_rendering` | `x.fmt()` and `println(x)` render identically |
| `a_user_display_impl_dispatches_through_the_bound` | a user `impl Display` reached generically |
| `core_and_user_bounds_coexist` / `..._in_either_order` | `T: Display + Named` and `T: Named + Display` behave identically |
| `multiple_bounds_with_only_one_provider` | a non-contributing second bound does not disturb the first |
| `nested_generic_forwarding` | `outer<T: Display>` → `inner<U: Display>`, primitive and nominal |
| `an_impl_head_bound_reaches_a_core_trait` | `impl<T: Display> Wrap<T>` (WP-C6.2b-F5) |
| `qualified_calls_disambiguate_the_two_traits` | front end + oracle only — see DEV-168 |
| `release_and_debug_native_agree` | the fourth execution mode |

### Negative

| case | expected |
| --- | --- |
| `a_missing_bound_names_the_trait_to_add` | E0302, message names `T: Display` |
| `a_wrong_bound_still_names_display` | E0302 with `T: Named` present, still names `Display` |
| `an_unknown_method_on_a_parameter_is_still_not_found` | plain "not found", NOT a missing-bound message |
| `a_concrete_type_without_display_is_rejected` | E0302 — reachability from a bound did not make it reachable from nothing |
| `a_core_bound_method_checks_its_arity` | E0005 for `x.fmt(1)` |
| `same_name_from_two_bounds_is_ambiguous` | E0203 in BOTH bound orders |

Receiver mismatch is covered structurally rather than by a case: `Display::fmt`'s contract receiver
is `Ref`, `lower_display_fmt` reads the receiver place by copy or through a shared borrow, and
`mir/verify.rs` types the runtime call's operand as the scalar. A lowering that produced an
exclusive borrow or a move would fail verification.

---

## 5. Evidence

### Ownership

`fmt_borrows_and_does_not_consume_*` are the regression guards, and they fail against a consuming
lowering in three different ways: the move checker rejects the second `x.fmt()`; MIR verification
rejects the second read of a moved local (V-MOVE-1); the affine case's `Drop` line moves in the
output. The `released` line landing between the format and the `5` is what shows the value was
still alive at formatting time.

### Engine parity

Every positive case runs `agree_completing_with_stdout`, which compares the HIR oracle, the MIR
interpreter and the native binary field by field — stdout bytes, exit status, Drop log, returned
observation — and then pins stdout to a string stated in the test rather than taken from an engine.
Three engines agreeing on the wrong text fails.

### Native backend

`native_selects_stark_formatting_not_rusts` reads the generated crate's `src/main.rs` and requires
`stark_runtime::format::fmt_i64` to be present and `format!`, `std::fmt::Display`, `std::fmt::Debug`,
`#[derive(Debug`, and `ToString` to be absent. The rendering itself is
`stark_runtime::format`'s — the same module `println` emits into and the same one
`interp::canonical_float` delegates to, so a generic `fmt()` cannot render a value differently from
a `println` of it in any engine.

One honest qualification: `fmt_i64`/`fmt_u64` produce their digits through Rust's
`i64::to_string`, inside the shared canonical formatter that pre-dates this work and that the HIR
oracle also calls. That is STARK's rendering authority, not Rust trait dispatch deciding what a
STARK `Display` means — the selection of `Display::fmt` was made by STARK's trait machinery at
lowering. `fmt_bool`, `fmt_char` and `fmt_unit` build their strings by `push`/`push_str` and owe
nothing to a Rust trait impl.

### Ambiguity

`same_name_from_two_bounds_is_ambiguous` runs both bound orders. Both are rejected, and the
diagnostic names both traits. Neither being compiler-known nor being written first is a tie-breaker.

---

## 6. Transitional compromises

**One, and it is smaller than the work package anticipated.** The fallback the WP sanctioned — a
`CoreTraitMethod` metadata bridge — was already in the tree as `core_trait_contract`, written by
WP-C7.9 Packet B to check user `impl` blocks against a Core trait's required shape. This work
package reads that existing table rather than adding a parallel one, so there is exactly one
statement of a Core trait's signatures and it is the one implementations are checked against.

It is still not an ordinary trait DECLARATION, and the WP's preferred architecture asks for one.
Recording that plainly: **Core trait method metadata must eventually be derived from ordinary trait
declarations** — a prelude `trait Display { fn fmt(&self) -> String; }` carried as a real HIR item
with a lang-item-like classification attached, at which point `core_trait_contract` and
`BoundTrait::Core` both disappear and `BoundTrait` collapses to a single `ItemId`. That is a
resolver-bootstrap change (the prelude has no source file today) and is out of scope here.
Follow-up: **remove the duplicate Core-trait signature registry once the prelude carries real trait
items.**

---

## 7. Deferred follow-ups

| id | what |
| --- | --- |
| — | Derive Core-trait method metadata from ordinary prelude trait declarations; delete `core_trait_contract` and `BoundTrait::Core`. |
| DEV-167 | Method-form `to_string()` — needs blanket implementations. `stark-fmt` ships the free function. |
| DEV-168 | `Display::fmt(&x)` has no MIR lowering. The spec names this call as the disambiguation mechanism, and it runs in one engine of three. |
| DEV-169 | Explicit `.drop()` is accepted. Pre-existing, in the concrete path; needs a spec-vs-implementation ruling. |
| — | `Clone::clone`, `Hash::hash`, `Iterator::next` and `Into::into` are now callable through a bound at the FRONT END, and their concrete lowering is uneven (`Ord::cmp` is complete; the others are not). A program using them generically fails at lowering rather than at type checking — a worse diagnostic position than before, for shapes that were rejected outright before. Worth a follow-up pass; not a regression in what compiles. |

---

## 8. Does this satisfy the REST server's formatting prerequisite?

**Yes, for what a server observable surface needs: rendering values into text.** A log line, a
status line, a header value and a small JSON scalar can all be built today with
`packages/stark-fmt`, over any primitive and any user type carrying an `impl Display`, in the
interpreter and natively, in debug and release. That was not possible before this work package —
not awkward, not possible.

Two limits to plan around rather than discover:

* There is no format-string syntax, no interpolation, no padding, width, precision or alignment.
  A rendered table or a fixed-width log column has to be assembled by hand. Deliberate:
  the work package excludes all of it.
* `Display` is not a serialisation format. It has no escaping, so composing JSON from
  `Line::value` is wrong for any string that can contain a quote. The REST work needs a real
  serialiser (`stark-json` already exists) for payloads; `stark-fmt` is for human-readable text.

Recommendation: treat the prerequisite as **met**, and scope the REST server's observable surface
to `Display`-rendered text plus `stark-json` for payloads. Do not schedule format strings as a
REST-server dependency — nothing in an observable surface needs them, and pulling them in would
reopen the ergonomics questions this work package deliberately closed.
