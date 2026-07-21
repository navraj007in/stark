# Native Provider ABI v0.1 — CE4 Amendment 1 (PROPOSED)

**Status**: PROPOSED — awaiting owner CE4 decision. Nothing in this document has been applied.
**Raised**: 2026-07-21, from external review of head `37828a07`.
**Amends**: `STARKLANG/docs/compiler/native-provider-abi-v0.1.md` (approved CD-046).
**Affects**: `starkc/src/backend/provider_abi.rs`, `starkc/stark-runtime/src/provider_abi.rs`.
**Recorded in**: `COMPILER-STATE.md` CD-052.

## Why this is an amendment and not a fix

The Native Provider ABI v0.1 was approved by the owner under CD-046 (commit `c9eaa53`, "owner CE4
approval of Native Provider ABI v0.1 — WP-C5.1 CLOSED"). Items 1 and 2 below are contradictions
*between the approved document and the code that implements it* — and, as it turns out, within
the approved document itself. Either would ordinarily be a straightforward bug fix, except that
resolving one requires choosing which side is authoritative, and both sides are owner-approved.
That choice is a CE4 decision. Item 3 is editorial and needs no decision.

Items 1 and 2 are raised now, together, because the cost of correcting them is currently near zero
and rises sharply later: **no provider executes in the C5 MVP** (§17, per `WP-C5-ENTRY.md` §10.2),
so there is no deployed provider to migrate, no generated code depending on the current shapes,
and no compatibility surface to preserve. Both become breaking changes the moment provider
execution lands.

Neither is a defect in anything that runs today. Both are latent: they describe an ABI that cannot
be expressed, and an ownership rule that cannot be enforced, in code that is not yet exercised.

---

## Item 1 — `FunctionDecl`'s return shape contradicts the document's own return convention

### The contradiction

`native-provider-abi-v0.1.md` §11 states:

> Every provider function returns `ProviderStatus`, never a value directly; results are written
> through an out-parameter (`BorrowedBufferMut` or a `*mut` scalar/`ResourceHandle` slot passed by
> the caller)

§11 also gives the rationale: this is the ordinary C ABI idiom (status code plus out-parameters),
chosen specifically so the boundary needs no hand-written unsafe access to a Rust-shaped tagged
union on either side.

**The document contradicts itself before the code is even consulted.** §8 ("Ownership transfer in
both directions") describes the out-direction as:

> **Out of the provider**: a provider function that *returns* a `ResourceHandle` hands STARK a
> freshly-owned resource […]

and §7 says a handle may be constructed only "as a provider function's return value". Both assume
direct returns; §11 forbids them. So the choice below is not merely "document vs. code" — the
document must pick a side regardless.

The metadata model in `starkc/src/backend/provider_abi.rs` does not express that convention:

```rust
pub struct FunctionDecl {
    pub name: String,
    pub capability: String,
    pub params: Vec<AbiType>,
    pub returns: AbiType,        // <- a free choice of return type
    pub is_close_for: Option<String>,
    pub may_block: bool,
}
```

There is no representation of an out-parameter at all, and `returns` is unconstrained. The
validator's own `valid_example_kv` fixture — the document's §17 mock, the thing that demonstrates
what a conforming provider looks like — contradicts §11 directly:

| Function | `returns` in the fixture | §11 requires |
| --- | --- | --- |
| `kv_open` | `ResourceHandle` | `ProviderStatus`, handle via out-parameter |
| `kv_get` | `ProviderStatus` | `ProviderStatus` ✓ (but the retrieved value has nowhere to go) |
| `kv_set` | `ProviderStatus` | `ProviderStatus` ✓ |
| `kv_close` | `ProviderStatus` | `ProviderStatus` ✓ |

`kv_open` returns a value directly, which §11 forbids. `kv_get` returns the right status but has
no way to express *where the retrieved value is written* — its `BorrowedBuffer` parameter is an
input (the key), and no output slot exists in the model.

So the metadata model can neither express the ABI the document specifies nor reject a provider
that violates it. A validator whose canonical positive fixture violates the contract it validates
is not enforcing that contract.

### Recommended resolution

Make the physical return fixed and represent values separately. Two shapes work; the first is
recommended because it also expresses `InOut`, which a buffer-filling call needs:

**(a) Directional parameters (recommended)**

```rust
pub enum Direction { In, Out, InOut }

pub struct Param {
    pub direction: Direction,
    pub ty: AbiType,
}

pub struct FunctionDecl {
    pub name: String,
    pub capability: String,
    pub params: Vec<Param>,
    // No `returns` field: the physical ABI return is always ProviderStatus.
    pub is_close_for: Option<String>,
    pub may_block: bool,
}
```

**(b) Split input/output lists**

```rust
pub inputs: Vec<AbiType>,
pub output: Option<AbiType>,
```

Shape (b) is simpler but cannot express a caller-allocated buffer that the provider fills, which
is the `BorrowedBufferMut` case §11 explicitly names — so (a) is the better fit for the ABI as
written.

Under either shape the validator gains checks it cannot currently perform:

- the physical return is `ProviderStatus` by construction (unrepresentable otherwise);
- a close function has the mandatory shape `close(In ResourceHandle) -> ProviderStatus`;
- `Out`/`InOut` parameters are the only way a function yields a value.

### Alternative, if the owner prefers to amend the document instead

If direct returns are in fact wanted, §11 should be rewritten to permit them and to state which
types may be returned directly, and the rationale paragraph (no hand-written unsafe union access
at the boundary) needs revisiting, since it is the argument *for* the status-plus-out-parameter
form. This is the less attractive direction: the C ABI idiom §11 chose is the reason the boundary
is simple.

---

## Item 2 — `ResourceHandle` is `Copy` despite exclusive-ownership semantics

### The contradiction

`native-provider-abi-v0.1.md` §8 states that ownership is never shared or aliased, and that passing
a handle by value into a provider function transfers ownership — STARK's generated code "must not
use it again afterward", explicitly "the same discipline as an ordinary STARK move". §13 requires
the resource's `is_close_for` function to be called **exactly once**, tied to the `Drop`
terminator, matching `WP-C5-ENTRY.md` §7.5's "exactly once, never repeated automatically"
invariant.

`starkc/stark-runtime/src/provider_abi.rs` derives the opposite:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceHandle {
```

`Copy` means every use is a duplication and the compiler will never object to a use-after-transfer
— the exact thing §8 says must not happen. Two apparent owners of one resource is a double-close
waiting for provider execution to land. The document itself points the same way: §7 says generated
code "may copy it, pass it, store it, and drop it", one section before §8 imposes exclusive-move
semantics. That sentence should be corrected as part of this amendment.

Note the shape of this one: it is not a bug that can be hit today, because nothing closes anything
yet. It is a type-level permission that makes the invariant unenforceable *by construction* the
moment it starts mattering.

### Recommended resolution

- Remove `Clone` and `Copy` from the owning handle type.
- A raw, C-compatible, `Copy` representation may remain as an internal FFI value (it has to cross
  the boundary as a scalar), but generated STARK code should only ever see a non-`Copy` owning
  wrapper. This is the same split the entry plan already uses elsewhere: an unsafe-adjacent raw
  form confined to the boundary, a safe owning form everywhere else.
- Duplicating a resource, if any provider ever needs it, must be an explicit provider operation
  (`duplicate` / `clone_handle`) with defined failure and ownership semantics — not an implicit
  language-level copy.

---

---

## Item 3 — a dangling cross-reference in §8 (editorial)

Not a design question; noted here so it is corrected in the same pass rather than found again.

§8 requires STARK to call "that resource type's `is_close_for` function (**§12**) exactly once".
Close/`Drop` semantics are **§13**; §12 is "Distinction among provider error, STARK trap, and host
failure". The pointer should read §13.

---

## What is being asked of the owner

1. **Item 1**: adopt shape (a), adopt shape (b), amend §11 to permit direct returns, or reject.
   Note that §7/§8 currently assume direct returns and §11 forbids them, so *some* edit to the
   document is required under every option except "reject and leave the contradiction".
2. **Item 2**: approve removing `Clone`/`Copy` from the owning handle (with the raw/wrapper split),
   or reject. Includes correcting §7's "may copy it" sentence.
3. **Item 3**: editorial, assumed approved unless the owner objects.
4. If Item 1 or 2 is approved, whether the ABI version stays `0.1` (defensible — nothing has
   shipped against it) or bumps.

Until this is decided, `provider_abi.rs` (both crates) and the approved ABI document are unchanged.
