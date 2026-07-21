# Native Provider ABI v0.1 — CE4 Amendment 1 (APPROVED, revision 3)

**Status**: **APPROVED** — CE4 Amendment 1 to Native Provider ABI v0.1, approved by the owner
2026-07-21 under CD-054, conditional on the required changes recorded in §0.1, which revision 3
incorporates. The approved ABI document and both implementation files are updated in the same
commit that records this approval.
**Raised**: 2026-07-21, from external review of head `37828a07`.
**Revised**: revision 2 on owner direction (CD-053, revision 1 not approved); revision 3 on owner
direction (CD-054 — revision 2's design approved *with required changes*).
**Amends**: `STARKLANG/docs/compiler/native-provider-abi-v0.1.md` (approved CD-046).
**Affects**: `starkc/src/backend/provider_abi.rs`, `starkc/stark-runtime/src/provider_abi.rs`.
**ABI version**: stays **`0.1`** (owner decision — nothing has shipped or executed against this
ABI, so correcting a pre-execution contract is an amendment, not a version bump).
**Recorded in**: `COMPILER-STATE.md` CD-052 (raised), CD-053 (revision 2), CD-054 (this revision
and its approval).

---

## 0. Revision history

Revision 1 raised two contradictions and recommended fixes. The owner approved the *principles*
but rejected the *shape*, because revision 1 left four issues unresolved and its recommended
parameter model conflated two independent concepts. Revision 2 keeps the approved principles
verbatim, resolves the four omissions, and replaces the recommended model.

| | Revision 1 | Revision 2 |
|---|---|---|
| Physical return | `ProviderStatus` always | unchanged — **approved** |
| Value results | out-parameters | explicit output channels — **approved**, now enumerated |
| Owning handle | not `Clone`/`Copy` | unchanged — **approved** |
| Raw FFI handle | may stay `Copy` at the boundary | unchanged — **approved** |
| Owning handle `Drop` | *not addressed* | **must not** implement Rust `Drop`; verified MIR keeps the exactly-once obligation |
| Buffers | §8 left them as "ownership transfer" | §8 corrected: buffers are **borrowed call-duration views**, never ownership transfer |
| Resource ops on a handle | consuming only (v0.1 §8 forbade borrowed handles) | **borrowed (non-consuming) handle form added**; consuming form reserved for close and explicitly ownership-taking operations |
| Handle typing | untyped `AbiType::ResourceHandle` | every handle parameter and handle output **names its declared resource type** |
| Parameter model | `Direction × AbiType` product | **closed parameter enum** — direction and resource ownership are separate concepts and are no longer multiplied together |
| Close-function check | shape `close(In ResourceHandle)` | exactly one **consumed handle of the declared resource type**, and **no ordinary value output** |

Revision 1's Item 3 (a dangling `§12` cross-reference that should read `§13`) is editorial, was
approved, and is carried forward unchanged in §6 below.

### 0.1 What changed in revision 3 (owner-required, CD-054)

Revision 2's design was approved — the closed `AbiParam` model, the fixed physical
`ProviderStatus` return, explicit output channels, typed borrowed/consumed/output handles,
borrowed buffer semantics, the `RawResourceHandle`/`OwnedResourceHandle` separation, owning
handles being non-`Clone`/non-`Copy`/no-`Drop`, ABI version `0.1`, and the corrected
example-provider shapes. Five changes were required before approval could be recorded, and are
incorporated here:

1. **The flagged close-function question is ruled.** A close function takes **exactly one
   parameter** — the consumed handle — and nothing else (§4.4). Revision 2's permissive reading
   (additional pure inputs allowed) is withdrawn.
2. **Consumed-handle error rule** — new §4.6.
3. **Output initialisation rule** — new §4.7.
4. **Close-failure rule** — new §4.8.
5. **Physical ABI mapping** — new §4.9, including the requirement that all raw↔owned conversions
   go through isolated reviewed boundary helpers rather than generated ad hoc field access.

---

---

## 1. The two contradictions being amended (unchanged from revision 1)

Restated compactly; the full argument is unchanged and is not re-litigated here.

**1a. `FunctionDecl`'s return shape contradicts the document's own return convention.** §11 states
that every provider function returns `ProviderStatus`, never a value directly, with results
written through an out-parameter — and gives the rationale (the ordinary C ABI idiom, chosen so
the boundary needs no hand-written unsafe access to a Rust-shaped tagged union on either side).
But §7 says a handle may be constructed only "as a provider function's return value" and §8
describes the out-direction as a function that *returns* a `ResourceHandle`. The document
contradicts itself before the code is consulted. The metadata model then expresses neither
convention: `FunctionDecl.returns: AbiType` is a free choice with no representation of an
out-parameter at all, and §17's own `valid_example_kv` fixture has `kv_open` returning
`ResourceHandle` directly (which §11 forbids) and `kv_get` returning `ProviderStatus` with nowhere
to put the retrieved value. A validator whose canonical positive fixture violates the contract it
validates is not enforcing that contract.

**1b. `ResourceHandle` is `Copy` despite exclusive-ownership semantics.** §8 says ownership is
never shared or aliased and that passing a handle by value transfers it ("the same discipline as
an ordinary STARK move"); §13 requires the close function to run exactly once. The runtime type
derives `Clone, Copy`, so every use is a duplication and no use-after-transfer can ever be
diagnosed. §7's "may copy it, pass it, store it, and drop it" points the same wrong way, one
section before §8 imposes exclusive-move semantics.

Neither is a defect in anything that runs today — no provider executes in the C5 MVP (§17, per
`WP-C5-ENTRY.md` §10.2). Both are latent: an ABI that cannot be expressed, and an ownership rule
that cannot be enforced. Both become breaking changes the moment provider execution lands, which
is why they are corrected now, at near-zero cost.

---

## 2. Approved principles (owner, CD-053)

These are settled and are restated here so the revised design has a fixed frame. They are not
reopened by this revision.

1. **Every physical provider function returns `ProviderStatus`.** No function returns a value
   directly.
2. **Result values travel through explicit output channels.** A value a provider produces is
   named as such in the function's declaration; it is never the physical return.
3. **The generated owning resource representation must not be `Clone` or `Copy`.**
4. **A separate raw, C-compatible handle may remain `Copy` inside the isolated FFI boundary.** It
   has to cross as a scalar; it is confined to the boundary and generated STARK code never sees
   it.
5. **The owning wrapper must not use Rust `Drop`; verified MIR remains responsible for
   exactly-once close.**

Principle 5 is the one revision 1 never addressed, and it is not a detail. If the owning wrapper
implemented Rust `Drop`, the exactly-once close obligation would have two independent enforcers:
MIR's `Drop` terminator (which `WP-C5-ENTRY.md` §7.5 makes the *verified* one) and the generated
Rust type system. Two enforcers means either a double close on every resource, or an authority
split where the answer to "what guarantees close runs exactly once?" depends on which engine you
ask. The MIR verifier is the authority; the generated Rust type must not silently become a second
one. The cost is explicit and accepted: a wrapper with no `Drop` can be *leaked* by generated Rust
that fails to call close — which is exactly the property MIR verification is responsible for
excluding, and exactly the property that would become unfalsifiable if Rust papered over it.

---

## 3. The four omitted issues, resolved

### 3.1 Buffers are borrowed call-duration views, not ownership-transferred values (§8 correction)

§9 already states the correct rule — both buffer types are "valid **only for the duration of the
call that received them**", and a provider must not retain the pointer past return. §8 then
contradicts it by listing buffers alongside handles as things that transfer ownership when passed
by value: "a `ResourceHandle` **or buffer (§9)** passed *by value* into a provider function
transfers ownership".

That is wrong, and it is wrong in the direction that matters: it says STARK's generated code
"must not use it again afterward" — but a caller-allocated buffer is precisely a thing the caller
*does* use again afterward. That is the entire point of `BorrowedBufferMut`: the caller allocates,
the provider fills, **the caller reads the result**. Under §8 as written, reading the buffer you
just passed to `kv_get` is a use-after-transfer.

**Resolution.** §8 covers `ResourceHandle` ownership only. Buffers are removed from it and stay
governed by §9's borrow contract: the pointer is valid for the call's duration, the provider may
not retain it, and ownership never moves in either direction. The word "ownership" does not apply
to a buffer parameter at all.

### 3.2 Ordinary resource operations need a non-consuming borrowed-handle form

§8's last paragraph currently states: "No function signature in v0.1 may take or return a
*borrowed* `ResourceHandle` — only full ownership transfer", deferring a borrowed convention until
some later package has a real use case.

The real use case is already in the document. §17's own mock provider has `kv_get` and `kv_set`,
both of which take the store's handle. With consuming-only handles, `kv_get(store, key, out)`
**consumes the store** — one read and the handle is gone, and the second call in any realistic
program is a use-after-transfer. The v0.1 deferral was not a conservative choice; it made the
canonical example unexpressible.

**Resolution (owner direction).** The deferral is lifted. Two handle input forms exist:

- **borrowed** — the caller retains ownership; the provider may use the resource for the duration
  of the call and must not retain the handle past return (the same lifetime rule §9 gives
  buffers). This is the form ordinary operations use.
- **consumed** — ownership transfers into the provider; the caller must not use the handle again.
  This is the form close uses, and the only other legitimate users are operations that explicitly
  take ownership (e.g. a hypothetical `kv_into_snapshot` that ends the store's life).

The default for any ordinary operation is *borrowed*. Consuming is the exception and must be
justified by the operation's semantics.

### 3.3 Every handle parameter and handle output must name its declared resource type

`AbiType::ResourceHandle` is untyped: nothing in a `FunctionDecl` distinguishes a `KvStore` handle
from any other provider resource's handle. The runtime struct carries a `resource_type: u32`
field, but that is a *runtime value*, not something the compile-time validator can check a
declaration against — so today the validator cannot reject a provider declaring
`kv_close(ResourceHandle)` where the handle is, per its own declared function table, some other
resource type entirely. §13 makes exactly that a contract violation ("calling a resource's close
function on ... a handle from a different resource type"), and §17's validator has no way to see
it.

**Resolution.** Every handle-carrying parameter and every handle output names a declared resource
type (a `String` matching an entry in `ProviderMetadata.resource_types`, checked by the
validator). A handle naming an undeclared resource type is a new violation class.

### 3.4 Parameter direction and resource ownership are separate concepts

This is why revision 1's recommended shape (a) — `Param { direction: Direction, ty: AbiType }` —
is **rejected rather than refined**. It is a product of two orthogonal axes, and most of the
product is meaningless:

| | `In` | `Out` | `InOut` |
|---|---|---|---|
| `U8`…`F64` | copied scalar input ✓ | scalar output slot ✓ | in/out slot ✓ |
| `BorrowedBuffer` | immutable borrowed view ✓ | **meaningless** — an immutable view cannot be an output | **meaningless** |
| `BorrowedBufferMut` | mutable borrowed view ✓ | ✓ (same thing, spelled twice) | ✓ (same thing, spelled a third time) |
| `ResourceHandle` | borrowed **or** consumed — *the axis cannot say which* | newly-owned output ✓ | **meaningless** |
| `ProviderStatus` | **meaningless** — it is the physical return | **meaningless** | **meaningless** |

Fifteen combinations; six are meaningful, three are the same case spelled three ways, and the one
distinction that actually matters — borrowed vs. consumed handle — is the one the model *cannot
express*, because ownership is not a direction. A validator over this shape would spend its
existence rejecting combinations the type system should never have allowed to be written, and
would still need a second field to carry ownership.

**Resolution (owner direction).** A closed parameter model in which invalid combinations are
unrepresentable.

---

## 4. The revised metadata model

### 4.1 Parameter forms

```rust
/// The scalar widths that cross the boundary by value. Split out of the old flat `AbiType`:
/// these are the only members of that enum that were ever *types*; the rest were parameter
/// FORMS, which is precisely the conflation §3.4 removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarTy {
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    Bool,
    F32, F64,
}

/// §6 (amended): the closed parameter vocabulary. Every variant is one of the seven forms the
/// owner enumerated; there is no product of independent axes, so every representable parameter
/// is meaningful and every meaningful parameter is representable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiParam {
    /// 1. Copied scalar input.
    ScalarIn(ScalarTy),

    /// 2a. Scalar output slot: caller-allocated, provider writes, caller reads after return.
    ScalarOut(ScalarTy),
    /// 2b. Scalar in/out slot: caller writes, provider reads AND overwrites, caller reads back.
    ScalarInOut(ScalarTy),

    /// 3. Immutable borrowed buffer. Call-duration view (§9); never ownership (§3.1).
    BufferIn,
    /// 4. Mutable borrowed buffer. Call-duration view; the provider may fill and/or modify it,
    ///    and the CALLER reads it after return -- which is why §8 must not call this a
    ///    transfer.
    BufferInOut,

    /// 5. Borrowed typed resource handle -- the default for ordinary resource operations. The
    ///    caller retains ownership; the provider must not retain the handle past return.
    HandleBorrowed { resource_type: String },
    /// 6. Consumed typed resource handle: ownership transfers in, the caller must not use it
    ///    again. Close functions, and operations that explicitly end a resource's life.
    HandleConsumed { resource_type: String },
    /// 7. Newly-owned typed resource-handle output slot: the provider writes a handle the
    ///    CALLER now solely owns and must eventually close exactly once (§13).
    HandleOut { resource_type: String },
}
```

Three properties follow structurally, with no validator rule required:

- **A buffer can never be "owned" or "consumed"** — no variant says so.
- **A handle's ownership is always stated** — borrowed, consumed, or newly-owned-out; there is no
  handle form that leaves it open.
- **`ProviderStatus` is not a parameter form and not a type** — it is the fixed physical return
  (§4.2), so it cannot be written as a parameter at all.

### 4.2 `FunctionDecl`

```rust
pub struct FunctionDecl {
    pub name: String,
    pub capability: String,
    pub params: Vec<AbiParam>,
    // No `returns` field. The physical ABI return is ALWAYS `ProviderStatus` (§11), so it is a
    // property of the ABI rather than a per-function choice -- unrepresentable otherwise, which
    // is what makes §11 enforced by construction instead of by a check nobody wrote.
    pub is_close_for: Option<String>,
    pub may_block: bool,
}
```

### 4.3 The runtime types

```rust
// stark-runtime/src/provider_abi.rs

/// The raw, C-compatible handle. `Copy` because it crosses the boundary as a scalar and has to
/// be. CONFINED to the FFI boundary -- generated STARK code never names this type.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawResourceHandle {
    pub id: u64,
    pub resource_type: u32,
}

/// The owning form generated code sees. NOT `Clone`, NOT `Copy` (§8's exclusive ownership) -- so a
/// use-after-transfer is a Rust compile error in the generated crate rather than a silent
/// double-close. And NO `impl Drop` -- the exactly-once close obligation (§13) belongs
/// to verified MIR's `Drop` terminator (`WP-C5-ENTRY.md` §7.5), and a Rust destructor here
/// would either double it or quietly take it over.
#[derive(Debug)]
pub struct OwnedResourceHandle(RawResourceHandle);
```

`ProviderStatus`, `BorrowedBuffer`, and `BorrowedBufferMut` keep their current `#[repr(C)]`
definitions unchanged.

Duplicating a resource, if any provider ever needs it, must be an explicit provider operation
(`duplicate`/`clone_handle`) with defined failure and ownership semantics — returning a new
`HandleOut` — never an implicit language-level copy.

### 4.4 New and changed validator rules

The validator gains checks it structurally could not perform before:

| Rule | Enforced by |
|---|---|
| §11: the physical return is `ProviderStatus` | **construction** — no `returns` field exists |
| §10: no internal generated-Rust aggregate crosses | **construction** — `AbiParam`/`ScalarTy` are closed |
| §15: no callbacks | **construction** — no function-pointer variant |
| A value result is declared as an output form | **construction** — outputs are parameter forms |
| Every handle names a declared resource type | new check → `HandleResourceTypeUndeclared` |
| A close function's shape | new check → `CloseFunctionShape` |

```rust
pub enum AbiViolation {
    // ... existing variants unchanged ...

    /// §6/§13: a `HandleBorrowed`/`HandleConsumed`/`HandleOut` parameter names a resource
    /// type the provider never declared in `resource_types`, so §13's wrong-resource-type rule
    /// could not be checked against it.
    HandleResourceTypeUndeclared {
        function: String,
        resource_type: String,
    },

    /// §13 (amended): a function with `is_close_for: Some(rt)` does not have the mandatory
    /// close shape -- see below for the exact rule.
    CloseFunctionShape {
        function: String,
        resource_type: String,
        problem: CloseShapeProblem,
    },
}

/// Which clause of the close-function rule a declaration broke. Named per clause rather than
/// collapsed into one string, so a violation says what to fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloseShapeProblem {
    /// The function does not take exactly one parameter. A close function's whole signature is
    /// the resource being closed -- see the rule below for why this is architectural.
    NotExactlyOneParameter { found: usize },
    /// The single parameter is not a `HandleConsumed` -- e.g. it is `HandleBorrowed`, which
    /// would close a resource the caller still believes it owns.
    ParameterIsNotAConsumedHandle,
    /// The consumed handle's resource type is not the one `is_close_for` names.
    ConsumedHandleWrongResourceType { found: String },
}
```

**The close-function rule (owner ruling, CD-054).** A function with `is_close_for: Some(rt)` must
take **exactly one parameter**:

```rust
HandleConsumed { resource_type: rt }
```

No additional scalar, buffer, handle, or output parameter of any kind. Its only result is the
`ProviderStatus` every function returns.

The reason is architectural rather than stylistic, and it is worth stating because the permissive
reading revision 2 floated (allow pure inputs like `flush: Bool`) looks harmless until you ask who
would supply them. **MIR's `Drop(place)` terminator supplies only the resource being dropped.**
There is no argument list at a drop site and no place to put one — the drop is implicit in the
STARK program's control flow, not a call the programmer wrote. A close function with a second
parameter is therefore a function the generated code *cannot call*: every additional argument
would have to be invented by the backend out of nothing.

The consequence is a design rule, not just a validation rule: **any flush option, completion mode,
or other fallible operation that needs arguments must be a separate, explicitly invoked provider
function, called before Drop.** That is also the only way such an operation can report failure
usefully — see §4.8, where a close function's own failure has nowhere to go.

### 4.5 The §17 fixtures, corrected

`valid_example_kv` currently violates the contract it is the positive fixture for. Under the
revised model it becomes:

| Function | Parameters | `is_close_for` |
|---|---|---|
| `kv_open` | `BufferIn` (path), `HandleOut { "KvStore" }` | — |
| `kv_get` | `HandleBorrowed { "KvStore" }`, `BufferIn` (key), `BufferInOut` (value out), `ScalarOut(U64)` (bytes written) | — |
| `kv_set` | `HandleBorrowed { "KvStore" }`, `BufferIn` (key), `BufferIn` (value) | — |
| `kv_close` | `HandleConsumed { "KvStore" }` | `Some("KvStore")` |

Every function's physical return is `ProviderStatus`, by construction. `kv_open`'s handle now has
a declared destination; `kv_get`'s retrieved value now has somewhere to go; `kv_get`/`kv_set` no
longer consume the store they operate on; `kv_close` consumes it exactly once, and the validator
can now say so.

New invalid fixtures accompany each new violation class — a handle naming an undeclared resource
type, and one per `CloseShapeProblem` variant (a close taking a second parameter, a close whose
single parameter is `HandleBorrowed` rather than `HandleConsumed`, and a close consuming the wrong
resource type) — matching §17's existing one-fixture-per-violation-class discipline.

### 4.6 Consumed-handle error rule (owner-required, CD-054)

**Ownership transfers at provider-call entry.** A `HandleConsumed` value is dead to generated
STARK code the moment the call is made — *regardless of what `ProviderStatus` reports*. There is
no "the call failed, so you still own it" path.

This is the only rule that makes the consumed form analysable at all. The alternative — ownership
returning to the caller on failure — would make a handle's liveness depend on a runtime value, so
whether a later use is a use-after-transfer could not be decided by MIR verification, only
observed at runtime. Exactly-once close would stop being a static property.

An operation that wants ownership back on failure must say so structurally, and has two ways to:

- declare an explicit `HandleOut` channel, so a failed call can hand back a *fresh* owned handle
  (a new value, not the resurrection of a dead one); or
- take the handle as `HandleBorrowed`, so ownership never moved and the question does not arise.

The second is the right answer for almost every operation. `HandleConsumed` should be rare: close,
and operations that genuinely end a resource's life.

### 4.7 Output initialisation rule (owner-required, CD-054)

`ScalarOut` and `HandleOut` storage is **uninitialised before the call** and **valid only when
`ProviderStatus` reports success**. Generated code must:

- allocate the raw output slot through `MaybeUninit` — never a fabricated zero value, which would
  be indistinguishable from a real result the provider wrote;
- **never read or wrap it on failure**, and never run any conversion over it;
- for `HandleOut`, **validate the returned raw handle's `resource_type` against the declared
  resource type before constructing the owning wrapper** (§4.9's boundary helper is where this
  check lives, so it cannot be skipped by a code path that forgets it). A mismatch is a provider
  contract violation — a STARK trap per §12's middle row, not a provider error.

`ScalarInOut` and `BufferInOut` are different and stay different: both are **caller-initialised
and caller-owned across the call**. The provider reads and may overwrite them; it never allocates
them, and their validity does not depend on the status code.

The asymmetry is the point. An `Out` slot is a promise the provider only keeps on success; an
`InOut` slot is the caller's own memory, lent for the duration of the call.

### 4.8 Close-failure rule (owner-required, CD-054)

**A close function's nonzero `ProviderStatus` cannot become a recoverable `Result::Err`.** MIR's
`Drop` terminator has no result destination — there is no place in the STARK program for the value
to land, because the program never wrote the call. §12's first row (provider error → recoverable
`Result::Err`) therefore does not apply to close, and this is the one place where a nonzero status
is not an ordinary expected failure.

Treat it as a distinct **fatal provider-close/host failure**:

- **abort without unwinding** (`WP-C5-ENTRY.md` §7.7's trap discipline);
- **do not retry close** — the handle's state after a failed close is undefined by this ABI;
- **consider the handle consumed** regardless (§4.6 applies to close like any other consuming
  call);
- **do not run further pending Drop glue** — the same rule a trap already follows, and for the
  same reason: once a close has failed, the runtime's resource invariants can no longer be
  trusted, so continuing to close *other* resources is guesswork.

Explicit recoverable work — flushing, committing, syncing, anything whose failure the program
should be able to handle — must be a **separate provider operation performed before close**, where
its status has somewhere to go. This is the same conclusion §4.4's ruling reaches from the other
direction: close takes no arguments *and* returns nothing recoverable, so anything that needs
either is not close.

### 4.9 Physical ABI mapping (owner-required, CD-054)

Every `AbiParam` variant maps to exactly one C ABI parameter. Stated explicitly so no
implementation has to infer it:

| `AbiParam` variant | C ABI parameter |
|---|---|
| `ScalarIn(T)` | `T` |
| `ScalarOut(T)` | `*mut T` |
| `ScalarInOut(T)` | `*mut T` |
| `BufferIn` | `BorrowedBuffer` |
| `BufferInOut` | `BorrowedBufferMut` |
| `HandleBorrowed { .. }` | `RawResourceHandle` (boundary-confined) |
| `HandleConsumed { .. }` | `RawResourceHandle` (boundary-confined) |
| `HandleOut { .. }` | `*mut RawResourceHandle` |
| *physical function return* | `ProviderStatus` |

Two observations this table makes plain, both intentional:

- **`ScalarOut` and `ScalarInOut` are physically identical** (`*mut T`). Their difference is a
  *contract* difference (§4.7: uninitialised-and-valid-only-on-success vs. caller-initialised),
  not a representational one. That is exactly why they are separate `AbiParam` variants rather
  than one — the metadata carries a distinction the C signature cannot.
- **`HandleBorrowed` and `HandleConsumed` are also physically identical** (a `RawResourceHandle`
  by value). Their difference is the ownership contract (§4.6). A provider author reading only
  the C signature cannot tell them apart; the declaration is what says which it is, and the
  validator is what enforces it.

**Boundary-helper requirement.** All raw→owned and owned→raw conversions must go through the
isolated, reviewed boundary helpers in `stark-runtime` — never through generated ad hoc field
access on `RawResourceHandle`. Generated code must not read or write `id`/`resource_type`
directly. This is what keeps the unsafe-adjacent surface reviewable: the resource-type validation
of §4.7, and the "confined to the boundary" property of the raw form, are both properties of a
small named set of helpers rather than of every generated call site.

---

## 5. Document sections this amendment edits

| § | Change |
|---|---|
| §6 | `FunctionDecl` loses `returns`; `params` becomes `Vec<AbiParam>`; `AbiType` is replaced by `AbiParam` + `ScalarTy` |
| §6.1 (new) | the physical ABI mapping table (§4.9) and the boundary-helper requirement |
| §7 | "may copy it, pass it, store it, and drop it" corrected — the owning form is non-`Copy`, non-`Clone`, and has no Rust `Drop`; the raw `Copy` form is boundary-confined; "constructed only as a provider function's return value" becomes "written only into a `HandleOut` parameter, through the boundary helper, after resource-type validation" |
| §8 | buffers removed from the ownership rules (§3.1); the borrowed-handle prohibition lifted (§3.2); the consumed-handle error rule added (§4.6); the `§12` cross-reference corrected to `§13` (§6 below) |
| §9 | unchanged in substance; §8 no longer contradicts it |
| §10 | the closed-vocabulary statement restated over `AbiParam`/`ScalarTy`; `ProviderStatus` is named as the fixed return rather than a member of the type vocabulary |
| §11 / §11.1 (new) | the output initialisation rule added as §11.1 (§4.7); the existing rationale paragraph stands as written |
| §12 | gains the close-failure exception — a failed close is the fatal provider-close/host failure of §13.2, not the table's first (recoverable) row |
| §13 / §13.1 / §13.2 (new) | the close-function shape rule as §13.1 (§4.4) and the close-failure rule as §13.2 (§4.8); the exactly-once obligation stated as belonging to verified MIR, with the no-Rust-`Drop` consequence |
| §17 | `valid_example_kv` rewritten (§4.5); new invalid fixtures per new violation class; the validator's rule list updated |
| §18 | the "what may generated code do with it", "how does a provider report success vs. failure", and "what must be true about a resource type's close function" questions re-answered against the revised model |

---

## 6. Item 3 — the dangling cross-reference (editorial, approved)

§8 requires STARK to call "that resource type's `is_close_for` function (**§12**) exactly once".
Close/`Drop` semantics are **§13**; §12 is "Distinction among provider error, STARK trap, and host
failure". The pointer reads §13 after this amendment.

---

## 7. Approval and implementation

Approval is recorded in the header: **CE4 Amendment 1 to Native Provider ABI v0.1, APPROVED
2026-07-21 (CD-054)**, conditional on the §0.1 changes, which this revision incorporates. Nothing
in this amendment remains open for decision.

Implementation lands in a single commit, per the owner's direction:

1. `native-provider-abi-v0.1.md` updated per §5's edit list (the ABI document remains the
   normative contract; this amendment is the record of *why* it changed);
2. `starkc/src/backend/provider_abi.rs` — `ScalarTy`, `AbiParam`, the `returns`-less
   `FunctionDecl`, the new violation classes, and the validator rules of §4.4;
3. `starkc/stark-runtime/src/provider_abi.rs` — the `RawResourceHandle`/`OwnedResourceHandle`
   split and the §4.9 boundary helpers, which are the only sanctioned raw↔owned conversions;
4. §17's fixtures rewritten to conform, with one negative fixture per violation class.

Note on scope, unchanged from revision 1: no provider executes in the C5 MVP (§17, per
`WP-C5-ENTRY.md` §10.2). This amendment corrects a contract *before* anything is built against
it — which is why it is cheap now and why every rule in §4.6-§4.9 is a statement about code that
does not exist yet. The validator, the type definitions, and the fixtures are what exist; the
call-site code generation that must obey §4.7 and §4.9 belongs to whichever package first makes a
provider execute.
