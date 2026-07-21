# STARK Native Provider ABI v0.1

**Status: APPROVED (CE4, 2026-07-21, CD-046), as amended by CE4 Amendment 1 (2026-07-21,
CD-054).** Drafted 2026-07-21 under WP-C5.1c; CD-042 had approved *writing* this document as part
of `WP-C5-ENTRY.md`'s recommended §19 choices ("Provider ABI | Specify v0.1 in C5.1; execution not
required for MVP"), and the owner's CD-046 review approved this document's technical content as
drafted. WP-C5.1 is closed.

**Amendment 1 (CD-054).** An external review of head `37828a07` found that this document
contradicted itself on how a provider function returns a value (§11 required status-plus-output
while §7/§8 assumed direct returns), and that `ResourceHandle` was `Copy` despite §8's exclusive
ownership. Correcting those surfaced four further gaps: buffers described as ownership transfers,
no non-consuming handle form (which made this document's own §17 example unusable after one
call), untyped handles the validator could not check, and a parameter model conflating direction
with ownership. The amendment record — every contradiction, the rejected alternatives, and the
reasoning — is `native-provider-abi-v0.1-CE4-amendment-1.md`. **The ABI version stays `0.1`**:
nothing had shipped or executed against it, so this corrects a pre-execution contract rather than
breaking a live one. Sections carrying amended text are marked *(amended, CD-054)*.

**Scope boundary (WP-C5-ENTRY.md §10.2).** This document specifies the v0.1 ABI shape and ships a
compile-time metadata validator plus mock fixtures (§17). It does **not** implement dynamic
loading, real `extern "C"` linkage, or any provider actually executing — that is deferred to the
package that first needs a real provider (file/network/etc.), which must build against this
contract rather than inventing its own. No C5 exit claim may say providers run natively.

## 1. Why a provider boundary at all

Core v1 has no FFI, no concurrency, and no raw pointers (03-Type-System.md; charter §1.6
non-goals). Everything the language itself can express is closed: `Vec`, `HashMap`, `String`,
`File`/`IOError` are STARK-observable types with STARK-defined semantics, implemented however the
runtime likes. A **provider** is how the native backend eventually reaches outside that closed
world (real OS file handles, real sockets, hardware) without STARK source ever seeing a raw
pointer or an FFI-shaped type. The ABI in this document is the seam: generated Rust and a
provider's implementation agree on it; nothing else about a provider's internals is STARK's
concern.

## 2. Provider identity and semantic version (§10.1 point 1)

Every provider exports one fixed metadata record:

```text
ProviderIdentity {
    name: str                 // e.g. "example-kv" -- stable, provider-chosen
    semver: (u32, u32, u32)   // the PROVIDER's own version, not this ABI's
    abi_version: str          // MUST equal "0.1" for this contract
}
```

`abi_version` is checked before anything else in §14's compatibility gate. A provider targeting a
different ABI version is rejected outright — v0.1 makes no forward- or backward-compatibility
promise across ABI versions (that is a later-gate concern if provider ABI versioning is ever
needed simultaneously with multiple STARK releases).

## 3. Integrity hash and origin metadata (§10.1 point 2)

```text
ProviderOrigin {
    content_hash: [u8; 32]    // SHA-256 of the provider artifact, computed by the compiler
                               // at the point it accepts the provider, not asserted by the
                               // provider itself
    origin: str               // where it came from: a local path in v0.1 (a registry/URL
                               // origin is out of scope until package distribution for
                               // providers is designed, which this document does not do)
}
```

The compiler records `content_hash`/`origin` in the build manifest (§11's `build.json` in
`WP-C5-ENTRY.md`, extended with a `providers` array once a real provider exists) so a build is
reproducibly tied to the exact provider bytes it linked, not just a name and version string a
provider could lie about.

## 4. Supported target triples (§10.1 point 3)

```text
target_triples: [str]   // e.g. ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"]
```

Non-empty; the compiler rejects a provider whose declared list does not contain the build's target
triple **before** invoking `cargo build` (§16), not as a link-time surprise.

## 5. Capability declaration (§10.1 point 4)

```text
capabilities: [str]   // a closed, compiler-known vocabulary in later gates; v0.1 treats this
                       // as an opaque non-empty label set, since no capability is consumed by
                       // anything yet
```

A provider declares which STARK-facing capabilities it implements (illustrative, not a committed
v0.1 set: `"filesystem"`, `"clock"`, `"env"`). §17's validator checks only that the set is
non-empty and that every exported function (§6) is reachable from at least one declared
capability — it does not yet check capability names against a fixed enum, since no consumer of
that enum exists before a real provider package is designed.

## 6. Exported function table (§10.1 point 5) *(amended, CD-054)*

```text
FunctionDecl {
    name: str
    capability: str            // which §5 capability this function belongs to
    params: [AbiParam]
    // No `returns` field: the physical ABI return is ALWAYS ProviderStatus (§11).
    is_close_for: Option<str>  // Some(resource_type_name) if this function is THE close
                                // function for that resource type (§13); at most one function
                                // per resource type may set this, and its shape is constrained
    may_block: bool             // §14
}
```

`AbiParam` is a **closed** parameter vocabulary — one variant per admitted form, with no product
of independent axes:

```text
AbiParam =
    | ScalarIn(ScalarTy)        // copied scalar input
    | ScalarOut(ScalarTy)       // caller-allocated slot; valid only on success (§11)
    | ScalarInOut(ScalarTy)     // caller-initialised, caller-owned across the call
    | BufferIn                  // immutable borrowed view (§9)
    | BufferInOut               // mutable borrowed view (§9)
    | HandleBorrowed { resource_type }   // caller retains ownership -- the DEFAULT (§8)
    | HandleConsumed { resource_type }   // ownership transfers in at call entry (§8)
    | HandleOut { resource_type }        // newly-owned handle written on success (§8)

ScalarTy = U8 | U16 | U32 | U64 | I8 | I16 | I32 | I64 | Bool | F32 | F64
```

Three properties hold **structurally**, requiring no validator rule:

- a buffer can never be "owned" or "consumed" — no variant says so;
- a handle's ownership is always stated — borrowed, consumed, or newly-owned-out; there is no
  form that leaves it open;
- `ProviderStatus` is not a parameter form and not a type, so it cannot be written as a parameter.

There is no variant for an internal generated-Rust aggregate and none for a callback or function
pointer — see §10 and §15, both enforced by this closure rather than by a separate runtime check.

Every handle-carrying parameter names a resource type, which must appear in the provider's
declared `resource_types` (§17's validator checks this). Without it, §13's "closing a handle from
a different resource type is a contract violation" rule cannot be checked against a declaration at
all.

### 6.1 Physical ABI mapping *(added, CD-054)*

Each `AbiParam` variant maps to exactly one C ABI parameter. Stated explicitly so no
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

Two pairs are physically identical and deliberately kept distinct in the metadata, because the C
signature cannot carry the difference:

- `ScalarOut` vs. `ScalarInOut` — both `*mut T`; they differ in the **initialisation contract**
  (§11).
- `HandleBorrowed` vs. `HandleConsumed` — both a raw handle by value; they differ in the
  **ownership contract** (§8). A provider author reading only the C signature cannot tell them
  apart; the declaration says which, and the validator enforces it.

**Boundary-helper requirement.** All raw→owned and owned→raw conversions must go through the
isolated, reviewed helpers in `stark-runtime` (`OwnedResourceHandle::from_raw_checked`/`as_raw`/
`into_raw`) — never through generated ad hoc field access on `RawResourceHandle`. Generated code
must not read or write `id`/`resource_type` directly. This keeps the unsafe-adjacent surface
reviewable: §11's resource-type validation and the raw form's boundary confinement are properties
of a small named set of helpers, not of every generated call site.

## 7. Opaque resource-handle representation (§10.1 point 6) *(amended, CD-054)*

A resource handle exists in **two forms**, and the separation carries the safety story:

```rust
/// Boundary-confined. Crosses the FFI boundary as a scalar, so it must be Copy.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawResourceHandle {
    pub id: u64,
    pub resource_type: u32,   // a compiler-assigned index into the provider's declared
                               // resource-type list, not a pointer or provider-chosen tag
}

/// What generated STARK code holds. NOT Copy, NOT Clone, and NO Drop impl.
#[derive(Debug, PartialEq, Eq)]
pub struct OwnedResourceHandle(RawResourceHandle);
```

Each of those three negatives is deliberate:

- **not `Copy`/`Clone`** — §8 makes a handle exclusively owned. With `Copy`, every use is a
  duplication, a use-after-transfer is invisible to the compiler, and a double close becomes a
  matter of time rather than a diagnosable error.
- **no Rust `Drop`** — the exactly-once close obligation belongs to **verified MIR's `Drop`
  terminator** (`WP-C5-ENTRY.md` §7.5). A destructor here would either double the close or
  quietly take over an invariant the MIR verifier owns, and "which layer guarantees
  exactly-once?" must have exactly one answer.

The accepted cost is explicit: an `OwnedResourceHandle` that generated code never closes is
*leaked*. That is the property MIR verification exists to exclude — and the property that would
become unfalsifiable if a Rust destructor papered over it.

A handle is opaque to generated Rust: STARK code may hold it, pass it (borrowed or consumed), and
give it up, but may never read `id`/`resource_type` directly, and may construct one only through
§6.1's boundary helper, from a handle a provider wrote into a `HandleOut` slot, after
resource-type validation. Duplicating a resource — if any provider ever needs it — must be an
explicit provider operation with defined failure and ownership semantics, never an implicit
language-level copy.

## 8. Ownership transfer in both directions (§10.1 point 7) *(amended, CD-054)*

This section governs **resource handles only**. Buffers (§9) are borrowed call-duration views and
never transfer ownership in either direction — a caller that passes a `BorrowedBufferMut` reads it
again after the call, which is the whole point of the form.

Three handle forms, never shared or aliased ownership:

- **Borrowed into the provider** (`HandleBorrowed`) — **the default for ordinary operations.** The
  caller retains ownership; the provider may use the resource for the duration of the call and
  must not retain the handle past return (the same lifetime rule §9 gives buffers). A `get`/`set`
  shaped function uses this: it must not consume the store it reads.
- **Consumed into the provider** (`HandleConsumed`) — ownership transfers **at call entry**;
  generated code must not use the handle again (the same discipline as an ordinary STARK move,
  §7.3 of `WP-C5-ENTRY.md`). Reserved for close and for operations that explicitly end a
  resource's life; rare by design.
- **Out of the provider** (`HandleOut`) — on success the provider writes a freshly-owned handle
  into a caller-supplied slot; STARK is now solely responsible for calling that resource type's
  `is_close_for` function (§13) exactly once.

**Consumed-handle error rule.** A `HandleConsumed` value is dead to generated STARK code
regardless of whether `ProviderStatus` reports success or failure. There is no "the call failed,
so you still own it" path — ownership returning on failure would make a handle's liveness depend
on a runtime value, so whether a later use is a use-after-transfer could not be decided by MIR
verification, only observed at runtime, and exactly-once close would stop being a static property.
An operation that wants ownership back on failure must declare an explicit `HandleOut` channel
(handing back a *fresh* handle, not resurrecting a dead one) or take the handle as
`HandleBorrowed` so ownership never moved.

## 9. Borrowed buffer representations (§10.1 points 8-9)

```rust
#[repr(C)]
pub struct BorrowedBuffer {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct BorrowedBufferMut {
    pub ptr: *mut u8,
    pub len: usize,
}
```

Both are valid **only for the duration of the call that received them** — a provider must not
retain the pointer past return (v0.1 has no way to enforce this beyond the contract statement
itself; a later gate may add a generation/epoch check if a real provider needs one).
`BorrowedBufferMut` is admitted in v0.1 (§10.1 point 9 is conditional on admission) because a
`read`-shaped provider function needs somewhere to write bytes into.

## 10. No direct crossing of internal generated Rust aggregates (§10.1 point 17)

`AbiParam` (§6) is closed over exactly the eight forms §6 lists, and `ScalarTy` over exactly the
primitive scalar widths (`U8`/`U16`/.../`I64`/`Bool`/`F32`/`F64`). `ProviderStatus` is **the fixed
physical return** (§11), not a member of the parameter vocabulary. There is no
`AbiParam::Aggregate(MirTy)` or equivalent escape hatch. A provider function that needs
structured data decomposes it into these primitives and buffers explicitly — the same discipline
`WP-C5-ENTRY.md` §6.3 already applies internally ("Rust layout is backend-, target-, and
toolchain-version dependent... not a stable ABI"): a generated `#[repr(Rust)]` struct's layout is
not something a separately-compiled provider may assume, so it must never cross this boundary.

## 11. Error return representation (§10.1 point 10)

Every provider function returns `ProviderStatus`, never a value directly; results are written
through an explicit output channel — a `ScalarOut`/`ScalarInOut` slot, a `BufferInOut`, or a
`HandleOut` (§6, §6.1):

```rust
#[repr(C)]
pub struct ProviderStatus {
    pub code: u32,   // 0 = success; nonzero = provider-error, meaning defined per-provider
}
```

This is the ordinary C ABI idiom (status code + out-parameters) rather than a Rust-shaped tagged
union, because a `#[repr(C)]` union carrying a `BorrowedBuffer` payload has no safe discriminant
story without hand-written unsafe access on both sides of the boundary — the out-parameter form
needs none.

### 11.1 Output initialisation rule *(added, CD-054)*

`ScalarOut` and `HandleOut` storage is **uninitialised before the call** and **valid only when
`ProviderStatus` reports success**. Generated code must:

- allocate the raw output slot through `MaybeUninit` — never a fabricated zero value, which would
  be indistinguishable from a real result the provider wrote;
- **never read or wrap it on failure**, and run no conversion over it;
- for `HandleOut`, **validate the returned raw handle's `resource_type` against the declared
  resource type before constructing the owning wrapper**. This check lives inside §6.1's boundary
  helper precisely so it cannot be skipped by a call site that forgets it. A mismatch is a
  provider **contract violation** — a STARK trap per §12's middle row, not a provider error.

`ScalarInOut` and `BufferInOut` are different and stay different: both are **caller-initialised
and caller-owned across the call**. The provider reads and may overwrite them; it never allocates
them, and their validity does not depend on the status code.

The asymmetry is the point. An `Out` slot is a promise the provider keeps only on success; an
`InOut` slot is the caller's own memory, lent for the duration of one call.

## 12. Distinction among provider error, STARK trap, and host failure (§10.1 point 11)

Three channels, never conflated:

| Channel | Meaning | STARK-observable as |
|---|---|---|
| Provider error (`ProviderStatus.code != 0`) | An ordinary, expected failure the provider itself defines (file not found, permission denied) | `Result::Err` — recoverable STARK code handles it like any other error |
| STARK trap | The **provider violated this contract** (wrote a handle of the wrong resource type into a `HandleOut` slot — see §11.1's validation — wrote past a `BorrowedBufferMut`'s `len`, etc.) | An abort, same as any other MIR trap (`WP-C5-ENTRY.md` §7.7) — never recoverable, because a contract violation means the runtime's invariants can no longer be trusted |
| Host failure | An environment-level condition neither STARK nor the provider caused (OS resource exhaustion, the provider's own process/library aborting) | Also an abort, but classified distinctly in diagnostics (§12.5's build-time analogue: never presented as an ordinary user-source error) |

**One exception to the first row, added by CD-054:** a **close** function's nonzero status is
*not* a recoverable `Result::Err`. It is the fatal provider-close/host failure of §13.2, because a
MIR `Drop` terminator has no result destination for it to land in.

Only the compile-time validator (§17) exists in v0.1 scope; the runtime-side enforcement of the
trap/host-failure rows (detecting a contract violation at call time) is real provider-execution
work, explicitly deferred (see the scope boundary above).

## 13. Resource close/Drop semantics (§10.1 point 12) *(amended, CD-054)*

Every `resource_type` a provider declares must have **exactly one** `FunctionDecl` with
`is_close_for: Some(that_type)` (checked by §17's validator, not left to convention). STARK's MIR
`Drop` terminator on a provider-resource-typed local calls that function exactly once — the same
"exactly once, never repeated automatically" invariant `WP-C5-ENTRY.md` §7.5 already requires for
ordinary STARK Drop, extended across the provider boundary rather than given a second set of
rules. That obligation belongs to **verified MIR**, which is why the owning handle type carries no
Rust `Drop` impl (§7).

Calling a resource's close function on an already-closed handle, or on a handle from a different
resource type, is a contract violation — a STARK trap (§12's middle row), not a provider error.

### 13.1 The close-function shape *(added, CD-054)*

A close function takes **exactly one parameter**:

```text
HandleConsumed { resource_type: <the type it closes> }
```

No additional scalar, buffer, handle, or output parameter of any kind. Its only result is the
`ProviderStatus` every function returns.

The reason is architectural, not stylistic. **MIR's `Drop(place)` terminator supplies only the
resource being dropped.** There is no argument list at a drop site and no place to put one — the
drop is implicit in the program's control flow, not a call the programmer wrote. A close function
with a second parameter is therefore one the generated code *cannot call*: every additional
argument would have to be invented by the backend out of nothing.

The consequence is a design rule, not merely a validation rule: **any flush option, completion
mode, or other fallible operation needing arguments must be a separate, explicitly invoked
provider function, called before Drop.**

### 13.2 Close failure *(added, CD-054)*

A close function's nonzero `ProviderStatus` **cannot become a recoverable `Result::Err`**, because
the `Drop` terminator has no result destination. Treat it as a distinct **fatal
provider-close/host failure**:

- **abort without unwinding** (§7.7 of `WP-C5-ENTRY.md`);
- **do not retry close** — a handle's state after a failed close is undefined by this ABI;
- **consider the handle consumed** regardless (§8's consumed-handle error rule applies to close
  like any other consuming call);
- **do not run further pending Drop glue** — the same rule a trap already follows, for the same
  reason: once a close has failed, the runtime's resource invariants are no longer trustworthy, so
  continuing to close *other* resources is guesswork.

Explicit recoverable work — flushing, committing, syncing, anything whose failure the program
should handle — must be a separate provider operation performed **before** close, where its status
has somewhere to go. This is §13.1's conclusion from the other direction: close takes no arguments
*and* returns nothing recoverable, so anything needing either is not close.

## 14. Blocking declaration (§10.1 point 13)

`FunctionDecl.may_block: bool` (§6). Core v1 has no concurrency or async (charter non-goals), so
v0.1 does not schedule around this field — it exists so a function's blocking behavior is a
declared fact from day one rather than retrofitted when a concurrency extension needs to know
which calls are safe to run without stalling everything else.

## 15. Callback prohibition (§10.1 points 14 and 16)

**No provider function in v0.1 may receive a function pointer.** `AbiParam` (§6) has no
function-pointer variant, so this is enforced structurally, the same way §10 is: there is no form
in the closed parameter vocabulary that could name a callback. §17's validator additionally rejects a raw
`FunctionDecl` construction that tries to smuggle one in outside the type system (defense in
depth, not the primary enforcement). Because callbacks are categorically absent, "no concurrent
callbacks" (§10.1 point 16) holds vacuously in v0.1 — restated here as a standing invariant for
whichever future gate first considers calls back into STARK, not because v0.1 does anything active
to schedule around concurrent callbacks that cannot occur.

## 16. Compiler/runtime/provider compatibility checks (§10.1 point 15)

Three checks, all before any provider function is ever called, all fatal on mismatch (never a
silent downgrade):

1. `ProviderIdentity.abi_version == "0.1"` (§2) — this document's own version.
2. The build's target triple is in the provider's `target_triples` (§4).
3. The provider's declared `capabilities` (§5) are a subset of what the STARK program's
   dependency graph actually requires — an unclaimed capability used by the program is rejected
   before backend invocation, the same "reject before backend invocation" discipline
   `WP-C5-ENTRY.md` §3.2 already applies to every other deferred feature.

This is deliberately independent of `stark_runtime::version::check` (`WP-C5-ENTRY.md` §9.2): a
provider version mismatch and a runtime version mismatch are different failure classes with
different likely causes (a stale provider binary vs. a stale toolchain install) and should not be
collapsed into one diagnostic.

## 17. C5 implementation boundary: compile-time validator and mock fixtures (§10.2) *(amended, CD-054)*

Delivered in this WP, no provider feature expansion beyond it:

- `starkc/src/backend/provider_abi.rs` — `ProviderMetadata`, `FunctionDecl`, `AbiParam`,
  `ScalarTy`, and `validate(&ProviderMetadata) -> Result<(), Vec<AbiViolation>>`, checking every
  mechanically checkable rule above: §2's `abi_version`, §4's non-empty triples, §5's non-empty
  capabilities and function-reachability, §6's every-handle-names-a-declared-resource-type, §13's
  exactly-one-close-per-resource-type, and §13.1's close-function shape. §10's no-aggregate rule,
  §11's fixed return, and §15's callback exclusion need no check at all — they are unrepresentable
  in `AbiParam`/`FunctionDecl`. A **compile-time** check: it validates a provider's declared
  metadata, not a running provider.
- `stark-runtime/src/provider_abi.rs` — the `#[repr(C)]` type definitions from §7, §9, and §11
  (`RawResourceHandle`, `BorrowedBuffer`, `BorrowedBufferMut`, `ProviderStatus`), the owning
  `OwnedResourceHandle`, and §6.1's boundary helpers (`from_raw_checked`, `as_raw`, `into_raw`) —
  the only sanctioned raw↔owned conversions. No `extern "C"` linkage, dynamic loading, or
  invocation logic — the owning later package's job.
- Mock fixtures (`starkc/src/backend/provider_abi.rs` test module): one valid illustrative
  provider (`example-kv`, a fictional key-value store — not a committed capability and not tied to
  any real STARK stdlib type):

  | Function | Parameters | `is_close_for` |
  |---|---|---|
  | `kv_open` | `BufferIn` (path), `HandleOut { KvStore }` | — |
  | `kv_get` | `HandleBorrowed { KvStore }`, `BufferIn` (key), `BufferInOut` (value), `ScalarOut(U64)` (bytes written) | — |
  | `kv_set` | `HandleBorrowed { KvStore }`, `BufferIn` (key), `BufferIn` (value) | — |
  | `kv_close` | `HandleConsumed { KvStore }` | `Some(KvStore)` |

  plus one deliberately invalid fixture per violation class (missing close function; a capability
  with no reachable function; a handle naming an undeclared resource type; and one per
  close-shape problem — an extra parameter, an added output, a borrowed rather than consumed
  handle, and a consumed handle of the wrong resource type), proving the validator catches each.

  **The pre-amendment fixture violated the contract it was the positive example for**, which is
  how CD-052's review found the return-shape contradiction: `kv_open` returned a `ResourceHandle`
  directly (§11 forbids any direct return), `kv_get` had nowhere to put the value it retrieved,
  and `kv_get`/`kv_set` consumed the store they operated on — so the document's own example
  provider was unusable after a single read. A validator whose canonical positive fixture violates
  its own contract is not enforcing that contract.

## 18. Definition of done for this document

A later implementation session (real provider execution, whichever package owns it) can answer,
without inventing policy:

- What identity/version/origin/integrity data must a provider declare, and when is it checked?
- What are the two forms a native resource handle takes across the boundary (§7's raw/owning
  split), and what may generated code do with each — including what it may *not* do (copy it,
  read its fields, rely on a Rust destructor)?
- What are the only two buffer shapes, and what is their lifetime contract?
- How does a provider report success vs. failure (§11's fixed `ProviderStatus` return plus
  explicit output channels), when is an output slot valid (§11.1), and how is a provider error
  distinguished from a STARK trap or a host failure?
- What must be true about a resource type's close function — its exactly-one-consumed-handle
  shape (§13.1) and what its own failure means (§13.2) — and what happens if that discipline is
  violated?
- Why can a provider never receive a callback in v0.1, and is that a permanent rule or a v0.1
  scope limit?
- What three checks gate a provider before its first function is ever called?

If any of those still requires an architectural guess, this document is not ready for CE4 approval
as drafted.
