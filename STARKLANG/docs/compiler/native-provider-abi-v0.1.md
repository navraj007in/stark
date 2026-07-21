# STARK Native Provider ABI v0.1

**Status: APPROVED (CE4, 2026-07-21, CD-046) — approved as drafted, no changes.** Drafted
2026-07-21 under WP-C5.1c; CD-042 had approved *writing* this document as part of
`WP-C5-ENTRY.md`'s recommended §19 choices ("Provider ABI | Specify v0.1 in C5.1; execution not
required for MVP"), and the owner's CD-046 review approved this document's actual technical
content as drafted, with no required changes. WP-C5.1 is closed.

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

## 6. Exported function table (§10.1 point 5)

```text
FunctionDecl {
    name: str
    capability: str            // which §5 capability this function belongs to
    params: [AbiType]
    returns: AbiType
    is_close_for: Option<str>  // Some(resource_type_name) if this function is THE close
                                // function for that resource type (§12); at most one function
                                // per resource type may set this
    may_block: bool             // §13
}
```

`AbiType` (§7-9) is a **closed** enum with no variant for an internal generated-Rust aggregate —
see §10, which is enforced structurally by this closure rather than by a separate runtime check.

## 7. Opaque resource-handle representation (§10.1 point 6)

```rust
#[repr(C)]
pub struct ResourceHandle {
    pub id: u64,
    pub resource_type: u32,   // a compiler-assigned index into the provider's declared
                               // resource-type list, not a pointer or provider-chosen tag
}
```

A handle is opaque to generated Rust: STARK code (and the generated Rust that implements it) may
copy it, pass it, store it, and drop it, but may never read `id`/`resource_type` or construct one
except as a provider function's return value. This is the ABI's only cross-boundary
resource-carrying type (§10).

## 8. Ownership transfer in both directions (§10.1 point 7)

Two directions, never shared/aliased ownership:

- **Into the provider**: a `ResourceHandle` or buffer (§9) passed *by value* into a provider
  function transfers ownership; STARK's generated code must not use it again afterward (the same
  discipline as an ordinary STARK move, §7.3 of `WP-C5-ENTRY.md`).
- **Out of the provider**: a provider function that returns a `ResourceHandle` hands STARK a
  freshly-owned resource; STARK is now solely responsible for eventually calling that resource
  type's `is_close_for` function (§12) exactly once.

No function signature in v0.1 may take or return a *borrowed* `ResourceHandle` — only full
ownership transfer. A borrowed-handle convention is deferred to whichever later package first
needs one (e.g. "peek without closing"), since designing it without a real use case risks guessing
wrong.

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

`AbiType` (§6) is closed over exactly: the primitive scalar widths (`U8`/`U16`/.../`I64`/`Bool`/
`F32`/`F64`), `ResourceHandle`, `BorrowedBuffer`, `BorrowedBufferMut`, and `ProviderStatus` (§11).
There is no `AbiType::Aggregate(MirTy)` or equivalent escape hatch. A provider function that needs
structured data decomposes it into these primitives and buffers explicitly — the same discipline
`WP-C5-ENTRY.md` §6.3 already applies internally ("Rust layout is backend-, target-, and
toolchain-version dependent... not a stable ABI"): a generated `#[repr(Rust)]` struct's layout is
not something a separately-compiled provider may assume, so it must never cross this boundary.

## 11. Error return representation (§10.1 point 10)

Every provider function returns `ProviderStatus`, never a value directly; results are written
through an out-parameter (`BorrowedBufferMut` or a `*mut` scalar/`ResourceHandle` slot passed by
the caller):

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

## 12. Distinction among provider error, STARK trap, and host failure (§10.1 point 11)

Three channels, never conflated:

| Channel | Meaning | STARK-observable as |
|---|---|---|
| Provider error (`ProviderStatus.code != 0`) | An ordinary, expected failure the provider itself defines (file not found, permission denied) | `Result::Err` — recoverable STARK code handles it like any other error |
| STARK trap | The **provider violated this contract** (returned a `ResourceHandle` for an undeclared resource type, wrote past a `BorrowedBufferMut`'s `len`, etc.) | An abort, same as any other MIR trap (`WP-C5-ENTRY.md` §7.7) — never recoverable, because a contract violation means the runtime's invariants can no longer be trusted |
| Host failure | An environment-level condition neither STARK nor the provider caused (OS resource exhaustion, the provider's own process/library aborting) | Also an abort, but classified distinctly in diagnostics (§12.5's build-time analogue: never presented as an ordinary user-source error) |

Only the compile-time validator (§17) exists in v0.1 scope; the runtime-side enforcement of the
trap/host-failure rows (detecting a contract violation at call time) is real provider-execution
work, explicitly deferred (see the scope boundary above).

## 13. Resource close/Drop semantics (§10.1 point 12)

Every `resource_type` a provider declares must have **exactly one** `FunctionDecl` with
`is_close_for: Some(that_type)` (checked by §17's validator, not left to convention). STARK's MIR
`Drop` terminator on a provider-resource-typed local calls that function exactly once — the same
"exactly once, never repeated automatically" invariant `WP-C5-ENTRY.md` §7.5 already requires for
ordinary STARK Drop, extended across the provider boundary rather than given a second set of
rules. Calling a resource's close function on an already-closed handle, or on a handle from a
different resource type, is a contract violation — a STARK trap (§12's middle row), not a provider
error.

## 14. Blocking declaration (§10.1 point 13)

`FunctionDecl.may_block: bool` (§6). Core v1 has no concurrency or async (charter non-goals), so
v0.1 does not schedule around this field — it exists so a function's blocking behavior is a
declared fact from day one rather than retrofitted when a concurrency extension needs to know
which calls are safe to run without stalling everything else.

## 15. Callback prohibition (§10.1 points 14 and 16)

**No provider function in v0.1 may receive a function pointer.** `AbiType` (§6) has no
function-pointer variant, so this is enforced structurally, the same way §10 is: there is no type
in the closed vocabulary that could name a callback. §17's validator additionally rejects a raw
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

## 17. C5 implementation boundary: compile-time validator and mock fixtures (§10.2)

Delivered in this WP, no provider feature expansion beyond it:

- `starkc/src/backend/provider_abi.rs` — `ProviderMetadata`, `FunctionDecl`, `AbiType`, and
  `validate(&ProviderMetadata) -> Result<(), Vec<AbiViolation>>`, checking every mechanically
  checkable rule above (§2's `abi_version`, §4's non-empty triples, §5's non-empty capabilities
  and function-reachability, §13's exactly-one-close-per-resource-type, §15's callback exclusion).
  A **compiler-side, compile-time** check — it validates a provider's declared metadata, not a
  running provider.
- `stark-runtime/src/provider_abi.rs` — the `#[repr(C)]` type definitions from §7, §9, and §11
  (`ResourceHandle`, `BorrowedBuffer`, `BorrowedBufferMut`, `ProviderStatus`) that a real
  provider's implementation and a generated binary would both compile against. No `extern "C"`
  linkage, dynamic loading, or invocation logic — those are the owning later package's job.
- Mock fixtures (`starkc/src/backend/provider_abi.rs` test module): one valid illustrative
  provider (`example-kv`, a fictional key-value store — not a committed capability and not tied to
  any real STARK stdlib type) exercising a resource handle, a paired open/close pair, a
  `BorrowedBuffer`-taking function, and one deliberately invalid fixture per violation class
  (missing close function; a capability with no reachable function), proving the validator catches
  each.

## 18. Definition of done for this document

A later implementation session (real provider execution, whichever package owns it) can answer,
without inventing policy:

- What identity/version/origin/integrity data must a provider declare, and when is it checked?
- What is the only type that may name a native resource crossing the boundary, and what may
  generated code do with it?
- What are the only two buffer shapes, and what is their lifetime contract?
- How does a provider report success vs. failure, and how is that distinguished from a STARK trap
  or a host failure?
- What must be true about a resource type's close function, and what happens if that discipline is
  violated?
- Why can a provider never receive a callback in v0.1, and is that a permanent rule or a v0.1
  scope limit?
- What three checks gate a provider before its first function is ever called?

If any of those still requires an architectural guess, this document is not ready for CE4 approval
as drafted.
