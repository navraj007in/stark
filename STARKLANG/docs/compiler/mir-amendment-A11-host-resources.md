# MIR Amendment A11 — a representation for package-declared host resources

**Status:** **APPROVED under CE3, 2026-07-29 (CD-224)** — `MIR_VERSION` → `0.2`, subject to §9's
requalification. Implementation may proceed.
**Governs:** Packet 6 / Route B (CD-220).
**Scope class:** **MIR shape change**, not a runtime-surface revision. §1 argues why, and the
distinction decides everything else in this document.

---

## 1. This is not an A10-shaped change

A10 added `Callee::Provider` and bumped `MIR_RUNTIME_SURFACE` `0.1-A9` → `0.1-A10`, leaving
`MIR_VERSION` at `0.1`. **That precedent does not carry**, and the difference is structural rather
than a matter of degree.

A `Callee` variant appears in exactly one place: a `Terminator::Call` payload. A consumer that does
not know it fails to match on one enum, at one site, and can be made to fail loudly there.

**A `MirTy` variant is different in kind.** Types flow through every part of MIR: local
declarations, projections, operand typing, aggregate construction, `TypeContext`, layout queries,
`copy_types`, drop planning, the textual dump, and every backend's type emission. A consumer meeting
an unknown `MirTy` has no single site to fail at — it has as many sites as it has type logic, and
most of them will do something plausible rather than refuse.

The set of representable **types** is the representation contract. Changing it is a shape change.

### Disposition asked for (Q1)

> **Increment `MIR_VERSION` to `0.2`.** A11 is not backward-compatible within `0.1`.
>
> **APPROVED (CD-224)**, conditional on §9's bounded requalification and on §3's rule that closed
> gate evidence is not rewritten.

The alternative — keeping `0.1` and relying on the runtime surface — is rejected because a
`0.1`-era consumer would find the version string it expects, accept the artifact, and then meet a
type it cannot represent. That is exactly the failure mode versioning exists to prevent, and it
would be silent in the consumers most likely to hit it.

---

## 2. How older consumers reject (Q2, Q3)

**Rejection is on `MIR_VERSION`, and the mechanism already exists.** `verify_program`'s first act
is the §A1 gate:

```rust
if program.mir_version != MIR_VERSION || program.runtime_surface != MIR_RUNTIME_SURFACE {
    return Err(vec![MirError { code: "MIR-0017", .. }]);   // before consuming any body
}
```

It is an **equality** check, not a range, so a `0.1` build rejects a `0.2` artifact and a `0.2`
build rejects a `0.1` one. Nothing new is needed for Q2; the gate needs the constant to move, which
is Q1.

**Version, not surface, and not both (Q3).** `MIR_RUNTIME_SURFACE` stays at `0.1-A10`:

- A11 adds **no `RuntimeFn` member**. A host resource's close is a provider call reached through
  MIR's `Drop` terminator (ABI §13), not a runtime operation.
- Bumping the surface as well would misreport the change's nature to anyone reading the two
  constants, implying new runtime operations that do not exist.

The one exception to state: if implementation discovers a runtime helper is genuinely required, the
surface bumps **too**, and this document is revised rather than the helper being added quietly.

---

## 3. What must change with the version (Q4)

`mir_version` is not decorative — it is threaded into the build's identity. Every one of these
follows from a `0.1` → `0.2` increment and must land in the same change:

| Surface | Effect |
| --- | --- |
| `MIR_VERSION` (`mir/mod.rs`) | `"0.1"` → `"0.2"` |
| Build key (`build.rs`, `mir={}`) | every key changes; **all cached builds invalidate** — correct, since the representation changed |
| Artifact contract (`build.json`'s `mir_version`) | new value recorded per artifact |
| `BuildVersions` / installed-runtime check (`backend/version.rs`, `stark_runtime::version::check`) | a runtime built against `0.1` must refuse a `0.2` binary; this is the check's purpose and it needs no new logic |
| MIR dump snapshots | any `.snap` recording a version header re-pins |
| Frozen corpus locks (`tests/exec_snapshots`, `tests/c6-corpus`) | re-pin if they carry the version; verified at implementation, not assumed |
| `mir.md` (CD-028, amended CD-029) | the contract document states the version it defines |
| C6.4 tier-1 evidence | **not** rewritten — see §9 |

**The cache invalidation is a feature, not a cost.** A build key that ignored a representation
change would serve a cached artifact produced under different type rules.

---

## 4. The form

```rust
pub enum MirTy {
    // … existing variants unchanged …
    /// A11: a host resource. Established by the COMPILER for a Core resource (`File`) or by a
    /// PACKAGE declaration for a package resource (`TcpStream`) -- one representation, two
    /// authorities (CD-224).
    HostResource {
        /// The STARK nominal this resource is, e.g. the item for `TcpListener`.
        nominal: crate::hir::ItemId,
        /// §2 identity of the provider that owns the resource type.
        provider: String,
        /// §13 resource-type name as that provider declares it, e.g. `"tcp_stream"`.
        resource: String,
    },
}
```

### One representation, two authorities (CD-224)

A **Core** resource and a **package** resource both lower to this form. What differs is who may
establish the binding, and that difference lives at resolution, not in the type:

| | Core resource | Package resource |
| --- | --- | --- |
| authority | compiler / specification | package declaration |
| declared in | `ResourceRegistry::builtin()` | `provider_api.resources` |
| `nominal` resolves to | the Core item (`File`) | the package item (`TcpStream`) |
| a package may redefine it | **no** | yes |

`file → CoreType::File` therefore stays compiler-owned and undeclarable by any package, while
`tcp_listener`/`tcp_stream` are package-declared. Both produce a `HostResource` whose `nominal` names
the type the programmer wrote. Packet 4 is preserved on both sides: Core keeps authority over its
types, packages keep authority over theirs.

**Implementation note.** `ResourceRegistry` currently maps a resource name to a `MirTy` directly.
Under A11 it maps a resource name to a **nominal identity**, and the `HostResource` is constructed at
resolution, when the provider is known — the same point that already produces `ValidatedProviderCall`.
A registry entry cannot carry a provider, because the provider is a property of the build.

**Both identities are retained, as Packet 6 requires.** `nominal` is what diagnostics and the source
language talk about; `provider`/`resource` is what the ABI talks about. Neither can be derived from
the other, and dropping either loses something a later stage needs.

### Canonical, deterministic identity (Q5)

Ordering and equality are **structural over the three fields**, in declaration order
(`nominal`, `provider`, `resource`), consistent with every other `MirTy` variant's derived `Ord`.

The canonical **rendering**, for dumps, diagnostics and any content hash, is:

```text
hostres#<provider>/<resource>@<nominal-canonical-path>
```

`<nominal-canonical-path>` is the same content path `key_symbol` already produces for nominals
(`struct#liba::A`), **not** the `ItemId` index — CD-108 established that ordering-dependent indices
must not reach canonical identity, and a host resource is no exception. `<provider>` and
`<resource>` are the validated metadata strings verbatim; they are already constrained to be
non-empty by ABI §5/§13 validation.

Two different nominals bound to the same provider resource render differently, and the same nominal
bound through different providers renders differently. Both are deliberate: §7's negative cases turn
on being able to tell those apart.

### Nominal identity in diagnostics and codegen (Q6)

- **Diagnostics** resolve `nominal` to its source name, so a message says
  `expected TcpStream, found TcpListener` rather than naming a provider resource string the
  programmer never wrote.
- **Codegen type selection** does *not* consult `nominal`: every `HostResource`, whatever its
  nominal, emits as `stark_runtime::provider_abi::OwnedResourceHandle`. The nominal distinction is
  a **STARK type-system** distinction, enforced before emission; at the ABI boundary all handles
  are the same shape, and their runtime distinction is the `resource_type` field validated by
  `from_raw_checked`.

That split is the point: **static distinctness in STARK, dynamic validation at the boundary.**

---

## 5. Drop planning and the close function (Q7)

A `HostResource` local's `Drop` terminator must call that resource's close, exactly once.

**The close is selected from validated metadata, at resolution time, not at drop time.** For each
`HostResource` a program uses, resolution records a `ValidatedProviderClose`:

```rust
pub struct ValidatedProviderClose {
    pub resource: MirTy,             // the HostResource form
    pub close: ProviderCallId,       // the is_close_for function, already validated
}
```

interned in the same program-level arena as provider calls. `drop_plan` then looks the resource up
and emits a provider call to that id, rather than searching metadata during lowering.

Rules, all enforceable before emission:

1. every `HostResource` reachable in a body has exactly one recorded close (ABI §13's
   exactly-one-close-per-resource-type, checked at resolution);
2. the recorded close's declaration is `is_close_for: Some(resource)` for **that** resource name;
3. its parameter list is exactly one `HandleConsumed` of that resource (ABI §13.1);
4. no other call site may invoke a close directly — a package cannot bind one
   (`WP-C7.8.8-PACKAGE-API-DESIGN.md` §2), and the verifier rejects a `Callee::Provider` whose
   declaration is `is_close_for` outside a `Drop` lowering.

Rule 4 is what keeps "exactly once" true: MIR owns the only path.

---

## 6. What a host resource is not (Q8)

A `HostResource` is opaque. Each of these is a verifier rule with its own rejection, not a
convention:

| Forbidden | Rule |
| --- | --- |
| structural construction | `Rvalue::Aggregate` whose type is a `HostResource` — there are no fields to construct from |
| field projection | `Projection::Field`/`VariantField` on a `HostResource` base |
| `Copy` | never admitted to `TypeContext::copy_types`; `Operand::Copy` of one is rejected |
| `Clone` | no `Clone` impl may be recorded for its nominal |
| ordinary struct emission | emission takes the `HostResource` arm; the struct path is unreachable for it, and `emit_nominal_definitions` emits **no** Rust `struct` for a host-resource nominal |
| Rust `Drop` | the emitted type is `OwnedResourceHandle`, which deliberately has none (CE4 Amendment 1) |

The last two are why Route C was rejected: as an ordinary `Struct` every one of these would be a
check someone had to remember, and the default behaviour — emit a struct with fields and derives —
is exactly wrong.

---

## 7. Required regression evidence

- A `0.1`-pinned consumer rejects a `0.2` artifact with MIR-0017, **before** any body is consumed.
- A `0.2` build rejects a `0.1` artifact symmetrically.
- Build keys differ across the version change for an otherwise identical program.
- The canonical rendering is stable across clean rebuild, relocation and dependency reordering, and
  contains no `ItemId` index.
- Two nominals on one provider resource, and one nominal through two providers, are distinguishable
  by canonical identity.
- A `HostResource` local's drop emits the recorded close, exactly once, on every exit path.
- A `Callee::Provider` naming an `is_close_for` function outside a `Drop` lowering is rejected.
- Aggregate construction, field projection, `Operand::Copy` and a recorded `Clone` impl are each
  rejected with their own diagnostic.
- `emit_nominal_definitions` emits no Rust struct for a host-resource nominal, and the emitted type
  is `OwnedResourceHandle` with no `Drop` impl.
- An ordinary `Struct` nominal is entirely unaffected: still constructed, projected, copied where
  `Copy`, and emitted as a struct.

---

## 8. Open for the owner

**8.1 and 8.2 are RESOLVED by CD-224** — the increment is approved and §9 records the
immutable-evidence rule. **8.3 remains open.**

**8.3** §5's `ValidatedProviderClose` adds a second program-level arena. It could instead be a field
on the `HostResource` form — rejected here because a `ProviderCallId` inside a *type* would make two
structurally identical resources compare unequal when resolved in different programs. Confirm the
arena.

---

## 9. Historical evidence and requalification (CD-224)

> **Historical gate evidence remains immutable and valid for the version and commit it records. A
> representation-contract version increment does not retroactively reopen the gate, but current
> compiler claims that rely on the changed representation must be requalified under the new
> version.**

**Closed C6 evidence is not rewritten, regenerated, or reinterpreted as though produced under
`0.2`.** Those records are tied to their commit, compiler version and the MIR `0.1` contract, and
they remain valid claims about the closed gate at that point in time.

**A version bump alone is not a gate-reopening condition.** Gate C6 reopens only if the
non-regression run below finds an actual regression in a C6 *closure claim* — not because the
version string moved.

Required consequences of the increment, all in scope for the implementing slice:

1. every build key includes `mir_version = 0.2`;
2. MIR snapshots and **current** corpus locks are re-pinned;
3. installed-runtime and artifact consumers reject `0.1`/`0.2` mismatches **explicitly**, in both
   directions;
4. serializers and validators recognise the new type form;
5. cache invalidation is **tested**, not assumed;
6. current differential and native regression suites run under `0.2`;
7. a **bounded C6 non-regression suite** confirms the representation change did not alter previously
   admitted ownership, `Drop` and native semantics.

Item 7 is the one that carries the claim. It is bounded — it re-runs C6's ownership/Drop/native
behaviour under `0.2`, not the whole gate — and its purpose is to distinguish "the contract version
moved" from "behaviour moved".
