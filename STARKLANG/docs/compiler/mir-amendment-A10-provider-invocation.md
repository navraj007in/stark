# MIR v0.1 Amendment A10 — Provider Invocation as a Distinct Call Form

Status: **APPROVED under CE3 as an additive MIR v0.1 amendment, runtime surface `0.1-A10`**
(owner decision, 2026-07-28). Implementation of WP-C7.8.2 may begin against this document.

Scope class: **narrow additive amendment to MIR v0.1** (`mir.md`, APPROVED CD-028, amended
CD-029). It adds **one `Callee` variant** and the validated record it resolves to. It adds **no
`RuntimeFn` members**, changes no existing variant, and alters no MIR shape version.

**This amendment is separate from A1 deliberately.** Surfaces `0.1-A1` through `0.1-A9` are
revisions of `mir-amendment-A1-strings-runtime.md`, each enumerating additional `RuntimeFn`
members in a versioned appendix table. A10 is not that. It introduces a **distinct invocation
category** with its own trust and verification model, which does not belong in a document whose
subject is the string/collection runtime surface. A1's header and §11 revision log record the
counter advancing to `0.1-A10` here, so the surface version stays traceable from where readers
look for it.

Governing decisions: `WP-C7.8.1-DECISION-PACKETS.md` Packet 2 (CE3, this amendment) and Packet 1
(CE4, the invocation model this serves).

---

## 1. Why a distinct call form rather than more `RuntimeFn`

`RuntimeFn` and provider calls have materially different trust and verification models, and the
difference is structural rather than stylistic.

A **runtime intrinsic** is compiler-owned:

- its identity is closed and known to the compiler;
- its semantics are defined by the compiler/runtime contract;
- it requires no provider selection;
- it relies on no external metadata;
- target compatibility is implicit in the runtime implementation.

A **provider call** is externally declared, and carries:

- provider identity;
- capability identity;
- a validated function declaration;
- target applicability;
- ABI parameter and output-slot shape;
- ownership-transfer rules;
- borrowed-buffer constraints;
- resource-type identity;
- a declared recoverable status vocabulary;
- failure-channel rules.

Encoding provider calls as `RuntimeFn` would either erase those distinctions or force provider
metadata into `RuntimeFn` indirectly, at which point the distinction exists implicitly and less
safely. A dedicated callee variant is the clearer and safer representation.

**`RuntimeFn` remains reserved for compiler-owned runtime operations.** A10 adds nothing to it.

## 2. The addition

`Callee` (`starkc/src/mir/mod.rs`) gains a fourth variant. The existing three are unchanged:

```rust
pub enum Callee {
    Instance(Instance),
    /// Indirect call through a `FnPtr`-typed operand (CD-021/CD-027).
    FnValue(Operand),
    Runtime(RuntimeFn),
    /// A10: resolves to a validated provider `FunctionDecl` carrying the full ABI contract.
    Provider(ProviderCallId),
}
```

**Note on the approving sketch.** The disposition's illustrative shape wrote `Function(FunctionId)`
and omitted `FnValue`. Both are artefacts of the sketch, not directives: `Instance(Instance)` is
this repository's name for the direct-call form, and `FnValue` carries function values closed under
CD-021/CD-027 and WP-C5.4. A10 is purely additive — it removes and renames nothing.

`ProviderCallId` must identify a **compiler-owned validated record, never a string**:

```rust
pub struct ValidatedProviderCall {
    pub provider_id: ProviderId,
    pub capability_id: CapabilityId,
    pub function: FunctionDecl,
    pub target: TargetTriple,
    pub status_binding: StatusBindingId,
}
```

Field names may follow repository conventions, but this semantic content must remain available to
both MIR verification and backend emission. A bare symbol would be insufficient: every invariant in
§4 is checked *against the declaration*, and a call site that carried only a name would force the
verifier to reconstruct the contract it is supposed to be checking.

## 3. Binding sequence

Provider calls are resolved **before MIR verification**, never during backend emission:

```text
package operation
→ capability requirement
→ provider selection for target
→ metadata validation
→ FunctionDecl resolution
→ ValidatedProviderCall allocation
→ Callee::Provider(ProviderCallId)
```

**The backend must never perform first-time provider selection or interpret unvalidated metadata.**
By the time emission runs, selection has happened, metadata has been validated, and the call form
already names a record that satisfies every §4 invariant. This is the same "reject before backend
invocation" discipline `WP-C5-ENTRY.md` §3.2 applies to every other deferred feature.

## 4. Verifier invariants (binding)

1. **Provider existence** — the referenced provider is part of the selected provider set.
2. **Target applicability** — the provider metadata admits the compilation target.
3. **Function membership** — the function belongs to the selected provider and capability.
4. **Verbatim symbol validity** — the symbol was validated and is emitted unchanged.
5. **Input contract** — MIR argument types, mutability, ownership and ABI shapes match
   `FunctionDecl`.
6. **Borrow validity** — borrowed strings and buffers remain live and stable for the complete call
   (ABI §9's call-duration lifetime).
7. **Consumed-resource invalidation** — a consumed handle becomes unusable immediately, on every
   path, per ABI §8's consumed-handle error rule.
8. **Output-slot discipline** — output values are read only after success; failure paths never
   inspect uninitialised output storage (ABI §11.1).
9. **Result and channel discipline** — success, declared recoverable status, undeclared status,
   contract violation and host failure follow their assigned channels and **cannot collapse into
   each other**.

**Additionally:** a resource-producing call must validate `resource_type` before constructing an
owned STARK resource (ABI §11.1).

## 5. Status handling

The Packet 1 §1.2 rule is binding on both the verifier and the generated binding:

```text
status == success
    → validate and read outputs

status is declared by the package binding
    → construct channel-one Result::Err

status is undeclared
    → provider contract violation
```

**An undeclared status must not fall through to `IOError::Other`, `NetworkError::Other`, or any
other generic package error.** That fallthrough is the failure mode where a provider and its
package drift apart while remaining physically ABI-compatible: nothing crashes, and the meaning
quietly changes.

## 6. Backend emission

For the static first-party path (Packet 1, option B), `Callee::Provider` lowers to a direct
`extern "C"` declaration and call using the validated `FunctionDecl.name` **verbatim**.

The backend may generate Rust helper code. That helper must **not**:

- rename or sanitise the symbol;
- catch panics;
- reinterpret unknown status codes as recoverable;
- read output slots before a successful status;
- create Rust-owned destruction semantics for ABI resources;
- bypass the approved boundary helpers
  (`OwnedResourceHandle::from_raw_checked`/`as_raw`/`into_raw`).

The panic prohibition is not merely a rule to follow: the generated workspace already sets
`panic = "abort"` in both profiles (`starkc/src/backend/generated_rust/build.rs:298`, `:314`), so a
`catch_unwind` here would misclassify a provider defect as recoverable, split panic semantics
within one workspace, and obscure a guarantee that already holds.

## 7. Surface version

`MIR_RUNTIME_SURFACE` advances `0.1-A9` → `0.1-A10` (`starkc/src/mir/mod.rs:44`).

A10 records both the syntax-level addition and the semantic contract:

```text
A10:
- adds Callee::Provider(ProviderCallId);
- ProviderCallId resolves to validated provider metadata;
- provider calls are target-resolved before verification;
- verifier enforces ABI ownership, output-slot, resource-type and failure-channel rules;
- generated-Rust backend emits direct static extern "C" calls;
- RuntimeFn remains reserved for compiler-owned runtime operations.
```

**A10 is not "several new `RuntimeFn` members".** The consequential change is the new invocation
category. A consumer pinned to `0.1-A9` must reject a `0.1-A10` artifact loudly under V-RT-1, which
is the intended behaviour of a versioned surface rather than a regression.

## 8. Required regression evidence

Before WP-C7.8.2 closes:

- a valid `stark-time` provider call represented as `Callee::Provider`;
- generated code calls the exact metadata symbol;
- target-incompatible provider rejected **before backend invocation**;
- duplicate provider selection rejected;
- function not belonging to the selected provider rejected;
- malformed or non-prefixed symbol rejected **without sanitisation**;
- consumed handle reused after a provider call rejected;
- borrowed input invalidated too early rejected;
- output slot read on non-success rejected;
- incorrect returned `resource_type` classified as **contract violation**;
- undeclared nonzero status classified as **contract violation**;
- declared nonzero status converted to the package error;
- provider close failure classified as **host failure**;
- **no provider call represented as `RuntimeFn`**;
- the mechanism-independent conformance suite passes against the static path;
- a `0.1-A9`-pinned consumer rejects a `0.1-A10` artifact with the V-RT-1 diagnostic.

## 9. Explicitly out of A10

This amendment does not disposition, and implementation must not prejudge:

- Packet 4's Core-versus-package API placement (CE1, OPEN);
- Packet 5's trust-boundary policy (CE9, OPEN);
- dynamic provider loading;
- HTTP;
- package-specific status vocabularies.

A10 provides the MIR and backend structure WP-C7.8.2 needs to begin, and nothing beyond it.

## 10. Implementation status

| Slice | State |
| --- | --- |
| C7.8.2a — MIR representation and versioning | **LANDED** |
| C7.8.2b — provider resolution and validated call records | next |
| C7.8.2c — verifier invariants and negative fixtures | pending |
| C7.8.2d — generated-Rust static extern bindings | pending |
| C7.8.2e — `stark-time` end-to-end execution | pending |
| C7.8.2f — full A10 regression evidence | pending |

**What C7.8.2a landed.** `Callee::Provider(ProviderCallId)`, `ProviderCallId`,
`ValidatedProviderCall`, and the program-level `MirProgram::provider_calls` arena;
`MIR_RUNTIME_SURFACE` at `0.1-A10`; the dump form `provider:<provider>:<symbol>`; and a defined
refusal at every consumer that cannot yet honour a provider call — verification (MIR-0020), the
MIR interpreter, and generated-Rust emission. Evidence: `starkc/tests/a10_provider_call.rs`
(9 cases).

**A provider call is representable but not yet admitted.** That is the intended C7.8.2a state:
§4's invariants land in C7.8.2c, and until they do, verification refuses rather than accepting an
unchecked contract. A dangling `ProviderCallId` reports separately (MIR-0019) so an
arena-construction defect cannot hide behind the blanket refusal that C7.8.2c removes.

**One layering change was required.** The ABI declaration types moved from
`starkc/src/backend/provider_abi.rs` to the crate root (`starkc/src/provider_abi.rs`), because
`ValidatedProviderCall` puts a `FunctionDecl` inside MIR and `crate::mir` is deliberately
backend-independent. `backend::provider_abi` remains as a re-export shim, so
`stark-time/native/src/lib.rs` — which compiles against `starkc::backend::provider_abi::*` — needs
**no edit at all**, which is the stronger form of Packet 1's exit condition.

**The MIR interpreter will never execute provider calls.** It is a pure semantic oracle with no
provider linked into it, so a host call has no meaning it could reproduce. Providers run only on
the native path, which makes differential comparison for provider-backed programs native-only by
construction rather than by omission.

## 11. Revision log

**Rev. 1 (2026-07-28) — A10 approved under CE3.** Adds `Callee::Provider(ProviderCallId)` and
`ValidatedProviderCall`; nine binding verifier invariants plus the `resource_type` rule; the
pre-verification binding sequence; backend emission prohibitions; `MIR_RUNTIME_SURFACE` →
`0.1-A10`. No `RuntimeFn` additions. No MIR shape version change.

**Rev. 2 (2026-07-28) — C7.8.2a landed.** §10 records implementation state, the
representable-but-not-admitted position, the `provider_abi` layering move and its shim, and the
interpreter's permanent exclusion. No contract change.
