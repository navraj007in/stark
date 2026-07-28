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
subject is the string/collection runtime surface. A1's header and revision log record the
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

`MIR_RUNTIME_SURFACE` advances `0.1-A9` → `0.1-A10` (`starkc/src/mir/mod.rs`, `MIR_RUNTIME_SURFACE`).

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
| C7.8.2b — provider resolution and validated call records | **LANDED** |
| C7.8.2c — verifier invariants and negative fixtures | **LANDED** (1–5 of 9; 6–9 deferred, see below) |
| C7.8.2d-1 — provider binding plan | **LANDED** |
| C7.8.2d-2 — scalar and buffer emission (closes invariant 6) | **LANDED** |
| C7.8.2d-3 — output and status control flow (closes invariants 8, 9) | **LANDED** |
| C7.8.2d-4 — resource framework, no `File` admission (structures invariant 7) | **LANDED** |
| C7.8.2e — `stark-time` end-to-end execution | next |
| C7.8.2f — full A10 regression evidence | pending |

**What C7.8.2a landed.** `Callee::Provider(ProviderCallId)`, `ProviderCallId`,
`ValidatedProviderCall`, and the program-level `MirProgram::provider_calls` arena;
`MIR_RUNTIME_SURFACE` at `0.1-A10`; the dump form `provider:<provider>:<symbol>`; and a defined
refusal at every consumer that cannot yet honour a provider call — verification (MIR-0020), the
MIR interpreter, and generated-Rust emission. Evidence: `starkc/tests/a10_provider_call.rs`
(9 cases).

**What C7.8.2b landed.** `starkc/src/provider_resolve.rs`: `DeclaredProvider`, `ProviderSet`,
`ResolveError`, and `ProviderCallArena`. `ProviderSet::select` runs §3's sequence — metadata
validation, symbol-grammar validation, target applicability, cross-provider symbol uniqueness, and
exactly-one-supplier-per-capability — returning *every* failure rather than the first.
`ProviderSet::resolve` produces a `ValidatedProviderCall`; the arena interns them, deduplicating so
one function called from many bodies yields one record and one id. Evidence:
`starkc/tests/a10_provider_resolve.rs` (16 cases), including `stark-time`'s real metadata resolving
**unmodified**, ambiguity rejected in **both** declaration orders, and invalid symbols reported
verbatim.

One finding from building it needs an owner decision — recorded in §11.

**What C7.8.2c landed.** Invariants **1–5** are enforced, replacing C7.8.2a's blanket refusal
(MIR-0020, retired and not reused). New codes: MIR-0021 target applicability, MIR-0022 capability
membership, MIR-0023 symbol validity, MIR-0024 a resource type not bound to a MIR type, MIR-0025 ABI
version. Invariant 5 is enforced by *derivation* rather than a separate rule —
`mir::provider_sig` turns the declared `AbiParam` list into the expected MIR signature, and the
verifier's existing arity and per-argument checks do the rest.

The MIR signature mirrors the physical ABI one-for-one rather than inventing a STARK-shaped one:
`dest` receives `ProviderStatus.code` as `UInt32`, and every declared parameter — output slots
included — is an argument, following §6.1's mapping (`ScalarOut(T)`/`ScalarInOut(T)` → `&mut T`,
`BufferIn` → `&[UInt8]`, `BufferInOut` → `&mut [UInt8]`). Converting a status into a STARK
`Result` is the binding layer's job; doing it here would collapse §12's three channels at exactly
the layer that must keep them apart.

`ValidatedProviderCall` gained `provider_target_triples` so invariant 2 is a *check* rather than an
assumption — without the declared list there would be nothing to verify `target_triple` against,
and "re-checks rather than trusts" would have been a comment rather than a rule.

**Invariants 6–9 are deferred to C7.8.2d/e, and cannot be pulled forward.** Borrow validity,
consumed-resource invalidation, output-slot discipline and channel discipline all constrain the
*binding layer's* generated control flow — reads on failure paths, handle liveness after a
consuming call — and no lowering produces that structure yet. C7.8.2d-1 builds the structure they
will be checked against.

**MIR-0024, stated narrowly.** Resource-bearing provider calls are structurally defined but remain
inadmissible until a provider resource type is **bound to a concrete MIR type**. It does not say
MIR cannot support resources. The diagnostic therefore **outlives the empty registry**: once
C7.8.4 registers `"file" → Core(File, [])`, that call is admitted while an unknown
`"custom-db-session"` still reports MIR-0024. Do not delete it when `File` lands.

Evidence: `starkc/tests/a10_provider_verify.rs` (10 cases, each breaking exactly one thing in a
verified baseline) plus the positive case in `a10_provider_call.rs`.

**What C7.8.2d-1 landed.** `starkc/src/provider_bind.rs`: `ProviderBindingPlan`,
`ProviderInputPlan`, `ProviderOutputPlan`, `StatusBinding`, `StatusOutcome`, and
`ResourceRegistry`. Provider emission stops being a single raw call expression, because that shape
has nowhere to put invariants 6, 8 and 9. Evidence: `starkc/tests/a10_provider_bind.rs`.

Three classification rules are load-bearing and were easy to get wrong:

- **`ScalarInOut` and `BufferInOut` are inputs, never output slots.** ABI §11.1 makes them
  caller-initialised and caller-owned, so applying `MaybeUninit` semantics would be wrong in both
  directions — forbidding a legitimate read after failure, and implying the provider allocates
  storage it does not. Only `ScalarOut` and `HandleOut` are uninitialised-until-success.
- **Every declared parameter produces exactly one plan item, carrying its declared index**, so
  interleaved inputs and outputs cannot be reordered by an emitter that walks inputs then outputs.
  `ProviderBindingPlan::covers` holds this as a checkable property rather than a construction
  comment.
- **An empty status vocabulary is legal and precise**: every nonzero status is then a contract
  violation. `stark-time` is exactly this case.

**The resource framework is complete; only the bindings are absent.** `ResourceRegistry` maps a
provider `resource_type` to a `MirTy`, and `HandleConsumed`/`HandleOut` planning works today —
proven with a synthetic resource type, so landing `File` in C7.8.4 is a *registration* rather than
new machinery. `ResourceRegistry::builtin()` is empty in C7.8.2d.

**What C7.8.2d-2 landed.** `starkc/src/backend/generated_rust/emit_provider.rs`: one `extern "C"`
block per program, deduplicated by symbol and **sorted**, plus call-site emission for every
non-resource parameter form. Symbols are emitted verbatim — never through `mangle::sanitize_symbol`,
since a provider symbol is not a MIR instance and the same name must resolve under a future
`dlsym`. Evidence: `starkc/tests/a10_provider_emit.rs` (12 cases).

**Invariant 6 is closed for the admitted forms, and closed by construction.** Every argument is
bound to a *named* local before the call, and pointers are taken from that binding:

```rust
let __prov_a0 = <operand>;                       // named, live across the call
let __prov_status: u32 = unsafe { f(__prov_a0 as *mut u64) }.code;
```

The shape this rules out — `f(make_bytes(v).as_ptr(), …)` — compiles, links, and is undefined
behaviour at runtime, so no later stage would catch it. A provider call is therefore emitted as a
**statement sequence**, not an expression, which is why it is separated from `emit_call` rather
than folded into it.

`ValidatedProviderCall` records reach emission through `TyEnv`, which was already threaded to every
emitter. The field defaults to empty rather than becoming a required constructor argument, so the
many existing `TyEnv::new` call sites — all of which predate providers — are untouched.

**d-2 does not interpret the status.** The raw `ProviderStatus.code` reaches the MIR destination
as `UInt32`, matching what `provider_sig` types it as, and a test asserts no branch on it exists
yet. Dispatch is d-3's, because channel policy belongs in the binding plan d-1 built for it rather
than in the emitter.

**What C7.8.2d-3 landed.** Output-slot discipline and status dispatch, closing invariants 8 and 9
for non-resource calls. `ValidatedProviderCall` gained `status_binding`, matching §2's sketch.

**Invariant 8 is stronger than "do not read on failure".** A `ScalarOut` is emitter-owned
`MaybeUninit` storage; the provider writes *there*, and the value reaches the MIR-visible local
only inside the success arm:

```rust
let mut __prov_o0 = std::mem::MaybeUninit::<u64>::uninit();
let __prov_status: u32 = unsafe { f(__prov_o0.as_mut_ptr()) }.code;
match __prov_status {
    0u32 => { *__prov_a0 = unsafe { __prov_o0.assume_init() }; }
    unknown => stark_runtime::provider_abi::contract_violation_unknown_status(..., unknown),
}
```

There is no failure path on which a read could occur, because the only write to the MIR local is in
`0u32 =>`. A provider that writes its slot and *then* reports failure still has its output ignored.
`ScalarInOut` and `BufferInOut` keep taking their pointer from the argument binding — §11.1 makes
them caller-owned, so `MaybeUninit` would be wrong for them.

**Invariant 9: the wildcard is the contract-violation channel, not a fallback.** An undeclared
status calls `contract_violation_unknown_status`, which aborts — so **it never becomes a STARK
value at all**, and the MIR destination is only ever written with success or a declared code. There
is no `_ => SomeError::Other` arm, which is the collapse Packet 1 §1.2 forbids. Declared arms are
emitted in ascending code order, so generated source does not depend on declaration order.

**What invariant 8 cannot enforce, stated rather than implied.** A provider that returns success
*without* writing its output slot is undetectable for scalar outputs: `MaybeUninit` has no
queryable initialised bit, and a sentinel value would be indistinguishable from a legitimate
result. That case is the provider's §11.1 obligation, not something generated code can check. The
analogous check *is* possible for `HandleOut` — `resource_type` validation — and lands with the
resource work.

Evidence: `starkc/tests/a10_provider_emit.rs` (21 cases).

**What C7.8.2d-4 landed.** The resource framework, structured for invariant 7 and proven with a
**synthetic** resource type — `File` is not admitted and nothing here should be read as saying it
is. Evidence: `starkc/tests/a10_provider_resource.rs`.

**Emission now consumes the binding plan** rather than re-walking `AbiParam`s. That was d-1's
purpose: classification, resource-type resolution and the status vocabulary are decided once, so
the emitter cannot quietly disagree with the verifier about which parameters are outputs.

**Invariant 7's lowering contract.** A `HandleConsumed` argument is converted with `into_raw()`
*before* the call, consuming the owning handle at call entry. The value is moved, so there is no
restore path on any arm — success, declared error, or contract violation. The shape ABI §8 rules
out (`call; if success { mark moved } else { restore }`) is not merely avoided but unrepresentable
in the emitted sequence.

**`HandleOut` validates before constructing.** The raw handle is read from `MaybeUninit` on the
success arm only, then passed to `OwnedResourceHandle::from_raw_checked` with the expected §7 id; a
mismatch calls `contract_violation_resource_type`, which aborts. Wrapping a mistyped handle would
hand generated code an owning value for a resource of unknown kind, and every later operation on it
— including its close — would act on the wrong thing.

**The §7 type id is the index into the provider's own declared list**, not a global id and not a
registry index. `ValidatedProviderCall` gained `provider_resource_types` so the id is derivable and
a returned handle is checkable; the test fixture declares an unrelated type *first*, so an
implementation that returned 0 or a registry index fails.

Two refusals are now distinct: `UnboundResourceType` is the *compiler* lacking a binding
(MIR-0024), `UndeclaredResourceType` is the *provider's own metadata* being internally inconsistent
— a type with no id to assign, so nothing could validate a handle against it.

**`extern "C"` declarations are shape-only.** A handle crosses as `RawResourceHandle` whatever type
it carries, so declarations need no registry and the resource type never appears in the C
signature. The refusal therefore lives at planning, and `ResourceRegistry::builtin()` remains empty.

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

## 11. The symbol prefix rule — RESOLVED (dropped, 2026-07-29)

Packet 1 §1.3 requires the validator to reject "any symbol lacking an approved provider-identity
prefix". **That rule as literally derived rejects the only real provider we have.**

`stark-time/native/src/lib.rs` declares identity name **`stark-std-time`** and symbols
**`stark_time_monotonic_now_ns`** / **`stark_time_unix_now`**. A prefix mechanically derived from
the identity would be `stark_std_time_`, which neither symbol carries. Enforcing it would force a
change to that crate's declared function names — which are ABI-facing, and therefore exactly what
Packet 1's own exit condition forbids changing.

C7.8.2b therefore implements the two rules that are exact and unambiguous, and **does not** invent
a prefix policy:

- **symbol grammar** — `[A-Za-z_][A-Za-z0-9_]*`, rejected verbatim, never sanitised;
- **uniqueness** — no two selected providers export the same C symbol, and no provider exports one
  twice.

Note that uniqueness is the *actual* anti-collision guarantee. A prefix convention only makes
collision unlikely; the duplicate check makes it impossible. The prefix rule's residual value is
social (a symbol reads as belonging to its provider), not structural.

Options for the owner:

| | Option | Consequence |
| --- | --- | --- |
| A | **Drop the prefix requirement**; keep grammar + uniqueness | No provider changes. Loses a readability convention that nothing structural depended on |
| B | **Declared, not derived** — a provider declares its own symbol prefix in metadata | Accommodates `stark-std-time` → `stark_time_`, but adds a `ProviderMetadata` field, which breaks the struct literal in `stark-time/native/src/lib.rs` and so trips the same exit condition from the other side |
| C | **Compiler-side approved-prefix registry** for first-party providers | Matches §1.3's word "approved" exactly, needs no metadata change and no provider change. Cost: a registry the compiler carries, and a policy question for third-party providers later |
| D | **Rename `stark-time`'s symbols** to match a derived prefix | Keeps the rule as drafted, at the cost of an ABI-facing change to the reference provider — the thing Packet 1 named as the drift signal |

**RESOLVED — owner decision, 2026-07-29: option A. The prefix requirement is dropped.**

Packet 1 §1.3 is amended accordingly. Grammar and the two duplicate checks (cross-provider and
intra-provider, kept distinct because their remediations differ) are the binding symbol rules.
`stark-time` needs no change, which was the point. Grammar and uniqueness are enforced today; only the prefix check is
absent.

## 12. Revision log

**Rev. 1 (2026-07-28) — A10 approved under CE3.** Adds `Callee::Provider(ProviderCallId)` and
`ValidatedProviderCall`; nine binding verifier invariants plus the `resource_type` rule; the
pre-verification binding sequence; backend emission prohibitions; `MIR_RUNTIME_SURFACE` →
`0.1-A10`. No `RuntimeFn` additions. No MIR shape version change.

**Rev. 2 (2026-07-28) — C7.8.2a landed.** §10 records implementation state, the
representable-but-not-admitted position, the `provider_abi` layering move and its shim, and the
interpreter's permanent exclusion. No contract change.

**Rev. 9 (2026-07-29) — C7.8.2d-4 landed.** §10 records emission moving onto the binding plan,
invariant 7's consumed-handle contract, `HandleOut` type validation before construction, the §7 id
derivation, and the two distinct resource refusals. `ValidatedProviderCall` gained
`provider_resource_types`. No contract change; `File` remains unadmitted.

**Rev. 8 (2026-07-29) — C7.8.2d-3 landed.** §10 records `MaybeUninit` output slots, success-only
reads, the three-channel status dispatch, and the one case invariant 8 cannot enforce.
`ValidatedProviderCall` gained `status_binding` per §2's sketch. No contract change.

**Rev. 7 (2026-07-29) — C7.8.2d-2 landed.** §10 records `extern "C"` declaration and call
emission, invariant 6 closed by named borrow bindings for non-resource forms, and the deliberate
absence of status interpretation. No contract change.

**Rev. 6 (2026-07-29) — C7.8.2d-1 landed.** §10 records the provider binding plan, the
input/output classification rules, the status-channel dispatch, and the resource registry.
MIR-0024's meaning narrowed to "resource type not bound", which outlives the empty registry. No
contract change.

**Rev. 5 (2026-07-29) — C7.8.2c landed.** §10 records invariants 1–5 enforced, the derived MIR
signature, `provider_target_triples`, and why 6–9 cannot precede a binding layer. MIR-0020 retired.
No contract change.

**Rev. 4 (2026-07-29) — prefix requirement dropped (owner, option A).** §11 resolved; Packet 1
§1.3 amended. Grammar plus cross- and intra-provider uniqueness are the binding symbol rules. No
contract change.

**Rev. 3 (2026-07-29) — C7.8.2b landed.** §10 records provider selection, resolution and the
call arena. §11 raises the symbol-prefix conflict: Packet 1 §1.3's prefix rule, derived
mechanically, rejects `stark-time`'s existing symbols. Grammar and uniqueness are implemented;
the prefix check is deliberately not, pending an owner decision. No contract change.
