# WP-C7.8.1 — decision packets for the first-party native host capability model

Five dispositions the owner must record before any WP-C7.8 implementation begins (2026-07-28).
Each states root cause, normative requirement, choices, recommendation, compatibility impact,
implementation surface and required regression evidence.

These packets replace §5 ("Required design decisions") of the superseded root-level
`WP-C7.8-Native-Host-Capability-Foundation.md`. That document's §5.1 presented the native
resource model as an open choice and recommended generated-Rust wrappers with Rust `Drop`. That
question is **not open**: it was decided by **CE4 Amendment 1 (CD-054, approved 2026-07-21)** and
is implemented in `starkc/stark-runtime/src/provider_abi.rs`. Native Provider ABI v0.1 and CE4
Amendment 1 are **normative inputs** to these packets, not subjects of them.

The framing for all five:

> Apply the approved Native Provider ABI v0.1 resource, ownership, buffer, result, and
> failure-channel semantics to first-party host capabilities.

Packets 1 and 2 are sequential prerequisites for everything else. Packets 3, 4 and 5 may be
dispositioned in parallel but all five must be recorded before WP-C7.8.3 starts.

---

# Packet 1 — CE4: first-party provider invocation model

## Root cause

Native Provider ABI v0.1 specifies the complete semantic contract for host-backed resources —
§7's raw/owning handle split, §8's three ownership forms, §9's borrowed buffers, §11's
`ProviderStatus`-plus-output-slots return shape, §12's three failure channels, §13's
MIR-owned exactly-once close — and then explicitly declines to implement the mechanism that
would let any of it execute:

> `stark-runtime/src/provider_abi.rs` — the `#[repr(C)]` type definitions from §7, §9, and §11
> …No `extern "C"` linkage, dynamic loading, or invocation logic — **the owning later package's
> job**. (§17)

WP-C7.8 is that owning package. The gap is not the resource model; it is **where a provider
function physically lives and how a MIR body reaches it**.

This is already blocking a real consumer. `stark-time/` exists, ships a provider crate at
`stark-time/native/` whose metadata passes the v0.1 validator
(`provider_metadata_validates_against_abi_v0_1`), and is classified `READY_PACKAGE_PROVIDER` with
exactly one blocker recorded in `stark-time/BLOCKERS.md`:

> Real `extern "C"` provider linkage/invocation: STARK generated code calling
> `stark_time_monotonic_now_ns` / `stark_time_unix_now` through Native Provider ABI v0.1 and
> observing their `ProviderStatus`/output-slot results.

That file also records that `WP-TIME-A §1.1` **forbids the time package from inventing the
mechanism itself**. So the seam has a waiting consumer, a written specification, and a standing
prohibition on being solved bottom-up. It needs a CE4 disposition.

## Normative requirement

- **CHARTER §2.3, CE4** — "Runtime ABI, value layout, drop glue, panic/trap, and native resource
  model" is owner-escalated. Invocation placement is part of the runtime/provider boundary.
- **ABI v0.1 §16** — three compatibility checks (`abi_version == "0.1"`, target triple in
  `target_triples`, declared capabilities a superset of what the program requires) must run
  **before any provider function is ever called**, all fatal, never a silent downgrade.
- **ABI v0.1 §6.1** — every raw↔owned conversion goes through
  `OwnedResourceHandle::from_raw_checked`/`as_raw`/`into_raw`. Generated code must never touch
  `RawResourceHandle`'s fields.
- **`WP-C5-ENTRY.md` §3.2** — reject before backend invocation, rather than failing inside rustc.

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **Full provider boundary now** — first-party capabilities ship as dynamically loaded providers, discovered and version-checked at build time, called through `extern "C"` | One architecture for first- and third-party providers, and the ABI is proven end-to-end by its first real consumers. But dynamic loading, provider discovery, packaging and per-platform loader behaviour all have to be built before a single `File::open` executes — and each is its own cross-platform surface. Scope grows well past the P1 unblock |
| B | **Statically linked, ABI-semantic** — first-party providers are ordinary Rust crates linked into the produced binary, reached by direct symbol reference, but conforming exactly to §7/§8/§9/§11/§12/§13 semantics and constructed through §6.1's boundary helpers | Smallest step that unblocks P1 without reopening CE4's semantics. Handle split, status/out-slot discipline, failure channels and MIR-owned drop are all preserved and testable. Cost: two physical invocation mechanisms coexist until dynamic loading lands, and the ABI remains unproven through an actual dynamic boundary |
| C | **Runtime-intrinsic** — host operations become new `RuntimeFn` variants like every other runtime call, with no provider concept at all | Cheapest to build (the `Callee::Runtime` bridge already exists) and the worst outcome architecturally: it re-implements exactly-once close, failure classification and handle ownership in a second place, leaves ABI v0.1 a specification with no consumer, and strands `stark-time`'s already-written provider crate |

Option C is listed because it is the path of least resistance from the current code and should be
explicitly rejected rather than drifted into.

## Recommendation

**B — statically linked first-party providers, with exact semantic conformance to Native Provider
ABI v0.1. Dynamic loading becomes a separate later work package.**

B contains scope without conceding architecture. Every property CE4 Amendment 1 was written to
guarantee is preserved and independently testable; what is deferred is only the *physical*
boundary, which is the part with the largest cross-platform cost and the smallest semantic
content. Migration A-ward later is a change of linkage, not of contract, precisely because the
handle/status/buffer/drop rules are unchanged.

The drift risk B carries is real and must be answered structurally, not by intention: **the same
conformance test suite must run against both mechanisms**, so a statically linked provider that
violates §8's consumed-handle rule or §11.1's output-initialisation rule fails the same test a
dynamic one would.

## Compatibility impact

- **ABI version stays `0.1`.** Nothing about the contract changes; a first executing consumer
  appears. `COMPILER-STATE.md`'s "No provider executes. §10.2's boundary is unchanged" statement
  becomes stale on this WP's closure and must be updated there, not silently.
- **`stark-time` unblocks** without modification to its provider crate — the strongest available
  evidence that B implements the ABI rather than an approximation of it. If `stark-time/native/`
  needs edits to work under B, that is a signal B has drifted.
- **§16's three checks still apply** to a statically linked provider. Target-triple and capability
  checks are build-time; `abi_version` becomes a compile-time constant assertion.
- **No Core language semantics change.** No spec document is touched by this packet.

## Implementation surface

```text
starkc/src/backend/generated_rust/           provider call emission: symbol reference,
                                             MaybeUninit out-slots, status check, boundary-helper
                                             conversion. New module; do not extend emit_runtime.rs
                                             (that is the RuntimeFn bridge, a different mechanism)
starkc/src/backend/provider_abi.rs           §16 checks promoted from metadata validation to
                                             build-gating; capability-requirement computation
                                             from the program's dependency graph
starkc/stark-runtime/src/provider_abi.rs     unchanged — this is the point
starkc/src/backend/generated_rust/build.rs   link the selected provider crates into the generated
                                             workspace's Cargo.toml (per-target selection)
stark-time/native/                           unchanged; becomes the first executing provider
```

## Required regression evidence

- `stark-time`'s existing provider metadata validates **and** its two functions execute from a
  native STARK program, observing `ProviderStatus` and output slots.
- A conformance suite over §8/§11.1/§13.1 that is mechanism-agnostic by construction.
- Negative: a provider whose declared target triples exclude the build target is rejected **before
  backend invocation** (§3.2), with a diagnostic distinct from `stark_runtime::version::check`'s
  (§16's independence requirement).
- Negative: a handle written to a `HandleOut` slot with the wrong `resource_type` traps
  (§11.1/§12 middle row) rather than being wrapped.

---

# Packet 2 — CE3: MIR runtime surface amendment for provider invocation

## Root cause

`RuntimeFn` (`starkc/src/mir/mod.rs:384`) is a **closed, versioned** enum of 72 variants, and its
own documentation states the rule:

> every extension of this enum is an extension of the MIR version's runtime contract, and an
> unknown variant must fail loudly at any backend (V-RT-1).

`MIR_RUNTIME_SURFACE` is currently `"0.1-A9"` (`starkc/src/mir/mod.rs:44`). All 72 existing
variants are **value** operations. None carries a resource handle, none has an out-parameter, and
none has a failure channel other than trap-or-succeed. Provider invocation is a different shape,
so it cannot be added as more of the same.

Separately, `MirTy::Core(CoreType, Vec<MirTy>)` already exists and `CoreType::File` is already
declared (`starkc/src/hir.rs:153`). The front end types `File` today; the backend refuses it,
because `emit_types.rs`'s per-`CoreType` match (lines ~234–287) has no arm for it and the program
falls through to the generic refusal at `starkc/src/bin/stark.rs:391`/`:431`. **No new `MirTy`
variant is required** — this is an admission and lowering gap, not a type-system gap.

## Normative requirement

- **V-RT-1** — an unknown runtime variant must fail loudly at any backend.
- **MIR contract §7** — the runtime surface is closed and versioned; consumers reject a surface
  they do not recognise.
- Precedent: `PrintFloat32`/`PrintlnFloat32` were added at `0.1-A9` under an approved CE3
  (CD-138), with the operation's identity carrying the declared width. Surface additions are
  enumerated and dispositioned, never incidental.

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **A distinct `Callee::Provider` calling form**, parallel to `Callee::Runtime`, carrying the provider/capability/function identity and its `AbiParam` signature | Keeps the closed `RuntimeFn` enum genuinely closed and makes the two mechanisms structurally distinguishable in MIR — a verifier can enforce ABI rules on provider calls and value-op rules on runtime calls without a discriminating side table. Cost: a new `Callee` variant, verifier rules, and a surface-version bump |
| B | **Model every provider function as a new `RuntimeFn` variant** | Reuses the existing bridge, and dissolves the distinction the ABI depends on: `RuntimeFn` would have to grow handle-carrying, out-parameter-bearing variants, at which point its "closed value surface" property is gone and §8/§11.1 have no structural home |
| C | **No MIR change** — emit provider calls as backend-only constructs invisible to MIR | Unverifiable. MIR's `Drop` terminator is what owns exactly-once close (§13); a resource MIR cannot see is a resource MIR cannot drop |

## Recommendation

**A.** The ABI's guarantees are enforced by *shape* — §6's closed `AbiParam` vocabulary makes
aggregates and callbacks unrepresentable rather than merely forbidden. That property only survives
into MIR if provider calls have their own form. B would trade it away for reuse of an emission
path, and C forfeits the drop guarantee entirely.

Bump `MIR_RUNTIME_SURFACE` to `0.1-A10`. Enumerate every added operation with signature and
ownership form in the amendment, as `mir-amendment-A1-strings-runtime.md` did for its surface.

## Compatibility impact

- **Any consumer pinned to `0.1-A9` rejects `0.1-A10` loudly**, which is the intended behaviour of
  a versioned surface, not a regression.
- **Snapshot/corpus re-pin**: any evidence recording `mir_runtime_surface` must be refreshed
  (`starkc/src/backend/version.rs:26` writes it into the artifact contract).
- **No change to the 72 existing `RuntimeFn` variants**, so no existing lowering is perturbed.

## Implementation surface

```text
starkc/src/mir/mod.rs:44          MIR_RUNTIME_SURFACE -> "0.1-A10"
starkc/src/mir/mod.rs             Callee::Provider { .. }; no RuntimeFn additions
starkc/src/mir/verify.rs          provider-call admission: AbiParam conformance, consumed-handle
                                  liveness (§8), out-slot never read on failure (§11.1),
                                  exactly-one-close-per-resource-type reachability (§13)
starkc/src/backend/generated_rust/emit_types.rs   admit MirTy::Core(CoreType::File, _) and the
                                  new host resource core types (~lines 234-287)
STARKLANG/docs/compiler/mir-amendment-A10-provider-invocation.md   the enumerated amendment
```

## Required regression evidence

- Positive and negative fixtures per added operation.
- A `0.1-A9`-pinned consumer rejects a `0.1-A10` artifact with the V-RT-1 diagnostic.
- Verifier negative: a consumed handle used after the call is rejected **in MIR verification**,
  not by rustc.
- Verifier negative: a resource type reachable in a body with no reachable close is rejected.

---

# Packet 3 — CE2: STD-IO-001's drop-close rule versus ABI §13.2

## Root cause

Two approved documents give different answers for the same event — an open `File` is dropped and
the host close fails.

**`06-Standard-Library.md`, STD-IO-001 (normative):**

> Dropping an open file attempts close but **cannot surface a new language trap**.

**Native Provider ABI v0.1 §13.2 (approved CE4):**

> - **abort without unwinding**;
> - **do not retry close**;
> - **consider the handle consumed** regardless;
> - **do not run further pending Drop glue**.

Read naively these are contradictory, and the contradiction sits directly under the drop path
every host resource in WP-C7.8 depends on. It must be resolved before C7.8.4, not discovered
during it.

## Normative requirement

Both of the above, plus **ABI §12**, which is the key to the resolution: §12 defines **three**
channels and classifies them separately —

| Channel | STARK-observable as |
| --- | --- |
| Provider error | `Result::Err` |
| STARK trap (contract violation) | abort, same as any other MIR trap |
| **Host failure** | **also an abort, but classified distinctly in diagnostics** |

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **Reconcile by classification** — a failed drop-close is a §12 **host failure**, not a language trap. STD-IO-001 forbids a *language trap*; it does not speak to host failure, which §12 already holds distinct | No document changes; both texts hold as written. Requires the implementation and the diagnostic to actually classify it as host failure — if it renders as a trap, the reconciliation is fiction. This is a CE2 ambiguity resolution, the cheapest honest outcome |
| B | **Amend STD-IO-001** (CE1) to state explicitly that a failed close on the drop path is a host failure and aborts | Removes all doubt at the cost of a normative Core edit and a spec regeneration. Strictly clearer than A; strictly more expensive |
| C | **Swallow close failure on the drop path** for `File` specifically, honouring STD-IO-001 literally | Contradicts §13.2's actual reasoning — that once a close has failed the runtime's resource invariants are untrustworthy, so continuing is guesswork. Trades a loud, rare failure for silent data loss on exactly the operation where data loss is the risk |

## Recommendation

**A, with B as the fallback if the owner reads STD-IO-001's "cannot surface a new language trap"
as forbidding any abort whatsoever.** A costs nothing and both texts survive intact; the burden it
creates is a real one and belongs in the test suite — the failure must be *observably* classified
as host failure, in the diagnostic and in the evidence, not merely asserted to be one.

C should be rejected explicitly. It is the option that looks most compliant and is least safe.

## Compatibility impact

- Under A: none to any document. One new required property in the diagnostic contract.
- Under B: `06-Standard-Library.md` edit plus `python3 STARKLANG/tools/build-core-spec.py`
  regeneration of `STARK-Core-v1.md` (+ HTML/PDF), per the repo's spec-editing convention.
- Either way, `stark-hex`-style package evidence recording IO failure text is unaffected; this is
  a drop-path-only rule.

## Implementation surface

```text
starkc/src/backend/generated_rust/            drop-path close: status check, host-failure
                                              classification, no retry, no subsequent drop glue
starkc/src/diagnostics.rs (or equivalent)     host-failure rendering distinct from trap rendering
STARKLANG/docs/spec/06-Standard-Library.md    ONLY under option B
```

## Required regression evidence

- A provider whose close returns nonzero, dropped implicitly: process aborts, the observation is
  classified host failure, no further drop glue runs (assert a second resource's close does *not*
  execute).
- The same case does not render as a language trap in diagnostics or in trap-provenance evidence.

---

# Packet 4 — CE1: normative surface placement for the five capabilities

## Root cause

The superseded proposal presented its `args`/`env`/`File`/time/TCP signatures as "minimum public
surface", with names left to "existing package conventions". They are not equivalent to each
other, and the difference is normative.

**`File` is already Core.** `06-Standard-Library.md:551–560` defines it, and STD-IO-001 makes it
normative under the `std-full` profile:

```stark
impl File {
    fn open(path: &str) -> Result<File, IOError>;
    fn create(path: &str) -> Result<File, IOError>;
    fn read_to_string(&mut self) -> Result<String, IOError>;
    fn write(&mut self, data: &[UInt8]) -> Result<UInt64, IOError>;
    fn write_str(&mut self, text: &str) -> Result<UInt64, IOError>;
    fn close(self) -> Result<Unit, IOError>;
}

enum IOError { NotFound, PermissionDenied, AlreadyExists, InvalidInput, Other(String) }
```

Four consequences the superseded proposal missed:

1. `read_to_end`, `write_all` and `flush` **are not in the spec**. The spec's `write` returns
   `UInt64` bytes-accepted and STD-IO-001 says "callers must handle a short write" — a *different
   contract* from `write_all`, not a rename of it.
2. `close(self) -> Result<Unit, IOError>` **is already normative**, so "omit public `close` and
   rely on ownership plus `Drop`" is a spec deletion, not a design choice.
3. `IOError` has **five** variants. `ConnectionRefused`, `TimedOut`, `Interrupted`,
   `UnexpectedEof`, `AddressUnavailable`, `Unsupported` and `Closed` are all absent.
4. `File` is non-`Copy`, movable, not cloneable — already normative, and already exactly what the
   ABI handle model provides.

**Nothing else is Core.** `06-Standard-Library.md` contains no `args`, no `env`, no `Instant`, no
`SystemTime`, no `sleep`, and no TCP type of any kind. Meanwhile `stark-time/` already exists as a
*package* (`starkpkg.json`, `stark-time/native/` provider), and
`STARK-Standard-Package-Roadmap.md` explicitly separates "pure libraries", "host-backed standard
packages that require operating-system providers", and "language/runtime capabilities".

So the established pattern is already decided by precedent: **host capabilities are packages.**
`File` is the sole exception, because it was placed in Core before that pattern existed.

## Normative requirement

- **CHARTER §2.3, CE1** — "Normative Core or tensor semantic change" is owner-escalated.
- **STD-PROFILE-001** — `std-full` is "an optional, indivisible advertised capability"; a host
  facility's absence "prevents the `std-full` claim rather than changing an API's" shape.
- **Repo convention** — new language features land in the spec first, extensions second.

## Choices

| | Option | Consequence |
| --- | --- | --- |
| A | **`File` stays Core and is implemented to the spec as written**; `args`/`env`/time/TCP ship as host-backed packages following the `stark-time` precedent | No CE1 at all for four of five capabilities. `File` needs no spec change either, provided the implementation delivers `write`'s short-write contract rather than substituting `write_all`. A `write_all` convenience may be added **in a package** over the Core primitive |
| B | **Extend Core** with `read_to_end`/`write_all`/`flush`, extra `IOError` variants, and new `args`/`env`/time/TCP APIs | One uniform surface, at the cost of a substantial CE1 touching the profile definition, plus spec regeneration, plus fixture re-triage — for capabilities the package system was built to carry |
| C | **Move `File` out of Core** into a package alongside the others | Most consistent end state, and a breaking normative removal from `std-full` for a type that has been specified since Core v1 froze. Not justified by this WP's needs |

## Recommendation

**A.** It is the only option that unblocks P1 without a normative Core change, and it is the one
the codebase already votes for — `stark-time` is a package, `CoreType::File` is in the front end,
and `STARK-Standard-Package-Roadmap.md` was written to hold exactly this class of work.

Two conditions attach:

- **C7.8.4 implements `write`, not `write_all`.** The short-write contract is the normative one.
  A `write_all` loop belongs in a package or in the STARK-level prelude over it.
- **`NetworkError`, `ProcessError` and `TimeError` are package types**, never additions to Core's
  `IOError`. Packet 5 governs what may appear inside them.

`args()`'s fallibility follows from the recorded invalid-text policy: if the ABI's string-transfer
rule can reject non-UTF-8 platform arguments, `args()` is fallible; if the policy is lossy
replacement, it is infallible. Decide that in C7.8.3 and record it — do not leave it implied by a
signature.

## Compatibility impact

- **No spec edit, no `build-core-spec.py` regeneration, no fixture re-triage** under A.
- Gate C6's recorded exclusion — `File` (5) EXCLUDED, "deferred to the I/O gate" — is discharged
  by C7.8.4 against the spec surface, not against a new one.
- Package qualification labels from `STARK-Standard-Package-Roadmap.md` P0
  (`HOST-BACKED`, `PLATFORM-SPECIFIC`) apply to the new packages.

## Implementation surface

```text
STARKLANG/docs/spec/06-Standard-Library.md    UNCHANGED under A
starkc/src/hir.rs:46,47,153                   CoreType::File, ReadFile, WriteFile already exist
stark-env/, stark-net/ (new packages)         args/env and TcpListener/TcpStream
stark-time/                                   already exists; unblocked by Packet 1
```

## Required regression evidence

- A short write is observable from STARK: `write` returns fewer bytes than supplied and the
  program can act on it.
- `IOError`'s five variants are produced for the four distinguishable host conditions, with
  `Other(String)` carrying stable context otherwise (STD-IO-001's exact rule).
- No new symbol appears in `std-full` that is not in `06-Standard-Library.md`.

---

# Packet 5 — CE9: trust boundary for file, environment and network capability

## Root cause

WP-C7.8 introduces the first compiler-generated code that opens files, reads process environment,
and originates and accepts network connections. `CHARTER §2.3` classes this as CE9
("Security-sensitive compiler behaviour — archive extraction, process execution, code generation,
native linking, trust boundaries"). The superseded proposal listed sound security *requirements*
in its §11 but did not flag the escalation, so the rules would have landed as implementation
choices rather than an owner-recorded boundary.

Packet 1's Option B raises a second CE9 surface the proposal did not contemplate at all: linking a
provider crate into the produced binary is **native linking** under the same clause.

## Normative requirement

- **CHARTER §2.3, CE9.**
- **ABI §16** — provider identity, integrity hash and origin metadata (§2/§3) are checked before
  any call.
- **ABI §15** — no provider function may receive a function pointer; enforced structurally by
  `AbiParam`'s closure.

## Choices

Not an either/or. The disposition records the boundary; the question is where each line sits.

| Rule | Recommended position |
| --- | --- |
| Environment mutation | **Excluded.** Read-only `env` in C7.8; no `set_env` at any point in this WP |
| Process/shell execution | **Excluded** from C7.8 entirely. It is the CE9 clause's own example and deserves its own disposition |
| Inbound network | **Admitted, explicitly** — `TcpListener::bind`/`accept` is required by P1 (Packet 4, C7.8.6). The superseded §11's "no implicit network listener" is retained in its correct form: no listener is ever created implicitly; a program must call `bind` |
| Path handling | No path concatenation, normalisation or resolution inside the runtime. Paths pass through as opaque byte strings to the host call |
| Provider linking | Only crates named in the build's own workspace; no discovery from environment variables or ambient search paths |
| `unsafe` | Confined to §6.1's named boundary helpers, which already exist and are already reviewed |
| Input sizing | Bounded before allocation on every read path; an unbounded `read_to_string` on a device file must fail rather than exhaust memory |
| Test network use | Loopback only; no external network dependency in any test |

## Recommendation

Record the table above as the CE9 disposition for WP-C7.8, with **inbound TCP explicitly
admitted** — that is the substantive change from the superseded document, and it should be an
owner decision rather than a consequence of P1's requirements.

## Compatibility impact

- No existing behaviour changes; this establishes a boundary for new behaviour.
- The "no process execution" line should be recorded as **deferred**, not permanent — P1 does not
  need it, and a future package may. Recording it as permanent would over-claim.

## Implementation surface

Enforcement is distributed rather than centralised: absence of a `set_env` provider function,
absence of a process-execution capability in any declared capability set, `bind` as an explicit
call in every listener test, and the §16 checks at build gating.

## Required regression evidence

- No provider in the C7.8 set declares an environment-mutating or process-executing function; the
  validator's capability-reachability check is the mechanical assertion.
- A read against a source with no defined end fails on the size bound rather than allocating
  without limit.
- Full test suite runs with no external network reachable.

---

# Sequencing note

Packets 1 and 2 gate everything. Packet 3 gates C7.8.4 specifically. Packets 4 and 5 gate the
public surface of C7.8.3 through C7.8.6.

Evidence for these packets comes from hosted CI on all three Tier-1 platforms, not from local
runs — the cross-platform claims in WP-C7.8's exit criteria cannot be established any other way.
