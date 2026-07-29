# WP-C7.8 — First-Party Native Host Capabilities

**Status:** IN PROGRESS (CD-212). C7.8.0–C7.8.2 closed; all five packets dispositioned;
C7.8.3–C7.8.6 unblocked.
**Parent gate:** C7
**Supersedes:** `WP-C7.8-Native-Host-Capability-Foundation.md` (repo root), dispositioned
**REVISE — conflicts with approved CE4 Native Provider ABI v0.1 and does not fully unblock P1.**
**Decisions:** `WP-C7.8.1-DECISION-PACKETS.md` — **all five dispositioned** (CE4, CE3, CE2 on 2026-07-28; CE1, CE9 on 2026-07-29)
**MIR amendment:** `mir-amendment-A10-provider-invocation.md` (rev. 1, surface `0.1-A10`)
**Primary targets:** macOS, Linux, Windows

---

## 1. Why this document replaces the previous one

The superseded proposal was correct about the blocker and wrong about the foundation. Its §5.1
presented the native resource model as an open choice between generated-Rust owned wrappers and a
runtime handle table, and recommended the former with destruction via Rust `Drop`.

That question was decided on 2026-07-21 by **CE4 Amendment 1 (CD-054)** and is implemented in
`starkc/stark-runtime/src/provider_abi.rs`. The approved model is:

- `RawResourceHandle` at the provider boundary — `#[repr(C)]`, `Copy`, boundary-confined;
- `OwnedResourceHandle` in the compiler/runtime model — not `Copy`, not `Clone`, **no Rust
  `Drop`**;
- exactly-once release generated and verified through **MIR `Drop` terminators**.

The no-`Drop` property is load-bearing, in the ABI's own words: a Rust destructor "would either
double the close or quietly take over an invariant the MIR verifier is supposed to own — and
'which layer guarantees exactly-once?' must have one answer." The superseded recommendation
reversed that. It was also a reversal the document could not have argued, because it never cited
the ABI at all.

This document therefore does not re-derive a foundation. It applies one:

> Apply the approved Native Provider ABI v0.1 resource, ownership, buffer, result, and
> failure-channel semantics to first-party host capabilities.

Native Provider ABI v0.1 and CE4 Amendment 1 are **normative inputs**. Nothing in this WP reopens
them.

## 2. What was also missing

**The previous scope did not unblock P1.** Its TCP surface was `TcpStream::connect` — client only
— while P1's exit criteria (`COMPILER-ROADMAP.md` §4.2) require "TCP listener and stream", pure-STARK
HTTP/1.1 routing, and three working REST endpoints. Three REST endpoints cannot be served over an
outbound socket. `TcpListener::bind`/`accept` is mandatory; HTTP correctly stays out, because P1
intends to write it in pure STARK.

**The seam already has a waiting consumer.** `stark-time/` exists, ships a provider crate at
`stark-time/native/` whose metadata passes the v0.1 validator, is classified
`READY_PACKAGE_PROVIDER`, and records exactly one blocker: real `extern "C"` linkage and
invocation. `stark-time/BLOCKERS.md` further records that `WP-TIME-A §1.1` forbids that package
from inventing the mechanism. So the missing piece is not a design; it is an owner disposition and
the invocation seam behind it.

**Four capabilities are not Core and one is.** `06-Standard-Library.md:551–583` already defines
`File` and `IOError` normatively under STD-IO-001; it defines no `args`, no `env`, no `Instant`,
no `sleep`, and no TCP type. Packet 4 governs the consequences.

## 3. Objective

After WP-C7.8, native STARK programs can read command-line arguments and environment variables,
open/create/read/write files, obtain monotonic time and sleep, and bind, accept, read and write
TCP connections — with host-backed values that are move-only, deterministically released under
verified MIR, panic-contained, and mapped onto the ABI's three failure channels, on macOS, Linux
and Windows.

## 4. Non-goals

HTTP/HTTPS (P1 implements it in pure STARK); TLS; DNS beyond what the connect/bind surface needs;
asynchronous I/O; event loops; filesystem watching; directory traversal beyond test needs; advanced
file metadata; UDP; raw descriptor exposure; process or shell execution; environment mutation;
dynamic provider loading (Packet 1 defers it to a later WP); a direct Cranelift or LLVM backend;
general FFI.

Process execution is recorded as **deferred, not permanently excluded** (Packet 5).

## 5. Structure

```text
C7.8.0  Scope approval and roadmap/state insertion
C7.8.1  CE4 disposition: first-party provider invocation model
C7.8.2  MIR runtime surface amendment and ABI bindings
C7.8.3  Arguments and environment
C7.8.4  File I/O
C7.8.5  Monotonic time and sleep
C7.8.6  TCP listener and stream
C7.8.7  Cross-platform verification and P1 unblock assessment
C7.8.8  Source/package provider integration   <- the critical path to P1 (CD-220)
```

**C7.8.8 was not in the original structure, and its absence was the plan's largest error.** Slices
2e through 6 proved the backend and ABI by hand-building MIR. None of them made a capability
reachable from STARK source: `lower.rs` produces no `Callee::Provider` at all. A capability that
only a hand-authored MIR body can call is backend evidence, not a language feature, and P1 is
written in STARK.

C7.8.0 and C7.8.1 gate everything, and both are **closed**. C7.8.2 gates C7.8.3 onward. Within C7.8.3–C7.8.6, file
ownership and destruction must be proven before TCP — file I/O is the simpler resource-lifecycle
test bed, and a listener adds an accepted-resource lifetime on top of it.

---

## 5.0 WP-C7.8.0 — Governance and scope

**CLOSED 2026-07-28 (CD-201).** WP-C7.8 is admitted to the roadmap and recorded in compiler state;
the C7.7 open question is resolved; the C8 concurrency boundary is written down. **C7.8.2 is
authorised to begin.** C7.8.3 onward remains gated by Packets 4 and 5.

Required (all discharged):

1. ✅ C7.8 added to `COMPILER-ROADMAP.md` — §4.1's mandatory path between C7.1–C7.4 and P1, a
   `### WP-C7.8` entry after WP-C7.7, and §4.2's checkpoint gains P1's precondition.
2. ✅ `COMPILER-STATE.md` records **APPROVED — PRE-IMPLEMENTATION**, deliberately *not*
   `IN PROGRESS`: no implementation has started.
3. ✅ `WP-C7.7-GATE-EXIT.md` §6 records the owner decision resolving its open question — a
   preceding work package — **without changing the C7 exit verdict**.
4. ◻ Three of five dispositions recorded in `WP-C7.8.1-DECISION-PACKETS.md`: CE4 (CD-198/199),
   CE3 (CD-200), CE2 (CD-197). CE1 and CE9 remain open and gate C7.8.3 onward, not C7.8.2.
5. ✅ C8 concurrency boundary written down (§7, and `COMPILER-STATE.md`).

**Exit:** met for admission. **C7.8.2 is authorised to begin.** The remaining two dispositions are
tracked as gates on C7.8.3–C7.8.6 rather than on admission, since neither bears on the MIR
provider-call surface.

## 5.1 WP-C7.8.1 — First-party provider invocation model

The CE4 decision, **dispositioned 2026-07-28 as option B**. Full packet in
`WP-C7.8.1-DECISION-PACKETS.md` Packet 1.

**B — statically linked, ABI-semantic.** First-party capabilities are ordinary Rust crates linked
into the produced binary, reached by direct `extern "C"` symbol reference, conforming exactly to
ABI v0.1 semantics and constructed only through §6.1's boundary helpers. Dynamic loading is a
separate later work package. This unblocks P1 without reopening CE4's semantics; migration toward
A later is a change of linkage, not of contract.

**Panic containment is already structural under B and needs no new work.** The generated workspace
sets `panic = "abort"` in both profiles (`starkc/src/backend/generated_rust/build.rs:298`, `:314`)
as defence-in-depth for DROP-ABORT-001, and B compiles the provider into that same workspace under
that same profile — so a provider panic aborts rather than unwinding into generated STARK code. The
superseded document's §4.5 requirement is therefore satisfied by construction, not implemented.
**This does not survive Option A**: a dynamically loaded provider may be built `panic = "unwind"`,
so the future dynamic WP must supply boundary `catch_unwind` explicitly, inheriting none from the
build profile.

**No `catch_unwind` wrapper may be added to the static path** — it would misclassify a provider
defect as a recoverable result, give provider calls different panic semantics from the rest of the
generated workspace, and obscure the abort guarantee that already exists.

The four sub-decisions are **confirmed**, with two clarifications:

- **Status binding.** The provider code space encodes ABI channel one **only** — contract
  violations are detected caller-side, host failures are terminations, so neither ever travels as a
  code. Numeric ranges are deliberately *not* reserved for channels two and three. The code→error
  mapping lives in the package's binding layer rather than in provider metadata, requiring no ABI
  change. **Clarification: an undeclared status code is a contract violation**, never a generic
  recoverable `Other`, unless the package specification explicitly defines that fallback.
- **Symbol representation.** `FunctionDecl.name` verbatim, never through `mangle.rs`, with
  provider-identity prefixes and cross-selected-set uniqueness validation. No STARK↔provider
  collision is possible since the backend emits no `#[no_mangle]`. **Clarification: names are
  validated verbatim and never sanitised** — repairing one would make the metadata name differ from
  the linkage name, which must never be true when the same field drives a future `dlsym`. Admitted
  grammar `[A-Za-z_][A-Za-z0-9_]*`.
- **Platform selection.** Capability plus target triple; per-platform variation internal to the
  provider crate; ambiguity a hard error before backend invocation, naming both providers and their
  metadata locations. No priority mechanism — it would reintroduce implicit selection.
- **Conformance testing.** One mechanism-agnostic suite, runtime-violation fixtures, and
  `stark-time` as the real-provider case. **Each fixture asserts the exact channel**, not merely
  that the process failed.

B's drift risk is answered structurally, not by intention: **one mechanism-agnostic conformance
suite**, so a statically linked provider that violates §8's consumed-handle rule or §11.1's
output-initialisation rule fails the same test a dynamic one would, and the dynamic WP inherits the
suite unchanged.

This sub-WP explicitly inherits, without restatement or variation:

| Inherited | Source |
| --- | --- |
| Handle ownership — borrowed / consumed / out, never shared or aliased | ABI §8 |
| Consumed-handle error rule — dead regardless of status | ABI §8 |
| Out-parameter initialisation — `MaybeUninit`, valid only on success, never read on failure | ABI §11.1 |
| Borrowed-buffer lifetime — valid only for the call that received them | ABI §9 |
| Three failure channels | ABI §12 |
| MIR-owned exactly-once close | ABI §13 |
| Close-function shape — one `HandleConsumed` parameter, no others | ABI §13.1 |
| Close failure — fatal, no retry, no further drop glue | ABI §13.2 |
| Raw↔owned conversion only through named boundary helpers | ABI §6.1 |
| Three pre-call compatibility checks | ABI §16 |

**Exit:** CE4 recorded; `stark-time`'s two provider functions execute from a native STARK program
observing `ProviderStatus` and output slots, **with no semantic or ABI-facing source change to
`stark-time/native/`**. Permissible edits are limited to integration metadata, build plumbing and
symbol registration; an alteration to its signatures, ownership contract, status protocol or
provider metadata means the integration model has drifted from ABI v0.1.

## 5.2 WP-C7.8.2 — MIR runtime surface amendment and ABI bindings

The CE3 decision, **dispositioned 2026-07-28**. Packet 2 in `WP-C7.8.1-DECISION-PACKETS.md`; the
full amendment is `STARKLANG/docs/compiler/mir-amendment-A10-provider-invocation.md`.

`RuntimeFn` is a closed, versioned enum of 72 value operations (`starkc/src/mir/mod.rs:384`) whose
own contract states that "every extension of this enum is an extension of the MIR version's
runtime contract" (V-RT-1). Provider calls carry handles, out-parameters and a failure channel, so
they are a different shape and get a distinct `Callee::Provider` form rather than more
`RuntimeFn` variants.

`MirTy::Core(CoreType, Vec<MirTy>)` and `CoreType::File` (`starkc/src/hir.rs:153`) already exist —
this is an admission and lowering gap in `emit_types.rs`, not a type-system gap. No new `MirTy`
variant is required.

The declaration and nine verifier invariants were enumerated before the disposition, per the
owner's requirement that they be settled before implementation, and are **binding** under it.
`Callee` gains a fourth variant, `Provider(ProviderCallId)`, resolving to a validated
`FunctionDecl` carrying the full ABI contract — not a bare symbol, because every invariant is
checked against the declaration rather than reconstructed at the call site. Provider calls are
target-resolved **before** MIR verification; the backend never performs first-time provider
selection or interprets unvalidated metadata.

Required:

- enumerate every added runtime operation with signature and ownership form;
- update verifier admission rules (`AbiParam` conformance, consumed-handle liveness, out-slot
  discipline, close reachability);
- bump `MIR_RUNTIME_SURFACE` from `0.1-A9` to `0.1-A10`;
- the amendment is published at `mir-amendment-A10-provider-invocation.md` (rev. 1, approved);
- positive and negative fixtures per operation;
- record the CE3 disposition.

**Exit:** amendment published; a `0.1-A9`-pinned consumer rejects a `0.1-A10` artifact with the
V-RT-1 diagnostic; verifier rejects a post-consumption handle use in MIR, not in rustc.

## 5.3 WP-C7.8.3 — Arguments and environment

No host resource is required — no handle, no drop obligation — but the status and output-buffer
model still applies. This is the smallest slice that exercises Packet 1's seam end to end, which
is why it precedes file I/O.

Surface (package types per Packet 4; `stark-env` or equivalent, never Core):

```stark
fn args() -> Result<Vec<String>, ProcessError>
fn env(name: &str) -> Result<Option<String>, ProcessError>
```

`args()`'s fallibility is not free-floating: it follows from the recorded invalid-text policy. If
non-UTF-8 platform arguments are rejected, `args()` is fallible; if the policy is lossy
replacement, it is infallible. **Decide and record it in this sub-WP** rather than leaving it
implied by a signature.

Semantics: deterministic order; documented whether the executable path is included; owned strings
that never alias runtime-owned buffers; `Ok(None)` for an absent variable, `Err` only for invalid
input, encoding, or provider failure; no environment mutation (Packet 5).

Tests: zero/one/many arguments; spaces; Unicode; absent variable; present-but-empty variable;
normal variable; Unicode value where supported; invalid variable name.

**Exit:** a native program reads one argument and one environment variable on all three platforms.

## 5.4 WP-C7.8.4 — File I/O

Governed by Packet 3, **dispositioned 2026-07-28 under CE2**, and constrained by Packet 4.

**Implement the spec surface as written** (`06-Standard-Library.md:553–559`) — this is normative
Core, not new API:

```stark
impl File {
    fn open(path: &str) -> Result<File, IOError>;
    fn create(path: &str) -> Result<File, IOError>;
    fn read_to_string(&mut self) -> Result<String, IOError>;
    fn write(&mut self, data: &[UInt8]) -> Result<UInt64, IOError>;
    fn write_str(&mut self, text: &str) -> Result<UInt64, IOError>;
    fn close(self) -> Result<Unit, IOError>;
}
```

Three consequences, each a correction to the superseded proposal:

- **`write`, not `write_all`.** STD-IO-001's contract is bytes-accepted with "callers must handle
  a short write". A `write_all` loop is a package-level convenience over this primitive, not a
  replacement for it. `read_to_end` and `flush` are likewise absent from Core and may not be added
  here (Packet 4). The package `write_all` treats a **successful zero-byte write with bytes
  remaining as an error**, never a retry — the alternative is an unbounded loop.
- **`close(self) -> Result<Unit, IOError>` is already normative**, so it is implemented, not
  omitted. Per the Packet 3 disposition it **consumes `self` at call entry, unconditionally**: the
  recoverable completion operation runs first (ABI §13.1 — "any flush option, completion mode, or
  other fallible operation needing arguments must be a separate, explicitly invoked provider
  function, called before Drop"), its failure returns `Err(IOError)`, and the consumed resource
  passes through MIR's `Drop` terminator on **both** arms so the ABI close is attempted exactly
  once. A nonzero status there is a fatal host failure per §13.2, not a second `Result`.
  Consume-on-success-only is rejected: it would leak the resource precisely when completion failed.
- **No Rust `Drop`.** Destruction stays MIR-owned. This is the CE4 property the superseded
  document would have discarded.
- **Two documented consequences of the close model.** `close()` returning `Err(IOError)` is a
  report, not a recovery point — the resource is already gone, and no retry or reclaim is possible.
  And if completion fails *and* the mandatory ABI close then also fails, execution terminates
  before the `Err` reaches the caller, so `close()` has one path that returns no value at all.
- **Open question to decide in this sub-WP, not during it.** Core exposes no byte-oriented read —
  `read_to_string` is UTF-8-validating and whole-file — so a package `read_to_end` cannot be built
  over Core's `File` alone. Either the package binds the provider read directly (Core unchanged) or
  Core gains a byte read (CE1).

`File` is non-`Copy`, movable, not cloneable — already normative under STD-IO-001 and exactly what
the ABI handle model provides. Path strings pass through as opaque bytes; the runtime performs no
concatenation, normalisation or resolution (Packet 5). Record accepted encoding, relative-path
base, Windows separators and drive prefixes, and invalid-path behaviour.

Tests — positive: empty file; text; binary; create-and-write each of those; filenames with spaces;
Unicode filenames where supported; file returned from a function; file moved between locals;
implicit close on scope exit; explicit close; **short write observable from STARK**. Negative:
missing file; permission denied where reliably constructible; directory-as-file where platform
behaviour is defined; invalid UTF-8 through `read_to_string`; use-after-move rejected in MIR
verification; the `Core(File, [])` backend refusal gone. Cross-engine: HIR/native error-category
parity where HIR supports the operation; byte-for-byte output parity.

**Exit:** a native program reads a path from an argument, opens it, reads it, creates a second
file, writes the bytes, and releases both safely, with identical output bytes on all three
platforms. Gate C6's recorded `File` (5) exclusion is discharged **against the spec surface**.

## 5.5 WP-C7.8.5 — Monotonic time and sleep

Largely unblocked by C7.8.1 rather than newly built: `stark-time/` already exists with its
provider crate. Reuse it; do not introduce a parallel time model.

Required for P1: `Instant::now` and `sleep`. Wall-clock time is useful and already present in the
package, but it should not silently enlarge the minimum blocker-removal scope — include it only to
the extent the package roadmap already requires.

Semantics: monotonic within host guarantees; not calendar time; elapsed duration through existing
checked operations; `sleep` rejects negative durations if representable, defines zero-duration
behaviour, promises no exact wake time, and returns a defined provider error if unsupported.

Tests: two monotonic samples non-decreasing; valid elapsed duration; zero-duration sleep succeeds;
small sleep does not return before a documented tolerance; overflow and invalid-duration cases;
provider panic containment.

**Exit:** a native program records a start time, sleeps, records an end time, and computes a
non-negative elapsed duration on all three platforms — with `stark-time/BLOCKERS.md` discharged
and its classification updated from `READY_PACKAGE_PROVIDER`.

## 5.6 WP-C7.8.6 — TCP listener and stream

The listener is mandatory; the superseded scope's omission of it is why that document would not
have unblocked P1.

Surface (package types per Packet 4; `stark-net` or equivalent, never Core):

```stark
struct TcpListener;
struct TcpStream;

impl TcpListener {
    fn bind(address: &str) -> Result<TcpListener, NetworkError>;
    fn accept(&mut self) -> Result<TcpStream, NetworkError>;
}

impl TcpStream {
    fn connect(address: &str) -> Result<TcpStream, NetworkError>;
    fn read(&mut self, max_bytes: UInt64) -> Result<Vec<UInt8>, NetworkError>;
    fn write_all(&mut self, data: &[UInt8]) -> Result<Unit, NetworkError>;
}
```

Both are ABI resource types under the approved handle model with MIR-owned destruction. `accept`
is the first operation that **produces** a resource from a borrowed one (`HandleBorrowed` in,
`HandleOut` out), so it exercises a lifetime shape file I/O does not.

`NetworkError` is a package type. It may carry `ConnectionRefused`, `TimedOut`, `NotFound`,
`PermissionDenied`, `InvalidInput` and similar categories — but only inside ABI §12's **channel
one**. It must never absorb a contract violation (channel two) or a host failure (channel three).
Core's `IOError` is not extended (Packet 4).

Semantics: `127.0.0.1:8080`-style addresses suffice for this slice; hostname support only if it
comes naturally from the provider; reads return zero bytes on orderly peer closure; writes handle
partial OS writes; no listener is ever created implicitly — a program must call `bind`, with a
deliberate address and **no hidden default such as `0.0.0.0`** (Packet 5); no raw descriptor is
exposed; blocking I/O is acceptable.

Tests, against a controlled local server, loopback only — positive: bind; accept; connect; send;
receive; payload containing zero bytes; payload larger than one likely OS chunk; orderly close from
each side; stream moved between locals; stream returned from a function; accepted stream outliving
the accept call; implicit and explicit release. Negative: connection refused; malformed address;
invalid port; bind to an occupied port; peer reset where reliably testable; use-after-move rejected.

**Exit:** a native program binds a listener, accepts a connection from a second native program,
echoes a payload byte-identically in both directions, and releases both resources safely on all
three platforms.

## 5.7 WP-C7.8.7 — Cross-platform verification and P1 unblock assessment

Every mandatory capability test executes on macOS, Linux and Windows via hosted CI. No platform's
behaviour is inferred from another's.

Leak and lifecycle, at minimum: repeated file open/release; repeated TCP connect/release; repeated
accept/release; early return with a live resource; `?`-propagation with a live resource; resource
moved through a function call; explicit close followed by the destructor path.

The capability matrix (`§12` of the superseded document, retained) must distinguish
frontend-supported, HIR-supported, MIR-lowered, native-runtime-supported, and cross-platform
verified. A single "supported" column would reproduce exactly the over-claiming C7.2 was corrected
for.

**The closure claim, amended 2026-07-29 (CD-220).** The original wording would have been satisfied
by hand-built MIR, which is why it is replaced:

> The native backend and provider ABI support the host capabilities required by P1, and each
> mandatory capability is reachable from ordinary STARK source through a package API, typed HIR,
> provider MIR lowering and native execution on macOS, Linux and Windows.

**Until the source-to-provider lowering path exists, C7.8 has not removed P1's host-capability
precondition** — regardless of whether a provider can be executed from hand-built MIR.

**Narrowed 2026-07-30.** That path now exists. `c788_source_time_e2e.rs` compiles a `.stark`
program calling a manifest-bound function with ordinary syntax, lowers it to `Callee::Provider`,
links `stark-time-native`, executes it and asserts the printed monotonic reading is nonzero. The
general blocker — `lower_program` emitting no provider call at all — is gone.

Stated as precisely as this section's own correction demands: that path runs through the compiler
**library**. `native_build.rs` still calls plain `lower_program` and never invokes synthesis, so
`starkc build` on a package with a `provider_api` block does not yet work. Every component exists
and is tested; the driver wiring is the gap.

The precondition is **narrowed, not lifted**, and the claim above still requires *each mandatory*
capability. What remains is now specific per capability rather than architectural: `stark-env` needs
only its manifest binding and an e2e -- the recoverable-status `Err` arm lowers as of 2026-07-30
(design §16.1); `stark-file` and `stark-net` still need a resource-nominal mechanism (design §3.1). Cross-platform verification of the source path is
C7.8.7's, and one capability on one host is not it.

It does **not** claim P1 is complete, and it does **not** close Gate C7.

**Hand-authored MIR tests are not source-language capability completion.** They remain backend and
ABI evidence, and C7.8.7 must report them as such rather than counting them toward capability
coverage.

---

## 6. Gate relationship

```text
WP-C7.8 (C7.8.0 … C7.8.7)
    ↓
P1 — Native Systems Baseline  (pure-STARK JSON, HTTP/1.1 routing, three REST endpoints,
                               trap-abort operational report)
    ↓
WP-C7.5 re-opens  (steady-state runtime; debug/release ratio; interpreter/native ratio)
WP-C7.4 re-opens  (do the folding passes ever fire on realistic code?)
WP-C7.6 re-opens  (does a generated-code deficit justify opening CE6?)
    ↓
WP-C7.7 re-assessment → Gate C7 closure
```

Gate C7 remains `CANDIDATE-COMPLETE-BLOCKED-BY-P1` throughout this WP.

## 7. Concurrency with C8

C8 (semantic language services) may run in parallel — `COMPILER-ROADMAP.md` §4.3 permits it after
C2 — and is currently active (`WP-C8.0-BASELINE.md`, LSP and VS Code changes in the working tree).
Ownership boundaries, written down rather than assumed:

| Owner | Surface |
| --- | --- |
| C8 | LSP server, editor-facing compiler work, `starkc/src/lsp/`, `starkc/src/analysis*`, `editors/vscode/` |
| C7.8 | MIR runtime surface, provider ABI application, host capability lowering, backend emission, `stark-runtime`, host packages |

- **C8 must not add or modify provider ABI or MIR runtime-surface entries.** `MIR_RUNTIME_SURFACE`
  belongs to C7.8 for the duration.
- **C7.8 must not alter LSP protocol or editor-facing behaviour**, except where exposing
  already-approved diagnostics.
- **Changes to common MIR enums require coordination.** C8 compiles against `Callee` and `MirTy`
  even though it does not semantically use provider calls, so A10's added variant is a cross-track
  change even where it is not a cross-track *semantic* one.
- **Shared roadmap/state files.** No lease mechanism exists in `COMPILER-CHARTER.md` or
  `COMPILER-ROADMAP.md` today, so the operative rule is the weaker one already in use: updates to
  `COMPILER-STATE.md` are **additive to distinct sections**, never rewrites of a shared one, with
  each track appending under its own heading. A lease mechanism must be specified before it can be
  cited.

## 8. Escalations

| Class | Subject | Packet |
| --- | --- | --- |
| CE4 | First-party provider invocation model and ABI application | 1 — **DISPOSITIONED 2026-07-28** |
| CE3 | New MIR runtime surface (`0.1-A9` → `0.1-A10`) | 2 — **DISPOSITIONED 2026-07-28** |
| CE2 | STD-IO-001 drop-close versus ABI §13.2 | 3 — **DISPOSITIONED 2026-07-28** |
| CE1 | Normative surface placement for the five capabilities | 4 — **DISPOSITIONED 2026-07-29** |
| CE9 | File, environment and network trust boundaries; provider linking | 5 — **DISPOSITIONED 2026-07-29** |
| CE3 | MIR representation for package-declared host resources (Route B) | 6 — **DISPOSITIONED 2026-07-29** |

Packet 4 was dispositioned as the option requiring **no** CE1 change. That was still a CE1 decision:
the alternatives were Core changes, and declining them is the ruling. Packet 5's boundary table
admits inbound TCP explicitly rather than letting P1's requirements imply it.

## 9. Exit criteria

**Governance**

- [x] C7.8 present in `COMPILER-ROADMAP.md` and `COMPILER-STATE.md` (CD-201).
- [x] The `WP-C7.7-GATE-EXIT.md` §6 open question is answered by a recorded owner decision (CD-201).
- [x] All five packets dispositioned (CE4, CE3, CE2 2026-07-28; CE1, CE9 2026-07-29, CD-212).
- [x] C8 concurrency boundary written down (CD-201).

**Architecture**

- [ ] Provider invocation model recorded and implemented per Packet 1.
- [ ] `MIR_RUNTIME_SURFACE` amended and published per Packet 2.
- [ ] Resource representation is the approved ABI model — no Rust `Drop` on any host resource.
- [ ] Destruction is MIR-owned and verified exactly-once.
- [ ] All three ABI failure channels are distinguishable in observation and diagnostics.
- [ ] `stark-time/native/` executes unmodified.

**Capabilities**

- [ ] Arguments and environment read natively.
- [ ] File open/create/read/write to the STD-IO-001 surface, short write observable.
- [ ] Monotonic time and sleep; `stark-time/BLOCKERS.md` discharged.
- [ ] TCP bind, accept, connect, read, write.
- [ ] Every host resource is move-only, with use-after-move rejected in MIR verification.

**Quality**

- [ ] Hosted CI green on macOS, Linux and Windows.
- [ ] Formatting and strict clippy clean.
- [ ] No `Core(File, [])`-class backend refusal remains for the admitted surface.
- [ ] Runtime panics contained; no panic unwinds through generated code.
- [ ] Error categories stable and tested.
- [ ] Leak and lifecycle tests pass.
- [ ] Capability matrix updated with all five layer columns.

## 10. Closure statement

> WP-C7.8 CLOSED. First-party native host capabilities — arguments, environment, file I/O,
> monotonic time and sleep, TCP listening, accepting and stream I/O — execute natively through the
> approved Native Provider ABI v0.1 semantics, statically linked per the CE4 disposition. Host
> resources are move-only, released exactly once under verified MIR, panic-contained, and mapped
> onto the ABI's three failure channels. The admitted surface is verified on macOS, Linux and
> Windows. P1 is no longer blocked by missing native host capability. P1 is not complete and Gate
> C7 does not close.


---

## 11. WP-C7.8.8 — Source/package provider integration (CD-220)

The path from STARK source to a native provider call. **This is the critical path to P1**, and TCP
sits behind it rather than in front of it: `tcp_listener`/`tcp_stream` need the host-resource
representation Packet 6 dispositioned, which the source path needs anyway.

### The path, in outline

1. the package manifest declares required capabilities;
2. a package API declaration binds source-level functions and resource types to provider
   capabilities, symbols and provider resource names;
3. type checking resolves an ordinary package call;
4. the provider binding survives to lowering;
5. lowering creates `ValidatedProviderCall` and emits `Callee::Provider`;
6. package resource nominals lower to the MIR host-resource representation (Packet 6);
7. generated Rust uses `OwnedResourceHandle`.

**The implementation order is `WP-C7.8.8-PACKAGE-API-DESIGN.md` §16, approved as CD-225** — eight
numbered steps, and the authority for status. The outline above is the *shape* of the path and
deliberately does not renumber against it; where the two are read together, §16 governs.

Position (2026-07-30): outline step 1 done (CD-213); §16 steps 1, 3, **6 and 8 done** (step 2 for
functions only) — the monotonic clock executes from ordinary STARK source with no hand-built MIR
(`c788_source_time_e2e.rs`). Step 3 collapsed into step 2, because synthesis emits STARK source and
the ordinary front end builds the HIR (design §3.1).

§16 steps 4, 5 and 7 were blocked on a resource-nominal mechanism; **CD-234 dispositions it** (design
§3.2: a synthesized zero-variant enum, opaque structurally rather than by a checker rule), and A11's
`MirTy::HostResource` is now implemented at MIR `0.2`. What remains on that path is synthesis of the
nominals, the registry change, resolution-time construction, and the drop/close lifecycle — see A11
§8.5 and `COMPILER-STATE.md`.

### Proof order, by increasing complexity

Each is a STARK **source** program, compiled and run:

1. time;
2. args/env;
3. `File` create/write/close;
4. TCP bind/connect;
5. accept;
6. full native echo.

### What does not count

**Hand-authored MIR does not demonstrate source-language capability.** The existing e2e suites
(`a10_stark_time_e2e`, `c783_env_e2e`, `c784_file_e2e`, `c78_buffer_e2e`, `c785_time_closeout`) are
backend and ABI evidence and stay valuable as such — they are what proved the emission, ownership
and channel rules. They must not be counted toward capability completion, and C7.8.7 must keep the
two columns apart.