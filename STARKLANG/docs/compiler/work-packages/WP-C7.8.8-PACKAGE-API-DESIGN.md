# WP-C7.8.8 step 2 — package API declaration design

**Status:** rev. 5 — **fully dispositioned (CD-224, CD-225)**, and **partially implemented**.
Implementation proceeds in the order recorded in §16. Steps 1, 3, 6 and 8 are done; step 2 is done
**for functions only**; steps 4, 5 and 7 are in progress. The monotonic clock executes from ordinary
STARK source (`c788_source_time_e2e.rs`), and the resource-nominal mechanism §3.1 was blocked on is
**dispositioned in §3.2 (CD-234)** — a synthesized zero-variant enum, with A11's
`MirTy::HostResource` implemented at MIR `0.2`.

Rev. 5 adds implementation findings and corrects three sections against them. The design's decisions
all stand; what changed is *how* some of them are realised, and each correction is marked inline
rather than by silent edit:

- **§3.1** — synthesis is generated source text, not constructed HIR, and an ordinary nominal form
  cannot be used for a resource. **Resolved by §3.2 (CD-234): a zero-variant enum.**
- **§3.2** — the resource nominal is a synthesized `enum X {}`, opaque structurally rather than by a
  checker rule, with `MirTy::HostResource` owning the MIR and native representation.
- **§16.1** — what step 6 lowers. Recoverable statuses **are** lowered, via a `SwitchInt` whose
  `otherwise` edge is `Unreachable` rather than a fallback error.
- **§16.2** — the `starkc build` driver is **not** wired; the proof runs through the compiler
  library.
- **§16.3** — the one backend change required: a zero-variant enum's Rust representation carries a
  placeholder variant, because locals are default-initialised eagerly.
- **§7.2 carries a clarification**: the compiler generates the raw error enum itself, since the
  manifest deliberately carries no code→variant table. An empty vocabulary yields an *uninhabited*
  type, so `Err` cannot be constructed.
- **§10 and §11 carry correction notices**: the binding is a side table rather than an HIR field,
  and the status→`Result` construction happens in lowering rather than in the synthesized body.

**Governs:** step 2 of WP-C7.8.8 — binding source-level functions and resource types to provider
capabilities, symbols and provider resource names.
**Inputs:** Packet 4 (no Core change), Packet 5 (admission and trust boundary), Packet 6 (Route B
host-resource representation), FUTURE-FFI-001.

---

## 1. Two constraints decide most of this

**The binding cannot be STARK syntax.** `extern` is a *reserved* word (`01-Lexical-Grammar.md` §9),
and FUTURE-FFI-001 states Core v1 "exposes no public `unsafe`, raw pointer, general FFI, external
calling-convention, or arbitrary dynamic-library interface". Adding an `extern fn` item form, or any
attribute syntax, is a **grammar change to Core** — the CE1 that Packet 4 exists to avoid. So the
binding lives in package *metadata*, not in `.stark` source.

**FUTURE-FFI-001 also says what the boundary must look like**, and it is the boundary we already
have: "Host access occurs only through an approved native-provider boundary whose package/artifact
metadata identifies provider and artifact identity, origin, integrity hash and version; the
versioned ABI and supported targets". That is `ProviderMetadata` plus the registry. The design below
adds no new trust surface — it names, in the package manifest, which of the *already-validated*
declarations a package's API is bound to.

---

## 2. Where the binding lives

`starkpkg.json`, extending the `capabilities` array added in CD-213:

```json
{
  "name": "std-time",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "capabilities": ["clock"],
  "provider_api": {
    "functions": {
      "monotonic_now_ns": { "capability": "clock", "symbol": "stark_time_monotonic_now_ns" },
      "unix_now":         { "capability": "clock", "symbol": "stark_time_unix_now" }
    },
    "resources": {}
  }
}
```

and for a resource-bearing package:

```json
{
  "name": "std-net",
  "capabilities": ["tcp"],
  "provider_api": {
    "resources": {
      "TcpListener": { "capability": "tcp", "resource": "tcp_listener" },
      "TcpStream":   { "capability": "tcp", "resource": "tcp_stream" }
    },
    "functions": {
      "TcpListener::bind":   { "capability": "tcp", "symbol": "stark_tcp_listener_bind" },
      "TcpListener::accept": { "capability": "tcp", "symbol": "stark_tcp_listener_accept" },
      "TcpStream::connect":  { "capability": "tcp", "symbol": "stark_tcp_stream_connect" },
      "TcpStream::read":     { "capability": "tcp", "symbol": "stark_tcp_stream_read" },
      "TcpStream::write":    { "capability": "tcp", "symbol": "stark_tcp_stream_write" }
    }
  }
}
```

**Closes are not bound and must not be.** `stark_tcp_listener_close` is `is_close_for` in the
provider's metadata; MIR's `Drop` terminator selects it. A package naming a close in `functions`
would create a second path to a close the ABI already owns exactly once — rejected at manifest
validation.

**Every `capability` named here must appear in the package's own `capabilities` array.** Packet 5
admits providers only through declared capability requirements, and a binding is a use.

---

## 3. How the items reach source

**The compiler synthesizes them.** The package writes no declaration for a bound function; the
manifest entry *is* the declaration, and the compiler injects a corresponding item into the
package's module namespace before name resolution.

This follows an existing Core precedent rather than inventing one: PRINT-DISPLAY-001 already
describes `print`/`println` as "implementation-provided generic functions, **not** syntax hooks",
resolved by ordinary means. A provider-bound function is the same kind of thing.

Consequences, all of which are the point:

- source calls them with **ordinary syntax** — `Instant::now()`, `TcpStream::connect(addr)`;
- name resolution, visibility, and type checking need **no special case**: by the time they run,
  these are ordinary items with ordinary signatures;
- there is **no bodyless-`fn` grammar**, so `02-Syntax-Grammar.md` is untouched.

The alternative — the package declaring a signature that the compiler then replaces — was rejected:
it requires a bodyless function form (CE1 grammar) and creates two sources of truth for the
signature, which can disagree.

### 3.1 The mechanism is source text, and functions only (implementation finding, step 3)

Injection is implemented as **generated STARK source** (`starkc/src/provider_synth.rs`), parsed by
the ordinary front end. This is forced rather than chosen: every name in HIR is a `Span` into a
`SourceFile` (`hir::FnSig::name`, `ItemKind::Struct::name`), so an item constructed directly would
carry either a fabricated span — surfacing in every diagnostic that touches it — or a second name
representation threaded through the whole front end. Generating source makes §3's claim *literally*
true instead of approximately: the items are ordinary because the ordinary parser built them.

A generated body is `panic("provider binding not lowered")` — type `!`, so it satisfies any return
type — with **no trailing semicolon**, since `panic(…);` is a statement and would give the block
value `Unit`. Lowering never reaches the body; it emits `Callee::Provider` from the binding (step 6).

**Resource nominals cannot use this mechanism.** Every source form that declares a nominal is
constructible — `struct S;` and `struct S {}` both admit a value at a use site — and a host resource
must be opaque (§6, A11 §6). Generating source for one would hand programs a way to forge a handle
no provider produced, and `from_raw_checked` would not catch it: the `resource_type` would be
whatever the forger wrote.

`synthesize()` therefore **refuses** any signature carrying a receiver or a resource type, rather
than emitting something weaker. The consequence for §16's order: steps 1–3 and 8 are reachable for
`clock` (scalar-only, and both its signatures compile), and steps 4–7 needed a resource-nominal
mechanism decided first. This is why CD-225 put the monotonic-time proof before resource
capabilities; the finding confirms that order was load-bearing, not merely convenient.

### 3.2 RESOLVED (CD-234): the nominal is a synthesized zero-variant enum

The owner dispositioned §3.1's gap. Neither option §3.1 named was taken:

- a **compiler-injected spanless item** is rejected — it reintroduces the fabricated spans and second
  name representation that generating source exists to avoid;
- an **ordinary struct plus a "do not construct this one" marker** is rejected — soundness would rest
  on a checker rule every future construction path must remember, which is the same hidden special
  case Packet 6 already rejected for the MIR representation question.

Instead the nominal is generated as:

```stark
enum TcpStream {}
```

A **zero-variant enum** is opaque *structurally*, not by prohibition. It has no fields, no variants,
no constructor expression, and no pattern capable of manufacturing a value — there is simply no
variant to name and no struct-literal form. So it yields a normally parsed item with real source
spans and ordinary name resolution and type checking, and it needs no marker that any consumer could
forget.

**The soundness condition attached to the approval:**

> A synthesized zero-variant enum may provide the source-level nominal identity, but a
> provider-bound instance of that nominal lowers to `MirTy::HostResource` and must never receive the
> backend representation or default-initialisation behaviour of an ordinary zero-variant enum.

The required split:

```text
ordinary zero-variant enum   →  ordinary enum MIR/backend representation
                             →  placeholder permitted where local-init machinery needs one

provider-bound nominal       →  MirTy::HostResource
                             →  OwnedResourceHandle representation
                             →  no placeholder, no default value, slot begins dead
```

A `HostResource` local becomes live **only** through a successful `HandleOut`, a move from an
already-live resource, or an argument/return carrying one. No declaration, default initialisation,
aggregate construction or backend placeholder may make it live.

Drop flags still decide whether a *live* resource is closed, but they must not be used to excuse a
forged placeholder existing. The guarantee is stronger than that: **a dead host-resource slot
contains no semantically valid STARK value, and native code must never read or close it.** CD-234 is
explicit that a placeholder-backed host-resource local is forbidden even where current drop flags
appear to make it unreachable.

This is a **CE3 clarification to A11**, not a new Core feature and not a reversal of the decision
against marked ordinary structs. The zero-variant enum carries no hidden resource semantics through
every compiler consumer; the provider binding causes lowering to replace its representation with the
explicit `HostResource` form at the established boundary.

Implementation status is in `mir-amendment-A11-host-resources.md` §8.5. The type, its identity, the
codegen rule and CD-234's refusals are done (`a11_host_resource.rs`); synthesis of the nominals, the
registry change, resolution-time construction, and the drop/close lifecycle are not.

---

## 4. The signature contract, which is the hard part

A STARK signature and an `AbiParam` list are different shapes. The provider says
`(BufferIn, ScalarOut(U64), ScalarOut(Bool)) -> ProviderStatus`; the package wants
`fn env(name: &str) -> Result<Option<String>, ProcessError>`.

**Proposal: the compiler derives the STARK signature from the validated `AbiParam` list**, by a
fixed, total rule — so the manifest names a symbol and nothing more, and the two can never disagree
because there is only one of them.

| `AbiParam` | STARK signature position |
| --- | --- |
| `ScalarIn(T)` | parameter, in declaration order |
| `BufferIn` | parameter, `&[UInt8]` |
| `BufferInOut` | parameter, `&mut [UInt8]` |
| `ScalarInOut(T)` | parameter, `&mut T` |
| `HandleBorrowed{r}` | parameter, `&R` where `R` is the bound nominal |
| `HandleConsumed{r}` | parameter, `R` by value |
| `ScalarOut(T)` | **result component** |
| `HandleOut{r}` | **result component**, `R` |
| `ProviderStatus` | the `Result` itself |

The return type is then:

- no out-slots → `Result<Unit, E>`;
- one out-slot → `Result<T, E>`;
- several → `Result<(T1, T2, …), E>` in declaration order.

`E` is the package's declared error type, named once per capability:

```json
"errors": { "clock": "TimeError", "process.env": "ProcessError" }
```

`stark_time_unix_now` therefore derives `fn unix_now() -> Result<(Int64, UInt32), TimeError>` —
two out-slots, a tuple, in declared order. **That is the signature whose absence caused CD-219's
defect**: a hand-written mirror said one slot, and only execution disagreed. Derivation removes the
class.

### DISPOSITION (CD-224): derived, and the invariant is standing

> **There is one authoritative callable signature: validated provider metadata. The package
> declaration exposes and names that callable surface but does not mirror its physical or ownership
> signature.**

The package declaration identifies:

- the **capability**;
- the **provider symbol**;
- the **public package function or method identity**;
- the **associated package resource type**, where applicable;
- the **error/status type mapping**, where it is not derivable from provider metadata.

It does **not** repeat ABI parameter types or ownership modes. There is no second copy to drift.

### Why derivation rather than declaration

A declared signature is a second copy of information the ABI already carries, and the two drift.
CD-219 is the worked example: the registry's copy of `unix_now` had one out-slot where the crate
declared two, metadata validation could not see it (a one-slot declaration is internally
consistent), and the provider's null check caught it at runtime. Deriving means there is nothing to
drift *from*.

**The cost, stated:** package authors cannot choose ergonomic signatures. `env` derives as
`fn env(name: &[UInt8]) -> Result<(Bool, UInt64), ProcessError>`, not the
`Result<Option<String>, ProcessError>` a user wants. That is deliberate — the derived items are the
**raw binding layer**, and the ergonomic API is ordinary STARK written *over* it, in the package,
with no compiler involvement. Packet 4 already put `read_to_end`/`write_all` in exactly that
position.

---

## 5. What HIR carries (Packet 6)

Each synthesized item carries its binding, and the binding is the reason the item exists:

```text
HIR fn item      -> ProviderBinding { capability, symbol }
HIR nominal item -> HostResourceBinding { capability, resource_name }
```

A nominal with a `HostResourceBinding` is a **host resource** — Packet 6's Route B — and lowers to
the MIR host-resource form rather than to a struct. It has no fields to lower, and any attempt to
declare fields for it in source is rejected at manifest validation (the nominal is synthesized, so
the package cannot declare one).

Name resolution and type checking treat both as ordinary items. The binding is *carried*, never
*consulted*, until lowering — which is what keeps steps 3 and 4 free of special cases.

---

## 6. Validation, at manifest load

All of it before any code is generated, per Packet 5's "reject before backend invocation":

1. every `capability` named in `provider_api` is in the package's `capabilities`;
2. every `symbol` exists in the selected provider's validated metadata for that capability;
3. every `resource` exists in that provider's declared `resource_types`;
4. no bound symbol is an `is_close_for` function;
5. every `HandleBorrowed`/`HandleConsumed`/`HandleOut` in a bound function's declaration names a
   resource the package also binds — otherwise the derived signature would reference a nominal that
   does not exist;
6. each capability with a bound function has an `errors` entry;
7. no two bindings name the same item path.

Failures are `BuildCommandError::Capability`, the class CD-213 added.

---

## 7. Dispositioned (CD-225)

### 7.1 Associated placement — approved

Resource operations are associated with their resource type:

```stark
TcpListener::bind_raw(addr)
listener.accept_raw()
TcpStream::connect_raw(addr)
stream.read_raw(buf)
```

It matches Core `File`, keeps operations discoverable beside the type, and avoids an arbitrary
free-function namespace.

**Associated placement is API organisation only.** It must not imply structural methods, fields,
constructors, or ordinary nominal representation for a host resource. A host resource has no fields
to reach and no constructor to call; §6's opacity rules are unchanged by where its operations are
named.

### 7.2 Status mapping stays in ordinary STARK — approved

The manifest carries **only the minimum raw error identity** needed to derive the binding. It does
not carry a status-code→public-variant table.

The synthesized raw API exposes the validated provider status vocabulary through a **package-private
raw error type**. The package then translates that raw result into its public `IOError`,
`ProcessError`, `NetworkError` or other ergonomic type using ordinary STARK.

The division this preserves:

- **the compiler** validates physical status declarations and produces the raw typed result;
- **package code** owns public error semantics, grouping and convenience behaviour.

> **CLARIFIED at implementation (2026-07-30): the compiler generates the raw error enum itself.**
> This section says the manifest carries "only the minimum raw error identity" and **no**
> status-code→variant table, and that the compiler "produces the raw typed result". Those leave no
> way for a *package-declared* enum to tell the compiler which variant means status 3 — so the raw
> type has to be derived from the same validated vocabulary the emitter dispatches on. The division
> is unchanged: the compiler owns the raw type, package code maps it to a public `IOError` or
> `ProcessError` in ordinary STARK.
>
> One variant per declared status code, named by the vocabulary and ordered by code, so generated
> source and the variant indices are a function of the vocabulary alone.
>
> **An empty vocabulary generates an uninhabited enum** — `enum RawTimeError { }` — and that is the
> point rather than a degenerate case. `clock` declares no recoverable status, so `Err` **cannot be
> constructed**, and the type system states what this section's last paragraph states in prose.
>
> Two capabilities may share a raw error type while they agree on every code; a disagreement is
> refused, since it would give one status code two meanings in one enum.
>
> **Declaring status 0 as a recoverable error is refused**, because 0 is success (ABI §11).
> Tolerating it would fail twice without naming the mistake: `ProviderBindingPlan::classify` tests
> success first, so the declaration would be silently shadowed, and §16.1's `SwitchInt` would carry
> two arms for the same value.
>
> A capability with **no vocabulary entry at all** is also refused rather than defaulted to empty.
> "No recoverable statuses" and "nobody supplied the vocabulary" are different claims, and
> defaulting would quietly convert the second into the first.
>
> **The variant name is the vocabulary name's final segment.** A vocabulary names a status with the
> *package-facing* error it corresponds to, and the first-party registry writes those qualified —
> `stark-env` declares code 1 as `ProcessError::InvalidName`. `ProcessError` is, by this section's own
> account, the **public** type package code maps the raw result *to*, so the raw variant is
> `InvalidName`. Emitting the qualified form verbatim would generate
> `enum RawEnvError { ProcessError::InvalidName, … }`, which does not parse.
>
> Because that is interpretation rather than data, the derived name is **validated**: a final segment
> that is not a legal STARK identifier is refused at synthesis, where the offending vocabulary entry
> can be named, instead of producing source that fails to parse with nothing pointing back to its
> cause.

**Only declared recoverable statuses reach the mapping layer.** Ordinary STARK cannot see, catch, or
reinterpret a contract violation or a host failure — those channels abort, and no package code runs
on them. That is Packet 1 §1.2's three-channel separation surviving into the source language: a
package chooses how to *name* a recoverable error, never whether a violation is recoverable.

### 7.3 Synthesized items are package-private — approved

Every synthesized provider item and synthesized host-resource nominal is **package-private**. A
package must explicitly expose a curated wrapper API; appearing in `provider_api` never makes a
function public.

So an application calls `TcpStream::connect(addr)`, not `TcpStream::connect_raw(addr)` — unless the
package author deliberately re-exports the raw layer.

**The compiler rejects any attempt to make provider crate names, raw ABI symbols or physical ABI
forms application-visible**, which is §8's rule enforced at the visibility boundary rather than only
at the manifest.

---

## 8. The application-source rule

**Application source calls package APIs and never names a provider crate, a capability, or an ABI
symbol.** A program says `TcpStream::connect(addr)`. It does not name `stark-net-native`, `"tcp"`,
or `stark_tcp_stream_connect`.

**The rule, as it must be enforced:**

> Application source and ordinary package APIs may name **capabilities and package declarations
> only**. Provider crate identities, raw symbols and physical ABI parameter forms are not part of
> application-visible STARK source.

This is a validation rule, not a convention:

- a `provider_api` block is only honoured in a package that **declares** the capability, so an
  application cannot bind anything it did not first require;
- nothing in `.stark` source can name a symbol — §1 established that there is no syntax for it;
- the derived items carry their binding in HIR, so an application observes a signature, never a
  linkage detail.

The consequence worth stating: **a package is the only place a provider binding can exist**, which
is what makes the capability set of a program equal to the union of its packages' declarations.

## 9. Worked declarations

Each shows the manifest entry and the STARK signature the compiler derives from validated metadata
(§4). No signature is written by hand anywhere.

### 9.1 Monotonic time

```json
{ "name": "std-time", "capabilities": ["clock"],
  "provider_api": {
    "errors": { "clock": "TimeError" },
    "functions": {
      "Instant::now_ns": { "capability": "clock", "symbol": "stark_time_monotonic_now_ns" },
      "SystemTime::unix_now": { "capability": "clock", "symbol": "stark_time_unix_now" }
    } } }
```

Provider declares `[ScalarOut(U64)]` and `[ScalarOut(I64), ScalarOut(U32)]`, so:

```stark
fn now_ns() -> Result<UInt64, TimeError>;
fn unix_now() -> Result<(Int64, UInt32), TimeError>;
```

The tuple is not a design choice — it is the provider's two out-slots, in declared order. **This is
the shape CD-219's hand-written mirror got wrong.**

### 9.2 Environment lookup

```json
"functions": {
  "env::var_len":  { "capability": "process.env", "symbol": "stark_env_var_len" },
  "env::var_fill": { "capability": "process.env", "symbol": "stark_env_var_fill" }
}
```

```stark
fn var_len(name: &[UInt8]) -> Result<(Bool, UInt64), ProcessError>;
fn var_fill(name: &[UInt8], out: &mut [UInt8]) -> Result<UInt64, ProcessError>;
```

Deliberately unergonomic: this is the **raw binding layer**. The package writes
`fn var(name: &str) -> Result<Option<String>, ProcessError>` over it in ordinary STARK, and that is
the function an application calls.

### 9.3 `File` — a **Core** resource, bound by source-function lowering only

**DISPOSITIONED (CD-224): Core `File` and package resources are distinct binding mechanisms**, and
neither changes the other's authority.

`File` is normative Core (STD-IO-001). Its resource binding stays compiler/spec-owned in the
built-in registry —

```text
file → CoreType::File          (compiler-owned; NOT declarable by any package)
```

— so `std-file`'s manifest binds **functions only**, and declares no `resources` block at all. A
package cannot redefine what `File` is:

```json
{ "name": "std-file", "capabilities": ["filesystem"],
  "provider_api": {
    "errors": { "filesystem": "IOError" },
    "functions": {
      "File::open_raw":     { "capability": "filesystem", "symbol": "stark_file_open" },
      "File::create_raw":   { "capability": "filesystem", "symbol": "stark_file_create" },
      "File::read_raw":     { "capability": "filesystem", "symbol": "stark_file_read" },
      "File::write_raw":    { "capability": "filesystem", "symbol": "stark_file_write" },
      "File::complete_raw": { "capability": "filesystem", "symbol": "stark_file_complete" }
    } } }
```

Derived, with `File` resolving to the **Core** nominal:

```stark
fn open_raw(path: &[UInt8]) -> Result<File, IOError>;
fn read_raw(f: &File, out: &mut [UInt8]) -> Result<(UInt64, Bool), IOError>;
fn write_raw(f: &File, data: &[UInt8]) -> Result<UInt64, IOError>;
fn complete_raw(f: &File) -> Result<Unit, IOError>;
```

`stark_file_close` is absent and must be (§2): it is `is_close_for`, and MIR's `Drop` terminator
owns it.

**So this example demonstrates source-function-to-provider lowering over an existing Core resource** —
step 5 of WP-C7.8.8 — and deliberately *not* package nominal declaration, which §9.4/§9.5 covers.

**The two authorities, and what they share.** Both lower to the same explicit MIR host-resource
representation (A11); what differs is who may establish the binding:

| | Core resource | Package resource |
| --- | --- | --- |
| binding authority | compiler / specification | package declaration |
| declared where | `ResourceRegistry::builtin()` | `provider_api.resources` |
| validated against | provider metadata | provider metadata |
| a package may redefine it | **no** | yes, it owns it |
| MIR representation | A11 host-resource form | A11 host-resource form |

Packet 4 is preserved exactly: `File` stays normative Core, TCP stays package-owned, and neither
mechanism reaches into the other.

### 9.4 `TcpListener`, 9.5 `TcpStream`

```json
{ "name": "std-net", "capabilities": ["tcp"],
  "provider_api": {
    "errors": { "tcp": "NetworkError" },
    "resources": {
      "TcpListener": { "capability": "tcp", "resource": "tcp_listener" },
      "TcpStream":   { "capability": "tcp", "resource": "tcp_stream" }
    },
    "functions": {
      "TcpListener::bind_raw":   { "capability": "tcp", "symbol": "stark_tcp_listener_bind" },
      "TcpListener::accept_raw": { "capability": "tcp", "symbol": "stark_tcp_listener_accept" },
      "TcpStream::connect_raw":  { "capability": "tcp", "symbol": "stark_tcp_stream_connect" },
      "TcpStream::read_raw":     { "capability": "tcp", "symbol": "stark_tcp_stream_read" },
      "TcpStream::write_raw":    { "capability": "tcp", "symbol": "stark_tcp_stream_write" }
    } } }
```

```stark
fn bind_raw(addr: &[UInt8]) -> Result<TcpListener, NetworkError>;
fn accept_raw(l: &TcpListener) -> Result<TcpStream, NetworkError>;
fn connect_raw(addr: &[UInt8]) -> Result<TcpStream, NetworkError>;
fn read_raw(s: &TcpStream, out: &mut [UInt8]) -> Result<UInt64, NetworkError>;
fn write_raw(s: &TcpStream, data: &[UInt8]) -> Result<UInt64, NetworkError>;
```

`accept_raw` is the shape nothing else has: it **borrows** a listener and **produces** a stream, so
one call keeps one resource and creates another. Both closes are absent; both are `is_close_for`.

Packet 5's inbound rule survives into the source language unchanged: a listener exists only because a
program called `bind_raw` with an address it supplied.

## 10. Typed HIR representation

```text
HirItem::Fn      + ProviderBinding      { capability, symbol }
HirItem::Nominal + HostResourceBinding  { capability, resource }
```

Both are **carried, not consulted**, until lowering. Name resolution and type checking see ordinary
items with ordinary signatures, which is what keeps steps 3 and 4 of WP-C7.8.8 free of special cases.

> **CORRECTED at implementation (2026-07-30).** The binding is **not** a field on the HIR item. §3.1
> forced synthesis to be generated source, so HIR is built by the ordinary parser and carries no
> provider vocabulary at all — adding a `ProviderBinding` field would mean the parser producing a
> field it has no input for. The binding rides in a **side table** instead
> (`SynthesizedLayer::bindings`, item path → `(capability, symbol)`), which lowering resolves to
> item ids once, up front (`mir/provider_lower.rs`).
>
> "Carried, not consulted" survives intact, and is in fact stronger: the front end does not merely
> decline to consult the binding, it never sees one. The `HostResourceBinding` row is unimplemented
> — the `HostResourceBinding` row is dispositioned by §3.2 (CD-234) and partly implemented.

A nominal with a `HostResourceBinding` is **opaque in the type system**: it has no fields, so field
access fails by ordinary name resolution rather than by a special rule, and it is never `Copy` or
`Clone`, so ordinary move checking rejects reuse after a consuming call.

## 11. MIR lowering contract

| HIR | MIR |
| --- | --- |
| call to a `ProviderBinding` fn | resolve → `ValidatedProviderCall`, intern, emit `Callee::Provider` |
| a `HostResourceBinding` nominal type | `MirTy::HostResource { nominal, provider, resource }` (A11) |
| the fn's `Result` return | the call's `UInt32` status destination, plus the out-slot destinations §4 derived |
| a host-resource local going out of scope | `Terminator::Drop` → the recorded `ValidatedProviderClose` (A11 §5) |

The status→`Result` construction is the **binding layer's**, not MIR's: MIR carries the raw status,
and the derived item's body — synthesized alongside its declaration — performs the three-channel
dispatch A10 §5 specifies. That keeps channel policy in one place and out of the type system.

> **CORRECTED at implementation (2026-07-30).** The synthesized body cannot do this, and the reason
> is structural rather than incidental. For the body to dispatch on a status, the body would have to
> *contain* the provider call — and the call is what lowering emits at the **call site** from the
> binding. A body that also called the provider would mean either two calls or a body that lowers to
> the very thing it is supposed to wrap. So the body stays `panic(…)` and is never lowered (§3.1),
> and the `Result` is constructed **in lowering**, immediately after the call terminator, from the
> out-slot locals.
>
> Channel policy is still in one place, just not the place this paragraph named: the **emitter**
> owns it (`emit_provider.rs` — slots written back only on status zero, declared codes matched,
> undeclared codes aborted). Lowering builds the `Ok` arm from the slots; for a capability with a
> non-empty status vocabulary it builds the `Err` arm from §7.2's generated raw error enum (§16.1).
>
> The third row of the table above is therefore right about *what* MIR carries — the `UInt32` status
> destination plus out-slot destinations — and wrong only about who turns it into a `Result`.

## 12. Diagnostics

All at manifest load, before any code is generated, as `BuildCommandError::Capability`:

| Case | Diagnostic |
| --- | --- |
| symbol outside its declared capability | names the symbol, the capability it was bound to, and the capability the provider declares for it |
| resource absent from provider metadata | names the resource and lists the provider's declared resource types |
| two nominals bound to one resource | names both nominals and the resource; a warning is insufficient — §13.3 explains |
| one nominal bound through incompatible providers | names the nominal and both providers |
| capability used but not declared | names the capability and points at the manifest's `capabilities` array |
| binding a close | names the symbol and its `is_close_for` resource |
| derived signature references an unbound resource | names the function, the resource, and that the package must bind it |

Type-level failures — an ordinary struct where a host resource is required, or projection of a host
resource — are ordinary type errors, reported by the existing checker with no provider vocabulary in
them.

## 13. Negative cases

Each is a *test*, and the phrasing states what would happen without the check.

**13.1 Function binds to a symbol outside its declared capability.** `"File::open_raw"` bound with
`"capability": "clock"`. Rejected: the symbol is not in the clock provider's declarations. Without
the check the build would select the wrong provider and link a function whose signature happens to
match.

**13.2 Resource name absent from provider metadata.** `"resource": "directory"` where the provider
declares only `"file"`. Rejected at load — otherwise A11's `HostResource` would carry a resource name
with no §7 id behind it, and `from_raw_checked` would have nothing to validate against.

**13.3 Two package nominals bind to one provider resource.** `TcpStream` and `Socket` both bound to
`"tcp_stream"`. **Rejected, not warned.** They would be distinct STARK types that are the same
resource at the boundary: a `Socket` would satisfy a `TcpStream` parameter dynamically while failing
statically, and each would record its own close for one resource — breaking exactly-once.

**13.4 One nominal binds through incompatible providers.** `TcpStream` bound in two packages, to
different providers. Rejected: the canonical identity (A11 §5) differs, so two "same" types would not
compare equal, and a value from one could reach the other's close.

**13.5 Derivation failures — replacing the withdrawn signature-mismatch case (CD-224).** A package
declares no signature, so disagreement is structurally impossible. What *can* fail is derivation
itself, and each failure is its own diagnostic:

- **13.5a** the `AbiParam` sequence cannot be mapped to an admitted STARK package signature;
- **13.5b** an `AbiParam` form is unsupported for source lowering;
- **13.5c** the return/error shape cannot be derived unambiguously;
- **13.5d** the bound symbol's derived ownership form conflicts with the package API category — e.g.
  a `HandleConsumed` receiver bound as a method on a type the call would consume, where the package
  declared it as a borrowing operation;
- **13.5e** the provider's status binding cannot be represented by the declared package error type;
- **13.5f** two ABI-distinct functions derive to an ambiguous package API declaration.

**The residual risk is moved, not removed.** A provider crate whose metadata drifts from its own
`extern "C"` definitions is still possible; that is each provider crate's own test's job, and every
first-party provider has one. What derivation eliminates is the *compiler-side* mirror that
CD-219 proved drifts.

**13.6 Package uses a capability it did not declare.** A `provider_api` entry naming `"tcp"` in a
package whose `capabilities` omits it. Rejected — Packet 5 admits providers only through declared
requirements, and a binding is a use.

**13.7 Ordinary struct passed where a host resource is required.** An ordinary `struct Handle {}`
passed to `read_raw(s: &TcpStream, …)`. An ordinary type error: the two are different nominals. No
provider vocabulary appears in the message.

**13.8 Host resource structurally constructed or projected.** `TcpStream { }` or `s.fd`. Rejected by
name resolution and type checking before MIR — the nominal has no fields and no constructor — and by
A11 §6's verifier rules if MIR is ever constructed directly. Two independent layers, deliberately.

## 14. Package graph and build key

**Bindings enter the graph with their package.** `PackageGraph` already loads every manifest; the
`provider_api` block is parsed with the rest and validated (§6) after provider selection, since
validation needs the selected provider's metadata.

**Bindings enter the build key.** A changed binding changes the generated code, so the key must
change or a stale artifact is served. The key must cover, in deterministic order:

- each bound item path → `(capability, symbol)`;
- each bound nominal → `(capability, resource)`;
- the `errors` map;
- the resolved provider identity and semver for every capability used.

The last is what makes a provider *upgrade* invalidate the cache: the same binding against different
metadata is a different program.

Sorted by item path, so manifest key order cannot reach the key — the same property CD-213 gave the
capability list and CD-205 gave the status vocabulary.

## 15. What this does not decide

Dynamic loading, capability sandboxing, allowlists and deployment policy remain deferred (Packet 5).
Nothing here changes the ABI, the runtime surface, or any Core specification document.


---

## 16. Implementation order (CD-225)

1. manifest parsing and validation for `provider_api` — **done, CD-226**
   (`package.rs`; `c788_provider_api_manifest.rs`);
2. synthesis of private package items and resource nominals — **functions done, CD-227**
   (`provider_derive.rs`, `provider_synth.rs`; `c788_derive.rs`, `c788_synth.rs`);
   **resource nominals dispositioned** by §3.2 (CD-234), synthesis of them not yet written;
3. typed HIR bindings — **done, CD-228**, subsumed by step 2: synthesis emits source, so the front
   end builds the HIR itself and the binding rides alongside as `SynthesizedLayer::bindings`;
4. resource-name-to-nominal registry — **unblocked by §3.2**; A11's type exists, the registry change does not yet;
5. resolution-time construction of `MirTy::HostResource` — **unblocked by §3.2**; the type and its refusals exist (A11 §8.5), construction does not yet;
6. `Callee::Provider` lowering — **done for scalar signatures, CD-229**
   (`mir/provider_lower.rs`, `mir/lower.rs`); resource and recoverable-status forms refused
   explicitly, see §16.1;
7. close-arena population and verifier rules — **partly done**: `MIR-0026` (nothing manufactures a resource) landed; drop-flag and close-arena rules did not;
8. **source-level monotonic-time proof** — **DONE, CD-229** (`c788_source_time_e2e.rs`): a `.stark`
   program calls the bound function with ordinary syntax, is compiled by the ordinary front end,
   lowered to `Callee::Provider`, linked against `stark-time-native`, executed, and the printed
   nanosecond reading is asserted nonzero. Through the compiler **library**; the `starkc build`
   driver is not yet wired (§16.2).

Numbering is fixed; steps are annotated rather than renumbered, because CD-225 approved this order
by number.

> **Decision-ID correction (2026-07-30).** The commits landing steps 3, 6 and 8 were written with
> subjects `CD-196` and `CD-197`, which are **already taken** — CD-196 is "WP-C7.8 REVISE" (4419d6c)
> and CD-197 is "Packet 3 dispositioned under CE2" (9aa7482). Their correct identities are
> **CD-228** (step 3, commit cdba7c8) and **CD-229** (steps 6 and 8, commit ee85652), and this
> document is the authority for that mapping.
>
> The commit subjects are **not** rewritten: they are pushed, a parallel session works the same
> repository, and force-pushing shared history to fix a label risks losing that session's work —
> a worse outcome than a subject line that needs this note to read correctly.

**TCP is not first, and neither is `File`.** The first acceptance test compiles an ordinary STARK
source call through package resolution, typed HIR and `Callee::Provider`, then links and executes
the time provider — **with no hand-built MIR**. Time has no resource, no buffer and one out-slot, so
a failure in that test is a failure in the source path itself rather than in anything it carries.

§3.1's finding makes that ordering load-bearing rather than convenient: steps 4–7 all touch resource
nominals, which had no mechanism until CD-234 (§3.2), while step 8's target needs none. The remaining path to the
proof was therefore **step 6 alone** — lowering a call to a synthesized item into `Callee::Provider`,
which the emitter and linker already execute (`a10_stark_time_e2e.rs`, from hand-built MIR). That is
now closed, and **the source path is proven end to end for a scalar capability**.

### 16.1 What step 6 lowers, and what it refuses

Lowering is hooked at `Res::Item` in `lower_call`, after name resolution, type checking and borrow
checking have all seen an **ordinary function**. That placement is the design's claim made
operational: nothing before lowering knows a provider exists.

The emitted shape follows what `emit_provider.rs` already does, rather than a parallel convention:

| `AbiParam` | lowered as |
| --- | --- |
| `ScalarIn`, `ScalarInOut`, `BufferIn`, `BufferInOut` | the STARK call's own argument, in order |
| `ScalarOut(t)` | a caller-owned local, zero-initialised, passed as `&mut` |
| `HandleBorrowed`, `HandleConsumed`, `HandleOut` | **refused** — §3.1 |

The call's `dest` receives the raw `ProviderStatus` code, **not** the STARK value; the emitter writes
out-slots back only on status zero and aborts on any undeclared code. So the `Result` is built after
the call from the slots: no slots → `Ok(Unit)`, one → `Ok(v)`, several → `Ok((v1, …))`.

**One refusal remains: a resource in any position, per §3.1.**

**Recoverable statuses are lowered (2026-07-30).** A capability with a declared vocabulary gets a
`SwitchInt` on the status: the zero arm builds `Ok` from the out-slots, one arm per declared code
builds `Err(RawE::V)` from §7.2's generated enum, and `otherwise` is **`Unreachable`** — never a
fallback error. An undeclared nonzero code has already aborted inside the emitted call, so no value
reaches that edge, and a `_ =>` arm mapped to some generic package error is exactly the channel
collapse Packet 1 §1.2 forbids. Each declared code gets its **own** block, because each constructs a
different variant.

If the vocabulary is non-empty and no error mapping reached lowering, the call is still refused:
emitting `Ok` regardless would turn a declared recoverable failure into a successful call returning
an unwritten slot.

`clock` has an empty vocabulary, so no branch is emitted at all — every nonzero status is a contract
violation the emitter aborts on, control reaching the code after the call *means* status zero and
written slots, and `Ok` is the only reachable outcome. That is a fact about the emitted Rust rather
than an optimistic assumption, and §7.2's uninhabited `RawTimeError` states it in the type system too.

### 16.2 The driver is not wired yet

The step 8 proof drives the pipeline through the compiler **library**: it parses, resolves,
typechecks, calls `lower_program_with_providers`, emits, links and runs. `native_build.rs` — what
`starkc build` actually uses — still calls plain `lower_program` and never invokes synthesis at all.
So a package with a `provider_api` block in its manifest does **not** build from the command line.

Recorded rather than glossed, because CD-220 had to correct an over-claim of exactly this shape:
"executes natively" had meant *hand-built MIR runs*, which was true and was not what a reader took
it to mean. "Reachable from STARK source" must not now quietly mean *reachable if you drive the
compiler as a library*.

What remains is integration, not design — every component exists and is tested:

1. read `provider_api` from the manifest (`package.rs`, done);
2. `derive_all` against the selected provider's validated metadata (done);
3. `synthesize` the raw layer (done);
4. prepend it to the package's compilation unit — **the only genuinely new piece**, and the place to
   decide how a generated unit reports spans in diagnostics;
5. `ProviderLowering::build` from the bindings (done);
6. call `lower_program_with_providers` instead of `lower_program`.

### 16.3 One backend change was required

An uninhabited STARK enum previously had no generated-Rust representation: `default_value_expr`
rejected a zero-variant enum, because it has no value to default a local to. That surfaced the moment
§7.2's uninhabited raw error type met a program binding it (`Err(e) => …`), since the CFG dispatch
loop default-initialises every local **eagerly**.

An aborting expression does not work here — unlike the `FnPtr` sentinel it sits beside, which is a
named aborting *function*, an aborting expression would fire on function entry rather than on misuse.
So a zero-variant enum's Rust declaration now carries a single placeholder variant, and that is its
default.

The placeholder is invisible to STARK: the front end sees zero variants, so no STARK program can
construct or match one, and MIR never reads a local of the type. It exists solely so the eager
default-init has something to write.
