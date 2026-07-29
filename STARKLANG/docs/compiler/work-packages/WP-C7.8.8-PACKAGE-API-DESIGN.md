# WP-C7.8.8 step 2 — package API declaration design

**Status:** rev. 4 — **fully dispositioned (CD-224, CD-225)**. No open questions. Implementation
proceeds in the order recorded in §16. No parser, HIR or MIR implementation until the
declaration shape and A11's MIR-version disposition are both approved.
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

1. manifest parsing and validation for `provider_api`;
2. synthesis of private package items and resource nominals;
3. typed HIR bindings;
4. resource-name-to-nominal registry;
5. resolution-time construction of `MirTy::HostResource`;
6. `Callee::Provider` lowering;
7. close-arena population and verifier rules;
8. **source-level monotonic-time proof** before any resource capability.

**TCP is not first, and neither is `File`.** The first acceptance test compiles an ordinary STARK
source call through package resolution, typed HIR and `Callee::Provider`, then links and executes
the time provider — **with no hand-built MIR**. Time has no resource, no buffer and one out-slot, so
a failure in that test is a failure in the source path itself rather than in anything it carries.
