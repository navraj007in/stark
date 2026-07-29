# WP-C7.8.8 step 2 — package API declaration design

**Status:** DRAFT, for owner disposition. No implementation until ruled on.
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

## 7. Open questions for the owner

**7.1 Item paths.** `"TcpListener::bind"` implies associated-function placement on a synthesized
nominal. Is that the intended surface, or should bound functions be free functions in a module
(`tcp::bind(listener)`)? Associated placement reads better and matches STD-IO-001's `File`; free
functions are simpler to synthesize.

**7.2 Error mapping.** §4 derives `Result<_, E>` with `E` named per capability, but the code→variant
mapping is Packet 1 §1.2's "package binding layer". Where does *that* live — a manifest table, or
ordinary STARK the package writes over the raw layer? A manifest table is checkable; STARK is more
flexible and keeps the compiler out of package semantics.

**7.3 Visibility.** Are synthesized items public by default, or private so the package must
re-export a curated surface? Private-by-default keeps the raw layer out of a package's public API,
which suits §4's "raw binding layer" framing.

**7.4 `File`.** Core already specifies `File` (STD-IO-001), so `std-file` binds an existing Core
nominal rather than a synthesized one. Does the manifest bind `resources: { "File": … }` against the
Core type, or is Core `File` special-cased as it is today in `ResourceRegistry::builtin()`? The
second keeps Core and package resources visibly distinct; the first is one mechanism.

---

## 8. What this does not decide

Dynamic loading, capability sandboxing, allowlists and deployment policy remain deferred (Packet 5).
Nothing here changes the ABI, the runtime surface, or any Core specification document.
