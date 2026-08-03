# Native Provider ABI v0.1 — Amendment 2 (CD-360): cross-provider resource transfer

**APPROVED 2026-08-03 (CD-360).** Nothing in this amendment remains open for decision.

---

## 1. The ruling

> **A cross-provider `HandleConsumed` transfer consumes the source handle regardless of whether the
> provider operation succeeds or fails. Failure does not restore the source resource. The consuming
> provider is responsible for releasing any underlying native resource when it fails before
> producing the destination handle.**

`HandleConsumed<T>` therefore keeps the meaning it has always had:

```text
HandleConsumed<T>  =  ownership leaves the caller unconditionally
```

## 2. Why unconditional

Preserving that meaning is the value of the ruling. It means the ABI needs:

* no conditional ownership restoration;
* no branch-dependent move state in MIR;
* **no change to drop elaboration**;
* no source handle live on one result arm and dead on another;
* no reconstruction of ownership after a provider may have partially used the resource;
* no ambiguity over whether a failed TLS handshake leaves a reusable TCP stream.

The rejected alternative — returning the source on failure — would require a different ABI shape
(`Result<TlsStream, (TlsError, TcpStream)>` or an equivalent conditional handle output) and would
introduce conditional move restoration across provider boundaries. That is ownership machinery, not
an ergonomic improvement, and it is not justified by making failed handshakes recoverable. It
remains available as a future ABI extension if a real capability needs recoverable transfer.

## 3. The obligation this places on the consuming provider

**Normative, and unenforceable by the compiler.** A provider function that consumes a foreign
resource and returns a failure status **must leave no live native resource behind**: on every
failure path it must have released or invalidated the underlying resource the consumed handle
referred to.

This is the price of the ruling. Because ownership left the caller unconditionally, the caller
cannot clean up and nothing else will. A provider that returns failure without releasing has
leaked, and no check can detect it — for the same reason §5 obligation 5 refuses a closeless
resource: the provider never learns the handle was abandoned.

It is stated here, rather than left implicit, because it is the one part of the contract that
review has to carry.

## 4. Public shape

```stark
fn connect_tls(stream: TcpStream, config: &TlsConfig) -> Result<TlsStream, TlsError>;
```

After the call, `stream` is unavailable on **both** `Ok` and `Err`.

```text
on success                      on failure
----------                      ----------
TCP resource consumed           TCP resource consumed
TLS resource created            no TlsStream slot written
TlsStream owns final release    provider released/invalidated the socket  (§3)
                                caller receives only TlsError
```

The failure column's third line is §3's obligation; the second is already guaranteed by
`HandleOut`'s existing "writes the slot only on success" rule.

## 5. Declaration

A provider declares what it may consume but does not own:

```text
foreign_resources: [ { provider: "stark-std-net", resource: "tcp_stream" } ]

stark_tls_client_connect(
    HandleConsumed { resource_type: "tcp_stream" },   # foreign, owned by stark-std-net
    BufferIn,                                          # server name
    HandleOut     { resource_type: "tls_stream" },     # owned by this provider
) -> ProviderStatus
```

**Explicit, not inferred.** Treating "any handle type I did not declare" as foreign would silently
accept a misspelled resource type and defer the typo to a link failure. Naming the owning provider
keeps the check at the three-part identity `{nominal, provider, resource}` the type system uses.

## 6. Validator rules

Added:

1. A foreign resource type may appear **only** in `HandleConsumed` position.
2. Naming a foreign resource type creates **no** close obligation in the consuming provider.
3. A function consuming a foreign resource must produce a `HandleOut` of a resource it **owns** —
   otherwise it is a close of another provider's resource wearing a different name.
4. **At most one** foreign consumed resource per function. Two would require an ordering rule for
   the failure path, and §1 is written for exactly one source.
5. A declared foreign resource that no function consumes is refused, as an unreachable capability
   already is — a dead declaration grants silent permission.

Unchanged, and deliberately so:

* a provider may not **produce** another provider's resource;
* a provider may not declare a **close** for a resource it does not own;
* a **borrowed** foreign resource stays prohibited — no owner story, no identified caller need;
* declaring a foreign type in `resource_types` still demands a close for it, which is what stops
  one resource acquiring two competing closes;
* resource identity remains structural over `{nominal, provider, resource}`.

## 7. What this amendment is not

Not a new transfer mechanism. Probing established that most of the contract already existed:
`HandleOut` writes only on success, close is selected per resource and a closeless resource is
refused, identity already carries provider, and every function returns `ProviderStatus`. What was
missing was **permission**:

> A provider function may reference a foreign provider's resource type only in a consuming handle
> position, without inheriting or redefining that resource's close operation.

## 8. Implementation status

| | status |
| --- | --- |
| `ForeignResource` declaration and `ProviderMetadata` field | DONE (CD-360) |
| Validator rules §6.1–§6.5 with the full negative matrix | DONE — `tests/cd360_cross_provider_transfer.rs`, 11 tests |
| Build-time resolution: a foreign resource resolves to exactly one owning provider in the selected set | **OPEN** |
| MIR/native representation: source dead on every edge, source close never emitted, identity changes across the call | **OPEN** |
| Runtime proving case, both outcomes | **OPEN** — needs a TLS peer with a controlled certificate chain |

The last three belong to the remainder of `WP-PROVIDER-HANDLE-TRANSFER`. §3 is a provider-author
obligation no compiler check can enforce, and is recorded here so review can carry it.
