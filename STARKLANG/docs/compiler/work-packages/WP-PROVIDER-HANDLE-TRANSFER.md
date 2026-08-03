# WP-PROVIDER-HANDLE-TRANSFER — cross-provider resource ownership transfer

**Status:** DESIGN, not started. **Priority:** P0. **Blocks:** HC9 (TLS), HC10 (HTTPS), and any
provider that wraps another provider's resource.

> **Governing claim.** A resource may cross provider boundaries only through an ABI-declared
> ownership transfer that preserves exactly-one-owner and exactly-once-release semantics.

---

## 1. Why this exists

TLS wraps TCP. The TLS provider must take a `TcpStream` that the *net* provider created and produce
a `TlsStream` that the *TLS* provider owns and releases. The ABI has no way to express that, and
the validator refuses both ways of attempting it.

Without a frozen rule, an implementation would take one of five bad paths — duplicate ownership,
smuggle raw handles past the type system, bypass the validator, fuse TCP and TLS into one provider,
or leave Drop authority unclear. Each weakens the resource model that CD-234/CD-237/CD-240 and the
A11 packet exist to guarantee.

## 2. Current state, verified rather than assumed

Probed directly against `provider_abi::validate` (2026-08-02). Both attempts fail:

```text
# TLS provider declares tcp_stream so it can name it in a signature
ResourceTypeMissingClose { resource_type: "tcp_stream" }
        -> every declared resource type must have a close IN THE SAME PROVIDER,
           so declaring it would give tcp_stream a second, competing close

# TLS provider does not declare it, and names it anyway
HandleResourceTypeUndeclared { function: "stark_tls_client_connect",
                               resource_type: "tcp_stream" }
        -> a provider may only reference resource types it declares
```

**These two rules are correct and should survive.** Together they are what makes "exactly one
provider owns and releases each resource type" true by construction. The packet must not weaken
them; it must add a third, narrower rule alongside.

## 3. What the ABI already provides

More of the contract exists than the refusal suggests. Established by inspection and by existing
tests:

| already true | where | consequence for transfer |
| --- | --- | --- |
| Resource identity is structural over **all three** of `{nominal, provider, resource}` | `a11_host_resource.rs::identity_is_structural_over_all_three_fields` | provider identity is part of the TYPE, so a transfer is a genuine type change, not a re-tag |
| `HandleOut` writes its slot **only on success** | `c788_resource_lifecycle.rs::handle_out_emission_writes_the_slot_only_on_success` | the destination's failure disposition is already settled: on failure no `TlsStream` exists |
| Every function returns `ProviderStatus`; no direct returns (§11) | `provider_abi.rs` | `Result<HandleOut<TlsStream>, TlsError>` is already the natural shape |
| Close is selected per resource at build time; a resource with no close is **refused** | `native_build.rs` §5 obligation 5 | "which provider releases" is answered structurally — whoever declares the type declares its close |
| `HandleConsumed` means the caller's handle is gone and its MIR close must never run | DEV-146 / A11 | the source-side mechanism exists |

**So the missing pieces are two, and only two:**

1. **Permission** for a provider to *name* a resource type another provider owns, in the specific
   position of a consuming transfer — without inheriting the obligation to close it.
2. **The failure disposition of the consumed source.** This is the hard part and the reason the
   packet exists.

## 4. The decision to freeze

### 4.1 The transfer operation

Conceptually:

```text
HandleConsumed<TcpStream>  ->  Result<HandleOut<TlsStream>, TlsError>
```

Concretely, in existing ABI vocabulary — no new parameter forms required:

```text
fn stark_tls_client_connect(
    tcp:         HandleConsumed { resource_type: "tcp_stream" },   # FOREIGN, owned by stark-std-net
    server_name: BufferIn,
    out:         HandleOut { resource_type: "tls_stream" },        # OWNED by this provider
) -> ProviderStatus
```

### 4.2 The invariant

> On success, the source handle is permanently consumed and the destination provider owns the
> underlying resource.

That half is uncontroversial and matches what `HandleConsumed` + `HandleOut` already mean.

### 4.3 The failure rule — THE decision this packet exists to make

**It must not be ambiguous.** Three candidates:

**(A) Failure also consumes the source.** The TLS provider is responsible for closing the TCP
socket on any failure path. The caller's handle is gone either way.
*For:* one rule, no conditional ownership, no state for MIR to track — the source slot is dead
after the call unconditionally, exactly as `HandleConsumed` already means today. Drop elaboration
needs no change whatsoever.
*Against:* a caller cannot retry the handshake on the same socket. In practice nobody does — a
failed TLS handshake leaves the socket unusable — so the cost is theoretical.

**(B) Failure returns ownership.** The source handle is live again if the call fails.
*For:* no resource is destroyed by a failed operation.
*Against:* ownership becomes **conditional on a runtime value**. MIR would have to model a place as
live-on-one-edge and dead-on-another, and drop elaboration would need a status-dependent close.
That is a significant widening of the resource model, and it is the shape most likely to produce
a double-close or a leak under a path nobody tested.

**(C) Two-phase prepare/commit.** Validate and configure without consuming, then commit.
*For:* no ambiguity; failure before commit is clean.
*Against:* protocol complexity, a new intermediate state, and a new way to get it wrong (commit
never called → leak). Not justified at Core v1.

**Recommendation: (A).** It is the only option that requires **no change to drop elaboration**,
because it keeps `HandleConsumed` meaning exactly what it means today — unconditionally consumed.
(B) makes ownership depend on a runtime value, which is precisely the kind of conditional invariant
this compiler has repeatedly failed to get right on first attempt. (A) also states the real-world
truth: a failed handshake does not leave you a usable socket.

If (A) is adopted, the ABI must say so **in the function declaration**, not in prose, so the rule is
machine-checkable and visible at the call site.

### 4.4 The questions this packet must answer in full

```text
resource ownership before transfer      source provider owns; caller holds the handle
resource ownership after transfer       destination provider owns
does transfer consume the source        YES, unconditionally (recommendation A)
can wrapping fail                       YES — ProviderStatus discriminates
what happens on failure                 FROZEN BY 4.3 — no TlsStream exists; source consumed
which provider performs final release   the destination, via its own is_close_for
is the original handle usable after     NO, on every path
is provider identity preserved          NO — it changes, and that IS the transfer
```

## 5. Validator rules to add

A third rule, narrow enough not to weaken the two in §2:

1. A function parameter may name a **foreign** resource type **only** in `HandleConsumed` position,
   and only when the same function has a `HandleOut` of a resource type the provider **does** own.
   A foreign `HandleBorrowed` stays refused — borrowing across providers has no owner story and no
   caller need identified.
2. Naming a foreign resource type must **not** create the `ResourceTypeMissingClose` obligation. The
   declaring provider still owns the close; that is the point.
3. The declared foreign resource type must be **resolvable in the selected provider set** at build
   time, and must resolve to exactly one owning provider. An unresolvable or ambiguous foreign
   resource is a build refusal, not a link error.
4. A function may perform **at most one** transfer. Two consumed foreign handles in one call is
   refused until a use case exists.
5. `is_close_for` shape is unchanged: exactly one parameter, a `HandleConsumed` of the type the
   provider declares. A transfer function is never a close function.

## 6. MIR and generated-Rust representation

* The transfer call is an ordinary provider call. The source place becomes **dead** after it on
  every edge, which is what `HandleConsumed` already causes — this is the payoff of recommendation
  (A).
* The source's MIR close must never be emitted for the transferred place. Already true for
  `HandleConsumed`; the transfer must not create an exception.
* The destination handle is written on success only, and its close is the destination provider's.
* `MirTy::HostResource` changes identity across the call — `{net, tcp_stream}` in, `{tls,
  tls_stream}` out. MIR-0004/0005 verification should confirm the two are distinct types and that
  no path treats them as interchangeable.

## 7. Negative tests — the load-bearing half

A transfer mechanism that fires too eagerly is a double-free. Each must be refused:

```text
a foreign resource in HandleBorrowed position
a foreign resource in HandleOut position (manufacturing another provider's resource)
a transfer function that also declares is_close_for
two foreign HandleConsumed parameters in one function
a foreign resource type that no provider in the set declares
a foreign resource type that two providers declare
use of the source handle after a transfer call, on the success path
use of the source handle after a transfer call, on the FAILURE path
a MIR body that emits the source provider's close for a transferred handle
```

The last three are the ones that would ship a double-close if missed.

## 8. Proving case

TCP-to-TLS, end to end, under the qualification gate:

* a native consumer that connects, transfers, speaks TLS to a real peer, and releases exactly once;
* observed release — CD-347/348 require lifecycle evidence, not lowering evidence;
* the failure path executed too: a handshake that fails against a peer presenting a bad
  certificate, asserting the source is consumed and nothing leaks.

Note this needs a TLS peer in CI with a controlled certificate chain — a heavier fixture than
`echo_peer`/`http_peer`. Budget for it in the packet, not in HC9.

## 9. Exit criteria

1. The failure disposition is frozen in writing, with its rationale, and is machine-checkable from
   the declaration.
2. Validator rules implemented with the §7 negatives green.
3. A resource transferred across providers is released exactly once, observed natively.
4. The source handle is unusable after transfer on every path, enforced by the front end.
5. No change to the two rules in §2 for non-transfer functions.

## 10. Explicit non-goals

* Borrowing a foreign resource.
* Returning a foreign resource (`HandleOut` of a type you do not own).
* Transfer chains longer than one hop in a single call.
* Reversing a transfer.
* Sandboxing or isolating provider code — see `WP-EXTERNAL-PROVIDERS.md` §trust.
