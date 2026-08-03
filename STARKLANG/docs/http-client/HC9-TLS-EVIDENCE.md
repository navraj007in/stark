# HC9 — `stark-tls`: closure record and evidence

**Status:** CLOSED 2026-08-03
**Engine:** rustls `0.23.43` over aws-lc-rs `1.17.3` (aws-lc-sys `0.43.0`), **Profile N**
**Depends on:** CD-360 (cross-provider transfer), CD-361 (backend selection), CD-363/364 (provider
manifests)
**Enables:** HC10 (HTTPS client)

---

## 1. The claim, and its exact boundary

> A STARK program can establish a **verified** TLS 1.2 or TLS 1.3 stream to a named host over a
> `stark-net` TCP connection, exchange application bytes, and release both layers exactly once —
> without touching a raw ABI symbol.

Proved natively on macOS/arm64 at closure; the Linux and Windows lanes run the same suites in CI.

### What this does NOT claim

| | |
| --- | --- |
| system trust store | **not implemented.** `SystemRoots` and `BundledRoots` are declared and REFUSED (`Unsupported`). HC10's. |
| Profile F (FIPS) | **not qualified.** Needs CMake and Go, neither present at closure; qualified separately per CD-361 §2. |
| HTTPS | HC10. `stark-http-client` does not yet select TLS from a URL scheme. |
| client certificates, custom verifiers, revocation | out of scope by design — see §3. |
| non-x86_64/aarch64 targets | the manifest declares four triples; nothing beyond them is claimed. |

---

## 2. The architecture, and the one thing it needed that did not exist

```text
stark-tls (STARK)          TlsStream, TlsClientConfig, TlsError
        ↓ provider_api
stark-tls-native (Rust)    rustls + aws-lc-rs, one resource table
        ↓ transfer
stark-net-native           the socket, handed over and forgotten
```

CD-360 settled what a transfer MEANS: `HandleConsumed` of a foreign resource takes ownership at call
entry, on success and on failure alike, and the consumer owes the release. Implementing it surfaced
**two gaps the ruling did not reach**, both closed here.

### 2.1 A package could not NAME another package's resource

The derived signature for `stark_tls_stream_connect` has a `TcpStream` first parameter — and
`TcpStream` is `stark-net`'s nominal. Derivation failed with `UnboundResourceInSignature`: a
transfer declarable in a *provider* manifest and not in a *package* one.

`provider_api.foreign_resources` closes it:

```json
"foreign_resources": {
  "tcp_stream": { "package": "stark_net", "nominal": "TcpStream" }
}
```

It resolves to the qualified path `stark_net::TcpStream`, and **synthesizes nothing**. The two
alternatives are both quietly wrong:

* binding `tcp_stream` as an ordinary resource generates a SECOND `enum TcpStream {}` — a distinct
  `ItemId`, a different type with the same spelling, and a handle the program cannot pass anywhere;
* inferring the owner from the dependency graph makes a typo (`tcp_strem`) resolve to nothing, far
  from its cause.

Pinned by `starkc/tests/hc9_foreign_resource_nominals.rs` (11 tests).

### 2.2 The MIR verifier was a FOURTH site for CD-360's rule

CD-360 found the ownership rule implemented in three places and fixed each separately. The verifier
was a fourth, missed because CD-360's fixture built its `ValidatedProviderCall` by hand and never
ran the verifier over a transfer. HC9's first native build:

```text
MIR-0005 stark_tls::connect bb53: call argument:
  expected HostResource(… provider: "stark-std-tls", resource: "tcp_stream"),
  found    HostResource(… provider: "stark-std-net", resource: "tcp_stream")
```

The planner was right and the verifier was wrong — **a correct program refused by the compiler**.
The rule now lives in exactly one function, `mir::provider_sig::owner_of`, which both callers use, so
a fifth site cannot restate it slightly differently. Pinned by
`starkc/tests/hc9_transfer_verification.rs` (5 tests), including one that asserts the planner's and
the verifier's types are *the same value*.

### 2.3 How the socket physically crosses — `stark_tcp_stream_detach`

CD-360 conveyed ownership but not the object: a `RawResourceHandle` is an index into the OWNER's
private table, which the consumer cannot read. The convention added here, documented on
`stark_provider_abi::RawOsHandle`:

```text
stark_<resource>_detach(RawResourceHandle, *mut RawOsHandle) -> ProviderStatus
```

* **Not in the provider manifest.** A manifest describes the STARK-callable surface — what
  `provider_api` may bind and what lowering may emit. `detach` is neither. Putting it there would
  place a permanently unreachable symbol into the surface the validator governs. The declaration that
  IS compiler-visible is the consumer's `consumes: [{provider, resource}]`, checked from both ends.
* **Resolved by the linker, not by Cargo.** Every provider is statically linked into one generated
  binary, so an `extern "C"` declaration resolves with no dependency edge and no path assumption — the
  coupling CD-363 spent its effort deleting.

**Recorded limitation:** a missing detach symbol is a LINK error naming a symbol, not a compiler
diagnostic. The compiler knows a transfer was declared; it does not know the owner published the
means. Closing that needs the manifest to carry a transfer surface distinct from the callable
surface, which is larger than HC9 should decide.

### 2.4 Ordering inside `stark_tls_stream_connect` is load-bearing

```text
detach the socket FIRST, into an owned Rust TcpStream
        ↓
validate the configuration
        ↓
handshake
```

The handle is consumed by the ABI whatever the function returns, so **any early return before the
socket is adopted strands it** in the net provider's table — unreachable from STARK and never closed.
Detaching first makes every later error path a plain Rust drop. There is no cleanup code in that
function, and its absence is the design.

---

## 3. Security posture

Absent rather than defaulted-off: **no** "accept invalid certificate" switch, **no** verification
callback, **no** way to disable hostname checking, **no** client-certificate path. A flag that exists
is a flag that gets set in a hurry and never unset; a callback is a verifier written by someone who
is not writing a verifier.

Hostname verification and SNI are not optional — both follow from `ServerName` parsing, which rustls
verifies against directly. TLS 1.0/1.1 are unnameable in `TlsVersion`, so a version range cannot be
configured down to them.

**Each certificate failure carries its own status.** A client that reports a clock problem, a trust
problem and a name problem identically sends operators to the wrong place:

```text
22 CertificateExpired        23 CertificateNotYetValid   24 CertificateUnknownIssuer
25 HostnameMismatch          21 CertificateInvalid       26 ProtocolVersionUnsupported
27 HandshakeTimeout          28 PeerClosedDuringHandshake
```

**The handshake bound is a TOTAL deadline on a monotonic clock**, not a per-read socket timeout. HC4
states the distinction and this is where it first bites: a per-read bound is an idle bound, so a peer
dribbling one handshake byte at a time stays under it forever. `Duration::zero()` is refused rather
than read as "unbounded" — an unbounded handshake has no spelling in this package.

---

## 4. Evidence

| what | where | count |
| --- | --- | --- |
| provider unit + certificate matrix + lifecycle | `stark-tls/native/src/lib.rs` `mod tests` | 19 |
| foreign-resource nominals (compiler) | `starkc/tests/hc9_foreign_resource_nominals.rs` | 11 |
| transfer verification (compiler) | `starkc/tests/hc9_transfer_verification.rs` | 5 |
| package pure surface | `stark-tls/src/tests.stark` | 8 |
| executed native lifecycle | `stark-tls-consumer`, under the package gate | 1 program |

### The certificate matrix

Each fixture differs from the happy path in **exactly one** property, so a failure names one cause.
Certificates are checked in with **absolute** validity windows (`stark-tls/fixtures/generate.sh`), so
an "expired" fixture stays expired and a "not yet valid" one does not quietly become valid. Nothing
depends on `openssl` or the `cryptography` module at test time — only the standard library.

| case | fixture | expected |
| --- | --- | --- |
| trusted chain | `server` | SUCCESS |
| untrusted root | `untrusted` (signed by `rogue-ca`) | `CertificateUnknownIssuer` |
| valid chain, wrong anchor | `server` + `rogue-ca` roots | `CertificateUnknownIssuer` |
| expired | `expired` (2020–2021) | `CertificateExpired` |
| not yet valid | `not-yet-valid` (2040–2045) | `CertificateNotYetValid` |
| hostname mismatch | `wrong-host` (`other.test`) | `HostnameMismatch` |
| missing intermediate | `chained` alone | `CertificateUnknownIssuer` |
| **control** for the above | `chained-fullchain` | SUCCESS |
| TLS 1.3 only peer | `server` | SUCCESS, `peer_version` = Tls13 |
| TLS 1.2 only peer | `server` | SUCCESS, `peer_version` = Tls12 |
| silent peer | — | `HandshakeTimeout`, bounded |
| peer closes mid-handshake | — | `PeerClosedDuringHandshake` |

Every negative case also asserts `live_stream_count()` is unchanged: **no TLS resource survives a
failed handshake.**

### The executed lifecycle

`stark-tls-consumer` runs natively against three controlled peers and prints:

```text
  tls: TLS 1.3 session verified, used and closed explicitly
  tls: TLS 1.2 session verified, used and closed explicitly
  tls: untrusted root rejected as certificate issuer is not trusted
  tls: drop released the session and the socket under it
STARK_TLS_RESOURCE_OK
```

Both release paths are exercised — explicit `close()` and release by drop — which is CD-348's bar for
a resource-shaped package. Naming both protocol versions is deliberate: "TLS 1.3 was negotiated" is
only evidence if another peer could have answered 1.2 and the client can tell them apart.

**CD-360's runtime proving case, left open by that decision, is now closed by this program**: a real
transfer executed against a live peer, on both the success and failure paths, with release observed
exactly once. (A double release aborts in the provider, so a zero exit status is the assertion.)

`stark-tls` is the 16th package in the first-party qualification gate, and the declared-surface check
reports **14 public callables, all called**.

---

## 5. Findings recorded, not fixed here

| id | what | why not here |
| --- | --- | --- |
| **DEV-156** | `stark fmt` evicts a doc comment from a struct FIELD or an impl METHOD to after the item. `field_def` never consumes leading comments, so they fall through to the next item's flush; `delimited_list`'s flat form has no comment awareness. Full reproducer and cause in `COMPILER-STATE.md`'s CD-365 entry. | Fixing it changes the canonical form repo-wide, so every affected package must be reformatted in the same commit. That is a formatter work package, not an HC9 one — and this checkout is shared. `stark-tls` uses the surviving placement, with the reason recorded in the source. |
| **DEV-157** | The native backend has no representation for `MirTy::Never`, so `Err(_) => panic(..)` in match-arm VALUE position checks and then fails to build. | Known C5.3 gap (WP-C5.1's MirTy matrix). `stark-tls-consumer` nests instead, as `stark-net-resource-consumer` already does. |
| — | `c788_resource_lifecycle::build_driver_selects_closes_for_bound_resource_nominals` fails in this checkout with "Cargo succeeded but the expected binary is missing". **Confirmed pre-existing on HEAD** (verified by stashing every HC9 change). Environmental, tied to the shared `target/` directory. | Not an HC9 regression. |
| — | A provider manifest's `targets` cannot express *toolchain* prerequisites, so `stark-tls-native` declares four triples it can only build on with a C/C++ compiler present (CD-361 §6). | Already recorded against WP-EXTERNAL-PROVIDERS. |
| — | `stark-tls` declares only the `tls` capability. The wall-clock and secure-randomness the engine uses are internal to the linked provider and are **not** separately declared, so the roadmap's "these requirements must be visible in package/provider metadata" is met for network (via `consumes`) but not for the other two. | The ABI has no way to declare an internally-satisfied dependency. Making it visible needs a manifest field; recorded rather than faked. |

---

## 6. Versions pinned

CD-361: *you pin what you qualified.*

```text
rustls        =0.23.43     default-features = false, features = ["aws_lc_rs", "tls12", "std"]
rustls-pemfile =2.2.0
rustls-pki-types =1.15.1
aws-lc-rs      1.17.3      (transitive; held by stark-tls/native/Cargo.lock)
aws-lc-sys     0.43.0      (transitive)
rustls-webpki  0.103.13    (transitive)
```

`default-features = false` is not tidiness: the default set also pulls `ring`, and two cryptographic
backends in one binary is precisely the "unrelated crypto stacks" outcome CD-361 exists to prevent.
