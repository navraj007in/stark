# HC13 — platform matrix

**Status:** current as of 2026-08-03

---

## 1. Tier-1 platforms

Every row runs in CI on every push, in `first-party package qualification (${{ matrix.name }})`,
with `fail-fast: false` so one platform failing does not hide the others.

| platform | runner | HTTP | HTTPS | DNS | TCP | timeouts | 16-package gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linux-x64 | `ubuntu-latest` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| macos-arm64 | `macos-14` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| windows-x64 | `windows-latest` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**There is no platform gating in the harness.** `qualify-first-party-packages.py` contains no
`sys.platform` branch: every lane runs every peer. And every peer **asserts its bind rather than
attempting it** —

> *"Binding is asserted, as with every other peer: a skipped peer would silently downgrade lifecycle
> evidence to lowering evidence while the gate still reported success."*

That matters most on Windows, where a loopback TLS listener is likeliest to fail to bind. An
attempt-and-skip would have reported green while proving nothing.

---

## 2. The divergence this matrix actually caught

**DEV-163.** A socket read whose `SO_RCVTIMEO` expires reports:

```text
Unix     EAGAIN        -> std::io::ErrorKind::WouldBlock
Windows  WSAETIMEDOUT  -> std::io::ErrorKind::TimedOut
```

The provider passed both through distinctly, and `stark-net` mapped `WouldBlock` to
`NetworkError::Interrupted`. So a configured `read_timeout` surfaced as **"the connection failed"**
on Linux and macOS, and **"timed out reading the response"** on Windows — for the identical peer,
from identical STARK source.

Neither platform was wrong on its own terms; the *same event* had two names. This is the class of
defect a single-platform test suite cannot see, and it survived HC0–HC12 because every test until
HC13 used a peer that answers.

Fixed in the provider, where the socket mode is known: a stream in this provider is always blocking,
so `WouldBlock` from a read or write can only mean the deadline expired. Both platforms now report
`STATUS_TIMED_OUT`.

---

## 3. Tier-2 and untested

| platform | status |
| --- | --- |
| linux-arm64 | untested. Nothing platform-specific is expected, and "expected" is not evidence. |
| windows-arm64 | untested. |
| FreeBSD, other BSDs | untested. `std::net` should suffice; unproven. |
| any cross-compiled target | **refused.** `stark build --target` validates the triple and then declines. Every Tier-1 binary is built natively on its own platform. |

---

## 4. Native provider identities and versions

CD-361 (Profile N): one cryptographic foundation, pinned exactly, because two crypto backends
compiled into one binary is a supply-chain and correctness hazard rather than redundancy.

| crate | version | pinned by |
| --- | --- | --- |
| `rustls` | `=0.23.43` | `stark-tls/native/Cargo.toml` |
| `rustls-pemfile` | `=2.2.0` | `stark-tls/native/Cargo.toml` |
| `rustls-pki-types` | `=1.15.1` | `stark-tls/native/Cargo.toml` |
| `rustls-native-certs` | `=0.8.2` | `stark-tls/native/Cargo.toml` |
| `aws-lc-rs` | `1.17.3` | `Cargo.lock` |
| `aws-lc-sys` | `0.43.0` | `Cargo.lock` |

`rustls` is `default-features = false` with `aws_lc_rs`, `tls12`, `std` — so `ring` is not compiled
in, and the feature set is a decision rather than a default.

**Profile N vs Profile F.** Profile N is what ships: it builds with a stock Rust toolchain on all
three Tier-1 runners. Profile F (FIPS) requires CMake and Go on the build host and is **not** built
or tested here.

Providers in this stack:

```text
stark-net/native      DNS, TCP, socket deadlines      std::net
stark-tls/native      TLS 1.2/1.3 client              rustls + aws-lc-rs
stark-time/native     monotonic and wall clock        std::time
stark-random/native   CSPRNG                          getrandom via std
stark-env/native      environment and arguments       std::env
stark-file/native     filesystem                      std::fs
```

---

## 5. What the matrix does not cover

- **`Connect` and `Resolve` timeout phases** on any platform. See HC13-KNOWN-LIMITATIONS.md §1.1.
- **Non-Tier-1 architectures.** No arm64 Linux runner exists in this CI.
- **Locale, IPv6-only hosts, and split-horizon DNS.** Peers are IPv4 loopback.
- **Long-running behaviour.** Every test is a single short exchange; nothing here proves stability
  over hours or thousands of connections.
