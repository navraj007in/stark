# HC2-EVIDENCE — `stark-net` TCP client package

**Stage:** HC2 of `WP-HTTP-CLIENT-ROADMAP.md`.
**Commit:** `56a78b4 Implement HC2 stark-net package`.

## Implemented surface

`stark-net` now exists as an importable STARK package and binds the existing TCP stream provider
resource:

```text
IpAddress::{V4,V6}
Ipv4Address
Ipv6Address
SocketAddress
TcpStream
NetworkError
connect(address, timeout)
connect_no_timeout(address)
read(&mut TcpStream, &mut [UInt8])
write(&mut TcpStream, &[UInt8])
write_all(&mut TcpStream, &[UInt8])
close(TcpStream)
TcpStream::{connect,read,write,write_all,shutdown_write}
```

Provider bindings:

```text
stark_tcp_stream_connect
stark_tcp_stream_read
stark_tcp_stream_write
stark_tcp_stream_close
```

## Contract rulings

### Timeout

HC2 does not implement socket timeouts. `connect(address, timeout)` accepts the HC2-shaped API, but a
non-zero timeout returns `NetworkError::Unsupported`. Zero-duration connect and
`connect_no_timeout(address)` use the existing blocking provider.

Actual connect/read/write timeout support remains HC4 work and requires provider support.

### Close

`TcpStream` is an affine host resource. `close(stream)` consumes the stream and returns `Ok(())`;
MIR drop elaboration emits the declared `stark_tcp_stream_close` provider close exactly once. The raw
close binding is intentionally not called from package code, because that would create a second
destruction path.

### Read/write mutability

The public free functions and methods both require `&mut TcpStream`. The native provider accepts a
borrowed resource handle, but TCP stream I/O is semantically stateful, so the package surface exposes
mutable access.

### IPv6 rendering

IPv6 socket address formatting emits eight uncompressed hexadecimal segments inside brackets, for
example:

```text
[0:0:0:0:0:0:0:1]:443
```

This is valid address text for the provider; HC2 does not attempt canonical IPv6 compression.

### Error mapping

The provider vocabulary is mapped into the HC2 public `NetworkError` vocabulary:

| Raw provider error | Public error |
| --- | --- |
| `ConnectionRefused` | `ConnectionRefused` |
| `TimedOut` | `TimedOut` |
| `NotFound` | `AddressNotAvailable` |
| `PermissionDenied` | `PermissionDenied` |
| `AddressInUse` | `AddressNotAvailable` |
| `InvalidInput` | `InvalidAddress` |
| `ConnectionReset` | `ConnectionReset` |
| `BrokenPipe` | `ConnectionReset` |
| `WouldBlock` | `Interrupted` |
| `Unsupported` | `Unsupported` |
| `Other` | `ProviderFailure(11)` |

`11` is the provider's declared `STATUS_OTHER_DECLARED` value in `stark-net/native`.

## Evidence

Local evidence at implementation time:

```text
cargo build -p starkc --bin stark

stark-net:          stark check
stark-net:          stark test                 2 passed
stark-net:          stark fmt --check
stark-net-consumer: stark check
stark-net-consumer: stark run                  STARK_NET_CONSUMER_OK
stark-net-consumer: stark build --no-build-cache
stark-net-consumer: native artifact run        STARK_NET_CONSUMER_OK
```

Full first-party package qualification passed with `stark-net` included:

```text
python3 starkc/scripts/qualify-first-party-packages.py --stark starkc/target/debug/stark --repo-root .
```

Existing native TCP loopback evidence remains in:

```text
starkc/tests/c788_starkc_build.rs
```

That test builds a STARK TCP server and client from source, connects over loopback, exchanges bytes,
and verifies the echoed payload. It predates the package wrapper but exercises the same
`stark-net/native` provider connect/read/write/close path.

## Remaining HC2 gaps

- Package-level loopback tests cannot run under `stark run` because interpreter provider calls use
  synthesized panic bodies; provider execution is native-only.
- Timeout support is explicitly deferred to HC4.
- `shutdown_write` returns `Unsupported` until a `stark_tcp_stream_shutdown` provider function lands.
- Cross-platform native evidence must be confirmed in CI.
