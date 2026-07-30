# C7 P1 REST workload

This package is a bounded, single-threaded HTTP/1.1 service implemented in STARK. The native host
surface is limited to TCP bind/accept/read/write and an optional environment lookup for the bind
address. HTTP parsing, JSON validation, routing, response construction, byte-length calculation,
and `write_all` are STARK code.

## Run

```sh
cd starkc/tests/workloads/c7-p1-rest
../../../target/debug/stark build --no-build-cache
STARK_P1_BIND=127.0.0.1:39091 ./target/stark/debug/c7-p1-rest
```

The server handles exactly 24 accepted connections, then exits. `STARK_P1_BIND` may select another
explicit loopback address and port; the default is `127.0.0.1:39091`.

## Frozen protocol

- `GET /health` → `200`, `{"status":"ok"}`
- `GET /items/1` and `/items/2` → deterministic fixture objects
- valid but absent item IDs → `404`; malformed/overflowing IDs → `400`
- `POST /items` with exactly one non-empty string `name` member → `201` with fixed ID 3
- unknown routes → `404`; unsupported methods → `405`
- HTTP/1.1, origin-form target, strict CRLF, required non-empty `Host`
- case-insensitive `Host`, `Content-Length`, and `Transfer-Encoding` names
- required `Content-Length` for POST; no transfer coding; one request per connection
- every response is JSON with byte-exact length and `Connection: close`

## Bounds and JSON

The package enforces the work-package recommended bounds: 2 KiB request line, 8 KiB headers, 32
headers, 64 KiB body, 1 KiB path, JSON depth 32, and decoded JSON string length 16 KiB. The total
fixed request buffer is 72 KiB.

The reusable JSON validator accepts objects, arrays, strings, integer numbers, booleans, and null.
It accepts all required simple escapes. `\uXXXX` is accepted for non-surrogate BMP values; surrogate
pairs are deliberately rejected. Floating-point and exponent number forms are deliberately rejected
for P1. Direct UTF-8 is structurally checked and preserved byte-for-byte. POST responses preserve
the validated JSON string token, which retains its exact decoded value and guarantees valid escaping.

## Compatibility inventory

| Need | Existing surface used |
|---|---|
| package/module layout | `starkpkg.json`, sibling `mod` files |
| bounded bytes | fixed `[UInt8; 73728]`, slices, `Vec<UInt8>` |
| strings | string literal `.bytes()` only on the native request path |
| TCP | synthesized `TcpListener`, `TcpStream`, bind/accept/read/write |
| address selection | `var_len`/`var_fill` for `STARK_P1_BIND` |
| selected ephemeral port | not exposed; harness supplies a free loopback port |
| deterministic stop | fixed 24-connection test workload |

No provider-close function is declared or called by package source. Resource destruction is left to
the existing generated-Rust Drop path.

## Qualification commands

```sh
python3 scripts/pure_tests.py
../../../target/debug/stark build --no-build-cache --verbose
python3 scripts/e2e.py
python3 scripts/measure.py
```

The pure runner creates a temporary capability-free package and executes the seven byte, JSON, and
HTTP tests through `stark test`. The raw-socket runner launches the native artifact, validates 24
responses byte-for-byte, and verifies bounded clean shutdown. The measurement runner retains its
machine-readable observations in `measurements/latest.json`.
