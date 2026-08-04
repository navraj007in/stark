# stark-time-native

The native clock provider for `stark-time` v0.1, implementing Native Provider ABI v0.1
(`STARKLANG/docs/compiler/native-provider-abi-v0.1.md`, as amended by CE4 Amendment 1).

**Not wired into `stark build`.** This crate builds and tests standalone. Nothing in the
compiler links against it: the repository has no owner-approved provider execution/linkage
mechanism yet. See `../BLOCKERS.md`.

## Provider identity

```text
name:         stark-std-time
semver:       0.1.0
abi_version:  0.1
capabilities: [clock]
resource_types: []
```

## Functions

### `stark_time_monotonic_now_ns`

```text
capability: clock
params:     [ScalarOut(U64)]
may_block:  false
```

Writes nanoseconds elapsed since a process-local origin established on first call
(`std::time::Instant`, lazily initialized via `OnceLock`). The origin is unspecified, not the
Unix epoch, and not stable across process runs.

### `stark_time_unix_now`

```text
capability: clock
params:     [ScalarOut(I64), ScalarOut(U32)]
may_block:  false
```

Writes normalized Unix seconds and nanoseconds (`std::time::SystemTime` against
`std::time::UNIX_EPOCH`), floor-based for times before the epoch (see the normalization table
below).

## Status codes

```text
0 = success
1 = clock unavailable   (ProviderStatus::CLOCK_UNAVAILABLE -- declared, not currently reachable:
                          std::time::Instant/SystemTime::now() do not themselves fail on any
                          currently supported target)
2 = value out of STARK v0.1 range (ProviderStatus::OUT_OF_RANGE)
```

Every other nonzero code is reserved for a future provider-defined failure and would map to
`TimeError::ProviderFailure(code)` at the STARK boundary (not implemented here — see
`../BLOCKERS.md`). Both functions write every declared output before returning success and write
no output at all on failure (unit-tested: `ffi_monotonic_writes_output_only_on_success`,
`ffi_unix_now_writes_canonical_output`).

## Wall-clock before-epoch normalization

`stark_time_unix_now` produces **floor** seconds, not truncation-toward-zero:

| Host reading (before epoch) | seconds | nanoseconds |
|---|---:|---:|
| exactly 1 second before epoch | `-1` | `0` |
| 1.5 seconds before epoch | `-2` | `500_000_000` |

## Use of Rust `std::time`

Exclusively `std::time::{Instant, SystemTime, UNIX_EPOCH}` from the Rust standard library. No
third-party crate: `Cargo.toml`'s `[dependencies]` table is empty (the `starkc`/`stark-runtime`
path dependencies are `[dev-dependencies]` only, used to validate this provider's metadata
against the repository's real ABI v0.1 validator — they are never linked into the provider
functions themselves).

## No resources, no callbacks

This provider declares `resource_types: []` and has no close function (unit-tested:
`provider_metadata_has_no_resource_types`). Neither function accepts a function-pointer/callback
parameter — the ABI's closed `AbiParam` vocabulary has no such form to accept.

## Panic containment

Both `extern "C"` functions wrap their pure computation in `std::panic::catch_unwind` and call
`std::process::abort()` if it panics, rather than letting the panic unwind across the `extern
"C"` boundary (undefined behavior) or converting it into an ordinary `ProviderStatus` (which
would misrepresent a host failure as a recoverable clock error). The same abort policy applies to
a null output pointer, checked explicitly before any write.

## Unsafe boundary audit

The only `unsafe` code in this crate is the two output-pointer writes, each preceded by a
non-null check and performed only after all fallible computation (overflow guards, the
`catch_unwind` panic boundary) has already succeeded — so the write is unconditionally the last
thing either function does before returning `ProviderStatus::SUCCESS`. No other `unsafe` appears
anywhere in the crate. The pure computations behind both functions (`monotonic_now_ns`,
`unix_now_normalized`) are ordinary safe Rust and are exercised directly by unit tests, without
going through the FFI boundary at all.

## Supported targets

Declared in `provider_metadata()` (`src/lib.rs`, test-only): `aarch64-apple-darwin`,
`x86_64-apple-darwin`, `x86_64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`. Only `std::time` is
used, which supports all four; see `../EVIDENCE.md` for which of these actually have build/test
evidence from this session versus being declared on the strength of `std::time`'s own portability.
