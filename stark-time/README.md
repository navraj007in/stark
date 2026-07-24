# stark-time

Deterministic duration values and checked duration arithmetic, plus the value types for
process-local monotonic time and Unix wall-clock time, for STARK Core v1.

## Current status: PARTIAL — WAITING_PROVIDER_EXECUTION

This package's pure logic is complete and tested (40 `stark test` cases). Its native clock
provider crate (`native/`) is complete and unit-tested (13 `cargo test` cases) but is **not
wired into any STARK program**: the repository does not yet have an owner-approved mechanism for
generated STARK code to link against and call a native provider function. See `BLOCKERS.md` for
the exact blocked requirement and evidence.

Concretely, this means:

- `Duration`, `Instant`, and `UnixTimestamp` all exist and their non-clock-reading operations
  work today.
- `Instant::now()`, `UnixTimestamp::now()`, and `Instant::elapsed()` are **not implemented** —
  they are the three frozen-API members that require a live clock reading through the native
  provider, and this package must not invent a way to call that provider itself.

## `Duration` vs. `Instant` vs. `UnixTimestamp`

- **`Duration`** is a span of time — how long something took, or how long to wait. It has no
  notion of "when": `Duration::from_seconds(5)` means five seconds of elapsed time, full stop.
- **`Instant`** is a single monotonic clock reading, meant only for measuring elapsed time between
  two readings in the *same process* (`checked_duration_since`, `elapsed`). It is not a
  human-meaningful timestamp.
- **`UnixTimestamp`** is a point in wall-clock time, expressed as seconds and nanoseconds relative
  to the Unix epoch (1970-01-01T00:00:00Z). It is human-meaningful (`to_unix_millis()`) but is
  **not** suitable for measuring elapsed time.

## Monotonic vs. wall clocks

The monotonic clock behind `Instant` is guaranteed nondecreasing within one process and
unaffected by wall-clock adjustments (NTP sync, manual clock changes, leap-second smearing) — use
it whenever you need "how much time passed," never "what time is it."

The wall clock behind `UnixTimestamp` answers "what time is it" but **may jump backwards** at any
moment because of host clock adjustments. Never use it to measure elapsed time: two consecutive
`UnixTimestamp::now()` calls are not guaranteed to be nondecreasing.

## `Instant`'s origin is unspecified

An `Instant`'s internal tick count is relative to a provider-private origin that is:

- unspecified (do not guess or depend on its numeric value);
- **not** the Unix epoch;
- not stable across separate program runs;
- not meaningful outside the process/provider instance that produced it.

The field is private specifically so callers cannot come to depend on the numeric origin. Compare
two `Instant`s only through `checked_duration_since`/`elapsed`, never by reading their internals
(which you cannot do — they are private).

## What this package deliberately does not do

No time zones, no calendar dates, no date arithmetic, no DST rules, no UTC offset lookup, no
formatting or parsing of timestamps (ISO 8601, RFC 3339, or otherwise), no leap-second
representation, no NTP/clock synchronization, and no `sleep`, timers, intervals, alarms,
deadlines, or scheduler integration of any kind. All of these are explicitly out of scope for
v0.1 (see the work package's §4.2) and recorded here as possible future work, not implemented.

## Checked conversion behavior

Every arithmetic operation that could overflow or underflow is checked and returns
`Option<T>`/`None` rather than trapping or wrapping: `Duration::checked_add`, `checked_sub`,
`as_millis`, `as_micros`, `as_nanos`, and `UnixTimestamp::to_unix_millis`. None of `Duration`'s or
`UnixTimestamp`'s public operations can trap.

`UnixTimestamp::from_unix_millis` normalizes negative millisecond inputs to floor-based canonical
seconds/nanoseconds (STARK's integer division truncates toward zero, so this is done explicitly
rather than relying on the division itself) — see `native/README.md` and `BLOCKERS.md` for the
worked table.

## Provider / native trust disclosure

`Instant::now()` and `UnixTimestamp::now()` (once implemented under a future work package) read
the host clock through a native Rust provider (`native/`), not through anything expressible in
pure STARK — Core v1 has no FFI, no raw pointers, and no `unsafe`. The provider crate itself uses
only `std::time` from the Rust standard library; see `native/README.md` for its exact contract,
panic-containment policy, and unsafe-boundary audit.

## Supported targets

The native provider crate's source (`std::time` only) targets:

- `aarch64-apple-darwin`
- `x86_64-apple-darwin`
- `x86_64-unknown-linux-gnu`
- `x86_64-pc-windows-msvc`

Build/test evidence exists only for the host this package was developed on; see `EVIDENCE.md`'s
cross-platform table for exactly which targets have real evidence versus which are declared but
unverified.
