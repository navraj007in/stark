# STARK `std-time` v0.1 — Gemini Implementation Work Package

**Package:** `std-time`  
**Version:** `0.1.0`  
**Public module:** `std::time` or the current approved package alias equivalent  
**Implementation:** Pure STARK value logic plus one approved Rust native provider  
**Native Provider ABI:** STARK Native Provider ABI v0.1  
**Core syntax changes:** Prohibited  
**Core semantic changes:** Prohibited  
**Unrestricted FFI:** Prohibited  
**Async/concurrency work:** Prohibited  
**Status:** Implementation specification — public API and observable behaviour are frozen for this work package  
**Repository baseline inspected:** STARK `main` at `ef3bf5adf63b6e3f6796ba61114b3ff73d2590d5`, 24 July 2026  
**Important current limitation:** the approved provider ABI has metadata validation and boundary types, but the inspected baseline does not yet provide real provider linkage/invocation. This work package contains an explicit blocker boundary for that seam.

---

## 1. Instruction to Gemini

You are the implementation engineer for one bounded STARK ecosystem package.

Implement **only** the work authorised by this document. Treat every statement containing **MUST**, **MUST NOT**, **SHALL**, **SHALL NOT**, **EXACTLY**, or **ONLY** as binding.

Do not redesign the public API. Do not add convenience functions. Do not add syntax. Do not alter ownership, borrowing, traits, arithmetic, package identity, or provider ABI policy.

Your job is to:

1. inspect the current repository and classify the prerequisites in Section 5;
2. implement the pure STARK time value types and arithmetic;
3. implement and unit-test the bounded Rust clock provider;
4. describe and validate the provider metadata against ABI v0.1;
5. integrate the provider only when a current owner-approved provider-execution mechanism already exists;
6. otherwise stop at the integration boundary and produce the required blocker record;
7. create the complete deterministic and live-clock test corpus;
8. run all supported checks;
9. record exact evidence;
10. stop without widening scope.

### 1.1 Absolute rule

The inspected provider ABI specification states that real `extern "C"` linkage, loading, and invocation are deferred. Therefore:

> **You MUST NOT invent a provider loader, linker contract, manifest schema, symbol-discovery mechanism, Cargo integration policy, or generated-code call shape inside this package.**

If the current repository still lacks an owner-approved provider execution mechanism, complete the pure package and provider crate portions that can be completed independently, mark the package `PARTIAL — WAITING_PROVIDER_EXECUTION`, and follow Section 24.

---

## 2. Objective

Create the smallest useful first-party native-backed STARK package.

`std-time` v0.1 shall provide:

- deterministic duration values and checked duration arithmetic;
- process-local monotonic clock readings for measuring elapsed time;
- wall-clock readings represented as normalized Unix timestamps;
- a narrow two-function native provider;
- no resource handles;
- no callbacks;
- no allocation requirement at the provider boundary;
- no time-zone, calendar, formatting, sleeping, timer, async, or scheduler features.

This package is intended to validate the approved Native Provider ABI with the lowest-risk real host capability.

It must demonstrate that:

1. scalar outputs cross the provider boundary correctly;
2. provider failures become typed STARK errors;
3. invalid provider output becomes a contract trap rather than an ordinary error;
4. public APIs remain STARK-defined and do not expose Rust implementation terminology;
5. the same API works across supported operating systems;
6. a standard package can use native functionality without adding a language feature.

---

## 3. Governing repository contracts

Before implementation, read the current versions of:

- `COMPILER-STATE.md`;
- `STARKLANG/docs/compiler/native-provider-abi-v0.1.md`;
- `STARKLANG/docs/proposals/CORE_PACKAGES_ECOSYSTEM_ROADMAP.md`;
- current ecosystem charter, roadmap, state, and active work package files, when present;
- `STARKLANG/docs/spec/03-Type-System.md`;
- `STARKLANG/docs/spec/06-Standard-Library.md`;
- `STARKLANG/docs/spec/07-Modules-and-Packages.md`;
- current package-manifest implementation and package examples;
- `starkc/src/backend/provider_abi.rs`;
- `stark-runtime/src/provider_abi.rs`;
- current native generated-Rust backend and runtime linkage code, but only to determine whether provider execution already exists.

The repository contracts take precedence over implementation guesses. This document freezes the package-level API and semantics; it does not authorise contradicting a newer owner-approved repository contract.

If a newer approved decision conflicts with this work package, stop and report the exact conflict. Do not silently choose one.

---

## 4. Scope

### 4.1 Included

- `Duration` representation and checked arithmetic.
- Monotonic `Instant` values.
- Wall-clock `UnixTimestamp` values.
- Current monotonic time.
- Current Unix wall time.
- Millisecond, microsecond, and nanosecond duration conversions.
- Unix-second and Unix-millisecond timestamp conversions.
- Typed clock errors.
- Native provider metadata.
- Native Rust provider implementation.
- Deterministic fake-provider tests.
- Live-provider smoke and invariant tests.
- Package documentation.
- Cross-platform evidence for macOS, Linux, and Windows when CI or machines are available.
- A provider-execution blocker report when the approved execution seam is absent.

### 4.2 Explicitly excluded

Do not implement:

- `sleep`;
- timers;
- intervals;
- alarms;
- deadlines;
- event loops;
- async/await;
- threads;
- tasks;
- cancellation;
- scheduler integration;
- native callbacks into STARK;
- local time;
- time zones;
- daylight-saving rules;
- UTC offset lookup;
- calendar dates;
- date arithmetic;
- parsing or formatting timestamps;
- ISO 8601 or RFC 3339;
- leap-second representation;
- NTP or clock synchronization;
- stopwatch convenience types;
- CPU time;
- process time;
- performance counters other than the monotonic reading;
- high-resolution platform-specific APIs when Rust `std::time` already provides the requirement;
- serialization of `Instant`;
- comparison of `Instant` values from different processes or provider instances;
- a general FFI mechanism;
- a general provider framework redesign;
- dynamic provider discovery;
- plugin loading;
- package build scripts;
- install scripts;
- compiler plugins;
- a public registry;
- any compiler intrinsic named for time.

Record excluded requests as future work. Do not implement them in v0.1.

---

## 5. Preconditions and readiness classification

Inspect the current repository. Do not assume the baseline SHA remains current.

Confirm these capabilities:

1. ordinary public structs and enums work cross-package;
2. private struct fields are enforceable across packages;
3. `UInt32`, `UInt64`, and `Int64` work in the intended engines;
4. integer division, remainder, comparison, and checked arithmetic semantics are understood;
5. `Option` and `Result` work natively for the required shapes;
6. reference receivers such as `&self` work;
7. associated functions and methods work cross-package;
8. package library entries work;
9. `stark check`, `stark test`, and `stark fmt --check` work for ordinary packages;
10. Native Provider ABI metadata validation exists;
11. runtime ABI boundary types exist;
12. a real owner-approved provider execution/linkage mechanism either:
    - exists and is documented; or
    - is absent and must be recorded as the package blocker.

### 5.1 Readiness outcomes

Classify the repository into exactly one outcome:

#### `READY_FULL`

All language, package, provider build, provider validation, linkage, invocation, error mapping, and native execution mechanisms required by this work package already exist.

Proceed through the full work package.

#### `READY_PACKAGE_PROVIDER`

The pure STARK package logic and Rust provider can be implemented and unit-tested, but real provider linkage/invocation is absent.

Proceed with:

- value types;
- pure arithmetic;
- package tests that do not call the live provider;
- provider crate;
- provider metadata;
- provider unit tests;
- deterministic wrapper tests possible through existing mocks.

Then stop with:

```text
PARTIAL — WAITING_PROVIDER_EXECUTION
```

Do not claim `std-time` works from STARK source.

#### `BLOCKED_LANGUAGE_OR_PACKAGE`

A required existing Core or package capability is absent.

Do not modify the compiler. Produce the blocker report in Section 24.

---

## 6. Work-package split

This specification contains two bounded parts.

### WP-TIME-A — Package semantics and provider implementation

Always authorised:

- pure STARK API and arithmetic;
- provider metadata;
- Rust provider implementation;
- unit tests;
- documentation;
- evidence;
- integration audit.

### WP-TIME-B — Provider execution integration

Authorised **only** when either:

1. the repository already contains an owner-approved provider execution design and implementation path; or
2. the owner explicitly supplies a separate approved work package naming the files, contracts, and escalation class.

This document alone does **not** authorise WP-TIME-B architecture.

---

## 7. Package placement and allowed files

The owner or current ecosystem state shall assign the package directory. Do not create a new top-level repository layout without approval.

Expected package structure:

```text
std-time/
├── starkpkg.json
├── README.md
├── EVIDENCE.md
├── BLOCKERS.md              # create only when blockers exist
├── src/
│   ├── lib.stark
│   └── tests.stark
└── native/
    ├── Cargo.toml
    ├── README.md
    └── src/
        └── lib.rs
```

If current package conventions use a different approved layout, use the current layout and record the difference in `EVIDENCE.md`.

### 7.1 Allowed modifications in WP-TIME-A

- files inside the assigned `std-time` package;
- package-local tests;
- package-local documentation;
- package-local Rust provider crate;
- a package-local deterministic fake provider;
- a package-local metadata fixture in the format already approved by the repository.

### 7.2 Prohibited modifications in WP-TIME-A

Do not modify:

- Core syntax;
- Core type rules;
- ownership or borrow checking;
- MIR;
- MIR verifier;
- generated-Rust slot representation;
- provider ABI types or rules;
- package manifest schema;
- root resolver;
- runtime loader/linker;
- native backend provider call generation;
- root compiler state or gate decisions;
- unrelated conformance fixtures;
- release tooling;
- CI outside package-local or owner-assigned files.

A needed change in any prohibited area is a blocker or a separate owner-approved WP.

---

## 8. Manifest

Use the current approved package manifest schema.

The logical package identity is:

```json
{
  "name": "std-time",
  "version": "0.1.0",
  "entry": "src/lib.stark",
  "dependencies": {}
}
```

The package has no STARK package dependencies in v0.1.

Native provider metadata and root-application native-code consent must use only the current approved package/provider mechanism. If no such manifest fields are approved, do not invent them.

---

## 9. Frozen public API

Implement exactly this public API unless a syntax spelling must be adjusted to match the current parser without changing meaning:

```stark
pub enum TimeError {
    ClockUnavailable,
    ClockWentBackwards,
    OutOfRange,
    ProviderFailure(UInt32)
}

pub struct Duration {
    seconds: UInt64,
    nanoseconds: UInt32
}

impl Duration {
    pub fn zero() -> Duration;
    pub fn from_seconds(seconds: UInt64) -> Duration;
    pub fn from_millis(milliseconds: UInt64) -> Duration;
    pub fn from_micros(microseconds: UInt64) -> Duration;
    pub fn from_nanos(nanoseconds: UInt64) -> Duration;

    pub fn seconds(&self) -> UInt64;
    pub fn subsec_nanos(&self) -> UInt32;
    pub fn is_zero(&self) -> Bool;

    pub fn checked_add(&self, other: &Duration) -> Option<Duration>;
    pub fn checked_sub(&self, other: &Duration) -> Option<Duration>;

    pub fn as_millis(&self) -> Option<UInt64>;
    pub fn as_micros(&self) -> Option<UInt64>;
    pub fn as_nanos(&self) -> Option<UInt64>;
}

pub struct Instant {
    ticks_nanos: UInt64
}

impl Instant {
    pub fn now() -> Result<Instant, TimeError>;
    pub fn checked_duration_since(&self, earlier: &Instant) -> Option<Duration>;
    pub fn elapsed(&self) -> Result<Duration, TimeError>;
}

pub struct UnixTimestamp {
    seconds: Int64,
    nanoseconds: UInt32
}

impl UnixTimestamp {
    pub fn now() -> Result<UnixTimestamp, TimeError>;

    pub fn from_unix_seconds(seconds: Int64) -> UnixTimestamp;
    pub fn from_unix_millis(milliseconds: Int64) -> UnixTimestamp;

    pub fn seconds(&self) -> Int64;
    pub fn subsec_nanos(&self) -> UInt32;
    pub fn to_unix_millis(&self) -> Option<Int64>;
}
```

### 9.1 API restrictions

- Fields MUST remain private outside the package.
- Do not add public constructors from raw monotonic ticks.
- Do not expose provider status codes except through `ProviderFailure(UInt32)`.
- Do not expose Rust types or crate names.
- Do not expose an operating-system clock identifier.
- Do not add `sleep`.
- Do not add calendar conversion.
- Do not add formatting.
- Do not add operator overloads unless a separate owner decision requests them.
- Do not add `Display`, `Hash`, serialization, or parsing implementations.
- Do not add `Clone`/`Copy` policy declarations unless current STARK semantics require explicit declarations. These values contain no resources and may follow ordinary scalar-aggregate value semantics.
- No public API may mention “Rust”, “SystemTime”, “OnceLock”, “FFI”, “provider”, or target-specific terminology.

---

## 10. Constants and canonical representation

Use these exact constants:

```text
NANOS_PER_SECOND = 1_000_000_000
NANOS_PER_MILLI  = 1_000_000
NANOS_PER_MICRO  = 1_000
MILLIS_PER_SECOND = 1_000
MICROS_PER_SECOND = 1_000_000
```

### 10.1 `Duration`

A `Duration` represents a nonnegative span.

Canonical invariant:

```text
0 <= nanoseconds < 1_000_000_000
```

The represented value is:

```text
seconds + nanoseconds / 1_000_000_000
```

No public operation may produce a noncanonical `Duration`.

### 10.2 `Instant`

`ticks_nanos` is a process-local monotonic reading measured in nanoseconds from a provider-private origin.

Its origin is:

- unspecified;
- not the Unix epoch;
- not stable across program runs;
- not serializable;
- not meaningful outside the current process/provider instance.

The public field is private so callers cannot depend on the numeric origin.

### 10.3 `UnixTimestamp`

Canonical invariant:

```text
0 <= nanoseconds < 1_000_000_000
```

The represented timestamp is:

```text
seconds + nanoseconds / 1_000_000_000
```

`seconds` is the mathematical floor relative to the Unix epoch, not truncation toward zero.

Example:

```text
1969-12-31T23:59:59.500Z
seconds     = -1
nanoseconds = 500_000_000
```

Unix time in this package does not represent leap seconds. Time zones and civil calendars are outside v0.1.

---

## 11. `Duration` semantics

### 11.1 `zero`

Returns exactly:

```text
seconds = 0
nanoseconds = 0
```

### 11.2 `from_seconds`

Returns:

```text
seconds = input
nanoseconds = 0
```

This constructor cannot overflow.

### 11.3 `from_millis`

For `milliseconds`:

```text
seconds = milliseconds / 1_000
nanoseconds = (milliseconds % 1_000) * 1_000_000
```

This constructor cannot overflow its canonical representation.

### 11.4 `from_micros`

```text
seconds = microseconds / 1_000_000
nanoseconds = (microseconds % 1_000_000) * 1_000
```

### 11.5 `from_nanos`

```text
seconds = nanoseconds / 1_000_000_000
subsec = nanoseconds % 1_000_000_000
```

### 11.6 Accessors

- `seconds()` returns the whole-second component.
- `subsec_nanos()` returns the canonical subsecond component.
- `is_zero()` is true only when both components are zero.

### 11.7 `checked_add`

Add two canonical durations without trapping.

Algorithm:

1. Add the second components with an explicit overflow guard.
2. Add the nanosecond components in a type wide enough for a maximum of `1_999_999_998`.
3. If nanoseconds are at least `1_000_000_000`, subtract that amount and add one second.
4. Guard the carry addition.
5. Return `None` on any overflow.
6. Otherwise return a canonical `Duration`.

Do not rely on checked arithmetic trapping and then catch the trap. Prevent the trap.

### 11.8 `checked_sub`

Subtract `other` from `self`.

- Return `None` when `other > self`.
- Borrow one second when `self.nanoseconds < other.nanoseconds`.
- Return a canonical value.
- Do not trap on underflow.

Comparison is lexicographic by `(seconds, nanoseconds)`.

### 11.9 `as_millis`

Compute:

```text
seconds * 1_000 + nanoseconds / 1_000_000
```

Return `None` if the multiplication or addition cannot fit in `UInt64`.

Sub-millisecond precision is truncated.

### 11.10 `as_micros`

Compute:

```text
seconds * 1_000_000 + nanoseconds / 1_000
```

Return `None` on overflow.

### 11.11 `as_nanos`

Compute:

```text
seconds * 1_000_000_000 + nanoseconds
```

Return `None` on overflow.

---

## 12. `Instant` semantics

### 12.1 Clock requirements

The monotonic provider reading MUST be:

- nondecreasing within one process/provider instance;
- unaffected by wall-clock adjustments;
- suitable for elapsed-time measurement;
- allowed to return the same value in consecutive calls;
- represented in nanoseconds;
- bounded to `UInt64`.

No resolution guarantee is made beyond the provider returning nanosecond units. The underlying clock may have coarser real resolution.

### 12.2 `Instant::now`

Call the monotonic provider function.

On provider status:

- `0` → construct `Instant`;
- `1` → `Err(TimeError::ClockUnavailable)`;
- `2` → `Err(TimeError::OutOfRange)`;
- any other nonzero code → `Err(TimeError::ProviderFailure(code))`.

The output slot is invalid on failure and MUST NOT be read.

### 12.3 `checked_duration_since`

If `self.ticks_nanos < earlier.ticks_nanos`, return `None`.

Otherwise:

```text
delta = self.ticks_nanos - earlier.ticks_nanos
return Some(Duration::from_nanos(delta))
```

### 12.4 `elapsed`

1. Call `Instant::now()`.
2. Propagate a provider error unchanged.
3. Compute `now.checked_duration_since(self)`.
4. If it returns `None`, return `Err(TimeError::ClockWentBackwards)`.
5. Otherwise return the duration.

A clock regression is not `OutOfRange` and not an unknown provider status.

---

## 13. `UnixTimestamp` semantics

### 13.1 Wall-clock behaviour

The wall clock represents Unix time relative to:

```text
1970-01-01T00:00:00Z
```

The wall clock:

- may move forwards or backwards because of host clock adjustment;
- is not suitable for elapsed-time measurement;
- is not monotonic;
- carries no time-zone information;
- ignores leap-second representation;
- may represent times before the epoch.

### 13.2 `UnixTimestamp::now`

Call the wall-clock provider function with two output slots:

```text
seconds: Int64
nanoseconds: UInt32
```

On provider status:

- `0` → validate and construct;
- `1` → `Err(TimeError::ClockUnavailable)`;
- `2` → `Err(TimeError::OutOfRange)`;
- any other nonzero code → `Err(TimeError::ProviderFailure(code))`.

Both output slots are invalid on failure and MUST NOT be read.

On success:

```text
nanoseconds < 1_000_000_000
```

MUST hold.

A successful provider response with `nanoseconds >= 1_000_000_000` is a provider contract violation and MUST use the existing STARK provider-contract trap channel. It MUST NOT become `TimeError`.

### 13.3 `from_unix_seconds`

Returns:

```text
seconds = input
nanoseconds = 0
```

### 13.4 `from_unix_millis`

STARK integer division may truncate toward zero, so normalize negative values explicitly.

Let:

```text
q = milliseconds / 1_000
r = milliseconds % 1_000
```

If `r >= 0`:

```text
seconds = q
nanoseconds = r * 1_000_000
```

If `r < 0`:

```text
seconds = q - 1
nanoseconds = (r + 1_000) * 1_000_000
```

This produces floor-based canonical timestamps.

Examples:

| Milliseconds | Seconds | Nanoseconds |
|---:|---:|---:|
| `0` | `0` | `0` |
| `1` | `0` | `1_000_000` |
| `999` | `0` | `999_000_000` |
| `1000` | `1` | `0` |
| `-1` | `-1` | `999_000_000` |
| `-999` | `-1` | `1_000_000` |
| `-1000` | `-1` | `0` |
| `-1001` | `-2` | `999_000_000` |

### 13.5 `to_unix_millis`

Compute:

```text
seconds * 1_000 + nanoseconds / 1_000_000
```

Use explicit overflow guards.

Return `None` when the result cannot fit in `Int64`.

The result truncates precision below one millisecond. Because the timestamp representation uses floor seconds, negative subsecond values convert correctly:

```text
(-1 seconds, 500_000_000 nanos) -> -500 milliseconds
```

---

## 14. Error semantics

```stark
pub enum TimeError {
    ClockUnavailable,
    ClockWentBackwards,
    OutOfRange,
    ProviderFailure(UInt32)
}
```

### 14.1 `ClockUnavailable`

Use when the native clock source cannot provide the requested reading.

### 14.2 `ClockWentBackwards`

Use only when a monotonic reading used by `elapsed` is earlier than the stored `Instant`.

Do not use this for wall-clock regression.

### 14.3 `OutOfRange`

Use when the provider obtains a valid host value that cannot fit the v0.1 STARK representation.

Examples:

- monotonic elapsed nanoseconds exceed `UInt64`;
- Unix seconds cannot fit `Int64`.

### 14.4 `ProviderFailure(code)`

Use for a nonzero provider code not assigned above.

Do not expose Rust error strings, OS error objects, panic text, or host-specific type names.

### 14.5 Contract violations

These are traps, not `TimeError`:

- provider reports success but does not initialize an output;
- wall-clock nanoseconds are not canonical;
- provider writes outside an output;
- provider ABI metadata does not match the called function;
- any other violation already classified as a provider contract violation by ABI v0.1.

### 14.6 Host failures

Provider panics, aborts, or environment failures classified as host failures MUST NOT be presented as ordinary `TimeError`.

---

## 15. Native provider contract

Provider logical identity:

```text
name: "stark-std-time"
semver: 0.1.0
abi_version: "0.1"
capabilities: ["clock"]
resource_types: []
```

Supported target triples MUST be the exact current supported triples for which provider artifacts are actually built. Do not claim an untested target.

Minimum intended validation targets:

```text
aarch64-apple-darwin
x86_64-apple-darwin
x86_64-unknown-linux-gnu
x86_64-pc-windows-msvc
```

A target may be omitted temporarily when no build evidence exists. Record the omission.

### 15.1 Function 1 — monotonic time

Logical declaration:

```text
name: "stark_time_monotonic_now_ns"
capability: "clock"
params:
  - ScalarOut(U64)
is_close_for: None
may_block: false
physical return: ProviderStatus
```

### 15.2 Function 2 — Unix wall time

Logical declaration:

```text
name: "stark_time_unix_now"
capability: "clock"
params:
  - ScalarOut(I64)
  - ScalarOut(U32)
is_close_for: None
may_block: false
physical return: ProviderStatus
```

### 15.3 Status codes

```text
0 = success
1 = clock unavailable
2 = value out of STARK v0.1 range
```

All other nonzero codes are reserved for provider-defined failures and map to `ProviderFailure(code)`.

Do not use code `0` without initializing every declared output.

### 15.4 No resources

This provider declares no resource type and no close function.

`Instant`, `UnixTimestamp`, and `Duration` are ordinary values, not provider handles.

### 15.5 No buffers

No string or byte buffer crosses the provider boundary.

### 15.6 No callbacks

No function pointer or callback is accepted.

### 15.7 No blocking

Both functions declare `may_block: false`.

---

## 16. Rust provider implementation

Use Rust stable and `std::time`. Do not add a third-party dependency.

### 16.1 Monotonic source

Use a process-local origin based on `std::time::Instant`.

A suitable internal design is:

```rust
static ORIGIN: OnceLock<std::time::Instant> = OnceLock::new();
```

For each call:

1. obtain or initialize the origin;
2. compute `origin.elapsed()`;
3. convert the elapsed nanoseconds to `u64`;
4. return status `2` if the `u128` nanosecond count exceeds `u64::MAX`;
5. write the output only on success;
6. return status `0`.

The public STARK API must not expose the origin or its numeric meaning.

### 16.2 Wall-clock source

Use:

```rust
std::time::SystemTime::now()
std::time::UNIX_EPOCH
```

For a time at or after the epoch:

```text
seconds = duration.as_secs()
nanoseconds = duration.subsec_nanos()
```

Check that seconds fit `i64`.

For a time before the epoch, let:

```text
d = UNIX_EPOCH.duration_since(now)
s = d.as_secs()
n = d.subsec_nanos()
```

Normalize as:

If `n == 0`:

```text
seconds = -s
nanoseconds = 0
```

Otherwise:

```text
seconds = -s - 1
nanoseconds = 1_000_000_000 - n
```

All signed conversions and negations MUST be guarded. Return status `2` when the value cannot fit `Int64`.

### 16.3 Output pointers

The physical ABI uses out-pointers. Follow the approved runtime/provider ABI helpers and safety rules.

At minimum:

- reject or classify an impossible null output pointer according to the approved provider-execution contract;
- never dereference an output pointer before all fallible computation needed for success has completed;
- write every output exactly once on success;
- write no output on ordinary provider failure;
- do not retain pointers after return.

Do not invent a null-pointer policy if the provider execution WP already defines one. Use the approved policy.

### 16.4 Panic containment

A Rust panic MUST NOT unwind across the provider boundary.

Use the owner-approved provider panic policy. When no more specific policy exists:

- catch unwinding at the exported boundary;
- classify the panic as a host failure;
- abort without unwinding rather than converting it to an ordinary provider error.

Do not expose panic text to STARK.

### 16.5 Unsafe code

Keep unsafe operations:

- minimal;
- isolated to ABI output writes;
- documented;
- covered by Rust unit tests where possible.

Do not use unsafe code for clock arithmetic.

### 16.6 Allocation

The provider functions SHOULD allocate nothing after the monotonic origin has been initialized.

No allocation guarantee is part of the public STARK API, but unnecessary allocation is a defect.

---

## 17. STARK/provider wrapper requirements

The wrapper between provider calls and public STARK values MUST:

1. allocate scalar outputs using the approved uninitialized-output mechanism;
2. invoke the exact declared provider function;
3. inspect `ProviderStatus` before reading outputs;
4. never read outputs on failure;
5. map codes exactly as Section 14 defines;
6. validate wall nanoseconds after success;
7. construct private STARK value representations;
8. expose no provider or Rust implementation detail.

If the current generated-Rust backend cannot emit provider calls, do not implement an ad hoc direct Rust call in package source. Report the integration blocker.

---

## 18. Deterministic fake provider

Live clocks cannot support exact expected-value tests. Supply a deterministic test provider or the current approved equivalent.

The fake provider must support scripted outcomes for:

- monotonic success with a chosen `UInt64`;
- wall success with chosen `Int64` seconds and `UInt32` nanoseconds;
- status `1`;
- status `2`;
- arbitrary unknown status;
- successful wall status with invalid nanoseconds for contract-trap testing.

Recommended scripted monotonic readings:

```text
10_000_000_000
10_000_000_000
10_000_000_123
```

Recommended wall readings:

```text
0 seconds, 0 nanos
1 second, 500_000_000 nanos
-1 seconds, 500_000_000 nanos
```

The fake provider is test-only and MUST NOT ship as the production provider.

Do not add callbacks to inject the fake provider. Use the existing test/provider selection mechanism. If none exists, test the wrapper at the lowest deterministic layer available and record the missing injection capability without redesigning the ABI.

---

## 19. Test corpus — pure `Duration`

Add tests for all of the following.

### 19.1 Constructors

| Input | Expected seconds | Expected nanos |
|---|---:|---:|
| zero | `0` | `0` |
| `from_seconds(5)` | `5` | `0` |
| `from_millis(1)` | `0` | `1_000_000` |
| `from_millis(999)` | `0` | `999_000_000` |
| `from_millis(1000)` | `1` | `0` |
| `from_millis(1001)` | `1` | `1_000_000` |
| `from_micros(1)` | `0` | `1_000` |
| `from_micros(999_999)` | `0` | `999_999_000` |
| `from_micros(1_000_000)` | `1` | `0` |
| `from_nanos(1)` | `0` | `1` |
| `from_nanos(999_999_999)` | `0` | `999_999_999` |
| `from_nanos(1_000_000_000)` | `1` | `0` |
| `from_nanos(1_000_000_001)` | `1` | `1` |

### 19.2 Zero

- zero is zero;
- a nonzero seconds value is not zero;
- a nonzero nanoseconds value is not zero.

### 19.3 Checked addition

Test:

- zero plus zero;
- zero plus value;
- addition without nanosecond carry;
- addition with exact carry;
- addition with carry and remainder;
- second-component overflow;
- overflow caused only by nanosecond carry;
- operands remain unchanged.

### 19.4 Checked subtraction

Test:

- equal values produce zero;
- subtraction without borrow;
- subtraction with nanosecond borrow;
- subtraction to zero;
- underflow returns `None`;
- operands remain unchanged.

### 19.5 Unit conversion

Test exact values and overflow:

- seconds to millis/micros/nanos;
- mixed seconds and nanoseconds;
- truncation below the requested unit;
- maximum value that fits;
- first value that overflows;
- no conversion traps.

---

## 20. Test corpus — `UnixTimestamp`

### 20.1 Seconds constructor

Test:

- zero;
- positive;
- negative;
- `Int64` minimum;
- `Int64` maximum.

All have zero nanoseconds.

### 20.2 Millisecond constructor

Pin every row from Section 13.4, plus:

```text
1_001
-1_001
Int64::MIN
Int64::MAX
```

For all constructed values:

```text
nanoseconds < 1_000_000_000
```

### 20.3 Millisecond conversion

Test:

- zero;
- positive exact;
- positive sub-millisecond truncation;
- negative exact;
- negative sub-millisecond conversion;
- overflow at positive range;
- overflow at negative range.

### 20.4 Round trips

For representative millisecond values:

```text
to_unix_millis(from_unix_millis(x)) == Some(x)
```

Include positive, negative, zero, and boundary-adjacent values.

---

## 21. Test corpus — deterministic provider/wrapper

When the repository supports deterministic provider execution or wrapper injection, test:

1. monotonic success maps to `Instant`;
2. repeated equal readings are accepted;
3. increasing reading produces the exact duration;
4. earlier `self`/later `earlier` returns `None`;
5. `elapsed` maps a regression to `ClockWentBackwards`;
6. wall epoch maps to `(0,0)`;
7. positive fractional wall time maps correctly;
8. negative fractional wall time maps correctly;
9. status `1` maps to `ClockUnavailable`;
10. status `2` maps to `OutOfRange`;
11. an unknown status maps to `ProviderFailure(code)`;
12. outputs are not read on failure;
13. wall nanoseconds equal to `1_000_000_000` trap as a provider contract violation;
14. provider metadata validates;
15. a wrong ABI version is rejected;
16. a missing capability is rejected;
17. an unsupported target is rejected before invocation;
18. no resource close function is expected because no resource type is declared;
19. both functions are declared `may_block: false`.

---

## 22. Test corpus — live provider

Live tests must assert invariants, not exact clock values.

### 22.1 Monotonic smoke

1. call `Instant::now()` twice;
2. require the second reading not to be earlier than the first;
3. allow equality;
4. require `checked_duration_since` to return `Some`;
5. do not sleep merely to force progress.

### 22.2 Elapsed smoke

1. capture an `Instant`;
2. perform bounded ordinary computation;
3. call `elapsed`;
4. require success and a canonical duration;
5. do not assert a minimum nonzero duration.

### 22.3 Wall-clock sandwich

At the Rust/provider integration test layer:

1. capture host wall time immediately before the provider call;
2. call the provider;
3. capture host wall time immediately after;
4. convert all three to the same normalized Unix representation;
5. assert the provider value lies within the inclusive before/after interval.

Do not use an arbitrary five-second tolerance when a sandwich assertion is possible.

### 22.4 Canonicality

Every successful live wall reading must satisfy:

```text
nanoseconds < 1_000_000_000
```

### 22.5 No relation between clocks

Do not test or claim a stable relation between monotonic ticks and Unix wall time.

---

## 23. Cross-engine and cross-platform evidence

### 23.1 Pure logic

Run deterministic duration and timestamp arithmetic through every available relevant engine:

- HIR interpreter;
- MIR interpreter;
- native generated-Rust backend.

Require exact agreement.

### 23.2 Live provider

Do not compare exact clock readings across sequential engines. Each engine observes a different instant.

Instead compare:

- success/failure classification;
- canonical output invariants;
- monotonic non-regression within one engine run;
- provider status mapping using the deterministic fake provider.

### 23.3 Required platforms

Target evidence:

- macOS;
- Linux;
- Windows.

For each platform record:

- target triple;
- compiler version;
- Rust version;
- provider build result;
- metadata validation result;
- pure tests;
- fake-provider tests;
- live-provider tests;
- package native consumer result.

Do not mark an untested platform supported.

---

## 24. Blocker and escalation protocol

When blocked, create `BLOCKERS.md` with this exact structure:

```markdown
# std-time v0.1 blockers

## Classification

READY_FULL | READY_PACKAGE_PROVIDER | BLOCKED_LANGUAGE_OR_PACKAGE

## Repository head

<exact SHA>

## Blocked requirement

<one precise capability>

## Evidence

- command:
- source:
- expected:
- actual:
- diagnostic:

## Why this package cannot fix it

<name the prohibited compiler/runtime/provider area>

## Existing approved owner

<work package, gate, or "none found">

## Minimum next decision

<one bounded decision; no implementation proposal wider than necessary>

## Work completed safely

<list completed package/provider files and tests>

## Closure status

PARTIAL — WAITING_PROVIDER_EXECUTION
or
BLOCKED — WAITING_<CAPABILITY>
```

### 24.1 Mandatory provider-execution blocker

If real provider execution is still absent, the blocker must state:

- ABI v0.1 metadata validation exists;
- ABI runtime boundary types exist;
- real linkage/invocation is explicitly deferred in the approved ABI document;
- `std-time` cannot claim native STARK execution until an owner-approved execution seam lands;
- no loader/linker/manifest policy was invented.

### 24.2 Escalations

Stop and escalate if implementation would require:

- a provider ABI change;
- a new manifest field;
- a package identity change;
- a compiler intrinsic;
- MIR changes;
- a runtime error-channel redesign;
- direct aggregate crossing of the ABI;
- callback support;
- async/concurrency;
- a new host-failure policy;
- a public API change;
- a new supported target without evidence.

---

## 25. Implementation order

Follow this order.

### Step 1 — Audit

- inspect current head;
- classify readiness;
- record relevant files and decisions;
- identify package directory.

### Step 2 — Freeze local test fixtures

Before implementation, write tests for:

- duration constructors;
- negative Unix milliseconds;
- overflow;
- error mapping;
- invalid successful wall nanoseconds;
- monotonic equality.

### Step 3 — Implement pure values

Implement:

- constants;
- `Duration`;
- `UnixTimestamp` conversion methods;
- private canonical construction helpers.

Run pure tests.

### Step 4 — Implement provider metadata

Create the exact two-function metadata and run ABI validation.

### Step 5 — Implement Rust provider

Implement:

- monotonic source;
- Unix wall source;
- normalization;
- status codes;
- panic containment;
- unit tests.

### Step 6 — Implement wrapper only through approved seam

- use existing approved invocation;
- map status;
- protect output initialization;
- validate output;
- expose public methods.

If no seam exists, stop this step and follow Section 24.

### Step 7 — Deterministic provider tests

Run fake-provider mapping tests where the repository supports them.

### Step 8 — Live tests

Run invariant-based live provider tests.

### Step 9 — Cross-package consumer

When native execution works, create or use an owner-assigned tiny consumer that:

- obtains a monotonic instant;
- obtains a wall timestamp;
- performs bounded work;
- obtains elapsed duration;
- validates results with assertions;
- does not rely on printed output.

### Step 10 — Full evidence

Run package checks and the required repository regression set for any authorised shared files.

### Step 11 — Documentation and status

Update package-local `README.md`, `EVIDENCE.md`, and `BLOCKERS.md` when needed.

Stop.

---

## 26. Required commands

Use the repository’s current commands. At minimum attempt and record:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

For the package, use the current equivalents of:

```bash
stark check
stark test
stark fmt --check
stark doc
stark build
stark run
```

Do not claim a command passed when it was not run.

If a command is unavailable or unrelated to the current package stage, record:

```text
NOT RUN — <reason>
```

---

## 27. Documentation requirements

### 27.1 Package README

Explain:

- difference between `Duration`, `Instant`, and `UnixTimestamp`;
- monotonic versus wall clocks;
- process-local unspecified `Instant` origin;
- wall clock may move backwards;
- no time zones or calendar conversion;
- no `sleep`;
- checked conversion behaviour;
- provider/native trust disclosure;
- supported targets with actual evidence;
- current package status.

### 27.2 Native README

Explain:

- provider identity;
- ABI version;
- exact functions;
- status codes;
- use of Rust `std::time`;
- no resources;
- no callbacks;
- no third-party crates;
- panic containment;
- supported targets;
- unsafe boundary audit.

### 27.3 API examples

Include examples equivalent to:

```stark
fn measure() -> Result<Duration, TimeError> {
    let start = Instant::now()?;
    let mut sum: UInt64 = 0;
    let mut i: UInt64 = 0;
    while i < 1000 {
        sum = sum + i;
        i = i + 1;
    }
    start.elapsed()
}
```

and:

```stark
fn current_unix_millis() -> Result<Int64, TimeError> {
    let timestamp = UnixTimestamp::now()?;
    match timestamp.to_unix_millis() {
        Option::Some(value) => Result::Ok(value),
        Option::None => Result::Err(TimeError::OutOfRange),
    }
}
```

Adjust only for current valid STARK syntax.

---

## 28. Evidence file

Create `EVIDENCE.md` containing:

```markdown
# std-time v0.1 evidence

## Repository head

<sha>

## Readiness classification

<classification>

## Files changed

<list>

## Public API

<exact API>

## Provider metadata

<identity, ABI, capabilities, functions, targets>

## Pure test results

<commands and counts>

## Provider unit-test results

<commands and counts>

## Fake-provider results

<commands and counts or blocker>

## Live-provider results

<commands and counts or blocker>

## Cross-engine evidence

<HIR/MIR/native>

## Cross-platform evidence

<platform table>

## Formatting and lint

<exact commands>

## Full regression

<exact command and result>

## Known limitations

<none, or exact blockers>

## Final status

COMPLETE
or
PARTIAL — WAITING_PROVIDER_EXECUTION
or
BLOCKED — WAITING_<CAPABILITY>
```

---

## 29. Definition of done

### 29.1 `WP-TIME-A COMPLETE`

All of the following:

- frozen public API implemented for all pure operations;
- every pure arithmetic test passes;
- provider metadata validates;
- Rust provider unit tests pass;
- provider uses no third-party dependency;
- provider has no resource types or callbacks;
- provider output normalization is correct before and after Unix epoch;
- package documentation is complete;
- evidence is recorded;
- no prohibited file was modified.

This status does not claim STARK programs can call the live provider.

### 29.2 `std-time v0.1 COMPLETE`

All of the following:

- `WP-TIME-A COMPLETE`;
- an owner-approved provider execution seam exists;
- STARK native code invokes both functions through ABI v0.1;
- output slots are never read on failure;
- status mapping is exact;
- invalid successful output traps;
- deterministic fake-provider tests pass;
- live-provider invariant tests pass;
- pure HIR/MIR/native agreement passes;
- at least one cross-package native STARK consumer passes;
- macOS, Linux, and Windows provider artifacts and tests pass, or unsupported targets are explicitly omitted from metadata;
- formatting, clippy, package tests, and required regression suite pass;
- `EVIDENCE.md` declares `COMPLETE`;
- no unresolved blocker remains.

### 29.3 Invalid completion claims

Do not claim complete when only:

- Rust provider unit tests pass;
- HIR interpreter tests pass;
- metadata validates;
- generated Rust calls `std::time` through an ad hoc compiler special case;
- one platform was tested while metadata claims three;
- live clock values were manually inspected;
- exact-time tests pass only through sleeps or wide arbitrary tolerances;
- provider execution is mocked but not real.

---

## 30. Review checklist

Before reporting completion, answer every item:

### Scope

- [ ] No `sleep`.
- [ ] No calendar or time-zone work.
- [ ] No async/concurrency.
- [ ] No new syntax.
- [ ] No compiler intrinsic.
- [ ] No general FFI.
- [ ] No provider ABI redesign.

### API

- [ ] Public API exactly matches Section 9.
- [ ] Fields are private.
- [ ] No Rust terminology leaks.
- [ ] `Instant` origin is unspecified and hidden.
- [ ] Negative Unix milliseconds normalize correctly.

### Arithmetic

- [ ] All `Duration` results are canonical.
- [ ] Checked add cannot trap.
- [ ] Checked subtract cannot trap.
- [ ] Unit conversions return `None` on overflow.
- [ ] Unix millisecond conversion handles negatives.

### Provider

- [ ] ABI version is `0.1`.
- [ ] Capability is `clock`.
- [ ] No resource types.
- [ ] Exactly two functions.
- [ ] Both `may_block: false`.
- [ ] Outputs written only on success.
- [ ] Panic does not unwind across boundary.
- [ ] No third-party dependency.
- [ ] Supported targets match evidence.

### Integration

- [ ] Status checked before output read.
- [ ] Unknown code becomes `ProviderFailure`.
- [ ] Invalid wall nanoseconds trap.
- [ ] No ad hoc provider loader was invented.
- [ ] No aggregate crosses the ABI.

### Tests

- [ ] Pure deterministic tests.
- [ ] Negative timestamp tests.
- [ ] Overflow tests.
- [ ] Fake-provider tests.
- [ ] Live monotonic invariants.
- [ ] Live wall-clock sandwich.
- [ ] Cross-engine pure agreement.
- [ ] Cross-platform evidence.

### Evidence

- [ ] Exact repository SHA.
- [ ] Exact commands.
- [ ] Exact test counts.
- [ ] Exact unsupported items.
- [ ] Honest final status.

---

## 31. Required final report from Gemini

Return a concise report in this format:

```markdown
## Status

COMPLETE
or
PARTIAL — WAITING_PROVIDER_EXECUTION
or
BLOCKED — WAITING_<CAPABILITY>

## Repository head

<sha>

## Implemented

- ...

## Files changed

- ...

## Validation

- `<command>` — PASS/FAIL/NOT RUN
- ...

## Provider execution

- existing approved seam used: yes/no
- real STARK native call proven: yes/no

## Cross-platform

| Target | Build | Tests | Live provider |
|---|---|---|---|
| ... | ... | ... | ... |

## Blockers or limitations

- ...

## Scope confirmation

- no Core changes;
- no provider ABI changes;
- no manifest-schema invention;
- no async/concurrency;
- no time-zone/calendar work.
```

Do not say “done” unless every criterion for the reported status is satisfied.

---

## 32. Owner review points

The owner should review these before authorising full integration:

1. Is the Section 9 public API accepted as the official `std-time` v0.1 API?
2. Does the current ecosystem package format have an approved native-provider declaration?
3. Does an owner-approved provider execution/linkage design now exist?
4. Which target triples are officially supported for v0.1?
5. What exact runtime mechanism raises provider-contract traps?
6. What exact policy handles provider panics and null output pointers?
7. Where should the first-party `std-time` package live?
8. Which CI jobs supply macOS, Linux, and Windows evidence?

Until points 2, 3, 5, and 6 have approved answers in the repository, Gemini must not invent them.
